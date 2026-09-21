"""貪欲法(forward selection)による特徴量探索(2026-09-20追加、ユーザー提案「ベースから
貪欲法で検証したい」)。

FEATURE_TOGGLESの個別「改善/悪化」コメントは過去の設定(label_mode/timescale等が今と違った
時点)でのアドホックなテスト結果で、しかも多くはTrueの候補を複数同時に組み合わせたテスト構成
での結果であり、単一特徴量の効果として綺麗に切り分けられていない。現在の設定
(risk_adjusted_blend/gap=5/hp=5/fused AdamW等)のもとで、1個ずつクリーンに再検証する。

アルゴリズム:
    1. 選択集合を BASELINE_STOCK_COLS + BASELINE_MACRO_COLS (現在の本採用ベースライン)から
       開始する。参照PFはこのベースラインのアンサンブル日次Top-N PF
       (baseline_model_cacheがあれば再学習せず再利用)。
    2. 各ラウンドで、選択集合に残り候補を1個ずつ追加した構成を全部並列学習する
       (候補数 x len(SEEDS) 個のジョブを1つの並列プールに投げる。parallel_seed_sweep.pyの
       run_jobs_parallel/worker_train_onyをそのまま流用——config_labelを候補名にするだけで
       追加の変更は不要)。
    3. 各候補についてアンサンブル(len(SEEDS)体)の日次Top-N PFを評価する
       (stage3_toggle_experiment.py本体の[判定・日次Top-N]と同じ指標)。
    4. そのラウンドで最もPFが高かった候補が参照PFを上回っていれば選択集合に追加して
       次ラウンドへ進む——このとき、そのラウンドで既に学習済みのモデルをそのまま次の
       参照として使い回す(再学習しない)。誰も参照PFを上回らなければ探索終了。

設定(MODEL_ARCH/ENTRY_CONVENTION/MAX_PAIR_GAP/HOLDING_PERIOD/SEQ_LEN/TRAIN_BATCH_SIZE/
LABEL_MODE等)はすべてstage3_toggle_experiment.pyの現在の値をそのまま使う。FEATURE_TOGGLES
のOn/Off状態そのものは無視する(貪欲法が独自に候補集合を管理するため)。

使い方:
    python greedy_feature_search.py                  # 全候補、改善マージン0
    python greedy_feature_search.py --margin 0.01     # 改善マージンを設定(seedノイズ対策)
    python greedy_feature_search.py --max-concurrent 8
    python greedy_feature_search.py --candidates market_above_ma25_ratio,market_turnover_z
                                                       # 候補プールを指定した特徴量だけに絞る
                                                       # (既存の全FEATURE_TOGGLES再検証を避けたい場合)
"""
import sys
import os
import time
import pickle
import tempfile
import shutil

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_WAIT_POLICY"] = "PASSIVE"

try:
    sys.stdout.reconfigure(line_buffering=True, encoding='utf-8')
    sys.stderr.reconfigure(line_buffering=True, encoding='utf-8')
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RESULT_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "greedy_search_log.parquet")


def augment_macro_pool_df_all(m, macro_pool_df):
    """m.augment_macro_pool_dfは現在のTEST_MACRO_COLS|BASELINE_MACRO_COLSに含まれる
    computed列しか追加しない。貪欲法は現在Falseな候補(SHAP交互作用等)も含めて全候補を
    試すため、MACRO_COMPUTE_FNSの全エントリを無条件に追加しておく必要がある
    (2026-09-20、round1実行時にKeyError: 'cta_net_norm_x_market_vol_regime' not in index
    で発覚)。"""
    macro_pool_df = macro_pool_df.copy()
    for f, fn in m.MACRO_COMPUTE_FNS.items():
        if f not in macro_pool_df.columns:
            macro_pool_df[f] = fn(macro_pool_df)
    return macro_pool_df


def build_shared_pools(m):
    with open(m.UNIVERSE_PATH, "r", encoding="utf-8") as f:
        tickers = [line.strip() for line in f if line.strip()]
    macro_df = m.load_macro_slim5(return_source="nk225")
    pool = m.get_full_feature_pool_df(tickers, macro_df)
    margin_pool = m.get_margin_pool_df(tickers, macro_df)
    sector_pool = m.get_sector_relative_pool_df(tickers, macro_df)
    macro_pool_df = augment_macro_pool_df_all(m, m.get_full_macro_pool_df(tickers))
    loss_regime_labels = m.compute_regime_labels_expanding(macro_pool_df) if m.USE_REGIME_AWARE_LOSS else None

    _base_per_ticker = m.build_per_ticker(pool, margin_pool, sector_pool, macro_pool_df, m.BASELINE_STOCK_COLS)
    _base_valid_dates = m.valid_cross_section_dates(_base_per_ticker, m.BASELINE_STOCK_COLS, m.MIN_CROSS_SECTION)
    shared_split_date = m.compute_global_split_date(_base_valid_dates, None)
    return pool, margin_pool, sector_pool, macro_pool_df, loss_regime_labels, shared_split_date


def build_config_dataset(m, stock_cols, macro_cols, pool, margin_pool, sector_pool, macro_pool_df,
                          loss_regime_labels, shared_split_date):
    tr, va = m.build_dataset(macro_cols, stock_cols, pool, margin_pool, sector_pool, macro_pool_df,
                              regime_filter=None, regime_labels_for_loss=loss_regime_labels,
                              global_split_date_override=shared_split_date)
    pool_bt = m.prepare_backtest_pool(macro_cols, stock_cols, pool, margin_pool, sector_pool, macro_pool_df,
                                       regime_filter=None, global_split_date_override=shared_split_date)
    return tr, va, pool_bt


def ensemble_daily_pf(m, models, pool_bt):
    d_ens = m.run_backtest_inference_ensemble(models, pool_bt)
    r = m.evaluate_daily_topn(d_ens, "greedy", n_per_day=m.DAILY_TOPN, print_detail=False,
                               min_reliable_n=m.MIN_RELIABLE_N, max_concurrent=m.MAX_CONCURRENT_POSITIONS)
    return r['pf'], r['n']


def train_configs_parallel(m, pss, configs, max_concurrent):
    """configs: dict[name] -> (tr, va, s_cols, m_cols)。全nameのjobをフラットな1つの
    並列プールにまとめて投げる(GPUに余裕があるため、候補間でも並列化する)。"""
    tmpdir = tempfile.mkdtemp(prefix="greedy_feature_search_")
    shared_data_path = os.path.join(tmpdir, "shared_data.pkl")
    shared = {}
    for name, (tr, va, s_cols, mc_cols) in configs.items():
        shared[f"tr_{name}"] = tr
        shared[f"va_{name}"] = va
        shared[f"s_cols_{name}"] = s_cols
        shared[f"m_cols_{name}"] = mc_cols
    with open(shared_data_path, "wb") as f:
        pickle.dump(shared, f)

    jobs, out_paths = [], {}
    for name in configs:
        for seed in m.SEEDS:
            out_path = os.path.join(tmpdir, f"state_{name}_{seed}.pt")
            jobs.append((name, seed, shared_data_path, out_path))
            out_paths[(name, seed)] = out_path

    script_path = os.path.abspath(pss.__file__)
    t0 = time.time()
    pss.run_jobs_parallel(jobs, script_path, max_concurrent)
    elapsed = time.time() - t0
    return out_paths, tmpdir, elapsed


def load_baseline_reference(m, pss, pool_base, tr0, va0, shared_split_date, max_concurrent):
    """既存のbaseline_model_cache機構をそのまま使う。キャッシュ済みならそれを読み込むだけ、
    無ければ学習してキャッシュに保存する(stage3_toggle_experiment.py::__main__と同じロジック)。"""
    import torch
    fp = m.baseline_cache_fingerprint(len(tr0[2]), len(va0[2]), shared_split_date)
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "baseline_model_cache", fp)
    os.makedirs(cache_dir, exist_ok=True)
    cached_seeds = {s for s in m.SEEDS if os.path.exists(os.path.join(cache_dir, f"state_{s}.pt"))}
    missing = [s for s in m.SEEDS if s not in cached_seeds]
    if cached_seeds:
        m.log(f"[greedy] baselineキャッシュ命中(fingerprint={fp}): seed={sorted(cached_seeds)}")
    if missing:
        m.log(f"[greedy] baseline未キャッシュのseed={missing}を学習します...")
        configs = {"base": (tr0, va0, m.BASELINE_STOCK_COLS, m.BASELINE_MACRO_COLS)}
        tmpdir = tempfile.mkdtemp(prefix="greedy_baseline_")
        shared_data_path = os.path.join(tmpdir, "shared_data.pkl")
        with open(shared_data_path, "wb") as f:
            pickle.dump({"tr_base": tr0, "va_base": va0, "s_cols_base": m.BASELINE_STOCK_COLS,
                         "m_cols_base": m.BASELINE_MACRO_COLS}, f)
        jobs = [("base", s, shared_data_path, os.path.join(tmpdir, f"state_{s}.pt")) for s in missing]
        pss.run_jobs_parallel(jobs, os.path.abspath(pss.__file__), max_concurrent)
        for s in missing:
            shutil.copy2(os.path.join(tmpdir, f"state_{s}.pt"), os.path.join(cache_dir, f"state_{s}.pt"))
        try:
            shutil.rmtree(tmpdir)
        except Exception:
            pass

    models = []
    for s in m.SEEDS:
        model = m.build_model(m.BASELINE_STOCK_COLS, m.BASELINE_MACRO_COLS)
        model.load_state_dict(torch.load(os.path.join(cache_dir, f"state_{s}.pt"), map_location=m.DEVICE))
        model.eval()
        models.append(model)
    pf, n = ensemble_daily_pf(m, models, pool_base)
    return models, pf, n


def run_greedy_search(margin=0.0, max_concurrent=10, candidates=None):
    import torch
    import pandas as pd
    import stage3_toggle_experiment as m
    import parallel_seed_sweep as pss

    t_start = time.time()
    m.log("[greedy] データプール構築(1回だけ)...")
    pool, margin_pool, sector_pool, macro_pool_df, loss_regime_labels, shared_split_date = build_shared_pools(m)

    tr0, va0, pool_base = build_config_dataset(m, m.BASELINE_STOCK_COLS, m.BASELINE_MACRO_COLS,
                                                pool, margin_pool, sector_pool, macro_pool_df,
                                                loss_regime_labels, shared_split_date)
    m.log(f"[greedy] baseline: train={len(tr0[2])} val={len(va0[2])} backtest_pool={len(pool_base)}")

    reference_models, reference_pf, reference_n = load_baseline_reference(
        m, pss, pool_base, tr0, va0, shared_split_date, max_concurrent)
    m.log(f"[greedy] round 0 (baseline): アンサンブル日次Top-{m.DAILY_TOPN} PF={reference_pf:.3f} (n={reference_n})")

    selected_stock = list(m.BASELINE_STOCK_COLS)
    selected_macro = list(m.BASELINE_MACRO_COLS)
    candidate_pool = candidates if candidates is not None else list(m.FEATURE_TOGGLES)
    remaining = [f for f in candidate_pool if f not in selected_stock and f not in selected_macro]
    m.log(f"[greedy] 候補プール: {len(remaining)}個 -> {remaining}")

    history = [dict(round=0, candidate="(baseline)", side=None, pf=reference_pf, reference_pf=reference_pf,
                     margin=margin, n_candidates=len(remaining), adopted=True,
                     selected_stock=str(selected_stock), selected_macro=str(selected_macro),
                     elapsed_sec=time.time() - t_start)]
    _save_history(history)

    round_num = 0
    while remaining:
        round_num += 1
        t_round = time.time()
        configs, meta = {}, {}
        for cand in remaining:
            side = m.FEATURE_CATALOG[cand]['side']
            if side == 'stock':
                s_cols, mc_cols = selected_stock + [cand], selected_macro
            else:
                s_cols, mc_cols = selected_stock, selected_macro + [cand]
            tr, va, pool_bt = build_config_dataset(m, s_cols, mc_cols, pool, margin_pool, sector_pool,
                                                    macro_pool_df, loss_regime_labels, shared_split_date)
            configs[cand] = (tr, va, s_cols, mc_cols)
            meta[cand] = (side, s_cols, mc_cols, pool_bt)

        m.log(f"[greedy] round {round_num}: {len(configs)}候補 x {len(m.SEEDS)}seed = "
              f"{len(configs) * len(m.SEEDS)}ジョブを並列学習します(最大{max_concurrent}並列)...")
        out_paths, tmpdir, train_elapsed = train_configs_parallel(m, pss, configs, max_concurrent)
        m.log(f"[greedy] round {round_num}: 学習完了 実時間={train_elapsed:.1f}秒")

        best_cand, best_pf, best_n, best_models = None, -1e18, None, None
        round_rows = []
        for cand in remaining:
            side, s_cols, mc_cols, pool_bt = meta[cand]
            models = []
            for seed in m.SEEDS:
                model = m.build_model(s_cols, mc_cols)
                model.load_state_dict(torch.load(out_paths[(cand, seed)], map_location=m.DEVICE))
                model.eval()
                models.append(model)
            pf, n = ensemble_daily_pf(m, models, pool_bt)
            m.log(f"    候補={cand}({side}): アンサンブルPF={pf:.3f} (n={n}) [参照={reference_pf:.3f}]")
            round_rows.append(dict(candidate=cand, side=side, pf=pf, n=n))
            if pf > best_pf:
                if best_models is not None:
                    del best_models
                best_pf, best_cand, best_n, best_models = pf, cand, n, models
            else:
                del models
            torch.cuda.empty_cache()

        adopted = best_pf > reference_pf + margin
        history.append(dict(round=round_num, candidate=best_cand,
                             side=meta[best_cand][0], pf=best_pf, reference_pf=reference_pf, margin=margin,
                             n_candidates=len(remaining), adopted=adopted,
                             selected_stock=str(meta[best_cand][1]), selected_macro=str(meta[best_cand][2]),
                             elapsed_sec=time.time() - t_start))
        _save_history(history)

        if adopted:
            m.log(f"[greedy] round {round_num}: 採用 -> {best_cand} "
                  f"(PF {reference_pf:.3f} -> {best_pf:.3f}, +{best_pf - reference_pf:.3f})")
            for old_model in reference_models:
                del old_model
            torch.cuda.empty_cache()
            reference_models, reference_pf, reference_n = best_models, best_pf, best_n
            selected_stock, selected_macro = meta[best_cand][1], meta[best_cand][2]
            remaining.remove(best_cand)
        else:
            m.log(f"[greedy] round {round_num}: 改善候補なし(最良={best_cand} PF={best_pf:.3f} "
                  f"<= 参照PF={reference_pf:.3f}+margin({margin:.3f}))。探索終了")
            del best_models
            torch.cuda.empty_cache()
            try:
                shutil.rmtree(tmpdir)
            except Exception:
                pass
            break

        try:
            shutil.rmtree(tmpdir)
        except Exception:
            pass

    total_elapsed = time.time() - t_start
    m.log("=" * 90)
    m.log(f"[greedy] 探索終了: 最終選択集合(stock={len(selected_stock)}, macro={len(selected_macro)})")
    m.log(f"[greedy]   stock={selected_stock}")
    m.log(f"[greedy]   macro={selected_macro}")
    m.log(f"[greedy]   最終アンサンブル日次Top-{m.DAILY_TOPN} PF={reference_pf:.3f} "
          f"(baseline={history[0]['pf']:.3f}からの改善={reference_pf - history[0]['pf']:.3f})")
    m.log(f"[greedy]   総実時間={total_elapsed:.1f}秒 ({total_elapsed/60:.1f}分)")
    return dict(selected_stock=selected_stock, selected_macro=selected_macro,
                final_pf=reference_pf, baseline_pf=history[0]['pf'], history=history)


def _save_history(history):
    import pandas as pd
    pd.DataFrame(history).to_parquet(RESULT_LOG_PATH, index=False)


if __name__ == "__main__":
    margin_arg, max_concurrent_arg, candidates_arg = 0.0, 10, None
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--margin":
            margin_arg = float(args[i + 1]); i += 2
        elif args[i] == "--max-concurrent":
            max_concurrent_arg = int(args[i + 1]); i += 2
        elif args[i] == "--candidates":
            candidates_arg = args[i + 1].split(","); i += 2
        else:
            i += 1
    run_greedy_search(margin=margin_arg, max_concurrent=max_concurrent_arg, candidates=candidates_arg)
