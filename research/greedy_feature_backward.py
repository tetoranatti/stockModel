"""後退法(backward elimination)による特徴量探索(2026-09-20追加、ユーザー提案「まずベースから
減らしていって確認するのはどうかな」)。

BARRIER_MODE="trailing"採用によりラベル定義が大きく変わったため、旧ラベル(固定バリア)で
確立されたBASELINE_STOCK_COLS/BASELINE_MACRO_COLS(11+5=16特徴量)が新ラベルの下でも
全部必要とは限らない。greedy_feature_search.py(前進法、baselineに候補を1個ずつ足す)の
鏡像として、baselineから特徴量を1個ずつ引いて、外した方が良くなる特徴量が無くなるまで
続ける。

アルゴリズム:
    1. 選択集合 = BASELINE_STOCK_COLS + BASELINE_MACRO_COLS(16個)から開始。
       参照PFは、この全16特徴量構成を新ラベルの下でフレッシュに学習したもの
       (旧ラベル用のbaseline_model_cacheは再利用できない——フィンガープリントに
       barrier_mode等が含まれるため自動的に別キャッシュになる)。
    2. 各ラウンドで、選択集合から残っている特徴量を1個ずつ抜いた構成を全部並列学習・評価する。
    3. 最もPFが高かった「1個抜いた構成」が参照PFを上回っていれば、その特徴量を
       選択集合から恒久的に除外して次ラウンドへ進む(そのラウンドで学習済みのモデルを
       そのまま次の参照として使い回す)。誰も参照PFを上回らなければ探索終了。
    4. 片側(stock/macro)がゼロになる除外は安全のためスキップする。

greedy_feature_search.pyのbuild_shared_pools/build_config_dataset/train_configs_parallel/
ensemble_daily_pfをそのまま再利用する。

使い方:
    python greedy_feature_backward.py
    python greedy_feature_backward.py --margin 0.01
"""
import sys
import os
import time
import pickle
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import greedy_feature_search as g

RESULT_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "greedy_backward_log.parquet")


def run_backward_elimination(margin=0.0, max_concurrent=10,
                              build_dataset_fn=None, prepare_backtest_pool_fn=None,
                              result_log_path=None):
    import torch
    import pandas as pd
    import stage3_exp.experiment_context as m
    import parallel_seed_sweep as pss

    result_log_path = result_log_path or RESULT_LOG_PATH

    t_start = time.time()
    m.log("[backward] データプール構築(1回だけ)...")
    pool, margin_pool, sector_pool, macro_pool_df, loss_regime_labels, shared_split_date = g.build_shared_pools(m)

    selected_stock = list(m.BASELINE_STOCK_COLS)
    selected_macro = list(m.BASELINE_MACRO_COLS)

    # round 0: 現在の全16特徴量構成を、現行ラベル設定(BARRIER_MODE等)でフレッシュに学習する
    # (旧ラベルで作られたbaseline_model_cacheはfingerprintが変わっているので自動的に別物になる)。
    tr0, va0, pool0 = g.build_config_dataset(m, selected_stock, selected_macro, pool, margin_pool, sector_pool,
                                              macro_pool_df, loss_regime_labels, shared_split_date,
                                              build_dataset_fn, prepare_backtest_pool_fn)
    m.log(f"[backward] 全{len(selected_stock)+len(selected_macro)}特徴量構成: "
          f"train={len(tr0[2])} val={len(va0[2])} pool={len(pool0)}")

    out_paths, tmpdir, elapsed = g.train_configs_parallel(
        m, pss, {"full": (tr0, va0, selected_stock, selected_macro)}, max_concurrent)
    m.log(f"[backward] round 0 (全特徴量): 学習完了 実時間={elapsed:.1f}秒")

    import torch as _torch
    reference_models = []
    for seed in m.SEEDS:
        model = m.build_model(selected_stock, selected_macro)
        model.load_state_dict(_torch.load(out_paths[("full", seed)], map_location=m.DEVICE))
        model.eval()
        reference_models.append(model)
    reference_pf, reference_n = g.ensemble_daily_pf(m, reference_models, pool0)
    try:
        shutil.rmtree(tmpdir)
    except Exception:
        pass
    m.log(f"[backward] round 0 (全特徴量, 参照): アンサンブル日次Top-{m.DAILY_TOPN} PF={reference_pf:.3f} (n={reference_n})")

    history = [dict(round=0, removed="(none)", side=None, pf=reference_pf, reference_pf=reference_pf,
                     margin=margin, n_candidates=len(selected_stock) + len(selected_macro), adopted=True,
                     selected_stock=str(selected_stock), selected_macro=str(selected_macro),
                     elapsed_sec=time.time() - t_start)]
    _save_history(history, result_log_path)

    round_num = 0
    while True:
        candidates = list(selected_stock) + list(selected_macro)
        if len(candidates) <= 1:
            m.log("[backward] 残り候補が1個以下になったため探索終了")
            break
        round_num += 1
        t_round = time.time()
        configs, meta = {}, {}
        for feat in candidates:
            if feat in selected_stock:
                s_cols = [f for f in selected_stock if f != feat]
                mc_cols = list(selected_macro)
                side = 'stock'
            else:
                s_cols = list(selected_stock)
                mc_cols = [f for f in selected_macro if f != feat]
                side = 'macro'
            if not s_cols or not mc_cols:
                m.log(f"[backward] {feat}({side})を外すと片側がゼロになるためスキップ")
                continue
            tr, va, pool_bt = g.build_config_dataset(m, s_cols, mc_cols, pool, margin_pool, sector_pool,
                                                       macro_pool_df, loss_regime_labels, shared_split_date,
                                                       build_dataset_fn, prepare_backtest_pool_fn)
            configs[feat] = (tr, va, s_cols, mc_cols)
            meta[feat] = (side, s_cols, mc_cols, pool_bt)

        if not configs:
            m.log("[backward] このラウンドで除外可能な候補が無いため探索終了")
            break

        m.log(f"[backward] round {round_num}: 残り{len(candidates)}特徴量のうち{len(configs)}個を1個ずつ除外した"
              f"構成 x {len(m.SEEDS)}seed = {len(configs) * len(m.SEEDS)}ジョブを並列学習します"
              f"(最大{max_concurrent}並列)...")
        out_paths, tmpdir, train_elapsed = g.train_configs_parallel(m, pss, configs, max_concurrent)
        m.log(f"[backward] round {round_num}: 学習完了 実時間={train_elapsed:.1f}秒")

        best_feat, best_pf, best_n, best_models = None, -1e18, None, None
        for feat in configs:
            side, s_cols, mc_cols, pool_bt = meta[feat]
            models = []
            for seed in m.SEEDS:
                model = m.build_model(s_cols, mc_cols)
                model.load_state_dict(_torch.load(out_paths[(feat, seed)], map_location=m.DEVICE))
                model.eval()
                models.append(model)
            pf, n = g.ensemble_daily_pf(m, models, pool_bt)
            m.log(f"    {feat}({side})を除外: アンサンブルPF={pf:.3f} (n={n}) [参照={reference_pf:.3f}]")
            if pf > best_pf:
                if best_models is not None:
                    del best_models
                best_pf, best_feat, best_n, best_models = pf, feat, n, models
            else:
                del models
            _torch.cuda.empty_cache()

        adopted = best_pf > reference_pf + margin
        history.append(dict(round=round_num, removed=best_feat, side=meta[best_feat][0], pf=best_pf,
                             reference_pf=reference_pf, margin=margin, n_candidates=len(configs), adopted=adopted,
                             selected_stock=str(meta[best_feat][1]), selected_macro=str(meta[best_feat][2]),
                             elapsed_sec=time.time() - t_start))
        _save_history(history, result_log_path)

        if adopted:
            m.log(f"[backward] round {round_num}: {best_feat}を除外 (PF {reference_pf:.3f} -> {best_pf:.3f}, "
                  f"+{best_pf - reference_pf:.3f})")
            for old_model in reference_models:
                del old_model
            _torch.cuda.empty_cache()
            reference_models, reference_pf, reference_n = best_models, best_pf, best_n
            selected_stock, selected_macro = meta[best_feat][1], meta[best_feat][2]
        else:
            m.log(f"[backward] round {round_num}: 除外しても改善する特徴量なし(最良={best_feat} "
                  f"PF={best_pf:.3f} <= 参照PF={reference_pf:.3f}+margin({margin:.3f}))。探索終了")
            del best_models
            _torch.cuda.empty_cache()
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
    m.log(f"[backward] 探索終了: 最終選択集合(stock={len(selected_stock)}, macro={len(selected_macro)})")
    m.log(f"[backward]   stock={selected_stock}")
    m.log(f"[backward]   macro={selected_macro}")
    m.log(f"[backward]   最終アンサンブル日次Top-{m.DAILY_TOPN} PF={reference_pf:.3f} "
          f"(全特徴量={history[0]['pf']:.3f}からの変化={reference_pf - history[0]['pf']:.3f})")
    m.log(f"[backward]   総実時間={total_elapsed:.1f}秒 ({total_elapsed/60:.1f}分)")
    return dict(selected_stock=selected_stock, selected_macro=selected_macro,
                final_pf=reference_pf, full_pf=history[0]['pf'], history=history)


def _save_history(history, result_log_path=None):
    import pandas as pd
    pd.DataFrame(history).to_parquet(result_log_path or RESULT_LOG_PATH, index=False)


if __name__ == "__main__":
    margin_arg, max_concurrent_arg = 0.0, 10
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--margin":
            margin_arg = float(args[i + 1]); i += 2
        elif args[i] == "--max-concurrent":
            max_concurrent_arg = int(args[i + 1]); i += 2
        else:
            i += 1
    run_backward_elimination(margin=margin_arg, max_concurrent=max_concurrent_arg)
