"""複数seedを並列プロセスで学習・評価する汎用ランナー(2026-09-19追加、同日中にユーザー指摘で
6点改良)。

これまでこのセッションで多用してきた「1スクリプト内でseedをforループして逐次学習」する
diag_*_variance.py系パターンの代替。並列化の実現性調査(3seed: 139s→57s=2.44x、
5seed: 推定230s→77s=約2.9倍、逐次/並列で結果が完全一致することを確認済み、
[[feedback_nn_training_performance]]参照)を受けて、実際に使える汎用ツールとしてresearch/
直下に置く(スクラッチパッドではなく恒久的なツールとして)。

設計(初版からの改良点、6項目):
    1. データ構築を親プロセスで1回だけ行い(_build_data)、全workerで共有する
       (以前は各workerが独立に再構築しており、5workerで5回冗長にキャッシュ読込+
       データセット構築をしていた)。
    2. 同時起動するworker数をmax_concurrentで制限する(ローリングプール方式——
       枠が空き次第すぐ次のseedを起動する。全部一度に起動してGPUメモリ/プロセス数を
       圧迫しないようにする)。
    3. workerはtrain_model()のみを実行し、学習済みstate_dictだけを返す(データ構築も
       推論もしない)。推論・アンサンブル評価は親プロセスが全workerの学習完了後に
       一括で行う——バッチ推論は既に十分高速なので、ここを並列化する意味は薄い
       ([[feedback_nn_training_performance]]参照)。
    4. スイープ結果を`sweep_results_log.parquet`に追記保存し、このセッション/今後の
       セッションを通じた実験履歴として蓄積する。
    5. 全workerが同じ(親プロセスで1回だけ構築した)pool_baseを使うため、
       run_backtest_inference_ensemble()をそのまま複数モデルに渡せる——以前のような
       (ticker,date)キーでのDataFrame mergeによるp_win/p_stop平均化が不要になった。
    6. 全seedの推論・アンサンブル評価が終わった後、モデルを明示的に破棄し
       torch.cuda.empty_cache()でGPUメモリを解放する。

設定(MODEL_ARCH/ENTRY_CONVENTION/MAX_PAIR_GAP/HOLDING_PERIOD/SEQ_LEN/TRAIN_BATCH_SIZE等)は
すべてstage3_toggle_experiment.pyの現在の値をそのまま使う——このスクリプトは設定を
一切上書きしない。特定の設定で比較したい場合は、呼び出し側でstage3_toggle_experiment.py
自体のトグルを変えてから実行する。

使い方:
    python parallel_seed_sweep.py                       # SEEDS=[42,43,44,45,46]、最大5並列
    python parallel_seed_sweep.py --seeds 42 43 44        # seedsを明示指定
    python parallel_seed_sweep.py --max-concurrent 3      # 同時起動数を変更
    python parallel_seed_sweep.py --worker-train <seed> <shared_data_path> <out_state_path>  # 内部ワーカーモード
"""
import sys
import os
import time
import pickle
import subprocess

# ============================================================
# 複数worker実行時のCPU過剰並列を防止
# NumPy / PyTorch / pandas のimport前に設定する
# ============================================================
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_WAIT_POLICY"] = "PASSIVE"

try:
    # バックグラウンド実行時、リダイレクト先がcp932/Shift-JISにfall backして日本語print文が
    # 文字化けするのを防ぐ(stage3_toggle_experiment.pyと同じ対策)。
    sys.stdout.reconfigure(line_buffering=True, encoding='utf-8')
    sys.stderr.reconfigure(line_buffering=True, encoding='utf-8')
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

WORKER_FLAG = "--worker-train"
DEFAULT_SEEDS = [42, 43, 44, 45, 46]
DEFAULT_MAX_CONCURRENT = 5
RESULTS_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sweep_results_log.parquet")


def _build_data():
    """stage3_toggle_experiment.pyの現在の設定でbaseline(既存14固定)のdataset/backtest_poolを
    構築する。親プロセスが1回だけ呼ぶ(改良点1: 以前はworkerごとに冗長に呼んでいた)。"""
    import stage3_toggle_experiment as m

    with open(m.UNIVERSE_PATH, "r", encoding="utf-8") as f:
        tickers = [line.strip() for line in f if line.strip()]

    macro_df = m.load_macro_slim5(return_source="nk225")
    pool = m.get_full_feature_pool_df(tickers, macro_df)
    margin_pool = m.get_margin_pool_df(tickers, macro_df)
    sector_pool = m.get_sector_relative_pool_df(tickers, macro_df)
    macro_pool_df = m.augment_macro_pool_df(m.get_full_macro_pool_df())

    loss_regime_labels = m.compute_regime_labels_expanding(macro_pool_df) if m.USE_REGIME_AWARE_LOSS else None

    _base_per_ticker = m.build_per_ticker(pool, margin_pool, sector_pool, macro_pool_df, m.BASELINE_STOCK_COLS)
    _base_valid_dates = m.valid_cross_section_dates(_base_per_ticker, m.BASELINE_STOCK_COLS, m.MIN_CROSS_SECTION)
    shared_split_date = m.compute_global_split_date(_base_valid_dates, None)

    tr0, va0 = m.build_dataset(m.BASELINE_MACRO_COLS, m.BASELINE_STOCK_COLS, pool, margin_pool, sector_pool,
                                macro_pool_df, regime_filter=None, regime_labels_for_loss=loss_regime_labels,
                                global_split_date_override=shared_split_date)
    pool_base = m.prepare_backtest_pool(m.BASELINE_MACRO_COLS, m.BASELINE_STOCK_COLS, pool, margin_pool,
                                         sector_pool, macro_pool_df, regime_filter=None,
                                         global_split_date_override=shared_split_date)
    return m, tr0, va0, pool_base


def worker_train_only(config_label, seed, shared_data_path, out_state_path):
    """workerプロセス: train_model()のみを実行し、学習済みstate_dictを保存する
    (改良点3: データ構築・推論はしない、親プロセスが用意した共有データを読むだけ)。

    config_label: 共有データpickle内のキー接尾辞(例: "base"なら tr_base/va_base/
    s_cols_base/m_cols_base を読む)。stage3_toggle_experiment.py::__main__のように
    base/test複数構成を同時に並列学習したい場合に使う(2026-09-19、ユーザー提案
    「本体スクリプトのbaseline vs テスト構成比較も並列化したい」)。単一構成のみの
    従来のrun_parallel_sweep()はconfig_label="base"固定で呼ぶ。"""
    import torch
    # 1プロセスあたりのPyTorch CPUスレッドを制限する。
    # workerを複数並列にするため、worker内部の多重並列は原則不要。
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # 初期化後に呼ばれた場合に備える。
        pass

    import stage3_toggle_experiment as m

    with open(shared_data_path, "rb") as f:
        shared = pickle.load(f)
    tr0 = shared[f"tr_{config_label}"]
    va0 = shared[f"va_{config_label}"]
    s_cols = shared[f"s_cols_{config_label}"]
    m_cols = shared[f"m_cols_{config_label}"]

    t0 = time.time()
    model = m.train_model(seed, tr0, va0, s_cols, m_cols, f"parallel_sweep({config_label},seed={seed})")
    elapsed = time.time() - t0
    torch.save(model.state_dict(), out_state_path)
    print(f"[worker {config_label} seed={seed}] train_model elapsed={elapsed:.1f}s", flush=True)

    # 改良点6: workerプロセス終了時にも明示的に解放しておく(プロセス終了自体でも
    # OS側は回収するが、同一プロセスが複数モデルを扱う設計に将来変わっても安全なように)。
    del model
    torch.cuda.empty_cache()


def run_jobs_parallel(jobs, script_path, max_concurrent):
    """改良点2: ローリングプール方式で同時起動数をmax_concurrentに制限する
    (枠が空き次第、すぐ次のジョブを起動する——固定バッチ単位で待つより効率的)。

    jobs: [(config_label, seed, shared_data_path, out_state_path), ...] のリスト。
    stage3_toggle_experiment.py::__main__からも直接importして呼べる汎用関数
    (base/test×seedsを1個のフラットなjobリストとして渡せば、両方まとめて並列学習できる)。"""
    pending = list(jobs)
    running = []  # [((config_label, seed), proc), ...]
    while pending or running:
        while pending and len(running) < max_concurrent:
            config_label, seed, shared_data_path, out_state_path = pending.pop(0)
            proc = subprocess.Popen([sys.executable, script_path, WORKER_FLAG, config_label, str(seed),
                                      shared_data_path, out_state_path])
            running.append(((config_label, seed), proc))
        still_running = []
        for key, proc in running:
            ret = proc.poll()
            if ret is None:
                still_running.append((key, proc))
            elif ret != 0:
                raise RuntimeError(f"{key}のworkerプロセスが異常終了しました(exit code {ret})")
        if len(still_running) == len(running):
            time.sleep(0.5)  # 誰も完了していなければ少し待つ(ビジーループ回避)
        running = still_running


def run_parallel_sweep(seeds=None, max_concurrent=DEFAULT_MAX_CONCURRENT):
    seeds = seeds or DEFAULT_SEEDS
    script_path = os.path.abspath(__file__)

    print(f"[*] データ構築(親プロセスで1回だけ、全worker共有)...")
    m, tr0, va0, pool_base = _build_data()
    print(f"[+] train={len(tr0[2])} val={len(va0[2])} backtest_pool={len(pool_base)}")

    import tempfile
    tmpdir = tempfile.mkdtemp(prefix="parallel_seed_sweep_")
    shared_data_path = os.path.join(tmpdir, "shared_data.pkl")
    with open(shared_data_path, "wb") as f:
        pickle.dump({"tr_base": tr0, "va_base": va0, "s_cols_base": m.BASELINE_STOCK_COLS,
                     "m_cols_base": m.BASELINE_MACRO_COLS}, f)

    seed_and_paths = [(s, shared_data_path, os.path.join(tmpdir, f"state_{s}.pt")) for s in seeds]
    jobs = [("base", s, shared_data_path, out_path) for s, _, out_path in seed_and_paths]

    print(f"[*] seeds={seeds} を最大{max_concurrent}並列で学習します(train_modelのみ並列化)...")
    t0 = time.time()
    run_jobs_parallel(jobs, script_path, max_concurrent)
    train_elapsed = time.time() - t0
    print(f"[*] 全{len(seeds)}seed学習完了: 実時間={train_elapsed:.1f}秒\n")

    # --- 親プロセスで推論・アンサンブル評価を一括実行(改良点3・5) ---
    import torch
    import numpy as np
    import pandas as pd

    models = []
    daily_pfs = []
    for seed, _, out_state_path in seed_and_paths:
        model = m.build_model(m.BASELINE_STOCK_COLS, m.BASELINE_MACRO_COLS)
        model.load_state_dict(torch.load(out_state_path, map_location=m.DEVICE))
        d = m.run_backtest_inference(model, pool_base)
        r_daily = m.evaluate_daily_topn(d, f"seed={seed}", n_per_day=m.DAILY_TOPN, print_detail=False,
                                         min_reliable_n=m.MIN_RELIABLE_N, max_concurrent=m.MAX_CONCURRENT_POSITIONS)
        print(f"[*] seed={seed}: 日次Top-{m.DAILY_TOPN} PF={r_daily['pf']:.3f}(n={r_daily['n']})")
        daily_pfs.append(r_daily['pf'])
        models.append(model)

    daily_pfs = np.array(daily_pfs)
    print(f"\n[*] 日次Top-{m.DAILY_TOPN}: seed別PF={list(np.round(daily_pfs, 3))}")
    print(f"    mean={daily_pfs.mean():.3f} std={(daily_pfs.std(ddof=1) if len(daily_pfs) > 1 else 0.0):.3f}")

    # 改良点5: 全workerが同じpool_baseを使っているので、mergeせずそのまま
    # run_backtest_inference_ensemble()に複数モデルを渡せる(本番run_dynamic_regime_screening_v8.py
    # と同じsoftmax確率平均ロジック)。
    d_ens = m.run_backtest_inference_ensemble(models, pool_base)
    r_ens = m.evaluate_daily_topn(d_ens, "アンサンブル", n_per_day=m.DAILY_TOPN, print_detail=False,
                                   min_reliable_n=m.MIN_RELIABLE_N, max_concurrent=m.MAX_CONCURRENT_POSITIONS)
    print(f"[*] アンサンブル({len(seeds)}seed平均) 日次Top-{m.DAILY_TOPN} PF={r_ens['pf']:.3f}(n={r_ens['n']})")

    total_elapsed = time.time() - t0
    print(f"\n[*] 学習実時間={train_elapsed:.1f}秒 + 推論/評価={total_elapsed - train_elapsed:.1f}秒 "
          f"= 合計{total_elapsed:.1f}秒")

    # 改良点6: 全評価完了後にモデルを破棄してGPUメモリを解放
    del models
    torch.cuda.empty_cache()

    # 改良点4: 結果をparquetに追記保存(実験履歴として蓄積、今後の比較に使う)
    row = {
        "timestamp": pd.Timestamp.now(), "seeds": str(seeds), "n_seeds": len(seeds),
        "model_arch": m.MODEL_ARCH, "entry_convention": m.ENTRY_CONVENTION,
        "max_pair_gap": m.MAX_PAIR_GAP, "holding_period": m.HOLDING_PERIOD, "seq_len": m.SEQ_LEN,
        "train_batch_size": m.TRAIN_BATCH_SIZE, "use_regime_aware_loss": m.USE_REGIME_AWARE_LOSS,
        "label_mode": m.LABEL_MODE,
        "daily_pf_mean": float(daily_pfs.mean()),
        "daily_pf_std": float(daily_pfs.std(ddof=1)) if len(daily_pfs) > 1 else 0.0,
        "ensemble_pf": float(r_ens['pf']),
        "train_elapsed_sec": train_elapsed, "total_elapsed_sec": total_elapsed,
    }
    new_row_df = pd.DataFrame([row])
    if os.path.exists(RESULTS_LOG_PATH):
        combined = pd.concat([pd.read_parquet(RESULTS_LOG_PATH), new_row_df], ignore_index=True)
    else:
        combined = new_row_df
    combined.to_parquet(RESULTS_LOG_PATH, index=False)
    print(f"[*] 結果を{RESULTS_LOG_PATH}に追記しました(累計{len(combined)}件)")

    try:
        import shutil
        shutil.rmtree(tmpdir)
    except Exception:
        pass

    return dict(daily_pfs=daily_pfs, ensemble_pf=r_ens['pf'], train_elapsed=train_elapsed,
                total_elapsed=total_elapsed)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == WORKER_FLAG:
        worker_train_only(sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5])
    else:
        seeds_arg, max_concurrent_arg = None, DEFAULT_MAX_CONCURRENT
        args = sys.argv[1:]
        i = 0
        while i < len(args):
            if args[i] == "--seeds":
                seeds_arg = []
                i += 1
                while i < len(args) and not args[i].startswith("--"):
                    seeds_arg.append(int(args[i]))
                    i += 1
            elif args[i] == "--max-concurrent":
                max_concurrent_arg = int(args[i + 1])
                i += 2
            else:
                i += 1
        run_parallel_sweep(seeds_arg, max_concurrent_arg)
