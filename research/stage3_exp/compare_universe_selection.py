"""150銘柄ユニバースの選び方(現行 vs クラスタ均等)を比較する実験
(2026-09-23追加、ユーザー提案)。

既存の_feature_cache_utils.pyのキャッシュ(_cache_base_per_ticker_df.pkl等)や
train_universe_bars.parquetは現行150銘柄前提で他の実験からも参照されているため、
汚染しないよう本スクリプトはプロセス内だけでキャッシュパス定数を書き換えて
(モジュール属性の一時的な上書き)別ファイルに退避させる。本番コード・既存キャッシュは
一切変更しない。

学習はparallel_seed_sweep.pyの部品(_build_data相当・run_jobs_parallel)を再利用し、
現行のBASELINE_STOCK_COLS/BASELINE_MACRO_COLS(トグル変更なし)のままクラスタ均等
ユニバースで5seed学習・アンサンブル評価する。業種制約/クラスタ制約両方のTOPN別PFを
出力し、既存150銘柄での実験結果(このセッションの直前の実行ログ)と比較できるようにする。

実行方法:
    python research/stage3_exp/compare_universe_selection.py
"""
import os
import pickle
import sys
import tempfile
import time

sys.path.insert(0, r"F:\stockModel")
sys.path.insert(0, r"F:\stockModel\research")

try:
    sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")
    sys.stderr.reconfigure(line_buffering=True, encoding="utf-8")
except Exception:
    pass

import pandas as pd

BASE_DIR = r"F:\stockModel"
CACHE_DIR = os.path.join(BASE_DIR, "data", "cache")
RESEARCH_CACHE_DIR = os.path.join(BASE_DIR, "research", "cache")
CLUSTERED_UNIVERSE_PATH = os.path.join(BASE_DIR, "universe_150_tickers_clustered.txt")
CLUSTERED_BARS_PATH = os.path.join(CACHE_DIR, "train_universe_bars_clustered150.parquet")
LONG_HISTORY_CACHE_PATH = os.path.join(CACHE_DIR, "cluster_daily_bars_raw.parquet")
CACHE_TAG = "clustered150"


def _rebuild_clustered_universe_list():
    """現在のクラスタ設定(config.pyのN_CLUSTERS等)でクラスタ均等150銘柄を選び直し、
    universe_150_tickers_clustered.txtへ書き出す(前回生成分は古いクラスタ設定の
    可能性があるため、比較実験の直前に必ず再生成する)。"""
    from stage3_exp.cluster_utils import select_universe_evenly

    tickers, cluster_map = select_universe_evenly(n_target=150)
    with open(CLUSTERED_UNIVERSE_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(tickers) + "\n")
    counts = pd.Series(cluster_map).loc[tickers].value_counts().sort_index()
    print(f"[*] クラスタ均等150銘柄を再生成しました: {CLUSTERED_UNIVERSE_PATH}")
    print(counts.to_string())
    return tickers


def _build_clustered_universe_bars(tickers):
    """cluster_daily_bars_raw.parquet(長期・分割調整済み・市場全体)から指定銘柄だけを
    抜き出し、train_universe_bars.parquetと同じワイド形式(MultiIndex columns=
    (ticker, field))で保存する。"""
    long_df = pd.read_parquet(LONG_HISTORY_CACHE_PATH)
    long_df = long_df[long_df["ticker"].isin(tickers)].copy()
    long_df["Date"] = pd.to_datetime(long_df["Date"])

    missing = sorted(set(tickers) - set(long_df["ticker"].unique()))
    if missing:
        print(f"[!] 長期キャッシュに無い銘柄({len(missing)}件、除外): {missing}")

    wide = long_df.set_index(["Date", "ticker"]).unstack("ticker")
    wide = wide.swaplevel(0, 1, axis=1).sort_index(axis=1)
    wide.to_parquet(CLUSTERED_BARS_PATH)
    print(
        f"[+] クラスタ均等150銘柄用の価格ワイドキャッシュ保存: {CLUSTERED_BARS_PATH} "
        f"({wide.columns.get_level_values(0).nunique()}銘柄, "
        f"{wide.index.min().date()}〜{wide.index.max().date()})"
    )


def _patch_cache_paths(fcu):
    """_feature_cache_utils側のキャッシュパス定数をこのプロセス内だけ差し替える
    (既存の現行150銘柄用キャッシュ・train_universe_bars.parquetは変更しない)。"""
    fcu.UNIVERSE_BARS_CACHE_PATH = CLUSTERED_BARS_PATH
    fcu.SCRATCH_CACHE_PATH = os.path.join(
        RESEARCH_CACHE_DIR, f"_cache_base_per_ticker_df_{CACHE_TAG}.pkl"
    )
    fcu.POOL_CACHE_PATH = os.path.join(
        RESEARCH_CACHE_DIR, f"_cache_full_feature_pool_{CACHE_TAG}.pkl"
    )
    fcu.MACRO_POOL_CACHE_PATH = os.path.join(
        RESEARCH_CACHE_DIR, f"_cache_full_macro_pool_{CACHE_TAG}.pkl"
    )
    fcu.SECTOR_POOL_CACHE_PATH = os.path.join(
        RESEARCH_CACHE_DIR, f"_cache_sector_relative_pool_{CACHE_TAG}.pkl"
    )
    fcu.MARGIN_POOL_CACHE_PATH = os.path.join(
        RESEARCH_CACHE_DIR, f"_cache_margin_pool_{CACHE_TAG}.pkl"
    )


def main():
    tickers = _rebuild_clustered_universe_list()
    _build_clustered_universe_bars(tickers)

    import _feature_cache_utils as fcu
    _patch_cache_paths(fcu)

    import stage3_exp.experiment_context as m
    m.UNIVERSE_PATH = CLUSTERED_UNIVERSE_PATH  # _build_data()はm.UNIVERSE_PATHを直接読む

    import parallel_seed_sweep as pss

    print("[*] データ構築(クラスタ均等150銘柄ユニバース、force_rebuild=Trueでキャッシュ新規生成)...")
    with open(m.UNIVERSE_PATH, "r", encoding="utf-8") as f:
        tickers = [line.strip() for line in f if line.strip()]

    macro_df = m.load_macro_slim5(return_source="nk225")
    pool = m.get_full_feature_pool_df(tickers, macro_df, force_rebuild=True)
    margin_pool = m.get_margin_pool_df(tickers, macro_df, force_rebuild=True)
    sector_pool = m.get_sector_relative_pool_df(tickers, macro_df, force_rebuild=True)
    macro_pool_df = m.augment_macro_pool_df(
        m.get_full_macro_pool_df(tickers, force_rebuild=True)
    )

    loss_regime_labels = (
        m.compute_regime_labels_expanding(macro_pool_df) if m.USE_REGIME_AWARE_LOSS else None
    )

    _base_per_ticker = m.build_per_ticker(
        pool, margin_pool, sector_pool, macro_pool_df, m.BASELINE_STOCK_COLS
    )
    _base_valid_dates = m.valid_cross_section_dates(
        _base_per_ticker, m.BASELINE_STOCK_COLS, m.MIN_CROSS_SECTION
    )
    shared_split_date = m.compute_global_split_date(_base_valid_dates, None)
    print(f"[*] split date: {shared_split_date.date()}")

    tr0, va0 = m.build_dataset(
        m.BASELINE_MACRO_COLS, m.BASELINE_STOCK_COLS, pool, margin_pool, sector_pool,
        macro_pool_df, regime_filter=None, regime_labels_for_loss=loss_regime_labels,
        global_split_date_override=shared_split_date,
    )
    close_price_lookup = {}
    pool_base = m.prepare_backtest_pool(
        m.BASELINE_MACRO_COLS, m.BASELINE_STOCK_COLS, pool, margin_pool, sector_pool,
        macro_pool_df, regime_filter=None, global_split_date_override=shared_split_date,
        close_price_out=close_price_lookup,
    )
    print(f"[+] train={len(tr0[2])} val={len(va0[2])} backtest_pool={len(pool_base)}")

    tmpdir = tempfile.mkdtemp(prefix="compare_universe_")
    shared_data_path = os.path.join(tmpdir, "shared_data.pkl")
    pss.dump_shared_training_data(
        shared_data_path,
        {
            "tr_base": tr0, "va_base": va0,
            "s_cols_base": m.BASELINE_STOCK_COLS, "m_cols_base": m.BASELINE_MACRO_COLS,
        },
    )

    seeds = m.SEEDS
    seed_and_paths = [
        (s, os.path.join(tmpdir, f"state_{s}.pt")) for s in seeds
    ]
    jobs = [("base", s, shared_data_path, out_path) for s, out_path in seed_and_paths]

    print(f"[*] seeds={seeds} を最大{m.PARALLEL_MAX_CONCURRENT}並列で学習します...")
    t0 = time.time()
    pss.run_jobs_parallel(jobs, os.path.abspath(pss.__file__), m.PARALLEL_MAX_CONCURRENT)
    print(f"[+] 全{len(seeds)}seed学習完了: 実時間={time.time() - t0:.1f}秒")

    import torch
    models = []
    for seed, out_path in seed_and_paths:
        model = m.build_model(m.BASELINE_STOCK_COLS, m.BASELINE_MACRO_COLS)
        model.load_state_dict(torch.load(out_path, map_location=m.DEVICE))
        models.append(model)

    d_ens = m.run_backtest_inference_ensemble(models, pool_base)

    r_ens = m.evaluate(
        d_ens, f"クラスタ均等150銘柄(アンサンブル{len(seeds)}seed) score Top-K",
        min_reliable_n=m.MIN_RELIABLE_N, common_sample_k=m.COMMON_SAMPLE_K,
        max_concurrent=m.MAX_CONCURRENT_POSITIONS, close_price_lookup=close_price_lookup,
    )

    print("\n" + "=" * 90)
    print(f"【TOPN別PF分析(業種制約 max_per_sector={m.MAX_PER_SECTOR}) - クラスタ均等150銘柄ユニバース】")
    print("=" * 90)
    topn_sector_df = m.evaluate_topn_curve(
        d_ens, topn_list=[1, 3, 5, 10, 20, 30], max_per_sector=m.MAX_PER_SECTOR,
        close_price_lookup=close_price_lookup,
    )
    print(topn_sector_df.to_string(index=False))

    print("\n" + "=" * 90)
    print(f"【TOPN別PF分析(クラスタ制約 max_per_cluster={m.MAX_PER_CLUSTER}) - クラスタ均等150銘柄ユニバース】")
    print("=" * 90)
    topn_cluster_df = m.evaluate_topn_curve(
        d_ens, topn_list=[1, 3, 5, 10, 20, 30], max_per_cluster=m.MAX_PER_CLUSTER,
        close_price_lookup=close_price_lookup,
    )
    print(topn_cluster_df.to_string(index=False))

    del models
    torch.cuda.empty_cache()

    try:
        import shutil
        shutil.rmtree(tmpdir)
    except Exception as e:
        print(f"[!] 一時ディレクトリを削除できませんでした: {tmpdir} ({e})")


if __name__ == "__main__":
    main()
