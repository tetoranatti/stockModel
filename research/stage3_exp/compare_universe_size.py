"""学習銘柄拡大(現行150→300銘柄、追加のみ)の効果を検証する実験
(2026-09-23、ユーザー提案)。

universe_300_tickers_expanded.txt / train_universe_bars_expanded300.parquet は
research/stage3_exp/backfill_expanded_universe.py で事前に作成済み(現行150銘柄は
そのまま維持し、適格銘柄のうち流動性上位150銘柄を追加。新規銘柄もfetch_ticker_jquants
で本番と同じ3年分をバックフィル済みなので、cluster_diversification_2026-09-23の
「クラスタ均等150銘柄」実験で起きた「新規銘柄の履歴が短くサンプル数/split dateが
交絡する」問題を回避している)。

compare_universe_selection.pyと同様、_feature_cache_utils.pyのキャッシュパス定数を
プロセス内だけ書き換えて別ファイルに退避させ、本番キャッシュ・現行150銘柄の実験結果には
一切影響しない。現行のBASELINE_STOCK_COLS/BASELINE_MACRO_COLS(トグル変更なし、
クラスタEmbedding等の新規特徴量も無し)のまま300銘柄ユニバースで5seed学習・アンサンブル
評価し、既存150銘柄での基準値(業種制約 top5/10/20 PF=1.632/1.584/1.649、
クラスタ制約 PF=1.775/1.804/1.759)と比較する。

実行方法:
    python research/stage3_exp/compare_universe_size.py
"""
import os
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

BASE_DIR = r"F:\stockModel"
CACHE_DIR = os.path.join(BASE_DIR, "data", "cache")
RESEARCH_CACHE_DIR = os.path.join(BASE_DIR, "research", "cache")
EXPANDED_UNIVERSE_PATH = os.path.join(BASE_DIR, "universe_300_tickers_expanded.txt")
EXPANDED_BARS_PATH = os.path.join(CACHE_DIR, "train_universe_bars_expanded300.parquet")
CACHE_TAG = "expanded300"


def _patch_cache_paths(fcu):
    """_feature_cache_utils側のキャッシュパス定数をこのプロセス内だけ差し替える
    (既存の現行150銘柄用キャッシュ・train_universe_bars.parquetは変更しない)。"""
    fcu.UNIVERSE_BARS_CACHE_PATH = EXPANDED_BARS_PATH
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
    import _feature_cache_utils as fcu
    _patch_cache_paths(fcu)

    import stage3_exp.experiment_context as m
    m.UNIVERSE_PATH = EXPANDED_UNIVERSE_PATH  # _build_data()はm.UNIVERSE_PATHを直接読む

    import parallel_seed_sweep as pss

    with open(m.UNIVERSE_PATH, "r", encoding="utf-8") as f:
        tickers = [line.strip() for line in f if line.strip()]
    print(f"[*] 拡大ユニバース: {len(tickers)}銘柄")

    print("[*] データ構築(300銘柄ユニバース、force_rebuild=Trueでキャッシュ新規生成)...")
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

    tmpdir = tempfile.mkdtemp(prefix="compare_universe_size_")
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
        d_ens, f"拡大300銘柄(アンサンブル{len(seeds)}seed) score Top-K",
        min_reliable_n=m.MIN_RELIABLE_N, common_sample_k=m.COMMON_SAMPLE_K,
        max_concurrent=m.MAX_CONCURRENT_POSITIONS, close_price_lookup=close_price_lookup,
    )

    print("\n" + "=" * 90)
    print(f"【TOPN別PF分析(業種制約 max_per_sector={m.MAX_PER_SECTOR}) - 拡大300銘柄ユニバース】")
    print("=" * 90)
    topn_sector_df = m.evaluate_topn_curve(
        d_ens, topn_list=[1, 3, 5, 10, 20, 30], max_per_sector=m.MAX_PER_SECTOR,
        close_price_lookup=close_price_lookup,
    )
    print(topn_sector_df.to_string(index=False))

    print("\n" + "=" * 90)
    print(f"【TOPN別PF分析(クラスタ制約 max_per_cluster={m.MAX_PER_CLUSTER}) - 拡大300銘柄ユニバース】")
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
