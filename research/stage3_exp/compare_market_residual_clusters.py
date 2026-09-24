"""市場残差リターン(候補ユニバース横断面平均に対するOLS回帰残差)ベースの階層クラスタリング
(sklearn AgglomerativeClustering, distance=1-corr, cluster_utils.py::market_residual_corr)
による、TOPNクラスタ制約(MAX_PER_CLUSTER)の比較実験(2026-09-23追加、ユーザー提案)。

n_clusters=10/15/20を比較し、現行のmonthly_consensus(config.N_CLUSTERS)によるクラスタ制約
とも並べて評価する。ユニバースは既存の現行150銘柄(universe_150_tickers.txt)をそのまま使い、
再選定は行わない(compare_universe_selection.pyで別途検証済み・棄却済みのため対象外)。

実行方法:
    python research/stage3_exp/compare_market_residual_clusters.py
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

import pandas as pd

N_CLUSTERS_CANDIDATES = [10, 15, 20]


def main():
    import stage3_exp.experiment_context as m
    import stage3_exp.diagnostics as diagnostics
    from stage3_exp.cluster_utils import (
        compute_market_residual_cluster_map,
        compute_market_residual_spectral_cluster_map,
    )
    import parallel_seed_sweep as pss

    with open(m.UNIVERSE_PATH, "r", encoding="utf-8") as f:
        tickers = [line.strip() for line in f if line.strip()]
    print(f"[*] ユニバース: {len(tickers)}銘柄 ({m.UNIVERSE_PATH})")

    macro_df = m.load_macro_slim5(return_source="nk225")
    pool = m.get_full_feature_pool_df(tickers, macro_df, force_rebuild=False)
    margin_pool = m.get_margin_pool_df(tickers, macro_df, force_rebuild=False)
    sector_pool = m.get_sector_relative_pool_df(tickers, macro_df, force_rebuild=False)
    macro_pool_df = m.augment_macro_pool_df(
        m.get_full_macro_pool_df(tickers, force_rebuild=False)
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

    tmpdir = tempfile.mkdtemp(prefix="compare_market_resid_cluster_")
    shared_data_path = os.path.join(tmpdir, "shared_data.pkl")
    pss.dump_shared_training_data(
        shared_data_path,
        {
            "tr_base": tr0, "va_base": va0,
            "s_cols_base": m.BASELINE_STOCK_COLS, "m_cols_base": m.BASELINE_MACRO_COLS,
        },
    )

    seeds = m.SEEDS
    seed_and_paths = [(s, os.path.join(tmpdir, f"state_{s}.pt")) for s in seeds]
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

    print("\n" + "=" * 90)
    print("【ベースライン(制約なし) TOPN別PF】")
    print("=" * 90)
    baseline_df = m.evaluate_topn_curve(
        d_ens, topn_list=[1, 3, 5, 10, 20, 30], close_price_lookup=close_price_lookup,
    )
    print(baseline_df.to_string(index=False))

    print("\n" + "=" * 90)
    print(f"【現行クラスタ制約(method={m.CLUSTER_METHOD}, N_CLUSTERS={m.N_CLUSTERS}, "
          f"max_per_cluster={m.MAX_PER_CLUSTER}) TOPN別PF】")
    print("=" * 90)
    current_df = m.evaluate_topn_curve(
        d_ens, topn_list=[1, 3, 5, 10, 20, 30], max_per_cluster=m.MAX_PER_CLUSTER,
        close_price_lookup=close_price_lookup,
    )
    print(current_df.to_string(index=False))

    def _run_variant(label, cluster_map):
        diagnostics.TICKER_TO_CLUSTER = cluster_map

        sizes = pd.Series(cluster_map).value_counts().sort_index()
        print(f"\n[*] {label} クラスタサイズ分布(候補ユニバース全体、"
              f"最大クラスタ占有率={sizes.max() / sizes.sum():.1%}):")
        print(sizes.to_string())

        print("\n" + "=" * 90)
        print(f"【{label} クラスタ制約(max_per_cluster={m.MAX_PER_CLUSTER}) TOPN別PF】")
        print("=" * 90)
        df = m.evaluate_topn_curve(
            d_ens, topn_list=[1, 3, 5, 10, 20, 30], max_per_cluster=m.MAX_PER_CLUSTER,
            close_price_lookup=close_price_lookup,
        )
        print(df.to_string(index=False))

    for k in N_CLUSTERS_CANDIDATES:
        cluster_map = compute_market_residual_cluster_map(n_clusters=k, force_rebuild=True)
        _run_variant(f"market_residual_corr(linkage=average) k={k}", cluster_map)

    for k in [15, 20]:
        cluster_map = compute_market_residual_cluster_map(
            n_clusters=k, linkage="complete", force_rebuild=True
        )
        _run_variant(f"market_residual_corr(linkage=complete) k={k}", cluster_map)

    for k in N_CLUSTERS_CANDIDATES:
        cluster_map = compute_market_residual_spectral_cluster_map(
            n_clusters=k, force_rebuild=True
        )
        _run_variant(f"market_residual_spectral k={k}", cluster_map)

    del models
    torch.cuda.empty_cache()

    try:
        import shutil
        shutil.rmtree(tmpdir)
    except Exception as e:
        print(f"[!] 一時ディレクトリを削除できませんでした: {tmpdir} ({e})")


if __name__ == "__main__":
    main()
