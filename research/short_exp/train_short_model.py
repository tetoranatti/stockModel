"""空売りランキングモデルの学習(2026-09-24追加)。ロング版と全く同じ特徴量集合
(BASELINE_STOCK_COLS/BASELINE_MACRO_COLS)・アーキテクチャ(build_model)・損失関数
(NearPairRankingLossRegimeAware)を使い、ラベルだけをdata_builder_short.build_dataset_short
(空売り損益で符号付け)に差し替える。学習だけを行い、チェックポイントを永続キャッシュへ
保存する——バックテストはbacktest_short_model.py側で、再学習せずにキャッシュを読むだけで
何度でも試せるようにする(ロング版のbaseline_model_cacheと同じ発想)。

実行方法:
    python research/short_exp/train_short_model.py
"""
import sys
import os
import time
import shutil

sys.path.insert(0, r"F:\stockModel")
sys.path.insert(0, r"F:\stockModel\research")

try:
    sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")
    sys.stderr.reconfigure(line_buffering=True, encoding="utf-8")
except Exception:
    pass

SHORT_MODEL_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "cache", "short_model_cache"
)
MAX_CONCURRENT = 10


def main():
    import stage3_exp.experiment_context as m
    import parallel_seed_sweep as pss
    import greedy_feature_search as gfs
    from short_exp.data_builder_short import build_dataset_short

    with open(m.UNIVERSE_PATH, "r", encoding="utf-8") as f:
        tickers = [line.strip() for line in f if line.strip()]
    print(f"[*] ユニバース: {len(tickers)}銘柄")
    print(f"[*] 特徴量: stock={m.BASELINE_STOCK_COLS}")
    print(f"[*] 特徴量: macro={m.BASELINE_MACRO_COLS}")

    macro_df = m.load_macro_slim5(return_source="nk225")
    pool = m.get_full_feature_pool_df(tickers, macro_df, force_rebuild=False)
    margin_pool = m.get_margin_pool_df(tickers, macro_df, force_rebuild=False)
    sector_pool = m.get_sector_relative_pool_df(tickers, macro_df, force_rebuild=False)
    macro_pool_df = m.augment_macro_pool_df(
        m.get_full_macro_pool_df(tickers, force_rebuild=False)
    )

    _base_per_ticker = m.build_per_ticker(
        pool, margin_pool, sector_pool, macro_pool_df, m.BASELINE_STOCK_COLS
    )
    _base_valid_dates = m.valid_cross_section_dates(
        _base_per_ticker, m.BASELINE_STOCK_COLS, m.MIN_CROSS_SECTION
    )
    shared_split_date = m.compute_global_split_date(_base_valid_dates, None)
    print(f"[*] split date: {shared_split_date.date()}")

    tr0, va0 = build_dataset_short(
        m.BASELINE_MACRO_COLS, m.BASELINE_STOCK_COLS, pool, margin_pool, sector_pool,
        macro_pool_df, regime_filter=None, global_split_date_override=shared_split_date,
    )
    print(f"[+] train={len(tr0[2])} val={len(va0[2])}")
    print(
        f"[*] train target(空売り損益) mean={tr0[2].mean():.4f} std={tr0[2].std():.4f} "
        f"(参考: 正=空売りで儲かった方向)"
    )

    configs = {"short": (tr0, va0, m.BASELINE_STOCK_COLS, m.BASELINE_MACRO_COLS)}
    print(f"\n[*] {len(m.SEEDS)}seedを最大{MAX_CONCURRENT}並列で学習します...")
    t0 = time.time()
    out_paths, tmpdir, elapsed = gfs.train_configs_parallel(m, pss, configs, MAX_CONCURRENT)
    print(f"[+] 学習完了: 実時間={elapsed:.1f}秒")

    os.makedirs(SHORT_MODEL_CACHE_DIR, exist_ok=True)
    for seed in m.SEEDS:
        shutil.copy2(
            out_paths[("short", seed)],
            os.path.join(SHORT_MODEL_CACHE_DIR, f"state_{seed}.pt"),
        )
    print(f"[+] チェックポイントを保存しました: {SHORT_MODEL_CACHE_DIR}")

    try:
        shutil.rmtree(tmpdir)
    except Exception:
        pass


if __name__ == "__main__":
    main()
