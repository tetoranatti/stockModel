"""空売りランキングモデル v2 学習(2026-09-24追加)。backward_elim_short.pyの結果
(dist_from_high20除外でPF 1.085->1.512, +0.427)を反映した特徴量集合(9 stock + 5 macro、
macroは変更無し)で最終モデルを学習する。train_short_model.py(ロングから流用しただけの
10+5構成、v1)は比較用に残す。

実行方法:
    python research/short_exp/train_short_model_v2.py
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

# backward_elim_short.py 2026-09-24の結果: dist_from_high20を除外(PF 1.085->1.512)。
# それ以外の除外は改善せず(round2で収束)。macroは既存BASELINE_MACRO_COLSのまま変更無し。
SHORT_STOCK_COLS_V2 = [
    'stock_ret_1d', 'stock_ret_20d', 'rolling_beta', 'vol_ratio_5d', 'overnight_gap',
    'dist_from_low20', 'gap_strength_5', 'atr_accel', 'dist_from_high60',
]

SHORT_MODEL_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "cache", "short_model_cache_v2"
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
    print(f"[*] 特徴量: stock={SHORT_STOCK_COLS_V2}")
    print(f"[*] 特徴量: macro={m.BASELINE_MACRO_COLS}")

    macro_df = m.load_macro_slim5(return_source="nk225")
    pool = m.get_full_feature_pool_df(tickers, macro_df, force_rebuild=False)
    margin_pool = m.get_margin_pool_df(tickers, macro_df, force_rebuild=False)
    sector_pool = m.get_sector_relative_pool_df(tickers, macro_df, force_rebuild=False)
    macro_pool_df = m.augment_macro_pool_df(
        m.get_full_macro_pool_df(tickers, force_rebuild=False)
    )

    _base_per_ticker = m.build_per_ticker(
        pool, margin_pool, sector_pool, macro_pool_df, SHORT_STOCK_COLS_V2
    )
    _base_valid_dates = m.valid_cross_section_dates(
        _base_per_ticker, SHORT_STOCK_COLS_V2, m.MIN_CROSS_SECTION
    )
    shared_split_date = m.compute_global_split_date(_base_valid_dates, None)
    print(f"[*] split date: {shared_split_date.date()}")

    tr0, va0 = build_dataset_short(
        m.BASELINE_MACRO_COLS, SHORT_STOCK_COLS_V2, pool, margin_pool, sector_pool,
        macro_pool_df, regime_filter=None, global_split_date_override=shared_split_date,
    )
    print(f"[+] train={len(tr0[2])} val={len(va0[2])}")

    configs = {"short_v2": (tr0, va0, SHORT_STOCK_COLS_V2, m.BASELINE_MACRO_COLS)}
    print(f"\n[*] {len(m.SEEDS)}seedを最大{MAX_CONCURRENT}並列で学習します...")
    t0 = time.time()
    out_paths, tmpdir, elapsed = gfs.train_configs_parallel(m, pss, configs, MAX_CONCURRENT)
    print(f"[+] 学習完了: 実時間={elapsed:.1f}秒")

    os.makedirs(SHORT_MODEL_CACHE_DIR, exist_ok=True)
    for seed in m.SEEDS:
        shutil.copy2(
            out_paths[("short_v2", seed)],
            os.path.join(SHORT_MODEL_CACHE_DIR, f"state_{seed}.pt"),
        )
    print(f"[+] チェックポイントを保存しました: {SHORT_MODEL_CACHE_DIR}")

    try:
        shutil.rmtree(tmpdir)
    except Exception:
        pass


if __name__ == "__main__":
    main()
