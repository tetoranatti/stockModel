"""空売りモデルv2のWalk-forward検証(2026-09-24、ユーザー提案「walk-forwardから先に」)。
stage3_exp/walk_forward_validation.py(ロング側、[[walk_forward_validation_2026-09-23]])の
5-fold expanding window方式をそのまま空売りv2(SHORT_STOCK_COLS_V2、9 stock+5 macro、
dist_from_high20除外済み)に適用する。データ構築・ターゲットはbuild_dataset_short/
prepare_backtest_pool_short(compute_simulated_short_targets、正=空売り利益)を使う点のみ
ロング版と異なり、fold分割・train/val/testロジック・評価方法は完全に同一。

実行方法:
    python research/short_exp/walk_forward_validation_short.py
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

from train_short_model_v2 import SHORT_STOCK_COLS_V2

FOLD_START_PERCENTILES = [0.55, 0.65, 0.75, 0.85, 0.95]
INNER_VAL_PERCENTILE = 0.85
MAX_CONCURRENT = 10


def main():
    import pandas as pd
    import torch
    import stage3_exp.experiment_context as m
    import parallel_seed_sweep as pss
    import greedy_feature_search as gfs
    from short_exp.data_builder_short import build_dataset_short, prepare_backtest_pool_short

    with open(m.UNIVERSE_PATH, "r", encoding="utf-8") as f:
        tickers = [line.strip() for line in f if line.strip()]
    print(f"[*] ユニバース: {len(tickers)}銘柄")
    print(f"[*] 特徴量: stock={SHORT_STOCK_COLS_V2}")

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
    valid_dates = m.valid_cross_section_dates(
        _base_per_ticker, SHORT_STOCK_COLS_V2, m.MIN_CROSS_SECTION
    )
    sorted_dates = pd.DatetimeIndex(sorted(valid_dates))
    print(f"[*] 有効取引日: {len(sorted_dates)}日 ({sorted_dates[0].date()} 〜 {sorted_dates[-1].date()})")

    boundary_dates = [
        sorted_dates[int(len(sorted_dates) * p)] for p in FOLD_START_PERCENTILES
    ]

    configs, fold_meta = {}, {}
    close_price_lookup = {}
    for i, outer_start in enumerate(boundary_dates):
        name = f"fold{i}"
        outer_end = boundary_dates[i + 1] if i + 1 < len(boundary_dates) else None
        dates_before = sorted_dates[sorted_dates < outer_start]
        inner_split = m.compute_global_split_date(dates_before, None, percentile=INNER_VAL_PERCENTILE)

        tr, va = build_dataset_short(
            m.BASELINE_MACRO_COLS, SHORT_STOCK_COLS_V2, pool, margin_pool, sector_pool,
            macro_pool_df, regime_filter=None,
            global_split_date_override=inner_split, val_end_date_override=outer_start,
        )
        pool_base_full = prepare_backtest_pool_short(
            m.BASELINE_MACRO_COLS, SHORT_STOCK_COLS_V2, pool, margin_pool, sector_pool,
            macro_pool_df, regime_filter=None, global_split_date_override=outer_start,
            close_price_out=close_price_lookup,
        )
        pool_test = [row for row in pool_base_full if outer_end is None or row[1] < outer_end]

        configs[name] = (tr, va, SHORT_STOCK_COLS_V2, m.BASELINE_MACRO_COLS)
        fold_meta[name] = dict(
            outer_start=outer_start, outer_end=outer_end,
            inner_split=inner_split, pool_test=pool_test,
            n_train=len(tr[2]), n_val=len(va[2]),
        )
        end_str = outer_end.date() if outer_end is not None else sorted_dates[-1].date()
        print(
            f"[*] {name}: train<{inner_split.date()} val[{inner_split.date()},{outer_start.date()}) "
            f"test[{outer_start.date()},{end_str}) train_n={len(tr[2])} val_n={len(va[2])} test_n={len(pool_test)}"
        )

    print(f"\n[*] {len(configs)}fold x {len(m.SEEDS)}seed = {len(configs) * len(m.SEEDS)}ジョブを"
          f"並列学習します(最大{MAX_CONCURRENT}並列)...")
    t0 = time.time()
    out_paths, tmpdir, elapsed = gfs.train_configs_parallel(m, pss, configs, MAX_CONCURRENT)
    print(f"[+] 全ジョブ学習完了: 実時間={elapsed:.1f}秒")

    rows = []
    for name in configs:
        models = []
        for seed in m.SEEDS:
            model = m.build_model(SHORT_STOCK_COLS_V2, m.BASELINE_MACRO_COLS)
            model.load_state_dict(torch.load(out_paths[(name, seed)], map_location=m.DEVICE))
            model.eval()
            models.append(model)

        pool_test = fold_meta[name]["pool_test"]
        d_ens = m.run_backtest_inference_ensemble(models, pool_test)

        r_capacity = m.evaluate_daily_topn(
            d_ens, name, n_per_day=m.DAILY_TOPN, print_detail=False,
            min_reliable_n=m.MIN_RELIABLE_N, max_concurrent=m.MAX_CONCURRENT_POSITIONS,
            close_price_lookup=close_price_lookup,
        )
        curve_df = m.evaluate_topn_curve(
            d_ens, topn_list=[5], close_price_lookup=close_price_lookup,
        )
        curve5 = curve_df.iloc[0]

        rows.append(dict(
            fold=name,
            test_start=fold_meta[name]["outer_start"].date(),
            test_end=(fold_meta[name]["outer_end"].date() if fold_meta[name]["outer_end"] is not None
                      else sorted_dates[-1].date()),
            train_n=fold_meta[name]["n_train"],
            capacity_pf=r_capacity["pf"], capacity_n=r_capacity["n"],
            unconstrained_pf=curve5["pf"], unconstrained_trades=curve5["trades"],
            max_dd=curve5["max_dd"],
        ))
        del models
        torch.cuda.empty_cache()

    try:
        shutil.rmtree(tmpdir)
    except Exception:
        pass

    result_df = pd.DataFrame(rows)
    print("\n" + "=" * 90)
    print("【空売りモデルv2 Walk-forward検証結果(5-fold, expanding window)】")
    print("=" * 90)
    print(result_df.to_string(index=False))
    print(
        f"\n  capacity_pf: mean={result_df['capacity_pf'].mean():.3f} std={result_df['capacity_pf'].std():.3f}"
    )
    print(
        f"  unconstrained_pf: mean={result_df['unconstrained_pf'].mean():.3f} "
        f"std={result_df['unconstrained_pf'].std():.3f}"
    )
    n_pf_gt1 = int((result_df["unconstrained_pf"] > 1.0).sum())
    print(f"  unconstrained_pf > 1.0: {n_pf_gt1}/{len(result_df)}fold")

    result_df.to_csv(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cache", "walk_forward_result_short.csv"),
        index=False,
    )


if __name__ == "__main__":
    main()
