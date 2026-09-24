"""Walk-forward検証(2026-09-23、ユーザー提案「実運用に耐えうるモデルと言えるか」の
確認)。既存のバックテストは単一の固定75/25分割([[backtest_methodology_2026-09-23]])で、
2025-12-18以降の約9ヶ月の1区間しか見ていない——たまたまその期間で良かっただけなのか、
時期を問わず安定しているのかを区別できない。

方式: expanding window(訓練期間を毎回広げる)walk-forward、5-fold。有効な取引日734日
(2024-01〜2026-09)に対して、各foldのテスト開始点をFOLD_START_PERCENTILESの百分位点
(55/65/75/85/95%)に置き、次の境界までをテスト期間とする(最終foldのみ終端まで、
約37日と短め)。fold3のテスト開始点(75%点=2025-12-18)は現行の固定splitと一致する
——既存の単一分割結果はこのwalk-forwardの1foldとして包含される形になる。

各foldの訓練データはさらにINNER_VAL_PERCENTILE(85%)点でtrain/val(early stopping用)に
分割する(既存のcompute_global_split_dateをそのまま再利用)。5fold×5seed=25ジョブを
1つのフラットな並列プールにまとめて学習する(greedy_feature_search.pyのtrain_configs_
parallelを再利用)。

実行方法:
    python research/stage3_exp/walk_forward_validation.py
"""
import sys
import os
import time
import tempfile
import shutil

sys.path.insert(0, r"F:\stockModel")
sys.path.insert(0, r"F:\stockModel\research")

try:
    sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")
    sys.stderr.reconfigure(line_buffering=True, encoding="utf-8")
except Exception:
    pass

FOLD_START_PERCENTILES = [0.55, 0.65, 0.75, 0.85, 0.95]
INNER_VAL_PERCENTILE = 0.85
MAX_CONCURRENT = 10


def main():
    import pandas as pd
    import torch
    import stage3_exp.experiment_context as m
    import parallel_seed_sweep as pss
    import greedy_feature_search as gfs

    with open(m.UNIVERSE_PATH, "r", encoding="utf-8") as f:
        tickers = [line.strip() for line in f if line.strip()]
    print(f"[*] ユニバース: {len(tickers)}銘柄")

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
    valid_dates = m.valid_cross_section_dates(
        _base_per_ticker, m.BASELINE_STOCK_COLS, m.MIN_CROSS_SECTION
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

        tr, va = m.build_dataset(
            m.BASELINE_MACRO_COLS, m.BASELINE_STOCK_COLS, pool, margin_pool, sector_pool,
            macro_pool_df, regime_filter=None, regime_labels_for_loss=loss_regime_labels,
            global_split_date_override=inner_split, val_end_date_override=outer_start,
        )
        pool_base_full = m.prepare_backtest_pool(
            m.BASELINE_MACRO_COLS, m.BASELINE_STOCK_COLS, pool, margin_pool, sector_pool,
            macro_pool_df, regime_filter=None, global_split_date_override=outer_start,
            close_price_out=close_price_lookup,
        )
        pool_test = [row for row in pool_base_full if outer_end is None or row[1] < outer_end]

        configs[name] = (tr, va, m.BASELINE_STOCK_COLS, m.BASELINE_MACRO_COLS)
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
            model = m.build_model(m.BASELINE_STOCK_COLS, m.BASELINE_MACRO_COLS)
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
    print("【Walk-forward検証結果(5-fold, expanding window)】")
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
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cache", "walk_forward_result.csv"),
        index=False,
    )


if __name__ == "__main__":
    main()
