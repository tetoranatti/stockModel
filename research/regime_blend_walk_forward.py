"""ロング/ショート レジームゲート配分のwalk-forward検証(2026-09-24追加)。

regime_blend_simulation.pyは単一9か月split(2025-12-19〜2026-09-18)のみでの
検証だったため、「たまたまこのサンプルに効いただけ」の可能性を排除できていなかった。
本スクリプトはstage3_exp.walk_forward_validation.py/short_exp.walk_forward_validation_
short.pyと同じ5-fold expanding window(FOLD_START_PERCENTILES=[0.55,0.65,0.75,0.85,0.95])
で、ロング(BASELINE_STOCK_COLS)・ショートv2(SHORT_STOCK_COLS_V2)の両方を同一の
fold境界(ロング側のvalid_datesを基準日程として共有——ブレンド比較のため両モデルが
全く同じテスト期間で評価される必要がある)で学習・バックテストし、各foldごとに
レジームゲート配分 vs 固定50/50 vs 単体のCAGR/MaxDDを比較する。

レジームラベルはshort_exp.regime_momentum.compute_expanding_momentum_regime_labels
(NK225 20日モメンタム、拡大窓3分位、look-ahead無し)をそのまま使う。

実行方法:
    python research/regime_blend_walk_forward.py
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

FOLD_START_PERCENTILES = [0.55, 0.65, 0.75, 0.85, 0.95]
INNER_VAL_PERCENTILE = 0.85
MAX_CONCURRENT = 10

GATE_SCHEMES = {
    "gate_mild_0.3_0.5_0.7": {"HIGH": 0.3, "MID": 0.5, "LOW": 0.7},
    "gate_strong_0.0_0.5_1.0": {"HIGH": 0.0, "MID": 0.5, "LOW": 1.0},
}


def daily_equity_curve(sub_selected, max_concurrent, close_price_lookup):
    """stage3_exp.diagnostics.compute_max_drawdownの日次時価評価ロジックをそのまま
    流用し、maxddスカラーではなく(日付, 資金)の時系列(pd.Series)を返す版。"""
    import pandas as pd

    if len(sub_selected) == 0:
        return pd.Series(dtype=float)

    sub = sub_selected.reset_index(drop=True)
    entry_dates = pd.to_datetime(sub["entry_date"])
    exit_dates = pd.to_datetime(sub["exit_date"])

    tickers_involved = sub["ticker"].unique()
    cal_start, cal_end = entry_dates.min(), exit_dates.max()
    master_calendar = sorted(
        set().union(*(s.index for t, s in close_price_lookup.items() if t in set(tickers_involved)))
    )
    master_calendar = pd.DatetimeIndex(master_calendar)
    master_calendar = master_calendar[(master_calendar >= cal_start) & (master_calendar <= cal_end)]

    close_reindexed = {
        t: close_price_lookup[t].reindex(master_calendar).ffill()
        for t in tickers_involved if t in close_price_lookup
    }

    entry_prices = {}
    for pos in range(len(sub)):
        t, ed = sub.at[pos, "ticker"], entry_dates.iloc[pos]
        series = close_reindexed.get(t)
        entry_prices[pos] = series.loc[ed] if series is not None and ed in series.index else None

    entries_by_date, exits_by_date = {}, {}
    for pos in range(len(sub)):
        entries_by_date.setdefault(entry_dates.iloc[pos], []).append(pos)
        exits_by_date.setdefault(exit_dates.iloc[pos], []).append(pos)

    def _mark_return(pos, day):
        entry_p = entry_prices.get(pos)
        if entry_p is None or entry_p == 0:
            return 0.0
        t = sub.at[pos, "ticker"]
        series = close_reindexed.get(t)
        if series is None or day not in series.index:
            return 0.0
        px = series.loc[day]
        if pd.isna(px):
            return 0.0
        return (px - entry_p) / entry_p

    cash = 1.0
    open_positions = {}
    equity_curve = []
    for day in master_calendar:
        for pos in exits_by_date.get(day, []):
            alloc = open_positions.pop(pos, None)
            if alloc is not None:
                cash += alloc * (1.0 + sub.at[pos, "ret_pct"])
        for pos in entries_by_date.get(day, []):
            if len(open_positions) < max_concurrent:
                current_equity = cash + sum(a * (1.0 + _mark_return(p, day)) for p, a in open_positions.items())
                alloc = min(current_equity / max_concurrent, cash)
                if alloc > 0:
                    cash -= alloc
                    open_positions[pos] = alloc
        total_equity = cash + sum(a * (1.0 + _mark_return(p, day)) for p, a in open_positions.items())
        equity_curve.append(total_equity)

    return pd.Series(equity_curve, index=master_calendar)


def select_daily_topn(d_all, n_per_day):
    if len(d_all) == 0:
        return d_all
    return (
        d_all.sort_values(["date", "score", "ticker"], ascending=[True, False, True])
        .groupby("date", group_keys=False)
        .head(n_per_day)
    )


def compute_max_drawdown_from_curve(equity_curve):
    peak, max_dd = -float("inf"), 0.0
    for eq in equity_curve:
        peak = max(peak, eq)
        if peak > 0:
            max_dd = min(max_dd, (eq - peak) / peak)
    return max_dd


def compute_cagr_from_curve(final_equity, n_days):
    if n_days <= 0 or final_equity <= 0:
        return float("nan")
    years = n_days / 365.25
    return final_equity ** (1.0 / years) - 1.0 if years > 0 else float("nan")


def build_curve(daily_returns):
    curve = [1.0]
    for r in daily_returns:
        curve.append(curve[-1] * (1.0 + r))
    return curve[1:]


def main():
    import pandas as pd
    import torch
    import stage3_exp.experiment_context as m
    import parallel_seed_sweep as pss
    import greedy_feature_search as gfs
    from short_exp.data_builder_short import build_dataset_short, prepare_backtest_pool_short
    from short_exp.train_short_model_v2 import SHORT_STOCK_COLS_V2
    from short_exp.regime_momentum import compute_expanding_momentum_regime_labels

    with open(m.UNIVERSE_PATH, "r", encoding="utf-8") as f:
        tickers = [line.strip() for line in f if line.strip()]
    print(f"[*] ユニバース: {len(tickers)}銘柄")

    macro_df = m.load_macro_slim5(return_source="nk225")
    pool = m.get_full_feature_pool_df(tickers, macro_df, force_rebuild=False)
    margin_pool = m.get_margin_pool_df(tickers, macro_df, force_rebuild=False)
    sector_pool = m.get_sector_relative_pool_df(tickers, macro_df, force_rebuild=False)
    macro_pool_df = m.augment_macro_pool_df(m.get_full_macro_pool_df(tickers, force_rebuild=False))

    regime_labels = compute_expanding_momentum_regime_labels(macro_pool_df)

    # --- ロング側のvalid_datesをfold境界の共通基準にする(ブレンド比較のため
    #     両モデルを完全に同じテスト期間で評価する必要がある) ---
    _base_per_ticker_long = m.build_per_ticker(pool, margin_pool, sector_pool, macro_pool_df, m.BASELINE_STOCK_COLS)
    valid_dates_long = m.valid_cross_section_dates(_base_per_ticker_long, m.BASELINE_STOCK_COLS, m.MIN_CROSS_SECTION)
    sorted_dates = pd.DatetimeIndex(sorted(valid_dates_long))
    print(f"[*] 共通有効取引日(ロング基準): {len(sorted_dates)}日 ({sorted_dates[0].date()} 〜 {sorted_dates[-1].date()})")

    boundary_dates = [sorted_dates[int(len(sorted_dates) * p)] for p in FOLD_START_PERCENTILES]

    configs, fold_meta = {}, {}
    close_price_lookup = {}
    for i, outer_start in enumerate(boundary_dates):
        outer_end = boundary_dates[i + 1] if i + 1 < len(boundary_dates) else None
        dates_before = sorted_dates[sorted_dates < outer_start]
        inner_split = m.compute_global_split_date(dates_before, None, percentile=INNER_VAL_PERCENTILE)

        # ロング
        tr_l, va_l = m.build_dataset(
            m.BASELINE_MACRO_COLS, m.BASELINE_STOCK_COLS, pool, margin_pool, sector_pool,
            macro_pool_df, regime_filter=None, global_split_date_override=inner_split,
            val_end_date_override=outer_start,
        )
        pool_base_full_l = m.prepare_backtest_pool(
            m.BASELINE_MACRO_COLS, m.BASELINE_STOCK_COLS, pool, margin_pool, sector_pool,
            macro_pool_df, regime_filter=None, global_split_date_override=outer_start,
            close_price_out=close_price_lookup,
        )
        pool_test_l = [row for row in pool_base_full_l if outer_end is None or row[1] < outer_end]

        # ショート(ロングと全く同じinner_split/outer_start/outer_endを使い、テスト期間を揃える)
        tr_s, va_s = build_dataset_short(
            m.BASELINE_MACRO_COLS, SHORT_STOCK_COLS_V2, pool, margin_pool, sector_pool,
            macro_pool_df, regime_filter=None, global_split_date_override=inner_split,
            val_end_date_override=outer_start,
        )
        pool_base_full_s = prepare_backtest_pool_short(
            m.BASELINE_MACRO_COLS, SHORT_STOCK_COLS_V2, pool, margin_pool, sector_pool,
            macro_pool_df, regime_filter=None, global_split_date_override=outer_start,
            close_price_out=close_price_lookup,
        )
        pool_test_s = [row for row in pool_base_full_s if outer_end is None or row[1] < outer_end]

        configs[f"long_fold{i}"] = (tr_l, va_l, m.BASELINE_STOCK_COLS, m.BASELINE_MACRO_COLS)
        configs[f"short_fold{i}"] = (tr_s, va_s, SHORT_STOCK_COLS_V2, m.BASELINE_MACRO_COLS)
        fold_meta[i] = dict(
            outer_start=outer_start, outer_end=outer_end, inner_split=inner_split,
            pool_test_l=pool_test_l, pool_test_s=pool_test_s,
        )
        end_str = outer_end.date() if outer_end is not None else sorted_dates[-1].date()
        print(f"[*] fold{i}: train<{inner_split.date()} val[{inner_split.date()},{outer_start.date()}) "
              f"test[{outer_start.date()},{end_str}) long_train_n={len(tr_l[2])} short_train_n={len(tr_s[2])} "
              f"long_test_n={len(pool_test_l)} short_test_n={len(pool_test_s)}")

    print(f"\n[*] {len(configs)}config x {len(m.SEEDS)}seed = {len(configs) * len(m.SEEDS)}ジョブを"
          f"最大{MAX_CONCURRENT}並列で学習します...")
    t0 = time.time()
    out_paths, tmpdir, elapsed = gfs.train_configs_parallel(m, pss, configs, MAX_CONCURRENT)
    print(f"[+] 全ジョブ学習完了: 実時間={elapsed:.1f}秒")

    fold_rows = []
    for i in sorted(fold_meta.keys()):
        meta = fold_meta[i]

        models_l = []
        for seed in m.SEEDS:
            model = m.build_model(m.BASELINE_STOCK_COLS, m.BASELINE_MACRO_COLS)
            model.load_state_dict(torch.load(out_paths[(f"long_fold{i}", seed)], map_location=m.DEVICE))
            model.eval()
            models_l.append(model)
        d_ens_long = m.run_backtest_inference_ensemble(models_l, meta["pool_test_l"])
        del models_l
        torch.cuda.empty_cache()

        models_s = []
        for seed in m.SEEDS:
            model = m.build_model(SHORT_STOCK_COLS_V2, m.BASELINE_MACRO_COLS)
            model.load_state_dict(torch.load(out_paths[(f"short_fold{i}", seed)], map_location=m.DEVICE))
            model.eval()
            models_s.append(model)
        d_ens_short = m.run_backtest_inference_ensemble(models_s, meta["pool_test_s"])
        del models_s
        torch.cuda.empty_cache()

        long_selected = select_daily_topn(d_ens_long, m.DAILY_TOPN)
        short_selected = select_daily_topn(d_ens_short, m.DAILY_TOPN)
        eq_long = daily_equity_curve(long_selected, m.MAX_CONCURRENT_POSITIONS, close_price_lookup)
        eq_short = daily_equity_curve(short_selected, m.MAX_CONCURRENT_POSITIONS, close_price_lookup)

        if len(eq_long) < 5 or len(eq_short) < 5:
            print(f"[!] fold{i}: 取引数が少なすぎるためスキップ (long={len(eq_long)}日 short={len(eq_short)}日)")
            continue

        ret_long = eq_long.pct_change().dropna()
        ret_short = eq_short.pct_change().dropna()
        common_idx = ret_long.index.intersection(ret_short.index)
        ret_long = ret_long.reindex(common_idx)
        ret_short = ret_short.reindex(common_idx)
        regime_aligned = regime_labels.reindex(common_idx).ffill()
        n_days = (common_idx.max() - common_idx.min()).days if len(common_idx) > 1 else 0

        print(f"\n--- fold{i}: test[{meta['outer_start'].date()}, "
              f"{meta['outer_end'].date() if meta['outer_end'] is not None else sorted_dates[-1].date()}) "
              f"共通リターン日数={len(common_idx)} ---")
        print("  regime分布:", regime_aligned.value_counts().to_dict())

        def eval_scheme(scheme_name, w_short_series):
            blended = (1.0 - w_short_series) * ret_long + w_short_series * ret_short
            curve = build_curve(blended.values)
            dd = compute_max_drawdown_from_curve(curve)
            cagr = compute_cagr_from_curve(curve[-1] if curve else 1.0, n_days)
            print(f"    {scheme_name:<26s} CAGR={cagr*100:7.1f}%  MaxDD={dd*100:6.1f}%  最終資産={curve[-1] if curve else float('nan'):.3f}")
            fold_rows.append(dict(fold=i, scheme=scheme_name, cagr=cagr, max_dd=dd,
                                   final=curve[-1] if curve else float("nan"), n_days=len(common_idx)))

        eval_scheme("long_only", pd.Series(0.0, index=common_idx))
        eval_scheme("short_only", pd.Series(1.0, index=common_idx))
        eval_scheme("fixed_50_50", pd.Series(0.5, index=common_idx))
        for name, gate in GATE_SCHEMES.items():
            w_short_series = regime_aligned.map(gate).fillna(0.5)
            eval_scheme(name, w_short_series)

    try:
        shutil.rmtree(tmpdir)
    except Exception:
        pass

    result_df = pd.DataFrame(fold_rows)
    print("\n" + "=" * 100)
    print("【walk-forward集計: スキーム別 CAGR/MaxDD (5-fold平均)】")
    print("=" * 100)
    summary = result_df.groupby("scheme").agg(
        cagr_mean=("cagr", "mean"), cagr_std=("cagr", "std"),
        max_dd_mean=("max_dd", "mean"), max_dd_worst=("max_dd", "min"),
        n_folds=("fold", "count"),
    )
    print(summary.to_string())

    out_path = r"F:\stockModel\research\cache\_cache_regime_blend_walk_forward_result.csv"
    result_df.to_csv(out_path, index=False)
    print(f"\n[+] fold別結果を保存: {out_path}")


if __name__ == "__main__":
    main()
