"""ロング/ショート2モデルのレジーム連動ブレンド設計向け、事前分析(2026-09-24追加)。

ユーザー確認事項:
  1. 空売りモデルv2のTopN(1〜5)を、実運用と同じ同時保有制約込み(evaluate_daily_topn)で
     比較する。研究ハーネスのTOPNカーブ(_log_backtest_short_v2.txt)はtrades/pf/avg_retが
     制約無視の「純粋なランキング品質」であり、実際の約定集合とはズレる
     (see research/cache/_log_backtest_short_v2.txt個別行コメント)。
  2. ロングモデル(baseline、150銘柄・DAILY_TOPN=5)とショートモデルv2の日次資金曲線を
     同じclose_price_lookupベースの時価評価ロジックで構築し、日次リターンの相関、および
     ロング側の主要ドローダウン局面でショート側がどう動くかを見る。

実行方法:
    python research/regime_blend_short_long_correlation.py
"""
import sys
import os

sys.path.insert(0, r"F:\stockModel")
sys.path.insert(0, r"F:\stockModel\research")

try:
    sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")
    sys.stderr.reconfigure(line_buffering=True, encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd


def daily_equity_curve(sub_selected, max_concurrent, close_price_lookup):
    """stage3_exp.diagnostics.compute_max_drawdownのclose_price_lookup分岐(日次時価評価)
    をそのまま流用しつつ、maxddスカラーではなく(日付, 資金)の時系列を返す版。
    ロジック自体はdiagnostics.pyから変更していない(2026-09-23導入の日次時価評価版)。"""
    if len(sub_selected) == 0:
        return pd.Series(dtype=float)

    sub = sub_selected.reset_index(drop=True)
    entry_dates = pd.to_datetime(sub["entry_date"])
    exit_dates = pd.to_datetime(sub["exit_date"])

    tickers_involved = sub["ticker"].unique()
    cal_start, cal_end = entry_dates.min(), exit_dates.max()
    master_calendar = sorted(
        set().union(
            *(s.index for t, s in close_price_lookup.items() if t in set(tickers_involved))
        )
    )
    master_calendar = pd.DatetimeIndex(master_calendar)
    master_calendar = master_calendar[(master_calendar >= cal_start) & (master_calendar <= cal_end)]

    close_reindexed = {
        t: close_price_lookup[t].reindex(master_calendar).ffill()
        for t in tickers_involved
        if t in close_price_lookup
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
                final_ret = sub.at[pos, "ret_pct"]
                cash += alloc * (1.0 + final_ret)
        for pos in entries_by_date.get(day, []):
            if len(open_positions) < max_concurrent:
                current_equity = cash + sum(
                    a * (1.0 + _mark_return(p, day)) for p, a in open_positions.items()
                )
                alloc = min(current_equity / max_concurrent, cash)
                if alloc > 0:
                    cash -= alloc
                    open_positions[pos] = alloc
                    executed_positions_append = pos  # noqa: F841 (readability only)
        total_equity = cash + sum(
            a * (1.0 + _mark_return(p, day)) for p, a in open_positions.items()
        )
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


def build_short_d_ens():
    import torch
    import stage3_exp.experiment_context as m
    from short_exp.data_builder_short import prepare_backtest_pool_short
    from short_exp.train_short_model_v2 import SHORT_STOCK_COLS_V2, SHORT_MODEL_CACHE_DIR

    with open(m.UNIVERSE_PATH, "r", encoding="utf-8") as f:
        tickers = [line.strip() for line in f if line.strip()]

    macro_df = m.load_macro_slim5(return_source="nk225")
    pool = m.get_full_feature_pool_df(tickers, macro_df, force_rebuild=False)
    margin_pool = m.get_margin_pool_df(tickers, macro_df, force_rebuild=False)
    sector_pool = m.get_sector_relative_pool_df(tickers, macro_df, force_rebuild=False)
    macro_pool_df = m.augment_macro_pool_df(m.get_full_macro_pool_df(tickers, force_rebuild=False))

    _base_per_ticker = m.build_per_ticker(pool, margin_pool, sector_pool, macro_pool_df, SHORT_STOCK_COLS_V2)
    _base_valid_dates = m.valid_cross_section_dates(_base_per_ticker, SHORT_STOCK_COLS_V2, m.MIN_CROSS_SECTION)
    shared_split_date = m.compute_global_split_date(_base_valid_dates, None)
    print(f"[short] split date: {shared_split_date.date()}")

    close_price_lookup = {}
    pool_test = prepare_backtest_pool_short(
        m.BASELINE_MACRO_COLS, SHORT_STOCK_COLS_V2, pool, margin_pool, sector_pool,
        macro_pool_df, regime_filter=None, global_split_date_override=shared_split_date,
        close_price_out=close_price_lookup,
    )
    print(f"[short] backtest_pool={len(pool_test)}")

    models = []
    for seed in m.SEEDS:
        ckpt = os.path.join(SHORT_MODEL_CACHE_DIR, f"state_{seed}.pt")
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"{ckpt}が見つかりません。先にtrain_short_model_v2.pyを実行してください。")
        model = m.build_model(SHORT_STOCK_COLS_V2, m.BASELINE_MACRO_COLS)
        model.load_state_dict(torch.load(ckpt, map_location=m.DEVICE))
        model.eval()
        models.append(model)

    d_ens = m.run_backtest_inference_ensemble(models, pool_test)
    del models
    torch.cuda.empty_cache()
    return d_ens, close_price_lookup


def build_long_d_ens():
    import tempfile
    import time
    import shutil
    import torch
    import stage3_exp.experiment_context as m
    import parallel_seed_sweep as pss

    with open(m.UNIVERSE_PATH, "r", encoding="utf-8") as f:
        tickers = [line.strip() for line in f if line.strip()]

    macro_df = m.load_macro_slim5(return_source="nk225")
    pool = m.get_full_feature_pool_df(tickers, macro_df, force_rebuild=False)
    margin_pool = m.get_margin_pool_df(tickers, macro_df, force_rebuild=False)
    sector_pool = m.get_sector_relative_pool_df(tickers, macro_df, force_rebuild=False)
    macro_pool_df = m.augment_macro_pool_df(m.get_full_macro_pool_df(tickers, force_rebuild=False))

    loss_regime_labels = m.compute_regime_labels_expanding(macro_pool_df) if m.USE_REGIME_AWARE_LOSS else None

    _base_per_ticker = m.build_per_ticker(pool, margin_pool, sector_pool, macro_pool_df, m.BASELINE_STOCK_COLS)
    _base_valid_dates = m.valid_cross_section_dates(_base_per_ticker, m.BASELINE_STOCK_COLS, m.MIN_CROSS_SECTION)
    shared_split_date = m.compute_global_split_date(_base_valid_dates, None)
    print(f"[long] split date: {shared_split_date.date()}")

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
    print(f"[long] train={len(tr0[2])} val={len(va0[2])} backtest_pool={len(pool_base)}")

    tmpdir = tempfile.mkdtemp(prefix="regime_blend_long_baseline_")
    shared_data_path = os.path.join(tmpdir, "shared_data.pkl")
    pss.dump_shared_training_data(
        shared_data_path,
        {"tr_base": tr0, "va_base": va0, "s_cols_base": m.BASELINE_STOCK_COLS, "m_cols_base": m.BASELINE_MACRO_COLS},
    )

    seeds = m.SEEDS
    seed_and_paths = [(s, os.path.join(tmpdir, f"state_{s}.pt")) for s in seeds]
    jobs = [("base", s, shared_data_path, out_path) for s, out_path in seed_and_paths]

    print(f"[long] seeds={seeds} を最大{m.PARALLEL_MAX_CONCURRENT}並列で学習します...")
    t0 = time.time()
    pss.run_jobs_parallel(jobs, os.path.abspath(pss.__file__), m.PARALLEL_MAX_CONCURRENT)
    print(f"[long] 全{len(seeds)}seed学習完了: 実時間={time.time() - t0:.1f}秒")

    models = []
    for seed, out_path in seed_and_paths:
        model = m.build_model(m.BASELINE_STOCK_COLS, m.BASELINE_MACRO_COLS)
        model.load_state_dict(torch.load(out_path, map_location=m.DEVICE))
        models.append(model)

    d_ens = m.run_backtest_inference_ensemble(models, pool_base)
    del models
    torch.cuda.empty_cache()

    try:
        shutil.rmtree(tmpdir)
    except Exception as e:
        print(f"[!] 一時ディレクトリを削除できませんでした: {tmpdir} ({e})")

    return d_ens, close_price_lookup


def main():
    import stage3_exp.experiment_context as m

    print("=" * 90)
    print("【確認1: 空売りモデルv2、TopN(1〜5)を同時保有制約込みで比較】")
    print("=" * 90)
    d_ens_short, close_lookup_short = build_short_d_ens()

    rows = []
    for n in [1, 2, 3, 4, 5]:
        r = m.evaluate_daily_topn(
            d_ens_short, f"空売りv2 daily-top{n}", n_per_day=n, print_detail=False,
            min_reliable_n=m.MIN_RELIABLE_N, max_concurrent=m.MAX_CONCURRENT_POSITIONS,
            close_price_lookup=close_lookup_short,
        )
        r["n_per_day"] = n
        rows.append(r)
        print(f"  top{n}: n={r['n']:4d} win_rate={r['win_rate']:.1f}% PF={r['pf']:.3f} "
              f"avg_ret={r['avg_ret']:+.4f} MaxDD={r['max_dd']*100:.1f}% HHI={r['hhi']:.3f}")

    print("\n" + "=" * 90)
    print("【確認2: ロング(baseline top5) vs ショートv2 の日次資金曲線・相関】")
    print("=" * 90)
    d_ens_long, close_lookup_long = build_long_d_ens()

    long_selected = select_daily_topn(d_ens_long, m.DAILY_TOPN)
    short_selected = select_daily_topn(d_ens_short, 5)

    eq_long = daily_equity_curve(long_selected, m.MAX_CONCURRENT_POSITIONS, close_lookup_long)
    eq_short = daily_equity_curve(short_selected, m.MAX_CONCURRENT_POSITIONS, close_lookup_short)

    print(f"[long] equity curve: {eq_long.index.min().date()} 〜 {eq_long.index.max().date()} ({len(eq_long)}日)")
    print(f"[short] equity curve: {eq_short.index.min().date()} 〜 {eq_short.index.max().date()} ({len(eq_short)}日)")

    common_idx = eq_long.index.intersection(eq_short.index)
    print(f"[共通期間] {common_idx.min().date()} 〜 {common_idx.max().date()} ({len(common_idx)}日)")

    ret_long = eq_long.reindex(common_idx).pct_change().dropna()
    ret_short = eq_short.reindex(common_idx).pct_change().dropna()
    common_ret_idx = ret_long.index.intersection(ret_short.index)
    ret_long = ret_long.reindex(common_ret_idx)
    ret_short = ret_short.reindex(common_ret_idx)

    corr = ret_long.corr(ret_short)
    print(f"\n[日次リターン相関(ロングtop5 vs ショートv2 top5)] r={corr:.3f} (n={len(common_ret_idx)}日)")

    # ロング側の資金曲線からドローダウン(ピーク比)を計算し、深いDD局面でショート側が
    # プラスに効いているか(ヘッジとして機能しているか)を見る。
    eq_long_common = eq_long.reindex(common_idx).ffill()
    peak = eq_long_common.cummax()
    dd_long = (eq_long_common - peak) / peak

    dd_threshold = -0.03
    deep_dd_dates = dd_long[dd_long <= dd_threshold].index
    if len(deep_dd_dates) > 0:
        short_ret_during_long_dd = ret_short.reindex(deep_dd_dates).dropna()
        long_ret_during_long_dd = ret_long.reindex(deep_dd_dates).dropna()
        print(f"\n[ロングがDD<={dd_threshold*100:.0f}%の日({len(deep_dd_dates)}日)における日次リターン]")
        print(f"  ロング平均: {long_ret_during_long_dd.mean():+.4f} / ショート平均: {short_ret_during_long_dd.mean():+.4f}")
        print(f"  ショート勝率(その日プラス): {(short_ret_during_long_dd > 0).mean()*100:.1f}%")
    else:
        print(f"\n[ロングのDDが{dd_threshold*100:.0f}%を下回る日は共通期間内に無し]")

    out_df = pd.DataFrame({"eq_long_top5": eq_long, "eq_short_top5": eq_short})
    out_path = os.path.join(m.CACHE_DIR, "_cache_long_short_equity_curves.pkl") if hasattr(m, "CACHE_DIR") else \
        r"F:\stockModel\research\cache\_cache_long_short_equity_curves.pkl"
    out_df.to_pickle(out_path)
    print(f"\n[+] 資金曲線を保存: {out_path}")


if __name__ == "__main__":
    main()
