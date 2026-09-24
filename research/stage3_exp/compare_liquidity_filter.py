"""流動性フィルタ(直近20日平均売買代金≥10億円)を適用したバックテストの効果を検証する
実験(2026-09-23、ユーザー提案「実運用に近い直近売買代金10億円以上の銘柄でバックテスト
してみたい」)。

再学習はせず、既存のbaseline_model_cache(BASELINE_STOCK_COLS/BASELINE_MACRO_COLS、
5seed)をそのまま使う——モデルは変えず、バックテスト時の候補プール(d_ens)を日次で
「その銘柄のその日時点の直近20日平均売買代金(Close×Volume)が10億円未満なら除外」で
事後フィルタしてから、フィルタ無し版と同じTOPN別PF分析(evaluate_topn_curve)で比較する。
学習データやモデルは一切変えないので、セクター/クラスタ制約実験と同じく「見せ方(選抜方法)
の違い」の検証であり、モデルの再訓練を伴う[[universe_expansion_300]]とはスコープが異なる。

実行方法:
    python research/stage3_exp/compare_liquidity_filter.py
"""
import sys
import time

sys.path.insert(0, r"F:\stockModel")
sys.path.insert(0, r"F:\stockModel\research")

try:
    sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")
    sys.stderr.reconfigure(line_buffering=True, encoding="utf-8")
except Exception:
    pass

TURNOVER_WINDOW = 20
TURNOVER_THRESHOLD = 1_000_000_000  # 10億円


def _print_topn_compare(m, title, d_all, d_liquid, **kwargs):
    import pandas as pd

    cols = [
        "top_n", "trades", "pf", "avg_ret", "cagr", "max_dd",
        "worst_month_pf", "median_month_pf", "pf_lt_05_months",
    ]
    all_df = m.evaluate_topn_curve(
        d_all, topn_list=[1, 3, 5, 10, 20, 30], **kwargs,
    )[cols]
    liquid_df = m.evaluate_topn_curve(
        d_liquid, topn_list=[1, 3, 5, 10, 20, 30], **kwargs,
    )[cols]
    diff_df = pd.DataFrame({
        "top_n": all_df["top_n"].values,
        "pf_all": all_df["pf"].values,
        "pf_liquid": liquid_df["pf"].values,
        "pf_diff": liquid_df["pf"].values - all_df["pf"].values,
    })
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)
    print("  -- 全候補(流動性フィルタ無し) --")
    print(all_df.to_string(index=False))
    print("  -- 流動性フィルタ後(直近20日平均売買代金≥10億円) --")
    print(liquid_df.to_string(index=False))
    print("  -- 差分(pf_liquid - pf_all) --")
    print(diff_df.to_string(index=False))


def main():
    import os
    import torch
    import pandas as pd
    import stage3_exp.experiment_context as m
    import parallel_seed_sweep as pss

    with open(m.UNIVERSE_PATH, "r", encoding="utf-8") as f:
        tickers = [line.strip() for line in f if line.strip()]
    print(f"[*] ユニバース: {len(tickers)}銘柄 ({m.UNIVERSE_PATH})")
    print(f"[*] 流動性フィルタ: 直近{TURNOVER_WINDOW}日平均売買代金 >= {TURNOVER_THRESHOLD:,}円")

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

    # baseline_model_cacheの再利用(既存のexperiment_runner.py/greedy_feature_search.pyと
    # 同じフィンガープリント機構——特徴量・split dateが同じなら再学習しない)。
    fp = m.baseline_cache_fingerprint(len(tr0[2]), len(va0[2]), shared_split_date)
    cache_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "cache", "baseline_model_cache", fp
    )
    os.makedirs(cache_dir, exist_ok=True)
    cached_seeds = {s for s in m.SEEDS if os.path.exists(os.path.join(cache_dir, f"state_{s}.pt"))}
    missing = [s for s in m.SEEDS if s not in cached_seeds]
    if cached_seeds:
        print(f"[*] baselineキャッシュ命中(fingerprint={fp}): seed={sorted(cached_seeds)}")
    if missing:
        print(f"[*] baseline未キャッシュのseed={missing}を学習します...")
        import tempfile, shutil
        tmpdir = tempfile.mkdtemp(prefix="compare_liquidity_filter_")
        shared_data_path = os.path.join(tmpdir, "shared_data.pkl")
        pss.dump_shared_training_data(shared_data_path, {
            "tr_base": tr0, "va_base": va0,
            "s_cols_base": m.BASELINE_STOCK_COLS, "m_cols_base": m.BASELINE_MACRO_COLS,
        })
        jobs = [("base", s, shared_data_path, os.path.join(tmpdir, f"state_{s}.pt")) for s in missing]
        t0 = time.time()
        pss.run_jobs_parallel(jobs, os.path.abspath(pss.__file__), m.PARALLEL_MAX_CONCURRENT)
        print(f"[+] {len(missing)}seed学習完了: 実時間={time.time() - t0:.1f}秒")
        for s in missing:
            shutil.copy2(os.path.join(tmpdir, f"state_{s}.pt"), os.path.join(cache_dir, f"state_{s}.pt"))
        try:
            shutil.rmtree(tmpdir)
        except Exception:
            pass

    models = []
    for s in m.SEEDS:
        model = m.build_model(m.BASELINE_STOCK_COLS, m.BASELINE_MACRO_COLS)
        model.load_state_dict(torch.load(os.path.join(cache_dir, f"state_{s}.pt"), map_location=m.DEVICE))
        model.eval()
        models.append(model)

    d_ens = m.run_backtest_inference_ensemble(models, pool_base)

    # 銘柄ごとの直近TURNOVER_WINDOW日平均売買代金(Close×Volume)を(ticker,date)で
    # d_ensへ結合する。poolは各銘柄dfがOpen/High/Low/Close/Volumeを保持している前提
    # (get_base_per_ticker_dfのdocstring参照)。
    turnover_rows = []
    for t, df in pool.items():
        s = (df["Close"] * df["Volume"]).rolling(TURNOVER_WINDOW).mean()
        turnover_rows.append(pd.DataFrame({
            "ticker": t, "date": s.index, "turnover_20d": s.values,
        }))
    turnover_df = pd.concat(turnover_rows, ignore_index=True)
    d_ens = d_ens.merge(turnover_df, on=["ticker", "date"], how="left")

    n_missing_turnover = d_ens["turnover_20d"].isna().sum()
    if n_missing_turnover:
        print(f"[!] turnover_20dが引けなかった候補: {n_missing_turnover}/{len(d_ens)}件(除外扱い)")

    d_liquid = d_ens[d_ens["turnover_20d"] >= TURNOVER_THRESHOLD].copy()
    print(
        f"[*] 候補行数: 全体={len(d_ens)} 流動性フィルタ後={len(d_liquid)} "
        f"({len(d_liquid) / len(d_ens) * 100:.1f}%)"
    )
    per_day_liquid_n = d_liquid.groupby("date").size()
    print(
        f"[*] 日次の流動性適格銘柄数: min={per_day_liquid_n.min()} "
        f"median={per_day_liquid_n.median():.0f} max={per_day_liquid_n.max()}"
    )
    qualifying_tickers = sorted(d_liquid["ticker"].unique())
    print(f"[*] 一度でも適格になった銘柄数: {len(qualifying_tickers)}/{len(tickers)}")

    _print_topn_compare(
        m, "【TOPN別PF分析(流動性フィルタ有無、制約無し)】", d_ens, d_liquid,
        close_price_lookup=close_price_lookup,
    )
    _print_topn_compare(
        m, f"【TOPN別PF分析(流動性フィルタ有無、業種制約 max_per_sector={m.MAX_PER_SECTOR})】",
        d_ens, d_liquid, max_per_sector=m.MAX_PER_SECTOR, close_price_lookup=close_price_lookup,
    )
    if m.TICKER_TO_CLUSTER:
        _print_topn_compare(
            m, f"【TOPN別PF分析(流動性フィルタ有無、クラスタ制約 max_per_cluster={m.MAX_PER_CLUSTER})】",
            d_ens, d_liquid, max_per_cluster=m.MAX_PER_CLUSTER, close_price_lookup=close_price_lookup,
        )

    del models
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
