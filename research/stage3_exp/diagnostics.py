"""Top-N and overlap diagnostics."""

import numpy as np
import pandas as pd
from .config import MAX_CONCURRENT_POSITIONS

def calc_pf(ret):
    profit = ret[ret > 0].sum()
    loss = -ret[ret < 0].sum()

    if loss <= 0:
        return np.inf

    return profit / loss

def calc_cagr(selected, max_concurrent=MAX_CONCURRENT_POSITIONS):
    """
    各取引へ総資金の1/max_concurrentを配分し、
    exit_dateに損益を反映する簡易ポートフォリオCAGR。

    selectedに必要な列:
        date, entry_date, exit_date, ret_pct
    """
    if selected.empty:
        return np.nan

    work = selected.copy()
    work["entry_date"] = pd.to_datetime(work["entry_date"])
    work["exit_date"] = pd.to_datetime(work["exit_date"])

    start_date = work["entry_date"].min()
    end_date = work["exit_date"].max()
    elapsed_days = (end_date - start_date).days

    if elapsed_days <= 0:
        return np.nan

    # 1取引あたり総資金の1/max_concurrentを配分
    work["portfolio_pnl"] = (
        work["ret_pct"] / float(max_concurrent)
    )

    # 決済日単位でポートフォリオ損益を集約
    daily_return = (
        work.groupby("exit_date")["portfolio_pnl"]
        .sum()
        .sort_index()
    )

    # 決済日に確定損益を複利反映
    ending_equity = (1.0 + daily_return).prod()

    if ending_equity <= 0:
        return -1.0

    return ending_equity ** (365.25 / elapsed_days) - 1.0

def evaluate_topn_curve(df, topn_list=(1, 3, 5, 10, 20, 30)):
    rows = []

    for top_n in topn_list:

        selected = (
            df.sort_values(
                ["date", "score"],
                ascending=[True, False]
            )
            .groupby("date")
            .head(top_n)
            .copy()
        )

        selected["month"] = (
            pd.to_datetime(selected["date"])
            .dt.to_period("M")
        )

        monthly_pf = (
            selected
            .groupby("month")["ret_pct"]
            .apply(calc_pf)
        )

        cagr = calc_cagr(
            selected,
            max_concurrent=MAX_CONCURRENT_POSITIONS,
        )

        rows.append({
            "top_n": top_n,
            "trades": len(selected),
            "pf": calc_pf(selected["ret_pct"]),
            "avg_ret": selected["ret_pct"].mean(),
            "cagr": cagr,
            "worst_month_pf": monthly_pf.min(),
            "median_month_pf": monthly_pf.median(),
            "pf_lt_05_months": int((monthly_pf < 0.5).sum()),
        })

    return pd.DataFrame(rows)

def monthly_topn_pf_matrix(df, topn_list=(1, 3, 5, 10, 20, 30)):
    monthly_results = {}

    for top_n in topn_list:
        selected = (
            df.sort_values(
                ["date", "score", "ticker"],
                ascending=[True, False, True],
            )
            .groupby("date", sort=False)
            .head(top_n)
            .copy()
        )

        selected["month"] = pd.to_datetime(
            selected["date"]
        ).dt.to_period("M")

        monthly_results[f"top_{top_n}"] = (
            selected.groupby("month")["ret_pct"]
            .apply(calc_pf)
        )

    result = pd.DataFrame(monthly_results)
    result["mean_pf"] = result.mean(axis=1)
    result["min_pf"] = result.min(axis=1)

    return result.sort_values("mean_pf")

def daily_detail_for_month(df, target_month, top_n=5):
    selected = (
        df.sort_values(
            ["date", "score", "ticker"],
            ascending=[True, False, True],
        )
        .groupby("date", sort=False)
        .head(top_n)
        .copy()
    )

    selected["date"] = pd.to_datetime(selected["date"])
    selected["month"] = selected["date"].dt.to_period("M")

    month_df = selected[
        selected["month"] == pd.Period(target_month)
    ].copy()

    daily = (
        month_df.groupby("date")
        .agg(
            trades=("ret_pct", "size"),
            total_ret=("ret_pct", "sum"),
            avg_ret=("ret_pct", "mean"),
            wins=("ret_pct", lambda x: int((x > 0).sum())),
            losses=("ret_pct", lambda x: int((x < 0).sum())),
        )
    )

    daily["win_rate"] = daily["wins"] / daily["trades"]
    daily["pf"] = (
        month_df.groupby("date")["ret_pct"]
        .apply(calc_pf)
    )

    return daily.sort_values("total_ret")

def evaluate_daily_topn_overlap(
    df,
    top_n=5,
    lookback_days=5,
    target_month=None,
):
    """
    日別TopNの銘柄重複を集計する。

    Parameters
    ----------
    df : pd.DataFrame
        必須列:
        date, ticker, score, ret_pct

    top_n : int
        日ごとに選択する上位銘柄数。

    lookback_days : int
        直近何取引日との重複を調べるか。

    target_month : str or None
        例: "2026-07"
        指定時は、TopN選択後に対象月だけ表示する。
        前日・過去期間との重複計算は全期間を使う。

    Returns
    -------
    pd.DataFrame
    """
    required_cols = {
        "date",
        "ticker",
        "score",
        "ret_pct",
    }
    missing_cols = required_cols - set(df.columns)

    if missing_cols:
        raise ValueError(
            f"必要な列が不足しています: {sorted(missing_cols)}"
        )

    work = df.copy()
    work["date"] = pd.to_datetime(work["date"])

    # 同点時も結果が再現可能になるようtickerをtie-breakに使用
    selected = (
        work.sort_values(
            ["date", "score", "ticker"],
            ascending=[True, False, True],
        )
        .groupby("date", sort=True)
        .head(top_n)
        .copy()
    )

    grouped = list(
        selected.groupby("date", sort=True)
    )

    rows = []
    history = []

    for date, group in grouped:
        current_tickers = set(group["ticker"])

        # 直前の取引日との重複
        if history:
            previous_tickers = history[-1]["tickers"]
            previous_overlap = (
                current_tickers & previous_tickers
            )
        else:
            previous_overlap = set()

        # 直近lookback_days取引日に一度でも入った銘柄との重複
        recent_history = history[-lookback_days:]

        recent_union = set()
        for item in recent_history:
            recent_union.update(item["tickers"])

        lookback_overlap = (
            current_tickers & recent_union
        )

        # 日次PF
        daily_pf = calc_pf(group["ret_pct"])

        rows.append({
            "date": date,
            "trades": len(group),
            "tickers": ",".join(
                sorted(map(str, current_tickers))
            ),
            "previous_overlap_count": len(
                previous_overlap
            ),
            "previous_overlap_ratio": (
                len(previous_overlap) / len(current_tickers)
                if current_tickers else np.nan
            ),
            "lookback_overlap_count": len(
                lookback_overlap
            ),
            "lookback_overlap_ratio": (
                len(lookback_overlap) / len(current_tickers)
                if current_tickers else np.nan
            ),
            "avg_score": group["score"].mean(),
            "min_score": group["score"].min(),
            "max_score": group["score"].max(),
            "total_ret": group["ret_pct"].sum(),
            "avg_ret": group["ret_pct"].mean(),
            "wins": int((group["ret_pct"] > 0).sum()),
            "losses": int((group["ret_pct"] < 0).sum()),
            "win_rate": (
                (group["ret_pct"] > 0).mean()
            ),
            "pf": daily_pf,
        })

        history.append({
            "date": date,
            "tickers": current_tickers,
        })

    result = pd.DataFrame(rows)

    if target_month is not None:
        target_period = pd.Period(
            target_month,
            freq="M",
        )

        result = result[
            result["date"].dt.to_period("M")
            == target_period
        ].copy()

    return result.reset_index(drop=True)
