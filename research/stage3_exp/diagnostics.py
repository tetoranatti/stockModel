"""Evaluation metrics, regime helpers, Top-N and overlap diagnostics."""

import numpy as np
import pandas as pd

from .config import (
    COMMON_SAMPLE_K,
    MAX_CONCURRENT_POSITIONS,
    MIN_RELIABLE_N,
)
from _feature_cache_utils import _load_sector_map

TICKER_TO_SECTOR = _load_sector_map()


def calc_pf(data):
    """PF（プロフィットファクター）を返す。

    ``ret_pct`` Seriesまたは``ret_pct``列を持つDataFrameを受け取る。
    空データはNaN、損失合計が0以下ならinfを返す。
    """
    if isinstance(data, pd.DataFrame):
        ret = data["ret_pct"]
    else:
        ret = data

    if len(ret) == 0:
        return np.nan

    profit = ret[ret > 0].sum()
    loss = -ret[ret < 0].sum()
    if loss <= 0:
        return np.inf
    return profit / loss


# ============================================================================
# Core evaluation and regime helpers
# ============================================================================


def compute_max_drawdown(sub, max_concurrent=MAX_CONCURRENT_POSITIONS):
    """同時保有上限max_concurrent・均等配分の資金曲線シミュレーションで最大ドローダウンを
    算出する(本番backtest/run_event_driven_backtest_v8_exp.pyと同じロジック、2026-09-19追加)。

    entry_date(D+1引け、2026-09-19修正——以前はdate=シグナル当日を使っており、実際の
    約定日より1日早いタイミングで資金を拘束していた)を使う。

    戻り値: (max_dd, executed_positions)。max_ddは負の値(例: -0.128 = -12.8%)。
    executed_positionsはsub内で実際に枠が空いて約定した行の0始まり位置のlist
    (2026-09-19追加、ユーザー指摘「同時保有制約後の同じ約定集合でPF・勝率・MaxDDを
    計算」——枠が無くて約定しなかったトレードもPF/win_rateには数えてしまうと、
    MaxDDだけ現実的な制約を反映し、PF/win_rateは反映しないという矛盾が生じるため、
    呼び出し側でこの集合を使ってPF/win_rateも計算し直す)。"""
    if len(sub) == 0:
        return float("nan"), []
    events = []
    for pos, row in enumerate(sub.itertuples(index=False)):
        events.append((row.entry_date, 1, pos, row.ret_pct))
        events.append((row.exit_date, 0, pos, row.ret_pct))
    events.sort(key=lambda e: (e[0], e[1]))

    equity, open_slots, equity_curve = 1.0, {}, [1.0]
    executed_positions = []
    for _, kind, pos, ret_pct in events:
        if kind == 0:
            size = open_slots.pop(pos, None)
            if size is not None:
                equity += size * ret_pct
                equity_curve.append(equity)
        else:
            if len(open_slots) < max_concurrent:
                open_slots[pos] = equity / max_concurrent
                executed_positions.append(pos)

    peak, max_dd = -float("inf"), 0.0
    for eq in equity_curve:
        peak = max(peak, eq)
        if peak > 0:
            max_dd = min(max_dd, (eq - peak) / peak)
    return max_dd, executed_positions


def compute_concentration(sub):
    """特定銘柄への集中度(2026-09-19追加)。上位1銘柄・上位5銘柄がトレード件数に占める
    割合(%)と、HHI(Herfindahl指数、0〜1・1に近いほど少数銘柄に集中)を返す。"""
    if len(sub) == 0:
        return float("nan"), float("nan"), float("nan")
    shares = sub["ticker"].value_counts() / len(sub)
    top1_pct = shares.iloc[0] * 100
    top5_pct = shares.iloc[:5].sum() * 100
    hhi = (shares**2).sum()
    return top1_pct, top5_pct, hhi


def compute_expanding_vol_regime_labels(macro_pool_df, min_periods=120):
    """LOW/MID/HIGH/UNKNOWNを同一の実現ボラ系列・同一関数から一括生成する(2026-09-19、
    ユーザー指摘「LOW/MID/HIGHを同一関数、同一ボラ系列から生成」)。以前はis_low
    (production regime_risk_model.py::_compute_vol_regime_is_lowを呼ぶ)とis_high
    (realized_volをここで再計算)が別々に実現ボラを計算しており、どちらかの定義だけ
    将来変更されても気づかないまま食い違うリスクがあった。realized_vol
    (=NK_Ret.abs().rolling(20).mean()、production _compute_vol_regime_is_lowと同じ式)を
    ここで1回だけ計算し、拡大窓の1/3・2/3分位点から3値を導出する。

    拡大窓の分位点がまだ計算できない最初のmin_periods日は'UNKNOWN'にする(2026-09-19、
    ユーザー指摘「初期120日をMIDではなくUNKNOWNにする」——以前はLOW/HIGHどちらでもない
    日を一律'MID'にしており、単にウォームアップで未分類なだけの日が本物のMID(中間ボラ)
    と区別できなかった)。REGIME_ID_MAPに'UNKNOWN'は無いため、NearPairRankingLoss
    RegimeAwareのペアタグ付けでは自動的に-1(常に除外)扱いになる。

    戻り値: date->'LOW'/'MID'/'HIGH'/'UNKNOWN'のpd.Series。"""
    realized_vol = macro_pool_df["NK_Ret"].abs().rolling(20).mean()
    q1_exp = realized_vol.expanding(min_periods=min_periods).quantile(1 / 3)
    q2_exp = realized_vol.expanding(min_periods=min_periods).quantile(2 / 3)
    known = q1_exp.notna() & q2_exp.notna()
    labels = pd.Series("UNKNOWN", index=macro_pool_df.index)
    labels[known & (realized_vol < q1_exp)] = "LOW"
    labels[known & (realized_vol >= q1_exp) & (realized_vol < q2_exp)] = "MID"
    labels[known & (realized_vol >= q2_exp)] = "HIGH"
    return labels


def compute_is_low_regime_filter(macro_pool_df, min_periods=120):
    """regime限定モデルの学習/バックテストのサンプル選別用(2026-09-19追加、後にcompute_
    expanding_vol_regime_labelsの薄いラッパーに統一)。date->bool(True=LOWレジーム)。"""
    return compute_expanding_vol_regime_labels(macro_pool_df, min_periods) == "LOW"


def compute_is_high_regime_filter(macro_pool_df, min_periods=120):
    """compute_is_low_regime_filterのHIGH版。"""
    return compute_expanding_vol_regime_labels(macro_pool_df, min_periods) == "HIGH"


def compute_is_mid_regime_filter(macro_pool_df, min_periods=120):
    """compute_is_low_regime_filterのMID版(拡大窓ウォームアップ中のUNKNOWN日は含まない)。"""
    return compute_expanding_vol_regime_labels(macro_pool_df, min_periods) == "MID"


def compute_regime_labels(macro_pool_df):
    """market_vol_regime(市場実現ボラの60日z-score、キャッシュ済み)の全期間3分位で
    date->'LOW'/'MID'/'HIGH'のマッピングを作る(2026-09-19追加、ユーザー提案の
    レジーム別評価用)。診断目的の事後集計であり、本番のregime_risk_model
    (modules/regime_risk_model.py::_compute_vol_regime_is_low)が使う拡大窓版とは異なり、
    全期間固定の分位点を使う(未来情報を使うため学習特徴量には使えないが、過去の
    バックテスト結果をレジーム別に見るだけの集計には問題ない)。"""
    vol = macro_pool_df["market_vol_regime"]
    q1, q2 = vol.quantile([1 / 3, 2 / 3])
    labels = pd.Series("MID", index=macro_pool_df.index)
    labels[vol < q1] = "LOW"
    labels[vol >= q2] = "HIGH"
    return labels


def compute_regime_labels_expanding(macro_pool_df, min_periods=120):
    """compute_regime_labelsのlook-ahead無し版。compute_regime_labelsは全期間固定の
    分位点を使うため診断・事後集計には十分だが、NearPairRankingLossRegimeAware
    (学習に使う)のペアタグ付けにそのまま使うと訓練プロセス自体に未来情報が混ざる。
    compute_expanding_vol_regime_labelsの薄いラッパー(2026-09-19、同一関数への統一)。"""
    return compute_expanding_vol_regime_labels(macro_pool_df, min_periods=min_periods)


def regime_breakdown(sub, label, regime_labels, min_reliable_n=MIN_RELIABLE_N):
    """monthly_breakdownと同じ形式で、月の代わりにボラティリティregime(LOW/MID/HIGH)
    別にn・勝率・PFを出す。少数サンプルのバケットはmonthly_breakdownと同じ閾値
    (min_reliable_n)で警告する。"""
    sub = sub.copy()
    sub["regime"] = pd.to_datetime(sub["date"]).map(regime_labels)
    print(f"\n  [regime別内訳: {label}]")
    for regime in ["LOW", "MID", "HIGH"]:
        g = sub[sub["regime"] == regime]
        if len(g) == 0:
            print(f"    {regime}: n=0")
            continue
        win_rate = len(g[g["ret_pct"] > 0]) / len(g) * 100
        pf = calc_pf(g)
        flag = (
            f" [!少数トレード(<{min_reliable_n}件)]" if len(g) < min_reliable_n else ""
        )
        print(
            f"    {regime}: n={len(g):4d} win_rate={win_rate:5.1f}% PF={pf:5.2f}{flag}"
        )


def monthly_breakdown(sub, label, verbose=True):
    """verbose=Falseだと表を印字せずconsistencyだけ計算して返す(2026-09-19追加、
    ユーザー指摘「month_consistencyを常に返す」——evaluate()がprint_detail=Falseの
    ときもmonth_consistencyを結果dictに含められるようにするため)。"""
    sub = sub.copy()
    sub["month"] = pd.to_datetime(sub["date"]).dt.to_period("M")
    if verbose:
        print(f"\n  [月別内訳: {label}]")
    n_positive, n_total = 0, 0
    for month, g in sub.groupby("month"):
        win_rate = len(g[g["ret_pct"] > 0]) / len(g) * 100
        pf = calc_pf(g)
        n_total += 1
        if pf > 1.0:
            n_positive += 1
        if verbose:
            print(f"    {month}: n={len(g):4d} win_rate={win_rate:5.1f}% PF={pf:5.2f}")
    consistency = (n_positive / n_total * 100) if n_total else float("nan")
    if verbose:
        print(
            f"    -> 月別一貫性: {n_positive}/{n_total}ヶ月でPF>1 ({consistency:.0f}%)"
        )
    return consistency


def _evaluate_from_selection(
    sub_selected, label, print_detail, min_reliable_n, max_concurrent
):
    """evaluate()とevaluate_daily_topn()の共通後半処理(2026-09-19追加、日次Top-N評価を
    足す際に重複を避けるため抽出)。sub_selectedは呼び出し側が既に選抜済み(全期間
    score上位K件、または日次score上位N件)の候補集合。

    2段階構成: 1. sub_selected自体のPFがtop_k_pf/top_k_n(同時保有制約を無視した
    「純粋なランキング品質」)。2. 同時保有上限max_concurrentの資金曲線シミュレーション
    (compute_max_drawdown、entry_date=D+1引けを使用)で実際に枠が空いて約定した集合
    だけを使い、n/win_rate/pf/avg_ret/月別一貫性/銘柄集中度を計算し直す(2026-09-19、
    ユーザー指摘「同時保有制約後の同じ約定集合でPF・勝率・MaxDDを計算」)——制約を
    無視したPFと制約込みのMaxDDが矛盾しないようにするため。pfとtop_k_pfが大きく
    乖離する場合、同時保有枠がボトルネックになっている(機会損失がある)ことを示す。

    month_consistencyはprint_detailの値に関わらず常に結果dictに含める(2026-09-19、
    ユーザー指摘。以前はprint_detail=Falseのとき欠落していた)。"""
    top_k_n = len(sub_selected)
    top_k_pf = calc_pf(sub_selected)

    if top_k_n == 0:
        if print_detail:
            print(f"  {label}: n=0")
        return dict(
            label=label,
            n=0,
            win_rate=np.nan,
            pf=np.nan,
            avg_ret=np.nan,
            max_dd=np.nan,
            top_k_pf=top_k_pf,
            top_k_n=top_k_n,
            month_consistency=np.nan,
            n_months=0,
            top1_pct=np.nan,
            top5_pct=np.nan,
            hhi=np.nan,
            few_trades_flag=True,
        )

    max_dd, executed_positions = compute_max_drawdown(
        sub_selected, max_concurrent=max_concurrent
    )
    sub = (
        sub_selected.iloc[executed_positions]
        if executed_positions
        else sub_selected.iloc[0:0]
    )
    n = len(sub)

    if n == 0:
        if print_detail:
            print(
                f"  {label}: n=0(同時保有枠に一件も入らず) top_k_pf(制約無視、上位{top_k_n}件)={top_k_pf:.2f}"
            )
        return dict(
            label=label,
            n=0,
            win_rate=np.nan,
            pf=np.nan,
            avg_ret=np.nan,
            max_dd=max_dd,
            top_k_pf=top_k_pf,
            top_k_n=top_k_n,
            month_consistency=np.nan,
            n_months=0,
            top1_pct=np.nan,
            top5_pct=np.nan,
            hhi=np.nan,
            few_trades_flag=True,
        )

    win_rate = len(sub[sub["ret_pct"] > 0]) / n * 100
    pf = calc_pf(sub)
    avg_ret = sub["ret_pct"].mean()
    top1_pct, top5_pct, hhi = compute_concentration(sub)
    few_trades_flag = n < min_reliable_n
    consistency = monthly_breakdown(sub, label, verbose=print_detail)

    if print_detail:
        flag_str = f" [!少数トレード(<{min_reliable_n}件)]" if few_trades_flag else ""
        print(
            f"  {label}: n={n:5d}(約定、候補{top_k_n}件中) win_rate={win_rate:5.1f}% PF={pf:5.2f} "
            f"avg_ret={avg_ret:+.4f} MaxDD={max_dd*100:5.1f}%{flag_str}"
        )
        print(
            f"    top_k_pf(同時保有制約無視、候補{top_k_n}件)={top_k_pf:.2f} | "
            f"銘柄集中: 上位1銘柄{top1_pct:4.1f}% 上位5銘柄{top5_pct:4.1f}% HHI={hhi:.3f}"
        )

    return dict(
        label=label,
        n=n,
        win_rate=win_rate,
        pf=pf,
        avg_ret=avg_ret,
        max_dd=max_dd,
        top_k_pf=top_k_pf,
        top_k_n=top_k_n,
        top1_pct=top1_pct,
        top5_pct=top5_pct,
        hhi=hhi,
        few_trades_flag=few_trades_flag,
        month_consistency=consistency,
    )


def evaluate(
    d_all,
    label,
    print_detail=True,
    min_reliable_n=MIN_RELIABLE_N,
    common_sample_k=COMMON_SAMPLE_K,
    max_concurrent=MAX_CONCURRENT_POSITIONS,
):
    """標準指標(n・勝率・PF・平均リターン)に加え、top_k_pf・MaxDD・月別一貫性・
    少数トレードへの偏り・銘柄集中度までまとめて算出する(2026-09-19追加、ユーザー要望の
    9指標に対応、その後複数回の指摘で改修)。

    「全期間score上位common_sample_k件」を候補として選ぶ(p_winは絶対確率として校正
    されていないことをcalibration checkで確認済みのため、固定閾値ではなくscore
    (呼び出し側定義、既定はEV_WIN_WEIGHT*p_win-EV_STOP_WEIGHT*p_stop=本番のev_scoreと
    同じ2:1重み。stage3_toggle_experiment.py参照)のランキングで選ぶ)。
    同点処理は決定論的(2026-09-19、ユーザー指摘
    「Top-K同点処理を決定論的にする」): score降順→ticker昇順→date昇順で明示的に
    ソートしてから先頭K件を取る(pandas nlargestのkeep='first'は入力行順という
    暗黙の基準に依存するため)。

    全期間を通した選抜なので、シグナルが特定の数日に偏っていると実際には日々N件しか
    執行できない運用では再現できない結果になりうる——日々の執行に忠実な評価は
    evaluate_daily_topn()を使うこと(2026-09-19、ユーザー指摘「全期間Top-Kに加えて
    日次Top-Nを主運用評価として追加」)。"""
    top_k = min(common_sample_k, len(d_all))
    sub_selected = (
        d_all.sort_values(
            ["score", "ticker", "date"], ascending=[False, True, True]
        ).head(top_k)
        if top_k > 0
        else d_all.iloc[0:0]
    )
    return _evaluate_from_selection(
        sub_selected, label, print_detail, min_reliable_n, max_concurrent
    )


def evaluate_daily_topn(
    d_all,
    label,
    n_per_day=5,
    print_detail=True,
    min_reliable_n=MIN_RELIABLE_N,
    max_concurrent=MAX_CONCURRENT_POSITIONS,
):
    """日次Top-N運用評価(2026-09-19追加、ユーザー指摘「全期間Top-Kに加えて日次Top-Nを
    主運用評価として追加」)。evaluate()の「全期間score上位K件」は、シグナルが特定の
    数日に偏っていると、実際には毎日決まった件数しか執行できない運用では再現不可能な
    結果になりうる(例: ある1週間に候補200件中150件が集中していても、そんなに一度に
    建てられない)。日ごとにscore上位n_per_day件だけを選び、それを全期間通して
    積み上げてPF等を計算する——より運用に忠実な評価。同点処理はevaluate()と同じく
    決定論的(score降順→ticker昇順)。

    残りの指標(top_k_pf/MaxDD/月別一貫性/銘柄集中度等)の定義・同時保有制約の扱いは
    evaluate()と共通(_evaluate_from_selectionを共有)。"""
    if len(d_all) == 0:
        sub_selected = d_all
    else:
        sub_selected = (
            d_all.sort_values(
                ["date", "score", "ticker"], ascending=[True, False, True]
            )
            .groupby("date", group_keys=False)
            .head(n_per_day)
        )
    return _evaluate_from_selection(
        sub_selected, label, print_detail, min_reliable_n, max_concurrent
    )


# ============================================================================
# Top-N, overlap and sector diagnostics
# ============================================================================


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
    work["portfolio_pnl"] = work["ret_pct"] / float(max_concurrent)

    # 決済日単位でポートフォリオ損益を集約
    daily_return = work.groupby("exit_date")["portfolio_pnl"].sum().sort_index()

    # 決済日に確定損益を複利反映
    ending_equity = (1.0 + daily_return).prod()

    if ending_equity <= 0:
        return -1.0

    return ending_equity ** (365.25 / elapsed_days) - 1.0


def evaluate_topn_curve(
    df,
    topn_list=(1, 3, 5, 10, 20, 30),
    max_per_sector=None,
):
    rows = []

    for top_n in topn_list:
        if max_per_sector is None:
            # 従来のベースライン
            selected = (
                df.sort_values(
                    ["date", "score", "ticker"],
                    ascending=[True, False, True],
                )
                .groupby("date", sort=False)
                .head(top_n)
                .copy()
            )
        else:
            # セクター上限制御あり
            selected_parts = []

            for _, day_df in df.groupby("date", sort=False):
                day_selected = select_topn_with_sector_limit(
                    day_df,
                    top_n=top_n,
                    max_per_sector=max_per_sector,
                )
                selected_parts.append(day_selected)

            selected = (
                pd.concat(selected_parts, ignore_index=True)
                if selected_parts
                else df.iloc[:0].copy()
            )

        selected["month"] = pd.to_datetime(selected["date"]).dt.to_period("M")

        monthly_pf = selected.groupby("month")["ret_pct"].apply(calc_pf)

        cagr = calc_cagr(
            selected,
            max_concurrent=MAX_CONCURRENT_POSITIONS,
        )

        rows.append(
            {
                "top_n": top_n,
                "max_per_sector": max_per_sector,
                "trades": len(selected),
                "pf": calc_pf(selected["ret_pct"]),
                "avg_ret": selected["ret_pct"].mean(),
                "cagr": cagr,
                "worst_month_pf": monthly_pf.min(),
                "median_month_pf": monthly_pf.median(),
                "pf_lt_05_months": int((monthly_pf < 0.5).sum()),
            }
        )

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

        selected["month"] = pd.to_datetime(selected["date"]).dt.to_period("M")

        monthly_results[f"top_{top_n}"] = selected.groupby("month")["ret_pct"].apply(
            calc_pf
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

    month_df = selected[selected["month"] == pd.Period(target_month)].copy()

    daily = month_df.groupby("date").agg(
        trades=("ret_pct", "size"),
        total_ret=("ret_pct", "sum"),
        avg_ret=("ret_pct", "mean"),
        wins=("ret_pct", lambda x: int((x > 0).sum())),
        losses=("ret_pct", lambda x: int((x < 0).sum())),
    )

    daily["win_rate"] = daily["wins"] / daily["trades"]
    daily["pf"] = month_df.groupby("date")["ret_pct"].apply(calc_pf)

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
        raise ValueError(f"必要な列が不足しています: {sorted(missing_cols)}")

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

    grouped = list(selected.groupby("date", sort=True))

    rows = []
    history = []

    for date, group in grouped:
        current_tickers = set(group["ticker"])

        # 直前の取引日との重複
        if history:
            previous_tickers = history[-1]["tickers"]
            previous_overlap = current_tickers & previous_tickers
        else:
            previous_overlap = set()

        # 直近lookback_days取引日に一度でも入った銘柄との重複
        recent_history = history[-lookback_days:]

        recent_union = set()
        for item in recent_history:
            recent_union.update(item["tickers"])

        lookback_overlap = current_tickers & recent_union

        # 日次PF
        daily_pf = calc_pf(group["ret_pct"])

        rows.append(
            {
                "date": date,
                "trades": len(group),
                "tickers": ",".join(sorted(map(str, current_tickers))),
                "previous_overlap_count": len(previous_overlap),
                "previous_overlap_ratio": (
                    len(previous_overlap) / len(current_tickers)
                    if current_tickers
                    else np.nan
                ),
                "lookback_overlap_count": len(lookback_overlap),
                "lookback_overlap_ratio": (
                    len(lookback_overlap) / len(current_tickers)
                    if current_tickers
                    else np.nan
                ),
                "avg_score": group["score"].mean(),
                "min_score": group["score"].min(),
                "max_score": group["score"].max(),
                "total_ret": group["ret_pct"].sum(),
                "avg_ret": group["ret_pct"].mean(),
                "wins": int((group["ret_pct"] > 0).sum()),
                "losses": int((group["ret_pct"] < 0).sum()),
                "win_rate": ((group["ret_pct"] > 0).mean()),
                "pf": daily_pf,
            }
        )

        history.append(
            {
                "date": date,
                "tickers": current_tickers,
            }
        )

    result = pd.DataFrame(rows)

    if target_month is not None:
        target_period = pd.Period(
            target_month,
            freq="M",
        )

        result = result[result["date"].dt.to_period("M") == target_period].copy()

    return result.reset_index(drop=True)


def evaluate_sector_distribution(
    df,
    top_n=10,
):
    """
    日別TopNのセクター分布を集計

    必須列:
        date
        ticker
        sector
        score
    """

    selected = (
        df.sort_values(
            ["date", "score", "ticker"],
            ascending=[True, False, True],
        )
        .groupby("date", sort=False)
        .head(top_n)
        .copy()
    )

    daily_sector = selected.groupby(["date", "sector"]).size().reset_index(name="count")

    summary = (
        daily_sector.groupby("date")["count"]
        .agg(
            max_sector_count="max",
            sector_count="size",
            mean_sector_count="mean",
        )
        .reset_index()
    )

    summary["max_sector_ratio"] = summary["max_sector_count"] / top_n

    return summary


def select_topn_with_sector_limit(
    day_df,
    top_n=10,
    max_per_sector=2,
):
    """
    日次候補から、1セクター当たりの採用上限を適用してTop-Nを選択する。

    必須列:
        ticker
        score
    """

    required_cols = {"ticker", "score"}
    missing_cols = required_cols - set(day_df.columns)

    if missing_cols:
        raise ValueError(
            f"select_topn_with_sector_limit: "
            f"必要な列が不足しています: {sorted(missing_cols)}"
        )

    work = day_df.copy()

    # ticker型の不一致を避けるため、マップ側とDataFrame側を文字列へ統一
    normalized_sector_map = {
        str(ticker): sector for ticker, sector in TICKER_TO_SECTOR.items()
    }

    work["sector"] = (
        work["ticker"].astype(str).map(normalized_sector_map).fillna("UNKNOWN")
    )

    ordered = work.sort_values(
        ["score", "ticker"],
        ascending=[False, True],
        kind="mergesort",
    )

    selected_indices = []
    sector_counts = {}

    for idx, row in ordered.iterrows():
        sector = row["sector"]

        if sector_counts.get(sector, 0) >= max_per_sector:
            continue

        selected_indices.append(idx)
        sector_counts[sector] = sector_counts.get(sector, 0) + 1

        if len(selected_indices) >= top_n:
            break

    if not selected_indices:
        return ordered.iloc[:0].copy()

    return ordered.loc[selected_indices].copy()
