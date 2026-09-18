# modules/tracking.py
# ライブスクリーニングの推奨(ポートフォリオ採用銘柄)を実際の値動きと突き合わせて
# 記録するトラッキング機能。バックテストは同じ約11ヶ月分の過去データを何度も
# 再分析するだけで新しい情報が増えないため、実運用でしか得られない「本当の意味での
# アウトオブサンプル検証」をここで蓄積する。
import os
import datetime
import pandas as pd

BASE_DIR = r"F:\stockModel"
TRACKING_LOG_CSV = os.path.join(BASE_DIR, "data", "tracking_log.csv")

HOLDING_PERIOD_DAYS = 10  # PFモデルの学習・バックテストと同じ保有期間(営業日)

TRACKING_COLS = [
    "entry_date", "ticker", "action", "entry_price", "fill_date", "target_price", "stop_price",
    "prob_win", "prob_win_raw", "prob_stop", "ev_score", "recommended_shares",
    "estimated_cost_yen", "status", "exit_date", "exit_price", "exit_reason",
    "actual_ret_pct", "holding_days_elapsed",
]
# entry_date: シグナル当日(スクリーニング実行日、target/stop算出の基準終値の日)
# fill_date: 実際に約定した日(entry_dateの翌営業日の引成注文で、その日の大引けに約定する前提)
# entry_price: fill_date の終値(実際の約定価格)。シグナル当日終値ではない
#   (エントリー価格基準の検証により、終値で即約定可能という前提は非現実的である一方、
#   翌朝始値の成行だと寄り天で高値掴みするケースを拾ってしまいモデルのエッジが
#   大きく縮小することが判明。翌営業日に引成(大引け成行)注文を出す方式なら、
#   再学習不要で理想値に近い成績を維持できることをバックテストで確認したため、
#   本番の実行方式として採用。fill_date当日は寄り付きから引けまで一切ポジションを
#   持たないため、利確/損切判定はfill_date当日には行わず、翌日(fill_date+1)から開始する)
# status: AWAITING_ENTRY(シグナル済み・引成注文待ち) -> PENDING(約定済み・保有中) -> RESOLVED(手仕舞い済み)


def load_tracking_log():
    if not os.path.exists(TRACKING_LOG_CSV):
        return pd.DataFrame(columns=TRACKING_COLS)
    df = pd.read_csv(TRACKING_LOG_CSV, encoding='utf-8-sig')
    for col in TRACKING_COLS:
        if col not in df.columns:
            df[col] = None
    return df[TRACKING_COLS]


def save_tracking_log(df):
    os.makedirs(os.path.dirname(TRACKING_LOG_CSV), exist_ok=True)
    df.to_csv(TRACKING_LOG_CSV, index=False, encoding='utf-8-sig')


def append_new_entries(df, portfolio_rows, entry_date):
    """今日のポートフォリオ採用銘柄をAWAITING_ENTRY行(未約定)として追記する。
    entry_priceは翌営業日の引成(大引け成行)注文が約定するまで確定しない(resolve_entries()が埋める)。
    target_price/stop_priceはスクリーニング側で算出済みの絶対価格(当日終値基準)を
    そのまま使う(実際にOCO注文として出す価格そのものであり、約定価格が
    ずれても水準自体は動かさないのが実運用上自然なため)。
    同じ日・同じ銘柄が既にログにあれば重複追加しない(同日に複数回実行した場合の対策)。"""
    entry_date_str = pd.Timestamp(entry_date).strftime("%Y-%m-%d")
    existing_keys = set(zip(df["entry_date"].astype(str), df["ticker"].astype(str)))

    new_rows = []
    for p in portfolio_rows:
        key = (entry_date_str, str(p["ticker"]))
        if key in existing_keys:
            continue
        new_rows.append({
            "entry_date": entry_date_str,
            "ticker": p["ticker"],
            "action": p.get("action", ""),
            "entry_price": None,
            "fill_date": None,
            "target_price": float(p["target_price"]),
            "stop_price": float(p["stop_price"]),
            "prob_win": p.get("prob_win"),
            "prob_win_raw": p.get("prob_win_raw"),
            "prob_stop": p.get("prob_stop"),
            "ev_score": p.get("ev_score"),
            "recommended_shares": p.get("recommended_shares"),
            "estimated_cost_yen": p.get("estimated_cost_yen"),
            "status": "AWAITING_ENTRY",
            "exit_date": None, "exit_price": None, "exit_reason": None,
            "actual_ret_pct": None, "holding_days_elapsed": 0,
        })
    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    return df, len(new_rows)


def _get_latest_bar(all_prices_df, ticker, date):
    """all_prices_df(fetch_all_tickers_data()の戻り値)から指定銘柄・指定日の
    Open/High/Low/Closeを取得する。無ければNoneを返す。"""
    try:
        has_multi = isinstance(all_prices_df.columns, pd.MultiIndex)
        sub = all_prices_df[ticker] if has_multi else all_prices_df
        idx = pd.to_datetime(sub.index).tz_localize(None)
        sub = sub.set_axis(idx)
        target = pd.Timestamp(date)
        if target not in sub.index:
            return None
        row = sub.loc[target]
        if pd.isna(row.get("Open")) or pd.isna(row.get("High")) or pd.isna(row.get("Low")) or pd.isna(row.get("Close")):
            return None
        return {
            "open": float(row["Open"]), "high": float(row["High"]),
            "low": float(row["Low"]), "close": float(row["Close"]),
        }
    except Exception:
        return None


def resolve_entries(df, all_prices_df, today):
    """AWAITING_ENTRY行について、翌営業日に引成(大引け成行)注文を出した前提で、
    その日の終値をentry_price、その日をfill_dateとして確定する。寄り付きから
    引けまでポジションを一切持たない(引成注文はその日の大引けで初めて約定する)ため、
    fill_date当日は利確/損切判定を行わない。翌日以降の判定はresolve_pending()に委ねる
    (run_event_driven_backtest_v8_exp.pyの entry_p=closes[idx+1] と同じ約定規則)。"""
    today_str = pd.Timestamp(today).strftime("%Y-%m-%d")
    n_filled = 0

    for idx, row in df[df["status"] == "AWAITING_ENTRY"].iterrows():
        ticker = row["ticker"]
        entry_date = pd.Timestamp(row["entry_date"])
        if pd.Timestamp(today_str) <= entry_date:
            continue  # シグナル当日はまだ翌営業日になっていない

        bar = _get_latest_bar(all_prices_df, ticker, today_str)
        if bar is None:
            continue  # 翌営業日データがまだ無い(休日等) -> 次回実行時に再判定

        df.at[idx, "entry_price"] = bar["close"]
        df.at[idx, "fill_date"] = today_str
        df.at[idx, "holding_days_elapsed"] = 0
        df.at[idx, "status"] = "PENDING"
        n_filled += 1

    return df, n_filled


def resolve_pending(df, all_prices_df, today):
    """PENDING行(約定済み・保有中)について、今日のHigh/Lowが利確/損切ラインに
    到達したかを判定し、到達していれば状態を更新する。保有期間
    (HOLDING_PERIOD_DAYS営業日、約定日の翌日を1日目として起算)を超えたらタイムアウトとして
    終値で手仕舞う。run_event_driven_backtest_v8_exp.pyと同じ判定順
    (同日に両方到達した場合は損切優先)を踏襲する。"""
    today_str = pd.Timestamp(today).strftime("%Y-%m-%d")
    n_resolved = 0

    for idx, row in df[df["status"] == "PENDING"].iterrows():
        ticker = row["ticker"]
        fill_date = pd.Timestamp(row["fill_date"])
        if pd.Timestamp(today_str) <= fill_date:
            continue  # 約定日(引成で引けに買った日)は寄り付き~引けの保有が無いため判定しない

        bar = _get_latest_bar(all_prices_df, ticker, today_str)
        if bar is None:
            continue

        holding_days = int(row["holding_days_elapsed"]) + 1
        df.at[idx, "holding_days_elapsed"] = holding_days

        target_price = float(row["target_price"])
        stop_price = float(row["stop_price"])
        hit_target = bar["high"] >= target_price
        hit_stop = bar["low"] <= stop_price

        if hit_stop:
            exit_price, exit_reason = stop_price, ("STOP_LOSS (Both)" if hit_target else "STOP_LOSS")
        elif hit_target:
            exit_price, exit_reason = target_price, "TAKE_PROFIT"
        elif holding_days >= HOLDING_PERIOD_DAYS:
            exit_price, exit_reason = bar["close"], "TIME_OUT"
        else:
            continue  # まだ未確定、継続保有

        entry_price = float(row["entry_price"])
        df.at[idx, "status"] = "RESOLVED"
        df.at[idx, "exit_date"] = today_str
        df.at[idx, "exit_price"] = exit_price
        df.at[idx, "exit_reason"] = exit_reason
        df.at[idx, "actual_ret_pct"] = round((exit_price - entry_price) / entry_price, 5)
        n_resolved += 1

    return df, n_resolved


def compute_summary(df):
    resolved = df[df["status"] == "RESOLVED"].copy()
    awaiting_entry = df[df["status"] == "AWAITING_ENTRY"]
    pending = df[df["status"] == "PENDING"]
    if resolved.empty:
        return {
            "n_awaiting_entry": len(awaiting_entry), "n_pending": len(pending), "n_resolved": 0,
            "win_rate": None, "pf": None, "avg_ret_pct": None,
        }
    resolved["actual_ret_pct"] = resolved["actual_ret_pct"].astype(float)
    wins = resolved[resolved["actual_ret_pct"] > 0]
    losses = resolved[resolved["actual_ret_pct"] < 0]
    win_rate = len(wins) / len(resolved) * 100.0
    pf = (wins["actual_ret_pct"].sum() / abs(losses["actual_ret_pct"].sum())
          if len(losses) > 0 and losses["actual_ret_pct"].sum() != 0 else float("inf"))
    return {
        "n_awaiting_entry": len(awaiting_entry),
        "n_pending": len(pending),
        "n_resolved": len(resolved),
        "win_rate": round(win_rate, 1),
        "pf": round(pf, 2) if pf != float("inf") else None,
        "avg_ret_pct": round(resolved["actual_ret_pct"].mean(), 4),
    }
