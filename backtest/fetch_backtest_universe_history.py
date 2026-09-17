# fetch_backtest_universe_history.py
# バックテスト専用の拡張ユニバース(universe_backtest_tickers.txt、売買代金10億円以上の
# 564銘柄)のうち、train_universe_bars.parquetにまだ無い銘柄の3年分の四本値を取得し、
# 同じキャッシュファイルに追記する。ワイド形式(銘柄がトップレベル列)なので、新規銘柄を
# 追加しても既存の150銘柄分のデータには影響しない。
# 学習用ユニバース(universe_150_tickers.txt)自体は変更しない一回限りのバックフィル。
import os
import time
import threading
import datetime
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import pandas as pd
from dotenv import load_dotenv

BASE_DIR = r"F:\stockModel"
CACHE_DIR = os.path.join(BASE_DIR, "data", "cache")
TRAIN_CACHE_PATH = os.path.join(CACHE_DIR, "train_universe_bars.parquet")
BACKTEST_UNIVERSE_PATH = os.path.join(BASE_DIR, "universe_backtest_tickers.txt")
YEARS_BACK = 3
FETCH_WORKERS = 8
REQUESTS_PER_MINUTE = 110

load_dotenv(os.path.join(BASE_DIR, ".env"))
JQUANTS_API_KEY = os.environ.get("JQUANTS_API_KEY")
JQUANTS_BASE_URL = "https://api.jquants.com/v2"


class RateLimiter:
    def __init__(self, max_calls, period=60.0):
        self.max_calls = max_calls
        self.period = period
        self.calls = deque()
        self.lock = threading.Lock()

    def acquire(self):
        while True:
            with self.lock:
                now = time.monotonic()
                while self.calls and now - self.calls[0] > self.period:
                    self.calls.popleft()
                if len(self.calls) < self.max_calls:
                    self.calls.append(now)
                    return
                wait_sec = self.period - (now - self.calls[0])
            time.sleep(max(wait_sec, 0.01))


def fetch_ticker_jquants(session, code, from_date, to_date, rate_limiter=None):
    clean_code = code.replace(".T", "").strip()
    url = f"{JQUANTS_BASE_URL}/equities/bars/daily"
    params = {"code": clean_code, "from": from_date.replace("-", ""), "to": to_date.replace("-", "")}
    try:
        res = None
        for attempt in range(4):
            if rate_limiter is not None:
                rate_limiter.acquire()
            res = session.get(url, params=params, timeout=15)
            if res.status_code == 429:
                time.sleep(2.0 * (attempt + 1))
                continue
            break
        if res is None or res.status_code != 200:
            return None
        data = res.json().get("data", [])
        if not data:
            return None
        df = pd.DataFrame(data)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        col_map = {'AdjO': 'Open', 'AdjH': 'High', 'AdjL': 'Low', 'AdjC': 'Close', 'AdjVo': 'Volume'}
        for k, v in [('O', 'Open'), ('H', 'High'), ('L', 'Low'), ('C', 'Close'), ('Vo', 'Volume')]:
            if k in df.columns and col_map[f"Adj{k}"] not in df.columns:
                col_map[f"Adj{k}"] = v
        available_cols = [c for c in col_map.keys() if c in df.columns]
        return df[available_cols].rename(columns=col_map).astype(float)
    except Exception:
        return None


def main():
    print("=" * 75)
    print("【バックテスト拡張ユニバース 過去データ バックフィル】")
    print("=" * 75)

    with open(BACKTEST_UNIVERSE_PATH, "r", encoding="utf-8") as f:
        target_tickers = [line.strip() for line in f if line.strip()]

    existing_df = None
    existing_tickers = set()
    if os.path.exists(TRAIN_CACHE_PATH):
        existing_df = pd.read_parquet(TRAIN_CACHE_PATH)
        existing_tickers = set(existing_df.columns.get_level_values(0))

    missing = [t for t in target_tickers if t not in existing_tickers]
    print(f"[*] 対象: {len(target_tickers)}銘柄 / 既存キャッシュ済み: {len(target_tickers) - len(missing)} / 新規取得: {len(missing)}")
    if not missing:
        print("[+] 追加分なし。完了。")
        return

    today_d = datetime.date.today()
    end_date = today_d.strftime("%Y-%m-%d")
    start_date = (today_d - datetime.timedelta(days=365 * YEARS_BACK)).strftime("%Y-%m-%d")

    session = requests.Session()
    session.headers.update({"x-api-key": JQUANTS_API_KEY})
    rate_limiter = RateLimiter(max_calls=REQUESTS_PER_MINUTE)

    print(f"[*] {len(missing)}銘柄の四本値を並列取得中 ({start_date} 〜 {end_date}, {FETCH_WORKERS}並列)...")
    ticker_dfs = {}
    with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as executor:
        futures = {executor.submit(fetch_ticker_jquants, session, t, start_date, end_date, rate_limiter): t for t in missing}
        done_count = 0
        for future in as_completed(futures):
            t = futures[future]
            df_single = future.result()
            if df_single is not None and not df_single.empty:
                ticker_dfs[t] = df_single
            done_count += 1
            if done_count % 25 == 0 or done_count == len(missing):
                print(f"  --> {done_count}/{len(missing)} 銘柄完了")

    if not ticker_dfs:
        print("[-] 新規取得データがありませんでした。")
        return

    new_df = pd.concat(ticker_dfs, axis=1)
    combined_df = pd.concat([existing_df, new_df], axis=1) if existing_df is not None else new_df
    combined_df = combined_df.sort_index(axis=1)
    combined_df.to_parquet(TRAIN_CACHE_PATH)
    print(f"\n[+] 保存完了: {TRAIN_CACHE_PATH} (銘柄数: {len(combined_df.columns.get_level_values(0).unique())})")


if __name__ == "__main__":
    main()
