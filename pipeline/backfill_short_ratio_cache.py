# backfill_short_ratio_cache.py
# J-Quants /markets/short-ratio (業種別空売り比率) から市場全体の空売り比率を
# 日次で算出しキャッシュする。33業種+その他(9999)の売買代金を合算して
# market_short_ratio = 空売り代金 / 総売り代金 を求める。
# このエンドポイントは2018年まで遡って取得できることを確認済みのため、
# CTA/JNETフロー(2025-09-01以降のみ)と違い学習期間を制限しない。
import os
import time
import threading
import datetime
from collections import deque
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

BASE_DIR = r"F:\stockModel"
CACHE_PATH = os.path.join(BASE_DIR, "data", "short_ratio_daily_cache.csv")
# TOPIX/VIXキャッシュの開始日(2023-09-19)より少し前から余裕を持って遡及する
ARCHIVE_START = datetime.date(2023, 8, 1)
# 初回実行(FETCH_WORKERS=8, REQUESTS_PER_MINUTE=110)では818日中271日が同時実行過多で
# 失敗(個別に再取得すると正常に取得できることを確認済み)。このエンドポイントは他の
# バルク系エンドポイントより実効レート制限が厳しいようなので、大幅に控えめにする。
FETCH_WORKERS = 3
REQUESTS_PER_MINUTE = 40

load_dotenv(os.path.join(BASE_DIR, ".env"))
JQUANTS_API_KEY = os.environ.get("JQUANTS_API_KEY")
JQUANTS_BASE_URL = "https://api.jquants.com/v2"


class RateLimiter:
    """直近60秒あたりのリクエスト数を上限以内に抑えるトークンバケット風リミッター。
    (初回バックフィル実行時、これが無くレート制限に引っかかり818日中106日しか
    取得できなかった不具合があったため追加。429時のリトライも併せて実装。)"""
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


def fetch_short_ratio(session, date_str, rate_limiter, retry_count=6):
    """1日分の業種別空売り比率を取得し、市場全体の空売り比率(float)を返す。データが無ければNone。"""
    for attempt in range(retry_count):
        try:
            rate_limiter.acquire()
            res = session.get(
                f"{JQUANTS_BASE_URL}/markets/short-ratio",
                params={"date": date_str}, timeout=15,
            )
            if res.status_code == 429:
                time.sleep(2.0 * (attempt + 1))
                continue
            if res.status_code != 200:
                return None
            break
        except Exception:
            time.sleep(1.0)
    else:
        return None

    try:
        data = res.json().get("data", [])
        if not data:
            return None
        sell_ex = sum(r.get("SellExShortVa", 0.0) or 0.0 for r in data)
        shrt_res = sum(r.get("ShrtWithResVa", 0.0) or 0.0 for r in data)
        shrt_nores = sum(r.get("ShrtNoResVa", 0.0) or 0.0 for r in data)
        total_short = shrt_res + shrt_nores
        total_sell = sell_ex + total_short
        if total_sell <= 0:
            return None
        return total_short / total_sell
    except Exception:
        return None


def get_business_days(start_date, end_date):
    days = []
    curr = start_date
    while curr <= end_date:
        if curr.weekday() < 5:
            days.append(curr.strftime("%Y%m%d"))
        curr += datetime.timedelta(days=1)
    return days


def main():
    print("=" * 70)
    print("【市場全体 空売り比率 日次キャッシュ バックフィル】")
    print("=" * 70)

    session = requests.Session()
    session.headers.update({"x-api-key": JQUANTS_API_KEY})
    rate_limiter = RateLimiter(max_calls=REQUESTS_PER_MINUTE)

    today = datetime.date.today()
    target_days = get_business_days(ARCHIVE_START, today)

    existing_dates = set()
    if os.path.exists(CACHE_PATH):
        existing_df = pd.read_csv(CACHE_PATH, encoding='utf-8-sig')
        existing_dates = set(existing_df["date"].unique())

    missing_days = [d for d in target_days if f"{d[:4]}-{d[4:6]}-{d[6:]}" not in existing_dates]
    print(f"[*] 対象営業日候補: {len(target_days)}日 / 未キャッシュ: {len(missing_days)}日")

    if not missing_days:
        print("[+] 追加分なし。完了。")
        return

    results = {}
    with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as executor:
        futures = {executor.submit(fetch_short_ratio, session, d, rate_limiter): d for d in missing_days}
        done_count = 0
        for future in as_completed(futures):
            d = futures[future]
            ratio = future.result()
            if ratio is not None:
                results[d] = ratio
            done_count += 1
            if done_count % 50 == 0 or done_count == len(missing_days):
                print(f"  --> {done_count}/{len(missing_days)} 日完了")

    if not results:
        print("[-] 取得できたデータがありませんでした。")
        return

    new_rows = [
        {"date": f"{d[:4]}-{d[4:6]}-{d[6:]}", "market_short_ratio": ratio}
        for d, ratio in sorted(results.items())
    ]
    new_df = pd.DataFrame(new_rows)

    if os.path.exists(CACHE_PATH):
        existing_df = pd.read_csv(CACHE_PATH, encoding='utf-8-sig')
        existing_df = existing_df[~existing_df["date"].isin(new_df["date"])]
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined_df = new_df

    combined_df = combined_df.sort_values("date")
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    combined_df.to_csv(CACHE_PATH, index=False, encoding='utf-8-sig')
    print(f"\n[+] バックフィル完了: {CACHE_PATH} ({len(results)}日分追加、累計{combined_df['date'].nunique()}日)")


if __name__ == "__main__":
    main()
