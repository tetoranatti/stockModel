import os
import re
import json
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "margin")
os.makedirs(DATA_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def get_current_week_cache_path():
    """現在の年と週番号（ISO週）に基づいたキャッシュファイルパスを生成"""
    now = datetime.now()
    year, week, _ = now.isocalendar()
    return os.path.join(DATA_DIR, f"margin_cache_{year}_w{week:02d}.json")

def load_cache():
    cache_path = get_current_week_cache_path()
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_cache(cache_data):
    cache_path = get_current_week_cache_path()
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[警告] キャッシュ保存失敗: {e}")

def clean_num(val):
    if val is None or pd.isna(val):
        return 0.0
    s = str(val).replace(",", "").replace("+", "").replace("倍", "").replace("株", "").strip()
    try:
        return float(s)
    except ValueError:
        return 0.0

def fetch_single_margin(ticker_code):
    """株探の個別銘柄ページから直近の信用残を取得"""
    code = str(ticker_code).replace(".T", "").strip()
    url = f"https://kabutan.jp/stock/?code={code}"

    try:
        # 過度な負荷を避けるためスレッドごとに微小待機
        time.sleep(0.1)
        res = requests.get(url, headers=HEADERS, timeout=8)
        if res.status_code != 200:
            return None
        
        soup = BeautifulSoup(res.text, "html.parser")

        target_table = None
        for table in soup.find_all("table"):
            text = table.get_text()
            if "売り残" in text and "買い残" in text and "倍率" in text:
                target_table = table
                break

        if not target_table:
            return None

        rows = target_table.find_all("tr")
        history = []
        for r in rows:
            tds = [td.get_text(strip=True) for td in r.find_all(["td", "th"])]
            if len(tds) >= 4 and any(char.isdigit() for char in tds[0]):
                date_str = tds[0]
                short_val = clean_num(tds[1])
                buy_val = clean_num(tds[2])
                ratio_val = clean_num(tds[3])
                history.append({
                    "date": date_str,
                    "short": short_val,
                    "buy": buy_val,
                    "ratio": ratio_val
                })

        if not history:
            return None

        latest = history[0]
        if len(history) >= 2:
            prev = history[1]
            buy_diff = latest["buy"] - prev["buy"]
            buy_pct = round((buy_diff / prev["buy"]) * 100, 2) if prev["buy"] > 0 else 0.0
            short_diff = latest["short"] - prev["short"]
        else:
            buy_diff = 0
            buy_pct = 0.0
            short_diff = 0

        is_accumulating = (buy_pct <= -3.0) and (latest["ratio"] <= 5.0)

        return {
            "ticker": f"{code}.T",
            "code": code,
            "margin_date": latest["date"],
            "margin_buy": int(latest["buy"]),
            "margin_buy_diff": int(buy_diff),
            "margin_buy_pct": buy_pct,
            "margin_short": int(latest["short"]),
            "margin_ratio": latest["ratio"],
            "is_accumulating": is_accumulating
        }

    except Exception:
        return None

def update_screened_with_margin(max_workers=5):
    """
    キャッシュを併用しつつ並列処理で需給データをマージ
    """
    csv_path = os.path.join(BASE_DIR, "final_regime_screened_v6.csv")
    if not os.path.exists(csv_path):
        print(f"CSVが見つかりません: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    target_tickers = df["ticker"].dropna().unique().tolist()
    
    # 1. キャッシュのロード
    cache = load_cache()
    cached_count = sum(1 for t in target_tickers if t in cache)
    missing_tickers = [t for t in target_tickers if t not in cache]

    print(f"対象銘柄: {len(target_tickers)}件 (キャッシュ済み: {cached_count}件, 新規取得: {len(missing_tickers)}件)")

    # 2. 未キャッシュ銘柄のみ並列スクレイピング
    if missing_tickers:
        print(f"{max_workers}並列で新規データ取得中...")
        completed = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ticker = {executor.submit(fetch_single_margin, t): t for t in missing_tickers}
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                completed += 1
                try:
                    res = future.result()
                    if res:
                        cache[ticker] = res
                except Exception as e:
                    pass
                print(f"進捗: {completed}/{len(missing_tickers)} 完了", end="\r")

        print("\nスクレイピング完了。キャッシュを保存中...")
        save_cache(cache)
    else:
        print("今週分のデータは全てキャッシュされているため、ネットワーク通信をスキップします。")

    # 3. CSVへの結合
    results = [cache[t] for t in target_tickers if t in cache]
    if not results:
        print("結合対象のデータがありませんでした。")
        return

    margin_df = pd.DataFrame(results)

    # 既存のCSVにマージ済みの旧需給列があれば上書きのために一度除去
    drop_cols = ["margin_date", "margin_buy", "margin_buy_diff", "margin_buy_pct", 
                 "margin_short", "margin_ratio", "is_accumulating", "code"]
    existing_drops = [c for c in drop_cols if c in df.columns]
    if existing_drops:
        df = df.drop(columns=existing_drops)

    merged_df = pd.merge(df, margin_df, on="ticker", how="left")
    merged_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print(f"需給データを正常にマージしました: {csv_path}")

if __name__ == "__main__":
    update_screened_with_margin()