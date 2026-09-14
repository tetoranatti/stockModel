# modules/data_loader.py
import os
import glob
import json
import shutil
import datetime
import pandas as pd
import yfinance as yf

BASE_DIR = r"F:\stockModel"
DATA_DIR = os.path.join(BASE_DIR, "data")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
MARGIN_DIR = os.path.join(DATA_DIR, "margin")
DOWNLOADS_DIR = os.path.join(os.environ.get("USERPROFILE", ""), "Downloads")
SCREENER_CSV = os.path.join(BASE_DIR, "screener_result.csv")
JPX_DB_PATH = os.path.join(BASE_DIR, "jpx_daily_features_db.csv")
SECTOR_SENTIMENT_JSON = os.path.join(DATA_DIR, "sector_sentiment.json")
SECTOR_MASTER_JSON = os.path.join(DATA_DIR, "jpx_sector_master.json")

os.makedirs(CACHE_DIR, exist_ok=True)

def sync_latest_screener_csv():
    pattern = os.path.join(DOWNLOADS_DIR, "*screener*.csv")
    files = glob.glob(pattern)
    if files:
        latest = max(files, key=os.path.getmtime)
        shutil.copy2(latest, SCREENER_CSV)
        print(f"[+] 最新スクリーニングCSVを自動同期: {os.path.basename(latest)}")

def load_screener_tickers():
    sync_latest_screener_csv()
    if not os.path.exists(SCREENER_CSV):
        return []
    try:
        raw_df = pd.read_csv(SCREENER_CSV, encoding='cp932')
    except Exception:
        raw_df = pd.read_csv(SCREENER_CSV, encoding='utf-8')

    code_col = [c for c in raw_df.columns if "コード" in str(c)][0]
    import re
    tickers = [
        f"{str(c).strip().upper()}.T" 
        for c in raw_df[code_col] 
        if re.match(r"^[0-9]{4}$|^[0-9]{3}[A-Z]$", str(c).strip().upper())
    ]
    return tickers

def fetch_all_tickers_data(tickers, m_start):
    today_str = datetime.date.today().strftime("%Y%m%d")
    cache_file = os.path.join(CACHE_DIR, f"prices_{today_str}.parquet")

    if os.path.exists(cache_file):
        try:
            print(f"[*] 本日分の株価ローカルキャッシュを読み込み中 ({cache_file})...")
            return pd.read_parquet(cache_file)
        except Exception:
            pass

    print(f"[*] 全 {len(tickers)} 銘柄の株価データを一括並列ダウンロード中 (threads=True)...")
    all_df = yf.download(
        tickers=tickers,
        start=m_start,
        interval="1d",
        auto_adjust=True,   # ★ 株式分割・併合・配当落ちを過去全期間に遡って自動補正
        group_by="ticker",
        threads=True,
        progress=True
    )
    if not all_df.empty:
        try:
            all_df.to_parquet(cache_file)
            print(f"[+] 当日株価キャッシュを保存しました: {cache_file}")
        except Exception as e:
            print(f"[!] キャッシュ保存スキップ: {e}")
    return all_df

def load_margin_cache():
    now = datetime.datetime.now()
    year, week, _ = now.isocalendar()
    cache_path = os.path.join(MARGIN_DIR, f"margin_cache_{year}_w{week:02d}.json")
    if not os.path.exists(cache_path):
        files = glob.glob(os.path.join(MARGIN_DIR, "margin_cache_*.json"))
        if files:
            cache_path = max(files, key=os.path.getmtime)
        else:
            return {}
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def load_sector_data():
    sentiment_map = {}
    master_map = {}
    if os.path.exists(SECTOR_SENTIMENT_JSON):
        try:
            with open(SECTOR_SENTIMENT_JSON, "r", encoding="utf-8") as f:
                sentiment_map = json.load(f)
        except Exception as e:
            print(f"[!] センチメント読み込みスキップ: {e}")

    if os.path.exists(SECTOR_MASTER_JSON):
        try:
            with open(SECTOR_MASTER_JSON, "r", encoding="utf-8") as f:
                master_map = json.load(f)
        except Exception as e:
            print(f"[!] セクターマスター読み込みスキップ: {e}")

    return sentiment_map, master_map