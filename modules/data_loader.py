# modules/data_loader.py
import os
import glob
import json
import datetime
import pandas as pd

BASE_DIR = r"F:\stockModel"
DATA_DIR = os.path.join(BASE_DIR, "data")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
MARGIN_DIR = os.path.join(DATA_DIR, "margin")
SCREENER_CSV = os.path.join(BASE_DIR, "screener_result.csv")
JPX_DB_PATH = os.path.join(BASE_DIR, "jpx_daily_features_db.csv")
SECTOR_SENTIMENT_JSON = os.path.join(DATA_DIR, "sector_sentiment.json")
SECTOR_MASTER_JSON = os.path.join(DATA_DIR, "jpx_sector_master.json")

os.makedirs(CACHE_DIR, exist_ok=True)

def load_screener_tickers():
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

def fetch_all_tickers_data():
    """build_jquants_cache.py が生成する当日分のJ-Quants株価キャッシュを読み込む"""
    today_str = datetime.date.today().strftime("%Y%m%d")
    cache_file = os.path.join(CACHE_DIR, f"prices_{today_str}.parquet")

    if not os.path.exists(cache_file):
        raise FileNotFoundError(
            f"[!] {cache_file} が見つかりません。"
            f" 先に build_jquants_cache.py を実行して当日分のキャッシュを生成してください。"
        )
    print(f"[*] 本日分の株価ローカルキャッシュを読み込み中 ({cache_file})...")
    return pd.read_parquet(cache_file)

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