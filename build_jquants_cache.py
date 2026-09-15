# build_jquants_cache.py
import os
import datetime
import time
import json
import requests
import pandas as pd
from dotenv import load_dotenv

BASE_DIR = r"F:\stockModel"
DATA_DIR = os.path.join(BASE_DIR, "data")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
UNIVERSE_PATH = os.path.join(BASE_DIR, "universe_150_tickers.txt")
TRAIN_CACHE_PATH = os.path.join(CACHE_DIR, "train_universe_bars.parquet")
TOPIX_CACHE_PATH = os.path.join(CACHE_DIR, "train_topix_bars.parquet")
NK225_UNDERLYING_CACHE_PATH = os.path.join(CACHE_DIR, "train_nk225_underlying.parquet")

os.makedirs(CACHE_DIR, exist_ok=True)
load_dotenv()

# APIキー設定
JQUANTS_API_KEY = os.environ.get("JQUANTS_API_KEY")
if not JQUANTS_API_KEY:
    raise ValueError("[!] JQUANTS_API_KEY が設定されていません。.env ファイルを確認してください。")

JQUANTS_BASE_URL = "https://api.jquants.com/v2"

MIN_TURNOVER = 10e8  # 10億円 (v8基準)
FETCH_DAYS = 45      # 日次スクリーニング用の直近営業日数
YEARS_BACK = 3       # 学習用データセットの期間 (年)

def get_recent_business_days(n_days=45):
    """直近の営業日候補（平日）リストを生成"""
    today = datetime.date.today()
    days = []
    curr = today
    while len(days) < n_days:
        if curr.weekday() < 5:  # 土日除外
            days.append(curr.strftime("%Y%m%d"))
        curr -= datetime.timedelta(days=1)
    return sorted(days)

def fetch_daily_all(session, date_str, retry_count=3):
    """1日分の全上場銘柄四本値を一括取得 (1リクエスト)"""
    url = f"{JQUANTS_BASE_URL}/equities/bars/daily"
    params = {"date": date_str}
    for attempt in range(retry_count):
        try:
            res = session.get(url, params=params, timeout=15)
            if res.status_code == 200:
                data = res.json().get("data", [])
                return data
            elif res.status_code == 429:
                wait_sec = 2.0 * (attempt + 1)
                print(f"[!] 429 Rate Limit ({date_str}): {wait_sec}秒待機して再試行...")
                time.sleep(wait_sec)
                continue
            else:
                print(f"[!] HTTP Error {res.status_code} ({date_str}): {res.text}")
                return []
        except Exception as e:
            print(f"[!] {date_str} 例外エラー: {e}")
            time.sleep(1.0)
    return []

def fetch_ticker_jquants(session, code, from_date, to_date):
    """J-Quants V2 API から個別銘柄の長期四本値を取得"""
    clean_code = code.replace(".T", "").strip()
    url = f"{JQUANTS_BASE_URL}/equities/bars/daily"
    params = {
        "code": clean_code,
        "from": from_date.replace("-", ""),
        "to": to_date.replace("-", "")
    }
    try:
        res = session.get(url, params=params, timeout=15)
        if res.status_code != 200:
            return None
        data = res.json().get("data", [])
        if not data:
            return None

        df = pd.DataFrame(data)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)

        col_map = {
            'AdjO': 'Open',
            'AdjH': 'High',
            'AdjL': 'Low',
            'AdjC': 'Close',
            'AdjVo': 'Volume'
        }
        for k, v in [('O', 'Open'), ('H', 'High'), ('L', 'Low'), ('C', 'Close'), ('Vo', 'Volume')]:
            if k in df.columns and col_map[f"Adj{k}"] not in df.columns:
                col_map[f"Adj{k}"] = v

        available_cols = [c for c in col_map.keys() if c in df.columns]
        return df[available_cols].rename(columns=col_map).astype(float)
    except Exception as e:
        print(f"[!] {code} 取得エラー: {e}")
        return None

def fetch_nk225_underlying(session, date_str):
    """J-Quants V2 API から日経225オプション四本値を取得し、当日の原証券価格(日経225現物相当)のみ抜き出す"""
    url = f"{JQUANTS_BASE_URL}/derivatives/bars/daily/options/225"
    params = {"date": date_str}
    try:
        res = session.get(url, params=params, timeout=15)
        if res.status_code != 200:
            return None
        data = res.json().get("data", [])
        for rec in data:
            under_px = rec.get("UnderPx")
            if under_px is not None:
                return float(under_px)
        return None
    except Exception as e:
        print(f"[!] {date_str} 日経225原証券価格取得エラー: {e}")
        return None

def fetch_index_jquants(session, index_code="0000", from_date=None, to_date=None):
    """J-Quants V2 API からTOPIX等の指数四本値を取得"""
    url = f"{JQUANTS_BASE_URL}/indices/bars/daily"
    params = {"code": index_code}
    if from_date:
        params["from"] = from_date.replace("-", "")
    if to_date:
        params["to"] = to_date.replace("-", "")
    try:
        res = session.get(url, params=params, timeout=15)
        if res.status_code != 200:
            return None
        data = res.json().get("data", [])
        if not data:
            return None
        df = pd.DataFrame(data)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        rename_map = {'O': 'Open', 'H': 'High', 'L': 'Low', 'C': 'Close'}
        return df[list(rename_map.keys())].rename(columns=rename_map).astype(float)
    except Exception as e:
        print(f"[!] 指数取得例外: {e}")
        return None

def main():
    session = requests.Session()
    session.headers.update({"x-api-key": JQUANTS_API_KEY})

    # =========================================================================
    # PART 1: 当日スクリーニング ＆ UI用 キャッシュ生成（直近45日バルク取得）
    # =========================================================================
    print("=" * 75)
    print("【1/2】日次スクリーニング用キャッシュ生成 (直近45営業日 / 代金10億円以上)")
    print("=" * 75)

    target_days = get_recent_business_days(FETCH_DAYS)
    all_records = []

    print(f"[*] 直近 {len(target_days)} 営業日分の全市場データを取得開始...")
    for i, d_str in enumerate(target_days):
        records = fetch_daily_all(session, d_str)
        if records:
            all_records.extend(records)
            print(f"  --> [{i+1}/{len(target_days)}] {d_str}: {len(records)} 銘柄取得完了")
        time.sleep(0.55)  # Standard 120req/min 安全域

    if not all_records:
        print("[-] 日次バルクデータが取得できませんでした。")
        return

    print("[*] データ結合・整形中...")
    raw_df = pd.DataFrame(all_records)
    raw_df['Date'] = pd.to_datetime(raw_df['Date'])
    
    raw_df = raw_df[raw_df['Code'].astype(str).str.match(r"^[0-9]{4}0?$|^[0-9]{3}[A-Z]0?$")]
    raw_df['ticker'] = raw_df['Code'].astype(str).str.slice(0, 4) + ".T"

    col_map = {
        'AdjO': 'Open',
        'AdjH': 'High',
        'AdjL': 'Low',
        'AdjC': 'Close',
        'AdjVo': 'Volume',
        'TurnoverValue': 'Turnover'
    }
    for k in ['O', 'H', 'L', 'C', 'Vo', 'Va']:
        if k in raw_df.columns:
            target_k = 'Turnover' if k == 'Va' else col_map.get(f"Adj{k}", k)
            if target_k not in raw_df.columns:
                raw_df[target_k] = raw_df[k]

    sub_df = raw_df[['Date', 'ticker', 'Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        sub_df[col] = sub_df[col].astype(float)

    # 10億円スクリーニング
    print("[*] 直近5日平均 売買代金 10億円以上の銘柄を抽出中...")
    sub_df['Turnover_calc'] = sub_df['Close'] * sub_df['Volume']
    latest_turnover_5d = (
        sub_df.sort_values('Date')
        .groupby('ticker')['Turnover_calc']
        .apply(lambda s: s.tail(5).mean())
    )

    qualified_tickers = latest_turnover_5d[latest_turnover_5d >= MIN_TURNOVER].index.tolist()
    print(f"[+] 条件合致（売買代金 >= 10億円）: {len(qualified_tickers)} 銘柄抽出")

    filtered_df = sub_df[sub_df['ticker'].isin(qualified_tickers)].drop(columns=['Turnover_calc'])
    filtered_df.set_index(['Date', 'ticker'], inplace=True)
    pivot_df = filtered_df.unstack(level='ticker')
    pivot_df = pivot_df.swaplevel(0, 1, axis=1).sort_index(axis=1)

    today_str = datetime.date.today().strftime("%Y%m%d")
    cache_file = os.path.join(CACHE_DIR, f"prices_{today_str}.parquet")
    pivot_df.to_parquet(cache_file)
    print(f"[+] 当日株価キャッシュ保存完了: {cache_file}")

    screener_out = os.path.join(BASE_DIR, "screener_result.csv")
    pd.DataFrame({"コード": [t.replace(".T", "") for t in qualified_tickers]}).to_csv(screener_out, index=False, encoding="cp932")
    print(f"[+] スクリーナー銘柄リスト更新完了: {screener_out}")

    # UI描画用ローカルJSON保存
    chart_cache_file = os.path.join(CACHE_DIR, f"chart_candles_{today_str}.json")
    chart_dict = {}
    for ticker in qualified_tickers:
        if ticker in pivot_df.columns.levels[0]:
            df_t = pivot_df[ticker].dropna()
            bars = []
            for dt, row in df_t.iterrows():
                bars.append({
                    "time": dt.strftime("%Y-%m-%d"),
                    "open": float(row["Open"]),
                    "high": float(row["High"]),
                    "low": float(row["Low"]),
                    "close": float(row["Close"]),
                    "volume": float(row["Volume"])
                })
            chart_dict[ticker] = bars

    with open(chart_cache_file, "w", encoding="utf-8") as f:
        json.dump(chart_dict, f)
    print(f"[+] チャート描画用JSON保存完了: {chart_cache_file}")

    # =========================================================================
    # PART 2: モデル学習用 キャッシュ生成（universe_150 の過去3年分）
    # =========================================================================
    print("\n" + "=" * 75)
    print(f"【2/2】モデル学習用ユニバース長期データキャッシュ生成 (過去 {YEARS_BACK} 年分)")
    print("=" * 75)

    if not os.path.exists(UNIVERSE_PATH):
        print(f"[!] {UNIVERSE_PATH} が見つかりません。学習用キャッシュ作成をスキップします。")
        return

    with open(UNIVERSE_PATH, "r", encoding="utf-8") as f:
        universe_tickers = [line.strip() for line in f if line.strip()]

    print(f"[*] 学習ユニバース対象: {len(universe_tickers)} 銘柄")
    end_date = datetime.date.today().strftime("%Y-%m-%d")
    start_date = (datetime.date.today() - datetime.timedelta(days=365 * YEARS_BACK)).strftime("%Y-%m-%d")

    # 1. TOPIX指数の取得・保存
    print(f"[*] TOPIX指数 (0000) を取得中 ({start_date} ～ {end_date})...")
    topix_df = fetch_index_jquants(session, index_code="0000", from_date=start_date, to_date=end_date)
    if topix_df is not None and not topix_df.empty:
        topix_df.to_parquet(TOPIX_CACHE_PATH)
        print(f"[+] TOPIX学習用キャッシュ保存完了: {TOPIX_CACHE_PATH}")
    time.sleep(0.55)

    # 2. ユニバース全銘柄の過去3年分を取得
    print(f"[*] 全 {len(universe_tickers)} 銘柄の過去3年四本値を取得中...")
    ticker_train_dfs = {}
    for i, t in enumerate(universe_tickers):
        df_single = fetch_ticker_jquants(session, t, start_date, end_date)
        if df_single is not None and not df_single.empty:
            ticker_train_dfs[t] = df_single

        time.sleep(0.55)  # Standard 120req/min 安全域

        if (i + 1) % 25 == 0 or (i + 1) == len(universe_tickers):
            print(f"  --> 学習用データ進捗: {i + 1}/{len(universe_tickers)} 銘柄完了")

    if ticker_train_dfs:
        all_train_df = pd.concat(ticker_train_dfs, axis=1)
        all_train_df.to_parquet(TRAIN_CACHE_PATH)
        print(f"[+] 学習用3年分キャッシュ保存完了: {TRAIN_CACHE_PATH}")
    else:
        print("[-] 学習用データを取得できませんでした。")

    # =========================================================================
    # PART 3: 日経225原証券価格（マクロ特徴量用）キャッシュ生成
    # J-Quantsは日経225現物指数を配信していないため、日経225オプション四本値
    # レスポンスに含まれる UnderPx (原証券価格) を日経225終値の代替として使う
    # =========================================================================
    print("\n" + "=" * 75)
    print(f"【3/3】日経225原証券価格キャッシュ生成 (過去 {YEARS_BACK} 年分, 1営業日ずつ取得)")
    print("=" * 75)

    business_days = []
    curr = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
    end_d = datetime.datetime.strptime(end_date, "%Y-%m-%d").date()
    while curr <= end_d:
        if curr.weekday() < 5:
            business_days.append(curr.strftime("%Y%m%d"))
        curr += datetime.timedelta(days=1)

    print(f"[*] 対象営業日数: {len(business_days)} 日 (概算所要時間: 約{len(business_days) * 0.55 / 60:.1f}分)")
    nk225_records = []
    for i, d_str in enumerate(business_days):
        under_px = fetch_nk225_underlying(session, d_str)
        if under_px is not None:
            nk225_records.append({"Date": pd.to_datetime(d_str), "NK_Close": under_px})

        time.sleep(0.55)  # Standard 120req/min 安全域

        if (i + 1) % 100 == 0 or (i + 1) == len(business_days):
            print(f"  --> 日経225原証券価格 取得進捗: {i + 1}/{len(business_days)} 日完了")

    if nk225_records:
        nk225_df = pd.DataFrame(nk225_records).set_index("Date").sort_index()
        nk225_df.to_parquet(NK225_UNDERLYING_CACHE_PATH)
        print(f"[+] 日経225原証券価格キャッシュ保存完了: {NK225_UNDERLYING_CACHE_PATH} ({len(nk225_df)} 日分)")
    else:
        print("[-] 日経225原証券価格を取得できませんでした（Standardプラン契約・APIキーをご確認ください）。")

    print("\n[+] 全キャッシュ生成パイプラインが正常に完了しました。")

if __name__ == "__main__":
    main()