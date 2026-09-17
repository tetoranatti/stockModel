# build_jquants_cache.py
import os
import datetime
import time
import json
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import pandas as pd
from dotenv import load_dotenv

BASE_DIR = r"F:\stockModel"
DATA_DIR = os.path.join(BASE_DIR, "data")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
MARGIN_DIR = os.path.join(DATA_DIR, "margin")
UNIVERSE_PATH = os.path.join(BASE_DIR, "universe_150_tickers.txt")
TRAIN_CACHE_PATH = os.path.join(CACHE_DIR, "train_universe_bars.parquet")
TOPIX_CACHE_PATH = os.path.join(CACHE_DIR, "train_topix_bars.parquet")
NK225_UNDERLYING_CACHE_PATH = os.path.join(CACHE_DIR, "train_nk225_underlying.parquet")

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(MARGIN_DIR, exist_ok=True)
load_dotenv()

# APIキー設定
JQUANTS_API_KEY = os.environ.get("JQUANTS_API_KEY")
if not JQUANTS_API_KEY:
    raise ValueError("[!] JQUANTS_API_KEY が設定されていません。.env ファイルを確認してください。")

JQUANTS_BASE_URL = "https://api.jquants.com/v2"

MIN_TURNOVER = 10e8  # 10億円 (v8基準)
# compute_stock_features()のrolling_betaが90営業日窓を使うため、
# run_dynamic_regime_screening_v8.py側のSEQ_LEN(10)+95日フィルタを満たすには
# 最低105営業日分が必要。45日だと全銘柄がフィルタで弾かれ0件になる(実際に発生した不具合)。
FETCH_DAYS = 130     # 日次スクリーニング用の直近営業日数
YEARS_BACK = 3       # 学習用データセットの期間 (年)

FETCH_WORKERS = 8          # 並列フェッチのスレッド数
REQUESTS_PER_MINUTE = 110  # Standardプラン上限(120req/分)に対する安全マージン

class RateLimiter:
    """複数スレッドで共有し、直近60秒あたりのリクエスト数を上限以内に抑えるトークンバケット風リミッター"""
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

def fetch_daily_all(session, date_str, retry_count=3, rate_limiter=None):
    """1日分の全上場銘柄四本値を一括取得 (1リクエスト)"""
    url = f"{JQUANTS_BASE_URL}/equities/bars/daily"
    params = {"date": date_str}
    for attempt in range(retry_count):
        try:
            if rate_limiter is not None:
                rate_limiter.acquire()
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

def fetch_ticker_jquants(session, code, from_date, to_date, rate_limiter=None):
    """J-Quants V2 API から個別銘柄の長期四本値を取得"""
    clean_code = code.replace(".T", "").strip()
    url = f"{JQUANTS_BASE_URL}/equities/bars/daily"
    params = {
        "code": clean_code,
        "from": from_date.replace("-", ""),
        "to": to_date.replace("-", "")
    }
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
            if res is not None and res.status_code != 200:
                print(f"[!] {code} 取得エラー: HTTP {res.status_code}")
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

def fetch_nk225_underlying(session, date_str, rate_limiter=None):
    """J-Quants V2 API から日経225オプション四本値を取得し、当日の原証券価格(日経225現物相当)のみ抜き出す"""
    url = f"{JQUANTS_BASE_URL}/derivatives/bars/daily/options/225"
    params = {"date": date_str}
    try:
        if rate_limiter is not None:
            rate_limiter.acquire()
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

def fetch_margin_interest_by_date(session, date_str, rate_limiter=None):
    """J-Quants V2 API から信用取引週末残高(全銘柄、1リクエスト)を取得。未公表日はNone"""
    url = f"{JQUANTS_BASE_URL}/markets/margin-interest"
    try:
        if rate_limiter is not None:
            rate_limiter.acquire()
        res = session.get(url, params={"date": date_str}, timeout=15)
        if res.status_code != 200:
            return None
        data = res.json().get("data", [])
        return data if data else None
    except Exception as e:
        print(f"[!] {date_str} 信用取引週末残高取得エラー: {e}")
        return None

def fetch_latest_margin_interest(session, start_date, rate_limiter=None, max_lookback_days=10):
    """start_date から遡って、直近で公表済みの信用取引週末残高(全銘柄)を探索して取得する
    (週次公表のため、通常は直近の金曜日が見つかる)"""
    for offset in range(max_lookback_days):
        d = start_date - datetime.timedelta(days=offset)
        data = fetch_margin_interest_by_date(session, d.strftime("%Y%m%d"), rate_limiter)
        if data:
            return d, data
    return None, []

def fetch_earnings_dates(session, code, rate_limiter=None):
    """J-Quants V2 API から銘柄コード指定で決算発表予定日の履歴(過去〜直近)を取得"""
    url = f"{JQUANTS_BASE_URL}/fins/earnings-date"
    try:
        if rate_limiter is not None:
            rate_limiter.acquire()
        res = session.get(url, params={"code": code}, timeout=15)
        if res.status_code != 200:
            return []
        return res.json().get("data", [])
    except Exception as e:
        print(f"[!] {code} 決算発表日取得エラー: {e}")
        return []

def fetch_equities_master_by_date(session, date_str, rate_limiter=None):
    """J-Quants V2 API から指定日時点の上場銘柄マスター(全銘柄、ETF/REIT含む、1リクエスト)を取得"""
    url = f"{JQUANTS_BASE_URL}/equities/master"
    try:
        if rate_limiter is not None:
            rate_limiter.acquire()
        res = session.get(url, params={"date": date_str}, timeout=15)
        if res.status_code != 200:
            return []
        return res.json().get("data", [])
    except Exception as e:
        print(f"[!] {date_str} 上場銘柄マスター取得エラー: {e}")
        return []

def fetch_latest_equities_master(session, start_date, rate_limiter=None, max_lookback_days=10):
    """start_date から遡って、直近で取得可能な上場銘柄マスターを探索して取得する"""
    for offset in range(max_lookback_days):
        d = start_date - datetime.timedelta(days=offset)
        data = fetch_equities_master_by_date(session, d.strftime("%Y%m%d"), rate_limiter)
        if data:
            return d, data
    return None, []

def load_existing_cache(cache_path):
    """既存キャッシュを読み込み、(DataFrame, 最終日付) を返す。存在しなければ (None, None)"""
    if not os.path.exists(cache_path):
        return None, None
    try:
        df = pd.read_parquet(cache_path)
        if df.empty:
            return None, None
        return df, pd.to_datetime(df.index.max()).date()
    except Exception:
        return None, None

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
    rate_limiter = RateLimiter(max_calls=REQUESTS_PER_MINUTE)

    # 上場銘柄マスターを先に取得し、PART1(スクリーニング・学習ユニバース双方の
    # 母集団)から普通株式以外(ETF/REIT/外国株等)とグロース市場銘柄を除外する
    # のに使う。PART6のキャッシュ生成でもこの結果を使い回す。
    print("[*] 上場銘柄マスターを取得中 (ETF/REIT・グロース市場除外用)...")
    today_d_for_master = datetime.date.today()
    master_date, master_records = fetch_latest_equities_master(session, today_d_for_master, rate_limiter)
    equity_master_map = {rec["Code"]: rec for rec in master_records} if master_records else {}

    def is_eligible_ticker(ticker: str) -> bool:
        """普通株式(ProdCat=011)かつグロース市場(Mkt=0113)以外のみ対象とする"""
        code5_candidates = [ticker.replace(".T", "") + "0", ticker.replace(".T", "")]
        rec = next((equity_master_map[c] for c in code5_candidates if c in equity_master_map), None)
        if rec is None:
            return True  # マスターに無い銘柄は判定不能のため従来通り対象に含める
        return rec.get("ProdCat") == "011" and rec.get("Mkt") != "0113"

    # =========================================================================
    # PART 1: 当日スクリーニング ＆ UI用 キャッシュ生成（直近45日バルク取得）
    # =========================================================================
    print("=" * 75)
    print("【1/6】日次スクリーニング用キャッシュ生成 (直近45営業日 / 代金10億円以上)")
    print("=" * 75)

    target_days = get_recent_business_days(FETCH_DAYS)
    all_records = []

    print(f"[*] 直近 {len(target_days)} 営業日分の全市場データを{FETCH_WORKERS}並列で取得開始...")
    with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as executor:
        futures = {
            executor.submit(fetch_daily_all, session, d_str, 3, rate_limiter): d_str
            for d_str in target_days
        }
        done_count = 0
        for future in as_completed(futures):
            d_str = futures[future]
            records = future.result()
            if records:
                all_records.extend(records)
            done_count += 1
            print(f"  --> [{done_count}/{len(target_days)}] {d_str}: {len(records)} 銘柄取得完了")

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

    turnover_qualified = latest_turnover_5d[latest_turnover_5d >= MIN_TURNOVER].index.tolist()
    qualified_tickers = [t for t in turnover_qualified if is_eligible_ticker(t)]
    excluded_count = len(turnover_qualified) - len(qualified_tickers)
    print(f"[+] 条件合致（売買代金 >= 10億円）: {len(turnover_qualified)} 銘柄"
          f" -> ETF/REIT・グロース市場除外後: {len(qualified_tickers)} 銘柄 ({excluded_count} 銘柄除外)")

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
    print(f"【2/6】モデル学習用ユニバース長期データキャッシュ生成 (過去 {YEARS_BACK} 年分)")
    print("=" * 75)

    if not os.path.exists(UNIVERSE_PATH):
        print(f"[!] {UNIVERSE_PATH} が見つかりません。学習用キャッシュ作成をスキップします。")
        return

    with open(UNIVERSE_PATH, "r", encoding="utf-8") as f:
        universe_tickers = [line.strip() for line in f if line.strip()]

    print(f"[*] 学習ユニバース対象: {len(universe_tickers)} 銘柄")
    today_d = datetime.date.today()
    end_date = today_d.strftime("%Y-%m-%d")
    full_start_date = (today_d - datetime.timedelta(days=365 * YEARS_BACK)).strftime("%Y-%m-%d")

    # 1. TOPIX指数の取得・保存（差分更新: 既存キャッシュの最終日翌日から取得）
    existing_topix_df, topix_last_date = load_existing_cache(TOPIX_CACHE_PATH)
    if topix_last_date is not None and topix_last_date >= today_d:
        print(f"[*] TOPIXキャッシュは既に最新です (最終日: {topix_last_date})。スキップ")
    else:
        topix_fetch_start = (topix_last_date + datetime.timedelta(days=1)).strftime("%Y-%m-%d") if topix_last_date else full_start_date
        print(f"[*] TOPIX指数 (0000) を取得中 ({topix_fetch_start} ～ {end_date})...")
        new_topix_df = fetch_index_jquants(session, index_code="0000", from_date=topix_fetch_start, to_date=end_date)
        if new_topix_df is not None and not new_topix_df.empty:
            topix_df = pd.concat([existing_topix_df, new_topix_df]) if existing_topix_df is not None else new_topix_df
            topix_df = topix_df[~topix_df.index.duplicated(keep='last')].sort_index()
            topix_df.to_parquet(TOPIX_CACHE_PATH)
            print(f"[+] TOPIX学習用キャッシュ保存完了: {TOPIX_CACHE_PATH} ({len(topix_df)} 日分)")
        else:
            print("[*] TOPIX: 新規データなし")
    time.sleep(0.55)

    # 2. ユニバース全銘柄の四本値を取得（差分更新: 既存キャッシュの最終日翌日から取得）
    existing_train_df, train_last_date = load_existing_cache(TRAIN_CACHE_PATH)
    if train_last_date is not None and train_last_date >= today_d:
        print(f"[*] ユニバース四本値キャッシュは既に最新です (最終日: {train_last_date})。スキップ")
    else:
        train_fetch_start = (train_last_date + datetime.timedelta(days=1)).strftime("%Y-%m-%d") if train_last_date else full_start_date
        print(f"[*] 全 {len(universe_tickers)} 銘柄の四本値を並列取得中 ({train_fetch_start} ～ {end_date}, {FETCH_WORKERS}並列)...")
        ticker_train_dfs = {}
        with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as executor:
            futures = {
                executor.submit(fetch_ticker_jquants, session, t, train_fetch_start, end_date, rate_limiter): t
                for t in universe_tickers
            }
            done_count = 0
            for future in as_completed(futures):
                t = futures[future]
                df_single = future.result()
                if df_single is not None and not df_single.empty:
                    ticker_train_dfs[t] = df_single

                done_count += 1
                if done_count % 25 == 0 or done_count == len(universe_tickers):
                    print(f"  --> 学習用データ進捗: {done_count}/{len(universe_tickers)} 銘柄完了")

        if ticker_train_dfs:
            new_train_df = pd.concat(ticker_train_dfs, axis=1)
            all_train_df = pd.concat([existing_train_df, new_train_df]) if existing_train_df is not None else new_train_df
            all_train_df = all_train_df[~all_train_df.index.duplicated(keep='last')].sort_index()
            all_train_df.to_parquet(TRAIN_CACHE_PATH)
            print(f"[+] 学習用キャッシュ保存完了: {TRAIN_CACHE_PATH} ({len(all_train_df)} 日分)")
        elif existing_train_df is not None:
            print("[*] ユニバース四本値: 新規データなし（既存キャッシュを維持）")
        else:
            print("[-] 学習用データを取得できませんでした。")

    # =========================================================================
    # PART 3: 日経225原証券価格（マクロ特徴量用）キャッシュ生成
    # J-Quantsは日経225現物指数を配信していないため、日経225オプション四本値
    # レスポンスに含まれる UnderPx (原証券価格) を日経225終値の代替として使う
    # 差分更新: 既存キャッシュの最終日翌日から取得（初回のみ全期間分を取得）
    # =========================================================================
    print("\n" + "=" * 75)
    print("【3/6】日経225原証券価格キャッシュ生成")
    print("=" * 75)

    existing_nk225_df, nk225_last_date = load_existing_cache(NK225_UNDERLYING_CACHE_PATH)
    if nk225_last_date is not None and nk225_last_date >= today_d:
        print(f"[*] 日経225原証券価格キャッシュは既に最新です (最終日: {nk225_last_date})。スキップ")
    else:
        nk225_fetch_start_d = (nk225_last_date + datetime.timedelta(days=1)) if nk225_last_date else datetime.datetime.strptime(full_start_date, "%Y-%m-%d").date()

        business_days = []
        curr = nk225_fetch_start_d
        while curr <= today_d:
            if curr.weekday() < 5:
                business_days.append(curr.strftime("%Y%m%d"))
            curr += datetime.timedelta(days=1)

        print(f"[*] 対象営業日数: {len(business_days)} 日 (このAPIは1日1リクエストのため、{FETCH_WORKERS}並列で取得します)")
        nk225_records = []
        with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as executor:
            futures = {
                executor.submit(fetch_nk225_underlying, session, d_str, rate_limiter): d_str
                for d_str in business_days
            }
            done_count = 0
            for future in as_completed(futures):
                d_str = futures[future]
                under_px = future.result()
                if under_px is not None:
                    nk225_records.append({"Date": pd.to_datetime(d_str), "NK_Close": under_px})

                done_count += 1
                if done_count % 50 == 0 or done_count == len(business_days):
                    print(f"  --> 日経225原証券価格 取得進捗: {done_count}/{len(business_days)} 日完了")

        if nk225_records:
            new_nk225_df = pd.DataFrame(nk225_records).set_index("Date").sort_index()
            nk225_df = pd.concat([existing_nk225_df, new_nk225_df]) if existing_nk225_df is not None else new_nk225_df
            nk225_df = nk225_df[~nk225_df.index.duplicated(keep='last')].sort_index()
            nk225_df.to_parquet(NK225_UNDERLYING_CACHE_PATH)
            print(f"[+] 日経225原証券価格キャッシュ保存完了: {NK225_UNDERLYING_CACHE_PATH} ({len(nk225_df)} 日分)")
        elif existing_nk225_df is not None:
            print("[*] 日経225原証券価格: 新規データなし（既存キャッシュを維持）")
        else:
            print("[-] 日経225原証券価格を取得できませんでした（Standardプラン契約・APIキーをご確認ください）。")

    # =========================================================================
    # PART 4: 信用取引週末残高キャッシュ生成 (需給・大口買い集め検知シグナル用)
    # 週次公表のため、今週分のキャッシュが既にあればスキップする。
    # run_dynamic_regime_screening_v8.py はこのキャッシュを読んで判定に使うため、
    # 必ずスクリーニング実行より前にこのキャッシュを最新化しておく必要がある。
    # =========================================================================
    print("\n" + "=" * 75)
    print("【4/6】信用取引週末残高キャッシュ生成")
    print("=" * 75)

    margin_year, margin_week, _ = today_d.isocalendar()
    margin_cache_path = os.path.join(MARGIN_DIR, f"margin_cache_{margin_year}_w{margin_week:02d}.json")

    if os.path.exists(margin_cache_path):
        print(f"[*] 信用取引週末残高キャッシュは既に今週分があります ({os.path.basename(margin_cache_path)})。スキップ")
    else:
        latest_date, latest_data = fetch_latest_margin_interest(session, today_d, rate_limiter)
        if not latest_data:
            print("[-] 信用取引週末残高を取得できませんでした。")
        else:
            prev_date, prev_data = fetch_latest_margin_interest(
                session, latest_date - datetime.timedelta(days=1), rate_limiter
            )
            prev_map = {rec["Code"]: rec for rec in prev_data} if prev_data else {}

            margin_dict = {}
            for rec in latest_data:
                code = str(rec["Code"])
                ticker = f"{code[:4]}.T"
                long_vol = float(rec.get("LongVol") or 0.0)
                shrt_vol = float(rec.get("ShrtVol") or 0.0)
                ratio = round(long_vol / shrt_vol, 2) if shrt_vol > 0 else 999.0

                prev_rec = prev_map.get(code)
                if prev_rec:
                    prev_long = float(prev_rec.get("LongVol") or 0.0)
                    buy_diff = long_vol - prev_long
                    buy_pct = round((buy_diff / prev_long) * 100, 2) if prev_long > 0 else 0.0
                else:
                    buy_diff = 0.0
                    buy_pct = 0.0

                is_accumulating = (buy_pct <= -3.0) and (ratio <= 5.0)

                margin_dict[ticker] = {
                    "margin_date": latest_date.strftime("%Y-%m-%d"),
                    "margin_buy": long_vol,
                    "margin_buy_diff": buy_diff,
                    "margin_buy_pct": buy_pct,
                    "margin_short": shrt_vol,
                    "margin_ratio": ratio,
                    "is_accumulating": is_accumulating
                }

            with open(margin_cache_path, "w", encoding="utf-8") as f:
                json.dump(margin_dict, f, ensure_ascii=False, indent=2)
            print(f"[+] 信用取引週末残高キャッシュ保存完了: {margin_cache_path} ({len(margin_dict)} 銘柄, 基準日 {latest_date})")

    # =========================================================================
    # PART 5: 決算発表予定日キャッシュ生成 (UIの決算バッジ表示用)
    # 銘柄ごとに1リクエスト必要なため、スクリーニング対象銘柄(qualified_tickers)
    # のみを対象に、今日以降で最も近い予定日だけを抽出してキャッシュする。
    # =========================================================================
    print("\n" + "=" * 75)
    print("【5/6】決算発表予定日キャッシュ生成")
    print("=" * 75)

    earnings_cache_path = os.path.join(CACHE_DIR, f"earnings_calendar_{today_d.strftime('%Y%m%d')}.json")
    if os.path.exists(earnings_cache_path):
        print(f"[*] 決算発表予定日キャッシュは本日分が既にあります。スキップ")
    elif not qualified_tickers:
        print("[-] スクリーニング対象銘柄が無いため、決算発表予定日の取得をスキップします。")
    else:
        today_str = today_d.strftime("%Y-%m-%d")
        print(f"[*] 全 {len(qualified_tickers)} 銘柄の決算発表予定日を{FETCH_WORKERS}並列で取得中...")
        earnings_dict = {}
        with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as executor:
            futures = {
                executor.submit(fetch_earnings_dates, session, t.replace(".T", ""), rate_limiter): t
                for t in qualified_tickers
            }
            done_count = 0
            for future in as_completed(futures):
                ticker = futures[future]
                records = future.result()
                future_records = [r for r in records if r.get("SchDate") and r["SchDate"] >= today_str]
                if future_records:
                    nearest = min(future_records, key=lambda r: r["SchDate"])
                    earnings_dict[ticker] = {
                        "next_earnings_date": nearest["SchDate"],
                        "fq_name": nearest.get("FQName", "")
                    }
                done_count += 1
                if done_count % 100 == 0 or done_count == len(qualified_tickers):
                    print(f"  --> 取得進捗: {done_count}/{len(qualified_tickers)} 銘柄完了")

        with open(earnings_cache_path, "w", encoding="utf-8") as f:
            json.dump(earnings_dict, f, ensure_ascii=False, indent=2)
        print(f"[+] 決算発表予定日キャッシュ保存完了: {earnings_cache_path} ({len(earnings_dict)}/{len(qualified_tickers)} 銘柄で予定日判明)")

    # =========================================================================
    # PART 6: 上場銘柄マスター(会社名)キャッシュ生成 (UI表示用)
    # ETF/REITも含む全銘柄が対象。jpx_sector_master.json(株式のみ・セクター
    # センチメント用)とは別に、UIの会社名表示専用として全銘柄をカバーする。
    # =========================================================================
    print("\n" + "=" * 75)
    print("【6/6】上場銘柄マスター(会社名)キャッシュ生成")
    print("=" * 75)

    company_master_path = os.path.join(CACHE_DIR, f"company_master_{today_d.strftime('%Y%m%d')}.json")
    if os.path.exists(company_master_path):
        print("[*] 上場銘柄マスターキャッシュは本日分が既にあります。スキップ")
    else:
        # PART1冒頭で既に取得済みのマスターを使い回す(APIリクエスト節約)
        if not master_records:
            print("[-] 上場銘柄マスターを取得できませんでした。")
        else:
            company_dict = {}
            for rec in master_records:
                code = str(rec["Code"])
                ticker = f"{code[:4]}.T"
                company_dict[ticker] = {
                    "name": rec.get("CoName", ""),
                    "name_en": rec.get("CoNameEn", "")
                }
            with open(company_master_path, "w", encoding="utf-8") as f:
                json.dump(company_dict, f, ensure_ascii=False, indent=2)
            print(f"[+] 上場銘柄マスターキャッシュ保存完了: {company_master_path} ({len(company_dict)} 銘柄, 基準日 {master_date})")

    print("\n[+] 全キャッシュ生成パイプラインが正常に完了しました。")

if __name__ == "__main__":
    main()