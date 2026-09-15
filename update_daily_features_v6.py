import os
import re
import time
import datetime
import urllib.request
import unicodedata
import numpy as np
import pandas as pd
import openpyxl
import requests
from dotenv import load_dotenv

# 平日祝日の自動判定用（jpholidayが未インストールの場合はフォールバック）
try:
    import jpholiday
    HAS_JPHOLIDAY = True
except ImportError:
    HAS_JPHOLIDAY = False

BASE_DIR = r"F:\stockModel"
DB_PATH = os.path.join(BASE_DIR, "jpx_daily_features_db.csv")
WEEKLY_DIR = os.path.join(BASE_DIR, "jpx_weekly_oi")
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

load_dotenv()
JQUANTS_API_KEY = os.environ.get("JQUANTS_API_KEY")
JQUANTS_BASE_URL = "https://api.jquants.com/v2"

def is_market_holiday(d: datetime.date) -> bool:
    """土日・日本の祝日・年末年始（12/31〜1/3）を判定"""
    if d.weekday() >= 5:
        return True
    # 年末年始休暇
    if (d.month == 12 and d.day == 31) or (d.month == 1 and d.day in [1, 2, 3]):
        return True
    # 祝日判定
    if HAS_JPHOLIDAY and jpholiday.is_holiday(d):
        return True
    return False

# --- 1. 週次建玉Excelの自動探索 & ダウンロード ---
def fetch_latest_weekly_oi(target_date: datetime.date, file_type: str):
    os.makedirs(WEEKLY_DIR, exist_ok=True)

    # 直近の金曜日から探索
    days_back = (target_date.weekday() - 4) % 7
    # 月曜日の公表前（17時前）なら前週の金曜分へシフト
    if target_date.weekday() == 0 and datetime.datetime.now().hour < 17:
        days_back += 7

    for offset in range(days_back, days_back + 21, 7):
        # 祝日シフト考慮（金曜が祝日の場合の木・水・火）
        for day_shift in [0, 1, 2, 3]:
            ref_d = target_date - datetime.timedelta(days=offset + day_shift)
            year = ref_d.strftime("%Y")  # 年跨ぎ（1月実行時）も基準日ベースでパスを構築
            ymd = ref_d.strftime("%Y%m%d")
            fname = f"{ymd}_{file_type}_oi_by_tp.xlsx"
            save_path = os.path.join(WEEKLY_DIR, fname)

            # 既存ファイルが有効サイズ（5KB以上）であれば利用
            if os.path.exists(save_path) and os.path.getsize(save_path) > 5000:
                return save_path

            url = f"https://www.jpx.co.jp/automation/markets/derivatives/open-interest/files/{year}/{fname}"
            try:
                req = urllib.request.Request(url, headers=HEADERS)
                with urllib.request.urlopen(req, timeout=10) as resp:
                    content = resp.read()
                    
                    # 404等のHTMLエラーページ混入チェック
                    if b"<!DOCTYPE html" in content[:150] or b"<html" in content[:150]:
                        continue
                    if len(content) < 5000:
                        continue

                    with open(save_path, "wb") as f:
                        f.write(content)

                print(f"  [+] 新規週次建玉ファイル取得成功 ({file_type}): {fname}")
                return save_path
            except Exception:
                continue

    return None

# --- 2. J-Quants日経225オプションAPIからの建玉壁計算 ---
def fetch_op_walls_jquants(target_date: datetime.date, session=None, rate_limiter=None):
    """J-Quants日経225オプション四本値(市場全体の合計建玉)から、直近限月のコール/プット壁を計算"""
    if not JQUANTS_API_KEY:
        return None, None
    date_str = target_date.strftime("%Y%m%d")
    url = f"{JQUANTS_BASE_URL}/derivatives/bars/daily/options/225"
    try:
        if session is None:
            session = requests.Session()
            session.headers.update({"x-api-key": JQUANTS_API_KEY})

        res = None
        for attempt in range(4):
            if rate_limiter is not None:
                rate_limiter.acquire()
            res = session.get(url, params={"date": date_str}, timeout=15)
            if res.status_code == 429:
                wait_sec = 2.0 * (attempt + 1)
                time.sleep(wait_sec)
                continue
            break

        if res is None or res.status_code != 200:
            print(f"  [!] J-Quantsオプション取得エラー: HTTP {res.status_code if res is not None else '?'} ({date_str})")
            return None, None
        data = res.json().get("data", [])
        if not data:
            return None, None

        df = pd.DataFrame(data)
        front_month = sorted(df['CM'].unique())[0]
        front_df = df[df['CM'] == front_month]

        put_df = front_df[front_df['PCDiv'] == '1']
        call_df = front_df[front_df['PCDiv'] == '2']
        if put_df.empty or call_df.empty:
            return None, None

        p_wall = float(put_df.loc[put_df['OI'].idxmax(), 'Strike'])
        c_wall = float(call_df.loc[call_df['OI'].idxmax(), 'Strike'])
        return c_wall, p_wall
    except Exception as e:
        print(f"  [!] J-Quantsオプション解析エラー: {e}")
        return None, None

# --- 3. CTA先物純建玉パーサー（正規化・記号ゆれ完全対応版） ---
def parse_fut_cta(filepath: str):
    if not filepath or not os.path.exists(filepath):
        return None
    try:
        wb = openpyxl.load_workbook(filepath, data_only=True)
        ws = wb.active
        
        target_cta = ['ABNクリアリン', 'バークレイズ', 'ソシエテ']
        current_product = None
        cta_net = 0.0

        for row in ws.iter_rows(values_only=True):
            # 文字列をNFKC正規化した上で判定
            col0 = unicodedata.normalize('NFKC', str(row[0] or "")).strip()

            # 括弧の全角・半角に依存せずキーワードで判定
            if "日経225先物" in col0 and "mini" not in col0.lower():
                current_product = 'large'
                continue
            elif "日経225mini" in col0.lower() or "日経225ミニ" in col0:
                current_product = 'mini'
                continue
            elif "<" in col0 or "＜" in col0:
                current_product = 'other'
                continue

            if current_product in ['large', 'mini']:
                scale = 1.0 if current_product == 'large' else 0.1
                seller = unicodedata.normalize('NFKC', str(row[3] or "")).strip()
                buyer = unicodedata.normalize('NFKC', str(row[6] or "")).strip()
                
                s = float(row[4]) * scale if len(row) > 4 and isinstance(row[4], (int, float)) else 0.0
                b = float(row[7]) * scale if len(row) > 7 and isinstance(row[7], (int, float)) else 0.0

                if any(k in buyer for k in target_cta):
                    cta_net += b
                if any(k in seller for k in target_cta):
                    cta_net -= s

        return float(cta_net)

    except Exception as e:
        print(f"  [!] 先物解析エラー: {e}")
        return None

# --- 4. 日次同期メイン実行 ---
def update_daily():
    today = datetime.date.today()
    today_str = today.strftime("%Y-%m-%d")

    print("=" * 65)
    print(f"[*] 日次マクロデータベース同期実行: {today_str}")
    print("=" * 65)

    # 休場日判定（土日/祝日/年末年始）
    if is_market_holiday(today):
        print(f"  [-] 本日は市場休場日（土日/祝日/年末年始）のため、DB追記をスキップします。")
        return

    # DBが存在しない場合は初期化
    if not os.path.exists(DB_PATH):
        print(f"  [*] {DB_PATH} が存在しないため、新規作成します。")
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        df_db = pd.DataFrame(columns=['call_oi_wall', 'put_oi_wall', 'gamma_flip', 'cta_net_futures'])
        df_db.index.name = 'Date'
        last_valid = {}
    else:
        df_db = pd.read_csv(DB_PATH, index_col=0, parse_dates=True)
        last_valid = df_db.iloc[-1].to_dict() if len(df_db) > 0 else {}

    # コール/プット壁: J-Quants日経225オプションAPI(市場全体の合計建玉)から計算
    c_wall, p_wall = fetch_op_walls_jquants(today)

    # CTA先物ネットポジション: J-Quantsに投資部門別建玉の代替がない(J-Quants Pro限定)ため、
    # 引き続きJPXウェブサイトの週次確定Excelから取得
    p_fut = fetch_latest_weekly_oi(today, "indexfut")
    cta_net = parse_fut_cta(p_fut)

    # 取得できない場合は前営業日値またはデフォルト値を使用(取得元不明を防ぐため明示的に警告)
    if c_wall is None:
        c_wall = last_valid.get('call_oi_wall', 40000.0)
        print(f"  [!] Call壁を取得できなかったため、フォールバック値を使用します: {c_wall:.0f}")
    if p_wall is None:
        p_wall = last_valid.get('put_oi_wall', 38000.0)
        print(f"  [!] Put壁を取得できなかったため、フォールバック値を使用します: {p_wall:.0f}")
    if cta_net is None:
        cta_net = last_valid.get('cta_net_futures', 0.0)
        print(f"  [!] CTA先物を取得できなかったため、フォールバック値を使用します: {cta_net:+.1f}")
    g_flip = (c_wall + p_wall) / 2.0

    print(f"  [+] 反映建玉水準: Call壁={c_wall:.0f} / Put壁={p_wall:.0f} / Flip={g_flip:.0f}")
    print(f"  [+] 反映CTA先物 : {cta_net:+.1f} 枚 (ラージ換算)")

    # レコード反映
    df_db.loc[pd.to_datetime(today_str)] = {
        'call_oi_wall': float(c_wall),
        'put_oi_wall': float(p_wall),
        'gamma_flip': float(g_flip),
        'cta_net_futures': float(cta_net)
    }

    df_db.sort_index(inplace=True)
    df_db.to_csv(DB_PATH)
    print(f"\n[★] {DB_PATH} の更新完了（総日数: {len(df_db)} 営業日）")
    print(df_db.tail(3))

if __name__ == "__main__":
    update_daily()