import os
import re
import datetime
import urllib.request
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import pdfplumber

BASE_DIR = r"F:\stockModel"
DB_PATH = os.path.join(BASE_DIR, "jpx_daily_features_db.csv")
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# --- 1. 空売り比率（当日分）の取得 ---
def fetch_today_short_ratio(target_date):
    yymmdd = target_date.strftime("%y%m%d")
    fname = f"{yymmdd}-m.pdf"
    pdf_save_path = os.path.join(BASE_DIR, "jpx_short_pdf", fname)
    os.makedirs(os.path.dirname(pdf_save_path), exist_ok=True)

    top_url = "https://www.jpx.co.jp/markets/statistics-equities/short-selling/index.html"
    try:
        req = urllib.request.Request(top_url, headers=HEADERS)
        with urllib.request.urlopen(req) as resp:
            soup = BeautifulSoup(resp.read().decode('utf-8', errors='ignore'), 'html.parser')
        
        pdf_url = None
        for a in soup.find_all('a', href=True):
            if fname in a['href']:
                href = a['href']
                pdf_url = href if href.startswith("http") else "https://www.jpx.co.jp" + (href if href.startswith('/') else '/' + href)
                break
        
        if not pdf_url:
            print(f"  [-] 本日の空売りPDFが見つかりません（16:00以降に公開されます）")
            return None

        req_pdf = urllib.request.Request(pdf_url, headers=HEADERS)
        with urllib.request.urlopen(req_pdf) as resp:
            with open(pdf_save_path, "wb") as f:
                f.write(resp.read())
        print(f"  [+] 空売りPDF取得成功: {fname}")

        with pdfplumber.open(pdf_save_path) as pdf:
            text = "".join([p.extract_text() or "" for p in pdf.pages])
            matches = re.findall(r'(\d{1,2}\.\d)\s*%', text)
            if len(matches) >= 3:
                # [実注文比率(a), 規制あり(b), 規制なし(c)]
                b = float(matches[1])
                c = float(matches[2])
                return round(b + c, 2)
    except Exception as e:
        print(f"  [!] 空売り取得エラー: {e}")
    return None

# --- 2. 手口・壁（当日分）の取得 ---
def parse_option_symbol(symbol_str):
    s = str(symbol_str).strip()
    if "225" not in s and "NK" not in s:
        return None, None
    m = re.search(r'([CPcp])\d{4}-(\d{4,6})', s)
    if m:
        cp = 'call' if m.group(1).upper() == 'C' else 'put'
        strike = float(m.group(2))
        if 20000.0 <= strike <= 85000.0:
            return cp, strike
    return None, None

def parse_jpx_excel(filepath):
    c_vol, p_vol = {}, {}
    if not os.path.exists(filepath):
        return c_vol, p_vol
    try:
        xls = pd.ExcelFile(filepath)
        for sheet in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet)
            if df.shape[1] < 8:
                continue
            for _, row in df.iterrows():
                try:
                    vol = float(str(row.iloc[7]).replace(',', ''))
                    if vol <= 0:
                        continue
                except Exception:
                    continue
                cp, strike = parse_option_symbol(row.iloc[2])
                if cp and strike:
                    if cp == 'call':
                        c_vol[strike] = c_vol.get(strike, 0.0) + vol
                    else:
                        p_vol[strike] = p_vol.get(strike, 0.0) + vol
    except Exception as e:
        print(f"  [!] Excel解析エラー ({os.path.basename(filepath)}): {e}")
    return c_vol, p_vol

def fetch_today_walls(target_date, last_valid):
    ymd = target_date.strftime("%Y%m%d")
    ym = target_date.strftime("%Y%m")
    save_dir = os.path.join(BASE_DIR, "jpx_raw_year")
    os.makedirs(save_dir, exist_ok=True)

    p_auc = os.path.join(save_dir, f"{ymd}_volume_by_participant_whole_day.xlsx")
    p_jnet = os.path.join(save_dir, f"{ymd}_volume_by_participant_whole_day_J-NET.xlsx")

    # 【正規エンドポイント】automation/markets/derivatives/participant-volume/files/daily/YYYYMM/
    base_url = f"https://www.jpx.co.jp/automation/markets/derivatives/participant-volume/files/daily/{ym}/"
    
    for p, suffix in [(p_auc, "whole_day.xlsx"), (p_jnet, "whole_day_J-NET.xlsx")]:
        if not os.path.exists(p) or os.path.getsize(p) < 1000:
            try:
                url = f"{base_url}{ymd}_volume_by_participant_{suffix}"
                req = urllib.request.Request(url, headers=HEADERS)
                with urllib.request.urlopen(req) as resp:
                    with open(p, "wb") as f:
                        f.write(resp.read())
                print(f"  [+] ダウンロード成功: {os.path.basename(p)}")
            except Exception:
                print(f"  [-] ダウンロード未完了 ({suffix}): 17:15以降の公開をお待ちください")

    c_auc, p_auc_map = parse_jpx_excel(p_auc)
    c_jnet, p_jnet_map = parse_jpx_excel(p_jnet)

    all_call = {s: c_auc.get(s, 0.0) + c_jnet.get(s, 0.0) for s in set(c_auc.keys()) | set(c_jnet.keys())}
    all_put = {s: p_auc_map.get(s, 0.0) + p_jnet_map.get(s, 0.0) for s in set(p_auc_map.keys()) | set(p_jnet_map.keys())}

    call_wall = max(all_call, key=all_call.get) if all_call and max(all_call.values()) > 0 else last_valid['call_oi_wall']
    put_wall = max(all_put, key=all_put.get) if all_put and max(all_put.values()) > 0 else last_valid['put_oi_wall']
    gamma_flip = (call_wall + put_wall) / 2.0

    return float(call_wall), float(put_wall), float(gamma_flip)

# --- 3. 統合実行関数 ---
def update_daily():
    today = datetime.date.today()
    today_str = today.strftime("%Y-%m-%d")

    print("=" * 65)
    print(f"[*] 日次データベース更新バッチ実行: {today_str}")
    print("=" * 65)

    if not os.path.exists(DB_PATH):
        print(f"[!] DBが見つかりません: {DB_PATH}")
        return

    df_db = pd.read_csv(DB_PATH, index_col=0, parse_dates=True)
    last_valid = df_db.iloc[-1].to_dict()

    # 空売り比率
    short_ratio = fetch_today_short_ratio(today)
    if short_ratio is None:
        short_ratio = last_valid.get('short_selling_ratio', 40.0)
        print(f"  [!] 空売り比率は前日値（{short_ratio}%）を引き継ぎます")
    else:
        print(f"  [+] 本日の空売り比率: {short_ratio}%")

    # 手口・壁データ
    c_wall, p_wall, g_flip = fetch_today_walls(today, last_valid)
    print(f"  [+] 本日の壁・ガンマ: Call={c_wall:.0f} / Put={p_wall:.0f} / Flip={g_flip:.0f}")

    # DBに当日行を更新（既存日付なら上書き、新規なら追記）
    df_db.loc[pd.to_datetime(today_str)] = {
        'call_oi_wall': c_wall,
        'put_oi_wall': p_wall,
        'gamma_flip': g_flip,
        'short_selling_ratio': short_ratio
    }

    df_db.sort_index(inplace=True)
    df_db.to_csv(DB_PATH)
    print(f"\n[★] {DB_PATH} の更新が完了しました。（総日数: {len(df_db)} 日）")

if __name__ == "__main__":
    update_daily()