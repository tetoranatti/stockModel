import io
import pandas as pd
import requests

def build_universe_with_jpx(input_csv_path="screener_result.csv", output_txt="universe_150_tickers.txt"):
    # 1. JPX公式マスター直リンク
    jpx_url = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xlsx"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    
    print("[*] Downloading JPX master...")
    resp = requests.get(jpx_url, headers=headers)
    resp.raise_for_status()

    # JPXのExcel読み込み (JPX公式フォーマット: 列1=コード, 列9=33業種区分)
    df_jpx_raw = pd.read_excel(io.BytesIO(resp.content))
    
    # 列名ではなく列の番号（インデックス）で安全に取得してリネーム
    # JPXのエクセルは通常：0:日付, 1:コード, 2:銘柄名, 3:市場区分, 9:33業種区分
    c_code = df_jpx_raw.columns[1]
    c_sec = df_jpx_raw.columns[9] if len(df_jpx_raw.columns) > 9 else df_jpx_raw.columns[len(df_jpx_raw.columns)//2]
    
    df_jpx = df_jpx_raw[[c_code, c_sec]].copy()
    df_jpx.columns = ['code', 'sector']
    df_jpx['clean_code'] = df_jpx['code'].astype(str).str.extract(r'(\d{4})')[0] + ".T"
    df_jpx = df_jpx.dropna(subset=['clean_code']).drop_duplicates(subset=['clean_code'])
    print(f"[+] JPX Master Loaded: {len(df_jpx)} rows")

    # 2. SBI証券の806銘柄CSV読み込み
    try:
        df_sbi = pd.read_csv(input_csv_path, encoding="cp932")
    except Exception:
        df_sbi = pd.read_csv(input_csv_path, encoding="utf-8")

    # 銘柄コード列の特定（4桁の数字が含まれる列を自動判別）
    code_col = None
    for c in df_sbi.columns:
        if df_sbi[c].astype(str).str.contains(r'^\d{4}$').any():
            code_col = c
            break
    if code_col is None:
        code_col = df_sbi.columns[0]

    # 売買代金列の特定（数値化して最も中央値が大きい列を代金と判定）
    numeric_cols = []
    for c in df_sbi.columns:
        s = df_sbi[c].astype(str).str.replace(',', '').str.strip()
        converted = pd.to_numeric(s, errors='coerce')
        if converted.notnull().sum() > len(df_sbi) * 0.5:
            numeric_cols.append((c, converted.median()))
    
    # 最も金額規模が大きい列を売買代金と判定
    numeric_cols.sort(key=lambda x: x[1], reverse=True)
    val_col = numeric_cols[0][0]
    
    df_sbi['clean_code'] = df_sbi[code_col].astype(str).str.extract(r'(\d{4})')[0] + ".T"
    df_sbi['val'] = pd.to_numeric(df_sbi[val_col].astype(str).str.replace(',', ''), errors='coerce')
    df_sbi = df_sbi.dropna(subset=['clean_code', 'val']).drop_duplicates(subset=['clean_code'])

    # 3. 突合（Left Join）
    df_merged = pd.merge(df_sbi, df_jpx, on='clean_code', how='left')
    df_merged['sector'] = df_merged['sector'].fillna('Other')
    df_merged = df_merged.sort_values(by='val', ascending=False).reset_index(drop=True)
    print(f"[*] Merged: {len(df_merged)} stocks / Unique sectors: {df_merged['sector'].nunique()}")

    # 4. 150銘柄選定
    selected = []

    # ① 超大型・マクロ連動枠 (トップ50)
    tier1 = df_merged.iloc[:50]['clean_code'].tolist()
    selected.extend(tier1)
    remaining = df_merged.iloc[50:].copy()

    # ② セクター分散枠 (各セクター上位均等抽出: 70銘柄)
    sectors = [s for s in remaining['sector'].unique() if s not in ['-', 'Other']]
    n_per_sec = max(2, 70 // max(1, len(sectors)))
    tier2 = []
    for sec in sectors:
        sec_df = remaining[remaining['sector'] == sec]
        tier2.extend(sec_df.head(n_per_sec)['clean_code'].tolist())
        if len(tier2) >= 70:
            break
    selected.extend(tier2[:70])

    # ③ 中小型反発枠 (下位層から30銘柄サンプリング)
    tier3_pool = df_merged[~df_merged['clean_code'].isin(selected)].copy()
    tier3 = tier3_pool.tail(80).sample(n=min(30, len(tier3_pool)), random_state=42)['clean_code'].tolist()
    selected.extend(tier3)

    final_tickers = list(dict.fromkeys(selected))[:150]
    print(f"[+] Successfully selected: {len(final_tickers)} tickers")

    with open(output_txt, "w") as f:
        f.write("\n".join(final_tickers))
    print(f"[+] Saved to: {output_txt}")

if __name__ == "__main__":
    build_universe_with_jpx()