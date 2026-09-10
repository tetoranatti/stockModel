import os
import re
import glob
import shutil
import numpy as np
import pandas as pd
import torch
import yfinance as yf
from train_model_v6_universe_v5 import DualStreamGRU_v6_Slim5, load_macro_slim5

# =============================================================================
# 1. パス & バックテスト実証ハイパーパラメータ
# =============================================================================
BASE_DIR = r"F:\stockModel"
MODEL_WEIGHTS = os.path.join(BASE_DIR, "swing_model_v6_crossattn_universe_v5.pt")
SCREENER_CSV = os.path.join(BASE_DIR, "screener_result.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "direct_screening_results_v5.csv")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN = 10
# 変更前 (5億円)
# MIN_TURNOVER = 5e8

# 変更後 (10億円)
MIN_TURNOVER = 10e8

# バックテスト最適化カットオフ基準
THRESHOLD_BUY = 0.37    # 本命エントリーライン (PF: 2.42 / 勝率: 56.7%)
THRESHOLD_WATCH = 0.355 # 打診・監視候補ライン (PF: 1.61水準 / 勝率: 51.9%)
MIN_VOL_RATIO = 1.0     # 出来高モメンタムフィルター (5日平均比)

# =============================================================================
# 2. CSV自動同期 & 銘柄コード読み込み
# =============================================================================
def sync_latest_csv(target_path):
    downloads_dir = os.path.join(os.path.expanduser("~"), "Downloads")
    pattern = os.path.join(downloads_dir, "*screener*.csv")
    sbi_files = glob.glob(pattern)
    if sbi_files:
        latest_dl = max(sbi_files, key=os.path.getmtime)
        if not os.path.exists(target_path) or os.path.getmtime(latest_dl) > os.path.getmtime(target_path):
            try:
                shutil.copy2(latest_dl, target_path)
                print(f"[*] Downloadsフォルダから最新CSVを同期: {os.path.basename(latest_dl)}")
            except Exception:
                pass
    return target_path

def load_tickers(csv_path):
    csv_path = sync_latest_csv(csv_path)
    if not os.path.exists(csv_path):
        univ_file = os.path.join(BASE_DIR, "universe_150_tickers.txt")
        if os.path.exists(univ_file):
            with open(univ_file, "r", encoding="utf-8") as f:
                return [line.strip() for line in f if line.strip()]
        return ["7203.T", "6758.T", "8035.T", "8306.T", "9432.T"]

    df = None
    for enc in ['cp932', 'shift_jis', 'utf-8']:
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            break
        except Exception:
            continue

    candidate_cols = ['コード', '銘柄コード', 'Code', 'code', 'Ticker', 'ticker']
    target_col = next((c for c in candidate_cols if any(c in str(col) for col in df.columns)), df.columns[0])

    tickers = []
    for val in df[target_col].dropna():
        s = str(val).strip()
        match = re.match(r"^(\d{4})", s)
        if match:
            tickers.append(f"{match.group(1)}.T")
        elif s.endswith(".T"):
            tickers.append(s)
    return sorted(list(dict.fromkeys(tickers)))

# =============================================================================
# 3. 1パス ダイレクト推論メインルーチン
# =============================================================================
def main():
    print("=" * 80)
    print("【1パス・ダイレクト推論】800銘柄全件スキャン & バックテスト実証判定")
    print("=" * 80)

    if not os.path.exists(MODEL_WEIGHTS):
        print(f"[!] モデル重みが見つかりません: {MODEL_WEIGHTS}")
        return

    checkpoint = torch.load(MODEL_WEIGHTS, map_location=DEVICE)
    stock_cols = checkpoint['stock_cols']
    macro_cols = checkpoint['macro_cols']
    hidden_dim = checkpoint.get('hidden_dim', 24)

    model = DualStreamGRU_v6_Slim5(
        stock_dim=len(stock_cols),
        macro_dim=len(macro_cols),
        hidden_dim=hidden_dim,
        num_classes=3
    ).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    macro_df = load_macro_slim5()
    tickers = load_tickers(SCREENER_CSV)
    print(f"[*] スキャン対象母集団: {len(tickers)} 銘柄\n")

    results = []
    m_start = (macro_df.index.min() - pd.Timedelta(days=40)).strftime("%Y-%m-%d")

    for i, ticker in enumerate(tickers):
        try:
            df = yf.download(ticker, start=m_start, interval="1d", progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df = df.dropna(subset=['Close', 'High', 'Low', 'Volume'])

            if len(df) < SEQ_LEN + 20:
                continue

            # 売買代金チェック（5日平均 >= 10億円）
            turnover_5d = (df['Close'] * df['Volume']).rolling(5).mean().iloc[-1]
            if turnover_5d < MIN_TURNOVER:
                continue

            # 特徴量生成 (10次元極限スリム版と同期)
            df['stock_ret_1d'] = df['Close'].pct_change(1, fill_method=None).fillna(0.0)
            df['stock_ret_5d'] = df['Close'].pct_change(5, fill_method=None).fillna(0.0)
            vol_5d = df['Volume'].rolling(5).mean()
            df['vol_ratio_5d'] = (df['Volume'] / (vol_5d + 1e-7)).fillna(1.0)

            hl = df['High'] - df['Low']
            h_cp = (df['High'] - df['Close'].shift(1)).abs()
            l_cp = (df['Low'] - df['Close'].shift(1)).abs()
            tr = pd.concat([hl, h_cp, l_cp], axis=1).max(axis=1)
            atr = tr.rolling(14).mean()
            df['ATR'] = atr
            df['atr_ratio'] = (atr / (df['Close'] + 1e-7)).fillna(0.0)

            aligned_nk = macro_df['NK_Ret'].reindex(df.index).fillna(0.0)
            cov = df['stock_ret_1d'].rolling(20).cov(aligned_nk)
            var = aligned_nk.rolling(20).var()
            df['rolling_beta'] = (cov / (var + 1e-7)).fillna(1.0)

            df = df.join(macro_df, how='inner')
            df = df.dropna(subset=['ATR', 'rolling_beta'] + macro_cols)
            if len(df) < SEQ_LEN:
                continue

            # 直近10日のZ-Score標準化
            w_s = df[stock_cols].iloc[-SEQ_LEN:].values
            w_m = df[macro_cols].iloc[-SEQ_LEN:].values
            w_s_norm = (w_s - w_s.mean(axis=0)) / (w_s.std(axis=0) + 1e-7)
            w_m_norm = (w_m - w_m.mean(axis=0)) / (w_m.std(axis=0) + 1e-7)

            t_s = torch.tensor(w_s_norm, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            t_m = torch.tensor(w_m_norm, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                logits = model(t_s, t_m)
                probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

            p_loss = float(probs[0])
            p_win = float(probs[2])

            curr_close = float(df['Close'].iloc[-1])
            curr_atr = float(df['ATR'].iloc[-1])
            vol_ratio = float(df['vol_ratio_5d'].iloc[-1])
            beta_val = float(df['rolling_beta'].iloc[-1])

            # 目標株価・損切ライン (+2ATR / -1ATR)
            target_p = curr_close + (2.0 * curr_atr)
            stop_p = curr_close - (1.0 * curr_atr)

            # バックテスト検証判定
            is_volume_ok = vol_ratio >= MIN_VOL_RATIO
            if p_win >= THRESHOLD_BUY and is_volume_ok:
                decision = "🎯 BUY"
            elif p_win >= THRESHOLD_WATCH and is_volume_ok:
                decision = "👀 WATCH"
            else:
                decision = "⏸️ WAIT"

            # 監視・買い候補、または利確確率上位を記録
            results.append({
                "Ticker": ticker,
                "現在値": round(curr_close, 1),
                "利確目標(+2ATR)": round(target_p, 1),
                "損切撤退(-1ATR)": round(stop_p, 1),
                "生利確確率": f"{p_win * 100:.1f}%",
                "生損切確率": f"{p_loss * 100:.1f}%",
                "出来高倍率": f"{vol_ratio:.2f}x",
                "5日売買代金(億)": round(float(turnover_5d) / 1e8, 1),
                "ベータ": round(beta_val, 2),
                "AI判定": decision,
                "_p_win": p_win
            })
        except Exception:
            continue

        if (i + 1) % 100 == 0 or (i + 1) == len(tickers):
            print(f"  --> {i + 1}/{len(tickers)} 銘柄 走査完了 (有効抽出: {len(results)} 件)")

    df_res = pd.DataFrame(results)
    if not df_res.empty:
        # 利確確率の高い順にソート
        df_res = df_res.sort_values(by="_p_win", ascending=False).drop(columns=["_p_win"])
        df_res.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

        print("\n" + "=" * 95)
        print("【スクリーニング結果上位 30 銘柄】")
        print("=" * 95)
        print(df_res.head(30).to_string(index=False))
        print("=" * 95)
        print(f"[+] 全件の解析結果を保存しました: {OUTPUT_CSV}")

        buys = df_res[df_res['AI判定'] == "🎯 BUY"]
        watches = df_res[df_res['AI判定'] == "👀 WATCH"]
        print(f"\n[★] 本日の『🎯 BUY』銘柄: {len(buys)} 件")
        print(f"[★] 本日の『👀 WATCH』銘柄: {len(watches)} 件")
    else:
        print("[!] 解析結果が得られませんでした。")

if __name__ == "__main__":
    main()