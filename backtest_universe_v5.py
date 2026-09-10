import os
import numpy as np
import pandas as pd
import torch
import yfinance as yf
from train_model_v6_universe_v5 import DualStreamGRU_v6_Slim5, load_macro_slim5

# =============================================================================
# 1. 設定 & パス
# =============================================================================
BASE_DIR = r"F:\stockModel"
MODEL_WEIGHTS = os.path.join(BASE_DIR, "swing_model_v6_crossattn_universe_v5.pt")
UNIVERSE_PATH = os.path.join(BASE_DIR, "universe_150_tickers.txt")
OUTPUT_CSV = os.path.join(BASE_DIR, "backtest_universe_v5_results.csv")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN = 10
HORIZON = 5
THRESHOLDS = [0.34, 0.35, 0.36, 0.37, 0.38, 0.40]

# =============================================================================
# 2. 単一銘柄のバックテスト評価関数 (ATRトリプルバリア)
# =============================================================================
def evaluate_ticker(model, ticker, macro_df, stock_cols, macro_cols):
    try:
        m_start = (macro_df.index.min() - pd.Timedelta(days=40)).strftime("%Y-%m-%d")
        df = yf.download(ticker, start=m_start, interval="1d", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df = df.dropna(subset=['Close', 'High', 'Low', 'Volume'])
        if len(df) < SEQ_LEN + HORIZON + 20:
            return []

        # 特徴量生成 (学習時と完全一致)
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
        if len(df) < SEQ_LEN + HORIZON + 5:
            return []

        vals_s = df[stock_cols].values
        vals_m = df[macro_cols].values
        close_p = df['Close'].values
        high_p = df['High'].values
        low_p = df['Low'].values
        atr_p = df['ATR'].values
        open_p = df['Open'].values if 'Open' in df.columns else close_p

        # 検証区間 (時系列スプリットの後半25%をテスト区間に設定)
        test_start = int(len(df) * 0.75)
        signals = []

        for idx in range(max(SEQ_LEN, test_start), len(df) - HORIZON):
            w_s = vals_s[idx - SEQ_LEN:idx]
            w_m = vals_m[idx - SEQ_LEN:idx]
            w_s_norm = (w_s - w_s.mean(axis=0)) / (w_s.std(axis=0) + 1e-7)
            w_m_norm = (w_m - w_m.mean(axis=0)) / (w_m.std(axis=0) + 1e-7)

            t_s = torch.tensor(w_s_norm, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            t_m = torch.tensor(w_m_norm, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                probs = torch.softmax(model(t_s, t_m), dim=-1).squeeze(0).cpu().numpy()

            p_win = float(probs[2])
            entry_p = open_p[idx + 1]
            curr_atr = atr_p[idx]
            upper_barrier = entry_p + (2.0 * curr_atr)
            lower_barrier = entry_p - (1.0 * curr_atr)

            # エグジット判定
            exit_ret = 0.0
            is_win = False
            for h in range(1, HORIZON + 1):
                cur_low = low_p[idx + h]
                cur_high = high_p[idx + h]

                if cur_low <= lower_barrier and cur_high >= upper_barrier:
                    exit_ret = (lower_barrier - entry_p) / entry_p
                    break
                elif cur_high >= upper_barrier:
                    exit_ret = (upper_barrier - entry_p) / entry_p
                    is_win = True
                    break
                elif cur_low <= lower_barrier:
                    exit_ret = (lower_barrier - entry_p) / entry_p
                    break
                elif h == HORIZON:
                    exit_ret = (close_p[idx + h] - entry_p) / entry_p
                    if exit_ret > 0:
                        is_win = True

            signals.append({
                "p_win": p_win,
                "ret": exit_ret,
                "is_win": is_win,
                "idx": idx
            })

        return signals
    except Exception:
        return []

# =============================================================================
# 3. バックテスト集計メインルーチン
# =============================================================================
def main():
    print("=" * 70)
    print("[*] v5モデル バックテスト実行 (10次元極限スリム / ATRトリプルバリア)")
    print("=" * 70)

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

    with open(UNIVERSE_PATH, "r", encoding="utf-8") as f:
        tickers = [line.strip() for line in f if line.strip()]

    print(f"[*] 対象銘柄数: {len(tickers)} 銘柄")
    print("[*] 検証シグナルを抽出中...")

    all_signals = []
    for i, t in enumerate(tickers):
        sigs = evaluate_ticker(model, t, macro_df, stock_cols, macro_cols)
        all_signals.extend(sigs)
        if (i + 1) % 30 == 0 or (i + 1) == len(tickers):
            print(f"  --> {i + 1}/{len(tickers)} 銘柄完了 (取得シグナル数: {len(all_signals)})")

    if not all_signals:
        print("[!] 有効なシグナルが抽出できませんでした。")
        return

    # しきい値ごとの集計
    summary = []
    for th in THRESHOLDS:
        # 重複エントリーを防止するため、同一銘柄内でのインデックス重複を考慮
        th_trades = [s for s in all_signals if s['p_win'] >= th]
        total_trades = len(th_trades)

        if total_trades == 0:
            continue

        wins = [s['ret'] for s in th_trades if s['ret'] > 0]
        losses = [abs(s['ret']) for s in th_trades if s['ret'] < 0]

        win_rate = (len(wins) / total_trades) * 100
        gross_profit = sum(wins)
        gross_loss = sum(losses)
        pf = (gross_profit / (gross_loss + 1e-7)) if gross_loss > 0 else 99.9
        avg_ret = (sum(s['ret'] for s in th_trades) / total_trades) * 100

        summary.append({
            "Threshold": th,
            "Trades": total_trades,
            "WinRate(%)": round(win_rate, 2),
            "PF": round(pf, 2),
            "AvgRet(%)": round(avg_ret, 2)
        })

    sum_df = pd.DataFrame(summary)
    print("\n" + "=" * 65)
    print("【バックテスト検証結果サマリ】")
    print("=" * 65)
    print(sum_df.to_string(index=False))
    print("=" * 65)

    sum_df.to_csv(OUTPUT_CSV, index=False)
    print(f"[+] 結果を保存しました: {OUTPUT_CSV}")

    if not sum_df.empty:
        best_row = sum_df.sort_values(by="PF", ascending=False).iloc[0]
        print(f"[★] 推奨最適しきい値: {best_row['Threshold']} (PF: {best_row['PF']}, 勝率: {best_row['WinRate(%)']}%)")

if __name__ == "__main__":
    main()