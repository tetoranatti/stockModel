import os
import time
import argparse
import pandas as pd
import numpy as np
import yfinance as yf
import torch
import torch.nn as nn
from datetime import datetime

# =============================================================================
# 1. AIモデル定義 (DualStream GRU + Attention)
# =============================================================================
class TemporalSelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(TemporalSelfAttention, self).__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, rnn_outputs):
        scores = self.attn(rnn_outputs)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(weights * rnn_outputs, dim=1)
        return context, weights

class DualStreamGRU_v5_Attention(nn.Module):
    def __init__(self, stock_dim=4, macro_dim=14, hidden_dim=44, num_classes=3):
        super(DualStreamGRU_v5_Attention, self).__init__()
        self.stock_gru = nn.GRU(stock_dim, hidden_dim, batch_first=True, num_layers=2, dropout=0.2)
        self.macro_gru = nn.GRU(macro_dim, hidden_dim, batch_first=True, num_layers=2, dropout=0.2)
        self.stock_attn = TemporalSelfAttention(hidden_dim)
        self.macro_attn = TemporalSelfAttention(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )

    def forward(self, x_stock, x_macro):
        out_stock, _ = self.stock_gru(x_stock)
        out_macro, _ = self.macro_gru(x_macro)
        context_stock, _ = self.stock_attn(out_stock)
        context_macro, _ = self.macro_attn(out_macro)
        combined = torch.cat([context_stock, context_macro], dim=-1)
        logits = self.classifier(combined)
        return logits

# =============================================================================
# 2. SBI証券 CSVパース
# =============================================================================
def load_sbi_csv(csv_path: str) -> list:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"SBI証券のCSVファイルが見つかりません: {csv_path}")

    df = None
    for enc in ['cp932', 'shift_jis', 'utf-8']:
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            break
        except Exception:
            continue

    if df is None:
        raise ValueError("CSVの読み込みに失敗しました（文字コードをご確認ください）。")

    code_col = None
    for col in df.columns:
        if any(keyword in str(col) for keyword in ['コード', 'code', 'Code']):
            code_col = col
            break

    if not code_col:
        code_col = df.columns[0]

    tickers = []
    for val in df[code_col]:
        c = str(val).strip()
        if len(c) >= 4 and c[:4].isdigit():
            tickers.append(f"{c[:4]}.T")

    tickers = list(dict.fromkeys(tickers))
    print(f"[+] SBI CSVロード完了: {len(tickers)} 銘柄を対象母数として読み込みました。")
    return tickers

# =============================================================================
# 3. テクニカル判定（調整版: 押し目反発初動捕捉）
# =============================================================================
def evaluate_technical(df: pd.DataFrame):
    if len(df) < 220:
        return False, None

    close = df['Close']
    high = df['High']
    low = df['Low']

    # ATR(14) & 比率
    hl = high - low
    h_cp = (high - close.shift(1)).abs()
    l_cp = (low - close.shift(1)).abs()
    atr = pd.concat([hl, h_cp, l_cp], axis=1).max(axis=1).rolling(14).mean()
    atr_pct = (atr.iloc[-1] / close.iloc[-1]) * 100

    # 移動平均 & 乖離率
    sma_25 = close.rolling(25).mean().iloc[-1]
    sma_200 = close.rolling(200).mean().iloc[-1]
    dev_25 = ((close.iloc[-1] - sma_25) / sma_25) * 100
    dev_200 = ((close.iloc[-1] - sma_200) / sma_200) * 100

    # RSI(14)
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rsi_val = (100 - (100 / (1 + (gain / (loss + 1e-7))))).iloc[-1]

    # ボリンジャーバンド %B (20日)
    rolling_mean = close.rolling(20).mean().iloc[-1]
    rolling_std = close.rolling(20).std().iloc[-1]
    upper = rolling_mean + 2 * rolling_std
    lower = rolling_mean - 2 * rolling_std
    pct_b = (close.iloc[-1] - lower) / (upper - lower + 1e-7)

    # -------------------------------------------------------------
    # AIバックテストから導出された実戦スクリーニング条件
    # -------------------------------------------------------------
    # 1. 200日線乖離率: 強気トレンド維持 (10%以上)
    cond_trend = dev_200 >= 10.0

    # 2. 25日線乖離率: 押し目〜過熱前 (-1.5% 〜 +6.5%)
    cond_ma25 = (-1.5 <= dev_25 <= 6.5)

    # 3. RSI: 45〜66 (モメンタム維持ゾーン)
    cond_rsi = (45.0 <= rsi_val <= 66.0)

    # 4. ボリンジャー %B: センターライン近辺以上 (0.40 〜 0.90)
    cond_bb = (0.40 <= pct_b <= 0.90)

    # 5. ATR%: 適切なボラティリティ (1.8% 〜 3.5%)
    cond_vol = (1.8 <= atr_pct <= 3.5)

    if cond_trend and cond_ma25 and cond_rsi and cond_bb and cond_vol:
        return True, {
            'Close': float(close.iloc[-1]),
            'ATR': float(atr.iloc[-1]),
            'RSI': float(rsi_val),
            'SMA25_Dev': round(dev_25, 2),
            'SMA200_Dev': round(dev_200, 2)
        }

    return False, None

# =============================================================================
# 4. AI推論実行
# =============================================================================
def run_ai_validation(candidates, model_path="swing_model_v5_attn_7012.pt"):
    if not os.path.exists(model_path):
        print("[-] AIモデルファイルが見つかりません。テクニカル通過銘柄のみ出力します。")
        return pd.DataFrame(candidates)

    checkpoint = torch.load(model_path, map_location="cpu")
    model = DualStreamGRU_v5_Attention(
        stock_dim=len(checkpoint['stock_cols']),
        macro_dim=len(checkpoint['macro_cols']),
        hidden_dim=44,
        num_classes=3
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    start_date = "2024-01-01"
    n225 = yf.download("^N225", start=start_date, interval="1d", progress=False)
    if isinstance(n225.columns, pd.MultiIndex): n225.columns = n225.columns.get_level_values(0)
    fx = yf.download("USDJPY=X", start=start_date, interval="1d", progress=False)
    if isinstance(fx.columns, pd.MultiIndex): fx.columns = fx.columns.get_level_values(0)
    vix = yf.download("^VIX", start=start_date, interval="1d", progress=False)
    if isinstance(vix.columns, pd.MultiIndex): vix.columns = vix.columns.get_level_values(0)

    macro_df = pd.DataFrame(index=n225.index)
    macro_df['NK_Close'] = n225['Close']
    macro_df['NK_Ret'] = n225['Close'].pct_change().fillna(0.0)
    macro_df['FX_Ret'] = fx['Close'].pct_change().reindex(macro_df.index).fillna(0.0)
    macro_df['VIX_Close'] = vix['Close'].reindex(macro_df.index).ffill().bfill()

    if os.path.exists("jpx_daily_features_db.csv"):
        jpx_db = pd.read_csv("jpx_daily_features_db.csv", index_col=0, parse_dates=True)
        macro_df = macro_df.join(jpx_db, how='left')

    defaults = {
        'call_oi_wall': 66500.0, 'put_oi_wall': 65000.0, 'abn_net_flow': 0.0,
        'arbitrage_balance': 1000000.0, 'gamma_flip': 65500.0, 'call_volume': 1000.0,
        'put_volume': 1000.0, 'pcr_ratio': 1.0, 'us10y_yield': 4.0
    }
    for col, val in defaults.items():
        macro_df[col] = macro_df[col].fillna(val) if col in macro_df.columns else val

    macro_df['gamma_log_dist'] = np.log((macro_df['NK_Close'] + 1e-7) / (macro_df['gamma_flip'] + 1e-7)).fillna(0.0)
    macro_df['delta_call_oi'] = macro_df['call_oi_wall'].diff().fillna(0.0)

    final_results = []
    seq_len = 10

    for item in candidates:
        t = item['Ticker']
        df = yf.download(t, period="2y", interval="1d", progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)

        hl = df['High'] - df['Low']
        h_cp = (df['High'] - df['Close'].shift(1)).abs()
        l_cp = (df['Low'] - df['Close'].shift(1)).abs()
        df['ATR'] = pd.concat([hl, h_cp, l_cp], axis=1).max(axis=1).rolling(14).mean()
        df['stock_ret_1d'] = df['Close'].pct_change(1).fillna(0.0)
        df['stock_ret_5d'] = df['Close'].pct_change(5).fillna(0.0)
        df['atr_ratio'] = (df['ATR'] / (df['Close'] + 1e-7)).fillna(0.0)

        aligned_nk = macro_df['NK_Ret'].reindex(df.index).fillna(0.0)
        cov = df['stock_ret_1d'].rolling(20).cov(aligned_nk)
        var = aligned_nk.rolling(20).var()
        df['rolling_beta'] = (cov / (var + 1e-7)).fillna(1.0)

        df = df.join(macro_df[checkpoint['macro_cols']], how='inner')
        if len(df) < seq_len:
            continue

        seq_s = df[checkpoint['stock_cols']].iloc[-seq_len:].values
        seq_m = df[checkpoint['macro_cols']].iloc[-seq_len:].values
        seq_s = (seq_s - seq_s.mean(axis=0)) / (seq_s.std(axis=0) + 1e-7)
        seq_m = (seq_m - seq_m.mean(axis=0)) / (seq_m.std(axis=0) + 1e-7)

        with torch.no_grad():
            out = model(torch.tensor(seq_s, dtype=torch.float32).unsqueeze(0),
                        torch.tensor(seq_m, dtype=torch.float32).unsqueeze(0))
            probs = torch.softmax(out, dim=-1).numpy()[0]

        is_buy = (probs[2] >= 0.35) and (probs[2] >= probs[0] * 0.9)
        final_results.append({
            'Ticker': t,
            'Close': item['Close'],
            'ATR': round(item['ATR'], 1),
            'RSI': round(item['RSI'], 1),
            '利確確率': f"{probs[2]*100:.1f}%",
            '損切確率': f"{probs[0]*100:.1f}%",
            'AI判定': "🎯 BUY" if is_buy else "⏸️ WAIT"
        })

    return pd.DataFrame(final_results)

# =============================================================================
# 5. メインルーチン
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="screener_result.csv")
    args = parser.parse_args()

    print("=" * 80)
    print(f"[*] SBI証券CSV連動 スクリーニング＆AI推論パイプライン ({datetime.now().strftime('%Y-%m-%d')})")
    print("=" * 80)

    try:
        tickers = load_sbi_csv(args.csv)
    except Exception as e:
        print(f"[!] エラー: {e}")
        return

    CHUNK_SIZE = 30
    SLEEP_SEC = 2.0
    candidates = []

    print(f"[*] 抽出された {len(tickers)} 銘柄からテクニカル初動（MACD底打ち反転 / RSI 28-60）を精査中...")

    total_chunks = (len(tickers) + CHUNK_SIZE - 1) // CHUNK_SIZE
    for idx in range(0, len(tickers), CHUNK_SIZE):
        chunk = tickers[idx : idx + CHUNK_SIZE]
        print(f"  -> Batch [{(idx//CHUNK_SIZE)+1:02d}/{total_chunks:02d}] 取得中...", end="\r")

        try:
            data = yf.download(chunk, period="2y", interval="1d", group_by='ticker', progress=False, threads=False)
            if data.empty:
                continue

            for t in chunk:
                try:
                    # MultiIndex の正しいアンパック処理
                    if len(chunk) == 1:
                        df_single = data.copy()
                        if isinstance(df_single.columns, pd.MultiIndex):
                            df_single.columns = df_single.columns.get_level_values(0)
                    else:
                        if t not in data.columns.levels[0]:
                            continue
                        df_single = data[t].copy()

                    # 欠損値除去
                    df_single = df_single.dropna(subset=['Close', 'High', 'Low'])
                    
                    passed, meta = evaluate_technical(df_single)
                    if passed:
                        meta['Ticker'] = t
                        candidates.append(meta)
                        print(f"\n  [★ Hit] {t:7s} | 終値: {meta['Close']:>7.1f} | RSI: {meta['RSI']:>4.1f}")
                except Exception as err:
                    # エラーの握りつぶしを防ぐデバッグ用（必要に応じて解除可能）
                    # print(f"Error {t}: {err}")
                    continue
        except Exception:
            pass

        time.sleep(SLEEP_SEC)

    print(f"\n{'-'*80}")
    print(f"[+] 第1段階（テクニカル合致）: {len(candidates)} 銘柄")
    print(f"{'-'*80}")

    if candidates:
        print("[*] 第2段階: JPX実需給連動モデルによる最終推論を実行中...")
        res_df = run_ai_validation(candidates)
        print("\n" + "=" * 80)
        print("📊 本日の最終判定結果")
        print("=" * 80)
        print(res_df.to_string(index=False))
        print("=" * 80)
    else:
        print("[-] 条件に合致する銘柄はありませんでした。")

if __name__ == "__main__":
    main()