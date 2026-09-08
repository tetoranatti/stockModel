import os
import argparse
import time
import pandas as pd
import numpy as np
import yfinance as yf
import torch
import torch.nn as nn

# =============================================================================
# 1. モデル定義 (v6: Cross-Attention & DualStreamGRU)
# =============================================================================
class CrossAttentionBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, dropout=0.2):
        super(CrossAttentionBlock, self).__init__()
        self.mha = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key_value):
        # query: 個別株時系列, key_value: マクロ時系列
        attn_out, _ = self.mha(query, key_value, key_value)
        return self.norm(query + self.dropout(attn_out))

class DualStreamGRU_v6_CrossAttn(nn.Module):
    def __init__(self, stock_dim=4, macro_dim=14, hidden_dim=48, num_classes=3):
        super(DualStreamGRU_v6_CrossAttn, self).__init__()
        self.stock_gru = nn.GRU(stock_dim, hidden_dim, batch_first=True, num_layers=2, dropout=0.2)
        self.macro_gru = nn.GRU(macro_dim, hidden_dim, batch_first=True, num_layers=2, dropout=0.2)
        self.cross_attn = CrossAttentionBlock(hidden_dim, num_heads=4, dropout=0.2)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
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
        fused = self.cross_attn(out_stock, out_macro)
        pooled = torch.mean(fused, dim=1)
        return self.classifier(pooled)

# =============================================================================
# 2. テクニカル指標計算 & 第1段階フィルター (最適化統計閾値)
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

    # 統計最適化フィルター条件
    cond_trend = dev_200 >= 10.0
    cond_ma25 = (-1.5 <= dev_25 <= 6.5)
    cond_rsi = (45.0 <= rsi_val <= 66.0)
    cond_bb = (0.40 <= pct_b <= 0.90)
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
# 3. SBI スクリーニング CSV 読み込み
# =============================================================================
def load_sbi_csv(csv_path: str):
    if not os.path.exists(csv_path):
        print(f"[!] CSVファイルが見つかりません: {csv_path}")
        return []

    encodings = ['cp932', 'shift_jis', 'utf-8']
    df = None
    for enc in encodings:
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            break
        except Exception:
            continue

    if df is None:
        print("[!] CSVの読み込みに失敗しました。文字コードを確認してください。")
        return []

    code_col = None
    for c in df.columns:
        if 'コード' in str(c):
            code_col = c
            break

    if not code_col:
        print("[!] 銘柄コード列が見つかりません。")
        return []

    tickers = []
    for val in df[code_col].dropna():
        s = str(val).strip()
        if len(s) >= 4 and s[:4].isdigit():
            tickers.append(f"{s[:4]}.T")

    return sorted(list(set(tickers)))

# =============================================================================
# 4. 第2段階: v6 クロスアテンション AI 推論
# =============================================================================
def run_ai_validation(candidates: dict):
    if not candidates:
        return pd.DataFrame()

    model_path = "swing_model_v6_crossattn_universe.pt"
    if not os.path.exists(model_path):
        print(f"[!] v6モデルファイルが見つかりません: {model_path}")
        return pd.DataFrame()

    checkpoint = torch.load(model_path, map_location="cpu")
    stock_cols = checkpoint.get('stock_cols', ['stock_ret_1d', 'stock_ret_5d', 'atr_ratio', 'rolling_beta'])
    macro_cols = checkpoint.get('macro_cols', [
        'NK_Ret', 'FX_Ret', 'VIX_Close', 'call_oi_wall', 'put_oi_wall',
        'abn_net_flow', 'arbitrage_balance', 'gamma_flip', 'call_volume',
        'put_volume', 'pcr_ratio', 'us10y_yield', 'gamma_log_dist', 'delta_call_oi'
    ])

    model = DualStreamGRU_v6_CrossAttn(
        stock_dim=len(stock_cols),
        macro_dim=len(macro_cols),
        hidden_dim=48,
        num_classes=3
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print("[*] マクロ指標を取得中...")
    start_date = "2024-01-01"
    n225 = yf.download("^N225", start=start_date, interval="1d", progress=False)
    if isinstance(n225.columns, pd.MultiIndex): n225.columns = n225.columns.get_level_values(0)
    fx = yf.download("USDJPY=X", start=start_date, interval="1d", progress=False)
    if isinstance(fx.columns, pd.MultiIndex): fx.columns = fx.columns.get_level_values(0)
    vix = yf.download("^VIX", start=start_date, interval="1d", progress=False)
    if isinstance(vix.columns, pd.MultiIndex): vix.columns = vix.columns.get_level_values(0)

    macro_df = pd.DataFrame(index=n225.index)
    macro_df['NK_Close'] = n225['Close']
    macro_df['NK_Ret'] = n225['Close'].pct_change(fill_method=None).fillna(0.0)
    macro_df['FX_Ret'] = fx['Close'].pct_change(fill_method=None).reindex(macro_df.index).fillna(0.0)
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

    results = []
    seq_len = 10
    print(f"[*] 通過した {len(candidates)} 銘柄に対して v6 AI推論を実行中...")

    for ticker, info in candidates.items():
        try:
            df = yf.download(ticker, period="1y", interval="1d", progress=False)
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            df = df.dropna(subset=['Close', 'High', 'Low'])
            if len(df) < 30: continue

            # 特徴量生成
            df['stock_ret_1d'] = df['Close'].pct_change(1, fill_method=None).fillna(0.0)
            df['stock_ret_5d'] = df['Close'].pct_change(5, fill_method=None).fillna(0.0)
            
            hl = df['High'] - df['Low']
            h_cp = (df['High'] - df['Close'].shift(1)).abs()
            l_cp = (df['Low'] - df['Close'].shift(1)).abs()
            tr = pd.concat([hl, h_cp, l_cp], axis=1).max(axis=1)
            atr = tr.rolling(14).mean()
            df['atr_ratio'] = (atr / (df['Close'] + 1e-7)).fillna(0.0)

            aligned_nk = macro_df['NK_Ret'].reindex(df.index).fillna(0.0)
            cov = df['stock_ret_1d'].rolling(20).cov(aligned_nk)
            var = aligned_nk.rolling(20).var()
            df['rolling_beta'] = (cov / (var + 1e-7)).fillna(1.0)

            df = df.join(macro_df[macro_cols], how='inner')
            if len(df) < seq_len: continue

            sub_s = df[stock_cols].iloc[-seq_len:].values
            sub_m = df[macro_cols].iloc[-seq_len:].values
            sub_s = (sub_s - sub_s.mean(axis=0)) / (sub_s.std(axis=0) + 1e-7)
            sub_m = (sub_m - sub_m.mean(axis=0)) / (sub_m.std(axis=0) + 1e-7)

            with torch.no_grad():
                out = model(torch.tensor(sub_s, dtype=torch.float32).unsqueeze(0),
                            torch.tensor(sub_m, dtype=torch.float32).unsqueeze(0))
                probs = torch.softmax(out, dim=-1).numpy()[0]

            win_p = probs[2]
            loss_p = probs[0]
            is_buy = (win_p >= 0.35) and (win_p >= loss_p * 0.9)

            results.append({
                'Ticker': ticker,
                'Close': round(info['Close'], 2),
                'ATR': round(info['ATR'], 1),
                'RSI': round(info['RSI'], 1),
                '利確確率': f"{win_p * 100:.1f}%",
                '損切確率': f"{loss_p * 100:.1f}%",
                'AI判定': '🎯 BUY' if is_buy else '⏸️ WAIT'
            })
        except Exception:
            continue

    return pd.DataFrame(results)

# =============================================================================
# 5. メイン処理
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="SBI Screener Pipeline (v6 Cross-Attention)")
    parser.add_argument("--csv", type=str, default="screener_result.csv", help="Path to SBI CSV file")
    args = parser.parse_args()

    tickers = load_sbi_csv(args.csv)
    if not tickers:
        return

    print(f"[*] 母集団 {len(tickers)} 銘柄からテクニカルスクリーニング中...")
    candidates = {}
    
    # バッチダウンロード
    chunk_size = 50
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i + chunk_size]
        data = yf.download(chunk, period="2y", interval="1d", group_by='ticker', progress=False)
        for t in chunk:
            try:
                df_t = data[t].dropna(subset=['Close', 'High', 'Low']) if len(chunk) > 1 else data.dropna(subset=['Close', 'High', 'Low'])
                passed, info = evaluate_technical(df_t)
                if passed:
                    candidates[t] = info
            except Exception:
                continue

    print(f"[+] 第1段階通過: {len(candidates)} 銘柄")
    res_df = run_ai_validation(candidates)

    if not res_df.empty:
        print("\n" + "=" * 80)
        print("📊 [v6 Cross-Attention] 本日の最終判定結果")
        print("=" * 80)
        print(res_df.to_string(index=False))
        print("=" * 80)
        res_df.to_csv("sbi_screened_v6_results.csv", index=False)
        print("\n[+] 結果を 'sbi_screened_v6_results.csv' に保存しました。")

if __name__ == "__main__":
    main()