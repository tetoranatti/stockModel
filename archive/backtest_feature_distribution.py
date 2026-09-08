import os
import time
import pandas as pd
import numpy as np
import yfinance as yf
import torch
import torch.nn as nn

# =============================================================================
# 1. モデル定義
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
        return torch.sum(weights * rnn_outputs, dim=1), weights

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
        return self.classifier(torch.cat([context_stock, context_macro], dim=-1))

# =============================================================================
# 2. テクニカル指標計算
# =============================================================================
def calc_indicators(df):
    close = df['Close']
    high = df['High']
    low = df['Low']

    # ATR(14)
    tr = pd.concat([(high - low), (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()
    df['ATR_Pct'] = (df['ATR'] / close) * 100

    # 移動平均 & 乖離率
    df['SMA25'] = close.rolling(25).mean()
    df['SMA200'] = close.rolling(200).mean()
    df['SMA25_Dev'] = ((close - df['SMA25']) / df['SMA25']) * 100
    df['SMA200_Dev'] = ((close - df['SMA200']) / df['SMA200']) * 100

    # RSI(14)
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / (loss + 1e-7))))

    # MACD(12, 26, 9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    df['MACD_Hist_Slope'] = df['MACD_Hist'].diff(2)  # 2日間のヒストグラム傾き

    # ボリンジャーバンド %B
    rolling_mean = close.rolling(20).mean()
    rolling_std = close.rolling(20).std()
    upper = rolling_mean + 2 * rolling_std
    lower = rolling_mean - 2 * rolling_std
    df['BB_pct_b'] = (close - lower) / (upper - lower + 1e-7)

    return df

# =============================================================================
# 3. 過去バックテスト & 正例抽出
# =============================================================================
def main():
    model_path = "swing_model_v5_attn_7012.pt"
    if not os.path.exists(model_path):
        print(f"[!] モデルファイルが見つかりません: {model_path}")
        return

    checkpoint = torch.load(model_path, map_location="cpu")
    model = DualStreamGRU_v5_Attention(
        stock_dim=len(checkpoint['stock_cols']),
        macro_dim=len(checkpoint['macro_cols']),
        hidden_dim=44,
        num_classes=3
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # マクロデータ取得
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

    # 検証対象ユニバース（SBIのスクリーニングCSVから上位銘柄を抽出）
    target_tickers = []
    if os.path.exists("screener_result.csv"):
        for enc in ['cp932', 'shift_jis', 'utf-8']:
            try:
                df_sbi = pd.read_csv("screener_result.csv", encoding=enc)
                col = [c for c in df_sbi.columns if 'コード' in str(c)][0]
                target_tickers = [f"{str(v).strip()[:4]}.T" for v in df_sbi[col] if str(v).strip()[:4].isdigit()]
                break
            except Exception:
                continue
    
    if not target_tickers:
        # デフォルトの主力50銘柄
        target_tickers = ["7203.T", "6758.T", "8035.T", "9984.T", "6857.T", "7011.T", "8306.T", "8058.T", "4063.T", "6501.T"]

    # 過去バックテスト実行用の母集団（先頭60銘柄で十分なサンプル数を確保）
    sample_pool = target_tickers[:60]
    print(f"[*] {len(sample_pool)} 銘柄の過去2年間から「AI BUY ➔ 利確成功」サンプルを探索中...")

    success_records = []
    seq_len = 10
    holding_days = 7  # 7営業日スイング
    target_profit = 0.05  # +5%利確
    stop_loss = -0.03    # -3%損切

    for t in sample_pool:
        df = yf.download(t, period="2y", interval="1d", progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df = df.dropna(subset=['Close', 'High', 'Low'])
        if len(df) < 220:
            continue

        df = calc_indicators(df)
        df['stock_ret_1d'] = df['Close'].pct_change(1, fill_method=None).fillna(0.0)
        df['stock_ret_5d'] = df['Close'].pct_change(5, fill_method=None).fillna(0.0)
        df['atr_ratio'] = (df['ATR'] / (df['Close'] + 1e-7)).fillna(0.0)

        aligned_nk = macro_df['NK_Ret'].reindex(df.index).fillna(0.0)
        cov = df['stock_ret_1d'].rolling(20).cov(aligned_nk)
        var = aligned_nk.rolling(20).var()
        df['rolling_beta'] = (cov / (var + 1e-7)).fillna(1.0)

        df = df.join(macro_df[checkpoint['macro_cols']], how='inner')
        if len(df) < 60:
            continue

        # 過去各営業日ごとにローリング推論
        for i in range(seq_len, len(df) - holding_days):
            sub_s = df[checkpoint['stock_cols']].iloc[i - seq_len : i].values
            sub_m = df[checkpoint['macro_cols']].iloc[i - seq_len : i].values
            sub_s = (sub_s - sub_s.mean(axis=0)) / (sub_s.std(axis=0) + 1e-7)
            sub_m = (sub_m - sub_m.mean(axis=0)) / (sub_m.std(axis=0) + 1e-7)

            with torch.no_grad():
                out = model(torch.tensor(sub_s, dtype=torch.float32).unsqueeze(0),
                            torch.tensor(sub_m, dtype=torch.float32).unsqueeze(0))
                probs = torch.softmax(out, dim=-1).numpy()[0]

            is_buy = (probs[2] >= 0.35) and (probs[2] >= probs[0] * 0.9)
            if not is_buy:
                continue

            # 保有期間内の成否判定
            entry_price = df['Close'].iloc[i]
            future_high = df['High'].iloc[i + 1 : i + 1 + holding_days].max()
            future_low = df['Low'].iloc[i + 1 : i + 1 + holding_days].min()

            max_gain = (future_high - entry_price) / entry_price
            max_drop = (future_low - entry_price) / entry_price

            # +5%達成かつ損切(-3%)に先に引っかかっていないものを「利確成功」と定義
            if max_gain >= target_profit and max_drop > stop_loss:
                row = df.iloc[i]
                success_records.append({
                    'Ticker': t,
                    'Date': df.index[i].strftime('%Y-%m-%d'),
                    'RSI': row['RSI'],
                    'SMA25_Dev': row['SMA25_Dev'],
                    'SMA200_Dev': row['SMA200_Dev'],
                    'BB_pct_b': row['BB_pct_b'],
                    'MACD_Hist': row['MACD_Hist'],
                    'MACD_Slope': row['MACD_Hist_Slope'],
                    'ATR_Pct': row['ATR_Pct']
                })

    print(f"\n[+] 収集完了: {len(success_records)} 件の利確成功サンプルを抽出しました。")
    if not success_records:
        print("[-] 条件を満たす成功サンプルが不足しています。")
        return

    df_res = pd.DataFrame(success_records)
    
    # 統計集計（中央値・第1四分位数・第3四分位数）
    stats = []
    features = [
        ('RSI (14日)', 'RSI'),
        ('25日線乖離率 (%)', 'SMA25_Dev'),
        ('200日線乖離率 (%)', 'SMA200_Dev'),
        ('ボリンジャー %B (20日)', 'BB_pct_b'),
        ('MACDヒストグラム2日傾き', 'MACD_Slope'),
        ('ボラティリティ (ATR %)', 'ATR_Pct')
    ]

    for label, col in features:
        q25 = df_res[col].quantile(0.25)
        med = df_res[col].median()
        q75 = df_res[col].quantile(0.75)
        stats.append({
            '指標名': label,
            '25%点 (下限目安)': round(q25, 2),
            '中央値 (代表値)': round(med, 2),
            '75%点 (上限目安)': round(q75, 2),
            '推奨スクリーニング範囲': f"{q25:.1f} 〜 {q75:.1f}"
        })

    summary_df = pd.DataFrame(stats)
    print("\n" + "=" * 80)
    print("📈 AI利確成功サンプルから逆算した「最適なテクニカル条件」")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print("=" * 80)

    df_res.to_csv("ai_success_features_distribution.csv", index=False)
    print("\n[+] 全サンプル詳細を 'ai_success_features_distribution.csv' に保存しました。")

if __name__ == "__main__":
    main()