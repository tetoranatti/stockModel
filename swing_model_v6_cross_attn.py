import os
import argparse
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

# =============================================================================
# 1. モデル定義: Cross-Attention 付き DualStream GRU (v6)
# =============================================================================
class MultiHeadCrossAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super(MultiHeadCrossAttention, self).__init__()
        self.mha = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key_value):
        # query: (B, T_stock, H), key_value: (B, T_macro, H)
        attn_out, _ = self.mha(query=query, key=key_value, value=key_value)
        # 残差接続 & LayerNorm
        out = self.norm(query + self.dropout(attn_out))
        return out

class DualStreamGRU_v6_CrossAttention(nn.Module):
    def __init__(self, stock_dim=4, macro_dim=14, hidden_dim=48, num_heads=4, num_classes=3):
        super(DualStreamGRU_v6_CrossAttention, self).__init__()
        # 各ストリームのGRU
        self.stock_gru = nn.GRU(stock_dim, hidden_dim, batch_first=True, num_layers=2, dropout=0.2)
        self.macro_gru = nn.GRU(macro_dim, hidden_dim, batch_first=True, num_layers=2, dropout=0.2)
        
        # Cross-Attention: 個別株(Query) が マクロ(Key/Value) を参照
        self.cross_attn = MultiHeadCrossAttention(hidden_dim=hidden_dim, num_heads=num_heads)
        
        # 分類ヘッド
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
        h_stock, _ = self.stock_gru(x_stock)  # (B, T, H)
        h_macro, _ = self.macro_gru(x_macro)  # (B, T, H)

        # マクロ情報を注入した個別株表現
        attended_stock = self.cross_attn(query=h_stock, key_value=h_macro)  # (B, T, H)

        # 時間軸の平均プーリング
        pooled = torch.mean(attended_stock, dim=1)  # (B, H)
        logits = self.classifier(pooled)
        return logits

# =============================================================================
# 2. 損失関数 & データセット
# =============================================================================
class AsymmetricPenaltyLoss(nn.Module):
    def __init__(self, false_buy_penalty: float = 3.0):
        super(AsymmetricPenaltyLoss, self).__init__()
        self.penalty = false_buy_penalty
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, targets):
        base_loss = self.ce(logits, targets)
        probs = torch.softmax(logits, dim=-1)
        is_actual_loss = (targets == 0)
        p_profit = probs[:, 2]
        multiplier = torch.ones_like(base_loss)
        multiplier[is_actual_loss] += (self.penalty - 1.0) * p_profit[is_actual_loss]
        return (base_loss * multiplier).mean()

class SingleStockDataset(Dataset):
    def __init__(self, stock_arr, macro_arr, targets, seq_len=10):
        self.samples = []
        for i in range(seq_len, len(targets)):
            s_seq = stock_arr[i - seq_len : i]
            m_seq = macro_arr[i - seq_len : i]
            s_seq = (s_seq - s_seq.mean(axis=0)) / (s_seq.std(axis=0) + 1e-7)
            m_seq = (m_seq - m_seq.mean(axis=0)) / (m_seq.std(axis=0) + 1e-7)
            self.samples.append((
                torch.tensor(s_seq, dtype=torch.float32),
                torch.tensor(m_seq, dtype=torch.float32),
                torch.tensor(int(targets[i]), dtype=torch.long)
            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# =============================================================================
# 3. 特徴量抽出
# =============================================================================
def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df['High'] - df['Low']
    high_cp = (df['High'] - df['Close'].shift(1)).abs()
    low_cp = (df['Low'] - df['Close'].shift(1)).abs()
    tr = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def load_and_preprocess(ticker: str, start_date: str = "2024-01-01"):
    df = yf.download(ticker, start=start_date, interval="1d", progress=False)
    if df.empty or len(df) < 50:
        return pd.DataFrame(), None, None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df['ATR'] = calculate_atr(df, 14)

    n225 = yf.download("^N225", start=start_date, interval="1d", progress=False)
    if isinstance(n225.columns, pd.MultiIndex): n225.columns = n225.columns.get_level_values(0)
    fx = yf.download("USDJPY=X", start=start_date, interval="1d", progress=False)
    if isinstance(fx.columns, pd.MultiIndex): fx.columns = fx.columns.get_level_values(0)
    vix = yf.download("^VIX", start=start_date, interval="1d", progress=False)
    if isinstance(vix.columns, pd.MultiIndex): vix.columns = vix.columns.get_level_values(0)

    df['NK_Close'] = n225['Close'].reindex(df.index).ffill()
    df['NK_Ret'] = n225['Close'].pct_change().reindex(df.index).fillna(0.0)
    df['FX_Ret'] = fx['Close'].pct_change().reindex(df.index).fillna(0.0)
    df['VIX_Close'] = vix['Close'].reindex(df.index).ffill().bfill()

    df['stock_ret_1d'] = df['Close'].pct_change(1).fillna(0.0)
    df['stock_ret_5d'] = df['Close'].pct_change(5).fillna(0.0)
    df['atr_ratio'] = (df['ATR'] / (df['Close'] + 1e-7)).fillna(0.0)
    cov = df['stock_ret_1d'].rolling(20).cov(df['NK_Ret'])
    var = df['NK_Ret'].rolling(20).var()
    df['rolling_beta'] = (cov / (var + 1e-7)).fillna(1.0)

    if os.path.exists("jpx_daily_features_db.csv"):
        jpx_db = pd.read_csv("jpx_daily_features_db.csv", index_col=0, parse_dates=True)
        df = df.join(jpx_db, how='left')

    defaults = {
        'call_oi_wall': 66500.0, 'put_oi_wall': 65000.0, 'abn_net_flow': 0.0,
        'arbitrage_balance': 1000000.0, 'gamma_flip': 65500.0, 'call_volume': 1000.0,
        'put_volume': 1000.0, 'pcr_ratio': 1.0, 'us10y_yield': 4.0
    }
    for col, val in defaults.items():
        df[col] = df[col].fillna(val) if col in df.columns else val

    df['gamma_log_dist'] = np.log((df['NK_Close'] + 1e-7) / (df['gamma_flip'] + 1e-7)).fillna(0.0)
    df['delta_call_oi'] = df['call_oi_wall'].diff().fillna(0.0)

    stock_cols = ['stock_ret_1d', 'stock_ret_5d', 'atr_ratio', 'rolling_beta']
    macro_cols = [
        'NK_Ret', 'FX_Ret', 'VIX_Close',
        'call_oi_wall', 'put_oi_wall', 'abn_net_flow', 'arbitrage_balance',
        'gamma_flip', 'call_volume', 'put_volume', 'pcr_ratio', 'us10y_yield',
        'gamma_log_dist', 'delta_call_oi'
    ]

    return df.dropna(subset=['ATR', 'rolling_beta']).copy(), stock_cols, macro_cols

def generate_triple_barrier_labels(df: pd.DataFrame, holding_period: int = 10):
    labels = []
    closes = df['Close'].values
    highs = df['High'].values
    lows = df['Low'].values
    atrs = df['ATR'].values
    n = len(df)

    for i in range(n):
        if i + holding_period >= n:
            labels.append(np.nan)
            continue
        entry_p = closes[i]
        upper_p = entry_p + (2.0 * atrs[i])
        lower_p = entry_p - (1.0 * atrs[i])
        outcome = 1
        for h in range(1, holding_period + 1):
            if lows[i + h] <= lower_p and highs[i + h] >= upper_p:
                outcome = 0
                break
            elif highs[i + h] >= upper_p:
                outcome = 2
                break
            elif lows[i + h] <= lower_p:
                outcome = 0
                break
        labels.append(outcome)

    df['Target'] = labels
    return df.dropna(subset=['Target']).copy()

# =============================================================================
# 4. 学習 & 推論ルーチン
# =============================================================================
def train_and_predict_v6(ticker: str, epochs: int = 15, patience: int = 3):
    ticker_clean = ticker.split('.')[0]
    model_save_path = f"swing_model_v6_crossattn_{ticker_clean}.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*70}")
    print(f"[*] Cross-Attention 搭載モデル (v6): {ticker}")
    print(f"[*] 使用デバイス: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"{'='*70}")

    df, stock_cols, macro_cols = load_and_preprocess(ticker)
    if len(df) < 50:
        return

    labeled_df = generate_triple_barrier_labels(df.copy())
    stock_data = labeled_df[stock_cols].values
    macro_data = labeled_df[macro_cols].values
    targets = labeled_df['Target'].values

    train_size = int(len(labeled_df) * 0.7)
    train_ds = SingleStockDataset(stock_data[:train_size], macro_data[:train_size], targets[:train_size])
    val_ds = SingleStockDataset(stock_data[train_size:], macro_data[train_size:], targets[train_size:])

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

    model = DualStreamGRU_v6_CrossAttention(
        stock_dim=len(stock_cols), 
        macro_dim=len(macro_cols), 
        hidden_dim=48, 
        num_heads=4, 
        num_classes=3
    ).to(device)

    criterion = AsymmetricPenaltyLoss(false_buy_penalty=3.0)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    best_val_loss = float('inf')
    best_weights = None
    patience_counter = 0

    print("[*] 学習開始（Early Stopping: Patience=3）...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for x_s, x_m, y in train_loader:
            x_s, x_m, y = x_s.to(device), x_m.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x_s, x_m)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_s, x_m, y in val_loader:
                x_s, x_m, y = x_s.to(device), x_m.to(device), y.to(device)
                out = model(x_s, x_m)
                loss = criterion(out, y)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        print(f"  Epoch [{epoch:02d}/{epochs}] - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}", end="")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
            print("  --> [Best Val Loss 更新 ★]")
        else:
            patience_counter += 1
            print(f"  (Patience: {patience_counter}/{patience})")
            if patience_counter >= patience:
                print(f"[*] Early Stopping 発動: Epoch {epoch} で学習を打ち切ります。")
                break

    if best_weights is not None:
        model.load_state_dict(best_weights)

    # 保存
    model_cpu = model.to("cpu")
    torch.save({
        "model_state_dict": model_cpu.state_dict(),
        "stock_cols": stock_cols,
        "macro_cols": macro_cols,
        "ticker": ticker
    }, model_save_path)
    print(f"[+] 最適モデル重みを保存しました: {model_save_path}")

    # 全期間推論で動的閾値を導出
    model_cpu.eval()
    seq_len = 10
    n = len(df)
    hist_p_profit = []

    for i in range(seq_len, n):
        seq_s = df[stock_cols].iloc[i - seq_len : i].values
        seq_m = df[macro_cols].iloc[i - seq_len : i].values
        seq_s = (seq_s - seq_s.mean(axis=0)) / (seq_s.std(axis=0) + 1e-7)
        seq_m = (seq_m - seq_m.mean(axis=0)) / (seq_m.std(axis=0) + 1e-7)

        t_s = torch.tensor(seq_s, dtype=torch.float32).unsqueeze(0)
        t_m = torch.tensor(seq_m, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            logits = model_cpu(t_s, t_m)
            p = torch.softmax(logits, dim=-1).numpy()[0]
            hist_p_profit.append(p[2])

    hist_p_profit = np.array(hist_p_profit)
    adaptive_threshold = round(float(np.percentile(hist_p_profit, 85.0)), 3)
    applied_threshold = min(0.40, max(0.33, adaptive_threshold))

    # 直近推論
    seq_s_latest = df[stock_cols].iloc[-seq_len:].values
    seq_m_latest = df[macro_cols].iloc[-seq_len:].values
    seq_s_latest = (seq_s_latest - seq_s_latest.mean(axis=0)) / (seq_s_latest.std(axis=0) + 1e-7)
    seq_m_latest = (seq_m_latest - seq_m_latest.mean(axis=0)) / (seq_m_latest.std(axis=0) + 1e-7)
    with torch.no_grad():
        out_latest = model_cpu(torch.tensor(seq_s_latest, dtype=torch.float32).unsqueeze(0),
                               torch.tensor(seq_m_latest, dtype=torch.float32).unsqueeze(0))
        latest_probs = torch.softmax(out_latest, dim=-1).numpy()[0]

    p_loss, p_wait, p_profit = latest_probs[0], latest_probs[1], latest_probs[2]
    is_buy = (p_profit >= applied_threshold) and (p_profit >= p_loss * 0.8)

    latest_close = float(df['Close'].iloc[-1])
    atr_val = float(df['ATR'].iloc[-1])
    target_profit = round(latest_close + (2.0 * atr_val), 1)
    stop_loss = round(latest_close - (1.0 * atr_val), 1)

    print(f"\n{'-'*70}")
    print(f"📊 【{ticker}】 Cross-Attention (v6) 最新推論結果 ({datetime.now().strftime('%Y-%m-%d')})")
    print(f"{'-'*70}")
    print(f"現在値 (終値)       : {latest_close:,.1f} 円 (ATR: {atr_val:,.1f})")
    print(f"シグナル判定         : {'【 🎯 BUY点灯 (買い推奨) 】' if is_buy else '【 ⏸️ WAIT (待機・静観) 】'}")
    print(f"利確確率 (+2.0ATR)   : {p_profit * 100:.1f} %")
    print(f"損切確率 (-1.0ATR)   : {p_loss * 100:.1f} %")
    print(f"保合確率 (WAIT)      : {p_wait * 100:.1f} %")
    print(f"適用エントリー閾値   : {applied_threshold * 100:.1f} % (過去上位15%適応: {adaptive_threshold*100:.1f}%)")
    print(f"利確ターゲット(+2ATR): {target_profit:,.1f} 円")
    print(f"損切ライン (-1ATR)   : {stop_loss:,.1f} 円")
    print(f"{'-'*70}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-Attention (v6) 学習＆推論")
    parser.add_argument("--ticker", type=str, default="7012.T")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=3)
    args = parser.parse_args()

    train_and_predict_v6(ticker=args.ticker, epochs=args.epochs, patience=args.patience)