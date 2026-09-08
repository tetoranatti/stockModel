import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

# =============================================================================
# 1. モデル定義 (v4 DualStreamGRU)
# =============================================================================
class DualStreamGRU_v4(nn.Module):
    def __init__(self, stock_dim=4, macro_dim=14, hidden_dim=44, num_classes=3):
        super(DualStreamGRU_v4, self).__init__()
        self.stock_gru = nn.GRU(stock_dim, hidden_dim, batch_first=True, num_layers=2, dropout=0.2)
        self.macro_gru = nn.GRU(macro_dim, hidden_dim, batch_first=True, num_layers=2, dropout=0.2)
        
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
        _, h_stock = self.stock_gru(x_stock)
        _, h_macro = self.macro_gru(x_macro)
        combined = torch.cat([h_stock[-1], h_macro[-1]], dim=-1)
        logits = self.classifier(combined)
        return logits

# =============================================================================
# 2. 非対称ペナルティ損失関数 (誤爆防止)
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

# =============================================================================
# 3. 特徴量エンジニアリング (テクニカル + JPX需給)
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

    # マクロ指標
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

    # 銘柄側4次元
    df['stock_ret_1d'] = df['Close'].pct_change(1).fillna(0.0)
    df['stock_ret_5d'] = df['Close'].pct_change(5).fillna(0.0)
    df['atr_ratio'] = (df['ATR'] / (df['Close'] + 1e-7)).fillna(0.0)
    cov = df['stock_ret_1d'].rolling(20).cov(df['NK_Ret'])
    var = df['NK_Ret'].rolling(20).var()
    df['rolling_beta'] = (cov / (var + 1e-7)).fillna(1.0)

    # JPX日次データベース
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

    # v4拡張特徴量
    df['gamma_log_dist'] = np.log((df['NK_Close'] + 1e-7) / (df['gamma_flip'] + 1e-7)).fillna(0.0)
    df['delta_call_oi'] = df['call_oi_wall'].diff().fillna(0.0)

    stock_cols = ['stock_ret_1d', 'stock_ret_5d', 'atr_ratio', 'rolling_beta']
    macro_cols = [
        'NK_Ret', 'FX_Ret', 'VIX_Close',
        'call_oi_wall', 'put_oi_wall', 'abn_net_flow', 'arbitrage_balance',
        'gamma_flip', 'call_volume', 'put_volume', 'pcr_ratio', 'us10y_yield',
        'gamma_log_dist', 'delta_call_oi'
    ]

    df = df.dropna(subset=['ATR', 'rolling_beta']).copy()
    return df, stock_cols, macro_cols

# =============================================================================
# 4. トリプルバリア・ラベリング & Dataset
# =============================================================================
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
# 5. メインルーチン（学習 & 本番推論）
# =============================================================================
def train_and_predict(ticker: str, force_train: bool = True, epochs: int = 8):
    ticker_clean = ticker.split('.')[0]
    model_save_path = f"swing_model_v4_{ticker_clean}.pt"

    print(f"\n{'='*70}")
    print(f"[*] 単一銘柄 独立学習・推論システム: {ticker}")
    print(f"[*] 保存先モデル: {model_save_path}")
    print(f"{'='*70}")

    df, stock_cols, macro_cols = load_and_preprocess(ticker)
    if len(df) < 50:
        print("[-] データ取得件数が不足しています。")
        return

    model = DualStreamGRU_v4(stock_dim=len(stock_cols), macro_dim=len(macro_cols), hidden_dim=44, num_classes=3)

    # 学習フェーズ
    if force_train or not os.path.exists(model_save_path):
        labeled_df = generate_triple_barrier_labels(df.copy())
        stock_data = labeled_df[stock_cols].values
        macro_data = labeled_df[macro_cols].values
        targets = labeled_df['Target'].values

        split_idx = int(len(labeled_df) * 0.8)
        train_ds = SingleStockDataset(stock_data[:split_idx], macro_data[:split_idx], targets[:split_idx])
        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)

        criterion = AsymmetricPenaltyLoss(false_buy_penalty=3.0)
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

        print("[*] モデルの個別最適化学習を開始...")
        model.train()
        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            for x_s, x_m, y in train_loader:
                optimizer.zero_grad()
                out = model(x_s, x_m)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if epoch % 2 == 0 or epoch == epochs:
                print(f"    Epoch [{epoch}/{epochs}] - Loss: {total_loss/len(train_loader):.4f}")

        # 重み保存
        torch.save({
            "model_state_dict": model.state_dict(),
            "stock_cols": stock_cols,
            "macro_cols": macro_cols,
            "ticker": ticker
        }, model_save_path)
        print(f"[+] モデルを正常に保存しました: {model_save_path}")
    else:
        print(f"[*] 既存の学習済み重みをロードします: {model_save_path}")
        checkpoint = torch.load(model_save_path, map_location=torch.device('cpu'), weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

    # 最新データによる本番推論フェーズ
    model.eval()
    seq_stock = df[stock_cols].iloc[-10:].values
    seq_macro = df[macro_cols].iloc[-10:].values

    s_mean, s_std = seq_stock.mean(axis=0), seq_stock.std(axis=0) + 1e-7
    m_mean, m_std = seq_macro.mean(axis=0), seq_macro.std(axis=0) + 1e-7
    seq_stock = (seq_stock - s_mean) / s_std
    seq_macro = (seq_macro - m_mean) / m_std

    t_stock = torch.tensor(seq_stock, dtype=torch.float32).unsqueeze(0)
    t_macro = torch.tensor(seq_macro, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        logits = model(t_stock, t_macro)
        probs = torch.softmax(logits, dim=-1).numpy()[0]

    p_loss, p_wait, p_profit = probs[0], probs[1], probs[2]

    # 銘柄別閾値判定 (旭化成 3407.T は 35%、その他は 40%)
    profit_threshold = 0.35 if "3407" in ticker else 0.40
    is_buy = (p_profit >= profit_threshold) and (p_profit >= p_loss * 0.8)

    latest_close = float(df['Close'].iloc[-1])
    atr_val = float(df['ATR'].iloc[-1])
    target_profit = round(latest_close + (2.0 * atr_val), 1)
    stop_loss = round(latest_close - (1.0 * atr_val), 1)

    print(f"\n{'-'*70}")
    print(f"📊 最新推論結果サマリー ({datetime.now().strftime('%Y-%m-%d')})")
    print(f"{'-'*70}")
    print(f"対象銘柄           : {ticker}")
    print(f"現在値 (直近終値)  : {latest_close:,.1f} 円 (ATR: {atr_val:,.1f})")
    print(f"シグナル判定       : {'【 🎯 BUY点灯 (買いエントリー推奨) 】' if is_buy else '【 ⏸️ WAIT (静観・待機) 】'}")
    print(f"利確確率 (+2.0ATR) : {p_profit * 100:.1f} %  (エントリー閾値: {profit_threshold*100:.0f}%)")
    print(f"損切確率 (-1.0ATR) : {p_loss * 100:.1f} %")
    print(f"保合確率 (WAIT)    : {p_wait * 100:.1f} %")
    print(f"利確目標価格 (+2ATR): {target_profit:,.1f} 円")
    print(f"防衛撤退ライン(-1ATR): {stop_loss:,.1f} 円")
    print(f"{'-'*70}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="指定銘柄の個別学習＆推論スクリプト")
    parser.add_argument("--ticker", type=str, default="7012.T", help="銘柄コード (例: 7012.T, 3407.T, 8227.T, 8593.T)")
    parser.add_argument("--no-train", action="store_true", help="再学習を行わずに既存モデルで推論のみ実行")
    parser.add_argument("--epochs", type=int, default=8, help="学習エポック数")
    args = parser.parse_args()

    train_and_predict(ticker=args.ticker, force_train=not args.no_train, epochs=args.epochs)