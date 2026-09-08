import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import yfinance as yf

# =============================================================================
# 1. 実験用アーキテクチャ (DualStreamGRU v4)
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
# 2. 非対称ペナルティ損失関数 (Asymmetric Cross Entropy Loss)
# =============================================================================
class AsymmetricPenaltyLoss(nn.Module):
    def __init__(self, false_buy_penalty: float = 3.0):
        """
        false_buy_penalty: 実際はLOSS(クラス0)なのにモデルがPROFIT(クラス2)と
                           過信した誤爆に対するペナルティ倍率
        """
        super(AsymmetricPenaltyLoss, self).__init__()
        self.penalty = false_buy_penalty
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, targets):
        base_loss = self.ce(logits, targets)
        probs = torch.softmax(logits, dim=-1)
        
        # 実際がクラス0(LOSS) かつ 予測でクラス2(PROFIT)を優位としたサンプルを抽出
        is_actual_loss = (targets == 0)
        p_profit = probs[:, 2]
        
        # 誤爆の度合いに応じてペナルティ係数を動的適用
        multiplier = torch.ones_like(base_loss)
        multiplier[is_actual_loss] += (self.penalty - 1.0) * p_profit[is_actual_loss]
        
        return (base_loss * multiplier).mean()

# =============================================================================
# 3. 特徴量生成および前処理（ΔOI・ガンマ対数距離の追加）
# =============================================================================
def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df['High'] - df['Low']
    high_cp = (df['High'] - df['Close'].shift(1)).abs()
    low_cp = (df['Low'] - df['Close'].shift(1)).abs()
    tr = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def load_and_preprocess_v4(ticker: str, start_date: str = "2024-01-01"):
    df = yf.download(ticker, start=start_date, interval="1d", progress=False)
    if df.empty:
        return pd.DataFrame(), None, None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df['ATR'] = calculate_atr(df, 14)
    
    # 指数・為替・VIX
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

    # 銘柄側4次元特徴量
    df['stock_ret_1d'] = df['Close'].pct_change(1).fillna(0.0)
    df['stock_ret_5d'] = df['Close'].pct_change(5).fillna(0.0)
    df['atr_ratio'] = (df['ATR'] / (df['Close'] + 1e-7)).fillna(0.0)
    cov = df['stock_ret_1d'].rolling(20).cov(df['NK_Ret'])
    var = df['NK_Ret'].rolling(20).var()
    df['rolling_beta'] = (cov / (var + 1e-7)).fillna(1.0)

    # JPX日次データベースのマージ
    if os.path.exists("jpx_daily_features_db.csv"):
        jpx_db = pd.read_csv("jpx_daily_features_db.csv", index_col=0, parse_dates=True)
        df = df.join(jpx_db, how='left')

    # 基本マクロ列の補完
    defaults = {
        'call_oi_wall': 66500.0,
        'put_oi_wall': 65000.0,
        'abn_net_flow': 0.0,
        'arbitrage_balance': 1000000.0,
        'gamma_flip': 65500.0,
        'call_volume': 1000.0,
        'put_volume': 1000.0,
        'pcr_ratio': 1.0,
        'us10y_yield': 4.0
    }
    for col, val in defaults.items():
        if col not in df.columns:
            df[col] = val
        else:
            df[col] = df[col].fillna(val)

    # --- v4拡張特徴量 ---
    # 1. ガンマフリップ境界との対数距離 ln(Price / GammaFlip)
    df['gamma_log_dist'] = np.log((df['NK_Close'] + 1e-7) / (df['gamma_flip'] + 1e-7)).fillna(0.0)
    
    # 2. 建玉前日比 (ΔOI)
    df['delta_call_oi'] = df['call_oi_wall'].diff().fillna(0.0)
    df['delta_put_oi'] = df['put_oi_wall'].diff().fillna(0.0)

    stock_cols = ['stock_ret_1d', 'stock_ret_5d', 'atr_ratio', 'rolling_beta']
    macro_cols = [
        'NK_Ret', 'FX_Ret', 'VIX_Close',
        'call_oi_wall', 'put_oi_wall', 'abn_net_flow', 'arbitrage_balance',
        'gamma_flip', 'call_volume', 'put_volume', 'pcr_ratio', 'us10y_yield',
        'gamma_log_dist', 'delta_call_oi'  # 14次元
    ]

    df = df.dropna(subset=['ATR', 'rolling_beta']).copy()
    return df, stock_cols, macro_cols

# =============================================================================
# 4. トリプルバリア法によるラベリング
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
        atr_val = atrs[i]
        upper_p = entry_p + (2.0 * atr_val)
        lower_p = entry_p - (1.0 * atr_val)

        outcome = 1 # デフォルトは保ち合い(WAIT)
        for h in range(1, holding_period + 1):
            cur_high = highs[i + h]
            cur_low = lows[i + h]

            if cur_low <= lower_p and cur_high >= upper_p:
                outcome = 0 # LOSS
                break
            elif cur_high >= upper_p:
                outcome = 2 # PROFIT
                break
            elif cur_low <= lower_p:
                outcome = 0 # LOSS
                break

        labels.append(outcome)

    df['Target'] = labels
    return df.dropna(subset=['Target']).copy()

# =============================================================================
# 5. データセットクラス
# =============================================================================
class TimeSeriesDataset(Dataset):
    def __init__(self, stock_arr, macro_arr, targets, seq_len=10):
        self.seq_len = seq_len
        self.samples = []
        for i in range(seq_len, len(targets)):
            s_seq = stock_arr[i - seq_len : i]
            m_seq = macro_arr[i - seq_len : i]
            # シーケンス内標準化
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
# 6. 学習および推論検証エンジン
# =============================================================================
def train_and_evaluate_v4(ticker: str = "7012.T", save_path: str = "swing_model_v4_test.pt", epochs: int = 8):
    print(f"\n{'='*60}")
    print(f"[*] 実験用モデル (v4) 学習パイプライン開始: {ticker}")
    print(f"[*] 保存先: {save_path} (本番ファイルは保護されます)")
    print(f"{'='*60}")

    df, stock_cols, macro_cols = load_and_preprocess_v4(ticker)
    if len(df) < 60:
        print("[-] データ数が不足しています。")
        return

    df = generate_triple_barrier_labels(df)
    
    stock_data = df[stock_cols].values
    macro_data = df[macro_cols].values
    targets = df['Target'].values

    # Train / Val 分割 (時系列順 80:20)
    split_idx = int(len(df) * 0.8)
    train_ds = TimeSeriesDataset(stock_data[:split_idx], macro_data[:split_idx], targets[:split_idx])
    val_ds = TimeSeriesDataset(stock_data[split_idx:], macro_data[split_idx:], targets[split_idx:])

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

    model = DualStreamGRU_v4(
        stock_dim=len(stock_cols),
        macro_dim=len(macro_cols),
        hidden_dim=44,
        num_classes=3
    )

    # 非対称ペナルティ損失関数 (誤爆ペナルティ 3.0倍)
    criterion = AsymmetricPenaltyLoss(false_buy_penalty=3.0)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    # 学習ループ
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
        
        avg_loss = total_loss / len(train_loader)
        if epoch % 2 == 0 or epoch == epochs:
            print(f"Epoch [{epoch}/{epochs}] - Loss: {avg_loss:.4f}")

    # チェックポイント保存 (メタデータ付き)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "stock_cols": stock_cols,
        "macro_cols": macro_cols,
        "model_version": "v4_experimental",
        "description": "DualStreamGRU with Asymmetric Loss & Extended Macro"
    }
    torch.save(checkpoint, save_path)
    print(f"\n[+] 実験用モデルを '{save_path}' として正常に保存しました。")

    # 検証セットでの推論確率分布の確認
    model.eval()
    val_profit_probs = []
    val_loss_probs = []
    with torch.no_grad():
        for x_s, x_m, _ in val_loader:
            logits = model(x_s, x_m)
            probs = torch.softmax(logits, dim=-1).numpy()
            val_loss_probs.extend(probs[:, 0])
            val_profit_probs.extend(probs[:, 2])

    print(f"[*] 検証期間 推論分布:")
    print(f"    平均利確確率: {np.mean(val_profit_probs)*100:.1f}% (最大: {np.max(val_profit_probs)*100:.1f}%)")
    print(f"    平均損切確率: {np.mean(val_loss_probs)*100:.1f}%")

if __name__ == "__main__":
    # まずは値動きと出来高が活発な川崎重工(7012.T)で検証
    train_and_evaluate_v4(ticker="7012.T", save_path="swing_model_v4_test.pt", epochs=8)