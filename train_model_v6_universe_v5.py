import os
import random
import datetime
import calendar
import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# =============================================================================
# 設定・パス・乱数シード固定（再現性の担保）
# =============================================================================
BASE_DIR = r"F:\stockModel"
DB_PATH = os.path.join(BASE_DIR, "jpx_daily_features_db.csv")
UNIVERSE_PATH = os.path.join(BASE_DIR, "universe_150_tickers.txt")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "swing_model_v6_crossattn_universe_v5.pt")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# =============================================================================
# 1. 改修版v6 モデルアーキテクチャ (軽量Cross-Attention 24次元)
# =============================================================================
class DecayPooling(nn.Module):
    def __init__(self, seq_len=10):
        super(DecayPooling, self).__init__()
        weights = np.exp(np.linspace(-1.5, 0.0, seq_len))
        weights = weights / weights.sum()
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32).unsqueeze(0).unsqueeze(-1))

    def forward(self, x):
        return torch.sum(x * self.weights, dim=1)

class CrossAttentionBlock(nn.Module):
    def __init__(self, hidden_dim=24, num_heads=2, dropout=0.3):
        super(CrossAttentionBlock, self).__init__()
        self.mha = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key_value):
        attn_out, _ = self.mha(query=query, key=key_value, value=key_value)
        return self.norm(query + self.dropout(attn_out))

class DualStreamGRU_v6_Slim5(nn.Module):
    def __init__(self, stock_dim=5, macro_dim=5, hidden_dim=24, num_heads=2, num_classes=3):
        super(DualStreamGRU_v6_Slim5, self).__init__()
        self.stock_gru = nn.GRU(stock_dim, hidden_dim, batch_first=True, num_layers=1)
        self.macro_gru = nn.GRU(macro_dim, hidden_dim, batch_first=True, num_layers=1)
        self.cross_attn = CrossAttentionBlock(hidden_dim, num_heads=num_heads, dropout=0.3)
        self.pool = DecayPooling(seq_len=10)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, num_classes)
        )

    def forward(self, x_stock, x_macro):
        out_stock, _ = self.stock_gru(x_stock)
        out_macro, _ = self.macro_gru(x_macro)
        fused = self.cross_attn(out_stock, out_macro)
        pooled = self.pool(fused)
        return self.classifier(pooled)

class AsymmetricPenaltyLoss(nn.Module):
    def __init__(self, false_buy_penalty: float = 1.3, class_weights=None):
        super(AsymmetricPenaltyLoss, self).__init__()
        self.penalty = false_buy_penalty
        self.ce = nn.CrossEntropyLoss(weight=class_weights, reduction='none')

    def forward(self, logits, targets):
        base_loss = self.ce(logits, targets)
        probs = torch.softmax(logits, dim=-1)
        is_actual_loss = (targets == 0)
        p_profit = probs[:, 2]  # 買い（2）と誤認した確率
        multiplier = torch.ones_like(base_loss)
        multiplier[is_actual_loss] += (self.penalty - 1.0) * p_profit[is_actual_loss]
        return (base_loss * multiplier).mean()

class UniverseDataset(Dataset):
    def __init__(self, X_stock, X_macro, y):
        self.X_stock = torch.tensor(X_stock, dtype=torch.float32)
        self.X_macro = torch.tensor(X_macro, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_stock[idx], self.X_macro[idx], self.y[idx]

# =============================================================================
# 2. マクロ環境データ構築
# =============================================================================
def load_macro_slim5(db_path=DB_PATH):
    print("[*] マクロ・需給特徴量（確定建玉＋CTA先物 5次元）を構築中...")
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"[!] {db_path} が見つかりません。")

    jpx_db = pd.read_csv(db_path, index_col=0, parse_dates=True)
    jpx_db.index = pd.to_datetime(jpx_db.index).tz_localize(None)
    start_date = (jpx_db.index.min() - datetime.timedelta(days=20)).strftime("%Y-%m-%d")

    n225 = yf.download("^N225", start=start_date, interval="1d", progress=False)
    if isinstance(n225.columns, pd.MultiIndex):
        n225.columns = n225.columns.get_level_values(0)
    n225.index = pd.to_datetime(n225.index).tz_localize(None)

    macro_df = pd.DataFrame(index=n225.index)
    macro_df['NK_Close'] = n225['Close']
    macro_df['NK_Ret'] = n225['Close'].pct_change(fill_method=None).fillna(0.0)

    # 過去リークを防ぐため、前方補間のみで欠損を処理
    macro_df = macro_df.join(jpx_db, how='inner').ffill().fillna(0.0)

    pin_strike = (macro_df['call_oi_wall'] + macro_df['put_oi_wall']) / 2.0
    macro_df['pin_dist_ratio'] = ((macro_df['NK_Close'] - pin_strike) / (pin_strike + 1e-5)) / 0.02
    macro_df['wall_spread'] = ((macro_df['call_oi_wall'] - macro_df['put_oi_wall']).abs() / (pin_strike + 1e-5)) / 0.02

    cta_mean = macro_df['cta_net_futures'].rolling(60, min_periods=10).mean()
    cta_std = macro_df['cta_net_futures'].rolling(60, min_periods=10).std() + 1e-5
    macro_df['cta_net_norm'] = ((macro_df['cta_net_futures'] - cta_mean) / cta_std).fillna(0.0)

    macro_df['cta_momentum'] = (macro_df['cta_net_futures'] - macro_df['cta_net_futures'].shift(5)).fillna(0.0) / 5000.0
    macro_df['nk_ret_norm'] = macro_df['NK_Ret'] / 0.015

    return macro_df

# =============================================================================
# 3. データセット構築（生リターン基準ベータ ＆ 10日保有ターゲット）
# =============================================================================
def build_universe_dataset_slim(tickers, macro_df, seq_len=10, holding_period=10):
    stock_cols = ['stock_ret_1d', 'stock_ret_5d', 'atr_ratio', 'rolling_beta', 'vol_ratio_5d']
    macro_cols = ['pin_dist_ratio', 'wall_spread', 'cta_net_norm', 'cta_momentum', 'nk_ret_norm']

    tr_x_s, tr_x_m, tr_y = [], [], []
    va_x_s, va_x_m, va_y = [], [], []

    print(f"[*] 全 {len(tickers)} 銘柄からデータセット構築中...")
    m_start = (macro_df.index.min() - datetime.timedelta(days=40)).strftime("%Y-%m-%d")

    for i, t in enumerate(tickers):
        try:
            df = yf.download(t, start=m_start, interval="1d", progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df = df.dropna(subset=['Close', 'High', 'Low', 'Volume'])

            # 欠損・売買停止・データ不足ガード
            if len(df) < seq_len + holding_period + 25 or (df['Volume'] == 0).all():
                continue

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

            # 生リターン基準のベータ
            aligned_nk = macro_df['NK_Ret'].reindex(df.index).fillna(0.0)
            cov = df['stock_ret_1d'].rolling(20).cov(aligned_nk)
            var = aligned_nk.rolling(20).var()
            df['rolling_beta'] = (cov / (var + 1e-7)).fillna(1.0)

            df = df.join(macro_df[macro_cols], how='inner')
            df = df.dropna(subset=['ATR', 'rolling_beta'] + macro_cols)
            if len(df) < seq_len + holding_period + 10:
                continue

            closes = df['Close'].values
            highs = df['High'].values
            lows = df['Low'].values
            atrs = df['ATR'].values
            n_bars = len(df)

            targets = np.full(n_bars, np.nan)
            for idx in range(n_bars - holding_period):
                entry_p = closes[idx]
                upper_p = entry_p + (2.0 * atrs[idx])
                lower_p = entry_p - (1.0 * atrs[idx])
                outcome = 1
                for h in range(1, holding_period + 1):
                    if lows[idx + h] <= lower_p and highs[idx + h] >= upper_p:
                        outcome = 0
                        break
                    elif highs[idx + h] >= upper_p:
                        outcome = 2
                        break
                    elif lows[idx + h] <= lower_p:
                        outcome = 0
                        break
                targets[idx] = outcome

            df['Target'] = targets
            df = df.dropna(subset=['Target'])

            vals_s = df[stock_cols].values
            vals_m = df[macro_cols].values
            vals_y = df['Target'].values.astype(int)

            n_samples = len(df)
            split_idx = int(n_samples * 0.75)

            for idx in range(seq_len - 1, n_samples):
                w_s = vals_s[idx - seq_len + 1 : idx + 1].copy()
                w_m = vals_m[idx - seq_len + 1 : idx + 1].copy()

                w_s = (w_s - w_s.mean(axis=0)) / (w_s.std(axis=0) + 1e-7)
                target = vals_y[idx]

                if idx < split_idx:
                    tr_x_s.append(w_s)
                    tr_x_m.append(w_m)
                    tr_y.append(target)
                else:
                    va_x_s.append(w_s)
                    va_x_m.append(w_m)
                    va_y.append(target)

        except Exception:
            continue

        if (i + 1) % 50 == 0 or (i + 1) == len(tickers):
            print(f"  --> {i + 1}/{len(tickers)} 銘柄 完了")

    tr_x_s, tr_x_m, tr_y = np.array(tr_x_s), np.array(tr_x_m), np.array(tr_y)
    va_x_s, va_x_m, va_y = np.array(va_x_s), np.array(va_x_m), np.array(va_y)
    return (tr_x_s, tr_x_m, tr_y), (va_x_s, va_x_m, va_y), stock_cols, macro_cols

# =============================================================================
# 4. 学習ループ (改修版v6 黄金比安定構成)
# =============================================================================
def train_universe_model_v5():
    if os.path.exists(UNIVERSE_PATH):
        with open(UNIVERSE_PATH, "r", encoding="utf-8") as f:
            tickers = [line.strip() for line in f if line.strip()]
    else:
        tickers = ["7203.T", "6758.T", "8035.T", "8306.T", "9432.T", "7167.T", "4519.T", "5726.T"]

    macro_df = load_macro_slim5()
    print(f"[+] マクロ実データ整合期間: {macro_df.index.min().date()} 〜 {macro_df.index.max().date()} ({len(macro_df)} 営業日)")

    train_data, val_data, s_cols, m_cols = build_universe_dataset_slim(tickers, macro_df, seq_len=10, holding_period=10)

    tr_x_s, tr_x_m, tr_y = train_data
    va_x_s, va_x_m, va_y = val_data

    print(f"[+] データセット構築完了: Trainサンプル数 = {len(tr_y)}, Valサンプル数 = {len(va_y)}")

    train_ds = UniverseDataset(tr_x_s, tr_x_m, tr_y)
    val_ds = UniverseDataset(va_x_s, va_x_m, va_y)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)

    class_counts = np.bincount(tr_y)
    weights = len(tr_y) / (len(class_counts) * class_counts + 1e-5)
    class_weights = torch.tensor(weights, dtype=torch.float32)

    model = DualStreamGRU_v6_Slim5(
        stock_dim=len(s_cols), macro_dim=len(m_cols), hidden_dim=24, num_heads=2, num_classes=3
    ).to(DEVICE)

    criterion = AsymmetricPenaltyLoss(false_buy_penalty=1.3, class_weights=class_weights.to(DEVICE))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00015, weight_decay=1e-2)

    best_val_loss = float('inf')
    patience = 6
    patience_cnt = 0

    print("\n[*] [改修版v6 決定版] 確定建玉＋CTA先物・非対称損失関数による学習開始 (lr=0.00015)...")
    epochs = 25

    for epoch in range(1, epochs + 1):
        model.train()
        total_tr_loss = 0.0
        for b_xs, b_xm, b_y in train_loader:
            b_xs, b_xm, b_y = b_xs.to(DEVICE), b_xm.to(DEVICE), b_y.to(DEVICE)
            optimizer.zero_grad()
            out = model(b_xs, b_xm)
            loss = criterion(out, b_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_tr_loss += loss.item() * len(b_y)

        train_loss = total_tr_loss / len(train_ds)

        model.eval()
        total_va_loss = 0.0
        with torch.no_grad():
            for b_xs, b_xm, b_y in val_loader:
                b_xs, b_xm, b_y = b_xs.to(DEVICE), b_xm.to(DEVICE), b_y.to(DEVICE)
                out = model(b_xs, b_xm)
                loss = criterion(out, b_y)
                total_va_loss += loss.item() * len(b_y)

        val_loss = total_va_loss / len(val_ds)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_cnt = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'stock_cols': s_cols,
                'macro_cols': m_cols,
                'hidden_dim': 24,
                'val_loss': best_val_loss
            }, MODEL_SAVE_PATH)
            print(f"  Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}  --> [Best Val Loss 更新 ★]")
        else:
            patience_cnt += 1
            print(f"  Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}  (Patience: {patience_cnt}/{patience})")
            if patience_cnt >= patience:
                print(f"[*] Early Stopping 発動: Epoch {epoch} で学習を終了します。")
                break

    print(f"\n[+] 最適モデル重みを保存しました: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_universe_model_v5()