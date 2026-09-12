import os
import datetime
import calendar
import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix

BASE_DIR = r"F:\stockModel"
DB_PATH = os.path.join(BASE_DIR, "jpx_daily_features_db.csv")
MODEL_PATH = os.path.join(BASE_DIR, "swing_model_v6_crossattn_universe_v5.pt")
UNIVERSE_PATH = os.path.join(BASE_DIR, "universe_150_tickers.txt")

# --- モデル定義 ---
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

class UniverseDataset(Dataset):
    def __init__(self, X_stock, X_macro, y):
        self.X_stock = torch.tensor(X_stock, dtype=torch.float32)
        self.X_macro = torch.tensor(X_macro, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_stock[idx], self.X_macro[idx], self.y[idx]

# --- マクロデータ構築 ---
def load_macro_slim5(db_path=DB_PATH):
    jpx_db = pd.read_csv(db_path, index_col=0, parse_dates=True)
    jpx_db.index = pd.to_datetime(jpx_db.index).tz_localize(None)
    start_date = (jpx_db.index.min() - datetime.timedelta(days=20)).strftime("%Y-%m-%d")

    n225 = yf.download("^N225", start=start_date, interval="1d", progress=False)
    if isinstance(n225.columns, pd.MultiIndex): n225.columns = n225.columns.get_level_values(0)
    n225.index = pd.to_datetime(n225.index).tz_localize(None)

    fx = yf.download("USDJPY=X", start=start_date, interval="1d", progress=False)
    if isinstance(fx.columns, pd.MultiIndex): fx.columns = fx.columns.get_level_values(0)
    fx.index = pd.to_datetime(fx.index).tz_localize(None)

    macro_df = pd.DataFrame(index=n225.index)
    macro_df['NK_Close'] = n225['Close']
    macro_df['NK_Ret'] = n225['Close'].pct_change(fill_method=None).fillna(0.0)

    macro_df = macro_df.join(jpx_db, how='inner').ffill().bfill()

    pin_strike = (macro_df['call_oi_wall'] + macro_df['put_oi_wall']) / 2.0
    macro_df['pin_dist_ratio'] = ((macro_df['NK_Close'] - pin_strike) / (pin_strike + 1e-5)) / 0.02
    macro_df['wall_spread'] = ((macro_df['call_oi_wall'] - macro_df['put_oi_wall']).abs() / (pin_strike + 1e-5)) / 0.02

    cta_mean = macro_df['cta_net_futures'].rolling(60, min_periods=10).mean()
    cta_std = macro_df['cta_net_futures'].rolling(60, min_periods=10).std() + 1e-5
    macro_df['cta_net_norm'] = ((macro_df['cta_net_futures'] - cta_mean) / cta_std).fillna(0.0)

    macro_df['cta_momentum'] = (macro_df['cta_net_futures'] - macro_df['cta_net_futures'].shift(5)).fillna(0.0) / 5000.0
    macro_df['nk_ret_norm'] = macro_df['NK_Ret'] / 0.015

    slim_cols = ['pin_dist_ratio', 'wall_spread', 'cta_net_norm', 'cta_momentum', 'nk_ret_norm']
    return macro_df[slim_cols]

# --- 評価データセット構築 ---
def build_universe_dataset_slim(tickers, macro_df, seq_len=10, holding_period=10):
    stock_cols = ['stock_ret_1d', 'stock_ret_5d', 'atr_ratio', 'rolling_beta', 'vol_ratio_5d']
    macro_cols = list(macro_df.columns)

    tr_x_s, tr_x_m, tr_y = [], [], []
    va_x_s, va_x_m, va_y = [], [], []

    m_start = (macro_df.index.min() - datetime.timedelta(days=40)).strftime("%Y-%m-%d")

    for i, t in enumerate(tickers):
        try:
            df = yf.download(t, start=m_start, interval="1d", progress=False)
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df = df.dropna(subset=['Close', 'High', 'Low', 'Volume'])
            if len(df) < seq_len + holding_period + 20: continue

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

            aligned_nk = macro_df['nk_ret_norm'].reindex(df.index).fillna(0.0)
            cov = df['stock_ret_1d'].rolling(20).cov(aligned_nk)
            var = aligned_nk.rolling(20).var()
            df['rolling_beta'] = (cov / (var + 1e-7)).fillna(1.0)

            df = df.join(macro_df, how='inner')
            df = df.dropna(subset=['ATR', 'rolling_beta'] + macro_cols)
            if len(df) < seq_len + holding_period + 5: continue

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

    return (np.array(tr_x_s), np.array(tr_x_m), np.array(tr_y)), \
           (np.array(va_x_s), np.array(va_x_m), np.array(va_y)), stock_cols, macro_cols

# --- 評価実行関数 ---
def evaluate_best_model():
    if not os.path.exists(MODEL_PATH):
        print(f"[!] モデル重みが見つかりません: {MODEL_PATH}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(MODEL_PATH, map_location=device)

    print("[*] 評価用データセットを構築中...")
    macro_df = load_macro_slim5()
    
    if os.path.exists(UNIVERSE_PATH):
        with open(UNIVERSE_PATH, "r", encoding="utf-8") as f:
            tickers = [line.strip() for line in f if line.strip()]
    else:
        tickers = ["7203.T", "6758.T", "8035.T", "8306.T", "9432.T", "7167.T", "4519.T", "5726.T"]

    _, val_data, s_cols, m_cols = build_universe_dataset_slim(tickers, macro_df)
    va_x_s, va_x_m, va_y = val_data

    val_ds = UniverseDataset(va_x_s, va_x_m, va_y)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)

    model = DualStreamGRU_v6_Slim5(
        stock_dim=len(s_cols), macro_dim=len(m_cols), hidden_dim=checkpoint['hidden_dim'], num_heads=2, num_classes=3
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    all_preds = []
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for b_xs, b_xm, b_y in val_loader:
            b_xs, b_xm = b_xs.to(device), b_xm.to(device)
            logits = model(b_xs, b_xm)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(b_y.numpy())

    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)

    print("\n" + "=" * 55)
    print(f"[*] モデル評価レポート (Best Val Loss: {checkpoint.get('val_loss', 0.0):.4f})")
    print("=" * 55)

    print("\n【混同行列 (行: 実際 / 列: 予測)】")
    cm = confusion_matrix(all_targets, all_preds)
    cm_df = pd.DataFrame(cm, index=['実際_0(下落)', '実際_1(揉合)', '実際_2(上昇)'], 
                             columns=['予測_0', '予測_1', '予測_2'])
    print(cm_df)

    print("\n【詳細分類レポート】")
    print(classification_report(all_targets, all_preds, target_names=['0(下落)', '1(揉合)', '2(上昇)'], digits=4))

    print("\n【クラス2（買い）の確信度（閾値）別 勝率】")
    for th in [0.35, 0.40, 0.45, 0.50, 0.55]:
        mask = all_probs[:, 2] >= th
        if mask.sum() > 0:
            precision = (all_targets[mask] == 2).sum() / mask.sum()
            loss_rate = (all_targets[mask] == 0).sum() / mask.sum()
            print(f"  確信度 >= {th:.2f} | 件数: {mask.sum():5d} | 利確到達率: {precision*100:5.1f}% | 損切り率: {loss_rate*100:5.1f}%")

if __name__ == "__main__":
    evaluate_best_model()