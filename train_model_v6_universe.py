import os
import time
import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

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
# 2. PyTorch Dataset 定義
# =============================================================================
class StockMacroDataset(Dataset):
    def __init__(self, X_stock, X_macro, y):
        self.X_stock = torch.tensor(X_stock, dtype=torch.float32)
        self.X_macro = torch.tensor(X_macro, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_stock[idx], self.X_macro[idx], self.y[idx]

# =============================================================================
# 3. マクロ指標の取得 & 結合
# =============================================================================
def load_macro_features(start_date="2021-01-01"):
    print("[*] マクロ・先物指標を取得中...")
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

    # JPX需給データベースの結合
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

    return macro_df

# =============================================================================
# 4. 複数銘柄のウィンドウ系列化 & データセット統合
# =============================================================================
def build_universe_dataset(tickers, macro_df, seq_len=10, horizon=5, tp_pct=0.04, sl_pct=-0.02):
    stock_cols = ['stock_ret_1d', 'stock_ret_5d', 'atr_ratio', 'rolling_beta']
    macro_cols = [
        'NK_Ret', 'FX_Ret', 'VIX_Close', 'call_oi_wall', 'put_oi_wall',
        'abn_net_flow', 'arbitrage_balance', 'gamma_flip', 'call_volume',
        'put_volume', 'pcr_ratio', 'us10y_yield', 'gamma_log_dist', 'delta_call_oi'
    ]

    all_x_stock, all_x_macro, all_y = [], [], []
    print(f"[*] 全 {len(tickers)} 銘柄から学習データを抽出・生成中...")

    for i, t in enumerate(tickers):
        try:
            df = yf.download(t, start="2021-01-01", interval="1d", progress=False)
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            df = df.dropna(subset=['Close', 'High', 'Low'])
            if len(df) < seq_len + horizon + 30: continue

            # 個別特徴量
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

            # マクロ指標と内部結合
            df = df.join(macro_df[macro_cols], how='inner')
            if len(df) < seq_len + horizon: continue

            # ラベル付け: 向こう horizon 日間の将来リターン
            future_max = df['High'].rolling(horizon).max().shift(-horizon)
            future_min = df['Low'].rolling(horizon).min().shift(-horizon)
            gain = (future_max - df['Close']) / df['Close']
            loss = (future_min - df['Close']) / df['Close']

            # 0: 損切, 1: 様子見, 2: 利確
            labels = pd.Series(1, index=df.index)
            labels[loss <= sl_pct] = 0
            labels[gain >= tp_pct] = 2

            df['label'] = labels
            df = df.dropna(subset=['label'])

            vals_s = df[stock_cols].values
            vals_m = df[macro_cols].values
            vals_y = df['label'].values.astype(int)

            # スライディングウィンドウ作成
            for idx in range(seq_len, len(df) - horizon):
                w_s = vals_s[idx-seq_len:idx]
                w_m = vals_m[idx-seq_len:idx]
                
                # Z-Score 正規化（ウィンドウ単位）
                w_s = (w_s - w_s.mean(axis=0)) / (w_s.std(axis=0) + 1e-7)
                w_m = (w_m - w_m.mean(axis=0)) / (w_m.std(axis=0) + 1e-7)

                all_x_stock.append(w_s)
                all_x_macro.append(w_m)
                all_y.append(vals_y[idx])

        except Exception:
            continue

        if (i + 1) % 10 == 0:
            print(f"  --> {i + 1}/{len(tickers)} 銘柄 処理完了")

    all_x_stock = np.array(all_x_stock)
    all_x_macro = np.array(all_x_macro)
    all_y = np.array(all_y)

    print(f"[+] 総サンプル数: {len(all_y)} (利確: {np.sum(all_y == 2)}, 損切: {np.sum(all_y == 0)}, 様子見: {np.sum(all_y == 1)})")
    return all_x_stock, all_x_macro, all_y, stock_cols, macro_cols

# =============================================================================
# 5. モデル学習メイン処理 (Early Stopping)
# =============================================================================
def train_universe_model():
    # 銘柄プールの読み込み（先ほどのスクリーニング結果CSVまたは固定リスト）
    if os.path.exists("sbi_screened_v6_results.csv"):
        df_screened = pd.read_csv("sbi_screened_v6_results.csv")
        tickers = df_screened['Ticker'].tolist()
        print(f"[*] 'sbi_screened_v6_results.csv' より {len(tickers)} 銘柄を学習対象としてロードしました。")
    else:
        # デフォルト代表銘柄プール（セクター分散）
        tickers = [
            "7203.T", "6758.T", "8035.T", "8306.T", "8725.T", "8766.T",
            "9432.T", "9502.T", "2810.T", "4205.T", "5401.T", "8058.T"
        ]
        print(f"[*] デフォルト代表 {len(tickers)} 銘柄を使用します。")

    macro_df = load_macro_features()
    X_s, X_m, y, s_cols, m_cols = build_universe_dataset(tickers, macro_df)

    if len(y) == 0:
        print("[!] 学習データが不足しています。")
        return

    # Train / Val 分割 (時系列構造を壊さないようシャッフルありで層化抽出)
    X_s_tr, X_s_va, X_m_tr, X_m_va, y_tr, y_va = train_test_split(
        X_s, X_m, y, test_size=0.2, random_state=42, stratify=y
    )

    train_ds = StockMacroDataset(X_s_tr, X_m_tr, y_tr)
    val_ds = StockMacroDataset(X_s_va, X_m_va, y_va)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)

    # クラス不均衡ペナルティ (Class Weights)
    class_counts = np.bincount(y_tr)
    total_samples = len(y_tr)
    weights = total_samples / (len(class_counts) * class_counts + 1e-5)
    class_weights = torch.tensor(weights, dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualStreamGRU_v6_CrossAttn(
        stock_dim=len(s_cols), macro_dim=len(m_cols), hidden_dim=48, num_classes=3
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    best_val_loss = float('inf')
    patience = 4
    patience_cnt = 0
    save_path = "swing_model_v6_crossattn_universe.pt"

    print("\n[*] 複数銘柄ユニバース学習開始 (Patience=4)...")
    epochs = 15

    for epoch in range(1, epochs + 1):
        model.train()
        total_tr_loss = 0.0
        for b_xs, b_xm, b_y in train_loader:
            b_xs, b_xm, b_y = b_xs.to(device), b_xm.to(device), b_y.to(device)
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
                b_xs, b_xm, b_y = b_xs.to(device), b_xm.to(device), b_y.to(device)
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
                'hidden_dim': 48,
                'val_loss': best_val_loss
            }, save_path)
            print(f"  Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}  --> [Best Val Loss 更新 ★]")
        else:
            patience_cnt += 1
            print(f"  Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}  (Patience: {patience_cnt}/{patience})")
            if patience_cnt >= patience:
                print(f"[*] Early Stopping 発動: Epoch {epoch} で学習を終了します。")
                break

    print(f"\n[+] 最適モデル重みを保存しました: {save_path}")

if __name__ == "__main__":
    train_universe_model()