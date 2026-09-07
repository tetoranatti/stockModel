import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
import yfinance as yf

np.random.seed(42)
torch.manual_seed(42)

DB_FILE = "jpx_daily_features_db.csv"

# ==============================================================================
# 1. ATR動的トリプルバリア法（先10営業日判定）
# ==============================================================================
def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df['high']
    low = df['low']
    close_prev = df['close'].shift(1)
    tr1 = high - low
    tr2 = (high - close_prev).abs()
    tr3 = (low - close_prev).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return true_range.ewm(alpha=1/period, min_periods=period, adjust=False).mean()


def apply_dynamic_triple_barrier(
    df: pd.DataFrame, max_holding_period: int = 10, atr_period: int = 14,
    tp_multiplier: float = 2.0, sl_multiplier: float = 1.0
) -> pd.DataFrame:
    df = df.copy()
    df['atr'] = calculate_atr(df, period=atr_period)
    labels = [np.nan] * len(df)
    high_vals, low_vals, close_vals, atr_vals = df['high'].values, df['low'].values, df['close'].values, df['atr'].values
    n = len(df)
    
    for i in range(n):
        if i + max_holding_period >= n or np.isnan(atr_vals[i]):
            continue
        entry_price = close_vals[i]
        upper_barrier = entry_price + (atr_vals[i] * tp_multiplier)
        lower_barrier = entry_price - (atr_vals[i] * sl_multiplier)
        
        future_high = high_vals[i + 1 : i + 1 + max_holding_period]
        future_low = low_vals[i + 1 : i + 1 + max_holding_period]
        
        hit_tp = np.where(future_high >= upper_barrier)[0]
        hit_sl = np.where(future_low <= lower_barrier)[0]
        
        first_tp = hit_tp[0] if len(hit_tp) > 0 else np.inf
        first_sl = hit_sl[0] if len(hit_sl) > 0 else np.inf
        
        if first_tp < first_sl:
            labels[i] = 1
        elif first_sl <= first_tp and first_sl != np.inf:
            labels[i] = -1
        else:
            labels[i] = 0
            
    df['label'] = labels
    return df


# ==============================================================================
# 2. 完全実データ取得 & 堅牢なハイブリッド蓄積DB結合 (計16次元)
# ==============================================================================
def fetch_100pct_real_market_data(ticker: str = "7012.T", period: str = "2y") -> pd.DataFrame:
    print(f"市場データを同期中 ({ticker}, 先物, 為替, VIX, 手口DB / 期間: {period}) ...")
    
    # 1. 個別株実データ
    stock_df = yf.Ticker(ticker).history(period=period, auto_adjust=True).reset_index()
    stock_df['date'] = pd.to_datetime(stock_df['Date']).dt.tz_localize(None).dt.floor('D')
    stock_df = stock_df.rename(columns={'Open':'open', 'High':'high', 'Low':'low', 'Close':'close', 'Volume':'volume'})
    stock_df['atr'] = calculate_atr(stock_df, period=14)
    stock_df['stock_ret_1d'] = np.log(stock_df['close'] / stock_df['close'].shift(1))
    stock_df['stock_ret_5d'] = np.log(stock_df['close'] / stock_df['close'].shift(5))
    stock_df['atr_ratio'] = stock_df['atr'] / stock_df['close']

    # 2. 日経先物 (NKD=F)
    fut_df = yf.Ticker("NKD=F").history(period=period, auto_adjust=True).reset_index()
    fut_df['date'] = pd.to_datetime(fut_df['Date']).dt.tz_localize(None).dt.floor('D')
    fut_df['fut_ret_1d'] = np.log(fut_df['Close'] / fut_df['Close'].shift(1))
    fut_df['fut_ret_5d'] = np.log(fut_df['Close'] / fut_df['Close'].shift(5))
    fut_df['fut_vol_ratio'] = fut_df['Volume'] / fut_df['Volume'].rolling(5, min_periods=1).mean()
    fut_features = fut_df[['date', 'Close', 'Volume', 'fut_ret_1d', 'fut_ret_5d', 'fut_vol_ratio']].rename(
        columns={'Close': 'fut_close', 'Volume': 'fut_volume'}
    )

    # 3. 為替 ドル円 (JPY=X)
    usdjpy_df = yf.Ticker("JPY=X").history(period=period, auto_adjust=True).reset_index()
    usdjpy_df['date'] = pd.to_datetime(usdjpy_df['Date']).dt.tz_localize(None).dt.floor('D')
    usdjpy_df['usdjpy_ret_1d'] = np.log(usdjpy_df['Close'] / usdjpy_df['Close'].shift(1)).shift(1)

    # 4. 米国VIX (^VIX)
    vix_df = yf.Ticker("^VIX").history(period=period, auto_adjust=True).reset_index()
    vix_df['date'] = pd.to_datetime(vix_df['Date']).dt.tz_localize(None).dt.floor('D')
    vix_df['vix_level'] = (vix_df['Close'] / 100.0).shift(1)

    # 5. 日本市場ボラティリティ（日経先物の20日実測HVから直接算出・404エラー完全排除）
    fut_df_sorted = fut_features.sort_values('date').copy()
    fut_hv20 = fut_df_sorted['fut_ret_1d'].rolling(window=20, min_periods=5).std() * np.sqrt(250)
    jniv_features = pd.DataFrame({
        'date': fut_df_sorted['date'],
        'jniv_level': fut_hv20.fillna(0.20).values,
        'jniv_ret_1d': np.log(fut_hv20 / fut_hv20.shift(1)).fillna(0.0).values
    })

    # 基本マージ
    merged = stock_df.merge(fut_features, on='date', how='left')
    merged = merged.merge(usdjpy_df[['date', 'usdjpy_ret_1d']], on='date', how='left')
    merged = merged.merge(vix_df[['date', 'vix_level']], on='date', how='left')
    merged = merged.merge(jniv_features, on='date', how='left')
    merged['stock_fut_spread'] = merged['stock_ret_1d'] - merged['fut_ret_1d']

    # 6. JPX蓄積DBのマージ
    if os.path.exists(DB_FILE):
        try:
            jpx_db = pd.read_csv(DB_FILE)
            jpx_db['date'] = pd.to_datetime(jpx_db['date']).dt.tz_localize(None).dt.floor('D')
            merged = merged.merge(jpx_db, on='date', how='left')
            print(f"[*] JPX蓄積DBを検出: {len(jpx_db)} 日分の実測データを結合")
        except Exception as e:
            print(f"[!] JPX蓄積DBの読み込みをスキップ: {e}")
            for col in ['call_wall_strike', 'put_wall_strike', 'abn_net_trade', 'arbitrage_buy_stocks']:
                merged[col] = np.nan
    else:
        for col in ['call_wall_strike', 'put_wall_strike', 'abn_net_trade', 'arbitrage_buy_stocks']:
            merged[col] = np.nan

    # 実測値があれば優先採用、過去日・未蓄積日は近似式でフォールバック補完
    merged['dist_call_wall'] = np.where(
        merged['call_wall_strike'].notnull(),
        np.log(merged['fut_close'] / merged['call_wall_strike']),
        np.log(merged['fut_close'] / 66500.0)
    )
    merged['dist_put_wall'] = np.where(
        merged['put_wall_strike'].notnull(),
        np.log(merged['fut_close'] / merged['put_wall_strike']),
        np.log(merged['fut_close'] / 65000.0)
    )
    
    vol_spike = (merged['fut_volume'] > (merged['fut_volume'].rolling(20, min_periods=5).mean() * 1.8)) & (merged['fut_ret_1d'].abs() > 0.012)
    merged['abn_shock_flag'] = np.where(
        merged['abn_net_trade'].notnull(),
        (merged['abn_net_trade'].abs() > 1000).astype(float),
        vol_spike.astype(float)
    )

    merged['arbitrage_pressure'] = np.where(
        merged['arbitrage_buy_stocks'].notnull(),
        merged['arbitrage_buy_stocks'].pct_change().fillna(0.0),
        merged['stock_fut_spread'].ewm(span=10).mean()
    )

    # ベータ値算出
    aligned = pd.concat([
        stock_df.set_index('date')['stock_ret_1d'].rename('stock_ret'),
        fut_features.set_index('date')['fut_ret_1d'].rename('fut_ret')
    ], axis=1, join='inner').dropna()
    rolling_cov = aligned['stock_ret'].rolling(window=60, min_periods=15).cov(aligned['fut_ret'])
    rolling_var = aligned['fut_ret'].rolling(window=60, min_periods=15).var()
    aligned_beta = (rolling_cov / (rolling_var + 1e-6)).reset_index()
    aligned_beta.columns = ['date', 'rolling_beta']
    merged = merged.merge(aligned_beta[['date', 'rolling_beta']], on='date', how='left')

    # 【重要】欠損を前方・後方補完してから dropna
    merged['rolling_beta'] = merged['rolling_beta'].ffill().bfill().fillna(1.0)
    merged = merged.ffill().bfill()
    merged = merged.dropna().reset_index(drop=True)
    print(f"[*] データセット作成完了: 有効レコード数 = {len(merged)} 行")
    return merged


# ==============================================================================
# 3. Dataset & 2系統ブランチ型GRUモデル (16次元)
# ==============================================================================
class FinancialTimeSeriesDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray, seq_len: int = 20):
        self.seq_len = seq_len
        self.X_samples = []
        self.y_samples = []
        for i in range(seq_len - 1, len(features)):
            if np.isnan(labels[i]): continue
            self.X_samples.append(features[i - seq_len + 1 : i + 1])
            self.y_samples.append(labels[i])
        self.X_samples = torch.tensor(np.array(self.X_samples), dtype=torch.float32)
        self.y_samples = torch.tensor(np.array(self.y_samples), dtype=torch.long)

    def __len__(self): return len(self.X_samples)
    def __getitem__(self, idx): return self.X_samples[idx], self.y_samples[idx]


class MacroInformedSwingNet(nn.Module):
    def __init__(self, stock_dim: int, macro_dim: int, hidden_dim: int = 44, num_classes: int = 3):
        super().__init__()
        self.stock_dim = stock_dim
        self.stock_gru = nn.GRU(input_size=stock_dim, hidden_size=hidden_dim, num_layers=2, batch_first=True, dropout=0.2)
        self.macro_gru = nn.GRU(input_size=macro_dim, hidden_size=hidden_dim, num_layers=2, batch_first=True, dropout=0.2)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x_stock = x[:, :, :self.stock_dim]
        x_macro = x[:, :, self.stock_dim:]
        out_stock, _ = self.stock_gru(x_stock)
        out_macro, _ = self.macro_gru(x_macro)
        fused = torch.cat([out_stock[:, -1, :], out_macro[:, -1, :]], dim=1)
        return self.classifier(fused)


# ==============================================================================
# 4. 学習 & 推論ルーチン
# ==============================================================================
def train_pipeline(model, train_loader, val_loader, class_weights, epochs=10, lr=7e-4, device="cpu"):
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    model.to(device)
    print(f"\n--- 本番モデル学習開始 (Device: {device}) ---")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * len(y_batch)
            correct += (torch.argmax(logits, dim=1) == y_batch).sum().item()
            total += len(y_batch)
            
        model.eval()
        val_loss, val_correct, val_total, tp_correct, tp_total = 0.0, 0, 0, 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                val_loss += loss.item() * len(y_batch)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == y_batch).sum().item()
                val_total += len(y_batch)
                tp_mask = (preds == 2)
                tp_correct += ((preds == y_batch) & tp_mask).sum().item()
                tp_total += tp_mask.sum().item()
                
        print(f"Epoch [{epoch:02d}/{epochs:02d}] Train Loss: {total_loss/total:.4f} Acc: {correct/total:.3f} | Val Loss: {val_loss/val_total:.4f} Acc: {val_correct/val_total:.3f} | 利確適合率: {(tp_correct/tp_total) if tp_total>0 else 0.0:.3f}")


def predict_tomorrow_signal(df: pd.DataFrame, ticker: str, checkpoint_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    feature_cols = checkpoint['feature_cols']
    scaler = checkpoint['scaler']
    
    model = MacroInformedSwingNet(stock_dim=len(checkpoint['stock_cols']), macro_dim=len(checkpoint['macro_cols']), hidden_dim=44, num_classes=3)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    latest_window_raw = df[feature_cols].iloc[-20:].values
    x_input = torch.tensor(scaler.transform(latest_window_raw), dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = F.softmax(model(x_input), dim=1).cpu().numpy()[0]

    prob_loss, prob_neutral, prob_profit = probs[0], probs[1], probs[2]
    latest = df.iloc[-1]
    
    data_source_mode = "JPX実測蓄積データ" if ('call_wall_strike' in df.columns and pd.notnull(latest.get('call_wall_strike'))) else "実データ派生近似ロジック"
    target_tp = latest['close'] + (latest['atr'] * 2.0)
    target_sl = latest['close'] - (latest['atr'] * 1.0)

    print("\n" + "="*70)
    print(f"【JPX蓄積パイプライン連動・実戦推論レポート】 対象: {ticker}")
    print(f"基準日: {latest['date'].strftime('%Y-%m-%d')} | 終値: {latest['close']:.1f} 円 | ATR(14): {latest['atr']:.1f} 円")
    print(f"需給データ適用モード: [ {data_source_mode} ]")
    print(f"市場感応度(実ベータ): {latest['rolling_beta']:.2f} | ボラティリティ: {latest['jniv_level']*100:.2f}%")
    print("-" * 70)
    print("■ シナリオ確率 (先10営業日):")
    print(f"  ・利確到達確率 (+1):  {prob_profit * 100:.2f} %")
    print(f"  ・保ち合い/停滞 (0):   {prob_neutral * 100:.2f} %")
    print(f"  ・損切到達確率 (-1):  {prob_loss * 100:.2f} %")
    print("-" * 70)
    
    signal = "BUY (買いエントリー推奨)" if (prob_profit >= 0.42 and prob_profit > prob_loss * 1.5) else ("AVOID / SHORT (下落警戒)" if prob_loss >= 0.50 else "WAIT (シグナルなし・静観)")
    print(f"判定シグナル: >>> {signal} <<<")
    print(f"  ・利確目標ライン (+2.0 ATR): {target_tp:.1f} 円 (+{(target_tp/latest['close'] - 1)*100:.1f}%)")
    print(f"  ・防衛損切ライン (-1.0 ATR): {target_sl:.1f} 円 (-{(1 - target_sl/latest['close'])*100:.1f}%)")
    print("="*70 + "\n")


# ==============================================================================
# 5. メイン実行ブロック
# ==============================================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    CHECKPOINT_FILE = "swing_model_production.pt"
    TARGET_TICKER = "8593.T"
    
    real_market_df = fetch_100pct_real_market_data(ticker=TARGET_TICKER, period="2y")
    labeled_df = apply_dynamic_triple_barrier(real_market_df, max_holding_period=10)
    
    stock_cols = ['stock_ret_1d', 'stock_ret_5d', 'atr_ratio', 'rolling_beta']
    macro_cols = [
        'fut_ret_1d', 'fut_ret_5d', 'fut_vol_ratio', 'stock_fut_spread',
        'usdjpy_ret_1d', 'vix_level', 'jniv_level', 'jniv_ret_1d',
        'dist_call_wall', 'dist_put_wall', 'abn_shock_flag', 'arbitrage_pressure'
    ]
    feature_cols = stock_cols + macro_cols
    
    labeled_df['target_class'] = labeled_df['label'].map({-1: 0, 0: 1, 1: 2})
    
    # 欠損を除外した有効行で分割
    valid_df = labeled_df.dropna(subset=['target_class']).reset_index(drop=True)
    split_idx = int(len(valid_df) * 0.8)
    
    scaler = RobustScaler()
    train_features = scaler.fit_transform(valid_df.iloc[:split_idx][feature_cols].values)
    val_features = scaler.transform(valid_df.iloc[split_idx:][feature_cols].values)
    
    train_dataset = FinancialTimeSeriesDataset(train_features, valid_df.iloc[:split_idx]['target_class'].values, seq_len=20)
    val_dataset = FinancialTimeSeriesDataset(val_features, valid_df.iloc[split_idx:]['target_class'].values, seq_len=20)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    class_counts = np.bincount(train_dataset.y_samples.numpy(), minlength=3)
    weights = torch.tensor(len(train_dataset) / (3.0 * np.maximum(class_counts, 1)), dtype=torch.float32)
    
    model = MacroInformedSwingNet(stock_dim=len(stock_cols), macro_dim=len(macro_cols), hidden_dim=44, num_classes=3)
    train_pipeline(model, train_loader, val_loader, weights, epochs=10, lr=7e-4, device=device)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler': scaler,
        'stock_cols': stock_cols,
        'macro_cols': macro_cols,
        'feature_cols': feature_cols
    }, CHECKPOINT_FILE)
    
    predict_tomorrow_signal(real_market_df, TARGET_TICKER, CHECKPOINT_FILE)