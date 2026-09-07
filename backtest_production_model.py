import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import yfinance as yf

# =============================================================================
# 1. 本番モデル定義 (チェックポイントと完全一致)
# =============================================================================
class DualStreamGRU(nn.Module):
    def __init__(self, stock_dim=4, macro_dim=12, hidden_dim=44, num_classes=3):
        super(DualStreamGRU, self).__init__()
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
# 2. 特徴量生成関数（本番学習時の特徴量を完全再現）
# =============================================================================
def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df['High'] - df['Low']
    high_cp = (df['High'] - df['Close'].shift(1)).abs()
    low_cp = (df['Low'] - df['Close'].shift(1)).abs()
    tr = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def load_data_and_features(ticker: str, start_date: str = "2024-01-01"):
    """銘柄およびマクロ特徴量の一括取得・生成"""
    df = yf.download(ticker, start=start_date, interval="1d", progress=False)
    if df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # ATR計算
    df['ATR'] = calculate_atr(df, 14)

    # マクロ指標 (^N225, USDJPY=X, ^VIX)
    n225 = yf.download("^N225", start=start_date, interval="1d", progress=False)
    if isinstance(n225.columns, pd.MultiIndex): n225.columns = n225.columns.get_level_values(0)
    fx = yf.download("USDJPY=X", start=start_date, interval="1d", progress=False)
    if isinstance(fx.columns, pd.MultiIndex): fx.columns = fx.columns.get_level_values(0)
    vix = yf.download("^VIX", start=start_date, interval="1d", progress=False)
    if isinstance(vix.columns, pd.MultiIndex): vix.columns = vix.columns.get_level_values(0)

    df['NK_Ret'] = n225['Close'].pct_change().reindex(df.index).fillna(0.0)
    df['FX_Ret'] = fx['Close'].pct_change().reindex(df.index).fillna(0.0)
    df['VIX_Close'] = vix['Close'].reindex(df.index).ffill().bfill()

    # -------------------------------------------------------------
    # 本番モデルで要求される4つの銘柄特徴量
    # -------------------------------------------------------------
    df['stock_ret_1d'] = df['Close'].pct_change(1).fillna(0.0)
    df['stock_ret_5d'] = df['Close'].pct_change(5).fillna(0.0)
    df['atr_ratio'] = (df['ATR'] / (df['Close'] + 1e-7)).fillna(0.0)
    
    # 日経平均に対する20日ローリング・ベータ値
    cov = df['stock_ret_1d'].rolling(20).cov(df['NK_Ret'])
    var = df['NK_Ret'].rolling(20).var()
    df['rolling_beta'] = (cov / (var + 1e-7)).fillna(1.0)

    # 従来の指標（フォールバック用）
    df['SMA5'] = df['Close'].rolling(5).mean()
    df['SMA25'] = df['Close'].rolling(25).mean()
    df['Dev25'] = (df['Close'] - df['SMA25']) / (df['SMA25'] + 1e-7)
    df['Volume_Ratio'] = df['Volume'] / (df['Volume'].rolling(20).mean() + 1e-7)

    # JPX需給データベースが存在すればマージ
    if os.path.exists("jpx_daily_features_db.csv"):
        jpx_db = pd.read_csv("jpx_daily_features_db.csv", index_col=0, parse_dates=True)
        df = df.join(jpx_db, how='left')

    df = df.dropna(subset=['ATR', 'rolling_beta']).copy()
    return df

# =============================================================================
# 3. バックテスト実行エンジン
# =============================================================================
def run_model_backtest(ticker: str, model_path: str = "swing_model_production.pt", fee_rate: float = 0.001):
    print(f"\n{'='*60}")
    print(f"[*] 本番モデル・バックテスト開始: {ticker}")
    print(f"{'='*60}")

    df = load_data_and_features(ticker)
    if len(df) < 50:
        print("[-] データ数が不足しています。")
        return None

    if not os.path.exists(model_path):
        print(f"[-] エラー: モデルファイル '{model_path}' が見つかりません。")
        return None

    # チェックポイントの読み込み
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        stock_cols = checkpoint.get("stock_cols", ['stock_ret_1d', 'stock_ret_5d', 'atr_ratio', 'rolling_beta'])
        macro_cols = checkpoint.get("macro_cols", None)
    else:
        state_dict = checkpoint
        stock_cols = ['stock_ret_1d', 'stock_ret_5d', 'atr_ratio', 'rolling_beta']
        macro_cols = None

    # 次元数の取得
    stock_dim = state_dict['stock_gru.weight_ih_l0'].shape[1]
    macro_dim = state_dict['macro_gru.weight_ih_l0'].shape[1]
    hidden_dim = state_dict['stock_gru.weight_hh_l0'].shape[1]

    # マクロ列の安全なセット
    if macro_cols is None or len(macro_cols) != macro_dim:
        macro_cols = [
            'NK_Ret', 'FX_Ret', 'VIX_Close',
            'call_oi_wall', 'put_oi_wall', 'abn_net_flow', 'arbitrage_balance',
            'gamma_flip', 'call_volume', 'put_volume', 'pcr_ratio', 'us10y_yield'
        ]

    # マクロ列の欠損補完
    for col in macro_cols:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = df[col].fillna(0.0)

    # 銘柄特徴量列の存在確認と補完
    for col in stock_cols:
        if col not in df.columns:
            df[col] = 0.0

    # モデル構築と重み適用
    model = DualStreamGRU(stock_dim=stock_dim, macro_dim=macro_dim, hidden_dim=hidden_dim, num_classes=3)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"[+] 学習済みモデル '{model_path}' を完全適合でロードしました。")

    seq_len = 10
    trades = []
    holding = False
    entry_price = 0.0
    upper_barrier = 0.0
    lower_barrier = 0.0
    holding_days = 0
    entry_date = None

    dates = df.index
    n = len(df)
    profit_probs = []

    for i in range(seq_len, n - 10):
        current_date = dates[i]

        # ポジション未保有時：モデル推論
        if not holding:
            seq_stock = df[stock_cols].iloc[i - seq_len + 1 : i + 1].values
            seq_macro = df[macro_cols].iloc[i - seq_len + 1 : i + 1].values

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
            profit_probs.append(p_profit)

            # バックテスト用判定閾値（36%以上 かつ 損切確率より優位）
            is_buy = (p_profit >= 0.36) and (p_profit > p_loss * 1.1)

            if is_buy:
                holding = True
                holding_days = 0
                entry_date = dates[i + 1]
                entry_price = float(df['Open'].iloc[i + 1] * (1.0 + fee_rate))
                atr_val = float(df['ATR'].iloc[i])
                upper_barrier = entry_price + (2.0 * atr_val)
                lower_barrier = entry_price - (1.0 * atr_val)
                continue

        # ポジション保有中：トリプルバリア判定
        if holding:
            holding_days += 1
            high_p = float(df['High'].iloc[i])
            low_p = float(df['Low'].iloc[i])
            close_p = float(df['Close'].iloc[i])

            hit_profit = high_p >= upper_barrier
            hit_loss = low_p <= lower_barrier

            exit_price = None
            exit_type = None

            if hit_profit and hit_loss:
                exit_price = lower_barrier * (1.0 - fee_rate)
                exit_type = "LOSS (同日接触)"
            elif hit_profit:
                exit_price = upper_barrier * (1.0 - fee_rate)
                exit_type = "PROFIT (利確)"
            elif hit_loss:
                exit_price = lower_barrier * (1.0 - fee_rate)
                exit_type = "LOSS (損切)"
            elif holding_days >= 10:
                exit_price = close_p * (1.0 - fee_rate)
                exit_type = "TIMEOUT (期限)"

            if exit_type:
                ret = (exit_price - entry_price) / entry_price
                trades.append({
                    "entry_date": entry_date.strftime('%Y-%m-%d'),
                    "exit_date": current_date.strftime('%Y-%m-%d'),
                    "entry_price": round(entry_price, 1),
                    "exit_price": round(exit_price, 1),
                    "holding_days": holding_days,
                    "exit_type": exit_type,
                    "return_pct": round(ret * 100, 2),
                    "return_factor": 1.0 + ret
                })
                holding = False

    avg_p = np.mean(profit_probs) if profit_probs else 0.0
    max_p = np.max(profit_probs) if profit_probs else 0.0
    print(f"[*] モデル推論状況: 平均利確確率 = {avg_p*100:.1f}%, 最大利確確率 = {max_p*100:.1f}%")

    res_df = pd.DataFrame(trades)
    if len(res_df) == 0:
        print("[-] 該当期間中にモデルのBUY判定条件を満たすシグナルは発生しませんでした。")
        return None

    # 集計
    total_trades = len(res_df)
    win_trades = res_df[res_df['return_pct'] > 0]
    loss_trades = res_df[res_df['return_pct'] <= 0]
    win_rate = (len(win_trades) / total_trades) * 100
    total_gain = win_trades['return_pct'].sum()
    total_loss = abs(loss_trades['return_pct'].sum())
    profit_factor = (total_gain / total_loss) if total_loss > 0 else np.nan
    cum_return = (res_df['return_factor'].prod() - 1.0) * 100

    print(f"総トレード数       : {total_trades} 回")
    print(f"勝率 (利確/全取引) : {win_rate:.1f} %")
    print(f"プロフィットファクター: {profit_factor:.2f}")
    print(f"通算累積リターン   : {cum_return:+.2f} %")
    print(f"平均保有日数       : {res_df['holding_days'].mean():.1f} 営業日")
    print("\n[直近トレード履歴]")
    print(res_df[['entry_date', 'exit_date', 'exit_type', 'holding_days', 'return_pct']].tail(5).to_string(index=False))
    return res_df

# =============================================================================
# 4. 実行部
# =============================================================================
if __name__ == "__main__":
    targets = {
        "川崎重工業": "7012.T",
        "しまむら": "8227.T",
        "旭化成": "3407.T",
        "三菱HCキャピタル": "8593.T"
    }
    for name, sym in targets.items():
        run_model_backtest(sym)