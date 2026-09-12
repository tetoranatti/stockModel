import os
import random
import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn

# =============================================================================
# 設定・乱数シード固定
# =============================================================================
BASE_DIR = r"F:\stockModel"
DB_PATH = os.path.join(BASE_DIR, "jpx_daily_features_db.csv")
UNIVERSE_PATH = os.path.join(BASE_DIR, "universe_150_tickers.txt")
MODEL_PATH = os.path.join(BASE_DIR, "swing_model_v6_crossattn_universe_v5.pt")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# =============================================================================
# 1. モデル定義 (v6 安定版 Cross-Attention)
# =============================================================================
class DecayPooling(nn.Module):
    def __init__(self, seq_len=10):
        super().__init__()
        weights = np.exp(np.linspace(-1.5, 0.0, seq_len))
        weights = weights / weights.sum()
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32).unsqueeze(0).unsqueeze(-1))

    def forward(self, x):
        return torch.sum(x * self.weights, dim=1)

class CrossAttentionBlock(nn.Module):
    def __init__(self, hidden_dim=24, num_heads=2, dropout=0.3):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key_value):
        attn_out, _ = self.mha(query=query, key=key_value, value=key_value)
        return self.norm(query + self.dropout(attn_out))

class DualStreamGRU_v6_Slim5(nn.Module):
    def __init__(self, stock_dim=5, macro_dim=5, hidden_dim=24, num_heads=2, num_classes=3):
        super().__init__()
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

# =============================================================================
# 2. マクロ環境データ構築
# =============================================================================
def load_macro_slim5(db_path=DB_PATH):
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

    # 過去リークを防ぐ ffill 優先結合
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
# 3. バックテスト本体
# =============================================================================
def run_multi_threshold_backtest():
    print("=" * 85)
    print("【v6 スイングモデル ターゲット完全同期＆閾値探索バックテスト】")
    print("=" * 85)

    if not os.path.exists(MODEL_PATH):
        print(f"[!] モデル重みが見つかりません: {MODEL_PATH}")
        return

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    stock_cols = checkpoint['stock_cols']
    macro_cols = checkpoint['macro_cols']

    model = DualStreamGRU_v6_Slim5(
        stock_dim=len(stock_cols),
        macro_dim=len(macro_cols),
        hidden_dim=checkpoint.get('hidden_dim', 24),
        num_heads=2,
        num_classes=3
    ).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    macro_df = load_macro_slim5()
    macro_feed = macro_df[macro_cols]

    if os.path.exists(UNIVERSE_PATH):
        with open(UNIVERSE_PATH, "r", encoding="utf-8") as f:
            tickers = [line.strip() for line in f if line.strip()]
    else:
        tickers = ["7203.T", "6758.T", "8035.T", "8306.T", "9432.T", "7167.T", "4519.T", "5726.T"]

    m_start = (macro_df.index.min() - datetime.timedelta(days=40)).strftime("%Y-%m-%d")
    holding_period = 10
    seq_len = 10

    print(f"[*] 全 {len(tickers)} 銘柄のテスト区間（時系列スプリット後25%）推論キャッシュ作成中...")
    cached_records = []

    for i, t in enumerate(tickers):
        try:
            df = yf.download(t, start=m_start, interval="1d", progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df = df.dropna(subset=['Close', 'High', 'Low', 'Volume'])

            # データ欠損・取引停止ガード
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

            df = df.join(macro_feed, how='inner')
            df = df.dropna(subset=['ATR', 'rolling_beta'] + macro_cols)

            n_samples = len(df)
            split_idx = int(n_samples * 0.75)
            vals_s = df[stock_cols].values
            vals_m = df[macro_cols].values

            closes = df['Close'].values
            highs = df['High'].values
            lows = df['Low'].values
            atrs = df['ATR'].values

            for idx in range(split_idx, n_samples - holding_period):
                w_s = vals_s[idx - seq_len + 1 : idx + 1].copy()
                w_m = vals_m[idx - seq_len + 1 : idx + 1].copy()
                w_s = (w_s - w_s.mean(axis=0)) / (w_s.std(axis=0) + 1e-7)

                t_s = torch.tensor(w_s, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                t_m = torch.tensor(w_m, dtype=torch.float32).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    probs = torch.softmax(model(t_s, t_m), dim=-1).squeeze(0).cpu().numpy()

                p_stop, _, p_win = float(probs[0]), float(probs[1]), float(probs[2])

                entry_p = closes[idx]
                upper_p = entry_p + (2.0 * atrs[idx])
                lower_p = entry_p - (1.0 * atrs[idx])

                exit_p = None
                exit_reason = None
                h_days = 0

                for h in range(1, holding_period + 1):
                    bar_h = highs[idx + h]
                    bar_l = lows[idx + h]
                    h_days = h

                    if bar_l <= lower_p and bar_h >= upper_p:
                        exit_p = lower_p
                        exit_reason = "STOP_LOSS (Both)"
                        break
                    elif bar_h >= upper_p:
                        exit_p = upper_p
                        exit_reason = "TAKE_PROFIT"
                        break
                    elif bar_l <= lower_p:
                        exit_p = lower_p
                        exit_reason = "STOP_LOSS"
                        break

                if exit_p is None:
                    exit_p = closes[idx + holding_period]
                    exit_reason = "TIME_OUT"

                ret_pct = (exit_p - entry_p) / entry_p
                cached_records.append({
                    'ticker': t,
                    'date': df.index[idx],
                    'p_win': p_win,
                    'p_stop': p_stop,
                    'ret_pct': ret_pct,
                    'exit_reason': exit_reason,
                    'holding_days': h_days
                })
        except Exception:
            continue

        if (i + 1) % 50 == 0 or (i + 1) == len(tickers):
            print(f"  --> {i + 1}/{len(tickers)} 銘柄 完了")

    df_all = pd.DataFrame(cached_records)
    if df_all.empty:
        print("[-] 有効な検証データが取得できませんでした。")
        return

    print(f"\n[★] 推論キャッシュ生成完了: 総サンプル数 = {len(df_all)}")
    print(f"  --> p_win 分布: Min={df_all['p_win'].min():.3f} | Median={df_all['p_win'].median():.3f} | Max={df_all['p_win'].max():.3f}")

    # =========================================================================
    # 閾値グリッド集計テーブル出力
    # =========================================================================
    print("\n" + "=" * 85)
    print(f"{'買確信度 (th)':<12} | {'件数':<6} | {'勝率 (%)':<8} | {'利確到達率':<10} | {'損切率':<8} | {'損益比':<6} | {'PF':<6}")
    print("=" * 85)

    thresholds = [0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.40]
    for th in thresholds:
        sub = df_all[(df_all['p_win'] >= th) & (df_all['p_win'] > df_all['p_stop'])].copy()
        if len(sub) == 0:
            print(f"{th:<12.2f} | {0:<6} | {'-':<8} | {'-':<10} | {'-':<8} | {'-':<6} | {'-':<6}")
            continue

        wins = sub[sub['ret_pct'] > 0]
        losses = sub[sub['ret_pct'] < 0]
        win_rate = len(wins) / len(sub) * 100.0
        tp_rate = (sub['exit_reason'] == 'TAKE_PROFIT').mean() * 100.0
        sl_rate = sub['exit_reason'].str.startswith('STOP_LOSS').mean() * 100.0

        avg_w = wins['ret_pct'].mean() if len(wins) > 0 else 0.0
        avg_l = abs(losses['ret_pct'].mean()) if len(losses) > 0 else 0.0
        rr = avg_w / avg_l if avg_l > 0 else 0.0
        pf = wins['ret_pct'].sum() / abs(losses['ret_pct'].sum()) if len(losses) > 0 and losses['ret_pct'].sum() != 0 else float('inf')

        print(f"{th:<12.2f} | {len(sub):<6d} | {win_rate:<8.1f} | {tp_rate:<10.1f}% | {sl_rate:<8.1f}% | {rr:<6.2f} | {pf:<6.2f}")

    print("=" * 85)

    # 推奨閾値 (0.36) のドローダウン・資産曲線詳細
    target_th = 0.36
    sub_best = df_all[(df_all['p_win'] >= target_th) & (df_all['p_win'] > df_all['p_stop'])].copy()
    if not sub_best.empty:
        sub_best = sub_best.sort_values('date').reset_index(drop=True)
        # 1銘柄あたり資金の10%投入を想定した累積資産推移
        sub_best['nav'] = (1.0 + sub_best['ret_pct'] * 0.10).cumprod()
        peak = sub_best['nav'].cummax()
        drawdown = (sub_best['nav'] - peak) / peak
        max_dd = drawdown.min() * 100.0

        print(f"\n[★] 推奨閾値 ({target_th:.2f}) 実運用シミュレーション (1トレード10%均等配分):")
        print(f"  - 最終累積資産倍率: {sub_best['nav'].iloc[-1]:.2f} 倍")
        print(f"  - 最大ドローダウン  : {max_dd:.2f} %")
        print(f"  - 平均保有日数      : {sub_best['holding_days'].mean():.1f} 営業日")

if __name__ == "__main__":
    run_multi_threshold_backtest()