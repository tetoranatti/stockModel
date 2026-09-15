import os
import random
import datetime
import numpy as np
import pandas as pd
import torch

from modules.model_arch import DualStream_GRU_PreLN_Transformer

BASE_DIR = r"F:\stockModel"
DB_PATH = os.path.join(BASE_DIR, "jpx_daily_features_db.csv")
UNIVERSE_PATH = os.path.join(BASE_DIR, "universe_150_tickers.txt")
MODEL_PATH = os.path.join(BASE_DIR, "swing_model_v8_timeout_refined.pt")
CACHE_DIR = os.path.join(BASE_DIR, "data", "cache")
UNIVERSE_BARS_CACHE_PATH = os.path.join(CACHE_DIR, "train_universe_bars.parquet")
NK225_UNDERLYING_CACHE_PATH = os.path.join(CACHE_DIR, "train_nk225_underlying.parquet")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

def load_macro_slim5(db_path=DB_PATH):
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"[!] {db_path} が見つかりません。")
    if not os.path.exists(NK225_UNDERLYING_CACHE_PATH):
        raise FileNotFoundError(
            f"[!] {NK225_UNDERLYING_CACHE_PATH} が見つかりません。"
            f" 先に build_jquants_cache.py を実行してキャッシュを生成してください。"
        )

    jpx_db = pd.read_csv(db_path, index_col=0, parse_dates=True)
    jpx_db.index = pd.to_datetime(jpx_db.index).tz_localize(None)

    n225 = pd.read_parquet(NK225_UNDERLYING_CACHE_PATH)
    n225.index = pd.to_datetime(n225.index).tz_localize(None)

    macro_df = pd.DataFrame(index=n225.index)
    macro_df['NK_Close'] = n225['NK_Close']
    macro_df['NK_Ret'] = macro_df['NK_Close'].pct_change(fill_method=None).fillna(0.0)

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

def run_backtest():
    print("=" * 85)
    print("【v8 TIME_OUT改訂版 閾値探索バックテスト】")
    print("=" * 85)

    if not os.path.exists(MODEL_PATH):
        print(f"[!] モデル重みが見つかりません: {MODEL_PATH}")
        return

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    stock_cols = checkpoint['stock_cols']
    macro_cols = checkpoint['macro_cols']
    hidden_dim = checkpoint.get('hidden_dim', 20)
    num_heads = checkpoint.get('num_heads', 1)
    dropout = checkpoint.get('dropout', 0.2)

    model = DualStream_GRU_PreLN_Transformer(
        stock_dim=len(stock_cols),
        macro_dim=len(macro_cols),
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_classes=3,
        dropout=dropout
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

    m_start = macro_df.index.min() - datetime.timedelta(days=40)
    holding_period = 10
    seq_len = 10

    if not os.path.exists(UNIVERSE_BARS_CACHE_PATH):
        raise FileNotFoundError(
            f"[!] {UNIVERSE_BARS_CACHE_PATH} が見つかりません。"
            f" 先に build_jquants_cache.py を実行してキャッシュを生成してください。"
        )
    universe_bars = pd.read_parquet(UNIVERSE_BARS_CACHE_PATH)
    universe_bars.index = pd.to_datetime(universe_bars.index).tz_localize(None)

    print(f"[*] 全 {len(tickers)} 銘柄の推論検証中...")
    cached_records = []

    for i, t in enumerate(tickers):
        try:
            if t not in universe_bars.columns.get_level_values(0):
                continue
            df = universe_bars[t].loc[universe_bars.index >= m_start].copy()
            df = df.dropna(subset=['Close', 'High', 'Low', 'Volume'])

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

            aligned_nk = macro_df['NK_Ret'].reindex(df.index).fillna(0.0)
            cov = df['stock_ret_1d'].rolling(20).cov(aligned_nk)
            var = aligned_nk.rolling(20).var()
            df['rolling_beta'] = (cov / (var + 1e-7)).fillna(1.0)

            df = df.join(macro_feed, how='inner')
            df = df.dropna(subset=['ATR', 'rolling_beta'] + macro_cols)

            n_samples = len(df)
            split_idx = int(n_samples * 0.75)
            vals_s, vals_m = df[stock_cols].values, df[macro_cols].values
            closes, highs, lows, atrs = df['Close'].values, df['High'].values, df['Low'].values, df['ATR'].values

            indices = list(range(split_idx, n_samples - holding_period))
            if not indices:
                continue

            batch_s, batch_m = [], []
            for idx in indices:
                w_s = vals_s[idx - seq_len + 1 : idx + 1].copy()
                w_m = vals_m[idx - seq_len + 1 : idx + 1].copy()
                w_s = (w_s - w_s.mean(axis=0)) / (w_s.std(axis=0) + 1e-7)
                batch_s.append(w_s)
                batch_m.append(w_m)

            t_s = torch.tensor(np.array(batch_s), dtype=torch.float32).to(DEVICE)
            t_m = torch.tensor(np.array(batch_m), dtype=torch.float32).to(DEVICE)

            with torch.no_grad():
                probs_all = torch.softmax(model(t_s, t_m), dim=-1).cpu().numpy()

            for k, idx in enumerate(indices):
                probs = probs_all[k]
                p_stop, _, p_win = float(probs[0]), float(probs[1]), float(probs[2])

                entry_p = closes[idx]
                upper_p = entry_p + (2.0 * atrs[idx])
                lower_p = entry_p - (1.0 * atrs[idx])

                exit_p, exit_reason = None, None
                h_days = 0

                for h in range(1, holding_period + 1):
                    bar_h, bar_l = highs[idx + h], lows[idx + h]
                    h_days = h

                    if bar_l <= lower_p and bar_h >= upper_p:
                        exit_p, exit_reason = lower_p, "STOP_LOSS (Both)"
                        break
                    elif bar_h >= upper_p:
                        exit_p, exit_reason = upper_p, "TAKE_PROFIT"
                        break
                    elif bar_l <= lower_p:
                        exit_p, exit_reason = lower_p, "STOP_LOSS"
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

    print(f"\n[★] 推論完了: 総サンプル数 = {len(df_all)}")
    print(f"  --> p_win 分布: Min={df_all['p_win'].min():.3f} | Median={df_all['p_win'].median():.3f} | Max={df_all['p_win'].max():.3f}")

    print("\n" + "=" * 85)
    print(f"{'買確信度 (th)':<12} | {'件数':<6} | {'勝率 (%)':<8} | {'利確到達率':<10} | {'損切率':<8} | {'損益比':<6} | {'PF':<6}")
    print("=" * 85)

    test_ths = [0.33, 0.335, 0.34, 0.345, 0.35, 0.355, 0.36, 0.37, 0.38]
    for th in test_ths:
        sub = df_all[(df_all['p_win'] >= th) & (df_all['p_win'] > df_all['p_stop'])].copy()
        if len(sub) == 0:
            print(f"{th:<12.3f} | {0:<6} | {'-':<8} | {'-':<10} | {'-':<8} | {'-':<6} | {'-':<6}")
            continue

        wins, losses = sub[sub['ret_pct'] > 0], sub[sub['ret_pct'] < 0]
        win_rate = len(wins) / len(sub) * 100.0
        tp_rate = (sub['exit_reason'] == 'TAKE_PROFIT').mean() * 100.0
        sl_rate = sub['exit_reason'].str.startswith('STOP_LOSS').mean() * 100.0

        avg_w = wins['ret_pct'].mean() if len(wins) > 0 else 0.0
        avg_l = abs(losses['ret_pct'].mean()) if len(losses) > 0 else 0.0
        rr = avg_w / avg_l if avg_l > 0 else 0.0
        pf = wins['ret_pct'].sum() / abs(losses['ret_pct'].sum()) if len(losses) > 0 and losses['ret_pct'].sum() != 0 else float('inf')

        print(f"{th:<12.3f} | {len(sub):<6d} | {win_rate:<8.1f} | {tp_rate:<10.1f}% | {sl_rate:<8.1f}% | {rr:<6.2f} | {pf:<6.2f}")

    print("=" * 85)

if __name__ == "__main__":
    run_backtest()