import os
import random
import datetime
import numpy as np
import pandas as pd
import torch

from modules.model_arch import DualStream_GRU_PreLN_Transformer
from modules.macro_features import load_macro_slim5
from modules.stock_features import compute_stock_features
from modules.cross_sectional_features import (
    apply_log_transform,
    compute_cross_sectional_stats,
    valid_cross_section_dates,
    normalize_cross_sectional,
)
from modules.model_inference import load_trained_models_ensemble, predict_probabilities_ensemble

BASE_DIR = r"F:\stockModel"
UNIVERSE_PATH = os.path.join(BASE_DIR, "universe_150_tickers.txt")
ENSEMBLE_SEEDS = [42, 43, 44, 45, 46]
MODEL_PATHS = [os.path.join(BASE_DIR, f"swing_model_v8_ensemble_seed{s}.pt") for s in ENSEMBLE_SEEDS]
CACHE_DIR = os.path.join(BASE_DIR, "data", "cache")
UNIVERSE_BARS_CACHE_PATH = os.path.join(CACHE_DIR, "train_universe_bars.parquet")
MIN_CROSS_SECTION = 20

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

def run_backtest():
    print("=" * 85)
    print("【v8 ランキング損失+横断面正規化+アンサンブル版 閾値探索バックテスト】")
    print("=" * 85)

    missing = [p for p in MODEL_PATHS if not os.path.exists(p)]
    if missing:
        print(f"[!] モデル重みが見つかりません: {missing}")
        return

    models, stock_cols, macro_cols = load_trained_models_ensemble(MODEL_PATHS)
    print(f"[*] アンサンブル {len(models)} モデルを読み込みました (seeds={ENSEMBLE_SEEDS})")

    macro_df = load_macro_slim5()
    macro_feed = macro_df[macro_cols]

    if os.path.exists(UNIVERSE_PATH):
        with open(UNIVERSE_PATH, "r", encoding="utf-8") as f:
            tickers = [line.strip() for line in f if line.strip()]
    else:
        tickers = ["7203.T", "6758.T", "8035.T", "8306.T", "9432.T", "7167.T", "4519.T", "5726.T"]

    m_start = macro_df.index.min() - datetime.timedelta(days=150)  # rolling_beta(90日窓)分の余裕を確保
    holding_period = 10
    seq_len = 10

    if not os.path.exists(UNIVERSE_BARS_CACHE_PATH):
        raise FileNotFoundError(
            f"[!] {UNIVERSE_BARS_CACHE_PATH} が見つかりません。"
            f" 先に build_jquants_cache.py を実行してキャッシュを生成してください。"
        )
    universe_bars = pd.read_parquet(UNIVERSE_BARS_CACHE_PATH)
    universe_bars.index = pd.to_datetime(universe_bars.index).tz_localize(None)

    # --- 1パス目: 銘柄ごとの生特徴量(ログ変換込み)を計算 ---
    print(f"[*] 全 {len(tickers)} 銘柄の特徴量を計算中...")
    per_ticker_df = {}
    for i, t in enumerate(tickers):
        try:
            if t not in universe_bars.columns.get_level_values(0):
                continue
            df = universe_bars[t].loc[universe_bars.index >= m_start].copy()
            df = df.dropna(subset=['Close', 'High', 'Low', 'Volume'])
            if len(df) < seq_len + holding_period + 95 or (df['Volume'] == 0).all():  # +95 = rolling_beta(90日窓)+余裕
                continue

            aligned_nk = macro_df['NK_Ret'].reindex(df.index).fillna(0.0)
            df = compute_stock_features(df, aligned_nk)
            df = df.join(macro_feed, how='inner')
            df = df.dropna(subset=['ATR', 'rolling_beta'] + macro_cols)
            if len(df) < seq_len + holding_period + 10:
                continue

            df = apply_log_transform(df)
            per_ticker_df[t] = df
        except Exception:
            continue
        if (i + 1) % 50 == 0 or (i + 1) == len(tickers):
            print(f"  --> {i + 1}/{len(tickers)} 銘柄 完了")

    # --- 横断面統計 ---
    print("[*] 横断面統計(同日の全銘柄基準)を計算中...")
    cross_mean, cross_std = compute_cross_sectional_stats(per_ticker_df, stock_cols)
    valid_dates = valid_cross_section_dates(per_ticker_df, stock_cols, MIN_CROSS_SECTION)
    print(f"[+] 横断面統計が有効な日数: {len(valid_dates)}")

    # --- 2パス目: 横断面正規化・アンサンブル推論 ---
    print(f"[*] 全 {len(per_ticker_df)} 銘柄の推論検証中...")
    cached_records = []

    for i, (t, df) in enumerate(per_ticker_df.items()):
        try:
            df = df.loc[df.index.isin(valid_dates)].copy()
            if len(df) < seq_len + holding_period + 10:
                continue

            norm_s = normalize_cross_sectional(df, cross_mean, cross_std, stock_cols)

            n_samples = len(df)
            split_idx = int(n_samples * 0.75)
            vals_s_norm = norm_s.values
            vals_m = df[macro_cols].values
            closes, highs, lows, atrs = df['Close'].values, df['High'].values, df['Low'].values, df['ATR'].values

            indices = list(range(split_idx, n_samples - holding_period))
            if not indices:
                continue

            for idx in indices:
                w_s = vals_s_norm[idx - seq_len + 1: idx + 1].copy()
                if np.isnan(w_s).any():
                    continue
                w_m = vals_m[idx - seq_len + 1: idx + 1].copy()

                p_win, p_stop, _ = predict_probabilities_ensemble(models, w_s, w_m)

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

        if (i + 1) % 50 == 0 or (i + 1) == len(per_ticker_df):
            print(f"  --> {i + 1}/{len(per_ticker_df)} 銘柄 完了")

    df_all = pd.DataFrame(cached_records)
    if df_all.empty:
        print("[-] 有効な検証データが取得できませんでした。")
        return

    print(f"\n[★] 推論完了: 総サンプル数 = {len(df_all)}")
    print(f"  --> p_win 分布: Min={df_all['p_win'].min():.3f} | Median={df_all['p_win'].median():.3f} | Max={df_all['p_win'].max():.3f}")

    print("\n" + "=" * 85)
    print(f"{'買確信度 (th)':<12} | {'件数':<6} | {'勝率 (%)':<8} | {'利確到達率':<10} | {'損切率':<8} | {'損益比':<6} | {'PF':<6}")
    print("=" * 85)

    p_lo, p_hi = df_all['p_win'].quantile(0.02), df_all['p_win'].quantile(0.98)
    test_ths = np.linspace(p_lo, p_hi, 12)
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
