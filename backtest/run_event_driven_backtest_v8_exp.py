import os
import random
import datetime
import numpy as np
import pandas as pd
import torch

# backtest/配下からでもmodules/を解決できるようにプロジェクトルートをsys.pathへ追加
import sys
sys.path.insert(0, r"F:\stockModel")
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
# バックテスト専用の拡張ユニバース(売買代金10億円以上、本番ライブスクリーニングと同じ
# 母集団規模=564銘柄)。学習用のuniverse_150_tickers.txtとは別ファイルにし、
# 学習・地合い危険度モデル・空売り機会モデルには影響させない。
UNIVERSE_PATH_BACKTEST = os.path.join(BASE_DIR, "universe_backtest_tickers.txt")
ENSEMBLE_SEEDS = [42, 43, 44, 45, 46]
MODEL_PATHS = [os.path.join(BASE_DIR, f"swing_model_v8_ensemble_seed{s}.pt") for s in ENSEMBLE_SEEDS]
MODEL_PATHS_TOPIX = [os.path.join(BASE_DIR, f"swing_model_v8_topix_seed{s}.pt") for s in ENSEMBLE_SEEDS]
CACHE_DIR = os.path.join(BASE_DIR, "data", "cache")
UNIVERSE_BARS_CACHE_PATH = os.path.join(CACHE_DIR, "train_universe_bars.parquet")
MIN_CROSS_SECTION = 20
MAX_CONCURRENT_POSITIONS = 20  # 資金曲線シミュレーション用の同時保有上限(均等配分)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

def simulate_equity_curve(sub, max_concurrent=MAX_CONCURRENT_POSITIONS):
    """選定トレードを時系列順に処理し、同時保有上限max_concurrentで資金を均等配分した
    エクイティカーブを返す。上限を超える新規シグナルは資金枠が空くまで見送る(現実的な
    資金制約を簡易再現)。トレード間の重複を考慮しない単純合算のPF/勝率とは異なり、
    実際の運用に近い資金推移とドローダウンを見るための補助指標。"""
    events = []
    for tid, row in enumerate(sub.itertuples(index=False)):
        events.append((row.date, 1, tid, row.ret_pct))       # entry: 同日ならexitの後
        events.append((row.exit_date, 0, tid, row.ret_pct))  # exit: 同日ならentryより先に処理し枠を空ける
    events.sort(key=lambda e: (e[0], e[1]))

    equity = 1.0
    open_slots = {}
    equity_curve = [equity]
    for _, kind, tid, ret_pct in events:
        if kind == 0:  # exit
            size = open_slots.pop(tid, None)
            if size is not None:
                equity += size * ret_pct
                equity_curve.append(equity)
        else:  # entry
            if len(open_slots) < max_concurrent:
                open_slots[tid] = equity / max_concurrent
    return equity_curve

def compute_max_drawdown(equity_curve):
    peak = -float('inf')
    max_dd = 0.0
    for eq in equity_curve:
        peak = max(peak, eq)
        if peak > 0:
            max_dd = min(max_dd, (eq - peak) / peak)
    return max_dd

def run_backtest(return_source="nk225", use_expanded_universe=False):
    print("=" * 85)
    print(f"【v8 ランキング損失+横断面正規化+アンサンブル版 閾値探索バックテスト "
          f"(指数ソース: {return_source}, 拡張ユニバース: {use_expanded_universe})】")
    print("=" * 85)

    model_paths = MODEL_PATHS_TOPIX if return_source == "topix" else MODEL_PATHS
    missing = [p for p in model_paths if not os.path.exists(p)]
    if missing:
        print(f"[!] モデル重みが見つかりません: {missing}")
        return

    models, stock_cols, macro_cols = load_trained_models_ensemble(model_paths)
    print(f"[*] アンサンブル {len(models)} モデルを読み込みました (seeds={ENSEMBLE_SEEDS})")

    macro_df = load_macro_slim5(return_source=return_source)
    macro_feed = macro_df[macro_cols]

    universe_path = UNIVERSE_PATH_BACKTEST if use_expanded_universe else UNIVERSE_PATH
    if os.path.exists(universe_path):
        with open(universe_path, "r", encoding="utf-8") as f:
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
            betas = df['rolling_beta'].values

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
                    'exit_date': df.index[idx + h_days],
                    'p_win': p_win,
                    'p_stop': p_stop,
                    'beta': betas[idx],
                    'ret_pct': ret_pct,
                    'exit_reason': exit_reason,
                    'holding_days': h_days,
                    # 想定資金での実株数シミュレーション用(UIのupdatePositionSize()と同じ計算に使う)
                    'entry_price': float(entry_p),
                    'atr': float(atrs[idx]),
                })
        except Exception:
            continue

        if (i + 1) % 50 == 0 or (i + 1) == len(per_ticker_df):
            print(f"  --> {i + 1}/{len(per_ticker_df)} 銘柄 完了")

    df_all = pd.DataFrame(cached_records)
    if df_all.empty:
        print("[-] 有効な検証データが取得できませんでした。")
        return

    # トレード単位の生データをキャッシュ(空売りモデル等、別の切り口で再分析する際に
    # PF推論(最も重い処理)をやり直さずに済むようにする)
    universe_suffix = "_universe564" if use_expanded_universe else ""
    trades_cache_path = os.path.join(BASE_DIR, "data", "cache", f"backtest_trades_{return_source}{universe_suffix}.parquet")
    df_all.to_parquet(trades_cache_path)
    print(f"[+] トレード単位データをキャッシュ: {trades_cache_path} ({len(df_all)}件)")

    print(f"\n[★] 推論完了: 総サンプル数 = {len(df_all)}")
    print(f"  --> p_win 分布: Min={df_all['p_win'].min():.3f} | Median={df_all['p_win'].median():.3f} | Max={df_all['p_win'].max():.3f}")
    print(f"  --> p_stop分布: Min={df_all['p_stop'].min():.3f} | Median={df_all['p_stop'].median():.3f} | Max={df_all['p_stop'].max():.3f}")

    print("\n" + "=" * 100)
    print(f"{'買確信度 (th)':<12} | {'件数':<6} | {'勝率 (%)':<8} | {'利確到達率':<10} | {'損切率':<8} | {'損益比':<6} | {'PF':<6} | {'MaxDD':<7}")
    print(f"  (MaxDD: 同時保有上限{MAX_CONCURRENT_POSITIONS}銘柄・均等配分の資金曲線シミュレーションでの最大ドローダウン)")
    print("=" * 100)

    p_lo, p_hi = df_all['p_win'].quantile(0.02), df_all['p_win'].quantile(0.98)
    test_ths = np.linspace(p_lo, p_hi, 12)
    for th in test_ths:
        sub = df_all[(df_all['p_win'] >= th) & (df_all['p_win'] > df_all['p_stop'])].copy()
        if len(sub) == 0:
            print(f"{th:<12.3f} | {0:<6} | {'-':<8} | {'-':<10} | {'-':<8} | {'-':<6} | {'-':<6} | {'-':<7}")
            continue

        wins, losses = sub[sub['ret_pct'] > 0], sub[sub['ret_pct'] < 0]
        win_rate = len(wins) / len(sub) * 100.0
        tp_rate = (sub['exit_reason'] == 'TAKE_PROFIT').mean() * 100.0
        sl_rate = sub['exit_reason'].str.startswith('STOP_LOSS').mean() * 100.0

        avg_w = wins['ret_pct'].mean() if len(wins) > 0 else 0.0
        avg_l = abs(losses['ret_pct'].mean()) if len(losses) > 0 else 0.0
        rr = avg_w / avg_l if avg_l > 0 else 0.0
        pf = wins['ret_pct'].sum() / abs(losses['ret_pct'].sum()) if len(losses) > 0 and losses['ret_pct'].sum() != 0 else float('inf')

        equity_curve = simulate_equity_curve(sub)
        max_dd = compute_max_drawdown(equity_curve)

        print(f"{th:<12.3f} | {len(sub):<6d} | {win_rate:<8.1f} | {tp_rate:<10.1f}% | {sl_rate:<8.1f}% | {rr:<6.2f} | {pf:<6.2f} | {max_dd*100:<6.1f}%")

    print("=" * 100)

    # --- BUY と STRONG BUY の成績差(排他区間) ---
    # 上の表は「p_win >= th」の累積集計なので、STRONG BUY銘柄がBUY銘柄の集計にも
    # 混ざってしまい、両者の純粋な成績差が見えない。ここではrisk_manager.pyの
    # evaluate_screening_gate()と同じ閾値で排他的に区切って比較する
    # (vol_ratio>=0.85のBUY追加条件はここでは未適用のため実運用よりわずかに緩い)。
    print(f"\n{'='*100}")
    print("【アクション階層別 成績比較 (STRONG BUY vs BUY vs WATCH, 排他区間)】")
    print("  区分: STRONG BUY: p_win>=0.700 | BUY: 0.590<=p_win<0.700 | WATCH: 0.470<=p_win<0.590 (共通: p_win>p_stop)")
    print("=" * 100)
    print(f"{'階層':<14} | {'件数':<6} | {'勝率 (%)':<8} | {'利確到達率':<10} | {'損切率':<8} | {'損益比':<6} | {'PF':<6} | {'平均ret_pct':<12} | {'MaxDD':<7}")
    print("-" * 110)
    for label, lo, hi in [("STRONG BUY", 0.700, float('inf')), ("BUY", 0.590, 0.700), ("WATCH", 0.470, 0.590)]:
        sub = df_all[(df_all['p_win'] >= lo) & (df_all['p_win'] < hi) & (df_all['p_win'] > df_all['p_stop'])].copy()
        if len(sub) == 0:
            print(f"{label:<14} | {0:<6} | {'-':<8} | {'-':<10} | {'-':<8} | {'-':<6} | {'-':<6} | {'-':<12} | {'-':<7}")
            continue
        wins, losses = sub[sub['ret_pct'] > 0], sub[sub['ret_pct'] < 0]
        win_rate = len(wins) / len(sub) * 100.0
        tp_rate = (sub['exit_reason'] == 'TAKE_PROFIT').mean() * 100.0
        sl_rate = sub['exit_reason'].str.startswith('STOP_LOSS').mean() * 100.0
        avg_w = wins['ret_pct'].mean() if len(wins) > 0 else 0.0
        avg_l = abs(losses['ret_pct'].mean()) if len(losses) > 0 else 0.0
        rr = avg_w / avg_l if avg_l > 0 else 0.0
        pf = wins['ret_pct'].sum() / abs(losses['ret_pct'].sum()) if len(losses) > 0 and losses['ret_pct'].sum() != 0 else float('inf')
        max_dd = compute_max_drawdown(simulate_equity_curve(sub))
        print(f"{label:<14} | {len(sub):<6d} | {win_rate:<8.1f} | {tp_rate:<10.1f}% | {sl_rate:<8.1f}% | {rr:<6.2f} | {pf:<6.2f} | {sub['ret_pct'].mean():<12.4f} | {max_dd*100:<6.1f}%")
    print("=" * 100)

    for th_label, th in [("strong_buy(0.700)", 0.700), ("buy(0.590)", 0.590), ("watch(0.470)", 0.470)]:
        print_monthly_breakdown(df_all, th, th_label)

    print(f"\n{'='*100}")
    print("【p_stop 閾値スキャン(risk_manager.pyのヘッジ判定 p_stop>=0.500 再検証用)】")
    print(f"  条件: p_stop >= th かつ p_stop > p_win のサブセットの実際の平均リターン")
    print("=" * 100)
    print(f"{'p_stop th':<12} | {'件数':<6} | {'勝率 (%)':<8} | {'平均ret_pct':<12} | {'損切率':<8} | {'平均p_win':<10} | {'平均p_neutral':<12}")
    print("-" * 90)
    ps_lo, ps_hi = df_all['p_stop'].quantile(0.50), df_all['p_stop'].quantile(0.99)
    for th in np.linspace(ps_lo, ps_hi, 10):
        sub = df_all[(df_all['p_stop'] >= th) & (df_all['p_stop'] > df_all['p_win'])].copy()
        if len(sub) == 0:
            print(f"{th:<12.3f} | {0:<6} | {'-':<8} | {'-':<12} | {'-':<8} | {'-':<10} | {'-':<12}")
            continue
        win_rate = (sub['ret_pct'] > 0).mean() * 100.0
        sl_rate = sub['exit_reason'].str.startswith('STOP_LOSS').mean() * 100.0
        p_win_avg = sub['p_win'].mean()
        p_neutral_avg = 1.0 - sub['p_stop'].mean() - p_win_avg
        print(f"{th:<12.3f} | {len(sub):<6d} | {win_rate:<8.1f} | {sub['ret_pct'].mean():<12.4f} | {sl_rate:<8.1f}% | {p_win_avg:<10.3f} | {p_neutral_avg:<12.3f}")
    print("=" * 100)

    # p_win>=0.6 かつ p_stop>=0.75 は p_win+p_stop+p_neutral=1 に反するため同時成立不可。
    # 「p_win/p_stopが両方高い(=p_neutralが低い、大きな値動きを予想)」という意図を汲み、
    # p_neutralの低さ + net bullish(p_win>p_stop) + 低ベータの3軸で代わりに検証する。
    print(f"\n{'='*100}")
    print("【修正版3軸判定: p_neutral低(高確信度) + p_win>p_stop(net bullish) + beta<=1.0】")
    print("=" * 100)
    print(f"{'p_neutral上限':<14} | {'件数':<6} | {'勝率 (%)':<8} | {'PF':<6} | {'MaxDD':<7}")
    print("-" * 60)
    p_neutral_all = 1.0 - df_all['p_win'] - df_all['p_stop']
    for pn_th in [0.30, 0.20, 0.15, 0.10, 0.05]:
        sub = df_all[(p_neutral_all <= pn_th) & (df_all['p_win'] > df_all['p_stop']) & (df_all['beta'] <= 1.0)].copy()
        if len(sub) == 0:
            print(f"{pn_th:<14.2f} | {0:<6} | {'-':<8} | {'-':<6} | {'-':<7}")
            continue
        wins, losses = sub[sub['ret_pct'] > 0], sub[sub['ret_pct'] < 0]
        win_rate = len(wins) / len(sub) * 100.0
        pf = wins['ret_pct'].sum() / abs(losses['ret_pct'].sum()) if len(losses) > 0 and losses['ret_pct'].sum() != 0 else float('inf')
        max_dd = compute_max_drawdown(simulate_equity_curve(sub))
        print(f"{pn_th:<14.2f} | {len(sub):<6d} | {win_rate:<8.1f} | {pf:<6.2f} | {max_dd*100:<6.1f}%")
    print("=" * 100)

def print_monthly_breakdown(df_all, th, label):
    """本番閾値ごとに、月別PF・MaxDDを算出する(月ごとに資金を1.0にリセットして
    その月の資金曲線を単独シミュレーション。月をまたぐ複利効果は含まない、
    月単位でのパフォーマンス安定性を見るための簡易指標)。"""
    sub = df_all[(df_all['p_win'] >= th) & (df_all['p_win'] > df_all['p_stop'])].copy()
    if sub.empty:
        print(f"\n[月別内訳: {label}] 該当サンプルなし")
        return
    sub['month'] = sub['date'].dt.to_period('M')

    print(f"\n[月別内訳: {label}, 総件数={len(sub)}]")
    print(f"{'年月':<10} | {'件数':<6} | {'勝率 (%)':<8} | {'PF':<6} | {'MaxDD':<7}")
    print("-" * 50)
    for month, g in sub.groupby('month'):
        wins, losses = g[g['ret_pct'] > 0], g[g['ret_pct'] < 0]
        win_rate = len(wins) / len(g) * 100.0
        pf = wins['ret_pct'].sum() / abs(losses['ret_pct'].sum()) if len(losses) > 0 and losses['ret_pct'].sum() != 0 else float('inf')
        max_dd = compute_max_drawdown(simulate_equity_curve(g))
        print(f"{str(month):<10} | {len(g):<6d} | {win_rate:<8.1f} | {pf:<6.2f} | {max_dd*100:<6.1f}%")

if __name__ == "__main__":
    return_source = "topix" if "--topix" in sys.argv else "nk225"
    use_expanded_universe = "--expanded-universe" in sys.argv
    run_backtest(return_source=return_source, use_expanded_universe=use_expanded_universe)
