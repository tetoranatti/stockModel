# train_regime_risk_model.py
# 日次集約の地合い危険度モデル(modules/regime_risk_model.py)の本番学習スクリプト。
# 個別銘柄の危険度は較正できなかったため、「その日全体でどれだけ損切りが
# 出やすいか」という日次集約ラベルに切り替えて学習する(検証: AUC≈0.64@5シード)。
import os
import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# training/配下からでもmodules/を解決できるようにプロジェクトルートをsys.pathへ追加
import sys
sys.path.insert(0, r"F:\stockModel")
import train_model_v8_exp as tm
from modules.macro_features import load_macro_slim5
from modules.stock_features import compute_stock_features
from modules.regime_risk_model import REGIME_COLS, TinyRegimeMLP, add_regime_features, compute_breadth_5d, DEVICE

BASE_DIR = r"F:\stockModel"
MODEL_SAVE_PATH_TEMPLATE = os.path.join(BASE_DIR, "regime_risk_model_seed{seed}.pt")
ENSEMBLE_SEEDS = [42, 43, 44, 45, 46]
SEQ_LEN, HOLDING_PERIOD = 10, 10
MIN_TICKERS_PER_DAY = 20


def _simulate_stop_flag(closes, highs, lows, atrs, holding_period):
    n_bars = len(closes)
    stop_flags = np.full(n_bars, np.nan)
    for idx in range(n_bars - holding_period):
        entry_p = closes[idx]
        upper_p = entry_p + (2.0 * atrs[idx])
        lower_p = entry_p - (1.0 * atrs[idx])
        is_stop = 0.0
        for h in range(1, holding_period + 1):
            bh, bl = highs[idx + h], lows[idx + h]
            if bl <= lower_p and bh >= upper_p:
                is_stop = 1.0
                break
            elif bh >= upper_p:
                is_stop = 0.0
                break
            elif bl <= lower_p:
                is_stop = 1.0
                break
        stop_flags[idx] = is_stop
    return stop_flags


def build_day_level_dataset():
    with open(tm.UNIVERSE_PATH, "r", encoding="utf-8") as f:
        tickers = [line.strip() for line in f if line.strip()]

    macro_df = load_macro_slim5()
    macro_df = add_regime_features(macro_df)

    universe_bars = pd.read_parquet(tm.UNIVERSE_BARS_CACHE_PATH)
    universe_bars.index = pd.to_datetime(universe_bars.index).tz_localize(None)
    m_start = macro_df.index.min() - datetime.timedelta(days=150)

    print(f"[*] 全 {len(tickers)} 銘柄のstop_flag・日次リターンを計算中...")
    per_ticker_stop, per_ticker_ret1d = {}, {}
    for i, t in enumerate(tickers):
        try:
            if t not in universe_bars.columns.get_level_values(0):
                continue
            df = universe_bars[t].loc[universe_bars.index >= m_start].copy()
            df = df.dropna(subset=['Close', 'High', 'Low', 'Volume'])
            if len(df) < SEQ_LEN + HOLDING_PERIOD + 95 or (df['Volume'] == 0).all():
                continue
            aligned_nk = macro_df['NK_Ret'].reindex(df.index).fillna(0.0)
            df = compute_stock_features(df, aligned_nk)
            df = df.dropna(subset=['ATR', 'rolling_beta'])
            if len(df) < SEQ_LEN + HOLDING_PERIOD + 10:
                continue
            closes, highs, lows, atrs = df['Close'].values, df['High'].values, df['Low'].values, df['ATR'].values
            stop_flags = _simulate_stop_flag(closes, highs, lows, atrs, HOLDING_PERIOD)
            per_ticker_stop[t] = pd.Series(stop_flags, index=df.index)
            per_ticker_ret1d[t] = df['stock_ret_1d']
        except Exception:
            continue
        if (i + 1) % 50 == 0 or (i + 1) == len(tickers):
            print(f"  --> {i + 1}/{len(tickers)} 銘柄 完了")

    macro_df['breadth_5d'] = compute_breadth_5d(per_ticker_ret1d, macro_df.index)

    stop_df = pd.DataFrame(per_ticker_stop)
    daily_stop_rate = stop_df.mean(axis=1, skipna=True)
    daily_n = stop_df.notna().sum(axis=1)
    valid_days = daily_n[daily_n >= MIN_TICKERS_PER_DAY].index

    day_df = pd.DataFrame({'y_stop_rate': daily_stop_rate}).reindex(valid_days).dropna()
    day_df = day_df.join(macro_df[REGIME_COLS], how='inner').dropna()
    day_df = day_df.sort_index()
    return day_df


def train_one_seed(seed, X_train, y_train_bin, X_val, y_val_bin):
    tm.set_seed(seed)
    model = TinyRegimeMLP(input_dim=len(REGIME_COLS)).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-2)

    X_t = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    y_t = torch.tensor(y_train_bin, dtype=torch.float32).to(DEVICE)
    X_v = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    y_v = torch.tensor(y_val_bin, dtype=torch.float32).to(DEVICE)

    best_val_loss, patience, patience_cnt, epochs, best_state = float('inf'), 10, 0, 100, None
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_t), y_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_v), y_v).item()
        if val_loss < best_val_loss:
            best_val_loss, patience_cnt = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break
    model.load_state_dict(best_state)
    print(f"  [regime_risk seed={seed}] stopped_epoch={epoch} best_val_loss={best_val_loss:.4f}")
    return model


def train():
    day_df = build_day_level_dataset()
    print(f"[+] 日次集約サンプル数: {len(day_df)}")
    print(f"[+] y_stop_rate分布: min={day_df['y_stop_rate'].min():.3f} median={day_df['y_stop_rate'].median():.3f} max={day_df['y_stop_rate'].max():.3f}")

    split_idx = int(len(day_df) * 0.75)
    train_df = day_df.iloc[:split_idx]
    val_df = day_df.iloc[split_idx:]

    threshold = train_df['y_stop_rate'].median()
    y_train_bin = (train_df['y_stop_rate'] > threshold).astype(int).values
    y_val_bin = (val_df['y_stop_rate'] > threshold).astype(int).values

    feat_mean = train_df[REGIME_COLS].mean().values.astype(np.float32)
    feat_std = (train_df[REGIME_COLS].std() + 1e-7).values.astype(np.float32)
    X_train = ((train_df[REGIME_COLS].values - feat_mean) / feat_std).astype(np.float32)
    X_val = ((val_df[REGIME_COLS].values - feat_mean) / feat_std).astype(np.float32)

    print(f"[+] Train={len(train_df)} Val={len(val_df)} | 二値化閾値={threshold:.3f}")

    for seed in ENSEMBLE_SEEDS:
        print(f"\n[*] [regime_risk アンサンブル seed={seed}] 学習開始...")
        model = train_one_seed(seed, X_train, y_train_bin, X_val, y_val_bin)
        save_path = MODEL_SAVE_PATH_TEMPLATE.format(seed=seed)
        torch.save({
            'model_state_dict': model.state_dict(),
            'regime_cols': REGIME_COLS,
            'feat_mean': torch.tensor(feat_mean),
            'feat_std': torch.tensor(feat_std),
            'threshold': float(threshold),
            'seed': seed,
        }, save_path)
        print(f"[+] 重み保存完了: {save_path}")


if __name__ == "__main__":
    train()
