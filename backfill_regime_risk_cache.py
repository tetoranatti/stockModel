# backfill_regime_risk_cache.py
# 地合い危険度モデルの日次キャッシュを過去全期間(既存データが揃っている範囲)で
# 一括計算する。ニュースセンチメントと違い、マクロ/VIX/USD/JPYは既に全履歴を
# 保有しているため、遡及取得の必要なく即座に全期間分を埋められる。
import os
import datetime
import pandas as pd

import train_model_v8_exp as tm
from modules.macro_features import load_macro_slim5
from modules.stock_features import compute_stock_features
from modules.regime_risk_model import (
    REGIME_COLS, add_regime_features, compute_breadth_5d,
    load_regime_risk_ensemble, predict_regime_risk, determine_regime_zone,
    append_regime_risk_cache,
)

BASE_DIR = r"F:\stockModel"
REGIME_MODEL_PATHS = [os.path.join(BASE_DIR, f"regime_risk_model_seed{s}.pt") for s in tm.ENSEMBLE_SEEDS]
BASE_K = 20


def main():
    print("=" * 80)
    print("【地合い危険度モデル 日次キャッシュ バックフィル(既存データ全期間)】")
    print("=" * 80)

    macro_df = load_macro_slim5()
    macro_df = add_regime_features(macro_df)

    with open(tm.UNIVERSE_PATH, "r", encoding="utf-8") as f:
        tickers = [line.strip() for line in f if line.strip()]
    universe_bars = pd.read_parquet(tm.UNIVERSE_BARS_CACHE_PATH)
    universe_bars.index = pd.to_datetime(universe_bars.index).tz_localize(None)
    m_start = macro_df.index.min() - datetime.timedelta(days=150)

    print("[*] 全銘柄の日次リターン(ブレッド指標用)を計算中...")
    per_ticker_ret1d = {}
    for i, t in enumerate(tickers):
        try:
            if t not in universe_bars.columns.get_level_values(0):
                continue
            df = universe_bars[t].loc[universe_bars.index >= m_start].copy()
            df = df.dropna(subset=['Close', 'High', 'Low', 'Volume'])
            if len(df) < 20 or (df['Volume'] == 0).all():
                continue
            aligned_nk = macro_df['NK_Ret'].reindex(df.index).fillna(0.0)
            df = compute_stock_features(df, aligned_nk)
            per_ticker_ret1d[t] = df['stock_ret_1d']
        except Exception:
            continue
        if (i + 1) % 50 == 0 or (i + 1) == len(tickers):
            print(f"  --> {i + 1}/{len(tickers)} 銘柄 完了")

    macro_df['breadth_5d'] = compute_breadth_5d(per_ticker_ret1d, macro_df.index)

    print("[*] 地合い危険度アンサンブルを読み込み中...")
    models, regime_cols, feat_mean, feat_std = load_regime_risk_ensemble(REGIME_MODEL_PATHS)

    valid_df = macro_df[REGIME_COLS].dropna()
    print(f"[+] 対象日数: {len(valid_df)} ({valid_df.index.min().date()} 〜 {valid_df.index.max().date()})")

    for i, (date, row) in enumerate(valid_df.iterrows()):
        risk_score = predict_regime_risk(models, regime_cols, feat_mean, feat_std, row)
        zone_info = determine_regime_zone(risk_score, base_k=BASE_K)
        append_regime_risk_cache(date, risk_score, zone_info)
        if (i + 1) % 100 == 0 or (i + 1) == len(valid_df):
            print(f"  --> {i + 1}/{len(valid_df)} 日 完了")

    print("\n[+] バックフィル完了: data/regime_risk_daily_cache.csv")


if __name__ == "__main__":
    main()
