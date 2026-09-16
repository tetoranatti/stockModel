import os
import pandas as pd

BASE_DIR = r"F:\stockModel"
DB_PATH = os.path.join(BASE_DIR, "jpx_daily_features_db.csv")
CACHE_DIR = os.path.join(BASE_DIR, "data", "cache")
NK225_UNDERLYING_CACHE_PATH = os.path.join(CACHE_DIR, "train_nk225_underlying.parquet")


def load_macro_slim5(db_path=DB_PATH, nk225_cache_path=NK225_UNDERLYING_CACHE_PATH):
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"[!] {db_path} が見つかりません。")
    if not os.path.exists(nk225_cache_path):
        raise FileNotFoundError(
            f"[!] {nk225_cache_path} が見つかりません。"
            f" 先に build_jquants_cache.py を実行してキャッシュを生成してください。"
        )

    jpx_db = pd.read_csv(db_path, index_col=0, parse_dates=True)
    jpx_db.index = pd.to_datetime(jpx_db.index).tz_localize(None)

    n225 = pd.read_parquet(nk225_cache_path)
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
