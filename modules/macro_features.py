import os
import pandas as pd

BASE_DIR = r"F:\stockModel"
DB_PATH = os.path.join(BASE_DIR, "jpx_daily_features_db.csv")
CACHE_DIR = os.path.join(BASE_DIR, "data", "cache")
NK225_UNDERLYING_CACHE_PATH = os.path.join(CACHE_DIR, "train_nk225_underlying.parquet")
TOPIX_CACHE_PATH = os.path.join(CACHE_DIR, "train_topix_bars.parquet")


def load_macro_slim5(db_path=DB_PATH, nk225_cache_path=NK225_UNDERLYING_CACHE_PATH, return_source="nk225"):
    """return_source="nk225"(既定、本番と同一挙動) or "topix"。
    pin_dist_ratio/wall_spreadは日経225オプションの建玉(権利行使価格が日経225の
    ポイント建て)由来のため、指数をTOPIXに切り替えてもNK_Close基準のまま変えない。
    切り替えるのはベータ計算・nk_ret_norm(モメンタム特徴量)に使うリターン系列のみ。
    (TOPIXの方が時価総額加重で個別株ユニバースとの整合性が良いのではという仮説の検証用)
    """
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

    if return_source == "topix":
        if not os.path.exists(TOPIX_CACHE_PATH):
            raise FileNotFoundError(f"[!] {TOPIX_CACHE_PATH} が見つかりません。")
        topix = pd.read_parquet(TOPIX_CACHE_PATH)
        topix.index = pd.to_datetime(topix.index).tz_localize(None)
        topix_ret = topix['Close'].pct_change(fill_method=None)
        ret_series = topix_ret.reindex(macro_df.index).ffill().fillna(0.0)
        ret_norm_divisor = 0.014  # TOPIX日次リターンの実測std(≈0.0137)に合わせて再較正
    else:
        ret_series = macro_df['NK_Close'].pct_change(fill_method=None).fillna(0.0)
        ret_norm_divisor = 0.015  # NK225日次リターンの実測std(≈0.0164)に近い既存の較正値

    macro_df['NK_Ret'] = ret_series

    macro_df = macro_df.join(jpx_db, how='inner').ffill().fillna(0.0)

    pin_strike = (macro_df['call_oi_wall'] + macro_df['put_oi_wall']) / 2.0
    macro_df['pin_dist_ratio'] = ((macro_df['NK_Close'] - pin_strike) / (pin_strike + 1e-5)) / 0.02
    macro_df['wall_spread'] = ((macro_df['call_oi_wall'] - macro_df['put_oi_wall']).abs() / (pin_strike + 1e-5)) / 0.02

    cta_mean = macro_df['cta_net_futures'].rolling(60, min_periods=10).mean()
    cta_std = macro_df['cta_net_futures'].rolling(60, min_periods=10).std() + 1e-5
    macro_df['cta_net_norm'] = ((macro_df['cta_net_futures'] - cta_mean) / cta_std).fillna(0.0)
    macro_df['cta_momentum'] = (macro_df['cta_net_futures'] - macro_df['cta_net_futures'].shift(5)).fillna(0.0) / 5000.0
    macro_df['nk_ret_norm'] = macro_df['NK_Ret'] / ret_norm_divisor

    return macro_df
