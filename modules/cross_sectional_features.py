# modules/cross_sectional_features.py
# 銘柄横断の(同日の全銘柄基準の)特徴量正規化。
# 「10日ウィンドウ内で自己正規化すると銘柄間の絶対水準の違いが消えてしまう」
# という問題への対応として、train_model_v8_exp.py / run_event_driven_backtest_v8_exp.py /
# run_dynamic_regime_screening_v8.py の3箇所で共有する。
import numpy as np
import pandas as pd

# 右に裾の長い正の比率指標。横断面統計を取る前にログ変換して歪みを緩和する。
LOG_TRANSFORM_COLS = ['vol_ratio_5d', 'atr_ratio']


def apply_log_transform(df):
    df = df.copy()
    for col in LOG_TRANSFORM_COLS:
        if col in df.columns:
            df[col] = np.log(df[col].clip(lower=1e-6))
    return df


def compute_cross_sectional_stats(per_ticker_dfs, stock_cols):
    """per_ticker_dfs: {ticker: DataFrame}(stock_colsを含み、apply_log_transform適用済み)。
    戻り値: (cross_mean, cross_std)、いずれも {col: 日付をindexとするSeries}。"""
    cross_mean, cross_std = {}, {}
    for col in stock_cols:
        wide = pd.DataFrame({t: df[col] for t, df in per_ticker_dfs.items() if col in df.columns})
        cross_mean[col] = wide.mean(axis=1)
        cross_std[col] = wide.std(axis=1)
    return cross_mean, cross_std


def valid_cross_section_dates(per_ticker_dfs, stock_cols, min_cross_section=20):
    """横断面統計が安定して計算できる(十分な銘柄数がある)日付だけを返す。"""
    ref_col = stock_cols[0]
    wide = pd.DataFrame({t: df[ref_col] for t, df in per_ticker_dfs.items() if ref_col in df.columns})
    count = wide.notna().sum(axis=1)
    return count[count >= min_cross_section].index


def normalize_cross_sectional(df, cross_mean, cross_std, stock_cols):
    """dfの各行(日付)を、その日のuniverse全体の平均・標準偏差で正規化する。"""
    out = pd.DataFrame(index=df.index)
    for col in stock_cols:
        cm = cross_mean[col].reindex(df.index)
        cs = cross_std[col].reindex(df.index)
        out[col] = (df[col] - cm) / (cs + 1e-7)
    return out
