# modules/stock_features.py
# 銘柄別テクニカル特徴量の算出ロジック。train_model_v8_exp.py /
# run_event_driven_backtest_v8_exp.py / run_dynamic_regime_screening_v8.py の
# 3箇所に手動複製されていた特徴量計算を一本化する。
import pandas as pd

STOCK_FEATURE_COLS = [
    'stock_ret_1d', 'stock_ret_5d', 'stock_ret_20d',
    'atr_ratio', 'rolling_beta', 'vol_ratio_5d',
    'overnight_gap', 'dist_from_high20', 'dist_from_low20',
]


def compute_stock_features(df, nk_ret_aligned):
    """OHLCV(Close/High/Low/Open/Volume)を持つ銘柄別DataFrameに特徴量列を追加して返す。

    nk_ret_aligned: 日経225リターンをdf.indexにreindexしたSeries(呼び出し側で
    macro_df['NK_Ret'].reindex(df.index).fillna(0.0) 等を渡す)。rolling_betaの算出に使う。
    'ATR'列(atr_ratioの元値)も副産物として追加し、呼び出し側の利確/損切りライン計算に使う。
    """
    df['stock_ret_1d'] = df['Close'].pct_change(1, fill_method=None).fillna(0.0)
    df['stock_ret_5d'] = df['Close'].pct_change(5, fill_method=None).fillna(0.0)
    df['stock_ret_20d'] = df['Close'].pct_change(20, fill_method=None).fillna(0.0)

    vol_5d = df['Volume'].rolling(5).mean()
    df['vol_ratio_5d'] = (df['Volume'] / (vol_5d + 1e-7)).fillna(1.0)

    hl = df['High'] - df['Low']
    h_cp = (df['High'] - df['Close'].shift(1)).abs()
    l_cp = (df['Low'] - df['Close'].shift(1)).abs()
    tr = pd.concat([hl, h_cp, l_cp], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    df['ATR'] = atr
    df['atr_ratio'] = (atr / (df['Close'] + 1e-7)).fillna(0.0)

    # betaは20日窓だと分母(日経リターンの分散)がたまたま小さい期間に
    # 当たった際に推定が不安定化し極端な値(|beta|>10等)が出ることを確認済み。
    # 90日窓では検証データ上この種の外れ値が完全に解消したため採用し、
    # 念のためクリップも残して二重に防御する。
    BETA_WINDOW = 90
    cov = df['stock_ret_1d'].rolling(BETA_WINDOW).cov(nk_ret_aligned)
    var = nk_ret_aligned.rolling(BETA_WINDOW).var()
    df['rolling_beta'] = (cov / (var + 1e-7)).fillna(1.0).clip(-5.0, 5.0)

    df['overnight_gap'] = (df['Open'] / df['Close'].shift(1) - 1.0).fillna(0.0)

    high20 = df['High'].rolling(20).max()
    low20 = df['Low'].rolling(20).min()
    df['dist_from_high20'] = ((df['Close'] - high20) / (high20 + 1e-7)).fillna(0.0)
    df['dist_from_low20'] = ((df['Close'] - low20) / (low20 + 1e-7)).fillna(0.0)

    return df
