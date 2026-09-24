"""特徴量実験の下準備(150銘柄分のcompute_stock_features()+基本マクロ結合)を
キャッシュする共通ユーティリティ。この部分は候補特徴量が何であれ共通なので、
実験ごとに再計算せず1回だけ計算してpickleに保存し、以降の実験はそこから
候補特徴量だけ追加する形にする(2026-09-18、ユーザー要望によりキャッシュ化)。

使い方:
    from _feature_cache_utils import get_base_per_ticker_df
    per_ticker_df = get_base_per_ticker_df(tickers, macro_df)
    for t, df in per_ticker_df.items():
        df = df.copy()
        df['my_new_feature'] = ...   # 実験固有の特徴量をここで追加
"""
import os
import pickle
import datetime
import numpy as np
import pandas as pd
import sys

sys.path.insert(0, r"F:\stockModel")
from modules.stock_features import compute_stock_features

BASE_DIR = r"F:\stockModel"
CACHE_DIR = os.path.join(BASE_DIR, "data", "cache")
UNIVERSE_BARS_CACHE_PATH = os.path.join(CACHE_DIR, "train_universe_bars.parquet")
SCRATCH_CACHE_PATH = os.path.join(
    r"F:\stockModel\research\cache",
    "_cache_base_per_ticker_df.pkl",
)
SEQ_LEN, HOLDING_PERIOD = 10, 10


def get_base_per_ticker_df(tickers, macro_df, force_rebuild=False):
    """{ticker: df} を返す。dfはcompute_stock_features()適用済み(9特徴量+ATR+rolling_beta)
    かつOpen/High/Low/Close/Volumeも保持したまま(実験側でTR再計算等に使えるように)。
    候補特徴量・横断面正規化・ログ変換・macro結合はまだ行っていない状態
    (それらは実験固有なのでここではやらない)。"""
    if not force_rebuild and os.path.exists(SCRATCH_CACHE_PATH):
        with open(SCRATCH_CACHE_PATH, "rb") as f:
            cached = pickle.load(f)
        print(f"[cache] ベース特徴量キャッシュを読み込みました: {len(cached)}銘柄 "
              f"({SCRATCH_CACHE_PATH})")
        return cached

    print("[cache] ベース特徴量キャッシュが無いため新規計算します(初回のみ、以降は再利用)...")
    universe_bars = pd.read_parquet(UNIVERSE_BARS_CACHE_PATH)
    universe_bars.index = pd.to_datetime(universe_bars.index).tz_localize(None)
    m_start = macro_df.index.min() - datetime.timedelta(days=150)

    per_ticker_df = {}
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
            per_ticker_df[t] = df
        except Exception:
            continue
        if (i + 1) % 50 == 0:
            print(f"  --> {i + 1}/{len(tickers)} 銘柄 完了")

    with open(SCRATCH_CACHE_PATH, "wb") as f:
        pickle.dump(per_ticker_df, f)
    print(f"[cache] ベース特徴量キャッシュを保存しました: {len(per_ticker_df)}銘柄 -> {SCRATCH_CACHE_PATH}")
    return per_ticker_df


# ============================================================================
# 特徴量プール: これまでの実験で試した候補特徴量を全部まとめて1回だけ計算・
# キャッシュしておく。今後どの組み合わせで再実験しても、ここから列を選ぶだけで
# 済むようにする(2026-09-18、ユーザー要望: 今後使わないかもしれない特徴量でも
# 頻繁に組み替える可能性があるため必ずキャッシュしておいてほしい、との指示)。
# ============================================================================
POOL_CACHE_PATH = os.path.join(
    r"F:\stockModel\research\cache",
    "_cache_full_feature_pool.pkl",
)
MACRO_POOL_CACHE_PATH = os.path.join(
    r"F:\stockModel\research\cache",
    "_cache_full_macro_pool.pkl",
)

# 銘柄レベルの候補特徴量プール(元: atr_vol_feature_ablation.py / kitchensink_feature_ablation.py /
# technical_batch_ablation.py で個別に実装していたものを統合)
POOL_STOCK_FEATURE_COLS = [
    'atr_term_ratio', 'atr_accel',            # ATR比率・ATR加速度
    'gap_avg_5d',                              # Gap率(直近5日平均|overnight_gap|)
    'volume_zscore',                           # 異常出来高(20日zスコア)
    'relative_strength_5d', 'rs20', 'rs60',    # 対日経相対強度(5/20/60日)
    'dist_from_high60',                        # 60日高値距離
    'momentum_accel',                          # モメンタム加速度
    'close_location_value', 'body_ratio',      # 終値位置率・実体率
    'adx14',                                   # ADX14
    'volume_accel',                            # 出来高加速度
    'efficiency_ratio20',                      # Efficiency Ratio(Kaufman、20日)
    'days_since_high20',                       # 20日高値からの経過日数
    'new_high20',                              # 20日ブレイクアウト(過去20日高値を上回ったか、二値)
    'positive_gap_ratio_5',                    # 直近5日でプラスギャップだった日の割合
    'gap_strength_5',                          # 直近5日の符号付きギャップ平均(方向性)
    'gap_follow_through',                      # ギャップ方向とその日の値動きが順張りしたか(5日平均)
    'volume_ma_ratio',                         # 出来高5日MA÷20日MA
    'vwap_distance',                           # 終値の20日出来高加重平均(VWAP)からの乖離率(2026-09-23追加)
    'ret_rank_20d', 'vol_rank_5d', 'high20_rank',  # ユニバース全体での横断パーセンタイル順位(2026-09-23追加)
]

# 横断(ユニバース全体)ランク特徴量: key=出力列名, value=順位化する元の列名。sector_rank_ret_5d
# (業種内順位、不採用済み)とはスコープが異なり、150銘柄ユニバース全体でのその日の順位
# (PAIR_MODE="cross_sectional"の学習方式との一致を狙う、2026-09-23追加)。
CROSS_SECTIONAL_RANK_COLS = {
    'ret_rank_20d': 'stock_ret_20d',
    'vol_rank_5d': 'vol_ratio_5d',
    'high20_rank': 'dist_from_high20',
}


def _add_cross_sectional_rank_features(pool):
    """CROSS_SECTIONAL_RANK_COLSの元列を、その日のユニバース全体でのパーセンタイル順位
    (-1〜+1)に変換した列を各銘柄dfに追加する(in-place)。poolの各dfは元列
    (stock_ret_20d/vol_ratio_5d/dist_from_high20、いずれもcompute_stock_features()で
    計算済みでNaN無し)を既に持っている前提。"""
    src_cols = list(CROSS_SECTIONAL_RANK_COLS.values())
    long_rows = []
    for t, df in pool.items():
        sub = df[src_cols].copy()
        sub['ticker'] = t
        sub['date'] = sub.index
        long_rows.append(sub)
    long_df = pd.concat(long_rows, ignore_index=True)

    for rank_col, src_col in CROSS_SECTIONAL_RANK_COLS.items():
        long_df[rank_col] = (long_df.groupby('date')[src_col].rank(pct=True) - 0.5) * 2

    lookup = long_df.set_index(['date', 'ticker'])[list(CROSS_SECTIONAL_RANK_COLS)]
    for t, df in pool.items():
        idx = pd.MultiIndex.from_arrays([df.index, [t] * len(df)], names=['date', 'ticker'])
        aligned = lookup.reindex(idx)
        for rank_col in CROSS_SECTIONAL_RANK_COLS:
            df[rank_col] = np.nan_to_num(aligned[rank_col].values, nan=0.0)
    return pool

# マクロレベルの候補特徴量プール
POOL_MACRO_FEATURE_COLS = [
    'market_vol_regime',                        # Volatility Regime(市場realized volのz score)
    'vix_overnight_chg',                         # VIXオーバーナイト変化率
    'usdjpy_chg',                                # USDJPY日次変化率
    'nk225_ret_5d', 'topix_ret_5d',              # 日経/TOPIXの5日リターン
    # regime別IC分析で符号反転が見られた特徴量の交互作用項(2026-09-18追加)
    'cta_momentum_x_mid', 'cta_net_norm_x_low',
    'market_vol_regime_x_high', 'usdjpy_chg_x_low',
]

# 市場breadth系の候補特徴量プール(2026-09-21追加、ユーザー提案: ユニバース全銘柄の
# 横断的な強弱を捉える指標。個別銘柄の特徴量プールとは独立に、universe_bars.parquetの
# 生OHLCVから直接計算する)
POOL_BREADTH_FEATURE_COLS = [
    'market_above_ma25_ratio',   # 25日移動平均を上回っている銘柄の比率
    'market_newhigh20_ratio',    # 過去20日高値を更新した銘柄の比率
    'market_newlow20_ratio',     # 過去20日安値を更新した銘柄の比率
    'market_breakout_score',     # 20日高値/安値を超過した幅(%)の銘柄横断平均(ブレイクアウトの強さ)
    'market_turnover_z',         # 市場全体売買代金(Close×Volume合計)5日平均の60日zスコア
]


def _compute_pool_stock_features(df, macro_df):
    """POOL_STOCK_FEATURE_COLS全部を計算して追加する。dfはcompute_stock_features()
    (ATR/rolling_beta含む)適用済みで、Open/High/Low/Close/Volumeも保持している前提。"""
    df = df.copy()
    hl = df['High'] - df['Low']
    h_cp = (df['High'] - df['Close'].shift(1)).abs()
    l_cp = (df['Low'] - df['Close'].shift(1)).abs()
    tr = pd.concat([hl, h_cp, l_cp], axis=1).max(axis=1)

    atr_5d = tr.rolling(5).mean()
    df['atr_term_ratio'] = (atr_5d / (df['ATR'] + 1e-7)).fillna(1.0)
    df['atr_accel'] = (df['ATR'] / (df['ATR'].shift(5) + 1e-7) - 1.0).fillna(0.0)

    overnight_gap = (df['Open'] / df['Close'].shift(1) - 1.0)
    df['gap_avg_5d'] = overnight_gap.abs().rolling(5).mean().fillna(0.0)

    vol_mean_20 = df['Volume'].rolling(20).mean()
    vol_std_20 = df['Volume'].rolling(20).std() + 1e-7
    df['volume_zscore'] = ((df['Volume'] - vol_mean_20) / vol_std_20).fillna(0.0)

    nk_close_aligned = macro_df['NK_Close'].reindex(df.index).ffill()
    nk_ret_5 = nk_close_aligned.pct_change(5)
    nk_ret_20 = nk_close_aligned.pct_change(20)
    nk_ret_60 = nk_close_aligned.pct_change(60)
    df['relative_strength_5d'] = (df['Close'].pct_change(5) - nk_ret_5).fillna(0.0)
    df['rs20'] = (df['Close'].pct_change(20) - nk_ret_20).fillna(0.0)
    df['rs60'] = (df['Close'].pct_change(60) - nk_ret_60).fillna(0.0)

    high60 = df['High'].rolling(60).max()
    df['dist_from_high60'] = ((df['Close'] - high60) / (high60 + 1e-7)).fillna(0.0)

    ret_5d = df['Close'].pct_change(5)
    df['momentum_accel'] = (ret_5d - ret_5d.shift(5)).fillna(0.0)

    rng = (df['High'] - df['Low']) + 1e-7
    df['close_location_value'] = (((df['Close'] - df['Low']) / rng) * 2 - 1).fillna(0.0)
    df['body_ratio'] = ((df['Close'] - df['Open']).abs() / rng).fillna(0.0)

    up_move = df['High'].diff()
    down_move = -df['Low'].diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)
    atr14 = tr.rolling(14).mean() + 1e-7
    plus_di = 100 * (plus_dm.rolling(14).mean() / atr14)
    minus_di = 100 * (minus_dm.rolling(14).mean() / atr14)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-7)
    df['adx14'] = dx.rolling(14).mean().fillna(0.0)

    vol_5d = df['Volume'].rolling(5).mean()
    df['volume_accel'] = (vol_5d / (vol_5d.shift(5) + 1e-7) - 1.0).fillna(0.0)

    # Efficiency Ratio(Kaufman, 20日): 直線的に動いた距離 / 実際に動いた総距離。
    # 1に近いほど一方向に効率よく動いている(トレンド)、0に近いほどノイズが多い(レンジ)。
    net_change = (df['Close'] - df['Close'].shift(20)).abs()
    total_change = df['Close'].diff().abs().rolling(20).sum()
    df['efficiency_ratio20'] = (net_change / (total_change + 1e-7)).fillna(0.0)

    # 20日高値からの経過日数: 直近の高値がどれだけ「古い」か。
    high20_argmax = df['High'].rolling(20).apply(lambda x: len(x) - 1 - np.argmax(x.values), raw=False)
    df['days_since_high20'] = high20_argmax.fillna(0.0)

    # 20日ブレイクアウト: 本日終値が「本日を含まない過去20日」の高値を上回ったか(二値)。
    prior_high20 = df['High'].shift(1).rolling(20).max()
    df['new_high20'] = (df['Close'] > prior_high20).astype(float).fillna(0.0)

    gap = (df['Open'] / df['Close'].shift(1) - 1.0)
    df['positive_gap_ratio_5'] = (gap > 0).rolling(5).mean().fillna(0.5)
    df['gap_strength_5'] = gap.rolling(5).mean().fillna(0.0)

    # ギャップ方向とその日の値動き(寄り引け)が同じ向きに継続したか(順張り継続度、5日平均)。
    # 正なら「ギャップした方向にその日も伸びる」傾向、負なら「ギャップを埋めがち」な傾向。
    daily_follow = np.sign(gap) * (df['Close'] / df['Open'] - 1.0)
    df['gap_follow_through'] = daily_follow.rolling(5).mean().fillna(0.0)

    vol_ma5 = df['Volume'].rolling(5).mean()
    vol_ma20 = df['Volume'].rolling(20).mean()
    df['volume_ma_ratio'] = (vol_ma5 / (vol_ma20 + 1e-7)).fillna(1.0)

    # 終値の20日出来高加重平均(VWAP)からの乖離率(日次OHLCVのみのためCloseベースの近似)。
    vwap20 = ((df['Close'] * df['Volume']).rolling(20).sum()
              / (df['Volume'].rolling(20).sum() + 1e-7))
    df['vwap_distance'] = ((df['Close'] - vwap20) / (vwap20 + 1e-7)).fillna(0.0)
    return df


def get_full_feature_pool_df(tickers, macro_df, force_rebuild=False):
    """銘柄レベルの候補特徴量(POOL_STOCK_FEATURE_COLS)を全部追加した{ticker: df}を返す。
    ベースキャッシュ(get_base_per_ticker_df)の上に構築し、こちらも同様にpickleキャッシュする。
    実験側は使いたい列だけ df[STOCK_FEATURE_COLS + [欲しい列...]] のように選んで使う。"""
    if not force_rebuild and os.path.exists(POOL_CACHE_PATH):
        with open(POOL_CACHE_PATH, "rb") as f:
            cached = pickle.load(f)
        print(f"[cache] 特徴量プールキャッシュを読み込みました: {len(cached)}銘柄 ({POOL_CACHE_PATH})")
        return cached

    print("[cache] 特徴量プールキャッシュが無いため新規計算します...")
    base = get_base_per_ticker_df(tickers, macro_df, force_rebuild=force_rebuild)
    pool = {}
    for i, (t, df) in enumerate(base.items()):
        try:
            pool[t] = _compute_pool_stock_features(df, macro_df)
        except Exception:
            continue
        if (i + 1) % 50 == 0:
            print(f"  --> {i + 1}/{len(base)} 銘柄 完了")
    pool = _add_cross_sectional_rank_features(pool)

    with open(POOL_CACHE_PATH, "wb") as f:
        pickle.dump(pool, f)
    print(f"[cache] 特徴量プールキャッシュを保存しました: {len(pool)}銘柄 -> {POOL_CACHE_PATH}")
    return pool


SECTOR_POOL_CACHE_PATH = os.path.join(
    r"F:\stockModel\research\cache",
    "_cache_sector_relative_pool.pkl",
)
SECTOR_MASTER_JSON = os.path.join(BASE_DIR, "data", "jpx_sector_master.json")
MIN_SECTOR_GROUP_SIZE = 5  # これ未満の同時点サンプル数ならsector相対特徴量は0(中立)にする

# セクター相対系の候補特徴量プール(2026-09-18追加、完全新規の情報源(セクター内相対位置)を検証)
POOL_SECTOR_FEATURE_COLS = [
    'sector_rel_ret_5d', 'sector_rel_ret_20d',    # 同業種平均との相対リターン(5日/20日)
    'sector_rel_vol_ratio_5d',                     # 同業種平均との相対出来高比率
    'sector_rank_ret_5d',                          # 同業種内での5日リターン順位(-1〜+1、0=中央)
]


def _load_sector_map():
    """ticker('XXXX.T') -> sector33業種名 の辞書を返す。"""
    import json
    with open(SECTOR_MASTER_JSON, "r", encoding="utf-8") as f:
        master = json.load(f)
    sector_map = {}
    for code, rec in master.items():
        sector_map[f"{code}.T"] = rec.get("sector33")
    return sector_map


def get_sector_relative_pool_df(tickers, macro_df, force_rebuild=False):
    """POOL_SECTOR_FEATURE_COLSを全部追加した{ticker: df}を返す。ベースキャッシュ
    (get_base_per_ticker_df)の上に構築。150銘柄ユニバース内でのセクター(33業種)別
    横断面統計を使うため、真の市場全体のセクター平均ではなく「ユニバース内相対」である点に注意。"""
    if not force_rebuild and os.path.exists(SECTOR_POOL_CACHE_PATH):
        with open(SECTOR_POOL_CACHE_PATH, "rb") as f:
            cached = pickle.load(f)
        print(f"[cache] セクター相対特徴量プールキャッシュを読み込みました: {len(cached)}銘柄 ({SECTOR_POOL_CACHE_PATH})")
        return cached

    print("[cache] セクター相対特徴量プールキャッシュが無いため新規計算します...")
    base = get_base_per_ticker_df(tickers, macro_df, force_rebuild=force_rebuild)
    sector_map = _load_sector_map()

    # 1. 全銘柄のstock_ret_5d/20d, vol_ratio_5dをlong形式に集約
    long_rows = []
    for t, df in base.items():
        sector = sector_map.get(t)
        if sector is None:
            continue
        sub = df[['stock_ret_5d', 'stock_ret_20d', 'vol_ratio_5d']].copy()
        sub['ticker'] = t
        sub['sector33'] = sector
        sub['date'] = sub.index
        long_rows.append(sub)
    long_df = pd.concat(long_rows, ignore_index=True)

    # 2. (date, sector33)別の平均を算出(グループサイズが小さい日は後でNaN扱いにする)
    grp = long_df.groupby(['date', 'sector33'])
    sector_stats = grp[['stock_ret_5d', 'stock_ret_20d', 'vol_ratio_5d']].mean()
    sector_stats.columns = ['sec_mean_ret_5d', 'sec_mean_ret_20d', 'sec_mean_vol_ratio_5d']
    sector_count = grp.size().rename('sec_n')
    sector_stats = sector_stats.join(sector_count)

    # 3. sector内順位(パーセンタイル、-1〜+1に変換)
    long_df['rank_ret_5d'] = long_df.groupby(['date', 'sector33'])['stock_ret_5d'].rank(pct=True)
    long_df['sector_rank_ret_5d'] = (long_df['rank_ret_5d'] - 0.5) * 2
    rank_lookup = long_df.set_index(['date', 'ticker'])['sector_rank_ret_5d']

    # 4. 各銘柄dfに結合
    pool = {}
    for i, (t, df) in enumerate(base.items()):
        sector = sector_map.get(t)
        if sector is None:
            continue
        try:
            df2 = df.copy()
            idx_pairs = pd.MultiIndex.from_arrays([df2.index, [sector] * len(df2)], names=['date', 'sector33'])
            stats_aligned = sector_stats.reindex(idx_pairs)
            stats_aligned.index = df2.index

            valid = stats_aligned['sec_n'] >= MIN_SECTOR_GROUP_SIZE
            df2['sector_rel_ret_5d'] = np.where(valid, df2['stock_ret_5d'] - stats_aligned['sec_mean_ret_5d'], 0.0)
            df2['sector_rel_ret_20d'] = np.where(valid, df2['stock_ret_20d'] - stats_aligned['sec_mean_ret_20d'], 0.0)
            df2['sector_rel_vol_ratio_5d'] = np.where(valid, df2['vol_ratio_5d'] - stats_aligned['sec_mean_vol_ratio_5d'], 0.0)

            rank_idx = pd.MultiIndex.from_arrays([df2.index, [t] * len(df2)], names=['date', 'ticker'])
            rank_vals = rank_lookup.reindex(rank_idx).values
            df2['sector_rank_ret_5d'] = np.where(valid.values, np.nan_to_num(rank_vals, nan=0.0), 0.0)

            pool[t] = df2
        except Exception:
            continue
        if (i + 1) % 50 == 0:
            print(f"  --> {i + 1}/{len(base)} 銘柄 完了")

    with open(SECTOR_POOL_CACHE_PATH, "wb") as f:
        pickle.dump(pool, f)
    print(f"[cache] セクター相対特徴量プールキャッシュを保存しました: {len(pool)}銘柄 -> {SECTOR_POOL_CACHE_PATH}")
    return pool


MARGIN_POOL_CACHE_PATH = os.path.join(
    r"F:\stockModel\research\cache",
    "_cache_margin_pool.pkl",
)
MARGIN_HISTORY_CSV = os.path.join(
    r"F:\stockModel\research\cache",
    "margin_history_long.csv",
)

# 信用取引残高系の候補特徴量プール(2026-09-18追加、J-Quants /markets/margin-interest を
# 週次バックフィルした margin_history_long.csv から算出。週次公表のため日次にffillして使う)
POOL_MARGIN_FEATURE_COLS = [
    'margin_ratio_level',       # 信用倍率(買い残/売り残)の生値(ffill済み)
    'margin_ratio_zscore_12w',  # 信用倍率の直近12週z-score(自分自身の履歴内での高低)
    'margin_buy_chg_1w',        # 信用買い残の前週比変化率
    'margin_short_chg_1w',      # 信用売り残の前週比変化率
]


def get_margin_pool_df(tickers, macro_df, force_rebuild=False):
    """POOL_MARGIN_FEATURE_COLSを全部追加した{ticker: df}を返す。週次データを日次にffillする。"""
    if not force_rebuild and os.path.exists(MARGIN_POOL_CACHE_PATH):
        with open(MARGIN_POOL_CACHE_PATH, "rb") as f:
            cached = pickle.load(f)
        print(f"[cache] マージン特徴量プールキャッシュを読み込みました: {len(cached)}銘柄 ({MARGIN_POOL_CACHE_PATH})")
        return cached

    print("[cache] マージン特徴量プールキャッシュが無いため新規計算します...")
    base = get_base_per_ticker_df(tickers, macro_df, force_rebuild=force_rebuild)

    margin_long = pd.read_csv(MARGIN_HISTORY_CSV, encoding='utf-8-sig')
    margin_long['date'] = pd.to_datetime(margin_long['date'])

    pool = {}
    for i, (t, df) in enumerate(base.items()):
        try:
            m = margin_long[margin_long['ticker'] == t].sort_values('date').set_index('date')
            if len(m) < 10:
                continue
            m['margin_buy_chg_1w'] = m['margin_buy'].pct_change().replace([np.inf, -np.inf], 0.0).fillna(0.0)
            m['margin_short_chg_1w'] = m['margin_short'].pct_change().replace([np.inf, -np.inf], 0.0).fillna(0.0)
            roll_mean = m['margin_ratio'].rolling(12, min_periods=4).mean()
            roll_std = m['margin_ratio'].rolling(12, min_periods=4).std() + 1e-7
            m['margin_ratio_zscore_12w'] = ((m['margin_ratio'] - roll_mean) / roll_std).fillna(0.0)
            m['margin_ratio_level'] = m['margin_ratio']

            df2 = df.copy()
            weekly_cols = ['margin_ratio_level', 'margin_ratio_zscore_12w', 'margin_buy_chg_1w', 'margin_short_chg_1w']
            m_aligned = m[weekly_cols].reindex(df2.index.union(m.index)).sort_index().ffill().reindex(df2.index)
            for c in weekly_cols:
                df2[c] = m_aligned[c].fillna(0.0).values
            pool[t] = df2
        except Exception:
            continue
        if (i + 1) % 50 == 0:
            print(f"  --> {i + 1}/{len(base)} 銘柄 完了")

    with open(MARGIN_POOL_CACHE_PATH, "wb") as f:
        pickle.dump(pool, f)
    print(f"[cache] マージン特徴量プールキャッシュを保存しました: {len(pool)}銘柄 -> {MARGIN_POOL_CACHE_PATH}")
    return pool


def _compute_market_breadth(tickers, calendar_index):
    """POOL_BREADTH_FEATURE_COLS(市場breadth系5特徴量)を、universe_bars.parquetの
    生OHLCVを全銘柄分ループして横断的に集計し、calendar_indexに合わせたdfとして返す
    (2026-09-21追加)。個別銘柄のcompute_stock_features適用済みデータ(get_base_per_ticker_df)
    は使わず生データから直接計算する——breadth指標はcompute_stock_features側の特徴量に
    依存しないため、こちらの方が依存関係が単純になる。"""
    universe_bars = pd.read_parquet(UNIVERSE_BARS_CACHE_PATH)
    universe_bars.index = pd.to_datetime(universe_bars.index).tz_localize(None)

    above_ma25, new_high20, new_low20, breakout_mag, turnover = {}, {}, {}, {}, {}
    for t in tickers:
        if t not in universe_bars.columns.get_level_values(0):
            continue
        sub = universe_bars[t][['Close', 'High', 'Low', 'Volume']].dropna(subset=['Close'])
        if len(sub) < 30:
            continue
        close, high, low, vol = sub['Close'], sub['High'], sub['Low'], sub['Volume']

        ma25 = close.rolling(25).mean()
        above_ma25[t] = (close > ma25).astype(float).where(ma25.notna())

        prior_high20 = high.shift(1).rolling(20).max()
        prior_low20 = low.shift(1).rolling(20).min()
        new_high20[t] = (close > prior_high20).astype(float).where(prior_high20.notna())
        new_low20[t] = (close < prior_low20).astype(float).where(prior_low20.notna())

        raw = np.where(close > prior_high20, close / prior_high20 - 1.0,
                        np.where(close < prior_low20, close / prior_low20 - 1.0, 0.0))
        breakout_mag[t] = pd.Series(raw, index=sub.index).where(prior_high20.notna() & prior_low20.notna())

        turnover[t] = close * vol

    above_ma25_df = pd.DataFrame(above_ma25)
    new_high20_df = pd.DataFrame(new_high20)
    new_low20_df = pd.DataFrame(new_low20)
    breakout_df = pd.DataFrame(breakout_mag)
    turnover_df = pd.DataFrame(turnover)

    def _ratio(bool_df):
        n_valid = bool_df.notna().sum(axis=1).replace(0, np.nan)
        return (bool_df.sum(axis=1) / n_valid)

    def _align(s, fill_value):
        combined = s.reindex(calendar_index.union(s.index)).sort_index().ffill()
        return combined.reindex(calendar_index).fillna(fill_value)

    result = pd.DataFrame(index=calendar_index)
    result['market_above_ma25_ratio'] = _align(_ratio(above_ma25_df), 0.5)
    result['market_newhigh20_ratio'] = _align(_ratio(new_high20_df), 0.0)
    result['market_newlow20_ratio'] = _align(_ratio(new_low20_df), 0.0)
    result['market_breakout_score'] = _align(breakout_df.mean(axis=1), 0.0)

    daily_turnover_5d = turnover_df.sum(axis=1).rolling(5).mean()
    tv_mean60 = daily_turnover_5d.rolling(60, min_periods=20).mean()
    tv_std60 = daily_turnover_5d.rolling(60, min_periods=20).std() + 1e-7
    result['market_turnover_z'] = _align((daily_turnover_5d - tv_mean60) / tv_std60, 0.0)

    return result


def get_full_macro_pool_df(tickers, force_rebuild=False):
    """POOL_MACRO_FEATURE_COLS(VIX/USDJPY/日経5日/TOPIX5日/市場ボラregime)+
    POOL_BREADTH_FEATURE_COLS(市場breadth系5特徴量)を全部追加したmacro_dfを返す
    (load_macro_slim5のNK225版をベースに追加)。tickersは市場breadth特徴量の集計対象
    ユニバース(通常はUNIVERSE_PATHの150銘柄)。"""
    if not force_rebuild and os.path.exists(MACRO_POOL_CACHE_PATH):
        with open(MACRO_POOL_CACHE_PATH, "rb") as f:
            cached = pickle.load(f)
        print(f"[cache] マクロ特徴量プールキャッシュを読み込みました ({MACRO_POOL_CACHE_PATH})")
        return cached

    from modules.macro_features import load_macro_slim5
    macro_df = load_macro_slim5(return_source="nk225")

    market_vol_raw = macro_df['NK_Ret'].abs().rolling(20).mean()
    vol_mean_60 = market_vol_raw.rolling(60, min_periods=20).mean()
    vol_std_60 = market_vol_raw.rolling(60, min_periods=20).std() + 1e-7
    macro_df['market_vol_regime'] = ((market_vol_raw - vol_mean_60) / vol_std_60).fillna(0.0)

    vix = pd.read_csv(os.path.join(BASE_DIR, "data", "vix_fred.csv"))
    vix['observation_date'] = pd.to_datetime(vix['observation_date'])
    vix = vix.set_index('observation_date')['VIXCLS'].dropna()
    vix_chg = vix.pct_change()
    vix_aligned = vix_chg.reindex(macro_df.index.union(vix_chg.index)).sort_index().ffill().reindex(macro_df.index)
    macro_df['vix_overnight_chg'] = vix_aligned.fillna(0.0)

    usdjpy = pd.read_csv(os.path.join(BASE_DIR, "data", "usdjpy_fred.csv"))
    usdjpy['observation_date'] = pd.to_datetime(usdjpy['observation_date'])
    usdjpy = usdjpy.set_index('observation_date')['DEXJPUS'].dropna()
    usdjpy_chg = usdjpy.pct_change()
    usdjpy_aligned = usdjpy_chg.reindex(macro_df.index.union(usdjpy_chg.index)).sort_index().ffill().reindex(macro_df.index)
    macro_df['usdjpy_chg'] = usdjpy_aligned.fillna(0.0)

    macro_df['nk225_ret_5d'] = macro_df['NK_Close'].pct_change(5).fillna(0.0)

    topix = pd.read_parquet(os.path.join(CACHE_DIR, "train_topix_bars.parquet"))
    topix.index = pd.to_datetime(topix.index).tz_localize(None)
    topix_ret_5d = topix['Close'].pct_change(5)
    macro_df['topix_ret_5d'] = topix_ret_5d.reindex(macro_df.index).ffill().fillna(0.0)

    # regime別IC分析(2026-09-18)で符号反転・regime限定シグナルが見られた特徴量の交互作用項。
    # is_low/mid/highは市場20日実現ボラの3分位(全期間の分位点で固定、compute_vol_regime_bucketと同じ定義)
    realized_vol = macro_df['NK_Ret'].abs().rolling(20).mean()
    q1, q2 = realized_vol.quantile([1/3, 2/3])
    is_low = (realized_vol < q1).astype(float)
    is_mid = ((realized_vol >= q1) & (realized_vol < q2)).astype(float)
    is_high = (realized_vol >= q2).astype(float)

    macro_df['cta_momentum_x_mid'] = macro_df['cta_momentum'] * is_mid
    macro_df['cta_net_norm_x_low'] = macro_df['cta_net_norm'] * is_low
    macro_df['market_vol_regime_x_high'] = macro_df['market_vol_regime'] * is_high
    macro_df['usdjpy_chg_x_low'] = macro_df['usdjpy_chg'] * is_low

    breadth_df = _compute_market_breadth(tickers, macro_df.index)
    for c in POOL_BREADTH_FEATURE_COLS:
        macro_df[c] = breadth_df[c]

    with open(MACRO_POOL_CACHE_PATH, "wb") as f:
        pickle.dump(macro_df, f)
    print(f"[cache] マクロ特徴量プールキャッシュを保存しました -> {MACRO_POOL_CACHE_PATH}")
    return macro_df
