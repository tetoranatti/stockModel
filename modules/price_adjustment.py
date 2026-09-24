# modules/price_adjustment.py
# J-Quantsの分割調整済み(Adj*)価格は「本日時点の発行済株数」を基準に過去へ遡って
# 再計算される仕組みだが、build_jquants_cache.pyのキャッシュは差分更新(未取得日だけ
# APIから取得し、取得済みの過去日はそのまま使い回す)方式のため、分割が起きると
# 既にキャッシュ済みの過去日が旧基準のAdj値のまま取り残される。結果として分割日の
# 前後で価格が不連続にジャンプするバグが生じる(2026-09-24、7649のチャートで発覚。
# 2026-08-27→08-28で終値が2678円→1321円、比率0.493≈1:2分割のジャンプ)。
#
# このモジュールは、既にキャッシュされたOHLCV時系列に対して事後的に同じ不整合を検出し、
# 分割日より前の全行を遡及的にスケールし直すための共通ロジック。
# pipeline/build_jquants_cache.py(通常運用の毎回の差分更新時)と
# scripts/repair_split_adjustment_cache.py(既存キャッシュの一括修復)の両方から使う。
import numpy as np
import pandas as pd

# 日本市場で実際に使われる分割・併合比率のみ(1.2倍/0.833倍のような「にじみ」の大きい
# 比率は普段の値動き(特に値幅制限いっぱいの日)と衝突しやすく、実データで605銘柄を
# 走査したところ257件もの誤検出を引き起こしたため候補から除外した)。close-to-closeの
# 変化比率がこのいずれかに2%以内で一致する場合のみ「分割由来の不整合ジャンプ」の候補
# とみなす。一致しなければ実際の値動き(暴落・急騰・TOB思惑等)として何もしない
# (誤って実際の価格変動を潰さないための安全弁)。
# 実データ(3年分・全銘柄でのバックフィル後)で検証したところ、本物の分割はほぼ全て
# 理論比率から2〜5%ずれていた(分割日の終値には分割そのものの比率に加えてその日の
# 通常の値動きも乗るため)。許容誤差を2%のままにすると、9434.T(ソフトバンク、
# 2024-09-27に1:10分割、実際の比率0.0968で理論値0.1から3.2%ずれ)のような実例を
# 取りこぼす。5%まで許容を広げても、価格下限(MIN_PRICE_FOR_SPLIT_DETECTION)と
# 反転ペア除去の2つの安全弁で誤検出は抑えられており、5%までの候補は分割の実施日が
# 日本企業の決算期末(3月末・9月末等)に集中する・出来高が分割日に急増する等、
# 実際の分割と整合する特徴を示すことを確認済み。
SPLIT_RATIO_CANDIDATES = [
    1 / 10, 1 / 5, 1 / 4, 1 / 3, 1 / 2, 1 / 1.5,
    1.5, 2, 3, 4, 5, 10,
]
SPLIT_RATIO_TOLERANCE = 0.05

# 低位株(値がさの小さい銘柄)は除外する。理由は2つ: (1) 呼値が粗く、日々の値動きが
# 偶然「綺麗な比率」になりやすい(実データで2743.Tが1円刻みで4→3→2→2→1円と推移した
# だけで0.667倍・0.5倍の「分割」に見えた)。(2) 値幅制限は円建て定額なので、株価が
# 100〜160円あたりの銘柄はストップ高/安が偶然ちょうど1.5倍前後になりやすい(実データで
# 100→150円、103→153円等、複数銘柄で同じパターンが多発した)。3年分・全銘柄を
# 走査した結果、この価格帯以下をturnoverチェック(後述、廃止)より確実に除外できた。
MIN_PRICE_FOR_SPLIT_DETECTION = 300.0

# 分割は一方向・恒久的な変化であり、数営業日〜数週間で逆比率に「戻る」ことは無い。
# 実データにはそういう往復(例: 6731.Tが0.5倍→2倍→0.5倍→2倍と数日おきに反復)が
# あり、これは分割ではなく単なる乱高下(値がさの薄商い銘柄等)だったため、近い時期に
# 互いに逆比率のジャンプが見つかった場合はペアごと分割候補から除外する。
REVERSAL_WINDOW_DAYS = 40
REVERSAL_PRODUCT_TOLERANCE = 0.05


def find_split_ratio(ratio):
    """ratioが既知の分割比率に近ければその比率を返す。一致しなければNone。"""
    if ratio is None or ratio <= 0:
        return None
    for cand in SPLIT_RATIO_CANDIDATES:
        if abs(ratio - cand) / cand <= SPLIT_RATIO_TOLERANCE:
            return cand
    return None


def backadjust_ticker_splits(df, price_cols=('Open', 'High', 'Low', 'Close'), volume_col='Volume'):
    """1銘柄分のOHLCV(DatetimeIndex)に対し、分割由来と判定できる不連続ジャンプを検出し、
    それより古い全行を検出比率でスケールし直す(価格は比率倍、出来高は逆数倍)。

    境界の検出は必ず「未調整の生Close」同士の比較だけで行う(調整後の値を使って
    隣の日と比較すると、一度調整した境界のすぐ隣がまた"ジャンプして見える"ため、
    存在しない分割を連鎖的に誤検出してしまう回帰バグがあった。検出を先に全て
    生の値だけで済ませ、スケールの適用は最後に累積してまとめて行うことで避けている)。
    複数回の本物の分割があっても、それぞれ独立に検出され、古い区間ほど正しく
    累積(多重)補正される。

    戻り値: (調整後df, [(検出日, 比率), ...])  detectedが空なら調整不要だったということ。
    """
    df = df.sort_index()
    n = len(df)
    if n < 2 or 'Close' not in df.columns:
        return df, []

    raw_close = df['Close'].to_numpy(dtype=float)

    # 1. 境界検出: 生のclose-to-close比率のみを見る(調整後の値は一切参照しない)。
    #    低位株(MIN_PRICE_FOR_SPLIT_DETECTION未満)は呼値・値幅制限の偶然一致が
    #    多いため除外する。
    boundaries = []  # [(i, ratio), ...] 「位置iより古い行にratioを掛ける」という意味
    for i in range(n - 1, 0, -1):
        prev_raw = raw_close[i - 1]
        curr_raw = raw_close[i]
        # 高い方の価格だけ閾値を満たせばよい(安い方は要求しない)。1:10分割のように
        # 分割後の株価が正当に閾値未満まで下がるケースを誤って除外しないため
        # (実データで9434.Tの1:10分割、1976.5円→191.3円が弾かれていたバグ)。
        if max(prev_raw, curr_raw) < MIN_PRICE_FOR_SPLIT_DETECTION:
            continue
        matched = find_split_ratio(curr_raw / prev_raw)
        if matched is None:
            continue
        boundaries.append((i, matched))

    if not boundaries:
        return df, []

    # 2. 反転ペアの除去: 近い時期(REVERSAL_WINDOW_DAYS営業日以内)に互いに逆比率の
    #    境界が見つかった場合、本物の分割ではない乱高下とみなして両方とも除外する。
    boundaries.sort(key=lambda x: x[0])
    keep = [True] * len(boundaries)
    for a in range(len(boundaries)):
        i_a, ratio_a = boundaries[a]
        for b in range(a + 1, len(boundaries)):
            i_b, ratio_b = boundaries[b]
            if i_b - i_a > REVERSAL_WINDOW_DAYS:
                break
            if abs(ratio_a * ratio_b - 1.0) < REVERSAL_PRODUCT_TOLERANCE:
                keep[a] = False
                keep[b] = False
    boundaries = [b for b, k in zip(boundaries, keep) if k]

    if not boundaries:
        return df, []

    # 3. 各行の累積スケールをまとめて計算してから、最後に一度だけ適用する
    scale = np.ones(n, dtype=float)
    for i, ratio in boundaries:
        scale[:i] *= ratio

    out_df = df.copy()
    for c in price_cols:
        if c in out_df.columns:
            out_df[c] = out_df[c].to_numpy(dtype=float) * scale
    if volume_col in out_df.columns:
        out_df[volume_col] = out_df[volume_col].to_numpy(dtype=float) / scale

    detected = [(df.index[i], ratio) for i, ratio in boundaries]
    return out_df, detected


def backadjust_long_format(df, date_col='Date', ticker_col='ticker',
                            price_cols=('Open', 'High', 'Low', 'Close'), volume_col='Volume'):
    """Date/ticker/OHLCVの縦持ち(1行=1銘柄1日)DataFrame全体に、銘柄ごとに
    backadjust_ticker_splits()を適用する。build_jquants_cache.pyのcombined_sub_df
    (daily_screening_bars_raw.parquet)用。

    戻り値: (調整後df, [(ticker, 検出日, 比率), ...])
    """
    parts = []
    all_detected = []
    for t, sub in df.groupby(ticker_col, sort=False):
        sub = sub.set_index(date_col)
        adjusted, detected = backadjust_ticker_splits(sub, price_cols, volume_col)
        for dt, ratio in detected:
            all_detected.append((t, dt, ratio))
        parts.append(adjusted.reset_index())
    result = pd.concat(parts, ignore_index=True) if parts else df
    return result, all_detected


def backadjust_wide_format(df, price_cols=('Open', 'High', 'Low', 'Close'), volume_col='Volume'):
    """(ticker, field)の横持ちMultiIndex列DataFrame全体に、銘柄ごとに
    backadjust_ticker_splits()を適用する。build_jquants_cache.pyのall_train_df
    (train_universe_bars.parquet)やprices_*.parquet用。dfは呼び出し元で書き換えられる。

    戻り値: (調整後df, [(ticker, 検出日, 比率), ...])
    """
    tickers = sorted(set(df.columns.get_level_values(0)))
    all_detected = []
    for t in tickers:
        if (t, 'Close') not in df.columns:
            continue
        sub = df[t]
        adjusted, detected = backadjust_ticker_splits(sub, price_cols, volume_col)
        for dt, ratio in detected:
            all_detected.append((t, dt, ratio))
        for c in adjusted.columns:
            df[(t, c)] = adjusted[c]
    return df, all_detected
