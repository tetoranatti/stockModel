"""リターンに基づく銘柄クラスタリング(2026-09-23追加、ユーザー提案)。

TSE33業種による分散制約(select_topn_with_sector_limit)は「同じ業種=分散されている」と
みなすが、実際の値動きの共動性とは必ずしも一致しない。売買代金10億円以上の広いユニバース
(pipeline/build_jquants_cache.pyが出力するdata/cache/prices_*.parquet、直近
FETCH_DAYS=130営業日・約600銘柄)のリターンからクラスタ分けし、(1) TopN選択時の
分散制約、(2) 150銘柄ユニバースの再選定、の両方に使う。

2種類の方式を用意:
    - "static_corr"(初期実装): 60日リターン系列の相関から一発で階層クラスタリング。
      直近半年の値動きが常に強く相関するペアをまとめて拾う一方、1回の相関計算だけに
      依存するため期間の取り方に結果が左右されやすい。
    - "monthly_consensus"(2026-09-23追加、ユーザー提案): 5日・20日・60日リターンを
      横断面で標準化した特徴ベクトルを使い、月ごとに「その月の強弱グループ」でKMeansへ
      分け、複数月にわたって同じグループに入り続けたペアほど近いとみなす合意(consensus)
      クラスタリング。単月のノイズに左右されにくく、半導体テーマのように短期〜中期の
      複数スケールで連動する銘柄群(公式業種分類には現れない)を拾いやすい。
      データソースは fetch_cluster_universe_history.py が作る約2年分の長期キャッシュ
      (data/cache/cluster_daily_bars_raw.parquet、分割調整済み)を優先し、無ければ
      直近6ヶ月分の prices_*.parquet にフォールバックする。

使い方:
    from .cluster_utils import load_cluster_map, select_universe_evenly
    cluster_map = load_cluster_map(method="monthly_consensus")  # {ticker: "C1"...}
    tickers, cluster_map = select_universe_evenly(n_target=150, method="monthly_consensus")
"""
import glob
import os
import pickle
import sys

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

BASE_DIR = r"F:\stockModel"
CACHE_DIR = os.path.join(BASE_DIR, "data", "cache")
CLUSTER_CACHE_PATH = os.path.join(
    r"F:\stockModel\research\cache", "_cache_return_clusters.pkl"
)
MONTHLY_CLUSTER_CACHE_PATH = os.path.join(
    r"F:\stockModel\research\cache", "_cache_monthly_consensus_clusters.pkl"
)
# fetch_cluster_universe_history.pyが作る長期(既定2年・分割調整済み)キャッシュ
LONG_HISTORY_CACHE_PATH = os.path.join(CACHE_DIR, "cluster_daily_bars_raw.parquet")
# pipeline/build_jquants_cache.pyが出力する「直近5日平均売買代金10億円以上・
# ETF/REIT/グロース市場除外後」の現在の適格銘柄リスト(長期キャッシュは
# fetch_daily_all由来で市場全体を含むため、これで絞り込む)
SCREENER_RESULT_PATH = os.path.join(BASE_DIR, "screener_result.csv")

N_CLUSTERS_DEFAULT = 12
RETURN_LOOKBACK_DEFAULT = 60
MIN_OVERLAP = 40  # 相関算出に必要な最小共通サンプル数(60日リターンは重複窓なので緩め)

N_FINAL_CLUSTERS_DEFAULT = 8
N_MONTHLY_BUCKETS_DEFAULT = 8
MONTHLY_RETURN_WINDOWS_DEFAULT = (1, 5, 20, 60)  # 複数horizon(2026-09-23、ユーザー提案)
MIN_SHARED_MONTHS_DEFAULT = 5  # co-occurrence算出に必要な最小共通月数(下落月限定で約10ヶ月分中)

# "all"/"down"/"up": consensusに使う月を市場全体の月次リターン(中央値)で絞り込む
# (2026-09-23、ユーザー提案: 分散投資が本当に効いてほしいのは下落局面で一緒に沈む銘柄を
# 避けたい場面なので、下落月だけでconsensusを取る)
MONTH_FILTER_DEFAULT = "down"
MARKET_RETURN_WINDOW_DEFAULT = 20  # 月間パフォーマンスの代表horizon(概ね1ヶ月相当)

# これ未満のhorizonは、月末1点だとその日のニュース等のノイズが乗りやすいため、
# 月内を等間隔でさかのぼって複数時点サンプリングし平均する(2026-09-23、ユーザー提案)。
# これ以上のhorizonはウィンドウ自体が月内を均しているので月末1点のみで十分。
SHORT_HORIZON_THRESHOLD = 10
SHORT_HORIZON_SAMPLE_STRIDE = 5  # 月内サンプリング間隔(営業日、週次相当)


def _latest_prices_path():
    """data/cache/prices_YYYYMMDD.parquet のうち最新日付のものを返す
    (build_jquants_cache.pyが売買代金10億円以上の全銘柄を出力するキャッシュ、150銘柄
    ユニバースより広い候補プール)。"""
    candidates = sorted(glob.glob(os.path.join(CACHE_DIR, "prices_*.parquet")))
    if not candidates:
        raise FileNotFoundError(
            f"{CACHE_DIR} に prices_*.parquet が見つかりません。"
            "先に pipeline/build_jquants_cache.py を実行してください。"
        )
    return candidates[-1]


def _compute_return_clusters(n_clusters, lookback_days, prices_path=None):
    """{ticker: "C<label>"} と {ticker: 平均売買代金} を返す。"""
    prices_path = prices_path or _latest_prices_path()
    bars = pd.read_parquet(prices_path)
    bars.index = pd.to_datetime(bars.index).tz_localize(None)

    tickers = sorted(bars.columns.get_level_values(0).unique())
    ret_wide = {}
    turnover_mean = {}
    for t in tickers:
        try:
            close = bars[t]["Close"]
            volume = bars[t]["Volume"]
        except KeyError:
            continue
        ret = close.pct_change(lookback_days, fill_method=None)
        if ret.notna().sum() < MIN_OVERLAP:
            continue
        ret_wide[t] = ret
        turnover_mean[t] = (close * volume).mean()

    ret_df = pd.DataFrame(ret_wide)
    corr = ret_df.corr(min_periods=MIN_OVERLAP).fillna(0.0)
    np.fill_diagonal(corr.values, 1.0)

    dist = (1.0 - corr).clip(lower=0.0)
    np.fill_diagonal(dist.values, 0.0)
    condensed = squareform(dist.values, checks=False)

    z = linkage(condensed, method="average")
    labels = fcluster(z, t=n_clusters, criterion="maxclust")

    cluster_map = {t: f"C{lbl}" for t, lbl in zip(corr.columns, labels)}
    liquidity = {t: turnover_mean[t] for t in corr.columns}

    return cluster_map, liquidity


def _qualified_tickers():
    """screener_result.csv(現在の適格銘柄、約590銘柄)を読み込み{'XXXX.T', ...}を返す。
    見つからなければNone(絞り込み無しにフォールバック)。"""
    if not os.path.exists(SCREENER_RESULT_PATH):
        return None
    try:
        df = pd.read_csv(SCREENER_RESULT_PATH, encoding="cp932")
        codes = df.iloc[:, 0].astype(str).str.zfill(4)
        return set(codes + ".T")
    except Exception:
        return None


def _load_close_volume_wide():
    """Close/Volumeの(date x ticker)ワイド形式を返す。fetch_cluster_universe_history.py
    が作る長期履歴キャッシュ(LONG_HISTORY_CACHE_PATH、約2年・分割調整済み)があれば
    優先して使い、無ければ従来のprices_*.parquet(直近6ヶ月分)にフォールバックする。

    長期キャッシュはfetch_daily_all由来で市場全体(約4600銘柄)を含むため、
    screener_result.csv記載の現在の適格銘柄(売買代金10億円以上、約590銘柄)へ
    絞り込む(2026-09-23、ユーザー提案どおり対象を適格銘柄のみに限定)。"""
    if os.path.exists(LONG_HISTORY_CACHE_PATH):
        long_df = pd.read_parquet(LONG_HISTORY_CACHE_PATH)
        long_df["Date"] = pd.to_datetime(long_df["Date"])
        close_wide = long_df.pivot(index="Date", columns="ticker", values="Close")
        volume_wide = long_df.pivot(index="Date", columns="ticker", values="Volume")
    else:
        prices_path = _latest_prices_path()
        bars = pd.read_parquet(prices_path)
        bars.index = pd.to_datetime(bars.index).tz_localize(None)
        tickers = sorted(bars.columns.get_level_values(0).unique())
        close_wide = pd.DataFrame(
            {t: bars[t]["Close"] for t in tickers if "Close" in bars[t].columns}
        )
        volume_wide = pd.DataFrame(
            {t: bars[t]["Volume"] for t in tickers if "Volume" in bars[t].columns}
        )

    qualified = _qualified_tickers()
    if qualified:
        keep_close = [c for c in close_wide.columns if c in qualified]
        keep_volume = [c for c in volume_wide.columns if c in qualified]
        close_wide = close_wide[keep_close]
        volume_wide = volume_wide[keep_volume]
    else:
        print(
            f"[!] {SCREENER_RESULT_PATH} が見つからないため、適格銘柄への絞り込み無しで"
            "全銘柄を対象にします(想定より広いユニバースになります)。"
        )

    return close_wide.sort_index(), volume_wide.sort_index()


def _monthly_feature_row(month_slice, window):
    """ある月・あるhorizonの特徴量(1銘柄1値)を返す。windowがSHORT_HORIZON_THRESHOLD
    未満(短期horizon)の場合、月末1点だけだとその日のニュース等のノイズが乗りやすいため、
    月内をSHORT_HORIZON_SAMPLE_STRIDE営業日おきに月末からさかのぼって複数時点サンプリングし
    平均する(2026-09-23、ユーザー提案)。window以上のhorizonはウィンドウ自体が月内を
    均しているので、従来通り月末1点のみを使う。"""
    if window >= SHORT_HORIZON_THRESHOLD:
        return month_slice.iloc[-1]
    sample_points = month_slice.iloc[::-1].iloc[::SHORT_HORIZON_SAMPLE_STRIDE]
    return sample_points.mean(axis=0)


def _compute_monthly_return_clusters(
    n_monthly_clusters,
    return_windows=MONTHLY_RETURN_WINDOWS_DEFAULT,
    month_filter=MONTH_FILTER_DEFAULT,
    market_return_window=MARKET_RETURN_WINDOW_DEFAULT,
):
    """複数horizon(既定1/5/20/60日)トレーリングリターンを横断面で標準化した特徴ベクトルを
    使い、月ごとにKMeansでn_monthly_clusters個のグループへ分ける(「その月、強かった/
    弱かったグループ」)。2026-09-23、ユーザー提案: 半導体テーマのように短期(数日〜数週間)
    と中期(数ヶ月)の両方で連動する銘柄群を、単一horizonより拾いやすくするため複数horizonを
    組み合わせる。短いhorizonは_monthly_feature_rowで月内複数時点平均によりノイズを抑える。

    month_filter="down"(既定): 市場全体の月次リターン(市場全体でmarket_return_window日
    リターンの横断面中央値)がマイナスの月だけをconsensusの対象にする。分散投資が本当に
    効いてほしいのは下落局面で一緒に沈む銘柄を避けたい場面のため(2026-09-23、ユーザー提案)。
    "up"なら上昇月のみ、"all"なら全月を使う(従来の挙動)。

    戻り値: {month_period: {ticker: bucket_id}}(bucket_idはそのクラスタの標準化リターン
    平均昇順、0=最も弱いグループ)。"""
    from sklearn.cluster import KMeans

    close_df, _ = _load_close_volume_wide()
    ret_dfs = [close_df.pct_change(w, fill_method=None) for w in return_windows]
    months = close_df.index.to_period("M")
    market_ret_df = close_df.pct_change(market_return_window, fill_method=None)

    monthly_clusters = {}
    for month in sorted(months.unique()):
        mask = months == month

        if month_filter != "all":
            market_ret = market_ret_df.loc[mask].iloc[-1].median()
            if pd.isna(market_ret):
                continue
            if month_filter == "down" and market_ret >= 0:
                continue
            if month_filter == "up" and market_ret < 0:
                continue

        rows = [
            _monthly_feature_row(df.loc[mask], w)
            for w, df in zip(return_windows, ret_dfs)
        ]
        feat = pd.concat(rows, axis=1)
        feat.columns = [f"ret_{w}d" for w in return_windows]
        feat = feat.dropna()
        if len(feat) < n_monthly_clusters * 3:
            continue  # サンプル数が少なすぎる月(データ開始直後等)はスキップ

        # horizon間でリターンの典型的な大きさが異なる(60日 > 20日 > 5日)ため、
        # 横断面z-score標準化してからKMeansへ渡す
        mean = feat.mean(axis=0)
        std = feat.std(axis=0).replace(0.0, 1.0)
        feat_z = (feat - mean) / std

        km = KMeans(n_clusters=n_monthly_clusters, n_init=10, random_state=42)
        labels = km.fit_predict(feat_z.values)

        # クラスタ番号がKMeansの内部初期化依存でバラつかないよう、
        # 3horizon平均の中心値昇順で振り直す(=総合的な強弱の順)
        order = np.argsort(km.cluster_centers_.mean(axis=1))
        remap = {old: new for new, old in enumerate(order)}
        monthly_clusters[month] = {t: remap[lbl] for t, lbl in zip(feat_z.index, labels)}

    return monthly_clusters


def _compute_consensus_clusters(monthly_clusters, n_final_clusters, min_shared_months):
    """月次クラスタ割当の一致率(co-occurrence)を銘柄間の「近さ」とみなし、最終的に
    n_final_clusters個へまとめ直す(2026-09-23、ユーザー提案「銘柄の分布と共通部分から
    分類」)。複数月にわたって同じ月内グループに入り続けたペアほど距離を近くする。"""
    all_tickers = sorted(set().union(*(set(m.keys()) for m in monthly_clusters.values())))
    idx = {t: i for i, t in enumerate(all_tickers)}
    n = len(all_tickers)

    same_count = np.zeros((n, n))
    shared_count = np.zeros((n, n))

    for assign in monthly_clusters.values():
        items = list(assign.items())
        for i in range(len(items)):
            ti, li = items[i]
            a = idx[ti]
            for j in range(i, len(items)):
                tj, lj = items[j]
                b = idx[tj]
                shared_count[a, b] += 1
                shared_count[b, a] += 1
                if li == lj:
                    same_count[a, b] += 1
                    same_count[b, a] += 1

    with np.errstate(invalid="ignore", divide="ignore"):
        co_occurrence = np.where(shared_count > 0, same_count / shared_count, 0.5)

    # 共有月数が足りない銘柄ペア(新規上場直後等)は中立(0.5)扱いにして偽の近さ/遠さを避ける
    co_occurrence = np.where(shared_count >= min_shared_months, co_occurrence, 0.5)
    np.fill_diagonal(co_occurrence, 1.0)

    dist = 1.0 - co_occurrence
    dist = (dist + dist.T) / 2.0  # 数値誤差での非対称を解消
    np.fill_diagonal(dist, 0.0)

    condensed = squareform(dist, checks=False)
    z = linkage(condensed, method="average")
    labels = fcluster(z, t=n_final_clusters, criterion="maxclust")

    cluster_map = {t: f"C{lbl}" for t, lbl in zip(all_tickers, labels)}
    return cluster_map


def compute_monthly_consensus_cluster_map(
    n_final_clusters=N_FINAL_CLUSTERS_DEFAULT,
    n_monthly_clusters=N_MONTHLY_BUCKETS_DEFAULT,
    return_windows=MONTHLY_RETURN_WINDOWS_DEFAULT,
    min_shared_months=MIN_SHARED_MONTHS_DEFAULT,
    month_filter=MONTH_FILTER_DEFAULT,
    market_return_window=MARKET_RETURN_WINDOW_DEFAULT,
    force_rebuild=False,
):
    """複数horizonリターンによる月次クラスタリング+複数月の合意(consensus)から
    最終クラスタマップを返す({ticker: "C<label>"})。month_filter="down"(既定)なら
    市場全体が下落した月だけでconsensusを取る(2026-09-23、ユーザー提案)。"""
    return_windows = tuple(return_windows)
    cache_key = (
        n_final_clusters,
        n_monthly_clusters,
        return_windows,
        min_shared_months,
        month_filter,
        market_return_window,
    )
    if not force_rebuild and os.path.exists(MONTHLY_CLUSTER_CACHE_PATH):
        with open(MONTHLY_CLUSTER_CACHE_PATH, "rb") as f:
            cached = pickle.load(f)
        if cached.get("key") == cache_key:
            return cached["cluster_map"]

    monthly_clusters = _compute_monthly_return_clusters(
        n_monthly_clusters, return_windows, month_filter, market_return_window
    )
    cluster_map = _compute_consensus_clusters(
        monthly_clusters, n_final_clusters, min_shared_months
    )
    with open(MONTHLY_CLUSTER_CACHE_PATH, "wb") as f:
        pickle.dump(
            {
                "key": cache_key,
                "cluster_map": cluster_map,
                "monthly_clusters": monthly_clusters,
            },
            f,
        )
    return cluster_map


def _liquidity_from_prices():
    """{ticker: 平均売買代金} を返す(_load_close_volume_wideと同じデータソース選択)。"""
    close_df, volume_df = _load_close_volume_wide()
    turnover = (close_df * volume_df).mean(axis=0)
    return turnover.to_dict()


def load_cluster_map(
    method="monthly_consensus",
    n_clusters=N_CLUSTERS_DEFAULT,
    lookback_days=RETURN_LOOKBACK_DEFAULT,
    n_final_clusters=N_FINAL_CLUSTERS_DEFAULT,
    n_monthly_clusters=N_MONTHLY_BUCKETS_DEFAULT,
    return_windows=MONTHLY_RETURN_WINDOWS_DEFAULT,
    min_shared_months=MIN_SHARED_MONTHS_DEFAULT,
    month_filter=MONTH_FILTER_DEFAULT,
    market_return_window=MARKET_RETURN_WINDOW_DEFAULT,
    force_rebuild=False,
):
    """ticker('XXXX.T') -> クラスタラベル('C1'等) の辞書を返す(_load_sector_mapと同じ形で
    diagnostics.pyのTICKER_TO_SECTORと並行利用できる)。

    method="static_corr": 60日リターン相関の一発階層クラスタリング(初期実装)。
    method="monthly_consensus"(既定): 複数horizonリターンの月次KMeans+複数月合意クラスタリング。
    """
    if method == "monthly_consensus":
        return compute_monthly_consensus_cluster_map(
            n_final_clusters=n_final_clusters,
            n_monthly_clusters=n_monthly_clusters,
            return_windows=return_windows,
            min_shared_months=min_shared_months,
            month_filter=month_filter,
            market_return_window=market_return_window,
            force_rebuild=force_rebuild,
        )
    if method != "static_corr":
        raise ValueError(f"未対応のmethodです: {method}")

    cache_key = (n_clusters, lookback_days)
    if not force_rebuild and os.path.exists(CLUSTER_CACHE_PATH):
        with open(CLUSTER_CACHE_PATH, "rb") as f:
            cached = pickle.load(f)
        if cached.get("key") == cache_key:
            return cached["cluster_map"]

    cluster_map, liquidity = _compute_return_clusters(n_clusters, lookback_days)
    with open(CLUSTER_CACHE_PATH, "wb") as f:
        pickle.dump(
            {"key": cache_key, "cluster_map": cluster_map, "liquidity": liquidity},
            f,
        )
    return cluster_map


def select_universe_evenly(
    n_target=150,
    method="monthly_consensus",
    n_clusters=N_CLUSTERS_DEFAULT,
    lookback_days=RETURN_LOOKBACK_DEFAULT,
    n_final_clusters=N_FINAL_CLUSTERS_DEFAULT,
    n_monthly_clusters=N_MONTHLY_BUCKETS_DEFAULT,
    return_windows=MONTHLY_RETURN_WINDOWS_DEFAULT,
    min_shared_months=MIN_SHARED_MONTHS_DEFAULT,
    month_filter=MONTH_FILTER_DEFAULT,
    market_return_window=MARKET_RETURN_WINDOW_DEFAULT,
):
    """クラスタごとに売買代金上位から順に1銘柄ずつ、クラスタを一巡させながらn_target銘柄選ぶ。
    小さいクラスタは先に枯渇するため、完全均等ではなく「クラスタサイズに応じた公平配分」になる
    (少数クラスタに過剰配分して流動性の低い銘柄まで拾ってしまうのを防ぐため)。

    戻り値: (選定tickerのリスト(流動性降順の巡回順で並ぶ), cluster_map)
    """
    if method == "monthly_consensus":
        cluster_map = compute_monthly_consensus_cluster_map(
            n_final_clusters=n_final_clusters,
            n_monthly_clusters=n_monthly_clusters,
            return_windows=return_windows,
            min_shared_months=min_shared_months,
            month_filter=month_filter,
            market_return_window=market_return_window,
        )
        liquidity = _liquidity_from_prices()
    elif method == "static_corr":
        cluster_map, liquidity = _compute_return_clusters(n_clusters, lookback_days)
    else:
        raise ValueError(f"未対応のmethodです: {method}")

    by_cluster = {}
    for t, c in cluster_map.items():
        by_cluster.setdefault(c, []).append(t)
    for c in by_cluster:
        by_cluster[c] = sorted(
            by_cluster[c], key=lambda t: liquidity.get(t, 0.0), reverse=True
        )

    selected = []
    cluster_names = sorted(by_cluster.keys())
    idx_per_cluster = {c: 0 for c in cluster_names}
    while len(selected) < n_target:
        progressed = False
        for c in cluster_names:
            i = idx_per_cluster[c]
            members = by_cluster[c]
            if i < len(members):
                selected.append(members[i])
                idx_per_cluster[c] = i + 1
                progressed = True
                if len(selected) >= n_target:
                    break
        if not progressed:
            break

    return selected, cluster_map


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")
        sys.stderr.reconfigure(line_buffering=True, encoding="utf-8")
    except Exception:
        pass

    tickers, cmap = select_universe_evenly()
    counts = pd.Series(cmap).loc[tickers].value_counts().sort_index()
    print(f"[*] 選定銘柄数: {len(tickers)}")
    print("[*] クラスタ別採用数:")
    print(counts.to_string())

    out_path = os.path.join(BASE_DIR, "universe_150_tickers_clustered.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(tickers) + "\n")
    print(f"[+] 書き出し完了: {out_path}")
