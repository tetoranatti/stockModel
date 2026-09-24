"""学習銘柄拡大の第一段: 現行150銘柄に加えて、適格銘柄(screener_result.csv、
売買代金10億円以上)の中から流動性上位のものを追加し、300銘柄ユニバースを作る
(2026-09-23、ユーザー提案)。

現行150銘柄はそのまま維持し(差し替えではなく追加)、新規追加分だけ
fetch_ticker_jquants(本番のtrain_universe_bars.parquet生成と同じper-ticker API)で
過去3年分を個別バックフィルする。

背景: cluster_diversification_2026-09-23の「クラスタ均等150銘柄への差し替え」実験は
fetch_daily_all由来のcluster_daily_bars_raw.parquet(ほとんどの銘柄が2024-08-19以降の
約2年分)を流用したため、本番の3年分(fetch_ticker_jquants由来)より短く、
学習サンプル減少・split date後ずれという交絡を生んでいた。今回は新規銘柄も
fetch_ticker_jquantsで揃えることでこの交絡を避ける。

本番ファイル(universe_150_tickers.txt, data/cache/train_universe_bars.parquet)は
一切変更しない。出力は全てuniverse_300_tickers.txt / train_universe_bars_expanded300.parquet
という別名の研究用ファイル。

実行方法:
    python research/stage3_exp/backfill_expanded_universe.py
"""
import datetime
import os
import pickle
import sys
import time

sys.path.insert(0, r"F:\stockModel")
sys.path.insert(0, r"F:\stockModel\research")

try:
    sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")
    sys.stderr.reconfigure(line_buffering=True, encoding="utf-8")
except Exception:
    pass

import pandas as pd
import requests

from pipeline.build_jquants_cache import (
    JQUANTS_API_KEY,
    RateLimiter,
    REQUESTS_PER_MINUTE,
    YEARS_BACK,
    fetch_ticker_jquants,
)

BASE_DIR = r"F:\stockModel"
CACHE_DIR = os.path.join(BASE_DIR, "data", "cache")
CURRENT_UNIVERSE_PATH = os.path.join(BASE_DIR, "universe_150_tickers.txt")
CURRENT_BARS_PATH = os.path.join(CACHE_DIR, "train_universe_bars.parquet")
LONG_HISTORY_CACHE_PATH = os.path.join(CACHE_DIR, "cluster_daily_bars_raw.parquet")
SCREENER_RESULT_PATH = os.path.join(BASE_DIR, "screener_result.csv")

EXPANDED_UNIVERSE_PATH = os.path.join(BASE_DIR, "universe_300_tickers_expanded.txt")
EXPANDED_BARS_PATH = os.path.join(CACHE_DIR, "train_universe_bars_expanded300.parquet")
NEW_TICKER_RAW_CACHE_PATH = os.path.join(
    r"F:\stockModel\research\cache", "_cache_expanded300_new_ticker_bars_raw.pkl"
)
N_TARGET_TOTAL = 300


def _load_current_150():
    with open(CURRENT_UNIVERSE_PATH, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _qualified_tickers():
    df = pd.read_csv(SCREENER_RESULT_PATH, encoding="cp932")
    codes = df.iloc[:, 0].astype(str).str.zfill(4)
    return set(codes + ".T")


def _liquidity_ranking(qualified):
    """cluster_daily_bars_raw.parquet(既存キャッシュ、約2年分)から平均売買代金を算出。
    追加銘柄の選定(流動性上位)にのみ使う値なので、この用途では2年平均で十分。"""
    long_df = pd.read_parquet(LONG_HISTORY_CACHE_PATH)
    long_df = long_df[long_df["ticker"].isin(qualified)].copy()
    turnover = (
        long_df.assign(to=long_df["Close"] * long_df["Volume"])
        .groupby("ticker")["to"]
        .mean()
        .sort_values(ascending=False)
    )
    return turnover


def build_expanded_universe_list():
    """現行150銘柄 + 適格銘柄のうち現行150に無い銘柄を流動性上位から
    N_TARGET_TOTAL - 150 件追加したリストを返す(追加のみ、差し替え無し)。"""
    current150 = _load_current_150()
    current_set = set(current150)
    qualified = _qualified_tickers()

    turnover = _liquidity_ranking(qualified)
    candidates = [t for t in turnover.index if t not in current_set]
    n_new = N_TARGET_TOTAL - len(current150)
    new_tickers = candidates[:n_new]

    print(f"[*] 現行ユニバース: {len(current150)}銘柄")
    print(f"[*] 適格銘柄(screener_result.csv): {len(qualified)}銘柄")
    print(f"[*] 新規追加(流動性上位、現行と重複無し): {len(new_tickers)}銘柄")

    expanded = current150 + new_tickers
    with open(EXPANDED_UNIVERSE_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(expanded) + "\n")
    print(f"[+] 拡大ユニバース書き出し完了: {EXPANDED_UNIVERSE_PATH} ({len(expanded)}銘柄)")
    return current150, new_tickers


def backfill_new_tickers(new_tickers):
    """新規銘柄のみ、fetch_ticker_jquantsで過去YEARS_BACK年分を個別取得する
    (本番のtrain_universe_bars.parquet生成ロジックと同じper-ticker APIを使い、
    3年分の履歴の長さを現行150銘柄と揃える)。"""
    session = requests.Session()
    session.headers.update({"x-api-key": JQUANTS_API_KEY})
    rate_limiter = RateLimiter(max_calls=REQUESTS_PER_MINUTE)

    today_d = datetime.date.today()
    end_date = today_d.strftime("%Y-%m-%d")
    full_start_date = (today_d - datetime.timedelta(days=365 * YEARS_BACK)).strftime("%Y-%m-%d")

    print(f"[*] 新規{len(new_tickers)}銘柄をバックフィル中 ({full_start_date} 〜 {end_date})...")
    t0 = time.time()
    new_dfs = {}
    for i, t in enumerate(new_tickers):
        df = fetch_ticker_jquants(session, t, full_start_date, end_date, rate_limiter)
        if df is not None and not df.empty:
            new_dfs[t] = df
        if (i + 1) % 25 == 0 or (i + 1) == len(new_tickers):
            print(f"  --> {i + 1}/{len(new_tickers)} 銘柄完了")

    missing = sorted(set(new_tickers) - set(new_dfs.keys()))
    if missing:
        print(f"[!] 取得失敗/データ無し({len(missing)}件、除外): {missing}")
    print(f"[+] バックフィル完了: {len(new_dfs)}銘柄, 実時間={time.time() - t0:.1f}秒")

    with open(NEW_TICKER_RAW_CACHE_PATH, "wb") as f:
        pickle.dump(new_dfs, f)
    print(f"[+] 生データを退避保存(マージ失敗時の再取得回避用): {NEW_TICKER_RAW_CACHE_PATH}")
    return new_dfs


def merge_and_save(current150, new_dfs):
    """現行のtrain_universe_bars.parquet(3年分)と新規バックフィル分を結合し、
    別名(train_universe_bars_expanded300.parquet)で保存する(本番キャッシュは変更しない)。

    注意: train_universe_bars.parquetは差分更新がpd.concat(axis=0)(行方向)で
    行われているため、過去にuniverse_150_tickers.txtへ載っていたが現在は外れた
    銘柄の列も(その銘柄がユニバースに居た当時の日付分だけ)残存している
    (実測: 現行150銘柄に対しファイル全体は534銘柄分の列を含んでいた)。
    今回の新規追加候補(現行150銘柄に「今」含まれない適格銘柄)の一部が、この
    残存列と重複してto_parquetがDuplicate column namesで失敗したため、
    current_dfは現行150銘柄の列だけに絞り込んでから結合する。"""
    current_df = pd.read_parquet(CURRENT_BARS_PATH)
    current_df = current_df[[c for c in current_df.columns if c[0] in set(current150)]]
    new_df = pd.concat(new_dfs, axis=1)
    merged = pd.concat([current_df, new_df], axis=1)
    merged = merged.sort_index(axis=1)
    merged.to_parquet(EXPANDED_BARS_PATH)

    first_dates = {}
    for t in new_dfs:
        first_dates[t] = new_dfs[t].index.min()
    print(f"[+] 拡大ユニバース価格キャッシュ保存完了: {EXPANDED_BARS_PATH}")
    print(f"    現行150銘柄: {current_df.index.min().date()} 〜 {current_df.index.max().date()}")
    print(
        f"    新規銘柄の履歴開始日: 最古={min(first_dates.values()).date()}, "
        f"最新(最も遅い)={max(first_dates.values()).date()}"
    )


def main():
    current150, new_tickers = build_expanded_universe_list()
    if os.path.exists(NEW_TICKER_RAW_CACHE_PATH):
        print(f"[*] 退避済みの生データを再利用します: {NEW_TICKER_RAW_CACHE_PATH}")
        with open(NEW_TICKER_RAW_CACHE_PATH, "rb") as f:
            new_dfs = pickle.load(f)
    else:
        new_dfs = backfill_new_tickers(new_tickers)
    merge_and_save(current150, new_dfs)


if __name__ == "__main__":
    main()
