"""クラスタリング用の長期(既定2年)市場全体日次四本値キャッシュを取得する
(2026-09-23追加、ユーザー提案)。

pipeline/build_jquants_cache.py PART1と同じ fetch_daily_all(1日・全銘柄一括取得)を
再利用するが、以下の理由で本番の data/cache/daily_screening_bars_raw.parquet は
シードに使わず、全期間を分割調整済み(Adj*)値で新規取得する:

  - 本番キャッシュは未調整の生値(O/H/L/C/Vo)を保存している(6ヶ月程度の窓では
    株式分割の影響が出にくく問題化していなかった)。
  - 2年分に伸ばすと株式分割を挟む銘柄が出てきて、生値のままリターンを計算すると
    分割日に不連続な異常値が発生し、クラスタリング結果を歪める。
  - そのためAdj*(分割・配当調整済み)フィールドを使い、生値キャッシュとは
    別ファイルとして完全に独立させる(本番パイプラインには一切影響しない)。

差分更新対応: 既にこのキャッシュにある日は再取得しない。2回目以降の実行は
新しく増えた営業日分だけ取得する。

実行方法:
    python research/stage3_exp/fetch_cluster_universe_history.py
"""
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

sys.path.insert(0, r"F:\stockModel")
from pipeline.build_jquants_cache import (
    CACHE_DIR,
    FETCH_WORKERS,
    JQUANTS_API_KEY,
    RateLimiter,
    REQUESTS_PER_MINUTE,
    fetch_daily_all,
    get_recent_business_days,
)

OUT_PATH = os.path.join(CACHE_DIR, "cluster_daily_bars_raw.parquet")
YEARS_BACK = 2
# 平日ベースの概算(約261営業日/年)+ 祝日分の余裕
N_TARGET_DAYS = int(261 * YEARS_BACK * 1.05)

ADJ_COL_MAP = {
    "AdjO": "Open",
    "AdjH": "High",
    "AdjL": "Low",
    "AdjC": "Close",
    "AdjVo": "Volume",
}
TICKER_CODE_PATTERN = r"^[0-9]{4}0?$|^[0-9]{3}[A-Z]0?$"


def _parse_records(records):
    """fetch_daily_allの生レコード(1日分・全銘柄)を分割調整済み四本値のDataFrameに変換する。"""
    if not records:
        return None
    raw_df = pd.DataFrame(records)

    missing_adj = [c for c in ADJ_COL_MAP if c not in raw_df.columns]
    if missing_adj:
        return None

    raw_df["Date"] = pd.to_datetime(raw_df["Date"])
    raw_df = raw_df[raw_df["Code"].astype(str).str.match(TICKER_CODE_PATTERN)]
    raw_df["ticker"] = raw_df["Code"].astype(str).str.slice(0, 4) + ".T"
    raw_df = raw_df.rename(columns=ADJ_COL_MAP)

    keep_cols = ["Date", "ticker", "Open", "High", "Low", "Close", "Volume"]
    out = raw_df[keep_cols].dropna()
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        out[c] = out[c].astype(float)
    return out


def main():
    try:
        sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")
        sys.stderr.reconfigure(line_buffering=True, encoding="utf-8")
    except Exception:
        pass

    print(f"[*] 直近{YEARS_BACK}年分(概算{N_TARGET_DAYS}営業日)を対象にします")
    target_days = get_recent_business_days(N_TARGET_DAYS)

    existing_df = None
    if os.path.exists(OUT_PATH):
        existing_df = pd.read_parquet(OUT_PATH)
        print(
            f"[*] 既存のクラスタ用ロング履歴キャッシュを検出: {OUT_PATH} "
            f"({existing_df['Date'].nunique()}日分, "
            f"{existing_df['Date'].min().date()}〜{existing_df['Date'].max().date()})"
        )

    cached_days = set()
    if existing_df is not None and not existing_df.empty:
        cached_days = set(existing_df["Date"].dt.strftime("%Y%m%d").unique())

    missing_days = [d for d in target_days if d not in cached_days]
    print(
        f"[*] 対象{len(target_days)}日中 {len(missing_days)}日が未取得。"
        f"{FETCH_WORKERS}並列でAPIから取得します(数分かかります)..."
    )

    if not missing_days:
        print("[*] 新規取得の必要なし。既存キャッシュのままです。")
        return

    session = requests.Session()
    session.headers.update({"x-api-key": JQUANTS_API_KEY})
    rate_limiter = RateLimiter(max_calls=REQUESTS_PER_MINUTE)

    new_parts = []
    empty_days = []
    with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as executor:
        futures = {
            executor.submit(fetch_daily_all, session, d, 3, rate_limiter): d
            for d in missing_days
        }
        done = 0
        for future in as_completed(futures):
            d = futures[future]
            records = future.result()
            parsed = _parse_records(records)
            if parsed is not None and not parsed.empty:
                new_parts.append(parsed)
            else:
                empty_days.append(d)  # 休場日等、データが無い日
            done += 1
            if done % 25 == 0 or done == len(missing_days):
                print(f"  --> {done}/{len(missing_days)} 日 完了")

    if empty_days:
        print(f"[*] データが無かった日(休場日等): {len(empty_days)}日")

    parts = ([existing_df] if existing_df is not None else []) + new_parts
    if not parts:
        print("[-] 取得できたデータがありません。")
        return

    combined = pd.concat(parts, ignore_index=True)
    combined = combined.drop_duplicates(subset=["Date", "ticker"], keep="last")
    combined = combined.sort_values(["Date", "ticker"]).reset_index(drop=True)
    combined.to_parquet(OUT_PATH)

    print(f"[+] 保存完了: {OUT_PATH}")
    print(
        f"    日数: {combined['Date'].nunique()}日 "
        f"({combined['Date'].min().date()} 〜 {combined['Date'].max().date()})"
    )
    print(f"    銘柄数: {combined['ticker'].nunique()}")


if __name__ == "__main__":
    main()
