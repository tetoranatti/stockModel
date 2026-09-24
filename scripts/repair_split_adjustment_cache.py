# scripts/repair_split_adjustment_cache.py
# 2026-09-24、7649のチャートに分割由来の不連続ジャンプが発覚(pipeline/build_jquants_cache.py
# の差分更新キャッシュが分割を遡及調整できていなかった)。恒久対応は
# pipeline/build_jquants_cache.py に組み込み済み(次回のキャッシュ更新から自動修復される)だが、
# 既に保存済みの当日キャッシュ(prices/daily_screening_bars_raw/train_universe_bars)は
# そのままでは直らないため、J-Quants APIを叩き直さずローカルのキャッシュファイルだけを
# その場で修復するワンオフスクリプト。
# (chart_candles_*.jsonの修復は同日中に廃止したため対象から外した。UI側はdaily_screening_
# bars_raw.parquetをhyparquetで直接読むようになったので、それが直っていれば十分)
#
# 実行方法: python scripts/repair_split_adjustment_cache.py
import glob
import os
import sys

import pandas as pd

sys.path.insert(0, r"F:\stockModel")
from modules.price_adjustment import backadjust_long_format, backadjust_wide_format

BASE_DIR = r"F:\stockModel"
CACHE_DIR = os.path.join(BASE_DIR, "data", "cache")
DAILY_SCREEN_RAW_CACHE_PATH = os.path.join(CACHE_DIR, "daily_screening_bars_raw.parquet")
TRAIN_CACHE_PATH = os.path.join(CACHE_DIR, "train_universe_bars.parquet")


def repair_long_format_parquet(path, label):
    if not os.path.exists(path):
        print(f"[-] {label}: {path} が見つからないためスキップ")
        return
    df = pd.read_parquet(path)
    fixed, detected = backadjust_long_format(df)
    if not detected:
        print(f"[*] {label}: 分割由来のジャンプは見つかりませんでした")
        return
    for t, dt, ratio in detected:
        print(f"  [!] {label}: {t} {dt.date()} (比率={ratio:.4f}) を遡及調整")
    fixed.to_parquet(path)
    print(f"[+] {label}: {len(detected)}件を修復して保存完了 ({path})")


def repair_wide_format_parquet(path, label):
    if not os.path.exists(path):
        print(f"[-] {label}: {path} が見つからないためスキップ")
        return
    df = pd.read_parquet(path)
    fixed, detected = backadjust_wide_format(df)
    if not detected:
        print(f"[*] {label}: 分割由来のジャンプは見つかりませんでした")
        return
    for t, dt, ratio in detected:
        print(f"  [!] {label}: {t} {dt.date()} (比率={ratio:.4f}) を遡及調整")
    fixed.to_parquet(path)
    print(f"[+] {label}: {len(detected)}件を修復して保存完了 ({path})")


def repair_latest_prices_parquet():
    files = sorted(glob.glob(os.path.join(CACHE_DIR, "prices_*.parquet")))
    if not files:
        print("[-] prices_*.parquet が見つからないためスキップ")
        return
    latest = files[-1]
    repair_wide_format_parquet(latest, f"prices ({os.path.basename(latest)})")


def main():
    print("=" * 75)
    print("分割調整キャッシュ修復スクリプト")
    print("=" * 75)
    repair_long_format_parquet(DAILY_SCREEN_RAW_CACHE_PATH, "daily_screening_bars_raw")
    repair_latest_prices_parquet()
    repair_wide_format_parquet(TRAIN_CACHE_PATH, "train_universe_bars")
    print("=" * 75)
    print("完了")


if __name__ == "__main__":
    main()
