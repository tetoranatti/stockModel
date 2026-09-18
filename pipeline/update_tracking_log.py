# update_tracking_log.py
# 日次トラッキング更新: (1)未約定(AWAITING_ENTRY)の推奨を翌営業日引成(大引け成行)で
# 約定判定し、(2)約定済み(PENDING)の推奨が利確/損切/タイムアウトに到達したか
# 本日の値動きで判定し、(3)本日のポートフォリオ採用銘柄を新規AWAITING_ENTRY行として
# 追記する。run_dynamic_regime_screening_v8.py の後に実行する前提(当日のポートフォリオ・
# 株価キャッシュの両方が必要なため)。
import os
import sys
import json
import datetime

sys.path.insert(0, r"F:\stockModel")
from modules.data_loader import fetch_all_tickers_data
from modules.tracking import (
    load_tracking_log, save_tracking_log, append_new_entries,
    resolve_entries, resolve_pending, compute_summary,
)

BASE_DIR = r"F:\stockModel"
PORTFOLIO_JSON = os.path.join(BASE_DIR, "data", "portfolio_latest.json")


def main():
    print("=" * 70)
    print("【トラッキングログ更新】")
    print("=" * 70)

    today = datetime.date.today()
    df = load_tracking_log()
    print(f"[*] 既存ログ: {len(df)}件")

    # --- 1. 未約定の推奨を翌営業日引成(大引け成行)で約定判定 ---
    try:
        all_prices_df = fetch_all_tickers_data()
        df, n_filled = resolve_entries(df, all_prices_df, today)
        print(f"[+] 本日約定した推奨: {n_filled}件")

        # --- 2. 約定済み(保有中)の推奨を本日の値動きで判定 ---
        df, n_resolved = resolve_pending(df, all_prices_df, today)
        print(f"[+] 本日確定した推奨: {n_resolved}件")
    except FileNotFoundError as e:
        print(f"[!] 本日分の株価キャッシュが無いため判定をスキップ: {e}")

    # --- 3. 本日のポートフォリオ採用銘柄を新規追記(未約定として) ---
    if os.path.exists(PORTFOLIO_JSON):
        with open(PORTFOLIO_JSON, "r", encoding="utf-8") as f:
            portfolio_data = json.load(f)
        positions = portfolio_data.get("positions", [])
        df, n_new = append_new_entries(df, positions, today)
        print(f"[+] 新規追記: {n_new}件 (本日のポートフォリオ採用{len(positions)}件中)")
    else:
        print("[!] data/portfolio_latest.json が見つからないため新規追記をスキップ")

    save_tracking_log(df)
    print(f"[+] 保存完了: data/tracking_log.csv (累計{len(df)}件)")

    summary = compute_summary(df)
    print("\n" + "=" * 70)
    print("【トラッキング実績サマリ】")
    print("=" * 70)
    print(f"  エントリー待ち: {summary['n_awaiting_entry']}件 / 保有中: {summary['n_pending']}件 / 確定済み: {summary['n_resolved']}件")
    if summary['n_resolved'] > 0:
        print(f"  勝率: {summary['win_rate']}% | PF: {summary['pf']} | 平均リターン: {summary['avg_ret_pct']*100:+.2f}%")
    else:
        print("  確定済みの推奨がまだありません(保有期間10営業日が経過すると確定します)")
    print("=" * 70)


if __name__ == "__main__":
    main()
