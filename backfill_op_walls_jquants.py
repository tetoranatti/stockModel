# backfill_op_walls_jquants.py
# jpx_daily_features_db.csv の call_oi_wall/put_oi_wall/gamma_flip を、
# J-Quants日経225オプションAPI(市場全体の合計建玉)ベースの値で過去分まとめて置き換える。
# cta_net_futures は対象外(J-Quantsに代替がないため従来通りJPX週次Excel由来のまま)。
#
# 再実行時は、まだ旧値のまま残っている日付(前回バックアップと一致する行)だけを
# 少ない並列数で再試行する(取得済みの日を無駄に叩き直してレート制限を誘発しないため)。
import os
import sys
import datetime
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

from build_jquants_cache import RateLimiter, REQUESTS_PER_MINUTE, FETCH_WORKERS
from update_daily_features_v6 import fetch_op_walls_jquants, DB_PATH

load_dotenv()
JQUANTS_API_KEY = os.environ.get("JQUANTS_API_KEY")

def main():
    retry_only = "--retry-only" in sys.argv
    workers = 3 if retry_only else FETCH_WORKERS

    if not os.path.exists(DB_PATH):
        print(f"[!] {DB_PATH} が見つかりません。")
        return

    df_db = pd.read_csv(DB_PATH, index_col=0, parse_dates=True)
    backup_path = DB_PATH + ".pre_jquants_walls_backfill.bak"

    if retry_only:
        if not os.path.exists(backup_path):
            print(f"[!] {backup_path} が見つかりません。--retry-only には前回のバックアップが必要です。")
            return
        df_bak = pd.read_csv(backup_path, index_col=0, parse_dates=True)
        unchanged = (df_db['call_oi_wall'] == df_bak['call_oi_wall']) & (df_db['put_oi_wall'] == df_bak['put_oi_wall'])
        dates = [d.date() for d in df_db.index[unchanged]]
        print(f"[*] 再試行対象(前回未取得分のみ): {len(dates)} 日 / 並列数: {workers}")
    else:
        dates = [d.date() for d in df_db.index]
        print(f"[*] バックフィル対象: {len(dates)} 日分 ({dates[0]} ～ {dates[-1]})")
        df_db.to_csv(backup_path)
        print(f"[*] バックフィル前のバックアップを保存: {backup_path}")

    if not dates:
        print("[+] 再試行対象なし。完了しています。")
        return

    session = requests.Session()
    session.headers.update({"x-api-key": JQUANTS_API_KEY})
    rate_limiter = RateLimiter(max_calls=REQUESTS_PER_MINUTE)

    results = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(fetch_op_walls_jquants, d, session, rate_limiter): d
            for d in dates
        }
        done_count = 0
        for future in as_completed(futures):
            d = futures[future]
            results[d] = future.result()
            done_count += 1
            if done_count % 50 == 0 or done_count == len(dates):
                print(f"  --> 取得進捗: {done_count}/{len(dates)} 日完了")

    updated_cnt = 0
    skipped_cnt = 0
    for d in dates:
        c_wall, p_wall = results[d]
        if c_wall is None or p_wall is None:
            skipped_cnt += 1
            continue
        ts = pd.Timestamp(d)
        df_db.loc[ts, 'call_oi_wall'] = c_wall
        df_db.loc[ts, 'put_oi_wall'] = p_wall
        df_db.loc[ts, 'gamma_flip'] = (c_wall + p_wall) / 2.0
        updated_cnt += 1

    df_db.sort_index(inplace=True)
    df_db.to_csv(DB_PATH)
    print(f"\n[+] 完了: 更新 {updated_cnt} 日 / 取得失敗(旧値維持) {skipped_cnt} 日")
    print(f"[+] 保存先: {DB_PATH}")

if __name__ == "__main__":
    main()
