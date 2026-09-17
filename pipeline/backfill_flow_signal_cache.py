# backfill_flow_signal_cache.py
# parse_flow_signal.py(CTA/J-NET大口手口フロー)の日次キャッシュを過去に遡って作る。
# JPXの日次手口ファイルアーカイブは2025-09-01頃までしか遡れないことを確認済みのため、
# その範囲でバックフィルする。
import os
import datetime
import pandas as pd

# pipeline/配下からでもmodules/を解決できるようにプロジェクトルートをsys.pathへ追加
import sys
sys.path.insert(0, r"F:\stockModel")
from parse_flow_signal import fetch_daily_volume_file, extract_nk225_volumes
from modules.constants import CTA_BROKER_KEYWORDS

BASE_DIR = r"F:\stockModel"
DAILY_CACHE_CSV = os.path.join(BASE_DIR, "data", "flow_signal_daily_cache.csv")
ARCHIVE_START = datetime.date(2025, 9, 1)  # JPXアーカイブの遡及可能な下限(確認済み)


def get_business_days(start_date, end_date):
    days = []
    curr = start_date
    while curr <= end_date:
        if curr.weekday() < 5:
            days.append(curr)
        curr += datetime.timedelta(days=1)
    return days


def classify_flow(cta_share, jnet_ratio, combined_tot):
    if cta_share >= 0.40:
        return "CTA_SURGE", "⚡ CTA手口急増（トレンド加速／逆張り警戒）"
    elif jnet_ratio >= 0.25:
        return "JNET_HIGH", "📦 機関大口クロス活発（ポジ移動／ロール注意）"
    elif cta_share < 0.20 and combined_tot > 0:
        return "QUIET", "💤 大口薄商い（ダマシ警戒／保ち合い気味）"
    return "NORMAL", "機関・大口フロー平常"


def append_flow_cache(date_str, cta_share, jnet_ratio, level, signal_desc):
    new_row = pd.DataFrame([{
        "date": date_str,
        "cta_share": round(cta_share, 4),
        "jnet_ratio": round(jnet_ratio, 4),
        "level": level,
        "signal_desc": signal_desc,
    }])
    if os.path.exists(DAILY_CACHE_CSV):
        existing_df = pd.read_csv(DAILY_CACHE_CSV, encoding='utf-8-sig')
        existing_df = existing_df[existing_df["date"] != date_str]
        combined_df = pd.concat([existing_df, new_row], ignore_index=True)
    else:
        combined_df = new_row
    combined_df = combined_df.sort_values("date")
    combined_df.to_csv(DAILY_CACHE_CSV, index=False, encoding='utf-8-sig')


def main():
    print("=" * 80)
    print(f"【大口手口フロー 日次キャッシュ バックフィル({ARCHIVE_START} 〜 今日)】")
    print("=" * 80)

    if os.path.exists(DAILY_CACHE_CSV):
        existing_dates = set(pd.read_csv(DAILY_CACHE_CSV, encoding='utf-8-sig')["date"].unique())
    else:
        existing_dates = set()

    today = datetime.date.today()
    target_days = get_business_days(ARCHIVE_START, today)
    print(f"[*] 対象営業日候補: {len(target_days)}日")

    n_ok, n_skip, n_fail = 0, 0, 0
    for i, target_date in enumerate(target_days):
        date_str = target_date.strftime("%Y-%m-%d")
        if date_str in existing_dates:
            n_skip += 1
            continue

        f_auc, ymd_auc = fetch_daily_volume_file(target_date, is_jnet=False)
        if not f_auc:
            n_fail += 1
            continue
        f_jnet, ymd_jnet = fetch_daily_volume_file(target_date, is_jnet=True)

        auc_tot, auc_cta = extract_nk225_volumes(f_auc, CTA_BROKER_KEYWORDS)
        jnet_tot, jnet_cta = extract_nk225_volumes(f_jnet, CTA_BROKER_KEYWORDS) if f_jnet else (0.0, 0.0)

        combined_tot = auc_tot + jnet_tot
        combined_cta = auc_cta + jnet_cta
        cta_share = (combined_cta / combined_tot) if combined_tot > 0 else 0.0
        jnet_ratio = (jnet_tot / combined_tot) if combined_tot > 0 else 0.0
        level, signal_desc = classify_flow(cta_share, jnet_ratio, combined_tot)

        # ymd_aucはファイル実取得日(遡り探索の結果)なので、実際に使われた日付で記録する
        actual_date_str = datetime.datetime.strptime(ymd_auc, "%Y%m%d").strftime("%Y-%m-%d") if ymd_auc else date_str
        append_flow_cache(actual_date_str, cta_share, jnet_ratio, level, signal_desc)
        n_ok += 1

        if (i + 1) % 20 == 0 or (i + 1) == len(target_days):
            print(f"  --> {i + 1}/{len(target_days)} 日 処理済み (成功={n_ok}, スキップ={n_skip}, 失敗={n_fail})")

    print(f"\n[+] バックフィル完了: {DAILY_CACHE_CSV}")
    print(f"    成功={n_ok} スキップ(既存)={n_skip} 失敗(ファイル取得不可)={n_fail}")


if __name__ == "__main__":
    main()
