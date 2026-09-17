# backfill_sector_sentiment.py
# check_sector_sentiment.py の日次キャッシュを過去に遡って埋めるバックフィルスクリプト。
# Google News RSSのafter:/before:日付範囲指定で過去日付のニュースを取得できることを
# 確認済み(英語・日本語クエリとも)。小規模(デフォルト15営業日)でまず試す。
import os
import re
import time
import datetime
import urllib.parse
import feedparser
import pandas as pd

from check_sector_sentiment import (
    SECTOR_CONFIGS, is_noisy_article, evaluate_sectors_with_gemini, DAILY_CACHE_CSV
)

N_DAYS_BACKFILL = 60  # 海運・小売クエリ改善後、3ヶ月規模に拡大
SLEEP_BETWEEN_DAYS_SEC = 2.0  # Google News RSSへの配慮(連続スクレイピング回避)


def get_recent_business_days(n_days, skip_today=True):
    today = datetime.date.today()
    days = []
    curr = today - datetime.timedelta(days=1) if skip_today else today
    while len(days) < n_days:
        if curr.weekday() < 5:
            days.append(curr)
        curr -= datetime.timedelta(days=1)
    return sorted(days)


def parse_feed_entries_historical(query, target_date, lang="ja", max_items=5, window_days_before=2, hours_limit=72):
    """target_date(過去日)を基準に after:/before: で日付範囲を絞り込んで取得する。
    hours_agoはtarget_date当日0:00 JST(=前日15:00 UTC、寄り付き前=まだその日のニュースは
    存在しない時点)を基準に計算する。これにより対象日の取引時間中・引け後に出たニュースが
    「既知の材料」として紛れ込むルックアヘッドバイアスを避ける。"""
    after_date = target_date - datetime.timedelta(days=window_days_before)
    before_date = target_date + datetime.timedelta(days=1)
    query_with_range = f"{query} after:{after_date.isoformat()} before:{before_date.isoformat()}"
    encoded_query = urllib.parse.quote(query_with_range)
    if lang == "en":
        rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
    else:
        rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=ja&gl=JP&ceid=JP:ja"

    feed = feedparser.parse(rss_url)
    reference_time = datetime.datetime.combine(
        target_date - datetime.timedelta(days=1), datetime.time(15, 0), tzinfo=datetime.timezone.utc
    )
    items = []

    for entry in feed.entries:
        raw_title = entry.title
        parts = raw_title.rsplit(" - ", 1)
        clean_title = parts[0].strip()
        source = parts[1].strip() if len(parts) > 1 else ("Global Media" if lang == "en" else "一般報道")

        raw_summary = entry.get('summary', '') or entry.get('description', '')
        clean_summary = re.sub(r'<[^>]+>', '', raw_summary).strip()
        clean_summary = re.sub(r'\s+', ' ', clean_summary)[:150]

        if is_noisy_article(clean_title, clean_summary):
            continue

        diff_hours = 24.0
        if hasattr(entry, 'published_parsed') and entry.published_parsed:
            try:
                pub_time = datetime.datetime(*entry.published_parsed[:6], tzinfo=datetime.timezone.utc)
                diff_hours = (reference_time - pub_time).total_seconds() / 3600.0
                if diff_hours < 0:
                    continue  # target_dateより未来の記事は除外(before:の取りこぼし対策)
            except Exception:
                diff_hours = 24.0

        if diff_hours > hours_limit:
            continue

        items.append({
            "title": clean_title,
            "source": source,
            "hours_ago": round(diff_hours, 1),
            "snippet": clean_summary if clean_summary else clean_title
        })
        if len(items) >= max_items:
            break

    return items


def fetch_sector_news_historical(conf, target_date, max_items=5):
    query = conf["query"]
    lang = conf.get("lang", "ja")

    articles = parse_feed_entries_historical(query, target_date, lang=lang, max_items=max_items,
                                              window_days_before=2, hours_limit=72)
    if len(articles) < 2:
        fallback = parse_feed_entries_historical(query, target_date, lang=lang, max_items=max_items,
                                                  window_days_before=4, hours_limit=120)
        existing_titles = {a["title"] for a in articles}
        for fb in fallback:
            if fb["title"] not in existing_titles:
                articles.append(fb)
            if len(articles) >= max_items:
                break
    return articles[:max_items]


def append_backfill_row(target_date, final_output):
    date_str = target_date.strftime("%Y-%m-%d")
    rows = []
    for sec_key, sec_eval in final_output.items():
        rows.append({
            "date": date_str,
            "sector_key": sec_key,
            "sector_name": sec_eval.get("name", ""),
            "score": sec_eval.get("score", 0.0),
            "shock_detected": sec_eval.get("shock_detected", False),
            "category": sec_eval.get("category", ""),
            "duration": sec_eval.get("duration", ""),
            "summary": sec_eval.get("summary", ""),
            "action_advice": sec_eval.get("action_advice", ""),
            "is_fallback": sec_eval.get("is_fallback", False),
        })
    new_df = pd.DataFrame(rows)

    if os.path.exists(DAILY_CACHE_CSV):
        existing_df = pd.read_csv(DAILY_CACHE_CSV, encoding='utf-8-sig')
        existing_df = existing_df[existing_df["date"] != date_str]
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined_df = new_df
    combined_df = combined_df.sort_values(["date", "sector_key"])
    combined_df.to_csv(DAILY_CACHE_CSV, index=False, encoding='utf-8-sig')


def main():
    print("=" * 80)
    print(f"【セクターセンチメント バックフィル(小規模テスト: 直近{N_DAYS_BACKFILL}営業日)】")
    print("=" * 80)

    if os.path.exists(DAILY_CACHE_CSV):
        existing_dates = set(pd.read_csv(DAILY_CACHE_CSV, encoding='utf-8-sig')["date"].unique())
    else:
        existing_dates = set()

    target_days = get_recent_business_days(N_DAYS_BACKFILL, skip_today=True)
    print(f"[*] 対象日: {target_days[0]} 〜 {target_days[-1]} ({len(target_days)}日)")

    for i, target_date in enumerate(target_days):
        date_str = target_date.strftime("%Y-%m-%d")
        if date_str in existing_dates:
            print(f"[{i+1}/{len(target_days)}] {date_str}: 既存データありスキップ")
            continue

        print(f"\n[{i+1}/{len(target_days)}] {date_str} のニュースを取得中...")
        sector_news = {}
        for sec_key, conf in SECTOR_CONFIGS.items():
            news_items = fetch_sector_news_historical(conf, target_date, max_items=5)
            sector_news[sec_key] = {
                "name": conf["name"],
                "lang": conf.get("lang", "ja"),
                "articles": news_items
            }
            print(f"  - {conf['name']}: {len(news_items)}件")

        total_articles = sum(len(v["articles"]) for v in sector_news.values())
        if total_articles == 0:
            print(f"  [!] {date_str}: 全セクターで記事0件のためGemini評価をスキップし中立値で記録")
            eval_results = {}
        else:
            eval_results = evaluate_sectors_with_gemini(sector_news)

        final_output = {}
        for sec_key, conf in SECTOR_CONFIGS.items():
            if sec_key not in eval_results:
                sec_eval = {
                    "score": 0.0, "shock_detected": False, "category": "GENERAL",
                    "duration": "NONE", "summary": "判定不能またはデータなし", "action_advice": "通常",
                    "is_fallback": True,
                }
            else:
                sec_eval = eval_results[sec_key]
                sec_eval["is_fallback"] = False
            sec_eval["name"] = conf["name"]
            final_output[sec_key] = sec_eval

        append_backfill_row(target_date, final_output)
        print(f"  --> {date_str} 分を保存")
        time.sleep(SLEEP_BETWEEN_DAYS_SEC)

    print(f"\n[+] バックフィル完了: {DAILY_CACHE_CSV}")


if __name__ == "__main__":
    main()
