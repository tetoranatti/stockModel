import os
import re
import json
import time
import datetime
import urllib.parse
import feedparser
from dotenv import load_dotenv
from google import genai
from google.genai import types

# =============================================================================
# 設定
# =============================================================================
BASE_DIR = r"F:\stockModel"
OUTPUT_JSON = os.path.join(BASE_DIR, "data", "sector_sentiment.json")
# プロジェクトルートの .env を読み込む
load_dotenv()

# 環境変数から取得（.env に定義された値が入る）
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("[!] GEMINI_API_KEY が設定されていません。.env ファイルを確認してください。")

SECTOR_CONFIGS = {
    "semiconductor": {
        "name": "半導体・ハイテク",
        # AI開発減速や投資抑制のニュースをダイレクトに捉えるクエリに強化
        "query": "半導体 OR SOX OR 製造装置 OR AI開発 OR AI投資 OR 対中規制",
        "tickers": ["6857", "8035", "6920", "6146", "7735", "6526", "285A", "4063", "4186", "4004", "5201"]
    },
    "defense_heavy": {
        "name": "防衛・重工",
        "query": "防衛 予算 OR 重工業 受注 OR 自衛隊 装備",
        "tickers": ["7011", "7012", "7013", "6503", "7721", "6946"]
    },
    "trading_companies": {
        "name": "総合商社・資源",
        "query": "総合商社 OR 商社 原油 OR コモディティ 商社",
        "tickers": ["8058", "8001", "8031", "8053", "8002", "2768", "1605"]
    },
    "automotive": {
        "name": "自動車・輸出",
        "query": "自動車 関税 OR 円高 自動車 OR EV 規制",
        "tickers": ["7203", "7267", "7269", "7201", "7270", "5108"]
    },
    "banking": {
        "name": "銀行・金融",
        "query": "銀行 OR 日銀 利上げ OR 長期金利 銀行 OR メガバンク",
        "tickers": ["8306", "8316", "8411", "8308", "7167", "8331"]
    },
    "electric_power": {
        "name": "電力・エネルギー",
        "query": "電力 原発再稼働 OR 規制委 電力 OR 関西電力 OR 東京電力",
        "tickers": ["9501", "9503", "9508", "9502", "9509"]
    },
    "shipping": {
        "name": "海運・物流",
        "query": "海運 運賃 OR 紅海 コンテナ OR バルチック海運指数",
        "tickers": ["9101", "9104", "9107", "9064"]
    },
    "retail_inbound": {
        "name": "小売・消費・インバウンド",
        "query": "小売 OR 百貨店 OR インバウンド OR 訪日客 免税",
        "tickers": ["9983", "3382", "3092", "8267", "9843", "3086", "2782"]
    },
    "food_staples": {
        "name": "食料品・ディフェンシブ",
        "query": "食品 原材料 OR 食品 値上げ OR ディフェンシブ 食品",
        "tickers": ["2802", "2914", "2502", "2503", "2801", "2267"]
    }
}

# =============================================================================
# 1. ニュース詳細収集 (経過時間 hours_ago 算出)
# =============================================================================
def parse_feed_entries(query_with_time, max_items=5, hours_limit=None):
    encoded_query = urllib.parse.quote(query_with_time)
    rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=ja&gl=JP&ceid=JP:ja"
    feed = feedparser.parse(rss_url)

    now_utc = datetime.datetime.now(datetime.timezone.utc)
    items = []

    for entry in feed.entries:
        raw_title = entry.title
        parts = raw_title.rsplit(" - ", 1)
        clean_title = parts[0].strip()
        source = parts[1].strip() if len(parts) > 1 else "一般報道"

        raw_summary = entry.get('summary', '') or entry.get('description', '')
        clean_summary = re.sub(r'<[^>]+>', '', raw_summary).strip()
        clean_summary = re.sub(r'\s+', ' ', clean_summary)[:120]

        diff_hours = 24.0
        if hasattr(entry, 'published_parsed') and entry.published_parsed:
            try:
                pub_time = datetime.datetime(*entry.published_parsed[:6], tzinfo=datetime.timezone.utc)
                diff_hours = max(0.0, (now_utc - pub_time).total_seconds() / 3600.0)
            except Exception:
                diff_hours = 24.0

        if hours_limit is not None and diff_hours > hours_limit:
            continue

        item_data = {
            "title": clean_title,
            "source": source,
            "hours_ago": round(diff_hours, 1),
            "snippet": clean_summary if clean_summary else clean_title
        }
        items.append(item_data)

        if len(items) >= max_items:
            break

    return items

def fetch_sector_news(query, max_items=5):
    articles = parse_feed_entries(f"{query} when:2d", max_items=max_items, hours_limit=48)
    if len(articles) < 2:
        fallback_articles = parse_feed_entries(f"{query} when:7d", max_items=max_items, hours_limit=None)
        existing_titles = {a["title"] for a in articles}
        for fb in fallback_articles:
            if fb["title"] not in existing_titles:
                articles.append(fb)
            if len(articles) >= max_items:
                break
    return articles[:max_items]

# =============================================================================
# 2. Gemini API によるセクターショック一括評価
# =============================================================================
def evaluate_sectors_with_gemini(sector_news_dict, max_retries=3):
    client = genai.Client(api_key=GEMINI_API_KEY)

    prompt = f"""
あなたは株式ヘッジファンドのシニアリスクアナリストです。
以下に提供された主要セクターの最新ニュースを分析し、翌週の日本株市場における『短期的な株価影響（セクターショックの有無）』を客観的に評価してください。

【時間経過と市場織り込みの割引ルール（重要）】
各ニュースには報道からの経過時間（hours_ago）が付与されています。
1. 速報（hours_ago <= 12.0時間）: フルスコアを反映してください。
2. 進行中（12.0時間 < hours_ago <= 36.0時間）: スコア絶対値を約50%減衰させて評価してください。
3. 織り込み済み（hours_ago > 36.0時間）: 特大材料を除きスコアを0.0（中立）付近に収束させてください。

【半導体・ハイテクの重点指示】
- 『AI開発競争の減速』『AIモデル学習投資の慎重姿勢』『GPU・サーバー設備投資の先送り』に関するニュースは、半導体製造装置や素材株全体のバリュエーションを直撃する重大な構造悪材料として厳しく（-0.5〜-1.0）評価してください。

【ニュースデータ】
{json.dumps(sector_news_dict, ensure_ascii=False, indent=2)}

【判定基準】
- スコア範囲: -1.0（極めて強いセクター悪材料・急落リスク）〜 +1.0（強力な業界好材料）
- 輸出規制強化、関税引き上げ、AI開発減速・AI投資抑制、需要急減、増資、地政学悪化: -0.5 〜 -1.0
- 防衛予算拡充、大型受注、原発再稼働容認、自社株買い、規制緩和: +0.5 〜 +1.0
- 一般的な市況解説、既知の動向、織り込み済みのもの: 0.0 前後
- shock_detected: スコアが -0.5 以下、または急落警戒が必要な場合に true

必ず以下のJSONフォーマットのみを返してください。
{{
  "<sector_key>": {{
    "score": <float: -1.0から+1.0>,
    "shock_detected": <boolean>,
    "summary": "<短評 35文字以内>",
    "action_advice": "<通常 / 打診30%抑制 / 見送り>"
  }}
}}
"""

    target_models = ["gemini-3.5-flash-lite", "gemini-3.6-flash"]

    for model_name in target_models:
        for attempt in range(1, max_retries + 1):
            try:
                print(f"[*] Gemini API 呼び出し中 [{model_name}] (試行 {attempt}/{max_retries})...")
                chat = client.chats.create(
                    model=model_name,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.1
                    )
                )
                response = chat.send_message(prompt)
                return json.loads(response.text.strip())
            except Exception as e:
                err_msg = str(e)
                print(f"[!] エラー ({model_name}): {err_msg}")
                if "503" in err_msg or "429" in err_msg or "overloaded" in err_msg.lower():
                    time.sleep(attempt * 3)
                else:
                    break

    return {}

# =============================================================================
# メイン処理
# =============================================================================
def main():
    print("=" * 80)
    print("【セクターショック検知 & センチメント自動解析 (AI投資感度強化版)】")
    print("=" * 80)

    sector_news = {}
    for sec_key, conf in SECTOR_CONFIGS.items():
        news_items = fetch_sector_news(conf["query"], max_items=5)
        sector_news[sec_key] = {
            "name": conf["name"],
            "articles": news_items
        }
        print(f"[*] {conf['name']}: {len(news_items)} 件の記事を取得")

    print("\n[*] Gemini API にセクター一括評価をリクエスト中...")
    eval_results = evaluate_sectors_with_gemini(sector_news)

    final_output = {}
    for sec_key, conf in SECTOR_CONFIGS.items():
        sec_eval = eval_results.get(sec_key, {
            "score": 0.0,
            "shock_detected": False,
            "summary": "判定不能またはデータなし",
            "action_advice": "通常"
        })
        sec_eval["name"] = conf["name"]
        sec_eval["tickers"] = conf["tickers"]
        final_output[sec_key] = sec_eval

        status_flag = "⚠️ SHOCK" if sec_eval.get("shock_detected") else "NORMAL"
        print(f"  - [{status_flag}] {conf['name']}: スコア {sec_eval.get('score', 0.0):+.2f} | {sec_eval.get('summary')} | 助言: {sec_eval.get('action_advice')}")

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)

    print(f"\n[+] 評価結果を保存しました: {OUTPUT_JSON}")

if __name__ == "__main__":
    main()