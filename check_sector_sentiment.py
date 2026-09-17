import os
import re
import json
import time
import datetime
import urllib.parse
import feedparser
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types

# =============================================================================
# 設定
# =============================================================================
BASE_DIR = r"F:\stockModel"
OUTPUT_JSON = os.path.join(BASE_DIR, "data", "sector_sentiment.json")
# 日次キャッシュ: 実行日ごとのセクター評価を蓄積し、将来のバックテストで
# 「その日時点のセクターセンチメント」を特徴量として使えるようにする。
# (Google News RSSは直近数日分しか取れないため過去分の遡及取得は不可。
#  このキャッシュは今後実行するたびに1日分ずつ積み上がっていく。)
DAILY_CACHE_CSV = os.path.join(BASE_DIR, "data", "sector_sentiment_daily_cache.csv")
load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("[!] GEMINI_API_KEY が設定されていません。.env ファイルを確認してください。")

# ノイズ・市況まとめ記事の除外キーワード（日本語・英語共通）
NOISE_KEYWORDS = [
    # 日本語ノイズ
    "値上がり率", "値下がり率", "今日の株価", "株価材料", "レーティング情報",
    "目標株価引き上げ", "目標株価引き下げ", "テクニカル分析", "寄り付き速報",
    "ストップ高", "ストップ安", "本日のランキング", "出来高上位",
    # 英語ノイズ
    "stock price today", "technical analysis", "price target", "bull of the day",
    "zacks", "motley fool", "market wrap", "stocks making the biggest moves"
]

SECTOR_CONFIGS = {
    "semiconductor": {
        "name": "半導体・ハイテク",
        # 英語一次ソース（米中規制、設備投資動向、SOX指数関連）
        "query": "(semiconductor OR chip equipment OR AI capex OR export controls OR ASML OR TSMC) (site:reuters.com OR site:bloomberg.com OR site:wsj.com)",
        "lang": "en",
        "tickers": ["6857", "8035", "6920", "6146", "7735", "6526", "285A", "4063", "4186", "4004", "5201"]
    },
    "automotive": {
        "name": "自動車・輸出",
        # 英語一次ソース（関税、輸入規制、EV政策、北米通商）
        "query": "(auto tariff OR car import duties OR EV subsidy OR auto trade dispute) (site:reuters.com OR site:bloomberg.com OR site:wsj.com)",
        "lang": "en",
        "tickers": ["7203", "7267", "7269", "7201", "7270", "5108"]
    },
    "defense_heavy": {
        "name": "防衛・重工",
        "query": "防衛予算 概算要求 OR 重工 受注 OR 自衛隊 装備品 OR 防衛装備移転",
        "lang": "ja",
        "tickers": ["7011", "7012", "7013", "6503", "7721", "6946"]
    },
    "trading_companies": {
        "name": "総合商社・資源",
        "query": "総合商社 資源 OR 原油市況 商社 OR 非鉄金属 権益 OR コモディティ 商社",
        "lang": "ja",
        "tickers": ["8058", "8001", "8031", "8053", "8002", "2768", "1605"]
    },
    "banking": {
        "name": "銀行・金融",
        "query": "日銀 追加利上げ OR 長期金利 上昇 銀行 OR 住宅ローン 金利引き上げ OR 銀行 利ざや",
        "lang": "ja",
        "tickers": ["8306", "8316", "8411", "8308", "7167", "8331"]
    },
    "electric_power": {
        "name": "電力・エネルギー",
        "query": "原発 再稼働容認 OR 原子力規制委員会 OR 電力 燃料費調整 OR データセンター 電力需要",
        "lang": "ja",
        "tickers": ["9501", "9503", "9508", "9502", "9509"]
    },
    "shipping": {
        "name": "海運・物流",
        # 旧クエリ(バルチック海運指数 急変動 等)はヒット率が低かったため、より一般的な語に拡張
        "query": "海運 運賃 OR 海運株 OR コンテナ船 需給 OR 物流 コスト OR スエズ運河 紅海",
        "lang": "ja",
        "tickers": ["9101", "9104", "9107", "9064"]
    },
    "retail_inbound": {
        "name": "小売・消費・インバウンド",
        # 旧クエリ(訪日外国人 免税改定 等)はヒット率が低かったため、より一般的な語に拡張
        "query": "インバウンド 消費 OR 訪日外国人 OR 百貨店 売上 OR 小売 業績 OR 個人消費",
        "lang": "ja",
        "tickers": ["9983", "3382", "3092", "8267", "9843", "3086", "2782"]
    },
    "food_staples": {
        "name": "食料品・ディフェンシブ",
        "query": "食品 原材料高騰 OR 食品 再値上げ OR 円安 原材料負担 食品",
        "lang": "ja",
        "tickers": ["2802", "2914", "2502", "2503", "2801", "2267"]
    }
}

# =============================================================================
# 1. ニュース収集（日英ハイブリッド＆ノイズ事前除去）
# =============================================================================
def is_noisy_article(title: str, snippet: str) -> bool:
    text = f"{title} {snippet}".lower()
    return any(k.lower() in text for k in NOISE_KEYWORDS)

def parse_feed_entries(query_with_time, lang="ja", max_items=5, hours_limit=None):
    encoded_query = urllib.parse.quote(query_with_time)
    if lang == "en":
        rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
    else:
        rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=ja&gl=JP&ceid=JP:ja"

    feed = feedparser.parse(rss_url)
    now_utc = datetime.datetime.now(datetime.timezone.utc)
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
                diff_hours = max(0.0, (now_utc - pub_time).total_seconds() / 3600.0)
            except Exception:
                diff_hours = 24.0

        if hours_limit is not None and diff_hours > hours_limit:
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

def fetch_sector_news(conf, max_items=5):
    query = conf["query"]
    lang = conf.get("lang", "ja")
    
    # 1. まず直近2日（48時間以内）の速報を優先取得
    articles = parse_feed_entries(f"{query} when:2d", lang=lang, max_items=max_items, hours_limit=48)
    
    # 2. 記事不足時のフォールバックも最大3日（72時間：土日跨ぎ対応）までに制限
    if len(articles) < 2:
        fallback_articles = parse_feed_entries(f"{query} when:3d", lang=lang, max_items=max_items, hours_limit=72)
        existing_titles = {a["title"] for a in articles}
        for fb in fallback_articles:
            if fb["title"] not in existing_titles:
                articles.append(fb)
            if len(articles) >= max_items:
                break
                
    return articles[:max_items]

# =============================================================================
# 2. Gemini API 多言語解釈・セクター影響評価
# =============================================================================
def evaluate_sectors_with_gemini(sector_news_dict, max_retries=3):
    client = genai.Client(api_key=GEMINI_API_KEY)

    prompt = f"""
あなたは日本の株式ヘッジファンドのシニア・リスクアナリストです。
提供された各セクターの最新ニュース（英語ニュースおよび日本語ニュース）を分析し、
今後1〜2週間の『日本株（東証上場銘柄）における株価インパクト、材料の性質、持続期間』を客観的に評価してください。

【重要：英語ニュースの評価方針】
- 半導体・ハイテクおよび自動車セクターには、海外の一次ニュース（Reuters/Bloomberg等）が含まれています。
- 米商務省の対中輸出規制（BIS規制）、関税措置、ビッグテックのAI設備投資（Capex）見直し等のニュースは、日本の製造装置（東エレク・ディスコ等）や大手自動車株に直撃する材料として厳密に評価してください。
- 出力するサマリ（summary）は、必ず自然な日本語（35文字以内）で記述してください。

【時間減衰ルール】
- 速報 (hours_ago <= 12h): 影響度を100%反映
- 進行中 (12h < hours_ago <= 36h): 影響度を約50%減衰
- 既知・織り込み (hours_ago > 36h): 特大構造材料を除き 0.0（中立）付近へ収束

【半導体・ハイテクの重点指示】
- 『AI開発減速』『AI投資抑制・回収懸念』『対中半導体規制の強化』は、セクター全体を直撃する構造悪材料（STRUCTURAL）としてスコア -0.6 〜 -1.0、advice: "見送り" を付与してください。

【ニュースデータ】
{json.dumps(sector_news_dict, ensure_ascii=False, indent=2)}

【出力JSONフォーマット（厳守）】
以下のスキーマで各セクターを評価してください。
{{
  "<sector_key>": {{
    "score": <float: -1.0から+1.0>,
    "shock_detected": <boolean: score <= -0.5 または急落リスクがある場合>,
    "category": "<REGULATION / AI_SLOWDOWN / GEOPOLITICS / POLICY / EARNINGS / COMMODITY / GENERAL>",
    "duration": "<TEMPORARY / STRUCTURAL / NONE>",
    "summary": "<日本語の短評 35文字以内>",
    "action_advice": "<通常 / 打診30%抑制 / 見送り>"
  }}
}}
"""

    # 404となった旧モデルから推奨最新モデルへ変更
    target_models = ["gemini-3.6-flash", "gemini-3.5-flash-lite", "gemini-3.1-pro-preview"]

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
    print("【セクターショック検知 & センチメント自動解析 (日英ハイブリッド版)】")
    print("=" * 80)

    sector_news = {}
    for sec_key, conf in SECTOR_CONFIGS.items():
        news_items = fetch_sector_news(conf, max_items=5)
        lang_label = "US(英語)" if conf.get("lang") == "en" else "JP(日本語)"
        sector_news[sec_key] = {
            "name": conf["name"],
            "lang": conf.get("lang", "ja"),
            "articles": news_items
        }
        print(f"[*] {conf['name']} [{lang_label}]: {len(news_items)} 件の記事取得")

    print("\n[*] Gemini API にセクター一括評価をリクエスト中...")
    eval_results = evaluate_sectors_with_gemini(sector_news)

    if not eval_results:
        print("[!] Gemini APIによるセクター評価が全モデル・全リトライで失敗しました。"
              "全セクターをデフォルト値(中立)で保存します。sector_sentiment.jsonの中立値は"
              "実際の市況ではなく解析失敗によるものである点に注意してください。")

    final_output = {}
    for sec_key, conf in SECTOR_CONFIGS.items():
        if sec_key not in eval_results:
            if eval_results:
                print(f"  [!] {conf['name']} の評価がGemini応答に含まれていません。デフォルト値(中立)を使用します。")
            sec_eval = {
                "score": 0.0,
                "shock_detected": False,
                "category": "GENERAL",
                "duration": "NONE",
                "summary": "判定不能またはデータなし",
                "action_advice": "通常"
            }
            sec_eval["is_fallback"] = True
        else:
            sec_eval = eval_results[sec_key]
            sec_eval["is_fallback"] = False
        sec_eval["name"] = conf["name"]
        sec_eval["tickers"] = conf["tickers"]
        final_output[sec_key] = sec_eval

        status_flag = "🛑 SHOCK" if sec_eval.get("shock_detected") else "NORMAL"
        cat_str = sec_eval.get("category", "")
        dur_str = sec_eval.get("duration", "")
        print(f"  - [{status_flag}] {conf['name']}: スコア {sec_eval.get('score', 0.0):+.2f} ({cat_str}/{dur_str}) | {sec_eval.get('summary')} | 助言: {sec_eval.get('action_advice')}")

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)

    print(f"\n[+] 評価結果を保存しました: {OUTPUT_JSON}")

    append_to_daily_cache(final_output)


def append_to_daily_cache(final_output):
    """今日分のセクター評価をDAILY_CACHE_CSVに追記する。同日に複数回実行した場合は
    その日の既存行を最新の評価で置き換える(重複防止)。"""
    today_str = datetime.date.today().strftime("%Y-%m-%d")
    rows = []
    for sec_key, sec_eval in final_output.items():
        rows.append({
            "date": today_str,
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
        existing_df = existing_df[existing_df["date"] != today_str]
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined_df = new_df

    combined_df = combined_df.sort_values(["date", "sector_key"])
    combined_df.to_csv(DAILY_CACHE_CSV, index=False, encoding='utf-8-sig')
    n_days = combined_df["date"].nunique()
    print(f"[+] 日次キャッシュに追記しました: {DAILY_CACHE_CSV} (累計{n_days}日分)")

if __name__ == "__main__":
    main()