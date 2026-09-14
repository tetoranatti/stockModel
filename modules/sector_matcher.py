# modules/sector_matcher.py
SEC33_TO_KEY = {
    "電気機器": "semiconductor",
    "輸送用機器": "automotive",
    "卸売業": "trading_companies",
    "銀行業": "banking",
    "電気・ガス業": "electric_power",
    "海運業": "shipping",
    "機械": "defense_heavy",
    "小売業": "retail_inbound",
    "食料品": "food_staples",
    "陸運業": "shipping",
    "空運業": "retail_inbound",
    "保険業": "banking",
    "証券、商品先物取引業": "banking"
}

SEC17_TO_KEY = {
    "金融（除く銀行）": "banking",
    "自動車・輸送機": "automotive",
    "小売": "retail_inbound",
    "食品": "food_staples",
    "運輸・物流": "shipping",
    "エネルギー資源": "trading_companies",
    "電機・精密": "semiconductor"
}

def get_ticker_sector_sentiment(ticker_code, sentiment_map, master_map):
    code_raw = str(ticker_code).replace(".T", "").strip()

    # 1. 代表銘柄照合
    for sec_key, sec_data in sentiment_map.items():
        if code_raw in sec_data.get("tickers", []):
            return sec_data

    # 2. 33業種照合
    m_info = master_map.get(code_raw, {})
    sec33 = m_info.get("sector33", "")
    sec17 = m_info.get("sector17", "")

    matched_key = SEC33_TO_KEY.get(sec33)
    if matched_key and matched_key in sentiment_map:
        return sentiment_map[matched_key]

    # 3. 17業種フォールバック
    fallback_key = SEC17_TO_KEY.get(sec17)
    if fallback_key and fallback_key in sentiment_map:
        return sentiment_map[fallback_key]

    return None