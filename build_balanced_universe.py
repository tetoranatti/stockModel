# build_balanced_universe.py
import os
import glob
import json
import pandas as pd

BASE_DIR = r"F:\stockModel"
DATA_DIR = os.path.join(BASE_DIR, "data")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
SECTOR_MASTER_JSON = os.path.join(DATA_DIR, "jpx_sector_master.json")
UNIVERSE_OUTPUT = os.path.join(BASE_DIR, "universe_150_tickers.txt")

MIN_TURNOVER = 10e8  # 10億円
TARGET_TOTAL = 150

def get_latest_cache():
    files = glob.glob(os.path.join(CACHE_DIR, "prices_*.parquet"))
    if not files:
        raise FileNotFoundError("prices_*.parquet が見つかりません。build_jquants_cache.py を先に実行してください。")
    return max(files, key=os.path.getmtime)

def main():
    print("=" * 70)
    print("【セクター均等配分】ユニバース150銘柄 生成 (売買代金 >= 10億円)")
    print("=" * 70)

    # 1. セクターマスター読込
    if not os.path.exists(SECTOR_MASTER_JSON):
        raise FileNotFoundError(f"{SECTOR_MASTER_JSON} が見つかりません。")
    with open(SECTOR_MASTER_JSON, "r", encoding="utf-8") as f:
        master_map = json.load(f)

    # 2. 直近の株価キャッシュ読込
    cache_file = get_latest_cache()
    print(f"[*] キャッシュ読込: {os.path.basename(cache_file)}")
    df = pd.read_parquet(cache_file)

    # 3. 銘柄ごとの5日平均売買代金とセクターの集計
    records = []
    tickers = df.columns.levels[0]

    for t in tickers:
        sub = df[t].dropna()
        if len(sub) < 5:
            continue
        turnover_5d = (sub['Close'] * sub['Volume']).tail(5).mean()
        if turnover_5d < MIN_TURNOVER:
            continue

        raw_code = t.replace(".T", "").strip()
        sec_info = master_map.get(raw_code, {})
        sector33 = sec_info.get("sector33", "その他")

        records.append({
            "ticker": t,
            "turnover": turnover_5d,
            "sector": sector33
        })

    cand_df = pd.DataFrame(records)
    print(f"[*] 条件合致（売買代金 >= 10億円）: {len(cand_df)} 銘柄")

    if cand_df.empty:
        print("[-] 該当銘柄がありません。")
        return

    # 4. セクター別に売買代金順ソート
    cand_df = cand_df.sort_values(by=["sector", "turnover"], ascending=[True, False])

    # 5. 各セクターから均等に選出（ラウンドロビン方式）
    sectors = cand_df["sector"].unique()
    selected = []
    
    # 枠が埋まるまで、各セクターの1位、2位、3位...と順番に拾い上げる
    rank = 0
    while len(selected) < TARGET_TOTAL:
        added_in_round = False
        for sec in sectors:
            sec_rows = cand_df[cand_df["sector"] == sec]
            if len(sec_rows) > rank:
                selected.append(sec_rows.iloc[rank]["ticker"])
                added_in_round = True
                if len(selected) == TARGET_TOTAL:
                    break
        rank += 1
        if not added_in_round:
            break

    # 150銘柄に満たない場合は残りの売買代金上位から補充
    if len(selected) < TARGET_TOTAL:
        remains = cand_df[~cand_df["ticker"].isin(selected)].sort_values(by="turnover", ascending=False)
        needed = TARGET_TOTAL - len(selected)
        selected.extend(remains["ticker"].head(needed).tolist())

    # 6. 保存とサマリ出力
    selected = sorted(selected)
    with open(UNIVERSE_OUTPUT, "w", encoding="utf-8") as f:
        for t in selected:
            f.write(f"{t}\n")

    print(f"\n[+] ユニバース保存完了: {UNIVERSE_OUTPUT} ({len(selected)} 銘柄)")
    
    # セクター別構成数の確認
    summary = cand_df[cand_df["ticker"].isin(selected)]["sector"].value_counts()
    print("\n【セクター別採用数】")
    for sec, count in summary.items():
        print(f"  - {sec:<12}: {count} 銘柄")

if __name__ == "__main__":
    main()