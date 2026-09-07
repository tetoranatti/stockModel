import os
import pandas as pd

DB_FILE = "jpx_daily_features_db.csv"

# 2026年9月7日時点のJPX最新手口・建玉公表データ
initial_records = [
    {
        'date': '2026-09-07',
        'call_wall_strike': 66500.0,        # 上値最大コール壁 (C66,500)
        'put_wall_strike': 65000.0,         # 下値最大プット壁 (P65,000)
        'max_pain': 63500.0,                # 9月限 Max Pain
        'pcr_oi': 1.12,                     # プット・コール・レシオ
        'abn_net_trade': -420,              # ABNクリアリン先物純売買高 (枚)
        'arbitrage_buy_stocks': 1180000.0   # 裁定買い残高 (千株)
    }
]

df = pd.DataFrame(initial_records)
df.to_csv(DB_FILE, index=False)
print(f"[OK] 初期データベースファイルを作成しました: {DB_FILE}")
print(df)