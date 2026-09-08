import pandas as pd
import yfinance as yf
from run_sbi_pipeline import load_sbi_csv

# 先頭100銘柄を抽出してテスト
tickers = load_sbi_csv("screener_result.csv")[:100]
print(f"[*] 先頭100銘柄を対象に各フィルターの脱落要因を分析中...\n")

reasons = {
    "1. データ本数不足(<220本)": 0,
    "2. 200日線未満(下降トレンド)": 0,
    "3. RSI範囲外(30〜55外)": 0,
    "4. MACD未合致(GC/反転なし)": 0,
    "5. 全条件クリア(通過)": 0
}

passed_candidates = []

for idx, t in enumerate(tickers, 1):
    print(f"[{idx:03d}/100] {t} をチェック中...", end="\r")
    
    # 期間不足を防ぐため period='2y' に指定
    df = yf.download(t, period="2y", interval="1d", progress=False)
    if df.empty:
        reasons["1. データ本数不足(<220本)"] += 1
        continue
        
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna()

    if len(df) < 220:
        reasons["1. データ本数不足(<220本)"] += 1
        continue

    close = df['Close']
    sma_200 = close.rolling(200).mean().iloc[-1]

    # 1. 200日移動平均線
    if close.iloc[-1] < sma_200:
        reasons["2. 200日線未満(下降トレンド)"] += 1
        continue

    # 2. RSI(14)
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rsi_val = (100 - (100 / (1 + (gain / (loss + 1e-7))))).iloc[-1]
    
    if not (30 <= rsi_val <= 55):
        reasons["3. RSI範囲外(30〜55外)"] += 1
        continue

    # 3. MACD反転/GC
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_hist = (ema12 - ema26) - (ema12 - ema26).ewm(span=9, adjust=False).mean()
    
    hist_recent = macd_hist.iloc[-3:]
    macd_cross = (macd_hist.iloc[-2] <= 0 and macd_hist.iloc[-1] > 0) or \
                 (macd_hist.iloc[-3] <= 0 and macd_hist.iloc[-2] > 0) or \
                 (hist_recent.iloc[-1] > hist_recent.iloc[-2] > hist_recent.iloc[-3] and hist_recent.iloc[-1] > 0)

    if not macd_cross:
        reasons["4. MACD未合致(GC/反転なし)"] += 1
        continue

    reasons["5. 全条件クリア(通過)"] += 1
    passed_candidates.append({
        "Ticker": t,
        "Close": round(float(close.iloc[-1]), 1),
        "RSI": round(float(rsi_val), 1)
    })

print("\n" + "=" * 50)
print("📊 フィルター別脱落内訳 (100銘柄中)")
print("=" * 50)
for k, v in reasons.items():
    print(f"  {k:30s}: {v:3d} 銘柄")
print("=" * 50)

if passed_candidates:
    print("\n[★ 通過した銘柄]")
    print(pd.DataFrame(passed_candidates).to_string(index=False))
else:
    print("\n[-] 100銘柄中、全条件を同時に満たした銘柄はありませんでした。")