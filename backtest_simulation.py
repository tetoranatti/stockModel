import pandas as pd
import numpy as np
import yfinance as yf

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR (Average True Range) を計算"""
    high_low = df['High'] - df['Low']
    high_cp = (df['High'] - df['Close'].shift(1)).abs()
    low_cp = (df['Low'] - df['Close'].shift(1)).abs()
    tr = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def run_backtest(ticker: str, start_date: str = "2024-01-01", fee_rate: float = 0.001):
    """
    トリプルバリア法によるバックテスト実行関数
    - 上限バリア: 翌日始値 + 2.0 * 当日ATR
    - 下限バリア: 翌日始値 - 1.0 * 当日ATR
    - 時間バリア: 最大10営業日
    """
    print(f"\n{'='*55}")
    print(f"[*] バックテスト実行: {ticker} (期間: {start_date} 〜 直近)")
    print(f"{'='*55}")

    # 株価データ取得
    df = yf.download(ticker, start=start_date, interval="1d", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if len(df) < 50:
        print("[-] データ数が不足しています。")
        return None

    # テクニカル指標の計算
    df['ATR'] = calculate_atr(df, 14)
    # 単純なモメンタム例（25日移動平均線より上、またはRSI等のトリガー）
    # ※ 本番モデル推論ログと紐付ける場合は、ここにBUYフラグ列を代入
    df['SMA25'] = df['Close'].rolling(25).mean()
    df['RSI'] = 100 - (100 / (1 + (df['Close'].diff().clip(lower=0).rolling(14).mean() / 
                                   (-df['Close'].diff().clip(upper=0)).rolling(14).mean())))
    
    # エントリー判定シグナル（例: 押し目・反発の初動）
    # 25日線上かつRSIが40以上60以下で上向き、または任意ロジック
    df['Signal'] = (df['Close'] > df['SMA25']) & (df['RSI'] > 45) & (df['RSI'] < 65)

    df = df.dropna().copy()
    trades = []
    holding = False
    entry_price = 0.0
    upper_barrier = 0.0
    lower_barrier = 0.0
    holding_days = 0
    entry_date = None

    dates = df.index
    n = len(df)

    for i in range(n - 10):
        current_date = dates[i]
        
        # エントリー判定（ポジション未保有時）
        if not holding:
            if df['Signal'].iloc[i]:
                holding = True
                holding_days = 0
                entry_date = dates[i + 1]
                # 翌営業日寄り付きでエントリー（スリッページ・手数料考慮）
                entry_price = float(df['Open'].iloc[i + 1] * (1.0 + fee_rate))
                atr_val = float(df['ATR'].iloc[i])
                upper_barrier = entry_price + (2.0 * atr_val)
                lower_barrier = entry_price - (1.0 * atr_val)
                continue

        # エグジット監視（保有中）
        if holding:
            holding_days += 1
            high_p = float(df['High'].iloc[i])
            low_p = float(df['Low'].iloc[i])
            close_p = float(df['Close'].iloc[i])

            hit_profit = high_p >= upper_barrier
            hit_loss = low_p <= lower_barrier

            exit_price = None
            exit_type = None

            if hit_profit and hit_loss:
                # 同日タッチは保守的に「損切」として扱う
                exit_price = lower_barrier * (1.0 - fee_rate)
                exit_type = "LOSS (同日接触)"
            elif hit_profit:
                exit_price = upper_barrier * (1.0 - fee_rate)
                exit_type = "PROFIT (利確)"
            elif hit_loss:
                exit_price = lower_barrier * (1.0 - fee_rate)
                exit_type = "LOSS (損切)"
            elif holding_days >= 10:
                # 10営業日経過による時間切れ手仕舞い
                exit_price = close_p * (1.0 - fee_rate)
                exit_type = "TIMEOUT (期限)"

            if exit_type:
                ret = (exit_price - entry_price) / entry_price
                trades.append({
                    "entry_date": entry_date.strftime('%Y-%m-%d'),
                    "exit_date": current_date.strftime('%Y-%m-%d'),
                    "entry_price": round(entry_price, 1),
                    "exit_price": round(exit_price, 1),
                    "holding_days": holding_days,
                    "exit_type": exit_type,
                    "return_pct": round(ret * 100, 2),
                    "return_factor": 1.0 + ret
                })
                holding = False

    res_df = pd.DataFrame(trades)
    if len(res_df) == 0:
        print("[-] 条件に一致するトレードが発生しませんでした。")
        return None

    # パフォーマンス集計
    total_trades = len(res_df)
    win_trades = res_df[res_df['return_pct'] > 0]
    loss_trades = res_df[res_df['return_pct'] <= 0]
    win_rate = (len(win_trades) / total_trades) * 100
    
    total_gain = win_trades['return_pct'].sum()
    total_loss = abs(loss_trades['return_pct'].sum())
    profit_factor = (total_gain / total_loss) if total_loss > 0 else np.nan

    cum_return = (res_df['return_factor'].prod() - 1.0) * 100
    avg_holding = res_df['holding_days'].mean()

    # 表示
    print(f"総トレード数       : {total_trades} 回")
    print(f"勝率 (利確/全取引) : {win_rate:.1f} %")
    print(f"プロフィットファクター: {profit_factor:.2f}")
    print(f"通算累積リターン   : {cum_return:+.2f} %")
    print(f"平均保有日数       : {avg_holding:.1f} 営業日")
    print("\n[直近5トレードの内訳]")
    print(res_df[['entry_date', 'exit_date', 'exit_type', 'holding_days', 'return_pct']].tail(5).to_string(index=False))
    return res_df

if __name__ == "__main__":
    # 保有銘柄・検討銘柄を順にバックテスト
    target_tickers = {
        "川崎重工業": "7012.T",
        "しまむら": "8227.T",
        "旭化成": "3407.T",
        "三菱HCキャピタル": "8593.T"
    }
    
    for name, sym in target_tickers.items():
        run_backtest(sym, start_date="2024-01-01")