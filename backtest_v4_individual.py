import os
import torch
import pandas as pd
import numpy as np
from swing_pipeline_v4_experimental import DualStreamGRU_v4, load_and_preprocess_v4

def run_single_model_backtest(ticker: str, name: str, fee_rate: float = 0.001):
    model_path = f"swing_model_v4_{ticker.split('.')[0]}.pt"
    if not os.path.exists(model_path):
        return None

    df, _, _ = load_and_preprocess_v4(ticker)
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    stock_cols = checkpoint["stock_cols"]
    macro_cols = checkpoint["macro_cols"]

    model = DualStreamGRU_v4(stock_dim=len(stock_cols), macro_dim=len(macro_cols), hidden_dim=44, num_classes=3)
    model.load_state_dict(state_dict)
    model.eval()

    seq_len = 10
    trades = []
    holding = False
    entry_price = 0.0
    upper_barrier = 0.0
    lower_barrier = 0.0
    holding_days = 0
    entry_date = None

    dates = df.index
    n = len(df)

    for i in range(seq_len, n - 10):
        current_date = dates[i]

        if not holding:
            seq_stock = df[stock_cols].iloc[i - seq_len + 1 : i + 1].values
            seq_macro = df[macro_cols].iloc[i - seq_len + 1 : i + 1].values

            s_mean, s_std = seq_stock.mean(axis=0), seq_stock.std(axis=0) + 1e-7
            m_mean, m_std = seq_macro.mean(axis=0), seq_macro.std(axis=0) + 1e-7
            seq_stock = (seq_stock - s_mean) / s_std
            seq_macro = (seq_macro - m_mean) / m_std

            t_stock = torch.tensor(seq_stock, dtype=torch.float32).unsqueeze(0)
            t_macro = torch.tensor(seq_macro, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                logits = model(t_stock, t_macro)
                probs = torch.softmax(logits, dim=-1).numpy()[0]

            p_loss, p_wait, p_profit = probs[0], probs[1], probs[2]

            # 判定基準
            if (p_profit >= 0.40) and (p_profit >= p_loss * 0.8):
                holding = True
                holding_days = 0
                entry_date = dates[i + 1]
                entry_price = float(df['Open'].iloc[i + 1] * (1.0 + fee_rate))
                atr_val = float(df['ATR'].iloc[i])
                upper_barrier = entry_price + (2.0 * atr_val)
                lower_barrier = entry_price - (1.0 * atr_val)
                continue

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
                exit_price = lower_barrier * (1.0 - fee_rate)
                exit_type = "LOSS (同日接触)"
            elif hit_profit:
                exit_price = upper_barrier * (1.0 - fee_rate)
                exit_type = "PROFIT (利確)"
            elif hit_loss:
                exit_price = lower_barrier * (1.0 - fee_rate)
                exit_type = "LOSS (損切)"
            elif holding_days >= 10:
                exit_price = close_p * (1.0 - fee_rate)
                exit_type = "TIMEOUT (期限)"

            if exit_type:
                ret = (exit_price - entry_price) / entry_price
                trades.append({
                    "ticker": ticker,
                    "name": name,
                    "holding_days": holding_days,
                    "exit_type": exit_type,
                    "return_pct": round(ret * 100, 2),
                    "return_factor": 1.0 + ret
                })
                holding = False

    return pd.DataFrame(trades)

if __name__ == "__main__":
    targets = {
        "7012.T": "川崎重工業",
        "8227.T": "しまむら",
        "3407.T": "旭化成",
        "8593.T": "三菱HCキャピタル"
    }

    all_summaries = []
    for sym, nm in targets.items():
        res_df = run_single_model_backtest(sym, nm)
        if res_df is None or len(res_df) == 0:
            all_summaries.append({"銘柄": f"{nm} ({sym})", "取引数": 0, "勝率(%)": 0.0, "PF": 0.0, "累積リターン(%)": 0.0})
            continue

        total = len(res_df)
        wins = res_df[res_df['return_pct'] > 0]
        losses = res_df[res_df['return_pct'] <= 0]
        win_rate = (len(wins) / total) * 100
        gain = wins['return_pct'].sum()
        loss = abs(losses['return_pct'].sum())
        pf = (gain / loss) if loss > 0 else 99.9
        cum_ret = (res_df['return_factor'].prod() - 1.0) * 100

        all_summaries.append({
            "銘柄": f"{nm} ({sym})",
            "取引数": total,
            "勝率(%)": round(win_rate, 1),
            "PF": round(pf, 2),
            "累積リターン(%)": round(cum_ret, 2)
        })

    print("\n" + "="*70)
    print("[*] 個別最適化 v4モデル バックテスト結果サマリー")
    print("="*70)
    print(pd.DataFrame(all_summaries).to_string(index=False))