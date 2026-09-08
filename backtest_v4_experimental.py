import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import yfinance as yf
from swing_pipeline_v4_experimental import DualStreamGRU_v4, load_and_preprocess_v4

# =============================================================================
# バックテスト実行関数 (1銘柄単位)
# =============================================================================
def evaluate_target(ticker: str, name: str, model, stock_cols, macro_cols, use_macro_filter: bool = True, fee_rate: float = 0.001):
    df, _, _ = load_and_preprocess_v4(ticker)
    if len(df) < 50:
        return None

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

        # ポジション未保有時：推論とエントリー判定
        if not holding:
            seq_stock = df[stock_cols].iloc[i - seq_len + 1 : i + 1].values
            seq_macro = df[macro_cols].iloc[i - seq_len + 1 : i + 1].values

            # シーケンス標準化
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

            # 基本シグナル: 利確確率40%以上 かつ 損切確率の0.8倍以上
            is_signal = (p_profit >= 0.40) and (p_profit >= p_loss * 0.8)

            # 地合いフィルター: ガンマ対数距離が -0.02以下(ネガティブガンマ急落地帯)の場合はエントリー拒絶
            if use_macro_filter:
                gamma_dist = df['gamma_log_dist'].iloc[i]
                is_safe_macro = (gamma_dist >= -0.02)
            else:
                is_safe_macro = True

            if is_signal and is_safe_macro:
                holding = True
                holding_days = 0
                entry_date = dates[i + 1]
                entry_price = float(df['Open'].iloc[i + 1] * (1.0 + fee_rate))
                atr_val = float(df['ATR'].iloc[i])
                upper_barrier = entry_price + (2.0 * atr_val)
                lower_barrier = entry_price - (1.0 * atr_val)
                continue

        # ポジション保有中：トリプルバリア決済判定
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

    return pd.DataFrame(trades)

# =============================================================================
# 全銘柄一括実行
# =============================================================================
def run_all_targets_backtest(model_path: str = "swing_model_v4_test.pt"):
    print(f"\n{'='*70}")
    print(f"[*] 実験用モデル (v4) 全4銘柄・一括バックテスト実行")
    print(f"{'='*70}")

    if not os.path.exists(model_path):
        print(f"[-] モデルファイル '{model_path}' が見つかりません。")
        return

    checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    stock_cols = checkpoint.get("stock_cols", ['stock_ret_1d', 'stock_ret_5d', 'atr_ratio', 'rolling_beta'])
    macro_cols = checkpoint.get("macro_cols", None)

    stock_dim = len(stock_cols)
    macro_dim = len(macro_cols)
    hidden_dim = state_dict['stock_gru.weight_hh_l0'].shape[1]

    model = DualStreamGRU_v4(stock_dim=stock_dim, macro_dim=macro_dim, hidden_dim=hidden_dim, num_classes=3)
    model.load_state_dict(state_dict)
    model.eval()

    targets = {
        "川崎重工業": "7012.T",
        "しまむら": "8227.T",
        "旭化成": "3407.T",
        "三菱HCキャピタル": "8593.T"
    }

    all_summaries = []

    for name, sym in targets.items():
        res_df = evaluate_target(sym, name, model, stock_cols, macro_cols, use_macro_filter=True)
        
        if res_df is None or len(res_df) == 0:
            all_summaries.append({
                "銘柄": f"{name} ({sym})",
                "取引回数": 0,
                "勝率 (%)": 0.0,
                "PF": 0.0,
                "通算リターン (%)": 0.0,
                "平均保有日数": 0.0
            })
            continue

        total_trades = len(res_df)
        win_trades = res_df[res_df['return_pct'] > 0]
        loss_trades = res_df[res_df['return_pct'] <= 0]
        win_rate = (len(win_trades) / total_trades) * 100
        total_gain = win_trades['return_pct'].sum()
        total_loss = abs(loss_trades['return_pct'].sum())
        profit_factor = (total_gain / total_loss) if total_loss > 0 else np.nan
        cum_return = (res_df['return_factor'].prod() - 1.0) * 100

        all_summaries.append({
            "銘柄": f"{name} ({sym})",
            "取引回数": total_trades,
            "勝率 (%)": round(win_rate, 1),
            "PF": round(profit_factor, 2) if not np.isnan(profit_factor) else 99.9,
            "通算リターン (%)": round(cum_return, 2),
            "平均保有日数": round(res_df['holding_days'].mean(), 1)
        })

    # 結果サマリーテーブル
    summary_table = pd.DataFrame(all_summaries)
    print("\n[v4モデル 全銘柄パフォーマンス比較表 (地合いフィルター適用)]")
    print(summary_table.to_string(index=False))

if __name__ == "__main__":
    run_all_targets_backtest()