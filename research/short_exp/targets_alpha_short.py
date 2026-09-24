"""5日後Alpha Return(指数相対)ターゲットとweighted Huber用サンプル重み(2026-09-24追加、
ユーザー提案)。バリア/トレーリングストップのパス依存を排除し、日経平均相対で市場ベータを
除去した単純な固定期間回帰ターゲットに切り替える。

target: alpha_5d = 個別銘柄の5営業日先(D+1始値エントリー基準、ENTRY_CONVENTION="d1_open"
       と同じ規約)リターン - 日経平均の同一期間(Close基準)リターン
weight: 1 + 2*max(0, -alpha_5d/0.03) —— alpha_5dがより深くマイナス(=空売りの本命)である
       ほど重みを増やし、ショート判断で重要な下方テールにHuber lossの学習容量を集中させる。
       (ユーザー指定の式そのまま)
"""
import numpy as np

WEIGHT_THRESHOLD = 0.03
WEIGHT_SCALE = 2.0


def compute_alpha5d_target_and_weight(df, nk_close_aligned, holding_period,
                                       weight_threshold=WEIGHT_THRESHOLD, weight_scale=WEIGHT_SCALE):
    """df: Open/Closeを含む銘柄df。nk_close_aligned: dfと同じdate indexに整列済みの日経平均Close
    (呼び出し側でreindex(df.index).ffill()済みのものを渡す)。"""
    closes, opens = df["Close"].values, df["Open"].values
    nk = nk_close_aligned.values
    n = len(closes)
    alpha = np.full(n, np.nan)
    for idx in range(n - 1 - holding_period):
        entry_p, exit_p = opens[idx + 1], closes[idx + 1 + holding_period]
        nk_entry, nk_exit = nk[idx + 1], nk[idx + 1 + holding_period]
        if nk_entry == 0 or np.isnan(nk_entry) or np.isnan(nk_exit) or np.isnan(entry_p):
            continue
        stock_ret = (exit_p - entry_p) / entry_p
        idx_ret = (nk_exit - nk_entry) / nk_entry
        alpha[idx] = stock_ret - idx_ret
    weight = 1.0 + weight_scale * np.maximum(0.0, -alpha / weight_threshold)
    return alpha, weight
