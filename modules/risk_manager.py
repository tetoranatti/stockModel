# modules/risk_manager.py

def determine_sizing_factor(ret_1d, is_bear_candle, days_to_clear, margin_ratio, sec_advice):
    size_factor = 1.0
    size_reason = "通常ロット"

    if ret_1d <= -0.015 or (is_bear_candle and ret_1d < 0):
        size_factor = 0.5
        size_reason = f"逆張りリバウンド警戒 (前日比{ret_1d*100:+.1f}%)"
    elif days_to_clear >= 3.0 and margin_ratio >= 8.0:
        size_factor = 0.6
        size_reason = f"需給滞留警戒 (消化{days_to_clear}日)"

    # セクター助言による追加ロット抑制
    if sec_advice == "打診30%抑制":
        size_factor = round(size_factor * 0.7, 2)
        size_reason += " ＋ セクター軟調(30%抑制)"

    return size_factor, size_reason

def evaluate_screening_gate(p_win, p_stop, ev_adj, beta, vol_ratio, days_to_clear,
                            is_bear_regime, strong_buy_th, buy_threshold, watch_threshold,
                            sec_shock, sec_advice, sec_summary):
    action = "⏸️ WAIT"
    gate_reason = "見送り"

    is_hedge_candidate = is_bear_regime and (beta >= 1.15) and (p_stop >= 0.500) and (p_stop - p_win >= 0.15)

    # 「見送り」のみを即時遮断（打診30%抑制はゲートを通す）
    if sec_shock or sec_advice == "見送り":
        action = "⏸️ WAIT"
        gate_reason = f"🛑 セクターショック警戒回避 [{sec_summary}]"
    elif is_hedge_candidate:
        action = "⚠️ SHORT / HEDGE"
        gate_reason = f"地合い連動下落ヘッジ(β={beta:.2f}, 損率={p_stop*100:.1f}%)"
    elif p_win >= strong_buy_th and p_win > p_stop:
        action = "🔥 STRONG BUY"
        gate_reason = f"本買い適合(補正勝率{p_win*100:.1f}%, EV={ev_adj:+.2f}R)"
    elif p_win >= buy_threshold and p_win > p_stop and vol_ratio >= 0.85:
        if is_bear_regime and beta >= 1.0:
            action = "⏸️ WAIT"
            gate_reason = f"地合い悪化時の高β見送り(β={beta:.2f})"
        else:
            action = "🎯 BUY"
            gate_reason = f"打診買い適合(補正勝率{p_win*100:.1f}%, 消化{days_to_clear}日)"
    elif p_win >= watch_threshold and p_win > p_stop:
        action = "👀 WATCH"
        gate_reason = f"監視対象(勝率{p_win*100:.1f}%)"

    return action, gate_reason

def calculate_target_stop_levels(curr_close, curr_atr, action):
    if action == "⚠️ SHORT / HEDGE":
        target_price = round(curr_close - 2.0 * curr_atr, 1)
        stop_price = round(curr_close + 1.0 * curr_atr, 1)
    else:
        target_price = round(curr_close + 2.0 * curr_atr, 1)
        stop_price = round(curr_close - 1.0 * curr_atr, 1)
    return target_price, stop_price