# modules/risk_manager.py

SECTOR_SIZE_WEIGHT_COEF = 0.3  # sec_score(-1.0〜+1.0)に対するサイズ重みの傾き

def sector_size_weight(sec_score):
    """セクターセンチメントスコア(-1.0〜+1.0)を連続的なサイズ倍率に変換する。
    score=-1.0で0.7倍、0で1.0倍、+1.0で1.3倍(SECTOR_SIZE_WEIGHT_COEF=0.3の場合)。
    旧実装は「打診30%抑制」時のみ一律0.7倍だったが、スコアの強弱を反映するよう連続化。"""
    return round(1.0 + sec_score * SECTOR_SIZE_WEIGHT_COEF, 3)


def determine_sizing_factor(ret_1d, is_bear_candle, days_to_clear, margin_ratio, sec_score):
    size_factor = 1.0
    size_reason = "通常ロット"

    if ret_1d <= -0.015 or (is_bear_candle and ret_1d < 0):
        size_factor = 0.5
        size_reason = f"逆張りリバウンド警戒 (前日比{ret_1d*100:+.1f}%)"
    elif days_to_clear >= 3.0 and margin_ratio >= 8.0:
        size_factor = 0.6
        size_reason = f"需給滞留警戒 (消化{days_to_clear}日)"

    # セクターセンチメントスコアに連動した連続的なロット調整
    sector_mult = sector_size_weight(sec_score)
    if sector_mult != 1.0:
        size_factor = round(size_factor * sector_mult, 2)
        size_reason += f" ＋ セクターセンチメント({sec_score:+.2f} -> {sector_mult:.2f}倍)"

    # 大口手口フロー(CTA/J-NET)によるロット調整は、1年分の実データでバックテストした結果
    # 撤廃(何も調整しない)が本番閾値でのリスク調整後リターンが最良だったため不採用。
    # 特にCTA_SURGE時の0.7倍抑制は、実際にはCTA比率が高い日ほど翌リターンが良い傾向
    # (上位20%で日次平均+1.4%)と逆方向の調整だった。データはdata/flow_signal_daily_cache.csv
    # に日次蓄積を継続しており、将来別の形で特徴量として再検討する。

    return size_factor, size_reason

def evaluate_screening_gate(p_win, p_stop, ev_adj, beta, vol_ratio, days_to_clear,
                            is_bear_regime, strong_buy_th, buy_threshold, watch_threshold,
                            sec_shock, sec_advice, sec_summary, flow_level="NORMAL"):
    action = "⏸️ WAIT"
    gate_reason = "見送り"

    # p_stopのヘッジ判定(p_stop>=0.500)は廃止。9特徴量+near-pairランキング損失版で
    # p_stopの絶対値と実際のリターンの関係を検証したところ、あらゆる学習方式
    # (3クラスsoftmax/単一sigmoid/独立2モデル/共有ヘッド+BCE較正、重み0〜0.3)を
    # 試しても「p_stopが高いほど実際のリターンが悪化する」という較正が一貫して
    # 得られなかった(むしろ逆の傾向)ため、絶対閾値によるヘッジ判定は機能しないと判断し無効化。
    if sec_shock or sec_advice == "見送り":
        action = "⏸️ WAIT"
        gate_reason = f"🛑 セクターショック警戒回避 [{sec_summary}]"
    elif p_win >= strong_buy_th and p_win > p_stop:
        if flow_level == "QUIET" and beta >= 1.2:
            action = "🎯 BUY"
            gate_reason = f"大口薄商いによる高β抑制(元本買い, β={beta:.2f})"
        else:
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

def calculate_target_stop_levels(curr_close, curr_atr):
    target_price = round(curr_close + 2.0 * curr_atr, 1)
    stop_price = round(curr_close - 1.0 * curr_atr, 1)
    return target_price, stop_price