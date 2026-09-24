# modules/risk_manager.py

# ポジションサイズ自動計算の既定値。ui/index.htmlの資金・リスク・信用倍率入力欄の
# 既定値と同じにしてある(UIで値を変えた場合はそちらが優先され、こちらはCLI/CSV出力用)。
DEFAULT_CAPITAL = 3_000_000
DEFAULT_RISK_PCT = 0.8
DEFAULT_LEVERAGE = 3.3  # 制度信用の実上限(SBI含め主要証券会社共通)
# 1銘柄あたりの投資額上限(資金に対する割合)。実資金シミュレーションで検証した結果、
# 上限が無いと低ATR銘柄1〜2つが信用余力を独占し(a)後続候補が軒並み0株スキップになる
# (b)少数銘柄への集中でMaxDDが悪化する、という問題があった。
# 単一バックテストでは15%が良さそうに見えたが、営業日ブロックのブートストラップ検証
# (300回リサンプリング)では10%の方が中央値の最終資金・MaxDDともに優れていたため
# (MaxDD中央値-8.6%→-4.6%、90%CIも-7.8%〜-2.5%とタイト)、10%を採用。
DEFAULT_MAX_POSITION_PCT = 0.10

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
                            sec_shock, sec_advice, sec_summary):
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
        # 旧: flow_level=="QUIET"かつ高ベータならBUYに格下げするルールがあったが、
        # CTA/JNET関連の大口手口フローは検証した限り全て「効果なし/逆効果」という
        # 結論だった(determine_sizing_factor側のコメント参照)ため、未検証のまま
        # 残っていたこのゲート版のルールも一貫性のため撤去した。
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
    """curr_close(シグナル当日終値)基準の絶対価格でOCO水準を算出する。
    実際の約定は翌営業日の引成(大引け成行)注文を想定しており、この基準値そのものが
    エントリー価格になるわけではないが、検証の結果この絶対水準を動かさずに
    翌営業日引けで約定させる方式が最も実績を維持できたため、水準の計算自体は
    curr_close基準のまま据え置いている(modules/tracking.pyのresolve_entries()参照)。"""
    target_price = round(curr_close + 2.0 * curr_atr, 1)
    stop_price = round(curr_close - 1.0 * curr_atr, 1)
    return target_price, stop_price


def calculate_target_stop_levels_short(curr_close, curr_atr):
    """calculate_target_stop_levels()の空売り版(2026-09-24追加)。ロングとは損益の向きが
    逆なので、targetはcurr_closeより下(値下がりで利益)、stopはcurr_closeより上
    (値上がりで損失)になる。ATR倍率(target 2.0倍・stop 1.0倍)はロング側と揃えている
    (research/short_exp側もロングと同じ固定バリア規約で検証済み、[[short_selling_model_2026-09-24]])。"""
    target_price = round(curr_close - 2.0 * curr_atr, 1)
    stop_price = round(curr_close + 1.0 * curr_atr, 1)
    return target_price, stop_price


def calculate_recommended_position(price, stop_price, size_factor,
                                    capital=DEFAULT_CAPITAL, risk_pct=DEFAULT_RISK_PCT,
                                    leverage=DEFAULT_LEVERAGE, max_position_pct=DEFAULT_MAX_POSITION_PCT,
                                    side="long"):
    """ui/src/app.js の updatePositionSize() と同じ計算式(リスクベース株数・
    信用余力(資金×レバレッジ倍率)上限株数・1銘柄あたり投資上限株数のうち
    最小のもの、100株単位丸め)で推奨ポジションサイズを算出する。UIを開かず
    スクリーニング結果だけで発注準備ができるように、CSV/JSON出力に直接含めるためのもの。

    side="short"(2026-09-24追加): 空売りはstop_priceがpriceより上にあるため、
    risk_per_shareの向きを反転する(price - stop_price ではなく stop_price - price)。"""
    risk_per_share = max(1.0, (stop_price - price) if side == "short" else (price - stop_price))
    max_risk = capital * (risk_pct / 100.0) * size_factor
    risk_based_shares = int(max_risk / (risk_per_share * 100)) * 100

    max_buying_power = capital * leverage
    leverage_cap_shares = int(max_buying_power / (price * 100)) * 100

    max_pos_yen = capital * max_position_pct
    max_pos_shares = int(max_pos_yen / (price * 100)) * 100

    shares = max(0, min(risk_based_shares, leverage_cap_shares, max_pos_shares))
    is_leverage_capped = leverage_cap_shares < risk_based_shares and shares > 0
    is_maxpos_capped = max_pos_shares < min(risk_based_shares, leverage_cap_shares) and shares > 0

    # 0株になった場合、どの上限が原因かを切り分ける(優先順位: risk -> leverage -> maxpos)。
    # UI/CSVで「なぜ見送りになったか」を表示するための診断情報。
    skip_reason = None
    if shares <= 0:
        if risk_based_shares <= 0:
            skip_reason = "risk"       # 1ATRの損切幅が広すぎてリスク予算内で100株すら買えない
        elif leverage_cap_shares <= 0:
            skip_reason = "leverage"   # 信用枠(資金×レバレッジ)を使い切っている
        elif max_pos_shares <= 0:
            skip_reason = "maxpos"     # 株価が高く、1銘柄上限%の枠内で100株すら買えない

    return {
        "recommended_shares": shares,
        "estimated_cost_yen": round(shares * price),
        "estimated_max_loss_yen": round(shares * risk_per_share),
        "leverage_capped": is_leverage_capped,
        "maxpos_capped": is_maxpos_capped,
        "skip_reason": skip_reason,
    }


def build_portfolio(candidates, capital=DEFAULT_CAPITAL, risk_pct=DEFAULT_RISK_PCT,
                     leverage=DEFAULT_LEVERAGE, max_positions=20,
                     max_position_pct=DEFAULT_MAX_POSITION_PCT, side="long"):
    """複数銘柄をまとめて買う前提で、資金・レバレッジ上限を全銘柄で共有しながら
    優先順位(candidatesに渡された順序)順に配分するポートフォリオを組む。

    calculate_recommended_position()は「その銘柄だけを買う」前提でフル資金を使って
    計算するため、複数のSTRONG BUY/BUYを同時に買うと資金・信用枠を使い過ぎてしまう
    (例: 5銘柄が同時にSTRONG BUYの日、各銘柄の推奨株数を素直に合計すると
    信用余力を5倍近く超過しうる)。この関数はそれを避けるため、残り資金枠を
    順に消費しながら株数を決める(データ想定資金シミュレーション: simulate_real_capital.py
    のK同時保有ロジックと同じ発想を1日分のスナップショットに適用したもの)。

    max_position_pct: 1銘柄あたりの投資額を資金の何%までに制限するか。実資金
    シミュレーションで検証した結果、上限が無いと低ATR銘柄1〜2つが信用余力を
    独占し(a)後続候補が軒並み0株スキップになる(b)集中でMaxDDが悪化する、
    という問題があった。15%に制限すると0株スキップ率68.9%→2.2%、
    MaxDD-21.2%→-16.7%、最終リターン+14.0%→+27.5%と全指標が改善したため導入。

    candidates: [{ticker, price, stop_price, size_factor, ...}, ...] (優先順位順)
    side="short"(2026-09-24追加): 空売り用。risk_per_shareの向きを反転する
    (calculate_recommended_position()と同じ)。ショート候補のcapitalは呼び出し側で
    別枠(regime_momentum.determine_capital_split()のcapital_short)を渡すこと。
    戻り値: (portfolio_rows, summary) — portfolio_rowsは採用された銘柄のみ、
    元のdictに recommended_shares/estimated_cost_yen/estimated_max_loss_yen/
    leverage_capped/maxpos_capped を追加したもの。
    """
    max_buying_power = capital * leverage
    max_pos_yen = capital * max_position_pct
    used_buying_power = 0.0
    total_risk_yen = 0.0
    portfolio_rows = []
    skip_reason_counts = {"risk": 0, "leverage": 0, "maxpos": 0, "position_limit": 0}

    for item in candidates:
        if len(portfolio_rows) >= max_positions:
            skip_reason_counts["position_limit"] += 1
            continue  # max_positionsに到達済み(以降の候補も採用できないが件数だけ集計する)

        price = float(item['price'])
        stop_price = float(item['stop_price'])
        size_factor = float(item.get('size_factor', 1.0))
        risk_per_share = max(1.0, (stop_price - price) if side == "short" else (price - stop_price))

        max_risk = capital * (risk_pct / 100.0) * size_factor
        risk_based_shares = int(max_risk / (risk_per_share * 100)) * 100

        remaining_buying_power = max(0.0, max_buying_power - used_buying_power)
        leverage_cap_shares = int(remaining_buying_power / (price * 100)) * 100

        max_pos_shares = int(max_pos_yen / (price * 100)) * 100

        shares = max(0, min(risk_based_shares, leverage_cap_shares, max_pos_shares))
        if shares <= 0:
            # どの上限が原因で見送りになったかを切り分ける(優先順位: risk -> leverage -> maxpos)。
            # UI/CSVで「なぜ見送りになったか」を表示するための診断情報
            # (calculate_recommended_position()のskip_reasonと同じロジック)。
            if risk_based_shares <= 0:
                skip_reason_counts["risk"] += 1
            elif leverage_cap_shares <= 0:
                skip_reason_counts["leverage"] += 1
            elif max_pos_shares <= 0:
                skip_reason_counts["maxpos"] += 1
            continue  # 残り資金枠が尽きた、1銘柄上限を超えている、または最低100株すら買えない

        cost = shares * price
        risk_yen = shares * risk_per_share
        used_buying_power += cost
        total_risk_yen += risk_yen

        row = dict(item)
        row.update({
            "recommended_shares": shares,
            "estimated_cost_yen": round(cost),
            "estimated_max_loss_yen": round(risk_yen),
            "leverage_capped": leverage_cap_shares < risk_based_shares,
            "maxpos_capped": max_pos_shares < min(risk_based_shares, leverage_cap_shares),
        })
        portfolio_rows.append(row)

    summary = {
        "side": side,
        "capital": capital,
        "leverage": leverage,
        "max_buying_power": max_buying_power,
        "used_buying_power_yen": round(used_buying_power),
        "buying_power_usage_pct": round(used_buying_power / max_buying_power * 100, 1) if max_buying_power > 0 else 0.0,
        "total_risk_yen": round(total_risk_yen),
        "n_positions": len(portfolio_rows),
        "n_candidates_evaluated": len(candidates),
        "skip_reason_counts": skip_reason_counts,
    }
    return portfolio_rows, summary