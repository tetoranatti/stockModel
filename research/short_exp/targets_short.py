"""空売りモデル用のラベル/バックテスト・シミュレーション(2026-09-24追加、ユーザー提案
「個別銘柄空売りランキングモデルをロング版から新規に作る」)。

stage3_exp.targets.simulate_ret_pct_d1_close(barrier_mode="trailing")の鏡像。ロング版は
「安値がストップを下回ったら損切り、高値の更新に応じてストップを切り上げる」だが、
空売り版は「高値がストップを上回ったら損切り(踏み上げ)、安値の更新に応じてストップを
切り下げる」——高値/安値の役割とストップの方向を全て反転させたもの。ロング版と同じく
保守的な判定順序(その日ストップを切り下げる前に、まず不利な方向(高値)がストップに
触れていないか確認する)を踏襲する。

v1では現行採用中のBARRIER_MODE="trailing"のみ対応(fixedモードは未実装、必要になれば
stage3_exp.targets.simulate_ret_pct_d1_closeのfixed分岐を同様に反転させて追加する)。
entry_convention="d1_open_pullback"/"d1_open_pullback_high"はロング買いの悲観シナリオ
(始値より不利な価格で約定したと仮定)として設計されたもので、空売り方向での妥当性は
未検証のままロング版と同じ式を流用している——"d1_open"(現行設定)以外を使う場合は
要再検討。

**exit_class(2026-09-24追加、ユーザー提案「TP/SL到達順序型ラベル」)**: 現行の
BARRIER_MODE="trailing"には固定の利確バリアが無い(トレーリングのみ)ため、素直な
「TPが先かSLが先か」の順序ラベルは作れない。代わりに決済理由で離散化する
(triple-barrier法のtimeout=0扱いに相当する考え方をトレーリングに適応):
    0=Stop  : 初期ストップ(トレーリング未発動)に引っかかって決済 -> 必ず損失側
    2=Win   : トレーリング発動後のストップに引っかかって決済 -> 発動後のストップは
              常にentry_pより有利側にあるため必ず利益側(下記の通り、活性化条件から
              数学的に保証される)
    1=Hold  : holding_period到達まで一度もストップに触れなかった(タイムアウト、
              含み損益の符号は問わない)
exit_p=None(タイムアウト)以外の全ケースでexit_classが定まる。目的関数を交差エントロピー
に変える場合に使う(NearPairRankingLossRegimeAwareのペアワイズ順位学習とは非対称で、
サンプル単位の分類問題になる)。
"""
import numpy as np


def simulate_short_ret_pct_d1_close(
    closes,
    highs,
    lows,
    atrs,
    holding_period,
    opens=None,
    entry_convention="d1_open",
    atr_barrier_upper=2.0,
    atr_barrier_lower=1.0,
    atr_trail_activation=1.0,
    atr_trail_distance=0.5,
):
    """空売りポジションのret_pct(正=空売り利益)とexit_h_daysを返す。
    stage3_exp.targets.simulate_ret_pct_d1_close(barrier_mode="trailing")と同じ
    シグネチャ・戻り値形式(targets, exit_h_days)。regime別バリアは未対応(fixedのみ
    対応していたロング版のUSE_REGIME_ATR_BARRIERと同様、v1では見送り)。"""
    n_bars = len(closes)
    targets = np.full(n_bars, np.nan)
    exit_h_days = np.full(n_bars, -1, dtype=int)
    exit_class = np.full(n_bars, -1, dtype=int)
    start_h = (
        1
        if entry_convention in ("d1_open", "d1_open_pullback", "d1_open_pullback_high")
        else 2
    )
    for idx in range(n_bars - holding_period):
        if entry_convention == "d1_open":
            entry_p = opens[idx + 1]
        elif entry_convention == "d1_open_pullback":
            entry_p = (opens[idx + 1] + lows[idx + 1]) / 2.0
        elif entry_convention == "d1_open_pullback_high":
            entry_p = (opens[idx + 1] + highs[idx + 1]) / 2.0
        else:
            entry_p = closes[idx + 1]

        # 初期ストップはentry_pより上(空売りは価格上昇で損失)。トレーリングは価格下落に
        # 応じてストップを切り下げる(利益を確保しながら追従)。
        stop = entry_p + atr_barrier_lower * atrs[idx]
        trough = entry_p
        exit_p, h_days = None, holding_period
        for h in range(start_h, holding_period + 1):
            hi, lo = highs[idx + h], lows[idx + h]
            if hi >= stop:
                exit_p, h_days = stop, h
                break
            if lo < trough:
                trough = lo
            if (entry_p - trough) >= atr_trail_activation * atrs[idx]:
                stop = min(stop, trough + atr_trail_distance * atrs[idx])

        if exit_p is None:
            exit_p = closes[idx + holding_period]
            exit_class[idx] = 1  # Hold: タイムアウト、ストップ未到達
        else:
            # 発動後のストップ(stop < entry_p)に触れた決済は活性化条件
            # (entry_p-trough >= activation*ATR、かつdistance < activation)から
            # 数学的にstop < entry_pが保証されるため必ず利益側 -> Win。
            # 初期ストップ(entry_pより上)のままの決済は必ず損失側 -> Stop。
            exit_class[idx] = 2 if stop < entry_p else 0
        targets[idx] = (entry_p - exit_p) / entry_p
        exit_h_days[idx] = h_days
    return targets, exit_h_days, exit_class


def compute_simulated_short_targets(df):
    """stage3_exp.targets.compute_simulated_targetsの空売り版。config.pyの現在値
    (ATR_BARRIER_UPPER/LOWER, ATR_TRAIL_ACTIVATION/DISTANCE, ENTRY_CONVENTION,
    HOLDING_PERIOD)をそのまま使う——ロング版と同じバリア幅・保有期間で比較できるようにする。"""
    from stage3_exp.config import (
        ATR_BARRIER_UPPER, ATR_BARRIER_LOWER, ATR_TRAIL_ACTIVATION, ATR_TRAIL_DISTANCE,
        ENTRY_CONVENTION, HOLDING_PERIOD,
    )
    closes, highs, lows, atrs = (
        df["Close"].values, df["High"].values, df["Low"].values, df["ATR"].values,
    )
    opens = (
        df["Open"].values
        if ENTRY_CONVENTION in ("d1_open", "d1_open_pullback", "d1_open_pullback_high")
        else None
    )
    return simulate_short_ret_pct_d1_close(
        closes, highs, lows, atrs, HOLDING_PERIOD,
        opens=opens, entry_convention=ENTRY_CONVENTION,
        atr_barrier_upper=ATR_BARRIER_UPPER, atr_barrier_lower=ATR_BARRIER_LOWER,
        atr_trail_activation=ATR_TRAIL_ACTIVATION, atr_trail_distance=ATR_TRAIL_DISTANCE,
    )
