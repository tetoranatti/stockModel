"""Target and exit-price simulation."""

import numpy as np
import pandas as pd
from .config import (ATR_BARRIER_UPPER, ATR_BARRIER_LOWER, USE_REGIME_ATR_BARRIER,
    ATR_BARRIER_BY_REGIME, ATR_TRAIL_ACTIVATION, ATR_TRAIL_DISTANCE, ENTRY_CONVENTION,
    HOLDING_PERIOD, BARRIER_MODE)

def simulate_ret_pct_d1_close(
    closes,
    highs,
    lows,
    atrs,
    holding_period,
    opens=None,
    entry_convention="d1_close",
    regime_labels=None,
    barrier_mode="fixed",
):
    """学習ターゲットとD+1引け(MOC)バックテストを同一関数に統合する。
    entry_convention: "d1_close"(既定、D+1引け/MOC、本番の確立済み執行方式) または
    "d1_open"(D+1始値/MOO)。日足OHLCしか使わず日中パスを再現できないため、"d1_open_pullback"/
    "d1_open_pullback_high"は始値と安値/高値の中間で約定したと仮定する悲観シナリオの近似。
    "d1_open"系はエントリー当日(idx+1)がまだ引けていないためバリア判定をh=1から開始し、
    "d1_close"はエントリー当日の引けで既に約定済みのためh=2から(オーバーナイトギャップの
    ズレは無い)。本番training/train_model_v8_exp.py::_simulate_ret_pctにはD+1引けの
    選択肢が無いため、学習ターゲット・バックテスト両方をこの1関数から作ることで統一する
    (本番側は意図的に未変更)。

    regime_labels: 'LOW'/'MID'/'HIGH'/'UNKNOWN'のndarray(closesと同じ長さ、idx=シグナル日基準)
    を渡すと、USE_REGIME_ATR_BARRIER=True時にATR_BARRIER_UPPER/LOWERの代わりに
    ATR_BARRIER_BY_REGIME[regime_labels[idx]]を使う。Noneまたは未知キーはATR_BARRIER_UPPER/
    LOWERにフォールバックする。
    barrier_mode: "fixed"(既定)または"trailing"。trailing時は固定upper_pを使わず、
    エントリー来の高値がATR_TRAIL_ACTIVATION*ATR伸びたらトレーリング発動、以降
    ストップ=高値-ATR_TRAIL_DISTANCE*ATR(切り上げのみ)を毎日追従させる。ある日の安値が
    「その日の高値でストップを切り上げる前」のストップを下回ったら即終了(楽観的に高値が
    先に付いたと仮定しない保守的な順序)。発動前は初期ストップ(entry_p-ATR_BARRIER_LOWER*ATR)
    のみが有効で、upper方向の終了条件は無い(タイムアウトのみ)。regime_labels/
    USE_REGIME_ATR_BARRIERはtrailing modeでは未対応(fixedのみ)。

    戻り値: (targets, exit_h_days)。targetsはret_pct(計算不能な末尾holding_period件はnan)。
    exit_h_daysは何営業日後に手仕舞いしたか(MaxDD計算のexit_date算出に必要、nanの日は-1)。"""
    n_bars = len(closes)
    targets = np.full(n_bars, np.nan)
    exit_h_days = np.full(n_bars, -1, dtype=int)
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
            # 始値〜高値の中間を使い、不利な価格で約定したと仮定する。lows/highsとも
            # 当日の実現値を使うためルックアヘッドは残るが、少なくとも悲観方向にはなる。
            entry_p = (opens[idx + 1] + highs[idx + 1]) / 2.0
        else:
            entry_p = closes[idx + 1]
        # 利確/損切りバリアはentry_p基準(closes[idx]基準だとオーバーナイトギャップ分
        # バリア位置がズレる。本番training/train_model_v8_exp.py::_simulate_ret_pctと同じ基準)。
        if USE_REGIME_ATR_BARRIER and regime_labels is not None:
            up_mult, lo_mult = ATR_BARRIER_BY_REGIME.get(
                regime_labels[idx], (ATR_BARRIER_UPPER, ATR_BARRIER_LOWER)
            )
        else:
            up_mult, lo_mult = ATR_BARRIER_UPPER, ATR_BARRIER_LOWER
        exit_p, h_days = None, holding_period
        if barrier_mode == "trailing":
            # トレーリングストップ。固定upper_pは使わない。各日、「その日の高値で
            # ストップを切り上げる前」の状態でまず安値がストップを下回っていないか確認する
            # (楽観的に高値が先に付いたと仮定しない保守的な順序)。
            stop = entry_p - lo_mult * atrs[idx]
            peak = entry_p
            for h in range(start_h, holding_period + 1):
                hi, lo = highs[idx + h], lows[idx + h]
                if lo <= stop:
                    exit_p, h_days = stop, h
                    break
                if hi > peak:
                    peak = hi
                if (peak - entry_p) >= ATR_TRAIL_ACTIVATION * atrs[idx]:
                    stop = max(stop, peak - ATR_TRAIL_DISTANCE * atrs[idx])
        else:
            upper_p, lower_p = (
                entry_p + up_mult * atrs[idx],
                entry_p - lo_mult * atrs[idx],
            )
            for h in range(start_h, holding_period + 1):
                hi, lo = highs[idx + h], lows[idx + h]
                if lo <= lower_p and hi >= upper_p:
                    exit_p, h_days = lower_p, h
                    break
                elif hi >= upper_p:
                    exit_p, h_days = upper_p, h
                    break
                elif lo <= lower_p:
                    exit_p, h_days = lower_p, h
                    break
        if exit_p is None:
            exit_p = closes[idx + holding_period]
        targets[idx] = (exit_p - entry_p) / entry_p
        exit_h_days[idx] = h_days
    return targets, exit_h_days

def compute_ret5_fixed_horizon_targets(closes, holding_days=5):
    """ATRバリア無しの単純な固定期間リターン。simulate_ret_pct_d1_closeと違い、期間中の
    利確/損切りラインへの早期到達は一切見ず、entry_p(D+1引け、既存の執行規約と統一)から
    holding_days営業日後の引けまでの単純な変化率のみを返す(タイムアウトのみ)。"""
    n_bars = len(closes)
    targets = np.full(n_bars, np.nan)
    for idx in range(n_bars - 1 - holding_days):
        entry_p = closes[idx + 1]
        exit_p = closes[idx + 1 + holding_days]
        targets[idx] = (exit_p - entry_p) / entry_p
    return targets

def compute_simulated_targets(df, barrier_regime_labels=None):
    """simulate_ret_pct_d1_closeの呼び出しをbuild_dataset/prepare_backtest_poolで共有する。
    barrier_regime_labels: date->'LOW'/'MID'/'HIGH'/'UNKNOWN'のpd.Series。USE_REGIME_ATR_BARRIER=
    True時、regime別のATRバリア倍率をsimulate_ret_pct_d1_closeに渡すために使う。"""
    closes, highs, lows, atrs = (
        df["Close"].values,
        df["High"].values,
        df["Low"].values,
        df["ATR"].values,
    )
    opens = (
        df["Open"].values
        if ENTRY_CONVENTION in ("d1_open", "d1_open_pullback", "d1_open_pullback_high")
        else None
    )
    regime_arr = (
        barrier_regime_labels.reindex(df.index).fillna("UNKNOWN").values
        if barrier_regime_labels is not None
        else None
    )
    return simulate_ret_pct_d1_close(
        closes,
        highs,
        lows,
        atrs,
        HOLDING_PERIOD,
        opens=opens,
        entry_convention=ENTRY_CONVENTION,
        regime_labels=regime_arr,
        barrier_mode=BARRIER_MODE,
    )
