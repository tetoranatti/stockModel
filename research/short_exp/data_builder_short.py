"""空売りモデル用のデータセット/バックテストプール構築(2026-09-24追加)。
stage3_exp.data_builder.build_dataset/prepare_backtest_poolのほぼそのままの複製で、
compute_simulated_targets(ロング)をtargets_short.compute_simulated_short_targets(空売り)
に差し替えただけ——targets[idx]の符号が空売り損益(価格下落で正)になる以外は、
特徴量・cross-sectional正規化・LABEL_MODE変換・purge・train/val分割ロジックは
ロング版と完全に同一(比較の公平性のため意図的に揃えている)。

build_per_ticker/prepare_ticker_data/compute_global_split_date自体は特徴量側のロジックで
ラベルに依存しないため、stage3_exp.data_builderから変更無しでそのままimportして使う
(重複させていない)。

USE_REGIME_ATR_BARRIER(regime別バリア)はv1では非対応(常に固定ATR_BARRIER_UPPER/LOWER)。
"""
import numpy as np
import pandas as pd
from training.train_model_v8_exp import MIN_CROSS_SECTION
from modules.cross_sectional_features import (
    apply_log_transform,
    compute_cross_sectional_stats,
    valid_cross_section_dates,
    normalize_cross_sectional,
)

from stage3_exp.config import (
    SEQ_LEN, HOLDING_PERIOD, LABEL_MODE, QUANTILE_LABEL_DOWN, QUANTILE_LABEL_UP,
    RET5_HOLDING_DAYS, RET5_DOWN_THRESH, RET5_UP_THRESH, RISK_BLEND_Z_WEIGHT,
    RISK_BLEND_RANK_WEIGHT, MACRO_CLIP, USE_SECTOR_EMBEDDING, REGIME_ID_MAP,
)
from stage3_exp.feature_catalog import TICKER_TO_SECTOR_ID
from stage3_exp.data_builder import build_per_ticker, prepare_ticker_data, compute_global_split_date
from stage3_exp.targets import compute_ret5_fixed_horizon_targets
from stage3_exp.runtime import log
from .targets_short import compute_simulated_short_targets


def build_dataset_short(
    macro_cols, stock_cols, pool, margin_pool, sector_pool, macro_pool_df,
    regime_filter=None, regime_labels_for_loss=None,
    global_split_date_override=None, val_end_date_override=None,
    use_exit_class_labels=False, train_start_date_override=None,
):
    """stage3_exp.data_builder.build_datasetの空売り版。引数・戻り値形式は完全に同一。

    use_exit_class_labels=True(2026-09-24追加、TP/SL到達順序型ラベルの代替案)にすると、
    config.pyのLABEL_MODE分岐を全て無視し、targets_short.compute_simulated_short_targetsが
    返すexit_class(0=Stop/1=Hold/2=Win)をそのまま教師ラベルにする。交差エントロピー損失
    (research/short_exp/trainer_ce.py)と組み合わせて使う想定——NearPairRankingLossRegimeAware
    (ペアワイズ順位学習)とは異なりサンプル単位の3クラス分類になる。

    train_start_date_override(2026-09-24追加、fixed-window walk-forward用): Noneなら
    従来通りtrain=「global_split_date未満の全履歴」(expanding)。日付を渡すと
    train=「[train_start_date_override, global_split_date)の固定長ウィンドウ」に絞る
    ——walk_forward_validation_short.pyのexpanding windowだとfoldが新しいほど学習データが
    増えるため、「新しいfoldほどPFが良い」結果が(a)学習データ量の増加(b)市場regime
    (モメンタムの強さ)の変化のどちらに起因するか切り分けられない。固定長にすれば(a)を
    排除して(b)だけを見られる。"""
    per_ticker_df, cross_mean, cross_std, valid_dates, global_split_date = (
        prepare_ticker_data(
            macro_cols, stock_cols, pool, margin_pool, sector_pool, macro_pool_df,
            regime_filter, global_split_date_override,
        )
    )
    calendar_tidx_map = {d: i for i, d in enumerate(sorted(macro_pool_df.index))}

    raw_targets = {}
    raw_exit_class = {}
    for t, df in per_ticker_df.items():
        targets, _, exit_class = compute_simulated_short_targets(df)
        raw_targets[t] = pd.Series(targets, index=df.index)
        exit_class_f = exit_class.astype(np.float64)
        exit_class_f[exit_class_f < 0] = np.nan  # loop範囲外(末尾holding_period件): raw_targetsのNaNと揃える
        raw_exit_class[t] = pd.Series(exit_class_f, index=df.index)

    train_targets = raw_targets
    if use_exit_class_labels:
        train_targets = raw_exit_class
    elif LABEL_MODE == "cross_sectional_quantile":
        wide = pd.DataFrame(raw_targets)
        q_low = wide.quantile(QUANTILE_LABEL_DOWN, axis=1)
        q_high = wide.quantile(1.0 - QUANTILE_LABEL_UP, axis=1)
        quantile_targets = {}
        for t, s in raw_targets.items():
            ql, qh = q_low.reindex(s.index), q_high.reindex(s.index)
            disc = pd.Series(1.0, index=s.index)
            disc[s <= ql] = 0.0
            disc[s >= qh] = 2.0
            disc[s.isna()] = np.nan
            quantile_targets[t] = disc
        train_targets = quantile_targets
    elif LABEL_MODE == "ret5_fixed_threshold":
        ret5_targets = {}
        for t, df in per_ticker_df.items():
            closes = df["Close"].values
            r5 = compute_ret5_fixed_horizon_targets(closes, holding_days=RET5_HOLDING_DAYS)
            s = pd.Series(r5, index=df.index)
            disc = pd.Series(1.0, index=s.index)
            disc[s <= RET5_DOWN_THRESH] = 0.0
            disc[s >= RET5_UP_THRESH] = 2.0
            disc[s.isna()] = np.nan
            ret5_targets[t] = disc
        train_targets = ret5_targets
    elif LABEL_MODE == "cross_sectional_rank":
        wide = pd.DataFrame(raw_targets)
        rank_pct = wide.rank(axis=1, pct=True)
        rank_targets = {}
        for t, s in raw_targets.items():
            r = (
                rank_pct[t].reindex(s.index) if t in rank_pct.columns
                else pd.Series(np.nan, index=s.index)
            )
            disc = r * 2.0 - 1.0
            disc[s.isna()] = np.nan
            rank_targets[t] = disc
        train_targets = rank_targets
    elif LABEL_MODE == "risk_adjusted_return":
        risk_adj_targets = {}
        for t, df in per_ticker_df.items():
            atr_pct_t = df["ATR"].values / np.clip(df["Close"].values, 1e-6, None)
            s = raw_targets[t]
            atr_pct_series = (
                pd.Series(atr_pct_t, index=df.index).reindex(s.index).replace(0, np.nan)
            )
            risk_adj_targets[t] = s / atr_pct_series
        train_targets = risk_adj_targets
    elif LABEL_MODE == "risk_adjusted_blend":
        risk_adj_targets = {}
        for t, df in per_ticker_df.items():
            atr_pct_t = df["ATR"].values / np.clip(df["Close"].values, 1e-6, None)
            s = raw_targets[t]
            atr_pct_series = (
                pd.Series(atr_pct_t, index=df.index).reindex(s.index).replace(0, np.nan)
            )
            risk_adj_targets[t] = s / atr_pct_series
        wide_risk_adj = pd.DataFrame(risk_adj_targets)
        daily_mean = wide_risk_adj.mean(axis=1)
        daily_std = wide_risk_adj.std(axis=1)
        rank_pct = wide_risk_adj.rank(axis=1, pct=True)
        blend_targets = {}
        for t, ra in risk_adj_targets.items():
            z = (ra - daily_mean.reindex(ra.index)) / daily_std.reindex(ra.index).replace(0, np.nan)
            r = (
                rank_pct[t].reindex(ra.index) if t in rank_pct.columns
                else pd.Series(np.nan, index=ra.index)
            ) * 2.0 - 1.0
            blended = RISK_BLEND_Z_WEIGHT * z + RISK_BLEND_RANK_WEIGHT * r
            blended[ra.isna()] = np.nan
            blend_targets[t] = blended
        train_targets = blend_targets
    elif LABEL_MODE != "continuous":
        raise ValueError(f"LABEL_MODE='{LABEL_MODE}'は未対応です")

    tr_x_s, tr_x_m, tr_y, tr_tid, tr_tidx, tr_regime = [], [], [], [], [], []
    va_x_s, va_x_m, va_y, va_tid, va_tidx, va_regime = [], [], [], [], [], []
    for i, (t, df) in enumerate(per_ticker_df.items()):
        try:
            if len(df) < SEQ_LEN + HOLDING_PERIOD + 10:
                continue
            norm_s = normalize_cross_sectional(df, cross_mean, cross_std, stock_cols)
            targets = raw_targets[t].values
            train_target_vals = train_targets[t].values
            vals_s_norm = norm_s.values
            vals_m = df[macro_cols].values
            if MACRO_CLIP is not None:
                vals_m = np.clip(vals_m, -MACRO_CLIP, MACRO_CLIP)
            n_samples = len(df)
            dates = df.index
            valid_ok = dates.isin(valid_dates)
            regime_ok = (
                regime_filter.reindex(dates).fillna(False).values
                if regime_filter is not None else None
            )
            if regime_labels_for_loss is not None:
                regime_id_arr = (
                    regime_labels_for_loss.reindex(dates).map(REGIME_ID_MAP)
                    .fillna(-1).astype(int).values
                )
            else:
                regime_id_arr = None
            cal_tidx_arr = np.array(
                [calendar_tidx_map.get(d, -1) for d in dates], dtype=np.int64
            )
            sec_id = TICKER_TO_SECTOR_ID.get(t, 0)

            for idx in range(SEQ_LEN - 1, n_samples):
                if not valid_ok[idx]:
                    continue
                if np.isnan(targets[idx]) or np.isnan(train_target_vals[idx]):
                    continue
                if regime_ok is not None and not regime_ok[idx]:
                    continue
                w_s = vals_s_norm[idx - SEQ_LEN + 1: idx + 1].copy()
                if np.isnan(w_s).any():
                    continue
                w_m = vals_m[idx - SEQ_LEN + 1: idx + 1].copy()
                if np.isnan(w_m).any():
                    continue
                if USE_SECTOR_EMBEDDING:
                    w_s = np.concatenate(
                        [w_s, np.full((SEQ_LEN, 1), sec_id, dtype=w_s.dtype)], axis=1
                    )
                target = train_target_vals[idx]
                r_id = int(regime_id_arr[idx]) if regime_id_arr is not None else -1
                cal_tidx = int(cal_tidx_arr[idx])
                is_train = dates[idx] < global_split_date
                if is_train and dates[idx + HOLDING_PERIOD] >= global_split_date:
                    continue
                if is_train and train_start_date_override is not None and dates[idx] < train_start_date_override:
                    continue
                if is_train:
                    tr_x_s.append(w_s); tr_x_m.append(w_m); tr_y.append(target)
                    tr_tid.append(i); tr_tidx.append(cal_tidx); tr_regime.append(r_id)
                else:
                    if val_end_date_override is not None and (
                        dates[idx] >= val_end_date_override
                        or dates[idx + HOLDING_PERIOD] >= val_end_date_override
                    ):
                        continue
                    va_x_s.append(w_s); va_x_m.append(w_m); va_y.append(target)
                    va_tid.append(i); va_tidx.append(cal_tidx); va_regime.append(r_id)
        except Exception as e:
            log(f"  [!] build_dataset_short: {t}をスキップ({type(e).__name__}: {e})")
            continue

    return (
        np.array(tr_x_s), np.array(tr_x_m), np.array(tr_y),
        np.array(tr_tid), np.array(tr_tidx), np.array(tr_regime),
    ), (
        np.array(va_x_s), np.array(va_x_m), np.array(va_y),
        np.array(va_tid), np.array(va_tidx), np.array(va_regime),
    )


def prepare_backtest_pool_short(
    macro_cols, stock_cols, pool, margin_pool, sector_pool, macro_pool_df,
    regime_filter=None, global_split_date_override=None, close_price_out=None,
):
    """stage3_exp.data_builder.prepare_backtest_poolの空売り版。ret_pctは
    compute_simulated_short_targets由来(正=空売り利益)。"""
    per_ticker_df, cross_mean, cross_std, valid_dates, global_split_date = (
        prepare_ticker_data(
            macro_cols, stock_cols, pool, margin_pool, sector_pool, macro_pool_df,
            regime_filter, global_split_date_override,
        )
    )

    pool_out = []
    for t, df in per_ticker_df.items():
        try:
            if len(df) < SEQ_LEN + HOLDING_PERIOD + 10:
                continue
            norm_s = normalize_cross_sectional(df, cross_mean, cross_std, stock_cols)
            n_samples = len(df)
            vals_s_norm = norm_s.values
            vals_m = df[macro_cols].values
            if MACRO_CLIP is not None:
                vals_m = np.clip(vals_m, -MACRO_CLIP, MACRO_CLIP)
            dates = df.index
            valid_ok = dates.isin(valid_dates)
            regime_ok = (
                regime_filter.reindex(dates).fillna(False).values
                if regime_filter is not None else None
            )
            ret_pcts, exit_h_days, _ = compute_simulated_short_targets(df)
            sec_id = TICKER_TO_SECTOR_ID.get(t, 0)
            if close_price_out is not None:
                close_price_out[t] = df["Close"]

            for idx in range(SEQ_LEN - 1, n_samples - HOLDING_PERIOD):
                if not valid_ok[idx]:
                    continue
                if dates[idx] < global_split_date:
                    continue
                if regime_ok is not None and not regime_ok[idx]:
                    continue
                if np.isnan(ret_pcts[idx]):
                    continue
                w_s = vals_s_norm[idx - SEQ_LEN + 1: idx + 1].copy()
                if np.isnan(w_s).any():
                    continue
                w_m = vals_m[idx - SEQ_LEN + 1: idx + 1].copy()
                if np.isnan(w_m).any():
                    continue
                if USE_SECTOR_EMBEDDING:
                    w_s = np.concatenate(
                        [w_s, np.full((SEQ_LEN, 1), sec_id, dtype=w_s.dtype)], axis=1
                    )
                pool_out.append((
                    t, dates[idx], dates[idx + 1], dates[idx + exit_h_days[idx]],
                    w_s, w_m, ret_pcts[idx],
                ))
        except Exception as e:
            log(f"  [!] prepare_backtest_pool_short: {t}をスキップ({type(e).__name__}: {e})")
            continue
    return pool_out
