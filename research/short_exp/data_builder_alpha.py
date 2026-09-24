"""5日後Alpha回帰モデル用のデータセット構築(2026-09-24追加)。stage3_exp.data_builderの
build_per_ticker/prepare_ticker_data(特徴量側、ラベル非依存)はそのまま再利用し、ラベル
部分だけをtargets_alpha_short.compute_alpha5d_target_and_weightに差し替える。LABEL_MODEの
ブレンド/離散化は一切行わない(target=alpha_5dそのもの)。NearPairRankingLoss用のregime/
tidペアリング構造も回帰には不要なため省略している。"""
import numpy as np
from stage3_exp.config import SEQ_LEN, HOLDING_PERIOD, MACRO_CLIP, USE_SECTOR_EMBEDDING
from stage3_exp.feature_catalog import TICKER_TO_SECTOR_ID
from stage3_exp.data_builder import prepare_ticker_data
from modules.cross_sectional_features import normalize_cross_sectional
from stage3_exp.runtime import log
from .targets_alpha_short import compute_alpha5d_target_and_weight


def build_alpha_dataset(
    macro_cols, stock_cols, pool, margin_pool, sector_pool, macro_pool_df, nk_close_full,
    regime_filter=None, global_split_date_override=None, val_end_date_override=None,
):
    per_ticker_df, cross_mean, cross_std, valid_dates, global_split_date = prepare_ticker_data(
        macro_cols, stock_cols, pool, margin_pool, sector_pool, macro_pool_df,
        regime_filter, global_split_date_override,
    )

    tr_x_s, tr_x_m, tr_y, tr_w = [], [], [], []
    va_x_s, va_x_m, va_y, va_w = [], [], [], []
    for t, df in per_ticker_df.items():
        try:
            if len(df) < SEQ_LEN + HOLDING_PERIOD + 10:
                continue
            nk_aligned = nk_close_full.reindex(df.index).ffill()
            alpha, weight = compute_alpha5d_target_and_weight(df, nk_aligned, HOLDING_PERIOD)
            norm_s = normalize_cross_sectional(df, cross_mean, cross_std, stock_cols)
            vals_s_norm = norm_s.values
            vals_m = df[macro_cols].values
            if MACRO_CLIP is not None:
                vals_m = np.clip(vals_m, -MACRO_CLIP, MACRO_CLIP)
            n_samples = len(df)
            dates = df.index
            valid_ok = dates.isin(valid_dates)
            regime_ok = (
                regime_filter.reindex(dates).fillna(False).values if regime_filter is not None else None
            )
            sec_id = TICKER_TO_SECTOR_ID.get(t, 0)

            for idx in range(SEQ_LEN - 1, n_samples):
                if not valid_ok[idx]:
                    continue
                if np.isnan(alpha[idx]):
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
                    w_s = np.concatenate([w_s, np.full((SEQ_LEN, 1), sec_id, dtype=w_s.dtype)], axis=1)
                is_train = dates[idx] < global_split_date
                if is_train and dates[idx + HOLDING_PERIOD] >= global_split_date:
                    continue
                if is_train:
                    tr_x_s.append(w_s); tr_x_m.append(w_m)
                    tr_y.append(alpha[idx]); tr_w.append(weight[idx])
                else:
                    if val_end_date_override is not None and (
                        dates[idx] >= val_end_date_override
                        or dates[idx + HOLDING_PERIOD] >= val_end_date_override
                    ):
                        continue
                    va_x_s.append(w_s); va_x_m.append(w_m)
                    va_y.append(alpha[idx]); va_w.append(weight[idx])
        except Exception as e:
            log(f"  [!] build_alpha_dataset: {t}をスキップ({type(e).__name__}: {e})")
            continue

    return (
        np.array(tr_x_s), np.array(tr_x_m), np.array(tr_y), np.array(tr_w),
    ), (
        np.array(va_x_s), np.array(va_x_m), np.array(va_y), np.array(va_w),
    )
