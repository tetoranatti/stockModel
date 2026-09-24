"""Dataset and backtest-pool construction."""

import numpy as np
import pandas as pd
from training.train_model_v8_exp import MIN_CROSS_SECTION
from modules.cross_sectional_features import (
    apply_log_transform,
    compute_cross_sectional_stats,
    valid_cross_section_dates,
    normalize_cross_sectional,
)

from .config import (
    SEQ_LEN,
    HOLDING_PERIOD,
    LABEL_MODE,
    QUANTILE_LABEL_DOWN,
    QUANTILE_LABEL_UP,
    RET5_HOLDING_DAYS,
    RET5_DOWN_THRESH,
    RET5_UP_THRESH,
    RISK_BLEND_Z_WEIGHT,
    RISK_BLEND_RANK_WEIGHT,
    MACRO_CLIP,
    USE_SECTOR_EMBEDDING,
    REGIME_ID_MAP,
)
from .feature_catalog import (
    FEATURE_CATALOG,
    MACRO_COMPUTE_FNS,
    STOCK_COMPUTE_FNS,
    TEST_MACRO_COLS,
    BASELINE_MACRO_COLS,
    TICKER_TO_SECTOR_ID,
)
from .targets import compute_ret5_fixed_horizon_targets, compute_simulated_targets
from .runtime import log


def augment_macro_pool_df(macro_pool_df):
    """テスト側で使うmacro側computed特徴量をmacro_pool_dfに1回だけ追加する。"""
    macro_pool_df = macro_pool_df.copy()
    for f in set(TEST_MACRO_COLS) | set(BASELINE_MACRO_COLS):
        if (
            FEATURE_CATALOG[f]["source"] == "computed"
            and f not in macro_pool_df.columns
        ):
            macro_pool_df[f] = MACRO_COMPUTE_FNS[f](macro_pool_df)
    return macro_pool_df


def build_per_ticker(pool, margin_pool, sector_pool, macro_pool_df, stock_cols):
    """pool_df(既存9+株側候補20が常に列として存在)をベースに、stock_colsに含まれる
    margin_pool/sector_pool/computed由来の列だけを追加したdfを返す。"""
    needed_extra = [
        f for f in stock_cols if FEATURE_CATALOG[f]["source"] != "stock_pool"
    ]
    per_ticker_df = {}
    for t, pool_df in pool.items():
        try:
            df = pool_df.copy()
            for f in needed_extra:
                src = FEATURE_CATALOG[f]["source"]
                if src == "margin_pool":
                    val = (
                        margin_pool[t][f].reindex(df.index)
                        if t in margin_pool
                        else pd.Series(0.0, index=df.index)
                    )
                    df[f] = val.fillna(0.0)
                elif src == "sector_pool":
                    val = (
                        sector_pool[t][f].reindex(df.index)
                        if t in sector_pool
                        else pd.Series(0.0, index=df.index)
                    )
                    df[f] = val.fillna(0.0)
                elif src == "computed":
                    df[f] = STOCK_COMPUTE_FNS[f](df, macro_pool_df)
            per_ticker_df[t] = df
        except Exception as e:
            log(f"  [!] build_per_ticker: {t}をスキップ({type(e).__name__}: {e})")
            continue
    return per_ticker_df


def compute_global_split_date(valid_dates, regime_filter=None, percentile=0.75):
    """train/valの分割を全銘柄共通のカレンダー日で行う(銘柄ごとの75%点だと分割日が
    最大約5.5ヶ月ずれ、out-of-sample性が銘柄間で一貫しなくなるため)。
    regime_filterを渡すと、regime該当日だけに絞ったpercentile点を返す。
    percentile(2026-09-23追加、ユーザー指摘「valとバックテスト期間の重複がどの程度
    成績に影響するか」を測る実験用): 既定0.75。0.875等を渡すと、val/backtest重複問題を
    切り分けるための第2の分割点(例: 75%点で真のブラインド期間の開始をさらに区切る)を
    同じロジックで計算できる。"""
    sorted_dates = pd.DatetimeIndex(sorted(valid_dates))
    if regime_filter is not None:
        regime_ok_global = regime_filter.reindex(sorted_dates).fillna(False).values
        eligible_dates = sorted_dates[regime_ok_global]
        if len(eligible_dates) == 0:
            return sorted_dates[-1] if len(sorted_dates) else pd.Timestamp.max
        return eligible_dates[int(len(eligible_dates) * percentile)]
    return sorted_dates[int(len(sorted_dates) * percentile)]


def prepare_ticker_data(
    macro_cols,
    stock_cols,
    pool,
    margin_pool,
    sector_pool,
    macro_pool_df,
    regime_filter=None,
    global_split_date_override=None,
):
    """build_dataset/prepare_backtest_poolに共通する前処理(per_ticker_df構築・macro結合・
    cross-sectional統計・split date計算)を1箇所にまとめる。
    macro結合はhow='left'(macro側の欠損日でOHLC/ATRの連続系列を削らない)。個々の特徴量の
    NaNは後段のサンプル単位window NaNチェックに任せ、ここではOHLC/ATRの欠損だけで行を落とす。
    global_split_date_override: 指定すると、このconfig自身のvalid_datesから計算せず
    渡された値をそのまま使う(base/testでsplit dateを完全共通化するため)。"""
    per_ticker_df = build_per_ticker(
        pool, margin_pool, sector_pool, macro_pool_df, stock_cols
    )
    macro_feed = macro_pool_df[macro_cols]
    for t in list(per_ticker_df.keys()):
        df = per_ticker_df[t].join(macro_feed, how="left")
        needed = [c for c in ["Close", "High", "Low", "ATR"] if c in df.columns]
        df = df.dropna(subset=needed)
        if len(df) < SEQ_LEN + HOLDING_PERIOD + 10:
            del per_ticker_df[t]
            continue
        per_ticker_df[t] = apply_log_transform(df)

    cross_mean, cross_std = compute_cross_sectional_stats(per_ticker_df, stock_cols)
    valid_dates = valid_cross_section_dates(
        per_ticker_df, stock_cols, MIN_CROSS_SECTION
    )
    global_split_date = (
        global_split_date_override
        if global_split_date_override is not None
        else compute_global_split_date(valid_dates, regime_filter)
    )
    return per_ticker_df, cross_mean, cross_std, valid_dates, global_split_date


def build_dataset(
    macro_cols,
    stock_cols,
    pool,
    margin_pool,
    sector_pool,
    macro_pool_df,
    regime_filter=None,
    regime_labels_for_loss=None,
    global_split_date_override=None,
    barrier_regime_labels=None,
    val_end_date_override=None,
):
    """regime_filter: date->bool(True=採用)のpd.Seriesを渡すと、その日付のサンプルだけ
    train/valに採用する(regime別モデル用)。SEQ_LEN分の価格系列window自体はregimeで
    フィルタせず連続したまま使う(不連続だと時系列モデルの入力が歪むため)。
    regime_labels_for_loss: date->'LOW'/'MID'/'HIGH'のpd.Seriesを渡すと、各サンプルの
    regime id(0/1/2、不明なら-1)を6番目の要素として返す(NearPairRankingLossRegimeAware用、
    regime_filterとは独立)。
    barrier_regime_labels: compute_simulated_targets参照。
    val_end_date_override(2026-09-23追加、ユーザー指摘「valとバックテストが同じ期間で
    early stopping選択にバックテスト期間の情報が漏れている」の影響度を測る実験用):
    指定すると、val採用条件に「dates[idx] < val_end_date_override」も追加し、
    それ以降の日付はtrain/valどちらにも入れず完全に除外する(=checkpoint選択が
    一切見ない、真にブラインドな期間を作れる)。Noneなら従来通りglobal_split_date以降
    全てがval。"""
    per_ticker_df, cross_mean, cross_std, valid_dates, global_split_date = (
        prepare_ticker_data(
            macro_cols,
            stock_cols,
            pool,
            margin_pool,
            sector_pool,
            macro_pool_df,
            regime_filter,
            global_split_date_override,
        )
    )
    # tidxは銘柄ごとのdf内の配列位置ではなく、macro_pool_df.index(全銘柄共通の取引日
    # カレンダー)上の位置を使う。dropnaで行が削れる銘柄があっても、tidx差が全銘柄で
    # 同じ意味の営業日数差になるようにするため(NearPairRankingLossのgap判定に必要)。
    calendar_tidx_map = {d: i for i, d in enumerate(sorted(macro_pool_df.index))}

    # raw ret_pctを銘柄ごとに1回だけ計算して使い回す(cross-sectional系LABEL_MODEでは
    # 全銘柄分のraw ret_pctが同時に必要なため)。
    raw_targets = {}
    for t, df in per_ticker_df.items():
        targets, _ = compute_simulated_targets(df, barrier_regime_labels)
        raw_targets[t] = pd.Series(targets, index=df.index)

    train_targets = raw_targets
    if LABEL_MODE == "cross_sectional_quantile":
        # 同日全銘柄のraw ret_pctを横断してcross-sectional分位点化する。学習ターゲットのみ
        # 置き換え、バックテスト側(prepare_backtest_pool)は連続値ret_pctのまま。
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
        # ATRバリア無しの固定期間(RET5_HOLDING_DAYS日)リターンを、固定の絶対閾値で3値化する。
        ret5_targets = {}
        for t, df in per_ticker_df.items():
            closes = df["Close"].values
            r5 = compute_ret5_fixed_horizon_targets(
                closes, holding_days=RET5_HOLDING_DAYS
            )
            s = pd.Series(r5, index=df.index)
            disc = pd.Series(1.0, index=s.index)
            disc[s <= RET5_DOWN_THRESH] = 0.0
            disc[s >= RET5_UP_THRESH] = 2.0
            disc[s.isna()] = np.nan
            ret5_targets[t] = disc
        train_targets = ret5_targets
    elif LABEL_MODE == "cross_sectional_rank":
        # 同日全銘柄のraw ret_pctを横断してcross-sectional百分位順位(連続値、離散化しない)に
        # 変換する。cross_sectional_quantile(離散3値)は同じbin同士のペアがdiff=0で除外され
        # 有効ペア数が減るため、順位を連続値のまま保つことでその副作用を避ける。
        wide = pd.DataFrame(raw_targets)
        rank_pct = wide.rank(axis=1, pct=True)  # (0,1]、同値は平均順位(pandas既定)
        rank_targets = {}
        for t, s in raw_targets.items():
            r = (
                rank_pct[t].reindex(s.index)
                if t in rank_pct.columns
                else pd.Series(np.nan, index=s.index)
            )
            disc = r * 2.0 - 1.0  # (-1,1]にスケール、ret_pctと同程度のオーダーに揃える
            disc[s.isna()] = np.nan
            rank_targets[t] = disc
        train_targets = rank_targets
    elif LABEL_MODE == "risk_adjusted_return":
        # ret_pctをそのシグナル時点のATR(価格比)で正規化した連続値。同じ%リターンでも
        # 高ボラ銘柄の方が「楽に」出せる値幅なので、額面%のままだと過大評価されうる
        # ——ATR/価格で割ることで補正する。
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
        # risk_adjusted_return(ATR正規化)のcross-sectional z-score(大きさの情報を保持)と
        # cross-sectional daily rank(順序情報のみ、外れ値に頑健)をRISK_BLEND_Z_WEIGHT:
        # RISK_BLEND_RANK_WEIGHTでブレンドする(rankはATR正規化後の値を順位化する点が
        # cross_sectional_rankモードと異なる)。
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
            z = (ra - daily_mean.reindex(ra.index)) / daily_std.reindex(
                ra.index
            ).replace(0, np.nan)
            r = (
                rank_pct[t].reindex(ra.index)
                if t in rank_pct.columns
                else pd.Series(np.nan, index=ra.index)
            ) * 2.0 - 1.0
            blended = RISK_BLEND_Z_WEIGHT * z + RISK_BLEND_RANK_WEIGHT * r
            blended[ra.isna()] = np.nan
            blend_targets[t] = blended
        train_targets = blend_targets
    elif LABEL_MODE != "continuous":
        raise ValueError(
            f"LABEL_MODE='{LABEL_MODE}'は未対応です(continuous/cross_sectional_quantile/"
            f"ret5_fixed_threshold/cross_sectional_rank/risk_adjusted_return/"
            f"risk_adjusted_blendのみ)"
        )

    tr_x_s, tr_x_m, tr_y, tr_tid, tr_tidx, tr_regime = [], [], [], [], [], []
    va_x_s, va_x_m, va_y, va_tid, va_tidx, va_regime = [], [], [], [], [], []
    for i, (t, df) in enumerate(per_ticker_df.items()):
        try:
            # valid_datesで行ごと削ると、SEQ_LEN分のlookback windowが不連続な日をまたいで
            # しまうため、判定日として採用するかどうかだけをvalid_okで絞り、window自体は
            # 削らず連続のまま使う。
            if len(df) < SEQ_LEN + HOLDING_PERIOD + 10:
                continue
            norm_s = normalize_cross_sectional(df, cross_mean, cross_std, stock_cols)
            targets = raw_targets[t].values  # NaN判定・purge判定は常に連続値基準
            train_target_vals = train_targets[
                t
            ].values  # 実際に学習へ渡す値(quantile化 or 連続値)
            vals_s_norm = norm_s.values
            vals_m = df[macro_cols].values
            if MACRO_CLIP is not None:
                vals_m = np.clip(vals_m, -MACRO_CLIP, MACRO_CLIP)
            n_samples = len(df)
            dates = df.index
            valid_ok = dates.isin(valid_dates)
            regime_ok = (
                regime_filter.reindex(dates).fillna(False).values
                if regime_filter is not None
                else None
            )
            if regime_labels_for_loss is not None:
                regime_id_arr = (
                    regime_labels_for_loss.reindex(dates)
                    .map(REGIME_ID_MAP)
                    .fillna(-1)
                    .astype(int)
                    .values
                )
            else:
                regime_id_arr = None
            # NearPairRankingLossのgap計算に使うtidxは配列位置ではなく全銘柄共通の
            # カレンダー日位置(calendar_tidx_map、上で定義)。
            cal_tidx_arr = np.array(
                [calendar_tidx_map.get(d, -1) for d in dates], dtype=np.int64
            )
            sec_id = TICKER_TO_SECTOR_ID.get(t, 0)  # USE_SECTOR_EMBEDDING用(銘柄固定値)

            for idx in range(SEQ_LEN - 1, n_samples):
                if not valid_ok[idx]:
                    continue
                # NaN判定はtargets(連続値、常にHOLDING_PERIOD基準)とtrain_target_vals(実際に
                # 学習へ渡す値、LABEL_MODEによって有効範囲が異なりうる)の両方をチェックする。
                if np.isnan(targets[idx]) or np.isnan(train_target_vals[idx]):
                    continue
                if regime_ok is not None and not regime_ok[idx]:
                    continue
                w_s = vals_s_norm[idx - SEQ_LEN + 1 : idx + 1].copy()
                if np.isnan(w_s).any():
                    continue
                w_m = vals_m[idx - SEQ_LEN + 1 : idx + 1].copy()
                if np.isnan(w_m).any():
                    continue
                if USE_SECTOR_EMBEDDING:
                    # sector_idをw_sの最終列に定数値として埋め込む(既存のtr_data/va_data
                    # タプル構造を変えずに済ませるため)。モデル側forward()でGRUに渡す前に切り離す。
                    w_s = np.concatenate(
                        [w_s, np.full((SEQ_LEN, 1), sec_id, dtype=w_s.dtype)], axis=1
                    )
                target = train_target_vals[idx]
                r_id = int(regime_id_arr[idx]) if regime_id_arr is not None else -1
                # 全銘柄共通のカレンダー日(global_split_date)で分割する(銘柄ごとの75%点だと
                # 分割日が最大約5.5ヶ月ずれ、out-of-sample性が銘柄間で一貫しなくなるため)。
                cal_tidx = int(cal_tidx_arr[idx])
                is_train = dates[idx] < global_split_date
                if is_train and dates[idx + HOLDING_PERIOD] >= global_split_date:
                    # purge: このサンプルのラベルはidx+HOLDING_PERIOD日後までの値動きで決まる
                    # (simulate_ret_pct_d1_close)。idxがtrain期間でもラベルの参照範囲がval期間に
                    # 食い込んでいれば、val期間の情報がtrainに漏れるため除外する(purged CV)。
                    continue
                if is_train:
                    tr_x_s.append(w_s)
                    tr_x_m.append(w_m)
                    tr_y.append(target)
                    tr_tid.append(i)
                    tr_tidx.append(cal_tidx)
                    tr_regime.append(r_id)
                else:
                    if val_end_date_override is not None and (
                        dates[idx] >= val_end_date_override
                        or dates[idx + HOLDING_PERIOD] >= val_end_date_override
                    ):
                        # val_end_date_override以降は完全ブラインド期間として除外。
                        # ラベル参照範囲(idx+HOLDING_PERIOD)がブラインド期間に食い込む
                        # サンプルもpurge(train側のpurgeと同じ理由——ブラインド期間の
                        # 情報がval経由でcheckpoint選択に漏れるのを防ぐ)。
                        continue
                    va_x_s.append(w_s)
                    va_x_m.append(w_m)
                    va_y.append(target)
                    va_tid.append(i)
                    va_tidx.append(cal_tidx)
                    va_regime.append(r_id)
        except Exception as e:
            log(f"  [!] build_dataset: {t}をスキップ({type(e).__name__}: {e})")
            continue

    return (
        np.array(tr_x_s),
        np.array(tr_x_m),
        np.array(tr_y),
        np.array(tr_tid),
        np.array(tr_tidx),
        np.array(tr_regime),
    ), (
        np.array(va_x_s),
        np.array(va_x_m),
        np.array(va_y),
        np.array(va_tid),
        np.array(va_tidx),
        np.array(va_regime),
    )


def prepare_backtest_pool(
    macro_cols,
    stock_cols,
    pool,
    margin_pool,
    sector_pool,
    macro_pool_df,
    regime_filter=None,
    global_split_date_override=None,
    barrier_regime_labels=None,
    close_price_out=None,
):
    per_ticker_df, cross_mean, cross_std, valid_dates, global_split_date = (
        prepare_ticker_data(
            macro_cols,
            stock_cols,
            pool,
            margin_pool,
            sector_pool,
            macro_pool_df,
            regime_filter,
            global_split_date_override,
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
                if regime_filter is not None
                else None
            )
            # build_datasetの学習ターゲットと同じ関数(compute_simulated_targets)を使い、
            # エントリー価格前提を統一する。exit_h_daysはMaxDD計算のexit_date算出に必要。
            ret_pcts, exit_h_days = compute_simulated_targets(df, barrier_regime_labels)
            sec_id = TICKER_TO_SECTOR_ID.get(t, 0)  # USE_SECTOR_EMBEDDING用(銘柄固定値)
            if close_price_out is not None:
                # diagnostics.compute_max_drawdownの日次時価評価(2026-09-23追加、
                # ユーザー指摘「未決済ポジションの含み損益」)用にClose価格を渡す。
                # 既存呼び出し元(close_price_outを渡さない)には一切影響しない。
                close_price_out[t] = df["Close"]

            for idx in range(SEQ_LEN - 1, n_samples - HOLDING_PERIOD):
                if not valid_ok[idx]:
                    continue
                if (
                    dates[idx] < global_split_date
                ):  # 全銘柄共通のカレンダー日で分割(build_datasetと同じ理由)
                    continue
                if regime_ok is not None and not regime_ok[idx]:
                    continue
                if np.isnan(ret_pcts[idx]):
                    continue
                w_s = vals_s_norm[idx - SEQ_LEN + 1 : idx + 1].copy()
                if np.isnan(w_s).any():
                    continue
                w_m = vals_m[idx - SEQ_LEN + 1 : idx + 1].copy()
                if np.isnan(w_m).any():
                    continue
                if USE_SECTOR_EMBEDDING:  # build_datasetと同じ埋め込み方
                    w_s = np.concatenate(
                        [w_s, np.full((SEQ_LEN, 1), sec_id, dtype=w_s.dtype)], axis=1
                    )
                # entry_date=dates[idx+1](D+1引け)・exit_dateはMaxDD計算(資金曲線シミュレーション)に必要。
                pool_out.append(
                    (
                        t,
                        dates[idx],
                        dates[idx + 1],
                        dates[idx + exit_h_days[idx]],
                        w_s,
                        w_m,
                        ret_pcts[idx],
                    )
                )
        except Exception as e:
            log(f"  [!] prepare_backtest_pool: {t}をスキップ({type(e).__name__}: {e})")
            continue
    return pool_out
