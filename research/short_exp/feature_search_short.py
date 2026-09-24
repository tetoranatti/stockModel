"""空売りモデルの新特徴量forward search(2026-09-24追加)。backward_elim_short.pyは
ロング用BASELINE_STOCK_COLSからの引き算(除外)のみで、空売りモデル向けの新規特徴量は
一度も試していなかった。greedy_feature_search.run_greedy_search()にbase_stock_cols/
base_macro_cols(2026-09-24追加)を渡し、SHORT_STOCK_COLS_V2(9 stock)+BASELINE_MACRO_COLS
(5 macro、train_short_model_v2.pyの本採用集合)を開始点にして、未検証の候補群を1個ずつ
追加検証する。

候補プールは4群(ユーザー選定、2026-09-24に2回に分けて検証):
  - CTA/regime交互作用(5個、regime_risk_modelのcta_net_norm_x_low等をこのNNモデルに
    転用できるか検証。[[regime_risk_model_cta_interaction]]はregime_risk_model側の
    確定済み勝ち筋だが、stage3_exp系モデルでは未検証) -> 1回目でALL REJECTED
  - トレンド/勢い・市場ブレッド系(26個、long/PFモデルでは一部却下済み
    [[feature_search_market_breadth_2026-09-21]] [[feature_search_vwap_distance_2026-09-23]]
    [[feature_search_cross_sectional_rank_2026-09-23]]だが、ラベルが空売り用に違うため
    再検証する) -> 1回目でALL REJECTED (詳細: [[short_feature_search_2026-09-24]])
  - 信用取引残高系(margin_pool、4個。空売り残高・信用買残の変化は空売り妙味/踏み上げ
    リスクに直結する最有力候補、空売りモデルでは未検証) -> 2回目で検証
  - セクター相対系(sector_pool、4個。セクター内相対的弱さは空売り選定の定番シグナル、
    空売りモデルでは未検証) -> 2回目で検証

--group で候補群を選択(margin_sector が2026-09-24 2回目の追加分)。省略時はcta_trend
(1回目、既にALL REJECTED済みなので再実行する意味は薄い)。

キャッシュ名前空間はcache_dir_name="short_feature_search_cache"で分離(ロング側の
baseline_model_cacheと衝突しない、baselineモデル自体は1回目と同じfingerprintなのでキャッシュ
再利用される)。ログはresult_log_pathでgroup別に分離(2回目を1回目の結果に上書きしない)。

実行方法:
    python research/short_exp/feature_search_short.py --group margin_sector
    python research/short_exp/feature_search_short.py --group cta_trend --margin 0.01
"""
import sys
import os

sys.path.insert(0, r"F:\stockModel")
sys.path.insert(0, r"F:\stockModel\research")

try:
    sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")
    sys.stderr.reconfigure(line_buffering=True, encoding="utf-8")
except Exception:
    pass

RESULT_LOG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "cache", "short_feature_search_log.parquet"
)

# CTA/regime交互作用(5)
CTA_REGIME_CANDIDATES = [
    "rolling_beta_x_cta_net_norm",
    "rolling_beta_x_cta_net_norm_x_market_vol_regime",
    "cta_net_norm_x_market_vol_regime",
    "pin_dist_ratio_x_cta_net_norm",
    "cta_net_norm_x_low_v2",
]

# トレンド/勢い系(stock、SHORT_STOCK_COLS_V2に既に入っているatr_accel/dist_from_high60/
# gap_strength_5は除く)
TREND_MOMENTUM_CANDIDATES = [
    "atr_term_ratio",
    "gap_avg_5d",
    "volume_zscore",
    "relative_strength_5d",
    "rs20",
    "rs60",
    "momentum_accel",
    "close_location_value",
    "body_ratio",
    "adx14",
    "volume_accel",
    "efficiency_ratio20",
    "days_since_high20",
    "new_high20",
    "positive_gap_ratio_5",
    "gap_follow_through",
    "volume_ma_ratio",
    "vwap_distance",
    "ret_rank_20d",
    "vol_rank_5d",
    "high20_rank",
]

# 市場ブレッド系(macro)
MARKET_BREADTH_CANDIDATES = [
    "market_above_ma25_ratio",
    "market_newhigh20_ratio",
    "market_newlow20_ratio",
    "market_breakout_score",
    "market_turnover_z",
]

# 信用取引残高系(stock、margin_pool)
MARGIN_CANDIDATES = [
    "margin_ratio_level",
    "margin_ratio_zscore_12w",
    "margin_buy_chg_1w",
    "margin_short_chg_1w",
]

# セクター相対系(stock、sector_pool)
SECTOR_RELATIVE_CANDIDATES = [
    "sector_rel_ret_5d",
    "sector_rel_ret_20d",
    "sector_rel_vol_ratio_5d",
    "sector_rank_ret_5d",
]

CANDIDATE_GROUPS = {
    "cta_trend": CTA_REGIME_CANDIDATES + TREND_MOMENTUM_CANDIDATES + MARKET_BREADTH_CANDIDATES,
    "margin_sector": MARGIN_CANDIDATES + SECTOR_RELATIVE_CANDIDATES,
}


def main():
    import greedy_feature_search as gfs
    from short_exp.train_short_model_v2 import SHORT_STOCK_COLS_V2
    from short_exp.data_builder_short import build_dataset_short, prepare_backtest_pool_short

    margin_arg, max_concurrent_arg, candidates_arg, group_arg = 0.0, 10, None, "cta_trend"
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--margin":
            margin_arg = float(args[i + 1]); i += 2
        elif args[i] == "--max-concurrent":
            max_concurrent_arg = int(args[i + 1]); i += 2
        elif args[i] == "--candidates":
            candidates_arg = args[i + 1].split(","); i += 2
        elif args[i] == "--group":
            group_arg = args[i + 1]; i += 2
        else:
            i += 1

    import stage3_exp.experiment_context as m
    candidates = candidates_arg if candidates_arg is not None else CANDIDATE_GROUPS[group_arg]
    result_log_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "cache",
        f"short_feature_search_log_{group_arg}.parquet",
    ) if candidates_arg is None else RESULT_LOG_PATH
    print(f"[*] group={group_arg}")
    print(f"[*] 開始集合: stock={SHORT_STOCK_COLS_V2}")
    print(f"[*] 開始集合: macro={m.BASELINE_MACRO_COLS}")
    print(f"[*] 候補プール({len(candidates)}個): {candidates}")

    gfs.run_greedy_search(
        margin=margin_arg, max_concurrent=max_concurrent_arg, candidates=candidates,
        build_dataset_fn=build_dataset_short, prepare_backtest_pool_fn=prepare_backtest_pool_short,
        cache_dir_name="short_feature_search_cache", result_log_path=result_log_path,
        base_stock_cols=SHORT_STOCK_COLS_V2, base_macro_cols=m.BASELINE_MACRO_COLS,
    )


if __name__ == "__main__":
    main()
