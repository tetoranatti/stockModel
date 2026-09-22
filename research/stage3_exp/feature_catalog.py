"""Feature catalog, computed features, and selected feature sets."""

from modules.stock_features import STOCK_FEATURE_COLS
from modules.regime_risk_model import _compute_vol_regime_is_low
from _feature_cache_utils import _load_sector_map
from .config import BASE_MACRO_COLS, FEATURE_TOGGLES

_SECTOR_MAP_RAW = _load_sector_map()
_SECTOR_NAMES_SORTED = sorted({v for v in _SECTOR_MAP_RAW.values() if v is not None})
SECTOR_NAME_TO_ID = {
    name: i + 1 for i, name in enumerate(_SECTOR_NAMES_SORTED)
}
TICKER_TO_SECTOR_ID = {
    t: SECTOR_NAME_TO_ID.get(v, 0) for t, v in _SECTOR_MAP_RAW.items()
}
NUM_SECTORS = len(_SECTOR_NAMES_SORTED) + 1
FEATURE_GROUPS = [
    (STOCK_FEATURE_COLS, "stock", "stock_pool"),
    (BASE_MACRO_COLS, "macro", "macro_pool"),
    (
        [
            "atr_term_ratio",
            "atr_accel",
            "gap_avg_5d",
            "volume_zscore",
            "relative_strength_5d",
            "rs20",
            "rs60",
            "dist_from_high60",
            "momentum_accel",
            "close_location_value",
            "body_ratio",
            "adx14",
            "volume_accel",
            "efficiency_ratio20",
            "days_since_high20",
            "new_high20",
            "positive_gap_ratio_5",
            "gap_strength_5",
            "gap_follow_through",
            "volume_ma_ratio",
        ],
        "stock",
        "stock_pool",
    ),
    (
        [
            "margin_ratio_level",
            "margin_ratio_zscore_12w",
            "margin_buy_chg_1w",
            "margin_short_chg_1w",
        ],
        "stock",
        "margin_pool",
    ),
    (
        [
            "sector_rel_ret_5d",
            "sector_rel_ret_20d",
            "sector_rel_vol_ratio_5d",
            "sector_rank_ret_5d",
        ],
        "stock",
        "sector_pool",
    ),
    (
        [
            "usdjpy_chg",
            "market_vol_regime",
            "vix_overnight_chg",
            "nk225_ret_5d",
            "topix_ret_5d",
        ],
        "macro",
        "macro_pool",
    ),
    (
        [
            "market_above_ma25_ratio",
            "market_newhigh20_ratio",
            "market_newlow20_ratio",
            "market_breakout_score",
            "market_turnover_z",
        ],
        "macro",
        "macro_pool",
    ),
]
FEATURE_CATALOG = {}
for _cols, _side, _source in FEATURE_GROUPS:
    for _f in _cols:
        FEATURE_CATALOG[_f] = {"side": _side, "source": _source}
FEATURE_CATALOG.update(
    {
        "rolling_beta_x_cta_net_norm": {"side": "stock", "source": "computed"},
        "rolling_beta_x_cta_net_norm_x_market_vol_regime": {
            "side": "stock",
            "source": "computed",
        },
        "cta_net_norm_x_market_vol_regime": {"side": "macro", "source": "computed"},
        "pin_dist_ratio_x_cta_net_norm": {"side": "macro", "source": "computed"},
        "cta_net_norm_x_low_v2": {"side": "macro", "source": "computed"},
    }
)
assert set(FEATURE_TOGGLES) == set(FEATURE_CATALOG), (
    f"FEATURE_TOGGLESとFEATURE_CATALOGのキーが一致しません: "
    f"toggles-catalog={set(FEATURE_TOGGLES)-set(FEATURE_CATALOG)}, catalog-toggles={set(FEATURE_CATALOG)-set(FEATURE_TOGGLES)}"
)
STOCK_COMPUTE_FNS = {
    "rolling_beta_x_cta_net_norm": lambda df, mpdf: df["rolling_beta"]
    * mpdf["cta_net_norm"].reindex(df.index).fillna(0.0),
    "rolling_beta_x_cta_net_norm_x_market_vol_regime": lambda df, mpdf: (
        df["rolling_beta"]
        * mpdf["cta_net_norm"].reindex(df.index).fillna(0.0)
        * mpdf["market_vol_regime"].reindex(df.index).fillna(0.0)
    ),
}
MACRO_COMPUTE_FNS = {
    "cta_net_norm_x_market_vol_regime": lambda mpdf: mpdf["cta_net_norm"]
    * mpdf["market_vol_regime"],
    "pin_dist_ratio_x_cta_net_norm": lambda mpdf: mpdf["pin_dist_ratio"]
    * mpdf["cta_net_norm"],
    # regime_risk_model.py::_compute_vol_regime_is_lowと同じ拡大窓(look-ahead無し)版のLOW判定を
    # 使う、cta_net_norm_x_lowの修正版。旧版(全期間固定分位点、look-ahead汚染)はmacro_feature_
    # scorecard.csvで全候補中最強だったが信頼できず却下されていた。修正版はPFモデルでは未テスト。
    "cta_net_norm_x_low_v2": lambda mpdf: mpdf["cta_net_norm"]
    * _compute_vol_regime_is_low(mpdf),
}
BASELINE_STOCK_COLS = [
    c for c in STOCK_FEATURE_COLS if c not in ("atr_ratio", "stock_ret_5d")
] + ["gap_strength_5", "atr_accel", "dist_from_high60"]
BASELINE_MACRO_COLS = list(BASE_MACRO_COLS)
TEST_FEATS = [f for f, on in FEATURE_TOGGLES.items() if on]
TEST_STOCK_COLS = [f for f in TEST_FEATS if FEATURE_CATALOG[f]["side"] == "stock"]
TEST_MACRO_COLS = [f for f in TEST_FEATS if FEATURE_CATALOG[f]["side"] == "macro"]
if not TEST_STOCK_COLS or not TEST_MACRO_COLS:
    raise ValueError(
        f"テスト側の特徴量が空です(stock={len(TEST_STOCK_COLS)}個, macro={len(TEST_MACRO_COLS)}個)。"
        f"FEATURE_TOGGLESで両側とも最低1つはTrueにしてください。"
    )
