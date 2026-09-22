"""Experiment configuration copied from stage3_toggle_experiment.py."""

MAX_PAIR_GAP = 5

EV_WIN_WEIGHT = 2.0

EV_STOP_WEIGHT = 1.0

BASE_MACRO_COLS = [
    "pin_dist_ratio",
    "wall_spread",
    "cta_net_norm",
    "cta_momentum",
    "nk_ret_norm",
]

FEATURE_TOGGLES = {
    # --- 既存9(株側、本番) ---
    # atr_ratio=FalseはBASELINE_STOCK_COLS(既に除外済み)との整合用——本番の9特徴量には依然含まれる。
    "stock_ret_1d": True,
    "stock_ret_5d": False,
    "stock_ret_20d": True,
    "atr_ratio": False,
    "rolling_beta": True,
    "vol_ratio_5d": True,
    "overnight_gap": True,
    "dist_from_high20": True,
    "dist_from_low20": True,
    # --- 既存5(マクロ側、本番) ---
    "pin_dist_ratio": True,
    "wall_spread": True,
    "cta_net_norm": True,
    "cta_momentum": True,
    "nk_ret_norm": True,
    # --- 株側候補(6項目スクリーニングIC通過) ---
    # gap_strength_5/atr_accel=TrueはBASELINE_STOCK_COLS(採用済み)との整合用。
    "rs60": False,  # ベースに追加したが効果なし
    "gap_strength_5": True,  # 改善ベース採用
    "atr_accel": True,  # 改善ベース採用
    # --- 株側候補(全体では不合格だがstock_feature_scorecard.csvでbest_regime=LOW) ---
    "atr_term_ratio": False,
    "dist_from_high60": True,  # 2026-09-20 貪欲法feature searchでbaseline採用(単体でPF+0.073)
    "gap_avg_5d": False,
    "relative_strength_5d": False,  # 悪化 positive_gap_ratio_5との組合せでわずかに改善
    "rs20": False,  # 悪化
    "momentum_accel": False,  # 悪化
    "volume_zscore": False,
    "volume_accel": False,  # 悪化
    "volume_ma_ratio": False,
    "close_location_value": False,
    "body_ratio": False,
    "adx14": False,  # 改善
    "efficiency_ratio20": False,  # 改善
    "days_since_high20": False,
    "new_high20": False,
    "positive_gap_ratio_5": False,  # 改善
    "gap_follow_through": False,  # 悪化
    # --- マクロ候補(usdjpy_chgのみIC通過) ---
    "usdjpy_chg": False,  # 悪化
    "market_vol_regime": False,
    "vix_overnight_chg": False,
    "nk225_ret_5d": False,  # 単一シード平均は改善するが、アンサンブル実運用は悪化
    "topix_ret_5d": False,
    # --- マージン残高候補(margin_ratio_levelのみIC通過) ---
    "margin_ratio_level": False,  # seed平均とMaxDDは悪化するが、アンサンブル実運用では改善
    "margin_ratio_zscore_12w": False,
    "margin_buy_chg_1w": False,
    "margin_short_chg_1w": False,
    # --- セクター相対候補(全不合格) ---
    "sector_rel_ret_5d": False,
    "sector_rel_ret_20d": False,
    "sector_rel_vol_ratio_5d": False,
    "sector_rank_ret_5d": False,
    # --- SHAP交互作用候補 ---
    "rolling_beta_x_cta_net_norm": False,
    "cta_net_norm_x_market_vol_regime": False,
    "pin_dist_ratio_x_cta_net_norm": False,
    "rolling_beta_x_cta_net_norm_x_market_vol_regime": False,
    # --- LOWボラ限定候補(regime_risk_modelの拡大窓is_low版、look-ahead無し) ---
    "cta_net_norm_x_low_v2": False,
    # --- 市場breadth候補(2026-09-21追加、ユーザー提案。未検証) ---
    "market_above_ma25_ratio": False,
    "market_newhigh20_ratio": False,
    "market_newlow20_ratio": False,
    "market_breakout_score": False,
    "market_turnover_z": False,
}

MODEL_ARCH = "dual_stream"

MODEL_HIDDEN_DIM = 20

MODEL_NUM_HEADS = 1

STOCK_HIDDEN_DIM = 32

MACRO_HIDDEN_DIM = (
    16  # "gru_macro_mlp_cross"専用: マクロ側MLPのhidden_dim(株の半分に縮小)
)

CROSS_ATTN_NUM_HEADS = 4

CROSS_ATTN_USE_FFN = (
    False  # "gru_macro_mlp_cross"専用: Cross-Attention後にFFNサブレイヤーを追加するか
)

USE_CROSS_FFN = True

USE_FINAL_NORM = False

NUM_SELF_ATTN_LAYERS = (
    1  # Self-Attention(stock_encoder/macro_encoder)の層数。dual_stream/transformer_only
)

USE_SECTOR_EMBEDDING = (
    False  # 銘柄のTSE33業種をnn.Embeddingで学習し、pooling後の特徴に結合するか
)

SECTOR_EMBED_DIM = 4

SEEDS = [42, 43, 44, 45, 46]

PARALLEL_MAX_CONCURRENT = 10

PROFILE_FIRST_EPOCH = False

USE_AMP = False

AMP_DTYPE = "bf16"

VAL_BATCH_SIZE = (
    1024  # 検証はbackward無しでメモリ圧が低いため学習batchより大きくできる(256->512で
)

TRAIN_BATCH_SIZE = (
    512  # 学習batch。旧MAX_PAIR_GAP=20時代は512でNaN化していたが、ペアマスクの
)

COMMON_SAMPLE_K = (
    200  # 「共通評価サンプル上のPF」: p_win上位k件(seedごとの候補数nに依存しない
)

MIN_RELIABLE_N = 30

MAX_CONCURRENT_POSITIONS = (
    60  # MaxDD計算用の資金曲線シミュレーション、同時保有上限(均等配分)。
)

DAILY_TOPN = (
    5  # 日次Top-N運用評価(evaluate_daily_topn)で1日あたり何件選ぶか。evaluate()の全期間
)

REGIME_FILTER = None

ENTRY_CONVENTION = (
    "d1_open"  # "d1_close"(D+1引け/MOC、本番の確立済み執行規約) / "d1_open"(既定、
)

ATR_BARRIER_UPPER = (
    2.0  # 利確バリア = entry_p + ATR_BARRIER_UPPER * ATR(元は本番から引き継いだ
)

ATR_BARRIER_LOWER = (
    1.0  # 損切りバリア = entry_p - ATR_BARRIER_LOWER * ATR(既定は本番と同じ2:1の
)

USE_REGIME_ATR_BARRIER = (
    False  # Trueにすると、ATR_BARRIER_UPPER/LOWERの代わりにATR_BARRIER_BY_REGIMEを
)

ATR_BARRIER_BY_REGIME = {  # USE_REGIME_ATR_BARRIER=True時のみ使う。(upper, lower)のタプル。
    # 既定は全regime均一(2.0, 1.0)=現行と同じ挙動になるよう初期化。
    "LOW": (2.0, 1.0),
    "MID": (2.0, 1.0),
    "HIGH": (2.0, 1.0),
    "UNKNOWN": (2.0, 1.0),
}

BARRIER_MODE = (
    "trailing"  # "fixed"(entry_p+upper_mult*ATR/entry_p-lower_multのどちらかに触れたら
)

ATR_TRAIL_ACTIVATION = (
    1.0  # BARRIER_MODE="trailing"時のみ。エントリー来の高値がentry_p+
)

ATR_TRAIL_DISTANCE = (
    0.5  # BARRIER_MODE="trailing"時のみ。発動後のストップ = 建値以降の高値 -
)

LABEL_MODE = "risk_adjusted_blend"

QUANTILE_LABEL_UP = 0.30

QUANTILE_LABEL_DOWN = (
    0.30  # "cross_sectional_quantile"専用: 下位何%をDown(離散値0)とするか(残りはHold=1)
)

RET5_HOLDING_DAYS = 5

RET5_DOWN_THRESH = -0.03

RET5_UP_THRESH = 0.03

RISK_BLEND_Z_WEIGHT = (
    0.70  # "risk_adjusted_blend"専用: risk_adjusted_returnのdaily z-scoreの重み
)

RISK_BLEND_RANK_WEIGHT = (
    0.30  # "risk_adjusted_blend"専用: risk_adjusted_returnのdaily rank(pct)の重み
)

USE_REGIME_AWARE_LOSS = (
    False  # Trueにすると、NearPairRankingLossの「同一銘柄・近傍日」ペアに
)

PAIR_MODE = "cross_sectional"

MACRO_CLIP = (
    None  # Noneなら未使用。数値(例: 3.0)を入れるとmacro特徴量をその範囲にclipする。
)

SEQ_LEN, HOLDING_PERIOD = 5, 5
