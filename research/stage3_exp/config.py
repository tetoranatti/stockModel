"""Experiment configuration copied from stage3_toggle_experiment.py.
既存14(株9+マクロ5)も含めて全特徴量がFEATURE_TOGGLESでOn/Off可能。

使い方:
    1. 下のFEATURE_TOGGLESで試したい構成に合わせてTrue/Falseを書き換える
       (既存は既定True。候補は既定False。既存を外す実験もここをFalseにするだけ)
    2. 必要ならMODEL_HIDDEN_DIM/MODEL_NUM_HEADSも変更する(既定は本番と同じ20/1)
    3. python stage3_toggle_experiment.py を実行
"""

# 研究ハーネス側の既定を本番の20から5へ変更(gap短縮ほどPF・seed間の安定性が改善したため採用
MAX_PAIR_GAP = 5

# Top-K/日次Top-N選定に使うscoreを両方ここに揃える旧実装は両方ともp_win-p_stopの1:1
# モデルが直接最適化する量ではあったが本番のev_score軸とは別物だった
EV_WIN_WEIGHT = 2.0
EV_STOP_WEIGHT = 1.0

BASE_MACRO_COLS = [
    "pin_dist_ratio",
    "wall_spread",
    "cta_net_norm",
    "cta_momentum",
    "nk_ret_norm",
]

# ============================================================================
# ここを編集して実験する。既存14もOn/Off可能(Falseにすれば「既存から減らす」実験になる)。
# ============================================================================
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

# "dual_stream" / "gru_only" / "transformer_only" / "gru_macro_mlp_cross"
MODEL_ARCH = "dual_stream"

# 複数シードでmean/stdを見る。単発でよければ[42]だけにする
SEEDS = [42, 43, 44, 45, 46]

# base/testのseedループを並列学習する際の同時起動worker数
PARALLEL_MAX_CONCURRENT = 10

# Trueにすると、最初のシードのベースライン学習のepoch=1だけ
# torch.profilerで計測し、処理時間トップ10を表示する(通常は計測オーバーヘッド無し)。
PROFILE_FIRST_EPOCH = False

# AMP(自動混合精度)を使うか
USE_AMP = False

# "bf16"(RTX 3080はネイティブ対応、GradScaler不要で安全) or "fp16"
AMP_DTYPE = "bf16"

# 検証はbackward無しでメモリ圧が低いため学習batchより大きくできる512以降は頭打ちのため余裕を持たせて1024
VAL_BATCH_SIZE = 1024

# 学習batch 512超を試す場合はNaN化を複数seedで必ず再確認すること
TRAIN_BATCH_SIZE = 512

# dual_stream専用
MODEL_HIDDEN_DIM = 20
MODEL_NUM_HEADS = 1
USE_CROSS_FFN = True
USE_FINAL_NORM = False

# gru_macro_mlp_cross専用
STOCK_HIDDEN_DIM = 32
MACRO_HIDDEN_DIM = 16
CROSS_ATTN_NUM_HEADS = 4
CROSS_ATTN_USE_FFN = False

# dual_stream/transformer_only専用
NUM_SELF_ATTN_LAYERS = 1

# 銘柄のTSE33業種をnn.Embeddingで学習し、pooling後の特徴に結合するか
USE_SECTOR_EMBEDDING = True
SECTOR_EMBED_DIM = 4

# 「共通評価サンプル上のPF」: p_win上位k件でPFを見る補助指標
COMMON_SAMPLE_K = 200

# STRONG BUY件数がこれ未満なら「少数トレードへの偏り」警告を出す
MIN_RELIABLE_N = 30

# MaxDD計算用の資金曲線シミュレーション、同時保有上限(均等配分)
MAX_CONCURRENT_POSITIONS = 60

# 日次Top-N運用評価(evaluate_daily_topn)で1日あたり何件選ぶか evaluate()の全期間
DAILY_TOPN = 5

# None 既定 / "LOW" / "HIGH": 学習・バックテスト両方をそのボラティリティレジーム日だけに絞る
REGIME_FILTER = None

# d1_close(D+1引け/MOC) / d1_open既定
# "d1_open_pullback":翌日の押し目買い
# "d1_open_pullback_high" :始値〜高値の中間を使い、不利な価格で約定したと仮定する
ENTRY_CONVENTION = "d1_open"

# 利確バリア/損切りバリア (±ATR_BARRIER_LOWER * ATR)
ATR_BARRIER_UPPER = 2.0
ATR_BARRIER_LOWER = 1.0

# Trueにすると、regime(LOW/MID/HIGH/UNKNOWN、look-ahead無しの拡大窓版)ごとに異なるバリア倍率を適用する
USE_REGIME_ATR_BARRIER = False

# USE_REGIME_ATR_BARRIER=True専用
ATR_BARRIER_BY_REGIME = {
    "LOW": (2.0, 1.0),
    "MID": (2.0, 1.0),
    "HIGH": (2.0, 1.0),
    "UNKNOWN": (2.0, 1.0),
}

# "fixed" entry_p+upper_mult*ATR/entry_p-lower_multのどちらかに触れたら即終了
# "trailing" 固定利確幅だと「まだ伸びる取引」と「もう頭打ちのを事前に区別できず、upper倍率をどう選んでも取りこぼしと巻き戻しの
# トレードオフが起きることが直接検証で確認された
BARRIER_MODE = "trailing"

# BARRIER_MODE="trailing"専用
ATR_TRAIL_ACTIVATION = 1.0
ATR_TRAIL_DISTANCE = 0.5

# "continuous": simulate_ret_pct_d1_closeの連続値ret_pctをそのまま使う
# "cross_sectional_quantile": 同日全銘柄横断でret_pctを分位点化した3値
# "ret5_fixed_threshold": ATRバリア無しの単純な固定期間(RET5_HOLDING_DAYS
#  日後)リターンを、固定の絶対閾値(RET5_DOWN_THRESH/RET5_UP_THRESH)で3値化する。
# "cross_sectional_rank": 同日全銘柄横断でret_pctを百分位順位(連続値、離散化しない)に変換する
# "risk_adjusted_return": ret_pctをシグナル時点のATR(価格比)で正規化した
# "risk_adjusted_blend" : RISK_BLEND_Z_WEIGHT:RISK_BLEND_RANK_WEIGHTでブレンドする
LABEL_MODE = "risk_adjusted_blend"

# "risk_adjusted_blend"専用:z-scoreの重み/daily rank(pct)の重み
RISK_BLEND_Z_WEIGHT = 0.70
RISK_BLEND_RANK_WEIGHT = 0.30

# "cross_sectional_quantile"専用: 上位何%をUp(離散値2)とするか/下位何%をDown(離散値0)とするか
QUANTILE_LABEL_UP = 0.30
QUANTILE_LABEL_DOWN = 0.30

# "ret5_fixed_threshold"専用: 固定保有日数
RET5_HOLDING_DAYS = 5
# Down(離散値0)/Up(離散値2)、残りはHold=1
RET5_DOWN_THRESH = -0.03
RET5_UP_THRESH = 0.03

# Trueにすると、NearPairRankingLossの「同一銘柄・近傍日」ペアに
USE_REGIME_AWARE_LOSS = False

# "same_ticker": SameTickerBatchSampler+「同一銘柄・近傍日」
# "cross_sectional": SameDateBatchSampler+「同一日・異なる銘柄」
PAIR_MODE = "cross_sectional"

# Noneなら未使用。数値(例: 3.0)を入れるとmacro特徴量をその範囲にclipする。
MACRO_CLIP = None

# 両方とも短くするほど安定性ともに改善したため
SEQ_LEN, HOLDING_PERIOD = 5, 5

# TOPN件からセクター制限内での選択
MAX_PER_SECTOR = 3

# TOPN件からリターンベースのクラスタ制限内での選択(2026-09-23追加、ユーザー提案。
# 業種分類より実際の値動きの共動性を反映した分散制約の候補として、セクター版と比較検証中)
# "monthly_consensus": 1/5/20/60日の複数horizonリターン(横断面標準化)で月ごとにKMeans分け、
#   複数月の一致度(consensus)で最終クラスタを決める(ユーザー提案、単月のノイズに
#   左右されにくく、半導体テーマのような業種横断の連動も拾える)。1日・5日は月内複数時点
#   サンプリング平均でノイズを抑える(cluster_utils.py::_monthly_feature_row)。
#   データソースはfetch_cluster_universe_history.pyが作る約2年分の長期キャッシュ
#   (data/cache/cluster_daily_bars_raw.parquet)を使う。無ければ直近6ヶ月版にフォールバック。
# "static_corr": 60日リターン系列の相関一発で階層クラスタリング(初期実装)。
CLUSTER_METHOD = "monthly_consensus"
N_CLUSTERS = 10                        # 最終クラスタ数(シルエットスコアはk=2が最高だが
                                        # ほぼ無意味な分割になるため、分散制約としての実用性
                                        # を優先しk=10を採用、2026-09-23)
N_MONTHLY_BUCKETS = 8                  # monthly_consensus専用: 月内を何グループに分けるか
CLUSTER_RETURN_WINDOWS = (1, 5, 20, 60)  # monthly_consensus専用: 月内クラスタリングに使う複数horizon
# 2026-09-23、ユーザー提案: 分散投資が本当に効いてほしいのは下落局面で一緒に沈む銘柄を
# 避けたい場面なので、市場全体(横断面中央値)が下落した月だけでconsensusを取る。
# 過去約25ヶ月中、下落月は10ヶ月のみだったためMIN_SHARED_MONTHSも5へ下げている。
CLUSTER_MONTH_FILTER = "down"          # "down" / "up" / "all"
CLUSTER_MARKET_RETURN_WINDOW = 20      # 月間パフォーマンス判定に使うhorizon
MIN_SHARED_MONTHS = 5                  # monthly_consensus専用: consensus算出に必要な最小共通月数(下落月10ヶ月分中)
CLUSTER_LOOKBACK_DAYS = 60             # static_corr専用: 相関を取るリターンの窓
MAX_PER_CLUSTER = 2

REGIME_ID_MAP = {
    "LOW": 0,
    "MID": 1,
    "HIGH": 2,
}
