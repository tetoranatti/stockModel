"""特徴量候補をTrue/Falseで手動切り替えして、実NN(DualStream_GRU_PreLN_Transformer)の
単一シード学習+D+1引け(MOC)バックテストで検証するための汎用実験フロー。
既存14(株9+マクロ5)も含めて全特徴量がFEATURE_TOGGLESでOn/Off可能。
「ベースライン」は常に既存14固定(比較の基準点)、「テスト」側はFEATURE_TOGGLESで
Trueにしたものだけで構成される(既存を減らす実験も、候補を足す実験もこれ1つでできる)。

使い方:
    1. 下のFEATURE_TOGGLESで試したい構成に合わせてTrue/Falseを書き換える
       (既存14は既定True。候補は既定False。既存を外す実験もここをFalseにするだけ)
    2. 必要ならMODEL_HIDDEN_DIM/MODEL_NUM_HEADSも変更する(既定は本番と同じ20/1)
    3. python stage3_toggle_experiment.py を実行
"""

import os
import sys
import time
import json
import hashlib
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, r"F:\stockModel")
sys.path.insert(0, r"F:\stockModel\research")

try:
    # バックグラウンド実行時のブロックバッファリング遅延を防ぐ。encoding='utf-8'を明示しないと
    # cp932にfall backしem dash等でUnicodeEncodeErrorが起きるため明示指定。
    sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")
    sys.stderr.reconfigure(line_buffering=True, encoding="utf-8")
except Exception:
    pass

_SCRIPT_START = time.time()


def elapsed():
    s = time.time() - _SCRIPT_START
    m, s = divmod(int(s), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def log(msg):
    print(f"[{elapsed()}] {msg}", flush=True)


from training.train_model_v8_exp import (
    UniverseDataset,
    SameTickerBatchSampler,
    MAX_PAIR_GAP,
    UNIVERSE_PATH,
    MIN_CROSS_SECTION,
)

# 研究ハーネス側の既定を本番の20から5へ変更(gap短縮ほどPF・seed間の安定性が改善した
# ため採用。本番training/train_model_v8_exp.pyは意図的に未変更)。
# クラス定義(NearPairRankingLossRegimeAwareのデフォルト引数)より前で代入が必要。
MAX_PAIR_GAP = 5
from modules.model_arch import (
    DualStream_GRU_PreLN_Transformer,
    GRUOnlyModel,
    TransformerOnlyModel,
    GRU_MacroMLP_CrossAttn_Model,
)
from torch.profiler import profile as torch_profile, ProfilerActivity
from modules.macro_features import load_macro_slim5
from modules.stock_features import STOCK_FEATURE_COLS
from modules.cross_sectional_features import (
    apply_log_transform,
    compute_cross_sectional_stats,
    valid_cross_section_dates,
    normalize_cross_sectional,
)
from _feature_cache_utils import (
    get_full_feature_pool_df,
    get_full_macro_pool_df,
    get_margin_pool_df,
    get_sector_relative_pool_df,
    _load_sector_map,
)
from modules.regime_risk_model import _compute_vol_regime_is_low
from modules.model_inference import predict_probabilities_ensemble_batch
from _eval_utils import (
    REGIME_ID_MAP,
    pf_of,
    compute_max_drawdown,
    compute_concentration,
    compute_is_low_regime_filter,
    compute_is_high_regime_filter,
    compute_is_mid_regime_filter,
    compute_regime_labels,
    compute_regime_labels_expanding,
    regime_breakdown,
    monthly_breakdown,
    evaluate,
    evaluate_daily_topn,
)

# 銘柄->TSE33業種idの対応表(USE_SECTOR_EMBEDDING用)。未分類はid=0。
# _load_sector_mapはユニバース非依存なのでモジュール読み込み時に1回だけ計算する。
_SECTOR_MAP_RAW = _load_sector_map()
_SECTOR_NAMES_SORTED = sorted({v for v in _SECTOR_MAP_RAW.values() if v is not None})
SECTOR_NAME_TO_ID = {
    name: i + 1 for i, name in enumerate(_SECTOR_NAMES_SORTED)
}  # 0=不明/未分類用に予約
TICKER_TO_SECTOR_ID = {
    t: SECTOR_NAME_TO_ID.get(v, 0) for t, v in _SECTOR_MAP_RAW.items()
}
NUM_SECTORS = len(_SECTOR_NAMES_SORTED) + 1

# Top-K/日次Top-N選定に使うscoreを両方ここに揃える旧実装は両方とも
# p_win-p_stopの1:1で、モデルが直接最適化する量ではあったが本番のev_score軸とは別物だった)。
EV_WIN_WEIGHT = 2.0
EV_STOP_WEIGHT = 1.0


class NearPairRankingLossFast(nn.Module):
    """NearPairRankingLossRegimeAware(regime=None)のO(B×max_gap)版(scoreの重みも含め
    数値的に厳密に同値。本番training/train_model_v8_exp.py::NearPairRankingLossとは
    score定義がEV_WIN_WEIGHT/EV_STOP_WEIGHT分だけ異なる点に注意——このファイル内では
    未使用で、正しさ検証用に残してあるだけ)。
    SameTickerBatchSampler+build_datasetにより各バッチ内は同一銘柄・tidx昇順で並ぶため、
    オフセットδ=1..max_gapだけ見ればgap<=max_gapの全ペアを漏れなく拾える。全ペア行列を
    作らずtorch.catで済ませ、O(B²)のメモリ確保・2Dマスクインデックスを回避する。
    tidxが昇順という前提が崩れる呼び出し方だと結果が変わる点に注意。"""

    def __init__(self, max_gap=MAX_PAIR_GAP):
        super().__init__()
        self.max_gap = max_gap

    def forward(self, logits, ret_pct, tidx):
        probs = torch.softmax(logits, dim=-1)
        score = EV_WIN_WEIGHT * probs[:, 2] - EV_STOP_WEIGHT * probs[:, 0]
        B = score.shape[0]
        parts = []
        for delta in range(1, min(self.max_gap, B - 1) + 1):
            s_diff = score[delta:] - score[:-delta]
            r_diff = ret_pct[delta:] - ret_pct[:-delta]
            t_diff = (tidx[delta:] - tidx[:-delta]).abs()
            sign = torch.sign(r_diff)
            mask = (sign != 0) & (t_diff <= self.max_gap)
            if mask.any():
                parts.append(torch.nn.functional.softplus(-sign[mask] * s_diff[mask]))
        if not parts:
            return score.sum() * 0.0
        return torch.cat(parts).mean()


class NearPairRankingLossRegimeAware(nn.Module):
    """NearPairRankingLossに「同一regime(LOW/MID/HIGH)のペアだけ比較する」制約を追加した版。
    同一銘柄・近傍日(|tidx_i-tidx_j|<=max_gap)ペアの42.7%がregimeを跨いでいた実測を受けて、
    regime由来のリターン格差が銘柄固有シグナルを汚染している可能性の検証用。
    regime引数はint(-1=対象外/不明で常に除外、Noneなら制約無し=構造的には本番training/
    train_model_v8_exp.py::NearPairRankingLossと同一)。

    scoreはEV_WIN_WEIGHT*p_win-EV_STOP_WEIGHT*p_stop(本番のev_score、pipeline/
    run_dynamic_regime_screening_v8.pyの候補順位・modules/model_inference.pyのev_raw
    と同じ2:1重み)。本番training/train_model_v8_exp.py::NearPairRankingLossは現状
    p_win-p_stopの1:1のままなので、この点はもう本番の学習コードと数値的に同一ではない
    (2026-09-21、ユーザー指摘「学習時のランキング軸と実運用時のランキング軸が不一致」を
    受けて、この実験ハーネス側を実運用の日次Top-N選定軸に合わせる方向に変更。本番の
    学習コード自体を変えるかどうかは、この実験で効果を確認してから別途判断する)。

    cross_sectional=Trueにすると、ペア制約が「同一銘柄・近傍日(0<gap<=max_gap)」から
    「同一日・異なるサンプル(gap==0)」に切り替わる(SameDateBatchSamplerと組み合わせて
    使う想定。PAIR_MODE="cross_sectional"参照)。実運用が実際に行っている「同日の複数
    銘柄を横断してランキングし日次Top-N件を選ぶ」タスクを学習損失として直接模したもの
    (2026-09-21追加、ユーザー指摘「学習時のペア軸(銘柄内・時系列)が実運用の同日横断
    ランキングと構造的に不一致」を受けて実装)。

    forward()は(loss_mean, n_pairs)を返す。バッチごとに有効ペア数が大きく変わりうるため、
    呼び出し側でn_pairsを使いエポック全体の平均をペア数加重にし、n_pairs=0のバッチ
    (有効ペア無し)を平均から自然に除外できるようにする。"""

    def __init__(self, max_gap=MAX_PAIR_GAP, cross_sectional=False):
        super().__init__()
        self.max_gap = max_gap
        self.cross_sectional = cross_sectional

    def forward(self, logits, ret_pct, tidx, regime=None):
        probs = torch.softmax(logits, dim=-1)
        score = EV_WIN_WEIGHT * probs[:, 2] - EV_STOP_WEIGHT * probs[:, 0]
        diff_score = score.unsqueeze(1) - score.unsqueeze(0)
        diff_ret = ret_pct.unsqueeze(1) - ret_pct.unsqueeze(0)
        sign = torch.sign(diff_ret)
        gap = (tidx.unsqueeze(1) - tidx.unsqueeze(0)).abs()
        if self.cross_sectional:
            mask = (sign != 0) & (gap == 0)  # 同一日(=異なる銘柄)のペアだけ比較
        else:
            mask = (sign != 0) & (gap > 0) & (gap <= self.max_gap)
        if regime is not None:
            same_regime = (regime.unsqueeze(1) == regime.unsqueeze(0)) & (
                regime.unsqueeze(1) >= 0
            )
            mask = mask & same_regime
        n_pairs = int(mask.sum().item())
        if n_pairs == 0:
            return diff_score.sum() * 0.0, 0
        return (
            torch.nn.functional.softplus(-sign[mask] * diff_score[mask]).mean(),
            n_pairs,
        )


class SameDateBatchSampler:
    """バッチを「同一取引日(tidx、全銘柄共通のカレンダー日位置)」だけで構成するサンプラー
    (SameTickerBatchSamplerの同日横断/cross-sectional版)。同じバッチ内の全サンプルが
    同じtidxになるため、NearPairRankingLossRegimeAware(cross_sectional=True)のペア制約
    (gap==0)をバッチ構成の時点で自動的に満たす。PAIR_MODE="cross_sectional"のとき
    SameTickerBatchSamplerの代わりに使う(2026-09-21追加)。1バッチ=1取引日の全銘柄横断
    サンプル(batch_size超なら分割)。SameTickerBatchSamplerと同じインターフェース
    (__iter__/__len__のみ、torch.utils.data.Samplerは継承しない——train_model()側も
    Dataset/DataLoaderを経由せずlist(sampler)で直接消費するため不要)。"""

    def __init__(self, tidx_array, batch_size, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        groups = {}
        for i, d in enumerate(tidx_array):
            groups.setdefault(int(d), []).append(i)
        self._batches = []
        for d, idxs in groups.items():
            for start in range(0, len(idxs), batch_size):
                chunk = idxs[start : start + batch_size]
                if len(chunk) >= 4:
                    self._batches.append(chunk)

    def __iter__(self):
        batches = self._batches
        if self.shuffle:
            batches = batches.copy()
            random.shuffle(batches)
        return iter(batches)

    def __len__(self):
        return len(self._batches)


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

MODEL_ARCH = "dual_stream"  # "dual_stream"(本番) / "gru_only" / "transformer_only" / "gru_macro_mlp_cross"
MODEL_HIDDEN_DIM = 20
MODEL_NUM_HEADS = 1
STOCK_HIDDEN_DIM = 32  # "gru_macro_mlp_cross"専用: 株側GRUのhidden_dim
MACRO_HIDDEN_DIM = (
    16  # "gru_macro_mlp_cross"専用: マクロ側MLPのhidden_dim(株の半分に縮小)
)
CROSS_ATTN_NUM_HEADS = 4  # "gru_macro_mlp_cross"専用: Cross-Attentionのhead数(STOCK_HIDDEN_DIMで割り切れる値に)
CROSS_ATTN_USE_FFN = (
    False  # "gru_macro_mlp_cross"専用: Cross-Attention後にFFNサブレイヤーを追加するか
)
# (dual_streamのUSE_CROSS_FFNで効果確認済みのため試験用)
USE_CROSS_FFN = True  # Cross-Attention後にFFNサブレイヤーを追加するか(dual_stream限定、本番採用済み)
USE_FINAL_NORM = False  # Cross-Attention後・pooling前に最終LayerNormを追加するか(dual_stream限定、既定=本番と同じ)
NUM_SELF_ATTN_LAYERS = (
    1  # Self-Attention(stock_encoder/macro_encoder)の層数。dual_stream/transformer_only
)
# 両対応(gru_onlyでは無視される)。既定1=本番と同じ
USE_SECTOR_EMBEDDING = (
    False  # 銘柄のTSE33業種をnn.Embeddingで学習し、pooling後の特徴に結合するか
)
# (dual_stream限定。sector33業種idをw_sの最終列に定数値として埋め込み、
# モデル側でGRUに渡す前に切り離してnn.Embeddingに通す実装)。
# 検証済み・不採用: baseline(10株+5マクロ)への追加で5seedアンサンブル
# PF -0.055、明確な悪化(sector_rel_*特徴量との重複+過学習リスク、詳細はmemory参照)。
SECTOR_EMBED_DIM = 4  # USE_SECTOR_EMBEDDING=True時のembedding次元
SEEDS = [42, 43, 44, 45, 46]  # 複数シードでmean/stdを見る。単発でよければ[42]だけにする
PARALLEL_MAX_CONCURRENT = 10  # base/testのseedループを並列学習する際の同時起動worker数
# (3seed=2.44x・5seed=2.9倍高速化、結果は逐次と完全一致を確認済み、詳細は
# research/parallel_seed_sweep.pyと[[feedback_nn_training_performance]]参照)。
# PROFILE_FIRST_EPOCH=True時はこの並列化は使われない(逐次ループにフォールバック)。
PROFILE_FIRST_EPOCH = False  # Trueにすると、最初のシードのベースライン学習のepoch=1だけ
# torch.profilerで計測し、処理時間トップ10を表示する(通常は計測オーバーヘッド無し)。
USE_AMP = False  # AMP(自動混合精度)を使うか。fp16/bf16の丸め誤差が入り数値は厳密一致しないので、
# 精度への影響を見てから使うこと
AMP_DTYPE = "bf16"  # "bf16"(RTX 3080はネイティブ対応、GradScaler不要で安全) or "fp16"
VAL_BATCH_SIZE = (
    1024  # 検証はbackward無しでメモリ圧が低いため学習batchより大きくできる(256->512で
)
# 約2倍高速化、512以降は頭打ちのため余裕を持たせて1024)。NearPairRankingLossは
# バッチ内ペアのみ比較するため、チャンクの切れ目でval_lossが僅かに変わる点に注意。
TRAIN_BATCH_SIZE = (
    512  # 学習batch。旧MAX_PAIR_GAP=20時代は512でNaN化していたが、ペアマスクの
)
# スパース性が原因だったとみられ、gap=5/hp=5/seq_len=5の現行設定では
# 512でも5seed全てNaN無し(256比で1.43倍高速・より安定・ensemble PFも改善)。
# 512超を試す場合はNaN化を複数seedで必ず再確認すること。
COMMON_SAMPLE_K = (
    200  # 「共通評価サンプル上のPF」: p_win上位k件(seedごとの候補数nに依存しない
)
# 固定件数)でPFを見る補助指標。STRONG BUY閾値だとseedでn=0〜数千件とばらつくため。
MIN_RELIABLE_N = 30  # STRONG BUY件数がこれ未満なら「少数トレードへの偏り」警告を出す
MAX_CONCURRENT_POSITIONS = (
    60  # MaxDD計算用の資金曲線シミュレーション、同時保有上限(均等配分)。
)
# 1日5件×最大保有5営業日の流入ペースに対し旧cap=20は容量不足で新規候補が
# 入れず、テスト構成で後半期間ずっと0件約定になっていた。cap=60で約定率
# 97%/95%まで回復し、PFがtop_k_pf(制約無視)とほぼ一致する水準に。
# 絶対PF水準はこのcap変更前後で比較できない点に注意(A/B相対比較の方向性は不変)。
DAILY_TOPN = (
    5  # 日次Top-N運用評価(evaluate_daily_topn)で1日あたり何件選ぶか。evaluate()の全期間
)
# score上位K件は特定の数日にシグナルが偏っていても検知できないため、実際の運用
# (毎日決まった件数しか執行できない)により忠実な評価として追加。
REGIME_FILTER = None  # None(全期間、既定) / "LOW" / "HIGH": 学習・バックテスト両方をそのボラティリティ
# レジーム(拡大窓is_low/is_high、look-ahead無し)の日だけに絞る(レジーム別モデル用)。
# ベースライン・テスト構成の両方に同じフィルタをかけるので比較の公平性は保たれる。
ENTRY_CONVENTION = (
    "d1_open"  # "d1_close"(D+1引け/MOC、本番の確立済み執行規約) / "d1_open"(既定、
)
# D+1始値/MOO。5seed比較で日次Top-5 PFがアンサンブルで0.927->1.201に改善した
# ため研究ハーネス側の既定として維持。本番の執行規約自体は未変更——寄成注文の
# 約定安定性を運用面で検証してから判断すべき)。学習ターゲット・バックテスト
# 両方に同じ規約が使われる(simulate_ret_pct_d1_close経由で共有)。
ATR_BARRIER_UPPER = (
    2.0  # 利確バリア = entry_p + ATR_BARRIER_UPPER * ATR(元は本番から引き継いだ
)
# ハードコード値。他のラベル設計toggleと同様に検証できるようトグル化)
ATR_BARRIER_LOWER = (
    1.0  # 損切りバリア = entry_p - ATR_BARRIER_LOWER * ATR(既定は本番と同じ2:1の
)
# 非対称リワード比。simulate_ret_pct_d1_close経由で学習ターゲット・バックテスト両方に使われる)
USE_REGIME_ATR_BARRIER = (
    False  # Trueにすると、ATR_BARRIER_UPPER/LOWERの代わりにATR_BARRIER_BY_REGIMEを
)
# 使い、regime(LOW/MID/HIGH/UNKNOWN、look-ahead無しの拡大窓版)ごとに異なる
# バリア倍率を適用する。regime-split retraining([[regime_specific_pf_model_2026-09-19]]
# で2/3のregimeが悪化・データ枯渇で失敗)とは違い、モデル・学習データは1つのまま
# ラベルの定義だけをregime条件付きにする軽量な変更。
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
# 即終了) / "trailing"(採用。固定利確幅だと「まだ伸びる取引」と「もう頭打ちの
# 取引」を事前に区別できず、upper倍率をどう選んでも取りこぼしと巻き戻しの
# トレードオフが起きることが直接検証で確認された——2.0ATR到達済み取引のうち
# 3.0ATRまで伸びたのは42.6%のみ。トレーリングストップは固定の利確ラインを置かず、
# 値動きに追従してストップを切り上げる。ATR_TRAIL_ACTIVATION/DISTANCEで挙動を
# 制御、ATR_BARRIER_LOWERは発動前の初期ストップとして引き続き使う。
# (1.0, 0.5)採用: 固定2.0/1.0比でensemble PF+0.204改善を確認済み。両方0.5未満に
# 縮めるとPFが際限なく上がる現象は、日足バックテストの非現実的な前提を突いた
# 見かけ倒しと判断し不採用(atr_barrier_investigation_2026-09-20.md参照)。
ATR_TRAIL_ACTIVATION = (
    1.0  # BARRIER_MODE="trailing"時のみ。エントリー来の高値がentry_p+
)
# ATR_TRAIL_ACTIVATION*ATRに到達するまではトレーリング未発動
# (初期ストップ=entry_p-ATR_BARRIER_LOWER*ATRのまま)。0.25/0.5は見かけ倒しの疑いで不採用。
ATR_TRAIL_DISTANCE = (
    0.5  # BARRIER_MODE="trailing"時のみ。発動後のストップ = 建値以降の高値 -
)
# ATR_TRAIL_DISTANCE*ATR(切り上げのみ)。exit理由診断で発動後の手仕舞いが
# 全件+0.44%以上の実質的な利益確保だったことを確認済み(0.25は見かけ倒しの疑いで不採用)。
LABEL_MODE = "risk_adjusted_blend"  # 学習ターゲット(tr_y/va_y)の定義。バックテスト側
# (prepare_backtest_pool)は常に連続値ret_pct(ATRバリア方式)のまま——ここは
# 学習ターゲットのみ変える。5seed比較でcontinuous(mean=1.069/ensemble=1.075)より
# risk_adjusted_return(mean=1.149/ensemble=1.204)が優れていたため、さらに
# z-score+rankでブレンドしたrisk_adjusted_blendを採用。選択肢:
#   "continuous": simulate_ret_pct_d1_closeの連続値ret_pctをそのまま使う。
#   "cross_sectional_quantile": 同日全銘柄横断でret_pctを分位点化した3値
#     (下位QUANTILE_LABEL_DOWN%=0/残り=1/上位QUANTILE_LABEL_UP%=2)。
#     market-beta方向とのノイズ混入(同日の市場全体の上げ下げ)を除く狙い。
#   "ret5_fixed_threshold": ATRバリア無しの単純な固定期間(RET5_HOLDING_DAYS
#     日後)リターンを、固定の絶対閾値(RET5_DOWN_THRESH/RET5_UP_THRESH)で3値化する。
#   "cross_sectional_rank": 同日全銘柄横断でret_pctを百分位順位(連続値、
#     離散化しない)に変換する。cross_sectional_quantileと同じ狙いだが、
#     離散化しないので同値ペアの除外による有効ペア数減少が起きにくい。
#   "risk_adjusted_return": ret_pctをシグナル時点のATR(価格比)で正規化した
#     連続値。高ボラ銘柄の額面%が過大評価されるのを補正する狙い。
# 離散3値モード(cross_sectional_quantile/ret5_fixed_threshold)は同じbin
# 同士のペアがdiff=0となりNearPairRankingLossから自動的に除外される
# (有効ペア数は減る)。連続値モード(continuous/cross_sectional_rank/
# risk_adjusted_return)は厳密な同値でない限り除外されない。
QUANTILE_LABEL_UP = 0.30  # "cross_sectional_quantile"専用: 上位何%をUp(離散値2)とするか
QUANTILE_LABEL_DOWN = (
    0.30  # "cross_sectional_quantile"専用: 下位何%をDown(離散値0)とするか(残りはHold=1)
)
RET5_HOLDING_DAYS = 5  # "ret5_fixed_threshold"専用: 固定保有日数
RET5_DOWN_THRESH = -0.03  # "ret5_fixed_threshold"専用: この値未満をDown(離散値0)
RET5_UP_THRESH = 0.03  # "ret5_fixed_threshold"専用: この値超をUp(離散値2)、残りはHold=1
RISK_BLEND_Z_WEIGHT = (
    0.70  # "risk_adjusted_blend"専用: risk_adjusted_returnのdaily z-scoreの重み
)
RISK_BLEND_RANK_WEIGHT = (
    0.30  # "risk_adjusted_blend"専用: risk_adjusted_returnのdaily rank(pct)の重み
)
USE_REGIME_AWARE_LOSS = (
    False  # Trueにすると、NearPairRankingLossの「同一銘柄・近傍日」ペアに
)
# gap=5/hp=5/seq_len=5/d1_open採用後で逆転
# Falseへ採用変更。gap/hp/seq_lenを短縮したことで比較対象ペア自体が元々近い
# 市場環境同士になり、regime制約が削るノイズよりペア数減少による学習信号減少の
# 方が相対的に効くようになったと考えられる。REGIME_FILTERとは独立。
PAIR_MODE = "cross_sectional"  # "same_ticker"(既定): SameTickerBatchSampler+「同一銘柄・近傍日」
# (現行)"cross_sectional": SameDateBatchSampler+「同一日・異なる銘柄」
MACRO_CLIP = (
    None  # Noneなら未使用。数値(例: 3.0)を入れるとmacro特徴量をその範囲にclipする。
)
# HIGHレジーム専用モデルの不安定化がこの正規化の粗さによるものか切り分けるためのtoggle。
# ============================================================================

ARCH_CLASSES = {
    "dual_stream": DualStream_GRU_PreLN_Transformer,
    "gru_only": GRUOnlyModel,
    "transformer_only": TransformerOnlyModel,
    "gru_macro_mlp_cross": GRU_MacroMLP_CrossAttn_Model,
}
if MODEL_ARCH not in ARCH_CLASSES:
    raise ValueError(
        f"MODEL_ARCH='{MODEL_ARCH}'は未対応です。選択肢: {list(ARCH_CLASSES)}"
    )

# 特徴量カタログ: side='stock'(銘柄横断正規化される側)/'macro'(全銘柄共通の日次値側)、
# source=どこから値を取得するか(stock_pool/macro_pool/margin_pool/sector_pool/computedのいずれか)。
# (対象列リスト, side, source)のグループごとに登録し、computed特徴量だけ個別に指定する。
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

# stock側の計算式(dfは銘柄ごと、macro_pool_dfは日次共通)。rolling_betaの生列は
# FEATURE_TOGGLES上のOn/Offに関わらずpool_df内に常に存在するので、計算自体はどの設定でも成立する。
STOCK_COMPUTE_FNS = {
    "rolling_beta_x_cta_net_norm": lambda df, mpdf: df["rolling_beta"]
    * mpdf["cta_net_norm"].reindex(df.index).fillna(0.0),
    "rolling_beta_x_cta_net_norm_x_market_vol_regime": lambda df, mpdf: (
        df["rolling_beta"]
        * mpdf["cta_net_norm"].reindex(df.index).fillna(0.0)
        * mpdf["market_vol_regime"].reindex(df.index).fillna(0.0)
    ),
}
# macro側の計算式(macro_pool_df自体に1回だけ追加する、日次共通値)
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

# BARRIER_MODE="trailing"採用後、後退法(greedy_feature_backward.py)で全16特徴量を再検証し
# stock_ret_5dを除外(アンサンブル日次Top-5 PF=1.424->1.604)。round2で残り14特徴量の除外を
# 試したが誰も上回れず探索収束。本番training/train_model_v8_exp.pyのSTOCK_FEATURE_COLSは
# 意図的に変更していない(研究ハーネスのみ)。
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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 両方とも短くするほど安定性ともに改善したため
SEQ_LEN, HOLDING_PERIOD = 5, 5


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def compute_global_split_date(valid_dates, regime_filter=None):
    """train/valの分割を全銘柄共通のカレンダー日で行う(銘柄ごとの75%点だと分割日が
    最大約5.5ヶ月ずれ、out-of-sample性が銘柄間で一貫しなくなるため)。
    regime_filterを渡すと、regime該当日だけに絞った75%点を返す。"""
    sorted_dates = pd.DatetimeIndex(sorted(valid_dates))
    if regime_filter is not None:
        regime_ok_global = regime_filter.reindex(sorted_dates).fillna(False).values
        eligible_dates = sorted_dates[regime_ok_global]
        if len(eligible_dates) == 0:
            return sorted_dates[-1] if len(sorted_dates) else pd.Timestamp.max
        return eligible_dates[int(len(eligible_dates) * 0.75)]
    return sorted_dates[int(len(sorted_dates) * 0.75)]


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
):
    """regime_filter: date->bool(True=採用)のpd.Seriesを渡すと、その日付のサンプルだけ
    train/valに採用する(regime別モデル用)。SEQ_LEN分の価格系列window自体はregimeで
    フィルタせず連続したまま使う(不連続だと時系列モデルの入力が歪むため)。
    regime_labels_for_loss: date->'LOW'/'MID'/'HIGH'のpd.Seriesを渡すと、各サンプルの
    regime id(0/1/2、不明なら-1)を6番目の要素として返す(NearPairRankingLossRegimeAware用、
    regime_filterとは独立)。
    barrier_regime_labels: compute_simulated_targets参照。"""
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


def baseline_cache_fingerprint(n_train, n_val, split_date):
    """baselineモデルの学習結果を左右する全設定をフィンガープリント化する。候補特徴量
    (テスト側)だけを変えて比較するワークフローではbaseline側の設定は変わらないことが
    多いため、前回と一致すればbaselineの学習をスキップしキャッシュ済みstate_dictを再利用
    する。n_train/n_val/split_dateも含めることで、データ更新時は自動的に無効化される。"""
    cfg = dict(
        baseline_stock_cols=sorted(BASELINE_STOCK_COLS),
        baseline_macro_cols=sorted(BASELINE_MACRO_COLS),
        model_arch=MODEL_ARCH,
        model_hidden_dim=MODEL_HIDDEN_DIM,
        model_num_heads=MODEL_NUM_HEADS,
        use_cross_ffn=USE_CROSS_FFN,
        use_final_norm=USE_FINAL_NORM,
        num_self_attn_layers=NUM_SELF_ATTN_LAYERS,
        use_sector_embedding=USE_SECTOR_EMBEDDING,
        sector_embed_dim=SECTOR_EMBED_DIM,
        stock_hidden_dim=STOCK_HIDDEN_DIM,
        macro_hidden_dim=MACRO_HIDDEN_DIM,
        cross_attn_num_heads=CROSS_ATTN_NUM_HEADS,
        cross_attn_use_ffn=CROSS_ATTN_USE_FFN,
        seq_len=SEQ_LEN,
        holding_period=HOLDING_PERIOD,
        max_pair_gap=MAX_PAIR_GAP,
        pair_mode=PAIR_MODE,
        entry_convention=ENTRY_CONVENTION,
        atr_barrier_upper=ATR_BARRIER_UPPER,
        atr_barrier_lower=ATR_BARRIER_LOWER,
        use_regime_atr_barrier=USE_REGIME_ATR_BARRIER,
        atr_barrier_by_regime=(
            sorted(ATR_BARRIER_BY_REGIME.items()) if USE_REGIME_ATR_BARRIER else None
        ),
        barrier_mode=BARRIER_MODE,
        atr_trail_activation=ATR_TRAIL_ACTIVATION,
        atr_trail_distance=ATR_TRAIL_DISTANCE,
        label_mode=LABEL_MODE,
        quantile_up=QUANTILE_LABEL_UP,
        quantile_down=QUANTILE_LABEL_DOWN,
        ret5_holding_days=RET5_HOLDING_DAYS,
        ret5_down=RET5_DOWN_THRESH,
        ret5_up=RET5_UP_THRESH,
        risk_blend_z=RISK_BLEND_Z_WEIGHT,
        risk_blend_rank=RISK_BLEND_RANK_WEIGHT,
        train_batch_size=TRAIN_BATCH_SIZE,
        val_batch_size=VAL_BATCH_SIZE,
        use_regime_aware_loss=USE_REGIME_AWARE_LOSS,
        regime_filter=REGIME_FILTER,
        macro_clip=MACRO_CLIP,
        ev_win_weight=EV_WIN_WEIGHT,
        ev_stop_weight=EV_STOP_WEIGHT,
        n_train=n_train,
        n_val=n_val,
        split_date=str(split_date),
    )
    key = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def build_model(s_cols, m_cols):
    """MODEL_ARCH等の現在のトグル値からモデルを構築する。train_model()と
    parallel_seed_sweep.pyの親プロセス側(state_dict読み込み前の再構築)の両方から使う
    ——モデル構築ロジックを2箇所に重複させないため。"""
    arch_cls = ARCH_CLASSES[MODEL_ARCH]
    if MODEL_ARCH == "gru_macro_mlp_cross":
        # 株64/マクロ32という非対称次元構成のため、他アーキテクチャ共通のMODEL_HIDDEN_DIM
        # ではなく専用toggle(STOCK_HIDDEN_DIM/MACRO_HIDDEN_DIM)を使う。
        model_kwargs = dict(
            stock_dim=len(s_cols),
            macro_dim=len(m_cols),
            stock_hidden=STOCK_HIDDEN_DIM,
            macro_hidden=MACRO_HIDDEN_DIM,
            num_heads=CROSS_ATTN_NUM_HEADS,
            num_classes=3,
            dropout=0.2,
            use_cross_ffn=CROSS_ATTN_USE_FFN,
            seq_len=SEQ_LEN,
        )
    else:
        stock_dim = len(s_cols) + (
            1 if (USE_SECTOR_EMBEDDING and MODEL_ARCH == "dual_stream") else 0
        )
        model_kwargs = dict(
            stock_dim=stock_dim,
            macro_dim=len(m_cols),
            hidden_dim=MODEL_HIDDEN_DIM,
            num_heads=MODEL_NUM_HEADS,
            num_classes=3,
            dropout=0.2,
            seq_len=SEQ_LEN,
        )
        if MODEL_ARCH == "dual_stream":
            model_kwargs["use_cross_ffn"] = USE_CROSS_FFN
            model_kwargs["use_final_norm"] = USE_FINAL_NORM
            if USE_SECTOR_EMBEDDING:
                # w_sの最終列に埋め込んだsector_idを切り離してnn.Embeddingに通す
                # (stock_dimは上でs_cols+1に増やし済み)。dual_stream限定。
                model_kwargs["num_sectors"] = NUM_SECTORS
                model_kwargs["sector_embed_dim"] = SECTOR_EMBED_DIM
        # gru_only/transformer_onlyにはuse_cross_ffn/use_final_norm概念が無いので渡さない
        # (**kwargsを持つので万一渡しても無視されるが、意図を明確にするため分岐している)
        if MODEL_ARCH in ("dual_stream", "transformer_only"):
            model_kwargs["num_self_attn_layers"] = NUM_SELF_ATTN_LAYERS
    return arch_cls(**model_kwargs).to(DEVICE)


def reorder_dataset_by_tidx(data):
    """datasetをtidx昇順へ1回だけ並べ替える。

    PAIR_MODE="cross_sectional"では同一日サンプルを連続配置し、
    学習・検証バッチを連続sliceで取得できるようにする。
    同一tidx内の元順序はstable sortで維持する。
    """
    x_s, x_m, y, tid, tidx, regime = data

    order = np.argsort(tidx, kind="stable")

    return (
        np.ascontiguousarray(x_s[order]),
        np.ascontiguousarray(x_m[order]),
        np.ascontiguousarray(y[order]),
        np.ascontiguousarray(tid[order]),
        np.ascontiguousarray(tidx[order]),
        np.ascontiguousarray(regime[order]),
    )


def build_same_date_ranges(tidx_array, batch_size):
    """tidx順に並んだ配列から、同一日だけで構成される連続範囲を作る。

    戻り値:
        [(start, end), ...]

    endはexclusive。
    同一日サンプル数がbatch_sizeを超える場合は、その日付内で分割する。
    4件未満のチャンクは従来のSameDateBatchSamplerと同様に除外する。
    """
    tidx_array = np.asarray(tidx_array)

    if len(tidx_array) == 0:
        return []

    change_points = np.flatnonzero(tidx_array[1:] != tidx_array[:-1]) + 1

    boundaries = np.concatenate(
        [
            np.array([0], dtype=np.int64),
            change_points,
            np.array([len(tidx_array)], dtype=np.int64),
        ]
    )

    ranges = []

    for group_start, group_end in zip(
        boundaries[:-1],
        boundaries[1:],
    ):
        for start in range(
            int(group_start),
            int(group_end),
            batch_size,
        ):
            end = min(
                start + batch_size,
                int(group_end),
            )

            if end - start >= 4:
                ranges.append((start, end))

    return ranges


def validate_same_date_ranges(tidx_array, ranges, label):
    """連続範囲内が同一tidxだけで構成されることを検証する。"""
    for start, end in ranges:
        if end - start < 4:
            raise AssertionError(
                f"{label}: 4件未満のバッチがあります: " f"start={start}, end={end}"
            )

        batch_tidx = tidx_array[start:end]

        if not np.all(batch_tidx == batch_tidx[0]):
            raise AssertionError(
                f"{label}: 異なる日付が同一バッチへ混入しています: "
                f"start={start}, end={end}, "
                f"unique_tidx={np.unique(batch_tidx).tolist()}"
            )


def train_model(
    seed,
    tr_data,
    va_data,
    s_cols,
    m_cols,
    label,
    profile_this_call=False,
):
    set_seed(seed)

    # cross-sectionalモードでは、同じtidxの日付サンプルを最初に1回だけ
    # 連続配置する。これにより各epochでは日付バッチの順番だけをshuffleし、
    # バッチ取得自体は連続sliceで行える。
    #
    # same_tickerモードは従来のSameTickerBatchSamplerとindex_select方式を
    # そのまま維持する。
    if PAIR_MODE == "cross_sectional":
        tr_data = reorder_dataset_by_tidx(tr_data)
        va_data = reorder_dataset_by_tidx(va_data)
    (
        tr_x_s,
        tr_x_m,
        tr_y,
        tr_tid,
        tr_tidx,
        tr_regime,
    ) = tr_data
    (
        va_x_s,
        va_x_m,
        va_y,
        va_tid,
        va_tidx,
        va_regime,
    ) = va_data

    # Dataset/DataLoaderを経由せず、テンソル化した全データを1回だけGPUへ載せる。
    #
    # memmap入力の場合もtorch.tensor()によって通常のGPU Tensorへコピーされる。
    # read-only memmapをtorch.from_numpy()で共有せず、既存どおり安全なコピーを使う。
    tr_xs_t = torch.tensor(
        tr_x_s,
        dtype=torch.float32,
        device=DEVICE,
    )
    tr_xm_t = torch.tensor(
        tr_x_m,
        dtype=torch.float32,
        device=DEVICE,
    )
    tr_y_t = torch.tensor(
        tr_y,
        dtype=torch.float32,
        device=DEVICE,
    )
    tr_tidx_t = torch.tensor(
        tr_tidx,
        dtype=torch.long,
        device=DEVICE,
    )

    # USE_REGIME_AWARE_LOSS=Falseでも既存コードとの構造を単純に保つため
    # 一度GPUへ載せる。ただしepochごとの並べ替えは行わない。
    tr_regime_t = torch.tensor(
        tr_regime,
        dtype=torch.long,
        device=DEVICE,
    )

    va_xs_t = torch.tensor(
        va_x_s,
        dtype=torch.float32,
        device=DEVICE,
    )
    va_xm_t = torch.tensor(
        va_x_m,
        dtype=torch.float32,
        device=DEVICE,
    )
    va_y_t = torch.tensor(
        va_y,
        dtype=torch.float32,
        device=DEVICE,
    )
    va_tidx_t = torch.tensor(
        va_tidx,
        dtype=torch.long,
        device=DEVICE,
    )
    va_regime_t = torch.tensor(
        va_regime,
        dtype=torch.long,
        device=DEVICE,
    )

    if PAIR_MODE == "cross_sectional":
        # データは上でtidx順に並べ替え済みなので、同一日の各バッチは
        # (start, end)の連続範囲として表現できる。
        train_ranges = build_same_date_ranges(
            tr_tidx,
            TRAIN_BATCH_SIZE,
        )
        val_ranges = build_same_date_ranges(
            va_tidx,
            VAL_BATCH_SIZE,
        )

        # 実装上の前提を学習開始前に検証する。
        validate_same_date_ranges(
            tr_tidx,
            train_ranges,
            f"{label}/train",
        )
        validate_same_date_ranges(
            va_tidx,
            val_ranges,
            f"{label}/validation",
        )

        train_sampler = None
        val_batches = None

        log(
            f"  [{label}] cross-sectional連続バッチ: "
            f"train={len(train_ranges)} val={len(val_ranges)}"
        )
    else:
        # same_ticker側は従来方式を維持する。
        train_ranges = None
        val_ranges = None

        train_sampler = SameTickerBatchSampler(
            tr_tid,
            TRAIN_BATCH_SIZE,
            shuffle=True,
        )
        val_batches = list(
            SameTickerBatchSampler(
                va_tid,
                VAL_BATCH_SIZE,
                shuffle=False,
            )
        )

    model = build_model(s_cols, m_cols)

    # pair数加重・ゼロペアバッチ除外のため、regime制約の有無に関わらず
    # NearPairRankingLossRegimeAwareを使用する。
    #
    # この修正ではcriterion自体は変更していないため、旧実装と同じく
    # B×Bの両方向ペアを計算する。上三角化は別変更として検証する。
    criterion = NearPairRankingLossRegimeAware(
        max_gap=MAX_PAIR_GAP,
        cross_sectional=(PAIR_MODE == "cross_sectional"),
    )

    # CUDAの場合はfused AdamWを使用する。
    fused_adamw = DEVICE.type == "cuda"

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0002,
        weight_decay=3e-2,
        fused=fused_adamw,
    )

    amp_dtype = torch.bfloat16 if AMP_DTYPE == "bf16" else torch.float16
    amp_enabled = USE_AMP and DEVICE.type == "cuda"

    # fp16だけGradScalerを使う。bf16はfp32と同じ指数範囲なので使用しない。
    use_scaler = amp_enabled and AMP_DTYPE == "fp16"

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_scaler,
    )

    best_val_loss = float("inf")
    best_state = None
    patience_cnt = 0

    patience = 7
    epochs = 30

    for epoch in range(1, epochs + 1):
        do_profile = profile_this_call and epoch == 1

        prof = None

        if do_profile:
            activities = [ProfilerActivity.CPU]

            if torch.cuda.is_available():
                activities.append(ProfilerActivity.CUDA)

            prof = torch_profile(
                activities=activities,
            )
            prof.__enter__()

        model.train()

        # ------------------------------------------------------------
        # epochのバッチ順序を構築
        # ------------------------------------------------------------
        if PAIR_MODE == "cross_sectional":
            # データ本体はtidx順に固定配置したまま、日付バッチの処理順だけを
            # epochごとにshuffleする。
            #
            # 各バッチ内のサンプル順序は固定だが、ランキング損失はバッチ内の
            # 全ペア比較なので、サンプル順序自体はlossへ影響しない。
            epoch_batches = train_ranges.copy()
            random.shuffle(epoch_batches)

            # cross-sectional側では全件index_selectを実行しない。
            xs_shuf = None
            xm_shuf = None
            y_shuf = None
            tidx_shuf = None
            regime_shuf = None
        else:
            # same_ticker側は従来方式を維持する。
            #
            # SameTickerBatchSamplerが決めたバッチ構成を1本へ連結して
            # GPU上で全体を1回だけindex_selectし、個々のバッチは
            # 連続sliceで取り出す。
            epoch_batches = list(train_sampler)

            flat_idx = [
                sample_idx
                for batch_indices in epoch_batches
                for sample_idx in batch_indices
            ]

            flat_idx_t = torch.tensor(
                flat_idx,
                dtype=torch.long,
                device=DEVICE,
            )

            xs_shuf = tr_xs_t.index_select(
                0,
                flat_idx_t,
            )
            xm_shuf = tr_xm_t.index_select(
                0,
                flat_idx_t,
            )
            y_shuf = tr_y_t.index_select(
                0,
                flat_idx_t,
            )
            tidx_shuf = tr_tidx_t.index_select(
                0,
                flat_idx_t,
            )

            if USE_REGIME_AWARE_LOSS:
                regime_shuf = tr_regime_t.index_select(
                    0,
                    flat_idx_t,
                )
            else:
                regime_shuf = None

        # エポック全体のtrain_lossは、有効ペア数による加重平均。
        total_tr = torch.zeros(
            (),
            device=DEVICE,
        )
        total_tr_pairs = 0
        n_tr_batches = 0

        # same_ticker側で、連結済みTensorから各バッチを切り出す位置。
        offset = 0

        for batch_spec in epoch_batches:
            if PAIR_MODE == "cross_sectional":
                start, end = batch_spec

                # 全て連続sliceなので、コピー無しのviewとして取得できる。
                b_xs = tr_xs_t[start:end]
                b_xm = tr_xm_t[start:end]
                b_y = tr_y_t[start:end]
                b_tidx = tr_tidx_t[start:end]

                if USE_REGIME_AWARE_LOSS:
                    b_regime = tr_regime_t[start:end]
                else:
                    b_regime = None
            else:
                batch_indices = batch_spec
                batch_size = len(batch_indices)
                next_offset = offset + batch_size

                b_xs = xs_shuf[offset:next_offset]
                b_xm = xm_shuf[offset:next_offset]
                b_y = y_shuf[offset:next_offset]
                b_tidx = tidx_shuf[offset:next_offset]

                if USE_REGIME_AWARE_LOSS:
                    b_regime = regime_shuf[offset:next_offset]
                else:
                    b_regime = None

                offset = next_offset

            with torch.autocast(
                device_type=DEVICE.type,
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                logits = model(
                    b_xs,
                    b_xm,
                )

                loss, n_pairs = criterion(
                    logits,
                    b_y,
                    b_tidx,
                    b_regime,
                )

            # 有効ペアがないバッチはbackward、optimizer.step、
            # epoch平均の全てから除外する。
            if n_pairs == 0:
                continue

            optimizer.zero_grad(
                set_to_none=True,
            )

            if use_scaler:
                scaler.scale(loss).backward()

                scaler.unscale_(optimizer)

                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                )

                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()

                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                )

                optimizer.step()

            total_tr += loss.detach() * n_pairs
            total_tr_pairs += n_pairs
            n_tr_batches += 1

        if total_tr_pairs > 0:
            train_loss = (total_tr / total_tr_pairs).item()
        else:
            train_loss = float("inf")

        if do_profile:
            prof.__exit__(
                None,
                None,
                None,
            )

            sort_key = (
                "self_cuda_time_total"
                if torch.cuda.is_available()
                else "self_cpu_time_total"
            )

            log(
                f"[Profiler] [{label}] "
                f"epoch1 学習ループ"
                f"({n_tr_batches}バッチ)"
                f"処理時間トップ15:"
            )

            print(
                prof.key_averages().table(
                    sort_by=sort_key,
                    row_limit=15,
                )
            )

        # ------------------------------------------------------------
        # validation
        # ------------------------------------------------------------
        model.eval()

        total_va = torch.zeros(
            (),
            device=DEVICE,
        )
        total_va_pairs = 0
        n_batches = 0

        if PAIR_MODE == "cross_sectional":
            validation_iterator = val_ranges
        else:
            validation_iterator = val_batches

        with torch.no_grad():
            for batch_spec in validation_iterator:
                if PAIR_MODE == "cross_sectional":
                    start, end = batch_spec

                    # validation側も連続sliceで取得する。
                    b_xs = va_xs_t[start:end]
                    b_xm = va_xm_t[start:end]
                    b_y = va_y_t[start:end]
                    b_tidx = va_tidx_t[start:end]

                    if USE_REGIME_AWARE_LOSS:
                        b_regime = va_regime_t[start:end]
                    else:
                        b_regime = None
                else:
                    batch_idx = batch_spec

                    # same_ticker側は従来どおりfancy indexing。
                    b_xs = va_xs_t[batch_idx]
                    b_xm = va_xm_t[batch_idx]
                    b_y = va_y_t[batch_idx]
                    b_tidx = va_tidx_t[batch_idx]

                    if USE_REGIME_AWARE_LOSS:
                        b_regime = va_regime_t[batch_idx]
                    else:
                        b_regime = None

                with torch.autocast(
                    device_type=DEVICE.type,
                    dtype=amp_dtype,
                    enabled=amp_enabled,
                ):
                    logits = model(
                        b_xs,
                        b_xm,
                    )

                    loss, n_pairs = criterion(
                        logits,
                        b_y,
                        b_tidx,
                        b_regime,
                    )

                total_va += loss * n_pairs
                total_va_pairs += n_pairs
                n_batches += 1

        if n_batches == 0:
            log(
                f"  [{label}] "
                f"epoch {epoch:2d}/{epochs}: "
                f"[!] 警告 検証バッチが0件です"
                f"(regime_filter等でval側の"
                f"サンプルが無くなっている可能性)"
            )
        elif total_va_pairs == 0:
            log(
                f"  [{label}] "
                f"epoch {epoch:2d}/{epochs}: "
                f"[!] 警告 検証バッチは"
                f"{n_batches}件あるが"
                f"有効ペアが1つもありません"
            )

        if total_va_pairs > 0:
            val_loss = (total_va / total_va_pairs).item()
        else:
            val_loss = float("inf")

        if np.isnan(val_loss) or np.isnan(train_loss):
            log(
                f"  [{label}] "
                f"epoch {epoch:2d}/{epochs}: "
                f"[!] 警告 train_loss/val_lossに"
                f"NaNが出ています"
                f"(train_loss={train_loss}, "
                f"val_loss={val_loss})"
                f"。学習が発散している可能性"
            )

        is_best = val_loss < best_val_loss

        if is_best:
            best_val_loss = val_loss

            best_state = {
                key: value.clone() for key, value in model.state_dict().items()
            }

            patience_cnt = 0
            marker = " <- best"
        else:
            patience_cnt += 1
            marker = f" (patience " f"{patience_cnt}/{patience})"

        is_last = patience_cnt >= patience or epoch == epochs

        # new best、5epochごと、最終epochだけ出力する。
        if is_best or is_last or epoch % 5 == 0:
            log(
                f"  [{label}] "
                f"epoch {epoch:2d}/{epochs}: "
                f"train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f}"
                f"{marker}"
            )

        if patience_cnt >= patience:
            break

    if best_state is None:
        raise RuntimeError(
            f"[{label} seed={seed}] "
            f"学習全エポックでval_lossが"
            f"一度も改善しませんでした"
            f"(best_state=None)。"
            f"val側のサンプル数が0、または"
            f"val_lossが常にNaN/infの"
            f"可能性があります。"
            f"regime_filter/特徴量構成/"
            f"学習データ量を確認してください。"
        )

    model.load_state_dict(best_state)
    model.eval()

    log(f"[+] [{label} seed={seed}] " f"学習完了 " f"best_val_loss={best_val_loss:.4f}")

    return model


def predict1(model, w_s, w_m):
    """1件ずつ推論する旧実装。正しさの検証用に残してある(通常はrun_backtest_inference
    のバッチ推論を使うので、これを直接呼ぶ必要はない)。"""
    t_s = torch.tensor(w_s, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    t_m = torch.tensor(w_m, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(t_s, t_m), dim=-1).squeeze(0).cpu().numpy()
    return float(probs[2]), float(probs[0])


def run_backtest_inference(model, pool, batch_size=256):
    """poolを1件ずつではなくbatch_size件まとめてforwardする(1件ずつだとカーネル起動+
    GPU-CPU同期が多発し非常に遅い)。LayerNormのみでBatchNorm/Dropoutはeval()で無効
    なので、バッチをどう区切っても1件ずつ推論した場合と数値的に完全一致する。"""
    model.eval()
    records = []
    with torch.no_grad():
        for i in range(0, len(pool), batch_size):
            chunk = pool[i : i + batch_size]
            w_s_batch = np.stack([item[4] for item in chunk])
            w_m_batch = np.stack([item[5] for item in chunk])
            t_s = torch.tensor(w_s_batch, dtype=torch.float32).to(DEVICE)
            t_m = torch.tensor(w_m_batch, dtype=torch.float32).to(DEVICE)
            probs = torch.softmax(model(t_s, t_m), dim=-1).cpu().numpy()
            for j, (t, date, entry_date, exit_date, w_s, w_m, ret_pct) in enumerate(
                chunk
            ):
                p_win, p_stop = float(probs[j, 2]), float(probs[j, 0])
                # scoreはNearPairRankingLoss(RegimeAware)が直接最適化している量そのもの
                # (EV_WIN_WEIGHT*p_win-EV_STOP_WEIGHT*p_stop、本番のev_scoreと同じ2:1重み)。
                # p_win単体は絶対確率として校正されていない(win_rateとの平均ギャップ0.112・
                # 非単調を実測確認済み)ため、ランキング用にはscoreを使う。
                records.append(
                    {
                        "ticker": t,
                        "date": date,
                        "entry_date": entry_date,
                        "exit_date": exit_date,
                        "p_win": p_win,
                        "p_stop": p_stop,
                        "score": EV_WIN_WEIGHT * p_win - EV_STOP_WEIGHT * p_stop,
                        "ret_pct": ret_pct,
                    }
                )
    return pd.DataFrame(records)


def run_backtest_inference_ensemble(models, pool, batch_size=256):
    """run_backtest_inferenceの複数モデル版。本番のmodules/model_inference.py::
    predict_probabilities_ensemble_batchと同じロジック(各モデルのsoftmax確率を平均
    してからp_win/p_stopを取り出す)で、単一seedの偏り(特定regimeに偏ってSTRONG BUYを
    出す現象、実測確認済み)を緩和する。本番は5seedアンサンブルだが、ここはSEEDSの数だけ使う。
    scoreにはpredict_probabilities_ensemble_batchが返すev(=EV_WIN_WEIGHT*p_win-
    EV_STOP_WEIGHT*p_stopと同じ2:1重み、本番のev_scoreそのもの)をそのまま使う
    """
    for model in models:
        model.eval()
    records = []
    for i in range(0, len(pool), batch_size):
        chunk = pool[i : i + batch_size]
        w_s_batch = np.stack([item[4] for item in chunk])
        w_m_batch = np.stack([item[5] for item in chunk])
        p_win_arr, p_stop_arr, ev_arr = predict_probabilities_ensemble_batch(
            models, w_s_batch, w_m_batch
        )
        for j, (t, date, entry_date, exit_date, w_s, w_m, ret_pct) in enumerate(chunk):
            p_win, p_stop, ev = (
                float(p_win_arr[j]),
                float(p_stop_arr[j]),
                float(ev_arr[j]),
            )
            records.append(
                {
                    "ticker": t,
                    "date": date,
                    "entry_date": entry_date,
                    "exit_date": exit_date,
                    "p_win": p_win,
                    "p_stop": p_stop,
                    "score": ev,
                    "ret_pct": ret_pct,
                }
            )
    return pd.DataFrame(records)


if __name__ == "__main__":
    removed_from_baseline = (set(BASELINE_STOCK_COLS) | set(BASELINE_MACRO_COLS)) - set(
        TEST_STOCK_COLS + TEST_MACRO_COLS
    )
    added_candidates = set(TEST_STOCK_COLS + TEST_MACRO_COLS) - (
        set(BASELINE_STOCK_COLS) | set(BASELINE_MACRO_COLS)
    )
    print(
        f"[*] テスト側 株{len(TEST_STOCK_COLS)}個 + マクロ{len(TEST_MACRO_COLS)}個 = 計{len(TEST_FEATS)}個"
    )
    print(
        f"    既存から外した: {sorted(removed_from_baseline) if removed_from_baseline else '(なし)'}"
    )
    print(
        f"    新規追加した候補: {sorted(added_candidates) if added_candidates else '(なし)'}"
    )
    # archごとに実際に使われるパラメータだけを表示する(MODEL_ARCHに関わらず常に
    # dual_stream用トグルを表示すると、gru_macro_mlp_cross使用時に無関係な値を表示してしまう)。
    if MODEL_ARCH == "gru_macro_mlp_cross":
        print(
            f"[*] モデル設定: arch={MODEL_ARCH}, stock_hidden={STOCK_HIDDEN_DIM}, macro_hidden={MACRO_HIDDEN_DIM}, "
            f"num_heads={CROSS_ATTN_NUM_HEADS}, use_cross_ffn={CROSS_ATTN_USE_FFN}, "
            f"train_batch_size={TRAIN_BATCH_SIZE}, seeds={SEEDS}\n"
        )
    else:
        print(
            f"[*] モデル設定: arch={MODEL_ARCH}, hidden_dim={MODEL_HIDDEN_DIM}, num_heads={MODEL_NUM_HEADS}, "
            f"use_cross_ffn={USE_CROSS_FFN}, use_final_norm={USE_FINAL_NORM}, "
            f"num_self_attn_layers={NUM_SELF_ATTN_LAYERS}, train_batch_size={TRAIN_BATCH_SIZE}, seeds={SEEDS}\n"
        )

    with open(UNIVERSE_PATH, "r", encoding="utf-8") as f:
        tickers = [line.strip() for line in f if line.strip()]

    macro_df = load_macro_slim5(return_source="nk225")
    pool = get_full_feature_pool_df(tickers, macro_df)
    margin_pool = get_margin_pool_df(tickers, macro_df)
    sector_pool = get_sector_relative_pool_df(tickers, macro_df)
    macro_pool_df = augment_macro_pool_df(get_full_macro_pool_df(tickers))

    if REGIME_FILTER == "LOW":
        is_target = compute_is_low_regime_filter(macro_pool_df)
        log(
            f"[*] REGIME_FILTER=LOW: 全{len(is_target)}日中{int(is_target.sum())}日({is_target.mean()*100:.0f}%)"
            f"がLOWレジーム(拡大窓is_low、look-ahead無し)"
        )
        train_regime_filter = is_target
    elif REGIME_FILTER == "HIGH":
        is_target = compute_is_high_regime_filter(macro_pool_df)
        log(
            f"[*] REGIME_FILTER=HIGH: 全{len(is_target)}日中{int(is_target.sum())}日({is_target.mean()*100:.0f}%)"
            f"がHIGHレジーム(拡大窓is_high、look-ahead無し)"
        )
        train_regime_filter = is_target
    elif REGIME_FILTER == "MID":
        is_target = compute_is_mid_regime_filter(macro_pool_df)
        log(
            f"[*] REGIME_FILTER=MID: 全{len(is_target)}日中{int(is_target.sum())}日({is_target.mean()*100:.0f}%)"
            f"がMIDレジーム(拡大窓、look-ahead無し)"
        )
        train_regime_filter = is_target
    elif REGIME_FILTER is None:
        train_regime_filter = None
    else:
        raise ValueError(
            f"REGIME_FILTER='{REGIME_FILTER}'は未対応です(None / 'LOW' / 'MID' / 'HIGH'のみ)"
        )

    regime_labels = compute_regime_labels(macro_pool_df)
    if USE_REGIME_AWARE_LOSS:
        log(
            "[*] USE_REGIME_AWARE_LOSS=True: NearPairRankingLossのペアを同一regime同士に限定します"
            "(ペアタグ付けはlook-ahead無しの拡大窓判定を使用)"
        )

    # データセット・バックテストプールはシードに依存しないので、シードループの外で1回だけ構築する
    log("[*] データセット・バックテストプール構築(シード非依存、1回だけ)...")
    # regime_labels(全期間固定分位、報告用regime_breakdownのみで使う)とは別に、学習の
    # ペアタグ付けにはlook-ahead無しの拡大窓版を使う。
    loss_regime_labels = (
        compute_regime_labels_expanding(macro_pool_df)
        if USE_REGIME_AWARE_LOSS
        else None
    )

    # split dateはbase/testで完全に共通化する(build_dataset/prepare_backtest_poolが
    # それぞれ自分のvalid_datesから独立に計算すると、候補特徴量のNaNパターン次第で分割日が
    # 微妙に食い違いうるため)。ベースライン(既存14、常に固定)のvalid_datesを基準に
    # 1回だけ計算し、base/test両方に同じ値を渡す。
    _base_per_ticker = build_per_ticker(
        pool, margin_pool, sector_pool, macro_pool_df, BASELINE_STOCK_COLS
    )
    _base_valid_dates = valid_cross_section_dates(
        _base_per_ticker, BASELINE_STOCK_COLS, MIN_CROSS_SECTION
    )
    shared_split_date = compute_global_split_date(
        _base_valid_dates, train_regime_filter
    )
    log(f"[*] 共通split date(base/test): {shared_split_date.date()}")

    tr0, va0 = build_dataset(
        BASELINE_MACRO_COLS,
        BASELINE_STOCK_COLS,
        pool,
        margin_pool,
        sector_pool,
        macro_pool_df,
        regime_filter=train_regime_filter,
        regime_labels_for_loss=loss_regime_labels,
        global_split_date_override=shared_split_date,
    )
    tr1, va1 = build_dataset(
        TEST_MACRO_COLS,
        TEST_STOCK_COLS,
        pool,
        margin_pool,
        sector_pool,
        macro_pool_df,
        regime_filter=train_regime_filter,
        regime_labels_for_loss=loss_regime_labels,
        global_split_date_override=shared_split_date,
    )
    pool_base = prepare_backtest_pool(
        BASELINE_MACRO_COLS,
        BASELINE_STOCK_COLS,
        pool,
        margin_pool,
        sector_pool,
        macro_pool_df,
        regime_filter=train_regime_filter,
        global_split_date_override=shared_split_date,
    )
    pool_test = prepare_backtest_pool(
        TEST_MACRO_COLS,
        TEST_STOCK_COLS,
        pool,
        margin_pool,
        sector_pool,
        macro_pool_df,
        regime_filter=train_regime_filter,
        global_split_date_override=shared_split_date,
    )
    log(
        f"[+] train0={len(tr0[2])} val0={len(va0[2])} / train1={len(tr1[2])} val1={len(va1[2])}"
    )
    log(f"[+] backtest_pool base={len(pool_base)} test={len(pool_test)}")

    results_base, results_test = [], []
    models_base, models_test = [], []
    if PROFILE_FIRST_EPOCH:
        # プロファイリングは並列学習と相性が悪い(複数プロセスの出力が混ざる)ため、
        # PROFILE_FIRST_EPOCH=Trueのときだけ元の逐次ループにフォールバックする。
        for si, seed in enumerate(SEEDS):
            log(f"=== seed={seed} ===")
            model_base = train_model(
                seed,
                tr0,
                va0,
                BASELINE_STOCK_COLS,
                BASELINE_MACRO_COLS,
                "ベースライン(既存14固定)",
                profile_this_call=(si == 0),
            )
            d_base = run_backtest_inference(model_base, pool_base)
            model_test = train_model(
                seed,
                tr1,
                va1,
                TEST_STOCK_COLS,
                TEST_MACRO_COLS,
                f"テスト構成({len(TEST_FEATS)}個)",
            )
            d_test = run_backtest_inference(model_test, pool_test)

            r_base = evaluate(
                d_base,
                f"seed={seed} ベースライン score Top-K",
                min_reliable_n=MIN_RELIABLE_N,
                common_sample_k=COMMON_SAMPLE_K,
                max_concurrent=MAX_CONCURRENT_POSITIONS,
            )
            r_test = evaluate(
                d_test,
                f"seed={seed} テスト構成 score Top-K",
                min_reliable_n=MIN_RELIABLE_N,
                common_sample_k=COMMON_SAMPLE_K,
                max_concurrent=MAX_CONCURRENT_POSITIONS,
            )
            r_base["seed"] = seed
            r_test["seed"] = seed
            results_base.append(r_base)
            results_test.append(r_test)
            models_base.append(model_base)
            models_test.append(model_test)
            print()
    else:
        # base×seeds + test×seeds = 2*len(SEEDS)個の独立した学習ジョブを、1つのフラットな
        # jobリストとして並列学習する(parallel_seed_sweep.pyで検証済みの方式——3seed 2.44x、
        # 5seed 2.9x、逐次/並列で結果が完全一致、[[feedback_nn_training_performance]]参照)。
        import tempfile, shutil as _shutil, torch as _torch
        import parallel_seed_sweep as _pss

        # baselineモデルのキャッシュ。設定が前回と変わっていなければbaseline側の学習を
        # スキップしてキャッシュ済みstate_dictを再利用する(test側は毎回学習する)。
        _fp = baseline_cache_fingerprint(len(tr0[2]), len(va0[2]), shared_split_date)
        _cache_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "cache",
            "baseline_model_cache",
            _fp,
        )
        os.makedirs(_cache_dir, exist_ok=True)
        _cached_seeds = {
            s
            for s in SEEDS
            if os.path.exists(os.path.join(_cache_dir, f"state_{s}.pt"))
        }
        if _cached_seeds:
            log(
                f"[*] baselineキャッシュ命中(fingerprint={_fp}): seed={sorted(_cached_seeds)} は学習をスキップします"
            )

        _tmpdir = tempfile.mkdtemp(prefix="stage3_main_parallel_")
        _shared_data_path = os.path.join(_tmpdir, "shared_data.pkl")

        _pss.dump_shared_training_data(
            _shared_data_path,
            {
                "tr_base": tr0,
                "va_base": va0,
                "s_cols_base": BASELINE_STOCK_COLS,
                "m_cols_base": BASELINE_MACRO_COLS,
                "tr_test": tr1,
                "va_test": va1,
                "s_cols_test": TEST_STOCK_COLS,
                "m_cols_test": TEST_MACRO_COLS,
            },
        )

        log(
            f"[+] worker共有データをmemmap化しました: "
            f"{os.path.join(_tmpdir, 'shared_arrays')}"
        )

        _jobs = []
        _out_paths = {}
        for seed in SEEDS:
            for label in ("base", "test"):
                if label == "base" and seed in _cached_seeds:
                    _out_paths[(label, seed)] = os.path.join(
                        _cache_dir, f"state_{seed}.pt"
                    )
                    continue
                out_path = os.path.join(_tmpdir, f"state_{label}_{seed}.pt")
                _jobs.append((label, seed, _shared_data_path, out_path))
                _out_paths[(label, seed)] = out_path

        log(
            f"[*] base×{len(SEEDS) - len(_cached_seeds)}seed(新規)+ test×{len(SEEDS)}seed = {len(_jobs)}ジョブを"
            f"並列学習します(最大{PARALLEL_MAX_CONCURRENT}並列)..."
        )
        _t0 = time.time()
        _pss.run_jobs_parallel(
            _jobs, os.path.abspath(_pss.__file__), PARALLEL_MAX_CONCURRENT
        )
        log(f"[+] 全{len(_jobs)}ジョブ学習完了: 実時間={time.time()-_t0:.1f}秒")

        # 新規学習したbaselineモデルをキャッシュに保存(次回以降の再利用のため)
        for seed in SEEDS:
            if seed not in _cached_seeds:
                _shutil.copy2(
                    _out_paths[("base", seed)],
                    os.path.join(_cache_dir, f"state_{seed}.pt"),
                )

        for seed in SEEDS:
            log(f"=== seed={seed}(推論・評価) ===")
            model_base = build_model(BASELINE_STOCK_COLS, BASELINE_MACRO_COLS)
            model_base.load_state_dict(
                _torch.load(_out_paths[("base", seed)], map_location=DEVICE)
            )
            model_base.eval()
            d_base = run_backtest_inference(model_base, pool_base)

            model_test = build_model(TEST_STOCK_COLS, TEST_MACRO_COLS)
            model_test.load_state_dict(
                _torch.load(_out_paths[("test", seed)], map_location=DEVICE)
            )
            model_test.eval()
            d_test = run_backtest_inference(model_test, pool_test)

            r_base = evaluate(
                d_base,
                f"seed={seed} ベースライン score Top-K",
                min_reliable_n=MIN_RELIABLE_N,
                common_sample_k=COMMON_SAMPLE_K,
                max_concurrent=MAX_CONCURRENT_POSITIONS,
            )
            r_test = evaluate(
                d_test,
                f"seed={seed} テスト構成 score Top-K",
                min_reliable_n=MIN_RELIABLE_N,
                common_sample_k=COMMON_SAMPLE_K,
                max_concurrent=MAX_CONCURRENT_POSITIONS,
            )
            r_base["seed"] = seed
            r_test["seed"] = seed
            results_base.append(r_base)
            results_test.append(r_test)
            models_base.append(model_base)
            models_test.append(model_test)
            print()

        try:
            import shutil

            shutil.rmtree(_tmpdir)
        except Exception as e:
            log(
                f"[!] 一時memmapディレクトリを削除できませんでした: "
                f"{_tmpdir} ({type(e).__name__}: {e})"
            )

    # アンサンブル評価: 単一seedはSTRONG BUYが特定regimeに偏る不安定性があるため、
    # 本番と同じsoftmax確率平均アンサンブル(predict_probabilities_ensemble_batch)での
    # 評価も追加する。SEEDS=1個でも単一seedと同じ結果になるだけなのでスキップしない。
    print("=" * 90)
    print(
        f"【アンサンブル評価({len(SEEDS)}seed平均、本番run_dynamic_regime_screening_v8.pyと同じロジック)】"
    )
    print("=" * 90)
    d_base_ens = run_backtest_inference_ensemble(models_base, pool_base)
    d_test_ens = run_backtest_inference_ensemble(models_test, pool_test)
    r_base_ens = evaluate(
        d_base_ens,
        f"ベースライン(アンサンブル{len(SEEDS)}seed) score Top-K",
        min_reliable_n=MIN_RELIABLE_N,
        common_sample_k=COMMON_SAMPLE_K,
        max_concurrent=MAX_CONCURRENT_POSITIONS,
    )
    r_test_ens = evaluate(
        d_test_ens,
        f"テスト構成(アンサンブル{len(SEEDS)}seed) score Top-K",
        min_reliable_n=MIN_RELIABLE_N,
        common_sample_k=COMMON_SAMPLE_K,
        max_concurrent=MAX_CONCURRENT_POSITIONS,
    )
    # score(=EV_WIN_WEIGHT*p_win-EV_STOP_WEIGHT*p_stop、本番のev_scoreと同じ2:1重み)
    # 上位K件で選ぶ。決定論的tie-break(evaluate()と同じ基準: score降順→ticker昇順→date昇順)
    sub_base_ens = d_base_ens.sort_values(
        ["score", "ticker", "date"], ascending=[False, True, True]
    ).head(min(COMMON_SAMPLE_K, len(d_base_ens)))
    sub_test_ens = d_test_ens.sort_values(
        ["score", "ticker", "date"], ascending=[False, True, True]
    ).head(min(COMMON_SAMPLE_K, len(d_test_ens)))
    if len(sub_base_ens) > 0:
        regime_breakdown(
            sub_base_ens,
            "ベースライン(アンサンブル)",
            regime_labels,
            min_reliable_n=MIN_RELIABLE_N,
        )
    if len(sub_test_ens) > 0:
        regime_breakdown(
            sub_test_ens,
            "テスト構成(アンサンブル)",
            regime_labels,
            min_reliable_n=MIN_RELIABLE_N,
        )
    print(
        f"\n  [判定・アンサンブル] baseline PF={r_base_ens['pf']:.2f} vs テスト構成 PF={r_test_ens['pf']:.2f} -> "
        f"{'テスト構成の方が良い' if r_test_ens['pf'] > r_base_ens['pf'] else 'ベースラインの方が良い'}"
        f"(単一seed毎の判定より、こちらの方が実運用のアンサンブル推論に近い)"
    )
    print()

    # 日次Top-N運用評価。全期間score上位K件は特定の数日にシグナルが偏っていても検知できない
    # ため、実際の運用(毎日DAILY_TOPN件しか執行できない)に忠実な評価をアンサンブル予測に追加。
    print("=" * 90)
    print(
        f"【日次Top-N運用評価(1日あたり{DAILY_TOPN}件、アンサンブル{len(SEEDS)}seed)】"
    )
    print("=" * 90)
    r_base_daily = evaluate_daily_topn(
        d_base_ens,
        f"ベースライン(日次Top-{DAILY_TOPN})",
        n_per_day=DAILY_TOPN,
        min_reliable_n=MIN_RELIABLE_N,
        max_concurrent=MAX_CONCURRENT_POSITIONS,
    )
    r_test_daily = evaluate_daily_topn(
        d_test_ens,
        f"テスト構成(日次Top-{DAILY_TOPN})",
        n_per_day=DAILY_TOPN,
        min_reliable_n=MIN_RELIABLE_N,
        max_concurrent=MAX_CONCURRENT_POSITIONS,
    )
    print(
        f"\n  [判定・日次Top-N] baseline PF={r_base_daily['pf']:.2f} vs テスト構成 PF={r_test_daily['pf']:.2f} -> "
        f"{'テスト構成の方が良い' if r_test_daily['pf'] > r_base_daily['pf'] else 'ベースラインの方が良い'}"
        f"(全期間Top-Kより日々の執行に忠実。全期間Top-Kの判定と食い違う場合、全期間Top-Kの"
        f"結果が特定の数日に偏っていた可能性が高い)"
    )
    print()

    # base/testで特徴量セットが違うとdropnaで落ちる(ticker,date)が微妙に異なり得るため、
    # 両方に共通するサンプルだけに絞った比較も出す(「モデルの質の差」と「評価サンプル自体の
    # 差」が混ざるのを避けるため)。
    key_base = set(zip(d_base_ens["ticker"], d_base_ens["date"]))
    key_test = set(zip(d_test_ens["ticker"], d_test_ens["date"]))
    common_keys = key_base & key_test
    log(
        f"[*] base/testの(ticker,date)集合: base={len(key_base)} test={len(key_test)} common={len(common_keys)} "
        f"(baseのみ={len(key_base - key_test)}件, testのみ={len(key_test - key_base)}件)"
    )
    if len(common_keys) > 0:
        mask_base_common = [
            k in common_keys for k in zip(d_base_ens["ticker"], d_base_ens["date"])
        ]
        mask_test_common = [
            k in common_keys for k in zip(d_test_ens["ticker"], d_test_ens["date"])
        ]
        print("=" * 90)
        print(
            f"【base/test共通(ticker,date)サブセットでの比較(アンサンブル、共通n={len(common_keys)}件)】"
        )
        print("=" * 90)
        r_base_common = evaluate(
            d_base_ens[mask_base_common],
            "ベースライン(共通サブセット)",
            min_reliable_n=MIN_RELIABLE_N,
            common_sample_k=COMMON_SAMPLE_K,
            max_concurrent=MAX_CONCURRENT_POSITIONS,
        )
        r_test_common = evaluate(
            d_test_ens[mask_test_common],
            "テスト構成(共通サブセット)",
            min_reliable_n=MIN_RELIABLE_N,
            common_sample_k=COMMON_SAMPLE_K,
            max_concurrent=MAX_CONCURRENT_POSITIONS,
        )
        print(
            f"\n  [判定・共通サブセット] baseline PF={r_base_common['pf']:.2f} vs テスト構成 PF={r_test_common['pf']:.2f} -> "
            f"{'テスト構成の方が良い' if r_test_common['pf'] > r_base_common['pf'] else 'ベースラインの方が良い'}"
            f"(特徴量差によるdropnaパターンの違いを排除した、最も公平な比較)"
        )
        print()

    # 異なるモデル(seed)の生予測を単純にpd.concatして1つの大きなサンプルとして扱うのは
    # 統計的に妥当でないため行わない。regimeごとの内訳は上のアンサンブル評価
    # (複数モデルの確率平均、本番と同じ手法)の regime_breakdown を使う。

    df_base = pd.DataFrame(results_base)
    df_test = pd.DataFrame(results_test)

    print("=" * 90)
    print("【シード別サマリ(標準指標)】")
    print("=" * 90)
    cols_main = ["seed", "n", "win_rate", "pf", "avg_ret", "max_dd", "few_trades_flag"]
    print("  -- ベースライン(既存14固定) --")
    print(df_base[cols_main].to_string(index=False))
    print("  -- テスト構成 --")
    print(df_test[cols_main].to_string(index=False))

    print("\n" + "=" * 90)
    print("【シード別サマリ(top_k_pf・月別一貫性・銘柄集中度)】")
    print("=" * 90)
    cols_extra = [
        "seed",
        "top_k_pf",
        "top_k_n",
        "month_consistency",
        "top1_pct",
        "top5_pct",
        "hhi",
    ]
    print("  -- ベースライン --")
    print(df_base[cols_extra].to_string(index=False))
    print("  -- テスト構成 --")
    print(df_test[cols_extra].to_string(index=False))

    print("\n" + "=" * 90)
    print("【seedごとのPF差分(テスト構成 - ベースライン)】")
    print("=" * 90)
    diff_df = pd.DataFrame(
        {
            "seed": df_base["seed"],
            "pf_base": df_base["pf"],
            "pf_test": df_test["pf"],
            "pf_diff": df_test["pf"] - df_base["pf"],
            "top_k_pf_base": df_base["top_k_pf"],
            "top_k_pf_test": df_test["top_k_pf"],
            "top_k_pf_diff": df_test["top_k_pf"] - df_base["top_k_pf"],
        }
    )
    print(diff_df.to_string(index=False))
    n_seeds_test_better = (diff_df["pf_diff"] > 0).sum()
    print(
        f"\n  テスト構成がベースラインを上回ったseed数: {n_seeds_test_better}/{len(SEEDS)}"
        f"(約定ベースPF基準) | {(diff_df['top_k_pf_diff'] > 0).sum()}/{len(SEEDS)}(top_k_pf基準)"
    )

    print("\n" + "=" * 90)
    print("【mean ± std(組み合わせごとのシードばらつき比較用)】")
    print("=" * 90)
    finite_base_pf = df_base["pf"][np.isfinite(df_base["pf"])]
    finite_test_pf = df_test["pf"][np.isfinite(df_test["pf"])]
    print(
        f"  ベースライン PF: mean={finite_base_pf.mean():.3f} std={finite_base_pf.std():.3f} "
        f"(seed毎: {[round(x, 2) for x in df_base['pf']]})"
    )
    print(
        f"  テスト構成   PF: mean={finite_test_pf.mean():.3f} std={finite_test_pf.std():.3f} "
        f"(seed毎: {[round(x, 2) for x in df_test['pf']]})"
    )
    print(
        f"  ベースライン win_rate: mean={df_base['win_rate'].mean():.1f}% std={df_base['win_rate'].std():.1f}%"
    )
    print(
        f"  テスト構成   win_rate: mean={df_test['win_rate'].mean():.1f}% std={df_test['win_rate'].std():.1f}%"
    )
    print(
        f"  ベースライン MaxDD: mean={df_base['max_dd'].mean()*100:.1f}% std={df_base['max_dd'].std()*100:.1f}%"
    )
    print(
        f"  テスト構成   MaxDD: mean={df_test['max_dd'].mean()*100:.1f}% std={df_test['max_dd'].std()*100:.1f}%"
    )

    n_few_base = df_base["few_trades_flag"].sum()
    n_few_test = df_test["few_trades_flag"].sum()
    if n_few_base or n_few_test:
        print(
            f"\n  [!] 少数トレード警告(n<{MIN_RELIABLE_N}件): "
            f"ベースライン {n_few_base}/{len(SEEDS)}シード、テスト構成 {n_few_test}/{len(SEEDS)}シード"
            f" — 該当シードのPFは信頼性が低い可能性"
        )

    print("\n" + "=" * 90)
    print(
        f"【判定・単一seed平均】baseline PF mean={finite_base_pf.mean():.2f} vs テスト構成 PF mean={finite_test_pf.mean():.2f} -> "
        f"{'テスト構成の方が良い' if finite_test_pf.mean() > finite_base_pf.mean() else 'ベースラインの方が良い(このテスト構成は不採用推奨)'}"
    )
    print(
        f"    std比較: テスト構成のstdが{'大きい(不安定化)' if finite_test_pf.std() > finite_base_pf.std() else '小さい(安定化)'} "
        f"(baseline std={finite_base_pf.std():.3f} vs test std={finite_test_pf.std():.3f})"
    )
    print(
        f"【判定・アンサンブル(実運用に近い)】baseline PF={r_base_ens['pf']:.2f} vs テスト構成 PF={r_test_ens['pf']:.2f} -> "
        f"{'テスト構成の方が良い' if r_test_ens['pf'] > r_base_ens['pf'] else 'ベースラインの方が良い'}"
        f" (単一seed平均とアンサンブルの判定が食い違う場合は、単一seedのばらつき自体が"
        f"regime依存の可能性があるため、アンサンブル側を優先すること)"
    )
    print("=" * 90)
