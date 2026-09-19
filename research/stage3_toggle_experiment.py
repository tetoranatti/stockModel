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
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, r"F:\stockModel")
sys.path.insert(0, r"F:\stockModel\research")

try:
    # バックグラウンド実行時、ログファイルへの書き出しがブロックバッファリングで遅延するのを防ぐ。
    # encoding='utf-8'も明示しないと、リダイレクト時のPythonはコンソールのcp932にfall backし、
    # em dash等cp932非対応文字でUnicodeEncodeErrorが起きる(2026-09-19、実際に発生・修正)。
    sys.stdout.reconfigure(line_buffering=True, encoding='utf-8')
    sys.stderr.reconfigure(line_buffering=True, encoding='utf-8')
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
    UniverseDataset, SameTickerBatchSampler,
    MAX_PAIR_GAP, UNIVERSE_PATH, MIN_CROSS_SECTION,
)
# 研究ハーネス側の既定を本番の20から5へ変更(2026-09-19採用、ユーザー提案で3seed×3値
# (5/10/20)を比較した結果、gapを狭めるほどPF・seed間の安定性ともに改善したため
# —— gap=5: mean=1.445/std=0.291/ensemble=1.244、gap=20(本番既定): mean=1.238/
# std=0.377/ensemble=1.142。本番training/train_model_v8_exp.pyのMAX_PAIR_GAPは
# 意図的に変更していない(研究ハーネスのみ)。クラス定義(NearPairRankingLossRegimeAware
# のデフォルト引数)より前でこの代入を行う必要がある。
MAX_PAIR_GAP = 5
from modules.model_arch import DualStream_GRU_PreLN_Transformer, GRUOnlyModel, TransformerOnlyModel, GRU_MacroMLP_CrossAttn_Model
from torch.profiler import profile as torch_profile, ProfilerActivity
from modules.macro_features import load_macro_slim5
from modules.stock_features import STOCK_FEATURE_COLS
from modules.cross_sectional_features import (
    apply_log_transform, compute_cross_sectional_stats, valid_cross_section_dates, normalize_cross_sectional,
)
from _feature_cache_utils import (
    get_full_feature_pool_df, get_full_macro_pool_df, get_margin_pool_df,
    get_sector_relative_pool_df,
)
from modules.regime_risk_model import _compute_vol_regime_is_low
from modules.model_inference import predict_probabilities_ensemble_batch
from _eval_utils import (
    REGIME_ID_MAP, pf_of, compute_max_drawdown, compute_concentration,
    compute_is_low_regime_filter, compute_is_high_regime_filter, compute_is_mid_regime_filter,
    compute_regime_labels, compute_regime_labels_expanding, regime_breakdown, monthly_breakdown,
    evaluate, evaluate_daily_topn,
)


class NearPairRankingLossFast(nn.Module):
    """NearPairRankingLossのO(B×max_gap)版(2026-09-18、Profilerでaten::_index_put_impl_が
    目立ったため追加)。本番のO(B²)版(全B×Bペアを作ってからmax_gap以内をマスク)と
    数値的に厳密に同値——SameTickerBatchSampler+build_datasetの構築により、各バッチ内は
    同一銘柄・tidx昇順で並ぶことが保証されているため、配列オフセットδ=1..max_gapだけ見れば
    |tidx_i-tidx_j|<=max_gapを満たす全ペアを漏れなく拾える。全ペア行列を作らずtorch.catで
    済ませることで、O(B²)のメモリ確保・2Dマスクインデックスを回避する。
    tidxが昇順という前提が崩れる呼び出し方をすると結果が変わるので注意(通常の
    SameTickerBatchSampler経由なら常に成立する)。"""
    def __init__(self, max_gap=MAX_PAIR_GAP):
        super().__init__()
        self.max_gap = max_gap

    def forward(self, logits, ret_pct, tidx):
        probs = torch.softmax(logits, dim=-1)
        score = probs[:, 2] - probs[:, 0]
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
    """NearPairRankingLossに「同一regime(LOW/MID/HIGH)のペアだけ比較する」制約を追加した版
    (2026-09-19、ユーザー提案「ペアワイズランキングがレジームの変化に合わないってことある？」
    の検証用)。同一銘柄・近傍日(|tidx_i-tidx_j|<=max_gap)ペアのうち、日付ベースの実測で
    42.7%がLOW/MID/HIGHのregimeを跨いでいた——つまり学習ペアの半分近くが「別の相場環境の
    日同士」を比較させられており、regime由来のリターン格差が銘柄固有のシグナルを汚染している
    可能性がある。regime引数(int、-1は「対象外/不明」で常に除外、Noneならregime制約無しで
    本番NearPairRankingLossと数値的に同一)。

    forward()は(loss_mean, n_pairs)のタプルを返す(2026-09-19追加、ユーザー指摘「有効ペア数
    加重」「ゼロペアバッチ除外」)。バッチごとに有効ペア数が大きく変わりうる
    (特にregime制約を付けると、あるバッチは同一regimeのサンプルばかりで多い/別バッチは
    regimeが割れていて少ない、ということが起きる)。呼び出し側でn_pairsを使って
    エポック全体の平均をペア数加重にし、n_pairs=0のバッチ(有効ペアが無い=情報を持たない)
    を平均から自然に除外できるようにする。"""
    def __init__(self, max_gap=MAX_PAIR_GAP):
        super().__init__()
        self.max_gap = max_gap

    def forward(self, logits, ret_pct, tidx, regime=None):
        probs = torch.softmax(logits, dim=-1)
        score = probs[:, 2] - probs[:, 0]
        diff_score = score.unsqueeze(1) - score.unsqueeze(0)
        diff_ret = ret_pct.unsqueeze(1) - ret_pct.unsqueeze(0)
        sign = torch.sign(diff_ret)
        gap = (tidx.unsqueeze(1) - tidx.unsqueeze(0)).abs()
        mask = (sign != 0) & (gap > 0) & (gap <= self.max_gap)
        if regime is not None:
            same_regime = (regime.unsqueeze(1) == regime.unsqueeze(0)) & (regime.unsqueeze(1) >= 0)
            mask = mask & same_regime
        n_pairs = int(mask.sum().item())
        if n_pairs == 0:
            return diff_score.sum() * 0.0, 0
        return torch.nn.functional.softplus(-sign[mask] * diff_score[mask]).mean(), n_pairs


BASE_MACRO_COLS = ['pin_dist_ratio', 'wall_spread', 'cta_net_norm', 'cta_momentum', 'nk_ret_norm']

# ============================================================================
# ここを編集して実験する。既存14もOn/Off可能(Falseにすれば「既存から減らす」実験になる)。
# ============================================================================
FEATURE_TOGGLES = {
    # --- 既存9(株側、本番) ---
    'stock_ret_1d': True, 'stock_ret_5d': True, 'stock_ret_20d': True,
    'atr_ratio': True, 'rolling_beta': True, 'vol_ratio_5d': True,
    'overnight_gap': True, 'dist_from_high20': True, 'dist_from_low20': True,
    # --- 既存5(マクロ側、本番) ---
    'pin_dist_ratio': True, 'wall_spread': True, 'cta_net_norm': True, 'cta_momentum': True, 'nk_ret_norm': True,

    # --- 株側候補(6項目スクリーニングIC通過) ---
    'rs60': True,
    'gap_strength_5': True,
    'atr_accel': True,
    # --- 株側候補(全体では不合格だがstock_feature_scorecard.csvでbest_regime=LOW) ---
    'atr_term_ratio': False,
    'dist_from_high60': False,
    'gap_avg_5d': False,
    'relative_strength_5d': False,  # fail_redundant(既存と相関0.68)のため除外
    'rs20': False,                  # fail_redundant(既存と相関0.74)のため除外
    'momentum_accel': False,
    'volume_zscore': False,
    'volume_accel': False,
    'volume_ma_ratio': False,
    'close_location_value': False,
    'body_ratio': False,
    'adx14': False,
    'efficiency_ratio20': False,
    'days_since_high20': False,
    'new_high20': False,
    'positive_gap_ratio_5': False,
    'gap_follow_through': False,
    # --- マクロ候補(usdjpy_chgのみIC通過) ---
    'usdjpy_chg': True,
    'market_vol_regime': False,
    'vix_overnight_chg': False,
    'nk225_ret_5d': False,
    'topix_ret_5d': False,
    # --- マージン残高候補(margin_ratio_levelのみIC通過) ---
    'margin_ratio_level': True,
    'margin_ratio_zscore_12w': False,
    'margin_buy_chg_1w': False,
    'margin_short_chg_1w': False,
    # --- セクター相対候補(全不合格) ---
    'sector_rel_ret_5d': False,
    'sector_rel_ret_20d': False,
    'sector_rel_vol_ratio_5d': False,
    'sector_rank_ret_5d': False,
    # --- SHAP交互作用候補 ---
    'rolling_beta_x_cta_net_norm': False,
    'cta_net_norm_x_market_vol_regime': False,
    'pin_dist_ratio_x_cta_net_norm': False,
    'rolling_beta_x_cta_net_norm_x_market_vol_regime': False,
    # --- LOWボラ限定候補(regime_risk_modelの拡大窓is_low版、look-ahead無し) ---
    'cta_net_norm_x_low_v2': False,
}

MODEL_ARCH = "gru_macro_mlp_cross"  # "dual_stream"(本番) / "gru_only" / "transformer_only" / "gru_macro_mlp_cross"
MODEL_HIDDEN_DIM = 20   # 20:本番と同じ既定値。超軽量版を試すならここを変える(num_headsで割り切れる値に)
MODEL_NUM_HEADS = 1     # 1:本番と同じ既定値。gru_onlyでは無視される
STOCK_HIDDEN_DIM = 32   # "gru_macro_mlp_cross"専用: 株側GRUのhidden_dim(2026-09-19追加、ユーザー提案)
MACRO_HIDDEN_DIM = 16   # "gru_macro_mlp_cross"専用: マクロ側MLPのhidden_dim(株の半分、ユーザー指摘
                         # 「隠れ層はもっと小さくていい」を反映し当初案64/32から縮小)
CROSS_ATTN_NUM_HEADS = 4  # "gru_macro_mlp_cross"専用: Cross-Attentionのhead数(STOCK_HIDDEN_DIMで割り切れる値に)
CROSS_ATTN_USE_FFN = False  # "gru_macro_mlp_cross"専用: Cross-Attention後にFFNサブレイヤーを追加するか
                             # (dual_streamのUSE_CROSS_FFNで効果が確認済みのため試す、2026-09-19追加)
USE_CROSS_FFN = True    # Cross-Attention後にFFNサブレイヤーを追加するか(dual_stream限定、2026-09-19に本番採用済み)
USE_FINAL_NORM = False  # Cross-Attention後・pooling前に最終LayerNormを追加するか(dual_stream限定、既定False=本番と同じ)
NUM_SELF_ATTN_LAYERS = 1  # Self-Attention(stock_encoder/macro_encoder)の層数。dual_stream/transformer_only
                           # 両対応(gru_onlyでは無視される)。既定1=本番と同じ(1層のみ)
SEEDS = [42, 43, 44]    # 複数シードでmean/stdを見る(組み合わせによってシード間のばらつきが
                        # 変わるか比較したい場合はここを増減する。単発でよければ[42]だけにする)
PROFILE_FIRST_EPOCH = False  # Trueにすると、最初のシードのベースライン学習の epoch=1 だけ
                             # torch.profilerで計測し、処理時間トップ10を表示する(それ以外は
                             # 計測オーバーヘッド無し。データ準備 vs 学習 vs 検証のどこが重いか
                             # 見たい時だけTrueにする)
USE_AMP = False        # AMP(自動混合精度)を使うか。他の最適化と違い数値は厳密一致しない
                        # (fp16/bf16の丸め誤差が入る)ので、精度への影響を見てから使うこと
AMP_DTYPE = "bf16"     # "bf16"(RTX 3080はネイティブ対応、GradScaler不要で安全) or "fp16"
VAL_BATCH_SIZE = 1024  # 検証はbackward無しでメモリ圧が低いため、学習batch(128)より大きくできる。
                        # 実測(1銘柄400サンプル規模)で256→512は約2倍高速化、512以降は頭打ち。
                        # 1024にして余裕を持たせてある。ただしNearPairRankingLossはバッチ内
                        # ペアのみ比較するため、チャンクの切れ目が変わるとval_lossの値も
                        # 僅かに変わる(小数点4桁目程度、early stoppingへの実用上の影響は
                        # ほぼ無いはず。他の最適化と違い厳密には不変ではない点に注意)。
TRAIN_BATCH_SIZE = 256  # 学習batch。速度は256/512でほぼ線形にスケールするが、勾配のノイズが
                        # 減るため収束の質(best_val_loss・バックテストPF)が変わりうる。
                        # 採用前に品質を実際に検証すること(verify_train_batch_size.py参照)。
                        # ⚠️ 512は実際に試したところ複数シードでNaN化・学習停滞を確認(2026-09-19)。
                        # 256は問題なし。512以上は避けること。
COMMON_SAMPLE_K = 200      # 「共通評価サンプル上のPF」: p_win上位k件(seedごとの候補数nに
                           # 依存しない固定件数)でPFを見る。STRONG BUY閾値だとseedによって
                           # n=0〜数千件までばらつき公平に比較できないための補助指標。
MIN_RELIABLE_N = 30        # STRONG BUY件数がこれ未満なら「少数トレードへの偏り」警告を出す
MAX_CONCURRENT_POSITIONS = 20  # MaxDD計算用の資金曲線シミュレーション、同時保有上限(均等配分)
DAILY_TOPN = 5  # 日次Top-N運用評価(evaluate_daily_topn)で1日あたり何件選ぶか(2026-09-19追加、
                # ユーザー指摘「全期間Top-Kに加えて日次Top-Nを主運用評価として追加」)。
                # evaluate()の全期間score上位K件は特定の数日にシグナルが偏っていても検知できない
                # ——実際の運用(毎日決まった件数しか執行できない)により忠実な評価として追加。
REGIME_FILTER = None  # None(全期間、既定) / "LOW" / "HIGH": 学習・バックテスト両方を
                        # そのボラティリティレジーム(拡大窓is_low/is_high、look-ahead無し)の
                        # 日だけに絞る(2026-09-19追加、レジーム別モデル用)。ベースライン・
                        # テスト構成の両方に同じフィルタをかけるので、比較の公平性は保たれる。
ENTRY_CONVENTION = "d1_open"  # "d1_close"(D+1引け/MOC、本番の確立済み執行規約) /
                        # "d1_open"(既定、D+1始値/MOO、2026-09-19追加・採用。5seed比較で
                        # 日次Top-5 PFがアンサンブルで0.927→1.201に改善したため、研究
                        # ハーネス側の既定として維持する。本番の執行規約自体はまだ変更していない
                        # ——寄成注文の約定安定性を運用面で検証してから判断すべき、ユーザー
                        # 提案「D+1引けという執行規約自体を見直す」)。学習ターゲット・
                        # バックテスト両方に同じ規約が使われる(simulate_ret_pct_d1_close経由で共有)。
LABEL_MODE = "continuous"  # 学習ターゲット(tr_y/va_y)の定義。バックテスト側(prepare_backtest_pool)
                        # は常に連続値ret_pct(ATRバリア方式)のまま——ここは学習ターゲットのみ
                        # 変える(2026-09-19追加、ユーザー提案「ラベル閾値を分位点/固定閾値ラベルで
                        # 試す」)。選択肢:
                        #   "continuous": 既定。simulate_ret_pct_d1_closeの連続値ret_pctをそのまま使う。
                        #   "cross_sectional_quantile": 同日全銘柄横断でret_pctを分位点化した3値
                        #     (下位QUANTILE_LABEL_DOWN%=0/残り=1/上位QUANTILE_LABEL_UP%=2)。
                        #     NearPairRankingLossは同一銘柄・近傍日のペア比較なので、日によって
                        #     市場全体が上げ下げしているとret_pctの差に銘柄選択スキルとmarket-beta
                        #     方向が混ざってしまう懸念に対応する狙い。
                        #   "ret5_fixed_threshold": ATRバリア無しの単純な固定期間(RET5_HOLDING_DAYS
                        #     日後)リターンを、固定の絶対閾値(RET5_DOWN_THRESH/RET5_UP_THRESH)で
                        #     3値化する。
                        # いずれのモードも、同じbin同士のペアはdiff=0となりNearPairRankingLossから
                        # 自動的に除外される(有効ペア数は減る)。
QUANTILE_LABEL_UP = 0.30    # "cross_sectional_quantile"専用: 上位何%をUp(離散値2)とするか
QUANTILE_LABEL_DOWN = 0.30  # "cross_sectional_quantile"専用: 下位何%をDown(離散値0)とするか(残りはHold=1)
RET5_HOLDING_DAYS = 5       # "ret5_fixed_threshold"専用: 固定保有日数
RET5_DOWN_THRESH = -0.03    # "ret5_fixed_threshold"専用: この値未満をDown(離散値0)
RET5_UP_THRESH = 0.03       # "ret5_fixed_threshold"専用: この値超をUp(離散値2)、残りはHold=1
USE_REGIME_AWARE_LOSS = True  # Trueにすると、NearPairRankingLossの「同一銘柄・近傍日」ペアに
                        # さらに「同一regime(LOW/MID/HIGH)」制約を加える(2026-09-19追加)。
                        # near-day(|tidx差|<=MAX_PAIR_GAP=20)ペアの42.7%がregimeを跨いでいる
                        # ことを実測で確認済み——ペアワイズ損失がregime変化と整合しない
                        # 可能性の検証用。REGIME_FILTERとは独立(全レジーム学習のままペアだけ
                        # regime内に絞れる)。
MACRO_CLIP = None       # Noneなら未使用。数値(例: 3.0)を入れるとmacro特徴量をその範囲に
                        # clipする(2026-09-19追加)。pin_dist_ratio/wall_spread/cta_momentum/
                        # nk_ret_normは固定の割り算定数で正規化されておりHIGHレジームに
                        # 適応していない(cta_net_normだけ60日ローリングz-scoreで適応的)。
                        # 実測でHIGH日はこれらがmin=-8.26等の極端値を取ることを確認済み——
                        # HIGHレジーム専用モデルの不安定化がこの正規化の粗さによるものか
                        # 切り分けるためのtoggle。
# ============================================================================

ARCH_CLASSES = {
    "dual_stream": DualStream_GRU_PreLN_Transformer,
    "gru_only": GRUOnlyModel,
    "transformer_only": TransformerOnlyModel,
    "gru_macro_mlp_cross": GRU_MacroMLP_CrossAttn_Model,
}
if MODEL_ARCH not in ARCH_CLASSES:
    raise ValueError(f"MODEL_ARCH='{MODEL_ARCH}'は未対応です。選択肢: {list(ARCH_CLASSES)}")

# 特徴量カタログ: side='stock'(銘柄横断正規化される側)/'macro'(全銘柄共通の日次値側)、
# source=どこから値を取得するか(stock_pool/macro_pool/margin_pool/sector_pool/computedのいずれか)
FEATURE_CATALOG = {}
for _f in STOCK_FEATURE_COLS:
    FEATURE_CATALOG[_f] = {'side': 'stock', 'source': 'stock_pool'}
for _f in BASE_MACRO_COLS:
    FEATURE_CATALOG[_f] = {'side': 'macro', 'source': 'macro_pool'}
for _f in ['atr_term_ratio', 'atr_accel', 'gap_avg_5d', 'volume_zscore', 'relative_strength_5d', 'rs20', 'rs60',
           'dist_from_high60', 'momentum_accel', 'close_location_value', 'body_ratio', 'adx14', 'volume_accel',
           'efficiency_ratio20', 'days_since_high20', 'new_high20', 'positive_gap_ratio_5', 'gap_strength_5',
           'gap_follow_through', 'volume_ma_ratio']:
    FEATURE_CATALOG[_f] = {'side': 'stock', 'source': 'stock_pool'}
for _f in ['margin_ratio_level', 'margin_ratio_zscore_12w', 'margin_buy_chg_1w', 'margin_short_chg_1w']:
    FEATURE_CATALOG[_f] = {'side': 'stock', 'source': 'margin_pool'}
for _f in ['sector_rel_ret_5d', 'sector_rel_ret_20d', 'sector_rel_vol_ratio_5d', 'sector_rank_ret_5d']:
    FEATURE_CATALOG[_f] = {'side': 'stock', 'source': 'sector_pool'}
for _f in ['usdjpy_chg', 'market_vol_regime', 'vix_overnight_chg', 'nk225_ret_5d', 'topix_ret_5d']:
    FEATURE_CATALOG[_f] = {'side': 'macro', 'source': 'macro_pool'}
FEATURE_CATALOG['rolling_beta_x_cta_net_norm'] = {'side': 'stock', 'source': 'computed'}
FEATURE_CATALOG['rolling_beta_x_cta_net_norm_x_market_vol_regime'] = {'side': 'stock', 'source': 'computed'}
FEATURE_CATALOG['cta_net_norm_x_market_vol_regime'] = {'side': 'macro', 'source': 'computed'}
FEATURE_CATALOG['pin_dist_ratio_x_cta_net_norm'] = {'side': 'macro', 'source': 'computed'}
FEATURE_CATALOG['cta_net_norm_x_low_v2'] = {'side': 'macro', 'source': 'computed'}

assert set(FEATURE_TOGGLES) == set(FEATURE_CATALOG), \
    f"FEATURE_TOGGLESとFEATURE_CATALOGのキーが一致しません: " \
    f"toggles-catalog={set(FEATURE_TOGGLES)-set(FEATURE_CATALOG)}, catalog-toggles={set(FEATURE_CATALOG)-set(FEATURE_TOGGLES)}"

# stock側の計算式(dfは銘柄ごと、macro_pool_dfは日次共通)。rolling_betaの生列は
# FEATURE_TOGGLES上のOn/Offに関わらずpool_df内に常に存在するので、計算自体はどの設定でも成立する。
STOCK_COMPUTE_FNS = {
    'rolling_beta_x_cta_net_norm': lambda df, mpdf: df['rolling_beta'] * mpdf['cta_net_norm'].reindex(df.index).fillna(0.0),
    'rolling_beta_x_cta_net_norm_x_market_vol_regime': lambda df, mpdf: (
        df['rolling_beta'] * mpdf['cta_net_norm'].reindex(df.index).fillna(0.0)
        * mpdf['market_vol_regime'].reindex(df.index).fillna(0.0)
    ),
}
# macro側の計算式(macro_pool_df自体に1回だけ追加する、日次共通値)
MACRO_COMPUTE_FNS = {
    'cta_net_norm_x_market_vol_regime': lambda mpdf: mpdf['cta_net_norm'] * mpdf['market_vol_regime'],
    'pin_dist_ratio_x_cta_net_norm': lambda mpdf: mpdf['pin_dist_ratio'] * mpdf['cta_net_norm'],
    # regime_risk_model.py::_compute_vol_regime_is_low と同じ拡大窓(look-ahead無し)版のLOW判定を
    # 使う、cta_net_norm_x_lowの修正版(2026-09-19)。macro_feature_scorecard.csvでは全期間固定
    # 分位点版(look-ahead汚染)がregime_ic_max=0.4221(LOW)と全候補中最強だったが、その値は
    # 汚染込みで信頼できないため未検証のまま却下されていた。修正版はPFモデルでは未テスト。
    'cta_net_norm_x_low_v2': lambda mpdf: mpdf['cta_net_norm'] * _compute_vol_regime_is_low(mpdf),
}

BASELINE_STOCK_COLS = list(STOCK_FEATURE_COLS)
BASELINE_MACRO_COLS = list(BASE_MACRO_COLS)

TEST_FEATS = [f for f, on in FEATURE_TOGGLES.items() if on]
TEST_STOCK_COLS = [f for f in TEST_FEATS if FEATURE_CATALOG[f]['side'] == 'stock']
TEST_MACRO_COLS = [f for f in TEST_FEATS if FEATURE_CATALOG[f]['side'] == 'macro']

if not TEST_STOCK_COLS or not TEST_MACRO_COLS:
    raise ValueError(
        f"テスト側の特徴量が空です(stock={len(TEST_STOCK_COLS)}個, macro={len(TEST_MACRO_COLS)}個)。"
        f"FEATURE_TOGGLESで両側とも最低1つはTrueにしてください。"
    )

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN, HOLDING_PERIOD = 10, 5  # HOLDING_PERIOD: 本番既定10から5へ変更(2026-09-19採用、
                        # ユーザー提案で3seed×3値(10/7/5)を比較した結果、短くするほどPF・
                        # seed間の安定性ともに改善したため —— hp=5: mean=1.482/std=0.089/
                        # ensemble=1.472、hp=10(本番既定): mean=1.445/std=0.291/
                        # ensemble=1.244。本番training/train_model_v8_exp.pyのHOLDING_PERIOD
                        # 相当の値は意図的に変更していない(研究ハーネスのみ)。


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def simulate_ret_pct_d1_close(closes, highs, lows, atrs, holding_period, opens=None, entry_convention="d1_close"):
    """学習ターゲットとD+1引け(MOC)バックテストを同一関数に統合する(2026-09-19、
    ユーザー指摘で発覚した不一致の修正)。

    entry_convention: "d1_close"(既定、D+1引け/MOC) または "d1_open"(D+1始値/MOO、
    2026-09-19追加、ユーザー提案「D+1引けという執行規約自体を見直す」——D+1引けは
    シグナル生成からエントリーまでに丸1日分の値動き(=シグナルの鮮度が失われる時間)が
    挟まるため、より早いD+1始値エントリーと比較検証する)。"d1_open"時はopens配列が必須。
    "d1_open"はエントリー当日(idx+1)がまだ引けていないため、バリア判定をその日
    (h=1)から開始する(production _simulate_ret_pctのnext_open_entryと同じ規約)。
    "d1_close"はエントリー当日の引けで既に約定済みのため、判定は翌日(h=2)から
    (エントリー日基準のオーバーナイトギャップ分のズレは無い)。

    production側のtraining/train_model_v8_exp.py::_simulate_ret_pctは
    entry_p=closes[idx](シグナル当日終値)が既定で、next_open_entry=Trueでも
    opens[idx+1](翌日始値)にしかならず、D+1引け(execution_convention_d1_moc.md
    で確立済みの本番執行方式)という選択肢が無い。このハーネスのバックテスト側
    (旧prepare_backtest_pool内)は既にentry_p=closes[idx+1]で計算していたので、
    学習ターゲット側だけ古い前提のままズレていた。ここではD+1引けに統一し、
    学習ターゲット・バックテストの両方をこの1つの関数から作る
    (production側のtraining/train_model_v8_exp.pyは意図的に触れていない——
    ユーザー判断で研究ハーネスのみ修正)。

    戻り値: (targets, exit_h_days)。targetsはret_pct(計算不能な末尾holding_period
    件はnan)。exit_h_daysは何営業日後に手仕舞いしたか(MaxDD計算のexit_date算出に必要、
    nanの日は-1)。"""
    n_bars = len(closes)
    targets = np.full(n_bars, np.nan)
    exit_h_days = np.full(n_bars, -1, dtype=int)
    start_h = 1 if entry_convention == "d1_open" else 2
    for idx in range(n_bars - holding_period):
        entry_p = opens[idx + 1] if entry_convention == "d1_open" else closes[idx + 1]
        # 利確/損切りバリアはentry_p基準(2026-09-19修正——以前はd_close=closes[idx]
        # 基準になっており、実際のエントリー価格entry_pとのオーバーナイトギャップ分だけ
        # バリア位置がズレていた。実測でギャップ/ATRは平均0.58・中央値0.43、損切り幅
        # 1ATRの50%以上ズレている日が43.7%と、無視できない規模だった。本番
        # training/train_model_v8_exp.py::_simulate_ret_pctは元々entry_p基準で正しく
        # 統一されており、こちらが統合時に合わせ損ねていた)。
        upper_p, lower_p = entry_p + 2.0 * atrs[idx], entry_p - 1.0 * atrs[idx]
        exit_p, h_days = None, holding_period
        for h in range(start_h, holding_period + 1):
            hi, lo = highs[idx + h], lows[idx + h]
            if lo <= lower_p and hi >= upper_p:
                exit_p, h_days = lower_p, h; break
            elif hi >= upper_p:
                exit_p, h_days = upper_p, h; break
            elif lo <= lower_p:
                exit_p, h_days = lower_p, h; break
        if exit_p is None:
            exit_p = closes[idx + holding_period]
        targets[idx] = (exit_p - entry_p) / entry_p
        exit_h_days[idx] = h_days
    return targets, exit_h_days


def compute_ret5_fixed_horizon_targets(closes, holding_days=5):
    """ATRバリア無しの単純な固定期間リターン(2026-09-19追加、ユーザー提案「Down ret5<-3%
    / Flat -3%~+3% / Up ret5>+3%」)。simulate_ret_pct_d1_closeと違い、期間中の利確/損切り
    ラインへの早期到達は一切見ず、entry_p(D+1引け、既存の執行規約と統一)からholding_days
    営業日後の引けまでの単純な変化率のみを返す(タイムアウトのみ)。"""
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
        if FEATURE_CATALOG[f]['source'] == 'computed' and f not in macro_pool_df.columns:
            macro_pool_df[f] = MACRO_COMPUTE_FNS[f](macro_pool_df)
    return macro_pool_df


def build_per_ticker(pool, margin_pool, sector_pool, macro_pool_df, stock_cols):
    """pool_df(既存9+株側候補20が常に列として存在)をベースに、stock_colsに含まれる
    margin_pool/sector_pool/computed由来の列だけを追加したdfを返す。"""
    needed_extra = [f for f in stock_cols if FEATURE_CATALOG[f]['source'] != 'stock_pool']
    per_ticker_df = {}
    for t, pool_df in pool.items():
        try:
            df = pool_df.copy()
            for f in needed_extra:
                src = FEATURE_CATALOG[f]['source']
                if src == 'margin_pool':
                    val = margin_pool[t][f].reindex(df.index) if t in margin_pool else pd.Series(0.0, index=df.index)
                    df[f] = val.fillna(0.0)
                elif src == 'sector_pool':
                    val = sector_pool[t][f].reindex(df.index) if t in sector_pool else pd.Series(0.0, index=df.index)
                    df[f] = val.fillna(0.0)
                elif src == 'computed':
                    df[f] = STOCK_COMPUTE_FNS[f](df, macro_pool_df)
            per_ticker_df[t] = df
        except Exception as e:
            log(f"  [!] build_per_ticker: {t}をスキップ({type(e).__name__}: {e})")
            continue
    return per_ticker_df


def compute_global_split_date(valid_dates, regime_filter=None):
    """train/valの分割を全銘柄共通のカレンダー日で行うための分割日を1回だけ計算する
    (2026-09-19修正、ユーザー指摘: 「分割は全銘柄共通のカレンダー日で行うべき」)。
    valid_datesは銘柄非依存の共通日付集合(valid_cross_section_datesの戻り値)。
    regime_filterを渡すと、その中でregime該当日だけの75%点を取る(regime_filter=None
    のときの通常の75%点と同じロジックを共有する)。"""
    sorted_dates = pd.DatetimeIndex(sorted(valid_dates))
    if regime_filter is not None:
        regime_ok_global = regime_filter.reindex(sorted_dates).fillna(False).values
        eligible_dates = sorted_dates[regime_ok_global]
        if len(eligible_dates) == 0:
            return sorted_dates[-1] if len(sorted_dates) else pd.Timestamp.max
        return eligible_dates[int(len(eligible_dates) * 0.75)]
    return sorted_dates[int(len(sorted_dates) * 0.75)]


def build_dataset(macro_cols, stock_cols, pool, margin_pool, sector_pool, macro_pool_df, regime_filter=None,
                   regime_labels_for_loss=None, global_split_date_override=None):
    """regime_filter: date->bool(True=採用)のpd.Seriesを渡すと、その日付のサンプルだけ
    train/valに採用する(2026-09-19追加、regime別モデル用)。SEQ_LEN分の価格系列window自体は
    regimeでフィルタせず連続したまま使う(不連続にすると時系列モデルの入力が歪むため)。
    idx時点(判定日)がregimeに合致するかどうかだけをフィルタする。
    regime_labels_for_loss: date->'LOW'/'MID'/'HIGH'のpd.Seriesを渡すと、各サンプルの
    regime id(0/1/2、不明なら-1)を6番目の要素として返す(2026-09-19追加、
    NearPairRankingLossRegimeAware用。regime_filterとは独立)。
    global_split_date_override: 指定すると、このconfig自身のvalid_datesから
    split dateを計算せず、渡された値をそのまま使う(2026-09-19追加、ユーザー指摘
    「split dateをbase/testで完全共通化」。base/testで候補特徴量のNaNパターンが違うと
    valid_dates自体がズレて、分割日がbase/testで微妙に食い違いうるため)。"""
    per_ticker_df = build_per_ticker(pool, margin_pool, sector_pool, macro_pool_df, stock_cols)
    macro_feed = macro_pool_df[macro_cols]
    for t in list(per_ticker_df.keys()):
        df = per_ticker_df[t]
        # how='left'(2026-09-19修正、旧'inner'): macro側にたまたま欠損日があっても
        # OHLC/ATRの連続系列を削らない(下のneeded絞り込みと同じ理由)。
        df = df.join(macro_feed, how='left')
        # 以前はstock_cols/macro_cols(候補特徴量含む)のNaNでも行ごと削除しており、
        # 候補特徴量のNaN(rolling window助走期間・欠損データ等)がOHLC/ATRの連続性
        # まで壊していた(2026-09-19修正、ユーザー指摘「特徴量のdropnaでOHLC時系列を
        # 圧縮しない」)。ここではOHLC/ATRという構造的に必須な列だけ要求し、個々の
        # 特徴量のNaNは後段のサンプル単位のwindow NaNチェック(w_s/w_m)に任せる。
        needed = [c for c in ['Close', 'High', 'Low', 'ATR'] if c in df.columns]
        df = df.dropna(subset=needed)
        if len(df) < SEQ_LEN + HOLDING_PERIOD + 10:
            del per_ticker_df[t]; continue
        per_ticker_df[t] = apply_log_transform(df)

    cross_mean, cross_std = compute_cross_sectional_stats(per_ticker_df, stock_cols)
    valid_dates = valid_cross_section_dates(per_ticker_df, stock_cols, MIN_CROSS_SECTION)
    global_split_date = global_split_date_override if global_split_date_override is not None \
        else compute_global_split_date(valid_dates, regime_filter)
    # tidxは以前、銘柄ごとのdf内の「配列位置」だった(2026-09-19修正前)。dropnaで行が
    # 削れる銘柄があると、同じtidx差でも銘柄によって実際のカレンダー日数差が異なり、
    # NearPairRankingLossのgap=|tidx_i-tidx_j|<=MAX_PAIR_GAPが「20営業日以内」を意味しなく
    # なっていた(ユーザー指摘「共通カレンダーtidxへ変更」)。macro_pool_df.index
    # (全銘柄共通の取引日カレンダー)上の位置を使うことで、tidx差が全銘柄で同じ意味の
    # 営業日数差になるようにする。
    calendar_tidx_map = {d: i for i, d in enumerate(sorted(macro_pool_df.index))}

    # raw ret_pctを銘柄ごとに1回だけ計算し、辞書に保持する(2026-09-19追加。以前はこのループ内で
    # 銘柄ごとに直接計算していたが、USE_QUANTILE_LABELSのcross-sectional分位点計算には
    # 全銘柄分のraw ret_pctが同時に必要なため、先に1回計算してから使い回す)。
    raw_targets = {}
    for t, df in per_ticker_df.items():
        closes, highs, lows, atrs = df['Close'].values, df['High'].values, df['Low'].values, df['ATR'].values
        opens = df['Open'].values if ENTRY_CONVENTION == "d1_open" else None
        targets, _ = simulate_ret_pct_d1_close(closes, highs, lows, atrs, HOLDING_PERIOD,
                                                opens=opens, entry_convention=ENTRY_CONVENTION)
        raw_targets[t] = pd.Series(targets, index=df.index)

    train_targets = raw_targets
    if LABEL_MODE == "cross_sectional_quantile":
        # 同日全銘柄のraw ret_pctを横断してcross-sectional分位点化する(2026-09-19追加、
        # ユーザー提案)。学習ターゲットのみ置き換え、バックテスト側(prepare_backtest_pool)は
        # 連続値ret_pctのまま。
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
        # ATRバリア無しの固定期間(RET5_HOLDING_DAYS日)リターンを、固定の絶対閾値で3値化する
        # (2026-09-19追加、ユーザー提案「Down ret5<-3% / Flat -3%~+3% / Up ret5>+3%」)。
        ret5_targets = {}
        for t, df in per_ticker_df.items():
            closes = df['Close'].values
            r5 = compute_ret5_fixed_horizon_targets(closes, holding_days=RET5_HOLDING_DAYS)
            s = pd.Series(r5, index=df.index)
            disc = pd.Series(1.0, index=s.index)
            disc[s <= RET5_DOWN_THRESH] = 0.0
            disc[s >= RET5_UP_THRESH] = 2.0
            disc[s.isna()] = np.nan
            ret5_targets[t] = disc
        train_targets = ret5_targets
    elif LABEL_MODE != "continuous":
        raise ValueError(f"LABEL_MODE='{LABEL_MODE}'は未対応です"
                          f"(continuous/cross_sectional_quantile/ret5_fixed_thresholdのみ)")

    tr_x_s, tr_x_m, tr_y, tr_tid, tr_tidx, tr_regime = [], [], [], [], [], []
    va_x_s, va_x_m, va_y, va_tid, va_tidx, va_regime = [], [], [], [], [], []
    for i, (t, df) in enumerate(per_ticker_df.items()):
        try:
            # valid_datesで行ごと削ると、その日が銘柄自身の時系列から物理的に消え、
            # SEQ_LEN分のlookback windowが実際には連続していない日をまたいでしまう
            # (regime_filterで既に修正した問題と同じ穴、2026-09-19修正)。判定日として
            # 採用するかどうかだけをvalid_okで絞り、window自体は削らず連続のまま使う。
            if len(df) < SEQ_LEN + HOLDING_PERIOD + 10:
                continue
            norm_s = normalize_cross_sectional(df, cross_mean, cross_std, stock_cols)
            targets = raw_targets[t].values             # NaN判定・purge判定は常に連続値基準
            train_target_vals = train_targets[t].values  # 実際に学習へ渡す値(quantile化 or 連続値)
            vals_s_norm = norm_s.values
            vals_m = df[macro_cols].values
            if MACRO_CLIP is not None:
                vals_m = np.clip(vals_m, -MACRO_CLIP, MACRO_CLIP)
            n_samples = len(df)
            dates = df.index
            valid_ok = dates.isin(valid_dates)
            regime_ok = regime_filter.reindex(dates).fillna(False).values if regime_filter is not None else None
            if regime_labels_for_loss is not None:
                regime_id_arr = regime_labels_for_loss.reindex(dates).map(REGIME_ID_MAP).fillna(-1).astype(int).values
            else:
                regime_id_arr = None
            # 2026-09-19追加: NearPairRankingLossのgap計算に使うtidxは、配列位置ではなく
            # 全銘柄共通のカレンダー日位置にする(calendar_tidx_map、上で定義)。
            cal_tidx_arr = np.array([calendar_tidx_map.get(d, -1) for d in dates], dtype=np.int64)

            for idx in range(SEQ_LEN - 1, n_samples):
                if not valid_ok[idx]:
                    continue
                # NaN判定はtargets(連続値、常にHOLDING_PERIOD基準でpurge日程と整合)と
                # train_target_vals(実際に学習へ渡す値、LABEL_MODEによって有効範囲が異なりうる)
                # の両方をチェックする(2026-09-19追加、ret5_fixed_thresholdはHOLDING_PERIODより
                # 短いRET5_HOLDING_DAYS基準のため、有効範囲がtargetsと厳密には一致しない)。
                if np.isnan(targets[idx]) or np.isnan(train_target_vals[idx]):
                    continue
                if regime_ok is not None and not regime_ok[idx]:
                    continue
                w_s = vals_s_norm[idx - SEQ_LEN + 1: idx + 1].copy()
                if np.isnan(w_s).any():
                    continue
                w_m = vals_m[idx - SEQ_LEN + 1: idx + 1].copy()
                if np.isnan(w_m).any():  # 2026-09-19追加: w_s側にしかNaNチェックが無かった
                    continue
                target = train_target_vals[idx]
                r_id = int(regime_id_arr[idx]) if regime_id_arr is not None else -1
                # 全銘柄共通のカレンダー日(global_split_date)で分割する(2026-09-19修正)。
                # 以前は各銘柄自身のサンプル数の75%点で切っていたため、銘柄ごとに実際の
                # 分割日が最大165日(約5.5ヶ月)もずれていた——上場時期やdropnaで欠損する
                # 行数が銘柄ごとに違うため。これだと、ある銘柄の「検証期間」の日付が、
                # 別の銘柄では依然として「学習期間」として使われてしまい、時系列上の
                # out-of-sample性が銘柄間で一貫しなくなる(ユーザー指摘で発覚)。
                cal_tidx = int(cal_tidx_arr[idx])
                is_train = dates[idx] < global_split_date
                if is_train and dates[idx + HOLDING_PERIOD] >= global_split_date:
                    # purge(2026-09-19追加、ユーザー指摘「Train/Val境界をまたぐラベルをpurge」):
                    # このサンプルのラベルはidx+HOLDING_PERIOD日後までの値動きを見て決まる
                    # (simulate_ret_pct_d1_close)。idxはtrain期間でも、ラベルの参照範囲が
                    # val期間に食い込んでいたら、実質的にval期間の情報がtrainに漏れている
                    # (purged cross-validationの考え方)。そのサンプルごと除外する。
                    continue
                if is_train:
                    tr_x_s.append(w_s); tr_x_m.append(w_m); tr_y.append(target); tr_tid.append(i); tr_tidx.append(cal_tidx); tr_regime.append(r_id)
                else:
                    va_x_s.append(w_s); va_x_m.append(w_m); va_y.append(target); va_tid.append(i); va_tidx.append(cal_tidx); va_regime.append(r_id)
        except Exception as e:
            log(f"  [!] build_dataset: {t}をスキップ({type(e).__name__}: {e})")
            continue

    return (np.array(tr_x_s), np.array(tr_x_m), np.array(tr_y), np.array(tr_tid), np.array(tr_tidx), np.array(tr_regime)), \
           (np.array(va_x_s), np.array(va_x_m), np.array(va_y), np.array(va_tid), np.array(va_tidx), np.array(va_regime))


def prepare_backtest_pool(macro_cols, stock_cols, pool, margin_pool, sector_pool, macro_pool_df, regime_filter=None,
                           global_split_date_override=None):
    per_ticker_df = build_per_ticker(pool, margin_pool, sector_pool, macro_pool_df, stock_cols)
    macro_feed = macro_pool_df[macro_cols]
    for t in list(per_ticker_df.keys()):
        df = per_ticker_df[t]
        # how='left'(2026-09-19修正、旧'inner'): macro側にたまたま欠損日があっても
        # OHLC/ATRの連続系列を削らない(下のneeded絞り込みと同じ理由)。
        df = df.join(macro_feed, how='left')
        # 以前はstock_cols/macro_cols(候補特徴量含む)のNaNでも行ごと削除しており、
        # 候補特徴量のNaN(rolling window助走期間・欠損データ等)がOHLC/ATRの連続性
        # まで壊していた(2026-09-19修正、ユーザー指摘「特徴量のdropnaでOHLC時系列を
        # 圧縮しない」)。ここではOHLC/ATRという構造的に必須な列だけ要求し、個々の
        # 特徴量のNaNは後段のサンプル単位のwindow NaNチェック(w_s/w_m)に任せる。
        needed = [c for c in ['Close', 'High', 'Low', 'ATR'] if c in df.columns]
        df = df.dropna(subset=needed)
        if len(df) < SEQ_LEN + HOLDING_PERIOD + 10:
            del per_ticker_df[t]; continue
        per_ticker_df[t] = apply_log_transform(df)

    cross_mean, cross_std = compute_cross_sectional_stats(per_ticker_df, stock_cols)
    valid_dates = valid_cross_section_dates(per_ticker_df, stock_cols, MIN_CROSS_SECTION)
    global_split_date = global_split_date_override if global_split_date_override is not None \
        else compute_global_split_date(valid_dates, regime_filter)

    pool_out = []
    for t, df in per_ticker_df.items():
        try:
            # build_datasetと同じ理由でvalid_datesによる行削除をやめ、連続windowを保つ
            # (2026-09-19修正)。判定日の採用可否だけをvalid_okで絞る。
            if len(df) < SEQ_LEN + HOLDING_PERIOD + 10:
                continue
            norm_s = normalize_cross_sectional(df, cross_mean, cross_std, stock_cols)
            n_samples = len(df)
            vals_s_norm = norm_s.values
            vals_m = df[macro_cols].values
            if MACRO_CLIP is not None:
                vals_m = np.clip(vals_m, -MACRO_CLIP, MACRO_CLIP)
            closes, highs, lows, atrs = df['Close'].values, df['High'].values, df['Low'].values, df['ATR'].values
            dates = df.index
            valid_ok = dates.isin(valid_dates)
            regime_ok = regime_filter.reindex(dates).fillna(False).values if regime_filter is not None else None
            # build_datasetの学習ターゲットと同じ関数を使う(2026-09-19、同一関数化・
            # エントリー価格前提の統一)。exit_h_daysはMaxDD計算のexit_date算出に必要。
            opens = df['Open'].values if ENTRY_CONVENTION == "d1_open" else None
            ret_pcts, exit_h_days = simulate_ret_pct_d1_close(closes, highs, lows, atrs, HOLDING_PERIOD,
                                                               opens=opens, entry_convention=ENTRY_CONVENTION)

            for idx in range(SEQ_LEN - 1, n_samples - HOLDING_PERIOD):
                if not valid_ok[idx]:
                    continue
                if dates[idx] < global_split_date:  # 全銘柄共通のカレンダー日で分割(2026-09-19修正、build_datasetと同じ理由)
                    continue
                if regime_ok is not None and not regime_ok[idx]:
                    continue
                if np.isnan(ret_pcts[idx]):
                    continue
                w_s = vals_s_norm[idx - SEQ_LEN + 1: idx + 1].copy()
                if np.isnan(w_s).any():
                    continue
                w_m = vals_m[idx - SEQ_LEN + 1: idx + 1].copy()
                if np.isnan(w_m).any():  # 2026-09-19追加: w_s側にしかNaNチェックが無かった
                    continue
                # exit_dateはMaxDD計算(資金曲線シミュレーション)に必要(2026-09-19追加)
                # entry_date=dates[idx+1](D+1引け)をMaxDD計算用に追加(2026-09-19修正、
                # ユーザー指摘「MaxDDのエントリー日をD+1へ変更」。以前はcompute_max_drawdownが
                # date(シグナル当日)をエントリー日として資金曲線を組んでおり、実際の約定日
                # (D+1)より1日早いタイミングで資金を拘束していた)。
                pool_out.append((t, dates[idx], dates[idx + 1], dates[idx + exit_h_days[idx]], w_s, w_m, ret_pcts[idx]))
        except Exception as e:
            log(f"  [!] prepare_backtest_pool: {t}をスキップ({type(e).__name__}: {e})")
            continue
    return pool_out


def train_model(seed, tr_data, va_data, s_cols, m_cols, label, profile_this_call=False):
    tr_x_s, tr_x_m, tr_y, tr_tid, tr_tidx, tr_regime = tr_data
    va_x_s, va_x_m, va_y, va_tid, va_tidx, va_regime = va_data
    set_seed(seed)

    # Dataset/DataLoaderを経由せず、テンソル化した全データを1回だけGPUに載せ、バッチは
    # そこから直接fancy indexingで取得する(2026-09-19、ユーザー提案。データは既に
    # メモリ上の配列でDataLoaderの遅延ロード/並列ワーカーの恩恵が無いため、__getitem__+
    # collateのオーバーヘッドと毎バッチのCPU->GPU転送を回避できる)。
    # SameTickerBatchSampler.__iter__は呼ぶたびに(shuffle=Trueなら)再シャッフルする
    # ため、元のDataLoader経由(毎epoch新しいイテレータ)と同じくepochごとに再シャッフル
    # される(サンプラー自体は使い回し、for batch_idx in train_sampler: で回す)。
    tr_xs_t = torch.tensor(tr_x_s, dtype=torch.float32, device=DEVICE)
    tr_xm_t = torch.tensor(tr_x_m, dtype=torch.float32, device=DEVICE)
    tr_y_t = torch.tensor(tr_y, dtype=torch.float32, device=DEVICE)
    tr_tidx_t = torch.tensor(tr_tidx, dtype=torch.long, device=DEVICE)
    tr_regime_t = torch.tensor(tr_regime, dtype=torch.long, device=DEVICE)
    va_xs_t = torch.tensor(va_x_s, dtype=torch.float32, device=DEVICE)
    va_xm_t = torch.tensor(va_x_m, dtype=torch.float32, device=DEVICE)
    va_y_t = torch.tensor(va_y, dtype=torch.float32, device=DEVICE)
    va_tidx_t = torch.tensor(va_tidx, dtype=torch.long, device=DEVICE)
    va_regime_t = torch.tensor(va_regime, dtype=torch.long, device=DEVICE)

    train_sampler = SameTickerBatchSampler(tr_tid, TRAIN_BATCH_SIZE, shuffle=True)
    val_batches = list(SameTickerBatchSampler(va_tid, VAL_BATCH_SIZE, shuffle=False))

    arch_cls = ARCH_CLASSES[MODEL_ARCH]
    if MODEL_ARCH == "gru_macro_mlp_cross":
        # 株64/マクロ32(縮小してSTOCK_HIDDEN_DIM=32/MACRO_HIDDEN_DIM=16)という非対称次元
        # 構成のため、他アーキテクチャ共通のMODEL_HIDDEN_DIMではなく専用toggleを使う
        # (2026-09-19追加、ユーザー提案「Technical GRU(64)+Macro MLP(32)+Cross Attention」)。
        model_kwargs = dict(stock_dim=len(s_cols), macro_dim=len(m_cols),
                             stock_hidden=STOCK_HIDDEN_DIM, macro_hidden=MACRO_HIDDEN_DIM,
                             num_heads=CROSS_ATTN_NUM_HEADS, num_classes=3, dropout=0.2,
                             use_cross_ffn=CROSS_ATTN_USE_FFN)
    else:
        model_kwargs = dict(stock_dim=len(s_cols), macro_dim=len(m_cols), hidden_dim=MODEL_HIDDEN_DIM,
                             num_heads=MODEL_NUM_HEADS, num_classes=3, dropout=0.2)
        if MODEL_ARCH == "dual_stream":
            model_kwargs['use_cross_ffn'] = USE_CROSS_FFN
            model_kwargs['use_final_norm'] = USE_FINAL_NORM
        # gru_only/transformer_onlyにはuse_cross_ffn/use_final_norm概念が無いので渡さない
        # (**kwargsを持つので万一渡しても無視されるが、意図を明確にするため分岐している)
        if MODEL_ARCH in ("dual_stream", "transformer_only"):
            model_kwargs['num_self_attn_layers'] = NUM_SELF_ATTN_LAYERS
    model = arch_cls(**model_kwargs).to(DEVICE)
    # 2026-09-19: pair数加重・ゼロペアバッチ除外のため、regime制約の有無に関わらず常に
    # NearPairRankingLossRegimeAware(regime=Noneなら本番NearPairRankingLossと同一)を使う。
    criterion = NearPairRankingLossRegimeAware(max_gap=MAX_PAIR_GAP)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0002, weight_decay=3e-2)

    amp_dtype = torch.bfloat16 if AMP_DTYPE == "bf16" else torch.float16
    amp_enabled = USE_AMP and DEVICE.type == "cuda"
    # fp16はアンダーフロー対策にGradScalerが要る(bf16はfp32と同じ指数範囲なので不要)。
    use_scaler = amp_enabled and AMP_DTYPE == "fp16"
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    best_val_loss, best_state, patience_cnt = float('inf'), None, 0
    patience, epochs = 7, 30
    for epoch in range(1, epochs + 1):
        do_profile = profile_this_call and epoch == 1
        prof = None
        if do_profile:
            activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if torch.cuda.is_available() else [])
            prof = torch_profile(activities=activities)
            prof.__enter__()

        model.train()
        # 検証ループと同じ理由(.item()は毎回GPU-CPU同期を発生させる)で、テンソルのまま
        # 累積し、学習ループ終了後に1回だけ.item()する(2026-09-18、ユーザー指摘で修正)。
        # バッチごとにtr_xs_t[batch_idx]でfancy indexingすると、バッチ数だけgatherカーネルが
        # 起動する(index変換のCPU->GPU転送込み)。epochの全バッチ列を1回だけ結合してgatherし、
        # 各バッチはその結果を連続スライスで取り出す(スライスはコピー無しのview)ことで、
        # gather呼び出しをバッチ数回→epochあたり4回(xs/xm/y/tidx)に減らす(2026-09-19、ユーザー提案)。
        # SameTickerBatchSamplerはlist(train_sampler)を呼んだ時点でshuffle済みの順序が
        # 確定するため、バッチ構成・ペア(NearPairRankingLossの対象)は元の実装と完全に同一。
        epoch_batches = list(train_sampler)
        flat_idx = [i for b in epoch_batches for i in b]
        flat_idx_t = torch.tensor(flat_idx, dtype=torch.long, device=DEVICE)
        xs_shuf = tr_xs_t.index_select(0, flat_idx_t)
        xm_shuf = tr_xm_t.index_select(0, flat_idx_t)
        y_shuf = tr_y_t.index_select(0, flat_idx_t)
        tidx_shuf = tr_tidx_t.index_select(0, flat_idx_t)
        regime_shuf = tr_regime_t.index_select(0, flat_idx_t)

        # 2026-09-19修正: エポック全体のtrain_lossをバッチ単純平均から「有効ペア数加重」の
        # 平均に変更(ユーザー指摘)。バッチごとに有効ペア数(NearPairRankingLossが実際に
        # 比較したペアの数)が大きく変わりうるため、単純平均だとペアの少ないバッチと
        # 多いバッチを同列に扱ってしまう。n_pairs=0のバッチ(有効ペア無し=学習情報を
        # 持たない)はbackward/optimizer.stepもスキップし、平均からも自然に除外する。
        total_tr = torch.zeros((), device=DEVICE)
        total_tr_pairs = 0
        n_tr_batches = 0
        offset = 0
        for b in epoch_batches:
            bs = len(b)
            b_xs, b_xm, b_y, b_tidx = xs_shuf[offset:offset + bs], xm_shuf[offset:offset + bs], y_shuf[offset:offset + bs], tidx_shuf[offset:offset + bs]
            b_regime = regime_shuf[offset:offset + bs]
            offset += bs
            with torch.autocast(device_type=DEVICE.type, dtype=amp_dtype, enabled=amp_enabled):
                loss, n_pairs = criterion(model(b_xs, b_xm), b_y, b_tidx, b_regime if USE_REGIME_AWARE_LOSS else None)
            if n_pairs == 0:
                continue  # 有効ペアが無いバッチはbackward/step/平均のいずれからも除外
            optimizer.zero_grad(set_to_none=True)
            if use_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            total_tr += loss.detach() * n_pairs; total_tr_pairs += n_pairs; n_tr_batches += 1
        train_loss = (total_tr / total_tr_pairs).item() if total_tr_pairs > 0 else float('inf')

        if do_profile:
            prof.__exit__(None, None, None)
            sort_key = "self_cuda_time_total" if torch.cuda.is_available() else "self_cpu_time_total"
            log(f"[Profiler] [{label}] epoch1 学習ループ({n_tr_batches}バッチ)処理時間トップ10:")
            print(prof.key_averages().table(sort_by=sort_key, row_limit=10))

        model.eval()
        # .item()は毎回GPU-CPU同期を発生させるので、バッチごとには呼ばずテンソルのまま
        # 累積し、検証ループ終了後に1回だけ.item()する(2026-09-18、ユーザー指摘で修正)。
        # 学習ループと同じ理由でpair数加重にする(2026-09-19修正)。n_pairs=0のバッチは
        # loss*0=0がtotal_vaに、0がtotal_va_pairsに足されるだけなので、明示的なifを
        # 書かなくても自然に平均から除外される。
        total_va = torch.zeros((), device=DEVICE)
        total_va_pairs = 0
        n_batches = 0
        with torch.no_grad():
            for batch_idx in val_batches:
                b_xs, b_xm, b_y, b_tidx = va_xs_t[batch_idx], va_xm_t[batch_idx], va_y_t[batch_idx], va_tidx_t[batch_idx]
                b_regime = va_regime_t[batch_idx]
                with torch.autocast(device_type=DEVICE.type, dtype=amp_dtype, enabled=amp_enabled):
                    loss, n_pairs = criterion(model(b_xs, b_xm), b_y, b_tidx, b_regime if USE_REGIME_AWARE_LOSS else None)
                total_va += loss * n_pairs; total_va_pairs += n_pairs; n_batches += 1
        if n_batches == 0:
            log(f"  [{label}] epoch {epoch:2d}/{epochs}: [!] 警告 検証バッチが0件です"
                f"(regime_filter等でval側のサンプルが無くなっている可能性)")
        elif total_va_pairs == 0:
            log(f"  [{label}] epoch {epoch:2d}/{epochs}: [!] 警告 検証バッチは{n_batches}件あるが"
                f"有効ペアが1つもありません")
        val_loss = (total_va / total_va_pairs).item() if total_va_pairs > 0 else float('inf')
        if np.isnan(val_loss) or np.isnan(train_loss):
            log(f"  [{label}] epoch {epoch:2d}/{epochs}: [!] 警告 train_loss/val_lossにNaNが出ています"
                f"(train_loss={train_loss}, val_loss={val_loss})——学習が発散している可能性")

        if val_loss < best_val_loss:
            best_val_loss, best_state, patience_cnt = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
            marker = " <- best"
        else:
            patience_cnt += 1
            marker = f" (patience {patience_cnt}/{patience})"
        log(f"  [{label}] epoch {epoch:2d}/{epochs}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}{marker}")
        if patience_cnt >= patience:
            break
    if best_state is None:
        # 2026-09-19追加: 以前はbest_state=Noneのままload_state_dictに渡し、
        # 「TypeError: Expected state_dict to be dict-like, got NoneType」という
        # 分かりにくいエラーで落ちていた(LOW限定regimeの初期バグで実際に発生)。
        # val_lossが1度も改善しなかった(=val側が空、またはval_lossが常にNaN/inf)ことを
        # 明示するエラーに変える。
        raise RuntimeError(
            f"[{label} seed={seed}] 学習全エポックでval_lossが一度も改善しませんでした(best_state=None)。"
            f"val側のサンプル数が0、またはval_lossが常にNaN/infの可能性があります。"
            f"regime_filter/特徴量構成/学習データ量を確認してください。"
        )
    model.load_state_dict(best_state); model.eval()
    log(f"[+] [{label} seed={seed}] 学習完了 best_val_loss={best_val_loss:.4f}")
    return model


def predict1(model, w_s, w_m):
    """1件ずつ推論する旧実装。正しさの検証用に残してある(run_backtest_inferenceは
    2026-09-18よりバッチ推論に変更済み、通常はこちらを直接呼ぶ必要はない)。"""
    t_s = torch.tensor(w_s, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    t_m = torch.tensor(w_m, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(t_s, t_m), dim=-1).squeeze(0).cpu().numpy()
    return float(probs[2]), float(probs[0])


def run_backtest_inference(model, pool, batch_size=256):
    """poolを1件ずつではなくbatch_size件まとめてforwardする(2026-09-18、1件ずつだと
    ~23000件でその都度カーネル起動+GPU-CPU同期が発生し非常に遅かったため)。
    LayerNormのみでBatchNorm/Dropoutはeval()で無効なので、バッチをどう区切っても
    1件ずつ推論した場合と数値的に完全一致する。"""
    model.eval()
    records = []
    with torch.no_grad():
        for i in range(0, len(pool), batch_size):
            chunk = pool[i:i + batch_size]
            w_s_batch = np.stack([item[4] for item in chunk])
            w_m_batch = np.stack([item[5] for item in chunk])
            t_s = torch.tensor(w_s_batch, dtype=torch.float32).to(DEVICE)
            t_m = torch.tensor(w_m_batch, dtype=torch.float32).to(DEVICE)
            probs = torch.softmax(model(t_s, t_m), dim=-1).cpu().numpy()
            for j, (t, date, entry_date, exit_date, w_s, w_m, ret_pct) in enumerate(chunk):
                p_win, p_stop = float(probs[j, 2]), float(probs[j, 0])
                # scoreはNearPairRankingLoss(RegimeAware)が直接最適化している量そのもの
                # (2026-09-19追加)。p_win単体は絶対確率として校正されていない(実測で
                # 10分位のcalibration checkを行い、win_rateとの平均ギャップ0.112・非単調を
                # 確認済み)ため、ランキング用にはscoreを使う。
                records.append({'ticker': t, 'date': date, 'entry_date': entry_date, 'exit_date': exit_date,
                                 'p_win': p_win, 'p_stop': p_stop, 'score': p_win - p_stop, 'ret_pct': ret_pct})
    return pd.DataFrame(records)


def run_backtest_inference_ensemble(models, pool, batch_size=256):
    """run_backtest_inferenceの複数モデル版(2026-09-19追加)。本番の
    modules/model_inference.py::predict_probabilities_ensemble_batchと同じロジック
    (各モデルのsoftmax確率を平均してからp_win/p_stopを取り出す)で、単一seedの
    偏り(あるseedがあるregimeに極端に偏ってSTRONG BUYを出す現象、2026-09-19に
    実測確認)を緩和する。本番は5seedアンサンブルだが、ここはSEEDSの数だけ使う。"""
    for model in models:
        model.eval()
    records = []
    for i in range(0, len(pool), batch_size):
        chunk = pool[i:i + batch_size]
        w_s_batch = np.stack([item[4] for item in chunk])
        w_m_batch = np.stack([item[5] for item in chunk])
        p_win_arr, p_stop_arr, _ = predict_probabilities_ensemble_batch(models, w_s_batch, w_m_batch)
        for j, (t, date, entry_date, exit_date, w_s, w_m, ret_pct) in enumerate(chunk):
            p_win, p_stop = float(p_win_arr[j]), float(p_stop_arr[j])
            records.append({'ticker': t, 'date': date, 'entry_date': entry_date, 'exit_date': exit_date,
                             'p_win': p_win, 'p_stop': p_stop, 'score': p_win - p_stop, 'ret_pct': ret_pct})
    return pd.DataFrame(records)


if __name__ == "__main__":
    removed_from_baseline = (set(BASELINE_STOCK_COLS) | set(BASELINE_MACRO_COLS)) - set(TEST_STOCK_COLS + TEST_MACRO_COLS)
    added_candidates = set(TEST_STOCK_COLS + TEST_MACRO_COLS) - (set(BASELINE_STOCK_COLS) | set(BASELINE_MACRO_COLS))
    print(f"[*] テスト側 株{len(TEST_STOCK_COLS)}個 + マクロ{len(TEST_MACRO_COLS)}個 = 計{len(TEST_FEATS)}個")
    print(f"    既存から外した: {sorted(removed_from_baseline) if removed_from_baseline else '(なし)'}")
    print(f"    新規追加した候補: {sorted(added_candidates) if added_candidates else '(なし)'}")
    print(f"[*] モデル設定: arch={MODEL_ARCH}, hidden_dim={MODEL_HIDDEN_DIM}, num_heads={MODEL_NUM_HEADS}, "
          f"use_cross_ffn={USE_CROSS_FFN}, use_final_norm={USE_FINAL_NORM}, "
          f"num_self_attn_layers={NUM_SELF_ATTN_LAYERS}, train_batch_size={TRAIN_BATCH_SIZE}, seeds={SEEDS}\n")

    with open(UNIVERSE_PATH, "r", encoding="utf-8") as f:
        tickers = [line.strip() for line in f if line.strip()]

    macro_df = load_macro_slim5(return_source="nk225")
    pool = get_full_feature_pool_df(tickers, macro_df)
    margin_pool = get_margin_pool_df(tickers, macro_df)
    sector_pool = get_sector_relative_pool_df(tickers, macro_df)
    macro_pool_df = augment_macro_pool_df(get_full_macro_pool_df())

    if REGIME_FILTER == "LOW":
        is_target = compute_is_low_regime_filter(macro_pool_df)
        log(f"[*] REGIME_FILTER=LOW: 全{len(is_target)}日中{int(is_target.sum())}日({is_target.mean()*100:.0f}%)"
            f"がLOWレジーム(拡大窓is_low、look-ahead無し)")
        train_regime_filter = is_target
    elif REGIME_FILTER == "HIGH":
        is_target = compute_is_high_regime_filter(macro_pool_df)
        log(f"[*] REGIME_FILTER=HIGH: 全{len(is_target)}日中{int(is_target.sum())}日({is_target.mean()*100:.0f}%)"
            f"がHIGHレジーム(拡大窓is_high、look-ahead無し)")
        train_regime_filter = is_target
    elif REGIME_FILTER == "MID":
        is_target = compute_is_mid_regime_filter(macro_pool_df)
        log(f"[*] REGIME_FILTER=MID: 全{len(is_target)}日中{int(is_target.sum())}日({is_target.mean()*100:.0f}%)"
            f"がMIDレジーム(拡大窓、look-ahead無し)")
        train_regime_filter = is_target
    elif REGIME_FILTER is None:
        train_regime_filter = None
    else:
        raise ValueError(f"REGIME_FILTER='{REGIME_FILTER}'は未対応です(None / 'LOW' / 'MID' / 'HIGH'のみ)")

    regime_labels = compute_regime_labels(macro_pool_df)
    if USE_REGIME_AWARE_LOSS:
        log("[*] USE_REGIME_AWARE_LOSS=True: NearPairRankingLossのペアを同一regime同士に限定します"
            "(ペアタグ付けはlook-ahead無しの拡大窓判定を使用、2026-09-19修正)")

    # データセット・バックテストプールはシードに依存しないので、シードループの外で1回だけ構築する
    log("[*] データセット・バックテストプール構築(シード非依存、1回だけ)...")
    # regime_labels(全期間固定分位、報告用regime_breakdownのみで使う)とは別に、学習の
    # ペアタグ付けにはlook-ahead無しの拡大窓版を使う(2026-09-19修正、本番化を見据えて)。
    loss_regime_labels = compute_regime_labels_expanding(macro_pool_df) if USE_REGIME_AWARE_LOSS else None

    # split dateはbase/testで完全に共通化する(2026-09-19修正、ユーザー指摘)。以前は
    # build_dataset/prepare_backtest_poolがそれぞれ自分のvalid_dates(候補特徴量の
    # NaNパターンに依存しうる)から独立に計算しており、base/testで分割日が微妙に
    # 食い違う可能性があった。ベースライン(既存14、常に固定)のvalid_datesを基準に
    # 1回だけ計算し、base/test両方に同じ値を渡す。
    _base_per_ticker = build_per_ticker(pool, margin_pool, sector_pool, macro_pool_df, BASELINE_STOCK_COLS)
    _base_valid_dates = valid_cross_section_dates(_base_per_ticker, BASELINE_STOCK_COLS, MIN_CROSS_SECTION)
    shared_split_date = compute_global_split_date(_base_valid_dates, train_regime_filter)
    log(f"[*] 共通split date(base/test): {shared_split_date.date()}")

    tr0, va0 = build_dataset(BASELINE_MACRO_COLS, BASELINE_STOCK_COLS, pool, margin_pool, sector_pool, macro_pool_df, regime_filter=train_regime_filter, regime_labels_for_loss=loss_regime_labels, global_split_date_override=shared_split_date)
    tr1, va1 = build_dataset(TEST_MACRO_COLS, TEST_STOCK_COLS, pool, margin_pool, sector_pool, macro_pool_df, regime_filter=train_regime_filter, regime_labels_for_loss=loss_regime_labels, global_split_date_override=shared_split_date)
    pool_base = prepare_backtest_pool(BASELINE_MACRO_COLS, BASELINE_STOCK_COLS, pool, margin_pool, sector_pool, macro_pool_df, regime_filter=train_regime_filter, global_split_date_override=shared_split_date)
    pool_test = prepare_backtest_pool(TEST_MACRO_COLS, TEST_STOCK_COLS, pool, margin_pool, sector_pool, macro_pool_df, regime_filter=train_regime_filter, global_split_date_override=shared_split_date)
    log(f"[+] train0={len(tr0[2])} val0={len(va0[2])} / train1={len(tr1[2])} val1={len(va1[2])}")
    log(f"[+] backtest_pool base={len(pool_base)} test={len(pool_test)}")

    results_base, results_test = [], []
    models_base, models_test = [], []
    for si, seed in enumerate(SEEDS):
        log(f"=== seed={seed} ===")
        model_base = train_model(seed, tr0, va0, BASELINE_STOCK_COLS, BASELINE_MACRO_COLS, "ベースライン(既存14固定)",
                                  profile_this_call=(PROFILE_FIRST_EPOCH and si == 0))
        d_base = run_backtest_inference(model_base, pool_base)
        model_test = train_model(seed, tr1, va1, TEST_STOCK_COLS, TEST_MACRO_COLS, f"テスト構成({len(TEST_FEATS)}個)")
        d_test = run_backtest_inference(model_test, pool_test)

        r_base = evaluate(d_base, f"seed={seed} ベースライン score Top-K", min_reliable_n=MIN_RELIABLE_N,
                           common_sample_k=COMMON_SAMPLE_K, max_concurrent=MAX_CONCURRENT_POSITIONS)
        r_test = evaluate(d_test, f"seed={seed} テスト構成 score Top-K", min_reliable_n=MIN_RELIABLE_N,
                           common_sample_k=COMMON_SAMPLE_K, max_concurrent=MAX_CONCURRENT_POSITIONS)
        r_base['seed'] = seed; r_test['seed'] = seed
        results_base.append(r_base); results_test.append(r_test)
        models_base.append(model_base); models_test.append(model_test)
        print()

    # アンサンブル評価(2026-09-19追加): 単一seedはSTRONG BUYが特定regimeに偏る不安定性が
    # あること(ユーザー提案で実測確認済み)を受けて、本番と同じsoftmax確率平均アンサンブル
    # (predict_probabilities_ensemble_batch)での評価も追加する。SEEDSが2個以上のときだけ
    # 意味があるので、1個なら単一seedと同じ結果になりスキップしない(比較の一貫性のため)。
    print("=" * 90)
    print(f"【アンサンブル評価({len(SEEDS)}seed平均、本番run_dynamic_regime_screening_v8.pyと同じロジック)】")
    print("=" * 90)
    d_base_ens = run_backtest_inference_ensemble(models_base, pool_base)
    d_test_ens = run_backtest_inference_ensemble(models_test, pool_test)
    r_base_ens = evaluate(d_base_ens, f"ベースライン(アンサンブル{len(SEEDS)}seed) score Top-K", min_reliable_n=MIN_RELIABLE_N,
                          common_sample_k=COMMON_SAMPLE_K, max_concurrent=MAX_CONCURRENT_POSITIONS)
    r_test_ens = evaluate(d_test_ens, f"テスト構成(アンサンブル{len(SEEDS)}seed) score Top-K", min_reliable_n=MIN_RELIABLE_N,
                          common_sample_k=COMMON_SAMPLE_K, max_concurrent=MAX_CONCURRENT_POSITIONS)
    # score(=p_win-p_stop)上位K件で選ぶ(2026-09-19、固定閾値0.7からの変更、evaluate()と同じ基準)
    # 決定論的tie-break(2026-09-19、evaluate()と同じ基準: score降順→ticker昇順→date昇順)
    sub_base_ens = d_base_ens.sort_values(['score', 'ticker', 'date'], ascending=[False, True, True]).head(min(COMMON_SAMPLE_K, len(d_base_ens)))
    sub_test_ens = d_test_ens.sort_values(['score', 'ticker', 'date'], ascending=[False, True, True]).head(min(COMMON_SAMPLE_K, len(d_test_ens)))
    if len(sub_base_ens) > 0:
        regime_breakdown(sub_base_ens, "ベースライン(アンサンブル)", regime_labels, min_reliable_n=MIN_RELIABLE_N)
    if len(sub_test_ens) > 0:
        regime_breakdown(sub_test_ens, "テスト構成(アンサンブル)", regime_labels, min_reliable_n=MIN_RELIABLE_N)
    print(f"\n  [判定・アンサンブル] baseline PF={r_base_ens['pf']:.2f} vs テスト構成 PF={r_test_ens['pf']:.2f} -> "
          f"{'テスト構成の方が良い' if r_test_ens['pf'] > r_base_ens['pf'] else 'ベースラインの方が良い'}"
          f"(単一seed毎の判定より、こちらの方が実運用のアンサンブル推論に近い)")
    print()

    # 日次Top-N運用評価(2026-09-19追加、ユーザー指摘「全期間Top-Kに加えて日次Top-Nを
    # 主運用評価として追加」)。全期間score上位K件は特定の数日にシグナルが偏っていても
    # 検知できないため、実際の運用(毎日DAILY_TOPN件しか執行できない)に忠実な評価を
    # アンサンブル予測に対して追加する。
    print("=" * 90)
    print(f"【日次Top-N運用評価(1日あたり{DAILY_TOPN}件、アンサンブル{len(SEEDS)}seed)】")
    print("=" * 90)
    r_base_daily = evaluate_daily_topn(d_base_ens, f"ベースライン(日次Top-{DAILY_TOPN})", n_per_day=DAILY_TOPN,
                                        min_reliable_n=MIN_RELIABLE_N, max_concurrent=MAX_CONCURRENT_POSITIONS)
    r_test_daily = evaluate_daily_topn(d_test_ens, f"テスト構成(日次Top-{DAILY_TOPN})", n_per_day=DAILY_TOPN,
                                        min_reliable_n=MIN_RELIABLE_N, max_concurrent=MAX_CONCURRENT_POSITIONS)
    print(f"\n  [判定・日次Top-N] baseline PF={r_base_daily['pf']:.2f} vs テスト構成 PF={r_test_daily['pf']:.2f} -> "
          f"{'テスト構成の方が良い' if r_test_daily['pf'] > r_base_daily['pf'] else 'ベースラインの方が良い'}"
          f"(全期間Top-Kより日々の執行に忠実。全期間Top-Kの判定と食い違う場合、全期間Top-Kの"
          f"結果が特定の数日に偏っていた可能性が高い)")
    print()

    # base/testで特徴量セットが違うとdropnaで落ちる(ticker,date)が微妙に異なり得るため、
    # 両方に共通するサンプルだけに絞った比較も出す(2026-09-19追加、ユーザー指摘。
    # これをしないと「モデルの質の差」と「評価サンプル自体の差」が混ざってしまう)。
    key_base = set(zip(d_base_ens['ticker'], d_base_ens['date']))
    key_test = set(zip(d_test_ens['ticker'], d_test_ens['date']))
    common_keys = key_base & key_test
    log(f"[*] base/testの(ticker,date)集合: base={len(key_base)} test={len(key_test)} common={len(common_keys)} "
        f"(baseのみ={len(key_base - key_test)}件, testのみ={len(key_test - key_base)}件)")
    if len(common_keys) > 0:
        mask_base_common = [k in common_keys for k in zip(d_base_ens['ticker'], d_base_ens['date'])]
        mask_test_common = [k in common_keys for k in zip(d_test_ens['ticker'], d_test_ens['date'])]
        print("=" * 90)
        print(f"【base/test共通(ticker,date)サブセットでの比較(アンサンブル、共通n={len(common_keys)}件)】")
        print("=" * 90)
        r_base_common = evaluate(d_base_ens[mask_base_common], "ベースライン(共通サブセット)", min_reliable_n=MIN_RELIABLE_N,
                                  common_sample_k=COMMON_SAMPLE_K, max_concurrent=MAX_CONCURRENT_POSITIONS)
        r_test_common = evaluate(d_test_ens[mask_test_common], "テスト構成(共通サブセット)", min_reliable_n=MIN_RELIABLE_N,
                                  common_sample_k=COMMON_SAMPLE_K, max_concurrent=MAX_CONCURRENT_POSITIONS)
        print(f"\n  [判定・共通サブセット] baseline PF={r_base_common['pf']:.2f} vs テスト構成 PF={r_test_common['pf']:.2f} -> "
              f"{'テスト構成の方が良い' if r_test_common['pf'] > r_base_common['pf'] else 'ベースラインの方が良い'}"
              f"(特徴量差によるdropnaパターンの違いを排除した、最も公平な比較)")
        print()

    # 以前はここで全seedの生予測を単純にpd.concatして「全seed pool」のregime別内訳を
    # 出していたが、異なるモデル(seed)の出力を1つの大きなサンプルとして扱うのは統計的に
    # 妥当でない(2026-09-19、ユーザー指摘で廃止)。regimeごとの内訳を安定して見たい場合は
    # 上のアンサンブル評価(複数モデルの確率平均、本番と同じ手法)の regime_breakdown を使う。

    df_base = pd.DataFrame(results_base)
    df_test = pd.DataFrame(results_test)

    print("=" * 90)
    print("【シード別サマリ(標準指標)】")
    print("=" * 90)
    cols_main = ['seed', 'n', 'win_rate', 'pf', 'avg_ret', 'max_dd', 'few_trades_flag']
    print("  -- ベースライン(既存14固定) --")
    print(df_base[cols_main].to_string(index=False))
    print("  -- テスト構成 --")
    print(df_test[cols_main].to_string(index=False))

    print("\n" + "=" * 90)
    print("【シード別サマリ(top_k_pf・月別一貫性・銘柄集中度)】")
    print("=" * 90)
    cols_extra = ['seed', 'top_k_pf', 'top_k_n', 'month_consistency', 'top1_pct', 'top5_pct', 'hhi']
    print("  -- ベースライン --")
    print(df_base[cols_extra].to_string(index=False))
    print("  -- テスト構成 --")
    print(df_test[cols_extra].to_string(index=False))

    print("\n" + "=" * 90)
    print("【seedごとのPF差分(テスト構成 - ベースライン)】")
    print("=" * 90)
    diff_df = pd.DataFrame({
        'seed': df_base['seed'],
        'pf_base': df_base['pf'], 'pf_test': df_test['pf'],
        'pf_diff': df_test['pf'] - df_base['pf'],
        'top_k_pf_base': df_base['top_k_pf'], 'top_k_pf_test': df_test['top_k_pf'],
        'top_k_pf_diff': df_test['top_k_pf'] - df_base['top_k_pf'],
    })
    print(diff_df.to_string(index=False))
    n_seeds_test_better = (diff_df['pf_diff'] > 0).sum()
    print(f"\n  テスト構成がベースラインを上回ったseed数: {n_seeds_test_better}/{len(SEEDS)}"
          f"(約定ベースPF基準) | {(diff_df['top_k_pf_diff'] > 0).sum()}/{len(SEEDS)}(top_k_pf基準)")

    print("\n" + "=" * 90)
    print("【mean ± std(組み合わせごとのシードばらつき比較用)】")
    print("=" * 90)
    finite_base_pf = df_base['pf'][np.isfinite(df_base['pf'])]
    finite_test_pf = df_test['pf'][np.isfinite(df_test['pf'])]
    print(f"  ベースライン PF: mean={finite_base_pf.mean():.3f} std={finite_base_pf.std():.3f} "
          f"(seed毎: {[round(x, 2) for x in df_base['pf']]})")
    print(f"  テスト構成   PF: mean={finite_test_pf.mean():.3f} std={finite_test_pf.std():.3f} "
          f"(seed毎: {[round(x, 2) for x in df_test['pf']]})")
    print(f"  ベースライン win_rate: mean={df_base['win_rate'].mean():.1f}% std={df_base['win_rate'].std():.1f}%")
    print(f"  テスト構成   win_rate: mean={df_test['win_rate'].mean():.1f}% std={df_test['win_rate'].std():.1f}%")
    print(f"  ベースライン MaxDD: mean={df_base['max_dd'].mean()*100:.1f}% std={df_base['max_dd'].std()*100:.1f}%")
    print(f"  テスト構成   MaxDD: mean={df_test['max_dd'].mean()*100:.1f}% std={df_test['max_dd'].std()*100:.1f}%")

    n_few_base = df_base['few_trades_flag'].sum()
    n_few_test = df_test['few_trades_flag'].sum()
    if n_few_base or n_few_test:
        print(f"\n  [!] 少数トレード警告(n<{MIN_RELIABLE_N}件): "
              f"ベースライン {n_few_base}/{len(SEEDS)}シード、テスト構成 {n_few_test}/{len(SEEDS)}シード"
              f" — 該当シードのPFは信頼性が低い可能性")

    print("\n" + "=" * 90)
    print(f"【判定・単一seed平均】baseline PF mean={finite_base_pf.mean():.2f} vs テスト構成 PF mean={finite_test_pf.mean():.2f} -> "
          f"{'テスト構成の方が良い' if finite_test_pf.mean() > finite_base_pf.mean() else 'ベースラインの方が良い(このテスト構成は不採用推奨)'}")
    print(f"    std比較: テスト構成のstdが{'大きい(不安定化)' if finite_test_pf.std() > finite_base_pf.std() else '小さい(安定化)'} "
          f"(baseline std={finite_base_pf.std():.3f} vs test std={finite_test_pf.std():.3f})")
    print(f"【判定・アンサンブル(実運用に近い)】baseline PF={r_base_ens['pf']:.2f} vs テスト構成 PF={r_test_ens['pf']:.2f} -> "
          f"{'テスト構成の方が良い' if r_test_ens['pf'] > r_base_ens['pf'] else 'ベースラインの方が良い'}"
          f" (単一seed平均とアンサンブルの判定が食い違う場合は、単一seedのばらつき自体が"
          f"regime依存の可能性があるため、アンサンブル側を優先すること — 2026-09-19確認)")
    print("=" * 90)
