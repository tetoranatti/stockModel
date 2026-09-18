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
    sys.stdout.reconfigure(line_buffering=True)  # バックグラウンド実行時、ログファイルへの
    sys.stderr.reconfigure(line_buffering=True)  # 書き出しがブロックバッファリングで遅延するのを防ぐ
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
    UniverseDataset, SameTickerBatchSampler, NearPairRankingLoss, _simulate_ret_pct,
    MAX_PAIR_GAP, UNIVERSE_PATH, MIN_CROSS_SECTION,
)
from modules.model_arch import DualStream_GRU_PreLN_Transformer, GRUOnlyModel, TransformerOnlyModel
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
    'rs60': False,
    'gap_strength_5': False,
    'atr_accel': False,
    # --- 株側候補(不合格、参考用) ---
    'atr_term_ratio': False,
    'dist_from_high60': False,
    'gap_avg_5d': False,
    'relative_strength_5d': False,
    'rs20': False,
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
    'usdjpy_chg': False,
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
    'rolling_beta_x_cta_net_norm_x_market_vol_regime': True,
}

MODEL_ARCH = "dual_stream"  # "dual_stream"(本番) / "gru_only" / "transformer_only"
MODEL_HIDDEN_DIM = 20   # 20:本番と同じ既定値。超軽量版を試すならここを変える(num_headsで割り切れる値に)
MODEL_NUM_HEADS = 1     # 1:本番と同じ既定値。gru_onlyでは無視される
USE_CROSS_FFN = False   # Cross-Attention後にFFNサブレイヤーを追加するか(dual_stream限定、既定False=本番と同じ)
USE_FINAL_NORM = False  # Cross-Attention後・pooling前に最終LayerNormを追加するか(dual_stream限定、既定False=本番と同じ)
SEEDS = [42]    # 複数シードでmean/stdを見る(組み合わせによってシード間のばらつきが
                        # 変わるか比較したい場合はここを増減する。単発でよければ[42]だけにする)
PROFILE_FIRST_EPOCH = False  # Trueにすると、最初のシードのベースライン学習の epoch=1 だけ
                             # torch.profilerで計測し、処理時間トップ10を表示する(それ以外は
                             # 計測オーバーヘッド無し。データ準備 vs 学習 vs 検証のどこが重いか
                             # 見たい時だけTrueにする)
# ============================================================================

ARCH_CLASSES = {
    "dual_stream": DualStream_GRU_PreLN_Transformer,
    "gru_only": GRUOnlyModel,
    "transformer_only": TransformerOnlyModel,
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
SEQ_LEN, HOLDING_PERIOD = 10, 10


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
        except Exception:
            continue
    return per_ticker_df


def build_dataset(macro_cols, stock_cols, pool, margin_pool, sector_pool, macro_pool_df):
    per_ticker_df = build_per_ticker(pool, margin_pool, sector_pool, macro_pool_df, stock_cols)
    macro_feed = macro_pool_df[macro_cols]
    for t in list(per_ticker_df.keys()):
        df = per_ticker_df[t]
        df = df.join(macro_feed, how='inner')
        needed = [c for c in dict.fromkeys(['ATR'] + list(stock_cols) + list(macro_cols)) if c in df.columns]
        df = df.dropna(subset=needed)
        if len(df) < SEQ_LEN + HOLDING_PERIOD + 10:
            del per_ticker_df[t]; continue
        per_ticker_df[t] = apply_log_transform(df)

    cross_mean, cross_std = compute_cross_sectional_stats(per_ticker_df, stock_cols)
    valid_dates = valid_cross_section_dates(per_ticker_df, stock_cols, MIN_CROSS_SECTION)

    tr_x_s, tr_x_m, tr_y, tr_tid, tr_tidx = [], [], [], [], []
    va_x_s, va_x_m, va_y, va_tid, va_tidx = [], [], [], [], []
    for i, (t, df) in enumerate(per_ticker_df.items()):
        try:
            df = df.loc[df.index.isin(valid_dates)].copy()
            if len(df) < SEQ_LEN + HOLDING_PERIOD + 10:
                continue
            norm_s = normalize_cross_sectional(df, cross_mean, cross_std, stock_cols)
            closes, highs, lows, atrs = df['Close'].values, df['High'].values, df['Low'].values, df['ATR'].values
            targets = _simulate_ret_pct(closes, highs, lows, atrs, HOLDING_PERIOD)
            vals_s_norm = norm_s.values
            vals_m = df[macro_cols].values
            n_samples = len(df)
            split_idx = int(n_samples * 0.75)

            for idx in range(SEQ_LEN - 1, n_samples):
                if np.isnan(targets[idx]):
                    continue
                w_s = vals_s_norm[idx - SEQ_LEN + 1: idx + 1].copy()
                if np.isnan(w_s).any():
                    continue
                w_m = vals_m[idx - SEQ_LEN + 1: idx + 1].copy()
                target = targets[idx]
                if idx < split_idx:
                    tr_x_s.append(w_s); tr_x_m.append(w_m); tr_y.append(target); tr_tid.append(i); tr_tidx.append(idx)
                else:
                    va_x_s.append(w_s); va_x_m.append(w_m); va_y.append(target); va_tid.append(i); va_tidx.append(idx)
        except Exception:
            continue

    return (np.array(tr_x_s), np.array(tr_x_m), np.array(tr_y), np.array(tr_tid), np.array(tr_tidx)), \
           (np.array(va_x_s), np.array(va_x_m), np.array(va_y), np.array(va_tid), np.array(va_tidx))


def prepare_backtest_pool(macro_cols, stock_cols, pool, margin_pool, sector_pool, macro_pool_df):
    per_ticker_df = build_per_ticker(pool, margin_pool, sector_pool, macro_pool_df, stock_cols)
    macro_feed = macro_pool_df[macro_cols]
    for t in list(per_ticker_df.keys()):
        df = per_ticker_df[t]
        df = df.join(macro_feed, how='inner')
        needed = [c for c in dict.fromkeys(['ATR'] + list(stock_cols) + list(macro_cols)) if c in df.columns]
        df = df.dropna(subset=needed)
        if len(df) < SEQ_LEN + HOLDING_PERIOD + 10:
            del per_ticker_df[t]; continue
        per_ticker_df[t] = apply_log_transform(df)

    cross_mean, cross_std = compute_cross_sectional_stats(per_ticker_df, stock_cols)
    valid_dates = valid_cross_section_dates(per_ticker_df, stock_cols, MIN_CROSS_SECTION)

    pool_out = []
    for t, df in per_ticker_df.items():
        try:
            df = df.loc[df.index.isin(valid_dates)].copy()
            if len(df) < SEQ_LEN + HOLDING_PERIOD + 10:
                continue
            norm_s = normalize_cross_sectional(df, cross_mean, cross_std, stock_cols)
            n_samples = len(df)
            split_idx = int(n_samples * 0.75)
            vals_s_norm = norm_s.values
            vals_m = df[macro_cols].values
            closes, highs, lows, atrs = df['Close'].values, df['High'].values, df['Low'].values, df['ATR'].values
            dates = df.index

            for idx in range(split_idx, n_samples - HOLDING_PERIOD):
                w_s = vals_s_norm[idx - SEQ_LEN + 1: idx + 1].copy()
                if np.isnan(w_s).any():
                    continue
                w_m = vals_m[idx - SEQ_LEN + 1: idx + 1].copy()

                d_close = closes[idx]
                entry_p = closes[idx + 1]  # D+1引け(MOC)
                upper_p, lower_p = d_close + 2.0 * atrs[idx], d_close - 1.0 * atrs[idx]
                exit_p = None
                for h in range(2, HOLDING_PERIOD + 1):
                    hi, lo = highs[idx + h], lows[idx + h]
                    if lo <= lower_p and hi >= upper_p:
                        exit_p = lower_p; break
                    elif hi >= upper_p:
                        exit_p = upper_p; break
                    elif lo <= lower_p:
                        exit_p = lower_p; break
                if exit_p is None:
                    exit_p = closes[idx + HOLDING_PERIOD]
                ret_pct = (exit_p - entry_p) / entry_p
                pool_out.append((t, dates[idx], w_s, w_m, ret_pct))
        except Exception:
            continue
    return pool_out


def train_model(seed, tr_data, va_data, s_cols, m_cols, label, profile_this_call=False):
    tr_x_s, tr_x_m, tr_y, tr_tid, tr_tidx = tr_data
    va_x_s, va_x_m, va_y, va_tid, va_tidx = va_data
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
    va_xs_t = torch.tensor(va_x_s, dtype=torch.float32, device=DEVICE)
    va_xm_t = torch.tensor(va_x_m, dtype=torch.float32, device=DEVICE)
    va_y_t = torch.tensor(va_y, dtype=torch.float32, device=DEVICE)
    va_tidx_t = torch.tensor(va_tidx, dtype=torch.long, device=DEVICE)

    train_sampler = SameTickerBatchSampler(tr_tid, 128, shuffle=True)
    val_batches = list(SameTickerBatchSampler(va_tid, 256, shuffle=False))

    arch_cls = ARCH_CLASSES[MODEL_ARCH]
    model_kwargs = dict(stock_dim=len(s_cols), macro_dim=len(m_cols), hidden_dim=MODEL_HIDDEN_DIM,
                         num_heads=MODEL_NUM_HEADS, num_classes=3, dropout=0.2)
    if MODEL_ARCH == "dual_stream":
        model_kwargs['use_cross_ffn'] = USE_CROSS_FFN
        model_kwargs['use_final_norm'] = USE_FINAL_NORM
    # gru_only/transformer_onlyにはuse_cross_ffn/use_final_norm概念が無いので渡さない
    # (**kwargsを持つので万一渡しても無視されるが、意図を明確にするため分岐している)
    model = arch_cls(**model_kwargs).to(DEVICE)
    criterion = NearPairRankingLoss(max_gap=MAX_PAIR_GAP)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0002, weight_decay=3e-2)

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
        total_tr = torch.zeros((), device=DEVICE)
        n_tr_batches = 0
        for batch_idx in train_sampler:
            b_xs, b_xm, b_y, b_tidx = tr_xs_t[batch_idx], tr_xm_t[batch_idx], tr_y_t[batch_idx], tr_tidx_t[batch_idx]
            optimizer.zero_grad()
            loss = criterion(model(b_xs, b_xm), b_y, b_tidx)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_tr += loss.detach(); n_tr_batches += 1
        train_loss = (total_tr / n_tr_batches).item() if n_tr_batches else float('inf')

        if do_profile:
            prof.__exit__(None, None, None)
            sort_key = "self_cuda_time_total" if torch.cuda.is_available() else "self_cpu_time_total"
            log(f"[Profiler] [{label}] epoch1 学習ループ({n_tr_batches}バッチ)処理時間トップ10:")
            print(prof.key_averages().table(sort_by=sort_key, row_limit=10))

        model.eval()
        # .item()は毎回GPU-CPU同期を発生させるので、バッチごとには呼ばずテンソルのまま
        # 累積し、検証ループ終了後に1回だけ.item()する(2026-09-18、ユーザー指摘で修正)。
        total_va = torch.zeros((), device=DEVICE)
        n_batches = 0
        with torch.no_grad():
            for batch_idx in val_batches:
                b_xs, b_xm, b_y, b_tidx = va_xs_t[batch_idx], va_xm_t[batch_idx], va_y_t[batch_idx], va_tidx_t[batch_idx]
                loss = criterion(model(b_xs, b_xm), b_y, b_tidx)
                total_va += loss; n_batches += 1
        val_loss = (total_va / n_batches).item() if n_batches else float('inf')

        if val_loss < best_val_loss:
            best_val_loss, best_state, patience_cnt = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
            marker = " <- best"
        else:
            patience_cnt += 1
            marker = f" (patience {patience_cnt}/{patience})"
        log(f"  [{label}] epoch {epoch:2d}/{epochs}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}{marker}")
        if patience_cnt >= patience:
            break
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
            w_s_batch = np.stack([item[2] for item in chunk])
            w_m_batch = np.stack([item[3] for item in chunk])
            t_s = torch.tensor(w_s_batch, dtype=torch.float32).to(DEVICE)
            t_m = torch.tensor(w_m_batch, dtype=torch.float32).to(DEVICE)
            probs = torch.softmax(model(t_s, t_m), dim=-1).cpu().numpy()
            for j, (t, date, w_s, w_m, ret_pct) in enumerate(chunk):
                records.append({'ticker': t, 'date': date, 'p_win': float(probs[j, 2]),
                                 'p_stop': float(probs[j, 0]), 'ret_pct': ret_pct})
    return pd.DataFrame(records)


def summarize(sub, label):
    if len(sub) == 0:
        print(f"  {label}: n=0")
        return 0, 0.0, float('nan')
    wins, losses = sub[sub['ret_pct'] > 0], sub[sub['ret_pct'] < 0]
    win_rate = len(wins) / len(sub) * 100
    pf = wins['ret_pct'].sum() / abs(losses['ret_pct'].sum()) if len(losses) and losses['ret_pct'].sum() != 0 else float('inf')
    print(f"  {label}: n={len(sub):5d} win_rate={win_rate:5.1f}% PF={pf:5.2f} avg_ret={sub['ret_pct'].mean():+.4f}")
    return len(sub), win_rate, pf


def monthly_breakdown(sub, label):
    sub = sub.copy()
    sub['month'] = pd.to_datetime(sub['date']).dt.to_period('M')
    print(f"\n  [月別内訳: {label}]")
    for month, g in sub.groupby('month'):
        wins, losses = g[g['ret_pct'] > 0], g[g['ret_pct'] < 0]
        win_rate = len(wins) / len(g) * 100
        pf = wins['ret_pct'].sum() / abs(losses['ret_pct'].sum()) if len(losses) and losses['ret_pct'].sum() != 0 else float('inf')
        print(f"    {month}: n={len(g):4d} win_rate={win_rate:5.1f}% PF={pf:5.2f}")


if __name__ == "__main__":
    removed_from_baseline = (set(BASELINE_STOCK_COLS) | set(BASELINE_MACRO_COLS)) - set(TEST_STOCK_COLS + TEST_MACRO_COLS)
    added_candidates = set(TEST_STOCK_COLS + TEST_MACRO_COLS) - (set(BASELINE_STOCK_COLS) | set(BASELINE_MACRO_COLS))
    print(f"[*] テスト側 株{len(TEST_STOCK_COLS)}個 + マクロ{len(TEST_MACRO_COLS)}個 = 計{len(TEST_FEATS)}個")
    print(f"    既存から外した: {sorted(removed_from_baseline) if removed_from_baseline else '(なし)'}")
    print(f"    新規追加した候補: {sorted(added_candidates) if added_candidates else '(なし)'}")
    print(f"[*] モデル設定: hidden_dim={MODEL_HIDDEN_DIM}, num_heads={MODEL_NUM_HEADS}, "
          f"use_cross_ffn={USE_CROSS_FFN}, use_final_norm={USE_FINAL_NORM}, seeds={SEEDS}\n")

    with open(UNIVERSE_PATH, "r", encoding="utf-8") as f:
        tickers = [line.strip() for line in f if line.strip()]

    macro_df = load_macro_slim5(return_source="nk225")
    pool = get_full_feature_pool_df(tickers, macro_df)
    margin_pool = get_margin_pool_df(tickers, macro_df)
    sector_pool = get_sector_relative_pool_df(tickers, macro_df)
    macro_pool_df = augment_macro_pool_df(get_full_macro_pool_df())

    # データセット・バックテストプールはシードに依存しないので、シードループの外で1回だけ構築する
    log("[*] データセット・バックテストプール構築(シード非依存、1回だけ)...")
    tr0, va0 = build_dataset(BASELINE_MACRO_COLS, BASELINE_STOCK_COLS, pool, margin_pool, sector_pool, macro_pool_df)
    tr1, va1 = build_dataset(TEST_MACRO_COLS, TEST_STOCK_COLS, pool, margin_pool, sector_pool, macro_pool_df)
    pool_base = prepare_backtest_pool(BASELINE_MACRO_COLS, BASELINE_STOCK_COLS, pool, margin_pool, sector_pool, macro_pool_df)
    pool_test = prepare_backtest_pool(TEST_MACRO_COLS, TEST_STOCK_COLS, pool, margin_pool, sector_pool, macro_pool_df)
    log(f"[+] train0={len(tr0[2])} val0={len(va0[2])} / train1={len(tr1[2])} val1={len(va1[2])}")
    log(f"[+] backtest_pool base={len(pool_base)} test={len(pool_test)}")

    results_base, results_test = [], []
    for si, seed in enumerate(SEEDS):
        log(f"=== seed={seed} ===")
        model_base = train_model(seed, tr0, va0, BASELINE_STOCK_COLS, BASELINE_MACRO_COLS, "ベースライン(既存14固定)",
                                  profile_this_call=(PROFILE_FIRST_EPOCH and si == 0))
        d_base = run_backtest_inference(model_base, pool_base)
        model_test = train_model(seed, tr1, va1, TEST_STOCK_COLS, TEST_MACRO_COLS, f"テスト構成({len(TEST_FEATS)}個)")
        d_test = run_backtest_inference(model_test, pool_test)

        sub_base = d_base[(d_base['p_win'] >= 0.700) & (d_base['p_win'] > d_base['p_stop'])]
        sub_test = d_test[(d_test['p_win'] >= 0.700) & (d_test['p_win'] > d_test['p_stop'])]

        n_base, wr_base, pf_base = summarize(sub_base, f"seed={seed} ベースライン STRONG BUY")
        n_test, wr_test, pf_test = summarize(sub_test, f"seed={seed} テスト構成 STRONG BUY")
        results_base.append({'seed': seed, 'n': n_base, 'win_rate': wr_base, 'pf': pf_base})
        results_test.append({'seed': seed, 'n': n_test, 'win_rate': wr_test, 'pf': pf_test})
        print()

    df_base = pd.DataFrame(results_base)
    df_test = pd.DataFrame(results_test)

    print("=" * 70)
    print("【シード別サマリ】")
    print("=" * 70)
    print("  -- ベースライン(既存14固定) --")
    print(df_base.to_string(index=False))
    print("  -- テスト構成 --")
    print(df_test.to_string(index=False))

    print("\n" + "=" * 70)
    print("【mean ± std(組み合わせごとのシードばらつき比較用)】")
    print("=" * 70)
    finite_base_pf = df_base['pf'][np.isfinite(df_base['pf'])]
    finite_test_pf = df_test['pf'][np.isfinite(df_test['pf'])]
    print(f"  ベースライン PF: mean={finite_base_pf.mean():.3f} std={finite_base_pf.std():.3f} "
          f"(seed毎: {[round(x, 2) for x in df_base['pf']]})")
    print(f"  テスト構成   PF: mean={finite_test_pf.mean():.3f} std={finite_test_pf.std():.3f} "
          f"(seed毎: {[round(x, 2) for x in df_test['pf']]})")
    print(f"  ベースライン win_rate: mean={df_base['win_rate'].mean():.1f}% std={df_base['win_rate'].std():.1f}%")
    print(f"  テスト構成   win_rate: mean={df_test['win_rate'].mean():.1f}% std={df_test['win_rate'].std():.1f}%")

    print("\n" + "=" * 70)
    print(f"【判定】baseline PF mean={finite_base_pf.mean():.2f} vs テスト構成 PF mean={finite_test_pf.mean():.2f} -> "
          f"{'テスト構成の方が良い' if finite_test_pf.mean() > finite_base_pf.mean() else 'ベースラインの方が良い(このテスト構成は不採用推奨)'}")
    print(f"    std比較: テスト構成のstdが{'大きい(不安定化)' if finite_test_pf.std() > finite_base_pf.std() else '小さい(安定化)'} "
          f"(baseline std={finite_base_pf.std():.3f} vs test std={finite_test_pf.std():.3f})")
    print("=" * 70)
