"""Baseline model cache fingerprint."""

import hashlib
import json
from .config import *
from .feature_catalog import BASELINE_STOCK_COLS, BASELINE_MACRO_COLS

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
