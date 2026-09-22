"""Model construction."""

from modules.model_arch import (DualStream_GRU_PreLN_Transformer, GRUOnlyModel,
    TransformerOnlyModel, GRU_MacroMLP_CrossAttn_Model)
from .config import (MODEL_ARCH, MODEL_HIDDEN_DIM, MODEL_NUM_HEADS, STOCK_HIDDEN_DIM,
    MACRO_HIDDEN_DIM, CROSS_ATTN_NUM_HEADS, CROSS_ATTN_USE_FFN, USE_CROSS_FFN,
    USE_FINAL_NORM, NUM_SELF_ATTN_LAYERS, USE_SECTOR_EMBEDDING, SECTOR_EMBED_DIM, SEQ_LEN)
from .feature_catalog import NUM_SECTORS
from .runtime import DEVICE

ARCH_CLASSES = {
    "dual_stream": DualStream_GRU_PreLN_Transformer,
    "gru_only": GRUOnlyModel,
    "transformer_only": TransformerOnlyModel,
    "gru_macro_mlp_cross": GRU_MacroMLP_CrossAttn_Model,
}
if MODEL_ARCH not in ARCH_CLASSES:
    raise ValueError(f"MODEL_ARCH='{MODEL_ARCH}'は未対応です。選択肢: {list(ARCH_CLASSES)}")

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
