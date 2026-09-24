"""5日後Alpha回帰モデルの構築(2026-09-24追加)。既存のDualStream_GRU_PreLN_Transformerを
num_classes=1(スカラー回帰出力、softmax無し)でそのまま使う——アーキテクチャ自体の変更は
不要。stage3_exp.model_factory.build_model(num_classes=3固定、3値分類用)とは別の
エントリポイントとして用意する。"""
from modules.model_arch import DualStream_GRU_PreLN_Transformer
from stage3_exp.config import (
    MODEL_HIDDEN_DIM, MODEL_NUM_HEADS, USE_CROSS_FFN, USE_FINAL_NORM,
    NUM_SELF_ATTN_LAYERS, USE_SECTOR_EMBEDDING, SECTOR_EMBED_DIM, SEQ_LEN,
)
from stage3_exp.feature_catalog import NUM_SECTORS
from stage3_exp.runtime import DEVICE


def build_alpha_model(s_cols, m_cols):
    stock_dim = len(s_cols) + (1 if USE_SECTOR_EMBEDDING else 0)
    model_kwargs = dict(
        stock_dim=stock_dim, macro_dim=len(m_cols), hidden_dim=MODEL_HIDDEN_DIM,
        num_heads=MODEL_NUM_HEADS, num_classes=1, dropout=0.2, seq_len=SEQ_LEN,
        use_cross_ffn=USE_CROSS_FFN, use_final_norm=USE_FINAL_NORM,
        num_self_attn_layers=NUM_SELF_ATTN_LAYERS,
    )
    if USE_SECTOR_EMBEDDING:
        model_kwargs["num_sectors"] = NUM_SECTORS
        model_kwargs["sector_embed_dim"] = SECTOR_EMBED_DIM
    return DualStream_GRU_PreLN_Transformer(**model_kwargs).to(DEVICE)
