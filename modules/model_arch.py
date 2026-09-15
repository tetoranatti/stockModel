# modules/model_arch.py
# train_model_v8_exp.py / run_event_driven_backtest_v8_exp.py / model_inference.py で
# 共有するモデルアーキテクチャ定義 (実績最強の Pre-LN Transformer + Cross-Attention)。
# 3箇所に手動複製されていたクラス定義をここに一本化する。
import numpy as np
import torch
import torch.nn as nn

class DecayPooling(nn.Module):
    def __init__(self, seq_len=10):
        super().__init__()
        weights = np.exp(np.linspace(-1.5, 0.0, seq_len))
        weights = weights / weights.sum()
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32).unsqueeze(0).unsqueeze(-1))

    def forward(self, x):
        return torch.sum(x * self.weights, dim=1)

class PreLN_SelfAttentionBlock(nn.Module):
    def __init__(self, hidden_dim=20, num_heads=1, dim_ff=32, dropout=0.2):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, dim_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, hidden_dim)
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        norm_x = self.norm1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x)
        x = x + self.drop1(attn_out)

        norm_x2 = self.norm2(x)
        ffn_out = self.ffn(norm_x2)
        x = x + self.drop2(ffn_out)
        return x

class PreLN_CrossAttentionBlock(nn.Module):
    def __init__(self, hidden_dim=20, num_heads=1, dropout=0.2):
        super().__init__()
        self.norm_q = nn.LayerNorm(hidden_dim)
        self.norm_kv = nn.LayerNorm(hidden_dim)
        self.mha = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key_value):
        q_norm = self.norm_q(query)
        kv_norm = self.norm_kv(key_value)
        attn_out, _ = self.mha(query=q_norm, key=kv_norm, value=kv_norm)
        return query + self.dropout(attn_out)

class DualStream_GRU_PreLN_Transformer(nn.Module):
    def __init__(self, stock_dim=5, macro_dim=5, hidden_dim=20, num_heads=1, num_classes=3, dropout=0.2):
        super().__init__()
        self.stock_gru = nn.GRU(stock_dim, hidden_dim, batch_first=True, num_layers=1)
        self.macro_gru = nn.GRU(macro_dim, hidden_dim, batch_first=True, num_layers=1)

        self.stock_encoder = PreLN_SelfAttentionBlock(hidden_dim=hidden_dim, num_heads=num_heads, dim_ff=32, dropout=dropout)
        self.macro_encoder = PreLN_SelfAttentionBlock(hidden_dim=hidden_dim, num_heads=num_heads, dim_ff=32, dropout=dropout)

        self.cross_attn = PreLN_CrossAttentionBlock(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
        self.pool = DecayPooling(seq_len=10)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, num_classes)
        )

    def forward(self, x_stock, x_macro):
        h_s, _ = self.stock_gru(x_stock)
        h_m, _ = self.macro_gru(x_macro)

        feat_s = self.stock_encoder(h_s)
        feat_m = self.macro_encoder(h_m)

        fused = self.cross_attn(feat_s, feat_m)
        pooled = self.pool(fused)
        return self.classifier(pooled)
