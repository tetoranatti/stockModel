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
    def __init__(self, hidden_dim=20, num_heads=1, dropout=0.2, use_ffn=False, dim_ff=32, kv_dim=None):
        super().__init__()
        # kv_dim: query側(hidden_dim)とkey/value側で次元が異なる場合に指定する(2026-09-19追加、
        # 株GRU(64)+マクロMLP(32)のような非対称次元構成の実験用)。Noneなら従来通り
        # hidden_dimと同じ(nn.MultiheadAttentionはkdim=vdim=embed_dimのとき単一in_proj_weightの
        # 高速パスを使うため、既存呼び出し元のパラメータ構造・チェックポイント互換性に影響しない)。
        kv_dim = kv_dim if kv_dim is not None else hidden_dim
        self.norm_q = nn.LayerNorm(hidden_dim)
        self.norm_kv = nn.LayerNorm(kv_dim)
        self.mha = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout,
                                          batch_first=True, kdim=kv_dim, vdim=kv_dim)
        self.dropout = nn.Dropout(dropout)

        # self-attention側(PreLN_SelfAttentionBlock)と違い、元々はattention+残差のみで
        # FFNサブレイヤーが無い非対称な構造だった。use_ffn=Trueで対称化できる(2026-09-18追加、
        # 既定Falseで既存の挙動・チェックポイントとの互換性を維持)。
        self.use_ffn = use_ffn
        if use_ffn:
            self.norm2 = nn.LayerNorm(hidden_dim)
            self.ffn = nn.Sequential(
                nn.Linear(hidden_dim, dim_ff),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim_ff, hidden_dim)
            )
            self.drop2 = nn.Dropout(dropout)

    def forward(self, query, key_value):
        q_norm = self.norm_q(query)
        kv_norm = self.norm_kv(key_value)
        attn_out, _ = self.mha(query=q_norm, key=kv_norm, value=kv_norm)
        x = query + self.dropout(attn_out)
        if self.use_ffn:
            norm_x2 = self.norm2(x)
            ffn_out = self.ffn(norm_x2)
            x = x + self.drop2(ffn_out)
        return x

class DualStream_GRU_PreLN_Transformer(nn.Module):
    def __init__(self, stock_dim=5, macro_dim=5, hidden_dim=20, num_heads=1, num_classes=3, dropout=0.2,
                 use_cross_ffn=False, use_final_norm=False, num_self_attn_layers=1):
        super().__init__()
        self.stock_gru = nn.GRU(stock_dim, hidden_dim, batch_first=True, num_layers=1)
        self.macro_gru = nn.GRU(macro_dim, hidden_dim, batch_first=True, num_layers=1)

        self.stock_encoder = PreLN_SelfAttentionBlock(hidden_dim=hidden_dim, num_heads=num_heads, dim_ff=32, dropout=dropout)
        self.macro_encoder = PreLN_SelfAttentionBlock(hidden_dim=hidden_dim, num_heads=num_heads, dim_ff=32, dropout=dropout)
        # 1層目(stock_encoder/macro_encoder)は既存名のまま維持し、既存チェックポイントとの
        # 互換性を保つ。2層目以降だけ別のModuleListに積む(2026-09-19追加、
        # num_self_attn_layers=1なら空リスト=パラメータ0個で、既存state_dictと完全一致する)。
        self.stock_encoder_extra = nn.ModuleList([
            PreLN_SelfAttentionBlock(hidden_dim=hidden_dim, num_heads=num_heads, dim_ff=32, dropout=dropout)
            for _ in range(max(0, num_self_attn_layers - 1))
        ])
        self.macro_encoder_extra = nn.ModuleList([
            PreLN_SelfAttentionBlock(hidden_dim=hidden_dim, num_heads=num_heads, dim_ff=32, dropout=dropout)
            for _ in range(max(0, num_self_attn_layers - 1))
        ])

        self.cross_attn = PreLN_CrossAttentionBlock(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout,
                                                      use_ffn=use_cross_ffn)
        self.pool = DecayPooling(seq_len=10)

        # Pre-LN構造は各サブレイヤーの入力側しか正規化しないため、残差ストリームの出力スケールが
        # 際限なく成長し得る。use_final_norm=Trueでpooling直前に最終LayerNormを挟める
        # (2026-09-18追加、既定Falseで既存の挙動・チェックポイントとの互換性を維持)。
        self.use_final_norm = use_final_norm
        if use_final_norm:
            self.final_norm = nn.LayerNorm(hidden_dim)

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
        for layer in self.stock_encoder_extra:
            feat_s = layer(feat_s)
        feat_m = self.macro_encoder(h_m)
        for layer in self.macro_encoder_extra:
            feat_m = layer(feat_m)

        fused = self.cross_attn(feat_s, feat_m)
        if self.use_final_norm:
            fused = self.final_norm(fused)
        pooled = self.pool(fused)
        return self.classifier(pooled)


# ============================================================================
# 以下2クラスは「seq_len=10のように短い系列ではGRU+Transformerの多重直列は
# 過剰で、ノイズ耐性・汎化性能を損なっているのでは」という仮説を検証するための
# 実験用アーキテクチャ(train_model_v8_exp.py --arch=gru_only / transformer_only)。
# 本番のDualStream_GRU_PreLN_Transformerはそのまま維持し、比較対象として追加する。
# ============================================================================

class GRUOnlyModel(nn.Module):
    """GRU単体案: SelfAttention/CrossAttentionを撤去し、2本のGRUの出力を
    DecayPoolingで集約後、単純結合(concat)して分類する。"""
    def __init__(self, stock_dim=5, macro_dim=5, hidden_dim=20, num_classes=3, dropout=0.2, **kwargs):
        super().__init__()
        self.stock_gru = nn.GRU(stock_dim, hidden_dim, batch_first=True, num_layers=1)
        self.macro_gru = nn.GRU(macro_dim, hidden_dim, batch_first=True, num_layers=1)
        self.pool = DecayPooling(seq_len=10)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, num_classes)
        )

    def forward(self, x_stock, x_macro):
        h_s, _ = self.stock_gru(x_stock)
        h_m, _ = self.macro_gru(x_macro)
        fused = torch.cat([self.pool(h_s), self.pool(h_m)], dim=-1)
        return self.classifier(fused)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, seq_len=10):
        super().__init__()
        pe = torch.zeros(seq_len, hidden_dim)
        position = torch.arange(0, seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * (-np.log(10000.0) / hidden_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TransformerOnlyModel(nn.Module):
    """Transformer単体案: GRUを撤去し、代わりに線形射影+正弦波Positional Encodingで
    時系列位置情報を与えた上でSelfAttention/CrossAttentionに通す。"""
    def __init__(self, stock_dim=5, macro_dim=5, hidden_dim=20, num_heads=1, num_classes=3, dropout=0.2,
                 num_self_attn_layers=1, **kwargs):
        super().__init__()
        self.stock_proj = nn.Linear(stock_dim, hidden_dim)
        self.macro_proj = nn.Linear(macro_dim, hidden_dim)
        self.pos_enc = SinusoidalPositionalEncoding(hidden_dim, seq_len=10)

        self.stock_encoder = PreLN_SelfAttentionBlock(hidden_dim=hidden_dim, num_heads=num_heads, dim_ff=32, dropout=dropout)
        self.macro_encoder = PreLN_SelfAttentionBlock(hidden_dim=hidden_dim, num_heads=num_heads, dim_ff=32, dropout=dropout)
        # DualStream_GRU_PreLN_Transformerと同じパターン(2026-09-19追加): 1層目は既存名のまま、
        # 2層目以降だけ別のModuleListに積む(num_self_attn_layers=1なら空リストで既定挙動と同じ)。
        self.stock_encoder_extra = nn.ModuleList([
            PreLN_SelfAttentionBlock(hidden_dim=hidden_dim, num_heads=num_heads, dim_ff=32, dropout=dropout)
            for _ in range(max(0, num_self_attn_layers - 1))
        ])
        self.macro_encoder_extra = nn.ModuleList([
            PreLN_SelfAttentionBlock(hidden_dim=hidden_dim, num_heads=num_heads, dim_ff=32, dropout=dropout)
            for _ in range(max(0, num_self_attn_layers - 1))
        ])
        self.cross_attn = PreLN_CrossAttentionBlock(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
        self.pool = DecayPooling(seq_len=10)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, num_classes)
        )

    def forward(self, x_stock, x_macro):
        h_s = self.pos_enc(self.stock_proj(x_stock))
        h_m = self.pos_enc(self.macro_proj(x_macro))

        feat_s = self.stock_encoder(h_s)
        for layer in self.stock_encoder_extra:
            feat_s = layer(feat_s)
        feat_m = self.macro_encoder(h_m)
        for layer in self.macro_encoder_extra:
            feat_m = layer(feat_m)

        fused = self.cross_attn(feat_s, feat_m)
        pooled = self.pool(fused)
        return self.classifier(pooled)


class GRU_MacroMLP_CrossAttn_Model(nn.Module):
    """株側GRU+マクロ側MLP(系列非依存、各日を独立変換)+Cross-Attention+MLP Headという案
    (2026-09-19追加、ユーザー提案)。GRUOnlyModelの反省(株・マクロを同じhidden_dimで
    扱っていた)を受け、株側とマクロ側で次元を分ける(既定32/16)。自己注意層は無く、
    Cross-Attentionのみで株シーケンスにマクロ情報を注入する構造。
    マクロ側はSelf-Attention/GRUのような時系列内の相互作用を仮定せず、各日の値を
    独立に(全時点で重み共有の)MLPで変換するだけ——マクロ変数自体の日次値の意味は
    時点をまたいで比較する必要が薄いという仮説に基づく。"""
    def __init__(self, stock_dim=5, macro_dim=5, stock_hidden=32, macro_hidden=16, num_heads=4,
                 num_classes=3, dropout=0.2, use_cross_ffn=False, **kwargs):
        super().__init__()
        self.stock_gru = nn.GRU(stock_dim, stock_hidden, batch_first=True, num_layers=1)
        self.macro_mlp = nn.Sequential(
            nn.Linear(macro_dim, macro_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(macro_hidden, macro_hidden),
        )
        # use_cross_ffn: dual_streamでFFNサブレイヤー追加が明確に効いた実績があるため
        # (2026-09-19)、同じ改良をこちらのCross-Attentionにも試せるようにする。
        self.cross_attn = PreLN_CrossAttentionBlock(hidden_dim=stock_hidden, num_heads=num_heads,
                                                      dropout=dropout, kv_dim=macro_hidden,
                                                      use_ffn=use_cross_ffn)
        self.pool = DecayPooling(seq_len=10)
        self.classifier = nn.Sequential(
            nn.Linear(stock_hidden, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, num_classes)
        )

    def forward(self, x_stock, x_macro):
        h_s, _ = self.stock_gru(x_stock)   # (B, T, stock_hidden)
        h_m = self.macro_mlp(x_macro)      # (B, T, macro_hidden)、時点間で重み共有の独立変換
        fused = self.cross_attn(h_s, h_m)  # query=株, key/value=マクロ
        pooled = self.pool(fused)
        return self.classifier(pooled)
