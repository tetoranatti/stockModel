# modules/model_inference.py
import numpy as np
import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DecayPooling(nn.Module):
    def __init__(self, seq_len=10):
        super().__init__()
        weights = np.exp(np.linspace(-1.5, 0.0, seq_len))
        weights = weights / weights.sum()
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32).unsqueeze(0).unsqueeze(-1))

    def forward(self, x):
        return torch.sum(x * self.weights, dim=1)

class CrossAttentionBlock(nn.Module):
    def __init__(self, hidden_dim=24, num_heads=2, dropout=0.3):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key_value):
        attn_out, _ = self.mha(query=query, key=key_value, value=key_value)
        return self.norm(query + self.dropout(attn_out))

class DualStreamGRU_v6_Slim5(nn.Module):
    def __init__(self, stock_dim=5, macro_dim=5, hidden_dim=24, num_heads=2, num_classes=3):
        super().__init__()
        self.stock_gru = nn.GRU(stock_dim, hidden_dim, batch_first=True, num_layers=1)
        self.macro_gru = nn.GRU(macro_dim, hidden_dim, batch_first=True, num_layers=1)
        self.cross_attn = CrossAttentionBlock(hidden_dim, num_heads=num_heads, dropout=0.3)
        self.pool = DecayPooling(seq_len=10)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, num_classes)
        )

    def forward(self, x_stock, x_macro):
        out_stock, _ = self.stock_gru(x_stock)
        out_macro, _ = self.macro_gru(x_macro)
        fused = self.cross_attn(out_stock, out_macro)
        pooled = self.pool(fused)
        return self.classifier(pooled)

def load_trained_model(weights_path):
    checkpoint = torch.load(weights_path, map_location=DEVICE)
    stock_cols = checkpoint['stock_cols']
    macro_cols = checkpoint['macro_cols']
    hidden_dim = checkpoint.get('hidden_dim', 24)

    model = DualStreamGRU_v6_Slim5(
        stock_dim=len(stock_cols),
        macro_dim=len(macro_cols),
        hidden_dim=hidden_dim,
        num_classes=3
    ).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, stock_cols, macro_cols

def predict_probabilities(model, w_s, w_m):
    w_s_norm = (w_s - w_s.mean(axis=0)) / (w_s.std(axis=0) + 1e-7)
    t_s = torch.tensor(w_s_norm, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    t_m = torch.tensor(w_m, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probs = torch.softmax(model(t_s, t_m), dim=-1).squeeze(0).cpu().numpy()
    p_stop_raw, _, p_win_raw = float(probs[0]), float(probs[1]), float(probs[2])
    ev_raw = round(2.0 * p_win_raw - 1.0 * p_stop_raw, 3)
    return p_win_raw, p_stop_raw, ev_raw

def compute_supply_demand_factor(margin_item, curr_close, turnover_5d):
    if not margin_item:
        return 1.0, 0.0

    margin_buy = float(margin_item.get("margin_buy", 0))
    margin_ratio = float(margin_item.get("margin_ratio", 1.0))
    buy_pct = float(margin_item.get("margin_buy_pct", 0.0))

    buy_value = margin_buy * curr_close
    days_to_clear = round(buy_value / (turnover_5d + 1e-7), 2)

    score = 0.0
    if buy_pct <= -5.0:
        score += 0.08
    elif buy_pct <= -2.0:
        score += 0.04
    elif buy_pct >= 10.0:
        score -= 0.08
    elif buy_pct >= 5.0:
        score -= 0.04

    if days_to_clear <= 0.5:
        score += 0.03
    elif days_to_clear >= 4.0:
        if margin_ratio >= 10.0:
            score -= 0.10
        elif margin_ratio >= 5.0:
            score -= 0.05
    else:
        if margin_ratio <= 1.5:
            score += 0.05
        elif margin_ratio >= 10.0:
            score -= 0.06

    weight = 1.0 + max(-0.15, min(0.15, score))
    return round(weight, 3), days_to_clear

def apply_odds_adjustment(p_win, p_stop, weight):
    if weight == 1.0 or p_win <= 0.0:
        return p_win, p_stop

    odds_win = p_win / max(1.0 - p_win, 1e-5)
    adj_odds_win = odds_win * weight
    adj_p_win = adj_odds_win / (1.0 + adj_odds_win)
    adj_p_stop = p_stop * (2.0 - weight)

    adj_p_win = max(0.01, min(0.99, adj_p_win))
    adj_p_stop = max(0.01, min(0.99, adj_p_stop))

    return round(adj_p_win, 4), round(adj_p_stop, 4)