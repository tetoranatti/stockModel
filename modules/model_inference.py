# modules/model_inference.py
import numpy as np
import torch

from modules.model_arch import DualStream_GRU_PreLN_Transformer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_trained_model(weights_path):
    checkpoint = torch.load(weights_path, map_location=DEVICE)
    stock_cols = checkpoint['stock_cols']
    macro_cols = checkpoint['macro_cols']
    hidden_dim = checkpoint.get('hidden_dim', 20)
    num_heads = checkpoint.get('num_heads', 1)
    dropout = checkpoint.get('dropout', 0.2)

    model = DualStream_GRU_PreLN_Transformer(
        stock_dim=len(stock_cols),
        macro_dim=len(macro_cols),
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_classes=3,
        dropout=dropout
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