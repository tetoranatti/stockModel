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
    # use_cross_ffn=Falseが既定なので、この鍵を持たない旧チェックポイントも
    # そのまま正しく再構築できる(2026-09-19、FFN本採用に伴い追加)。
    use_cross_ffn = checkpoint.get('use_cross_ffn', False)

    model = DualStream_GRU_PreLN_Transformer(
        stock_dim=len(stock_cols),
        macro_dim=len(macro_cols),
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_classes=3,
        dropout=dropout,
        use_cross_ffn=use_cross_ffn
    ).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, stock_cols, macro_cols

def load_trained_models_ensemble(weights_paths):
    """複数シードのチェックポイントを読み込み、モデルのリストを返す。
    全メンバーがstock_cols/macro_colsを共有している前提(学習時のstock_cols構成が同じ)。"""
    models = []
    stock_cols = macro_cols = None
    for path in weights_paths:
        model, s_cols, m_cols = load_trained_model(path)
        if stock_cols is None:
            stock_cols, macro_cols = s_cols, m_cols
        elif s_cols != stock_cols or m_cols != macro_cols:
            raise ValueError(f"[!] アンサンブルメンバー間でstock_cols/macro_colsが不一致: {path}")
        models.append(model)
    return models, stock_cols, macro_cols

def predict_probabilities(model, w_s, w_m):
    """w_s は呼び出し側で正規化済み(横断面正規化)であることを前提とする。
    このモジュール内では正規化を行わない(横断面統計はuniverse全体の文脈が
    必要なため、単一銘柄しか見ないこの関数では計算できない)。"""
    t_s = torch.tensor(w_s, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    t_m = torch.tensor(w_m, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(t_s, t_m), dim=-1).squeeze(0).cpu().numpy()
    p_stop_raw, _, p_win_raw = float(probs[0]), float(probs[1]), float(probs[2])
    ev_raw = round(2.0 * p_win_raw - 1.0 * p_stop_raw, 3)
    return p_win_raw, p_stop_raw, ev_raw

def predict_probabilities_ensemble(models, w_s, w_m):
    """アンサンブル全メンバーのsoftmax確率を平均してp_win/p_stopを算出する。
    w_s は呼び出し側で正規化済み(横断面正規化)であることを前提とする。"""
    t_s = torch.tensor(w_s, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    t_m = torch.tensor(w_m, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    probs_sum = None
    with torch.no_grad():
        for model in models:
            probs = torch.softmax(model(t_s, t_m), dim=-1).squeeze(0).cpu().numpy()
            probs_sum = probs if probs_sum is None else probs_sum + probs
    probs_avg = probs_sum / len(models)
    p_stop_raw, _, p_win_raw = float(probs_avg[0]), float(probs_avg[1]), float(probs_avg[2])
    ev_raw = round(2.0 * p_win_raw - 1.0 * p_stop_raw, 3)
    return p_win_raw, p_stop_raw, ev_raw

def predict_probabilities_batch(model, w_s_batch, w_m_batch):
    """predict_probabilitiesのバッチ版。w_s_batch/w_m_batchはshape (N, seq_len, dim)。
    戻り値はp_win/p_stop/evそれぞれshape (N,) のnumpy配列(2026-09-18追加、大量サンプルを
    1件ずつ推論していたループを一括化して高速化するため。.eval()+LayerNormのみ
    (BatchNorm不使用)なので、バッチの切り方に関わらず1件ずつ推論した場合と
    数値的に完全一致する)。"""
    t_s = torch.tensor(w_s_batch, dtype=torch.float32).to(DEVICE)
    t_m = torch.tensor(w_m_batch, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(t_s, t_m), dim=-1).cpu().numpy()
    p_stop_raw, p_win_raw = probs[:, 0], probs[:, 2]
    ev_raw = np.round(2.0 * p_win_raw - 1.0 * p_stop_raw, 3)
    return p_win_raw, p_stop_raw, ev_raw

def predict_probabilities_ensemble_batch(models, w_s_batch, w_m_batch):
    """predict_probabilities_ensembleのバッチ版。w_s_batch/w_m_batchはshape (N, seq_len, dim)。
    戻り値はp_win/p_stop/evそれぞれshape (N,) のnumpy配列(2026-09-18追加、理由・数値的
    等価性の根拠はpredict_probabilities_batchと同じ)。"""
    t_s = torch.tensor(w_s_batch, dtype=torch.float32).to(DEVICE)
    t_m = torch.tensor(w_m_batch, dtype=torch.float32).to(DEVICE)
    probs_sum = None
    with torch.no_grad():
        for model in models:
            probs = torch.softmax(model(t_s, t_m), dim=-1).cpu().numpy()
            probs_sum = probs if probs_sum is None else probs_sum + probs
    probs_avg = probs_sum / len(models)
    p_stop_raw, p_win_raw = probs_avg[:, 0], probs_avg[:, 2]
    ev_raw = np.round(2.0 * p_win_raw - 1.0 * p_stop_raw, 3)
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