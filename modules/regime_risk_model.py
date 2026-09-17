# modules/regime_risk_model.py
# 日次集約(銘柄横断ではなく「その日全体」)の地合い危険度モデル。
# 個別銘柄のp_stop較正はどの手法(3クラスsoftmax/独立sigmoid/ツリー等)でも
# 機能しなかった(AUC≈0.50〜0.57)一方、日次集約したSTOP率は市場全体の
# ボラティリティ・地合い指標(VIX/指数リターン/CTA/ブレッド)で予測でき
# (5シードアンサンブルAUC≈0.64)、この用途にはこちらを使う。
# USD/JPY特徴量(Δ1日/5日・20日ボラ・EMA(5,20)差=トレンド強度)を追加検証したところ
# AUC 0.641→0.668、PR-AUC 0.752→0.779まで改善したため採用(5シードアンサンブルで確認済み)。
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

BASE_DIR = r"F:\stockModel"
VIX_CSV_PATH = os.path.join(BASE_DIR, "data", "vix_fred.csv")
USDJPY_CSV_PATH = os.path.join(BASE_DIR, "data", "usdjpy_fred.csv")

REGIME_COLS = [
    'cta_net_norm', 'idx_ret_5d', 'idx_ret_20d', 'nk_ret_norm', 'vix_level_norm', 'vix_change_norm', 'breadth_5d',
    'usdjpy_ret_1d', 'usdjpy_ret_5d', 'usdjpy_vol_20d', 'usdjpy_ema_diff',
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TinyRegimeMLP(nn.Module):
    def __init__(self, input_dim, hidden=16, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def add_regime_features(macro_df):
    """load_macro_slim5()/load_macro_environment()の出力にidx_ret_5d/20d・VIX特徴量を追加する。
    (breadth_5dは銘柄横断データが要るため別途add_breadth_featureで追加する)"""
    macro_df = macro_df.copy()
    macro_df['idx_ret_5d'] = macro_df['NK_Close'].pct_change(5, fill_method=None).fillna(0.0)
    macro_df['idx_ret_20d'] = macro_df['NK_Close'].pct_change(20, fill_method=None).fillna(0.0)

    vix = pd.read_csv(VIX_CSV_PATH, index_col=0, parse_dates=True)['VIXCLS']
    vix = vix.reindex(macro_df.index).ffill()
    macro_df['vix_level_norm'] = (vix - 17.0) / 4.4
    macro_df['vix_change_norm'] = vix.diff().fillna(0.0) / 1.8

    usdjpy = pd.read_csv(USDJPY_CSV_PATH, index_col=0, parse_dates=True)['DEXJPUS']
    usdjpy = usdjpy.reindex(macro_df.index).ffill()
    ret_1d_raw = usdjpy.pct_change(1, fill_method=None).fillna(0.0)
    macro_df['usdjpy_ret_1d'] = ret_1d_raw / 0.006
    macro_df['usdjpy_ret_5d'] = (usdjpy.pct_change(5, fill_method=None).fillna(0.0)) / 0.013
    vol_20d = ret_1d_raw.rolling(20).std().fillna(0.006)
    macro_df['usdjpy_vol_20d'] = (vol_20d / 0.006) - 1.0
    ema_short = usdjpy.ewm(span=5, adjust=False).mean()
    ema_long = usdjpy.ewm(span=20, adjust=False).mean()
    macro_df['usdjpy_ema_diff'] = ((ema_short - ema_long) / usdjpy / 0.01).fillna(0.0)

    return macro_df


def compute_breadth_5d(per_ticker_ret1d, index):
    """per_ticker_ret1d: {ticker: stock_ret_1dのSeries} の辞書。
    同日の値上がり銘柄比率を5日平均で平滑化したSeries(index基準)を返す。"""
    ret1d_df = pd.DataFrame(per_ticker_ret1d)
    advance_ratio = (ret1d_df > 0).sum(axis=1) / ret1d_df.notna().sum(axis=1).clip(lower=1)
    return advance_ratio.reindex(index).rolling(5, min_periods=1).mean().fillna(0.5)


def load_regime_risk_ensemble(weights_paths):
    models = []
    feat_mean = feat_std = regime_cols = None
    for path in weights_paths:
        checkpoint = torch.load(path, map_location=DEVICE)
        r_cols = checkpoint['regime_cols']
        if regime_cols is None:
            regime_cols = r_cols
            feat_mean = checkpoint['feat_mean'].cpu().numpy()
            feat_std = checkpoint['feat_std'].cpu().numpy()
        elif r_cols != regime_cols:
            raise ValueError(f"[!] アンサンブルメンバー間でregime_colsが不一致: {path}")
        model = TinyRegimeMLP(input_dim=len(r_cols)).to(DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        models.append(model)
    return models, regime_cols, feat_mean, feat_std


def predict_regime_risk(models, regime_cols, feat_mean, feat_std, feature_row):
    """feature_row: regime_cols各値を持つdict または Series。0〜1の危険度スコア(アンサンブル平均)を返す。"""
    x = np.array([feature_row[c] for c in regime_cols], dtype=np.float32)
    x = (x - feat_mean) / feat_std
    t_x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    probs = []
    with torch.no_grad():
        for model in models:
            probs.append(torch.sigmoid(model(t_x)).item())
    return float(np.mean(probs))


REGIME_RISK_DAILY_CACHE_CSV = os.path.join(BASE_DIR, "data", "regime_risk_daily_cache.csv")


def append_regime_risk_cache(date, risk_score, zone_info, cache_path=REGIME_RISK_DAILY_CACHE_CSV):
    """地合い危険度スコアを日次キャッシュに追記する(将来の学習特徴量として再利用するため)。
    同日に複数回実行した場合はその日の既存行を置き換える。"""
    date_str = pd.Timestamp(date).strftime("%Y-%m-%d")
    new_row = pd.DataFrame([{
        "date": date_str,
        "regime_risk_score": risk_score,
        "zone": zone_info["zone"],
        "k": zone_info["k"],
        "size_mult": zone_info["size_mult"],
    }])
    if os.path.exists(cache_path):
        existing_df = pd.read_csv(cache_path, encoding='utf-8-sig')
        existing_df = existing_df[existing_df["date"] != date_str]
        combined_df = pd.concat([existing_df, new_row], ignore_index=True)
    else:
        combined_df = new_row
    combined_df = combined_df.sort_values("date")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    combined_df.to_csv(cache_path, index=False, encoding='utf-8-sig')
    return combined_df["date"].nunique()


def determine_regime_zone(risk_score, base_k=20):
    """危険度スコアに応じて採用銘柄数K・ロットサイズ倍率を決定する。"""
    if risk_score < 0.5:
        return {
            "zone": "SAFE",
            "k": round(base_k * 1.25),
            "size_mult": 1.1,
        }
    elif risk_score < 0.6:
        return {
            "zone": "NEUTRAL",
            "k": base_k,
            "size_mult": 1.0,
        }
    else:
        return {
            "zone": "DANGER",
            "k": round(base_k * 0.5),
            "size_mult": 0.5,
        }
