# modules/short_edge_model.py
# 日次集約型の「空売り機会」モデル。regime_risk_model.py(地合い危険度モデル)と
# 同じ日次集約の発想を踏襲するが、ラベルはより単純に「翌日TOPIXリターン<0」の
# 二値とする(地合い危険度モデルのようなSTOP率の連続値→中央値二値化は行わない)。
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

BASE_DIR = r"F:\stockModel"

# CTA/JNETフロー(flow_signal_daily_cache.csv)は2025-09-01以降しか無く、
# セクターセンチメント・空売り比率・日経VIは学習に十分な期間分の履歴が無い
# (それぞれ後日データが蓄積してから追加検討)ため、v1では以下の4系統のみを使う。
SHORT_EDGE_COLS = [
    # 指数下落モメンタム
    'idx_ret_1d', 'idx_ret_5d', 'idx_ret_20d',
    # CTAフロー(share/delta/trend/spike)
    'cta_share', 'cta_delta_1d', 'cta_ma5_diff', 'cta_zscore',
    # J-NETフロー(ratio/delta/spike)
    'jnet_ratio', 'jnet_delta_1d', 'jnet_ma5_diff', 'jnet_zscore',
    # VIX(level/delta/trend/spike)
    'vix_level_norm', 'vix_change_norm', 'vix_ma5_diff', 'vix_zscore',
    # 出来高の薄さ(銘柄横断集計)
    'cross_vol_ratio_mean', 'cross_vol_thin_pct',
    # PF逆スコア(モデルの強気度合いの日次集約)
    'pf_avg_p_win', 'pf_pct_bullish',
]

# CTA/JNETフロー(flow_signal_daily_cache.csv、2025-09-01以降しか無く学習期間を256日に
# 制限してしまう)を除いた版。jpx_daily_features_db.csv(2024-01-05〜)の範囲まで
# 学習期間を伸ばせるかを比較検証したところAUCはほぼ同等(0.692→0.694)でサンプル数だけ
# 2.6倍になったため、こちらを本線として採用。
# 追加検証: pin_dist_ratio/wall_spread/cta_net_norm/cta_momentum(オプション建玉由来、
# 新規データ取得不要の「無料」特徴量)を足して試したが AUC 0.694→0.673 と悪化したため不採用。
SHORT_EDGE_COLS_NO_FLOW = [
    c for c in SHORT_EDGE_COLS if not c.startswith('cta_') and not c.startswith('jnet_')
]

# 空売り比率(J-Quants /markets/short-ratio、2018年まで遡れるためNO_FLOW版の学習期間を
# 一切犠牲にしない)を追加した版。
SHORT_EDGE_COLS_SHORT_RATIO = SHORT_EDGE_COLS_NO_FLOW + [
    'short_ratio', 'short_ratio_delta_1d', 'short_ratio_ma5_diff', 'short_ratio_zscore',
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ShortEdgeMLP(nn.Module):
    """Input -> Dense(64,ReLU)+Dropout -> Dense(32,ReLU)+Dropout -> Dense(16,ReLU) -> Dense(1)
    (Sigmoidは推論側でtorch.sigmoid()を適用。学習はBCEWithLogitsLossでlogitsのまま扱う)"""
    def __init__(self, input_dim, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.drop1(self.relu(self.fc1(x)))
        x = self.drop2(self.relu(self.fc2(x)))
        x = self.relu(self.fc3(x))
        return self.fc4(x).squeeze(-1)


def _zscore(s, window=20, min_periods=10):
    mean = s.rolling(window, min_periods=min_periods).mean()
    std = s.rolling(window, min_periods=min_periods).std() + 1e-7
    return ((s - mean) / std).fillna(0.0)


def build_flow_features(flow_df):
    """flow_signal_daily_cache.csv(date, cta_share, jnet_ratio)からCTA/JNET派生特徴量を作る。"""
    df = flow_df.copy().sort_values('date').set_index('date')
    out = pd.DataFrame(index=df.index)
    out['cta_share'] = df['cta_share']
    out['cta_delta_1d'] = df['cta_share'].diff().fillna(0.0)
    out['cta_ma5_diff'] = (df['cta_share'] - df['cta_share'].rolling(5, min_periods=1).mean()).fillna(0.0)
    out['cta_zscore'] = _zscore(df['cta_share'])

    out['jnet_ratio'] = df['jnet_ratio']
    out['jnet_delta_1d'] = df['jnet_ratio'].diff().fillna(0.0)
    out['jnet_ma5_diff'] = (df['jnet_ratio'] - df['jnet_ratio'].rolling(5, min_periods=1).mean()).fillna(0.0)
    out['jnet_zscore'] = _zscore(df['jnet_ratio'])
    return out


def build_short_ratio_features(short_ratio_df):
    """short_ratio_daily_cache.csv(date, market_short_ratio)から空売り比率の派生特徴量を作る。"""
    df = short_ratio_df.copy().sort_values('date').set_index('date')
    out = pd.DataFrame(index=df.index)
    out['short_ratio'] = df['market_short_ratio']
    out['short_ratio_delta_1d'] = df['market_short_ratio'].diff().fillna(0.0)
    out['short_ratio_ma5_diff'] = (
        df['market_short_ratio'] - df['market_short_ratio'].rolling(5, min_periods=1).mean()
    ).fillna(0.0)
    out['short_ratio_zscore'] = _zscore(df['market_short_ratio'])
    return out


def build_vix_features(vix_csv_path):
    vix = pd.read_csv(vix_csv_path, index_col=0, parse_dates=True)['VIXCLS'].dropna()
    out = pd.DataFrame(index=vix.index)
    out['vix_level_norm'] = (vix - 17.0) / 4.4
    out['vix_change_norm'] = vix.diff().fillna(0.0) / 1.8
    out['vix_ma5_diff'] = (vix - vix.rolling(5, min_periods=1).mean()).fillna(0.0) / 1.8
    out['vix_zscore'] = _zscore(vix)
    return out


def build_idx_momentum_features(nk_close):
    out = pd.DataFrame(index=nk_close.index)
    out['idx_ret_1d'] = nk_close.pct_change(1, fill_method=None).fillna(0.0)
    out['idx_ret_5d'] = nk_close.pct_change(5, fill_method=None).fillna(0.0)
    out['idx_ret_20d'] = nk_close.pct_change(20, fill_method=None).fillna(0.0)
    return out


def compute_cross_vol_thinness(per_ticker_vol_ratio, index, thin_th=0.8):
    """per_ticker_vol_ratio: {ticker: vol_ratio_5dのSeries} の辞書。
    銘柄横断の出来高薄さを (平均vol_ratio_5d, 薄商い銘柄比率) として日次で返す。"""
    vr_df = pd.DataFrame(per_ticker_vol_ratio)
    mean_ratio = vr_df.mean(axis=1, skipna=True)
    thin_pct = (vr_df < thin_th).sum(axis=1) / vr_df.notna().sum(axis=1).clip(lower=1)
    out = pd.DataFrame(index=index)
    out['cross_vol_ratio_mean'] = mean_ratio.reindex(index).fillna(1.0)
    out['cross_vol_thin_pct'] = thin_pct.reindex(index).fillna(0.0)
    return out


def compute_pf_score_features(per_ticker_p_win, per_ticker_p_stop, index):
    """per_ticker_p_win/p_stop: {ticker: Series} の辞書。PFモデルの日次強気度合いを
    (横断平均p_win, p_win>p_stopの銘柄比率) として返す。"""
    pw_df = pd.DataFrame(per_ticker_p_win)
    ps_df = pd.DataFrame(per_ticker_p_stop)
    avg_p_win = pw_df.mean(axis=1, skipna=True)
    bullish_pct = (pw_df > ps_df).sum(axis=1) / pw_df.notna().sum(axis=1).clip(lower=1)
    out = pd.DataFrame(index=index)
    out['pf_avg_p_win'] = avg_p_win.reindex(index).fillna(0.5)
    out['pf_pct_bullish'] = bullish_pct.reindex(index).fillna(0.5)
    return out


# --- PF推論結果(日次集約)のキャッシュ ---
# 銘柄横断でPFモデル5シードアンサンブルを全期間推論するのが学習パイプライン中で
# 最も重い処理(実測で数分〜十数分)なため、日次集約後の値をCSVキャッシュしておき、
# 特徴量セットやアーキテクチャを変えて再学習する際の待ち時間を無くす。
# (地合い危険度モデルのregime_risk_daily_cache.csvと同じ発想)
PF_VOL_DAILY_CACHE_CSV = os.path.join(BASE_DIR, "data", "pf_vol_daily_cache.csv")
PF_VOL_CACHE_COLS = ['pf_avg_p_win', 'pf_pct_bullish', 'cross_vol_ratio_mean', 'cross_vol_thin_pct']


def load_pf_vol_cache(cache_path=PF_VOL_DAILY_CACHE_CSV):
    """キャッシュが無ければNoneを返す。あればdateをindexにしたDataFrameを返す。"""
    if not os.path.exists(cache_path):
        return None
    df = pd.read_csv(cache_path, encoding='utf-8-sig')
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()
    return df


def save_pf_vol_cache(pf_feat, vol_feat, cache_path=PF_VOL_DAILY_CACHE_CSV):
    """pf_feat/vol_feat(いずれもdate index)を結合し、既存キャッシュとupsertして保存する。"""
    new_df = pf_feat.join(vol_feat, how='outer')
    new_df.index.name = 'date'
    existing = load_pf_vol_cache(cache_path)
    if existing is not None:
        existing = existing[~existing.index.isin(new_df.index)]
        combined = pd.concat([existing, new_df]).sort_index()
    else:
        combined = new_df.sort_index()
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    out = combined.reset_index()
    out['date'] = out['date'].dt.strftime('%Y-%m-%d')
    out.to_csv(cache_path, index=False, encoding='utf-8-sig')
    return len(combined)


def load_short_edge_ensemble(weights_paths):
    models = []
    feat_mean = feat_std = feature_cols = None
    for path in weights_paths:
        checkpoint = torch.load(path, map_location=DEVICE)
        cols = checkpoint['feature_cols']
        if feature_cols is None:
            feature_cols = cols
            feat_mean = checkpoint['feat_mean'].cpu().numpy()
            feat_std = checkpoint['feat_std'].cpu().numpy()
        elif cols != feature_cols:
            raise ValueError(f"[!] アンサンブルメンバー間でfeature_colsが不一致: {path}")
        model = ShortEdgeMLP(input_dim=len(cols)).to(DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        models.append(model)
    return models, feature_cols, feat_mean, feat_std


def predict_short_edge(models, feature_cols, feat_mean, feat_std, feature_row):
    """feature_row: feature_cols各値を持つdict または Series。
    0〜1の「翌日TOPIX下落」確率(アンサンブル平均) p_short_edge を返す。"""
    x = np.array([feature_row[c] for c in feature_cols], dtype=np.float32)
    x = (x - feat_mean) / feat_std
    t_x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    probs = []
    with torch.no_grad():
        for model in models:
            probs.append(torch.sigmoid(model(t_x)).item())
    return float(np.mean(probs))


# --- ロングサイズへの適用(ゾーン方式) ---
# p_short_edge(=short_strength)の全期間分布の25%点・75%点を境に3ゾーン化する。
# 地合い危険度モデルと同じsize_mult=1.1/1.0/0.5を使い、実トレードでの資金曲線バックテスト
# (p_win>=0.590の高確信度帯)で比較した結果、この組み合わせがMaxDD改善(-13.6%→-11.1%)
# と最終資金の微増(2.447→2.473)を両立する良いバランスだったため採用。
# (連続式 size=1-alpha*p_short_edge は常に一律縮小するだけでリターンも一律減ったため不採用)
SHORT_EDGE_Q25 = 0.339
SHORT_EDGE_Q75 = 0.531


def determine_short_edge_size_mult(p_short_edge, q25=SHORT_EDGE_Q25, q75=SHORT_EDGE_Q75):
    """p_short_edge(翌日TOPIX下落確率)からロングサイズ倍率とゾーン名を決定する。"""
    if p_short_edge <= q25:
        return {"zone": "OPTIMISTIC", "size_mult": 1.1}
    elif p_short_edge <= q75:
        return {"zone": "NEUTRAL", "size_mult": 1.0}
    else:
        return {"zone": "PESSIMISTIC", "size_mult": 0.5}


SHORT_EDGE_DAILY_CACHE_CSV = os.path.join(BASE_DIR, "data", "short_edge_daily_cache.csv")


def append_short_edge_cache(date, p_short_edge, zone_info, cache_path=SHORT_EDGE_DAILY_CACHE_CSV):
    """空売り機会モデルのスコアを日次キャッシュに追記する(将来の学習特徴量として再利用するため)。
    同日に複数回実行した場合はその日の既存行を置き換える。"""
    date_str = pd.Timestamp(date).strftime("%Y-%m-%d")
    new_row = pd.DataFrame([{
        "date": date_str,
        "p_short_edge": p_short_edge,
        "zone": zone_info["zone"],
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
