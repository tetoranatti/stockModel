# train_short_edge_model.py
# 日次集約型「空売り機会」モデルの学習スクリプト。
# ラベル: 翌日TOPIXリターン<0 (二値、地合い危険度モデルのような中央値二値化は不要)。
# 特徴量: 指数下落モメンタム・CTA/JNETフロー・VIX・銘柄横断の出来高薄さ・PF逆スコア
# (セクターセンチメント・空売り比率・日経VIは学習に十分な履歴が無いためv1では見送り)。
import os
import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# training/配下からでもmodules/を解決できるようにプロジェクトルートをsys.pathへ追加
import sys
sys.path.insert(0, r"F:\stockModel")
import train_model_v8_exp as tm
from modules.macro_features import load_macro_slim5
from modules.stock_features import compute_stock_features
from modules.cross_sectional_features import (
    apply_log_transform, compute_cross_sectional_stats,
    valid_cross_section_dates, normalize_cross_sectional,
)
from modules.model_inference import load_trained_models_ensemble, predict_probabilities_ensemble_batch
from modules.short_edge_model import (
    SHORT_EDGE_COLS, SHORT_EDGE_COLS_NO_FLOW, SHORT_EDGE_COLS_SHORT_RATIO, ShortEdgeMLP, DEVICE,
    build_flow_features, build_vix_features, build_idx_momentum_features,
    build_short_ratio_features,
    compute_cross_vol_thinness, compute_pf_score_features,
    load_pf_vol_cache, save_pf_vol_cache,
)

BASE_DIR = r"F:\stockModel"
PF_MODEL_PATHS = [os.path.join(BASE_DIR, f"swing_model_v8_ensemble_seed{s}.pt") for s in tm.ENSEMBLE_SEEDS]
TOPIX_CACHE_PATH = os.path.join(BASE_DIR, "data", "cache", "train_topix_bars.parquet")
FLOW_CSV = os.path.join(BASE_DIR, "data", "flow_signal_daily_cache.csv")
VIX_CSV = os.path.join(BASE_DIR, "data", "vix_fred.csv")
SHORT_RATIO_CSV = os.path.join(BASE_DIR, "data", "short_ratio_daily_cache.csv")
# flowあり/flowなし/空売り比率追加版をそれぞれ別チェックポイントで比較できるようにする
# (flowありは2025-09-01以降の256日、flowなし系はjpx_daily_features_db.csvの範囲=
# 2024-01-05以降まで学習期間を伸ばせる。空売り比率は2018年まで遡れるため無制限)。
MODEL_SAVE_PATH_TEMPLATE = os.path.join(BASE_DIR, "short_edge_model_seed{seed}.pt")
MODEL_SAVE_PATH_TEMPLATE_NO_FLOW = os.path.join(BASE_DIR, "short_edge_noflow_model_seed{seed}.pt")
MODEL_SAVE_PATH_TEMPLATE_SHORT_RATIO = os.path.join(BASE_DIR, "short_edge_shortratio_model_seed{seed}.pt")
ENSEMBLE_SEEDS = [42, 43, 44, 45, 46]
SEQ_LEN = 10
MIN_CROSS_SECTION = 20


def build_day_level_dataset(include_flow=True, include_short_ratio=False):
    macro_df = load_macro_slim5()

    # PF推論(最も重い処理)は、後段のinner joinでどのみち切り捨てられる期間まで
    # 律儀に計算すると大半が無駄になり数倍遅くなる(実測済み)ため、最終的に効いてくる
    # 期間まであらかじめ絞ってから推論する。
    # - include_flow=True: CTA/JNETフロー(2025-09-01以降)が最も短くそこがボトルネック。
    # - include_flow=False: jpx_daily_features_db.csv由来のmacro_df(2024-01-05以降)がボトルネック。
    if include_flow:
        flow_df = pd.read_csv(FLOW_CSV, encoding='utf-8-sig')
        flow_df['date'] = pd.to_datetime(flow_df['date'])
        flow_feat = build_flow_features(flow_df)
        infer_start = flow_feat.index.min() - datetime.timedelta(days=int(SEQ_LEN * 2.5))
    else:
        flow_feat = None
        infer_start = macro_df.index.min()

    needed_dates = macro_df.index[macro_df.index >= infer_start]
    cache_df = load_pf_vol_cache()
    cache_hit = cache_df is not None and set(needed_dates).issubset(set(cache_df.index))

    if cache_hit:
        # PF推論・銘柄別特徴量計算を丸ごとスキップし、日次集約済みキャッシュから読む
        print(f"[*] PF/出来高の日次集約キャッシュがヒット(対象期間: {infer_start.date()} 〜)。推論をスキップします。")
        pf_feat = cache_df.reindex(macro_df.index)[['pf_avg_p_win', 'pf_pct_bullish']].fillna(0.5)
        vol_feat = cache_df.reindex(macro_df.index)[['cross_vol_ratio_mean', 'cross_vol_thin_pct']]
        vol_feat['cross_vol_ratio_mean'] = vol_feat['cross_vol_ratio_mean'].fillna(1.0)
        vol_feat['cross_vol_thin_pct'] = vol_feat['cross_vol_thin_pct'].fillna(0.0)
    else:
        pf_models, pf_stock_cols, pf_macro_cols = load_trained_models_ensemble(PF_MODEL_PATHS)
        with open(tm.UNIVERSE_PATH, "r", encoding="utf-8") as f:
            tickers = [line.strip() for line in f if line.strip()]
        universe_bars = pd.read_parquet(tm.UNIVERSE_BARS_CACHE_PATH)
        universe_bars.index = pd.to_datetime(universe_bars.index).tz_localize(None)
        m_start = macro_df.index.min() - datetime.timedelta(days=150)

        print(f"[*] 全 {len(tickers)} 銘柄の特徴量を計算中...")
        per_ticker_df = {}
        for i, t in enumerate(tickers):
            try:
                if t not in universe_bars.columns.get_level_values(0):
                    continue
                df = universe_bars[t].loc[universe_bars.index >= m_start].copy()
                df = df.dropna(subset=['Close', 'High', 'Low', 'Volume'])
                if len(df) < SEQ_LEN + 95 or (df['Volume'] == 0).all():
                    continue
                aligned_nk = macro_df['NK_Ret'].reindex(df.index).fillna(0.0)
                df = compute_stock_features(df, aligned_nk)
                df = df.join(macro_df[pf_macro_cols], how='inner')
                df = df.dropna(subset=['ATR', 'rolling_beta'] + pf_macro_cols)
                if len(df) < SEQ_LEN + 10:
                    continue
                per_ticker_df[t] = df
            except Exception:
                continue
            if (i + 1) % 50 == 0 or (i + 1) == len(tickers):
                print(f"  --> {i + 1}/{len(tickers)} 銘柄 完了")

        # 出来高薄さ(生のvol_ratio_5d、PF用log変換前)は先に確保しておく
        per_ticker_vol_ratio = {t: df['vol_ratio_5d'] for t, df in per_ticker_df.items()}

        log_df = {t: apply_log_transform(df) for t, df in per_ticker_df.items()}
        cross_mean, cross_std = compute_cross_sectional_stats(log_df, pf_stock_cols)
        valid_dates = valid_cross_section_dates(log_df, pf_stock_cols, MIN_CROSS_SECTION)
        valid_dates = valid_dates[valid_dates >= infer_start]

        print(f"[*] PFモデルで全銘柄を推論中(日次集約用、対象期間: {infer_start.date()} 〜)...")
        per_ticker_p_win, per_ticker_p_stop = {}, {}
        for i, (t, df) in enumerate(log_df.items()):
            try:
                df = df.loc[df.index.isin(valid_dates)].copy()
                if len(df) < SEQ_LEN:
                    continue
                norm_s = normalize_cross_sectional(df, cross_mean, cross_std, pf_stock_cols)
                vals_s_norm = norm_s.values
                vals_m = df[pf_macro_cols].values
                n_samples = len(df)

                p_win_arr = np.full(n_samples, np.nan)
                p_stop_arr = np.full(n_samples, np.nan)
                # 有効な(NaNでない)idxのw_s/w_mを先にまとめてバッチ推論する(2026-09-18、
                # 1件ずつ推論していたのを本銘柄分まとめて1回のforward呼び出しに変更。
                # .eval()+LayerNormのみ(BatchNorm不使用)なので結果は数値的に完全一致し、
                # 純粋な高速化になる)。
                valid_idx, w_s_list, w_m_list = [], [], []
                for idx in range(SEQ_LEN - 1, n_samples):
                    w_s = vals_s_norm[idx - SEQ_LEN + 1: idx + 1]
                    if np.isnan(w_s).any():
                        continue
                    valid_idx.append(idx)
                    w_s_list.append(w_s)
                    w_m_list.append(vals_m[idx - SEQ_LEN + 1: idx + 1])
                if valid_idx:
                    p_win_b, p_stop_b, _ = predict_probabilities_ensemble_batch(
                        pf_models, np.stack(w_s_list), np.stack(w_m_list)
                    )
                    for k, idx in enumerate(valid_idx):
                        p_win_arr[idx] = p_win_b[k]
                        p_stop_arr[idx] = p_stop_b[k]
                per_ticker_p_win[t] = pd.Series(p_win_arr, index=df.index)
                per_ticker_p_stop[t] = pd.Series(p_stop_arr, index=df.index)
            except Exception:
                continue
            if (i + 1) % 50 == 0 or (i + 1) == len(log_df):
                print(f"  --> {i + 1}/{len(log_df)} 銘柄 推論完了")

        vol_feat = compute_cross_vol_thinness(per_ticker_vol_ratio, macro_df.index)
        pf_feat = compute_pf_score_features(per_ticker_p_win, per_ticker_p_stop, macro_df.index)
        n_cached = save_pf_vol_cache(pf_feat, vol_feat)
        print(f"[+] PF/出来高の日次集約キャッシュを更新: {n_cached}日分")

    # --- 日次集約特徴量を組み立て(flow_featは上でPF推論の対象期間絞り込みに使ったものを再利用) ---
    idx_feat = build_idx_momentum_features(macro_df['NK_Close'])
    vix_feat = build_vix_features(VIX_CSV)

    # オプション建玉由来の「無料」特徴量(新規データ取得不要、macro_dfに既に計算済み)
    macro_extra_feat = macro_df[['pin_dist_ratio', 'wall_spread', 'cta_net_norm', 'cta_momentum']]

    feat_df = idx_feat.join(vol_feat, how='left').join(pf_feat, how='left').join(macro_extra_feat, how='left')
    if include_flow:
        feat_df = feat_df.join(flow_feat, how='inner')  # CTA/JNETが最も短い履歴のため内部結合で自然に絞られる
    feat_df = feat_df.join(vix_feat, how='inner')
    if include_short_ratio:
        # 空売り比率は2018年まで遡れるため、既存のinfer_start(macro_df起点)を狭めない
        short_ratio_df = pd.read_csv(SHORT_RATIO_CSV, encoding='utf-8-sig')
        short_ratio_df['date'] = pd.to_datetime(short_ratio_df['date'])
        short_ratio_feat = build_short_ratio_features(short_ratio_df)
        feat_df = feat_df.join(short_ratio_feat, how='left')

    # --- ラベル: 翌日TOPIXリターン<0 ---
    topix = pd.read_parquet(TOPIX_CACHE_PATH)
    topix.index = pd.to_datetime(topix.index).tz_localize(None)
    topix_ret_next = topix['Close'].pct_change(fill_method=None).shift(-1)
    label = (topix_ret_next < 0).astype(int).rename('y_short_edge')

    if include_short_ratio:
        cols = SHORT_EDGE_COLS_SHORT_RATIO
    elif include_flow:
        cols = SHORT_EDGE_COLS
    else:
        cols = SHORT_EDGE_COLS_NO_FLOW
    day_df = feat_df.join(label, how='inner').dropna()
    day_df = day_df[cols + ['y_short_edge']].sort_index()
    return day_df


def train_one_seed(seed, X_train, y_train, X_val, y_val, input_dim):
    tm.set_seed(seed)
    model = ShortEdgeMLP(input_dim=input_dim).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-2)

    X_t = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    y_t = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)
    X_v = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    y_v = torch.tensor(y_val, dtype=torch.float32).to(DEVICE)

    best_val_loss, patience, patience_cnt, epochs, best_state = float('inf'), 10, 0, 100, None
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_t), y_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_v), y_v).item()
        if val_loss < best_val_loss:
            best_val_loss, patience_cnt = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break
    model.load_state_dict(best_state)
    print(f"  [short_edge seed={seed}] stopped_epoch={epoch} best_val_loss={best_val_loss:.4f}")
    return model


def _auc(y_true, y_score):
    """sklearn無しでも動くMann-Whitney U統計量ベースのAUC計算。"""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n_pos, n_neg = y_true.sum(), len(y_true) - y_true.sum()
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    order = np.argsort(y_score)
    ranks = np.empty(len(y_score))
    ranks[order] = np.arange(1, len(y_score) + 1)
    # タイの平均順位補正
    _, inv, counts = np.unique(y_score, return_inverse=True, return_counts=True)
    sum_ranks = np.zeros(len(counts))
    cnt = np.zeros(len(counts))
    np.add.at(sum_ranks, inv, ranks)
    np.add.at(cnt, inv, 1)
    avg_rank_per_val = sum_ranks / cnt
    ranks = avg_rank_per_val[inv]
    sum_ranks_pos = ranks[y_true == 1].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def train(include_flow=True, include_short_ratio=False):
    if include_short_ratio:
        mode_label = "flowなし・長期+空売り比率"
        cols = SHORT_EDGE_COLS_SHORT_RATIO
        save_template = MODEL_SAVE_PATH_TEMPLATE_SHORT_RATIO
    elif include_flow:
        mode_label = "flowあり(256日)"
        cols = SHORT_EDGE_COLS
        save_template = MODEL_SAVE_PATH_TEMPLATE
    else:
        mode_label = "flowなし・長期(jpx_daily_features_db基準)"
        cols = SHORT_EDGE_COLS_NO_FLOW
        save_template = MODEL_SAVE_PATH_TEMPLATE_NO_FLOW

    print(f"[*] モード: {mode_label} | 特徴量数: {len(cols)}")
    day_df = build_day_level_dataset(include_flow=include_flow, include_short_ratio=include_short_ratio)
    print(f"\n[+] 日次集約サンプル数: {len(day_df)}")
    print(f"[+] ラベル分布: y=1(翌日TOPIX下落) {day_df['y_short_edge'].mean()*100:.1f}%"
          f" ({day_df['y_short_edge'].sum():.0f}/{len(day_df)}日)")

    split_idx = int(len(day_df) * 0.75)
    train_df = day_df.iloc[:split_idx]
    val_df = day_df.iloc[split_idx:]
    print(f"[+] Train={len(train_df)} Val={len(val_df)}"
          f" (期間: {day_df.index.min().date()} 〜 {day_df.index.max().date()})")

    feat_mean = train_df[cols].mean().values.astype(np.float32)
    feat_std = (train_df[cols].std() + 1e-7).values.astype(np.float32)
    X_train = ((train_df[cols].values - feat_mean) / feat_std).astype(np.float32)
    X_val = ((val_df[cols].values - feat_mean) / feat_std).astype(np.float32)
    y_train = train_df['y_short_edge'].values.astype(np.float32)
    y_val = val_df['y_short_edge'].values.astype(np.float32)

    val_probs_all = []
    for seed in ENSEMBLE_SEEDS:
        print(f"\n[*] [short_edge アンサンブル seed={seed}] 学習開始...")
        model = train_one_seed(seed, X_train, y_train, X_val, y_val, input_dim=len(cols))

        model.eval()
        with torch.no_grad():
            val_probs = torch.sigmoid(model(torch.tensor(X_val, dtype=torch.float32).to(DEVICE))).cpu().numpy()
        val_probs_all.append(val_probs)
        seed_auc = _auc(y_val, val_probs)
        seed_acc = ((val_probs >= 0.5).astype(int) == y_val).mean()
        print(f"  [short_edge seed={seed}] val_AUC={seed_auc:.3f} val_ACC={seed_acc*100:.1f}%")

        save_path = save_template.format(seed=seed)
        torch.save({
            'model_state_dict': model.state_dict(),
            'feature_cols': cols,
            'feat_mean': torch.tensor(feat_mean),
            'feat_std': torch.tensor(feat_std),
            'seed': seed,
        }, save_path)
        print(f"[+] 重み保存完了: {save_path}")

    ens_probs = np.mean(val_probs_all, axis=0)
    ens_auc = _auc(y_val, ens_probs)
    ens_acc = ((ens_probs >= 0.5).astype(int) == y_val).mean()
    print(f"\n{'='*60}")
    print(f"[★] 5シードアンサンブル検証成績({mode_label}): AUC={ens_auc:.3f} ACC={ens_acc*100:.1f}%"
          f" (ベースライン多数派={max(y_val.mean(), 1-y_val.mean())*100:.1f}%)")
    print(f"{'='*60}")


if __name__ == "__main__":
    include_short_ratio = "--short-ratio" in sys.argv
    include_flow = ("--no-flow" not in sys.argv) and not include_short_ratio
    train(include_flow=include_flow, include_short_ratio=include_short_ratio)
