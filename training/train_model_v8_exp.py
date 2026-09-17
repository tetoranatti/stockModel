import os
import random
import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler

# training/配下からでもmodules/を解決できるようにプロジェクトルートをsys.pathへ追加
import sys
sys.path.insert(0, r"F:\stockModel")
from modules.model_arch import DualStream_GRU_PreLN_Transformer
from modules.macro_features import load_macro_slim5
from modules.stock_features import compute_stock_features, STOCK_FEATURE_COLS
from modules.cross_sectional_features import (
    apply_log_transform,
    compute_cross_sectional_stats,
    valid_cross_section_dates,
    normalize_cross_sectional,
)

# =============================================================================
# 設定・パス・乱数シード固定
# =============================================================================
BASE_DIR = r"F:\stockModel"
UNIVERSE_PATH = os.path.join(BASE_DIR, "universe_150_tickers.txt")
MODEL_SAVE_PATH_TEMPLATE = os.path.join(BASE_DIR, "swing_model_v8_ensemble_seed{seed}.pt")
ENSEMBLE_SEEDS = [42, 43, 44, 45, 46]
CACHE_DIR = os.path.join(BASE_DIR, "data", "cache")
UNIVERSE_BARS_CACHE_PATH = os.path.join(CACHE_DIR, "train_universe_bars.parquet")

HIDDEN_DIM = 20
MIN_CROSS_SECTION = 20
MAX_PAIR_GAP = 20  # ペアワイズ損失で比較する同一銘柄内サンプルの最大営業日差

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# =============================================================================
# 1. 損失関数・データセット
# =============================================================================
class PairwiseRankingLoss(nn.Module):
    """RankNet型のpairwiseランキング損失。
    離散クラス(0/1/2)を経由せず、連続値ret_pctが大きいサンプルほど
    score(=p_win - p_stop)が高くなるよう直接学習する。CE損失とは異なり
    top-Kランキングの質を直接の最適化対象にできる。
    モデル出力は従来通り3クラスsoftmaxのままなので、p_win=probs[:,2] /
    p_stop=probs[:,0] を読む推論側(model_inference.py等)は無改造で使える。

    注: スコアを生ロジット差(logits[:,2]-logits[:,0])にする案を3シードで
    検証したが、K=1000以降で明確に悪化した(3シード平均PF: K=1000で1.49→1.25、
    K=2000で1.42→1.20、K=4000で1.27→1.15)ため、ソフトマックス経由に戻している。
    ロジットへのLayerNorm追加も試したが、num_classes=3という低次元では
    モデルが定数出力に潰れてしまい(PF全K帯で0)明確に悪化するため不採用。

    ペア範囲: 全ペア(銘柄混在・日付混在)で比較する版から、同一銘柄内かつ
    近傍日(MAX_PAIR_GAP日以内)のペアだけに絞る版(NearPairRankingLoss)に
    切り替えた。5シードの真のアンサンブル(予測平均)バックテストで
    全K帯で明確に改善(K=1000: PF 1.91→2.41、K=4000: 1.24→1.52)したため。
    異なる日をまたぐペアは市況差(レジーム)のノイズを含みやすいと考えられる。
    """
    def forward(self, logits, ret_pct):
        probs = torch.softmax(logits, dim=-1)
        score = probs[:, 2] - probs[:, 0]
        diff_score = score.unsqueeze(1) - score.unsqueeze(0)
        diff_ret = ret_pct.unsqueeze(1) - ret_pct.unsqueeze(0)
        sign = torch.sign(diff_ret)
        mask = sign != 0
        if mask.sum() == 0:
            return diff_score.sum() * 0.0
        return torch.nn.functional.softplus(-sign[mask] * diff_score[mask]).mean()

class NearPairRankingLoss(nn.Module):
    """PairwiseRankingLossと同じsoftmaxスコアだが、ペアを同一銘柄内かつ
    tidx(サンプル位置=営業日インデックス)の差がmax_gap以内のものだけに限定する。"""
    def __init__(self, max_gap=MAX_PAIR_GAP):
        super().__init__()
        self.max_gap = max_gap

    def forward(self, logits, ret_pct, tidx):
        probs = torch.softmax(logits, dim=-1)
        score = probs[:, 2] - probs[:, 0]
        diff_score = score.unsqueeze(1) - score.unsqueeze(0)
        diff_ret = ret_pct.unsqueeze(1) - ret_pct.unsqueeze(0)
        sign = torch.sign(diff_ret)
        gap = (tidx.unsqueeze(1) - tidx.unsqueeze(0)).abs()
        mask = (sign != 0) & (gap > 0) & (gap <= self.max_gap)
        if mask.sum() == 0:
            return diff_score.sum() * 0.0
        return torch.nn.functional.softplus(-sign[mask] * diff_score[mask]).mean()

class UniverseDataset(Dataset):
    def __init__(self, X_stock, X_macro, y, tid, tidx):
        self.X_stock = torch.tensor(X_stock, dtype=torch.float32)
        self.X_macro = torch.tensor(X_macro, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.tid = torch.tensor(tid, dtype=torch.long)
        self.tidx = torch.tensor(tidx, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_stock[idx], self.X_macro[idx], self.y[idx], self.tid[idx], self.tidx[idx]

class SameTickerBatchSampler(Sampler):
    """バッチを「同一銘柄の連続した期間」だけで構成するサンプラー。
    NearPairRankingLossのペア制約(同一銘柄・近傍日)を満たすバッチを作る。"""
    def __init__(self, tid_array, batch_size, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        groups = {}
        for i, t in enumerate(tid_array):
            groups.setdefault(int(t), []).append(i)
        self._batches = []
        for t, idxs in groups.items():
            for start in range(0, len(idxs), batch_size):
                chunk = idxs[start:start + batch_size]
                if len(chunk) >= 4:
                    self._batches.append(chunk)

    def __iter__(self):
        batches = self._batches
        if self.shuffle:
            batches = batches.copy()
            random.shuffle(batches)
        return iter(batches)

    def __len__(self):
        return len(self._batches)

# =============================================================================
# 3. データセット構築 (連続値ret_pctラベリング、横断面正規化、ランキング損失用)
# =============================================================================
def _simulate_ret_pct(closes, highs, lows, atrs, holding_period):
    n_bars = len(closes)
    targets = np.full(n_bars, np.nan)
    for idx in range(n_bars - holding_period):
        entry_p = closes[idx]
        upper_p = entry_p + (2.0 * atrs[idx])
        lower_p = entry_p - (1.0 * atrs[idx])
        exit_p = None

        # 期間内に利確/損切りラインに到達したらそこで手仕舞い
        for h in range(1, holding_period + 1):
            if lows[idx + h] <= lower_p and highs[idx + h] >= upper_p:
                exit_p = lower_p  # 同一足で両方到達した場合は損切り優先(保守的)
                break
            elif highs[idx + h] >= upper_p:
                exit_p = upper_p
                break
            elif lows[idx + h] <= lower_p:
                exit_p = lower_p
                break

        # タイムアウト: holding_period経過時点の終値で手仕舞い
        if exit_p is None:
            exit_p = closes[idx + holding_period]

        targets[idx] = (exit_p - entry_p) / entry_p
    return targets


def build_universe_dataset_slim(tickers, macro_df, seq_len=10, holding_period=10):
    stock_cols = STOCK_FEATURE_COLS
    macro_cols = ['pin_dist_ratio', 'wall_spread', 'cta_net_norm', 'cta_momentum', 'nk_ret_norm']

    if not os.path.exists(UNIVERSE_BARS_CACHE_PATH):
        raise FileNotFoundError(
            f"[!] {UNIVERSE_BARS_CACHE_PATH} が見つかりません。"
            f" 先に build_jquants_cache.py を実行してキャッシュを生成してください。"
        )
    universe_bars = pd.read_parquet(UNIVERSE_BARS_CACHE_PATH)
    universe_bars.index = pd.to_datetime(universe_bars.index).tz_localize(None)
    m_start = macro_df.index.min() - datetime.timedelta(days=150)  # rolling_beta(90日窓)分の余裕を確保

    # --- 1パス目: 銘柄ごとの生特徴量(ログ変換込み)を計算 ---
    print(f"[*] 全 {len(tickers)} 銘柄の特徴量を計算中...")
    per_ticker_df = {}
    for i, t in enumerate(tickers):
        try:
            if t not in universe_bars.columns.get_level_values(0):
                continue
            df = universe_bars[t].loc[universe_bars.index >= m_start].copy()
            df = df.dropna(subset=['Close', 'High', 'Low', 'Volume'])
            if len(df) < seq_len + holding_period + 95 or (df['Volume'] == 0).all():  # +95 = rolling_beta(90日窓)+余裕
                continue

            aligned_nk = macro_df['NK_Ret'].reindex(df.index).fillna(0.0)
            df = compute_stock_features(df, aligned_nk)
            df = df.join(macro_df[macro_cols], how='inner')
            df = df.dropna(subset=['ATR', 'rolling_beta'] + macro_cols)
            if len(df) < seq_len + holding_period + 10:
                continue

            df = apply_log_transform(df)
            per_ticker_df[t] = df
        except Exception:
            continue
        if (i + 1) % 50 == 0 or (i + 1) == len(tickers):
            print(f"  --> {i + 1}/{len(tickers)} 銘柄 完了")

    # --- 横断面(日付×特徴量)統計を計算 ---
    print("[*] 横断面統計(同日の全銘柄基準)を計算中...")
    cross_mean, cross_std = compute_cross_sectional_stats(per_ticker_df, stock_cols)
    valid_dates = valid_cross_section_dates(per_ticker_df, stock_cols, MIN_CROSS_SECTION)
    print(f"[+] 横断面統計が有効な日数: {len(valid_dates)}")

    # --- 2パス目: 横断面正規化・ラベル生成・ウィンドウ構築 ---
    # tid/tidxはNearPairRankingLoss用(同一銘柄・近傍日ペア制約の判定に使う)
    tr_x_s, tr_x_m, tr_y, tr_tid, tr_tidx = [], [], [], [], []
    va_x_s, va_x_m, va_y, va_tid, va_tidx = [], [], [], [], []

    for i, (t, df) in enumerate(per_ticker_df.items()):
        try:
            df = df.loc[df.index.isin(valid_dates)].copy()
            if len(df) < seq_len + holding_period + 10:
                continue

            norm_s = normalize_cross_sectional(df, cross_mean, cross_std, stock_cols)

            closes, highs, lows, atrs = df['Close'].values, df['High'].values, df['Low'].values, df['ATR'].values
            targets = _simulate_ret_pct(closes, highs, lows, atrs, holding_period)

            vals_s_norm = norm_s.values
            vals_m = df[macro_cols].values

            n_samples = len(df)
            split_idx = int(n_samples * 0.75)

            for idx in range(seq_len - 1, n_samples):
                if np.isnan(targets[idx]):
                    continue
                w_s = vals_s_norm[idx - seq_len + 1: idx + 1].copy()
                if np.isnan(w_s).any():
                    continue
                w_m = vals_m[idx - seq_len + 1: idx + 1].copy()
                target = targets[idx]

                if idx < split_idx:
                    tr_x_s.append(w_s)
                    tr_x_m.append(w_m)
                    tr_y.append(target)
                    tr_tid.append(i)
                    tr_tidx.append(idx)
                else:
                    va_x_s.append(w_s)
                    va_x_m.append(w_m)
                    va_y.append(target)
                    va_tid.append(i)
                    va_tidx.append(idx)
        except Exception:
            continue

        if (i + 1) % 50 == 0 or (i + 1) == len(per_ticker_df):
            print(f"  --> {i + 1}/{len(per_ticker_df)} 銘柄 完了")

    return (np.array(tr_x_s), np.array(tr_x_m), np.array(tr_y), np.array(tr_tid), np.array(tr_tidx)), \
           (np.array(va_x_s), np.array(va_x_m), np.array(va_y), np.array(va_tid), np.array(va_tidx)), \
           stock_cols, macro_cols

# =============================================================================
# 4. 学習ループ
# =============================================================================
def train():
    if os.path.exists(UNIVERSE_PATH):
        with open(UNIVERSE_PATH, "r", encoding="utf-8") as f:
            tickers = [line.strip() for line in f if line.strip()]
    else:
        tickers = ["7203.T", "6758.T", "8035.T", "8306.T", "9432.T", "7167.T", "4519.T", "5726.T"]

    macro_df = load_macro_slim5()
    train_data, val_data, s_cols, m_cols = build_universe_dataset_slim(tickers, macro_df)
    tr_x_s, tr_x_m, tr_y, tr_tid, tr_tidx = train_data
    va_x_s, va_x_m, va_y, va_tid, va_tidx = val_data

    print(f"[+] データ構築完了: Train = {len(tr_y)}, Val = {len(va_y)}")
    print(f"  - ret_pct分布: Train mean={tr_y.mean():.4f} std={tr_y.std():.4f} | "
          f"Val mean={va_y.mean():.4f} std={va_y.std():.4f}")

    # 単一シードだと運の良し悪しでPFが大きく振れる(複数シード検証で確認済み)ため、
    # 複数シードで学習してアンサンブル(予測平均)として本番運用する。
    for seed in ENSEMBLE_SEEDS:
        print(f"\n[*] [v8 アンサンブル seed={seed}] 学習開始...")
        train_one_seed(seed, tr_x_s, tr_x_m, tr_y, tr_tid, tr_tidx,
                        va_x_s, va_x_m, va_y, va_tid, va_tidx, s_cols, m_cols)


def train_one_seed(seed, tr_x_s, tr_x_m, tr_y, tr_tid, tr_tidx,
                    va_x_s, va_x_m, va_y, va_tid, va_tidx, s_cols, m_cols):
    set_seed(seed)
    save_path = MODEL_SAVE_PATH_TEMPLATE.format(seed=seed)

    train_ds = UniverseDataset(tr_x_s, tr_x_m, tr_y, tr_tid, tr_tidx)
    val_ds = UniverseDataset(va_x_s, va_x_m, va_y, va_tid, va_tidx)
    train_loader = DataLoader(train_ds, batch_sampler=SameTickerBatchSampler(tr_tid, 128, shuffle=True), pin_memory=True)
    val_loader = DataLoader(val_ds, batch_sampler=SameTickerBatchSampler(va_tid, 256, shuffle=False), pin_memory=True)

    model = DualStream_GRU_PreLN_Transformer(
        stock_dim=len(s_cols), macro_dim=len(m_cols),
        hidden_dim=HIDDEN_DIM, num_heads=1, num_classes=3, dropout=0.2
    ).to(DEVICE)

    criterion = NearPairRankingLoss(max_gap=MAX_PAIR_GAP)
    # lr=5e-05を試したが、真のアンサンブル(予測平均)バックテストで最上位帯
    # (n=461)のPFが0.94まで悪化する退行が判明したため0.0002に差し戻し。
    # (個別シードのtop-K評価だけでは見抜けなかった問題。要: 本番相当のアンサンブル
    # バックテストでの検証を今後も徹底する)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0002, weight_decay=3e-2)

    best_val_loss = float('inf')
    patience = 7
    patience_cnt = 0
    epochs = 30

    for epoch in range(1, epochs + 1):
        model.train()
        for b_xs, b_xm, b_y, b_tid, b_tidx in train_loader:
            b_xs, b_xm, b_y, b_tidx = (b_xs.to(DEVICE, non_blocking=True), b_xm.to(DEVICE, non_blocking=True),
                                        b_y.to(DEVICE, non_blocking=True), b_tidx.to(DEVICE, non_blocking=True))
            optimizer.zero_grad()
            loss = criterion(model(b_xs, b_xm), b_y, b_tidx)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        total_va_loss = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for b_xs, b_xm, b_y, b_tid, b_tidx in val_loader:
                b_xs, b_xm, b_y, b_tidx = (b_xs.to(DEVICE, non_blocking=True), b_xm.to(DEVICE, non_blocking=True),
                                            b_y.to(DEVICE, non_blocking=True), b_tidx.to(DEVICE, non_blocking=True))
                loss = criterion(model(b_xs, b_xm), b_y, b_tidx)
                total_va_loss += loss.item()
                n_val_batches += 1

        val_loss = total_va_loss / n_val_batches if n_val_batches > 0 else float('inf')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_cnt = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'stock_cols': s_cols,
                'macro_cols': m_cols,
                'hidden_dim': HIDDEN_DIM,
                'num_heads': 1,
                'dropout': 0.2,
                'val_loss': best_val_loss,
                'seed': seed,
            }, save_path)
            print(f"  Epoch [{epoch:02d}/{epochs:02d}] - Val: {val_loss:.4f}  --> [Best Val Loss 更新 ★]")
        else:
            patience_cnt += 1
            print(f"  Epoch [{epoch:02d}/{epochs:02d}] - Val: {val_loss:.4f}  (Patience: {patience_cnt}/{patience})")
            if patience_cnt >= patience:
                print(f"[*] Early Stopping 発動 (Epoch {epoch})")
                break

    print(f"[+] 重み保存完了: {save_path} (best_val_loss={best_val_loss:.4f})")

if __name__ == "__main__":
    train()