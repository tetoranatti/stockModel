import re
import os
import shutil
import glob
import random
import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yfinance as yf

# =============================================================================
# 設定・パス・乱数シード固定
# =============================================================================
BASE_DIR = r"F:\stockModel"
DOWNLOADS_DIR = os.path.join(os.environ["USERPROFILE"], "Downloads")
SCREENER_CSV = os.path.join(BASE_DIR, "screener_result.csv")
JPX_DB_PATH = os.path.join(BASE_DIR, "jpx_daily_features_db.csv")
MODEL_WEIGHTS = os.path.join(BASE_DIR, "swing_model_v6_crossattn_universe_v5.pt")
OUTPUT_CSV = os.path.join(BASE_DIR, "final_regime_screened_v6.csv")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MIN_TURNOVER = 10e8  # 5日平均売買代金10億円以上
SEQ_LEN = 10

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# =============================================================================
# 1. モデル定義
# =============================================================================
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

# =============================================================================
# 2. 最新CSV同期 & マクロ環境データ構築
# =============================================================================
def sync_latest_csv():
    pattern = os.path.join(DOWNLOADS_DIR, "*screener*.csv")
    files = glob.glob(pattern)
    if files:
        latest = max(files, key=os.path.getmtime)
        shutil.copy2(latest, SCREENER_CSV)
        print(f"[+] 最新スクリーニングCSVを自動同期: {os.path.basename(latest)}")

def load_macro_environment():
    if not os.path.exists(JPX_DB_PATH):
        raise FileNotFoundError(f"[!] {JPX_DB_PATH} が見つかりません。")

    jpx_db = pd.read_csv(JPX_DB_PATH, index_col=0, parse_dates=True)
    jpx_db.index = pd.to_datetime(jpx_db.index).tz_localize(None)
    start_date = (jpx_db.index.min() - datetime.timedelta(days=20)).strftime("%Y-%m-%d")

    n225 = yf.download("^N225", start=start_date, interval="1d", progress=False)
    if isinstance(n225.columns, pd.MultiIndex):
        n225.columns = n225.columns.get_level_values(0)
    n225.index = pd.to_datetime(n225.index).tz_localize(None)

    macro_df = pd.DataFrame(index=n225.index)
    macro_df['NK_Close'] = n225['Close']
    macro_df['NK_Ret'] = n225['Close'].pct_change(fill_method=None).fillna(0.0)

    # 過去リークを防ぐ前方補間
    macro_df = macro_df.join(jpx_db, how='inner').ffill().fillna(0.0)

    # 特徴量算出
    pin_strike = (macro_df['call_oi_wall'] + macro_df['put_oi_wall']) / 2.0
    macro_df['pin_dist_ratio'] = ((macro_df['NK_Close'] - pin_strike) / (pin_strike + 1e-5)) / 0.02
    macro_df['wall_spread'] = ((macro_df['call_oi_wall'] - macro_df['put_oi_wall']).abs() / (pin_strike + 1e-5)) / 0.02

    cta_mean = macro_df['cta_net_futures'].rolling(60, min_periods=10).mean()
    cta_std = macro_df['cta_net_futures'].rolling(60, min_periods=10).std() + 1e-5
    macro_df['cta_net_norm'] = ((macro_df['cta_net_futures'] - cta_mean) / cta_std).fillna(0.0)

    macro_df['cta_momentum'] = (macro_df['cta_net_futures'] - macro_df['cta_net_futures'].shift(5)).fillna(0.0) / 5000.0
    macro_df['nk_ret_norm'] = macro_df['NK_Ret'] / 0.015

    return macro_df

# =============================================================================
# 3. 実行メイン処理
# =============================================================================
def main():
    print("=" * 95)
    print("【v6スイングモデル 確定建玉＋CTA先物 統合型スクリーニング (EV最適化版)】")
    print("=" * 95)

    sync_latest_csv()

    if not os.path.exists(MODEL_WEIGHTS):
        print(f"[!] モデル重みが見つかりません: {MODEL_WEIGHTS}")
        return

    checkpoint = torch.load(MODEL_WEIGHTS, map_location=DEVICE)
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

    macro_df = load_macro_environment()
    latest_macro = macro_df.iloc[-1]

    pin_dist = latest_macro['pin_dist_ratio']
    cta_norm = latest_macro['cta_net_norm']
    cta_mom = latest_macro['cta_momentum']
    cta_raw = latest_macro.get('cta_net_futures', 0.0)

    # 地合いレジーム判定
    is_bear_regime = (pin_dist < -1.0) or (cta_norm < -0.8 and cta_mom < 0.0)
    
    # バックテスト検証に基づく最適閾値
    strong_buy_th = 0.365 if is_bear_regime else 0.360
    buy_threshold = 0.355 if is_bear_regime else 0.350
    watch_threshold = 0.345 if is_bear_regime else 0.340

    print(f"\n[★] 現在のマクロ需給環境認識:")
    print(f"  - 地合いレジーム: {'【警戒・下落加速リスク (BEAR)】' if is_bear_regime else '【通常・押し目有効 (BULL/NEUTRAL)】'}")
    print(f"  - ピン留め水準乖離: {pin_dist*2.0:+.2f}% ({'支持線割れ警戒' if pin_dist < -1.0 else '支持・引力圏内'})")
    print(f"  - CTA先物純建玉: {cta_raw:+.1f} 枚 (Z-Score: {cta_norm:+.2f}, 5日勢い: {cta_mom:+.2f})")
    print(f"  - 適用買閾値: STRONG BUY >= {strong_buy_th:.3f} | BUY >= {buy_threshold:.3f} | WATCH >= {watch_threshold:.3f}\n")

    try:
        raw_df = pd.read_csv(SCREENER_CSV, encoding='cp932')
    except Exception:
        raw_df = pd.read_csv(SCREENER_CSV, encoding='utf-8')

    code_col = [c for c in raw_df.columns if "コード" in str(c)][0]

    # 修正後（4桁数字、または数字3桁+英字1文字に対応）:
    tickers = [
        f"{str(c).strip()}.T" 
        for c in raw_df[code_col] 
        if re.match(r"^[0-9]{4}$|^[0-9]{3}[A-Z]$", str(c).strip().upper())
    ]

    print(f"[*] スキャン対象母集団: {len(tickers)} 銘柄 (売買代金10億円以上フィルター適用)")

    macro_feed = macro_df[macro_cols]
    m_start = (macro_df.index.min() - datetime.timedelta(days=40)).strftime("%Y-%m-%d")

    candidates = []

    for i, t in enumerate(tickers):
        try:
            df = yf.download(t, start=m_start, interval="1d", progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df = df.dropna(subset=['Close', 'High', 'Low', 'Volume'])
            
            # データ不足・取引停止スキップ
            if len(df) < SEQ_LEN + 20 or (df['Volume'] == 0).all():
                continue

            turnover_5d = (df['Close'] * df['Volume']).rolling(5).mean().iloc[-1]
            if turnover_5d < MIN_TURNOVER:
                continue

            df['stock_ret_1d'] = df['Close'].pct_change(1, fill_method=None).fillna(0.0)
            df['stock_ret_5d'] = df['Close'].pct_change(5, fill_method=None).fillna(0.0)
            vol_5d = df['Volume'].rolling(5).mean()
            df['vol_ratio_5d'] = (df['Volume'] / (vol_5d + 1e-7)).fillna(1.0)

            hl = df['High'] - df['Low']
            h_cp = (df['High'] - df['Close'].shift(1)).abs()
            l_cp = (df['Low'] - df['Close'].shift(1)).abs()
            tr = pd.concat([hl, h_cp, l_cp], axis=1).max(axis=1)
            atr = tr.rolling(14).mean()
            df['ATR'] = atr
            df['atr_ratio'] = (atr / (df['Close'] + 1e-7)).fillna(0.0)

            # 生リターン基準のベータ
            aligned_nk = macro_df['NK_Ret'].reindex(df.index).fillna(0.0)
            cov = df['stock_ret_1d'].rolling(20).cov(aligned_nk)
            var = aligned_nk.rolling(20).var()
            df['rolling_beta'] = (cov / (var + 1e-7)).fillna(1.0)

            df = df.join(macro_feed, how='inner')
            df = df.dropna(subset=['ATR', 'rolling_beta'] + macro_cols)
            if len(df) < SEQ_LEN:
                continue

            w_s = df[stock_cols].values[-SEQ_LEN:].copy()
            w_m = df[macro_cols].values[-SEQ_LEN:].copy()

            w_s_norm = (w_s - w_s.mean(axis=0)) / (w_s.std(axis=0) + 1e-7)

            t_s = torch.tensor(w_s_norm, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            t_m = torch.tensor(w_m, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                probs = torch.softmax(model(t_s, t_m), dim=-1).squeeze(0).cpu().numpy()

            p_stop, p_wait, p_win = float(probs[0]), float(probs[1]), float(probs[2])
            beta = float(df['rolling_beta'].iloc[-1])
            curr_close = float(df['Close'].iloc[-1])
            curr_atr = float(df['ATR'].iloc[-1])
            vol_ratio = float(df['vol_ratio_5d'].iloc[-1])

            # 期待値スコア (Rベース: 2.0 * Win - 1.0 * Stop)
            ev_score = round(2.0 * p_win - 1.0 * p_stop, 3)

            action = "⏸️ WAIT"
            gate_reason = "見送り"

            # 1. 空売り・ヘッジ判定 (⚠️ SHORT / HEDGE)
            is_hedge_candidate = is_bear_regime and (beta >= 1.15) and (p_stop >= 0.500) and (p_stop - p_win >= 0.15)

            if is_hedge_candidate:
                action = "⚠️ SHORT / HEDGE"
                gate_reason = f"地合い連動下落ヘッジ(β={beta:.2f}, 損率={p_stop*100:.1f}%)"

            # 2. 買いシグナル判定
            elif p_win >= strong_buy_th and p_win > p_stop:
                action = "🔥 STRONG BUY"
                gate_reason = f"本買い適合(勝率{p_win*100:.1f}%, EV={ev_score:+.2f}R)"
            
            elif p_win >= buy_threshold and p_win > p_stop and vol_ratio >= 0.85:
                if is_bear_regime and beta >= 1.0:
                    action = "⏸️ WAIT"
                    gate_reason = f"地合い悪化時の高β見送り(β={beta:.2f})"
                else:
                    action = "🎯 BUY"
                    gate_reason = f"打診買い適合(勝率{p_win*100:.1f}%, 出来高{vol_ratio:.2f}x)"

            elif p_win >= watch_threshold and p_win > p_stop:
                action = "👀 WATCH"
                gate_reason = f"監視対象(勝率{p_win*100:.1f}%)"

            if action == "⚠️ SHORT / HEDGE":
                target_price = round(curr_close - 2.0 * curr_atr, 1)
                stop_price = round(curr_close + 1.0 * curr_atr, 1)
            else:
                target_price = round(curr_close + 2.0 * curr_atr, 1)
                stop_price = round(curr_close - 1.0 * curr_atr, 1)

            candidates.append({
                'ticker': t,
                'price': curr_close,
                'target_price': target_price,
                'stop_price': stop_price,
                'prob_win': round(p_win, 4),
                'prob_stop': round(p_stop, 4),
                'ev_score': ev_score,
                'beta': round(beta, 2),
                'vol_ratio': round(vol_ratio, 2),
                'turnover_oku': round(turnover_5d / 1e8, 1),
                'action': action,
                'reason': gate_reason
            })
        except Exception:
            continue

        if (i + 1) % 50 == 0 or (i + 1) == len(tickers):
            print(f"  --> {i + 1}/{len(tickers)} 銘柄 スキャン完了")

    res_df = pd.DataFrame(candidates)
    if res_df.empty:
        print("[-] 条件を満たす銘柄はありませんでした。")
        return

    priority_map = {
        "🔥 STRONG BUY": 1,
        "🎯 BUY": 2,
        "⚠️ SHORT / HEDGE": 3,
        "👀 WATCH": 4,
        "⏸️ WAIT": 5
    }
    res_df['priority'] = res_df['action'].map(priority_map)
    # アクション優先度 -> 期待値スコア(ev_score)の降順で整列
    res_df = res_df.sort_values(by=['priority', 'ev_score'], ascending=[True, False]).drop(columns=['priority'])

    res_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')

    print("\n" + "=" * 95)
    print("【動的スクリーニング結果 (推奨・ヘッジ・監視ピックアップ)】")
    print("=" * 95)

    pickups = res_df[res_df['action'] != "⏸️ WAIT"]
    show_cols = ['ticker', 'price', 'target_price', 'stop_price', 'prob_win', 'ev_score', 'beta', 'vol_ratio', 'action', 'reason']
    if not pickups.empty:
        print(pickups.head(25)[show_cols].to_string(index=False))
    else:
        print(res_df.head(15)[show_cols].to_string(index=False))

    print("=" * 95)
    action_counts = res_df['action'].value_counts().to_dict()
    print(f"[*] 判定サマリ: {action_counts}")
    print(f"[+] 全結果を保存しました: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()