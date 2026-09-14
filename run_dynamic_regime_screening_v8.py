# run_dynamic_regime_screening_v8.py
import os
import random
import datetime
import json
import numpy as np
import pandas as pd
import torch

from modules.data_loader import (
    load_screener_tickers,
    fetch_all_tickers_data,
    load_margin_cache,
    load_sector_data
)
from modules.sector_matcher import get_ticker_sector_sentiment
from modules.regime_detector import load_macro_environment, detect_macro_regime
from modules.model_inference import (
    load_trained_model,
    predict_probabilities,
    compute_supply_demand_factor,
    apply_odds_adjustment
)
from modules.risk_manager import (
    determine_sizing_factor,
    evaluate_screening_gate,
    calculate_target_stop_levels
)

BASE_DIR = r"F:\stockModel"
# ★ v8 正式モデル重み（勝率62.1%, PF 3.33）
MODEL_WEIGHTS = os.path.join(BASE_DIR, "swing_model_v8_timeout_refined.pt")
OUTPUT_CSV = os.path.join(BASE_DIR, "final_regime_screened_v8.csv")
OUTPUT_JSON = os.path.join(BASE_DIR, "data", "screening_results_v8.json")

MIN_TURNOVER = 10e8
SEQ_LEN = 10

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    set_seed(42)
    print("=" * 95)
    print("【v8スイングモデル 高速一括スクリーニング (Pre-LN Transformer + 大口手口統合版)】")
    print("=" * 95)

    if not os.path.exists(MODEL_WEIGHTS):
        print(f"[!] モデル重みが見つかりません: {MODEL_WEIGHTS}")
        return

    # modules/model_inference.py 経由で v8 モデルを自動読み込み
    model, stock_cols, macro_cols = load_trained_model(MODEL_WEIGHTS)
    macro_df = load_macro_environment()

    regime = detect_macro_regime(macro_df)

    margin_cache = load_margin_cache()
    sentiment_map, master_map = load_sector_data()

    print(f"[*] 信用需給キャッシュ: {len(margin_cache)} 銘柄ロード済み")
    print(f"[*] セクターセンチメント: {len(sentiment_map)} セクター | 銘柄マスター: {len(master_map)} 件")
    print(f"\n[★] マクロ環境: {'【BEAR レジーム】' if regime['is_bear'] else '【BULL/NEUTRAL レジーム】'}")
    print(f"   - ピン留め乖離: {regime['pin_dist']*2.0:+.2f}% | CTA純建玉: {regime['cta_raw']:+.1f} 枚 (Z: {regime['cta_norm']:+.2f})")
    print(f"   - 大口手口フロー: {regime['flow_desc']} (CTAシェア: {regime['cta_share']*100:.1f}%, J-NET: {regime['jnet_ratio']*100:.1f}%)")
    print(f"   - 採用ゲート閾値: STRONG BUY >= {regime['strong_buy_th']:.3f} | BUY >= {regime['buy_threshold']:.3f} | WATCH >= {regime['watch_threshold']:.3f}\n")

    tickers = load_screener_tickers()
    if not tickers:
        print("[-] スキャン対象銘柄が存在しません。")
        return

    print(f"[*] スキャン対象母集団: {len(tickers)} 銘柄")
    macro_feed = macro_df[macro_cols]
    m_start = (macro_df.index.min() - datetime.timedelta(days=40)).strftime("%Y-%m-%d")
    all_prices_df = fetch_all_tickers_data(tickers, m_start)

    candidates = []
    has_multi_tickers = isinstance(all_prices_df.columns, pd.MultiIndex)
    print(f"[*] メモリ上で特徴量算出 & GPU推論を開始...")

    for i, t in enumerate(tickers):
        try:
            if has_multi_tickers:
                if t not in all_prices_df.columns.levels[0]:
                    continue
                df = all_prices_df[t].copy()
            else:
                df = all_prices_df.copy()

            df.index = pd.to_datetime(df.index).tz_localize(None)
            df = df.dropna(subset=['Close', 'High', 'Low', 'Volume'])

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

            aligned_nk = macro_df['NK_Ret'].reindex(df.index).fillna(0.0)
            cov = df['stock_ret_1d'].rolling(20).cov(aligned_nk)
            var = aligned_nk.rolling(20).var()
            df['rolling_beta'] = (cov / (var + 1e-7)).fillna(1.0)

            df = df.join(macro_feed, how='left')
            df[macro_cols] = df[macro_cols].ffill()
            df = df.dropna(subset=['ATR', 'rolling_beta'] + macro_cols)
            
            if len(df) < SEQ_LEN:
                continue

            w_s = df[stock_cols].values[-SEQ_LEN:].copy()
            w_m = df[macro_cols].values[-SEQ_LEN:].copy()

            # v8 Transformerモデルによる確率推論
            p_win_raw, p_stop_raw, ev_raw = predict_probabilities(model, w_s, w_m)

            curr_close = float(df['Close'].iloc[-1])
            curr_open = float(df['Open'].iloc[-1])
            curr_atr = float(df['ATR'].iloc[-1])
            beta = float(df['rolling_beta'].iloc[-1])
            vol_ratio = float(df['vol_ratio_5d'].iloc[-1])
            ret_1d = float(df['stock_ret_1d'].iloc[-1])
            is_bear_candle = curr_close < curr_open

            # 信用需給補正
            m_item = margin_cache.get(t)
            sd_weight, days_to_clear = compute_supply_demand_factor(m_item, curr_close, turnover_5d)
            p_win_adj, p_stop_adj = apply_odds_adjustment(p_win_raw, p_stop_raw, sd_weight)

            # セクターセンチメント補正
            sec_info = get_ticker_sector_sentiment(t, sentiment_map, master_map)
            sec_name = sec_info.get("name", "") if sec_info else ""
            sec_score = float(sec_info.get("score", 0.0)) if sec_info else 0.0
            sec_shock = bool(sec_info.get("shock_detected", False)) if sec_info else False
            sec_advice = str(sec_info.get("action_advice", "通常")) if sec_info else "通常"
            sec_summary = str(sec_info.get("summary", "")) if sec_info else ""

            if sec_score != 0.0:
                sec_weight = 1.0 + (sec_score * 0.08)
                p_win_adj, p_stop_adj = apply_odds_adjustment(p_win_adj, p_stop_adj, sec_weight)

            ev_adj = round(2.0 * p_win_adj - 1.0 * p_stop_adj, 3)

            # ロット調整係数（フローシグナル反映）
            margin_ratio_val = float(m_item.get("margin_ratio", 1.0) if m_item else 1.0)
            size_factor, size_reason = determine_sizing_factor(
                ret_1d, is_bear_candle, days_to_clear, margin_ratio_val, sec_advice,
                flow_level=regime['flow_level']
            )

            # ゲート採否判定（v8閾値反映）
            action, gate_reason = evaluate_screening_gate(
                p_win=p_win_adj,
                p_stop=p_stop_adj,
                ev_adj=ev_adj,
                beta=beta,
                vol_ratio=vol_ratio,
                days_to_clear=days_to_clear,
                is_bear_regime=regime['is_bear'],
                strong_buy_th=regime['strong_buy_th'],
                buy_threshold=regime['buy_threshold'],
                watch_threshold=regime['watch_threshold'],
                sec_shock=sec_shock,
                sec_advice=sec_advice,
                sec_summary=sec_summary,
                flow_level=regime['flow_level']
            )

            target_price, stop_price = calculate_target_stop_levels(curr_close, curr_atr, action)

            candidates.append({
                'ticker': t,
                'price': curr_close,
                'target_price': target_price,
                'stop_price': stop_price,
                'prob_win': p_win_adj,
                'prob_win_raw': round(p_win_raw, 4),
                'prob_stop': p_stop_adj,
                'prob_stop_raw': round(p_stop_raw, 4),
                'ev_score': ev_adj,
                'ev_score_raw': ev_raw,
                'sd_weight': sd_weight,
                'days_to_clear': days_to_clear,
                'size_factor': size_factor,
                'size_reason': size_reason,
                'beta': round(beta, 2),
                'vol_ratio': round(vol_ratio, 2),
                'turnover_oku': round(turnover_5d / 1e8, 1),
                'action': action,
                'reason': gate_reason,
                'sector_name': sec_name,
                'sector_score': sec_score,
                'sector_shock': sec_shock,
                'sector_advice': sec_advice,
                'sector_summary': sec_summary,
                'macro_flow': regime['flow_level']
            })
        except Exception:
            continue

        if (i + 1) % 50 == 0 or (i + 1) == len(tickers):
            print(f"  --> {i + 1}/{len(tickers)} 銘柄 完了")

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
    res_df = res_df.sort_values(by=['priority', 'ev_score'], ascending=[True, False]).drop(columns=['priority'])

    res_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')

    try:
        json_payload = {
            "updated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "macro_regime": {
                "is_bear": regime['is_bear'],
                "pin_dist": regime['pin_dist'],
                "cta_raw": regime['cta_raw'],
                "flow_level": regime['flow_level'],
                "cta_share": regime['cta_share'],
                "jnet_ratio": regime['jnet_ratio'],
                "flow_desc": regime['flow_desc']
            },
            "results": res_df.to_dict(orient="records")
        }
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(json_payload, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[!] JSON保存スキップ: {e}")

    print("\n" + "=" * 95)
    print("【動的スクリーニング結果 (v8)】")
    print("=" * 95)

    pickups = res_df[res_df['action'] != "⏸️ WAIT"]
    show_cols = ['ticker', 'price', 'ev_score', 'prob_win', 'days_to_clear', 'size_factor', 'action', 'sector_name']
    if not pickups.empty:
        print(pickups.head(25)[show_cols].to_string(index=False))
    else:
        print(res_df.head(15)[show_cols].to_string(index=False))

    print("=" * 95)
    print(f"[*] 判定サマリ: {res_df['action'].value_counts().to_dict()}")
    print(f"[+] 保存完了: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()