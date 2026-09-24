# run_dynamic_regime_screening_v8.py
import os
import random
import datetime
import json
import numpy as np
import pandas as pd
import torch

# pipeline/配下からでもmodules/を解決できるようにプロジェクトルートをsys.pathへ追加
import sys
sys.path.insert(0, r"F:\stockModel")
from modules.data_loader import (
    load_screener_tickers,
    fetch_all_tickers_data,
    load_margin_cache,
    load_sector_data
)
from modules.sector_matcher import get_ticker_sector_sentiment
from modules.regime_detector import load_macro_environment, detect_macro_regime
from modules.model_inference import (
    load_trained_models_ensemble,
    predict_probabilities_ensemble_batch,
    compute_supply_demand_factor,
    apply_odds_adjustment
)
from modules.risk_manager import (
    determine_sizing_factor,
    evaluate_screening_gate,
    calculate_target_stop_levels,
    calculate_target_stop_levels_short,
    calculate_recommended_position,
    build_portfolio,
    DEFAULT_CAPITAL,
    DEFAULT_RISK_PCT,
    DEFAULT_LEVERAGE,
    DEFAULT_MAX_POSITION_PCT,
)
from modules.regime_momentum import determine_capital_split
from modules.sector_embedding import TICKER_TO_SECTOR_ID
from modules.regime_risk_model import (
    add_regime_features,
    compute_breadth_5d,
    load_regime_risk_ensemble,
    predict_regime_risk,
    determine_regime_zone,
    append_regime_risk_cache,
)
from modules.short_edge_model import (
    SHORT_EDGE_COLS_NO_FLOW,
    build_idx_momentum_features,
    build_vix_features,
    load_short_edge_ensemble,
    predict_short_edge,
    determine_short_edge_size_mult,
    append_short_edge_cache,
)
from modules.stock_features import compute_stock_features, compute_short_model_extra_features
from modules.cross_sectional_features import (
    apply_log_transform,
    compute_cross_sectional_stats,
    valid_cross_section_dates,
    normalize_cross_sectional,
)

BASE_DIR = r"F:\stockModel"
# ★ v8 アンサンブルモデル(ランキング損失+横断面正規化)。旧モデルの閾値
#    (勝率52〜53%, PF1.8〜2.2程度)はp_win分布が異なるため参考にならない。
#    最新の閾値表はrun_event_driven_backtest_v8_exp.pyの実行結果を参照。
ENSEMBLE_SEEDS = [42, 43, 44, 45, 46]
MODEL_WEIGHTS_LIST = [os.path.join(BASE_DIR, f"swing_model_v8_ensemble_seed{s}.pt") for s in ENSEMBLE_SEEDS]
REGIME_RISK_WEIGHTS_LIST = [os.path.join(BASE_DIR, f"regime_risk_model_seed{s}.pt") for s in ENSEMBLE_SEEDS]
SHORT_EDGE_WEIGHTS_LIST = [os.path.join(BASE_DIR, f"short_edge_noflow_model_seed{s}.pt") for s in ENSEMBLE_SEEDS]
# ★ 空売りモデルv2(2026-09-24、実運用へ初投入。research/short_exp/train_short_model_v2.py、
#   9 stock + 5 macro、USE_SECTOR_EMBEDDING=True、SEQ_LEN=5)。SHORT_EDGE_*(翌日TOPIX下落
#   確率でロングサイズを絞るだけの既存モデル)とは別物——こちらは個別銘柄をランキングして
#   実際にショートポジションを建てる。scripts/export_short_model_v2_to_production.pyで
#   research/cache/short_model_cache_v2から本番形式に変換して配置する。
SHORT_MODEL_WEIGHTS_LIST = [os.path.join(BASE_DIR, f"short_model_v2_seed{s}.pt") for s in ENSEMBLE_SEEDS]
SHORT_STOCK_COLS_V2 = [
    'stock_ret_1d', 'stock_ret_20d', 'rolling_beta', 'vol_ratio_5d', 'overnight_gap',
    'dist_from_low20', 'gap_strength_5', 'atr_accel', 'dist_from_high60',
]
SHORT_SEQ_LEN = 5  # ロング(SEQ_LEN=10)とは別(research/stage3_exp/config.py::SEQ_LEN=5に合わせる)
SHORT_DAILY_TOPN = 5  # walk-forward検証済み設定([[regime_gated_long_short_blend_2026-09-24]])
VIX_CSV_PATH = os.path.join(BASE_DIR, "data", "vix_fred.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "final_regime_screened_v8.csv")
OUTPUT_CSV_SHORT = os.path.join(BASE_DIR, "final_regime_screened_v8_short.csv")
OUTPUT_JSON = os.path.join(BASE_DIR, "data", "screening_results_v8.json")

MIN_TURNOVER = 10e8
SEQ_LEN = 10
MIN_CROSS_SECTION = 20
BASE_K = 20  # 地合い危険度NEUTRAL時の採用銘柄数(STRONG BUY+BUY合算の上限)

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

    missing = [p for p in MODEL_WEIGHTS_LIST if not os.path.exists(p)]
    if missing:
        print(f"[!] モデル重みが見つかりません: {missing}")
        return

    # modules/model_inference.py 経由で v8 アンサンブルモデルを自動読み込み
    models, stock_cols, macro_cols = load_trained_models_ensemble(MODEL_WEIGHTS_LIST)
    print(f"[*] アンサンブル {len(models)} モデルを読み込みました (seeds={ENSEMBLE_SEEDS})")

    # 空売りモデルv2(見つからなければ実弾ショートはスキップし、ロングのみで継続)
    missing_short_model = [p for p in SHORT_MODEL_WEIGHTS_LIST if not os.path.exists(p)]
    short_models = None
    if missing_short_model:
        print(f"[!] 空売りモデルv2の重みが見つからないためショートは無効で継続: {missing_short_model}")
    else:
        short_models, short_stock_cols, short_macro_cols = load_trained_models_ensemble(SHORT_MODEL_WEIGHTS_LIST)
        print(f"[*] 空売りアンサンブル {len(short_models)} モデルを読み込みました (seeds={ENSEMBLE_SEEDS})")

    macro_df = load_macro_environment()

    regime = detect_macro_regime(macro_df)

    # モメンタムレジームゲート(2026-09-24追加、[[regime_gated_long_short_blend_2026-09-24]]で
    # 5-fold walk-forward検証済み)。ロング/ショートの資金配分をここで決める。
    capital_split = determine_capital_split(macro_df, DEFAULT_CAPITAL)
    print(f"[★] モメンタムレジーム: {capital_split['zone']} -> "
          f"ロング資金{capital_split['capital_long']:,}円(w={capital_split['w_long']:.2f}) / "
          f"ショート資金{capital_split['capital_short']:,}円(w={capital_split['w_short']:.2f})\n")

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
    all_prices_df = fetch_all_tickers_data()

    has_multi_tickers = isinstance(all_prices_df.columns, pd.MultiIndex)
    print(f"[*] メモリ上で特徴量算出中...")

    # --- 1パス目: 銘柄ごとの生特徴量(ログ変換込み)を計算 ---
    per_ticker_df = {}
    turnover_by_ticker = {}
    for t in tickers:
        try:
            if has_multi_tickers:
                if t not in all_prices_df.columns.levels[0]:
                    continue
                df = all_prices_df[t].copy()
            else:
                df = all_prices_df.copy()

            df.index = pd.to_datetime(df.index).tz_localize(None)
            df = df.dropna(subset=['Close', 'High', 'Low', 'Volume'])

            if len(df) < SEQ_LEN + 95 or (df['Volume'] == 0).all():  # +95 = rolling_beta(90日窓)+余裕
                continue

            # 必要最小限の窓(SEQ_LEN+95)にスライスしてから特徴量計算・ログ変換・横断面統計に
            # 回す。最終的にモデル入力に使うのは末尾SEQ_LEN行だけなので、フェッチ時の
            # ~130日分をそのまま保持する必要は無い。銘柄数が数百〜数千に増えても
            # per_ticker_df/apply_log_transform/横断面統計のメモリ・計算量が
            # 不必要に伸びないようにするため、ここで早めに切り詰める。
            df = df.tail(SEQ_LEN + 95).copy()

            turnover_5d = (df['Close'] * df['Volume']).rolling(5).mean().iloc[-1]
            if turnover_5d < MIN_TURNOVER:
                continue

            aligned_nk = macro_df['NK_Ret'].reindex(df.index).fillna(0.0)
            df = compute_stock_features(df, aligned_nk)
            # 空売りモデルv2(SHORT_STOCK_COLS_V2)専用の追加特徴量(gap_strength_5/atr_accel/
            # dist_from_high60)。ATR/overnight_gap列が必要なためcompute_stock_features()の後で
            # 呼ぶ(2026-09-24追加)。
            df = compute_short_model_extra_features(df)

            df = df.join(macro_feed, how='left')
            df[macro_cols] = df[macro_cols].ffill()
            df = df.dropna(subset=['ATR', 'rolling_beta'] + macro_cols)

            # 注: rolling_beta は compute_stock_features() 内で .fillna(1.0) されているため、
            # 90日窓のウォームアップ期間(未成熟な値)もNaNではなく1.0で埋まって上のdropnaを
            # すり抜ける(学習・バックテストと挙動を揃えるため意図的にそうなっている共有関数
            # なので、ここでは変更しない)。そのため末尾のみ明示的に切り出し、
            # ウォームアップ分を確実に除いた「本当に成熟したbeta値を持つ行」だけを保持する。
            # SEQ_LEN+95でスライスした時点でBETA_WINDOW(90日)分のウォームアップを差し引いても
            # 常にSEQ_LEN+5行以上残る計算のため、このtailで安全にウォームアップ後の範囲に収まる。
            df = df.tail(SEQ_LEN + 5)

            if len(df) < SEQ_LEN:
                continue

            per_ticker_df[t] = df
            turnover_by_ticker[t] = turnover_5d
        except Exception:
            continue

    # --- 横断面統計(同日の全銘柄基準) ---
    print(f"[*] 横断面統計(同日の全銘柄基準)を計算中... 対象{len(per_ticker_df)}銘柄")
    per_ticker_log = {t: apply_log_transform(df) for t, df in per_ticker_df.items()}
    cross_mean, cross_std = compute_cross_sectional_stats(per_ticker_log, stock_cols)
    valid_dates = valid_cross_section_dates(per_ticker_log, stock_cols, MIN_CROSS_SECTION)

    # --- 地合い危険度モデル: 日次集約STOP率(個別銘柄ではなく市場全体のリスク)を予測し、
    # 採用銘柄数K・ロットサイズを調整する ---
    regime_zone_info = {"zone": "NEUTRAL", "k": BASE_K, "size_mult": 1.0}
    regime_risk_score = None
    missing_regime = [p for p in REGIME_RISK_WEIGHTS_LIST if not os.path.exists(p)]
    if missing_regime:
        print(f"[!] 地合い危険度モデルの重みが見つからないためNEUTRAL固定で継続: {missing_regime}")
    else:
        try:
            macro_regime_df = add_regime_features(macro_df)
            per_ticker_ret1d = {t: df['stock_ret_1d'] for t, df in per_ticker_df.items()}
            macro_regime_df['breadth_5d'] = compute_breadth_5d(per_ticker_ret1d, macro_regime_df.index)
            latest_regime_row = macro_regime_df.iloc[-1]

            regime_models, regime_cols, feat_mean, feat_std = load_regime_risk_ensemble(REGIME_RISK_WEIGHTS_LIST)
            regime_risk_score = predict_regime_risk(regime_models, regime_cols, feat_mean, feat_std, latest_regime_row)
            regime_zone_info = determine_regime_zone(regime_risk_score, base_k=BASE_K)
            print(f"[★] 地合い危険度スコア: {regime_risk_score:.3f} -> {regime_zone_info['zone']}ゾーン "
                  f"(K={regime_zone_info['k']}, サイズ倍率={regime_zone_info['size_mult']:.2f})\n")

            n_cached_days = append_regime_risk_cache(latest_regime_row.name, regime_risk_score, regime_zone_info)
            print(f"[*] 地合い危険度スコアを日次キャッシュに追記(累計{n_cached_days}日分)\n")
        except Exception as e:
            print(f"[!] 地合い危険度モデルの推論に失敗したためNEUTRAL固定で継続: {e}")

    candidates = []
    print(f"[*] アンサンブル推論を開始...")

    # 空売り機会モデル(short_edge)用に、全銘柄の生p_win/p_stop・出来高比率を集めておく
    # (ロング候補に絞る前の全スキャン対象が対象。個別銘柄補正前の生値を使うのは
    # 学習時と定義を揃えるため)
    short_edge_p_win, short_edge_p_stop, short_edge_vol_ratio = {}, {}, {}

    # 全銘柄分のw_s/w_mを先にまとめてバッチ推論する(2026-09-18、1件ずつ推論していたのを
    # スキャン対象銘柄まとめて1回のforward呼び出しに変更。.eval()+LayerNormのみ
    # (BatchNorm不使用)なので結果は数値的に完全一致し、純粋な高速化になる)。
    batch_tickers, w_s_list, w_m_list = [], [], []
    for t in per_ticker_df.keys():
        try:
            df = per_ticker_df[t]
            df_log = per_ticker_log[t].loc[per_ticker_log[t].index.isin(valid_dates)]
            if len(df_log) < SEQ_LEN:
                continue
            norm_s = normalize_cross_sectional(df_log, cross_mean, cross_std, stock_cols)
            w_s = norm_s.values[-SEQ_LEN:]
            if np.isnan(w_s).any():
                continue
            w_m = df.loc[df_log.index, macro_cols].values[-SEQ_LEN:]
            batch_tickers.append(t)
            w_s_list.append(w_s)
            w_m_list.append(w_m)
        except Exception:
            continue

    pred_map = {}
    if batch_tickers:
        p_win_arr, p_stop_arr, ev_arr = predict_probabilities_ensemble_batch(
            models, np.stack(w_s_list), np.stack(w_m_list)
        )
        pred_map = {t: (float(p_win_arr[k]), float(p_stop_arr[k]), float(ev_arr[k]))
                    for k, t in enumerate(batch_tickers)}

    # --- 空売りモデルv2: ロングとは別のstock_cols(SHORT_STOCK_COLS_V2)・別の横断面統計・
    # 別のSEQ_LEN(5)でバッチ推論する(2026-09-24追加)。sector_idをw_sの最終列に定数値として
    # 埋め込む(research/stage3_exp/data_builder.py::build_dataset_shortと同じ規約、
    # DualStream_GRU_PreLN_Transformer.forward()側でこの最終列を切り離してnn.Embeddingに通す)。
    short_pred_map = {}
    if short_models is not None:
        short_cross_mean, short_cross_std = compute_cross_sectional_stats(per_ticker_log, SHORT_STOCK_COLS_V2)
        short_valid_dates = valid_cross_section_dates(per_ticker_log, SHORT_STOCK_COLS_V2, MIN_CROSS_SECTION)

        short_batch_tickers, short_w_s_list, short_w_m_list = [], [], []
        for t in per_ticker_df.keys():
            try:
                df = per_ticker_df[t]
                df_log = per_ticker_log[t].loc[per_ticker_log[t].index.isin(short_valid_dates)]
                if len(df_log) < SHORT_SEQ_LEN:
                    continue
                norm_s = normalize_cross_sectional(df_log, short_cross_mean, short_cross_std, SHORT_STOCK_COLS_V2)
                w_s = norm_s.values[-SHORT_SEQ_LEN:]
                if np.isnan(w_s).any():
                    continue
                sec_id = TICKER_TO_SECTOR_ID.get(t, 0)
                w_s = np.concatenate([w_s, np.full((SHORT_SEQ_LEN, 1), sec_id, dtype=w_s.dtype)], axis=1)
                w_m = df.loc[df_log.index, macro_cols].values[-SHORT_SEQ_LEN:]
                short_batch_tickers.append(t)
                short_w_s_list.append(w_s)
                short_w_m_list.append(w_m)
            except Exception:
                continue

        if short_batch_tickers:
            sp_win_arr, sp_stop_arr, sev_arr = predict_probabilities_ensemble_batch(
                short_models, np.stack(short_w_s_list), np.stack(short_w_m_list)
            )
            short_pred_map = {t: (float(sp_win_arr[k]), float(sp_stop_arr[k]), float(sev_arr[k]))
                               for k, t in enumerate(short_batch_tickers)}
        print(f"[*] 空売りモデルv2 推論完了: {len(short_pred_map)}銘柄")

    for i, t in enumerate(per_ticker_df.keys()):
        try:
            if t not in pred_map:
                continue
            df = per_ticker_df[t]
            df_log = per_ticker_log[t].loc[per_ticker_log[t].index.isin(valid_dates)]

            turnover_5d = turnover_by_ticker[t]

            # v8 アンサンブルモデルによる確率推論(予測平均、上でバッチ計算済み)
            p_win_raw, p_stop_raw, ev_raw = pred_map[t]

            # 表示・ゲート判定には正規化前の生の値を使う(解釈性のため)
            latest_date = df_log.index[-1]
            df_latest = df.loc[latest_date]
            curr_close = float(df_latest['Close'])
            curr_open = float(df_latest['Open'])
            curr_atr = float(df_latest['ATR'])
            beta = float(df_latest['rolling_beta'])
            vol_ratio = float(df_latest['vol_ratio_5d'])
            ret_1d = float(df_latest['stock_ret_1d'])
            is_bear_candle = curr_close < curr_open

            short_edge_p_win[t] = p_win_raw
            short_edge_p_stop[t] = p_stop_raw
            short_edge_vol_ratio[t] = vol_ratio

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

            # ロット調整係数（セクターセンチメント + 地合い危険度ゾーンによる調整）
            margin_ratio_val = float(m_item.get("margin_ratio", 1.0) if m_item else 1.0)
            size_factor, size_reason = determine_sizing_factor(
                ret_1d, is_bear_candle, days_to_clear, margin_ratio_val, sec_score
            )
            if regime_zone_info['size_mult'] != 1.0:
                size_factor = round(size_factor * regime_zone_info['size_mult'], 2)
                size_reason += f" ＋ 地合い危険度{regime_zone_info['zone']}({regime_zone_info['size_mult']:.2f}倍)"

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
            )

            target_price, stop_price = calculate_target_stop_levels(curr_close, curr_atr)

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

        if (i + 1) % 50 == 0 or (i + 1) == len(per_ticker_df):
            print(f"  --> {i + 1}/{len(per_ticker_df)} 銘柄 完了")

    res_df = pd.DataFrame(candidates)
    if res_df.empty:
        print("[-] 条件を満たす銘柄はありませんでした。")
        return

    priority_map = {
        "🔥 STRONG BUY": 1,
        "🎯 BUY": 2,
        "👀 WATCH": 3,
        "⏸️ WAIT": 4
    }
    res_df['priority'] = res_df['action'].map(priority_map)
    res_df = res_df.sort_values(by=['priority', 'ev_score'], ascending=[True, False]).drop(columns=['priority'])

    # 注: セクター単位のK上限(sec_score<0のセクターを3件までに絞る案)は、3ヶ月分の
    # 実データでバックテストしたところPFを明確に悪化させた(2.02->1.82、th=0.46)ため不採用。
    # 原因切り分けの結果、連続的なサイズ重み付け(determine_sizing_factor内)自体は無害
    # (むしろ微増)だったが、K上限はPFモデルが正しく評価していた優良銘柄を機械的に
    # 除外してしまっていた。そのためサイズ重み付けのみ残し、K上限は導入しない。

    # --- 地合い危険度ゾーンによるK(採用銘柄数)制限: STRONG BUY/BUYをev_score順にK件まで
    # 残し、それを超える分はWATCHへ降格する(情報は保持しつつ採用件数だけ絞る) ---
    k_final = regime_zone_info['k']
    actionable_mask = res_df['action'].isin(["🔥 STRONG BUY", "🎯 BUY"])
    actionable_idx = res_df[actionable_mask].index
    demote_idx = actionable_idx[k_final:]
    if len(demote_idx) > 0:
        res_df.loc[demote_idx, 'reason'] = res_df.loc[demote_idx, 'reason'] + f" ＋ 地合い危険度{regime_zone_info['zone']}によりK={k_final}超過で降格"
        res_df.loc[demote_idx, 'action'] = "👀 WATCH"

    # --- 空売り機会モデル(short_edge): 翌日TOPIX下落確率からロングサイズを事後調整 ---
    # (地合い危険度モデルと同じ日次集約の発想。市場悲観日はSTRONG BUYでも勝率が
    # 73%→44%まで落ちることをバックテストで確認済み。ここではロング縮小のみ適用し、
    # 空売り実行自体は行わない)
    short_edge_zone_info = {"zone": "NEUTRAL", "size_mult": 1.0}
    p_short_edge = None
    missing_short_edge = [p for p in SHORT_EDGE_WEIGHTS_LIST if not os.path.exists(p)]
    if missing_short_edge:
        print(f"[!] 空売り機会モデルの重みが見つからないためNEUTRAL固定で継続: {missing_short_edge}")
    elif not short_edge_p_win:
        print("[!] 空売り機会モデル用の推論データが無いためNEUTRAL固定で継続")
    else:
        try:
            idx_feat = build_idx_momentum_features(macro_df['NK_Close'])
            vix_feat = build_vix_features(VIX_CSV_PATH)
            latest_date = idx_feat.index.max()

            pw = np.array(list(short_edge_p_win.values()))
            ps = np.array([short_edge_p_stop[t] for t in short_edge_p_win.keys()])
            vr = np.array(list(short_edge_vol_ratio.values()))

            feature_row = {
                'idx_ret_1d': float(idx_feat.loc[latest_date, 'idx_ret_1d']),
                'idx_ret_5d': float(idx_feat.loc[latest_date, 'idx_ret_5d']),
                'idx_ret_20d': float(idx_feat.loc[latest_date, 'idx_ret_20d']),
                'vix_level_norm': float(vix_feat['vix_level_norm'].reindex(idx_feat.index).ffill().loc[latest_date]),
                'vix_change_norm': float(vix_feat['vix_change_norm'].reindex(idx_feat.index).ffill().loc[latest_date]),
                'vix_ma5_diff': float(vix_feat['vix_ma5_diff'].reindex(idx_feat.index).ffill().loc[latest_date]),
                'vix_zscore': float(vix_feat['vix_zscore'].reindex(idx_feat.index).ffill().loc[latest_date]),
                'cross_vol_ratio_mean': float(vr.mean()),
                'cross_vol_thin_pct': float((vr < 0.8).mean()),
                'pf_avg_p_win': float(pw.mean()),
                'pf_pct_bullish': float((pw > ps).mean()),
            }

            short_edge_models, se_cols, se_mean, se_std = load_short_edge_ensemble(SHORT_EDGE_WEIGHTS_LIST)
            p_short_edge = predict_short_edge(short_edge_models, se_cols, se_mean, se_std, feature_row)
            short_edge_zone_info = determine_short_edge_size_mult(p_short_edge)
            print(f"[★] 空売り機会スコア: {p_short_edge:.3f} -> {short_edge_zone_info['zone']}"
                  f" (ロングサイズ倍率={short_edge_zone_info['size_mult']:.2f})\n")

            n_cached_se = append_short_edge_cache(latest_date, p_short_edge, short_edge_zone_info)
            print(f"[*] 空売り機会スコアを日次キャッシュに追記(累計{n_cached_se}日分)\n")

            # 2026-09-24: ロングサイズを絞る効果は無効化した。同じ役割(上昇モメンタム悪化時に
            # ロングを抑える)を、実際にショートポジションを建てるモメンタムレジームゲート
            # (determine_capital_split、capital_long/capital_short)に統合したため
            # ([[regime_gated_long_short_blend_2026-09-24]])。p_short_edgeの算出・キャッシュ
            # 記録自体は監視用に残す([[feedback_keep_rejected_experiment_code]]の精神)。
        except Exception as e:
            print(f"[!] 空売り機会モデルの推論に失敗したためNEUTRAL固定で継続: {e}")

    # --- 推奨ポジションサイズの自動計算(UIのupdatePositionSize()と同じ式) ---
    # size_factor確定後(地合い危険度・空売り機会モデルの調整を全て反映した後)に計算する。
    # 注: これは「その銘柄だけを買う」前提のスタンドアロン参考値(standalone_*)。
    # 複数銘柄を同時に買う場合の資金配分は、この後のbuild_portfolio()側を見ること。
    position_rows = res_df.apply(
        lambda row: calculate_recommended_position(
            price=float(row['price']), stop_price=float(row['stop_price']), size_factor=float(row['size_factor'])
        ), axis=1
    )
    position_df = pd.DataFrame(list(position_rows)).rename(columns={
        'recommended_shares': 'standalone_shares',
        'estimated_cost_yen': 'standalone_cost_yen',
        'estimated_max_loss_yen': 'standalone_max_loss_yen',
        'leverage_capped': 'standalone_leverage_capped',
        'maxpos_capped': 'standalone_maxpos_capped',
    })
    res_df = pd.concat([res_df.reset_index(drop=True), position_df.reset_index(drop=True)], axis=1)
    n_leverage_capped = int(res_df['standalone_leverage_capped'].sum())
    print(f"[*] 推奨ポジションサイズ(単体参考値)計算完了(資金{DEFAULT_CAPITAL:,}円・リスク{DEFAULT_RISK_PCT}%"
          f"・レバレッジ{DEFAULT_LEVERAGE}倍換算、信用余力上限で縮小: {n_leverage_capped}件)")

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
            "regime_risk": {
                "score": regime_risk_score,
                "zone": regime_zone_info['zone'],
                "k": regime_zone_info['k'],
                "size_mult": regime_zone_info['size_mult'],
            },
            "short_edge": {
                "score": p_short_edge,
                "zone": short_edge_zone_info['zone'],
                "size_mult": short_edge_zone_info['size_mult'],
            },
            "momentum_regime": capital_split,
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
    if regime_risk_score is not None:
        print(f"[*] 地合い危険度: {regime_risk_score:.3f} ({regime_zone_info['zone']}ゾーン, K={regime_zone_info['k']}, サイズ倍率={regime_zone_info['size_mult']:.2f})")
    print(f"[*] 判定サマリ: {res_df['action'].value_counts().to_dict()}")

    # --- ポートフォリオ構築: 複数銘柄を同時に買う前提で資金・信用枠を共有しながら配分 ---
    # (standalone_sharesを単純合計すると資金・信用枠を超過しうるため、優先順位
    # [priority→ev_score、既にres_dfはこの順でソート済み]順に残り予算内で配分する)
    # max_positions=k_final を明示的に渡す(地合い危険度ゾーンによるK調整。渡さないと
    # build_portfolio()の既定値20に黙って再制限され、SAFEゾーンのK=25拡張が
    # 効かなくなるバグがあったため修正)
    # 資金はDEFAULT_CAPITAL全額ではなく、モメンタムレジームゲートが配分したcapital_longを使う
    # (2026-09-24追加。HIGH regimeではw_long=1.0でDEFAULT_CAPITALそのまま、LOW regimeでは
    # w_long=0.0になりロングは建てない——[[regime_gated_long_short_blend_2026-09-24]]で
    # walk-forward検証済みの強めのゲート)。
    actionable = res_df[res_df['action'].isin(["🔥 STRONG BUY", "🎯 BUY"])].copy()
    portfolio_rows, portfolio_summary = build_portfolio(
        actionable.to_dict(orient="records"), capital=capital_split['capital_long'], max_positions=k_final,
    )

    print("\n" + "=" * 95)
    print(f"【ロングポートフォリオ】(資金{capital_split['capital_long']:,}円[レジーム{capital_split['zone']} "
          f"w={capital_split['w_long']:.2f}]・リスク{DEFAULT_RISK_PCT}%・レバレッジ{DEFAULT_LEVERAGE}倍・"
          f"1銘柄上限{DEFAULT_MAX_POSITION_PCT*100:.0f}%・最大{k_final}銘柄[{regime_zone_info['zone']}])")
    print("=" * 95)
    if not portfolio_rows:
        print("  該当銘柄なし")
    else:
        pf_df = pd.DataFrame(portfolio_rows)
        pos_cols = ['ticker', 'price', 'action', 'recommended_shares', 'estimated_cost_yen', 'estimated_max_loss_yen', 'leverage_capped', 'maxpos_capped']
        print(pf_df[pos_cols].to_string(index=False))
        print(f"\n  採用銘柄数: {portfolio_summary['n_positions']} (候補{portfolio_summary['n_candidates_evaluated']}件中)")
        print(f"  信用余力使用: {portfolio_summary['used_buying_power_yen']:,}円 / "
              f"{portfolio_summary['max_buying_power']:,.0f}円 ({portfolio_summary['buying_power_usage_pct']:.1f}%)")
        print(f"  想定最大損失合計: {portfolio_summary['total_risk_yen']:,}円 "
              f"(資金比 {portfolio_summary['total_risk_yen']/max(1, capital_split['capital_long'])*100:.2f}%)")
        src = portfolio_summary.get('skip_reason_counts', {})
        skip_parts = []
        if src.get('risk'): skip_parts.append(f"損切幅超過{src['risk']}件")
        if src.get('leverage'): skip_parts.append(f"信用余力不足{src['leverage']}件")
        if src.get('maxpos'): skip_parts.append(f"1銘柄上限超過{src['maxpos']}件")
        if src.get('position_limit'): skip_parts.append(f"採用上限到達{src['position_limit']}件")
        if skip_parts:
            print(f"  見送り内訳: {' / '.join(skip_parts)}")
    print("=" * 95)

    try:
        portfolio_path = os.path.join(BASE_DIR, "data", "portfolio_latest.json")
        with open(portfolio_path, "w", encoding="utf-8") as f:
            json.dump({
                "updated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "summary": portfolio_summary,
                "positions": portfolio_rows,
            }, f, ensure_ascii=False, indent=2)
        print(f"[+] ポートフォリオ保存完了: {portfolio_path}")
    except Exception as e:
        print(f"[!] ポートフォリオ保存スキップ: {e}")

    # --- 空売りポートフォリオ構築(2026-09-24追加) ---
    # ロングと違い「勝率閾値によるゲート」は行わず、short_pred_mapのev_score降順で
    # 日次Top-N(SHORT_DAILY_TOPN)を選ぶ(research/short_exp側で検証済みの運用方式、
    # evaluate_daily_topnと同じ規約)。
    short_portfolio_rows, short_portfolio_summary = [], {}
    short_candidates_df = pd.DataFrame()
    if short_pred_map:
        short_rows = []
        for t, (p_win_s, p_stop_s, ev_s) in short_pred_map.items():
            df_latest = per_ticker_df[t].iloc[-1]
            curr_close = float(df_latest['Close'])
            curr_atr = float(df_latest['ATR'])
            target_price, stop_price = calculate_target_stop_levels_short(curr_close, curr_atr)
            short_rows.append({
                'ticker': t,
                'price': curr_close,
                'target_price': target_price,
                'stop_price': stop_price,
                'prob_win': p_win_s,
                'prob_stop': p_stop_s,
                'ev_score': ev_s,
                'size_factor': 1.0,
            })
        short_candidates_df = pd.DataFrame(short_rows).sort_values('ev_score', ascending=False)
        short_candidates_df.to_csv(OUTPUT_CSV_SHORT, index=False, encoding='utf-8-sig')

        short_topn = short_candidates_df.head(SHORT_DAILY_TOPN)
        short_portfolio_rows, short_portfolio_summary = build_portfolio(
            short_topn.to_dict(orient="records"), capital=capital_split['capital_short'],
            max_positions=SHORT_DAILY_TOPN, side="short",
        )

        print("\n" + "=" * 95)
        print(f"【ショートポートフォリオ】(資金{capital_split['capital_short']:,}円[レジーム{capital_split['zone']} "
              f"w={capital_split['w_short']:.2f}]・日次Top{SHORT_DAILY_TOPN})")
        print("=" * 95)
        if not short_portfolio_rows:
            print("  該当銘柄なし")
        else:
            spf_df = pd.DataFrame(short_portfolio_rows)
            print(spf_df[['ticker', 'price', 'ev_score', 'recommended_shares', 'estimated_cost_yen', 'estimated_max_loss_yen']].to_string(index=False))
            print(f"\n  採用銘柄数: {short_portfolio_summary['n_positions']} (候補{short_portfolio_summary['n_candidates_evaluated']}件中)")
        print("=" * 95)

        try:
            short_portfolio_path = os.path.join(BASE_DIR, "data", "portfolio_latest_short.json")
            with open(short_portfolio_path, "w", encoding="utf-8") as f:
                json.dump({
                    "updated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "regime": capital_split,
                    "summary": short_portfolio_summary,
                    "positions": short_portfolio_rows,
                }, f, ensure_ascii=False, indent=2)
            print(f"[+] ショートポートフォリオ保存完了: {short_portfolio_path}")
        except Exception as e:
            print(f"[!] ショートポートフォリオ保存スキップ: {e}")
    else:
        print("\n[!] 空売りモデルv2の推論結果が無いため、ショートポートフォリオはスキップ")

    print(f"[+] 保存完了: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()