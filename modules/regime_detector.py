# modules/regime_detector.py
import os
import json
import pandas as pd

BASE_DIR = r"F:\stockModel"
JPX_DB_PATH = os.path.join(BASE_DIR, "jpx_daily_features_db.csv")
FLOW_JSON_PATH = os.path.join(BASE_DIR, "macro_flow_signal.json")
NK225_UNDERLYING_CACHE_PATH = os.path.join(BASE_DIR, "data", "cache", "train_nk225_underlying.parquet")

def load_macro_environment():
    if not os.path.exists(JPX_DB_PATH):
        raise FileNotFoundError(f"[!] {JPX_DB_PATH} が見つかりません。")
    if not os.path.exists(NK225_UNDERLYING_CACHE_PATH):
        raise FileNotFoundError(
            f"[!] {NK225_UNDERLYING_CACHE_PATH} が見つかりません。"
            f" 先に build_jquants_cache.py を実行してキャッシュを生成してください。"
        )

    jpx_db = pd.read_csv(JPX_DB_PATH, index_col=0, parse_dates=True)
    jpx_db.index = pd.to_datetime(jpx_db.index).tz_localize(None)

    # 学習・バックテストと同じ日経225原証券価格(J-Quantsオプションのunderlying)を使う
    n225 = pd.read_parquet(NK225_UNDERLYING_CACHE_PATH)
    n225.index = pd.to_datetime(n225.index).tz_localize(None)

    # 日経平均ではなく jpx_db のインデックスを主軸にする
    macro_df = pd.DataFrame(index=jpx_db.index)

    # n225 から NK_Close を取得して左結合（未確定日は直近終値で ffill）
    macro_df = macro_df.join(n225[['NK_Close']], how='left')
    macro_df['NK_Close'] = macro_df['NK_Close'].ffill()
    macro_df['NK_Ret'] = macro_df['NK_Close'].pct_change(fill_method=None).fillna(0.0)

    # jpx_db の全特徴量を左結合
    macro_df = macro_df.join(jpx_db, how='left').ffill().fillna(0.0)

    pin_strike = (macro_df['call_oi_wall'] + macro_df['put_oi_wall']) / 2.0
    macro_df['pin_dist_ratio'] = ((macro_df['NK_Close'] - pin_strike) / (pin_strike + 1e-5)) / 0.02
    macro_df['wall_spread'] = ((macro_df['call_oi_wall'] - macro_df['put_oi_wall']).abs() / (pin_strike + 1e-5)) / 0.02

    cta_mean = macro_df['cta_net_futures'].rolling(60, min_periods=10).mean()
    cta_std = macro_df['cta_net_futures'].rolling(60, min_periods=10).std() + 1e-5
    macro_df['cta_net_norm'] = ((macro_df['cta_net_futures'] - cta_mean) / cta_std).fillna(0.0)

    macro_df['cta_momentum'] = (macro_df['cta_net_futures'] - macro_df['cta_net_futures'].shift(5)).fillna(0.0) / 5000.0
    macro_df['nk_ret_norm'] = macro_df['NK_Ret'] / 0.015

    return macro_df

def detect_macro_regime(macro_df):
    latest_macro = macro_df.iloc[-1]
    pin_dist = latest_macro['pin_dist_ratio']
    cta_norm = latest_macro['cta_net_norm']
    cta_mom = latest_macro['cta_momentum']
    cta_raw = latest_macro.get('cta_net_futures', 0.0)

    is_bear_regime = (pin_dist < -1.0) or (cta_norm < -0.8 and cta_mom < 0.0)

    flow_level = "NORMAL"
    cta_share = 0.0
    jnet_ratio = 0.0
    flow_desc = "手口平常"

    if os.path.exists(FLOW_JSON_PATH):
        try:
            with open(FLOW_JSON_PATH, "r", encoding="utf-8") as f:
                flow_data = json.load(f)
                flow_level = flow_data.get("level", "NORMAL")
                cta_share = float(flow_data.get("cta_share", 0.0))
                jnet_ratio = float(flow_data.get("jnet_ratio", 0.0))
                flow_desc = flow_data.get("signal_desc", "手口平常")
        except Exception:
            pass

    # v8アンサンブル(ランキング損失+横断面正規化、seeds=42-46)再学習に伴い、
    # p_win分布が旧CE分類モデル(0.18〜0.44)から大きくシフトしたため閾値を再スキャン
    # (run_event_driven_backtest_v8_exp.pyの実測値、全体p_win分布: Min=0.138 Median=0.365 Max=0.661):
    #   th=0.535 n=1078 勝率46.8% PF1.65 / th=0.498 n=2459 勝率42.1% PF1.36 / th=0.424 n=6946 勝率39.3% PF1.14
    # bear/bullの相対差(旧版+0.005)は未検証のため据え置きで比例スケールしただけの暫定値。
    strong_buy_th = 0.545 if is_bear_regime else 0.535   # ★ 特選ライン
    buy_threshold = 0.505 if is_bear_regime else 0.495   # 標準採用ライン
    watch_threshold = 0.430 if is_bear_regime else 0.420 # 監視ライン

    return {
        "is_bear": is_bear_regime,
        "pin_dist": pin_dist,
        "cta_norm": cta_norm,
        "cta_mom": cta_mom,
        "cta_raw": cta_raw,
        "flow_level": flow_level,
        "cta_share": cta_share,
        "jnet_ratio": jnet_ratio,
        "flow_desc": flow_desc,
        "strong_buy_th": strong_buy_th,
        "buy_threshold": buy_threshold,
        "watch_threshold": watch_threshold
    }