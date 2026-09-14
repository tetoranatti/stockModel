# modules/regime_detector.py
import os
import json
import datetime
import pandas as pd
import yfinance as yf

BASE_DIR = r"F:\stockModel"
JPX_DB_PATH = os.path.join(BASE_DIR, "jpx_daily_features_db.csv")
FLOW_JSON_PATH = os.path.join(BASE_DIR, "macro_flow_signal.json")

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

    # 日経平均ではなく jpx_db のインデックスを主軸にする
    macro_df = pd.DataFrame(index=jpx_db.index)
    
    # n225 から Close を取得して左結合（未確定日は直近終値で ffill）
    macro_df = macro_df.join(n225[['Close']], how='left')
    macro_df.rename(columns={'Close': 'NK_Close'}, inplace=True)
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

    # v8 (勝率62%, PF 3.33) の分布に合わせた閾値設定
    strong_buy_th = 0.365 if is_bear_regime else 0.360   # ★ 特選ライン (PF 3.33)
    buy_threshold = 0.360 if is_bear_regime else 0.355   # 標準採用ライン (PF 2.35)
    watch_threshold = 0.350 if is_bear_regime else 0.345 # 監視ライン

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