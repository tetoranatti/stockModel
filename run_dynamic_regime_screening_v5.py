import datetime
import glob
import os
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yfinance as yf

# =============================================================================
# 設定・パス
# =============================================================================
BASE_DIR = r"F:\stockModel"
DOWNLOADS_DIR = os.path.join(os.environ["USERPROFILE"], "Downloads")
SCREENER_CSV = os.path.join(BASE_DIR, "screener_result.csv")
JPX_DB_PATH = os.path.join(BASE_DIR, "jpx_daily_features_db.csv")
MODEL_WEIGHTS = os.path.join(BASE_DIR, "swing_model_v6_crossattn_universe_v5.pt")
OUTPUT_CSV = os.path.join(BASE_DIR, "final_regime_screened_v5.csv")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MIN_TURNOVER = 10e8  # 売買代金10億円以上
SEQ_LEN = 10


# =============================================================================
# 1. モデル定義 (v5確定版アーキテクチャ)
# =============================================================================
class DecayPooling(nn.Module):

  def __init__(self, seq_len=10):
    super(DecayPooling, self).__init__()
    weights = np.exp(np.linspace(-1.5, 0.0, seq_len))
    weights = weights / weights.sum()
    self.register_buffer(
        "weights",
        torch.tensor(weights, dtype=torch.float32).unsqueeze(0).unsqueeze(-1),
    )

  def forward(self, x):
    return torch.sum(x * self.weights, dim=1)


class CrossAttentionBlock(nn.Module):

  def __init__(self, hidden_dim=24, num_heads=2, dropout=0.2):
    super().__init__()
    self.mha = nn.MultiheadAttention(
        embed_dim=hidden_dim,
        num_heads=num_heads,
        dropout=dropout,
        batch_first=True,
    )
    self.norm = nn.LayerNorm(hidden_dim)
    self.dropout = nn.Dropout(dropout)

  def forward(self, query, key_value):
    attn_out, _ = self.mha(query=query, key=key_value, value=key_value)
    return self.norm(query + self.dropout(attn_out))


class DualStreamGRU_v6_Slim5(nn.Module):

  def __init__(
      self, stock_dim=5, macro_dim=5, hidden_dim=24, num_heads=2, num_classes=3
  ):
    super().__init__()
    self.stock_gru = nn.GRU(
        stock_dim, hidden_dim, batch_first=True, num_layers=1
    )
    self.macro_gru = nn.GRU(
        macro_dim, hidden_dim, batch_first=True, num_layers=1
    )
    self.cross_attn = CrossAttentionBlock(
        hidden_dim, num_heads=num_heads, dropout=0.2
    )
    self.pool = DecayPooling(seq_len=10)

    self.classifier = nn.Sequential(
        nn.Linear(hidden_dim, 16),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(16, num_classes),
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
    print(
        f"[+] 最新スクリーニングCSVを自動同期しました:"
        f" {os.path.basename(latest)}"
    )


def load_macro_environment():
  if not os.path.exists(JPX_DB_PATH):
    raise FileNotFoundError(f"[!] {JPX_DB_PATH} が見つかりません。")

  jpx_db = pd.read_csv(JPX_DB_PATH, index_col=0, parse_dates=True)
  jpx_db.index = pd.to_datetime(jpx_db.index).tz_localize(None)
  start_date = (jpx_db.index.min() - datetime.timedelta(days=15)).strftime(
      "%Y-%m-%d"
  )

  n225 = yf.download("^N225", start=start_date, interval="1d", progress=False)
  if isinstance(n225.columns, pd.MultiIndex):
    n225.columns = n225.columns.get_level_values(0)
  n225.index = pd.to_datetime(n225.index).tz_localize(None)

  fx = yf.download(
      "USDJPY=X", start=start_date, interval="1d", progress=False
  )
  if isinstance(fx.columns, pd.MultiIndex):
    fx.columns = fx.columns.get_level_values(0)
  fx.index = pd.to_datetime(fx.index).tz_localize(None)

  macro_df = pd.DataFrame(index=n225.index)
  macro_df["NK_Close"] = n225["Close"]
  macro_df["NK_Ret"] = (
      n225["Close"].pct_change(fill_method=None).fillna(0.0)
  )
  macro_df["FX_Ret"] = (
      fx["Close"]
      .pct_change(fill_method=None)
      .reindex(macro_df.index)
      .fillna(0.0)
  )

  macro_df = macro_df.join(jpx_db, how="inner").ffill().bfill()

  macro_df["gamma_log_dist"] = np.log(
      (macro_df["NK_Close"] + 1e-7) / (macro_df["gamma_flip"] + 1e-7)
  ).fillna(0.0)
  denom = (macro_df["call_oi_wall"] - macro_df["put_oi_wall"]).abs() + 1e-7
  macro_df["wall_dist_ratio"] = (
      (macro_df["NK_Close"] - macro_df["put_oi_wall"]) / denom
  ).clip(0.0, 1.0)
  macro_df["short_selling_norm"] = (
      macro_df["short_selling_ratio"] - 45.0
  ) / 5.0

  return macro_df


# =============================================================================
# 3. 実行メイン処理
# =============================================================================
def main():
  print("=" * 90)
  print(
      "【v5スイングモデル 動的レジーム制御型スクリーニング (ロング &"
      " ヘッジ統合版)】"
  )
  print("=" * 90)

  sync_latest_csv()

  if not os.path.exists(MODEL_WEIGHTS):
    print(f"[!] モデル重みが見つかりません: {MODEL_WEIGHTS}")
    return

  checkpoint = torch.load(MODEL_WEIGHTS, map_location=DEVICE)
  stock_cols = checkpoint["stock_cols"]
  macro_cols = checkpoint["macro_cols"]
  hidden_dim = checkpoint.get("hidden_dim", 24)

  model = DualStreamGRU_v6_Slim5(
      stock_dim=len(stock_cols),
      macro_dim=len(macro_cols),
      hidden_dim=hidden_dim,
      num_classes=3,
  ).to(DEVICE)
  model.load_state_dict(checkpoint["model_state_dict"])
  model.eval()

  macro_df = load_macro_environment()
  latest_macro = macro_df.iloc[-1]

  gamma_dist = latest_macro["gamma_log_dist"]
  wall_ratio = latest_macro["wall_dist_ratio"]
  short_ratio = latest_macro["short_selling_ratio"]

  is_negative_gamma = gamma_dist < 0
  near_call_wall = wall_ratio > 0.85
  base_threshold = 0.385 if is_negative_gamma else 0.370

  print(f"\n[★] 現在の相場環境認識 (マクロレジーム):")
  print(
      "  - ガンマ状態:"
      f" {'【ネガティブガンマ (急変警戒・ボラ拡大)】' if is_negative_gamma else '【ポジティブガンマ (安定推移)】'}"
      f" (乖離: {gamma_dist:.3f})"
  )
  print(
      f"  - 壁の位置関係: Put壁 [{latest_macro['put_oi_wall']:.0f}] <---"
      f" ({wall_ratio*100:.1f}%) ---> Call壁"
      f" [{latest_macro['call_oi_wall']:.0f}]"
  )
  print(
      f"  - 空売り比率: {short_ratio:.1f}% | 適用基本しきい値:"
      f" {base_threshold:.3f}\n"
  )

  try:
    raw_df = pd.read_csv(SCREENER_CSV, encoding="cp932")
  except Exception:
    raw_df = pd.read_csv(SCREENER_CSV, encoding="utf-8")

  code_col = [c for c in raw_df.columns if "コード" in str(c)][0]
  tickers = [
      f"{str(c).strip()}.T"
      for c in raw_df[code_col]
      if str(c).strip().isdigit()
  ]

  print(
      f"[*] スキャン対象母集団: {len(tickers)} 銘柄"
      " (売買代金10億円以上フィルター適用)"
  )

  macro_feed = macro_df[macro_cols]
  m_start = (macro_df.index.min() - datetime.timedelta(days=40)).strftime(
      "%Y-%m-%d"
  )

  candidates = []

  for i, t in enumerate(tickers):
    try:
      df = yf.download(t, start=m_start, interval="1d", progress=False)
      if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
      df.index = pd.to_datetime(df.index).tz_localize(None)
      df = df.dropna(subset=["Close", "High", "Low", "Volume"])
      if len(df) < SEQ_LEN + 20:
        continue

      turnover_5d = (df["Close"] * df["Volume"]).rolling(5).mean().iloc[-1]
      if turnover_5d < MIN_TURNOVER:
        continue

      df["stock_ret_1d"] = (
          df["Close"].pct_change(1, fill_method=None).fillna(0.0)
      )
      df["stock_ret_5d"] = (
          df["Close"].pct_change(5, fill_method=None).fillna(0.0)
      )
      vol_5d = df["Volume"].rolling(5).mean()
      df["vol_ratio_5d"] = (df["Volume"] / (vol_5d + 1e-7)).fillna(1.0)

      hl = df["High"] - df["Low"]
      h_cp = (df["High"] - df["Close"].shift(1)).abs()
      l_cp = (df["Low"] - df["Close"].shift(1)).abs()
      tr = pd.concat([hl, h_cp, l_cp], axis=1).max(axis=1)
      atr = tr.rolling(14).mean()
      df["ATR"] = atr
      df["atr_ratio"] = (atr / (df["Close"] + 1e-7)).fillna(0.0)

      aligned_nk = macro_df["NK_Ret"].reindex(df.index).fillna(0.0)
      cov = df["stock_ret_1d"].rolling(20).cov(aligned_nk)
      var = aligned_nk.rolling(20).var()
      df["rolling_beta"] = (cov / (var + 1e-7)).fillna(1.0)

      df = df.join(macro_feed, how="inner")
      df = df.dropna(subset=["ATR", "rolling_beta"] + macro_cols)
      if len(df) < SEQ_LEN:
        continue

      w_s = df[stock_cols].values[-SEQ_LEN:]
      w_m = df[macro_cols].values[-SEQ_LEN:]
      w_s_norm = (w_s - w_s.mean(axis=0)) / (w_s.std(axis=0) + 1e-7)
      w_m_norm = (w_m - w_m.mean(axis=0)) / (w_m.std(axis=0) + 1e-7)

      t_s = torch.tensor(w_s_norm, dtype=torch.float32).unsqueeze(0).to(DEVICE)
      t_m = torch.tensor(w_m_norm, dtype=torch.float32).unsqueeze(0).to(DEVICE)

      with torch.no_grad():
        probs = (
            torch.softmax(model(t_s, t_m), dim=-1).squeeze(0).cpu().numpy()
        )

      p_stop, p_wait, p_win = float(probs[0]), float(probs[1]), float(probs[2])
      beta = float(df["rolling_beta"].iloc[-1])
      curr_close = float(df["Close"].iloc[-1])
      curr_atr = float(df["ATR"].iloc[-1])
      vol_ratio = float(df["vol_ratio_5d"].iloc[-1])

      action = "⏸️ WAIT"
      gate_reason = "見送り"

      # -------------------------------------------------------------
      # 【最上流】空売り・ヘッジ判定 (⚠️ SHORT / HEDGE)
      # 出来高の制限を撤廃し、高ベータ × 下落確率支配的を捕捉
      # -------------------------------------------------------------
      is_hedge_candidate = (
          is_negative_gamma
          and beta >= 1.20
          and p_stop >= 0.550
          and (p_stop - p_win >= 0.20)
      )

      if is_hedge_candidate:
        action = "⚠️ SHORT / HEDGE"
        gate_reason = (
            f"下落加速ヘッジ(β={beta:.2f}, 下落率={p_stop*100:.1f}%,"
            f" 乖離={-(p_stop-p_win)*100:.1f}%)"
        )

      # -------------------------------------------------------------
      # CASE A: ネガティブガンマ環境 (悪地合い特化・防衛層)
      # -------------------------------------------------------------
      elif is_negative_gamma:
        if (
            beta <= 0.45
            and (p_win / (p_stop + 1e-5) >= 0.50)
            and p_win >= 0.300
            and vol_ratio >= 0.9
        ):
          action = "🛡️ DEFENSIVE BUY"
          gate_reason = f"避難買い(低β={beta:.2f}, 期待比={p_win/p_stop:.2f})"
        elif p_win >= 0.320 and vol_ratio >= 1.2:
          action = "⚡ REVERSAL BUY"
          gate_reason = (
              f"自律反発初動(勝率{p_win*100:.1f}%, 出来高{vol_ratio:.2f}x)"
          )
        elif p_win >= 0.300 and beta < 1.0:
          action = "👀 WATCH"
          gate_reason = f"地合い好転待ち(勝率{p_win*100:.1f}%)"
        elif beta >= 1.0:
          action = "⏸️ WAIT"
          gate_reason = f"地合い連動安警戒(β={beta:.2f})"
        else:
          action = "⏸️ WAIT"
          gate_reason = "確信度不足"

      # -------------------------------------------------------------
      # CASE B: ポジティブガンマ環境 (通常スイング層)
      # -------------------------------------------------------------
      else:
        if near_call_wall and p_win < 0.40:
          action = "⏸️ WAIT"
          gate_reason = "コール壁抵抗帯"
        elif p_win >= base_threshold and p_win > p_stop and vol_ratio >= 1.0:
          action = "🎯 BUY"
          gate_reason = "レジーム適合(合格)"
        elif (
            (p_win >= base_threshold - 0.020)
            and (p_win > p_stop * 1.1)
            and (vol_ratio >= 0.8)
        ):
          action = "👀 WATCH"
          gate_reason = f"押し目監視(勝率{p_win*100:.1f}%)"
        else:
          action = "⏸️ WAIT"
          gate_reason = "確信度不足"

      # 空売り時は利確と損切の方向を反転
      if action == "⚠️ SHORT / HEDGE":
        target_price = round(curr_close - 2.0 * curr_atr, 1)
        stop_price = round(curr_close + 1.0 * curr_atr, 1)
      else:
        target_price = round(curr_close + 2.0 * curr_atr, 1)
        stop_price = round(curr_close - 1.0 * curr_atr, 1)

      candidates.append({
          "ticker": t,
          "price": curr_close,
          "target_price": target_price,
          "stop_price": stop_price,
          "prob_win": round(p_win, 4),
          "prob_stop": round(p_stop, 4),
          "beta": round(beta, 2),
          "vol_ratio": round(vol_ratio, 2),
          "turnover_oku": round(turnover_5d / 1e8, 1),
          "action": action,
          "reason": gate_reason,
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
      "⚠️ SHORT / HEDGE": 1,
      "🎯 BUY": 2,
      "🛡️ DEFENSIVE BUY": 3,
      "⚡ REVERSAL BUY": 4,
      "👀 WATCH": 5,
      "⏸️ WAIT": 6,
  }
  res_df["priority"] = res_df["action"].map(priority_map)
  res_df = res_df.sort_values(
      by=["priority", "prob_win"], ascending=[True, False]
  ).drop(columns=["priority"])

  res_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

  print("\n" + "=" * 95)
  print("【動的スクリーニング結果 (推奨・ヘッジ・監視ピックアップ)】")
  print("=" * 95)

  pickups = res_df[res_df["action"] != "⏸️ WAIT"]
  if not pickups.empty:
    print(
        pickups.head(20)[[
            "ticker",
            "price",
            "target_price",
            "stop_price",
            "prob_win",
            "prob_stop",
            "beta",
            "vol_ratio",
            "action",
            "reason",
        ]].to_string(index=False)
    )
  else:
    print(
        res_df.head(15)[[
            "ticker",
            "price",
            "target_price",
            "stop_price",
            "prob_win",
            "prob_stop",
            "beta",
            "vol_ratio",
            "action",
            "reason",
        ]].to_string(index=False)
    )

  print("=" * 95)
  action_counts = res_df["action"].value_counts().to_dict()
  print(f"[*] 判定サマリ: {action_counts}")
  print(f"[+] 全結果を保存しました: {OUTPUT_CSV}")


if __name__ == "__main__":
  main()