"""Single-model and ensemble inference."""

import numpy as np
import pandas as pd
import torch
from modules.model_inference import predict_probabilities_ensemble_batch
from .config import EV_WIN_WEIGHT, EV_STOP_WEIGHT
from .runtime import DEVICE

def predict1(model, w_s, w_m):
    """1件ずつ推論する旧実装。正しさの検証用に残してある(通常はrun_backtest_inference
    のバッチ推論を使うので、これを直接呼ぶ必要はない)。"""
    t_s = torch.tensor(w_s, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    t_m = torch.tensor(w_m, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(t_s, t_m), dim=-1).squeeze(0).cpu().numpy()
    return float(probs[2]), float(probs[0])

def run_backtest_inference(model, pool, batch_size=256):
    """poolを1件ずつではなくbatch_size件まとめてforwardする(1件ずつだとカーネル起動+
    GPU-CPU同期が多発し非常に遅い)。LayerNormのみでBatchNorm/Dropoutはeval()で無効
    なので、バッチをどう区切っても1件ずつ推論した場合と数値的に完全一致する。"""
    model.eval()
    records = []
    with torch.no_grad():
        for i in range(0, len(pool), batch_size):
            chunk = pool[i : i + batch_size]
            w_s_batch = np.stack([item[4] for item in chunk])
            w_m_batch = np.stack([item[5] for item in chunk])
            t_s = torch.tensor(w_s_batch, dtype=torch.float32).to(DEVICE)
            t_m = torch.tensor(w_m_batch, dtype=torch.float32).to(DEVICE)
            probs = torch.softmax(model(t_s, t_m), dim=-1).cpu().numpy()
            for j, (t, date, entry_date, exit_date, w_s, w_m, ret_pct) in enumerate(
                chunk
            ):
                p_win, p_stop = float(probs[j, 2]), float(probs[j, 0])
                # scoreはNearPairRankingLoss(RegimeAware)が直接最適化している量そのもの
                # (EV_WIN_WEIGHT*p_win-EV_STOP_WEIGHT*p_stop、本番のev_scoreと同じ2:1重み)。
                # p_win単体は絶対確率として校正されていない(win_rateとの平均ギャップ0.112・
                # 非単調を実測確認済み)ため、ランキング用にはscoreを使う。
                records.append(
                    {
                        "ticker": t,
                        "date": date,
                        "entry_date": entry_date,
                        "exit_date": exit_date,
                        "p_win": p_win,
                        "p_stop": p_stop,
                        "score": EV_WIN_WEIGHT * p_win - EV_STOP_WEIGHT * p_stop,
                        "ret_pct": ret_pct,
                    }
                )
    return pd.DataFrame(records)

def run_backtest_inference_ensemble(models, pool, batch_size=256):
    """run_backtest_inferenceの複数モデル版。本番のmodules/model_inference.py::
    predict_probabilities_ensemble_batchと同じロジック(各モデルのsoftmax確率を平均
    してからp_win/p_stopを取り出す)で、単一seedの偏り(特定regimeに偏ってSTRONG BUYを
    出す現象、実測確認済み)を緩和する。本番は5seedアンサンブルだが、ここはSEEDSの数だけ使う。
    scoreにはpredict_probabilities_ensemble_batchが返すev(=EV_WIN_WEIGHT*p_win-
    EV_STOP_WEIGHT*p_stopと同じ2:1重み、本番のev_scoreそのもの)をそのまま使う
    """
    for model in models:
        model.eval()
    records = []
    for i in range(0, len(pool), batch_size):
        chunk = pool[i : i + batch_size]
        w_s_batch = np.stack([item[4] for item in chunk])
        w_m_batch = np.stack([item[5] for item in chunk])
        p_win_arr, p_stop_arr, ev_arr = predict_probabilities_ensemble_batch(
            models, w_s_batch, w_m_batch
        )
        for j, (t, date, entry_date, exit_date, w_s, w_m, ret_pct) in enumerate(chunk):
            p_win, p_stop, ev = (
                float(p_win_arr[j]),
                float(p_stop_arr[j]),
                float(ev_arr[j]),
            )
            records.append(
                {
                    "ticker": t,
                    "date": date,
                    "entry_date": entry_date,
                    "exit_date": exit_date,
                    "p_win": p_win,
                    "p_stop": p_stop,
                    "score": ev,
                    "ret_pct": ret_pct,
                }
            )
    return pd.DataFrame(records)
