"""Alpha回帰アンサンブル推論(2026-09-24追加)。stage3_exp.inference.
run_backtest_inference_ensemble(3値分類のp_win/p_stop/ev)の回帰版——各seedの予測alpha
を平均し、score=-avg_alphaとして既存のevaluate_topn_curve/evaluate_daily_topn
(scoreの高い順にTopN選択)へそのまま渡せる形式(ticker/date/score/ret_pct列を持つ
DataFrame)で返す。バックテストのret_pct自体は呼び出し側がprepare_backtest_pool_short
(トレーリングストップ込みの空売り損益)で用意したpoolをそのまま使う——学習ターゲット
(alpha_5d、固定期間)と実際の約定シミュレーション(バリア付き)を意図的に分けている。"""
import numpy as np
import pandas as pd
import torch
from stage3_exp.runtime import DEVICE


def run_backtest_inference_ensemble_alpha(models, pool, batch_size=256):
    for model in models:
        model.eval()
    records = []
    with torch.no_grad():
        for i in range(0, len(pool), batch_size):
            chunk = pool[i:i + batch_size]
            w_s_batch = np.stack([item[4] for item in chunk])
            w_m_batch = np.stack([item[5] for item in chunk])
            t_s = torch.tensor(w_s_batch, dtype=torch.float32, device=DEVICE)
            t_m = torch.tensor(w_m_batch, dtype=torch.float32, device=DEVICE)
            preds = [model(t_s, t_m).squeeze(-1).cpu().numpy() for model in models]
            avg_alpha = np.mean(preds, axis=0)
            for j, (t, date, entry_date, exit_date, w_s, w_m, ret_pct) in enumerate(chunk):
                records.append({
                    "ticker": t, "date": date, "entry_date": entry_date, "exit_date": exit_date,
                    "pred_alpha": float(avg_alpha[j]), "score": float(-avg_alpha[j]), "ret_pct": ret_pct,
                })
    return pd.DataFrame(records)
