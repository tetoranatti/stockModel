"""Pairwise ranking losses."""

import torch
import torch.nn as nn
from .config import EV_WIN_WEIGHT, EV_STOP_WEIGHT, MAX_PAIR_GAP

class NearPairRankingLossFast(nn.Module):
    """NearPairRankingLossRegimeAware(regime=None)のO(B×max_gap)版(scoreの重みも含め
    数値的に厳密に同値。本番training/train_model_v8_exp.py::NearPairRankingLossとは
    score定義がEV_WIN_WEIGHT/EV_STOP_WEIGHT分だけ異なる点に注意——このファイル内では
    未使用で、正しさ検証用に残してあるだけ)。
    SameTickerBatchSampler+build_datasetにより各バッチ内は同一銘柄・tidx昇順で並ぶため、
    オフセットδ=1..max_gapだけ見ればgap<=max_gapの全ペアを漏れなく拾える。全ペア行列を
    作らずtorch.catで済ませ、O(B²)のメモリ確保・2Dマスクインデックスを回避する。
    tidxが昇順という前提が崩れる呼び出し方だと結果が変わる点に注意。"""

    def __init__(self, max_gap=MAX_PAIR_GAP):
        super().__init__()
        self.max_gap = max_gap

    def forward(self, logits, ret_pct, tidx):
        probs = torch.softmax(logits, dim=-1)
        score = EV_WIN_WEIGHT * probs[:, 2] - EV_STOP_WEIGHT * probs[:, 0]
        B = score.shape[0]
        parts = []
        for delta in range(1, min(self.max_gap, B - 1) + 1):
            s_diff = score[delta:] - score[:-delta]
            r_diff = ret_pct[delta:] - ret_pct[:-delta]
            t_diff = (tidx[delta:] - tidx[:-delta]).abs()
            sign = torch.sign(r_diff)
            mask = (sign != 0) & (t_diff <= self.max_gap)
            if mask.any():
                parts.append(torch.nn.functional.softplus(-sign[mask] * s_diff[mask]))
        if not parts:
            return score.sum() * 0.0
        return torch.cat(parts).mean()

class NearPairRankingLossRegimeAware(nn.Module):
    """NearPairRankingLossに「同一regime(LOW/MID/HIGH)のペアだけ比較する」制約を追加した版。
    同一銘柄・近傍日(|tidx_i-tidx_j|<=max_gap)ペアの42.7%がregimeを跨いでいた実測を受けて、
    regime由来のリターン格差が銘柄固有シグナルを汚染している可能性の検証用。
    regime引数はint(-1=対象外/不明で常に除外、Noneなら制約無し=構造的には本番training/
    train_model_v8_exp.py::NearPairRankingLossと同一)。

    scoreはEV_WIN_WEIGHT*p_win-EV_STOP_WEIGHT*p_stop(本番のev_score、pipeline/
    run_dynamic_regime_screening_v8.pyの候補順位・modules/model_inference.pyのev_raw
    と同じ2:1重み)。本番training/train_model_v8_exp.py::NearPairRankingLossは現状
    p_win-p_stopの1:1のままなので、この点はもう本番の学習コードと数値的に同一ではない
    (2026-09-21、ユーザー指摘「学習時のランキング軸と実運用時のランキング軸が不一致」を
    受けて、この実験ハーネス側を実運用の日次Top-N選定軸に合わせる方向に変更。本番の
    学習コード自体を変えるかどうかは、この実験で効果を確認してから別途判断する)。

    cross_sectional=Trueにすると、ペア制約が「同一銘柄・近傍日(0<gap<=max_gap)」から
    「同一日・異なるサンプル(gap==0)」に切り替わる(SameDateBatchSamplerと組み合わせて
    使う想定。PAIR_MODE="cross_sectional"参照)。実運用が実際に行っている「同日の複数
    銘柄を横断してランキングし日次Top-N件を選ぶ」タスクを学習損失として直接模したもの
    (2026-09-21追加、ユーザー指摘「学習時のペア軸(銘柄内・時系列)が実運用の同日横断
    ランキングと構造的に不一致」を受けて実装)。

    forward()は(loss_mean, n_pairs)を返す。バッチごとに有効ペア数が大きく変わりうるため、
    呼び出し側でn_pairsを使いエポック全体の平均をペア数加重にし、n_pairs=0のバッチ
    (有効ペア無し)を平均から自然に除外できるようにする。"""

    def __init__(self, max_gap=MAX_PAIR_GAP, cross_sectional=False):
        super().__init__()
        self.max_gap = max_gap
        self.cross_sectional = cross_sectional

    def forward(self, logits, ret_pct, tidx, regime=None):
        probs = torch.softmax(logits, dim=-1)
        score = EV_WIN_WEIGHT * probs[:, 2] - EV_STOP_WEIGHT * probs[:, 0]
        diff_score = score.unsqueeze(1) - score.unsqueeze(0)
        diff_ret = ret_pct.unsqueeze(1) - ret_pct.unsqueeze(0)
        sign = torch.sign(diff_ret)
        gap = (tidx.unsqueeze(1) - tidx.unsqueeze(0)).abs()
        if self.cross_sectional:
            mask = (sign != 0) & (gap == 0)  # 同一日(=異なる銘柄)のペアだけ比較
        else:
            mask = (sign != 0) & (gap > 0) & (gap <= self.max_gap)
        if regime is not None:
            same_regime = (regime.unsqueeze(1) == regime.unsqueeze(0)) & (
                regime.unsqueeze(1) >= 0
            )
            mask = mask & same_regime
        n_pairs = int(mask.sum().item())
        if n_pairs == 0:
            return diff_score.sum() * 0.0, 0
        return (
            torch.nn.functional.softplus(-sign[mask] * diff_score[mask]).mean(),
            n_pairs,
        )
