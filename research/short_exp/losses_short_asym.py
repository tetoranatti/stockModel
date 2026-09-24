"""EV_WIN_WEIGHT:EV_STOP_WEIGHT非対称性を空売り専用に変えて検証するための損失
(2026-09-24追加、ユーザー提案)。stage3_exp.losses.NearPairRankingLossRegimeAwareの
コピーで、重みだけをコンストラクタ引数化した版——共有config.py(ロング本番にも使われる
EV_WIN_WEIGHT/EV_STOP_WEIGHT)は変更しない([[feedback_keep_rejected_experiment_code]]の
延長で、研究側の変更をロング側に波及させない方針を踏襲)。

動機: 空売りの損失側テール(踏み上げ)はロングの損失側(単純な下落、-100%が下限)より
理論上厄介(上昇は無制限)なため、p_stopをp_win以上に重く罰する(1:1や1:2)方が
空売りには適切かもしれない、という仮説の検証用。既定の2:1(EV_WIN_WEIGHT=2.0,
EV_STOP_WEIGHT=1.0)はstage3_exp.losses.NearPairRankingLossRegimeAwareと数値的に一致する。
"""
import torch
import torch.nn as nn
from stage3_exp.config import MAX_PAIR_GAP


class NearPairRankingLossShortAsym(nn.Module):
    def __init__(self, ev_win_weight, ev_stop_weight, max_gap=MAX_PAIR_GAP, cross_sectional=False):
        super().__init__()
        self.ev_win_weight = ev_win_weight
        self.ev_stop_weight = ev_stop_weight
        self.max_gap = max_gap
        self.cross_sectional = cross_sectional

    def forward(self, logits, ret_pct, tidx, regime=None):
        probs = torch.softmax(logits, dim=-1)
        score = self.ev_win_weight * probs[:, 2] - self.ev_stop_weight * probs[:, 0]
        diff_score = score.unsqueeze(1) - score.unsqueeze(0)
        diff_ret = ret_pct.unsqueeze(1) - ret_pct.unsqueeze(0)
        sign = torch.sign(diff_ret)
        gap = (tidx.unsqueeze(1) - tidx.unsqueeze(0)).abs()
        if self.cross_sectional:
            mask = (sign != 0) & (gap == 0)
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
