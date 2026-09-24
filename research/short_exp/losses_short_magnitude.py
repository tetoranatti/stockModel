"""リターンの大きさで重み付けしたペアワイズランキング損失(2026-09-24追加、ユーザー提案)。
既存のNearPairRankingLossRegimeAwareはsign(ret_pct差)しか見ておらず、僅差のペアも
圧勝ペアも同じ重みで学習される。大差のペアほど強く学習させる狙いで、|ret_pct差|に
比例した重みを掛ける。

外れ値対策(2026-09-24、空売り特有のテールリスクを踏まえた設計判断): 単純に|diff|を
そのまま重みにすると、踏み上げ級の1ペアがそのバッチの勾配を支配しかねない
(risk_adjusted_blendのz-score自体が既に同種の外れ値感応性を持つ問題として
[[short_selling_model_2026-09-24]]で指摘済み——ここでさらに増幅させたくない)。
そのためバッチ内平均で正規化した上でmax_weight倍にクリップし、外れ値ペアの寄与を
「相場並みの大差ペアの3倍」程度に頭打ちさせる(単純な比例重み付けより穏やかなロバスト版)。
"""
import torch
import torch.nn as nn
from stage3_exp.config import MAX_PAIR_GAP, EV_WIN_WEIGHT, EV_STOP_WEIGHT


class NearPairRankingLossMagnitudeWeighted(nn.Module):
    def __init__(self, ev_win_weight=EV_WIN_WEIGHT, ev_stop_weight=EV_STOP_WEIGHT,
                 max_gap=MAX_PAIR_GAP, cross_sectional=False, max_weight=3.0):
        super().__init__()
        self.ev_win_weight = ev_win_weight
        self.ev_stop_weight = ev_stop_weight
        self.max_gap = max_gap
        self.cross_sectional = cross_sectional
        self.max_weight = max_weight

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

        abs_diff = diff_ret[mask].abs()
        # バッチ内平均を1.0とする相対重み。max_weight倍にクリップして外れ値ペアの
        # 寄与を頭打ちにする(勾配はdetachされたweightに対しては流れない——ret_pctは
        # そもそも教師ラベルでモデル出力ではないため、通常のsoftplus項と同様に
        # backward対象はscore側のみで一貫している)。
        weight = (abs_diff / abs_diff.mean().clamp_min(1e-8)).clamp(max=self.max_weight)

        per_pair_loss = torch.nn.functional.softplus(-sign[mask] * diff_score[mask])
        loss = (per_pair_loss * weight).sum() / weight.sum().clamp_min(1e-8)
        return loss, n_pairs
