"""Weighted Huber Loss(2026-09-24追加、ユーザー指定)。"""
import torch
import torch.nn as nn


class WeightedHuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super().__init__()
        self.huber = nn.HuberLoss(delta=delta, reduction="none")

    def forward(self, pred, target, weight):
        per_sample = self.huber(pred, target)
        return (per_sample * weight).sum() / weight.sum().clamp_min(1e-8)
