"""
Triplet objective for sentence embeddings.
"""

import jittor as jt
from jittor import nn


class TripletLoss(nn.Module):
    """
    Triplet objective for sentence embeddings.

    Minimizes max(||a - p|| - ||a - n|| + margin, 0) using Euclidean distance.
    """
    def __init__(self, margin: float = 1.0, eps: float = 1e-9):
        super().__init__()
        self.margin = margin
        self.eps = eps

    def _pairwise_dist(self, x: jt.Var, y: jt.Var) -> jt.Var:
        return jt.sqrt(jt.sum((x - y) ** 2, dim=1) + self.eps)

    def execute(self, anchor: jt.Var, positive: jt.Var, negative: jt.Var) -> jt.Var:
        dist_ap = self._pairwise_dist(anchor, positive)
        dist_an = self._pairwise_dist(anchor, negative)
        loss = jt.maximum(dist_ap - dist_an + self.margin, 0.0)
        return jt.mean(loss)
