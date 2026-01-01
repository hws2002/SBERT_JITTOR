"""
Regression objective for SBERT STS-style training.
"""

import jittor as jt
from jittor import nn


class RegressionLoss(nn.Module):
    """
    Computes cosine similarity between sentence representations and
    applies MSE loss against target similarity scores.
    """

    def __init__(self, eps: float = 1e-9):
        super().__init__()
        self.eps = eps

    def execute(self, u: jt.Var, v: jt.Var, targets: jt.Var) -> jt.Var:
        dot = jt.sum(u * v, dim=1)
        u_norm = jt.sqrt(jt.sum(u * u, dim=1) + self.eps)
        v_norm = jt.sqrt(jt.sum(v * v, dim=1) + self.eps)
        cos_sim = dot / (u_norm * v_norm + self.eps)
        return jt.mean((cos_sim - targets) ** 2)