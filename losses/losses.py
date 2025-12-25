"""
Loss functions for SBERT-style objectives.
"""

import math
import jittor as jt
from jittor import nn


class RegressionLoss(nn.Module):
    """
    Regression objective for STS tasks.

    Computes cosine similarity between u and v, then applies MSE loss
    against the provided target scores.
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


class SoftmaxLoss(nn.Module):
    """
    SBERT-style softmax loss with a classifier inside the loss module.

    Uses [u; v; |u - v|] and cross-entropy loss for NLI training.
    """

    def __init__(self, model, num_labels: int, concatenation_sent_difference: bool = True):
        super().__init__()
        self.model = model
        self.num_labels = num_labels

        embedding_dim = model.output_dim
        num_vectors = 3 if concatenation_sent_difference else 2

        self.classifier = nn.Linear(embedding_dim * num_vectors, num_labels)
        self._init_classifier(self.classifier)
        self.loss_fct = nn.CrossEntropyLoss()

    @staticmethod
    def _init_classifier(layer: nn.Linear) -> None:
        # Match PyTorch nn.Linear default init (kaiming_uniform with a=sqrt(5)).
        fan_in = layer.weight.shape[1]
        bound = 1.0 / math.sqrt(fan_in)
        weight = jt.uniform(layer.weight.shape, low=-bound, high=bound)
        layer.weight.assign(weight)
        if layer.bias is not None:
            bias = jt.uniform(layer.bias.shape, low=-bound, high=bound)
            layer.bias.assign(bias)

    def execute(self, batch, labels):
        rep_a = self.model.encode(
            input_ids=batch["input_ids_a"],
            attention_mask=batch["attention_mask_a"],
            token_type_ids=batch.get("token_type_ids_a", None),
        )
        rep_b = self.model.encode(
            input_ids=batch["input_ids_b"],
            attention_mask=batch["attention_mask_b"],
            token_type_ids=batch.get("token_type_ids_b", None),
        )

        vectors = [rep_a, rep_b, jt.abs(rep_a - rep_b)]
        features = jt.concat(vectors, dim=1)
        logits = self.classifier(features)
        loss = self.loss_fct(logits, labels)
        return loss, logits
