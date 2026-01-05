"""
SBERT-style softmax loss with sentence-pair classifier head.
"""

import math
import jittor as jt
from jittor import nn


class SoftmaxLoss(nn.Module):
    """
    Implements the SBERT NLI objective using ablation features.

    ablation=0: [u; v; |u - v|] (default)
    ablation=1: [u; v]
    ablation=2: [|u - v|]
    ablation=3: [u * v]
    ablation=4: [|u - v|; u * v]
    ablation=5: [u; v; u * v]
    ablation=6: [u; v; |u - v|; u * v]
    """

    def __init__(
        self,
        model,
        num_labels: int,
        concatenation_sent_difference: bool = True,
        ablation: int = 0,
    ):
        super().__init__()
        self.model = model
        self.num_labels = num_labels
        self.ablation = ablation

        embedding_dim = model.output_dim
        if ablation == 1:
            num_vectors = 2
        elif ablation == 2:
            num_vectors = 1
        elif ablation == 3:
            num_vectors = 1
        elif ablation == 4:
            num_vectors = 2
        elif ablation == 5:
            num_vectors = 3
        elif ablation == 6:
            num_vectors = 4
        else:
            num_vectors = 3 if concatenation_sent_difference else 2

        self.classifier = nn.Linear(embedding_dim * num_vectors, num_labels)
        self._init_classifier(self.classifier)
        self.loss_fct = nn.CrossEntropyLoss()

    @staticmethod
    def _init_classifier(layer: nn.Linear) -> None:
        # Match PyTorch nn.Linear default init (kaiming_uniform with a=sqrt(5)).
        fan_in = layer.weight.shape[1]
        bound = 1.0 / math.sqrt(fan_in)
        weight = jt.rand(layer.weight.shape) * (2 * bound) - bound
        layer.weight.assign(weight)
        if layer.bias is not None:
            bias = jt.rand(layer.bias.shape) * (2 * bound) - bound
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

        if self.ablation == 1:
            vectors = [rep_a, rep_b]
        elif self.ablation == 2:
            vectors = [jt.abs(rep_a - rep_b)]
        elif self.ablation == 3:
            vectors = [rep_a * rep_b]
        elif self.ablation == 4:
            vectors = [jt.abs(rep_a - rep_b), rep_a * rep_b]
        elif self.ablation == 5:
            vectors = [rep_a, rep_b, rep_a * rep_b]
        elif self.ablation == 6:
            vectors = [rep_a, rep_b, jt.abs(rep_a - rep_b), rep_a * rep_b]
        else:
            vectors = [rep_a, rep_b, jt.abs(rep_a - rep_b)]
        fixed_vectors = []
        for vec in vectors:
            if vec.ndim == 1:
                vec = vec.unsqueeze(0)
            fixed_vectors.append(vec)
        features = jt.concat(fixed_vectors, dim=1)
        logits = self.classifier(features)
        loss = self.loss_fct(logits, labels)
        return loss, logits
