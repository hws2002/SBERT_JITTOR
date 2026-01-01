
import jittor as jt
from jittor import nn


class ComplexSoftmaxLoss(nn.Module):
    """
    SBERT-style softmax loss with a multi-layer classifier head inside the loss module.
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

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim * num_vectors, embedding_dim),
            nn.Tanh(),
            nn.Linear(embedding_dim, num_labels),
        )
        self.loss_fct = nn.CrossEntropyLoss()

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
        features = jt.concat(vectors, dim=1)
        logits = self.classifier(features)
        loss = self.loss_fct(logits, labels)
        return loss, logits
