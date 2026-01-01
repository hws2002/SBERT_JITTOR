
import jittor as jt
from jittor import nn


class ComplexSoftmaxLoss(nn.Module):
    """
    SBERT-style softmax loss with a multi-layer classifier head inside the loss module.
    """

    def __init__(self, model, num_labels: int, concatenation_sent_difference: bool = True):
        super().__init__()
        self.model = model
        self.num_labels = num_labels

        embedding_dim = model.output_dim
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim * 3, embedding_dim),
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

        diff = jt.abs(rep_a - rep_b)
        features = jt.concat([rep_a, rep_b, diff], dim=1)
        logits = self.classifier(features)
        loss = self.loss_fct(logits, labels)
        return loss, logits