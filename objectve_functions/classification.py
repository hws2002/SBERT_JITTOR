import jittor as jt
from jittor import nn


class SBERTClassificationHead(nn.Module):
    """
    SBERT Classification objective head.

    Takes two sentence embeddings u, v and returns logits for num_labels classes
    using the standard [u; v; |u - v|] projection.
    """

    def __init__(self, hidden_size: int, num_labels: int = 3):
        super().__init__()

        input_size = hidden_size * 3
        self.classifier = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, num_labels),
        )

    def execute(self, u: jt.Var, v: jt.Var) -> jt.Var:
        """
        Args:
            u: sentence embedding 1 (batch_size, hidden_size)
            v: sentence embedding 2 (batch_size, hidden_size)

        Returns:
            logits: (batch_size, num_labels)
        """
        diff = jt.abs(u - v)
        features = jt.concat([u, v, diff], dim=1)
        return self.classifier(features)
