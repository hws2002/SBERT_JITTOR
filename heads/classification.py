"""
Classification Head for SBERT NLI Training
"""

import jittor as jt
from jittor import nn


class ClassificationHead(nn.Module):
    """
    Classification head for sentence pair classification (e.g., NLI tasks).

    Takes two sentence embeddings u and v, concatenates them with their
    element-wise difference |u - v|, and projects to classification logits.

    Architecture: [u; v; |u - v|] -> Linear -> logits

    This matches the SBERT paper's classification objective.
    """

    def __init__(self, hidden_size: int, num_labels: int = 3):
        """
        Args:
            hidden_size: Dimension of sentence embeddings
            num_labels: Number of classification labels (default: 3 for NLI)
        """
        super().__init__()

        # Input is concatenation of [u, v, |u-v|]
        input_size = hidden_size * 3

        self.classifier = nn.Linear(input_size, num_labels)

        self.hidden_size = hidden_size
        self.num_labels = num_labels

    def execute(self, u: jt.Var, v: jt.Var) -> jt.Var:
        """
        Forward pass for sentence pair classification.

        Args:
            u: First sentence embeddings [batch_size, hidden_size]
            v: Second sentence embeddings [batch_size, hidden_size]

        Returns:
            logits: Classification logits [batch_size, num_labels]
        """
        # Compute element-wise absolute difference
        diff = jt.abs(u - v)

        # Concatenate [u, v, |u-v|]
        features = jt.concat([u, v, diff], dim=1)

        # Project to logits
        logits = self.classifier(features)

        return logits


"""
Classification Head for SBERT NLI Training
"""

import jittor as jt
from jittor import nn


class ComplexedClassificationHead(nn.Module):
    """
    Classification head for sentence pair classification (e.g., NLI tasks).

    Takes two sentence embeddings u and v, concatenates them with their
    element-wise difference |u - v|, and projects to classification logits.

    Architecture: [u; v; |u - v|] -> Linear -> Tanh -> Linear -> logits
    """

    def __init__(self, hidden_size: int, num_labels: int = 3):
        """
        Args:
            hidden_size: Dimension of sentence embeddings
            num_labels: Number of classification labels (default: 3 for NLI)
        """
        super().__init__()

        # Input is concatenation of [u, v, |u-v|]
        input_size = hidden_size * 3

        self.classifier = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, num_labels),
        )

        self.hidden_size = hidden_size
        self.num_labels = num_labels

    def execute(self, u: jt.Var, v: jt.Var) -> jt.Var:
        """
        Forward pass for sentence pair classification.

        Args:
            u: First sentence embeddings [batch_size, hidden_size]
            v: Second sentence embeddings [batch_size, hidden_size]

        Returns:
            logits: Classification logits [batch_size, num_labels]
        """
        # Compute element-wise absolute difference
        diff = jt.abs(u - v)

        # Concatenate [u, v, |u-v|]
        features = jt.concat([u, v, diff], dim=1)

        # Project to logits
        logits = self.classifier(features)

        return logits

