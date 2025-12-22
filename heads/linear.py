"""
Linear Projection Head
"""

import jittor as jt
from jittor import nn


class LinearHead(nn.Module):
    """
    Linear projection head.

    Projects sentence embeddings to a different dimension using a single linear layer.
    Useful for dimensionality reduction or alignment with specific tasks.
    """

    def __init__(self, input_dim: int, output_dim: int):
        """
        Args:
            input_dim: Dimension of input embeddings
            output_dim: Dimension of output embeddings
        """
        super().__init__()
        self.output_dim = output_dim
        self.linear = nn.Linear(input_dim, output_dim)

    def execute(self, x: jt.Var) -> jt.Var:
        """
        Args:
            x: Input embeddings [batch_size, input_dim]

        Returns:
            Projected embeddings [batch_size, output_dim]
        """
        return self.linear(x)
