"""
Identity Head - No projection (pass-through)
"""

import jittor as jt
from jittor import nn


class IdentityHead(nn.Module):
    """
    Identity projection head - passes embeddings through without modification.

    This is used when you want sentence embeddings directly from the encoder
    without any additional projection.
    """

    def __init__(self, input_dim: int):
        """
        Args:
            input_dim: Dimension of input embeddings (also the output dimension)
        """
        super().__init__()
        self.output_dim = input_dim

    def execute(self, x: jt.Var) -> jt.Var:
        """
        Args:
            x: Input embeddings [batch_size, input_dim]

        Returns:
            Same as input [batch_size, input_dim]
        """
        return x
