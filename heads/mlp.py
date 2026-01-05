"""
MLP Projection Head
"""

from typing import Optional
import jittor as jt
from jittor import nn


class MLPHead(nn.Module):
    """
    Multi-layer perceptron (MLP) projection head.

    Projects sentence embeddings through multiple hidden layers with
    non-linear activations and optional dropout.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: Optional[int] = None,
        num_layers: int = 2,
        activation: str = 'tanh',
        dropout: float = 0.1
    ):
        """
        Args:
            input_dim: Dimension of input embeddings
            output_dim: Dimension of output embeddings
            hidden_dim: Dimension of hidden layers (default: average of input and output)
            num_layers: Number of layers including output layer (minimum 2)
            activation: Activation function ('relu', 'gelu', 'tanh')
            dropout: Dropout probability (0 to disable)
        """
        super().__init__()
        self.output_dim = output_dim

        if num_layers < 2:
            raise ValueError("num_layers must be at least 2")

        if hidden_dim is None:
            hidden_dim = (input_dim + output_dim) // 2

        layers = []

        # First layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(self._get_activation(activation))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(self._get_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

    def _get_activation(self, activation: str):
        """Get activation module by name"""
        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'tanh': nn.Tanh(),
        }
        if activation not in activations:
            raise ValueError(f"Unknown activation: {activation}. Choose from {list(activations.keys())}")
        return activations[activation]

    def execute(self, x: jt.Var) -> jt.Var:
        """
        Args:
            x: Input embeddings [batch_size, input_dim]

        Returns:
            Projected embeddings [batch_size, output_dim]
        """
        return self.mlp(x)
