"""
Projection heads for SBERT models.

Available heads:
- IdentityHead: No projection (pass-through)
- LinearHead: Linear projection
- MLPHead: Multi-layer perceptron projection
- ClassificationHead: Two-sentence classification (for NLI training)
"""

from .identity import IdentityHead
from .linear import LinearHead
from .mlp import MLPHead
from .classification import ClassificationHead

__all__ = [
    'IdentityHead',
    'LinearHead',
    'MLPHead',
    'ClassificationHead',
]
