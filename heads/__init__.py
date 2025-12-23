"""
Projection heads for SBERT models.

Available heads:
- IdentityHead: No projection (pass-through)
- LinearHead: Linear projection
- MLPHead: Multi-layer perceptron projection
- ClassificationHead: Two-sentence classification (for NLI training)
- RegressionLoss: Cosine similarity + MSE loss
- TripletLoss: Euclidean triplet loss
"""

from .identity import IdentityHead
from .linear import LinearHead
from .mlp import MLPHead
from .classification import ClassificationHead
from .losses import RegressionLoss, TripletLoss

__all__ = [
    'IdentityHead',
    'LinearHead',
    'MLPHead',
    'ClassificationHead',
    'RegressionLoss',
    'TripletLoss',
]
