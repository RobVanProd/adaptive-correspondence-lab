"""Transparent experiments on correspondences between adaptive systems."""

from .bandit import ContextualBandit, NaturalPolicyGradient
from .categorical import (
    CategoricalNaturalGradient,
    MultiplicativeWeights,
    ReplicatorDynamics,
)
from .gaussian import DiagonalGaussianNaturalGradient, DiagonalGaussianState
from .schema import StepRecord, Trajectory

__all__ = [
    "CategoricalNaturalGradient",
    "ContextualBandit",
    "DiagonalGaussianNaturalGradient",
    "DiagonalGaussianState",
    "MultiplicativeWeights",
    "NaturalPolicyGradient",
    "ReplicatorDynamics",
    "StepRecord",
    "Trajectory",
]

__version__ = "0.1.0"
