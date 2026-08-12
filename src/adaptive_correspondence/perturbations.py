"""Named, validated violations of the categorical correspondence assumptions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .categorical import CategoricalEffects
from .simplex import validate_reward, validate_simplex

FloatArray = NDArray[np.float64]
PerturbationKind = Literal[
    "none",
    "euler",
    "reward-bias",
    "noise",
    "delay",
    "frequency",
    "mutation",
    "nonstationary",
    "finite-population",
    "constraint",
]


def canonical_direction(dimension: int) -> FloatArray:
    if dimension < 2:
        raise ValueError("dimension must be at least two")
    direction = np.linspace(-1.0, 1.0, dimension, dtype=np.float64)
    direction -= float(np.mean(direction))
    scale = float(np.max(np.abs(direction)))
    return direction / scale


def cyclic_payoff(dimension: int) -> FloatArray:
    """A deterministic antisymmetric cyclic interaction matrix."""
    if dimension < 2:
        raise ValueError("dimension must be at least two")
    if dimension == 2:
        return np.array([[0.0, -1.0], [1.0, 0.0]], dtype=np.float64)
    payoff = np.zeros((dimension, dimension), dtype=np.float64)
    for index in range(dimension):
        payoff[index, (index - 1) % dimension] = 1.0
        payoff[index, (index + 1) % dimension] = -1.0
    return payoff


def off_diagonal_mutation(dimension: int) -> FloatArray:
    if dimension < 2:
        raise ValueError("dimension must be at least two")
    matrix = np.full((dimension, dimension), 1.0 / (dimension - 1), dtype=np.float64)
    np.fill_diagonal(matrix, 0.0)
    return matrix


@dataclass(frozen=True)
class Perturbation:
    kind: PerturbationKind = "none"
    epsilon: float = 0.0

    def validate(self, dimension: int) -> None:
        allowed = {
            "none",
            "euler",
            "reward-bias",
            "noise",
            "delay",
            "frequency",
            "mutation",
            "nonstationary",
            "finite-population",
            "constraint",
        }
        if self.kind not in allowed:
            raise ValueError(f"unknown perturbation kind: {self.kind}")
        if not np.isfinite(self.epsilon) or self.epsilon < 0.0:
            raise ValueError("epsilon must be finite and non-negative")
        if self.kind in {"delay", "mutation"} and self.epsilon > 1.0:
            raise ValueError(f"{self.kind} epsilon must be in [0,1]")
        if self.kind == "constraint" and self.epsilon >= 1.0 / dimension:
            raise ValueError("constraint epsilon must be below 1/dimension")
        if self.kind == "finite-population" and self.epsilon > 1.0:
            raise ValueError("finite-population epsilon must be in [0,1]")

    def effects(self, dimension: int) -> CategoricalEffects:
        self.validate(dimension)
        if self.kind == "mutation" and self.epsilon > 0.0:
            return CategoricalEffects(
                mutation_rate=self.epsilon,
                mutation_matrix=off_diagonal_mutation(dimension),
            )
        if self.kind == "constraint" and self.epsilon > 0.0:
            return CategoricalEffects(probability_floor=self.epsilon)
        if self.kind == "finite-population" and self.epsilon > 0.0:
            population = int(np.ceil(self.epsilon**-2))
            return CategoricalEffects(finite_population=population)
        return CategoricalEffects()

    def apply_reward(
        self,
        current_reward: ArrayLike,
        previous_reward: ArrayLike,
        state: ArrayLike,
        *,
        step: int,
        rng: np.random.Generator,
    ) -> FloatArray:
        probability = validate_simplex(state)
        current = validate_reward(current_reward, probability.size)
        previous = validate_reward(previous_reward, probability.size)
        self.validate(probability.size)
        direction = canonical_direction(probability.size)

        if self.kind == "reward-bias":
            return current + self.epsilon * direction
        if self.kind == "noise":
            return current + rng.normal(0.0, self.epsilon, size=probability.size)
        if self.kind == "delay":
            return (1.0 - self.epsilon) * current + self.epsilon * previous
        if self.kind == "frequency":
            return current + self.epsilon * (cyclic_payoff(probability.size) @ probability)
        if self.kind == "nonstationary":
            signal = np.sin(0.73 * (step + 1))
            return current + self.epsilon * signal * direction
        return current
