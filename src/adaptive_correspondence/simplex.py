"""Numerically explicit operations on finite probability simplices."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def as_float_vector(value: ArrayLike, *, name: str = "vector") -> FloatArray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.ndim != 1 or vector.size < 2:
        raise ValueError(f"{name} must be a one-dimensional vector with at least two entries")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector.copy()


def validate_simplex(
    state: ArrayLike,
    *,
    name: str = "state",
    strictly_positive: bool = False,
    atol: float = 1e-12,
) -> FloatArray:
    probability = as_float_vector(state, name=name)
    if np.any(probability < 0.0):
        raise ValueError(f"{name} has a negative probability")
    if strictly_positive and np.any(probability <= 0.0):
        raise ValueError(f"{name} must be in the simplex interior")
    if not np.isclose(float(np.sum(probability)), 1.0, rtol=0.0, atol=atol):
        raise ValueError(f"{name} must sum to one within absolute tolerance {atol}")
    return probability


def validate_reward(reward: ArrayLike, dimension: int) -> FloatArray:
    vector = as_float_vector(reward, name="reward")
    if vector.size != dimension:
        raise ValueError(
            f"reward dimension {vector.size} does not match state dimension {dimension}"
        )
    return vector


def softmax(logits: ArrayLike) -> FloatArray:
    values = as_float_vector(logits, name="logits")
    shifted = values - float(np.max(values))
    weights = np.exp(shifted)
    return weights / float(np.sum(weights))


def centered_logits(probability: ArrayLike) -> FloatArray:
    state = validate_simplex(probability, strictly_positive=True)
    logits = np.log(state)
    return logits - float(np.mean(logits))


def exponential_update(probability: ArrayLike, reward: ArrayLike, eta: float) -> FloatArray:
    state = validate_simplex(probability)
    rewards = validate_reward(reward, state.size)
    if not np.isfinite(eta) or eta < 0.0:
        raise ValueError("eta must be finite and non-negative")
    centered_reward = rewards - float(np.max(rewards))
    log_factor = eta * centered_reward
    unnormalized = state * np.exp(log_factor)
    total = float(np.sum(unnormalized))
    if not np.isfinite(total) or total <= 0.0:
        raise FloatingPointError("exponential update has no finite positive mass")
    return unnormalized / total


def replicator_field(probability: ArrayLike, reward: ArrayLike) -> FloatArray:
    state = validate_simplex(probability)
    rewards = validate_reward(reward, state.size)
    return state * (rewards - float(np.dot(state, rewards)))


def categorical_logit_gradient(probability: ArrayLike, reward: ArrayLike) -> FloatArray:
    return replicator_field(probability, reward)


def categorical_fisher(probability: ArrayLike) -> FloatArray:
    state = validate_simplex(probability)
    return np.diag(state) - np.outer(state, state)


def negative_entropy_hessian(probability: ArrayLike) -> FloatArray:
    state = validate_simplex(probability, strictly_positive=True)
    return np.diag(1.0 / state)


def entropy(probability: ArrayLike) -> float:
    state = validate_simplex(probability)
    positive = state > 0.0
    return float(-np.sum(state[positive] * np.log(state[positive])))


def kl_divergence(left: ArrayLike, right: ArrayLike) -> float:
    p = validate_simplex(left, name="left")
    q = validate_simplex(right, name="right")
    if p.size != q.size:
        raise ValueError("KL arguments must have the same dimension")
    if np.any((p > 0.0) & (q <= 0.0)):
        raise ValueError("KL is infinite because right has zero mass on left support")
    support = p > 0.0
    return float(np.sum(p[support] * (np.log(p[support]) - np.log(q[support]))))


def l1_distance(left: ArrayLike, right: ArrayLike) -> float:
    p = validate_simplex(left, name="left")
    q = validate_simplex(right, name="right")
    if p.size != q.size:
        raise ValueError("distance arguments must have the same dimension")
    return float(np.sum(np.abs(p - q)))


def project_simplex_with_floor(value: ArrayLike, floor: float) -> FloatArray:
    """Euclidean projection onto {p: sum p=1, p_i>=floor}."""
    vector = as_float_vector(value, name="projection value")
    dimension = vector.size
    if not np.isfinite(floor) or floor < 0.0 or floor >= 1.0 / dimension:
        raise ValueError("floor must be finite and in [0, 1/dimension)")
    mass = 1.0 - dimension * floor
    shifted = vector - floor
    ordered = np.sort(shifted)[::-1]
    cumulative = np.cumsum(ordered)
    candidates = ordered - (cumulative - mass) / np.arange(1, dimension + 1) > 0.0
    if not np.any(candidates):
        raise FloatingPointError("simplex projection found no active coordinate")
    rho = int(np.nonzero(candidates)[0][-1])
    theta = float((cumulative[rho] - mass) / (rho + 1))
    projected = np.maximum(shifted - theta, 0.0) + floor
    return validate_simplex(projected, name="projected state", atol=5e-14)


def simplex_diagnostics(probability: ArrayLike, *, floor: float = 0.0) -> dict[str, float | bool]:
    state = np.asarray(probability, dtype=np.float64)
    finite = bool(np.all(np.isfinite(state)))
    return {
        "finite": finite,
        "sum_error": float(abs(np.sum(state) - 1.0)) if finite else float("inf"),
        "min_probability": float(np.min(state)) if finite else float("-inf"),
        "floor_shortfall": float(max(0.0, floor - np.min(state))) if finite else float("inf"),
    }
