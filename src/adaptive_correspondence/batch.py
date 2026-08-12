"""Bounded vectorized categorical transitions, kept separate from the reference path."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatMatrix = NDArray[np.float64]


def _validate_batch(states: ArrayLike, rewards: ArrayLike) -> tuple[FloatMatrix, FloatMatrix]:
    probability = np.asarray(states, dtype=np.float64)
    if probability.ndim != 2 or probability.shape[1] < 2:
        raise ValueError("states must have shape (batch, dimension>=2)")
    if not np.all(np.isfinite(probability)) or np.any(probability < 0.0):
        raise ValueError("states must contain finite non-negative probabilities")
    if not np.allclose(np.sum(probability, axis=1), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("every state row must sum to one")

    reward = np.asarray(rewards, dtype=np.float64)
    if reward.ndim == 1:
        if reward.shape[0] != probability.shape[1]:
            raise ValueError("reward dimension does not match states")
        reward = np.broadcast_to(reward, probability.shape)
    if reward.shape != probability.shape or not np.all(np.isfinite(reward)):
        raise ValueError("rewards must be finite and match states or their final dimension")
    return probability.copy(), np.asarray(reward, dtype=np.float64)


def exact_step(states: ArrayLike, rewards: ArrayLike, eta: float) -> FloatMatrix:
    probability, reward = _validate_batch(states, rewards)
    if not np.isfinite(eta) or eta < 0.0:
        raise ValueError("eta must be finite and non-negative")
    factors = eta * (reward - np.max(reward, axis=1, keepdims=True))
    weighted = probability * np.exp(factors)
    return weighted / np.sum(weighted, axis=1, keepdims=True)


def natural_gradient_step(states: ArrayLike, rewards: ArrayLike, eta: float) -> FloatMatrix:
    probability, reward = _validate_batch(states, rewards)
    if np.any(probability <= 0.0):
        raise ValueError("natural-gradient batch states must be in the simplex interior")
    if not np.isfinite(eta) or eta < 0.0:
        raise ValueError("eta must be finite and non-negative")
    logits = np.log(probability)
    logits -= np.mean(logits, axis=1, keepdims=True)
    direction = reward - np.mean(reward, axis=1, keepdims=True)
    updated = logits + eta * direction
    updated -= np.max(updated, axis=1, keepdims=True)
    weights = np.exp(updated)
    return weights / np.sum(weights, axis=1, keepdims=True)


def euler_step(states: ArrayLike, rewards: ArrayLike, eta: float) -> FloatMatrix:
    probability, reward = _validate_batch(states, rewards)
    if not np.isfinite(eta) or eta < 0.0:
        raise ValueError("eta must be finite and non-negative")
    mean_reward = np.sum(probability * reward, axis=1, keepdims=True)
    candidate = probability + eta * probability * (reward - mean_reward)
    if np.any(candidate < 0.0):
        raise ValueError("explicit Euler batch step left the simplex")
    if not np.allclose(np.sum(candidate, axis=1), 1.0, rtol=0.0, atol=5e-13):
        raise FloatingPointError("explicit Euler batch step lost simplex mass")
    return candidate


def run_terminal_states(
    initial_states: ArrayLike,
    reward_schedule: ArrayLike,
    eta: float,
    *,
    steps: int,
    method: Literal["exact", "natural-gradient", "euler"] = "exact",
    chunk_size: int = 4096,
) -> FloatMatrix:
    """Run terminal states in bounded chunks without retaining trajectories."""
    states = np.asarray(initial_states, dtype=np.float64)
    if states.ndim != 2:
        raise ValueError("initial_states must be a matrix")
    if isinstance(steps, bool) or steps < 0:
        raise ValueError("steps must be a non-negative integer")
    if isinstance(chunk_size, bool) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    schedule = np.asarray(reward_schedule, dtype=np.float64)
    if schedule.ndim == 1:
        schedule = np.broadcast_to(schedule, (steps, schedule.size))
    if schedule.shape != (steps, states.shape[1]):
        raise ValueError("reward_schedule must have shape (steps, dimension)")
    methods: dict[str, Callable[[ArrayLike, ArrayLike, float], FloatMatrix]] = {
        "exact": exact_step,
        "natural-gradient": natural_gradient_step,
        "euler": euler_step,
    }
    transition = methods[method]
    output = np.empty_like(states)
    for start in range(0, states.shape[0], chunk_size):
        stop = min(start + chunk_size, states.shape[0])
        chunk = states[start:stop].copy()
        for reward in schedule:
            chunk = transition(chunk, reward, eta)
        output[start:stop] = chunk
    return output
