"""Burg log-barrier mirror steps and zero-fit mutation sensitivity."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .simplex import validate_reward, validate_simplex

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class BurgSecondOrderTrace:
    states: FloatArray
    first: FloatArray
    second: FloatArray


def _validate_eta(eta: float) -> float:
    if not np.isfinite(eta) or eta < 0.0:
        raise ValueError("eta must be finite and non-negative")
    return float(eta)


def _validate_steps(steps: int) -> int:
    if isinstance(steps, bool) or not isinstance(steps, (int, np.integer)) or steps < 0:
        raise ValueError("steps must be a non-negative integer")
    return int(steps)


def _validate_mutation(mutation: ArrayLike, dimension: int) -> FloatArray:
    matrix = np.asarray(mutation, dtype=np.float64)
    if matrix.shape != (dimension, dimension):
        raise ValueError("mutation must be square with the state dimension")
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0.0):
        raise ValueError("mutation must be finite and non-negative")
    if not np.allclose(np.sum(matrix, axis=1), 1.0, atol=2e-14, rtol=0.0):
        raise ValueError("mutation must be row-stochastic")
    return matrix.copy()


def _dual_shift(state: FloatArray, reward: FloatArray, eta: float) -> float:
    poles = eta * reward - 1.0 / state
    boundary = float(np.max(poles))
    offset = max(1e-14, 1e-14 * max(1.0, abs(boundary)))
    lower = boundary + offset

    def residual(value: float) -> float:
        return float(np.sum(1.0 / (1.0 / state - eta * reward + value)) - 1.0)

    if residual(lower) <= 0.0:
        raise FloatingPointError("failed to bracket the Burg dual normalizer")
    width = 1.0
    upper = lower + width
    while residual(upper) > 0.0:
        width *= 2.0
        upper = lower + width
        if not np.isfinite(upper):
            raise FloatingPointError("Burg dual normalizer bracket diverged")
    for _ in range(160):
        midpoint = lower + 0.5 * (upper - lower)
        if midpoint in (lower, upper):
            break
        if residual(midpoint) > 0.0:
            lower = midpoint
        else:
            upper = midpoint
    return lower + 0.5 * (upper - lower)


def burg_mirror_step(probability: ArrayLike, reward: ArrayLike, *, eta: float) -> FloatArray:
    """Solve one constrained Burg-mirror reward-ascent step on the simplex."""
    state = validate_simplex(probability, strictly_positive=True)
    rewards = validate_reward(reward, state.size)
    eta_value = _validate_eta(eta)
    shift = _dual_shift(state, rewards, eta_value)
    updated = 1.0 / (1.0 / state - eta_value * rewards + shift)
    total = float(np.sum(updated))
    if not np.isfinite(total) or total <= 0.0 or np.any(updated <= 0.0):
        raise FloatingPointError("Burg mirror step left the simplex interior")
    updated /= total
    return validate_simplex(updated, strictly_positive=True, atol=2e-14)


def burg_mirror_step_polynomial_oracle(
    probability: ArrayLike, reward: ArrayLike, *, eta: float
) -> FloatArray:
    """Independent normalizer oracle from the constraint polynomial roots."""
    state = validate_simplex(probability, strictly_positive=True)
    rewards = validate_reward(reward, state.size)
    eta_value = _validate_eta(eta)
    constants = 1.0 / state - eta_value * rewards
    factors = [np.poly1d([1.0, value]) for value in constants]
    product = np.poly1d([1.0])
    for factor in factors:
        product *= factor
    reciprocal_sum = np.poly1d([0.0])
    for omitted in range(state.size):
        term = np.poly1d([1.0])
        for index, factor in enumerate(factors):
            if index != omitted:
                term *= factor
        reciprocal_sum += term
    roots = np.roots(product - reciprocal_sum)
    boundary = float(np.max(-constants))
    feasible = [
        float(root.real)
        for root in roots
        if abs(float(root.imag)) <= 2e-10 and float(root.real) > boundary
    ]
    if len(feasible) != 1:
        raise FloatingPointError("Burg polynomial oracle lacks a unique feasible root")
    updated = 1.0 / (constants + feasible[0])
    updated /= float(np.sum(updated))
    return validate_simplex(updated, strictly_positive=True, atol=2e-14)


def burg_directional_first(
    probability: ArrayLike, reward: ArrayLike, *, eta: float, direction: ArrayLike
) -> FloatArray:
    """Return ``D F_B(p)[v]`` by implicit differentiation of the dual shift."""
    state = validate_simplex(probability, strictly_positive=True)
    rewards = validate_reward(reward, state.size)
    tangent = np.asarray(direction, dtype=np.float64)
    if tangent.shape != state.shape or not np.all(np.isfinite(tangent)):
        raise ValueError("direction must be finite with the state shape")
    output = burg_mirror_step(state, rewards, eta=_validate_eta(eta))
    weighted = output**2
    shift_first = float(np.sum(weighted * tangent / state**2) / np.sum(weighted))
    derivative = weighted * (tangent / state**2 - shift_first)
    if abs(float(np.sum(derivative))) > 3e-13 + 3e-15 * float(
        np.linalg.norm(derivative, ord=1)
    ):
        raise FloatingPointError("Burg first derivative left the simplex tangent")
    return derivative


def burg_directional_second(
    probability: ArrayLike, reward: ArrayLike, *, eta: float, direction: ArrayLike
) -> FloatArray:
    """Return ``D^2 F_B(p)[v,v]`` from the implicit normalization equation."""
    state = validate_simplex(probability, strictly_positive=True)
    rewards = validate_reward(reward, state.size)
    tangent = np.asarray(direction, dtype=np.float64)
    if tangent.shape != state.shape or not np.all(np.isfinite(tangent)):
        raise ValueError("direction must be finite with the state shape")
    output = burg_mirror_step(state, rewards, eta=_validate_eta(eta))
    weighted = output**2
    shift_first = float(np.sum(weighted * tangent / state**2) / np.sum(weighted))
    denominator_first = -tangent / state**2 + shift_first
    numerator = np.sum(
        2.0 * output**3 * denominator_first**2
        - 2.0 * weighted * tangent**2 / state**3
    )
    shift_second = float(numerator / np.sum(weighted))
    derivative = 2.0 * output**3 * denominator_first**2
    derivative -= weighted * (2.0 * tangent**2 / state**3 + shift_second)
    if abs(float(np.sum(derivative))) > 2e-11 + 2e-14 * float(
        np.linalg.norm(derivative, ord=1)
    ):
        raise FloatingPointError("Burg second derivative left the simplex tangent")
    return derivative


def burg_second_order_sensitivity_trajectory(
    p0: ArrayLike,
    reward: ArrayLike,
    mutation: ArrayLike,
    *,
    eta: float,
    steps: int,
) -> BurgSecondOrderTrace:
    initial = validate_simplex(p0, strictly_positive=True)
    rewards = validate_reward(reward, initial.size)
    matrix = _validate_mutation(mutation, initial.size)
    eta_value = _validate_eta(eta)
    count = _validate_steps(steps)
    perturbation = matrix - np.eye(initial.size)
    states = np.empty((count + 1, initial.size), dtype=np.float64)
    first = np.zeros_like(states)
    second = np.zeros_like(states)
    states[0] = initial
    for index in range(count):
        state = states[index]
        clean = burg_mirror_step(state, rewards, eta=eta_value)
        selected_first = burg_directional_first(
            state, rewards, eta=eta_value, direction=first[index]
        )
        first[index + 1] = selected_first + clean @ perturbation
        second[index + 1] = burg_directional_first(
            state, rewards, eta=eta_value, direction=second[index]
        )
        second[index + 1] += burg_directional_second(
            state, rewards, eta=eta_value, direction=first[index]
        )
        second[index + 1] += 2.0 * (selected_first @ perturbation)
        if abs(float(np.sum(first[index + 1]))) > 5e-13 + 3e-15 * float(
            np.linalg.norm(first[index + 1], ord=1)
        ):
            raise FloatingPointError("Burg trajectory first derivative violates mass")
        if abs(float(np.sum(second[index + 1]))) > 3e-11 + 2e-14 * float(
            np.linalg.norm(second[index + 1], ord=1)
        ):
            raise FloatingPointError("Burg trajectory second derivative violates mass")
        states[index + 1] = clean
    return BurgSecondOrderTrace(states, first, second)


def burg_perturbed_trajectory(
    p0: ArrayLike,
    reward: ArrayLike,
    mutation: ArrayLike,
    *,
    eta: float,
    epsilon: float,
    steps: int,
) -> FloatArray:
    """Direct signed-epsilon path, independent of the sensitivity recurrence."""
    initial = validate_simplex(p0, strictly_positive=True)
    rewards = validate_reward(reward, initial.size)
    matrix = _validate_mutation(mutation, initial.size)
    eta_value = _validate_eta(eta)
    count = _validate_steps(steps)
    if not np.isfinite(epsilon):
        raise ValueError("epsilon must be finite")
    mixing = np.eye(initial.size) + float(epsilon) * (matrix - np.eye(initial.size))
    states = np.empty((count + 1, initial.size), dtype=np.float64)
    states[0] = initial
    for index in range(count):
        selected = burg_mirror_step(states[index], rewards, eta=eta_value)
        updated = selected @ mixing
        states[index + 1] = validate_simplex(
            updated, strictly_positive=True, atol=3e-13
        )
    return states


def burg_perturbed_trajectory_polynomial_oracle(
    p0: ArrayLike,
    reward: ArrayLike,
    mutation: ArrayLike,
    *,
    eta: float,
    epsilon: float,
    steps: int,
) -> FloatArray:
    """Direct perturbed path using only the polynomial normalizer oracle."""
    initial = validate_simplex(p0, strictly_positive=True)
    rewards = validate_reward(reward, initial.size)
    matrix = _validate_mutation(mutation, initial.size)
    eta_value = _validate_eta(eta)
    count = _validate_steps(steps)
    if not np.isfinite(epsilon):
        raise ValueError("epsilon must be finite")
    mixing = np.eye(initial.size) + float(epsilon) * (matrix - np.eye(initial.size))
    states = np.empty((count + 1, initial.size), dtype=np.float64)
    states[0] = initial
    for index in range(count):
        selected = burg_mirror_step_polynomial_oracle(
            states[index], rewards, eta=eta_value
        )
        states[index + 1] = validate_simplex(
            selected @ mixing, strictly_positive=True, atol=3e-13
        )
    return states


def five_point_trajectory_derivatives(
    p0: ArrayLike,
    reward: ArrayLike,
    mutation: ArrayLike,
    *,
    eta: float,
    steps: int,
    step: float,
) -> tuple[FloatArray, FloatArray]:
    """Independent symmetric five-point derivatives of the direct trajectory."""
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("finite-difference step must be positive")
    trajectories = {
        multiplier: burg_perturbed_trajectory(
            p0,
            reward,
            mutation,
            eta=eta,
            epsilon=multiplier * step,
            steps=steps,
        )
        for multiplier in (-2.0, -1.0, 0.0, 1.0, 2.0)
    }
    first = (
        trajectories[-2.0]
        - 8.0 * trajectories[-1.0]
        + 8.0 * trajectories[1.0]
        - trajectories[2.0]
    ) / (12.0 * step)
    second = (
        -trajectories[2.0]
        + 16.0 * trajectories[1.0]
        - 30.0 * trajectories[0.0]
        + 16.0 * trajectories[-1.0]
        - trajectories[-2.0]
    ) / (12.0 * step**2)
    return first, second
