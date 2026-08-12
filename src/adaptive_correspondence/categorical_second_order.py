"""Second-order sensitivity for the categorical selection-plus-mutation map.

The primary implementation differentiates the normalized categorical recurrence. An
independent matrix-polynomial implementation propagates unnormalized coefficients and
normalizes them only afterward. Neither function generates scientific outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .acl002 import categorical_map, row_jacobian
from .simplex import validate_reward, validate_simplex

FloatArray = NDArray[np.float64]
L1_COORDINATE_ZERO_TOLERANCE = 2e-14
FIRST_ORDER_MASS_TOLERANCE = 2e-13
SECOND_ORDER_MASS_TOLERANCE = 2e-11


@dataclass(frozen=True)
class SecondOrderTrace:
    """Clean states plus first and second epsilon derivatives."""

    states: FloatArray
    first: FloatArray
    second: FloatArray


def _validate_steps(steps: int) -> int:
    if isinstance(steps, bool) or not isinstance(steps, (int, np.integer)) or steps < 0:
        raise ValueError("steps must be a non-negative integer")
    return int(steps)


def _validate_eta(eta: float) -> float:
    if not np.isfinite(eta) or eta < 0.0:
        raise ValueError("eta must be finite and non-negative")
    return float(eta)


def _validate_mutation(mutation: ArrayLike, dimension: int) -> FloatArray:
    matrix = np.asarray(mutation, dtype=np.float64)
    if matrix.shape != (dimension, dimension):
        raise ValueError("mutation matrix must be square with the state dimension")
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0.0):
        raise ValueError("mutation matrix must be finite and non-negative")
    if not np.allclose(np.sum(matrix, axis=1), 1.0, rtol=0.0, atol=2e-14):
        raise ValueError("mutation matrix must be row-stochastic")
    return matrix.copy()


def _selection_factors(reward: FloatArray, eta: float) -> FloatArray:
    return np.exp(eta * (reward - float(np.max(reward))))


def row_hessian_quadratic(
    probability: ArrayLike,
    reward: ArrayLike,
    eta: float,
    direction: ArrayLike,
) -> FloatArray:
    """Return the row-vector contraction D^2 F(p)[direction,direction]."""
    state = validate_simplex(probability, name="probability", strictly_positive=True)
    rewards = validate_reward(reward, state.size)
    eta_value = _validate_eta(eta)
    tangent = np.asarray(direction, dtype=np.float64)
    if tangent.shape != state.shape or not np.all(np.isfinite(tangent)):
        raise ValueError("direction must be finite with the probability shape")
    factors = _selection_factors(rewards, eta_value)
    normalizer = float(np.dot(state, factors))
    beta = float(np.dot(tangent, factors) / normalizer)
    first_direction = tangent @ row_jacobian(state, rewards, eta_value)
    contracted = -2.0 * beta * first_direction
    if abs(float(np.sum(contracted))) > FIRST_ORDER_MASS_TOLERANCE:
        raise FloatingPointError("row Hessian contraction left the simplex tangent space")
    return contracted


def second_order_sensitivity_trajectory(
    p0: ArrayLike,
    reward: ArrayLike,
    mutation: ArrayLike,
    *,
    eta: float,
    steps: int,
) -> SecondOrderTrace:
    """Propagate p, s=dq/depsilon, and u=d^2q/depsilon^2 at epsilon zero."""
    initial = validate_simplex(p0, name="p0", strictly_positive=True)
    rewards = validate_reward(reward, initial.size)
    matrix = _validate_mutation(mutation, initial.size)
    eta_value = _validate_eta(eta)
    step_count = _validate_steps(steps)
    identity = np.eye(initial.size, dtype=np.float64)
    perturbation = matrix - identity
    states = np.empty((step_count + 1, initial.size), dtype=np.float64)
    first = np.empty_like(states)
    second = np.empty_like(states)
    states[0] = initial
    first[0] = 0.0
    second[0] = 0.0
    for step in range(step_count):
        current = states[step]
        jacobian = row_jacobian(current, rewards, eta_value)
        clean_next = categorical_map(current, rewards, eta_value)
        selected_first = first[step] @ jacobian
        first_next = selected_first + clean_next @ perturbation
        second_next = second[step] @ jacobian
        second_next += row_hessian_quadratic(current, rewards, eta_value, first[step])
        second_next += 2.0 * (selected_first @ perturbation)
        if abs(float(np.sum(first_next))) > FIRST_ORDER_MASS_TOLERANCE:
            raise FloatingPointError("first derivative left the simplex tangent space")
        if abs(float(np.sum(second_next))) > SECOND_ORDER_MASS_TOLERANCE:
            raise FloatingPointError("second derivative left the simplex tangent space")
        states[step + 1] = validate_simplex(clean_next, strictly_positive=True)
        first[step + 1] = first_next
        second[step + 1] = second_next
    return SecondOrderTrace(states=states, first=first, second=second)


def matrix_polynomial_second_order_trajectory(
    p0: ArrayLike,
    reward: ArrayLike,
    mutation: ArrayLike,
    *,
    eta: float,
    steps: int,
) -> SecondOrderTrace:
    """Independently differentiate normalize(p0 [D(I+epsilon B)]^t)."""
    initial = validate_simplex(p0, name="p0", strictly_positive=True)
    rewards = validate_reward(reward, initial.size)
    matrix = _validate_mutation(mutation, initial.size)
    eta_value = _validate_eta(eta)
    step_count = _validate_steps(steps)
    identity = np.eye(initial.size, dtype=np.float64)
    selection = np.diag(_selection_factors(rewards, eta_value))
    transition_zero = selection
    transition_one = selection @ (matrix - identity)

    states = np.empty((step_count + 1, initial.size), dtype=np.float64)
    first = np.empty_like(states)
    second = np.empty_like(states)
    coefficient_zero = initial.copy()
    coefficient_one = np.zeros_like(initial)
    coefficient_two = np.zeros_like(initial)
    states[0] = initial
    first[0] = 0.0
    second[0] = 0.0
    for step in range(1, step_count + 1):
        next_two = coefficient_two @ transition_zero + coefficient_one @ transition_one
        next_one = coefficient_one @ transition_zero + coefficient_zero @ transition_one
        next_zero = coefficient_zero @ transition_zero
        coefficient_zero, coefficient_one, coefficient_two = next_zero, next_one, next_two
        mass_zero = float(np.sum(coefficient_zero))
        mass_one = float(np.sum(coefficient_one))
        mass_two = float(np.sum(coefficient_two))
        if not np.isfinite(mass_zero) or mass_zero <= 0.0:
            raise FloatingPointError("polynomial oracle has no finite positive clean mass")
        clean = coefficient_zero / mass_zero
        tangent = coefficient_one / mass_zero
        tangent -= coefficient_zero * mass_one / mass_zero**2
        quadratic_coefficient = coefficient_two / mass_zero
        quadratic_coefficient -= coefficient_one * mass_one / mass_zero**2
        quadratic_coefficient += coefficient_zero * (
            mass_one**2 / mass_zero**3 - mass_two / mass_zero**2
        )
        curvature = 2.0 * quadratic_coefficient
        if abs(float(np.sum(tangent))) > FIRST_ORDER_MASS_TOLERANCE:
            raise FloatingPointError("polynomial first derivative violates tangent mass")
        if abs(float(np.sum(curvature))) > SECOND_ORDER_MASS_TOLERANCE:
            raise FloatingPointError("polynomial second derivative violates tangent mass")
        states[step] = validate_simplex(clean, strictly_positive=True)
        first[step] = tangent
        second[step] = curvature
    return SecondOrderTrace(states=states, first=first, second=second)


def signed_matrix_power_state(
    p0: ArrayLike,
    reward: ArrayLike,
    mutation: ArrayLike,
    *,
    eta: float,
    epsilon: float,
    steps: int,
) -> FloatArray:
    """Evaluate the analytic matrix expression at signed epsilon for derivative tests."""
    initial = validate_simplex(p0, name="p0", strictly_positive=True)
    rewards = validate_reward(reward, initial.size)
    matrix = _validate_mutation(mutation, initial.size)
    eta_value = _validate_eta(eta)
    step_count = _validate_steps(steps)
    if not np.isfinite(epsilon):
        raise ValueError("epsilon must be finite")
    selection = np.diag(_selection_factors(rewards, eta_value))
    mixing = np.eye(initial.size, dtype=np.float64) + float(epsilon) * (
        matrix - np.eye(initial.size, dtype=np.float64)
    )
    unnormalized = initial @ np.linalg.matrix_power(selection @ mixing, step_count)
    total = float(np.sum(unnormalized))
    if not np.isfinite(total) or total <= 0.0:
        raise FloatingPointError("signed matrix expression has no finite positive mass")
    state = unnormalized / total
    if not np.all(np.isfinite(state)) or np.any(state <= 0.0):
        raise FloatingPointError("signed matrix expression left the simplex interior")
    return state


def l1_second_order_coefficient(
    first: ArrayLike,
    second: ArrayLike,
    *,
    zero_tolerance: float = L1_COORDINATE_ZERO_TOLERANCE,
) -> tuple[float, tuple[int, ...]]:
    """Return the epsilon-squared L1 coefficient with explicit zero-coordinate branch."""
    tangent = np.asarray(first, dtype=np.float64)
    curvature = np.asarray(second, dtype=np.float64)
    if tangent.ndim != 1 or curvature.shape != tangent.shape:
        raise ValueError("first and second derivatives must be same-shaped vectors")
    if not np.all(np.isfinite(tangent)) or not np.all(np.isfinite(curvature)):
        raise ValueError("first and second derivatives must be finite")
    if not np.isfinite(zero_tolerance) or zero_tolerance < 0.0:
        raise ValueError("zero_tolerance must be finite and non-negative")
    zero_mask = np.abs(tangent) <= zero_tolerance
    nonzero_contribution = np.sum(np.sign(tangent[~zero_mask]) * curvature[~zero_mask])
    zero_contribution = np.sum(np.abs(curvature[zero_mask]))
    coefficient = 0.5 * float(nonzero_contribution + zero_contribution)
    zero_coordinates = tuple(int(index) for index in np.flatnonzero(zero_mask))
    return coefficient, zero_coordinates


def l1_truncated_prediction(first: ArrayLike, second: ArrayLike, *, epsilon: float) -> float:
    """Evaluate ||epsilon*s + epsilon^2*u/2||_1 without assuming fixed signs."""
    tangent = np.asarray(first, dtype=np.float64)
    curvature = np.asarray(second, dtype=np.float64)
    if tangent.ndim != 1 or curvature.shape != tangent.shape:
        raise ValueError("first and second derivatives must be same-shaped vectors")
    if not np.all(np.isfinite(tangent)) or not np.all(np.isfinite(curvature)):
        raise ValueError("first and second derivatives must be finite")
    if not np.isfinite(epsilon) or epsilon < 0.0:
        raise ValueError("epsilon must be finite and non-negative")
    displacement = float(epsilon) * tangent + 0.5 * float(epsilon) ** 2 * curvature
    return float(np.linalg.norm(displacement, ord=1))
