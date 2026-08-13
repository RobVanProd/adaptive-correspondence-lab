"""Exact finite-support bias analysis for the undamped plug-in NPG block."""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class BlockSpecification:
    """One contextual-bandit Fisher block and its joint sampling probability."""

    policy: ArrayLike
    reward: ArrayLike
    context_probability: float

    def arrays(self) -> tuple[FloatArray, FloatArray, float]:
        policy = np.asarray(self.policy, dtype=np.float64)
        reward = np.asarray(self.reward, dtype=np.float64)
        rho = float(self.context_probability)
        if policy.ndim != 1 or policy.size != 3:
            raise ValueError("support-bias enumeration requires exactly three actions")
        if reward.shape != policy.shape:
            raise ValueError("reward must match policy shape")
        if not np.all(np.isfinite(policy)) or not np.all(np.isfinite(reward)):
            raise ValueError("policy and reward must be finite")
        if np.any(policy <= 0.0) or not np.isclose(np.sum(policy), 1.0, atol=2e-14):
            raise ValueError("policy must be strictly interior and sum to one")
        if not np.isfinite(rho) or not 0.0 < rho < 1.0:
            raise ValueError("context probability must lie strictly between zero and one")
        return policy, reward, rho


@dataclass(frozen=True)
class ExactBlockMoments:
    sample_count: int
    probability_mass: float
    analytic_direction: FloatArray
    mean_direction: FloatArray
    covariance: FloatArray
    mean_support_loss: FloatArray
    mean_observed_support_perturbation: FloatArray
    expected_squared_error: float
    expected_squared_support_loss: float
    expected_squared_observed_support_perturbation: float
    truth_alignment_cosine: float
    rank_deficient_probability: float
    support_probabilities: dict[tuple[int, ...], float]


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _tangent_basis() -> FloatArray:
    return np.asarray(
        (
            (1.0 / np.sqrt(2.0), 1.0 / np.sqrt(6.0)),
            (-1.0 / np.sqrt(2.0), 1.0 / np.sqrt(6.0)),
            (0.0, -2.0 / np.sqrt(6.0)),
        ),
        dtype=np.float64,
    )


def _fisher(specification: BlockSpecification) -> FloatArray:
    policy, _, rho = specification.arrays()
    return rho * (np.diag(policy) - np.outer(policy, policy))


def fisher_cosine(
    specification: BlockSpecification, left: ArrayLike, right: ArrayLike
) -> float:
    metric = _fisher(specification)
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if left_array.shape != (3,) or right_array.shape != (3,):
        raise ValueError("directions must have three coordinates")
    numerator = float(left_array @ metric @ right_array)
    left_norm = float(left_array @ metric @ left_array)
    right_norm = float(right_array @ metric @ right_array)
    if left_norm <= 0.0 or right_norm <= 0.0:
        raise ValueError("Fisher cosine requires nonzero tangent directions")
    return float(np.clip(numerator / np.sqrt(left_norm * right_norm), -1.0, 1.0))


def support_pattern_probabilities(
    specification: BlockSpecification, *, sample_count: int
) -> dict[tuple[int, ...], float]:
    """Return exact probabilities for every observed action-support pattern."""
    sample_count = _positive_integer(sample_count, "sample_count")
    policy, _, rho = specification.arrays()
    action_probabilities = rho * policy
    outside_probability = 1.0 - rho
    result: dict[tuple[int, ...], float] = {}
    actions = range(policy.size)
    for size in range(policy.size + 1):
        for support in itertools.combinations(actions, size):
            probability = 0.0
            for retained_size in range(size + 1):
                for retained in itertools.combinations(support, retained_size):
                    sign = -1.0 if (size - retained_size) % 2 else 1.0
                    allowed = outside_probability + float(
                        np.sum(action_probabilities[list(retained)])
                    )
                    probability += sign * allowed**sample_count
            if probability < -5e-14:
                raise FloatingPointError("support probability is materially negative")
            result[support] = float(max(probability, 0.0))
    mass = math.fsum(result.values())
    if not np.isclose(mass, 1.0, rtol=0.0, atol=2e-13):
        raise FloatingPointError("support-pattern probability mass does not sum to one")
    return result


def _count_chunks(sample_count: int, chunk_size: int) -> Iterator[IntArray]:
    rows: list[tuple[int, int, int, int]] = []
    for first in range(sample_count + 1):
        for second in range(sample_count - first + 1):
            for third in range(sample_count - first - second + 1):
                outside = sample_count - first - second - third
                rows.append((first, second, third, outside))
                if len(rows) == chunk_size:
                    yield np.asarray(rows, dtype=np.int64)
                    rows = []
    if rows:
        yield np.asarray(rows, dtype=np.int64)


def _support_projectors(
    specification: BlockSpecification,
) -> tuple[dict[int, FloatArray], FloatArray, FloatArray, FloatArray]:
    policy, reward, _ = specification.arrays()
    basis = _tangent_basis()
    scores = np.eye(policy.size) - policy[None, :]
    score_coordinates = scores @ basis
    metric = _fisher(specification)
    tangent_metric = basis.T @ metric @ basis
    analytic = reward - np.mean(reward)
    analytic_coordinates = basis.T @ analytic
    projections: dict[int, FloatArray] = {}
    for mask in range(1 << policy.size):
        support = [action for action in range(policy.size) if mask & (1 << action)]
        if not support:
            projected_coordinates = np.zeros(2, dtype=np.float64)
        else:
            subspace = score_coordinates[support].T
            gram = subspace.T @ tangent_metric @ subspace
            projected_coordinates = (
                subspace
                @ np.linalg.pinv(gram, rcond=1e-12)
                @ subspace.T
                @ tangent_metric
                @ analytic_coordinates
            )
        projections[mask] = basis @ projected_coordinates
    return projections, basis, scores, analytic


def _directions_for_counts(
    *,
    counts: IntArray,
    reward: FloatArray,
    basis: FloatArray,
    scores: FloatArray,
    sample_count: int,
    rcond: float,
) -> FloatArray:
    action_counts = counts[:, :3].astype(np.float64)
    score_coordinates = scores @ basis
    score_outer = score_coordinates[:, :, None] * score_coordinates[:, None, :]
    empirical_fisher = np.einsum("ba,aij->bij", action_counts, score_outer) / sample_count
    empirical_gradient = (action_counts * reward[None, :]) @ score_coordinates / sample_count
    eigenvalues, eigenvectors = np.linalg.eigh(empirical_fisher)
    maximum = np.max(eigenvalues, axis=1, keepdims=True)
    inverse = np.divide(
        1.0,
        eigenvalues,
        out=np.zeros_like(eigenvalues),
        where=eigenvalues > rcond * maximum,
    )
    projected_gradient = np.einsum(
        "bij,bj->bi", eigenvectors.transpose(0, 2, 1), empirical_gradient
    )
    coordinates = np.einsum("bij,bj->bi", eigenvectors, inverse * projected_gradient)
    return coordinates @ basis.T


def exact_block_moments(
    specification: BlockSpecification,
    *,
    sample_count: int,
    rcond: float = 1e-12,
    enumeration_chunk_size: int = 50_000,
) -> ExactBlockMoments:
    """Enumerate the exact finite-N plug-in direction distribution for one block."""
    sample_count = _positive_integer(sample_count, "sample_count")
    chunk_size = _positive_integer(enumeration_chunk_size, "enumeration_chunk_size")
    if not np.isfinite(rcond) or rcond <= 0.0:
        raise ValueError("rcond must be finite and positive")
    policy, reward, rho = specification.arrays()
    probabilities = np.concatenate((rho * policy, [1.0 - rho]))
    log_probabilities = np.log(probabilities)
    log_factorials = np.asarray(
        [math.lgamma(value + 1.0) for value in range(sample_count + 1)],
        dtype=np.float64,
    )
    log_factorial_n = log_factorials[sample_count]
    projections, basis, scores, analytic = _support_projectors(specification)
    metric = _fisher(specification)
    support_probabilities = support_pattern_probabilities(
        specification, sample_count=sample_count
    )

    probability_mass = 0.0
    mean = np.zeros(3, dtype=np.float64)
    second = np.zeros((3, 3), dtype=np.float64)
    mean_support_loss = np.zeros(3, dtype=np.float64)
    mean_observed = np.zeros(3, dtype=np.float64)
    squared_total = 0.0
    squared_support = 0.0
    squared_observed = 0.0
    enumerated_support_mass = np.zeros(1 << policy.size, dtype=np.float64)

    for counts in _count_chunks(sample_count, chunk_size):
        log_mass = (
            log_factorial_n
            - np.sum(log_factorials[counts], axis=1)
            + counts @ log_probabilities
        )
        masses = np.exp(log_mass)
        directions = _directions_for_counts(
            counts=counts,
            reward=reward,
            basis=basis,
            scores=scores,
            sample_count=sample_count,
            rcond=rcond,
        )
        masks = np.sum(
            (counts[:, :3] > 0).astype(np.int64)
            * (1 << np.arange(policy.size, dtype=np.int64))[None, :],
            axis=1,
        )
        projected = np.asarray([projections[int(mask)] for mask in masks])
        support_loss = projected - analytic[None, :]
        observed = directions - projected
        total = directions - analytic[None, :]
        probability_mass += float(np.sum(masses))
        mean += masses @ directions
        second += np.einsum("b,bi,bj->ij", masses, directions, directions)
        mean_support_loss += masses @ support_loss
        mean_observed += masses @ observed
        squared_total += float(
            masses @ np.einsum("bi,ij,bj->b", total, metric, total)
        )
        squared_support += float(
            masses @ np.einsum("bi,ij,bj->b", support_loss, metric, support_loss)
        )
        squared_observed += float(
            masses @ np.einsum("bi,ij,bj->b", observed, metric, observed)
        )
        enumerated_support_mass += np.bincount(
            masks, weights=masses, minlength=1 << policy.size
        )

    if not np.isclose(probability_mass, 1.0, rtol=0.0, atol=5e-13):
        raise FloatingPointError("enumerated multinomial mass does not sum to one")
    for support, expected in support_probabilities.items():
        mask = sum(1 << action for action in support)
        if not np.isclose(
            enumerated_support_mass[mask], expected, rtol=5e-13, atol=5e-14
        ):
            raise FloatingPointError("enumerated and analytic support probabilities disagree")
    covariance = second - np.outer(mean, mean)
    covariance = 0.5 * (covariance + covariance.T)
    diagonal = np.diag(covariance)
    if np.any(diagonal < -2e-13) or not np.all(np.isfinite(covariance)):
        raise FloatingPointError("enumerated direction covariance is invalid")
    rank_deficient = math.fsum(
        probability
        for support, probability in support_probabilities.items()
        if len(support) < 2
    )
    return ExactBlockMoments(
        sample_count=sample_count,
        probability_mass=probability_mass,
        analytic_direction=analytic,
        mean_direction=mean,
        covariance=covariance,
        mean_support_loss=mean_support_loss,
        mean_observed_support_perturbation=mean_observed,
        expected_squared_error=squared_total,
        expected_squared_support_loss=squared_support,
        expected_squared_observed_support_perturbation=squared_observed,
        truth_alignment_cosine=fisher_cosine(specification, mean, analytic),
        rank_deficient_probability=rank_deficient,
        support_probabilities=support_probabilities,
    )


def self_consistency_limits(
    specification: BlockSpecification,
    *,
    estimator_mean: ArrayLike,
    analytic_direction: ArrayLike,
) -> dict[str, float]:
    """Return the SLLN split-half and truth-alignment cosine limits."""
    mean = np.asarray(estimator_mean, dtype=np.float64)
    analytic = np.asarray(analytic_direction, dtype=np.float64)
    fisher_cosine(specification, mean, mean)
    return {
        "split_half_limit": 1.0,
        "truth_alignment_limit": fisher_cosine(specification, mean, analytic),
    }
