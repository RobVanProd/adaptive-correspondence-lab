"""Exact and labeled-particle paths for a tiny sequential Bayesian filter."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class ParticleFilterSpecification:
    """True and approximate models for one three-state filtering sequence."""

    initial_belief: ArrayLike
    true_transition: ArrayLike
    filter_transition: ArrayLike
    true_likelihoods: ArrayLike
    filter_likelihoods: ArrayLike

    def arrays(
        self,
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
        initial = np.asarray(self.initial_belief, dtype=np.float64)
        true_transition = np.asarray(self.true_transition, dtype=np.float64)
        filter_transition = np.asarray(self.filter_transition, dtype=np.float64)
        true_likelihoods = np.asarray(self.true_likelihoods, dtype=np.float64)
        filter_likelihoods = np.asarray(self.filter_likelihoods, dtype=np.float64)
        if initial.shape != (3,):
            raise ValueError("initial belief must have three states")
        if true_transition.shape != (3, 3) or filter_transition.shape != (3, 3):
            raise ValueError("transition matrices must have shape (3, 3)")
        if (
            true_likelihoods.ndim != 2
            or true_likelihoods.shape[1] != 3
            or true_likelihoods.shape[0] < 1
            or filter_likelihoods.shape != true_likelihoods.shape
        ):
            raise ValueError("likelihood sequences must have matching shape (steps, 3)")
        arrays = (
            initial,
            true_transition,
            filter_transition,
            true_likelihoods,
            filter_likelihoods,
        )
        if not all(np.all(np.isfinite(array)) for array in arrays):
            raise ValueError("particle-filter specification must be finite")
        if np.any(initial <= 0.0) or not np.isclose(
            np.sum(initial), 1.0, atol=2e-14, rtol=0.0
        ):
            raise ValueError("initial belief must be interior and sum to one")
        for transition in (true_transition, filter_transition):
            if np.any(transition < 0.0) or not np.allclose(
                np.sum(transition, axis=1), 1.0, atol=2e-14, rtol=0.0
            ):
                raise ValueError("transition matrices must be row-stochastic")
        if np.any(true_likelihoods <= 0.0) or np.any(filter_likelihoods <= 0.0):
            raise ValueError("observation likelihoods must be strictly positive")
        return arrays


@dataclass(frozen=True)
class ExactParticleFilterMoments:
    particle_count: int
    probability_mass: float
    exact_belief_trajectory: FloatArray
    analytic_update: FloatArray
    mean_belief: FloatArray
    mean_update: FloatArray
    covariance: FloatArray
    truth_alignment_cosine: float
    terminal_missing_state_probability: FloatArray
    terminal_support_size_probabilities: dict[int, float]
    terminal_count_distribution: dict[tuple[int, int, int], float]


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def count_compositions(total: int) -> tuple[tuple[int, int, int], ...]:
    """Return all ordered three-cell count vectors summing to ``total``."""
    total = _positive_integer(total, "total")
    return tuple(
        (first, second, total - first - second)
        for first in range(total + 1)
        for second in range(total - first + 1)
    )


def _zero_inclusive_compositions(total: int) -> tuple[tuple[int, int, int], ...]:
    if isinstance(total, bool) or not isinstance(total, int) or total < 0:
        raise ValueError("composition total must be a non-negative integer")
    return tuple(
        (first, second, total - first - second)
        for first in range(total + 1)
        for second in range(total - first + 1)
    )


def _multinomial_probability(counts: tuple[int, int, int], probabilities: FloatArray) -> float:
    total = sum(counts)
    log_probability = math.lgamma(total + 1.0)
    for count in counts:
        log_probability -= math.lgamma(count + 1.0)
    for count, probability in zip(counts, probabilities, strict=True):
        if count:
            if probability <= 0.0:
                return 0.0
            log_probability += count * math.log(float(probability))
    return math.exp(log_probability)


def exact_bayes_trajectory(specification: ParticleFilterSpecification) -> FloatArray:
    """Compute the true-model filtering trajectory for the frozen observations."""
    initial, transition, _, likelihoods, _ = specification.arrays()
    belief = initial.copy()
    trajectory = [belief.copy()]
    for likelihood in likelihoods:
        belief = belief @ transition
        belief *= likelihood
        mass = float(np.sum(belief))
        if not np.isfinite(mass) or mass <= 0.0:
            raise FloatingPointError("Bayes update has invalid normalizing mass")
        belief /= mass
        trajectory.append(belief.copy())
    return np.asarray(trajectory, dtype=np.float64)


def grouped_transition_distribution(
    source_counts: ArrayLike, transition: ArrayLike
) -> dict[tuple[int, int, int], float]:
    """Exact predicted-count law for grouped source-state particles."""
    counts = np.asarray(source_counts)
    matrix = np.asarray(transition, dtype=np.float64)
    if counts.shape != (3,) or not np.issubdtype(counts.dtype, np.integer):
        raise ValueError("source counts must be three integers")
    if np.any(counts < 0) or int(np.sum(counts)) < 1:
        raise ValueError("source counts must be non-negative with positive total")
    if matrix.shape != (3, 3) or np.any(matrix < 0.0) or not np.allclose(
        np.sum(matrix, axis=1), 1.0, atol=2e-14, rtol=0.0
    ):
        raise ValueError("transition must be a row-stochastic (3, 3) matrix")
    distribution: dict[tuple[int, int, int], float] = {(0, 0, 0): 1.0}
    for source, group_count in enumerate(counts):
        if group_count == 0:
            continue
        group_distribution = {
            destination: _multinomial_probability(destination, matrix[source])
            for destination in _zero_inclusive_compositions(int(group_count))
        }
        combined: dict[tuple[int, int, int], float] = {}
        for accumulated, left_mass in distribution.items():
            for destination, right_mass in group_distribution.items():
                total = tuple(
                    left + right
                    for left, right in zip(accumulated, destination, strict=True)
                )
                combined[total] = combined.get(total, 0.0) + left_mass * right_mass
        distribution = combined
    mass = math.fsum(distribution.values())
    if not np.isclose(mass, 1.0, atol=3e-14, rtol=0.0):
        raise FloatingPointError("grouped transition probability does not sum to one")
    return distribution


def particle_filter_count_kernel(
    *, particle_count: int, transition: ArrayLike, likelihood: ArrayLike
) -> FloatArray:
    """Exact count-state transition: particle propagation then multinomial resampling."""
    particle_count = _positive_integer(particle_count, "particle_count")
    matrix = np.asarray(transition, dtype=np.float64)
    likelihood_array = np.asarray(likelihood, dtype=np.float64)
    if likelihood_array.shape != (3,) or not np.all(np.isfinite(likelihood_array)):
        raise ValueError("likelihood must contain three finite values")
    if np.any(likelihood_array <= 0.0):
        raise ValueError("likelihood must be strictly positive")
    states = count_compositions(particle_count)
    index = {counts: position for position, counts in enumerate(states)}
    kernel = np.zeros((len(states), len(states)), dtype=np.float64)
    for row, source in enumerate(states):
        predicted_distribution = grouped_transition_distribution(source, matrix)
        for predicted, predicted_mass in predicted_distribution.items():
            weighted = np.asarray(predicted, dtype=np.float64) * likelihood_array
            resampling_probabilities = weighted / np.sum(weighted)
            for terminal in states:
                kernel[row, index[terminal]] += predicted_mass * _multinomial_probability(
                    terminal, resampling_probabilities
                )
    row_mass = np.sum(kernel, axis=1)
    if np.any(kernel < -2e-15) or not np.allclose(
        row_mass, 1.0, atol=8e-14, rtol=0.0
    ):
        raise FloatingPointError("particle-filter count kernel is not row-stochastic")
    return np.maximum(kernel, 0.0)


def _euclidean_cosine(left: FloatArray, right: FloatArray) -> float:
    numerator = float(left @ right)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 0.0 or not np.isfinite(denominator):
        raise ValueError("Euclidean cosine requires nonzero finite directions")
    return float(np.clip(numerator / denominator, -1.0, 1.0))


def exact_particle_filter_moments(
    specification: ParticleFilterSpecification, *, particle_count: int
) -> ExactParticleFilterMoments:
    """Propagate the exact finite-particle count distribution over the sequence."""
    particle_count = _positive_integer(particle_count, "particle_count")
    initial, _, filter_transition, _, filter_likelihoods = specification.arrays()
    states = count_compositions(particle_count)
    distribution = np.asarray(
        [_multinomial_probability(counts, initial) for counts in states],
        dtype=np.float64,
    )
    for likelihood in filter_likelihoods:
        kernel = particle_filter_count_kernel(
            particle_count=particle_count,
            transition=filter_transition,
            likelihood=likelihood,
        )
        distribution = distribution @ kernel
    probability_mass = float(np.sum(distribution))
    if not np.isclose(probability_mass, 1.0, atol=2e-13, rtol=0.0):
        raise FloatingPointError("terminal count probability does not sum to one")
    beliefs = np.asarray(states, dtype=np.float64) / particle_count
    mean_belief = distribution @ beliefs
    second = np.einsum("b,bi,bj->ij", distribution, beliefs, beliefs)
    covariance = second - np.outer(mean_belief, mean_belief)
    covariance = 0.5 * (covariance + covariance.T)
    if np.any(np.diag(covariance) < -2e-13) or not np.all(np.isfinite(covariance)):
        raise FloatingPointError("terminal belief covariance is invalid")
    true_trajectory = exact_bayes_trajectory(specification)
    analytic_update = true_trajectory[-1] - initial
    mean_update = mean_belief - initial
    missing = np.asarray(
        [
            math.fsum(
                probability
                for counts, probability in zip(states, distribution, strict=True)
                if counts[state] == 0
            )
            for state in range(3)
        ],
        dtype=np.float64,
    )
    support_sizes = {
        size: math.fsum(
            probability
            for counts, probability in zip(states, distribution, strict=True)
            if sum(count > 0 for count in counts) == size
        )
        for size in (1, 2, 3)
    }
    return ExactParticleFilterMoments(
        particle_count=particle_count,
        probability_mass=probability_mass,
        exact_belief_trajectory=true_trajectory,
        analytic_update=analytic_update,
        mean_belief=mean_belief,
        mean_update=mean_update,
        covariance=covariance,
        truth_alignment_cosine=_euclidean_cosine(mean_update, analytic_update),
        terminal_missing_state_probability=missing,
        terminal_support_size_probabilities=support_sizes,
        terminal_count_distribution={
            counts: float(probability)
            for counts, probability in zip(states, distribution, strict=True)
        },
    )


def simulate_labeled_particle_filters(
    specification: ParticleFilterSpecification,
    *,
    particle_count: int,
    replications: int,
    rng: np.random.Generator,
    batch_size: int = 4096,
) -> tuple[FloatArray, IntArray]:
    """Run labeled particles without calling the count-state oracle."""
    particle_count = _positive_integer(particle_count, "particle_count")
    replications = _positive_integer(replications, "replications")
    batch_size = _positive_integer(batch_size, "batch_size")
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be a NumPy Generator")
    initial, _, transition, _, likelihoods = specification.arrays()
    belief_batches: list[FloatArray] = []
    particle_batches: list[IntArray] = []
    generated = 0
    initial_cdf = np.cumsum(initial)
    transition_cdf = np.cumsum(transition, axis=1)
    initial_cdf[-1] = 1.0
    transition_cdf[:, -1] = 1.0
    while generated < replications:
        size = min(batch_size, replications - generated)
        particles = np.sum(
            rng.random((size, particle_count, 1)) > initial_cdf[None, None, :],
            axis=2,
        ).astype(np.int64)
        for likelihood in likelihoods:
            selected_cdf = transition_cdf[particles]
            particles = np.sum(
                rng.random((size, particle_count, 1)) > selected_cdf,
                axis=2,
            ).astype(np.int64)
            weights = likelihood[particles]
            weights /= np.sum(weights, axis=1, keepdims=True)
            cumulative = np.cumsum(weights, axis=1)
            cumulative[:, -1] = 1.0
            uniforms = rng.random((size, particle_count))
            ancestors = np.empty((size, particle_count), dtype=np.int64)
            for draw in range(particle_count):
                ancestors[:, draw] = np.sum(
                    uniforms[:, draw, None] > cumulative, axis=1
                )
            particles = np.take_along_axis(particles, ancestors, axis=1)
        beliefs = np.mean(
            particles[:, :, None] == np.arange(3, dtype=np.int64)[None, None, :],
            axis=1,
        )
        belief_batches.append(beliefs.astype(np.float64, copy=False))
        particle_batches.append(particles)
        generated += size
    return np.concatenate(belief_batches), np.concatenate(particle_batches)
