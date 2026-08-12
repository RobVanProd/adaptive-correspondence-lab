"""Finite-lambda rank-mu expectation from Gaussian score and Fisher geometry."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def _finite_vector(value: ArrayLike, name: str) -> FloatArray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.ndim != 1 or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite vector")
    return vector.copy()


@dataclass(frozen=True)
class GaussianLinearBridgeState:
    """Diagonal Gaussian parameters and a linear ranking objective."""

    mean: ArrayLike
    log_std: ArrayLike
    objective: ArrayLike

    def arrays(self) -> tuple[FloatArray, FloatArray, FloatArray]:
        mean = _finite_vector(self.mean, "mean")
        log_std = _finite_vector(self.log_std, "log_std")
        objective = _finite_vector(self.objective, "objective")
        if mean.size < 2 or mean.shape != log_std.shape or mean.shape != objective.shape:
            raise ValueError("Gaussian bridge vectors must share dimension of at least two")
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            standard_deviation = np.exp(log_std)
        if not np.all(np.isfinite(standard_deviation)) or np.any(standard_deviation <= 0.0):
            raise ValueError("log_std must represent finite positive scales")
        if not np.any(objective != 0.0) or np.linalg.norm(objective * standard_deviation) == 0.0:
            raise ValueError("linear objective must induce a nonzero standardized axis")
        return mean, log_std, objective


def logarithmic_rank_weights(sample_count: int, parent_count: int) -> FloatArray:
    """Return frozen positive logarithmic weights for ranks 1 through mu."""
    if (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or sample_count < 2
    ):
        raise ValueError("sample_count must be an integer of at least two")
    if (
        isinstance(parent_count, bool)
        or not isinstance(parent_count, int)
        or not 1 <= parent_count <= sample_count
    ):
        raise ValueError("parent_count must be in [1, sample_count]")
    ranks = np.arange(1, parent_count + 1, dtype=np.float64)
    weights = np.log(parent_count + 0.5) - np.log(ranks)
    if np.any(weights <= 0.0):
        raise ValueError("rank weights must be strictly positive")
    weights /= float(np.sum(weights))
    return weights


def _normal_cdf(values: FloatArray) -> FloatArray:
    flat = np.ravel(values)
    result = np.array(
        [0.5 * (1.0 + math.erf(float(value) / math.sqrt(2.0))) for value in flat],
        dtype=np.float64,
    )
    return result.reshape(values.shape)


def expected_rank_utility(
    standardized_objective: ArrayLike,
    sample_count: int,
    selected_weights: ArrayLike,
) -> FloatArray:
    """Expected assigned rank weight conditional on one sample's objective coordinate."""
    t = np.asarray(standardized_objective, dtype=np.float64)
    weights = _finite_vector(selected_weights, "selected_weights")
    if sample_count < 2 or weights.size > sample_count or np.any(weights <= 0.0):
        raise ValueError("rank utility dimensions are invalid")
    if not np.isclose(np.sum(weights), 1.0, rtol=0.0, atol=2e-14):
        raise ValueError("selected weights must sum to one")
    lower_probability = _normal_cdf(t)
    higher_probability = 1.0 - lower_probability
    other_count = sample_count - 1
    utility = np.zeros_like(t, dtype=np.float64)
    for higher_count, weight in enumerate(weights):
        probability = (
            math.comb(other_count, higher_count)
            * higher_probability**higher_count
            * lower_probability ** (other_count - higher_count)
        )
        utility += weight * probability
    return utility


def analytic_finite_lambda_direction(
    state: GaussianLinearBridgeState,
    *,
    sample_count: int,
    parent_count: int,
    quadrature_order: int = 160,
) -> FloatArray:
    """Construct E[rank-mu tangent] without sampling or replaying the update."""
    if (
        isinstance(quadrature_order, bool)
        or not isinstance(quadrature_order, int)
        or not 32 <= quadrature_order <= 320
    ):
        raise ValueError("quadrature_order must be an integer in [32, 320]")
    _, log_std, objective = state.arrays()
    standard_deviation = np.exp(log_std)
    weights = logarithmic_rank_weights(sample_count, parent_count)
    t, integration_weights = np.polynomial.hermite_e.hermegauss(quadrature_order)
    utility = expected_rank_utility(t, sample_count, weights)
    normalizer = math.sqrt(2.0 * math.pi)
    first_moment = float(np.sum(integration_weights * utility * t) / normalizer)
    second_moment = float(
        np.sum(integration_weights * utility * (t**2 - 1.0)) / normalizer
    )

    standardized_axis = objective * standard_deviation
    standardized_axis /= np.linalg.norm(standardized_axis)
    mean_block = sample_count * standard_deviation * standardized_axis * first_moment
    covariance_block = (
        0.5 * sample_count * standardized_axis**2 * second_moment
    )
    direction = np.concatenate((mean_block, covariance_block))
    if not np.all(np.isfinite(direction)):
        raise FloatingPointError("analytic finite-lambda direction is non-finite")
    return direction


def sample_rank_mu_shadows(
    state: GaussianLinearBridgeState,
    *,
    sample_count: int,
    parent_count: int,
    replications: int,
    rng: np.random.Generator,
    batch_size: int = 4096,
) -> FloatArray:
    """Return independent realized rank-mu tangents; never calls the comparator."""
    if not isinstance(rng, np.random.Generator):
        raise ValueError("an explicit NumPy Generator is required")
    for value, name in ((replications, "replications"), (batch_size, "batch_size")):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    mean, log_std, objective = state.arrays()
    standard_deviation = np.exp(log_std)
    weights = logarithmic_rank_weights(sample_count, parent_count)
    output = np.empty((replications, 2 * mean.size), dtype=np.float64)
    start = 0
    while start < replications:
        count = min(batch_size, replications - start)
        standardized = rng.normal(size=(count, sample_count, mean.size))
        scores = np.sum(
            objective[None, None, :]
            * (mean[None, None, :] + standard_deviation[None, None, :] * standardized),
            axis=2,
        )
        ranks = np.argsort(scores, axis=1)[:, ::-1][:, :parent_count]
        selected = np.take_along_axis(standardized, ranks[:, :, None], axis=1)
        mean_block = np.sum(
            weights[None, :, None] * (standard_deviation[None, None, :] * selected),
            axis=1,
        )
        covariance_block = 0.5 * np.sum(
            weights[None, :, None] * (selected**2 - 1.0), axis=1
        )
        output[start : start + count, : mean.size] = mean_block
        output[start : start + count, mean.size :] = covariance_block
        start += count
    if not np.all(np.isfinite(output)):
        raise FloatingPointError("rank-mu shadow directions are non-finite")
    return output


def _metric_cosine(left: FloatArray, right: FloatArray, metric: FloatArray) -> float:
    left_norm = float(np.sqrt(np.sum(metric * left**2)))
    right_norm = float(np.sqrt(np.sum(metric * right**2)))
    if left_norm == 0.0 or right_norm == 0.0:
        raise ValueError("Fisher cosine requires nonzero directions")
    cosine = float(np.sum(metric * left * right) / (left_norm * right_norm))
    return float(np.clip(cosine, -1.0, 1.0))


def fisher_block_cosines(
    state: GaussianLinearBridgeState, left: ArrayLike, right: ArrayLike
) -> dict[str, float]:
    """Return separate mean and log-scale Fisher cosines."""
    mean, log_std, _ = state.arrays()
    left_vector = _finite_vector(left, "left direction")
    right_vector = _finite_vector(right, "right direction")
    if left_vector.shape != (2 * mean.size,) or right_vector.shape != left_vector.shape:
        raise ValueError("directions must contain matching mean and covariance blocks")
    dimension = mean.size
    mean_metric = np.exp(-2.0 * log_std)
    covariance_metric = np.full(dimension, 2.0)
    return {
        "mean": _metric_cosine(
            left_vector[:dimension], right_vector[:dimension], mean_metric
        ),
        "covariance": _metric_cosine(
            left_vector[dimension:], right_vector[dimension:], covariance_metric
        ),
    }
