import math
from itertools import product

import numpy as np
import pytest

from adaptive_correspondence.gaussian_rank_mu_bridge import (
    GaussianLinearBridgeState,
    analytic_finite_lambda_direction,
    expected_rank_utility,
    fisher_block_cosines,
    logarithmic_rank_weights,
    sample_rank_mu_shadows,
)


def test_log_rank_weights_are_positive_and_normalized() -> None:
    weights = logarithmic_rank_weights(sample_count=32, parent_count=16)
    assert weights.shape == (16,)
    assert np.all(weights > 0.0)
    assert np.sum(weights) == pytest.approx(1.0)
    assert np.all(np.diff(weights) < 0.0)


def test_expected_rank_utility_matches_rank_extremes() -> None:
    weights = logarithmic_rank_weights(8, 4)
    best = expected_rank_utility(np.array([12.0]), 8, weights)[0]
    worst = expected_rank_utility(np.array([-12.0]), 8, weights)[0]
    assert best == pytest.approx(weights[0], rel=0.0, abs=1e-14)
    assert worst == pytest.approx(0.0, rel=0.0, abs=1e-14)


def test_expected_rank_utility_matches_conditional_rank_monte_carlo() -> None:
    sample_count = 8
    weights = logarithmic_rank_weights(sample_count, 4)
    fixed_t = 0.37
    predicted = expected_rank_utility(np.array([fixed_t]), sample_count, weights)[0]
    rng = np.random.Generator(np.random.PCG64(9931))
    others = rng.normal(size=(200_000, sample_count - 1))
    better_counts = np.sum(others > fixed_t, axis=1)
    realized = np.where(better_counts < weights.size, weights[better_counts.clip(max=3)], 0.0)
    assert float(np.mean(realized)) == pytest.approx(predicted, abs=5e-4)


def test_closed_comparator_matches_independent_tensor_score_integral() -> None:
    state = GaussianLinearBridgeState(
        mean=(0.4, -0.7), log_std=np.log((0.6, 1.4)), objective=(1.2, -0.5)
    )
    sample_count = 8
    parent_count = 4
    weights = logarithmic_rank_weights(sample_count, parent_count)
    analytic = analytic_finite_lambda_direction(
        state, sample_count=sample_count, parent_count=parent_count, quadrature_order=160
    )

    nodes, node_weights = np.polynomial.hermite_e.hermegauss(36)
    total = np.zeros(4, dtype=np.float64)
    _mean, log_std, objective = state.arrays()
    standard_deviation = np.exp(log_std)
    axis = objective * standard_deviation
    axis /= np.linalg.norm(axis)
    for i, j in product(range(nodes.size), repeat=2):
        z = np.array([nodes[i], nodes[j]])
        t = float(axis @ z)
        utility = expected_rank_utility(np.array([t]), sample_count, weights)[0]
        natural_score = np.concatenate((standard_deviation * z, 0.5 * (z**2 - 1.0)))
        total += node_weights[i] * node_weights[j] * utility * natural_score
    oracle = sample_count * total / (2.0 * math.pi)

    np.testing.assert_allclose(analytic, oracle, rtol=2e-11, atol=2e-12)


def test_many_shadows_align_with_separate_analytic_blocks() -> None:
    state = GaussianLinearBridgeState(
        mean=(0.2, -0.4, 0.7),
        log_std=np.log((0.5, 1.1, 1.7)),
        objective=(1.0, -0.8, 0.35),
    )
    analytic = analytic_finite_lambda_direction(state, sample_count=32, parent_count=16)
    shadows = sample_rank_mu_shadows(
        state,
        sample_count=32,
        parent_count=16,
        replications=20_000,
        rng=np.random.Generator(np.random.PCG64(8041)),
        batch_size=2_000,
    )
    cosines = fisher_block_cosines(state, np.mean(shadows, axis=0), analytic)
    assert cosines["mean"] > 0.999
    assert cosines["covariance"] > 0.995


def test_analytic_comparator_converges_at_doubled_quadrature_order() -> None:
    state = GaussianLinearBridgeState(
        mean=(0.2, -0.4, 0.7),
        log_std=np.log((0.5, 1.1, 1.7)),
        objective=(1.0, -0.8, 0.35),
    )
    base = analytic_finite_lambda_direction(
        state, sample_count=32, parent_count=16, quadrature_order=160
    )
    doubled = analytic_finite_lambda_direction(
        state, sample_count=32, parent_count=16, quadrature_order=320
    )
    np.testing.assert_allclose(base, doubled, rtol=2e-9, atol=5e-12)


@pytest.mark.parametrize(
    "state",
    [
        GaussianLinearBridgeState((0.0,), (0.0,), (1.0,)),
        GaussianLinearBridgeState((0.0, 0.0), (0.0,), (1.0, 1.0)),
        GaussianLinearBridgeState((0.0, 0.0), (0.0, 1000.0), (1.0, 1.0)),
        GaussianLinearBridgeState((0.0, 0.0), (0.0, 0.0), (0.0, 0.0)),
    ],
)
def test_invalid_bridge_states_fail(state: GaussianLinearBridgeState) -> None:
    with pytest.raises(ValueError):
        state.arrays()
