from itertools import pairwise

import numpy as np

from adaptive_correspondence.gaussian import (
    DiagonalGaussianNaturalGradient,
    DiagonalGaussianState,
    DiagonalQuadraticObjective,
    run_gaussian_trajectory,
)


def _optimizer() -> DiagonalGaussianNaturalGradient:
    return DiagonalGaussianNaturalGradient(
        DiagonalQuadraticObjective(target=(0.0, 0.5), curvature=(1.0, 2.0))
    )


def test_analytic_gradient_matches_finite_difference() -> None:
    optimizer = _optimizer()
    state = DiagonalGaussianState(mean=(1.2, -0.3), log_std=np.log((0.8, 1.1)))
    canonical = state.canonical()
    numerical = np.empty_like(canonical)
    step = 1e-6
    for index in range(canonical.size):
        plus = canonical.copy()
        minus = canonical.copy()
        plus[index] += step
        minus[index] -= step
        dimension = canonical.size // 2
        plus_state = DiagonalGaussianState(plus[:dimension], plus[dimension:])
        minus_state = DiagonalGaussianState(minus[:dimension], minus[dimension:])
        numerical[index] = (
            optimizer.expected_objective(plus_state) - optimizer.expected_objective(minus_state)
        ) / (2.0 * step)
    np.testing.assert_allclose(optimizer.euclidean_gradient(state), numerical, atol=4e-10)


def test_fisher_maps_natural_direction_to_gradient() -> None:
    optimizer = _optimizer()
    state = DiagonalGaussianState(mean=(1.2, -0.3), log_std=np.log((0.8, 1.1)))
    np.testing.assert_allclose(
        optimizer.fisher(state) @ optimizer.natural_direction(state),
        optimizer.euclidean_gradient(state),
        atol=2e-16,
    )


def test_analytic_trajectory_improves_expected_objective() -> None:
    trajectory = run_gaussian_trajectory(
        initial_state=DiagonalGaussianState((1.5, -1.0), np.log((0.8, 1.2))),
        objective=DiagonalQuadraticObjective((0.0, 0.5), (1.0, 2.0)),
        eta=0.05,
        steps=10,
    )
    objectives = [record.potentials["expected_objective_after"] for record in trajectory.records]
    assert all(right > left for left, right in pairwise(objectives))


def test_rank_mu_trajectory_replays_exactly() -> None:
    kwargs = {
        "initial_state": DiagonalGaussianState((1.5, -1.0), np.log((0.8, 1.2))),
        "objective": DiagonalQuadraticObjective((0.0, 0.5), (1.0, 2.0)),
        "eta": 0.03,
        "steps": 4,
        "mode": "rank-mu",
        "seed": 91,
        "sample_count": 20,
    }
    left = run_gaussian_trajectory(**kwargs)
    right = run_gaussian_trajectory(**kwargs)
    assert left.to_dict() == right.to_dict()
    assert "no-clamps" in left.records[0].numerical_guards
