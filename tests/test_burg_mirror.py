import numpy as np

import adaptive_correspondence.burg_mirror as burg_module
from adaptive_correspondence.burg_mirror import (
    burg_mirror_step,
    burg_mirror_step_polynomial_oracle,
    burg_perturbed_trajectory,
    burg_second_order_sensitivity_trajectory,
    five_point_trajectory_derivatives,
)
from adaptive_correspondence.simplex import exponential_update


def _fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.array([0.46, 0.33, 0.21]),
        np.array([0.8, -0.35, 0.2]),
        np.array([[0.6, 0.25, 0.15], [0.1, 0.75, 0.15], [0.3, 0.2, 0.5]]),
    )


def test_burg_step_is_interior_and_not_entropy_relabeling() -> None:
    state, reward, _ = _fixture()
    burg = burg_mirror_step(state, reward, eta=0.08)
    entropy = exponential_update(state, reward, eta=0.08)
    assert np.all(burg > 0.0)
    np.testing.assert_allclose(np.sum(burg), 1.0, atol=2e-14, rtol=0.0)
    assert np.linalg.norm(burg - entropy) > 1e-4


def test_polynomial_normalizer_oracle_matches_bisection_step() -> None:
    state, reward, _ = _fixture()
    primary = burg_mirror_step(state, reward, eta=0.08)
    oracle = burg_mirror_step_polynomial_oracle(state, reward, eta=0.08)
    np.testing.assert_allclose(primary, oracle, atol=2e-14, rtol=2e-14)


def test_polynomial_oracle_does_not_call_bisection_normalizer(monkeypatch) -> None:
    state, reward, _ = _fixture()

    def forbidden(*args, **kwargs):
        raise AssertionError("polynomial oracle called bisection normalizer")

    monkeypatch.setattr(burg_module, "_dual_shift", forbidden)
    oracle = burg_mirror_step_polynomial_oracle(state, reward, eta=0.08)
    assert np.all(oracle > 0.0)


def test_second_order_recurrence_matches_independent_five_point_oracle() -> None:
    state, reward, mutation = _fixture()
    trace = burg_second_order_sensitivity_trajectory(
        state, reward, mutation, eta=0.06, steps=7
    )
    first, second = five_point_trajectory_derivatives(
        state, reward, mutation, eta=0.06, steps=7, step=5e-4
    )
    np.testing.assert_allclose(trace.first, first, atol=2e-10, rtol=2e-9)
    np.testing.assert_allclose(trace.second, second, atol=2e-8, rtol=2e-7)


def test_identity_mutation_has_zero_sensitivity() -> None:
    state, reward, _ = _fixture()
    trace = burg_second_order_sensitivity_trajectory(
        state, reward, np.eye(3), eta=0.05, steps=12
    )
    np.testing.assert_array_equal(trace.first, np.zeros_like(trace.first))
    np.testing.assert_array_equal(trace.second, np.zeros_like(trace.second))


def test_direct_signed_trajectory_preserves_simplex() -> None:
    state, reward, mutation = _fixture()
    trajectory = burg_perturbed_trajectory(
        state, reward, mutation, eta=0.05, epsilon=-1e-3, steps=10
    )
    assert np.all(trajectory > 0.0)
    np.testing.assert_allclose(np.sum(trajectory, axis=1), 1.0, atol=2e-14, rtol=0.0)
