import numpy as np
import pytest

from adaptive_correspondence.acl002_second_order_analysis import (
    evaluate_acl003_earning_rule,
    prediction_radius_comparison,
)
from adaptive_correspondence.categorical_second_order import (
    l1_second_order_coefficient,
    l1_truncated_prediction,
    matrix_polynomial_second_order_trajectory,
    row_hessian_quadratic,
    second_order_sensitivity_trajectory,
    signed_matrix_power_state,
)

P0 = np.array([0.2, 0.3, 0.5])
REWARD = np.array([0.7, -0.2, 0.1])
MUTATION = np.array(
    [
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0],
    ]
)


def test_row_hessian_quadratic_matches_centered_finite_difference() -> None:
    probability = np.array([0.25, 0.35, 0.40])
    direction = np.array([0.04, -0.03, -0.01])
    step = 2e-4

    from adaptive_correspondence.acl002 import categorical_map

    observed = (
        categorical_map(probability + step * direction, REWARD, 0.05)
        - 2.0 * categorical_map(probability, REWARD, 0.05)
        + categorical_map(probability - step * direction, REWARD, 0.05)
    ) / step**2

    analytic = row_hessian_quadratic(probability, REWARD, 0.05, direction)

    assert np.allclose(analytic, observed, rtol=0.0, atol=2e-8)
    assert abs(float(np.sum(analytic))) < 2e-13


@pytest.mark.parametrize("steps", [1, 2, 5, 20])
def test_recurrence_matches_independent_matrix_polynomial_oracle(steps: int) -> None:
    recurrence = second_order_sensitivity_trajectory(
        P0, REWARD, MUTATION, eta=0.05, steps=steps
    )
    oracle = matrix_polynomial_second_order_trajectory(
        P0, REWARD, MUTATION, eta=0.05, steps=steps
    )

    assert np.allclose(recurrence.states, oracle.states, rtol=0.0, atol=5e-14)
    assert np.allclose(recurrence.first, oracle.first, rtol=0.0, atol=5e-13)
    assert np.allclose(recurrence.second, oracle.second, rtol=0.0, atol=2e-11)
    assert np.max(np.abs(np.sum(recurrence.first, axis=1))) < 2e-13
    assert np.max(np.abs(np.sum(recurrence.second, axis=1))) < 2e-11


def test_five_point_signed_epsilon_difference_recovers_derivatives() -> None:
    trace = second_order_sensitivity_trajectory(P0, REWARD, MUTATION, eta=0.05, steps=5)
    step = 8e-4
    values = {
        multiplier: signed_matrix_power_state(
            P0,
            REWARD,
            MUTATION,
            eta=0.05,
            epsilon=multiplier * step,
            steps=5,
        )
        for multiplier in (-2, -1, 0, 1, 2)
    }
    first = (
        values[-2] - 8.0 * values[-1] + 8.0 * values[1] - values[2]
    ) / (12.0 * step)
    second = (
        -values[2]
        + 16.0 * values[1]
        - 30.0 * values[0]
        + 16.0 * values[-1]
        - values[-2]
    ) / (12.0 * step**2)

    assert np.allclose(first, trace.first[5], rtol=0.0, atol=2e-10)
    assert np.allclose(second, trace.second[5], rtol=0.0, atol=2e-8)


def test_l1_quadratic_coefficient_handles_zero_first_derivative() -> None:
    first = np.array([0.2, -0.2, 0.0])
    second = np.array([0.3, 0.1, -0.4])

    coefficient, zero_coordinates = l1_second_order_coefficient(first, second)

    assert zero_coordinates == (2,)
    assert coefficient == pytest.approx(0.3)
    assert l1_truncated_prediction(first, second, epsilon=0.01) == pytest.approx(
        0.004 + 0.3 * 0.01**2
    )


def test_one_step_l1_second_order_coefficient_is_zero() -> None:
    trace = second_order_sensitivity_trajectory(P0, REWARD, MUTATION, eta=0.05, steps=1)

    coefficient, zero_coordinates = l1_second_order_coefficient(
        trace.first[1], trace.second[1]
    )

    assert zero_coordinates == ()
    assert coefficient == pytest.approx(0.0, abs=2e-14)


def test_high_curvature_clean_case_uses_scale_aware_mass_guard() -> None:
    p0 = np.array([0.001, 0.099, 0.9])
    reward = np.array([1.5, 0.25, -0.4])
    mutation = np.full((3, 3), 1.0 / 3.0)

    recurrence = second_order_sensitivity_trajectory(
        p0, reward, mutation, eta=0.05, steps=50
    )
    oracle = matrix_polynomial_second_order_trajectory(
        p0, reward, mutation, eta=0.05, steps=50
    )

    assert np.allclose(recurrence.first, oracle.first, rtol=0.0, atol=5e-11)
    assert np.allclose(recurrence.second, oracle.second, rtol=0.0, atol=2e-9)


@pytest.mark.parametrize("epsilon", [float("nan"), float("inf")])
def test_signed_matrix_oracle_rejects_nonfinite_epsilon(epsilon: float) -> None:
    with pytest.raises(ValueError, match="epsilon"):
        signed_matrix_power_state(
            P0, REWARD, MUTATION, eta=0.05, epsilon=epsilon, steps=2
        )


def _prediction_row(
    landscape_id: str,
    epsilon: float,
    first_error: float,
    second_error: float,
) -> dict:
    return {
        "landscape_id": landscape_id,
        "split": "target",
        "stratum": "regular-sensitivity",
        "horizon": 20,
        "epsilon": epsilon,
        "region": (
            "confirmatory"
            if epsilon <= 0.001
            else "extended-local"
            if epsilon <= 0.01
            else "stress"
        ),
        "first_order_absolute_relative_error": first_error,
        "second_order_absolute_relative_error": second_error,
        "oracle_passed": True,
    }


def test_prediction_radius_comparison_uses_contiguous_prefixes() -> None:
    rows = [
        _prediction_row("T01", 1e-4, 0.01, 0.005),
        _prediction_row("T01", 3e-4, 0.06, 0.01),
        _prediction_row("T01", 1e-3, 0.02, 0.02),
        _prediction_row("T01", 3e-3, 0.03, 0.03),
    ]

    comparison = prediction_radius_comparison(rows, levels=(0.05,))

    assert comparison == [
        {
            "landscape_id": "T01",
            "split": "target",
            "horizon": 20,
            "relative_error_level": 0.05,
            "first_order_radius": 1e-4,
            "second_order_radius": 3e-3,
            "first_order_radius_index": 0,
            "second_order_radius_index": 3,
            "radius_index_improvement": 3,
        }
    ]


def test_acl003_earning_rule_requires_every_frozen_check() -> None:
    epsilons = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
    rows = []
    for index in range(12):
        for epsilon in epsilons:
            first_error = 0.01 if epsilon <= 1e-3 else 0.12 if epsilon <= 1e-2 else 0.3
            second_error = 0.005 if epsilon <= 1e-2 else 0.15
            rows.append(_prediction_row(f"T{index + 1:02d}", epsilon, first_error, second_error))

    result = evaluate_acl003_earning_rule(rows)

    assert result["passed"] is True
    assert all(check["passed"] for check in result["checks"].values())

    rows[0]["oracle_passed"] = False
    assert evaluate_acl003_earning_rule(rows)["passed"] is False
