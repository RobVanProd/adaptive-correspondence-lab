import numpy as np
import pytest

from adaptive_correspondence.simplex import (
    centered_logits,
    exponential_update,
    kl_divergence,
    project_simplex_with_floor,
    softmax,
    validate_simplex,
)


def test_centered_logits_round_trip() -> None:
    probability = np.array([0.2, 0.3, 0.5])
    np.testing.assert_allclose(softmax(centered_logits(probability)), probability, atol=2e-16)


@pytest.mark.parametrize(
    "invalid",
    [
        [0.2, 0.2],
        [-0.1, 1.1],
        [0.5, np.nan, 0.5],
        [[0.5, 0.5]],
    ],
)
def test_invalid_simplex_stops(invalid: list[float]) -> None:
    with pytest.raises(ValueError):
        validate_simplex(invalid)


def test_exponential_update_is_shift_invariant() -> None:
    state = [0.2, 0.3, 0.5]
    reward = np.array([1000.0, 999.0, 998.0])
    shifted = reward - 50_000.0
    np.testing.assert_allclose(
        exponential_update(state, reward, 0.2),
        exponential_update(state, shifted, 0.2),
        atol=2e-14,
        rtol=0.0,
    )


def test_floor_projection_satisfies_both_constraints() -> None:
    projected = project_simplex_with_floor([1.2, -0.1, -0.1], 0.05)
    assert np.min(projected) >= 0.05
    assert np.sum(projected) == pytest.approx(1.0, abs=5e-14)


def test_kl_rejects_missing_right_support() -> None:
    with pytest.raises(ValueError, match="zero mass"):
        kl_divergence([0.5, 0.5], [1.0, 0.0])
