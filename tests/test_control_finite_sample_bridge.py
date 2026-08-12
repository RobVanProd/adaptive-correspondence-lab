import numpy as np
import pytest

from adaptive_correspondence.control_finite_sample_bridge import (
    ControlBridgeState,
    context_fisher_cosines,
    exact_natural_direction,
    plugin_direction_from_counts,
    sample_plugin_npg_shadows,
)


def _state() -> ControlBridgeState:
    return ControlBridgeState(
        rewards=((0.8, 0.1, -0.3), (-0.2, 0.5, 1.0)),
        context_probabilities=(0.6, 0.4),
        logits=((0.2, -0.1, -0.1), (-0.3, 0.1, 0.2)),
    )


def test_fixed_counts_match_direct_score_and_fisher_accumulation() -> None:
    state = _state()
    counts = np.array([[11, 5, 3], [2, 7, 12]])
    direction = plugin_direction_from_counts(state, counts, rcond=1e-12)
    rewards, contexts, _logits, policy = state.arrays()
    direct = np.zeros_like(direction)
    total = int(np.sum(counts))
    for context in range(contexts.size):
        gradient = np.zeros(policy.shape[1])
        fisher = np.zeros((policy.shape[1], policy.shape[1]))
        for action in range(policy.shape[1]):
            score = -policy[context].copy()
            score[action] += 1.0
            gradient += counts[context, action] * rewards[context, action] * score / total
            fisher += counts[context, action] * np.outer(score, score) / total
        direct[context] = np.linalg.pinv(fisher, rcond=1e-12) @ gradient
        direct[context] -= np.mean(direct[context])
    np.testing.assert_allclose(direction, direct, atol=2e-15)


def test_analytic_fisher_maps_exact_direction_to_gradient_per_context() -> None:
    state = _state()
    rewards, contexts, _logits, policy = state.arrays()
    direction = exact_natural_direction(state)
    values = np.sum(policy * rewards, axis=1, keepdims=True)
    gradient = contexts[:, None] * policy * (rewards - values)
    for context in range(contexts.size):
        fisher = contexts[context] * (
            np.diag(policy[context]) - np.outer(policy[context], policy[context])
        )
        np.testing.assert_allclose(fisher @ direction[context], gradient[context], atol=3e-16)


def test_many_shadows_align_with_exact_context_blocks() -> None:
    state = _state()
    exact = exact_natural_direction(state)
    shadows = sample_plugin_npg_shadows(
        state,
        sample_count=128,
        replications=20_000,
        rng=np.random.Generator(np.random.PCG64(7183)),
        batch_size=2_000,
        rcond=1e-12,
    )
    cosines = context_fisher_cosines(state, np.mean(shadows, axis=0), exact)
    assert min(cosines) > 0.999


def test_missing_context_is_explicit_zero_block() -> None:
    direction = plugin_direction_from_counts(
        _state(), np.array([[12, 8, 4], [0, 0, 0]]), rcond=1e-12
    )
    np.testing.assert_array_equal(direction[1], np.zeros(3))


@pytest.mark.parametrize(
    "counts",
    [
        [[1, 2], [3, 4]],
        [[1, 2, -1], [3, 4, 5]],
        [[0, 0, 0], [0, 0, 0]],
    ],
)
def test_invalid_counts_fail(counts) -> None:
    with pytest.raises(ValueError):
        plugin_direction_from_counts(_state(), counts, rcond=1e-12)
