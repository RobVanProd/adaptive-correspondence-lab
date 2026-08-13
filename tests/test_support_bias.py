import itertools
import math

import numpy as np

from adaptive_correspondence.support_bias import (
    BlockSpecification,
    exact_block_moments,
    fisher_cosine,
    self_consistency_limits,
    support_pattern_probabilities,
)


def _direct_counts(spec: BlockSpecification, sample_count: int) -> tuple[np.ndarray, float]:
    policy, reward, rho = spec.arrays()
    scores = np.eye(policy.size) - policy[None, :]
    probabilities = np.concatenate((rho * policy, [1.0 - rho]))
    mean = np.zeros(policy.size)
    mass = 0.0
    for counts in itertools.product(range(sample_count + 1), repeat=4):
        if sum(counts) != sample_count:
            continue
        coefficient = math.factorial(sample_count)
        for count in counts:
            coefficient //= math.factorial(count)
        probability = coefficient * math.prod(
            value**count for value, count in zip(probabilities, counts, strict=True)
        )
        action_counts = np.asarray(counts[:3], dtype=np.float64)
        fisher = np.einsum("a,ai,aj->ij", action_counts, scores, scores) / sample_count
        gradient = (action_counts * reward) @ scores / sample_count
        direction = np.linalg.pinv(fisher, rcond=1e-12) @ gradient
        direction -= np.mean(direction)
        mean += probability * direction
        mass += probability
    return mean, mass


def _fixture(reward=(0.9, 0.1, -0.7)) -> BlockSpecification:
    return BlockSpecification(
        policy=(0.62, 0.27, 0.11),
        reward=reward,
        context_probability=0.43,
    )


def test_exact_moments_match_independent_direct_count_enumeration() -> None:
    spec = _fixture()
    direct_mean, direct_mass = _direct_counts(spec, sample_count=5)
    result = exact_block_moments(spec, sample_count=5, enumeration_chunk_size=7)
    np.testing.assert_allclose(direct_mass, 1.0, atol=3e-15)
    np.testing.assert_allclose(result.mean_direction, direct_mean, atol=3e-15, rtol=0.0)
    assert abs(result.probability_mass - 1.0) < 3e-15


def test_support_pattern_probabilities_are_exact_and_match_missing_cell_law() -> None:
    spec = _fixture()
    sample_count = 7
    probabilities = support_pattern_probabilities(spec, sample_count=sample_count)
    assert abs(sum(probabilities.values()) - 1.0) < 3e-15
    policy, _, rho = spec.arrays()
    empty = probabilities[()]
    np.testing.assert_allclose(empty, (1.0 - rho) ** sample_count)
    for action in range(3):
        absent = sum(
            probability
            for support, probability in probabilities.items()
            if action not in support
        )
        np.testing.assert_allclose(
            absent, (1.0 - rho * policy[action]) ** sample_count, atol=3e-15
        )


def test_support_and_observed_perturbation_components_close_exact_bias() -> None:
    result = exact_block_moments(_fixture(), sample_count=8, enumeration_chunk_size=11)
    np.testing.assert_allclose(
        result.mean_direction - result.analytic_direction,
        result.mean_support_loss + result.mean_observed_support_perturbation,
        atol=4e-15,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        result.expected_squared_error,
        result.expected_squared_support_loss
        + result.expected_squared_observed_support_perturbation,
        atol=4e-15,
        rtol=0.0,
    )


def test_reward_shift_preserves_ideal_tangent_but_can_change_plugin_expectation() -> None:
    base = exact_block_moments(_fixture(), sample_count=12)
    shifted = exact_block_moments(
        _fixture(reward=(2.9, 2.1, 1.3)), sample_count=12
    )
    np.testing.assert_allclose(base.analytic_direction, shifted.analytic_direction, atol=2e-15)
    assert np.linalg.norm(base.mean_direction - shifted.mean_direction) > 1e-3
    assert abs(base.truth_alignment_cosine - shifted.truth_alignment_cosine) > 1e-4


def test_self_consistency_limit_does_not_certify_truth_alignment() -> None:
    spec = _fixture(reward=(5.9, 5.1, 4.3))
    result = exact_block_moments(spec, sample_count=8)
    limits = self_consistency_limits(
        spec,
        estimator_mean=result.mean_direction,
        analytic_direction=result.analytic_direction,
    )
    assert limits["split_half_limit"] == 1.0
    assert limits["truth_alignment_limit"] == fisher_cosine(
        spec, result.mean_direction, result.analytic_direction
    )
    assert limits["truth_alignment_limit"] < 0.99


def test_equal_effective_minimum_count_does_not_determine_angular_bias() -> None:
    reward = (0.8, 0.0, -0.8)
    rare_context = BlockSpecification(
        policy=(0.4, 0.35, 0.25), reward=reward, context_probability=0.12
    )
    rare_action = BlockSpecification(
        policy=(0.6, 0.36, 0.04), reward=reward, context_probability=0.75
    )
    assert 0.12 * 0.25 == 0.75 * 0.04
    context_result = exact_block_moments(rare_context, sample_count=16)
    action_result = exact_block_moments(rare_action, sample_count=16)
    assert (
        context_result.truth_alignment_cosine
        - action_result.truth_alignment_cosine
        > 0.2
    )
    assert context_result.rank_deficient_probability > 100 * (
        action_result.rank_deficient_probability
    )
