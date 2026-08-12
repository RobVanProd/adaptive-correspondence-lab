import numpy as np
import pytest

from adaptive_correspondence.categorical import (
    CategoricalEffects,
    CategoricalNaturalGradient,
    MultiplicativeWeights,
    ReplicatorDynamics,
)
from adaptive_correspondence.perturbations import off_diagonal_mutation
from adaptive_correspondence.simplex import replicator_field

STATE = np.array([0.2, 0.3, 0.5])
REWARD = np.array([0.7, -0.2, 0.1])


def test_three_worlds_expose_same_expected_vector_field() -> None:
    expected = replicator_field(STATE, REWARD)
    worlds = [ReplicatorDynamics(), MultiplicativeWeights(), CategoricalNaturalGradient()]
    for world in worlds:
        np.testing.assert_allclose(world.expected_vector_field(STATE, REWARD), expected, atol=0.0)


def test_exact_three_way_next_state_equivalence() -> None:
    worlds = [ReplicatorDynamics(), MultiplicativeWeights(), CategoricalNaturalGradient()]
    states = [
        np.asarray(world.step(STATE, REWARD, 0.07, step_index=0).canonical_state_after)
        for world in worlds
    ]
    np.testing.assert_allclose(states[0], states[1], atol=0.0, rtol=0.0)
    np.testing.assert_allclose(states[0], states[2], atol=2e-16, rtol=0.0)


def test_record_contains_common_instrumentation() -> None:
    record = MultiplicativeWeights().step(STATE, REWARD, 0.05, step_index=4)
    payload = record.to_dict()
    assert payload["step"] == 4
    assert len(payload["geometry"]["fisher"]) == 3
    assert payload["geometry"]["bregman_generator"] == "negative-entropy"
    assert payload["constraint_violations"]["violated"] is False
    assert payload["rng_fingerprint"] == "none"
    assert set(payload) == {
        "domain",
        "step",
        "canonical_state_before",
        "canonical_state_after",
        "native_state_before",
        "native_state_after",
        "expected_vector_field",
        "analytic_gradient",
        "geometry",
        "realized_update",
        "step_size",
        "stochastic_error",
        "potentials",
        "regret",
        "constraint_violations",
        "numerical_guards",
        "rng_fingerprint",
        "rng_state",
    }


def test_euler_rejects_step_that_leaves_simplex() -> None:
    with pytest.raises(ValueError, match="negative probability"):
        ReplicatorDynamics("euler").step(STATE, REWARD, 20.0, step_index=0)


def test_mutation_is_explicit_post_update() -> None:
    effects = CategoricalEffects(
        mutation_rate=0.1,
        mutation_matrix=off_diagonal_mutation(3),
    )
    unmutated = MultiplicativeWeights().step(STATE, REWARD, 0.05, step_index=0)
    mutated = MultiplicativeWeights().step(
        STATE,
        REWARD,
        0.05,
        step_index=0,
        effects=effects,
    )
    expected = np.asarray(unmutated.canonical_state_after)
    expected = 0.9 * expected + 0.1 * (expected @ off_diagonal_mutation(3))
    np.testing.assert_allclose(mutated.canonical_state_after, expected, atol=2e-16)
    assert "row-stochastic-mutation-kernel" in mutated.numerical_guards


def test_finite_population_replays_from_seed() -> None:
    effects = CategoricalEffects(finite_population=37)
    left_rng = np.random.Generator(np.random.PCG64(99))
    right_rng = np.random.Generator(np.random.PCG64(99))
    left = ReplicatorDynamics().step(
        STATE, REWARD, 0.05, step_index=0, effects=effects, rng=left_rng
    )
    right = ReplicatorDynamics().step(
        STATE, REWARD, 0.05, step_index=0, effects=effects, rng=right_rng
    )
    assert left.to_dict() == right.to_dict()
    assert left.rng_fingerprint != "none"
    assert left.rng_state is not None


def test_finite_population_rejects_non_integer_size() -> None:
    effects = CategoricalEffects(finite_population=3.5)
    with pytest.raises(ValueError, match="positive integer"):
        MultiplicativeWeights().step(STATE, REWARD, 0.05, step_index=0, effects=effects)


def test_natural_gradient_rejects_boundary_native_state() -> None:
    with pytest.raises(ValueError, match="interior"):
        CategoricalNaturalGradient().step([0.0, 0.5, 0.5], REWARD, 0.1, step_index=0)
