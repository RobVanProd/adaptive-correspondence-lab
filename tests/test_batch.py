import numpy as np
import pytest

from adaptive_correspondence.batch import (
    euler_step,
    exact_step,
    natural_gradient_step,
    run_terminal_states,
)
from adaptive_correspondence.categorical import MultiplicativeWeights
from adaptive_correspondence.experiments import (
    CategoricalExperimentConfig,
    run_categorical_trajectory,
)


def test_vectorized_steps_agree() -> None:
    states = np.array([[0.2, 0.3, 0.5], [0.1, 0.7, 0.2]])
    rewards = np.array([0.7, -0.2, 0.1])
    np.testing.assert_allclose(
        exact_step(states, rewards, 0.05),
        natural_gradient_step(states, rewards, 0.05),
        atol=2e-16,
        rtol=0.0,
    )


def test_chunked_terminal_states_match_reference() -> None:
    config = CategoricalExperimentConfig(steps=17)
    reference = run_categorical_trajectory(MultiplicativeWeights(), config).terminal_state
    states = np.repeat(np.asarray(config.initial_state)[None, :], 11, axis=0)
    schedule = np.repeat(np.asarray(config.reward)[None, :], config.steps, axis=0)
    accelerated = run_terminal_states(
        states,
        schedule,
        config.eta,
        steps=config.steps,
        chunk_size=3,
    )
    expected = np.repeat(reference[None, :], states.shape[0], axis=0)
    np.testing.assert_allclose(accelerated, expected, atol=2e-15, rtol=0.0)


def test_batch_rejects_non_simplex_rows() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        exact_step([[0.4, 0.4]], [1.0, 0.0], 0.1)


def test_batch_euler_rejects_negative_candidate() -> None:
    with pytest.raises(ValueError, match="left the simplex"):
        euler_step([[0.2, 0.3, 0.5]], [0.7, -0.2, 0.1], 20.0)
