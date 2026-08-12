from itertools import pairwise

import numpy as np

from adaptive_correspondence.bandit import (
    BanditPolicyState,
    ContextualBandit,
    NaturalPolicyGradient,
    run_bandit_trajectory,
)


def _bandit() -> ContextualBandit:
    return ContextualBandit(
        rewards=((0.8, 0.1, -0.2), (-0.1, 0.4, 0.9)),
        context_probabilities=(0.6, 0.4),
    )


def test_analytic_fisher_maps_natural_direction_to_policy_gradient() -> None:
    optimizer = NaturalPolicyGradient(_bandit())
    state = BanditPolicyState(np.zeros((2, 3)))
    natural = optimizer.exact_natural_direction(state).ravel()
    gradient = optimizer.euclidean_gradient(state).ravel()
    np.testing.assert_allclose(optimizer.fisher(state) @ natural, gradient, atol=2e-16)


def test_exact_npg_improves_expected_return() -> None:
    trajectory = run_bandit_trajectory(
        bandit=_bandit(),
        initial_logits=np.zeros((2, 3)),
        eta=0.1,
        steps=12,
    )
    returns = [record.potentials["expected_return_after"] for record in trajectory.records]
    assert all(right > left for left, right in pairwise(returns))


def test_sampled_npg_replays_exactly() -> None:
    kwargs = {
        "bandit": _bandit(),
        "initial_logits": np.zeros((2, 3)),
        "eta": 0.03,
        "steps": 3,
        "mode": "sampled",
        "seed": 2026,
        "sample_count": 80,
    }
    left = run_bandit_trajectory(**kwargs)
    right = run_bandit_trajectory(**kwargs)
    assert left.to_dict() == right.to_dict()
    assert left.records[0].rng_fingerprint != "none"
