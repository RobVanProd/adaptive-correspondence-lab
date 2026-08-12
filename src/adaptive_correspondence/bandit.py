"""Tiny contextual-bandit natural policy gradient with analytic Fisher geometry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .schema import StepRecord, Trajectory, rng_fingerprint, rng_snapshot

FloatArray = NDArray[np.float64]


def _softmax_rows(logits: FloatArray) -> FloatArray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    weights = np.exp(shifted)
    return weights / np.sum(weights, axis=1, keepdims=True)


@dataclass(frozen=True)
class ContextualBandit:
    rewards: ArrayLike
    context_probabilities: ArrayLike

    def arrays(self) -> tuple[FloatArray, FloatArray]:
        rewards = np.asarray(self.rewards, dtype=np.float64)
        contexts = np.asarray(self.context_probabilities, dtype=np.float64)
        if rewards.ndim != 2 or min(rewards.shape) < 1 or not np.all(np.isfinite(rewards)):
            raise ValueError("rewards must be a finite context-by-action matrix")
        if contexts.shape != (rewards.shape[0],) or not np.all(np.isfinite(contexts)):
            raise ValueError("context probabilities must match the number of contexts")
        if np.any(contexts <= 0.0) or not np.isclose(
            float(np.sum(contexts)), 1.0, rtol=0.0, atol=1e-12
        ):
            raise ValueError("context probabilities must be strictly positive and sum to one")
        return rewards.copy(), contexts.copy()


@dataclass(frozen=True)
class BanditPolicyState:
    logits: ArrayLike

    def array(self, shape: tuple[int, int]) -> FloatArray:
        logits = np.asarray(self.logits, dtype=np.float64)
        if logits.shape != shape or not np.all(np.isfinite(logits)):
            raise ValueError("policy logits must be finite and match bandit shape")
        return logits - np.mean(logits, axis=1, keepdims=True)


class NaturalPolicyGradient:
    name = "contextual-bandit-natural-policy-gradient"

    def __init__(self, bandit: ContextualBandit) -> None:
        self.bandit = bandit
        self.rewards, self.context_probabilities = bandit.arrays()

    def policy(self, state: BanditPolicyState) -> FloatArray:
        return _softmax_rows(state.array(self.rewards.shape))

    def expected_return(self, state: BanditPolicyState) -> float:
        policy = self.policy(state)
        return float(np.sum(self.context_probabilities[:, None] * policy * self.rewards))

    def advantage(self, state: BanditPolicyState) -> FloatArray:
        policy = self.policy(state)
        value = np.sum(policy * self.rewards, axis=1, keepdims=True)
        return self.rewards - value

    def euclidean_gradient(self, state: BanditPolicyState) -> FloatArray:
        policy = self.policy(state)
        return self.context_probabilities[:, None] * policy * self.advantage(state)

    def fisher(self, state: BanditPolicyState) -> FloatArray:
        policy = self.policy(state)
        contexts, actions = policy.shape
        matrix = np.zeros((contexts * actions, contexts * actions), dtype=np.float64)
        for context in range(contexts):
            block = self.context_probabilities[context] * (
                np.diag(policy[context]) - np.outer(policy[context], policy[context])
            )
            start = context * actions
            matrix[start : start + actions, start : start + actions] = block
        return matrix

    def exact_natural_direction(self, state: BanditPolicyState) -> FloatArray:
        advantage = self.advantage(state)
        return advantage - np.mean(advantage, axis=1, keepdims=True)

    def _sampled_direction(
        self,
        state: BanditPolicyState,
        rng: np.random.Generator,
        sample_count: int,
    ) -> FloatArray:
        if isinstance(sample_count, bool) or sample_count < 1:
            raise ValueError("sample_count must be a positive integer")
        policy = self.policy(state)
        contexts, actions = policy.shape
        dimension = contexts * actions
        gradient = np.zeros(dimension, dtype=np.float64)
        fisher = np.zeros((dimension, dimension), dtype=np.float64)
        sampled_contexts = rng.choice(contexts, size=sample_count, p=self.context_probabilities)
        for context in sampled_contexts:
            action = int(rng.choice(actions, p=policy[context]))
            score = np.zeros((contexts, actions), dtype=np.float64)
            score[context] = -policy[context]
            score[context, action] += 1.0
            flat_score = score.ravel()
            gradient += self.rewards[context, action] * flat_score
            fisher += np.outer(flat_score, flat_score)
        gradient /= sample_count
        fisher /= sample_count
        direction = np.linalg.pinv(fisher, rcond=1e-12) @ gradient
        direction = direction.reshape(contexts, actions)
        return direction - np.mean(direction, axis=1, keepdims=True)

    def step(
        self,
        state: BanditPolicyState,
        eta: float,
        *,
        step_index: int,
        mode: Literal["exact", "sampled"] = "exact",
        rng: np.random.Generator | None = None,
        sample_count: int = 128,
        cumulative_regret_before: float = 0.0,
    ) -> StepRecord:
        logits = state.array(self.rewards.shape)
        if not np.isfinite(eta) or eta < 0.0:
            raise ValueError("eta must be finite and non-negative")
        exact_direction = self.exact_natural_direction(state)
        fingerprint = rng_fingerprint(rng)
        generator_state = rng_snapshot(rng)
        if mode == "exact":
            direction = exact_direction
        elif mode == "sampled":
            if rng is None:
                raise ValueError("sampled mode requires an explicit RNG")
            direction = self._sampled_direction(state, rng, sample_count)
        else:
            raise ValueError("mode must be 'exact' or 'sampled'")
        updated_logits = logits + eta * direction
        updated_logits -= np.mean(updated_logits, axis=1, keepdims=True)
        updated = BanditPolicyState(updated_logits)
        expected_updated_logits = logits + eta * exact_direction
        expected_updated_logits -= np.mean(expected_updated_logits, axis=1, keepdims=True)
        expected_updated = BanditPolicyState(expected_updated_logits)
        policy_before = self.policy(state)
        policy_after = self.policy(updated)
        expected_policy_after = self.policy(expected_updated)
        return_before = self.expected_return(state)
        return_after = self.expected_return(updated)
        optimal = float(np.sum(self.context_probabilities * np.max(self.rewards, axis=1)))
        instant_regret = optimal - return_before
        fisher = self.fisher(state)
        return StepRecord(
            domain=f"{self.name}-{mode}",
            step=step_index,
            canonical_state_before=policy_before.ravel().tolist(),
            canonical_state_after=policy_after.ravel().tolist(),
            native_state_before={"centered_logits": logits.tolist()},
            native_state_after={"centered_logits": updated_logits.tolist()},
            expected_vector_field=(policy_before * self.advantage(state)).ravel().tolist(),
            analytic_gradient=self.euclidean_gradient(state).ravel().tolist(),
            geometry={
                "fisher": fisher.tolist(),
                "fisher_rank": int(np.linalg.matrix_rank(fisher, tol=1e-12)),
                "context_probabilities": self.context_probabilities.tolist(),
            },
            realized_update=(policy_after - policy_before).ravel().tolist(),
            step_size=float(eta),
            stochastic_error=(policy_after - expected_policy_after).ravel().tolist(),
            potentials={
                "expected_return_before": return_before,
                "expected_return_after": return_after,
            },
            regret={
                "instantaneous": instant_regret,
                "cumulative": float(cumulative_regret_before + instant_regret),
            },
            constraint_violations={
                "maximum_row_sum_error": float(np.max(np.abs(np.sum(policy_after, axis=1) - 1.0))),
                "minimum_probability": float(np.min(policy_after)),
                "violated": False,
            },
            numerical_guards=[
                "centered-logit-gauge",
                "max-shifted-row-softmax",
                "explicit-moore-penrose-pseudoinverse" if mode == "sampled" else "analytic-fisher",
                "no-damping",
            ],
            rng_fingerprint=fingerprint,
            rng_state=generator_state,
        )


def run_bandit_trajectory(
    *,
    bandit: ContextualBandit,
    initial_logits: ArrayLike,
    eta: float,
    steps: int,
    mode: Literal["exact", "sampled"] = "exact",
    seed: int = 1729,
    sample_count: int = 128,
) -> Trajectory:
    if isinstance(steps, bool) or steps < 0:
        raise ValueError("steps must be a non-negative integer")
    optimizer = NaturalPolicyGradient(bandit)
    state = BanditPolicyState(initial_logits)
    rng = np.random.Generator(np.random.PCG64(seed))
    records = []
    cumulative_regret = 0.0
    for step in range(steps):
        record = optimizer.step(
            state,
            eta,
            step_index=step,
            mode=mode,
            rng=rng if mode == "sampled" else None,
            sample_count=sample_count,
            cumulative_regret_before=cumulative_regret,
        )
        records.append(record)
        cumulative_regret = record.regret["cumulative"]
        state = BanditPolicyState(record.native_state_after["centered_logits"])
    rewards, contexts = bandit.arrays()
    initial_policy = optimizer.policy(BanditPolicyState(initial_logits))
    return Trajectory(
        domain=f"{optimizer.name}-{mode}",
        config={
            "initial_state": initial_policy.ravel().tolist(),
            "initial_logits": np.asarray(initial_logits, dtype=np.float64).tolist(),
            "rewards": rewards.tolist(),
            "context_probabilities": contexts.tolist(),
            "eta": eta,
            "steps": steps,
            "mode": mode,
            "seed": seed,
            "sample_count": sample_count,
        },
        records=records,
    )
