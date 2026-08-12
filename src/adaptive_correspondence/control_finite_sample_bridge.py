"""Finite-sample plug-in NPG shadows for an exact contextual bandit."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .bandit import BanditPolicyState, ContextualBandit, NaturalPolicyGradient

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ControlBridgeState:
    rewards: ArrayLike
    context_probabilities: ArrayLike
    logits: ArrayLike

    def arrays(self) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        bandit = ContextualBandit(self.rewards, self.context_probabilities)
        rewards, contexts = bandit.arrays()
        state = BanditPolicyState(self.logits)
        logits = state.array(rewards.shape)
        policy = NaturalPolicyGradient(bandit).policy(state)
        if rewards.shape[0] < 2 or rewards.shape[1] < 2:
            raise ValueError("control bridge requires at least two contexts and actions")
        return rewards, contexts, logits, policy


def exact_natural_direction(state: ControlBridgeState) -> FloatArray:
    """Construct the exact centered NPG without calling sampled code."""
    rewards, _, _, policy = state.arrays()
    values = np.sum(policy * rewards, axis=1, keepdims=True)
    advantage = rewards - values
    return advantage - np.mean(advantage, axis=1, keepdims=True)


def _validated_counts(state: ControlBridgeState, counts: ArrayLike) -> FloatArray:
    rewards, _, _, _ = state.arrays()
    array = np.asarray(counts, dtype=np.float64)
    if (
        array.shape != rewards.shape
        or not np.all(np.isfinite(array))
        or np.any(array < 0.0)
        or not np.array_equal(array, np.floor(array))
        or np.sum(array) <= 0.0
    ):
        raise ValueError("counts must be a non-negative integer bandit table with positive total")
    return array


def plugin_direction_from_counts(
    state: ControlBridgeState, counts: ArrayLike, *, rcond: float = 1e-12
) -> FloatArray:
    """Compute one empirical-Fisher plug-in NPG direction from sufficient counts."""
    if not np.isfinite(rcond) or rcond <= 0.0:
        raise ValueError("rcond must be finite and positive")
    count_array = _validated_counts(state, counts)
    rewards, contexts, _, policy = state.arrays()
    total = float(np.sum(count_array))
    output = np.zeros_like(rewards)
    for context in range(contexts.size):
        scores = np.eye(policy.shape[1]) - policy[context][None, :]
        gradient = (count_array[context] * rewards[context]) @ scores / total
        score_outer = scores[:, :, None] * scores[:, None, :]
        fisher = np.sum(count_array[context, :, None, None] * score_outer, axis=0) / total
        output[context] = np.linalg.pinv(fisher, rcond=rcond) @ gradient
        output[context] -= np.mean(output[context])
    return output


def sample_plugin_npg_shadows(
    state: ControlBridgeState,
    *,
    sample_count: int,
    replications: int,
    rng: np.random.Generator,
    batch_size: int = 2048,
    rcond: float = 1e-12,
) -> FloatArray:
    """Draw independent plug-in NPG directions; never calls the exact comparator."""
    for value, name in (
        (sample_count, "sample_count"),
        (replications, "replications"),
        (batch_size, "batch_size"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not isinstance(rng, np.random.Generator):
        raise ValueError("an explicit NumPy Generator is required")
    if not np.isfinite(rcond) or rcond <= 0.0:
        raise ValueError("rcond must be finite and positive")
    rewards, contexts, _, policy = state.arrays()
    context_count, action_count = rewards.shape
    joint_probabilities = (contexts[:, None] * policy).ravel()
    output = np.empty((replications, context_count, action_count), dtype=np.float64)
    start = 0
    while start < replications:
        count = min(batch_size, replications - start)
        joint_counts = rng.multinomial(sample_count, joint_probabilities, size=count)
        tables = joint_counts.reshape(count, context_count, action_count).astype(np.float64)
        batch_directions = np.zeros_like(tables)
        for context in range(context_count):
            scores = np.eye(action_count) - policy[context][None, :]
            gradients = (
                (tables[:, context, :] * rewards[context][None, :]) @ scores / sample_count
            )
            score_outer = scores[:, :, None] * scores[:, None, :]
            fishers = np.einsum(
                "ba,aij->bij", tables[:, context, :], score_outer
            ) / sample_count
            inverses = np.linalg.pinv(fishers, rcond=rcond)
            directions = np.einsum("bij,bj->bi", inverses, gradients)
            directions -= np.mean(directions, axis=1, keepdims=True)
            batch_directions[:, context, :] = directions
        output[start : start + count] = batch_directions
        start += count
    if not np.all(np.isfinite(output)):
        raise FloatingPointError("control plug-in shadow directions are non-finite")
    return output


def _metric_cosine(left: FloatArray, right: FloatArray, metric: FloatArray) -> float:
    numerator = float(left @ metric @ right)
    denominator = float(np.sqrt((left @ metric @ left) * (right @ metric @ right)))
    if denominator == 0.0:
        raise ValueError("context Fisher cosine requires nonzero directions")
    return float(np.clip(numerator / denominator, -1.0, 1.0))


def context_fisher_cosines(
    state: ControlBridgeState, left: ArrayLike, right: ArrayLike
) -> list[float]:
    rewards, contexts, _, policy = state.arrays()
    left_array = np.asarray(left, dtype=np.float64).reshape(rewards.shape)
    right_array = np.asarray(right, dtype=np.float64).reshape(rewards.shape)
    if not np.all(np.isfinite(left_array)) or not np.all(np.isfinite(right_array)):
        raise ValueError("control directions must be finite")
    result = []
    for context in range(contexts.size):
        fisher = contexts[context] * (
            np.diag(policy[context]) - np.outer(policy[context], policy[context])
        )
        result.append(_metric_cosine(left_array[context], right_array[context], fisher))
    return result


def joint_fisher_cosine(
    state: ControlBridgeState, left: ArrayLike, right: ArrayLike
) -> float:
    rewards, contexts, _, policy = state.arrays()
    left_array = np.asarray(left, dtype=np.float64).reshape(rewards.shape)
    right_array = np.asarray(right, dtype=np.float64).reshape(rewards.shape)
    numerator = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for context in range(contexts.size):
        fisher = contexts[context] * (
            np.diag(policy[context]) - np.outer(policy[context], policy[context])
        )
        numerator += float(left_array[context] @ fisher @ right_array[context])
        left_norm += float(left_array[context] @ fisher @ left_array[context])
        right_norm += float(right_array[context] @ fisher @ right_array[context])
    if left_norm == 0.0 or right_norm == 0.0:
        raise ValueError("joint Fisher cosine requires nonzero directions")
    return float(np.clip(numerator / np.sqrt(left_norm * right_norm), -1.0, 1.0))
