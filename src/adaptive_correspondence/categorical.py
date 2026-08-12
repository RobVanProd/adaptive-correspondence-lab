"""Reference categorical adaptive systems with common instrumentation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .schema import StepRecord, rng_fingerprint, rng_snapshot
from .simplex import (
    categorical_fisher,
    categorical_logit_gradient,
    centered_logits,
    entropy,
    exponential_update,
    kl_divergence,
    negative_entropy_hessian,
    project_simplex_with_floor,
    replicator_field,
    simplex_diagnostics,
    softmax,
    validate_reward,
    validate_simplex,
)

FloatArray = NDArray[np.float64]


def _validate_mutation_matrix(matrix: ArrayLike | None, dimension: int) -> FloatArray | None:
    if matrix is None:
        return None
    kernel = np.asarray(matrix, dtype=np.float64)
    if kernel.shape != (dimension, dimension):
        raise ValueError("mutation matrix must be square with the state dimension")
    if not np.all(np.isfinite(kernel)) or np.any(kernel < 0.0):
        raise ValueError("mutation matrix must contain finite non-negative entries")
    if not np.allclose(np.sum(kernel, axis=1), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("each mutation-matrix row must sum to one")
    return kernel.copy()


@dataclass(frozen=True)
class CategoricalEffects:
    """Post-update mechanisms in a fixed, documented order."""

    mutation_rate: float = 0.0
    mutation_matrix: ArrayLike | None = None
    probability_floor: float = 0.0
    finite_population: int | None = None

    def validate(self, dimension: int) -> FloatArray | None:
        if not np.isfinite(self.mutation_rate) or not 0.0 <= self.mutation_rate <= 1.0:
            raise ValueError("mutation_rate must be in [0,1]")
        kernel = _validate_mutation_matrix(self.mutation_matrix, dimension)
        if self.mutation_rate > 0.0 and kernel is None:
            raise ValueError("positive mutation_rate requires a mutation_matrix")
        if not np.isfinite(self.probability_floor):
            raise ValueError("probability_floor must be finite")
        if self.probability_floor < 0.0 or self.probability_floor >= 1.0 / dimension:
            raise ValueError("probability_floor must be in [0, 1/dimension)")
        if self.finite_population is not None and (
            isinstance(self.finite_population, bool)
            or not isinstance(self.finite_population, (int, np.integer))
            or self.finite_population < 1
        ):
            raise ValueError("finite_population must be a positive integer")
        return kernel


class CategoricalWorld:
    """Shared reference transition shell for one categorical domain."""

    name = "categorical"

    def expected_vector_field(self, state: ArrayLike, reward: ArrayLike) -> FloatArray:
        return replicator_field(state, reward)

    def analytic_gradient(self, state: FloatArray, reward: FloatArray) -> FloatArray:
        return reward.copy()

    def native_state(self, state: FloatArray) -> dict[str, object]:
        return {"probability": state.tolist()}

    def transition(self, state: FloatArray, reward: FloatArray, eta: float) -> FloatArray:
        raise NotImplementedError

    def guards(self) -> list[str]:
        return []

    def step(
        self,
        state: ArrayLike,
        reward: ArrayLike,
        eta: float,
        *,
        step_index: int,
        effects: CategoricalEffects | None = None,
        rng: np.random.Generator | None = None,
        stochastic_error: ArrayLike | None = None,
        rng_fingerprint_override: str | None = None,
        rng_state_override: dict[str, object] | None = None,
        cumulative_regret_before: float = 0.0,
    ) -> StepRecord:
        probability = validate_simplex(state)
        rewards = validate_reward(reward, probability.size)
        if not np.isfinite(eta) or eta < 0.0:
            raise ValueError("eta must be finite and non-negative")
        applied_effects = effects or CategoricalEffects()
        mutation_kernel = applied_effects.validate(probability.size)

        fingerprint = rng_fingerprint_override or rng_fingerprint(rng)
        generator_state = rng_state_override or rng_snapshot(rng)
        guards = self.guards()
        proposed = self.transition(probability, rewards, eta)
        validate_simplex(proposed, name="unperturbed proposed state", atol=5e-13)

        after = proposed
        if applied_effects.mutation_rate > 0.0:
            assert mutation_kernel is not None
            mutated = after @ mutation_kernel
            after = (1.0 - applied_effects.mutation_rate) * after + (
                applied_effects.mutation_rate * mutated
            )
            after = validate_simplex(after, name="mutated state", atol=5e-13)
            guards.append("row-stochastic-mutation-kernel")

        if applied_effects.probability_floor > 0.0:
            after = project_simplex_with_floor(after, applied_effects.probability_floor)
            guards.append("euclidean-simplex-floor-projection")

        population_error = np.zeros_like(after)
        if applied_effects.finite_population is not None:
            if rng is None:
                raise ValueError("finite_population requires an explicit RNG")
            sampled = rng.multinomial(applied_effects.finite_population, after).astype(np.float64)
            sampled /= float(applied_effects.finite_population)
            population_error = sampled - after
            after = validate_simplex(sampled, name="finite-population state", atol=5e-13)
            guards.append("multinomial-frequency-simplex")

        external_error = (
            np.zeros_like(after)
            if stochastic_error is None
            else np.asarray(stochastic_error, dtype=np.float64)
        )
        if external_error.shape != after.shape or not np.all(np.isfinite(external_error)):
            raise ValueError("stochastic_error must be a finite vector matching the state")

        fisher = categorical_fisher(probability)
        geometry: dict[str, object] = {
            "fisher": fisher.tolist(),
            "fisher_rank": int(np.linalg.matrix_rank(fisher, tol=1e-12)),
            "bregman_generator": "negative-entropy",
        }
        if np.all(probability > 0.0):
            geometry["negative_entropy_hessian"] = negative_entropy_hessian(probability).tolist()
        else:
            geometry["negative_entropy_hessian"] = None
            guards.append("boundary-state-no-entropy-hessian")
        try:
            geometry["kl_after_before"] = kl_divergence(after, probability)
        except ValueError:
            geometry["kl_after_before"] = None
            guards.append("infinite-kl-not-serialized")

        reward_before = float(np.dot(probability, rewards))
        reward_after = float(np.dot(after, rewards))
        instant_regret = float(np.max(rewards) - reward_before)
        diagnostics = simplex_diagnostics(after, floor=applied_effects.probability_floor)
        violations = {
            **diagnostics,
            "violated": bool(
                (not diagnostics["finite"])
                or diagnostics["sum_error"] > 5e-13
                or diagnostics["min_probability"] < -5e-15
                or diagnostics["floor_shortfall"] > 5e-13
            ),
        }
        return StepRecord(
            domain=self.name,
            step=step_index,
            canonical_state_before=probability.tolist(),
            canonical_state_after=after.tolist(),
            native_state_before=self.native_state(probability),
            native_state_after=self.native_state(after),
            expected_vector_field=self.expected_vector_field(probability, rewards).tolist(),
            analytic_gradient=self.analytic_gradient(probability, rewards).tolist(),
            geometry=geometry,
            realized_update=(after - probability).tolist(),
            step_size=float(eta),
            stochastic_error=(external_error + population_error).tolist(),
            potentials={
                "expected_reward_before": reward_before,
                "expected_reward_after": reward_after,
                "entropy_before": entropy(probability),
                "entropy_after": entropy(after),
            },
            regret={
                "instantaneous": instant_regret,
                "cumulative": float(cumulative_regret_before + instant_regret),
            },
            constraint_violations=violations,
            numerical_guards=guards,
            rng_fingerprint=fingerprint,
            rng_state=generator_state,
        )


class ReplicatorDynamics(CategoricalWorld):
    """Replicator dynamics with either exact frozen-fitness flow or explicit Euler."""

    def __init__(self, integrator: Literal["exact", "euler"] = "exact") -> None:
        if integrator not in {"exact", "euler"}:
            raise ValueError("integrator must be 'exact' or 'euler'")
        self.integrator = integrator
        self.name = f"replicator-{integrator}"

    def transition(self, state: FloatArray, reward: FloatArray, eta: float) -> FloatArray:
        if self.integrator == "exact":
            return exponential_update(state, reward, eta)
        candidate = state + eta * replicator_field(state, reward)
        return validate_simplex(candidate, name="explicit Euler state", atol=5e-13)

    def guards(self) -> list[str]:
        if self.integrator == "exact":
            return ["max-shifted-exponential-normalization"]
        return ["explicit-euler-simplex-validity-check"]


class MultiplicativeWeights(CategoricalWorld):
    """Exponentiated-reward multiplicative weights on a simplex."""

    name = "multiplicative-weights"

    def transition(self, state: FloatArray, reward: FloatArray, eta: float) -> FloatArray:
        return exponential_update(state, reward, eta)

    def native_state(self, state: FloatArray) -> dict[str, object]:
        return {"normalized_weights": state.tolist()}

    def guards(self) -> list[str]:
        return ["max-shifted-exponential-normalization"]


class CategoricalNaturalGradient(CategoricalWorld):
    """Natural-gradient ascent in the centered categorical-logit gauge."""

    name = "categorical-natural-gradient"

    def transition(self, state: FloatArray, reward: FloatArray, eta: float) -> FloatArray:
        logits = centered_logits(state)
        natural_direction = reward - float(np.mean(reward))
        return softmax(logits + eta * natural_direction)

    def analytic_gradient(self, state: FloatArray, reward: FloatArray) -> FloatArray:
        return categorical_logit_gradient(state, reward)

    def native_state(self, state: FloatArray) -> dict[str, object]:
        return {"centered_logits": centered_logits(state).tolist()}

    def guards(self) -> list[str]:
        return ["centered-logit-gauge", "max-shifted-softmax"]
