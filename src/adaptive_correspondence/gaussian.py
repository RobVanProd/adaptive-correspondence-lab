"""Pure diagonal-Gaussian natural-gradient reference optimizer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .schema import StepRecord, Trajectory, rng_fingerprint, rng_snapshot

FloatArray = NDArray[np.float64]


def _finite_vector(value: ArrayLike, name: str) -> FloatArray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.ndim != 1 or vector.size < 1 or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a non-empty finite vector")
    return vector.copy()


@dataclass(frozen=True)
class DiagonalGaussianState:
    mean: ArrayLike
    log_std: ArrayLike

    def arrays(self) -> tuple[FloatArray, FloatArray]:
        mean = _finite_vector(self.mean, "mean")
        log_std = _finite_vector(self.log_std, "log_std")
        if mean.shape != log_std.shape:
            raise ValueError("mean and log_std must have the same shape")
        standard_deviation = np.exp(log_std)
        if not np.all(np.isfinite(standard_deviation)) or np.any(standard_deviation <= 0.0):
            raise ValueError("log_std must represent finite positive standard deviations")
        return mean, log_std

    def canonical(self) -> FloatArray:
        mean, log_std = self.arrays()
        return np.concatenate((mean, log_std))


@dataclass(frozen=True)
class DiagonalQuadraticObjective:
    target: ArrayLike
    curvature: ArrayLike

    def arrays(self) -> tuple[FloatArray, FloatArray]:
        target = _finite_vector(self.target, "target")
        curvature = _finite_vector(self.curvature, "curvature")
        if target.shape != curvature.shape:
            raise ValueError("target and curvature must have the same shape")
        if np.any(curvature <= 0.0):
            raise ValueError("quadratic curvature must be strictly positive")
        return target, curvature

    def evaluate(self, samples: ArrayLike) -> FloatArray:
        target, curvature = self.arrays()
        points = np.asarray(samples, dtype=np.float64)
        if points.shape[-1] != target.size or not np.all(np.isfinite(points)):
            raise ValueError("samples must be finite with objective dimension on the final axis")
        return -0.5 * np.sum(curvature * (points - target) ** 2, axis=-1)


class DiagonalGaussianNaturalGradient:
    """Analytic and finite-sample rank-mu updates with no hidden mechanisms."""

    name = "diagonal-gaussian-natural-gradient"

    def __init__(self, objective: DiagonalQuadraticObjective) -> None:
        self.objective = objective
        self.objective.arrays()

    def expected_objective(self, state: DiagonalGaussianState) -> float:
        mean, log_std = state.arrays()
        target, curvature = self.objective.arrays()
        if mean.shape != target.shape:
            raise ValueError("state and objective dimensions differ")
        variance = np.exp(2.0 * log_std)
        return float(-0.5 * np.sum(curvature * ((mean - target) ** 2 + variance)))

    def euclidean_gradient(self, state: DiagonalGaussianState) -> FloatArray:
        mean, log_std = state.arrays()
        target, curvature = self.objective.arrays()
        variance = np.exp(2.0 * log_std)
        gradient_mean = -curvature * (mean - target)
        gradient_log_std = -curvature * variance
        return np.concatenate((gradient_mean, gradient_log_std))

    def fisher(self, state: DiagonalGaussianState) -> FloatArray:
        _, log_std = state.arrays()
        variance = np.exp(2.0 * log_std)
        diagonal = np.concatenate((1.0 / variance, np.full(log_std.size, 2.0)))
        return np.diag(diagonal)

    def natural_direction(self, state: DiagonalGaussianState) -> FloatArray:
        _, log_std = state.arrays()
        gradient = self.euclidean_gradient(state)
        dimension = log_std.size
        variance = np.exp(2.0 * log_std)
        return np.concatenate((variance * gradient[:dimension], 0.5 * gradient[dimension:]))

    @staticmethod
    def _rank_mu_direction(
        state: DiagonalGaussianState,
        objective: DiagonalQuadraticObjective,
        rng: np.random.Generator,
        sample_count: int,
        parent_count: int | None,
    ) -> FloatArray:
        if isinstance(sample_count, bool) or sample_count < 2:
            raise ValueError("sample_count must be an integer of at least two")
        selected_count = sample_count // 2 if parent_count is None else parent_count
        if isinstance(selected_count, bool) or not 1 <= selected_count <= sample_count:
            raise ValueError("parent_count must be in [1, sample_count]")
        mean, log_std = state.arrays()
        standard_deviation = np.exp(log_std)
        standardized = rng.normal(size=(sample_count, mean.size))
        samples = mean + standard_deviation * standardized
        values = objective.evaluate(samples)
        selected = np.argsort(values)[::-1][:selected_count]
        ranks = np.arange(1, selected_count + 1, dtype=np.float64)
        weights = np.log(selected_count + 0.5) - np.log(ranks)
        weights /= float(np.sum(weights))
        mean_direction = np.sum(
            weights[:, None] * (samples[selected] - mean), axis=0
        )
        log_std_direction = 0.5 * np.sum(
            weights[:, None] * (standardized[selected] ** 2 - 1.0), axis=0
        )
        return np.concatenate((mean_direction, log_std_direction))

    @staticmethod
    def _kl_after_before(before: DiagonalGaussianState, after: DiagonalGaussianState) -> float:
        mean0, log_std0 = before.arrays()
        mean1, log_std1 = after.arrays()
        var0 = np.exp(2.0 * log_std0)
        var1 = np.exp(2.0 * log_std1)
        return float(
            0.5
            * np.sum((var1 + (mean1 - mean0) ** 2) / var0 - 1.0 + np.log(var0 / var1))
        )

    def step(
        self,
        state: DiagonalGaussianState,
        eta: float,
        *,
        step_index: int,
        mode: Literal["analytic", "rank-mu"] = "analytic",
        rng: np.random.Generator | None = None,
        sample_count: int = 32,
        parent_count: int | None = None,
        cumulative_regret_before: float = 0.0,
    ) -> StepRecord:
        mean, log_std = state.arrays()
        if not np.isfinite(eta) or eta < 0.0:
            raise ValueError("eta must be finite and non-negative")
        expected_direction = self.natural_direction(state)
        fingerprint = rng_fingerprint(rng)
        generator_state = rng_snapshot(rng)
        if mode == "analytic":
            direction = expected_direction
        elif mode == "rank-mu":
            if rng is None:
                raise ValueError("rank-mu mode requires an explicit RNG")
            direction = self._rank_mu_direction(
                state,
                self.objective,
                rng,
                sample_count,
                parent_count,
            )
        else:
            raise ValueError("mode must be 'analytic' or 'rank-mu'")

        dimension = mean.size
        updated = DiagonalGaussianState(
            mean=mean + eta * direction[:dimension],
            log_std=log_std + eta * direction[dimension:],
        )
        updated_mean, updated_log_std = updated.arrays()
        before_objective = self.expected_objective(state)
        after_objective = self.expected_objective(updated)
        instant_regret = -before_objective
        fisher = self.fisher(state)
        before_canonical = state.canonical()
        after_canonical = updated.canonical()
        return StepRecord(
            domain=f"{self.name}-{mode}",
            step=step_index,
            canonical_state_before=before_canonical.tolist(),
            canonical_state_after=after_canonical.tolist(),
            native_state_before={
                "mean": mean.tolist(),
                "standard_deviation": np.exp(log_std).tolist(),
                "log_standard_deviation": log_std.tolist(),
            },
            native_state_after={
                "mean": updated_mean.tolist(),
                "standard_deviation": np.exp(updated_log_std).tolist(),
                "log_standard_deviation": updated_log_std.tolist(),
            },
            expected_vector_field=expected_direction.tolist(),
            analytic_gradient=self.euclidean_gradient(state).tolist(),
            geometry={
                "fisher": fisher.tolist(),
                "parameterization": "mean-and-log-standard-deviation",
                "kl_after_before": self._kl_after_before(state, updated),
            },
            realized_update=(after_canonical - before_canonical).tolist(),
            step_size=float(eta),
            stochastic_error=(eta * (direction - expected_direction)).tolist(),
            potentials={
                "expected_objective_before": before_objective,
                "expected_objective_after": after_objective,
            },
            regret={
                "instantaneous": instant_regret,
                "cumulative": float(cumulative_regret_before + instant_regret),
            },
            constraint_violations={
                "finite": True,
                "minimum_standard_deviation": float(np.min(np.exp(updated_log_std))),
                "violated": False,
            },
            numerical_guards=[
                "log-standard-deviation-parameterization",
                "finite-positive-scale-validation",
                "no-clamps",
                "no-evolution-paths",
            ],
            rng_fingerprint=fingerprint,
            rng_state=generator_state,
        )


def run_gaussian_trajectory(
    *,
    initial_state: DiagonalGaussianState,
    objective: DiagonalQuadraticObjective,
    eta: float,
    steps: int,
    mode: Literal["analytic", "rank-mu"] = "analytic",
    seed: int = 1729,
    sample_count: int = 32,
    parent_count: int | None = None,
) -> Trajectory:
    if isinstance(steps, bool) or steps < 0:
        raise ValueError("steps must be a non-negative integer")
    rng = np.random.Generator(np.random.PCG64(seed))
    optimizer = DiagonalGaussianNaturalGradient(objective)
    state = initial_state
    records = []
    cumulative_regret = 0.0
    for step in range(steps):
        record = optimizer.step(
            state,
            eta,
            step_index=step,
            mode=mode,
            rng=rng if mode == "rank-mu" else None,
            sample_count=sample_count,
            parent_count=parent_count,
            cumulative_regret_before=cumulative_regret,
        )
        records.append(record)
        cumulative_regret = record.regret["cumulative"]
        canonical = np.asarray(record.canonical_state_after, dtype=np.float64)
        dimension = canonical.size // 2
        state = DiagonalGaussianState(canonical[:dimension], canonical[dimension:])
    target, curvature = objective.arrays()
    return Trajectory(
        domain=f"{optimizer.name}-{mode}",
        config={
            "initial_state": initial_state.canonical().tolist(),
            "target": target.tolist(),
            "curvature": curvature.tolist(),
            "eta": eta,
            "steps": steps,
            "mode": mode,
            "seed": seed,
            "sample_count": sample_count,
            "parent_count": parent_count,
        },
        records=records,
    )
