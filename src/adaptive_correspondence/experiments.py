"""Reproducible categorical comparison and prediction-transport protocols."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .batch import run_terminal_states
from .categorical import (
    CategoricalNaturalGradient,
    CategoricalWorld,
    MultiplicativeWeights,
    ReplicatorDynamics,
)
from .io import provenance
from .perturbations import Perturbation, canonical_direction
from .schema import Trajectory, rng_fingerprint, rng_snapshot
from .simplex import l1_distance, validate_reward, validate_simplex

FloatArray = NDArray[np.float64]
Metric = Literal["endpoint-l1", "max-path-l1"]


@dataclass(frozen=True)
class CategoricalExperimentConfig:
    initial_state: tuple[float, ...] = (0.2, 0.3, 0.5)
    reward: tuple[float, ...] = (0.7, -0.2, 0.1)
    eta: float = 0.05
    steps: int = 25
    seed: int = 1729
    common_schedule_amplitude: float = 0.0
    common_schedule_frequency: float = 0.41

    def validated(self) -> CategoricalExperimentConfig:
        state = validate_simplex(self.initial_state, name="initial_state", strictly_positive=True)
        validate_reward(self.reward, state.size)
        if not np.isfinite(self.eta) or self.eta < 0.0:
            raise ValueError("eta must be finite and non-negative")
        if (
            isinstance(self.steps, bool)
            or not isinstance(self.steps, (int, np.integer))
            or self.steps < 0
        ):
            raise ValueError("steps must be a non-negative integer")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, (int, np.integer))
            or self.seed < 0
        ):
            raise ValueError("seed must be a non-negative integer")
        if not np.isfinite(self.common_schedule_amplitude):
            raise ValueError("common_schedule_amplitude must be finite")
        if not np.isfinite(self.common_schedule_frequency):
            raise ValueError("common_schedule_frequency must be finite")
        return self

    def to_dict(self) -> dict[str, Any]:
        self.validated()
        return asdict(self)


def world_from_name(name: str) -> CategoricalWorld:
    worlds: dict[str, CategoricalWorld] = {
        "replicator-exact": ReplicatorDynamics("exact"),
        "replicator-euler": ReplicatorDynamics("euler"),
        "multiplicative-weights": MultiplicativeWeights(),
        "categorical-natural-gradient": CategoricalNaturalGradient(),
    }
    try:
        return worlds[name]
    except KeyError as error:
        raise ValueError(f"unknown categorical world: {name}") from error


def scheduled_reward(config: CategoricalExperimentConfig, step: int) -> FloatArray:
    base = np.asarray(config.reward, dtype=np.float64)
    if config.common_schedule_amplitude == 0.0:
        return base.copy()
    signal = np.sin(config.common_schedule_frequency * step)
    return base + config.common_schedule_amplitude * signal * canonical_direction(base.size)


def run_categorical_trajectory(
    world: CategoricalWorld,
    config: CategoricalExperimentConfig,
    *,
    perturbation: Perturbation | None = None,
    seed: int | None = None,
) -> Trajectory:
    config.validated()
    spec = perturbation or Perturbation()
    state = validate_simplex(config.initial_state, strictly_positive=True)
    spec.validate(state.size)
    run_seed = config.seed if seed is None else seed
    if (
        isinstance(run_seed, bool)
        or not isinstance(run_seed, (int, np.integer))
        or run_seed < 0
    ):
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.Generator(np.random.PCG64(run_seed))
    records = []
    cumulative_regret = 0.0

    for step in range(config.steps):
        current_reward = scheduled_reward(config, step)
        previous_reward = scheduled_reward(config, max(0, step - 1))
        fingerprint = rng_fingerprint(rng)
        generator_state = rng_snapshot(rng)
        observed_reward = spec.apply_reward(
            current_reward,
            previous_reward,
            state,
            step=step,
            rng=rng,
        )
        update_error = world.transition(state, observed_reward, config.eta) - world.transition(
            state, current_reward, config.eta
        )
        record = world.step(
            state,
            observed_reward,
            config.eta,
            step_index=step,
            effects=spec.effects(state.size),
            rng=rng,
            stochastic_error=update_error if spec.kind == "noise" else None,
            rng_fingerprint_override=fingerprint,
            rng_state_override=generator_state,
            cumulative_regret_before=cumulative_regret,
        )
        records.append(record)
        cumulative_regret = record.regret["cumulative"]
        state = np.asarray(record.canonical_state_after, dtype=np.float64)

    run_config = config.to_dict()
    run_config.update(
        {
            "seed": run_seed,
            "perturbation": asdict(spec),
            "mapping": "identity-on-simplex-probabilities",
        }
    )
    return Trajectory(domain=world.name, config=run_config, records=records)


def _path_distance(left: Trajectory, right: Trajectory) -> float:
    if len(left.records) != len(right.records):
        raise ValueError("trajectories must have the same horizon")
    initial_distance = l1_distance(left.config["initial_state"], right.config["initial_state"])
    distances = [initial_distance]
    distances.extend(
        l1_distance(a.canonical_state_after, b.canonical_state_after)
        for a, b in zip(left.records, right.records, strict=True)
    )
    return float(max(distances))


def trajectory_distance(left: Trajectory, right: Trajectory, metric: Metric) -> float:
    if metric == "endpoint-l1":
        return l1_distance(left.terminal_state, right.terminal_state)
    if metric == "max-path-l1":
        return _path_distance(left, right)
    raise ValueError(f"unknown metric: {metric}")


def run_equivalence(config: CategoricalExperimentConfig) -> dict[str, Any]:
    config.validated()
    worlds = [
        ReplicatorDynamics("exact"),
        MultiplicativeWeights(),
        CategoricalNaturalGradient(),
    ]
    trajectories = [run_categorical_trajectory(world, config) for world in worlds]
    reference = trajectories[0]
    pairwise = {}
    for trajectory in trajectories[1:]:
        pairwise[f"{reference.domain}__{trajectory.domain}"] = {
            "endpoint_l1": trajectory_distance(reference, trajectory, "endpoint-l1"),
            "max_path_l1": trajectory_distance(reference, trajectory, "max-path-l1"),
        }
    return {
        "kind": "categorical-exact-equivalence-verification",
        "claim_scope": "software reproduction under frozen assumptions only",
        "tolerance": {"absolute": 2e-14, "relative": 0.0},
        "passed": all(result["max_path_l1"] <= 2e-14 for result in pairwise.values()),
        "pairwise": pairwise,
        "trajectories": [trajectory.to_dict() for trajectory in trajectories],
        "provenance": provenance(),
    }


def _independent_seeds(master_seed: int, count: int) -> list[int]:
    if isinstance(count, bool) or count < 1:
        raise ValueError("seed_count must be a positive integer")
    sequence = np.random.SeedSequence(master_seed)
    return [int(child.generate_state(1, dtype=np.uint64)[0]) for child in sequence.spawn(count)]


def run_stability_sweep(
    config: CategoricalExperimentConfig,
    *,
    perturbation_kind: str,
    epsilons: ArrayLike,
    seed_count: int = 32,
    metric: Metric = "endpoint-l1",
    baseline_world: str = "multiplicative-weights",
    target_world: str = "replicator-exact",
) -> dict[str, Any]:
    config.validated()
    epsilon_values = np.asarray(epsilons, dtype=np.float64)
    if epsilon_values.ndim != 1 or epsilon_values.size < 1:
        raise ValueError("epsilons must be a non-empty vector")
    if not np.all(np.isfinite(epsilon_values)) or np.any(epsilon_values < 0.0):
        raise ValueError("epsilons must be finite and non-negative")
    sweep_config = config
    if perturbation_kind == "delay" and config.common_schedule_amplitude == 0.0:
        sweep_config = replace(config, common_schedule_amplitude=0.25)
    seeds = _independent_seeds(config.seed, seed_count)
    rows: list[dict[str, Any]] = []

    for epsilon in epsilon_values:
        spec = Perturbation(kind=perturbation_kind, epsilon=float(epsilon))
        spec.validate(len(config.initial_state))
        deltas = []
        for seed in seeds:
            actual_eta = float(epsilon) if perturbation_kind == "euler" else sweep_config.eta
            per_run_config = replace(sweep_config, eta=actual_eta)
            baseline = run_categorical_trajectory(
                world_from_name(baseline_world), per_run_config, seed=seed
            )
            effective_target = "replicator-euler" if perturbation_kind == "euler" else target_world
            target = run_categorical_trajectory(
                world_from_name(effective_target),
                per_run_config,
                perturbation=spec,
                seed=seed,
            )
            deltas.append(trajectory_distance(baseline, target, metric))
        values = np.asarray(deltas, dtype=np.float64)
        rows.append(
            {
                "epsilon": float(epsilon),
                "mean_delta": float(np.mean(values)),
                "std_delta": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
                "min_delta": float(np.min(values)),
                "median_delta": float(np.median(values)),
                "max_delta": float(np.max(values)),
                "seed_count": int(values.size),
            }
        )
    return {
        "kind": "epsilon-to-delta-stability-sweep",
        "perturbation": perturbation_kind,
        "metric": metric,
        "baseline_world": baseline_world,
        "target_world": "replicator-euler" if perturbation_kind == "euler" else target_world,
        "config": sweep_config.to_dict(),
        "seeds": seeds,
        "rows": rows,
        "provenance": provenance(),
    }


def fit_origin_slope(rows: list[dict[str, Any]]) -> float:
    epsilon = np.asarray([row["epsilon"] for row in rows], dtype=np.float64)
    delta = np.asarray([row["mean_delta"] for row in rows], dtype=np.float64)
    positive = epsilon > 0.0
    if not np.any(positive):
        raise ValueError("at least one positive epsilon is required to estimate a slope")
    denominator = float(np.dot(epsilon[positive], epsilon[positive]))
    return float(np.dot(epsilon[positive], delta[positive]) / denominator)


def run_transport_demo(
    config: CategoricalExperimentConfig,
    *,
    perturbation_kind: str = "reward-bias",
    source_epsilons: ArrayLike = (0.0005, 0.001, 0.002),
    target_epsilons: ArrayLike = (0.003, 0.006, 0.012),
    seed_count: int = 32,
) -> dict[str, Any]:
    source = run_stability_sweep(
        config,
        perturbation_kind=perturbation_kind,
        epsilons=source_epsilons,
        seed_count=seed_count,
        baseline_world="multiplicative-weights",
        target_world="replicator-exact",
    )
    coefficient = fit_origin_slope(source["rows"])
    target = run_stability_sweep(
        config,
        perturbation_kind=perturbation_kind,
        epsilons=target_epsilons,
        seed_count=seed_count,
        baseline_world="multiplicative-weights",
        target_world="categorical-natural-gradient",
    )
    comparisons = []
    for row in target["rows"]:
        prediction = coefficient * row["epsilon"]
        comparisons.append(
            {
                "epsilon": row["epsilon"],
                "predicted_delta": prediction,
                "observed_delta": row["mean_delta"],
                "residual": row["mean_delta"] - prediction,
            }
        )
    return {
        "kind": "transported-first-order-prediction-demonstration",
        "scientific_status": "plumbing demonstration; not a validated transport law",
        "fit": "origin-constrained least squares on source rows only",
        "source_coefficient": coefficient,
        "source": source,
        "target": target,
        "comparisons": comparisons,
        "provenance": provenance(),
    }


def software_verification(config: CategoricalExperimentConfig | None = None) -> dict[str, Any]:
    selected = config or CategoricalExperimentConfig(steps=10)
    equivalence = run_equivalence(selected)
    initial = np.asarray(selected.initial_state, dtype=np.float64)[None, :]
    batch_initial = np.repeat(initial, 5, axis=0)
    schedule = np.repeat(
        np.asarray(selected.reward, dtype=np.float64)[None, :], selected.steps, axis=0
    )
    batched = run_terminal_states(
        batch_initial,
        schedule,
        selected.eta,
        steps=selected.steps,
        method="exact",
        chunk_size=2,
    )
    reference = run_categorical_trajectory(MultiplicativeWeights(), selected).terminal_state
    batch_error = float(np.max(np.abs(batched - reference[None, :])))

    local_config = replace(selected, steps=1)
    errors = []
    for eta in (0.1, 0.05, 0.025):
        per_eta = replace(local_config, eta=eta)
        exact = run_categorical_trajectory(ReplicatorDynamics("exact"), per_eta)
        euler = run_categorical_trajectory(ReplicatorDynamics("euler"), per_eta)
        errors.append(l1_distance(exact.terminal_state, euler.terminal_state))
    order_ratios = [errors[index] / errors[index + 1] for index in range(len(errors) - 1)]
    passed = bool(
        equivalence["passed"]
        and batch_error <= 2e-14
        and all(3.5 < ratio < 4.5 for ratio in order_ratios)
    )
    return {
        "kind": "software-verification-summary",
        "passed": passed,
        "config": selected.to_dict(),
        "exact_equivalence": equivalence["pairwise"],
        "batch_max_absolute_error": batch_error,
        "euler_one_step_l1_errors": errors,
        "euler_halving_error_ratios": order_ratios,
        "provenance": provenance(),
    }
