"""Frozen ACL-002 mutation-sensitivity protocol and guarded future execution.

Importing this module never executes the preregistered experiment. Development tests
use separate toy landscapes. The confirmatory runner requires an explicitly approved
Git SHA, a clean tracked worktree, valid file locks, and a new output path.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .io import provenance, write_json
from .schema import json_safe
from .simplex import as_float_vector, validate_reward, validate_simplex

FloatArray = NDArray[np.float64]
SensitivityStratum = Literal[
    "analytic-zero",
    "low-sensitivity",
    "regular-sensitivity",
]

INHERITED_TOLERANCE = 2e-14
SAFETY_MULTIPLIER = 100
DELTA_FLOOR = INHERITED_TOLERANCE * SAFETY_MULTIPLIER
MAX_CONFIRMATORY_EPSILON = 1e-2
MEDIAN_GATE = 0.10
Q90_GATE = 0.20
ACL002_ETA = 0.05
ACL002_PRIMARY_HORIZON = 20
ACL002_SECONDARY_HORIZONS = (1, 5, 50)
ACL002_EPSILON_GRID = (0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)
ACL002_CONFIRMATORY_EPSILONS = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
ACL002_STRESS_EPSILONS = (3e-2, 1e-1)


@dataclass(frozen=True)
class SensitivityTrace:
    """Clean probability states and row sensitivities from time zero onward."""

    states: FloatArray
    sensitivities: FloatArray


@dataclass(frozen=True)
class AnalyticCoefficients:
    """First-order L1 and second-order oriented-KL coefficients at one horizon."""

    endpoint_l1: float
    path_l1: float
    kl_q_p: float


@dataclass(frozen=True)
class Landscape:
    identifier: str
    split: Literal["source", "target"]
    p0_name: str
    reward_name: str
    mutation_name: str
    p0: FloatArray
    reward: FloatArray
    mutation: FloatArray


@dataclass(frozen=True)
class ACL002Manifest:
    experiment_id: str
    eta: float
    primary_horizon: int
    secondary_horizons: tuple[int, ...]
    epsilon_grid: tuple[float, ...]
    confirmatory_epsilons: tuple[float, ...]
    stress_epsilons: tuple[float, ...]
    landscapes: tuple[Landscape, ...]
    expected_zero_ids: tuple[str, ...]
    expected_low_ids: tuple[str, ...]
    raw: dict[str, Any]

    @property
    def horizons(self) -> tuple[int, ...]:
        return tuple(sorted({self.primary_horizon, *self.secondary_horizons}))


@dataclass(frozen=True)
class GateResult:
    median: float
    q90: float
    passed: bool
    landscape_count: int


def _positive_mass_vector(value: ArrayLike, *, name: str) -> FloatArray:
    vector = as_float_vector(value, name=name)
    if np.any(vector < 0.0) or float(np.sum(vector)) <= 0.0:
        raise ValueError(f"{name} must have non-negative entries and positive mass")
    return vector


def categorical_map(probability: ArrayLike, reward: ArrayLike, eta: float) -> FloatArray:
    """Exact categorical map, extended to positive-mass ambient coordinates."""
    state = _positive_mass_vector(probability, name="probability")
    rewards = validate_reward(reward, state.size)
    if not np.isfinite(eta) or eta < 0.0:
        raise ValueError("eta must be finite and non-negative")
    centered_reward = rewards - float(np.max(rewards))
    factors = np.exp(eta * centered_reward)
    weighted = state * factors
    total = float(np.sum(weighted))
    if not np.isfinite(total) or total <= 0.0:
        raise FloatingPointError("categorical map has no finite positive mass")
    return weighted / total


def row_jacobian(probability: ArrayLike, reward: ArrayLike, eta: float) -> FloatArray:
    """Return J[i,j] = partial F_j / partial p_i for row-vector propagation."""
    state = _positive_mass_vector(probability, name="probability")
    rewards = validate_reward(reward, state.size)
    if not np.isfinite(eta) or eta < 0.0:
        raise ValueError("eta must be finite and non-negative")
    factors = np.exp(eta * (rewards - float(np.max(rewards))))
    normalizer = float(np.dot(state, factors))
    output = state * factors / normalizer
    scaled_factors = factors / normalizer
    jacobian = np.diag(scaled_factors) - np.outer(scaled_factors, output)
    if not np.allclose(np.sum(jacobian, axis=1), 0.0, rtol=0.0, atol=2e-14):
        raise FloatingPointError("row Jacobian violates output-mass conservation")
    return jacobian


def _validate_mutation(matrix: ArrayLike, dimension: int) -> FloatArray:
    mutation = np.asarray(matrix, dtype=np.float64)
    if mutation.shape != (dimension, dimension):
        raise ValueError("mutation matrix must be square with the state dimension")
    if not np.all(np.isfinite(mutation)) or np.any(mutation < 0.0):
        raise ValueError("mutation matrix must be finite and non-negative")
    if not np.allclose(np.sum(mutation, axis=1), 1.0, rtol=0.0, atol=2e-14):
        raise ValueError("mutation matrix must be row-stochastic")
    return mutation.copy()


def sensitivity_trajectory(
    p0: ArrayLike,
    reward: ArrayLike,
    mutation: ArrayLike,
    *,
    eta: float,
    steps: int,
) -> SensitivityTrace:
    initial = validate_simplex(p0, name="p0", strictly_positive=True)
    rewards = validate_reward(reward, initial.size)
    matrix = _validate_mutation(mutation, initial.size)
    if isinstance(steps, bool) or not isinstance(steps, (int, np.integer)) or steps < 0:
        raise ValueError("steps must be a non-negative integer")
    states = np.empty((steps + 1, initial.size), dtype=np.float64)
    sensitivities = np.empty_like(states)
    states[0] = initial
    sensitivities[0] = 0.0
    identity = np.eye(initial.size, dtype=np.float64)
    for step in range(steps):
        current = states[step]
        next_state = categorical_map(current, rewards, eta)
        sensitivity = sensitivities[step] @ row_jacobian(current, rewards, eta)
        sensitivity += next_state @ (matrix - identity)
        if abs(float(np.sum(sensitivity))) > 2e-13:
            raise FloatingPointError("row sensitivity left the simplex tangent space")
        states[step + 1] = validate_simplex(next_state, strictly_positive=True)
        sensitivities[step + 1] = sensitivity
    return SensitivityTrace(states=states, sensitivities=sensitivities)


def mutation_trajectory(
    p0: ArrayLike,
    reward: ArrayLike,
    mutation: ArrayLike,
    *,
    eta: float,
    epsilon: float,
    steps: int,
) -> FloatArray:
    initial = validate_simplex(p0, name="p0", strictly_positive=True)
    rewards = validate_reward(reward, initial.size)
    matrix = _validate_mutation(mutation, initial.size)
    if not np.isfinite(epsilon) or not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must be finite and in [0,1]")
    if isinstance(steps, bool) or not isinstance(steps, (int, np.integer)) or steps < 0:
        raise ValueError("steps must be a non-negative integer")
    states = np.empty((steps + 1, initial.size), dtype=np.float64)
    states[0] = initial
    for step in range(steps):
        clean_proposal = categorical_map(states[step], rewards, eta)
        next_state = (1.0 - epsilon) * clean_proposal + epsilon * (clean_proposal @ matrix)
        states[step + 1] = validate_simplex(
            next_state,
            name="mutation state",
            strictly_positive=True,
            atol=5e-13,
        )
    return states


def analytic_coefficients(trace: SensitivityTrace, *, horizon: int) -> AnalyticCoefficients:
    if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)) or horizon < 0:
        raise ValueError("horizon must be a non-negative integer")
    if trace.states.shape != trace.sensitivities.shape or trace.states.ndim != 2:
        raise ValueError("trace states and sensitivities must be same-shaped matrices")
    if horizon >= trace.states.shape[0]:
        raise ValueError("horizon exceeds the sensitivity trace")
    state = trace.states[horizon]
    sensitivity = trace.sensitivities[horizon]
    if np.any(state <= 0.0):
        raise ValueError("KL coefficient requires an interior clean state")
    endpoint = float(np.sum(np.abs(sensitivity)))
    path = float(np.max(np.sum(np.abs(trace.sensitivities[: horizon + 1]), axis=1)))
    kl = float(0.5 * np.sum((sensitivity**2) / state))
    return AnalyticCoefficients(endpoint_l1=endpoint, path_l1=path, kl_q_p=kl)


def classify_sensitivity(coefficient: float) -> SensitivityStratum:
    if not np.isfinite(coefficient) or coefficient < 0.0:
        raise ValueError("analytic sensitivity coefficient must be finite and non-negative")
    if coefficient <= INHERITED_TOLERANCE:
        return "analytic-zero"
    if coefficient * MAX_CONFIRMATORY_EPSILON < DELTA_FLOOR:
        return "low-sensitivity"
    return "regular-sensitivity"


def type7_quantile(values: ArrayLike, quantile: float) -> float:
    sample = np.asarray(values, dtype=np.float64)
    if sample.ndim != 1 or sample.size < 1 or not np.all(np.isfinite(sample)):
        raise ValueError("quantile values must be a non-empty finite vector")
    if not np.isfinite(quantile) or not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0,1]")
    return float(np.quantile(sample, quantile, method="linear"))


def per_landscape_alpha(coefficient: float, epsilons: ArrayLike, deltas: ArrayLike) -> float:
    if not np.isfinite(coefficient) or coefficient <= 0.0:
        raise ValueError("alpha estimation requires a positive finite coefficient")
    epsilon = np.asarray(epsilons, dtype=np.float64)
    observed = np.asarray(deltas, dtype=np.float64)
    if epsilon.ndim != 1 or epsilon.shape != observed.shape or epsilon.size < 1:
        raise ValueError("epsilons and deltas must be same-shaped non-empty vectors")
    if (
        not np.all(np.isfinite(epsilon))
        or not np.all(np.isfinite(observed))
        or np.any(epsilon <= 0.0)
        or np.any(observed < 0.0)
    ):
        raise ValueError(
            "alpha inputs must be finite with positive epsilons and non-negative deltas"
        )
    predictor = coefficient * epsilon
    denominator = float(np.dot(predictor, predictor))
    if denominator <= 0.0:
        raise ValueError("alpha fit has a zero denominator")
    return float(np.dot(predictor, observed) / denominator)


def median_source_alpha(per_landscape_alphas: ArrayLike) -> float:
    value = type7_quantile(per_landscape_alphas, 0.5)
    if value <= 0.0:
        raise ValueError("median source alpha must be positive")
    return value


def landscape_relative_score(
    coefficient: float,
    epsilons: ArrayLike,
    deltas: ArrayLike,
    *,
    alpha: float,
) -> float:
    epsilon = np.asarray(epsilons, dtype=np.float64)
    observed = np.asarray(deltas, dtype=np.float64)
    if epsilon.ndim != 1 or epsilon.shape != observed.shape or epsilon.size < 1:
        raise ValueError("relative-score inputs must be same-shaped non-empty vectors")
    prediction = alpha * coefficient * epsilon
    if not np.all(np.isfinite(prediction)) or np.any(prediction <= 0.0):
        raise ValueError("regular-target predictions must be finite and positive")
    if not np.all(np.isfinite(observed)) or np.any(observed < 0.0):
        raise ValueError("observed discrepancies must be finite and non-negative")
    errors = np.abs(observed - prediction) / prediction
    return type7_quantile(errors, 0.5)


def evaluate_gate(landscape_scores: ArrayLike) -> GateResult:
    scores = np.asarray(landscape_scores, dtype=np.float64)
    if scores.ndim != 1 or scores.size < 1 or not np.all(np.isfinite(scores)):
        raise ValueError("gate scores must be a non-empty finite vector")
    if np.any(scores < 0.0):
        raise ValueError("gate scores must be non-negative")
    median = type7_quantile(scores, 0.5)
    q90 = type7_quantile(scores, 0.9)
    return GateResult(
        median=median,
        q90=q90,
        passed=bool(median <= MEDIAN_GATE and q90 <= Q90_GATE),
        landscape_count=int(scores.size),
    )


def _as_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite_float_tuple(values: Any, name: str) -> tuple[float, ...]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 1 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a non-empty finite vector")
    return tuple(float(value) for value in array)


def validate_manifest_dict(payload: dict[str, Any]) -> ACL002Manifest:
    if not isinstance(payload, dict):
        raise ValueError("manifest must be a JSON object")
    if payload.get("schema_version") != 1:
        raise ValueError("manifest schema_version must be 1")
    experiment_id = payload.get("experiment_id")
    if not isinstance(experiment_id, str) or not experiment_id:
        raise ValueError("manifest experiment_id must be a non-empty string")
    if payload.get("randomness") != "none":
        raise ValueError("ACL-002 manifest must declare randomness as none")
    if payload.get("vector_convention") != "row":
        raise ValueError("ACL-002 manifest must use row vectors")
    expected_operation = "(1-epsilon)*F(p) + epsilon*(F(p) @ M)"
    if payload.get("mutation_operation") != expected_operation:
        raise ValueError("manifest mutation operation does not match the frozen row convention")
    eta = float(payload.get("eta", np.nan))
    if not np.isfinite(eta) or eta <= 0.0:
        raise ValueError("manifest eta must be finite and positive")
    primary_horizon = _as_positive_int(payload.get("primary_horizon"), "primary_horizon")
    secondary = tuple(
        _as_positive_int(value, "secondary horizon")
        for value in payload.get("secondary_horizons", [])
    )
    if not secondary or primary_horizon in secondary or len(set(secondary)) != len(secondary):
        raise ValueError("secondary horizons must be unique, non-empty, and exclude primary")
    epsilon_grid = _finite_float_tuple(payload.get("epsilon_grid"), "epsilon_grid")
    confirmatory = _finite_float_tuple(
        payload.get("confirmatory_epsilons"), "confirmatory_epsilons"
    )
    stress = _finite_float_tuple(payload.get("stress_epsilons"), "stress_epsilons")
    if epsilon_grid[0] != 0.0 or tuple(sorted(set(epsilon_grid))) != epsilon_grid:
        raise ValueError("epsilon_grid must be unique, sorted, and begin at zero")
    if any(value <= 0.0 for value in confirmatory + stress):
        raise ValueError("confirmatory and stress epsilons must be positive")
    if set(confirmatory) & set(stress) or set(confirmatory + stress) != set(epsilon_grid[1:]):
        raise ValueError("confirmatory and stress regions must partition positive epsilon_grid")
    if experiment_id == "ACL-002" and (
        eta != ACL002_ETA
        or primary_horizon != ACL002_PRIMARY_HORIZON
        or secondary != ACL002_SECONDARY_HORIZONS
        or epsilon_grid != ACL002_EPSILON_GRID
        or confirmatory != ACL002_CONFIRMATORY_EPSILONS
        or stress != ACL002_STRESS_EPSILONS
    ):
        raise ValueError("ACL-002 manifest design constants differ from the preregistration")

    numerical = payload.get("numerical_policy", {})
    expected_numerical = {
        "inherited_tolerance": INHERITED_TOLERANCE,
        "safety_multiplier": SAFETY_MULTIPLIER,
        "delta_floor": DELTA_FLOOR,
        "quantile_method": "linear",
        "quantile_definition": "Hyndman-Fan Type 7",
    }
    if numerical != expected_numerical:
        raise ValueError("manifest numerical policy differs from frozen ACL-002 constants")
    gates = payload.get("gates", {})
    if gates != {
        "target_landscape_median_relative_error_max": MEDIAN_GATE,
        "target_landscape_q90_relative_error_max": Q90_GATE,
    }:
        raise ValueError("manifest gates differ from frozen ACL-002 constants")

    states_payload = payload.get("states")
    rewards_payload = payload.get("rewards")
    matrices_payload = payload.get("mutation_matrices")
    catalogs = (states_payload, rewards_payload, matrices_payload)
    if not all(isinstance(catalog, dict) and catalog for catalog in catalogs):
        raise ValueError("state, reward, and mutation catalogs must be non-empty objects")
    states = {
        name: validate_simplex(value, name=f"state {name}", strictly_positive=True)
        for name, value in states_payload.items()
    }
    dimensions = {state.size for state in states.values()}
    if len(dimensions) != 1:
        raise ValueError("all manifest states must have one dimension")
    dimension = dimensions.pop()
    rewards = {
        name: validate_reward(value, dimension) for name, value in rewards_payload.items()
    }
    matrices = {
        name: _validate_mutation(value, dimension) for name, value in matrices_payload.items()
    }

    landscape_payload = payload.get("landscapes")
    if not isinstance(landscape_payload, list) or not landscape_payload:
        raise ValueError("landscapes must be a non-empty list")
    landscapes: list[Landscape] = []
    identifiers: set[str] = set()
    splits: set[str] = set()
    for item in landscape_payload:
        if not isinstance(item, dict):
            raise ValueError("each landscape must be an object")
        identifier = item.get("id")
        split = item.get("split")
        p0_name = item.get("p0")
        reward_name = item.get("reward")
        mutation_name = item.get("mutation")
        if not isinstance(identifier, str) or not identifier or identifier in identifiers:
            raise ValueError("landscape IDs must be non-empty and unique")
        if split not in {"source", "target"}:
            raise ValueError("landscape split must be source or target")
        if p0_name not in states or reward_name not in rewards or mutation_name not in matrices:
            raise ValueError(f"landscape {identifier} has an unknown catalog reference")
        identifiers.add(identifier)
        splits.add(split)
        landscapes.append(
            Landscape(
                identifier=identifier,
                split=split,
                p0_name=p0_name,
                reward_name=reward_name,
                mutation_name=mutation_name,
                p0=states[p0_name],
                reward=rewards[reward_name],
                mutation=matrices[mutation_name],
            )
        )
    if splits != {"source", "target"}:
        raise ValueError("manifest must contain source and target landscapes")
    if experiment_id == "ACL-002":
        source_count = sum(item.split == "source" for item in landscapes)
        target_count = sum(item.split == "target" for item in landscapes)
        if (source_count, target_count) != (14, 14):
            raise ValueError("ACL-002 must contain exactly 14 source and 14 target landscapes")

    expectations = payload.get("predeclared_stratum_expectations")
    if not isinstance(expectations, dict):
        raise ValueError("manifest must declare special-stratum expectations")
    zero_ids = tuple(expectations.get("analytic_zero", []))
    low_ids = tuple(expectations.get("low_sensitivity", []))
    if len(set(zero_ids)) != len(zero_ids) or len(set(low_ids)) != len(low_ids):
        raise ValueError("predeclared stratum IDs must be unique")
    if set(zero_ids) & set(low_ids) or not set(zero_ids + low_ids) <= identifiers:
        raise ValueError("predeclared strata must be disjoint known landscape IDs")
    if experiment_id == "ACL-002" and (
        zero_ids != ("S13", "T13") or low_ids != ("S14", "T14")
    ):
        raise ValueError("ACL-002 special-stratum expectations differ from preregistration")
    return ACL002Manifest(
        experiment_id=experiment_id,
        eta=eta,
        primary_horizon=primary_horizon,
        secondary_horizons=secondary,
        epsilon_grid=epsilon_grid,
        confirmatory_epsilons=confirmatory,
        stress_epsilons=stress,
        landscapes=tuple(landscapes),
        expected_zero_ids=zero_ids,
        expected_low_ids=low_ids,
        raw=payload,
    )


def load_manifest(path: str | Path) -> ACL002Manifest:
    manifest_path = Path(path)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read manifest: {manifest_path}") from error
    return validate_manifest_dict(payload)


def build_analytic_registry(manifest: ACL002Manifest) -> dict[str, Any]:
    """Compute only clean trajectories and analytic sensitivities—never outcomes."""
    entries: list[dict[str, Any]] = []
    observed_zero: list[str] = []
    observed_low: list[str] = []
    max_horizon = max(manifest.horizons)
    for landscape in manifest.landscapes:
        trace = sensitivity_trajectory(
            landscape.p0,
            landscape.reward,
            landscape.mutation,
            eta=manifest.eta,
            steps=max_horizon,
        )
        coefficients = {
            str(horizon): analytic_coefficients(trace, horizon=horizon)
            for horizon in manifest.horizons
        }
        primary = coefficients[str(manifest.primary_horizon)]
        stratum = classify_sensitivity(primary.endpoint_l1)
        if stratum == "analytic-zero":
            observed_zero.append(landscape.identifier)
        elif stratum == "low-sensitivity":
            observed_low.append(landscape.identifier)
        entries.append(
            {
                "id": landscape.identifier,
                "split": landscape.split,
                "p0": landscape.p0_name,
                "reward": landscape.reward_name,
                "mutation": landscape.mutation_name,
                "C_primary": primary.endpoint_l1,
                "K_primary": primary.kl_q_p,
                "stratum": stratum,
                "horizons": {
                    horizon: {
                        "C_endpoint_l1": value.endpoint_l1,
                        "C_max_path_l1": value.path_l1,
                        "K_kl_q_p": value.kl_q_p,
                    }
                    for horizon, value in coefficients.items()
                },
            }
        )
    if tuple(observed_zero) != manifest.expected_zero_ids:
        raise ValueError(
            "analytic-zero registry does not match predeclared expectation: "
            f"observed={observed_zero}, expected={list(manifest.expected_zero_ids)}"
        )
    if tuple(observed_low) != manifest.expected_low_ids:
        raise ValueError(
            "low-sensitivity registry does not match predeclared expectation: "
            f"observed={observed_low}, expected={list(manifest.expected_low_ids)}"
        )
    canonical_manifest = json.dumps(
        json_safe(manifest.raw), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return {
        "schema_version": 1,
        "experiment_id": manifest.experiment_id,
        "kind": "pre-outcome-analytic-registry",
        "outcomes_generated": False,
        "manifest_canonical_sha256": hashlib.sha256(
            canonical_manifest.encode("utf-8")
        ).hexdigest(),
        "row_vector_convention": True,
        "primary_horizon": manifest.primary_horizon,
        "delta_floor": DELTA_FLOOR,
        "landscapes": entries,
    }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_lock(lock_path: str | Path) -> dict[str, Any]:
    path = Path(lock_path).resolve()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read preregistration lock: {path}") from error
    if payload.get("schema_version") != 1 or not isinstance(payload.get("files"), dict):
        raise ValueError("invalid preregistration lock schema")
    if not payload["files"]:
        raise ValueError("preregistration lock must contain files")
    base = path.parent
    for relative, expected_hash in payload["files"].items():
        candidate = (base / relative).resolve()
        if base != candidate and base not in candidate.parents:
            raise ValueError("lock file path escapes the preregistration directory")
        actual_hash = sha256_file(candidate)
        if actual_hash != expected_hash:
            raise ValueError(f"preregistration hash mismatch for {relative}")
    return payload


def assert_execution_context(
    *,
    approved_sha: str,
    current_sha: str,
    tracked_dirty: bool,
    output_path: str | Path,
) -> None:
    if not approved_sha or current_sha != approved_sha:
        raise ValueError("current Git HEAD does not equal the explicitly approved SHA")
    if tracked_dirty:
        raise ValueError("tracked worktree must be clean before ACL-002 execution")
    if Path(output_path).exists():
        raise FileExistsError("ACL-002 output already exists and will not be overwritten")


def _git_execution_state(repo_path: str | Path) -> tuple[str, bool]:
    repo = Path(repo_path)
    try:
        current_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        ).stdout.strip()
        tracked_status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.SubprocessError) as error:
        raise ValueError("cannot verify Git execution state") from error
    return current_sha, bool(tracked_status)


def _oriented_kl_q_p(q: FloatArray, p: FloatArray) -> float:
    if np.any(q <= 0.0) or np.any(p <= 0.0):
        raise ValueError("oriented KL requires interior q and p")
    return float(np.sum(q * np.log(q / p)))


def generate_raw_rows(manifest: ACL002Manifest) -> list[dict[str, Any]]:
    """Generate locked outcomes. Do not call before the preregistration SHA is approved."""
    rows: list[dict[str, Any]] = []
    max_horizon = max(manifest.horizons)
    for landscape in manifest.landscapes:
        trace = sensitivity_trajectory(
            landscape.p0,
            landscape.reward,
            landscape.mutation,
            eta=manifest.eta,
            steps=max_horizon,
        )
        primary_c = analytic_coefficients(
            trace, horizon=manifest.primary_horizon
        ).endpoint_l1
        stratum = classify_sensitivity(primary_c)
        for epsilon in manifest.epsilon_grid:
            perturbed = mutation_trajectory(
                landscape.p0,
                landscape.reward,
                landscape.mutation,
                eta=manifest.eta,
                epsilon=epsilon,
                steps=max_horizon,
            )
            region = (
                "zero"
                if epsilon == 0.0
                else "confirmatory"
                if epsilon in manifest.confirmatory_epsilons
                else "stress"
            )
            for horizon in manifest.horizons:
                coefficients = analytic_coefficients(trace, horizon=horizon)
                clean_state = trace.states[horizon]
                perturbed_state = perturbed[horizon]
                endpoint_l1 = float(np.sum(np.abs(perturbed_state - clean_state)))
                path_l1 = float(
                    np.max(
                        np.sum(
                            np.abs(perturbed[: horizon + 1] - trace.states[: horizon + 1]),
                            axis=1,
                        )
                    )
                )
                oriented_kl = _oriented_kl_q_p(perturbed_state, clean_state)
                rows.append(
                    {
                        "landscape_id": landscape.identifier,
                        "split": landscape.split,
                        "p0": landscape.p0_name,
                        "reward": landscape.reward_name,
                        "mutation": landscape.mutation_name,
                        "epsilon": epsilon,
                        "region": region,
                        "horizon": horizon,
                        "stratum": stratum,
                        "clean_terminal": clean_state.tolist(),
                        "perturbed_terminal": perturbed_state.tolist(),
                        "endpoint_l1": endpoint_l1,
                        "max_path_l1": path_l1,
                        "kl_q_p": oriented_kl,
                        "C_endpoint_l1": coefficients.endpoint_l1,
                        "C_max_path_l1": coefficients.path_l1,
                        "K_kl_q_p": coefficients.kl_q_p,
                        "zero_fit_l1_prediction": coefficients.endpoint_l1 * epsilon,
                        "zero_fit_max_path_l1_prediction": coefficients.path_l1 * epsilon,
                        "zero_fit_kl_prediction": coefficients.kl_q_p * epsilon**2,
                        "kl_over_epsilon_squared": (
                            None if epsilon == 0.0 else oriented_kl / epsilon**2
                        ),
                        "kl_coefficient_error": (
                            None
                            if epsilon == 0.0
                            else oriented_kl / epsilon**2 - coefficients.kl_q_p
                        ),
                    }
                )
    return rows


def _registry_by_id(registry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    entries = registry.get("landscapes")
    if not isinstance(entries, list):
        raise ValueError("analytic registry has no landscape list")
    result = {entry.get("id"): entry for entry in entries if isinstance(entry, dict)}
    if len(result) != len(entries) or None in result:
        raise ValueError("analytic registry landscape IDs are invalid")
    return result


def _primary_local_rows(
    manifest: ACL002Manifest, rows: list[dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {
        landscape.identifier: [] for landscape in manifest.landscapes
    }
    for row in rows:
        if (
            row.get("horizon") == manifest.primary_horizon
            and row.get("epsilon") in manifest.confirmatory_epsilons
        ):
            identifier = row.get("landscape_id")
            if identifier not in grouped:
                raise ValueError("raw row has an unknown landscape")
            grouped[identifier].append(row)
    expected = list(manifest.confirmatory_epsilons)
    for identifier, selected in grouped.items():
        selected.sort(key=lambda row: row["epsilon"])
        if [row["epsilon"] for row in selected] != expected:
            raise ValueError(f"raw rows have an incomplete local grid for {identifier}")
    return grouped


def analyze_raw_rows(
    manifest: ACL002Manifest,
    registry: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    expected_count = len(manifest.landscapes) * len(manifest.epsilon_grid) * len(
        manifest.horizons
    )
    if len(rows) != expected_count:
        raise ValueError(f"raw row count {len(rows)} does not equal expected {expected_count}")
    registry_entries = _registry_by_id(registry)
    grouped = _primary_local_rows(manifest, rows)
    source_alphas: list[dict[str, float | str]] = []
    for landscape in manifest.landscapes:
        entry = registry_entries[landscape.identifier]
        if landscape.split != "source" or entry["stratum"] != "regular-sensitivity":
            continue
        selected = grouped[landscape.identifier]
        alpha = per_landscape_alpha(
            entry["C_primary"],
            [row["epsilon"] for row in selected],
            [row["endpoint_l1"] for row in selected],
        )
        source_alphas.append({"landscape_id": landscape.identifier, "alpha": alpha})
    alpha_source = median_source_alpha([item["alpha"] for item in source_alphas])

    layer_scores: dict[str, list[dict[str, float | str]]] = {
        "analytic": [],
        "transport": [],
    }
    target_prediction_rows: list[dict[str, Any]] = []
    special: list[dict[str, Any]] = []
    for landscape in manifest.landscapes:
        if landscape.split != "target":
            continue
        entry = registry_entries[landscape.identifier]
        selected = grouped[landscape.identifier]
        epsilons = [row["epsilon"] for row in selected]
        deltas = [row["endpoint_l1"] for row in selected]
        if entry["stratum"] == "regular-sensitivity":
            for layer, alpha in (("analytic", 1.0), ("transport", alpha_source)):
                score = landscape_relative_score(
                    entry["C_primary"], epsilons, deltas, alpha=alpha
                )
                layer_scores[layer].append(
                    {"landscape_id": landscape.identifier, "relative_error_median": score}
                )
                for row in selected:
                    prediction = alpha * entry["C_primary"] * row["epsilon"]
                    target_prediction_rows.append(
                        {
                            "landscape_id": landscape.identifier,
                            "layer": layer,
                            "epsilon": row["epsilon"],
                            "observed_delta": row["endpoint_l1"],
                            "predicted_delta": prediction,
                            "absolute_error": abs(row["endpoint_l1"] - prediction),
                            "relative_error": abs(row["endpoint_l1"] - prediction)
                            / prediction,
                        }
                    )
        else:
            checks = {}
            for layer, alpha in (("analytic", 1.0), ("transport", alpha_source)):
                predictions = alpha * entry["C_primary"] * np.asarray(epsilons)
                absolute_errors = np.abs(np.asarray(deltas) - predictions)
                checks[layer] = {
                    "absolute_errors": absolute_errors.tolist(),
                    "passed": bool(np.all(absolute_errors <= DELTA_FLOOR)),
                }
            special.append(
                {
                    "landscape_id": landscape.identifier,
                    "stratum": entry["stratum"],
                    "checks": checks,
                }
            )
    gates = {}
    for layer, scores in layer_scores.items():
        result = evaluate_gate([item["relative_error_median"] for item in scores])
        gates[layer] = {
            "alpha": 1.0 if layer == "analytic" else alpha_source,
            "median": result.median,
            "q90": result.q90,
            "passed": result.passed,
            "landscape_count": result.landscape_count,
            "landscape_scores": scores,
        }
    return {
        "experiment_id": manifest.experiment_id,
        "primary_horizon": manifest.primary_horizon,
        "confirmatory_epsilons": list(manifest.confirmatory_epsilons),
        "stress_results_gating": False,
        "source_landscape_alphas": source_alphas,
        "alpha_source": alpha_source,
        "primary_gates": gates,
        "target_prediction_rows": target_prediction_rows,
        "special_target_strata": special,
        "secondary_results_location": "raw_rows",
    }


def _registries_match(locked: dict[str, Any], computed: dict[str, Any]) -> bool:
    if set(locked) != set(computed):
        return False
    locked_entries = _registry_by_id(locked)
    computed_entries = _registry_by_id(computed)
    if set(locked_entries) != set(computed_entries):
        return False
    locked_without = {key: value for key, value in locked.items() if key != "landscapes"}
    computed_without = {key: value for key, value in computed.items() if key != "landscapes"}
    if locked_without != computed_without:
        return False
    for identifier in locked_entries:
        left = locked_entries[identifier]
        right = computed_entries[identifier]
        for key in ("id", "split", "p0", "reward", "mutation", "stratum"):
            if left.get(key) != right.get(key):
                return False
        for key in ("C_primary", "K_primary"):
            if not np.isclose(left[key], right[key], rtol=0.0, atol=INHERITED_TOLERANCE):
                return False
        for horizon in right["horizons"]:
            if horizon not in left["horizons"]:
                return False
            for key in ("C_endpoint_l1", "C_max_path_l1", "K_kl_q_p"):
                if not np.isclose(
                    left["horizons"][horizon][key],
                    right["horizons"][horizon][key],
                    rtol=0.0,
                    atol=INHERITED_TOLERANCE,
                ):
                    return False
        if set(left["horizons"]) != set(right["horizons"]):
            return False
    return True


def validate_analytic_registry(
    locked: dict[str, Any], computed: dict[str, Any]
) -> None:
    if not _registries_match(locked, computed):
        raise ValueError("analytic registry does not match clean recomputation")


def validate_preregistration_bundle(bundle_path: str | Path) -> dict[str, Any]:
    """Validate locks and clean analytic predictions without generating outcomes."""
    bundle = Path(bundle_path)
    lock = validate_lock(bundle / "LOCK.json")
    manifest = load_manifest(bundle / "manifest.json")
    try:
        locked_registry = json.loads(
            (bundle / "analytic_registry.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("cannot read locked analytic registry") from error
    computed_registry = build_analytic_registry(manifest)
    validate_analytic_registry(locked_registry, computed_registry)
    strata = {
        name: sum(entry["stratum"] == name for entry in locked_registry["landscapes"])
        for name in ("analytic-zero", "low-sensitivity", "regular-sensitivity")
    }
    return {
        "schema_version": 1,
        "experiment_id": manifest.experiment_id,
        "kind": "preregistration-only-validation",
        "valid": True,
        "outcomes_generated": False,
        "landscape_count": len(manifest.landscapes),
        "source_count": sum(item.split == "source" for item in manifest.landscapes),
        "target_count": sum(item.split == "target" for item in manifest.landscapes),
        "strata": strata,
        "locked_file_count": len(lock["files"]),
    }


def execute_confirmatory(
    *,
    repo_path: str | Path,
    manifest_path: str | Path,
    registry_path: str | Path,
    lock_path: str | Path,
    approved_sha: str,
    output_path: str | Path,
) -> Path:
    """Future one-shot runner. ACL-002 preregistration preparation must not call it."""
    current_sha, tracked_dirty = _git_execution_state(repo_path)
    assert_execution_context(
        approved_sha=approved_sha,
        current_sha=current_sha,
        tracked_dirty=tracked_dirty,
        output_path=output_path,
    )
    lock = validate_lock(lock_path)
    manifest = load_manifest(manifest_path)
    try:
        locked_registry = json.loads(Path(registry_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("cannot read locked analytic registry") from error
    computed_registry = build_analytic_registry(manifest)
    validate_analytic_registry(locked_registry, computed_registry)
    raw_rows = generate_raw_rows(manifest)
    analysis = analyze_raw_rows(manifest, locked_registry, raw_rows)
    payload = {
        "schema_version": 1,
        "experiment_id": manifest.experiment_id,
        "kind": "confirmatory-mutation-stability-result",
        "approved_preregistration_sha": approved_sha,
        "preregistration_lock": lock,
        "randomness_used": False,
        "raw_rows": raw_rows,
        "analysis": analysis,
        "provenance": provenance(),
    }
    return write_json(output_path, payload)
