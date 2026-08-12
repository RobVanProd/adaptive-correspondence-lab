"""ACL-003 new-value second-order categorical protocol.

Importing and validating this module never evaluates a perturbed ACL-003 trajectory.
The future confirmatory runner is guarded by an approved Git SHA, a completely clean
worktree, valid locks, and a nonexistent output path.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from .acl002 import (
    assert_execution_context,
    git_execution_state,
    matrix_power_oracle_trajectory,
    mutation_trajectory,
    sha256_file,
    type7_quantile,
    validate_lock,
)
from .categorical_second_order import (
    l1_second_order_coefficient,
    l1_truncated_prediction,
    matrix_polynomial_second_order_trajectory,
    second_order_sensitivity_trajectory,
)
from .io import provenance, write_json
from .simplex import validate_reward, validate_simplex

FloatArray = NDArray[np.float64]
LandscapeRole = Literal["confirmatory-target", "software-control"]

ACL003_ETA = 0.05
ACL003_PRIMARY_HORIZON = 20
ACL003_SECONDARY_HORIZONS = (1, 5, 50)
ACL003_EPSILON_GRID = (0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)
ACL003_NUMERICAL_CONTROL_EPSILONS = (1e-4, 3e-4)
ACL003_CONFIRMATORY_EPSILONS = (1e-3, 3e-3, 1e-2)
ACL003_STRESS_EPSILONS = (3e-2, 1e-1)
ACL003_MEDIAN_GATE = 0.10
ACL003_Q90_GATE = 0.20
ACL003_DELTA_FLOOR = 2e-12
ACL003_MATRIX_ORACLE_TOLERANCE = 5e-13
ACL003_STATE_ORACLE_TOLERANCE = 5e-13
ACL003_FIRST_ORACLE_TOLERANCE = 5e-11
ACL003_CURVATURE_ORACLE_TOLERANCE = 2e-9
CATALOG_NOVELTY_ATOL = 1e-15
ACL002_REFERENCE_MANIFEST_SHA256 = (
    "6a9e4e0a931277b1f5c464807d0bcacee3ccb684269843f8245a83ae88110741"
)
ACL003_LOCKED_FILES = frozenset(
    {
        "ANALYSIS_PLAN.md",
        "DERIVATION.md",
        "PREREGISTRATION.md",
        "README.md",
        "analytic_registry.json",
        "manifest.json",
    }
)


@dataclass(frozen=True)
class ACL003Landscape:
    identifier: str
    role: LandscapeRole
    p0_name: str
    reward_name: str
    mutation_name: str
    p0: FloatArray
    reward: FloatArray
    mutation: FloatArray


@dataclass(frozen=True)
class ACL003Manifest:
    experiment_id: str
    eta: float
    primary_horizon: int
    secondary_horizons: tuple[int, ...]
    epsilon_grid: tuple[float, ...]
    numerical_control_epsilons: tuple[float, ...]
    confirmatory_epsilons: tuple[float, ...]
    stress_epsilons: tuple[float, ...]
    landscapes: tuple[ACL003Landscape, ...]
    identity_control_ids: tuple[str, ...]
    expected_low_ids: tuple[str, ...]
    benchmark_scope: str
    inference_scope: str
    transport_scope: str
    novelty_reference_manifest_sha256: str | None
    raw: dict[str, Any]

    @property
    def horizons(self) -> tuple[int, ...]:
        return tuple(sorted({self.primary_horizon, *self.secondary_horizons}))


def _validate_mutation(value: Any, dimension: int, *, name: str) -> FloatArray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (dimension, dimension):
        raise ValueError(f"{name} must be square with state dimension")
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0.0):
        raise ValueError(f"{name} must be finite and non-negative")
    if not np.allclose(np.sum(matrix, axis=1), 1.0, rtol=0.0, atol=2e-14):
        raise ValueError(f"{name} must be row-stochastic")
    return matrix.copy()


def _float_tuple(payload: dict[str, Any], key: str) -> tuple[float, ...]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list")
    result = tuple(float(item) for item in value)
    if not all(np.isfinite(item) for item in result):
        raise ValueError(f"{key} must contain finite values")
    return result


def validate_manifest_dict(payload: dict[str, Any]) -> ACL003Manifest:
    """Validate generic toy manifests and activate exact constants for ACL-003."""
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported ACL-003 manifest schema")
    experiment_id = payload.get("experiment_id")
    if not isinstance(experiment_id, str) or not experiment_id:
        raise ValueError("experiment_id must be a non-empty string")
    if payload.get("randomness") != "none" or payload.get("vector_convention") != "row":
        raise ValueError("ACL-003 requires deterministic row-vector semantics")
    eta = float(payload.get("eta"))
    if not np.isfinite(eta) or eta < 0.0:
        raise ValueError("eta must be finite and non-negative")
    primary_horizon = payload.get("primary_horizon")
    secondary = payload.get("secondary_horizons")
    if (
        isinstance(primary_horizon, bool)
        or not isinstance(primary_horizon, int)
        or primary_horizon < 1
        or not isinstance(secondary, list)
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in secondary
        )
    ):
        raise ValueError("horizons must be positive integers")
    epsilon_grid = _float_tuple(payload, "epsilon_grid")
    numerical_control = _float_tuple(payload, "numerical_control_epsilons")
    confirmatory = _float_tuple(payload, "confirmatory_epsilons")
    stress = _float_tuple(payload, "stress_epsilons")
    if (
        tuple(sorted(set(epsilon_grid))) != epsilon_grid
        or epsilon_grid[0] != 0.0
        or set(epsilon_grid[1:]) != set(numerical_control + confirmatory + stress)
        or set(numerical_control) & set(confirmatory)
        or set(numerical_control) & set(stress)
        or set(confirmatory) & set(stress)
    ):
        raise ValueError("epsilon regions must be disjoint and partition positive grid")

    numerical = payload.get("numerical_policy")
    gates = payload.get("gates")
    if not isinstance(numerical, dict) or not isinstance(gates, dict):
        raise ValueError("numerical_policy and gates must be objects")
    required_numerical = {
        "delta_floor": ACL003_DELTA_FLOOR,
        "quantile_method": "linear",
        "quantile_definition": "Hyndman-Fan Type 7",
        "matrix_oracle_max_abs_tolerance": ACL003_MATRIX_ORACLE_TOLERANCE,
        "second_order_state_oracle_tolerance": ACL003_STATE_ORACLE_TOLERANCE,
        "second_order_first_oracle_tolerance": ACL003_FIRST_ORACLE_TOLERANCE,
        "second_order_curvature_oracle_tolerance": ACL003_CURVATURE_ORACLE_TOLERANCE,
    }
    if any(numerical.get(key) != value for key, value in required_numerical.items()):
        raise ValueError("ACL-003 numerical policy mismatch")
    required_gates = {
        "within_landscape_reduction": "maximum-over-confirmatory-epsilons",
        "target_landscape_median_relative_error_max": ACL003_MEDIAN_GATE,
        "target_landscape_q90_relative_error_max": ACL003_Q90_GATE,
    }
    if any(gates.get(key) != value for key, value in required_gates.items()):
        raise ValueError("ACL-003 gate mismatch")

    state_payload = payload.get("states")
    reward_payload = payload.get("rewards")
    mutation_payload = payload.get("mutation_matrices")
    catalogs = (state_payload, reward_payload, mutation_payload)
    if not all(isinstance(item, dict) and item for item in catalogs):
        raise ValueError("state, reward, and mutation catalogs must be non-empty objects")
    states = {
        name: validate_simplex(value, name=f"state {name}", strictly_positive=True)
        for name, value in state_payload.items()
    }
    dimensions = {value.size for value in states.values()}
    if len(dimensions) != 1:
        raise ValueError("all states must share a dimension")
    dimension = next(iter(dimensions))
    rewards = {
        name: validate_reward(value, dimension) for name, value in reward_payload.items()
    }
    mutations = {
        name: _validate_mutation(value, dimension, name=f"mutation {name}")
        for name, value in mutation_payload.items()
    }
    landscape_payload = payload.get("landscapes")
    if not isinstance(landscape_payload, list) or not landscape_payload:
        raise ValueError("landscapes must be a non-empty list")
    identifiers: set[str] = set()
    landscapes: list[ACL003Landscape] = []
    for item in landscape_payload:
        if not isinstance(item, dict):
            raise ValueError("each landscape must be an object")
        identifier = item.get("id")
        role = item.get("role")
        p0_name = item.get("p0")
        reward_name = item.get("reward")
        mutation_name = item.get("mutation")
        if not isinstance(identifier, str) or not identifier or identifier in identifiers:
            raise ValueError("landscape IDs must be non-empty and unique")
        if role not in {"confirmatory-target", "software-control"}:
            raise ValueError("landscape role is invalid")
        if p0_name not in states or reward_name not in rewards or mutation_name not in mutations:
            raise ValueError(f"landscape {identifier} has unknown catalog reference")
        identifiers.add(identifier)
        landscapes.append(
            ACL003Landscape(
                identifier=identifier,
                role=role,
                p0_name=p0_name,
                reward_name=reward_name,
                mutation_name=mutation_name,
                p0=states[p0_name],
                reward=rewards[reward_name],
                mutation=mutations[mutation_name],
            )
        )
    controls = tuple(payload.get("identity_control_ids", []))
    low_ids = tuple(payload.get("predeclared_low_sensitivity_ids", []))
    if not set(controls + low_ids) <= identifiers or set(controls) & set(low_ids):
        raise ValueError("control and low-sensitivity IDs must be disjoint known landscapes")
    for identifier in controls:
        landscape = next(item for item in landscapes if item.identifier == identifier)
        if landscape.role != "software-control" or not np.array_equal(
            landscape.mutation, np.eye(dimension)
        ):
            raise ValueError("identity control must use exact identity mutation")

    benchmark_scope = payload.get("benchmark_scope")
    inference_scope = payload.get("inference_scope")
    transport_scope = payload.get("transport_scope")
    scopes = (benchmark_scope, inference_scope, transport_scope)
    if not all(isinstance(value, str) and value for value in scopes):
        raise ValueError("scope fields must be non-empty strings")
    reference_hash = payload.get("novelty_reference_manifest_sha256")
    if reference_hash is not None and (
        not isinstance(reference_hash, str)
        or len(reference_hash) != 64
        or any(character not in "0123456789abcdef" for character in reference_hash)
    ):
        raise ValueError("novelty reference manifest hash must be lowercase SHA-256")
    if experiment_id == "ACL-003":
        design_matches = (
            eta == ACL003_ETA
            and primary_horizon == ACL003_PRIMARY_HORIZON
            and tuple(secondary) == ACL003_SECONDARY_HORIZONS
            and epsilon_grid == ACL003_EPSILON_GRID
            and numerical_control == ACL003_NUMERICAL_CONTROL_EPSILONS
            and confirmatory == ACL003_CONFIRMATORY_EPSILONS
            and stress == ACL003_STRESS_EPSILONS
            and sum(item.role == "confirmatory-target" for item in landscapes) == 16
            and sum(item.role == "software-control" for item in landscapes) == 1
            and controls == ("C01",)
            and benchmark_scope == "deterministic-new-value-held-out-benchmark"
            and inference_scope == "descriptive-criteria-not-population-confidence"
            and transport_scope == "zero-fit-new-value-within-categorical-class"
            and reference_hash == ACL002_REFERENCE_MANIFEST_SHA256
        )
        if not design_matches:
            raise ValueError("ACL-003 design constants mismatch")
    return ACL003Manifest(
        experiment_id=experiment_id,
        eta=eta,
        primary_horizon=primary_horizon,
        secondary_horizons=tuple(secondary),
        epsilon_grid=epsilon_grid,
        numerical_control_epsilons=numerical_control,
        confirmatory_epsilons=confirmatory,
        stress_epsilons=stress,
        landscapes=tuple(landscapes),
        identity_control_ids=controls,
        expected_low_ids=low_ids,
        benchmark_scope=benchmark_scope,
        inference_scope=inference_scope,
        transport_scope=transport_scope,
        novelty_reference_manifest_sha256=reference_hash,
        raw=payload,
    )


def load_manifest(path: str | Path) -> ACL003Manifest:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read ACL-003 manifest: {path}") from error
    return validate_manifest_dict(payload)


def _catalog_values(payload: dict[str, Any], key: str) -> list[FloatArray]:
    return [np.asarray(value, dtype=np.float64) for value in payload.get(key, {}).values()]


def _matches_any(value: FloatArray, references: list[FloatArray]) -> bool:
    return any(
        value.shape == reference.shape
        and np.allclose(value, reference, rtol=0.0, atol=CATALOG_NOVELTY_ATOL)
        for reference in references
    )


def validate_catalog_novelty(
    manifest: ACL003Manifest, reference_payload: dict[str, Any]
) -> dict[str, Any]:
    """Reject numeric reuse by any hypothesis-bearing ACL-003 catalog value."""
    hypothesis = [item for item in manifest.landscapes if item.role == "confirmatory-target"]
    used_states = {item.p0_name: item.p0 for item in hypothesis}
    used_rewards = {item.reward_name: item.reward for item in hypothesis}
    used_mutations = {item.mutation_name: item.mutation for item in hypothesis}
    reference_states = _catalog_values(reference_payload, "states")
    reference_rewards = _catalog_values(reference_payload, "rewards")
    reference_mutations = _catalog_values(reference_payload, "mutation_matrices")
    overlaps = {
        "hypothesis_state_overlap_count": sum(
            _matches_any(value, reference_states) for value in used_states.values()
        ),
        "hypothesis_reward_overlap_count": sum(
            _matches_any(value, reference_rewards) for value in used_rewards.values()
        ),
        "hypothesis_mutation_overlap_count": sum(
            _matches_any(value, reference_mutations) for value in used_mutations.values()
        ),
    }
    if overlaps["hypothesis_state_overlap_count"]:
        raise ValueError("ACL-003 state catalog overlaps ACL-002 numerically")
    if overlaps["hypothesis_reward_overlap_count"]:
        raise ValueError("ACL-003 reward catalog overlaps ACL-002 numerically")
    if overlaps["hypothesis_mutation_overlap_count"]:
        raise ValueError("ACL-003 mutation catalog overlaps ACL-002 numerically")
    return {
        **overlaps,
        "hypothesis_state_count": len(used_states),
        "hypothesis_reward_count": len(used_rewards),
        "hypothesis_mutation_count": len(used_mutations),
        "comparison_atol": CATALOG_NOVELTY_ATOL,
        "identity_control_exempt": True,
    }


def _region(manifest: ACL003Manifest, epsilon: float) -> str:
    if epsilon == 0.0:
        return "zero"
    if epsilon in manifest.numerical_control_epsilons:
        return "numerical-control"
    if epsilon in manifest.confirmatory_epsilons:
        return "confirmatory"
    if epsilon in manifest.stress_epsilons:
        return "stress"
    raise ValueError("epsilon is outside ACL-003 regions")


def build_analytic_registry(manifest: ACL003Manifest) -> dict[str, Any]:
    """Compute clean first/second sensitivities only; never perturbed outcomes."""
    entries = []
    observed_low = []
    max_horizon = max(manifest.horizons)
    for landscape in manifest.landscapes:
        trace = second_order_sensitivity_trajectory(
            landscape.p0,
            landscape.reward,
            landscape.mutation,
            eta=manifest.eta,
            steps=max_horizon,
        )
        oracle = matrix_polynomial_second_order_trajectory(
            landscape.p0,
            landscape.reward,
            landscape.mutation,
            eta=manifest.eta,
            steps=max_horizon,
        )
        errors = {
            "state": float(np.max(np.abs(trace.states - oracle.states))),
            "first": float(np.max(np.abs(trace.first - oracle.first))),
            "second": float(np.max(np.abs(trace.second - oracle.second))),
        }
        if errors["state"] > ACL003_STATE_ORACLE_TOLERANCE:
            raise FloatingPointError("ACL-003 clean state oracle mismatch")
        if errors["first"] > ACL003_FIRST_ORACLE_TOLERANCE:
            raise FloatingPointError("ACL-003 clean first oracle mismatch")
        if errors["second"] > ACL003_CURVATURE_ORACLE_TOLERANCE:
            raise FloatingPointError("ACL-003 clean curvature oracle mismatch")
        primary_first = trace.first[manifest.primary_horizon]
        primary_second = trace.second[manifest.primary_horizon]
        c_primary = float(np.linalg.norm(primary_first, ord=1))
        if landscape.identifier in manifest.identity_control_ids:
            stratum = "identity-control"
        elif c_primary * max(manifest.confirmatory_epsilons) < ACL003_DELTA_FLOOR:
            stratum = "low-sensitivity"
            observed_low.append(landscape.identifier)
        else:
            stratum = "regular-sensitivity"
        horizons = {}
        for horizon in manifest.horizons:
            coefficient, zero_coordinates = l1_second_order_coefficient(
                trace.first[horizon], trace.second[horizon]
            )
            horizons[str(horizon)] = {
                "C_endpoint_l1": float(np.linalg.norm(trace.first[horizon], ord=1)),
                "second_order_l1_coefficient": coefficient,
                "second_derivative_l1": float(np.linalg.norm(trace.second[horizon], ord=1)),
                "zero_first_derivative_coordinates": list(zero_coordinates),
            }
        entries.append(
            {
                "id": landscape.identifier,
                "role": landscape.role,
                "p0": landscape.p0_name,
                "reward": landscape.reward_name,
                "mutation": landscape.mutation_name,
                "stratum": stratum,
                "C_primary": c_primary,
                "second_derivative_l1_primary": float(np.linalg.norm(primary_second, ord=1)),
                "horizons": horizons,
                "oracle_max_absolute_errors": errors,
            }
        )
    if tuple(observed_low) != manifest.expected_low_ids:
        raise ValueError(
            "ACL-003 analytic low-sensitivity IDs differ from predeclared IDs: "
            f"observed={observed_low}, expected={list(manifest.expected_low_ids)}"
        )
    return {
        "schema_version": 1,
        "experiment_id": manifest.experiment_id,
        "kind": "clean-second-order-analytic-registry",
        "prediction_kind": "zero-fit-second-order-truncated-vector",
        "outcomes_generated": False,
        "landscapes": entries,
    }


def _registry_entry(registry: dict[str, Any], identifier: str) -> dict[str, Any]:
    return next(item for item in registry["landscapes"] if item["id"] == identifier)


def generate_raw_rows(manifest: ACL003Manifest) -> list[dict[str, Any]]:
    """Generate outcomes; only the guarded runner may call this on ACL-003."""
    rows = []
    max_horizon = max(manifest.horizons)
    registry = build_analytic_registry(manifest)
    for landscape in manifest.landscapes:
        trace = second_order_sensitivity_trajectory(
            landscape.p0,
            landscape.reward,
            landscape.mutation,
            eta=manifest.eta,
            steps=max_horizon,
        )
        registry_entry = _registry_entry(registry, landscape.identifier)
        for epsilon in manifest.epsilon_grid:
            iterative = mutation_trajectory(
                landscape.p0,
                landscape.reward,
                landscape.mutation,
                eta=manifest.eta,
                epsilon=epsilon,
                steps=max_horizon,
            )
            oracle = matrix_power_oracle_trajectory(
                landscape.p0,
                landscape.reward,
                landscape.mutation,
                eta=manifest.eta,
                epsilon=epsilon,
                steps=max_horizon,
            )
            oracle_error = float(np.max(np.abs(iterative - oracle)))
            if oracle_error > ACL003_MATRIX_ORACLE_TOLERANCE:
                raise FloatingPointError("ACL-003 iterative and matrix oracle disagree")
            for horizon in manifest.horizons:
                differences = iterative[: horizon + 1] - trace.states[: horizon + 1]
                endpoint = float(np.linalg.norm(differences[horizon], ord=1))
                path = float(np.max(np.linalg.norm(differences, ord=1, axis=1)))
                first_prediction = float(
                    np.linalg.norm(epsilon * trace.first[horizon], ord=1)
                )
                second_prediction = l1_truncated_prediction(
                    trace.first[horizon], trace.second[horizon], epsilon=epsilon
                )
                first_path = max(
                    float(np.linalg.norm(epsilon * trace.first[step], ord=1))
                    for step in range(horizon + 1)
                )
                second_path = max(
                    l1_truncated_prediction(
                        trace.first[step], trace.second[step], epsilon=epsilon
                    )
                    for step in range(horizon + 1)
                )
                rows.append(
                    {
                        "landscape_id": landscape.identifier,
                        "role": landscape.role,
                        "stratum": registry_entry["stratum"],
                        "horizon": horizon,
                        "epsilon": epsilon,
                        "region": _region(manifest, epsilon),
                        "endpoint_l1": endpoint,
                        "max_path_l1": path,
                        "first_order_prediction": first_prediction,
                        "second_order_prediction": second_prediction,
                        "first_order_max_path_prediction": first_path,
                        "second_order_max_path_prediction": second_path,
                        "clean_terminal": trace.states[horizon].tolist(),
                        "perturbed_terminal": iterative[horizon].tolist(),
                        "matrix_oracle_max_absolute_error": oracle_error,
                    }
                )
    return sorted(rows, key=lambda row: (row["landscape_id"], row["horizon"], row["epsilon"]))


def _relative_error(observed: float, predicted: float) -> float | None:
    if predicted < ACL003_DELTA_FLOOR:
        return None
    return abs(observed - predicted) / predicted


def analyze_raw_rows(
    manifest: ACL003Manifest, registry: dict[str, Any], raw_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Apply only frozen ACL-003 gates; numerical-control and stress stay non-gating."""
    scores = []
    prediction_rows = []
    for landscape in manifest.landscapes:
        entry = _registry_entry(registry, landscape.identifier)
        landscape_rows = [
            row
            for row in raw_rows
            if row["landscape_id"] == landscape.identifier
            and row["horizon"] == manifest.primary_horizon
        ]
        for row in landscape_rows:
            if row["epsilon"] == 0.0:
                continue
            prediction_rows.append(
                {
                    "landscape_id": landscape.identifier,
                    "epsilon": row["epsilon"],
                    "region": row["region"],
                    "first_order_absolute_relative_error": _relative_error(
                        row["endpoint_l1"], row["first_order_prediction"]
                    ),
                    "second_order_absolute_relative_error": _relative_error(
                        row["endpoint_l1"], row["second_order_prediction"]
                    ),
                }
            )
        if landscape.role != "confirmatory-target" or entry["stratum"] != "regular-sensitivity":
            continue
        confirmatory = [
            row for row in landscape_rows if row["epsilon"] in manifest.confirmatory_epsilons
        ]
        errors = [
            _relative_error(row["endpoint_l1"], row["second_order_prediction"])
            for row in confirmatory
        ]
        if any(value is None for value in errors):
            raise ValueError("regular ACL-003 target has undefined primary relative error")
        scores.append(
            {
                "landscape_id": landscape.identifier,
                "relative_error_max": max(float(value) for value in errors if value is not None),
            }
        )
    score_values = [row["relative_error_max"] for row in scores]
    if not score_values:
        raise ValueError("ACL-003 has no regular confirmatory target scores")
    median = type7_quantile(score_values, 0.5)
    q90 = type7_quantile(score_values, 0.9)
    controls = []
    for identifier in manifest.identity_control_ids:
        values = [
            row["endpoint_l1"] for row in raw_rows if row["landscape_id"] == identifier
        ]
        maximum = max(values)
        controls.append(
            {
                "landscape_id": identifier,
                "maximum_absolute_endpoint_l1": maximum,
                "passed": maximum <= ACL003_DELTA_FLOOR,
            }
        )
    controls_passed = all(item["passed"] for item in controls)
    primary_passed = median <= ACL003_MEDIAN_GATE and q90 <= ACL003_Q90_GATE
    verdict = "INVALID" if not controls_passed else "PASS" if primary_passed else "FAIL"
    return {
        "experiment_id": manifest.experiment_id,
        "primary_horizon": manifest.primary_horizon,
        "primary_gate": {
            "landscape_scores": scores,
            "landscape_count": len(scores),
            "median": median,
            "q90": q90,
            "median_threshold": ACL003_MEDIAN_GATE,
            "q90_threshold": ACL003_Q90_GATE,
            "passed": primary_passed,
        },
        "prediction_rows": prediction_rows,
        "software_controls": controls,
        "software_controls_passed": controls_passed,
        "instrument_valid": controls_passed,
        "verdict": verdict,
        "numerical_control_results_gating": False,
        "stress_results_gating": False,
        "secondary_results_location": "raw_rows",
    }


def validate_analytic_registry(
    locked: dict[str, Any], computed: dict[str, Any]
) -> None:
    if locked != computed:
        raise ValueError("ACL-003 analytic registry does not match clean recomputation")


def validate_preregistration_bundle(
    bundle_path: str | Path,
    *,
    reference_path: str | Path = "preregistrations/ACL-002/manifest.json",
) -> dict[str, Any]:
    bundle = Path(bundle_path)
    lock = validate_lock(bundle / "LOCK.json")
    if lock.get("outcomes_generated") is not False:
        raise ValueError("ACL-003 lock must declare outcomes_generated=false")
    manifest = load_manifest(bundle / "manifest.json")
    if manifest.experiment_id == "ACL-003" and (
        lock.get("experiment_id") != "ACL-003"
        or lock.get("kind") != "preregistration-bundle-lock"
        or set(lock["files"]) != ACL003_LOCKED_FILES
    ):
        raise ValueError("ACL-003 lock must contain the exact frozen file set")
    try:
        reference_hash = sha256_file(reference_path)
        locked_registry = json.loads(
            (bundle / "analytic_registry.json").read_text(encoding="utf-8")
        )
        reference = json.loads(Path(reference_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("cannot read ACL-003 registry or reference manifest") from error
    if (
        manifest.novelty_reference_manifest_sha256 is not None
        and reference_hash != manifest.novelty_reference_manifest_sha256
    ):
        raise ValueError("ACL-003 novelty reference manifest hash mismatch")
    computed = build_analytic_registry(manifest)
    validate_analytic_registry(locked_registry, computed)
    novelty = validate_catalog_novelty(manifest, reference)
    strata = {entry["stratum"]: 0 for entry in computed["landscapes"]}
    for entry in computed["landscapes"]:
        strata[entry["stratum"]] += 1
    return {
        "schema_version": 1,
        "experiment_id": manifest.experiment_id,
        "kind": "preregistration-only-validation",
        "valid": True,
        "outcomes_generated": False,
        "landscape_count": len(manifest.landscapes),
        "confirmatory_target_count": sum(
            item.role == "confirmatory-target" for item in manifest.landscapes
        ),
        "software_control_count": sum(
            item.role == "software-control" for item in manifest.landscapes
        ),
        "horizons": list(manifest.horizons),
        "numerical_control_epsilons": list(manifest.numerical_control_epsilons),
        "confirmatory_epsilons": list(manifest.confirmatory_epsilons),
        "stress_epsilons": list(manifest.stress_epsilons),
        "strata": dict(sorted(strata.items())),
        "catalog_novelty": novelty,
        "locked_file_count": len(lock["files"]),
    }


def execute_confirmatory(
    *,
    repo_path: str | Path,
    bundle_path: str | Path,
    reference_path: str | Path,
    approved_sha: str,
    output_path: str | Path,
) -> Path:
    """Execute ACL-003 once only after exact-SHA approval."""
    current_sha, worktree_dirty = git_execution_state(repo_path)
    assert_execution_context(
        approved_sha=approved_sha,
        current_sha=current_sha,
        worktree_dirty=worktree_dirty,
        output_path=output_path,
    )
    bundle = Path(bundle_path)
    validation = validate_preregistration_bundle(bundle, reference_path=reference_path)
    manifest = load_manifest(bundle / "manifest.json")
    registry = json.loads((bundle / "analytic_registry.json").read_text(encoding="utf-8"))
    raw_rows = generate_raw_rows(manifest)
    analysis = analyze_raw_rows(manifest, registry, raw_rows)
    payload = {
        "schema_version": 1,
        "experiment_id": manifest.experiment_id,
        "kind": "confirmatory-new-value-second-order-result",
        "approved_preregistration_sha": approved_sha,
        "preregistration_validation": validation,
        "randomness_used": False,
        "target_refit": False,
        "benchmark_scope": manifest.benchmark_scope,
        "inference_scope": manifest.inference_scope,
        "transport_scope": manifest.transport_scope,
        "raw_rows": raw_rows,
        "analysis": analysis,
        "provenance": provenance(),
    }
    return write_json(output_path, payload)
