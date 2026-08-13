"""ACL-008 no-refit second-order transport into Burg mirror geometry."""

from __future__ import annotations

import json
import platform
from pathlib import Path
from typing import Any

import numpy as np

from .acl002 import (
    assert_execution_context,
    git_execution_state,
    sha256_file,
    validate_lock,
)
from .acl003 import (
    ACL003_CONFIRMATORY_EPSILONS,
    ACL003_CURVATURE_ORACLE_TOLERANCE,
    ACL003_DELTA_FLOOR,
    ACL003_EPSILON_GRID,
    ACL003_ETA,
    ACL003_FIRST_ORACLE_TOLERANCE,
    ACL003_MATRIX_ORACLE_TOLERANCE,
    ACL003_MEDIAN_GATE,
    ACL003_NUMERICAL_CONTROL_EPSILONS,
    ACL003_PRIMARY_HORIZON,
    ACL003_Q90_GATE,
    ACL003_SECONDARY_HORIZONS,
    ACL003_STATE_ORACLE_TOLERANCE,
    ACL003_STRESS_EPSILONS,
    ACL003Manifest,
    analyze_raw_rows,
    validate_catalog_novelty,
)
from .acl003 import (
    validate_manifest_dict as validate_acl003_manifest_dict,
)
from .burg_mirror import (
    burg_perturbed_trajectory,
    burg_perturbed_trajectory_polynomial_oracle,
    burg_second_order_sensitivity_trajectory,
)
from .categorical_second_order import (
    l1_second_order_coefficient,
    l1_truncated_prediction,
)
from .io import provenance, write_json

ACL008_SOURCE_EVIDENCE = {
    "approved_preregistration_sha": "501464f3f6be07f6d813d94aefb818c461a3d5c7",
    "evidence_commit": "b15d77600369d559cb586a3bb54924737758e038",
    "evidence_artifact": (
        "evidence/ACL-003-confirmatory-"
        "501464f3f6be07f6d813d94aefb818c461a3d5c7.json"
    ),
    "evidence_sha256": "1f80c3f5aba4089c67bbfec1ddd6eff53f7a6d42c658436dff0f7c82a1cf8c99",
    "report_summary": "analysis/ACL-003-confirmatory/summary.json",
    "report_summary_sha256": (
        "eee21af8f75c7eb5d3a35fdb9d53b1549f275eca52c745a85c130538583128f4"
    ),
    "source_verdict": "PASS",
    "source_median": 0.0014843120912351297,
    "source_q90": 0.007387117284289386,
}
ACL008_REFERENCE_MANIFEST_SHA256 = (
    "47b2f2bc2ecd75b17d53286334e5268df93654c84c8e556ae93e129a0bc0c37a"
)
ACL008_ENVIRONMENT = {
    "python_implementation": "CPython",
    "python_version": "3.13.14",
    "numpy_version": "2.5.2",
    "platform_system": "Windows",
    "platform_machine": "AMD64",
}
ACL008_TARGET_IDS = tuple([f"B{index:02d}" for index in range(1, 17)] + ["C01"])
ACL008_REGISTRY_ATOL = 2e-12
ACL008_REGISTRY_RTOL = 2e-12
ACL008_LOCKED_FILES = frozenset(
    {
        "ANALYSIS_PLAN.md",
        "DERIVATION.md",
        "PREREGISTRATION.md",
        "README.md",
        "analytic_registry.json",
        "manifest.json",
    }
)


def validate_manifest_dict(payload: dict[str, Any]) -> ACL003Manifest:
    manifest = validate_acl003_manifest_dict(payload)
    if manifest.experiment_id == "ACL-008":
        numerical = payload.get("numerical_policy", {})
        exact = (
            payload.get("source_experiment") == "ACL-003"
            and payload.get("source_evidence") == ACL008_SOURCE_EVIDENCE
            and manifest.novelty_reference_manifest_sha256
            == ACL008_REFERENCE_MANIFEST_SHA256
            and payload.get("native_geometry")
            == "Burg-log-barrier-Hessian-diag-one-over-p-squared"
            and manifest.eta == ACL003_ETA
            and manifest.primary_horizon == ACL003_PRIMARY_HORIZON
            and manifest.secondary_horizons == ACL003_SECONDARY_HORIZONS
            and manifest.epsilon_grid == ACL003_EPSILON_GRID
            and manifest.numerical_control_epsilons
            == ACL003_NUMERICAL_CONTROL_EPSILONS
            and manifest.confirmatory_epsilons == ACL003_CONFIRMATORY_EPSILONS
            and manifest.stress_epsilons == ACL003_STRESS_EPSILONS
            and tuple(item.identifier for item in manifest.landscapes)
            == ACL008_TARGET_IDS
            and manifest.identity_control_ids == ("C01",)
            and not manifest.expected_low_ids
            and manifest.benchmark_scope
            == "deterministic-new-value-non-fisher-mirror-benchmark"
            and manifest.inference_scope
            == "descriptive-criteria-not-population-confidence"
            and manifest.transport_scope
            == "ACL-003-to-Burg-unchanged-zero-fit-second-order-rule"
            and payload.get("confirmatory_environment") == ACL008_ENVIRONMENT
            and numerical.get("analytic_registry_atol") == ACL008_REGISTRY_ATOL
            and numerical.get("analytic_registry_rtol") == ACL008_REGISTRY_RTOL
        )
        if not exact:
            raise ValueError("ACL-008 design constants mismatch")
    return manifest


def load_manifest(path: str | Path) -> ACL003Manifest:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("cannot read ACL-008 manifest") from error
    return validate_manifest_dict(payload)


def _registry_entry(registry: dict[str, Any], identifier: str) -> dict[str, Any]:
    return next(item for item in registry["landscapes"] if item["id"] == identifier)


def _region(manifest: ACL003Manifest, epsilon: float) -> str:
    if epsilon == 0.0:
        return "zero"
    if epsilon in manifest.numerical_control_epsilons:
        return "numerical-control"
    if epsilon in manifest.confirmatory_epsilons:
        return "confirmatory"
    if epsilon in manifest.stress_epsilons:
        return "stress"
    raise ValueError("epsilon is outside ACL-008 regions")


def build_analytic_registry(manifest: ACL003Manifest) -> dict[str, Any]:
    """Build clean/zero-epsilon derivatives without target outcomes."""
    entries: list[dict[str, Any]] = []
    observed_low: list[str] = []
    max_horizon = max(manifest.horizons)
    for landscape in manifest.landscapes:
        trace = burg_second_order_sensitivity_trajectory(
            landscape.p0,
            landscape.reward,
            landscape.mutation,
            eta=manifest.eta,
            steps=max_horizon,
        )
        clean_oracle = burg_perturbed_trajectory_polynomial_oracle(
            landscape.p0,
            landscape.reward,
            landscape.mutation,
            eta=manifest.eta,
            epsilon=0.0,
            steps=max_horizon,
        )
        state_error = float(np.max(np.abs(trace.states - clean_oracle)))
        if state_error > ACL003_STATE_ORACLE_TOLERANCE:
            raise FloatingPointError("ACL-008 clean state oracle mismatch")
        first = trace.first[manifest.primary_horizon]
        second = trace.second[manifest.primary_horizon]
        c_primary = float(np.linalg.norm(first, ord=1))
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
                "second_derivative_l1": float(
                    np.linalg.norm(trace.second[horizon], ord=1)
                ),
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
                "second_derivative_l1_primary": float(
                    np.linalg.norm(second, ord=1)
                ),
                "horizons": horizons,
                "clean_polynomial_oracle_max_absolute_error": state_error,
            }
        )
    if tuple(observed_low) != manifest.expected_low_ids:
        raise ValueError(
            "ACL-008 analytic low-sensitivity IDs differ from predeclared IDs: "
            f"observed={observed_low}, expected={list(manifest.expected_low_ids)}"
        )
    return {
        "schema_version": 1,
        "experiment_id": manifest.experiment_id,
        "kind": "clean-Burg-second-order-analytic-registry",
        "prediction_kind": "zero-fit-second-order-truncated-vector",
        "native_geometry": "Burg-log-barrier",
        "outcomes_generated": False,
        "target_refit": False,
        "landscapes": entries,
    }


def generate_raw_rows(manifest: ACL003Manifest) -> list[dict[str, Any]]:
    """Generate epsilon outcomes; only the approved guarded runner may call this."""
    rows: list[dict[str, Any]] = []
    max_horizon = max(manifest.horizons)
    registry = build_analytic_registry(manifest)
    for landscape in manifest.landscapes:
        trace = burg_second_order_sensitivity_trajectory(
            landscape.p0,
            landscape.reward,
            landscape.mutation,
            eta=manifest.eta,
            steps=max_horizon,
        )
        entry = _registry_entry(registry, landscape.identifier)
        for epsilon in manifest.epsilon_grid:
            iterative = burg_perturbed_trajectory(
                landscape.p0,
                landscape.reward,
                landscape.mutation,
                eta=manifest.eta,
                epsilon=epsilon,
                steps=max_horizon,
            )
            oracle = burg_perturbed_trajectory_polynomial_oracle(
                landscape.p0,
                landscape.reward,
                landscape.mutation,
                eta=manifest.eta,
                epsilon=epsilon,
                steps=max_horizon,
            )
            oracle_error = float(np.max(np.abs(iterative - oracle)))
            if oracle_error > ACL003_MATRIX_ORACLE_TOLERANCE:
                raise FloatingPointError("ACL-008 iterative and polynomial oracle disagree")
            for horizon in manifest.horizons:
                differences = iterative[: horizon + 1] - trace.states[: horizon + 1]
                rows.append(
                    {
                        "landscape_id": landscape.identifier,
                        "role": landscape.role,
                        "stratum": entry["stratum"],
                        "horizon": horizon,
                        "epsilon": epsilon,
                        "region": _region(manifest, epsilon),
                        "endpoint_l1": float(
                            np.linalg.norm(differences[horizon], ord=1)
                        ),
                        "max_path_l1": float(
                            np.max(np.linalg.norm(differences, ord=1, axis=1))
                        ),
                        "first_order_prediction": float(
                            np.linalg.norm(epsilon * trace.first[horizon], ord=1)
                        ),
                        "second_order_prediction": l1_truncated_prediction(
                            trace.first[horizon],
                            trace.second[horizon],
                            epsilon=epsilon,
                        ),
                        "first_order_max_path_prediction": max(
                            float(np.linalg.norm(epsilon * trace.first[index], ord=1))
                            for index in range(horizon + 1)
                        ),
                        "second_order_max_path_prediction": max(
                            l1_truncated_prediction(
                                trace.first[index],
                                trace.second[index],
                                epsilon=epsilon,
                            )
                            for index in range(horizon + 1)
                        ),
                        "clean_terminal": trace.states[horizon].tolist(),
                        "perturbed_terminal": iterative[horizon].tolist(),
                        "polynomial_oracle_max_absolute_error": oracle_error,
                    }
                )
    return sorted(
        rows, key=lambda row: (row["landscape_id"], row["horizon"], row["epsilon"])
    )


def _numeric_equivalent(
    locked: Any, computed: Any, *, path: str = "$"
) -> None:
    if isinstance(locked, bool) or isinstance(computed, bool):
        if locked is not computed:
            raise ValueError(f"ACL-008 registry differs at {path}")
    elif isinstance(locked, (int, float)) and isinstance(computed, (int, float)):
        if not np.isclose(
            float(locked),
            float(computed),
            atol=ACL008_REGISTRY_ATOL,
            rtol=ACL008_REGISTRY_RTOL,
        ):
            raise ValueError(f"ACL-008 registry differs at {path}")
    elif isinstance(locked, str) and isinstance(computed, str):
        if locked != computed:
            raise ValueError(f"ACL-008 registry differs at {path}")
    elif isinstance(locked, list) and isinstance(computed, list):
        if len(locked) != len(computed):
            raise ValueError(f"ACL-008 registry length differs at {path}")
        for index, (left, right) in enumerate(zip(locked, computed, strict=True)):
            _numeric_equivalent(left, right, path=f"{path}[{index}]")
    elif isinstance(locked, dict) and isinstance(computed, dict):
        if set(locked) != set(computed):
            raise ValueError(f"ACL-008 registry keys differ at {path}")
        for key in sorted(locked):
            _numeric_equivalent(locked[key], computed[key], path=f"{path}.{key}")
    else:
        raise ValueError(f"ACL-008 registry types differ at {path}")


def validate_source_evidence(repo_path: str | Path, manifest: ACL003Manifest) -> dict[str, Any]:
    repo = Path(repo_path).resolve()
    source = manifest.raw.get("source_evidence")
    if source != ACL008_SOURCE_EVIDENCE:
        raise ValueError("ACL-008 source evidence mismatch")
    evidence = repo / source["evidence_artifact"]
    report = repo / source["report_summary"]
    if sha256_file(evidence) != source["evidence_sha256"]:
        raise ValueError("ACL-003 source evidence SHA-256 mismatch")
    if sha256_file(report) != source["report_summary_sha256"]:
        raise ValueError("ACL-003 source report SHA-256 mismatch")
    return {"valid": True, **source}


def validate_execution_environment(manifest: ACL003Manifest) -> dict[str, Any]:
    actual = {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
    }
    if actual != manifest.raw.get("confirmatory_environment"):
        raise ValueError("ACL-008 confirmatory environment mismatch")
    return {"valid": True, **actual, "dtype": "float64", "randomness": "none"}


def validate_preregistration_bundle(
    bundle_path: str | Path,
    *,
    reference_path: str | Path = "preregistrations/ACL-003/manifest.json",
) -> dict[str, Any]:
    bundle = Path(bundle_path)
    lock = validate_lock(bundle / "LOCK.json")
    if (
        lock.get("experiment_id") != "ACL-008"
        or lock.get("kind") != "preregistration-bundle-lock"
        or lock.get("outcomes_generated") is not False
        or set(lock.get("files", {})) != ACL008_LOCKED_FILES
    ):
        raise ValueError("ACL-008 lock must contain exact frozen file set")
    if {path.name for path in bundle.iterdir() if path.is_file()} != (
        ACL008_LOCKED_FILES | {"LOCK.json"}
    ):
        raise ValueError("ACL-008 bundle must have exact frozen directory contents")
    manifest = load_manifest(bundle / "manifest.json")
    if sha256_file(reference_path) != ACL008_REFERENCE_MANIFEST_SHA256:
        raise ValueError("ACL-008 novelty reference manifest hash mismatch")
    try:
        reference = json.loads(Path(reference_path).read_text(encoding="utf-8"))
        locked = json.loads(
            (bundle / "analytic_registry.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("cannot read ACL-008 registry/reference") from error
    computed = build_analytic_registry(manifest)
    _numeric_equivalent(locked, computed)
    if locked.get("outcomes_generated") is not False:
        raise ValueError("ACL-008 registry must remain analytic-only")
    novelty = validate_catalog_novelty(manifest, reference)
    regular = sum(
        entry["stratum"] == "regular-sensitivity" for entry in locked["landscapes"]
    )
    if regular != 16:
        raise ValueError("ACL-008 requires 16 regular confirmatory targets")
    return {
        "schema_version": 1,
        "experiment_id": "ACL-008",
        "kind": "preregistration-only-validation",
        "valid": True,
        "outcomes_generated": False,
        "landscape_count": len(manifest.landscapes),
        "confirmatory_target_count": 16,
        "software_control_count": 1,
        "regular_target_count": regular,
        "catalog_novelty": novelty,
        "registry_comparison": "numeric-tolerance",
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
    repo = Path(repo_path).resolve()
    output = Path(output_path)
    if not output.is_absolute():
        output = repo / output
    output = output.resolve()
    canonical = (repo / "evidence" / f"ACL-008-confirmatory-{approved_sha}.json").resolve()
    if output != canonical:
        raise ValueError("ACL-008 output must equal canonical SHA-derived evidence path")
    bundle = Path(bundle_path)
    if not bundle.is_absolute():
        bundle = repo / bundle
    bundle = bundle.resolve()
    if bundle != (repo / "preregistrations" / "ACL-008").resolve():
        raise ValueError("ACL-008 requires canonical preregistration bundle")
    current_sha, dirty = git_execution_state(repo)
    assert_execution_context(
        approved_sha=approved_sha,
        current_sha=current_sha,
        worktree_dirty=dirty,
        output_path=canonical,
    )
    validation = validate_preregistration_bundle(
        bundle, reference_path=reference_path
    )
    manifest = load_manifest(bundle / "manifest.json")
    source = validate_source_evidence(repo, manifest)
    environment = validate_execution_environment(manifest)
    registry = json.loads((bundle / "analytic_registry.json").read_text(encoding="utf-8"))
    rows = generate_raw_rows(manifest)
    analysis = analyze_raw_rows(manifest, registry, rows)
    payload = {
        "schema_version": 1,
        "experiment_id": "ACL-008",
        "kind": "confirmatory-non-fisher-second-order-transport",
        "approved_preregistration_sha": approved_sha,
        "preregistration_validation": validation,
        "source_evidence_validation": source,
        "confirmatory_environment": environment,
        "randomness_used": False,
        "target_refit": False,
        "frozen_design": manifest.raw,
        "locked_analytic_registry": registry,
        "preregistration_bundle_lock": json.loads(
            (bundle / "LOCK.json").read_text(encoding="utf-8")
        ),
        "raw_rows": rows,
        "analysis": analysis,
        "provenance": provenance(),
    }
    return write_json(canonical, payload)


SOURCE_GATES = {
    "eta": ACL003_ETA,
    "primary_horizon": ACL003_PRIMARY_HORIZON,
    "secondary_horizons": ACL003_SECONDARY_HORIZONS,
    "epsilon_grid": ACL003_EPSILON_GRID,
    "numerical_control_epsilons": ACL003_NUMERICAL_CONTROL_EPSILONS,
    "confirmatory_epsilons": ACL003_CONFIRMATORY_EPSILONS,
    "stress_epsilons": ACL003_STRESS_EPSILONS,
    "median_gate": ACL003_MEDIAN_GATE,
    "q90_gate": ACL003_Q90_GATE,
    "delta_floor": ACL003_DELTA_FLOOR,
    "matrix_oracle_tolerance": ACL003_MATRIX_ORACLE_TOLERANCE,
    "state_oracle_tolerance": ACL003_STATE_ORACLE_TOLERANCE,
    "first_oracle_tolerance": ACL003_FIRST_ORACLE_TOLERANCE,
    "curvature_oracle_tolerance": ACL003_CURVATURE_ORACLE_TOLERANCE,
}
