"""Deterministic post-confirmatory analysis of the immutable ACL-002 artifact.

This module reads preserved evidence. It intentionally does not import ACL-002's
trajectory generator or confirmatory runner. All scientific transforms operate on the
stored raw rows after a fail-closed identity and schema check.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import statistics
import subprocess
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

ACL002_APPROVED_SHA = "3f6a935942f43c7d3055582d123e58af5bf3f38b"
ACL002_EVIDENCE_COMMIT = "5caf47b510d70564415354f34ba729ff505f7ed4"
ACL002_ARTIFACT_NAME = f"ACL-002-confirmatory-{ACL002_APPROVED_SHA}.json"
ACL002_ARTIFACT_RELATIVE_PATH = f"evidence/{ACL002_ARTIFACT_NAME}"
ACL002_ARTIFACT_SHA256 = (
    "4d08e85b927a5d78a29078ff0d6549225d98069b20186b754629464739f29d74"
)
ACL002_ARTIFACT_ALPHA_SOURCE = 0.9951356698171323
ACL002_PROMPT_ALPHA_TRANSCRIPTION = 0.9951356718983256
ACL002_ROW_COUNT = 896
ACL002_ORACLE_TOLERANCE = 5e-13
ACL002_DELTA_FLOOR = 2e-12
ACL002_HORIZONS = (1, 5, 20, 50)
ACL002_REGIONS = {
    "zero": 112,
    "confirmatory": 336,
    "extended-local": 224,
    "stress": 224,
}
ACL002_LANDSCAPE_IDS = tuple(
    [f"S{index:02d}" for index in range(1, 15)]
    + [f"T{index:02d}" for index in range(1, 15)]
)
STABILITY_LEVELS = (0.01, 0.05, 0.10, 0.20)


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _type7(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot compute a quantile of an empty sequence")
    return float(np.quantile(np.asarray(values, dtype=np.float64), probability, method="linear"))


def _verify_evidence_commit(repo_path: Path, artifact_path: Path) -> None:
    root = repo_path.resolve()
    artifact = artifact_path.resolve()
    _require(root == artifact or root in artifact.parents, "artifact must be inside repository")
    relative = artifact.relative_to(root).as_posix()
    try:
        committed = subprocess.run(
            ["git", "show", f"{ACL002_EVIDENCE_COMMIT}:{relative}"],
            cwd=root,
            check=True,
            capture_output=True,
            timeout=10,
        ).stdout
    except (FileNotFoundError, subprocess.SubprocessError) as error:
        raise ValueError("cannot verify ACL-002 evidence commit") from error
    _require(
        _sha256_bytes(committed) == ACL002_ARTIFACT_SHA256,
        "evidence commit does not contain the immutable ACL-002 artifact",
    )


def verify_acl002_artifact(
    artifact_path: str | Path,
    *,
    repo_path: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load ACL-002 only after validating its immutable identity and frozen verdict."""
    path = Path(artifact_path)
    try:
        content = path.read_bytes()
    except OSError as error:
        raise ValueError(f"cannot read ACL-002 artifact: {path}") from error
    digest = _sha256_bytes(content)
    _require(
        digest == ACL002_ARTIFACT_SHA256,
        f"ACL-002 artifact SHA-256 mismatch: expected {ACL002_ARTIFACT_SHA256}, got {digest}",
    )
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as error:
        raise ValueError("ACL-002 artifact is not valid JSON") from error

    _require(payload.get("schema_version") == 1, "unexpected ACL-002 schema version")
    _require(payload.get("experiment_id") == "ACL-002", "unexpected experiment identifier")
    _require(
        payload.get("kind") == "confirmatory-mutation-stability-result",
        "artifact is not an ACL-002 confirmatory result",
    )
    _require(
        payload.get("approved_preregistration_sha") == ACL002_APPROVED_SHA,
        "approved preregistration SHA mismatch",
    )
    _require(payload.get("randomness_used") is False, "ACL-002 unexpectedly used randomness")
    _require(
        payload.get("benchmark_scope") == "deterministic-held-out-benchmark",
        "benchmark scope mismatch",
    )
    _require(
        payload.get("inference_scope") == "descriptive-criteria-not-population-confidence",
        "inference scope mismatch",
    )
    _require(
        payload.get("transport_scope") == "within-family-combinatorial-held-out-transport",
        "transport scope mismatch",
    )

    rows = payload.get("raw_rows")
    _require(isinstance(rows, list), "ACL-002 raw_rows must be a list")
    _require(len(rows) == ACL002_ROW_COUNT, "ACL-002 raw row count mismatch")
    region_counts = dict(sorted(Counter(row.get("region") for row in rows).items()))
    _require(region_counts == ACL002_REGIONS, "ACL-002 region counts mismatch")
    horizons = sorted({row.get("horizon") for row in rows})
    _require(horizons == list(ACL002_HORIZONS), "ACL-002 horizons mismatch")
    landscape_ids = sorted({row.get("landscape_id") for row in rows})
    _require(landscape_ids == sorted(ACL002_LANDSCAPE_IDS), "ACL-002 landscape IDs mismatch")
    unique_cells = {
        (row.get("landscape_id"), row.get("horizon"), row.get("epsilon")) for row in rows
    }
    _require(len(unique_cells) == ACL002_ROW_COUNT, "ACL-002 contains duplicate raw cells")

    oracle = payload.get("matrix_power_oracle", {})
    _require(oracle.get("passed") is True, "ACL-002 matrix-power oracle did not pass")
    _require(
        oracle.get("tolerance") == ACL002_ORACLE_TOLERANCE,
        "ACL-002 matrix-power oracle tolerance mismatch",
    )
    _require(
        float(oracle.get("maximum_absolute_error", math.inf)) <= ACL002_ORACLE_TOLERANCE,
        "ACL-002 matrix-power oracle error exceeds tolerance",
    )

    analysis = payload.get("analysis", {})
    _require(
        analysis.get("alpha_source") == ACL002_ARTIFACT_ALPHA_SOURCE,
        "ACL-002 frozen alpha_source mismatch",
    )
    source_alphas = [float(row["alpha"]) for row in analysis.get("source_landscape_alphas", [])]
    _require(len(source_alphas) == 12, "ACL-002 source alpha count mismatch")
    _require(
        statistics.median(source_alphas) == ACL002_ARTIFACT_ALPHA_SOURCE,
        "ACL-002 alpha_source is not the frozen landscape median",
    )
    gates = analysis.get("primary_gates", {})
    for layer in ("analytic", "transport"):
        _require(gates.get(layer, {}).get("passed") is True, f"ACL-002 {layer} gate changed")
        _require(
            gates[layer].get("landscape_count") == 12,
            f"ACL-002 {layer} landscape count mismatch",
        )
    special = analysis.get("special_target_strata", [])
    _require(len(special) == 2, "ACL-002 special target stratum count mismatch")
    _require(
        all(
            entry["checks"][layer]["passed"] is True
            for entry in special
            for layer in ("analytic", "transport")
        ),
        "ACL-002 special-stratum verdict changed",
    )

    if repo_path is not None:
        _verify_evidence_commit(Path(repo_path), path)

    verification = {
        "artifact_path": path.as_posix(),
        "artifact_sha256": digest,
        "approved_preregistration_sha": ACL002_APPROVED_SHA,
        "evidence_commit": ACL002_EVIDENCE_COMMIT,
        "row_count": len(rows),
        "region_counts": region_counts,
        "horizons": horizons,
        "landscape_ids": landscape_ids,
        "matrix_oracle": oracle,
        "frozen_primary_results": gates,
        "frozen_alpha_source": analysis["alpha_source"],
        "prompt_alpha_transcription": ACL002_PROMPT_ALPHA_TRANSCRIPTION,
        "alpha_transcription_absolute_difference": abs(
            ACL002_PROMPT_ALPHA_TRANSCRIPTION - ACL002_ARTIFACT_ALPHA_SOURCE
        ),
        "randomness_used": False,
    }
    return payload, verification


def _base_derived_row(row: dict[str, Any]) -> dict[str, Any]:
    epsilon = float(row["epsilon"])
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("posthoc derived rows require finite positive epsilon")
    clean = np.asarray(row["clean_terminal"], dtype=np.float64)
    if clean.ndim != 1 or clean.size < 2 or np.any(clean <= 0.0):
        raise ValueError("invalid stored clean categorical state")
    if not math.isclose(float(np.sum(clean)), 1.0, rel_tol=0.0, abs_tol=5e-13):
        raise ValueError("stored clean categorical state does not lie on simplex")
    return {
        "landscape_id": row["landscape_id"],
        "split": row["split"],
        "stratum": row["stratum"],
        "horizon": int(row["horizon"]),
        "epsilon": epsilon,
        "region": row["region"],
        "p0_id": str(row["p0"]),
        "reward_id": str(row["reward"]),
        "mutation_id": str(row["mutation"]),
        "clean_terminal": clean.tolist(),
        "clean_boundary_min_probability": float(np.min(clean)),
        "clean_boundary_pressure": float(-math.log(float(np.min(clean)))),
    }


def _finite_nonnegative(row: dict[str, Any], key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"stored {key} must be finite and non-negative")
    return value


def _finite(row: dict[str, Any], key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"stored {key} must be finite")
    return value


def derive_l1_rows(
    raw_rows: Iterable[dict[str, Any]],
    *,
    delta_floor: float = ACL002_DELTA_FLOOR,
) -> list[dict[str, Any]]:
    """Derive signed L1 remainders without fitting any target coefficient."""
    derived: list[dict[str, Any]] = []
    for raw in raw_rows:
        if float(raw["epsilon"]) == 0.0:
            continue
        base = _base_derived_row(raw)
        epsilon = base["epsilon"]
        endpoint = _finite_nonnegative(raw, "endpoint_l1")
        path = _finite_nonnegative(raw, "max_path_l1")
        c_endpoint = _finite_nonnegative(raw, "C_endpoint_l1")
        c_path = _finite_nonnegative(raw, "C_max_path_l1")
        endpoint_prediction = c_endpoint * epsilon
        path_prediction = c_path * epsilon
        endpoint_residual = endpoint - endpoint_prediction
        path_residual = path - path_prediction
        derived.append(
            {
                **base,
                "C_endpoint_l1": c_endpoint,
                "endpoint_l1": endpoint,
                "endpoint_prediction": endpoint_prediction,
                "endpoint_residual": endpoint_residual,
                "endpoint_residual_over_epsilon_squared": endpoint_residual / epsilon**2,
                "endpoint_relative_residual": (
                    endpoint_residual / endpoint_prediction
                    if endpoint_prediction >= delta_floor
                    else None
                ),
                "C_max_path_l1": c_path,
                "max_path_l1": path,
                "max_path_prediction": path_prediction,
                "max_path_residual": path_residual,
                "max_path_residual_over_epsilon_squared": path_residual / epsilon**2,
                "max_path_relative_residual": (
                    path_residual / path_prediction if path_prediction >= delta_floor else None
                ),
            }
        )
    return sorted(derived, key=lambda row: (row["landscape_id"], row["horizon"], row["epsilon"]))


def derive_kl_rows(
    raw_rows: Iterable[dict[str, Any]],
    *,
    delta_floor: float = ACL002_DELTA_FLOOR,
) -> list[dict[str, Any]]:
    """Derive the oriented-KL quadratic remainder and guarded cubic scale."""
    derived: list[dict[str, Any]] = []
    for raw in raw_rows:
        if float(raw["epsilon"]) == 0.0:
            continue
        base = _base_derived_row(raw)
        epsilon = base["epsilon"]
        observed = _finite(raw, "kl_q_p")
        if observed < -delta_floor:
            raise ValueError("stored kl_q_p is negative beyond the numerical floor")
        coefficient = _finite_nonnegative(raw, "K_kl_q_p")
        prediction = coefficient * epsilon**2
        residual = observed - prediction
        derived.append(
            {
                **base,
                "K_kl_q_p": coefficient,
                "kl_q_p": observed,
                "kl_prediction": prediction,
                "kl_residual": residual,
                "kl_over_epsilon_squared": observed / epsilon**2,
                "kl_over_epsilon_squared_minus_k": residual / epsilon**2,
                "kl_relative_residual": (
                    residual / prediction if prediction >= delta_floor else None
                ),
                "kl_residual_over_epsilon_cubed": (
                    residual / epsilon**3 if prediction >= delta_floor else None
                ),
                "cubic_normalization_numerically_stable": prediction >= delta_floor,
            }
        )
    return sorted(derived, key=lambda row: (row["landscape_id"], row["horizon"], row["epsilon"]))


def empirical_stability_radii(
    l1_rows: Iterable[dict[str, Any]],
    *,
    metric: str,
    levels: Sequence[float] = STABILITY_LEVELS,
) -> list[dict[str, Any]]:
    """Return cumulative-prefix descriptive radii for regular-sensitivity cells."""
    if metric not in {"endpoint", "max_path"}:
        raise ValueError("metric must be 'endpoint' or 'max_path'")
    relative_key = f"{metric}_relative_residual"
    groups: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in l1_rows:
        if row["stratum"] != "regular-sensitivity":
            continue
        groups[(row["landscape_id"], row["split"], row["stratum"], row["horizon"])].append(
            row
        )
    output: list[dict[str, Any]] = []
    for (landscape_id, split, stratum, horizon), rows in sorted(groups.items()):
        ordered = sorted(rows, key=lambda row: row["epsilon"])
        for level in levels:
            if not 0.0 < level < 1.0:
                raise ValueError("relative-error levels must lie in (0, 1)")
            accepted: list[dict[str, Any]] = []
            first_failure: float | None = None
            for row in ordered:
                residual = row[relative_key]
                if residual is None or abs(float(residual)) > level:
                    first_failure = float(row["epsilon"])
                    break
                accepted.append(row)
            last = accepted[-1] if accepted else None
            output.append(
                {
                    "landscape_id": landscape_id,
                    "split": split,
                    "stratum": stratum,
                    "horizon": horizon,
                    "metric": metric,
                    "relative_error_level": float(level),
                    "largest_tested_epsilon": float(last["epsilon"]) if last else None,
                    "largest_tested_region": last["region"] if last else None,
                    "accepted_positive_epsilon_count": len(accepted),
                    "first_failing_epsilon": first_failure,
                    "all_tested_positive_epsilons_pass": len(accepted) == len(ordered),
                }
            )
    return output


def summarize_source_alphas(payload: dict[str, Any]) -> dict[str, Any]:
    """Describe the already-frozen source alphas without estimating a replacement."""
    analysis = payload["analysis"]
    entries = analysis["source_landscape_alphas"]
    values = [float(entry["alpha"]) for entry in entries]
    if not values:
        raise ValueError("source alpha registry is empty")
    frozen = float(analysis["alpha_source"])
    median = float(statistics.median(values))
    _require(frozen == median, "frozen source alpha is not the landscape median")
    below = sum(value < 1.0 for value in values)
    equal = sum(value == 1.0 for value in values)
    above = sum(value > 1.0 for value in values)
    deviations = [abs(value - median) for value in values]
    return {
        "landscape_count": len(values),
        "frozen_alpha_source": frozen,
        "recomputed_median_for_verification": median,
        "mean": float(statistics.fmean(values)),
        "sample_standard_deviation": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
        "minimum": min(values),
        "q25_type7": _type7(values, 0.25),
        "median_type7": _type7(values, 0.5),
        "q75_type7": _type7(values, 0.75),
        "maximum": max(values),
        "median_absolute_deviation": float(statistics.median(deviations)),
        "count_below_one": below,
        "count_equal_one": equal,
        "count_above_one": above,
        "heterogeneous_correction_signs": below > 0 and above > 0,
        "maximum_absolute_deviation_from_one": max(abs(value - 1.0) for value in values),
    }


def _signed_class(values: Sequence[float], *, tolerance: float = ACL002_DELTA_FLOOR) -> str:
    signs = {1 if value > tolerance else -1 if value < -tolerance else 0 for value in values}
    nonzero = signs - {0}
    if not nonzero:
        return "numerical-zero"
    if len(nonzero) > 1:
        return "mixed"
    return "positive" if 1 in nonzero else "negative"


def _numeric_summary(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise ValueError("cannot summarize an empty sequence")
    return {
        "minimum": min(values),
        "q10_type7": _type7(values, 0.10),
        "median_type7": _type7(values, 0.50),
        "q90_type7": _type7(values, 0.90),
        "maximum": max(values),
        "mean": float(statistics.fmean(values)),
        "population_standard_deviation": float(statistics.pstdev(values)),
    }


def summarize_l1_residuals(l1_rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize signs and quadratic scales within each landscape/region cell."""
    groups: dict[tuple[str, str, str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in l1_rows:
        groups[
            (
                row["landscape_id"],
                row["split"],
                row["stratum"],
                row["horizon"],
                row["region"],
            )
        ].append(row)
    output: list[dict[str, Any]] = []
    for (landscape_id, split, stratum, horizon, region), rows in sorted(groups.items()):
        record: dict[str, Any] = {
            "landscape_id": landscape_id,
            "split": split,
            "stratum": stratum,
            "horizon": horizon,
            "region": region,
            "epsilon_count": len(rows),
            "epsilon_minimum": min(row["epsilon"] for row in rows),
            "epsilon_maximum": max(row["epsilon"] for row in rows),
        }
        for metric in ("endpoint", "max_path"):
            residuals = [float(row[f"{metric}_residual"]) for row in rows]
            scales = [float(row[f"{metric}_residual_over_epsilon_squared"]) for row in rows]
            relative = [
                float(row[f"{metric}_relative_residual"])
                for row in rows
                if row[f"{metric}_relative_residual"] is not None
            ]
            scale_summary = _numeric_summary(scales)
            record.update(
                {
                    f"{metric}_residual_sign": _signed_class(residuals),
                    f"{metric}_quadratic_scale_minimum": scale_summary["minimum"],
                    f"{metric}_quadratic_scale_median": scale_summary["median_type7"],
                    f"{metric}_quadratic_scale_maximum": scale_summary["maximum"],
                    f"{metric}_quadratic_scale_relative_range": (
                        (scale_summary["maximum"] - scale_summary["minimum"])
                        / max(abs(scale_summary["median_type7"]), ACL002_DELTA_FLOOR)
                    ),
                    f"{metric}_relative_residual_median": (
                        _type7(relative, 0.5) if relative else None
                    ),
                }
            )
        output.append(record)
    return output


def summarize_cross_landscape_l1(
    l1_rows: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Describe dimensionless and quadratic residual structure without target fitting."""
    groups: dict[tuple[str, int, float, str], list[dict[str, Any]]] = defaultdict(list)
    for row in l1_rows:
        if row["stratum"] != "regular-sensitivity":
            continue
        groups[(row["split"], row["horizon"], row["epsilon"], row["region"])].append(row)
    output: list[dict[str, Any]] = []
    for (split, horizon, epsilon, region), rows in sorted(groups.items()):
        record: dict[str, Any] = {
            "split": split,
            "horizon": horizon,
            "epsilon": epsilon,
            "region": region,
            "landscape_count": len(rows),
        }
        for metric in ("endpoint", "max_path"):
            relative = [float(row[f"{metric}_relative_residual"]) for row in rows]
            scales = [float(row[f"{metric}_residual_over_epsilon_squared"]) for row in rows]
            relative_summary = _numeric_summary(relative)
            scale_summary = _numeric_summary(scales)
            record.update(
                {
                    f"{metric}_relative_residual_median": relative_summary["median_type7"],
                    f"{metric}_relative_residual_q10": relative_summary["q10_type7"],
                    f"{metric}_relative_residual_q90": relative_summary["q90_type7"],
                    f"{metric}_relative_residual_positive_fraction": sum(
                        value > 0.0 for value in relative
                    )
                    / len(relative),
                    f"{metric}_quadratic_scale_median": scale_summary["median_type7"],
                    f"{metric}_quadratic_scale_q10": scale_summary["q10_type7"],
                    f"{metric}_quadratic_scale_q90": scale_summary["q90_type7"],
                }
            )
        output.append(record)
    return output


def summarize_kl_residuals(kl_rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize quadratic-law error and guarded cubic scales by landscape/region."""
    groups: dict[tuple[str, str, str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in kl_rows:
        groups[
            (
                row["landscape_id"],
                row["split"],
                row["stratum"],
                row["horizon"],
                row["region"],
            )
        ].append(row)
    output: list[dict[str, Any]] = []
    for (landscape_id, split, stratum, horizon, region), rows in sorted(groups.items()):
        residuals = [float(row["kl_residual"]) for row in rows]
        quadratic_errors = [float(row["kl_over_epsilon_squared_minus_k"]) for row in rows]
        cubic = [
            float(row["kl_residual_over_epsilon_cubed"])
            for row in rows
            if row["kl_residual_over_epsilon_cubed"] is not None
        ]
        quadratic_summary = _numeric_summary(quadratic_errors)
        cubic_summary = _numeric_summary(cubic) if cubic else None
        output.append(
            {
                "landscape_id": landscape_id,
                "split": split,
                "stratum": stratum,
                "horizon": horizon,
                "region": region,
                "epsilon_count": len(rows),
                "kl_residual_sign": _signed_class(residuals),
                "kl_quadratic_error_minimum": quadratic_summary["minimum"],
                "kl_quadratic_error_median": quadratic_summary["median_type7"],
                "kl_quadratic_error_maximum": quadratic_summary["maximum"],
                "stable_cubic_count": len(cubic),
                "kl_cubic_scale_minimum": cubic_summary["minimum"] if cubic_summary else None,
                "kl_cubic_scale_median": (
                    cubic_summary["median_type7"] if cubic_summary else None
                ),
                "kl_cubic_scale_maximum": cubic_summary["maximum"] if cubic_summary else None,
            }
        )
    return output


def horizon_feature_rows(l1_rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract one analytic sensitivity-feature row per landscape and horizon."""
    cells: dict[tuple[str, int], dict[str, Any]] = {}
    for row in l1_rows:
        key = (row["landscape_id"], row["horizon"])
        candidate = {
            "landscape_id": row["landscape_id"],
            "split": row["split"],
            "stratum": row["stratum"],
            "horizon": row["horizon"],
            "log_horizon": math.log(float(row["horizon"])),
            "C_endpoint_l1": row["C_endpoint_l1"],
            "C_max_path_l1": row["C_max_path_l1"],
            "p0_id": row["p0_id"],
            "reward_id": row["reward_id"],
            "mutation_id": row["mutation_id"],
            "clean_terminal": row["clean_terminal"],
            "clean_boundary_min_probability": row["clean_boundary_min_probability"],
            "clean_boundary_pressure": row["clean_boundary_pressure"],
        }
        if key in cells and cells[key] != candidate:
            raise ValueError("analytic landscape/horizon features vary across epsilon")
        cells[key] = candidate
    output = [cells[key] for key in sorted(cells)]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in output:
        groups[row["landscape_id"]].append(row)
    for landscape_rows in groups.values():
        ordered = sorted(landscape_rows, key=lambda row: row["horizon"])
        dimensions = {len(row["clean_terminal"]) for row in ordered}
        if len(dimensions) != 1:
            raise ValueError("clean-state dimension varies within a landscape")
        reference = next(iter(dimensions)) - 1
        times = np.asarray([row["horizon"] for row in ordered], dtype=np.float64)
        design = np.column_stack([np.ones(len(times), dtype=np.float64), times])
        slopes = [0.0]
        intercepts = [0.0]
        maximum_error = 0.0
        reference_values = np.asarray(
            [row["clean_terminal"][reference] for row in ordered], dtype=np.float64
        )
        for coordinate in range(reference):
            coordinate_values = np.asarray(
                [row["clean_terminal"][coordinate] for row in ordered], dtype=np.float64
            )
            log_odds = np.log(coordinate_values / reference_values)
            coefficients, *_ = np.linalg.lstsq(design, log_odds, rcond=None)
            intercepts.append(float(coefficients[0]))
            slopes.append(float(coefficients[1]))
            maximum_error = max(
                maximum_error,
                float(np.max(np.abs(log_odds - design @ coefficients))),
            )
        relative_initial = np.exp(np.asarray(intercepts, dtype=np.float64))
        inferred_initial = relative_initial / np.sum(relative_initial)
        selection_spread = max(slopes) - min(slopes)
        for row in landscape_rows:
            row["selection_log_factor_spread_per_step"] = selection_spread
            row["inferred_initial_boundary_min_probability"] = float(
                np.min(inferred_initial)
            )
            row["clean_log_odds_linear_max_abs_error"] = maximum_error
    return output


def _feature_value(row: dict[str, Any], feature: str) -> float:
    if feature == "horizon_x_reward":
        return float(row["log_horizon"] * row["selection_log_factor_spread_per_step"])
    if feature == "horizon_x_boundary":
        return float(row["log_horizon"] * row["clean_boundary_pressure"])
    if feature.startswith("mutation_id="):
        return float(row["mutation_id"] == feature.removeprefix("mutation_id="))
    if feature.startswith("horizon_x_mutation_id="):
        mutation_id = feature.removeprefix("horizon_x_mutation_id=")
        return float(row["log_horizon"] * (row["mutation_id"] == mutation_id))
    return float(row[feature])


def _standardized_design(
    train: Sequence[dict[str, Any]],
    evaluate: Sequence[dict[str, Any]],
    features: Sequence[str],
) -> tuple[FloatArray, FloatArray]:
    raw_train = np.asarray(
        [[_feature_value(row, feature) for feature in features] for row in train],
        dtype=np.float64,
    )
    raw_evaluate = np.asarray(
        [[_feature_value(row, feature) for feature in features] for row in evaluate],
        dtype=np.float64,
    )
    means = np.mean(raw_train, axis=0)
    scales = np.std(raw_train, axis=0)
    scales = np.where(scales > 1e-14, scales, 1.0)
    train_design = np.column_stack(
        [np.ones(len(train), dtype=np.float64), (raw_train - means) / scales]
    )
    evaluate_design = np.column_stack(
        [np.ones(len(evaluate), dtype=np.float64), (raw_evaluate - means) / scales]
    )
    return train_design, evaluate_design


def _r_squared(observed: FloatArray, predicted: FloatArray) -> float:
    denominator = float(np.sum((observed - np.mean(observed)) ** 2))
    if denominator <= 0.0:
        return 1.0 if np.array_equal(observed, predicted) else float("nan")
    return 1.0 - float(np.sum((observed - predicted) ** 2)) / denominator


def horizon_model_comparison(feature_rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Exploratory OLS and leave-one-landscape-out comparisons for sensitivity growth."""
    rows = [row for row in feature_rows if row["stratum"] == "regular-sensitivity"]
    if len({row["landscape_id"] for row in rows}) < 3:
        raise ValueError("horizon model comparison needs at least three landscapes")
    output: list[dict[str, Any]] = []
    mutation_ids = sorted({row["mutation_id"] for row in rows})
    mutation_dummies = tuple(f"mutation_id={identifier}" for identifier in mutation_ids[1:])
    mutation_interactions = tuple(
        f"horizon_x_mutation_id={identifier}" for identifier in mutation_ids[1:]
    )
    base_features = (
        "log_horizon",
        "selection_log_factor_spread_per_step",
        "clean_boundary_pressure",
        *mutation_dummies,
    )
    model_features: dict[str, tuple[str, ...]] = {
        "horizon-only": ("log_horizon",),
        "reward-intensity-only": ("selection_log_factor_spread_per_step",),
        "boundary-only": ("clean_boundary_pressure",),
        "mutation-id-only": mutation_dummies,
        "additive": base_features,
        "additive-plus-interactions": (
            *base_features,
            "horizon_x_reward",
            "horizon_x_boundary",
            *mutation_interactions,
        ),
    }
    for response in ("C_endpoint_l1", "C_max_path_l1"):
        if any(float(row[response]) <= 0.0 for row in rows):
            raise ValueError("regular-sensitivity coefficients must be positive")
        observed = np.log(np.asarray([row[response] for row in rows], dtype=np.float64))
        for model_name, features in model_features.items():
            design, _ = _standardized_design(rows, rows, features)
            coefficients, *_ = np.linalg.lstsq(design, observed, rcond=None)
            fitted = design @ coefficients

            cross_validated = np.empty_like(observed)
            for landscape_id in sorted({row["landscape_id"] for row in rows}):
                train_indices = [
                    index for index, row in enumerate(rows) if row["landscape_id"] != landscape_id
                ]
                test_indices = [
                    index for index, row in enumerate(rows) if row["landscape_id"] == landscape_id
                ]
                train = [rows[index] for index in train_indices]
                test = [rows[index] for index in test_indices]
                train_design, test_design = _standardized_design(train, test, features)
                train_observed = observed[train_indices]
                fold_coefficients, *_ = np.linalg.lstsq(
                    train_design, train_observed, rcond=None
                )
                cross_validated[test_indices] = test_design @ fold_coefficients

            output.append(
                {
                    "response": response,
                    "response_transform": "natural-log",
                    "model": model_name,
                    "features": "+".join(features),
                    "observation_count": len(rows),
                    "landscape_count": len({row["landscape_id"] for row in rows}),
                    "in_sample_r_squared": _r_squared(observed, fitted),
                    "leave_one_landscape_out_r_squared": _r_squared(observed, cross_validated),
                    "leave_one_landscape_out_rmse_log_units": float(
                        np.sqrt(np.mean((observed - cross_validated) ** 2))
                    ),
                }
            )
    return output


def source_alpha_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    frozen = float(payload["analysis"]["alpha_source"])
    return [
        {
            "landscape_id": entry["landscape_id"],
            "alpha": float(entry["alpha"]),
            "alpha_minus_one": float(entry["alpha"]) - 1.0,
            "alpha_minus_frozen_median": float(entry["alpha"]) - frozen,
            "correction_direction": (
                "above-one"
                if float(entry["alpha"]) > 1.0
                else "below-one"
                if float(entry["alpha"]) < 1.0
                else "exactly-one"
            ),
        }
        for entry in payload["analysis"]["source_landscape_alphas"]
    ]


def compare_source_target_l1(
    cross_rows: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Pair source and target descriptive summaries without estimating a mapping."""
    cells = {
        (row["split"], row["horizon"], row["epsilon"], row["region"]): row
        for row in cross_rows
    }
    output: list[dict[str, Any]] = []
    keys = sorted({(key[1], key[2], key[3]) for key in cells})
    for horizon, epsilon, region in keys:
        source = cells[("source", horizon, epsilon, region)]
        target = cells[("target", horizon, epsilon, region)]
        record: dict[str, Any] = {
            "horizon": horizon,
            "epsilon": epsilon,
            "region": region,
            "source_landscape_count": source["landscape_count"],
            "target_landscape_count": target["landscape_count"],
        }
        for metric in ("endpoint", "max_path"):
            for quantity in ("relative_residual_median", "quadratic_scale_median"):
                field = f"{metric}_{quantity}"
                source_value = float(source[field])
                target_value = float(target[field])
                record[f"source_{field}"] = source_value
                record[f"target_{field}"] = target_value
                record[f"target_minus_source_{field}"] = target_value - source_value
                record[f"target_over_source_{field}"] = (
                    target_value / source_value
                    if abs(source_value) >= ACL002_DELTA_FLOOR
                    else None
                )
        output.append(record)
    return output


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (list, dict, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty derived table: {path.name}")
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise ValueError(f"derived table has inconsistent columns: {path.name}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows([{key: _csv_value(value) for key, value in row.items()} for row in rows])


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _git_analysis_code_state(repo_path: Path) -> str:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        tracked_status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=repo_path,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.SubprocessError) as error:
        raise ValueError("cannot establish committed posthoc code provenance") from error
    _require(not tracked_status, "posthoc generation requires committed tracked analysis code")
    return commit


def _line_plot_svg(
    *,
    title: str,
    y_label: str,
    series: Sequence[tuple[str, Sequence[tuple[float, float]]]],
) -> str:
    if not series or any(not points for _, points in series):
        raise ValueError("SVG plot requires non-empty series")
    width, height = 1100, 650
    left, right, top, bottom = 90, 250, 65, 75
    plot_width = width - left - right
    plot_height = height - top - bottom
    x_values = [math.log10(x) for _, points in series for x, _ in points]
    y_values = [y for _, points in series for _, y in points]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    if math.isclose(y_min, y_max):
        y_min -= 1.0
        y_max += 1.0
    padding = 0.08 * (y_max - y_min)
    y_min -= padding
    y_max += padding

    def x_position(value: float) -> float:
        return left + (math.log10(value) - x_min) / (x_max - x_min) * plot_width

    def y_position(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_height

    palette = (
        "#0072B2",
        "#D55E00",
        "#009E73",
        "#CC79A7",
        "#E69F00",
        "#56B4E9",
        "#F0E442",
        "#000000",
        "#5E3C99",
        "#1B9E77",
        "#E7298A",
        "#7570B3",
    )
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="30" text-anchor="middle" '
        f'font-family="sans-serif" font-size="20">{html.escape(title)}</text>',
    ]
    for index in range(6):
        fraction = index / 5
        y_value = y_min + fraction * (y_max - y_min)
        y = y_position(y_value)
        parts.append(
            f'<line x1="{left}" y1="{y:.3f}" x2="{left + plot_width}" y2="{y:.3f}" '
            'stroke="#dddddd" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{left - 8}" y="{y + 4:.3f}" text-anchor="end" '
            f'font-family="monospace" font-size="11">{y_value:.4g}</text>'
        )
    epsilon_ticks = sorted({x for _, points in series for x, _ in points})
    for epsilon in epsilon_ticks:
        x = x_position(epsilon)
        parts.append(
            f'<line x1="{x:.3f}" y1="{top}" x2="{x:.3f}" y2="{top + plot_height}" '
            'stroke="#eeeeee" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{x:.3f}" y="{top + plot_height + 20}" text-anchor="middle" '
            f'font-family="monospace" font-size="11">{epsilon:g}</text>'
        )
    zero_y = y_position(0.0)
    if top <= zero_y <= top + plot_height:
        parts.append(
            f'<line x1="{left}" y1="{zero_y:.3f}" x2="{left + plot_width}" '
            f'y2="{zero_y:.3f}" stroke="#555555" stroke-width="1.5"/>'
        )
    parts.extend(
        [
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" '
            'stroke="black" stroke-width="1.5"/>',
            f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" '
            f'y2="{top + plot_height}" stroke="black" stroke-width="1.5"/>',
            f'<text x="{left + plot_width / 2}" y="{height - 20}" text-anchor="middle" '
            'font-family="sans-serif" font-size="14">epsilon (log scale)</text>',
            f'<text x="22" y="{top + plot_height / 2}" text-anchor="middle" '
            f'transform="rotate(-90 22 {top + plot_height / 2})" '
            f'font-family="sans-serif" font-size="14">{html.escape(y_label)}</text>',
        ]
    )
    for index, (label, points) in enumerate(series):
        color = palette[index % len(palette)]
        coordinates = " ".join(
            f"{x_position(x):.3f},{y_position(y):.3f}" for x, y in points
        )
        parts.append(
            f'<polyline points="{coordinates}" fill="none" stroke="{color}" '
            'stroke-width="2" stroke-linejoin="round"/>'
        )
        for x, y in points:
            parts.append(
                f'<circle cx="{x_position(x):.3f}" cy="{y_position(y):.3f}" r="2.5" '
                f'fill="{color}"/>'
            )
        legend_y = top + 18 * index
        parts.append(
            f'<line x1="{left + plot_width + 20}" y1="{legend_y}" '
            f'x2="{left + plot_width + 45}" y2="{legend_y}" stroke="{color}" '
            'stroke-width="2"/>'
        )
        parts.append(
            f'<text x="{left + plot_width + 52}" y="{legend_y + 4}" '
            f'font-family="monospace" font-size="12">{html.escape(label)}</text>'
        )
    parts.append("</svg>\n")
    return "\n".join(parts)


def _group_overview(
    rows: Sequence[dict[str, Any]],
    *,
    value_key: str,
    group_keys: Sequence[str],
    absolute: bool = False,
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in rows:
        value = row[value_key]
        if value is None:
            continue
        numeric = abs(float(value)) if absolute else float(value)
        groups[tuple(row[key] for key in group_keys)].append(numeric)
    output = []
    for group, values in sorted(groups.items()):
        output.append(
            {
                **dict(zip(group_keys, group, strict=True)),
                "count": len(values),
                **_numeric_summary(values),
            }
        )
    return output


def _radius_overview(radius_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in radius_rows:
        groups[
            (
                row["split"],
                row["horizon"],
                row["metric"],
                row["relative_error_level"],
            )
        ].append(row)
    output = []
    for (split, horizon, metric, level), rows in sorted(groups.items()):
        values = [row["largest_tested_epsilon"] for row in rows]
        numeric = [float(value) for value in values if value is not None]
        counts = Counter("none" if value is None else f"{float(value):g}" for value in values)
        output.append(
            {
                "split": split,
                "horizon": horizon,
                "metric": metric,
                "relative_error_level": level,
                "landscape_count": len(rows),
                "radius_counts": dict(sorted(counts.items())),
                "median_radius_type7": _type7(numeric, 0.5) if numeric else None,
                "minimum_radius": min(numeric) if numeric else None,
                "maximum_radius": max(numeric) if numeric else None,
            }
        )
    return output


def _phase1_summary(
    *,
    payload: dict[str, Any],
    verification: dict[str, Any],
    l1_rows: Sequence[dict[str, Any]],
    l1_summary: Sequence[dict[str, Any]],
    radius_rows: Sequence[dict[str, Any]],
    kl_rows: Sequence[dict[str, Any]],
    kl_summary: Sequence[dict[str, Any]],
    models: Sequence[dict[str, Any]],
    table_hashes: dict[str, str],
    plot_hashes: dict[str, str],
    analysis_code_commit: str,
) -> dict[str, Any]:
    regular_target_l1 = [
        row
        for row in l1_rows
        if row["split"] == "target" and row["stratum"] == "regular-sensitivity"
    ]
    regular_target_kl = [
        row
        for row in kl_rows
        if row["split"] == "target" and row["stratum"] == "regular-sensitivity"
    ]
    l1_region_overview = _group_overview(
        regular_target_l1,
        value_key="endpoint_relative_residual",
        group_keys=("horizon", "region"),
        absolute=True,
    )
    kl_region_overview = _group_overview(
        regular_target_kl,
        value_key="kl_relative_residual",
        group_keys=("horizon", "region"),
        absolute=True,
    )
    coefficient_stability = _group_overview(
        [
            row
            for row in l1_summary
            if row["split"] == "target"
            and row["stratum"] == "regular-sensitivity"
            and row["horizon"] > 1
        ],
        value_key="endpoint_quadratic_scale_relative_range",
        group_keys=("horizon", "region"),
    )
    l1_sign_counts = Counter(
        (str(row["horizon"]), row["region"], row["endpoint_residual_sign"])
        for row in l1_summary
        if row["split"] == "target" and row["stratum"] == "regular-sensitivity"
    )
    kl_sign_counts = Counter(
        (str(row["horizon"]), row["region"], row["kl_residual_sign"])
        for row in kl_summary
        if row["split"] == "target" and row["stratum"] == "regular-sensitivity"
    )
    best_models = {}
    for response in ("C_endpoint_l1", "C_max_path_l1"):
        candidates = [row for row in models if row["response"] == response]
        best_models[response] = max(
            candidates, key=lambda row: row["leave_one_landscape_out_r_squared"]
        )
    endpoint_equals_path = sum(
        row["endpoint_l1"] == row["max_path_l1"]
        and row["C_endpoint_l1"] == row["C_max_path_l1"]
        for row in l1_rows
    )
    return {
        "schema_version": 1,
        "analysis_id": "ACL-002-posthoc",
        "classification": "deterministic-post-confirmatory-exploratory-analysis",
        "analysis_code_commit": analysis_code_commit,
        "source_artifact_verification": verification,
        "confirmed_findings": {
            "acl002_primary_gates": payload["analysis"]["primary_gates"],
            "special_target_strata": payload["analysis"]["special_target_strata"],
            "matrix_power_oracle": payload["matrix_power_oracle"],
            "interpretive_limit": (
                "ACL-002 confirms first-order prediction only on its frozen deterministic "
                "within-family benchmark; posthoc results do not alter that verdict."
            ),
        },
        "exploratory_observations": {
            "l1_target_absolute_relative_error_by_horizon_region": l1_region_overview,
            "l1_target_quadratic_scale_relative_range_horizons_gt_one": (
                coefficient_stability
            ),
            "l1_target_residual_sign_counts": {
                "|".join(key): value for key, value in sorted(l1_sign_counts.items())
            },
            "endpoint_equals_max_path_row_count": endpoint_equals_path,
            "positive_epsilon_row_count": len(l1_rows),
            "empirical_stability_radius": _radius_overview(radius_rows),
            "kl_target_absolute_relative_error_by_horizon_region": kl_region_overview,
            "kl_target_residual_sign_counts": {
                "|".join(key): value for key, value in sorted(kl_sign_counts.items())
            },
            "source_alpha": summarize_source_alphas(payload),
            "horizon_model_comparison": list(models),
            "best_exploratory_horizon_models_by_lolo_r_squared": best_models,
            "artifact_identifiability": {
                "reward_measure": (
                    "per-step clean log-odds spread inferred from artifact states; equals "
                    "eta times reward spread"
                ),
                "mutation_measure": "nominal catalog ID only",
                "numeric_mutation_matrix_available": False,
            },
        },
        "new_hypotheses": [
            {
                "id": "ACL-PH1-H1",
                "status": "exploratory-candidate",
                "statement": (
                    "For horizons above one, an analytically derived second-order "
                    "sensitivity predicts the predominantly negative L1 correction and "
                    "extends the useful epsilon radius on new categorical landscapes."
                ),
            },
            {
                "id": "ACL-PH1-H2",
                "status": "exploratory-candidate",
                "statement": (
                    "Second-order coefficients are landscape dependent; a universal "
                    "scalar correction will not transport as well as a state-aware "
                    "analytic recurrence."
                ),
            },
            {
                "id": "ACL-PH1-H3",
                "status": "exploratory-candidate",
                "statement": (
                    "The oriented-KL cubic remainder is structured but more heterogeneous "
                    "than the L1 quadratic remainder."
                ),
            },
        ],
        "next_decision": (
            "Derive and independently verify the second-order row sensitivity recurrence "
            "before deciding whether it earns a new-landscape ACL-003 preregistration."
        ),
        "derived_table_sha256": table_hashes,
        "plot_sha256": plot_hashes,
    }


def _find_overview(
    rows: Sequence[dict[str, Any]], *, horizon: int, region: str
) -> dict[str, Any]:
    return next(row for row in rows if row["horizon"] == horizon and row["region"] == region)


def _posthoc_markdown(summary: dict[str, Any]) -> str:
    confirmed = summary["confirmed_findings"]
    observations = summary["exploratory_observations"]
    analytic = confirmed["acl002_primary_gates"]["analytic"]
    transport = confirmed["acl002_primary_gates"]["transport"]
    l1_overview = observations["l1_target_absolute_relative_error_by_horizon_region"]
    kl_overview = observations["kl_target_absolute_relative_error_by_horizon_region"]
    radii = observations["empirical_stability_radius"]
    target_t20_radii = [
        row
        for row in radii
        if row["split"] == "target" and row["horizon"] == 20 and row["metric"] == "endpoint"
    ]
    model_rows = observations["horizon_model_comparison"]
    source_alpha = observations["source_alpha"]

    lines = [
        "# ACL-002 deterministic post-confirmatory analysis",
        "",
        "## Status and provenance",
        "",
        "This package is a deterministic post-confirmatory analysis of the preserved "
        "ACL-002 artifact. It generated no new trajectories, did not invoke the "
        "confirmatory runner, did not refit target predictions, and does not change the "
        "ACL-002 verdict.",
        "",
        "- Source artifact SHA-256: "
        f"`{summary['source_artifact_verification']['artifact_sha256']}`",
        f"- Approved preregistration: `{ACL002_APPROVED_SHA}`",
        f"- Evidence commit: `{ACL002_EVIDENCE_COMMIT}`",
        f"- Analysis-code commit: `{summary['analysis_code_commit']}`",
        f"- Stored rows: {summary['source_artifact_verification']['row_count']}",
        "- Classification: post-confirmatory and exploratory except where the immutable "
        "  ACL-002 verdict is restated explicitly.",
        "",
        "The continuation prompt transcribed `alpha_source` as "
        f"`{ACL002_PROMPT_ALPHA_TRANSCRIPTION:.16f}`. The artifact stores "
        f"`{ACL002_ARTIFACT_ALPHA_SOURCE:.16f}`, which exactly equals the frozen median "
        "of its 12 source alphas. The artifact value is authoritative; neither value was "
        "altered.",
        "",
        "## Confirmed findings (restated, not re-tested)",
        "",
        "ACL-002's frozen zero-fit and calibrated predictions both passed. These are "
        "within-family deterministic benchmark results, not population confidence "
        "statements or cross-class transport.",
        "",
        "| Layer | Median max error | Type-7 Q90 | Verdict |",
        "| --- | ---: | ---: | --- |",
        f"| Zero-fit analytic | {100 * analytic['median']:.4f}% | "
        f"{100 * analytic['q90']:.4f}% | PASS |",
        f"| Frozen source calibration | {100 * transport['median']:.4f}% | "
        f"{100 * transport['q90']:.4f}% | PASS |",
        "",
        "The two special target strata and the independent matrix-power oracle also "
        "passed in the immutable artifact.",
        "",
        "## Exploratory observations: L1 residual structure",
        "",
        "For each stored positive-epsilon row this package computes "
        "`R = endpoint_l1 - C_endpoint_l1 * epsilon`, `R / epsilon^2`, and the "
        "dimensionless signed relative residual. No target residual is fitted away.",
        "",
        "At the primary horizon `T=20`, all 12 regular target landscapes have negative "
        "endpoint residuals at every tested positive epsilon. The absolute relative-error "
        "distribution grows monotonically by region:",
        "",
        "| Region | Median | Type-7 Q90 | Maximum |",
        "| --- | ---: | ---: | ---: |",
    ]
    for region in ("confirmatory", "extended-local", "stress"):
        row = _find_overview(l1_overview, horizon=20, region=region)
        lines.append(
            f"| {region} | {100 * row['median_type7']:.3f}% | "
            f"{100 * row['q90_type7']:.3f}% | {100 * row['maximum']:.3f}% |"
        )
    lines.extend(
        [
            "",
            "For `T=1`, the L1 update is affine in epsilon for the single mixing step, so "
            "the stored second-order residual is numerical zero. At `T=5,20,50`, every "
            "regular target has a negative residual throughout the strict-confirmatory "
            "and extended-local regions. Within-landscape `R/epsilon^2` is especially "
            "stable in the strict region and degrades smoothly as horizon and epsilon "
            "increase. Its magnitude varies substantially across landscapes, arguing "
            "against a universal scalar second-order correction.",
            "",
            "Endpoint and max-path L1 are identical in "
            f"{observations['endpoint_equals_max_path_row_count']} of "
            f"{observations['positive_epsilon_row_count']} positive-epsilon rows. The "
            "separate columns remain in the raw tables.",
            "",
            "### Exploratory empirical stability radius at T=20",
            "",
            "A radius is the largest tested positive epsilon for which the entire tested "
            "prefix stays below the stated zero-fit relative-error level. These values "
            "are descriptive and were not ACL-002 gates.",
            "",
            "| Error level | Median radius | Minimum | Maximum | Radius counts |",
            "| ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in target_t20_radii:
        counts = ", ".join(f"{key}: {value}" for key, value in row["radius_counts"].items())
        lines.append(
            f"| {100 * row['relative_error_level']:.0f}% | "
            f"{row['median_radius_type7']:g} | {row['minimum_radius']:g} | "
            f"{row['maximum_radius']:g} | {counts} |"
        )
    lines.extend(
        [
            "",
            "### Exploratory horizon/covariate description",
            "",
            "The artifact does not contain numeric catalog arrays. Reward intensity is "
            "therefore the per-step clean log-odds spread inferred from stored clean "
            "states (mathematically `eta * reward_spread`), and mutation structure is a "
            "nominal catalog ID. Numeric mutation-matrix attribution is not identifiable "
            "from this artifact alone.",
            "",
            "The following deterministic OLS comparisons use log sensitivity and "
            "leave-one-landscape-out prediction. They are exploratory model summaries, "
            "not inferential selection:",
            "",
            "| Response | Model | In-sample R2 | LOLO R2 | LOLO RMSE (log units) |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in model_rows:
        if row["model"] in {"horizon-only", "additive", "additive-plus-interactions"}:
            lines.append(
                f"| {row['response']} | {row['model']} | "
                f"{row['in_sample_r_squared']:.3f} | "
                f"{row['leave_one_landscape_out_r_squared']:.3f} | "
                f"{row['leave_one_landscape_out_rmse_log_units']:.3f} |"
            )
    lines.extend(
        [
            "",
            "Horizon is the strongest single descriptor. The additive model improves "
            "held-out-landscape description, while the predeclared interaction expansion "
            "does not improve LOLO R2. This does not identify causal importance because "
            "the deterministic catalog is small and covariates are correlated.",
            "",
            "## Exploratory observations: oriented KL",
            "",
            "For regular cases the package computes `KL - K * epsilon^2`, "
            "`KL / epsilon^2 - K`, and the cubic normalization only when the analytic "
            "quadratic prediction exceeds the frozen numerical floor. Low- and "
            "zero-sensitivity rows remain separate.",
            "",
            "At `T=20`, the absolute relative error of the quadratic KL prediction is:",
            "",
            "| Region | Median | Type-7 Q90 | Maximum |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for region in ("confirmatory", "extended-local", "stress"):
        row = _find_overview(kl_overview, horizon=20, region=region)
        lines.append(
            f"| {region} | {100 * row['median_type7']:.3f}% | "
            f"{100 * row['q90_type7']:.3f}% | {100 * row['maximum']:.3f}% |"
        )
    lines.extend(
        [
            "",
            "The KL cubic remainder is predominantly negative and locally structured, "
            "but its coefficient spans a much wider range across landscapes than a "
            "single transportable scalar would allow. This is exploratory; ACL-002 did "
            "not gate on KL.",
            "",
            "## Exploratory observations: frozen source calibration",
            "",
            f"The 12 frozen source alphas range from {source_alpha['minimum']:.6f} to "
            f"{source_alpha['maximum']:.6f}; 11 are below one and one is above one. The "
            f"median is {source_alpha['frozen_alpha_source']:.9f}, the sample standard "
            f"deviation is {source_alpha['sample_standard_deviation']:.6f}, and the "
            f"median absolute deviation is {source_alpha['median_absolute_deviation']:.6f}.",
            "",
            "Thus alpha near one is not merely an average of equally balanced positive "
            "and negative corrections: most source landscapes share a small negative "
            "finite-epsilon correction, with one opposing landscape and meaningful "
            "coefficient heterogeneity. The frozen alpha is described, never refitted.",
            "",
            "## New hypotheses (not confirmed)",
            "",
            "1. For horizons above one, an analytic second-order sensitivity recurrence "
            "   predicts the negative L1 correction and extends the usable epsilon radius "
            "   on entirely new categorical landscapes.",
            "2. A state-aware second-order coefficient is necessary; a universal scalar "
            "   correction will fail across heterogeneous landscapes.",
            "3. The oriented-KL cubic remainder is structured but less transportable than "
            "   the L1 quadratic correction.",
            "",
            "The next step is derivation and independent finite-difference/symbolic "
            "verification of the second-order row sensitivity. No ACL-003 outcome should "
            "be generated until that mechanism either earns or fails to earn a clean "
            "preregistration.",
            "",
            "## Files",
            "",
            "`summary.json` is the machine-readable index. CSV files contain every "
            "derived row and grouped summary. SVG files are deterministic views of target "
            "T=20 residuals. Their SHA-256 hashes are recorded in `summary.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def generate_posthoc_package(
    *,
    artifact_path: str | Path,
    output_dir: str | Path,
    repo_path: str | Path,
) -> list[Path]:
    """Generate a deterministic analysis package from the immutable artifact only."""
    repo = Path(repo_path).resolve()
    destination = Path(output_dir).resolve()
    _require(not destination.exists(), "posthoc output directory already exists")
    code_commit = _git_analysis_code_state(repo)
    payload, verification = verify_acl002_artifact(artifact_path, repo_path=repo)
    l1_rows = derive_l1_rows(payload["raw_rows"])
    kl_rows = derive_kl_rows(payload["raw_rows"])
    l1_summary = summarize_l1_residuals(l1_rows)
    cross_l1 = summarize_cross_landscape_l1(l1_rows)
    source_target_l1 = compare_source_target_l1(cross_l1)
    kl_summary = summarize_kl_residuals(kl_rows)
    features = horizon_feature_rows(l1_rows)
    models = horizon_model_comparison(features)
    radius_rows = [
        *empirical_stability_radii(l1_rows, metric="endpoint"),
        *empirical_stability_radii(l1_rows, metric="max_path"),
    ]
    alphas = source_alpha_rows(payload)

    temporary = destination.with_name(destination.name + ".tmp")
    _require(not temporary.exists(), "temporary posthoc output directory already exists")
    temporary.mkdir(parents=True)
    tables: dict[str, Sequence[dict[str, Any]]] = {
        "l1-residuals.csv": l1_rows,
        "l1-residual-summary.csv": l1_summary,
        "l1-cross-landscape-summary.csv": cross_l1,
        "l1-source-target-comparison.csv": source_target_l1,
        "empirical-stability-radii.csv": radius_rows,
        "kl-residuals.csv": kl_rows,
        "kl-residual-summary.csv": kl_summary,
        "horizon-features.csv": features,
        "horizon-model-comparison.csv": models,
        "source-alphas.csv": alphas,
    }
    for filename, rows in tables.items():
        _write_csv(temporary / filename, rows)
    _write_json(temporary / "artifact-verification.json", verification)

    target_t20_l1 = [
        row
        for row in l1_rows
        if row["split"] == "target"
        and row["stratum"] == "regular-sensitivity"
        and row["horizon"] == 20
    ]
    target_t20_kl = [
        row
        for row in kl_rows
        if row["split"] == "target"
        and row["stratum"] == "regular-sensitivity"
        and row["horizon"] == 20
    ]
    plot_specs = {
        "target-t20-l1-relative-residual.svg": (
            "ACL-002 target T=20: signed L1 relative residual",
            "(delta / (C epsilon) - 1) x 100 [%]",
            target_t20_l1,
            "endpoint_relative_residual",
            100.0,
        ),
        "target-t20-l1-quadratic-scale.svg": (
            "ACL-002 target T=20: L1 residual quadratic scale",
            "(delta - C epsilon) / epsilon^2",
            target_t20_l1,
            "endpoint_residual_over_epsilon_squared",
            1.0,
        ),
        "target-t20-kl-relative-residual.svg": (
            "ACL-002 target T=20: signed oriented-KL relative residual",
            "(KL / (K epsilon^2) - 1) x 100 [%]",
            [row for row in target_t20_kl if row["kl_relative_residual"] is not None],
            "kl_relative_residual",
            100.0,
        ),
    }
    for filename, (title, y_label, rows, value_key, multiplier) in plot_specs.items():
        plot_series = []
        for landscape_id in sorted({row["landscape_id"] for row in rows}):
            points = sorted(
                (
                    (float(row["epsilon"]), float(row[value_key]) * multiplier)
                    for row in rows
                    if row["landscape_id"] == landscape_id
                ),
                key=lambda point: point[0],
            )
            plot_series.append((landscape_id, points))
        (temporary / filename).write_text(
            _line_plot_svg(title=title, y_label=y_label, series=plot_series),
            encoding="utf-8",
            newline="\n",
        )

    table_hashes = {
        filename: _sha256_bytes((temporary / filename).read_bytes()) for filename in tables
    }
    table_hashes["artifact-verification.json"] = _sha256_bytes(
        (temporary / "artifact-verification.json").read_bytes()
    )
    plot_hashes = {
        filename: _sha256_bytes((temporary / filename).read_bytes()) for filename in plot_specs
    }
    summary = _phase1_summary(
        payload=payload,
        verification=verification,
        l1_rows=l1_rows,
        l1_summary=l1_summary,
        radius_rows=radius_rows,
        kl_rows=kl_rows,
        kl_summary=kl_summary,
        models=models,
        table_hashes=table_hashes,
        plot_hashes=plot_hashes,
        analysis_code_commit=code_commit,
    )
    _write_json(temporary / "summary.json", summary)
    (temporary / "ACL-002_POSTHOC.md").write_text(
        _posthoc_markdown(summary), encoding="utf-8", newline="\n"
    )
    (temporary / "README.md").write_text(
        "# ACL-002 posthoc package\n\n"
        "Deterministic derived analysis of the immutable ACL-002 artifact. See "
        "[ACL-002_POSTHOC.md](ACL-002_POSTHOC.md) and `summary.json`. Regenerate from "
        f"analysis-code commit `{code_commit}` with:\n\n"
        "```powershell\n"
        f"python -m adaptive_correspondence.acl002_posthoc --artifact "
        f"{ACL002_ARTIFACT_RELATIVE_PATH} --output-dir analysis/ACL-002-posthoc\n"
        "```\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(destination)
    return sorted(path for path in destination.iterdir() if path.is_file())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    outputs = generate_posthoc_package(
        artifact_path=args.artifact,
        output_dir=args.output_dir,
        repo_path=Path.cwd(),
    )
    print(json.dumps([path.as_posix() for path in outputs], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
