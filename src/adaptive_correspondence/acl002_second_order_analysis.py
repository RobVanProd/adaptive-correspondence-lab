"""Exploratory second-order evaluation on already-preserved ACL-002 rows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .acl002 import ACL002Manifest, load_manifest, validate_preregistration_bundle
from .acl002_posthoc import (
    ACL002_ARTIFACT_SHA256,
    ACL002_DELTA_FLOOR,
    STABILITY_LEVELS,
    verify_acl002_artifact,
)
from .categorical_second_order import (
    l1_second_order_coefficient,
    l1_truncated_prediction,
    matrix_polynomial_second_order_trajectory,
    second_order_sensitivity_trajectory,
)

ACL003_EARNING_LEVELS = (0.05, 0.10)
ACL003_MAX_RADIUS_LOSSES = 2
ACL003_MIN_MEDIAN_RADIUS_INDEX_IMPROVEMENT = 1.0
ACL003_MIN_EPSILON_001_MEDIAN_ERROR_REDUCTION = 0.50
SECOND_ORDER_STATE_ORACLE_TOLERANCE = 5e-13
SECOND_ORDER_FIRST_ORACLE_TOLERANCE = 5e-11
SECOND_ORDER_CURVATURE_ORACLE_TOLERANCE = 2e-9


def _type7(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot compute a quantile of an empty sequence")
    return float(np.quantile(np.asarray(values, dtype=np.float64), probability, method="linear"))


def _prefix_radius(
    rows: Sequence[dict[str, Any]], *, error_key: str, level: float
) -> tuple[float | None, int]:
    accepted_index = -1
    accepted_epsilon: float | None = None
    for index, row in enumerate(sorted(rows, key=lambda item: item["epsilon"])):
        error = row[error_key]
        if error is None or float(error) > level:
            break
        accepted_index = index
        accepted_epsilon = float(row["epsilon"])
    return accepted_epsilon, accepted_index


def prediction_radius_comparison(
    prediction_rows: Iterable[dict[str, Any]],
    *,
    levels: Sequence[float] = ACL003_EARNING_LEVELS,
) -> list[dict[str, Any]]:
    """Compare cumulative-prefix first- and second-order descriptive radii."""
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in prediction_rows:
        if row["stratum"] != "regular-sensitivity":
            continue
        groups[(row["landscape_id"], row["split"], int(row["horizon"]))].append(row)
    output: list[dict[str, Any]] = []
    for (landscape_id, split, horizon), rows in sorted(groups.items()):
        epsilons = [float(row["epsilon"]) for row in rows]
        if len(epsilons) != len(set(epsilons)):
            raise ValueError("duplicate epsilon in prediction-radius group")
        for level in levels:
            if not 0.0 < float(level) < 1.0:
                raise ValueError("radius levels must lie in (0,1)")
            first_radius, first_index = _prefix_radius(
                rows, error_key="first_order_absolute_relative_error", level=float(level)
            )
            second_radius, second_index = _prefix_radius(
                rows, error_key="second_order_absolute_relative_error", level=float(level)
            )
            output.append(
                {
                    "landscape_id": landscape_id,
                    "split": split,
                    "horizon": horizon,
                    "relative_error_level": float(level),
                    "first_order_radius": first_radius,
                    "second_order_radius": second_radius,
                    "first_order_radius_index": first_index,
                    "second_order_radius_index": second_index,
                    "radius_index_improvement": second_index - first_index,
                }
            )
    return output


def evaluate_acl003_earning_rule(prediction_rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Apply the frozen posthoc decision rule for whether ACL-003 is worth running."""
    rows = [
        row
        for row in prediction_rows
        if row["split"] == "target"
        and row["stratum"] == "regular-sensitivity"
        and int(row["horizon"]) == 20
    ]
    landscape_ids = sorted({row["landscape_id"] for row in rows})
    if len(landscape_ids) != 12:
        raise ValueError("ACL-003 earning rule requires 12 regular ACL-002 targets")
    radii = prediction_radius_comparison(rows, levels=ACL003_EARNING_LEVELS)
    checks: dict[str, dict[str, Any]] = {}
    checks["independent_oracles"] = {
        "passed": all(bool(row["oracle_passed"]) for row in rows),
        "failed_row_count": sum(not bool(row["oracle_passed"]) for row in rows),
    }
    for level in ACL003_EARNING_LEVELS:
        level_rows = [row for row in radii if row["relative_error_level"] == level]
        improvements = [float(row["radius_index_improvement"]) for row in level_rows]
        lost_count = sum(value < 0.0 for value in improvements)
        median_improvement = _type7(improvements, 0.5)
        label = f"radius_{int(100 * level)}_percent"
        checks[label] = {
            "passed": (
                median_improvement >= ACL003_MIN_MEDIAN_RADIUS_INDEX_IMPROVEMENT
                and lost_count <= ACL003_MAX_RADIUS_LOSSES
            ),
            "median_radius_index_improvement": median_improvement,
            "landscape_radius_loss_count": lost_count,
            "minimum_required_median_improvement": (
                ACL003_MIN_MEDIAN_RADIUS_INDEX_IMPROVEMENT
            ),
            "maximum_allowed_landscape_losses": ACL003_MAX_RADIUS_LOSSES,
        }
    epsilon_001 = [row for row in rows if float(row["epsilon"]) == 0.01]
    if len(epsilon_001) != 12:
        raise ValueError("ACL-003 earning rule requires epsilon=0.01 for every target")
    first_median = _type7(
        [float(row["first_order_absolute_relative_error"]) for row in epsilon_001], 0.5
    )
    second_median = _type7(
        [float(row["second_order_absolute_relative_error"]) for row in epsilon_001], 0.5
    )
    error_reduction = 1.0 - second_median / first_median if first_median > 0.0 else 0.0
    checks["epsilon_0.01_median_error_reduction"] = {
        "passed": error_reduction >= ACL003_MIN_EPSILON_001_MEDIAN_ERROR_REDUCTION,
        "first_order_median_absolute_relative_error": first_median,
        "second_order_median_absolute_relative_error": second_median,
        "relative_reduction": error_reduction,
        "minimum_required_relative_reduction": (
            ACL003_MIN_EPSILON_001_MEDIAN_ERROR_REDUCTION
        ),
    }
    strict = [row for row in rows if row["region"] == "confirmatory"]
    first_q90 = _type7(
        [float(row["first_order_absolute_relative_error"]) for row in strict], 0.90
    )
    second_q90 = _type7(
        [float(row["second_order_absolute_relative_error"]) for row in strict], 0.90
    )
    checks["strict_q90_nonworsening"] = {
        "passed": second_q90 <= first_q90,
        "first_order_q90_absolute_relative_error": first_q90,
        "second_order_q90_absolute_relative_error": second_q90,
    }
    return {
        "rule": "frozen-posthoc-acl003-earning-rule-v1",
        "target_landscape_count": len(landscape_ids),
        "checks": checks,
        "passed": all(check["passed"] for check in checks.values()),
    }


def _absolute_relative_error(observed: float, predicted: float) -> float | None:
    if predicted < ACL002_DELTA_FLOOR:
        return None
    return abs(observed - predicted) / predicted


def derive_second_order_prediction_rows(
    payload: dict[str, Any], manifest: ACL002Manifest
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, float]]:
    """Apply analytic second-order predictions to stored rows without regenerating outcomes."""
    landscapes = {landscape.identifier: landscape for landscape in manifest.landscapes}
    artifact_ids = {row["landscape_id"] for row in payload["raw_rows"]}
    if artifact_ids != set(landscapes):
        raise ValueError("artifact and manifest landscape IDs disagree")
    max_horizon = max(manifest.horizons)
    traces = {}
    coefficients: list[dict[str, Any]] = []
    oracle_maxima = {"state": 0.0, "first": 0.0, "second": 0.0}
    for landscape_id, landscape in sorted(landscapes.items()):
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
        for key, value in errors.items():
            oracle_maxima[key] = max(oracle_maxima[key], value)
        if errors["state"] > SECOND_ORDER_STATE_ORACLE_TOLERANCE:
            raise FloatingPointError(f"second-order state oracle mismatch for {landscape_id}")
        if errors["first"] > SECOND_ORDER_FIRST_ORACLE_TOLERANCE:
            raise FloatingPointError(f"second-order first oracle mismatch for {landscape_id}")
        if errors["second"] > SECOND_ORDER_CURVATURE_ORACLE_TOLERANCE:
            raise FloatingPointError(f"second-order curvature oracle mismatch for {landscape_id}")
        traces[landscape_id] = trace
        stratum = next(
            row["stratum"]
            for row in payload["raw_rows"]
            if row["landscape_id"] == landscape_id
        )
        for horizon in manifest.horizons:
            coefficient, zero_coordinates = l1_second_order_coefficient(
                trace.first[horizon], trace.second[horizon]
            )
            coefficients.append(
                {
                    "landscape_id": landscape_id,
                    "split": landscape.split,
                    "stratum": stratum,
                    "horizon": horizon,
                    "first_order_l1_coefficient": float(
                        np.linalg.norm(trace.first[horizon], ord=1)
                    ),
                    "second_order_l1_coefficient": coefficient,
                    "zero_first_derivative_coordinates": list(zero_coordinates),
                    "zero_first_derivative_coordinate_count": len(zero_coordinates),
                    "second_derivative_l1_norm": float(
                        np.linalg.norm(trace.second[horizon], ord=1)
                    ),
                    "state_oracle_max_absolute_error": errors["state"],
                    "first_oracle_max_absolute_error": errors["first"],
                    "second_oracle_max_absolute_error": errors["second"],
                }
            )

    predictions: list[dict[str, Any]] = []
    for raw in payload["raw_rows"]:
        epsilon = float(raw["epsilon"])
        if epsilon == 0.0:
            continue
        horizon = int(raw["horizon"])
        trace = traces[raw["landscape_id"]]
        first_prediction = float(np.linalg.norm(epsilon * trace.first[horizon], ord=1))
        second_prediction = l1_truncated_prediction(
            trace.first[horizon], trace.second[horizon], epsilon=epsilon
        )
        first_path_prediction = max(
            float(np.linalg.norm(epsilon * trace.first[step], ord=1))
            for step in range(horizon + 1)
        )
        second_path_prediction = max(
            l1_truncated_prediction(trace.first[step], trace.second[step], epsilon=epsilon)
            for step in range(horizon + 1)
        )
        observed = float(raw["endpoint_l1"])
        observed_path = float(raw["max_path_l1"])
        stored_first_prediction = float(raw["zero_fit_l1_prediction"])
        stored_path_prediction = float(raw["zero_fit_max_path_l1_prediction"])
        if abs(first_prediction - stored_first_prediction) > 5e-12:
            raise ValueError("analytic first-order endpoint prediction disagrees with artifact")
        if abs(first_path_prediction - stored_path_prediction) > 5e-12:
            raise ValueError("analytic first-order path prediction disagrees with artifact")
        coefficient, zero_coordinates = l1_second_order_coefficient(
            trace.first[horizon], trace.second[horizon]
        )
        coefficient_record = next(
            row
            for row in coefficients
            if row["landscape_id"] == raw["landscape_id"] and row["horizon"] == horizon
        )
        oracle_passed = (
            coefficient_record["state_oracle_max_absolute_error"]
            <= SECOND_ORDER_STATE_ORACLE_TOLERANCE
            and coefficient_record["first_oracle_max_absolute_error"]
            <= SECOND_ORDER_FIRST_ORACLE_TOLERANCE
            and coefficient_record["second_oracle_max_absolute_error"]
            <= SECOND_ORDER_CURVATURE_ORACLE_TOLERANCE
        )
        predictions.append(
            {
                "landscape_id": raw["landscape_id"],
                "split": raw["split"],
                "stratum": raw["stratum"],
                "horizon": horizon,
                "epsilon": epsilon,
                "region": raw["region"],
                "observed_endpoint_l1": observed,
                "first_order_prediction": first_prediction,
                "second_order_prediction": second_prediction,
                "second_order_asymptotic_l1_coefficient": coefficient,
                "zero_first_derivative_coordinates": list(zero_coordinates),
                "first_order_signed_error": observed - first_prediction,
                "second_order_signed_error": observed - second_prediction,
                "first_order_absolute_relative_error": _absolute_relative_error(
                    observed, first_prediction
                ),
                "second_order_absolute_relative_error": _absolute_relative_error(
                    observed, second_prediction
                ),
                "observed_max_path_l1": observed_path,
                "first_order_max_path_prediction": first_path_prediction,
                "second_order_max_path_prediction": second_path_prediction,
                "first_order_max_path_absolute_relative_error": _absolute_relative_error(
                    observed_path, first_path_prediction
                ),
                "second_order_max_path_absolute_relative_error": _absolute_relative_error(
                    observed_path, second_path_prediction
                ),
                "oracle_passed": oracle_passed,
            }
        )
    predictions.sort(key=lambda row: (row["landscape_id"], row["horizon"], row["epsilon"]))
    coefficients.sort(key=lambda row: (row["landscape_id"], row["horizon"]))
    return predictions, coefficients, oracle_maxima


def _numeric_summary(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise ValueError("cannot summarize empty numeric values")
    array = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(np.min(array)),
        "median_type7": _type7(list(array), 0.5),
        "q90_type7": _type7(list(array), 0.9),
        "maximum": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def _error_overview(predictions: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        if row["split"] == "target" and row["stratum"] == "regular-sensitivity":
            groups[(row["split"], row["horizon"], row["region"])].append(row)
    output = []
    for (split, horizon, region), rows in sorted(groups.items()):
        first = [float(row["first_order_absolute_relative_error"]) for row in rows]
        second = [float(row["second_order_absolute_relative_error"]) for row in rows]
        output.append(
            {
                "split": split,
                "horizon": horizon,
                "region": region,
                "row_count": len(rows),
                "first_order": _numeric_summary(first),
                "second_order": _numeric_summary(second),
            }
        )
    return output


def _radius_overview(radius_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, float], list[dict[str, Any]]] = defaultdict(list)
    for row in radius_rows:
        if row["split"] == "target":
            groups[(row["horizon"], row["relative_error_level"])].append(row)
    output = []
    for (horizon, level), rows in sorted(groups.items()):
        improvements = [float(row["radius_index_improvement"]) for row in rows]
        output.append(
            {
                "horizon": horizon,
                "relative_error_level": level,
                "landscape_count": len(rows),
                "median_radius_index_improvement_type7": _type7(improvements, 0.5),
                "q10_radius_index_improvement_type7": _type7(improvements, 0.1),
                "q90_radius_index_improvement_type7": _type7(improvements, 0.9),
                "landscape_loss_count": sum(value < 0.0 for value in improvements),
                "landscape_equal_count": sum(value == 0.0 for value in improvements),
                "landscape_gain_count": sum(value > 0.0 for value in improvements),
            }
        )
    return output


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write empty second-order table")
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            if list(row) != fields:
                raise ValueError("inconsistent second-order table columns")
            writer.writerow(
                {
                    key: (
                        json.dumps(value, separators=(",", ":"))
                        if isinstance(value, (list, dict))
                        else ""
                        if value is None
                        else value
                    )
                    for key, value in row.items()
                }
            )


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _git_code_commit(repo: Path) -> str:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.SubprocessError) as error:
        raise ValueError("cannot establish second-order analysis code provenance") from error
    if status:
        raise ValueError("second-order package generation requires committed tracked code")
    return commit


def _markdown(summary: dict[str, Any]) -> str:
    rule = summary["acl003_earning_rule"]
    overview = summary["target_error_overview"]
    t20 = [row for row in overview if row["horizon"] == 20]
    lines = [
        "# ACL-002 exploratory second-order mechanism evaluation",
        "",
        "## Status",
        "",
        "This is a post-confirmatory mechanism analysis. It uses the immutable ACL-002 "
        "rows and frozen manifest, generates no trajectories, fits no target coefficient, "
        "and cannot change the ACL-002 verdict.",
        "",
        f"- Source artifact SHA-256: `{ACL002_ARTIFACT_SHA256}`",
        f"- Analysis-code commit: `{summary['analysis_code_commit']}`",
        f"- Independent oracle maximum state error: "
        f"`{summary['oracle_max_absolute_errors']['state']:.3e}`",
        f"- Independent oracle maximum first-derivative error: "
        f"`{summary['oracle_max_absolute_errors']['first']:.3e}`",
        f"- Independent oracle maximum second-derivative error: "
        f"`{summary['oracle_max_absolute_errors']['second']:.3e}`",
        "",
        "## Derivation",
        "",
        "With `B=M-I`, `a_t=s_t J_F^R(p_t)`, and "
        "`q_t=p_t+epsilon*s_t+(epsilon^2/2)*u_t+O(epsilon^3)`, direct differentiation "
        "gives:",
        "",
        "```text",
        "u_{t+1} = u_t J_F^R(p_t) + D^2F(p_t)[s_t,s_t] + 2 a_t B.",
        "```",
        "",
        "The full derivation and the explicit nondifferentiable L1 zero-coordinate branch "
        "are in `docs/second_order_categorical.md`.",
        "",
        "## Exploratory stored-row comparison",
        "",
        "Target absolute relative errors at `T=20`:",
        "",
        "| Region | First median | Second median | First Q90 | Second Q90 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in t20:
        lines.append(
            f"| {row['region']} | {100 * row['first_order']['median_type7']:.4f}% | "
            f"{100 * row['second_order']['median_type7']:.4f}% | "
            f"{100 * row['first_order']['q90_type7']:.4f}% | "
            f"{100 * row['second_order']['q90_type7']:.4f}% |"
        )
    lines.extend(
        [
            "",
            "## Frozen ACL-003 earning rule",
            "",
            f"Overall result: **{'PASS' if rule['passed'] else 'FAIL'}**. This is only a "
            "decision about whether a new-landscape preregistration is informative; it is "
            "not confirmation of the second-order hypothesis.",
            "",
            "| Check | Result |",
            "| --- | --- |",
        ]
    )
    for name, check in rule["checks"].items():
        lines.append(f"| `{name}` | {'PASS' if check['passed'] else 'FAIL'} |")
    lines.extend(
        [
            "",
            "The local improvement does not extend uniformly into stress. At `T=20`, "
            "stress Q90 worsens from 62.17% at first order to 77.66% at second order, "
            "and the worst second-order relative error is 778.42%. This is a mapped "
            "finite-truncation failure boundary, not a reason to suppress the local result.",
            "",
            "## Interpretation",
            "",
            (
                "The earning rule passed, so the mechanism has earned an ACL-003 "
                "preregistration on entirely new categorical catalog values."
                if rule["passed"]
                else "The earning rule failed, so this mechanism does not earn ACL-003."
            ),
            "The ACL-002 improvement remains exploratory because the mechanism was "
            "selected after seeing ACL-002 residuals.",
            "",
        ]
    )
    return "\n".join(lines)


def generate_second_order_package(
    *,
    artifact_path: str | Path,
    bundle_path: str | Path,
    output_dir: str | Path,
    repo_path: str | Path,
) -> list[Path]:
    """Create a deterministic package from existing evidence and analytic inputs."""
    repo = Path(repo_path).resolve()
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise ValueError("second-order output directory already exists")
    code_commit = _git_code_commit(repo)
    payload, verification = verify_acl002_artifact(artifact_path, repo_path=repo)
    bundle = Path(bundle_path)
    bundle_validation = validate_preregistration_bundle(bundle)
    if bundle_validation["outcomes_generated"] is not False:
        raise ValueError("ACL-002 preregistration bundle unexpectedly records outcomes")
    manifest_path = bundle / "manifest.json"
    manifest = load_manifest(manifest_path)
    predictions, coefficients, oracle_maxima = derive_second_order_prediction_rows(
        payload, manifest
    )
    radii = prediction_radius_comparison(predictions, levels=STABILITY_LEVELS)
    earning_rule = evaluate_acl003_earning_rule(predictions)
    temporary = destination.with_name(destination.name + ".tmp")
    if temporary.exists():
        raise ValueError("temporary second-order output directory already exists")
    temporary.mkdir(parents=True)
    tables = {
        "second-order-predictions.csv": predictions,
        "second-order-coefficients.csv": coefficients,
        "radius-comparison.csv": radii,
    }
    for name, rows in tables.items():
        _write_csv(temporary / name, rows)
    table_hashes = {name: _sha256_file(temporary / name) for name in tables}
    summary = {
        "schema_version": 1,
        "analysis_id": "ACL-002-second-order",
        "classification": "post-confirmatory-exploratory-mechanism-evaluation",
        "outcomes_generated": False,
        "target_refit": False,
        "analysis_code_commit": code_commit,
        "source_artifact_verification": verification,
        "manifest_path": manifest_path.as_posix(),
        "manifest_sha256": _sha256_file(manifest_path),
        "bundle_validation": bundle_validation,
        "oracle_tolerances": {
            "state": SECOND_ORDER_STATE_ORACLE_TOLERANCE,
            "first": SECOND_ORDER_FIRST_ORACLE_TOLERANCE,
            "second": SECOND_ORDER_CURVATURE_ORACLE_TOLERANCE,
        },
        "oracle_max_absolute_errors": oracle_maxima,
        "prediction_row_count": len(predictions),
        "coefficient_row_count": len(coefficients),
        "target_error_overview": _error_overview(predictions),
        "target_radius_overview": _radius_overview(radii),
        "acl003_earning_rule": earning_rule,
        "derived_table_sha256": table_hashes,
        "decision": (
            "prepare-new-landscape-ACL-003-preregistration"
            if earning_rule["passed"]
            else "do-not-preregister-second-order-ACL-003"
        ),
    }
    stress_t20 = next(
        row
        for row in summary["target_error_overview"]
        if row["horizon"] == 20 and row["region"] == "stress"
    )
    summary["exploratory_failure_boundary"] = {
        "horizon": 20,
        "region": "stress",
        "first_order_q90_absolute_relative_error": stress_t20["first_order"]["q90_type7"],
        "second_order_q90_absolute_relative_error": stress_t20["second_order"]["q90_type7"],
        "second_order_maximum_absolute_relative_error": stress_t20["second_order"]["maximum"],
        "interpretation": (
            "second-order truncation is locally useful but not uniformly stress-stable"
        ),
    }
    _write_json(temporary / "summary.json", summary)
    (temporary / "ACL-002_SECOND_ORDER.md").write_text(
        _markdown(summary), encoding="utf-8", newline="\n"
    )
    (temporary / "README.md").write_text(
        "# ACL-002 second-order analysis\n\n"
        "Post-confirmatory mechanism evaluation on preserved ACL-002 rows. See "
        "`ACL-002_SECOND_ORDER.md` and `summary.json`. No new trajectories were "
        "generated.\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(destination)
    return sorted(path for path in destination.iterdir() if path.is_file())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--bundle", default="preregistrations/ACL-002")
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    outputs = generate_second_order_package(
        artifact_path=args.artifact,
        bundle_path=args.bundle,
        output_dir=args.output_dir,
        repo_path=Path.cwd(),
    )
    print(json.dumps([path.as_posix() for path in outputs], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
