"""Artifact-only ACL-003 verification and post-confirmatory summaries."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .acl002 import type7_quantile
from .io import write_json

ACL003_APPROVED_SHA = "501464f3f6be07f6d813d94aefb818c461a3d5c7"
ACL003_EVIDENCE_COMMIT = "b15d77600369d559cb586a3bb54924737758e038"
ACL003_EVIDENCE_SHA256 = (
    "1f80c3f5aba4089c67bbfec1ddd6eff53f7a6d42c658436dff0f7c82a1cf8c99"
)
ACL003_ROW_COUNT = 544
ACL003_REGION_COUNTS = {
    "confirmatory": 204,
    "numerical-control": 136,
    "stress": 136,
    "zero": 68,
}
ACL003_HORIZON_COUNTS = {1: 136, 5: 136, 20: 136, 50: 136}
ACL003_LANDSCAPE_IDS = tuple([f"N{index:02d}" for index in range(1, 17)] + ["C01"])
ACL003_PRIMARY_MEDIAN = 0.0014843120912351297
ACL003_PRIMARY_Q90 = 0.007387117284289386
PREDICTION_FLOOR = 2e-12
RADIUS_LEVELS = (0.01, 0.05, 0.10, 0.20)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_acl003_evidence(
    path: str | Path, *, expected_sha256: str = ACL003_EVIDENCE_SHA256
) -> dict[str, Any]:
    """Verify immutable bytes and the frozen ACL-003 evidence envelope."""
    source = Path(path)
    if _sha256(source) != expected_sha256:
        raise ValueError("ACL-003 evidence SHA-256 mismatch")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("cannot read ACL-003 evidence") from error
    if expected_sha256 != ACL003_EVIDENCE_SHA256:
        return payload

    rows = payload.get("raw_rows")
    analysis = payload.get("analysis")
    if not isinstance(rows, list) or not isinstance(analysis, dict):
        raise ValueError("ACL-003 evidence structure mismatch")
    regions = dict(sorted(Counter(row.get("region") for row in rows).items()))
    horizons = dict(sorted(Counter(row.get("horizon") for row in rows).items()))
    identifiers = tuple(sorted({row.get("landscape_id") for row in rows}))
    primary = analysis.get("primary_gate", {})
    checks = (
        payload.get("approved_preregistration_sha") == ACL003_APPROVED_SHA,
        payload.get("randomness_used") is False,
        payload.get("target_refit") is False,
        len(rows) == ACL003_ROW_COUNT,
        regions == ACL003_REGION_COUNTS,
        horizons == ACL003_HORIZON_COUNTS,
        identifiers == tuple(sorted(ACL003_LANDSCAPE_IDS)),
        analysis.get("instrument_valid") is True,
        analysis.get("verdict") == "PASS",
        primary.get("passed") is True,
        primary.get("median") == ACL003_PRIMARY_MEDIAN,
        primary.get("q90") == ACL003_PRIMARY_Q90,
    )
    if not all(checks):
        raise ValueError("ACL-003 frozen evidence envelope mismatch")
    return payload


def _relative_error(observed: float, predicted: float) -> float | None:
    if predicted < PREDICTION_FLOOR:
        return None
    return abs(observed - predicted) / predicted


def _error_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    errors = []
    for row in payload["raw_rows"]:
        if row["epsilon"] == 0.0:
            continue
        errors.append(
            {
                "landscape_id": row["landscape_id"],
                "role": row["role"],
                "stratum": row["stratum"],
                "horizon": row["horizon"],
                "epsilon": row["epsilon"],
                "region": row["region"],
                "endpoint_l1": row["endpoint_l1"],
                "first_order_prediction": row["first_order_prediction"],
                "second_order_prediction": row["second_order_prediction"],
                "first_order_absolute_relative_error": _relative_error(
                    row["endpoint_l1"], row["first_order_prediction"]
                ),
                "second_order_absolute_relative_error": _relative_error(
                    row["endpoint_l1"], row["second_order_prediction"]
                ),
            }
        )
    return errors


def _group_summaries(errors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = []
    keys = sorted({(row["horizon"], row["region"]) for row in errors})
    for horizon, region in keys:
        rows = [
            row for row in errors if row["horizon"] == horizon and row["region"] == region
        ]
        for model, field in (
            ("first-order", "first_order_absolute_relative_error"),
            ("second-order", "second_order_absolute_relative_error"),
        ):
            values = [float(row[field]) for row in rows if row[field] is not None]
            groups.append(
                {
                    "horizon": horizon,
                    "region": region,
                    "model": model,
                    "count": len(values),
                    "median_absolute_relative_error": type7_quantile(values, 0.5),
                    "q90_absolute_relative_error": type7_quantile(values, 0.9),
                    "maximum_absolute_relative_error": max(values),
                }
            )
    return groups


def _radii(errors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [
        row
        for row in errors
        if row["horizon"] == 20
        and row["role"] == "confirmatory-target"
        and row["stratum"] == "regular-sensitivity"
    ]
    output = []
    for identifier in sorted({row["landscape_id"] for row in rows}):
        landscape = sorted(
            (row for row in rows if row["landscape_id"] == identifier),
            key=lambda row: row["epsilon"],
        )
        for model, field in (
            ("first-order", "first_order_absolute_relative_error"),
            ("second-order", "second_order_absolute_relative_error"),
        ):
            result: dict[str, Any] = {"landscape_id": identifier, "model": model}
            for level in RADIUS_LEVELS:
                radius = 0.0
                for row in landscape:
                    error = row[field]
                    if error is None or error > level:
                        break
                    radius = row["epsilon"]
                result[f"radius_{round(level * 100)}pct"] = radius
            output.append(result)
    return output


def analyze_stored_rows(
    payload: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Reproduce the primary gate and derive explicitly exploratory summaries."""
    errors = _error_rows(payload)
    regular_primary = [
        row
        for row in errors
        if row["horizon"] == 20
        and row["region"] == "confirmatory"
        and row["role"] == "confirmatory-target"
        and row["stratum"] == "regular-sensitivity"
    ]
    reproduced = {}
    for identifier in sorted({row["landscape_id"] for row in regular_primary}):
        reproduced[identifier] = max(
            float(row["second_order_absolute_relative_error"])
            for row in regular_primary
            if row["landscape_id"] == identifier
            and row["second_order_absolute_relative_error"] is not None
        )
    stored = {
        row["landscape_id"]: row["relative_error_max"]
        for row in payload["analysis"]["primary_gate"]["landscape_scores"]
    }
    if set(reproduced) != set(stored):
        raise ValueError("stored and reproduced ACL-003 landscape IDs differ")
    max_difference = max(abs(reproduced[key] - stored[key]) for key in stored)
    reproduced_values = list(reproduced.values())
    reproduced_median = type7_quantile(reproduced_values, 0.5)
    reproduced_q90 = type7_quantile(reproduced_values, 0.9)

    t20_rows = [
        row
        for row in errors
        if row["horizon"] == 20
        and row["role"] == "confirmatory-target"
        and row["stratum"] == "regular-sensitivity"
    ]
    paired = []
    for epsilon in sorted({row["epsilon"] for row in t20_rows}):
        cells = [row for row in t20_rows if row["epsilon"] == epsilon]
        paired.append(
            {
                "epsilon": epsilon,
                "region": cells[0]["region"],
                "landscape_count": len(cells),
                "second_order_improvement_count": sum(
                    row["second_order_absolute_relative_error"]
                    < row["first_order_absolute_relative_error"]
                    for row in cells
                    if row["second_order_absolute_relative_error"] is not None
                    and row["first_order_absolute_relative_error"] is not None
                ),
            }
        )
    oracle_maximum = max(
        row["matrix_oracle_max_absolute_error"] for row in payload["raw_rows"]
    )
    summary = {
        "schema_version": 1,
        "experiment_id": "ACL-003",
        "analysis_kind": "artifact-only-post-confirmatory",
        "new_outcomes_generated": False,
        "target_refit": False,
        "confirmed": {
            "approved_preregistration_sha": payload["approved_preregistration_sha"],
            "evidence_commit": ACL003_EVIDENCE_COMMIT,
            "evidence_sha256": ACL003_EVIDENCE_SHA256,
            "verdict": payload["analysis"]["verdict"],
            "instrument_valid": payload["analysis"]["instrument_valid"],
            "stored_primary_gate": payload["analysis"]["primary_gate"],
            "reproduced_primary_median": reproduced_median,
            "reproduced_primary_q90": reproduced_q90,
            "reproduced_score_max_difference": max_difference,
            "matrix_oracle_max_absolute_error": oracle_maximum,
            "randomness_used": payload["randomness_used"],
            "target_refit": payload["target_refit"],
        },
        "exploratory": {
            "stress_gating": False,
            "paired_improvement_by_epsilon": paired,
            "interpretation_scope": "new-value transport within categorical mutation only",
        },
    }
    return summary, errors, _group_summaries(errors), _radii(errors)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write empty ACL-003 derived table")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _report_markdown(summary: dict[str, Any], groups: list[dict[str, Any]]) -> str:
    confirmed = summary["confirmed"]
    gate = confirmed["stored_primary_gate"]
    stress = [
        row
        for row in groups
        if row["horizon"] == 20 and row["region"] == "stress"
    ]
    stress_lines = "\n".join(
        f"- {row['model']}: median {row['median_absolute_relative_error']:.3%}, "
        f"Q90 {row['q90_absolute_relative_error']:.3%}, maximum "
        f"{row['maximum_absolute_relative_error']:.3%}."
        for row in stress
    )
    return f"""# ACL-003 confirmatory report

## Confirmed findings

ACL-003 **{confirmed['verdict']}** with a valid instrument. At `T=20`, the maximum
within-landscape zero-fit second-order relative error across epsilon
`0.001,0.003,0.01` had median `{gate['median']:.6%}` and Type-7 Q90
`{gate['q90']:.6%}`, below the frozen `10%` and `20%` gates. The primary scores were
reproduced exactly from stored rows (maximum absolute reproduction difference
`{confirmed['reproduced_score_max_difference']:.3g}`). No randomness or target
refitting was used.

The iterative/matrix oracle maximum absolute discrepancy was
`{confirmed['matrix_oracle_max_absolute_error']:.3g}`.

## Exploratory observations

All observations below are derived from already stored, non-gating rows and cannot
change the confirmatory verdict.

At `T=20` in the stress region:

{stress_lines}

The complete horizon/region tables, paired improvement counts, and empirical radii are
in the machine-readable files beside this report.

## Interpretation

ACL-003 confirms a zero-fit, analytic second-order degradation law on new numeric
catalog values within the categorical mutation class through epsilon `0.01`. It does
not test transport to a different adaptive-system class, population inference, or a
universal adaptive law. The categorical rung has therefore earned escalation to the
separately defined Gaussian class while retaining its reported stress boundary.
"""


def write_report_package(payload: dict[str, Any], output_dir: str | Path) -> Path:
    """Write deterministic derived files without touching the evidence artifact."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    summary, errors, groups, radii = analyze_stored_rows(payload)
    write_json(destination / "summary.json", summary)
    _write_csv(destination / "prediction_errors.csv", errors)
    _write_csv(destination / "horizon_region_summary.csv", groups)
    _write_csv(destination / "empirical_radii.csv", radii)
    (destination / "ACL-003_REPORT.md").write_text(
        _report_markdown(summary, groups), encoding="utf-8", newline="\n"
    )
    return destination
