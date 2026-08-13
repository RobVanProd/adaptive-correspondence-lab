"""Artifact-only ACL-008 verification and non-Fisher curvature summaries."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from .acl002 import type7_quantile
from .acl003 import analyze_raw_rows
from .acl008 import validate_manifest_dict
from .io import write_json

ACL008_APPROVED_SHA = "086c8187caa641a7699ee07cff540a7d8e77ba18"
ACL008_EVIDENCE_COMMIT = "c972d886edddc2dd36d60bd8229640a8eec405db"
ACL008_EVIDENCE_SHA256 = (
    "856be5ff685d65e19e029fc243a2ef40170ddf64a8b035dd1b543b484e0eba4f"
)
ACL008_EVIDENCE_PATH = Path(
    "evidence/ACL-008-confirmatory-086c8187caa641a7699ee07cff540a7d8e77ba18.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_acl008_evidence(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if _sha256(source) != ACL008_EVIDENCE_SHA256:
        raise ValueError("ACL-008 evidence SHA-256 mismatch")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("cannot read ACL-008 evidence") from error
    analysis = payload.get("analysis", {})
    registry = payload.get("locked_analytic_registry", {})
    checks = (
        payload.get("approved_preregistration_sha") == ACL008_APPROVED_SHA,
        payload.get("kind") == "confirmatory-non-fisher-second-order-transport",
        payload.get("randomness_used") is False,
        payload.get("target_refit") is False,
        analysis.get("verdict") == "PASS",
        analysis.get("instrument_valid") is True,
        registry.get("outcomes_generated") is False,
        registry.get("target_refit") is False,
        payload.get("preregistration_bundle_lock", {}).get("outcomes_generated")
        is False,
        payload.get("source_evidence_validation", {}).get("valid") is True,
        payload.get("confirmatory_environment", {}).get("valid") is True,
        len(payload.get("raw_rows", [])) == 544,
    )
    if not all(checks):
        raise ValueError("ACL-008 frozen evidence envelope mismatch")
    return payload


def analyze_stored_evidence(
    payload: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = validate_manifest_dict(payload["frozen_design"])
    registry = payload["locked_analytic_registry"]
    rows = payload["raw_rows"]
    recomputed = analyze_raw_rows(manifest, registry, rows)
    if recomputed != payload["analysis"]:
        raise ValueError("ACL-008 stored analysis does not reproduce")
    max_oracle = max(row["polynomial_oracle_max_absolute_error"] for row in rows)
    prediction_rows = [
        row
        for row in recomputed["prediction_rows"]
        if row["landscape_id"].startswith("B")
    ]
    improvement = []
    for epsilon in manifest.epsilon_grid[1:]:
        selected = [row for row in prediction_rows if row["epsilon"] == epsilon]
        improvement.append(
            {
                "epsilon": epsilon,
                "region": selected[0]["region"],
                "landscape_count": len(selected),
                "second_order_improvement_count": sum(
                    row["second_order_absolute_relative_error"]
                    < row["first_order_absolute_relative_error"]
                    for row in selected
                ),
                "second_order_median_relative_error": type7_quantile(
                    [row["second_order_absolute_relative_error"] for row in selected],
                    0.5,
                ),
                "second_order_maximum_relative_error": max(
                    row["second_order_absolute_relative_error"] for row in selected
                ),
            }
        )
    stress = [row for row in prediction_rows if row["region"] == "stress"]
    summary = {
        "schema_version": 1,
        "experiment_id": "ACL-008",
        "analysis_kind": "artifact-only-post-confirmatory",
        "new_trajectories_generated": False,
        "target_refit": False,
        "confirmed": {
            "approved_preregistration_sha": ACL008_APPROVED_SHA,
            "evidence_commit": ACL008_EVIDENCE_COMMIT,
            "evidence_sha256": ACL008_EVIDENCE_SHA256,
            "verdict": recomputed["verdict"],
            "instrument_valid": recomputed["instrument_valid"],
            "reproduced_primary_median": recomputed["primary_gate"]["median"],
            "reproduced_primary_q90": recomputed["primary_gate"]["q90"],
            "regular_target_count": recomputed["primary_gate"]["landscape_count"],
            "identity_control_passed": recomputed["software_controls_passed"],
            "polynomial_oracle_max_absolute_error": max_oracle,
        },
        "boundary": {
            "stress_results_gating": False,
            "stress_maximum_relative_error": max(
                row["second_order_absolute_relative_error"] for row in stress
            ),
            "stress_argmax": max(
                stress, key=lambda row: row["second_order_absolute_relative_error"]
            ),
            "paired_improvement_by_epsilon": improvement,
        },
        "classification": {
            "result_type": "preregistered-no-refit-cross-geometry-transport",
            "adds_distinct_geometry": True,
            "same_state_space_as_source": True,
            "supports_formal_taylor_law_only": False,
            "supports_uniform_stress_radius": False,
            "phase_ii_outcome_candidate": "U3-correspondence-lattice",
            "reason": (
                "local retraction curvature transports across entropy and Burg, while "
                "the ACL-006/007 stochastic diagnostic links a different set of systems; "
                "neither estimand is nontrivially defined across the entire graph"
            ),
        },
    }
    return summary, prediction_rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _markdown(summary: dict[str, Any]) -> str:
    confirmed = summary["confirmed"]
    boundary = summary["boundary"]
    argmax = boundary["stress_argmax"]
    return f"""# ACL-008 confirmatory report

## Confirmed cross-geometry result

ACL-008 **PASS**ed with no target fit. The maximum local error per landscape had
Type-7 median `{confirmed['reproduced_primary_median']:.9f}` and Q90
`{confirmed['reproduced_primary_q90']:.9f}`, against the copied `0.10` and `0.20`
gates. The identity control passed. The bisection and independent polynomial-root paths
agreed to maximum absolute error
`{confirmed['polynomial_oracle_max_absolute_error']:.3g}`.

The result is stronger than existence of a Taylor series: ACL-003's finite practical
radius through epsilon `0.01` transported unchanged to 16 new-value targets under a
different mirror Hessian and non-exponential retraction.

## Nonuniform boundary

Stress was explicitly non-gating. The largest stored second-order relative error was
`{boundary['stress_maximum_relative_error']:.6f}` at target
`{argmax['landscape_id']}`, epsilon `{argmax['epsilon']}`. The local law therefore does
not imply a uniform large-perturbation radius; curvature accumulation and boundary
proximity remain state dependent.

## Phase-II interpretation

ACL-008 adds a distinct non-Fisher geometry while retaining the source state space and
objective semantics. Together with ACL-007, the evidence supports a correspondence
lattice rather than one universal adaptive process: local deterministic retraction
sensitivity connects entropy and Burg mirrors; standardized stochastic mean/dissociation
connects empirical-Fisher control and sequential Euclidean inference. Each law becomes
undefined or vacuous on important nodes of the other island, so no single normalization
has earned global status.
"""


def write_report_package(payload: dict[str, Any], output_dir: str | Path) -> Path:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    summary, predictions = analyze_stored_evidence(payload)
    write_json(destination / "summary.json", summary)
    _write_csv(destination / "prediction_errors.csv", predictions)
    _write_csv(
        destination / "improvement_by_epsilon.csv",
        summary["boundary"]["paired_improvement_by_epsilon"],
    )
    (destination / "ACL-008_REPORT.md").write_text(
        _markdown(summary), encoding="utf-8", newline="\n"
    )
    return destination
