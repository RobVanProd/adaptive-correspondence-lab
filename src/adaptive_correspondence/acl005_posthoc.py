"""Artifact-only ACL-005 verification and cross-class transport summaries."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .acl002 import type7_quantile
from .acl005 import reproduce_stopped_mean, validate_manifest_dict
from .control_finite_sample_bridge import context_fisher_cosines, joint_fisher_cosine
from .io import write_json

ACL005_APPROVED_SHA = "c3ebc07a41e8dbb84a24c68cdbb4f75c36108c5b"
ACL005_EVIDENCE_COMMIT = "24d577f8a1d7bc6f4f45250f4bab3d5b2b925aeb"
ACL005_EVIDENCE_SHA256 = (
    "5400a12392609f5cdf79a8b4b380f84ad11e68330f8ee93f653439129aa5db5b"
)
ACL005_EVIDENCE_PATH = Path(
    "evidence/ACL-005-confirmatory-c3ebc07a41e8dbb84a24c68cdbb4f75c36108c5b.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_acl005_evidence(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if _sha256(source) != ACL005_EVIDENCE_SHA256:
        raise ValueError("ACL-005 evidence SHA-256 mismatch")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("cannot read ACL-005 evidence") from error
    analysis = payload.get("analysis", {})
    results = payload.get("landscape_results", [])
    registry = payload.get("locked_analytic_registry", {})
    lock = payload.get("preregistration_bundle_lock", {})
    source_validation = payload.get("source_evidence_validation", {})
    regular = [row for row in results if row.get("role") == "confirmatory-target"]
    stress = [row for row in results if row.get("role") == "stress-target"]
    checks = (
        payload.get("approved_preregistration_sha") == ACL005_APPROVED_SHA,
        payload.get("kind")
        == "confirmatory-cross-class-contextual-bandit-conditional-mean",
        payload.get("target_refit") is False,
        analysis.get("transport_verdict") == "PASS",
        analysis.get("all_regular_landscapes_converged") is True,
        analysis.get("stress_gating") is False,
        analysis.get("joint_cosine_gating") is False,
        len(regular) == 10,
        len(stress) == 4,
        all(row.get("stopped_replications") == 4096 for row in results),
        registry.get("outcomes_generated") is False,
        registry.get("shadow_count") == 0,
        lock.get("outcomes_generated") is False,
        source_validation.get("valid") is True,
        source_validation.get("evidence_sha256")
        == "3f97f7c4debbd65014e6ee337d9a8990500bad38ce0dfef2e8ad3048c74cd91a",
    )
    if not all(checks):
        raise ValueError("ACL-005 frozen evidence envelope mismatch")
    return payload


def _chunk_mean(chunks: list[dict[str, Any]]) -> np.ndarray:
    return reproduce_stopped_mean({"shadow_chunks": chunks})


def _uncertainty(chunks: list[dict[str, Any]]) -> np.ndarray:
    count = sum(int(chunk["count"]) for chunk in chunks)
    mean = _chunk_mean(chunks).ravel()
    outer = np.sum(
        np.asarray([chunk["direction_outer_sum"] for chunk in chunks], dtype=np.float64),
        axis=0,
    )
    covariance = (outer - count * np.outer(mean, mean)) / (count - 1)
    covariance = 0.5 * (covariance + covariance.T)
    return np.sqrt(np.maximum(np.diag(covariance), 0.0) / count)


def _h1_row(
    *,
    landscape_id: str,
    role: str,
    context: int,
    values: list[float | None],
) -> dict[str, Any]:
    defined = [float(value) for value in values if value is not None]
    row: dict[str, Any] = {
        "landscape_id": landscape_id,
        "role": role,
        "context": context,
        "count": len(values),
        "defined_count": len(defined),
        "undefined_count": len(values) - len(defined),
    }
    if defined:
        row.update(
            {
                "q10": type7_quantile(defined, 0.1),
                "median": type7_quantile(defined, 0.5),
                "q90": type7_quantile(defined, 0.9),
                "fraction_positive_among_defined": sum(value > 0.0 for value in defined)
                / len(defined),
            }
        )
    else:
        row.update(
            {
                "q10": None,
                "median": None,
                "q90": None,
                "fraction_positive_among_defined": None,
            }
        )
    return row


def analyze_stored_evidence(
    payload: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    manifest = validate_manifest_dict(payload["frozen_design"])
    landscapes = {landscape.identifier: landscape for landscape in manifest.landscapes}
    registry = {
        row["id"]: row for row in payload["locked_analytic_registry"]["landscapes"]
    }
    stored_h1 = {
        (row["landscape_id"], row["context"]): row
        for row in payload["analysis"]["h1_descriptive"]
    }
    context_rows = []
    checkpoint_rows = []
    h1_rows = []
    mean_errors = []
    cosine_errors = []
    uncertainty_errors = []
    h1_errors = []
    for result in payload["landscape_results"]:
        identifier = result["landscape_id"]
        landscape = landscapes[identifier]
        analytic = np.asarray(result["analytic_direction"], dtype=np.float64)
        stored_mean = np.asarray(result["stopped_mean_direction"], dtype=np.float64)
        reproduced = reproduce_stopped_mean(result)
        mean_error = float(np.max(np.abs(reproduced - stored_mean)))
        mean_errors.append(mean_error)
        context_cosines = context_fisher_cosines(landscape.state, reproduced, analytic)
        joint = joint_fisher_cosine(landscape.state, reproduced, analytic)
        cosine_error = max(
            max(
                abs(left - right)
                for left, right in zip(
                    context_cosines, result["final_context_cosines"], strict=True
                )
            ),
            abs(joint - result["joint_cosine"]),
        )
        cosine_errors.append(cosine_error)
        reproduced_se = _uncertainty(result["shadow_chunks"])
        stored_se = np.asarray(
            result["shadow_uncertainty"]["coordinate_standard_error_of_mean"],
            dtype=np.float64,
        ).ravel()
        uncertainty_errors.append(float(np.max(np.abs(reproduced_se - stored_se))))
        rewards, contexts, _, policy = landscape.state.arrays()
        for context, cosine in enumerate(context_cosines):
            context_rows.append(
                {
                    "landscape_id": identifier,
                    "role": result["role"],
                    "context": context,
                    "stopped_replications": result["stopped_replications"],
                    "minimum_expected_cell_count_in_context": float(
                        manifest.interaction_sample_count
                        * contexts[context]
                        * np.min(policy[context])
                    ),
                    "analytic_fisher_norm": registry[identifier]["context_fisher_norms"][
                        context
                    ],
                    "context_fisher_cosine": cosine,
                    "joint_fisher_cosine": joint,
                    "maximum_coordinate_standard_error": float(
                        np.max(reproduced_se.reshape(rewards.shape)[context])
                    ),
                    "stopped_mean_reproduction_error": mean_error,
                    "final_cosine_reproduction_error": cosine_error,
                }
            )
        chunks = result["shadow_chunks"]
        for checkpoint in result["checkpoint_history"]:
            count = checkpoint["replications"]
            used = chunks[: count // manifest.chunk_size]
            half = len(used) // 2
            first = _chunk_mean(used[:half])
            second = _chunk_mean(used[half:])
            reproduced_half = context_fisher_cosines(landscape.state, first, second)
            for context, cosine in enumerate(reproduced_half):
                stored = checkpoint["half_context_fisher_cosines"][context]
                checkpoint_rows.append(
                    {
                        "landscape_id": identifier,
                        "role": result["role"],
                        "replications": count,
                        "context": context,
                        "half_context_fisher_cosine": cosine,
                        "half_cosine_reproduction_error": abs(cosine - stored),
                    }
                )
        for context, values in enumerate(result["h1_context_cosines"]):
            row = _h1_row(
                landscape_id=identifier,
                role=result["role"],
                context=context,
                values=values,
            )
            stored = stored_h1[(identifier, context)]
            errors = [
                abs(row[key] - stored[key])
                for key in (
                    "q10",
                    "median",
                    "q90",
                    "fraction_positive_among_defined",
                )
                if row[key] is not None and stored[key] is not None
            ]
            errors.extend(
                abs(row[key] - stored[key])
                for key in ("count", "defined_count", "undefined_count")
            )
            error = max(errors, default=0.0)
            h1_errors.append(error)
            row["summary_reproduction_error"] = error
            h1_rows.append(row)

    regular_rows = [row for row in context_rows if row["role"] == "confirmatory-target"]
    stress_rows = [row for row in context_rows if row["role"] == "stress-target"]
    regular_h1 = [row for row in h1_rows if row["role"] == "confirmatory-target"]
    stress_h1 = [row for row in h1_rows if row["role"] == "stress-target"]
    analysis = payload["analysis"]
    summary = {
        "schema_version": 1,
        "experiment_id": "ACL-005",
        "analysis_kind": "artifact-only-post-confirmatory",
        "new_shadows_generated": False,
        "target_refit": False,
        "confirmed": {
            "approved_preregistration_sha": ACL005_APPROVED_SHA,
            "evidence_commit": ACL005_EVIDENCE_COMMIT,
            "evidence_sha256": ACL005_EVIDENCE_SHA256,
            "transport_verdict": analysis["transport_verdict"],
            "all_regular_landscapes_converged": analysis[
                "all_regular_landscapes_converged"
            ],
            "convergence_threshold": analysis["convergence_threshold"],
            "transport_threshold": analysis["transport_threshold"],
            "regular_minimum_context_fisher_cosine": min(
                row["context_fisher_cosine"] for row in regular_rows
            ),
            "all_regular_stopped_replications": sorted(
                {row["stopped_replications"] for row in regular_rows}
            ),
            "maximum_stopped_mean_reproduction_error": max(mean_errors),
            "maximum_final_cosine_reproduction_error": max(cosine_errors),
            "maximum_uncertainty_reproduction_error": max(uncertainty_errors),
            "maximum_h1_summary_reproduction_error": max(h1_errors),
            "stress_gating": False,
            "joint_cosine_gating": False,
        },
        "transport": {
            "source_class": "Gaussian finite-lambda rank-mu (ACL-004)",
            "target_class": "finite-state contextual-bandit empirical-Fisher NPG",
            "quantity": (
                "unchanged blockwise rule: disjoint-half Fisher cosine >= 0.98 "
                "then analytic Fisher cosine >= 0.99"
            ),
            "target_refit": False,
            "source_evidence_sha256": payload["source_evidence_validation"][
                "evidence_sha256"
            ],
        },
        "stress_boundary": {
            "gating": False,
            "landscape_count": 4,
            "all_stress_landscapes_converged": all(
                row["converged"]
                for row in payload["landscape_results"]
                if row["role"] == "stress-target"
            ),
            "minimum_context_fisher_cosine": min(
                row["context_fisher_cosine"] for row in stress_rows
            ),
            "context_blocks_below_transport_threshold": sum(
                row["context_fisher_cosine"] < manifest.h2_cosine_min
                for row in stress_rows
            ),
            "landscapes_below_transport_threshold": sorted(
                {
                    row["landscape_id"]
                    for row in stress_rows
                    if row["context_fisher_cosine"] < manifest.h2_cosine_min
                }
            ),
            "landscape_minimum_expected_cell_count_range": [
                min(
                    registry[row["landscape_id"]][
                        "minimum_expected_joint_cell_count_per_shadow"
                    ]
                    for row in stress_rows
                ),
                max(
                    registry[row["landscape_id"]][
                        "minimum_expected_joint_cell_count_per_shadow"
                    ]
                    for row in stress_rows
                ),
            ],
        },
        "descriptive_h1": {
            "regular_context_median_range": [
                min(row["median"] for row in regular_h1 if row["median"] is not None),
                max(row["median"] for row in regular_h1 if row["median"] is not None),
            ],
            "stress_context_median_range": [
                min(row["median"] for row in stress_h1 if row["median"] is not None),
                max(row["median"] for row in stress_h1 if row["median"] is not None),
            ],
            "regular_undefined_fraction": sum(row["undefined_count"] for row in regular_h1)
            / sum(row["count"] for row in regular_h1),
            "stress_undefined_fraction": sum(row["undefined_count"] for row in stress_h1)
            / sum(row["count"] for row in stress_h1),
        },
        "termination_assessment": {
            "candidate": "Outcome A - predictive unification survives in restricted form",
            "reason": (
                "a preregistered normalized law transported from Gaussian rank-mu to "
                "contextual-bandit NPG without target refitting, while frozen stress "
                "targets identify a rare-cell breakdown regime"
            ),
            "universal_claim_supported": False,
        },
    }
    return summary, context_rows, h1_rows, checkpoint_rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _markdown(summary: dict[str, Any]) -> str:
    confirmed = summary["confirmed"]
    stress = summary["stress_boundary"]
    h1 = summary["descriptive_h1"]
    return f"""# ACL-005 confirmatory report

## Confirmed cross-class finding

ACL-005 **{confirmed['transport_verdict']}**. All 10 regular control landscapes met
the source-domain disjoint-half convergence rule at 4096 shadows. The minimum over all
20 regular context blocks was `{confirmed['regular_minimum_context_fisher_cosine']:.9f}`
against the unchanged `0.99` analytic-alignment threshold. The `0.98` stopping rule,
replication schedule, and analytic gate came from ACL-004; no control-target refit
occurred. Joint cosine and stress targets were non-gating.

Stored chunk sums reproduce every stopped mean with maximum error
`{confirmed['maximum_stopped_mean_reproduction_error']:.3g}`; final cosines reproduce
within `{confirmed['maximum_final_cosine_reproduction_error']:.3g}`.

## Frozen stress boundary

All four stress landscapes also met the half-mean stopping rule at 4096, but
{stress['context_blocks_below_transport_threshold']} of eight stress context blocks
fell below `0.99`. The minimum was `{stress['minimum_context_fisher_cosine']:.6f}`;
affected landscapes were {', '.join(stress['landscapes_below_transport_threshold'])}.
Thus half-mean self-consistency does not guarantee truth alignment under rare-cell
empirical-Fisher inversion. This boundary was predeclared and cannot alter the regular
PASS.

## Descriptive single-shadow behavior

Regular per-context single-shadow median cosines range from
`{h1['regular_context_median_range'][0]:.3f}` to
`{h1['regular_context_median_range'][1]:.3f}`. Stress medians range from
`{h1['stress_context_median_range'][0]:.3f}` to
`{h1['stress_context_median_range'][1]:.3f}`. Undefined zero-norm directions comprise
`{100*h1['stress_undefined_fraction']:.2f}%` of stress H1 context shadows. These H1
summaries are descriptive.

## Scope

The evidence supports transport of one restricted, dimensionless expected-direction
diagnostic from Gaussian rank-mu to finite-state contextual-bandit NPG. It does not
support sequential MDPs, neural policies, PPO, sample-count scaling, or a universal
identity among adaptive systems. The stress result makes the scope condition material:
adequate joint-cell coverage is part of the bridge.
"""


def write_report_package(payload: dict[str, Any], output_dir: str | Path) -> Path:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    summary, contexts, h1_rows, checkpoints = analyze_stored_evidence(payload)
    write_json(destination / "summary.json", summary)
    _write_csv(destination / "context_alignment.csv", contexts)
    _write_csv(destination / "h1_single_shadow.csv", h1_rows)
    _write_csv(destination / "convergence_checkpoints.csv", checkpoints)
    (destination / "ACL-005_REPORT.md").write_text(
        _markdown(summary), encoding="utf-8", newline="\n"
    )
    return destination
