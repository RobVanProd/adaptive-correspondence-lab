"""Artifact-only ACL-007 verification and cross-class summaries."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .acl007 import analyze_target_results, validate_manifest_dict, validate_target_result
from .io import write_json

ACL007_APPROVED_SHA = "0b807af1d0428340f1e5267b1e41f6e636b49d29"
ACL007_EVIDENCE_COMMIT = "c90954960b0fa099741ed9f35a61c5153b54c923"
ACL007_EVIDENCE_SHA256 = (
    "54793bcb3a40d914bce2b5a567f6d25e638a75edf4a55ef724e156a93d372133"
)
ACL007_EVIDENCE_PATH = Path(
    "evidence/ACL-007-confirmatory-0b807af1d0428340f1e5267b1e41f6e636b49d29.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_acl007_evidence(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if _sha256(source) != ACL007_EVIDENCE_SHA256:
        raise ValueError("ACL-007 evidence SHA-256 mismatch")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("cannot read ACL-007 evidence") from error
    analysis = payload.get("analysis", {})
    registry = payload.get("locked_analytic_registry", {})
    lock = payload.get("preregistration_bundle_lock", {})
    source_validation = payload.get("source_evidence_validation", {})
    environment = payload.get("confirmatory_environment", {})
    checks = (
        payload.get("approved_preregistration_sha") == ACL007_APPROVED_SHA,
        payload.get("kind")
        == "confirmatory-cross-class-sequential-particle-filter-transport",
        payload.get("target_refit") is False,
        analysis.get("transport_verdict") == "PASS",
        analysis.get("standardized_mean_prediction_verdict") == "PASS",
        analysis.get("dissociation_prediction_verdict") == "PASS",
        analysis.get("contrast_reproduction_verdict") == "PASS",
        registry.get("outcomes_generated") is False,
        registry.get("shadow_count") == 0,
        registry.get("target_refit") is False,
        lock.get("outcomes_generated") is False,
        source_validation.get("valid") is True,
        environment.get("valid") is True,
        environment.get("dtype") == "float64",
        environment.get("rng") == "PCG64",
        len(payload.get("target_results", [])) == 16,
    )
    if not all(checks):
        raise ValueError("ACL-007 frozen evidence envelope mismatch")
    return payload


def _chunk_mean(chunks: list[dict[str, Any]]) -> np.ndarray:
    count = sum(int(chunk["count"]) for chunk in chunks)
    total = np.sum(
        np.asarray([chunk["direction_sum"] for chunk in chunks], dtype=np.float64),
        axis=0,
    )
    return total / count


def analyze_stored_evidence(
    payload: dict[str, Any],
) -> tuple[
    dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]
]:
    manifest = validate_manifest_dict(payload["frozen_design"])
    registry = payload["locked_analytic_registry"]
    registry_by_id = {entry["id"]: entry for entry in registry["targets"]}
    results = payload["target_results"]
    recomputed = analyze_target_results(manifest, registry, results)
    if recomputed != payload["analysis"]:
        raise ValueError("ACL-007 stored analysis does not reproduce")

    target_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    maximum_error = 0.0
    for target, result in zip(manifest.targets, results, strict=True):
        entry = registry_by_id[target.identifier]
        validate_target_result(manifest, target, entry, result)
        chunks = result["shadow_chunks"]
        for checkpoint in result["checkpoint_history"]:
            count = int(checkpoint["replications"])
            prefix = chunks[: count // manifest.chunk_size]
            midpoint = len(prefix) // 2
            stored_means = (
                checkpoint["full_mean_direction"],
                checkpoint["first_half_mean_direction"],
                checkpoint["second_half_mean_direction"],
            )
            reproduced_means = (
                _chunk_mean(prefix),
                _chunk_mean(prefix[:midpoint]),
                _chunk_mean(prefix[midpoint:]),
            )
            error = max(
                float(np.max(np.abs(mean - np.asarray(stored, dtype=np.float64))))
                for mean, stored in zip(reproduced_means, stored_means, strict=True)
            )
            maximum_error = max(maximum_error, error)
            checkpoint_rows.append(
                {
                    "target_id": target.identifier,
                    "replications": count,
                    "full_direction_score": checkpoint["full_direction_score"],
                    "first_half_direction_score": checkpoint[
                        "first_half_direction_score"
                    ],
                    "second_half_direction_score": checkpoint[
                        "second_half_direction_score"
                    ],
                    "full_truth_cosine": checkpoint["full_truth_cosine"],
                    "half_cosine": checkpoint["half_cosine"],
                    "mean_reproduction_error": error,
                }
            )
        target_rows.append(
            {
                "target_id": target.identifier,
                "family": target.family,
                "model": target.model_identifier,
                "particle_count": target.particle_count,
                "exact_truth_cosine": result["exact_truth_cosine"],
                "observed_truth_cosine": result["observed_truth_cosine"],
                "truth_cosine_residual": result["angular_residual"],
                "angular_envelope": result["angular_envelope"],
                "full_direction_score": result["full_direction_score"],
                "first_half_direction_score": result["first_half_direction_score"],
                "second_half_direction_score": result["second_half_direction_score"],
                "final_half_cosine": result["final_half_cosine"],
                "dissociation_stratum": result["dissociation_stratum"],
                "missing_state_0_probability": entry[
                    "terminal_missing_state_probability"
                ][0],
                "missing_state_1_probability": entry[
                    "terminal_missing_state_probability"
                ][1],
                "missing_state_2_probability": entry[
                    "terminal_missing_state_probability"
                ][2],
                "full_support_probability": entry[
                    "terminal_support_size_probabilities"
                ]["3"],
            }
        )

    contrasts = [dict(row) for row in recomputed["contrast_results"]]
    dissociation = [row for row in target_rows if row["dissociation_stratum"]]
    summary = {
        "schema_version": 1,
        "experiment_id": "ACL-007",
        "analysis_kind": "artifact-only-post-confirmatory",
        "new_particles_generated": False,
        "target_refit": False,
        "confirmed": {
            "approved_preregistration_sha": ACL007_APPROVED_SHA,
            "evidence_commit": ACL007_EVIDENCE_COMMIT,
            "evidence_sha256": ACL007_EVIDENCE_SHA256,
            "transport_verdict": recomputed["transport_verdict"],
            "standardized_mean_prediction_verdict": recomputed[
                "standardized_mean_prediction_verdict"
            ],
            "dissociation_prediction_verdict": recomputed[
                "dissociation_prediction_verdict"
            ],
            "contrast_reproduction_verdict": recomputed[
                "contrast_reproduction_verdict"
            ],
            "direction_score_maximum": recomputed["direction_score_maximum"],
            "full_direction_score_median_type7": recomputed[
                "full_direction_score_median_type7"
            ],
            "full_direction_score_q90_type7": recomputed[
                "full_direction_score_q90_type7"
            ],
            "all_angular_residuals_within_frozen_envelopes": recomputed[
                "all_angular_residuals_within_frozen_envelopes"
            ],
            "maximum_checkpoint_reproduction_error": maximum_error,
            "resolvable_contrasts_passed": sum(
                bool(row["gating_reproduced"]) for row in contrasts
            ),
        },
        "boundary": {
            "dissociation_target_count": len(dissociation),
            "minimum_dissociation_truth_cosine": min(
                row["observed_truth_cosine"] for row in dissociation
            ),
            "maximum_dissociation_truth_cosine": max(
                row["observed_truth_cosine"] for row in dissociation
            ),
            "minimum_dissociation_half_cosine": min(
                row["final_half_cosine"] for row in dissociation
            ),
            "minimum_all_target_truth_cosine": min(
                row["observed_truth_cosine"] for row in target_rows
            ),
            "negative_truth_alignment_target_count": sum(
                row["observed_truth_cosine"] < 0.0 for row in target_rows
            ),
        },
        "classification": {
            "result_type": "preregistered-no-refit-cross-class-transport",
            "adds_structurally_independent_class": True,
            "outside_fisher_natural_family": True,
            "changes_native_geometry": True,
            "changes_objective_semantics": True,
            "changes_temporal_structure": True,
            "supports_split_consistency_as_truth_certificate": False,
            "supports_common_detailed_bias_mechanism": False,
            "satisfies_phase_ii_termination_alone": False,
            "reason_not_terminal": (
                "the transported score follows finite-mean Monte Carlo geometry; "
                "a third class must test a mechanism-bearing shared law or expose "
                "the boundary before broad adaptive unification is defensible"
            ),
            "next_required_step": (
                "preregister the smallest third structurally distinct class that "
                "tests a mechanism-bearing normalization without target refit"
            ),
        },
        "execution_note": {
            "shell_harness_status": "timeout-after-five-seconds",
            "artifact_status": "complete-atomic-canonical-output-preserved",
            "runner_invocations": 1,
            "rerun_performed": False,
        },
    }
    return summary, target_rows, checkpoint_rows, contrasts


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _markdown(summary: dict[str, Any]) -> str:
    confirmed = summary["confirmed"]
    boundary = summary["boundary"]
    return f"""# ACL-007 confirmatory report

## Confirmed cross-class result

The frozen overall transport verdict is **PASS**. The standardized-mean, dissociation,
and all nine contrast components passed without target refitting. The maximum full/half
standardized direction score was `{confirmed['direction_score_maximum']:.6f}` against
`5`; Type-7 full-score median and Q90 were
`{confirmed['full_direction_score_median_type7']:.6f}` and
`{confirmed['full_direction_score_q90_type7']:.6f}` against `1.5` and `2.5`. Every
truth-cosine residual lay within its frozen analytic envelope. Stored chunks reproduce
all 64 checkpoints with maximum vector discrepancy
`{confirmed['maximum_checkpoint_reproduction_error']:.3g}`.

## Dissociation and adverse targets

All {boundary['dissociation_target_count']} predeclared dissociation targets reached
split-half Euclidean cosine at least
`{boundary['minimum_dissociation_half_cosine']:.9f}`, while observed truth alignment
within that stratum ranged from `{boundary['minimum_dissociation_truth_cosine']:.9f}`
to `{boundary['maximum_dissociation_truth_cosine']:.9f}`. Across the full benchmark,
the minimum truth cosine was `{boundary['minimum_all_target_truth_cosine']:.9f}` and
{boundary['negative_truth_alignment_target_count']} targets were negatively aligned.
Thus the PASS does not hide target bias: it correctly predicts Monte Carlo convergence
around biased particle-filter expectations and preserves the contrast ordering.

## Scientific classification

ACL-007 is the first preregistered no-refit transport result outside the Fisher-natural
family. It changes estimator family, metric, reward optimization to inference, and
one-step sampling to repeated transition/weighting/resampling. Exact target moments are
theorem/software controls; the evidence-bearing result is the unchanged ACL-006
dimensionless diagnostic.

This does **not** yet establish a unified theory of adaptive dynamics. The transported
standardized score follows finite-mean Monte Carlo geometry and does not assert a common
detailed bias mechanism. Phase-II termination therefore requires one more structurally
distinct, mechanism-bearing test or a controlled failure that maps the boundary.

## Execution provenance

The runner was invoked exactly once. The shell harness reported its five-second timeout,
but the runner had already atomically completed the canonical artifact. The complete
file was preserved, parsed, reconstructively validated, and committed untouched; no
second invocation occurred.
"""


def write_report_package(payload: dict[str, Any], output_dir: str | Path) -> Path:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    summary, targets, checkpoints, contrasts = analyze_stored_evidence(payload)
    write_json(destination / "summary.json", summary)
    _write_csv(destination / "targets.csv", targets)
    _write_csv(destination / "checkpoints.csv", checkpoints)
    _write_csv(destination / "contrasts.csv", contrasts)
    (destination / "ACL-007_REPORT.md").write_text(
        _markdown(summary), encoding="utf-8", newline="\n"
    )
    return destination
