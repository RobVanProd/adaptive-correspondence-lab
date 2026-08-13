"""Artifact-only ACL-006 verification and mechanism summaries."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .acl006 import (
    analyze_target_results,
    validate_manifest_dict,
    validate_target_result,
)
from .io import write_json

ACL006_APPROVED_SHA = "a8b42042e397f1422866a0ca9496ee07abe0a42a"
ACL006_EVIDENCE_COMMIT = "c94890dc8f361c0309802c0ef0173ec84e814d3d"
ACL006_EVIDENCE_SHA256 = (
    "740c541bbd69db77f6d02327ded34765a37345f907b048f8d3f3a91aebc23918"
)
ACL006_EVIDENCE_PATH = Path(
    "evidence/ACL-006-confirmatory-a8b42042e397f1422866a0ca9496ee07abe0a42a.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_acl006_evidence(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if _sha256(source) != ACL006_EVIDENCE_SHA256:
        raise ValueError("ACL-006 evidence SHA-256 mismatch")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("cannot read ACL-006 evidence") from error
    analysis = payload.get("analysis", {})
    registry = payload.get("locked_analytic_registry", {})
    lock = payload.get("preregistration_bundle_lock", {})
    environment = payload.get("confirmatory_environment", {})
    checks = (
        payload.get("approved_preregistration_sha") == ACL006_APPROVED_SHA,
        payload.get("kind") == "confirmatory-exact-support-conditioned-angular-bias",
        payload.get("target_refit") is False,
        analysis.get("exact_mean_prediction_verdict") == "PASS",
        analysis.get("dissociation_prediction_verdict") == "PASS",
        analysis.get("stochastic_contrast_reproduction_verdict") == "PASS",
        analysis.get("self_consistency_certifies_truth") is False,
        registry.get("outcomes_generated") is False,
        registry.get("shadow_count") == 0,
        registry.get("target_refit") is False,
        lock.get("outcomes_generated") is False,
        environment.get("valid") is True,
        environment.get("dtype") == "float64",
        environment.get("rng") == "PCG64",
        len(payload.get("target_results", [])) == 16,
    )
    if not all(checks):
        raise ValueError("ACL-006 frozen evidence envelope mismatch")
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
    recomputed_analysis = analyze_target_results(manifest, registry, results)
    if recomputed_analysis != payload["analysis"]:
        raise ValueError("ACL-006 stored analysis does not reproduce")

    target_rows = []
    checkpoint_rows = []
    maximum_error = 0.0
    for target, result in zip(manifest.targets, results, strict=True):
        entry = registry_by_id[target.identifier]
        validate_target_result(manifest, target, entry, result)
        chunks = result["shadow_chunks"]
        for checkpoint in result["checkpoint_history"]:
            count = int(checkpoint["replications"])
            prefix = chunks[: count // manifest.chunk_size]
            midpoint = len(prefix) // 2
            full = _chunk_mean(prefix)
            first = _chunk_mean(prefix[:midpoint])
            second = _chunk_mean(prefix[midpoint:])
            errors = (
                float(
                    np.max(
                        np.abs(
                            full
                            - np.asarray(
                                checkpoint["full_mean_direction"], dtype=np.float64
                            )
                        )
                    )
                ),
                float(
                    np.max(
                        np.abs(
                            first
                            - np.asarray(
                                checkpoint["first_half_mean_direction"], dtype=np.float64
                            )
                        )
                    )
                ),
                float(
                    np.max(
                        np.abs(
                            second
                            - np.asarray(
                                checkpoint["second_half_mean_direction"], dtype=np.float64
                            )
                        )
                    )
                ),
            )
            reproduction_error = max(errors)
            maximum_error = max(maximum_error, reproduction_error)
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
                    "mean_reproduction_error": reproduction_error,
                }
            )
        target_rows.append(
            {
                "target_id": target.identifier,
                "family": target.family,
                "sample_count": target.sample_count,
                "effective_minimum_count": entry["effective_minimum_count"],
                "positive_fisher_condition_number": entry[
                    "positive_fisher_condition_number"
                ],
                "rank_deficient_probability": entry["rank_deficient_probability"],
                "expected_squared_support_loss": entry[
                    "expected_squared_support_loss"
                ],
                "expected_squared_observed_support_perturbation": entry[
                    "expected_squared_observed_support_perturbation"
                ],
                "exact_truth_cosine": result["exact_truth_cosine"],
                "observed_truth_cosine": result["observed_truth_cosine"],
                "truth_cosine_residual": result["angular_residual"],
                "angular_envelope": result["angular_envelope"],
                "full_direction_score": result["full_direction_score"],
                "first_half_direction_score": result["first_half_direction_score"],
                "second_half_direction_score": result["second_half_direction_score"],
                "final_half_cosine": result["final_half_cosine"],
                "dissociation_stratum": result["dissociation_stratum"],
            }
        )

    contrast_rows = [dict(row) for row in recomputed_analysis["contrast_results"]]
    dissociation = [row for row in target_rows if row["dissociation_stratum"]]
    summary = {
        "schema_version": 1,
        "experiment_id": "ACL-006",
        "analysis_kind": "artifact-only-post-confirmatory",
        "new_shadows_generated": False,
        "target_refit": False,
        "confirmed": {
            "approved_preregistration_sha": ACL006_APPROVED_SHA,
            "evidence_commit": ACL006_EVIDENCE_COMMIT,
            "evidence_sha256": ACL006_EVIDENCE_SHA256,
            "exact_mean_prediction_verdict": recomputed_analysis[
                "exact_mean_prediction_verdict"
            ],
            "dissociation_prediction_verdict": recomputed_analysis[
                "dissociation_prediction_verdict"
            ],
            "stochastic_contrast_reproduction_verdict": recomputed_analysis[
                "stochastic_contrast_reproduction_verdict"
            ],
            "direction_score_maximum": recomputed_analysis[
                "direction_score_maximum"
            ],
            "full_direction_score_median_type7": recomputed_analysis[
                "full_direction_score_median_type7"
            ],
            "full_direction_score_q90_type7": recomputed_analysis[
                "full_direction_score_q90_type7"
            ],
            "all_angular_residuals_within_frozen_envelopes": recomputed_analysis[
                "all_angular_residuals_within_frozen_envelopes"
            ],
            "maximum_checkpoint_reproduction_error": maximum_error,
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
            "effective_minimum_count_only_law": recomputed_analysis[
                "effective_minimum_count_only_law"
            ],
            "support_and_fisher_spectrum_only_law": recomputed_analysis[
                "support_and_fisher_spectrum_only_law"
            ],
            "resolvable_contrast_count": sum(
                bool(row["resolvable"]) for row in contrast_rows
            ),
        },
        "classification": {
            "result_type": "theorem-software-reproduction-and-mechanism-validation",
            "adds_structurally_independent_class": False,
            "supports_split_consistency_as_truth_certificate": False,
            "supports_scalar_effective_count_law": False,
            "next_required_step": (
                "preregister no-refit transport into a structurally distinct "
                "non-Fisher class"
            ),
        },
    }
    return summary, target_rows, checkpoint_rows, contrast_rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _markdown(summary: dict[str, Any]) -> str:
    confirmed = summary["confirmed"]
    boundary = summary["boundary"]
    return f"""# ACL-006 confirmatory report

## Confirmed mechanism result

All three frozen verdicts **PASS**: exact finite-mean prediction, split-half/truth
dissociation, and stochastic contrast reproduction. The maximum normalized full/half
direction score was `{confirmed['direction_score_maximum']:.6f}` against `5`; the
Type-7 full-score median and Q90 were
`{confirmed['full_direction_score_median_type7']:.6f}` and
`{confirmed['full_direction_score_q90_type7']:.6f}` against `1.5` and `2.5`. Every
truth-cosine residual was within its pre-outcome analytic envelope. Stored chunk sums
reproduce checkpoints with maximum vector discrepancy
`{confirmed['maximum_checkpoint_reproduction_error']:.3g}`.

## Self-consistency boundary

All {boundary['dissociation_target_count']} predeclared dissociation targets reached
final split-half Fisher cosine at least
`{boundary['minimum_dissociation_half_cosine']:.9f}`, while observed truth alignment
ranged down to `{boundary['minimum_dissociation_truth_cosine']:.9f}` and no higher than
`{boundary['maximum_dissociation_truth_cosine']:.9f}` within that stratum. This confirms
that split-half convergence estimates variance around an estimator's own expectation;
it cannot by itself certify alignment with an independently defined truth.

## Falsified reductions

The exact matched-support-factorization contrasts falsify an `N p_min`-only angular-bias
law. The additive reward-shift contrasts falsify laws using support probabilities and
the analytic Fisher spectrum without reward/baseline geometry. The full finite
multinomial law remains predictive; the useful compact bound remains open.

## Classification and next decision

ACL-006 is theorem/software reproduction plus mechanism validation inside the existing
contextual-bandit empirical-Fisher estimator family. It does not add a structurally
independent adaptive class and cannot strengthen the breadth claim by itself. The next
experiment must carry a frozen quantity into a non-Fisher class without target
refitting, or record that no nontrivial quantity can be specified.
"""


def write_report_package(payload: dict[str, Any], output_dir: str | Path) -> Path:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    summary, targets, checkpoints, contrasts = analyze_stored_evidence(payload)
    write_json(destination / "summary.json", summary)
    _write_csv(destination / "targets.csv", targets)
    _write_csv(destination / "checkpoints.csv", checkpoints)
    _write_csv(destination / "contrasts.csv", contrasts)
    (destination / "ACL-006_REPORT.md").write_text(
        _markdown(summary), encoding="utf-8", newline="\n"
    )
    return destination
