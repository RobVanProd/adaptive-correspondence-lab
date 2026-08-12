"""Artifact-only ACL-004 verification and Gaussian bridge summaries."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .acl002 import type7_quantile
from .acl004 import reproduce_stopped_mean
from .gaussian_rank_mu_bridge import GaussianLinearBridgeState, fisher_block_cosines
from .io import write_json

ACL004_APPROVED_SHA = "3ba4be7ce1460a40c4ef0879018df58947c36edb"
ACL004_EVIDENCE_COMMIT = "355dd97472da4230eff877b9a3c8c7c4626057cd"
ACL004_EVIDENCE_SHA256 = (
    "3f97f7c4debbd65014e6ee337d9a8990500bad38ce0dfef2e8ad3048c74cd91a"
)
ACL004_EVIDENCE_PATH = Path(
    "evidence/ACL-004-confirmatory-3ba4be7ce1460a40c4ef0879018df58947c36edb.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_acl004_evidence(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if _sha256(source) != ACL004_EVIDENCE_SHA256:
        raise ValueError("ACL-004 evidence SHA-256 mismatch")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("cannot read ACL-004 evidence") from error
    analysis = payload.get("analysis", {})
    results = payload.get("landscape_results", [])
    registry = payload.get("locked_analytic_registry", {})
    lock = payload.get("preregistration_bundle_lock", {})
    checks = (
        payload.get("approved_preregistration_sha") == ACL004_APPROVED_SHA,
        payload.get("kind") == "confirmatory-gaussian-finite-lambda-conditional-mean",
        payload.get("target_refit") is False,
        payload.get("lambda_scaling_studied") is False,
        analysis.get("h2_verdict") == "PASS",
        analysis.get("all_landscapes_converged") is True,
        len(results) == 12,
        all(row.get("stopped_replications") == 4096 for row in results),
        registry.get("outcomes_generated") is False,
        registry.get("shadow_count") == 0,
        lock.get("outcomes_generated") is False,
    )
    if not all(checks):
        raise ValueError("ACL-004 frozen evidence envelope mismatch")
    return payload


def _joint_cosine(
    state: GaussianLinearBridgeState, left: np.ndarray, right: np.ndarray
) -> float:
    mean, log_std, _ = state.arrays()
    metric = np.concatenate((np.exp(-2.0 * log_std), np.full(mean.size, 2.0)))
    return float(
        np.sum(metric * left * right)
        / np.sqrt(np.sum(metric * left**2) * np.sum(metric * right**2))
    )


def _chunk_mean(chunks: list[dict[str, Any]]) -> np.ndarray:
    return reproduce_stopped_mean({"shadow_chunks": chunks})


def analyze_stored_evidence(
    payload: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    design = payload["frozen_design"]
    states = {
        item["id"]: GaussianLinearBridgeState(
            item["mean"], item["log_std"], item["objective"]
        )
        for item in design["landscapes"]
    }
    h2_rows = []
    checkpoint_rows = []
    mean_errors = []
    cosine_errors = []
    for result in payload["landscape_results"]:
        identifier = result["landscape_id"]
        state = states[identifier]
        analytic = np.asarray(result["analytic_direction"], dtype=np.float64)
        stored_mean = np.asarray(result["stopped_mean_direction"], dtype=np.float64)
        reproduced = reproduce_stopped_mean(result)
        mean_error = float(np.max(np.abs(reproduced - stored_mean)))
        mean_errors.append(mean_error)
        block = fisher_block_cosines(state, reproduced, analytic)
        joint = _joint_cosine(state, reproduced, analytic)
        stored_cosines = result["final_cosines"]
        cosine_error = max(
            abs(block["mean"] - stored_cosines["mean"]),
            abs(block["covariance"] - stored_cosines["covariance"]),
            abs(joint - stored_cosines["joint"]),
        )
        cosine_errors.append(cosine_error)
        coordinate_se = result["shadow_uncertainty"]["coordinate_standard_error_of_mean"]
        h2_rows.append(
            {
                "landscape_id": identifier,
                "stopped_replications": result["stopped_replications"],
                "mean_fisher_cosine": block["mean"],
                "covariance_fisher_cosine": block["covariance"],
                "joint_fisher_cosine": joint,
                "maximum_coordinate_standard_error": max(coordinate_se),
                "stopped_mean_reproduction_error": mean_error,
                "final_cosine_reproduction_error": cosine_error,
            }
        )
        chunks = result["shadow_chunks"]
        for checkpoint in result["checkpoint_history"]:
            count = checkpoint["replications"]
            used = chunks[: count // design["shadow_chunk_size"]]
            half = len(used) // 2
            first = _chunk_mean(used[:half])
            second = _chunk_mean(used[half:])
            reproduced_half = fisher_block_cosines(state, first, second)
            stored_half = checkpoint["half_fisher_cosines"]
            checkpoint_rows.append(
                {
                    "landscape_id": identifier,
                    "replications": count,
                    "mean_half_fisher_cosine": reproduced_half["mean"],
                    "covariance_half_fisher_cosine": reproduced_half["covariance"],
                    "maximum_half_cosine_reproduction_error": max(
                        abs(reproduced_half[key] - stored_half[key])
                        for key in ("mean", "covariance")
                    ),
                }
            )

    h1_rows = []
    stored_h1 = {
        (row["landscape_id"], row["block"]): row
        for row in payload["analysis"]["h1_descriptive"]
    }
    h1_errors = []
    for result in payload["landscape_results"]:
        for block in ("mean", "covariance"):
            values = result["h1"][f"{block}_cosines"]
            reproduced = {
                "landscape_id": result["landscape_id"],
                "block": block,
                "count": len(values),
                "q10": type7_quantile(values, 0.1),
                "median": type7_quantile(values, 0.5),
                "q90": type7_quantile(values, 0.9),
                "fraction_positive": sum(value > 0.0 for value in values) / len(values),
            }
            stored = stored_h1[(result["landscape_id"], block)]
            error = max(
                abs(reproduced[key] - stored[key])
                for key in ("q10", "median", "q90", "fraction_positive")
            )
            h1_errors.append(error)
            reproduced["summary_reproduction_error"] = error
            h1_rows.append(reproduced)

    analysis = payload["analysis"]
    summary = {
        "schema_version": 1,
        "experiment_id": "ACL-004",
        "analysis_kind": "artifact-only-post-confirmatory",
        "new_shadows_generated": False,
        "target_refit": False,
        "confirmed": {
            "approved_preregistration_sha": ACL004_APPROVED_SHA,
            "evidence_commit": ACL004_EVIDENCE_COMMIT,
            "evidence_sha256": ACL004_EVIDENCE_SHA256,
            "h2_verdict": analysis["h2_verdict"],
            "all_landscapes_converged": analysis["all_landscapes_converged"],
            "h2_threshold": analysis["h2_threshold"],
            "mean_block_minimum_fisher_cosine": min(
                row["mean_fisher_cosine"] for row in h2_rows
            ),
            "covariance_block_minimum_fisher_cosine": min(
                row["covariance_fisher_cosine"] for row in h2_rows
            ),
            "maximum_stopped_mean_reproduction_error": max(mean_errors),
            "maximum_final_cosine_reproduction_error": max(cosine_errors),
            "maximum_h1_summary_reproduction_error": max(h1_errors),
            "joint_cosine_gating": False,
        },
        "descriptive_h1": {
            "mean_median_range": [
                min(row["median"] for row in h1_rows if row["block"] == "mean"),
                max(row["median"] for row in h1_rows if row["block"] == "mean"),
            ],
            "covariance_median_range": [
                min(row["median"] for row in h1_rows if row["block"] == "covariance"),
                max(row["median"] for row in h1_rows if row["block"] == "covariance"),
            ],
            "mean_fraction_positive_range": [
                min(row["fraction_positive"] for row in h1_rows if row["block"] == "mean"),
                max(row["fraction_positive"] for row in h1_rows if row["block"] == "mean"),
            ],
            "covariance_fraction_positive_range": [
                min(
                    row["fraction_positive"]
                    for row in h1_rows
                    if row["block"] == "covariance"
                ),
                max(
                    row["fraction_positive"]
                    for row in h1_rows
                    if row["block"] == "covariance"
                ),
            ],
        },
        "scope": {
            "lambda_scaling_studied": False,
            "cross_class_quantity_transported": False,
            "interpretation": "restricted finite-lambda Gaussian expected-direction bridge",
        },
    }
    return summary, h2_rows, h1_rows, checkpoint_rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _markdown(summary: dict[str, Any]) -> str:
    confirmed = summary["confirmed"]
    h1 = summary["descriptive_h1"]
    return f"""# ACL-004 confirmatory report

## Confirmed finding

ACL-004 **{confirmed['h2_verdict']}**. All 12 landscapes met the preregistered
disjoint-half convergence rule at 4096 shadows. The minimum separate-block Fisher
cosines against the independent finite-lambda comparator were
`{confirmed['mean_block_minimum_fisher_cosine']:.9f}` for the mean block and
`{confirmed['covariance_block_minimum_fisher_cosine']:.9f}` for the covariance block,
above the frozen `0.99` threshold. Joint cosine was non-gating, and no target refit or
lambda scaling occurred.

Stored chunk sums reproduce every stopped mean with maximum error
`{confirmed['maximum_stopped_mean_reproduction_error']:.3g}`; recomputed final cosines
agree within `{confirmed['maximum_final_cosine_reproduction_error']:.3g}`.

## Descriptive H1

Single-shadow median mean-block cosines range from `{h1['mean_median_range'][0]:.3f}`
to `{h1['mean_median_range'][1]:.3f}`. Covariance-block medians range from
`{h1['covariance_median_range'][0]:.3f}` to
`{h1['covariance_median_range'][1]:.3f}`. These finite-single-step summaries are
descriptive and did not affect H2.

## Scope

This is preregistered evidence for a restricted Gaussian finite-lambda
expected-direction correspondence. It is a second adaptive-system class, but no
categorical coefficient or degradation law was transported into it. Cross-class
no-refit transport remains unresolved.
"""


def write_report_package(payload: dict[str, Any], output_dir: str | Path) -> Path:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    summary, h2_rows, h1_rows, checkpoints = analyze_stored_evidence(payload)
    write_json(destination / "summary.json", summary)
    _write_csv(destination / "h2_landscapes.csv", h2_rows)
    _write_csv(destination / "h1_single_shadow.csv", h1_rows)
    _write_csv(destination / "convergence_checkpoints.csv", checkpoints)
    (destination / "ACL-004_REPORT.md").write_text(
        _markdown(summary), encoding="utf-8", newline="\n"
    )
    return destination
