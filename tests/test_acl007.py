import copy
import shutil
from pathlib import Path

import numpy as np
import pytest

import adaptive_correspondence.acl007 as acl007_module
from adaptive_correspondence.acl007 import (
    analyze_target_results,
    build_analytic_registry,
    estimate_target,
    execute_confirmatory,
    reproduce_target_mean,
    validate_manifest_dict,
    validate_preregistration_bundle,
    validate_source_evidence,
    validate_target_result,
)


def _toy_manifest() -> dict:
    model = {
        "initial_belief": [0.5, 0.3, 0.2],
        "true_transition": [
            [0.75, 0.20, 0.05],
            [0.10, 0.75, 0.15],
            [0.05, 0.25, 0.70],
        ],
        "true_likelihoods": [[0.8, 0.3, 0.1], [0.2, 0.7, 0.9]],
    }
    return {
        "schema_version": 1,
        "experiment_id": "TOY-ACL007",
        "randomness": "PCG64-independent-target-streams",
        "states": 3,
        "replication_schedule": [8, 16],
        "shadow_chunk_size": 4,
        "direction_score_max": 5.0,
        "full_score_median_max": 1.5,
        "full_score_q90_max": 2.5,
        "dissociation_exact_truth_cosine_max": 0.9,
        "dissociation_observed_truth_cosine_max": 0.95,
        "dissociation_half_cosine_min": 0.995,
        "resolvable_contrast_gap_min": 0.1,
        "analytic_registry_atol": 2e-12,
        "analytic_registry_rtol": 2e-12,
        "source_experiment": "TOY-SOURCE",
        "source_rule": "unchanged-native-metric-standardized-mean-and-dissociation-law",
        "benchmark_scope": "toy-sequential-inference",
        "inference_scope": "descriptive-not-population-confidence",
        "transport_scope": "toy-cross-class",
        "confirmatory_environment": {
            "python_implementation": "CPython",
            "python_version": "3.13.14",
            "numpy_version": "2.5.2",
            "platform_system": "Windows",
            "platform_machine": "AMD64",
        },
        "models": {"M": model},
        "targets": [
            {
                "id": "T01",
                "family": "correct",
                "model": "M",
                "particle_count": 3,
                "filter_transition": model["true_transition"],
                "filter_likelihoods": model["true_likelihoods"],
                "seed": 701,
            },
            {
                "id": "T02",
                "family": "reversed",
                "model": "M",
                "particle_count": 3,
                "filter_transition": model["true_transition"],
                "filter_likelihoods": [[0.1, 0.3, 0.8], [0.9, 0.7, 0.2]],
                "seed": 702,
            },
        ],
        "contrasts": [
            {
                "id": "correct-vs-reversed",
                "kind": "observation-misspecification",
                "left": "T01",
                "right": "T02",
            }
        ],
    }


def test_registry_is_exact_outcome_free_and_uses_euclidean_geometry() -> None:
    manifest = validate_manifest_dict(_toy_manifest())
    registry = build_analytic_registry(manifest)
    assert registry["outcomes_generated"] is False
    assert registry["shadow_count"] == 0
    assert registry["native_metric"] == "centered-euclidean-belief-tangent"
    assert len(registry["targets"]) == 2
    assert registry["targets"][0]["exact_probability_mass"] == pytest.approx(1.0)


def test_sampled_path_does_not_call_exact_count_oracle(monkeypatch) -> None:
    manifest = validate_manifest_dict(_toy_manifest())
    registry = build_analytic_registry(manifest)
    target = manifest.targets[0]

    def forbidden(*args, **kwargs):
        raise AssertionError("sampled target path called exact count oracle")

    monkeypatch.setattr(acl007_module, "exact_particle_filter_moments", forbidden)
    result = estimate_target(manifest, target, registry["targets"][0])
    validate_target_result(manifest, target, registry["targets"][0], result)
    assert result["generated_replications"] == 16
    np.testing.assert_array_equal(
        reproduce_target_mean(result), result["observed_mean_direction"]
    )


def test_target_validation_rejects_corrupted_chunk_direction_sum() -> None:
    manifest = validate_manifest_dict(_toy_manifest())
    registry = build_analytic_registry(manifest)
    target = manifest.targets[0]
    result = estimate_target(manifest, target, registry["targets"][0])
    corrupted = copy.deepcopy(result)
    corrupted["shadow_chunks"][0]["direction_sum"] = [float("nan"), 0.0, 0.0]
    with pytest.raises(ValueError, match="chunk sufficient statistics"):
        validate_target_result(manifest, target, registry["targets"][0], corrupted)


def test_split_consistency_cannot_rescue_standardized_mean_failure() -> None:
    manifest = validate_manifest_dict(_toy_manifest())
    registry = build_analytic_registry(manifest)
    rows = []
    for index, entry in enumerate(registry["targets"]):
        rows.append(
            {
                "target_id": entry["id"],
                "full_direction_score": 6.0 if index == 0 else 0.1,
                "first_half_direction_score": 0.1,
                "second_half_direction_score": 0.1,
                "angular_residual": 0.0,
                "angular_envelope": entry["final_angular_envelope"],
                "observed_truth_cosine": entry["exact_truth_alignment_cosine"],
                "final_half_cosine": 1.0,
            }
        )
    analysis = analyze_target_results(manifest, registry, rows)
    assert analysis["transport_verdict"] == "FAIL"
    assert analysis["target_refit"] is False


def test_runner_requires_sha_derived_canonical_paths(tmp_path: Path, monkeypatch) -> None:
    approved_sha = "7" * 40
    monkeypatch.setattr(
        acl007_module, "git_execution_state", lambda repo_path: (approved_sha, False)
    )
    with pytest.raises(ValueError, match="canonical evidence path"):
        execute_confirmatory(
            repo_path=tmp_path,
            bundle_path=tmp_path / "missing",
            approved_sha=approved_sha,
            output_path=tmp_path / "elsewhere.json",
        )


def test_runner_requires_sha_bound_canonical_bundle(tmp_path: Path, monkeypatch) -> None:
    approved_sha = "8" * 40
    monkeypatch.setattr(
        acl007_module, "git_execution_state", lambda repo_path: (approved_sha, False)
    )
    canonical = tmp_path / "evidence" / f"ACL-007-confirmatory-{approved_sha}.json"
    with pytest.raises(ValueError, match="canonical preregistration bundle"):
        execute_confirmatory(
            repo_path=tmp_path,
            bundle_path=tmp_path / "external-bundle",
            approved_sha=approved_sha,
            output_path=canonical,
        )


def test_runner_rejects_dirty_tree_before_bundle_or_rng(tmp_path: Path, monkeypatch) -> None:
    approved_sha = "9" * 40
    monkeypatch.setattr(
        acl007_module, "git_execution_state", lambda repo_path: (approved_sha, True)
    )
    canonical = tmp_path / "evidence" / f"ACL-007-confirmatory-{approved_sha}.json"
    with pytest.raises(ValueError, match="worktree must be completely clean"):
        execute_confirmatory(
            repo_path=tmp_path,
            bundle_path=tmp_path / "preregistrations" / "ACL-007",
            approved_sha=approved_sha,
            output_path=canonical,
        )


def test_acl007_identifier_activates_exact_transported_design() -> None:
    payload = _toy_manifest()
    payload["experiment_id"] = "ACL-007"
    with pytest.raises(ValueError, match="ACL-007 design"):
        validate_manifest_dict(payload)


def test_real_bundle_is_source_anchored_and_outcome_free() -> None:
    validation = validate_preregistration_bundle("preregistrations/ACL-007")
    assert validation["outcomes_generated"] is False
    assert validation["target_count"] == 16
    assert validation["dissociation_target_count"] > 0
    manifest = acl007_module.load_manifest("preregistrations/ACL-007/manifest.json")
    source = validate_source_evidence(Path.cwd(), manifest)
    assert source["valid"] is True
    assert source["evidence_sha256"] == (
        "740c541bbd69db77f6d02327ded34765a37345f907b048f8d3f3a91aebc23918"
    )


def test_real_bundle_rejects_extra_neighbor_file(tmp_path: Path) -> None:
    bundle = tmp_path / "ACL-007"
    shutil.copytree("preregistrations/ACL-007", bundle)
    (bundle / "unlocked-extra.txt").write_text("must fail\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exact frozen directory contents"):
        validate_preregistration_bundle(bundle)
