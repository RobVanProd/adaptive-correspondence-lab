import copy
import shutil
from pathlib import Path

import numpy as np
import pytest

import adaptive_correspondence.acl006 as acl006_module
from adaptive_correspondence.acl006 import (
    analyze_target_results,
    batch_plugin_directions_from_counts,
    build_analytic_registry,
    direct_plugin_direction,
    estimate_target,
    execute_confirmatory,
    reproduce_target_mean,
    validate_execution_environment,
    validate_manifest_dict,
    validate_preregistration_bundle,
    validate_target_result,
)


def _toy_manifest() -> dict:
    return {
        "schema_version": 1,
        "experiment_id": "TOY-ACL006",
        "randomness": "PCG64-independent-target-streams",
        "actions": 3,
        "empirical_fisher_rcond": 1e-12,
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
        "benchmark_scope": "toy-fixed-benchmark",
        "inference_scope": "descriptive-not-population-confidence",
        "mechanism_scope": "toy-undamped-plugin-estimator",
        "confirmatory_environment": {
            "python_implementation": "CPython",
            "python_version": "3.13.14",
            "numpy_version": "2.5.2",
            "platform_system": "Windows",
            "platform_machine": "AMD64",
        },
        "targets": [
            {
                "id": "T01",
                "family": "toy-a",
                "sample_count": 4,
                "context_probability": 0.4,
                "policy": [0.6, 0.3, 0.1],
                "reward": [1.0, 0.0, -1.0],
                "seed": 601,
            },
            {
                "id": "T02",
                "family": "toy-b",
                "sample_count": 4,
                "context_probability": 0.4,
                "policy": [0.6, 0.3, 0.1],
                "reward": [4.0, 3.0, 2.0],
                "seed": 602,
            },
        ],
        "contrasts": [
            {
                "id": "reward-shift",
                "kind": "reward-shift",
                "left": "T01",
                "right": "T02",
            }
        ],
    }


def test_direct_full_coordinate_direction_matches_manual_pseudoinverse() -> None:
    policy = np.asarray([0.57, 0.31, 0.12], dtype=np.float64)
    reward = np.asarray([1.2, -0.4, 0.25], dtype=np.float64)
    counts = np.asarray([3, 1, 2, 4], dtype=np.int64)
    scores = np.eye(3) - policy[None, :]
    fisher = np.einsum("a,ai,aj->ij", counts[:3], scores, scores) / 10
    gradient = (counts[:3] * reward) @ scores / 10
    expected = np.linalg.pinv(fisher, rcond=1e-12, hermitian=True) @ gradient
    expected -= np.mean(expected)
    actual = direct_plugin_direction(
        policy=policy,
        reward=reward,
        counts=counts,
        sample_count=10,
        rcond=1e-12,
    )
    np.testing.assert_allclose(actual, expected, atol=2e-15, rtol=0.0)


def test_vectorized_sample_path_matches_scalar_full_coordinate_oracle() -> None:
    policy = np.asarray([0.57, 0.31, 0.12], dtype=np.float64)
    reward = np.asarray([1.2, -0.4, 0.25], dtype=np.float64)
    counts = np.asarray(
        [[3, 1, 2, 4], [0, 4, 0, 6], [1, 0, 7, 2]], dtype=np.int64
    )
    actual, masks = batch_plugin_directions_from_counts(
        policy=policy,
        reward=reward,
        counts=counts,
        sample_count=10,
        rcond=1e-12,
    )
    expected = np.asarray(
        [
            direct_plugin_direction(
                policy=policy,
                reward=reward,
                counts=row,
                sample_count=10,
                rcond=1e-12,
            )
            for row in counts
        ]
    )
    np.testing.assert_allclose(actual, expected, atol=3e-15, rtol=0.0)
    np.testing.assert_array_equal(masks, [7, 2, 5])


def test_analytic_registry_contains_no_rng_outcomes_and_freezes_exact_moments() -> None:
    manifest = validate_manifest_dict(_toy_manifest())
    registry = build_analytic_registry(manifest)
    assert registry["outcomes_generated"] is False
    assert registry["shadow_count"] == 0
    assert len(registry["targets"]) == 2
    assert registry["targets"][0]["exact_probability_mass"] == pytest.approx(1.0)
    assert registry["targets"][0]["final_fisher_rms_standard_error"] > 0.0


def test_primary_score_and_dissociation_are_separate_verdicts() -> None:
    manifest = validate_manifest_dict(_toy_manifest())
    registry = build_analytic_registry(manifest)
    rows = []
    for entry in registry["targets"]:
        rows.append(
            {
                "target_id": entry["id"],
                "full_direction_score": 1.0,
                "first_half_direction_score": 1.0,
                "second_half_direction_score": 1.0,
                "angular_residual": 0.0,
                "angular_envelope": entry["final_angular_envelope"],
                "observed_truth_cosine": entry["exact_truth_alignment_cosine"],
                "final_half_cosine": 1.0,
            }
        )
    analysis = analyze_target_results(manifest, registry, rows)
    assert analysis["exact_mean_prediction_verdict"] == "PASS"
    assert analysis["target_refit"] is False
    assert analysis["self_consistency_certifies_truth"] is False


def test_score_failure_cannot_be_rescued_by_split_consistency() -> None:
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
    assert analysis["exact_mean_prediction_verdict"] == "FAIL"


def test_sampled_path_is_independent_of_exact_enumerator(monkeypatch) -> None:
    manifest = validate_manifest_dict(_toy_manifest())
    registry = build_analytic_registry(manifest)
    target = manifest.targets[0]
    entry = registry["targets"][0]

    def forbidden(*args, **kwargs):
        raise AssertionError("sampled path called exact enumerator")

    monkeypatch.setattr(acl006_module, "exact_block_moments", forbidden)
    result = estimate_target(manifest, target, entry)
    validate_target_result(manifest, target, entry, result)
    assert result["generated_replications"] == 16
    np.testing.assert_array_equal(
        reproduce_target_mean(result), result["observed_mean_direction"]
    )


def test_registry_comparison_accepts_machine_precision_not_material_drift() -> None:
    manifest = validate_manifest_dict(_toy_manifest())
    expected = build_analytic_registry(manifest)
    within = copy.deepcopy(expected)
    within["targets"][0]["exact_truth_alignment_cosine"] += 1e-13
    acl006_module._assert_numeric_equivalence(
        within,
        expected,
        atol=manifest.analytic_registry_atol,
        rtol=manifest.analytic_registry_rtol,
    )
    outside = copy.deepcopy(expected)
    outside["targets"][0]["exact_truth_alignment_cosine"] += 1e-7
    with pytest.raises(ValueError, match="differs numerically"):
        acl006_module._assert_numeric_equivalence(
            outside,
            expected,
            atol=manifest.analytic_registry_atol,
            rtol=manifest.analytic_registry_rtol,
        )


def test_manifest_rejects_false_reward_shift_label() -> None:
    payload = _toy_manifest()
    payload["targets"][1]["reward"] = [4.0, 3.1, 2.0]
    with pytest.raises(ValueError, match="reward-shift contrast semantics"):
        validate_manifest_dict(payload)


def test_runner_requires_sha_derived_canonical_paths(tmp_path: Path, monkeypatch) -> None:
    approved_sha = "6" * 40
    monkeypatch.setattr(
        acl006_module, "git_execution_state", lambda repo_path: (approved_sha, False)
    )
    with pytest.raises(ValueError, match="canonical evidence path"):
        execute_confirmatory(
            repo_path=tmp_path,
            bundle_path=tmp_path / "missing",
            approved_sha=approved_sha,
            output_path=tmp_path / "elsewhere.json",
        )


def test_runner_requires_sha_bound_canonical_bundle(tmp_path: Path, monkeypatch) -> None:
    approved_sha = "7" * 40
    monkeypatch.setattr(
        acl006_module, "git_execution_state", lambda repo_path: (approved_sha, False)
    )
    canonical = tmp_path / "evidence" / f"ACL-006-confirmatory-{approved_sha}.json"
    with pytest.raises(ValueError, match="canonical preregistration bundle"):
        execute_confirmatory(
            repo_path=tmp_path,
            bundle_path=tmp_path / "external-self-consistent-bundle",
            approved_sha=approved_sha,
            output_path=canonical,
        )


def test_runner_rejects_dirty_tree_before_bundle_or_rng(tmp_path: Path, monkeypatch) -> None:
    approved_sha = "8" * 40
    monkeypatch.setattr(
        acl006_module, "git_execution_state", lambda repo_path: (approved_sha, True)
    )
    canonical_bundle = tmp_path / "preregistrations" / "ACL-006"
    canonical = tmp_path / "evidence" / f"ACL-006-confirmatory-{approved_sha}.json"
    with pytest.raises(ValueError, match="worktree must be completely clean"):
        execute_confirmatory(
            repo_path=tmp_path,
            bundle_path=canonical_bundle,
            approved_sha=approved_sha,
            output_path=canonical,
        )


def test_acl006_identifier_activates_frozen_design() -> None:
    payload = _toy_manifest()
    payload["experiment_id"] = "ACL-006"
    with pytest.raises(ValueError, match="ACL-006 design"):
        validate_manifest_dict(payload)


def test_real_bundle_is_analytic_only_and_tolerantly_reproducible() -> None:
    validation = validate_preregistration_bundle("preregistrations/ACL-006")
    assert validation["outcomes_generated"] is False
    assert validation["target_count"] == 16
    assert validation["dissociation_target_count"] > 0
    assert validation["registry_comparison"] == "numeric-tolerance"
    manifest = acl006_module.load_manifest("preregistrations/ACL-006/manifest.json")
    assert validate_execution_environment(manifest)["valid"] is True


def test_real_bundle_rejects_extra_neighbor_file(tmp_path: Path) -> None:
    bundle = tmp_path / "ACL-006"
    shutil.copytree("preregistrations/ACL-006", bundle)
    (bundle / "unlocked-extra.txt").write_text("must fail\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exact frozen directory contents"):
        validate_preregistration_bundle(bundle)


def test_reproduce_target_mean_uses_only_frozen_chunk_sums() -> None:
    result = {
        "shadow_chunks": [
            {"count": 2, "direction_sum": [2.0, 4.0, -6.0]},
            {"count": 2, "direction_sum": [6.0, 0.0, -6.0]},
        ]
    }
    np.testing.assert_array_equal(reproduce_target_mean(result), [2.0, 1.0, -3.0])
