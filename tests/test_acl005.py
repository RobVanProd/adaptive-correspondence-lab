import shutil
from pathlib import Path

import numpy as np
import pytest

import adaptive_correspondence.acl005 as acl005_module
from adaptive_correspondence.acl005 import (
    analyze_landscape_results,
    build_analytic_registry,
    estimate_landscape,
    execute_confirmatory,
    reproduce_stopped_mean,
    validate_manifest_dict,
    validate_preregistration_bundle,
    validate_source_evidence,
)


def _toy_manifest() -> dict:
    return {
        "schema_version": 1,
        "experiment_id": "TOY-ACL005",
        "randomness": "PCG64-independent-landscape-streams",
        "contexts": 2,
        "actions": 3,
        "interaction_sample_count": 32,
        "empirical_fisher_rcond": 1e-12,
        "replication_schedule": [32, 64],
        "shadow_chunk_size": 8,
        "half_convergence_fisher_cosine_min": 0.8,
        "h2_fisher_cosine_min": 0.9,
        "h1_shadow_count": 8,
        "regular_min_expected_cell_count": 1.0,
        "stress_max_expected_cell_count": 0.2,
        "source_experiment": "TOY-SOURCE",
        "source_rule": "unchanged-block-fisher-cosine-law",
        "benchmark_scope": "toy-control",
        "inference_scope": "none",
        "transport_scope": "toy-cross-class",
        "landscapes": [
            {
                "id": "R01",
                "role": "confirmatory-target",
                "rewards": [[0.8, 0.1, -0.3], [-0.2, 0.5, 1.0]],
                "context_probabilities": [0.6, 0.4],
                "logits": [[0.2, -0.1, -0.1], [-0.3, 0.1, 0.2]],
                "seed": 501,
            },
            {
                "id": "S01",
                "role": "stress-target",
                "rewards": [[0.7, 0.0, -0.4], [-0.5, 0.2, 0.9]],
                "context_probabilities": [0.98, 0.02],
                "logits": [[2.0, 0.0, -2.0], [2.0, 0.0, -2.0]],
                "seed": 502,
            },
        ],
    }


def test_analytic_registry_freezes_preoutcome_strata_without_shadows() -> None:
    manifest = validate_manifest_dict(_toy_manifest())
    registry = build_analytic_registry(manifest)
    assert registry["outcomes_generated"] is False
    assert registry["shadow_count"] == 0
    assert [row["stratum"] for row in registry["landscapes"]] == ["regular", "stress"]


def test_estimator_stops_at_first_all_context_checkpoint(monkeypatch) -> None:
    manifest = validate_manifest_dict(_toy_manifest())
    landscape = manifest.landscapes[0]
    analytic = build_analytic_registry(manifest)["landscapes"][0]["analytic_direction"]

    def aligned(*args, replications, **kwargs):
        direction = np.asarray(analytic).reshape(2, 3)
        return np.tile(direction, (replications, 1, 1))

    monkeypatch.setattr(acl005_module, "sample_plugin_npg_shadows", aligned)
    result = estimate_landscape(manifest, landscape)
    assert result["converged"] is True
    assert result["stopped_replications"] == 32
    np.testing.assert_allclose(reproduce_stopped_mean(result), np.asarray(analytic), atol=0.0)


def test_regular_context_failure_cannot_be_rescued_by_joint() -> None:
    manifest = validate_manifest_dict(_toy_manifest())
    results = [
        {
            "landscape_id": "R01",
            "role": "confirmatory-target",
            "converged": True,
            "stopped_replications": 32,
            "final_context_cosines": [0.89, 1.0],
            "joint_cosine": 0.99999,
            "h1_context_cosines": [[0.1], [0.2]],
        },
        {
            "landscape_id": "S01",
            "role": "stress-target",
            "converged": True,
            "stopped_replications": 32,
            "final_context_cosines": [-1.0, -1.0],
            "joint_cosine": -1.0,
            "h1_context_cosines": [[-0.1], [-0.2]],
        },
    ]
    analysis = analyze_landscape_results(manifest, results)
    assert analysis["transport_verdict"] == "FAIL"
    assert analysis["stress_gating"] is False
    assert analysis["joint_cosine_gating"] is False


def test_regular_nonconvergence_is_inconclusive_but_stress_is_not() -> None:
    manifest = validate_manifest_dict(_toy_manifest())
    results = [
        {
            "landscape_id": "R01",
            "role": "confirmatory-target",
            "converged": False,
            "stopped_replications": 64,
            "final_context_cosines": [1.0, 1.0],
            "joint_cosine": 1.0,
            "h1_context_cosines": [[0.1], [0.2]],
        },
        {
            "landscape_id": "S01",
            "role": "stress-target",
            "converged": False,
            "stopped_replications": 64,
            "final_context_cosines": [1.0, 1.0],
            "joint_cosine": 1.0,
            "h1_context_cosines": [[0.1], [0.2]],
        },
    ]
    assert analyze_landscape_results(manifest, results)["transport_verdict"] == "INCONCLUSIVE"
    results[0]["converged"] = True
    assert analyze_landscape_results(manifest, results)["transport_verdict"] == "PASS"


def test_undefined_context_from_nonconvergence_is_preserved_as_inconclusive() -> None:
    manifest = validate_manifest_dict(_toy_manifest())
    results = [
        {
            "landscape_id": "R01",
            "role": "confirmatory-target",
            "converged": False,
            "stopped_replications": 64,
            "final_context_cosines": [1.0, None],
            "joint_cosine": 1.0,
            "h1_context_cosines": [[0.1], [None]],
        },
        {
            "landscape_id": "S01",
            "role": "stress-target",
            "converged": False,
            "stopped_replications": 64,
            "final_context_cosines": [None, None],
            "joint_cosine": None,
            "h1_context_cosines": [[None], [None]],
        },
    ]
    analysis = analyze_landscape_results(manifest, results)
    assert analysis["transport_verdict"] == "INCONCLUSIVE"
    assert analysis["regular_minimum_context_fisher_cosine"] is None


def test_runner_requires_sha_derived_canonical_output(tmp_path: Path, monkeypatch) -> None:
    approved_sha = "c" * 40
    monkeypatch.setattr(
        acl005_module, "git_execution_state", lambda repo_path: (approved_sha, False)
    )
    with pytest.raises(ValueError, match="canonical evidence path"):
        execute_confirmatory(
            repo_path=tmp_path,
            bundle_path=tmp_path / "missing",
            approved_sha=approved_sha,
            output_path=tmp_path / "elsewhere.json",
        )


def test_runner_requires_sha_bound_canonical_bundle(tmp_path: Path, monkeypatch) -> None:
    approved_sha = "d" * 40
    monkeypatch.setattr(
        acl005_module, "git_execution_state", lambda repo_path: (approved_sha, False)
    )
    canonical_output = tmp_path / "evidence" / f"ACL-005-confirmatory-{approved_sha}.json"
    with pytest.raises(ValueError, match="canonical preregistration bundle"):
        execute_confirmatory(
            repo_path=tmp_path,
            bundle_path=tmp_path / "external-self-consistent-bundle",
            approved_sha=approved_sha,
            output_path=canonical_output,
        )


def test_acl005_identifier_activates_exact_transport_design() -> None:
    payload = _toy_manifest()
    payload["experiment_id"] = "ACL-005"
    with pytest.raises(ValueError, match="ACL-005 design"):
        validate_manifest_dict(payload)


def test_real_bundle_is_analytic_only_and_source_anchored() -> None:
    validation = validate_preregistration_bundle("preregistrations/ACL-005")
    assert validation["outcomes_generated"] is False
    assert validation["regular_landscape_count"] == 10
    assert validation["stress_landscape_count"] == 4
    manifest = acl005_module.load_manifest("preregistrations/ACL-005/manifest.json")
    assert manifest.raw["source_evidence"]["evidence_sha256"] == (
        "3f97f7c4debbd65014e6ee337d9a8990500bad38ce0dfef2e8ad3048c74cd91a"
    )
    source_validation = validate_source_evidence(Path.cwd(), manifest)
    assert source_validation["valid"] is True
    assert source_validation["evidence_sha256"] == (
        "3f97f7c4debbd65014e6ee337d9a8990500bad38ce0dfef2e8ad3048c74cd91a"
    )


def test_real_bundle_rejects_extra_neighbor_file(tmp_path: Path) -> None:
    bundle = tmp_path / "ACL-005"
    shutil.copytree("preregistrations/ACL-005", bundle)
    (bundle / "unlocked-extra.txt").write_text("must fail\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exact frozen directory contents"):
        validate_preregistration_bundle(bundle)


def test_source_evidence_missing_aborts_as_validation_error(tmp_path: Path) -> None:
    manifest = acl005_module.load_manifest("preregistrations/ACL-005/manifest.json")
    with pytest.raises(ValueError, match="cannot read frozen ACL-004 source evidence"):
        validate_source_evidence(tmp_path, manifest)
