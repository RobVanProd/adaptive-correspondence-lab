import shutil
from pathlib import Path

import numpy as np
import pytest

import adaptive_correspondence.acl004 as acl004_module
from adaptive_correspondence.acl004 import (
    analyze_landscape_results,
    build_analytic_registry,
    estimate_landscape,
    execute_confirmatory,
    reproduce_stopped_mean,
    validate_manifest_dict,
)


def _toy_manifest() -> dict:
    return {
        "schema_version": 1,
        "experiment_id": "TOY-ACL004",
        "randomness": "PCG64-independent-landscape-streams",
        "parameterization": "mean-and-log-standard-deviation",
        "dimension": 2,
        "sample_count": 8,
        "parent_count": 4,
        "mean_learning_rate": 0.2,
        "covariance_learning_rate": 0.1,
        "rank_weights": "log(parent_count+0.5)-log(rank), normalized",
        "quadrature_order": 64,
        "quadrature_oracle_order": 96,
        "quadrature_relative_tolerance": 1e-6,
        "quadrature_absolute_tolerance": 1e-9,
        "replication_schedule": [32, 64],
        "shadow_chunk_size": 8,
        "half_convergence_fisher_cosine_min": 0.8,
        "h2_fisher_cosine_min": 0.9,
        "h1_shadow_count": 8,
        "benchmark_scope": "toy-deterministic",
        "inference_scope": "none",
        "transport_scope": "within-gaussian-class",
        "landscapes": [
            {
                "id": "G01",
                "mean": [0.2, -0.4],
                "log_std": np.log([0.7, 1.3]).tolist(),
                "objective": [1.0, -0.6],
                "seed": 101,
            }
        ],
    }


def test_analytic_registry_contains_no_shadows_or_outcomes() -> None:
    manifest = validate_manifest_dict(_toy_manifest())
    registry = build_analytic_registry(manifest)
    assert registry["outcomes_generated"] is False
    assert registry["shadow_count"] == 0
    assert registry["landscapes"][0]["mean_block_fisher_norm"] > 0.0
    assert registry["landscapes"][0]["covariance_block_fisher_norm"] > 0.0


def test_estimator_stops_at_first_qualifying_checkpoint(monkeypatch) -> None:
    manifest = validate_manifest_dict(_toy_manifest())
    landscape = manifest.landscapes[0]
    analytic = build_analytic_registry(manifest)["landscapes"][0]["analytic_direction"]

    def aligned_shadows(*args, replications, **kwargs):
        return np.tile(np.asarray(analytic), (replications, 1))

    monkeypatch.setattr(acl004_module, "sample_rank_mu_shadows", aligned_shadows)
    result = estimate_landscape(manifest, landscape)

    assert result["converged"] is True
    assert result["stopped_replications"] == 32
    assert len(result["checkpoint_history"]) == 1
    np.testing.assert_allclose(reproduce_stopped_mean(result), analytic, atol=0.0)
    assert result["rng_state_after"]["bit_generator"] == "PCG64"
    assert len(result["shadow_uncertainty"]["coordinate_standard_error_of_mean"]) == 4


def test_separate_block_failure_cannot_be_rescued_by_joint_cosine() -> None:
    manifest = validate_manifest_dict(_toy_manifest())
    results = [
        {
            "landscape_id": "G01",
            "converged": True,
            "stopped_replications": 32,
            "final_cosines": {"mean": 0.89, "covariance": 1.0, "joint": 0.99999},
            "h1": {"mean_cosines": [0.1], "covariance_cosines": [0.2]},
        }
    ]
    analysis = analyze_landscape_results(manifest, results)
    assert analysis["h2_verdict"] == "FAIL"
    assert analysis["joint_cosine_gating"] is False


def test_nonconvergence_is_inconclusive_not_failure() -> None:
    manifest = validate_manifest_dict(_toy_manifest())
    results = [
        {
            "landscape_id": "G01",
            "converged": False,
            "stopped_replications": 64,
            "final_cosines": {"mean": 1.0, "covariance": 1.0, "joint": 1.0},
            "h1": {"mean_cosines": [0.1], "covariance_cosines": [0.2]},
        }
    ]
    assert analyze_landscape_results(manifest, results)["h2_verdict"] == "INCONCLUSIVE"


def test_runner_requires_sha_derived_canonical_output(tmp_path: Path, monkeypatch) -> None:
    approved_sha = "b" * 40
    monkeypatch.setattr(
        acl004_module, "git_execution_state", lambda repo_path: (approved_sha, False)
    )
    with pytest.raises(ValueError, match="canonical evidence path"):
        execute_confirmatory(
            repo_path=tmp_path,
            bundle_path=tmp_path / "missing",
            approved_sha=approved_sha,
            output_path=tmp_path / "elsewhere.json",
        )


def test_acl004_identifier_activates_exact_design() -> None:
    payload = _toy_manifest()
    payload["experiment_id"] = "ACL-004"
    with pytest.raises(ValueError, match="ACL-004 design"):
        validate_manifest_dict(payload)


def test_real_bundle_rejects_extra_neighbor_file(tmp_path: Path) -> None:
    bundle = tmp_path / "ACL-004"
    shutil.copytree("preregistrations/ACL-004", bundle)
    (bundle / "unlocked-extra.txt").write_text("must fail\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exact frozen directory contents"):
        acl004_module.validate_preregistration_bundle(bundle)
