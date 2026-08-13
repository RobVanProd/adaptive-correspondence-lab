import json
import shutil
from pathlib import Path

import pytest

import adaptive_correspondence.acl008 as acl008_module
from adaptive_correspondence.acl008 import (
    ACL008_REFERENCE_MANIFEST_SHA256,
    SOURCE_GATES,
    build_analytic_registry,
    execute_confirmatory,
    load_manifest,
    validate_preregistration_bundle,
    validate_source_evidence,
)


def test_real_registry_is_analytic_only_and_has_frozen_strata(monkeypatch) -> None:
    manifest = load_manifest("preregistrations/ACL-008/manifest.json")
    original = acl008_module.burg_perturbed_trajectory_polynomial_oracle

    def zero_only(*args, **kwargs):
        assert kwargs["epsilon"] == 0.0
        return original(*args, **kwargs)

    monkeypatch.setattr(
        acl008_module, "burg_perturbed_trajectory_polynomial_oracle", zero_only
    )
    registry = build_analytic_registry(manifest)
    assert registry["outcomes_generated"] is False
    assert registry["target_refit"] is False
    assert sum(
        entry["stratum"] == "regular-sensitivity"
        for entry in registry["landscapes"]
    ) == 16
    assert registry["landscapes"][-1]["stratum"] == "identity-control"


def test_source_rule_is_copied_exactly() -> None:
    manifest = load_manifest("preregistrations/ACL-008/manifest.json")
    payload = manifest.raw
    assert payload["eta"] == SOURCE_GATES["eta"]
    assert tuple(payload["epsilon_grid"]) == SOURCE_GATES["epsilon_grid"]
    assert tuple(payload["confirmatory_epsilons"]) == SOURCE_GATES[
        "confirmatory_epsilons"
    ]
    assert payload["gates"]["target_landscape_median_relative_error_max"] == 0.1
    assert payload["gates"]["target_landscape_q90_relative_error_max"] == 0.2
    assert payload["novelty_reference_manifest_sha256"] == (
        ACL008_REFERENCE_MANIFEST_SHA256
    )


def test_source_evidence_hashes_are_live() -> None:
    manifest = load_manifest("preregistrations/ACL-008/manifest.json")
    validation = validate_source_evidence(Path.cwd(), manifest)
    assert validation["valid"] is True
    assert validation["source_verdict"] == "PASS"


def test_runner_requires_canonical_sha_path_before_bundle(tmp_path: Path, monkeypatch) -> None:
    approved_sha = "a" * 40
    monkeypatch.setattr(
        acl008_module, "git_execution_state", lambda repo_path: (approved_sha, False)
    )
    with pytest.raises(ValueError, match="canonical SHA-derived"):
        execute_confirmatory(
            repo_path=tmp_path,
            bundle_path=tmp_path / "preregistrations" / "ACL-008",
            reference_path=tmp_path / "source.json",
            approved_sha=approved_sha,
            output_path=tmp_path / "wrong.json",
        )


def test_runner_rejects_dirty_tree_before_bundle_or_outcomes(
    tmp_path: Path, monkeypatch
) -> None:
    approved_sha = "b" * 40
    monkeypatch.setattr(
        acl008_module, "git_execution_state", lambda repo_path: (approved_sha, True)
    )
    output = tmp_path / "evidence" / f"ACL-008-confirmatory-{approved_sha}.json"
    with pytest.raises(ValueError, match="worktree must be completely clean"):
        execute_confirmatory(
            repo_path=tmp_path,
            bundle_path=tmp_path / "preregistrations" / "ACL-008",
            reference_path=tmp_path / "source.json",
            approved_sha=approved_sha,
            output_path=output,
        )


def test_real_bundle_is_closed_and_outcome_free() -> None:
    validation = validate_preregistration_bundle("preregistrations/ACL-008")
    assert validation["outcomes_generated"] is False
    assert validation["confirmatory_target_count"] == 16
    assert validation["regular_target_count"] == 16
    registry = json.loads(
        Path("preregistrations/ACL-008/analytic_registry.json").read_text(
            encoding="utf-8"
        )
    )
    assert registry["outcomes_generated"] is False


def test_real_bundle_rejects_unlocked_neighbor(tmp_path: Path) -> None:
    bundle = tmp_path / "ACL-008"
    shutil.copytree("preregistrations/ACL-008", bundle)
    (bundle / "unlocked.txt").write_text("forbidden\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exact frozen directory"):
        validate_preregistration_bundle(bundle)
