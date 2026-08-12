from pathlib import Path

import pytest

from adaptive_correspondence.acl006_posthoc import (
    ACL006_EVIDENCE_PATH,
    analyze_stored_evidence,
    verify_acl006_evidence,
    write_report_package,
)


def test_real_evidence_reconstructs_all_three_verdicts() -> None:
    payload = verify_acl006_evidence(ACL006_EVIDENCE_PATH)
    summary, targets, checkpoints, contrasts = analyze_stored_evidence(payload)
    assert summary["confirmed"]["exact_mean_prediction_verdict"] == "PASS"
    assert summary["confirmed"]["dissociation_prediction_verdict"] == "PASS"
    assert summary["confirmed"]["stochastic_contrast_reproduction_verdict"] == "PASS"
    assert summary["confirmed"]["maximum_checkpoint_reproduction_error"] < 5e-14
    assert summary["boundary"]["minimum_dissociation_truth_cosine"] < 0.5
    assert summary["boundary"]["minimum_dissociation_half_cosine"] > 0.99999
    assert summary["classification"]["adds_structurally_independent_class"] is False
    assert len(targets) == 16
    assert len(checkpoints) == 64
    assert len(contrasts) == 9


def test_wrong_hash_fails_before_analysis(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        verify_acl006_evidence(corrupt)


def test_report_package_is_deterministic(tmp_path: Path) -> None:
    payload = verify_acl006_evidence(ACL006_EVIDENCE_PATH)
    left = tmp_path / "left"
    right = tmp_path / "right"
    write_report_package(payload, left)
    write_report_package(payload, right)
    assert sorted(path.name for path in left.iterdir()) == sorted(
        path.name for path in right.iterdir()
    )
    for path in left.iterdir():
        assert path.read_bytes() == (right / path.name).read_bytes()
