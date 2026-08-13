from pathlib import Path

import pytest

from adaptive_correspondence.acl007_posthoc import (
    ACL007_EVIDENCE_PATH,
    analyze_stored_evidence,
    verify_acl007_evidence,
    write_report_package,
)


def test_real_evidence_reconstructs_cross_class_verdict() -> None:
    payload = verify_acl007_evidence(ACL007_EVIDENCE_PATH)
    summary, targets, checkpoints, contrasts = analyze_stored_evidence(payload)
    confirmed = summary["confirmed"]
    assert confirmed["transport_verdict"] == "PASS"
    assert confirmed["standardized_mean_prediction_verdict"] == "PASS"
    assert confirmed["dissociation_prediction_verdict"] == "PASS"
    assert confirmed["contrast_reproduction_verdict"] == "PASS"
    assert confirmed["maximum_checkpoint_reproduction_error"] < 5e-14
    assert summary["classification"]["adds_structurally_independent_class"] is True
    assert summary["classification"]["satisfies_phase_ii_termination_alone"] is False
    assert len(targets) == 16
    assert len(checkpoints) == 64
    assert len(contrasts) == 9


def test_wrong_hash_fails_before_analysis(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        verify_acl007_evidence(corrupt)


def test_report_package_is_deterministic(tmp_path: Path) -> None:
    payload = verify_acl007_evidence(ACL007_EVIDENCE_PATH)
    left = tmp_path / "left"
    right = tmp_path / "right"
    write_report_package(payload, left)
    write_report_package(payload, right)
    assert sorted(path.name for path in left.iterdir()) == sorted(
        path.name for path in right.iterdir()
    )
    for path in left.iterdir():
        assert path.read_bytes() == (right / path.name).read_bytes()
