from pathlib import Path

import pytest

from adaptive_correspondence.acl008_posthoc import (
    ACL008_EVIDENCE_PATH,
    analyze_stored_evidence,
    verify_acl008_evidence,
    write_report_package,
)


def test_real_evidence_reconstructs_non_fisher_transport() -> None:
    payload = verify_acl008_evidence(ACL008_EVIDENCE_PATH)
    summary, predictions = analyze_stored_evidence(payload)
    assert summary["confirmed"]["verdict"] == "PASS"
    assert summary["confirmed"]["reproduced_primary_median"] < 0.01
    assert summary["confirmed"]["reproduced_primary_q90"] < 0.01
    assert summary["boundary"]["stress_maximum_relative_error"] > 1.0
    assert summary["classification"]["adds_distinct_geometry"] is True
    assert len(predictions) == 112


def test_wrong_hash_fails_before_analysis(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        verify_acl008_evidence(corrupt)


def test_report_package_is_deterministic(tmp_path: Path) -> None:
    payload = verify_acl008_evidence(ACL008_EVIDENCE_PATH)
    left = tmp_path / "left"
    right = tmp_path / "right"
    write_report_package(payload, left)
    write_report_package(payload, right)
    for path in left.iterdir():
        assert path.read_bytes() == (right / path.name).read_bytes()
