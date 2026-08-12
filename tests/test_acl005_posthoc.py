from pathlib import Path

import pytest

from adaptive_correspondence.acl005_posthoc import (
    ACL005_EVIDENCE_PATH,
    analyze_stored_evidence,
    verify_acl005_evidence,
    write_report_package,
)


def test_real_evidence_reproduces_cross_class_verdict_and_stress_boundary() -> None:
    payload = verify_acl005_evidence(ACL005_EVIDENCE_PATH)
    summary, context_rows, h1_rows, checkpoints = analyze_stored_evidence(payload)
    confirmed = summary["confirmed"]
    assert confirmed["transport_verdict"] == "PASS"
    assert confirmed["all_regular_landscapes_converged"] is True
    assert confirmed["maximum_stopped_mean_reproduction_error"] == 0.0
    assert confirmed["maximum_final_cosine_reproduction_error"] < 2e-15
    assert confirmed["regular_minimum_context_fisher_cosine"] == pytest.approx(
        0.9998416299085249
    )
    assert summary["stress_boundary"]["minimum_context_fisher_cosine"] < 0.1
    assert summary["stress_boundary"]["gating"] is False
    assert len(context_rows) == 28
    assert len(h1_rows) == 28
    assert len(checkpoints) == 28


def test_wrong_hash_fails_before_analysis(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        verify_acl005_evidence(corrupt)


def test_report_package_is_deterministic(tmp_path: Path) -> None:
    payload = verify_acl005_evidence(ACL005_EVIDENCE_PATH)
    left = tmp_path / "left"
    right = tmp_path / "right"
    write_report_package(payload, left)
    write_report_package(payload, right)
    assert sorted(path.name for path in left.iterdir()) == sorted(
        path.name for path in right.iterdir()
    )
    for path in left.iterdir():
        assert path.read_bytes() == (right / path.name).read_bytes()
