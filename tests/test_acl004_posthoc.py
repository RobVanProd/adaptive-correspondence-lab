from pathlib import Path

import pytest

from adaptive_correspondence.acl004_posthoc import (
    ACL004_EVIDENCE_PATH,
    analyze_stored_evidence,
    verify_acl004_evidence,
    write_report_package,
)


def test_real_evidence_reproduces_chunk_means_and_block_cosines() -> None:
    payload = verify_acl004_evidence(ACL004_EVIDENCE_PATH)
    summary, h2_rows, h1_rows, checkpoints = analyze_stored_evidence(payload)

    assert summary["confirmed"]["h2_verdict"] == "PASS"
    assert summary["confirmed"]["maximum_stopped_mean_reproduction_error"] == 0.0
    assert summary["confirmed"]["maximum_final_cosine_reproduction_error"] < 2e-15
    assert len(h2_rows) == 12
    assert len(h1_rows) == 24
    assert len(checkpoints) == 12


def test_wrong_hash_fails_before_analysis(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        verify_acl004_evidence(corrupt)


def test_report_package_is_deterministic(tmp_path: Path) -> None:
    payload = verify_acl004_evidence(ACL004_EVIDENCE_PATH)
    left = tmp_path / "left"
    right = tmp_path / "right"
    write_report_package(payload, left)
    write_report_package(payload, right)
    assert sorted(path.name for path in left.iterdir()) == sorted(
        path.name for path in right.iterdir()
    )
    for path in left.iterdir():
        assert path.read_bytes() == (right / path.name).read_bytes()
