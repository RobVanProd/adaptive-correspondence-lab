import json
from pathlib import Path

import pytest

from adaptive_correspondence.acl003_posthoc import (
    analyze_stored_rows,
    verify_acl003_evidence,
    write_report_package,
)


def _row(landscape: str, epsilon: float, observed: float, first: float, second: float) -> dict:
    return {
        "landscape_id": landscape,
        "role": "confirmatory-target",
        "stratum": "regular-sensitivity",
        "horizon": 20,
        "epsilon": epsilon,
        "region": "confirmatory" if epsilon <= 0.01 else "stress",
        "endpoint_l1": observed,
        "max_path_l1": observed,
        "first_order_prediction": first,
        "second_order_prediction": second,
        "first_order_max_path_prediction": first,
        "second_order_max_path_prediction": second,
        "matrix_oracle_max_absolute_error": 1e-16,
    }


def _toy_payload() -> dict:
    rows = [
        _row("N01", 0.001, 1.01, 1.1, 1.0),
        _row("N01", 0.003, 3.06, 3.3, 3.0),
        _row("N01", 0.01, 10.5, 11.0, 10.0),
        _row("N01", 0.03, 18.0, 33.0, 30.0),
    ]
    return {
        "approved_preregistration_sha": "a" * 40,
        "randomness_used": False,
        "target_refit": False,
        "raw_rows": rows,
        "analysis": {
            "verdict": "PASS",
            "instrument_valid": True,
            "software_controls_passed": True,
            "primary_gate": {
                "landscape_count": 1,
                "median": 0.05,
                "q90": 0.05,
                "median_threshold": 0.1,
                "q90_threshold": 0.2,
                "passed": True,
                "landscape_scores": [
                    {"landscape_id": "N01", "relative_error_max": 0.05}
                ],
            },
        },
    }


def test_stored_row_analysis_keeps_stress_out_of_confirmation() -> None:
    summary, errors, groups, radii = analyze_stored_rows(_toy_payload())

    assert summary["confirmed"]["verdict"] == "PASS"
    assert summary["confirmed"]["reproduced_score_max_difference"] == pytest.approx(0.0)
    assert summary["exploratory"]["stress_gating"] is False
    stress = [row for row in errors if row["region"] == "stress"]
    assert stress[0]["second_order_absolute_relative_error"] == pytest.approx(0.4)
    assert any(row["region"] == "stress" for row in groups)
    second_radius = next(row for row in radii if row["model"] == "second-order")
    assert second_radius["radius_5pct"] == pytest.approx(0.01)


def test_verification_rejects_wrong_bytes(tmp_path: Path) -> None:
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps(_toy_payload()), encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256"):
        verify_acl003_evidence(path, expected_sha256="0" * 64)


def test_report_package_is_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    write_report_package(_toy_payload(), first)
    write_report_package(_toy_payload(), second)

    assert sorted(path.name for path in first.iterdir()) == sorted(
        path.name for path in second.iterdir()
    )
    for path in first.iterdir():
        assert path.read_bytes() == (second / path.name).read_bytes()
