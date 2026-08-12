import json
import math
from pathlib import Path

import pytest

import adaptive_correspondence.acl002_posthoc as posthoc
from adaptive_correspondence.acl002_posthoc import (
    ACL002_APPROVED_SHA,
    ACL002_ARTIFACT_SHA256,
    ACL002_EVIDENCE_COMMIT,
    derive_kl_rows,
    derive_l1_rows,
    empirical_stability_radii,
    generate_posthoc_package,
    horizon_feature_rows,
    summarize_source_alphas,
    verify_acl002_artifact,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    REPOSITORY_ROOT
    / "evidence"
    / "ACL-002-confirmatory-3f6a935942f43c7d3055582d123e58af5bf3f38b.json"
)


def _raw_row(
    *,
    epsilon: float,
    endpoint_l1: float,
    max_path_l1: float,
    kl_q_p: float,
    c_endpoint: float = 2.0,
    c_path: float = 3.0,
    k_kl: float = 4.0,
    stratum: str = "regular-sensitivity",
    horizon: int = 20,
    clean_terminal: list[float] | None = None,
) -> dict:
    return {
        "landscape_id": "T99",
        "split": "target",
        "stratum": stratum,
        "horizon": horizon,
        "epsilon": epsilon,
        "region": "confirmatory",
        "p0": "toy-state",
        "reward": "toy-reward",
        "mutation": "toy-mutation",
        "clean_terminal": clean_terminal or [0.1, 0.3, 0.6],
        "perturbed_terminal": [0.11, 0.295, 0.595],
        "C_endpoint_l1": c_endpoint,
        "C_max_path_l1": c_path,
        "K_kl_q_p": k_kl,
        "endpoint_l1": endpoint_l1,
        "max_path_l1": max_path_l1,
        "kl_q_p": kl_q_p,
    }


def test_immutable_acl002_artifact_contract() -> None:
    payload, verification = verify_acl002_artifact(ARTIFACT, repo_path=REPOSITORY_ROOT)

    assert verification["artifact_sha256"] == ACL002_ARTIFACT_SHA256
    assert verification["approved_preregistration_sha"] == ACL002_APPROVED_SHA
    assert verification["evidence_commit"] == ACL002_EVIDENCE_COMMIT
    assert verification["row_count"] == 896
    assert verification["region_counts"] == {
        "confirmatory": 336,
        "extended-local": 224,
        "stress": 224,
        "zero": 112,
    }
    assert verification["matrix_oracle"]["passed"] is True
    assert verification["matrix_oracle"]["tolerance"] == 5e-13
    assert payload["analysis"]["alpha_source"] == 0.9951356698171323


def test_artifact_verification_fails_closed_on_changed_bytes(tmp_path: Path) -> None:
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    payload["analysis"]["alpha_source"] = 1.0
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256"):
        verify_acl002_artifact(changed)


def test_l1_rows_preserve_signed_second_order_and_dimensionless_residual() -> None:
    row = _raw_row(
        epsilon=0.1,
        endpoint_l1=0.21,
        max_path_l1=0.315,
        kl_q_p=0.0405,
    )

    derived = derive_l1_rows([row])[0]

    assert derived["endpoint_prediction"] == pytest.approx(0.2)
    assert derived["endpoint_residual"] == pytest.approx(0.01)
    assert derived["endpoint_residual_over_epsilon_squared"] == pytest.approx(1.0)
    assert derived["endpoint_relative_residual"] == pytest.approx(0.05)
    assert derived["max_path_prediction"] == pytest.approx(0.3)
    assert derived["max_path_residual"] == pytest.approx(0.015)
    assert derived["max_path_relative_residual"] == pytest.approx(0.05)
    assert derived["clean_boundary_min_probability"] == pytest.approx(0.1)


def test_l1_relative_residual_is_absent_for_zero_sensitivity() -> None:
    row = _raw_row(
        epsilon=0.001,
        endpoint_l1=2e-16,
        max_path_l1=3e-16,
        kl_q_p=1e-18,
        c_endpoint=0.0,
        c_path=0.0,
        k_kl=0.0,
        stratum="analytic-zero",
    )

    derived = derive_l1_rows([row])[0]

    assert derived["endpoint_relative_residual"] is None
    assert derived["max_path_relative_residual"] is None


def test_empirical_radius_requires_a_contiguous_passing_prefix() -> None:
    rows = []
    for epsilon, relative_error in [
        (1e-4, 0.005),
        (3e-4, 0.20),
        (1e-3, 0.01),
        (3e-3, 0.03),
    ]:
        prediction = 2.0 * epsilon
        rows.extend(
            derive_l1_rows(
                [
                    _raw_row(
                        epsilon=epsilon,
                        endpoint_l1=prediction * (1.0 + relative_error),
                        max_path_l1=3.0 * epsilon,
                        kl_q_p=4.0 * epsilon**2,
                    )
                ]
            )
        )

    radii = empirical_stability_radii(rows, metric="endpoint")
    by_level = {row["relative_error_level"]: row for row in radii}

    assert by_level[0.01]["largest_tested_epsilon"] == pytest.approx(1e-4)
    assert by_level[0.05]["largest_tested_epsilon"] == pytest.approx(1e-4)
    assert by_level[0.10]["largest_tested_epsilon"] == pytest.approx(1e-4)
    assert by_level[0.20]["largest_tested_epsilon"] == pytest.approx(3e-3)


def test_kl_rows_use_quadratic_and_guard_cubic_normalizations() -> None:
    regular = _raw_row(
        epsilon=0.1,
        endpoint_l1=0.2,
        max_path_l1=0.3,
        kl_q_p=0.0405,
    )
    zero = _raw_row(
        epsilon=0.001,
        endpoint_l1=0.0,
        max_path_l1=0.0,
        kl_q_p=1e-18,
        c_endpoint=0.0,
        c_path=0.0,
        k_kl=0.0,
        stratum="analytic-zero",
    )

    derived = derive_kl_rows([regular, zero])
    regular_derived = next(row for row in derived if row["stratum"] == "regular-sensitivity")
    zero_derived = next(row for row in derived if row["stratum"] == "analytic-zero")

    assert regular_derived["kl_prediction"] == pytest.approx(0.04)
    assert regular_derived["kl_residual"] == pytest.approx(0.0005)
    assert regular_derived["kl_over_epsilon_squared_minus_k"] == pytest.approx(0.05)
    assert regular_derived["kl_residual_over_epsilon_cubed"] == pytest.approx(0.5)
    assert zero_derived["kl_residual_over_epsilon_cubed"] is None


def test_source_alpha_summary_does_not_refit() -> None:
    payload = {
        "analysis": {
            "alpha_source": 1.0,
            "source_landscape_alphas": [
                {"landscape_id": "S01", "alpha": 0.9},
                {"landscape_id": "S02", "alpha": 1.1},
                {"landscape_id": "S03", "alpha": 1.0},
            ],
        }
    }

    summary = summarize_source_alphas(payload)

    assert summary["frozen_alpha_source"] == 1.0
    assert summary["recomputed_median_for_verification"] == 1.0
    assert summary["count_below_one"] == 1
    assert summary["count_equal_one"] == 1
    assert summary["count_above_one"] == 1
    assert summary["heterogeneous_correction_signs"] is True


def test_clean_log_odds_identify_selection_spread_without_manifest() -> None:
    p0 = [0.2, 0.3, 0.5]
    per_step_log_factors = [0.10, 0.0, -0.05]
    raw_rows = []
    for horizon in [1, 5, 20, 50]:
        masses = [
            probability * math.exp(horizon * log_factor)
            for probability, log_factor in zip(p0, per_step_log_factors, strict=True)
        ]
        total = sum(masses)
        clean = [mass / total for mass in masses]
        raw_rows.append(
            _raw_row(
                epsilon=0.001,
                endpoint_l1=0.002,
                max_path_l1=0.003,
                kl_q_p=4e-6,
                horizon=horizon,
                clean_terminal=clean,
            )
        )

    features = horizon_feature_rows(derive_l1_rows(raw_rows))

    assert len(features) == 4
    assert [row["selection_log_factor_spread_per_step"] for row in features] == pytest.approx(
        [0.15] * 4
    )
    assert max(row["clean_log_odds_linear_max_abs_error"] for row in features) < 1e-12
    assert features[0]["inferred_initial_boundary_min_probability"] == pytest.approx(0.2)


def test_posthoc_package_is_deterministic_and_does_not_change_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(posthoc, "_git_analysis_code_state", lambda _: "test-code-commit")
    monkeypatch.setattr(posthoc, "_verify_evidence_commit", lambda _repo, _artifact: None)
    source_before = ARTIFACT.read_bytes()
    first = tmp_path / "first"
    second = tmp_path / "second"

    first_files = generate_posthoc_package(
        artifact_path=ARTIFACT, output_dir=first, repo_path=REPOSITORY_ROOT
    )
    second_files = generate_posthoc_package(
        artifact_path=ARTIFACT, output_dir=second, repo_path=REPOSITORY_ROOT
    )

    assert ARTIFACT.read_bytes() == source_before
    assert [path.name for path in first_files] == [path.name for path in second_files]
    assert {
        path.name: path.read_bytes() for path in first_files
    } == {path.name: path.read_bytes() for path in second_files}
    summary = json.loads((first / "summary.json").read_text(encoding="utf-8"))
    assert summary["classification"] == "deterministic-post-confirmatory-exploratory-analysis"
    assert summary["analysis_code_commit"] == "test-code-commit"
    assert len((first / "l1-residuals.csv").read_text(encoding="utf-8").splitlines()) == 785
    assert (first / "ACL-002_POSTHOC.md").is_file()
    assert (first / "target-t20-l1-relative-residual.svg").is_file()
