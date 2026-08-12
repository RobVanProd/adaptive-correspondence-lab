import json
import shutil
from pathlib import Path

import numpy as np
import pytest

import adaptive_correspondence.acl003 as acl003_module
from adaptive_correspondence.acl002 import sha256_file
from adaptive_correspondence.acl003 import (
    analyze_raw_rows,
    build_analytic_registry,
    execute_confirmatory,
    generate_raw_rows,
    validate_catalog_novelty,
    validate_manifest_dict,
    validate_preregistration_bundle,
)

P0 = [0.27, 0.18, 0.55]
REWARD = [0.85, -0.35, 0.15]
MUTATION = [
    [0.55, 0.30, 0.15],
    [0.20, 0.50, 0.30],
    [0.25, 0.15, 0.60],
]


def _toy_manifest() -> dict:
    return {
        "schema_version": 1,
        "experiment_id": "TOY-ACL003",
        "randomness": "none",
        "benchmark_scope": "toy-new-value-benchmark",
        "inference_scope": "none",
        "transport_scope": "zero-fit-within-categorical-class",
        "vector_convention": "row",
        "eta": 0.05,
        "primary_horizon": 3,
        "secondary_horizons": [1, 2],
        "epsilon_grid": [0.0, 0.001, 0.003, 0.01, 0.03],
        "numerical_control_epsilons": [],
        "confirmatory_epsilons": [0.001, 0.003, 0.01],
        "stress_epsilons": [0.03],
        "numerical_policy": {
            "delta_floor": 2e-12,
            "quantile_method": "linear",
            "quantile_definition": "Hyndman-Fan Type 7",
            "matrix_oracle_max_abs_tolerance": 5e-13,
            "second_order_state_oracle_tolerance": 5e-13,
            "second_order_first_oracle_tolerance": 5e-11,
            "second_order_curvature_oracle_tolerance": 2e-9,
        },
        "gates": {
            "within_landscape_reduction": "maximum-over-confirmatory-epsilons",
            "target_landscape_median_relative_error_max": 0.10,
            "target_landscape_q90_relative_error_max": 0.20,
        },
        "states": {"new-p": P0},
        "rewards": {"new-r": REWARD},
        "mutation_matrices": {
            "new-m": MUTATION,
            "identity-control": np.eye(3).tolist(),
        },
        "landscapes": [
            {
                "id": "N01",
                "role": "confirmatory-target",
                "p0": "new-p",
                "reward": "new-r",
                "mutation": "new-m",
            },
            {
                "id": "C01",
                "role": "software-control",
                "p0": "new-p",
                "reward": "new-r",
                "mutation": "identity-control",
            },
        ],
        "identity_control_ids": ["C01"],
        "predeclared_low_sensitivity_ids": [],
    }


def test_toy_manifest_builds_zero_fit_second_order_registry_without_outcomes() -> None:
    manifest = validate_manifest_dict(_toy_manifest())
    registry = build_analytic_registry(manifest)

    assert registry["outcomes_generated"] is False
    assert registry["prediction_kind"] == "zero-fit-second-order-truncated-vector"
    regular = next(row for row in registry["landscapes"] if row["id"] == "N01")
    control = next(row for row in registry["landscapes"] if row["id"] == "C01")
    assert regular["C_primary"] > 0.0
    assert regular["second_derivative_l1_primary"] > 0.0
    assert control["stratum"] == "identity-control"
    assert control["C_primary"] == pytest.approx(0.0, abs=2e-14)


def test_catalog_novelty_compares_numeric_arrays_not_names() -> None:
    new = validate_manifest_dict(_toy_manifest())
    reference = {
        "states": {"old-name": [0.1, 0.2, 0.7]},
        "rewards": {"old-name": [1.0, 0.0, -1.0]},
        "mutation_matrices": {"old-name": np.eye(3).tolist()},
    }
    result = validate_catalog_novelty(new, reference)
    assert result["hypothesis_state_overlap_count"] == 0
    assert result["hypothesis_reward_overlap_count"] == 0
    assert result["hypothesis_mutation_overlap_count"] == 0

    reference["states"]["different-name"] = P0
    with pytest.raises(ValueError, match="state catalog overlaps"):
        validate_catalog_novelty(new, reference)


def test_analysis_uses_max_local_score_and_keeps_stress_non_gating() -> None:
    manifest = validate_manifest_dict(_toy_manifest())
    registry = build_analytic_registry(manifest)
    rows = generate_raw_rows(manifest)
    for row in rows:
        if row["landscape_id"] == "N01" and row["region"] == "confirmatory":
            row["endpoint_l1"] = row["second_order_prediction"] * 1.05
        if row["landscape_id"] == "N01" and row["region"] == "stress":
            row["endpoint_l1"] = row["second_order_prediction"] * 100.0
    analysis = analyze_raw_rows(manifest, registry, rows)

    assert analysis["primary_gate"]["passed"] is True
    assert analysis["primary_gate"]["landscape_scores"] == [
        {"landscape_id": "N01", "relative_error_max": pytest.approx(0.05)}
    ]
    assert analysis["stress_results_gating"] is False
    assert analysis["software_controls"][0]["passed"] is True
    assert analysis["instrument_valid"] is True
    assert analysis["verdict"] == "PASS"


def test_software_control_failure_invalidates_scientific_verdict() -> None:
    manifest = validate_manifest_dict(_toy_manifest())
    registry = build_analytic_registry(manifest)
    rows = generate_raw_rows(manifest)
    for row in rows:
        if row["landscape_id"] == "C01" and row["epsilon"] == 0.01:
            row["endpoint_l1"] = 1e-6

    analysis = analyze_raw_rows(manifest, registry, rows)

    assert analysis["primary_gate"]["passed"] is True
    assert analysis["instrument_valid"] is False
    assert analysis["verdict"] == "INVALID"


def test_raw_generation_stops_if_matrix_oracle_disagrees(monkeypatch) -> None:
    manifest = validate_manifest_dict(_toy_manifest())
    original = acl003_module.matrix_power_oracle_trajectory

    def wrong_oracle(*args, **kwargs):
        states = original(*args, **kwargs)
        states[-1, 0] += 1e-6
        states[-1, 1] -= 1e-6
        return states

    monkeypatch.setattr(acl003_module, "matrix_power_oracle_trajectory", wrong_oracle)
    with pytest.raises(FloatingPointError, match="matrix oracle"):
        generate_raw_rows(manifest)


def test_acl003_identifier_activates_frozen_design_constants() -> None:
    payload = _toy_manifest()
    payload["experiment_id"] = "ACL-003"
    with pytest.raises(ValueError, match="ACL-003 design"):
        validate_manifest_dict(payload)


def test_preregistration_validation_is_analytic_only(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_toy_manifest()), encoding="utf-8")
    manifest = validate_manifest_dict(_toy_manifest())
    registry_path = tmp_path / "analytic_registry.json"
    registry_path.write_text(json.dumps(build_analytic_registry(manifest)), encoding="utf-8")
    reference_path = tmp_path / "reference_manifest.json"
    reference_path.write_text(
        json.dumps(
            {
                "states": {"old": [0.1, 0.2, 0.7]},
                "rewards": {"old": [1.0, 0.0, -1.0]},
                "mutation_matrices": {
                    "old": [[0.5, 0.3, 0.2], [0.2, 0.5, 0.3], [0.3, 0.2, 0.5]]
                },
            }
        ),
        encoding="utf-8",
    )
    preregistration = tmp_path / "PREREGISTRATION.md"
    preregistration.write_text("toy only\n", encoding="utf-8")
    lock_path = tmp_path / "LOCK.json"
    names = ["manifest.json", "analytic_registry.json", "PREREGISTRATION.md"]
    lock_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "experiment_id": "TOY-ACL003",
                "outcomes_generated": False,
                "files": {name: sha256_file(tmp_path / name) for name in names},
            }
        ),
        encoding="utf-8",
    )

    result = validate_preregistration_bundle(tmp_path, reference_path=reference_path)

    assert result["valid"] is True
    assert result["outcomes_generated"] is False
    assert result["confirmatory_target_count"] == 1


def test_real_acl003_bundle_rejects_wrong_reference_manifest(tmp_path: Path) -> None:
    wrong_reference = tmp_path / "wrong-reference.json"
    wrong_reference.write_text(
        json.dumps({"states": {}, "rewards": {}, "mutation_matrices": {}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="reference manifest hash"):
        validate_preregistration_bundle(
            "preregistrations/ACL-003", reference_path=wrong_reference
        )


def test_real_acl003_bundle_requires_exact_lock_file_set(tmp_path: Path) -> None:
    source = Path("preregistrations/ACL-003")
    bundle = tmp_path / "ACL-003"
    shutil.copytree(source, bundle)
    lock_path = bundle / "LOCK.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    del lock["files"]["README.md"]
    lock_path.write_text(json.dumps(lock), encoding="utf-8")

    with pytest.raises(ValueError, match="exact frozen file set"):
        validate_preregistration_bundle(bundle)


def test_acl003_runner_requires_sha_derived_canonical_output(
    tmp_path: Path, monkeypatch
) -> None:
    approved_sha = "a" * 40
    monkeypatch.setattr(
        acl003_module, "git_execution_state", lambda repo_path: (approved_sha, False)
    )

    with pytest.raises(ValueError, match="canonical evidence path"):
        execute_confirmatory(
            repo_path=tmp_path,
            bundle_path=tmp_path / "missing-bundle",
            reference_path=tmp_path / "missing-reference.json",
            approved_sha=approved_sha,
            output_path=tmp_path / "outside.json",
        )
