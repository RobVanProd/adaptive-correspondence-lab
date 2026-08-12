import json
from pathlib import Path

import numpy as np
import pytest

from adaptive_correspondence.acl002 import (
    DELTA_FLOOR,
    INHERITED_TOLERANCE,
    Landscape,
    analytic_coefficients,
    analyze_raw_rows,
    assert_execution_context,
    build_analytic_registry,
    categorical_map,
    classify_sensitivity,
    evaluate_gate,
    generate_raw_rows,
    landscape_relative_score,
    median_source_alpha,
    mutation_trajectory,
    per_landscape_alpha,
    row_jacobian,
    sensitivity_trajectory,
    sha256_file,
    type7_quantile,
    validate_lock,
    validate_manifest_dict,
    validate_preregistration_bundle,
)

P0 = np.array([0.2, 0.3, 0.5])
REWARD = np.array([0.7, -0.2, 0.1])
MUTATION = np.array(
    [
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0],
    ]
)


def _toy_manifest() -> dict:
    return {
        "schema_version": 1,
        "experiment_id": "TOY-ONLY",
        "randomness": "none",
        "vector_convention": "row",
        "mutation_operation": "(1-epsilon)*F(p) + epsilon*(F(p) @ M)",
        "eta": 0.05,
        "primary_horizon": 3,
        "secondary_horizons": [1, 2],
        "epsilon_grid": [0.0, 0.001, 0.01, 0.1],
        "confirmatory_epsilons": [0.001, 0.01],
        "stress_epsilons": [0.1],
        "numerical_policy": {
            "inherited_tolerance": 2e-14,
            "safety_multiplier": 100,
            "delta_floor": 2e-12,
            "quantile_method": "linear",
            "quantile_definition": "Hyndman-Fan Type 7",
        },
        "gates": {
            "target_landscape_median_relative_error_max": 0.1,
            "target_landscape_q90_relative_error_max": 0.2,
        },
        "states": {"p": P0.tolist()},
        "rewards": {"r": REWARD.tolist()},
        "mutation_matrices": {"m": MUTATION.tolist()},
        "landscapes": [
            {"id": "DEV-S", "split": "source", "p0": "p", "reward": "r", "mutation": "m"},
            {"id": "DEV-T", "split": "target", "p0": "p", "reward": "r", "mutation": "m"},
        ],
        "predeclared_stratum_expectations": {
            "analytic_zero": [],
            "low_sensitivity": [],
        },
    }


def test_row_jacobian_matches_ambient_finite_difference() -> None:
    jacobian = row_jacobian(P0, REWARD, 0.05)
    finite_difference = np.empty_like(jacobian)
    step = 1e-7
    for input_index in range(P0.size):
        plus = P0.copy()
        minus = P0.copy()
        plus[input_index] += step
        minus[input_index] -= step
        finite_difference[input_index] = (
            categorical_map(plus, REWARD, 0.05) - categorical_map(minus, REWARD, 0.05)
        ) / (2.0 * step)
    np.testing.assert_allclose(jacobian, finite_difference, atol=5e-10, rtol=0.0)
    np.testing.assert_allclose(np.sum(jacobian, axis=1), 0.0, atol=2e-15)


def test_row_sensitivity_matches_one_sided_epsilon_difference() -> None:
    trace = sensitivity_trajectory(P0, REWARD, MUTATION, eta=0.05, steps=8)
    epsilon = 1e-8
    perturbed = mutation_trajectory(P0, REWARD, MUTATION, eta=0.05, epsilon=epsilon, steps=8)
    finite_difference = (perturbed[-1] - trace.states[-1]) / epsilon
    np.testing.assert_allclose(trace.sensitivities[-1], finite_difference, atol=3e-7, rtol=0.0)
    np.testing.assert_allclose(np.sum(trace.sensitivities, axis=1), 0.0, atol=2e-14)


def test_analytic_l1_and_oriented_kl_coefficients_match_local_behavior() -> None:
    trace = sensitivity_trajectory(P0, REWARD, MUTATION, eta=0.05, steps=5)
    coefficients = analytic_coefficients(trace, horizon=5)
    epsilon = 1e-6
    perturbed = mutation_trajectory(P0, REWARD, MUTATION, eta=0.05, epsilon=epsilon, steps=5)
    clean = trace.states[-1]
    endpoint_l1 = float(np.sum(np.abs(perturbed[-1] - clean)))
    oriented_kl = float(np.sum(perturbed[-1] * np.log(perturbed[-1] / clean)))
    assert endpoint_l1 / epsilon == pytest.approx(coefficients.endpoint_l1, rel=2e-5)
    assert oriented_kl / epsilon**2 == pytest.approx(coefficients.kl_q_p, rel=2e-4)


def test_landscape_balanced_alpha_and_type7_quantiles_are_exactly_frozen() -> None:
    epsilons = np.array([1e-4, 3e-4, 1e-3, 3e-3, 1e-2])
    coefficient = 2.5
    deltas = 0.8 * coefficient * epsilons
    assert per_landscape_alpha(coefficient, epsilons, deltas) == pytest.approx(0.8)
    assert median_source_alpha([0.8, 1.0, 9.0]) == pytest.approx(1.0)
    assert type7_quantile([1.0, 2.0, 3.0, 4.0], 0.9) == pytest.approx(3.7)


def test_relative_scores_and_gates_reduce_by_landscape() -> None:
    epsilons = np.array([1e-4, 3e-4, 1e-3, 3e-3, 1e-2])
    coefficient = 3.0
    observed = coefficient * epsilons * np.array([1.01, 0.99, 1.03, 0.97, 1.02])
    score = landscape_relative_score(coefficient, epsilons, observed, alpha=1.0)
    assert score == pytest.approx(0.02)
    passing = evaluate_gate([0.02, 0.04, 0.08, 0.1])
    assert passing.passed is True
    assert passing.median <= 0.1
    failing = evaluate_gate([0.02, 0.04, 0.08, 0.3])
    assert failing.passed is False


def test_sensitivity_strata_use_only_frozen_analytic_thresholds() -> None:
    assert INHERITED_TOLERANCE == 2e-14
    assert DELTA_FLOOR == 2e-12
    assert classify_sensitivity(0.0) == "analytic-zero"
    assert classify_sensitivity(1e-13) == "low-sensitivity"
    assert classify_sensitivity(1e-8) == "regular-sensitivity"


def test_toy_manifest_validation_and_registry_do_not_need_outcomes() -> None:
    manifest = validate_manifest_dict(_toy_manifest())
    assert [landscape.identifier for landscape in manifest.landscapes] == ["DEV-S", "DEV-T"]
    assert isinstance(manifest.landscapes[0], Landscape)
    registry = build_analytic_registry(manifest)
    assert registry["outcomes_generated"] is False
    assert {entry["id"] for entry in registry["landscapes"]} == {"DEV-S", "DEV-T"}
    assert all(entry["C_primary"] > 0.0 for entry in registry["landscapes"])


def test_frozen_analysis_keeps_two_gates_and_stress_non_gating_on_toy_data() -> None:
    manifest = validate_manifest_dict(_toy_manifest())
    registry = build_analytic_registry(manifest)
    rows = generate_raw_rows(manifest)
    assert len(rows) == len(manifest.landscapes) * len(manifest.epsilon_grid) * len(
        manifest.horizons
    )
    analysis = analyze_raw_rows(manifest, registry, rows)
    positive_row = next(row for row in rows if row["epsilon"] > 0.0)
    assert positive_row["kl_over_epsilon_squared"] is not None
    assert positive_row["kl_coefficient_error"] is not None
    zero_row = next(row for row in rows if row["epsilon"] == 0.0)
    assert zero_row["kl_over_epsilon_squared"] is None
    assert set(analysis["primary_gates"]) == {"analytic", "transport"}
    assert analysis["stress_results_gating"] is False
    assert analysis["alpha_source"] > 0.0
    assert len(analysis["target_prediction_rows"]) == 4


def test_manifest_rejects_non_row_stochastic_mutation() -> None:
    payload = _toy_manifest()
    payload["mutation_matrices"]["m"][0][0] = 0.25
    with pytest.raises(ValueError, match="row-stochastic"):
        validate_manifest_dict(payload)


def test_acl002_id_activates_exact_frozen_design_constants() -> None:
    payload = _toy_manifest()
    payload["experiment_id"] = "ACL-002"
    with pytest.raises(ValueError, match="design constants"):
        validate_manifest_dict(payload)


def test_lock_hash_validation_detects_any_change(tmp_path: Path) -> None:
    preregistration = tmp_path / "PREREGISTRATION.md"
    preregistration.write_text("frozen\n", encoding="utf-8")
    lock = tmp_path / "LOCK.json"
    lock.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "experiment_id": "TOY-ONLY",
                "files": {"PREREGISTRATION.md": sha256_file(preregistration)},
            }
        ),
        encoding="utf-8",
    )
    validate_lock(lock)
    preregistration.write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_lock(lock)


def test_preregistration_bundle_validation_uses_toy_analytics_only(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_toy_manifest()), encoding="utf-8")
    manifest = validate_manifest_dict(_toy_manifest())
    registry_path = tmp_path / "analytic_registry.json"
    registry_path.write_text(json.dumps(build_analytic_registry(manifest)), encoding="utf-8")
    preregistration = tmp_path / "PREREGISTRATION.md"
    preregistration.write_text("toy only\n", encoding="utf-8")
    lock_path = tmp_path / "LOCK.json"
    locked_names = ["manifest.json", "analytic_registry.json", "PREREGISTRATION.md"]
    lock_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "experiment_id": "TOY-ONLY",
                "files": {
                    name: sha256_file(tmp_path / name) for name in locked_names
                },
            }
        ),
        encoding="utf-8",
    )
    result = validate_preregistration_bundle(tmp_path)
    assert result["valid"] is True
    assert result["outcomes_generated"] is False
    assert result["landscape_count"] == 2


def test_execution_context_requires_exact_sha_clean_tree_and_new_output(tmp_path: Path) -> None:
    output = tmp_path / "raw.json"
    assert_execution_context(
        approved_sha="abc123",
        current_sha="abc123",
        tracked_dirty=False,
        output_path=output,
    )
    with pytest.raises(ValueError, match="approved SHA"):
        assert_execution_context(
            approved_sha="abc123",
            current_sha="different",
            tracked_dirty=False,
            output_path=output,
        )
    with pytest.raises(ValueError, match="clean"):
        assert_execution_context(
            approved_sha="abc123",
            current_sha="abc123",
            tracked_dirty=True,
            output_path=output,
        )
    output.write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError):
        assert_execution_context(
            approved_sha="abc123",
            current_sha="abc123",
            tracked_dirty=False,
            output_path=output,
        )
