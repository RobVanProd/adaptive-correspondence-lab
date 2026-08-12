import numpy as np
import pytest

from adaptive_correspondence.experiments import (
    CategoricalExperimentConfig,
    fit_origin_slope,
    run_equivalence,
    run_stability_sweep,
    run_transport_demo,
    software_verification,
)


def test_exact_equivalence_over_full_path() -> None:
    result = run_equivalence(CategoricalExperimentConfig(steps=50, eta=0.07))
    assert result["passed"] is True
    assert all(item["max_path_l1"] <= 2e-14 for item in result["pairwise"].values())


def test_random_interior_cases_preserve_three_way_equivalence() -> None:
    rng = np.random.Generator(np.random.PCG64(771))
    for _ in range(50):
        initial = rng.dirichlet(np.ones(3))
        reward = rng.normal(size=3)
        config = CategoricalExperimentConfig(
            initial_state=tuple(initial),
            reward=tuple(reward),
            eta=float(rng.uniform(0.0, 0.3)),
            steps=5,
        )
        assert run_equivalence(config)["passed"] is True


def test_zero_mutation_has_zero_discrepancy() -> None:
    result = run_stability_sweep(
        CategoricalExperimentConfig(steps=8),
        perturbation_kind="mutation",
        epsilons=(0.0, 0.01),
        seed_count=3,
    )
    assert result["rows"][0]["mean_delta"] <= 2e-14
    assert result["rows"][1]["mean_delta"] > 0.0


def test_noise_sweep_replays_from_master_seed() -> None:
    kwargs = {
        "config": CategoricalExperimentConfig(steps=5, seed=23),
        "perturbation_kind": "noise",
        "epsilons": (0.02,),
        "seed_count": 4,
    }
    left = run_stability_sweep(**kwargs)
    right = run_stability_sweep(**kwargs)
    assert left["seeds"] == right["seeds"]
    assert left["rows"] == right["rows"]


@pytest.mark.parametrize(
    ("kind", "epsilon"),
    [
        ("euler", 0.02),
        ("reward-bias", 0.02),
        ("noise", 0.02),
        ("delay", 0.5),
        ("frequency", 0.02),
        ("mutation", 0.02),
        ("nonstationary", 0.02),
        ("finite-population", 0.2),
        ("constraint", 0.02),
    ],
)
def test_every_declared_perturbation_produces_finite_curve(kind: str, epsilon: float) -> None:
    result = run_stability_sweep(
        CategoricalExperimentConfig(steps=3),
        perturbation_kind=kind,
        epsilons=(epsilon,),
        seed_count=2,
    )
    assert np.isfinite(result["rows"][0]["mean_delta"])


def test_delay_sweep_declares_nonstationary_common_schedule() -> None:
    result = run_stability_sweep(
        CategoricalExperimentConfig(steps=4),
        perturbation_kind="delay",
        epsilons=(0.0, 0.5),
        seed_count=1,
    )
    assert result["config"]["common_schedule_amplitude"] == pytest.approx(0.25)
    assert result["rows"][1]["mean_delta"] > 0.0


def test_origin_slope_uses_declared_rows() -> None:
    rows = [{"epsilon": 0.0, "mean_delta": 10.0}, {"epsilon": 1.0, "mean_delta": 2.0}]
    assert fit_origin_slope(rows) == pytest.approx(2.0)


def test_transport_demo_labels_scope_and_freezes_source_coefficient() -> None:
    result = run_transport_demo(CategoricalExperimentConfig(steps=4), seed_count=2)
    assert "not a validated transport law" in result["scientific_status"]
    expected = fit_origin_slope(result["source"]["rows"])
    assert result["source_coefficient"] == expected
    for comparison in result["comparisons"]:
        assert comparison["predicted_delta"] == pytest.approx(
            expected * comparison["epsilon"]
        )


def test_software_verification_passes() -> None:
    result = software_verification()
    assert result["passed"] is True
    np.testing.assert_allclose(result["euler_halving_error_ratios"], [4.0, 4.0], rtol=0.1)
