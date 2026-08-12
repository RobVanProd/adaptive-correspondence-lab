"""Small API example: verify the exact bridge, then measure a mutation departure."""

from adaptive_correspondence.experiments import (
    CategoricalExperimentConfig,
    run_equivalence,
    run_stability_sweep,
)

config = CategoricalExperimentConfig(steps=20, eta=0.05, seed=1729)
equivalence = run_equivalence(config)
print("exact correspondence passed:", equivalence["passed"])

curve = run_stability_sweep(
    config,
    perturbation_kind="mutation",
    epsilons=(0.0, 0.002, 0.01, 0.05),
    seed_count=1,
)
for row in curve["rows"]:
    print(f"epsilon={row['epsilon']:.3g} delta={row['mean_delta']:.8f}")
