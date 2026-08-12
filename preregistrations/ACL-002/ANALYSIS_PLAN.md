# ACL-002 frozen analysis plan

This file is a compact implementation contract. The narrative preregistration is
authoritative where additional context is needed.

## Constants

| Quantity | Frozen value |
|---|---:|
| `eta` | `0.05` |
| primary horizon | `20` |
| secondary horizons | `1, 5, 50` |
| epsilon grid | `0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1` |
| positive confirmatory epsilons | `1e-4, 3e-4, 1e-3, 3e-3, 1e-2` |
| stress epsilons | `3e-2, 1e-1` |
| inherited numeric tolerance | `2e-14` |
| safety multiplier | `100` |
| absolute discrepancy floor | `2e-12` |
| target median relative-error gate | `0.10` |
| target Q0.90 relative-error gate | `0.20` |
| quantile convention | Hyndman-Fan Type 7 / NumPy `method="linear"` |

## Estimation and reduction

1. Use primary-horizon endpoint L1 only for alpha estimation and primary gates.
2. Exclude analytic-zero and low-sensitivity sources from alpha estimation.
3. Fit each remaining source landscape separately through the origin over all five
   positive confirmatory epsilons.
4. Set `alpha_source` to the Type-7 median of landscape alphas.
5. For each regular target and prediction layer, calculate five relative errors and
   reduce them to a Type-7 median landscape score.
6. Across regular target landscape scores, calculate Type-7 median and Q0.90.
7. Apply the `0.10/0.20` conjunction separately to zero-fit and calibrated layers.
8. If a prediction layer produces any nonpositive regular-target prediction, that
   layer fails; do not repair denominators.

## Special strata

Classify from primary-horizon `C` before outcomes:

```text
analytic-zero: C <= 2e-14
low:           C > 2e-14 and C*1e-2 < 2e-12
regular:       otherwise
```

Special landscapes do not enter relative fits or gates. For each prediction layer,
report every local absolute error and whether all are `<=2e-12`.

## Raw rows

Before summaries, preserve one row for each landscape, epsilon, and declared horizon.
Each row includes split, IDs, clean and perturbed terminal states, endpoint L1,
max-path L1, oriented KL, `C`, path `C`, `K`, zero-fit L1 prediction, zero-fit KL
prediction, region, and stratum. No row is dropped.

## Missing, invalid, or exceptional data

The deterministic implementation admits no missing-value policy. A non-finite value,
simplex violation, lock mismatch, registry mismatch, duplicate ID, incomplete grid, or
unexpected row count stops execution before summary verdicts. Failures are not imputed,
winsorized, clipped, projected, or silently omitted.

## Reporting order

1. Provenance and lock verification.
2. Manifest and pre-outcome strata.
3. Raw primary rows.
4. `H_analytic` target verdict.
5. Source landscape alphas and frozen `alpha_source`.
6. `H_transport` target verdict.
7. Special-stratum absolute checks.
8. Secondary horizons and max-path L1.
9. Oriented KL quadratic tables.
10. Stress results, labeled non-gating.
