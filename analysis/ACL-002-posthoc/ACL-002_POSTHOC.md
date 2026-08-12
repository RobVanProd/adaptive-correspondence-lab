# ACL-002 deterministic post-confirmatory analysis

## Status and provenance

This package is a deterministic post-confirmatory analysis of the preserved ACL-002 artifact. It generated no new trajectories, did not invoke the confirmatory runner, did not refit target predictions, and does not change the ACL-002 verdict.

- Source artifact SHA-256: `4d08e85b927a5d78a29078ff0d6549225d98069b20186b754629464739f29d74`
- Approved preregistration: `3f6a935942f43c7d3055582d123e58af5bf3f38b`
- Evidence commit: `5caf47b510d70564415354f34ba729ff505f7ed4`
- Analysis-code commit: `f85eb1bef950285ac80123eed89ed862a775d441`
- Stored rows: 896
- Classification: post-confirmatory and exploratory except where the immutable ACL-002 verdict is restated explicitly.

The continuation prompt transcribed `alpha_source` as `0.9951356718983256`. The artifact stores `0.9951356698171323`, which exactly equals the frozen median of its 12 source alphas. The artifact value is authoritative; neither value was altered.

## Confirmed findings (restated, not re-tested)

ACL-002's frozen zero-fit and calibrated predictions both passed. These are within-family deterministic benchmark results, not population confidence statements or cross-class transport.

| Layer | Median max error | Type-7 Q90 | Verdict |
| --- | ---: | ---: | --- |
| Zero-fit analytic | 0.4275% | 1.4730% | PASS |
| Frozen source calibration | 0.4677% | 0.9914% | PASS |

The two special target strata and the independent matrix-power oracle also passed in the immutable artifact.

## Exploratory observations: L1 residual structure

For each stored positive-epsilon row this package computes `R = endpoint_l1 - C_endpoint_l1 * epsilon`, `R / epsilon^2`, and the dimensionless signed relative residual. No target residual is fitted away.

At the primary horizon `T=20`, all 12 regular target landscapes have negative endpoint residuals at every tested positive epsilon. The absolute relative-error distribution grows monotonically by region:

| Region | Median | Type-7 Q90 | Maximum |
| --- | ---: | ---: | ---: |
| confirmatory | 0.154% | 0.893% | 2.186% |
| extended-local | 2.626% | 10.940% | 18.672% |
| stress | 22.640% | 62.165% | 74.217% |

For `T=1`, the L1 update is affine in epsilon for the single mixing step, so the stored second-order residual is numerical zero. At `T=5,20,50`, every regular target has a negative residual throughout the strict-confirmatory and extended-local regions. Within-landscape `R/epsilon^2` is especially stable in the strict region and degrades smoothly as horizon and epsilon increase. Its magnitude varies substantially across landscapes, arguing against a universal scalar second-order correction.

Endpoint and max-path L1 are identical in 759 of 784 positive-epsilon rows. The separate columns remain in the raw tables.

### Exploratory empirical stability radius at T=20

A radius is the largest tested positive epsilon for which the entire tested prefix stays below the stated zero-fit relative-error level. These values are descriptive and were not ACL-002 gates.

| Error level | Median radius | Minimum | Maximum | Radius counts |
| ---: | ---: | ---: | ---: | --- |
| 1% | 0.001 | 0.0003 | 0.003 | 0.0003: 3, 0.001: 4, 0.003: 5 |
| 5% | 0.01 | 0.001 | 0.03 | 0.001: 1, 0.003: 4, 0.01: 6, 0.03: 1 |
| 10% | 0.01 | 0.003 | 0.03 | 0.003: 3, 0.01: 4, 0.03: 5 |
| 20% | 0.03 | 0.01 | 0.1 | 0.01: 5, 0.03: 4, 0.1: 3 |

### Exploratory horizon/covariate description

The artifact does not contain numeric catalog arrays. Reward intensity is therefore the per-step clean log-odds spread inferred from stored clean states (mathematically `eta * reward_spread`), and mutation structure is a nominal catalog ID. Numeric mutation-matrix attribution is not identifiable from this artifact alone.

The following deterministic OLS comparisons use log sensitivity and leave-one-landscape-out prediction. They are exploratory model summaries, not inferential selection:

| Response | Model | In-sample R2 | LOLO R2 | LOLO RMSE (log units) |
| --- | --- | ---: | ---: | ---: |
| C_endpoint_l1 | horizon-only | 0.695 | 0.669 | 1.081 |
| C_endpoint_l1 | additive | 0.851 | 0.770 | 0.902 |
| C_endpoint_l1 | additive-plus-interactions | 0.885 | 0.747 | 0.946 |
| C_max_path_l1 | horizon-only | 0.699 | 0.673 | 1.073 |
| C_max_path_l1 | additive | 0.855 | 0.774 | 0.891 |
| C_max_path_l1 | additive-plus-interactions | 0.889 | 0.753 | 0.933 |

Horizon is the strongest single descriptor. The additive model improves held-out-landscape description, while the fixed exploratory interaction expansion does not improve LOLO R2. This does not identify causal importance because the deterministic catalog is small and covariates are correlated.

## Exploratory observations: oriented KL

For regular cases the package computes `KL - K * epsilon^2`, `KL / epsilon^2 - K`, and the cubic normalization only when the analytic quadratic prediction exceeds the frozen numerical floor. Low- and zero-sensitivity rows remain separate.

At `T=20`, the absolute relative error of the quadratic KL prediction is:

| Region | Median | Type-7 Q90 | Maximum |
| --- | ---: | ---: | ---: |
| confirmatory | 0.773% | 5.969% | 44.385% |
| extended-local | 11.241% | 37.304% | 86.383% |
| stress | 59.814% | 91.234% | 99.361% |

The KL cubic remainder is predominantly negative and locally structured, but its coefficient spans a much wider range across landscapes than a single transportable scalar would allow. This is exploratory; ACL-002 did not gate on KL.

## Exploratory observations: frozen source calibration

The 12 frozen source alphas range from 0.986253 to 1.018112; 11 are below one and one is above one. The median is 0.995135670, the sample standard deviation is 0.007783, and the median absolute deviation is 0.003143.

Thus alpha near one is not merely an average of equally balanced positive and negative corrections: most source landscapes share a small negative finite-epsilon correction, with one opposing landscape and meaningful coefficient heterogeneity. The frozen alpha is described, never refitted.

## New hypotheses (not confirmed)

1. For horizons above one, an analytic second-order sensitivity recurrence predicts the negative L1 correction and extends the usable epsilon radius on entirely new categorical landscapes.
2. A state-aware second-order coefficient is necessary; a universal scalar correction will fail across heterogeneous landscapes.
3. The oriented-KL cubic remainder is structured but less transportable than the L1 quadratic correction.

The next step is derivation and independent finite-difference/symbolic verification of the second-order row sensitivity. No ACL-003 outcome should be generated until that mechanism either earns or fails to earn a clean preregistration.

## Files

`summary.json` is the machine-readable index. CSV files contain every derived row and grouped summary. SVG files are deterministic views of target T=20 residuals. Their SHA-256 hashes are recorded in `summary.json`.
