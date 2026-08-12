# ACL-002 preregistration: categorical mutation stability

## Status and scope

This document preregisters one deterministic confirmatory experiment. At this
checkpoint no ACL-002 perturbed trajectory has been generated and no confirmatory
outcome exists. Software-development tests use separate toy fixtures only.

ACL-002 asks how accurately a first-order sensitivity calculation predicts departure
from the clean categorical correspondence when a frozen row-stochastic mutation
operator is applied after every exact categorical update.

The Gaussian and bandit rungs are out of scope. Frequency-dependent rewards,
finite-population sampling, delayed feedback, noise, and adaptive model changes are out
of scope.

## System and orientation

Every probability and sensitivity is a **row vector**. For fixed reward row vector
`r` and step size `eta`, define

```text
F(p)_j = p_j exp(eta r_j) / sum_k p_k exp(eta r_k).
```

The clean and mutation-perturbed recurrences are

```text
p_(t+1) = F(p_t)
q_(t+1) = (1-epsilon) F(q_t) + epsilon F(q_t) M,
```

where `M` is row-stochastic and therefore acts by right multiplication. The row
Jacobian is indexed as

```text
(J_F^R)_[i,j] = partial F_j / partial p_i.
```

The sensitivity remains a row vector and obeys

```text
s_0 = 0
s_(t+1) = s_t J_F^R(p_t) + p_(t+1) (M-I).
```

No implicit transpose or column-vector convention is permitted in ACL-002.

## Hypotheses

### Primary zero-fit hypothesis, H_analytic

At primary horizon `T=20`, with endpoint L1 discrepancy

```text
delta_l(epsilon) = ||q_(l,T) - p_(l,T)||_1,
```

the analytic first-order coefficient for landscape `l` is

```text
C_l = ||s_(l,T)||_1
delta_hat_0 = C_l epsilon.
```

`H_analytic` receives its own held-out verdict with `alpha=1`. It cannot be replaced or
obscured by a calibrated result.

### Primary calibrated-transport hypothesis, H_transport

For each regular-sensitivity source landscape, fit one origin-constrained coefficient
over the five positive confirmatory epsilon values:

```text
x_(l,e) = C_l e
alpha_l = sum_e x_(l,e) delta_(l,e) / sum_e x_(l,e)^2.
```

Each source landscape contributes one value. Freeze

```text
alpha_source = median_l(alpha_l)
delta_hat_transport = alpha_source C_l epsilon.
```

The median is the ordinary empirical median, equivalent to Hyndman-Fan Type 7 at
`q=0.5`. Target outcomes are not used to estimate or alter `alpha_source`.

If `alpha_source` is materially different from one but transports, the interpretation
is a transferable finite-epsilon calibration—not validation of the raw first-order
approximation over the region.

## Experimental units and manifest

Landscapes, not seeds, are experimental units. ACL-002 is deterministic and uses no RNG.
The locked manifest contains 14 source and 14 held-out target landscapes. Each resolves
to an explicit strictly interior `p0`, fixed reward vector `r`, and row-stochastic `M`.
Reward catalogs contain dominance, weak-selection, tied, neutral, compressed, reversed,
and additive-shift controls. They do not call fixed reward vectors cooperative or cyclic.

Source/target identity, catalogs, and landscape combinations are frozen in
`manifest.json`. The manifest may be reviewed after this checkpoint but any change
requires a new preregistration commit and invalidates this checkpoint for execution.

## Horizons and epsilon regions

- Primary horizon: `T=20`.
- Secondary endpoint horizons: `T=1,5,50`.
- Full epsilon grid: `0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1`.
- Confirmatory local region: the five positive values from `1e-4` through `1e-2`.
- Stress region: `3e-2` and `1e-1`.

The zero point is a consistency check. Stress outcomes are reported but are incapable
of changing either confirmatory verdict.

## Sensitivity strata and numerical floor

The inherited exact-correspondence absolute tolerance is `2e-14`. Before outcomes, a
100-fold safety multiplier defines

```text
delta_floor = 2e-12.
```

At the primary horizon:

- analytic-zero: `C_l <= 2e-14`;
- low-sensitivity: `C_l > 2e-14` and `C_l * 1e-2 < 2e-12`;
- regular-sensitivity: all remaining landscapes.

Strata use only clean analytic quantities and are frozen in `analytic_registry.json`.
Analytic-zero and low-sensitivity landscapes are excluded from relative-error fitting
and gates. They remain fully reported. For each prediction layer they pass their
separate absolute check only if every local-region absolute prediction error is at most
`delta_floor`.

## Primary gates

For each regular target landscape and prediction layer, calculate

```text
relative_error_(l,e) = |delta_(l,e) - delta_hat_(l,e)| / delta_hat_(l,e)
```

at all five positive confirmatory epsilons. A nonpositive prediction automatically
fails that prediction layer. Reduce the five errors to one landscape score with a
Hyndman-Fan Type-7 median. Across target-landscape scores, calculate the Type-7 median
and Type-7 0.90 quantile (`numpy.quantile(..., method="linear")`).

A prediction layer passes exactly when both hold:

```text
target median <= 0.10
target Q_0.90 <= 0.20.
```

Apply this gate independently to `H_analytic` (`alpha=1`) and `H_transport`
(`alpha=alpha_source`). There is no post hoc combined verdict.

## Secondary analyses

- Endpoint L1 at `T=1,5,50`.
- Max-path L1 through each horizon, with analytic path coefficient
  `max_(t<=T) ||s_t||_1`.
- Oriented relative entropy `D_KL(q_T || p_T)` only.

For the oriented KL secondary analysis,

```text
K_(l,T) = 0.5 sum_i s_(l,T,i)^2 / p_(l,T,i)
D_KL(q_T || p_T) = K_(l,T) epsilon^2 + O(epsilon^3).
```

Report `D_KL / epsilon^2`, `K`, and their error at every positive epsilon. KL and all
other secondary analyses are descriptive and cannot change primary verdicts.

## Frozen analysis sequence

1. Verify the approved Git SHA, clean tracked worktree, and preregistration file hashes.
2. Validate all manifest states, rewards, matrices, identities, counts, and regions.
3. Recompute the clean trajectories, row Jacobians, sensitivities, `C`, `K`, and strata;
   require exact agreement with the locked analytic registry within frozen tolerances.
4. Generate every declared perturbed trajectory exactly once, in manifest order.
5. Write raw per-landscape/per-epsilon/per-horizon measurements before summaries.
6. Estimate per-landscape source alphas, then their median.
7. Evaluate and report `H_analytic` and `H_transport` separately.
8. Report special strata, secondary analyses, and stress results without changing gates.
9. Record configuration, approved SHA, file hashes, platform, package versions, and a
   statement that no RNG was used.

The future execution command refuses an existing output path. Raw output is never
overwritten.

## Deviations and stopping

No model, manifest, estimator, threshold, orientation, stratum rule, exclusion, or
analysis code may change after confirmatory outcomes are generated. A necessary change
before execution requires a new public preregistration checkpoint. A discovered defect
after execution is reported as a deviation; the original raw artifact is preserved and
the confirmatory verdict is not silently recomputed.

ACL-002 must not be run until this checkpoint receives a final manifest and derivation
review and its public commit SHA is explicitly approved for execution.
