# ACL-002 exploratory second-order mechanism evaluation

## Status

This is a post-confirmatory mechanism analysis. It uses the immutable ACL-002 rows and frozen manifest, generates no trajectories, fits no target coefficient, and cannot change the ACL-002 verdict.

- Source artifact SHA-256: `4d08e85b927a5d78a29078ff0d6549225d98069b20186b754629464739f29d74`
- Analysis-code commit: `071373b9bc4dd9369f82f3bda9bb20363d3eabda`
- Independent oracle maximum state error: `4.441e-16`
- Independent oracle maximum first-derivative error: `1.137e-13`
- Independent oracle maximum second-derivative error: `5.821e-11`

## Derivation

With `B=M-I`, `a_t=s_t J_F^R(p_t)`, and `q_t=p_t+epsilon*s_t+(epsilon^2/2)*u_t+O(epsilon^3)`, direct differentiation gives:

```text
u_{t+1} = u_t J_F^R(p_t) + D^2F(p_t)[s_t,s_t] + 2 a_t B.
```

The full derivation and the explicit nondifferentiable L1 zero-coordinate branch are in `docs/second_order_categorical.md`.

## Exploratory stored-row comparison

Target absolute relative errors at `T=20`:

| Region | First median | Second median | First Q90 | Second Q90 |
| --- | ---: | ---: | ---: | ---: |
| confirmatory | 0.1545% | 0.0001% | 0.8930% | 0.0050% |
| extended-local | 2.6260% | 0.0453% | 10.9400% | 0.9441% |
| stress | 22.6395% | 5.4893% | 62.1652% | 77.6553% |

## Frozen ACL-003 earning rule

Overall result: **PASS**. This is only a decision about whether a new-landscape preregistration is informative; it is not confirmation of the second-order hypothesis.

| Check | Result |
| --- | --- |
| `independent_oracles` | PASS |
| `radius_5_percent` | PASS |
| `radius_10_percent` | PASS |
| `epsilon_0.01_median_error_reduction` | PASS |
| `strict_q90_nonworsening` | PASS |

The local improvement does not extend uniformly into stress. At `T=20`, stress Q90 worsens from 62.17% at first order to 77.66% at second order, and the worst second-order relative error is 778.42%. This is a mapped finite-truncation failure boundary, not a reason to suppress the local result.

## Interpretation

The earning rule passed, so the mechanism has earned an ACL-003 preregistration on entirely new categorical catalog values.
The ACL-002 improvement remains exploratory because the mechanism was selected after seeing ACL-002 residuals.
