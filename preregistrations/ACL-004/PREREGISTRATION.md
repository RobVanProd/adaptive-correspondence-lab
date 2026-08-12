# ACL-004 preregistration

## Status

This is a preregistration-only checkpoint. `outcomes_generated` is false and the
analytic registry contains zero shadows. No manifest landscape may be sampled until an
adversarial review approves an exact public Git SHA.

## Claim

ACL-004 tests the restricted finite-population hypothesis

\[
H_{2,\lambda}: E[\Delta\theta_\lambda\mid\theta]
\parallel g_\lambda^{\rm analytic}(\theta)
\]

for a diagonal Gaussian with linear ranking objective. The comparator is derived
independently from conditional rank utility, Gaussian score, and Fisher geometry. It
is not the raw-objective quadratic natural gradient and is not defined by replaying the
rank-mu update.

## Frozen design

- 12 deterministic 3D landscapes `G01` through `G12`;
- parameterization `(mean, log_std)`;
- `lambda=32`, `mu=16`;
- normalized weights `log(mu+0.5)-log(rank)` for ranks `1..mu`;
- mean learning rate `0.2`, covariance learning rate `0.1`;
- independent PCG64 stream per landscape with frozen unique seed;
- no state iteration, evolution paths, CSA, clamps, antithetic sampling, or lambda
  scaling;
- float64 throughout.

The learning rates are recorded, but positive block scaling does not alter the
separate block cosines. Joint cosine is non-gating because different block rates could
otherwise create artificial anisotropy.

## Comparator guard

The base Gauss-Hermite order is `160`, checked against doubled order `320` at relative
tolerance `2e-9` and absolute tolerance `5e-12`. Every analytic mean and covariance
block must have Fisher norm above `1e-12` before outcomes.

## Conditional-mean stopping rule

At cumulative shadow counts `4096,8192,16384,32768,65536`, form the first and second
disjoint-half means. Stop at the first checkpoint where both the mean-block and
covariance-block Fisher cosines between halves are at least `0.98`.

If any landscape fails to converge by `65536`, H2 is `INCONCLUSIVE`. It is not counted
as evidence for or against alignment, and lambda may not be increased after seeing it.

## Primary gate

For stopped estimates, report separate Fisher cosines against the independent analytic
direction. ACL-004 passes only if all landscapes converged and

\[
\min_\ell \cos_F^{\rm mean}(\ell)\ge0.99
\]

and

\[
\min_\ell \cos_F^{\rm covariance}(\ell)\ge0.99.
\]

If convergence succeeds but either minimum is below `0.99`, ACL-004 fails. There is no
fitted coefficient, target correction, landscape exclusion, or joint-cosine rescue.

## H1 descriptive layer

The first 2048 single-shadow cosines per landscape are retained for separate mean and
covariance Q10, median, Q90, and fraction-positive summaries. This practical
finite-lambda signal-to-noise layer is descriptive and cannot alter H2.

## Evidence and guards

Every 2048-shadow chunk retains count, tangent sum, and tangent outer-product sum. The
first H1 cosines are retained individually. Each landscape records its frozen seed and
terminal PCG64 state, and the evidence embeds the frozen manifest. These stored values
must reproduce stopped means and half means exactly.

Execution requires the approved exact Git SHA, completely empty full porcelain status,
valid exact lock and directory membership, exact analytic registry, and the previously
nonexistent sole path
`evidence/ACL-004-confirmatory-{approved_sha}.json`.

The benchmark is deterministic and not a random population sample. ACL-004 tests a
within-Gaussian-class bridge. It does not yet transport a categorical coefficient or
law into the Gaussian class.

After execution, preserve the first artifact exactly. A defect is documented and
tested in a future experiment; ACL-004 is never repaired or rerun in place.
