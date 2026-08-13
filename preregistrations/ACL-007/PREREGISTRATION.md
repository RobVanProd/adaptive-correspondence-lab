# ACL-007 preregistration

## Status

This is a preregistration-only checkpoint. `outcomes_generated` is false, the analytic
registry has `shadow_count: 0`, and no labeled target particle has been drawn. Execution
is forbidden until an adversarial review approves an exact public Git SHA.

## Scientific question

ACL-007 asks whether a complete dimensionless diagnostic confirmed in ACL-006 transports
without refitting from an empirical-Fisher natural-gradient estimator into a genuinely
different adaptive class: sequential Bayesian inference by a finite bootstrap particle
filter in centered Euclidean belief geometry.

The transported claim is not that both estimators have the same bias mechanism. It is:

> Once the estimator expectation and covariance are independently fixed, ACL-006's
> unchanged native-metric standardized-mean and split-consistency/truth-dissociation
> rule predicts a sequential particle-filter benchmark without target refitting.

## Frozen source rule

ACL-006 used checkpoints `8192,32768,131072,262144`, chunk size `4096`, a maximum full
or half standardized direction score of `5`, Type-7 full-score median `1.5`, and Q90
`2.5`. Its dissociation stratum required exact truth cosine at most `0.90`, five-score
upper bound at most `0.95`, and a two-half lower bound at least `0.995`; realized halves
then had to reach `0.995` while truth remained at most `0.95`. Its resolvable contrast
gap was `0.10`.

ACL-007 copies every value unchanged. The source artifact/report hashes and source
verdicts are in `manifest.json`. No target observation can modify a threshold,
replication count, metric, target, or exclusion.

## Target class and exact comparator

The true three-state hidden Markov model has initial belief `b_0`, row-stochastic
transition `P`, and three frozen positive observation-likelihood vectors. Exact Bayes
filtering defines terminal belief `b_T` and ideal update

\[
d=b_T-b_0.
\]

Each finite particle filter uses an explicitly frozen approximate transition and
likelihood sequence. Its realized update is

\[
\widehat d=\widehat b_T-b_0.
\]

The exact finite count-state Markov law freezes `m=E[hat(d)]`, covariance `Sigma`,
terminal support probabilities, and Euclidean truth cosine before outcomes. The target
native metric is the ordinary Euclidean metric on the centered belief tangent; no
Fisher matrix, reward, gradient, or natural-gradient update appears.

## Frozen target grid

Sixteen deterministic filters use two entirely new true HMMs.

- Model A: correct filters at `N=3,4,6,8`; reversed-observation filters at the same
  counts; state-independent flat-observation filters at `N=4,8`; and missing-final-
  observation filters at `N=4,8`.
- Model B: correct, reversed-observation, flat-observation, and missing-final-observation
  filters at `N=4`. These have deliberately less favorable mean/variance geometry and
  prevent the benchmark from containing only easy asymptotic cases.

Exact pre-outcome quantities place `A06,A07,A08` in the dissociation stratum. All nine
frozen contrasts are resolvable after subtracting both five-score angular envelopes.

## Independent sampled path

The confirmatory path simulates labeled particles. Every transition and every
resampled particle uses an individual inverse-CDF uniform. It does not call the exact
count-state kernel and does not draw count-level multinomials. Independent PCG64 streams
are frozen per target.

## Verdicts

The standardized-mean component passes only if every target's full and two half scores
is at most `5`, the Type-7 full-score median is at most `1.5`, Q90 at most `2.5`, and
all truth-cosine residuals lie within their frozen five-score envelopes.

The dissociation component passes only if all three locked members end with half cosine
at least `0.995` and truth cosine at most `0.95`.

Every resolvable contrast must reproduce the exact signed truth-cosine gap and retain
absolute observed gap at least `0.10`.

Overall transport PASS requires all three components to pass. A nonempty dissociation
stratum is mandatory. There is no outcome-dependent stopping or exclusion.

## Scope and meaning

A PASS supports reuse of one stochastic adaptive-estimator diagnostic across a change
in estimator family, native metric, optimization-versus-inference semantics, and
temporal structure. It does not show that the detailed source and target biases are the
same or that all adaptive systems share one law. Exact target moments are theorem/
software reproduction; only the no-refit diagnostic is cross-class evidence.

A FAIL narrows the diagnostic to its source island or identifies a sequential/non-
Gaussian boundary. Both outcomes are useful.

## Forbidden behavior

- no target particle before exact public-SHA approval;
- no source threshold or schedule change;
- no target refit, early stop, budget increase, or outcome exclusion;
- no Fisher metric substituted for the frozen Euclidean target metric;
- no exact-kernel call from the sampled path;
- no rerun, overwrite, regeneration, or replacement after first execution.
