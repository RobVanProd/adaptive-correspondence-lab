# ACL-005 preregistration

## Status

This is a preregistration-only checkpoint. `outcomes_generated` is false and the
analytic registry contains zero target shadows. No target in this manifest has been
sampled. Execution is forbidden until an adversarial review approves an exact public
Git SHA.

## Scientific question

ACL-005 asks whether the normalized conditional-mean direction law confirmed for the
finite-lambda Gaussian rank-mu system in ACL-004 transports, without target refitting,
to a genuinely different adaptive-system class: finite-state contextual-bandit
natural policy gradient with an empirical-Fisher plug-in estimator.

The hypothesis is deliberately narrower than universal optimization. It predicts
that the source-domain blockwise stopping and analytic-alignment thresholds survive a
change in state space, sampling operator, and estimator nonlinearity.

## Frozen source law

ACL-004 used the checkpoints `4096,8192,16384,32768,65536`. It stopped after both
Gaussian Fisher blocks reached disjoint-half cosine `0.98` and required each analytic
block cosine to reach `0.99`. Its preregistered H2 passed. ACL-005 copies that schedule,
stopping threshold, and analytic threshold unchanged; the immutable source hashes and
source minima are frozen in `manifest.json`.

## Frozen target design

- two contexts and three actions;
- deterministic rewards and centered categorical logits;
- one frozen policy state per landscape, with no trajectory or policy update;
- `N=128` independent joint context-action interactions per shadow;
- empirical score gradient and empirical score Fisher;
- undamped Moore-Penrose inverse with `rcond=1e-12`;
- independent PCG64 stream and unique seed per landscape;
- 10 regular targets (`R01`--`R10`) and four non-gating stress targets
  (`S01`--`S04`);
- float64 throughout.

Regular targets are predeclared by minimum expected joint-cell count at least 4.
Stress targets have minimum expected joint-cell count at most 0.75 and deliberately
probe rare-context or rare-action singularity. These strata use analytic manifest
quantities only.

## Primary hypothesis and verdict

At the first checkpoint where every context's disjoint-half Fisher cosine is at least
`0.98`, each regular context's stopped-mean Fisher cosine with the independently exact
NPG direction will be at least `0.99`.

PASS requires convergence of all 10 regular targets and a minimum of at least `0.99`
over all 20 context blocks. Successful convergence with any lower block is FAIL. Any
regular nonconvergence by 65536 is INCONCLUSIVE. Joint cosine and stress targets are
non-gating.

## Scope and content

This is a cross-class, no-target-refit prediction of a dimensionless normalized law.
A PASS would support reuse of this blockwise expected-direction diagnostic across the
two studied classes. A FAIL would locate a boundary caused by empirical-Fisher
inversion or control sampling. An INCONCLUSIVE result would show that the copied
estimation budget cannot resolve the target law.

The result cannot establish a shared stability coefficient, sequential-RL behavior,
neural-policy behavior, lambda/sample-count scaling, or a universal adaptive process.
The target landscapes are a deterministic held-out benchmark, not a random population
sample.

## One-shot rule

Execution requires the exact approved public SHA, a completely clean full porcelain
status, the canonical in-repository `preregistrations/ACL-005` bundle, valid exact
bundle hashes and membership, a matching analytic registry, exact SHA-256 matches for
the frozen ACL-004 evidence artifact and report, and a previously nonexistent path
`evidence/ACL-005-confirmatory-{approved_sha}.json`. The first artifact is immutable.
