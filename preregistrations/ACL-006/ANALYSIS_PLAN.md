# ACL-006 analysis plan

## Frozen sequence

1. Abort before RNG use unless the approved SHA, clean full porcelain state, canonical
   bundle path, canonical nonexistent evidence path, exact lock membership, pinned
   environment, and analytic registry validation all pass.
2. Recompute the exact registry and compare structure exactly and floats with absolute
   and relative tolerance `2e-12`.
3. For every target, generate all fixed checkpoints `8192,32768,131072,262144`; there is
   no stopping decision.
4. Store per-chunk counts, direction sums, outer-product sums, support-mask counts, and
   terminal PCG64 states. Reconstruct final means from those sums before analysis.
5. Apply the exact-mean, dissociation, and contrast gates separately and write one
   previously nonexistent SHA-derived artifact.

## Exact-mean prediction

For each target and for its two final disjoint halves, compute

\[
z_F=\frac{\|\bar d-m\|_F}
{\sqrt{\operatorname{tr}(F\Sigma)/L}}.
\]

PASS requires all three scores per target at most `5`, plus Type-7 median at most `1.5`
and Type-7 Q90 at most `2.5` across the 16 full-mean scores. It also requires

\[
|\cos_F(\bar d,d)-\cos_F(m,d)|
\]

not to exceed the target's locked five-score angular envelope. Any violation is FAIL;
split-half agreement cannot rescue it.

## Dissociation prediction

Membership is read only from the locked registry. PASS requires every member's final
half cosine at least `0.995` and final truth cosine at most `0.95`. Any violation is
FAIL. An empty locked stratum is INVALID, not a scientific pass, for the real ACL-006
design.

Checkpoint half and truth cosines are reported as the finite-replication approach to
the asymptotic limits. Checkpoints do not gate separately.

## Contrast reproduction

Only registry contrasts marked `resolvable` gate. For each, the observed signed truth-
cosine gap must match the exact sign and its absolute size must remain at least `0.10`.
All marked contrasts must pass. Unmarked contrasts are descriptive and cannot change a
verdict.

The exact matched-effective-count and reward-shift counterexamples are recorded as
falsifications of the corresponding scalar/support-only reductions. They are not
target regressions and no coefficient is fit.

## Reporting and exclusions

Report all 16 targets, every checkpoint, exact/observed means and cosines, standardized
scores, support-mask counts, decomposition quantities, contrast rows, and all separate
verdicts. There are no outcome exclusions, fake replicates, target refits, population
confidence intervals, joint metrics, or post-outcome threshold changes.

The overall interpretation is mechanism validation inside one finite estimator family.
Even a full PASS is not independent breadth evidence.
