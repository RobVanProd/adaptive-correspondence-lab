# ACL-006 preregistration

## Status

This is a preregistration-only checkpoint. `outcomes_generated` is false. The analytic
registry contains exact finite multinomial quantities and zero sampled target shadows.
Execution is forbidden until an adversarial review approves an exact public Git SHA.

## Scientific question

ACL-006 asks whether the finite-sample bias mechanism exposed by ACL-005 can be turned
into a zero-fit support-conditioned prediction for the original undamped empirical-
Fisher plug-in estimator. It also tests the logically separate prediction that two
independent estimator means can become nearly collinear with each other while their
common expectation remains materially misaligned with the analytic natural direction.

The experiment does not ask whether sparse data are generically harmful. It asks
whether an exact finite law predicts the direction and whether the simpler proposed
reductions—especially `N p_min` alone and support/Fisher spectrum without reward
geometry—survive controlled counterexamples.

## Frozen estimator and mapping

For one context block with action policy `pi`, context probability `rho`, deterministic
reward `r`, and `N` joint samples, define true-policy score rows

\[
s_a=e_a-\pi.
\]

The ideal objects are

\[
F=\rho\sum_a\pi_as_as_a^\top,
\qquad
g=\rho\sum_a\pi_ar_as_a,
\qquad
d=F^+g.
\]

The realized estimator uses the same count table for both empirical objects:

\[
\widehat F=N^{-1}\sum_a n_as_as_a^\top,
\qquad
\widehat g=N^{-1}\sum_a n_ar_as_a,
\qquad
\widehat d=\widehat F^+\widehat g.
\]

All directions use the centered-logit tangent. The pseudoinverse uses `rcond=1e-12`.
There is no damping, fitted baseline, clipping, policy update, or target refit.

## Zero-fit prediction

For every manifest target, all four-cell multinomial count tables are enumerated before
outcomes to freeze

\[
m=E[\widehat d],
\qquad
\Sigma=\operatorname{Cov}(\widehat d),
\qquad
\cos_F(m,d).
\]

The sampled path is independent computationally: PCG64 multinomial draws, direct
three-coordinate Fisher/gradient construction, and a Hermitian pseudoinverse. It does
not call the exact tangent-coordinate enumerator.

At the final fixed replication count `L=262144`, define

\[
z_F=\frac{\|\bar d_L-m\|_F}
{\sqrt{\operatorname{tr}(F\Sigma)/L}}.
\]

The same score is calculated for both disjoint halves with `L/2` in the denominator.
The primary exact-mean prediction passes only if every full and half score is at most
`5`, the Type-7 median of full scores is at most `1.5`, the Type-7 Q90 is at most `2.5`,
and every observed truth cosine lies within its locked five-score angular envelope.
These are deterministic benchmark criteria, not confidence statements.

## Self-consistency dissociation

A target enters the dissociation stratum before RNG use only if its exact truth cosine
is at most `0.90`, its locked full-mean angular upper bound is at most `0.95`, and the
geometric lower bound on two half-mean cosines under the five-score balls is at least
`0.995`.

The dissociation prediction passes only if every such target ends with

\[
\cos_F(\bar d^{(1)},\bar d^{(2)})\ge0.995
\]

and simultaneously

\[
\cos_F(\bar d,d)\le0.95.
\]

Split-half agreement cannot rescue a failed exact-mean gate or truth alignment.

## Frozen contrasts

The 16 targets use new values and comprise:

- four pairs with equal `N`, equal minimum joint-cell probability, and hence equal
  `N p_min`, but rare-context versus rare-action support factorization;
- three additive reward-shift pairs with identical policy, support, Fisher geometry,
  and analytic centered tangent;
- two pairs varying positive Fisher conditioning and reward orientation at fixed `N`,
  `rho`, minimum action probability, and reward.

A contrast gates stochastic reproduction only when its exact cosine gap minus both
locked angular envelopes is at least `0.10`. It must reproduce the predicted sign and
retain an observed gap of at least `0.10`. Other contrasts are reported but non-gating.

## Scope and interpretation

A PASS confirms the exact finite law through an independent sampled implementation and
validates the predicted self-consistency/truth dissociation. It is primarily theorem and
software reproduction plus a mapped mechanism boundary; it is not new independent
evidence for a broad adaptive-system unification.

The exact reward-shift and matched-effective-count counterexamples falsify support-only
and one-parameter `N p_min` versions within this estimator family. They do not establish
that no useful coarser bound exists with richer reward/support observables.

## Forbidden behavior

- no target shadow before public exact-SHA approval;
- no damping or fitted baseline;
- no early stopping or budget increase;
- no outcome-dependent exclusion, threshold, target, or contrast change;
- no external or self-consistent substitute preregistration bundle;
- no rerun, overwrite, regeneration, or replacement after the first execution.

PASS, FAIL, INVALID, and INCONCLUSIVE remain distinct. A post-execution defect preserves
the first artifact and requires a new experiment rather than changing ACL-006.
