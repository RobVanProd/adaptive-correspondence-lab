# Support-conditioned bias of an empirical-Fisher natural gradient

## Setup

Consider one three-action contextual-bandit block with interior policy `pi`, context
probability `rho`, deterministic reward vector `r`, and `N` joint observations. Let

\[
s_a=e_a-\pi,
\qquad
F=\rho\sum_a\pi_a s_as_a^\top,
\qquad
g=\rho\sum_a\pi_a r_a s_a.
\]

On the centered-logit tangent, the analytic direction is

\[
d=F^+g=r-\operatorname{mean}(r)\mathbf 1.
\]

For action counts `n_a` in this context,

\[
\widehat F=\frac1N\sum_a n_a s_as_a^\top,
\qquad
\widehat g=\frac1N\sum_a n_a r_a s_a,
\qquad
\widehat d=\widehat F^+\widehat g.
\]

All quantities use the same true-policy scores. No damping, baseline fit, clipping, or
state update is present.

## Exact support probabilities

Let `q_a=rho pi_a`, `q_0=1-rho`, and let `S` be the set of actions observed at least
once. Inclusion-exclusion gives

\[
P(S)=\sum_{T\subseteq S}(-1)^{|S|-|T|}
\left(q_0+\sum_{a\in T}q_a\right)^N.
\]

In particular,

\[
P(n_a=0)=(1-q_a)^N,
\]

not merely the approximation `exp(-N q_a)`. For three interior action scores, any two
distinct observed actions span the two-dimensional centered tangent. Therefore

\[
P(\operatorname{rank}\widehat F<2)
=\sum_a(q_0+q_a)^N-2q_0^N.
\]

The implementation checks these formulas against the probability mass accumulated by
exhaustive four-cell multinomial enumeration.

## Orthogonal support decomposition

For support pattern `S`, let `U_S=range(hat(F))` and let `P_S` be the orthogonal
projector onto `U_S` under the analytic Fisher inner product. Since `hat(d)` lies in
`U_S`, every count table satisfies the identity

\[
\widehat d-d
=
-\left(I-P_S\right)d
+
\left(\widehat d-P_Sd\right).
\]

The first term is missing-identifiable-subspace error. The second is observed-support
perturbation. They are Fisher-orthogonal for each count table, so

\[
\|\widehat d-d\|_F^2
=
\|(I-P_S)d\|_F^2
+
\|\widehat d-P_Sd\|_F^2.
\]

Averaging yields an exact bias-vector decomposition and an exact expected squared-error
decomposition. Support loss vanishes whenever at least two actions are observed, but
the observed-support term need not vanish even then.

## Full-rank perturbation relation

Choose tangent coordinates and whiten by `F`. On a full-rank empirical event, define

\[
A=F^{-1/2}\widehat F F^{-1/2},
\qquad
e=F^{-1/2}(\widehat g-\widehat Fd),
\qquad
z=F^{1/2}d.
\]

Then the empirical direction obeys the exact relation

\[
\widehat z-z=A^{-1}e.
\]

If `||A-I||_2 <= gamma < 1`, the usual inverse perturbation bound gives

\[
\|\widehat z-z\|_2\le\frac{\|e\|_2}{1-\gamma}.
\]

This identifies Fisher conditioning and the score/reward residual as the second
mechanism. A useful nonasymptotic bound still requires control of the complement event;
ACL-006 does not assume that a one-parameter `N p_min` bound exists.

## Reward-shift counterexample to support-only laws

For `r'=r+c 1`, the analytic direction is invariant:

\[
d(r')=d(r).
\]

The empirical gradient is not invariant:

\[
\widehat g(r')=\widehat g(r)+c\bar s,
\qquad
\bar s=\frac1N\sum_a n_a s_a.
\]

Consequently,

\[
E[\widehat d(r')]-E[\widehat d(r)]
=cE[\widehat F^+\bar s],
\]

which is generally nonzero. Two systems can therefore have identical `N`, `p_min`,
support-pattern probabilities, Fisher spectrum, and analytic tangent yet different
finite-sample angular bias. Reward/baseline geometry is necessary in any predictive
law for this estimator.

## Self-consistency converse

Let independent shadow directions have finite first moment and nonzero expectation
`m`. By the strong law of large numbers, disjoint sample means satisfy

\[
\bar d_n^{(1)}\to m,
\qquad
\bar d_n^{(2)}\to m
\quad\text{almost surely}.
\]

Continuity of Fisher cosine away from zero gives

\[
\cos_F(\bar d_n^{(1)},\bar d_n^{(2)})\to1,
\]

while

\[
\cos_F(\bar d_n,d)\to\cos_F(m,d),
\]

which may be arbitrarily poor. The deterministic estimator `hat(d)=m != d` is an
immediate converse example: every split is perfectly consistent for every sample size
and truth alignment still fails.

Thus no threshold using estimator-only split-half alignment can certify truth alignment
over an unrestricted biased-estimator class. A certificate needs an independently
defined comparator or a valid bias bound using additional observables such as support,
empirical rank/spectrum, and reward geometry.

## Exact finite-system prediction

For this three-action system the conditional mean and covariance can be computed
without fitting:

\[
m(N,q,r)=
\sum_{n_0+n_1+n_2+n_3=N}
\operatorname{Multinomial}(n;N,q_0,q_1,q_2,q_3)\,\widehat d(n).
\]

This finite sum is the strongest available zero-fit support-conditioned prediction.
ACL-006 will test it through an independent stochastic path and will separately test
whether simpler effective-count reductions survive equal-`N p_min` and reward-shift
contrasts.
