# ACL-006 derivation

The full pre-outcome derivation is in
[`docs/support_conditioned_bias.md`](../../docs/support_conditioned_bias.md). This file
freezes the pieces used directly by the confirmatory analysis.

## Analytic direction

For interior `pi`, deterministic reward `r`, and context probability `rho`,

\[
F=\rho(\operatorname{diag}(\pi)-\pi\pi^\top),
\qquad
g=\rho\sum_a\pi_ar_a(e_a-\pi).
\]

On the centered tangent,

\[
d=F^+g=r-\operatorname{mean}(r)\mathbf 1.
\]

This comparator is defined without invoking the sampled update operator.

## Exact support law and error decomposition

Let `q_a=rho pi_a` and `q_0=1-rho`. For observed action support `S`,

\[
P(S)=\sum_{T\subseteq S}(-1)^{|S|-|T|}
\left(q_0+\sum_{a\in T}q_a\right)^N.
\]

For three actions,

\[
P(\operatorname{rank}\widehat F<2)
=\sum_a(q_0+q_a)^N-2q_0^N.
\]

Let `P_S` be the analytic-Fisher orthogonal projector onto
`range(hat(F))`. Count table by count table,

\[
\widehat d-d
=-(I-P_S)d+(\widehat d-P_Sd).
\]

The two terms are Fisher-orthogonal, so their squared Fisher errors add. ACL-006
enumerates every multinomial count table to obtain the exact mean, covariance, support
loss, and observed-support perturbation without fitting.

## Standardized finite-mean prediction

For independent shadows with exact covariance `Sigma`,

\[
E\|\bar d_L-m\|_F^2
=\frac{\operatorname{tr}(F\Sigma)}{L}.
\]

This identity defines the zero-fit RMS scale used by the full and half direction scores.
The score thresholds are preregistered deterministic benchmark criteria; no claim of a
specific finite-sample score distribution is made.

## Angular envelope

If `||x-m||_F <= r < ||m||_F`, normalization in the Fisher norm gives

\[
\left\|\frac{x}{\|x\|_F}-\frac{m}{\|m\|_F}\right\|_F
\le \frac{2r}{\|m\|_F-r}.
\]

Therefore the absolute change in cosine with any fixed unit comparator is at most that
same quantity. The registry sets `r` to five exact RMS standard errors and freezes the
resulting envelope.

For two half means, let

\[
a=\frac{2r_{1/2}}{\|m\|_F-r_{1/2}}.
\]

If both lie in their five-score balls, their normalized directions differ by at most
`2a`, hence

\[
\cos_F(\bar d^{(1)},\bar d^{(2)})\ge1-2a^2.
\]

This lower bound determines dissociation membership before outcomes.

## Why self-consistency cannot certify truth

For independent finite-moment shadows with nonzero mean `m`, the strong law gives two
disjoint sample means converging almost surely to `m`. Their Fisher cosine converges to
one, while truth alignment converges to `cos_F(m,d)`, which may be arbitrarily poor.

## Why support alone is insufficient

For `r'=r+c 1`, the analytic centered direction is unchanged, but

\[
E[\widehat d(r')]-E[\widehat d(r)]
=cE[\widehat F^+\bar s],
\qquad
\bar s=N^{-1}\sum_a n_as_a,
\]

which is generally nonzero. Thus `N p_min`, support probabilities, and Fisher spectrum
without reward/baseline geometry cannot determine this estimator's angular bias.
