# ACL-005 frozen derivation

The implementation-level derivation and count-table oracle are in
[`docs/control_finite_sample_bridge.md`](../../docs/control_finite_sample_bridge.md).

For context `c`, policy `pi_c`, reward vector `r_c`, and context probability `rho_c`,
the exact logit score is

\[
z_{c,a}=e_a-\pi_c.
\]

The analytic policy-gradient and Fisher block are

\[
g_c=\rho_c\sum_a\pi_{c,a}r_{c,a}z_{c,a},\qquad
F_c=\rho_c\left(\operatorname{diag}(\pi_c)-\pi_c\pi_c^\top\right).
\]

On the centered-logit tangent space, the exact comparator is the unique centered
solution of `F_c d_c = g_c`, equivalently

\[
d_c=(r_c-\langle\pi_c,r_c\rangle\mathbf 1)
-\operatorname{mean}(r_c-\langle\pi_c,r_c\rangle\mathbf 1)\mathbf 1.
\]

One sampled shadow draws a multinomial joint context-action count table `n_ca` from
`N=128` interactions, constructs

\[
\widehat g_c=N^{-1}\sum_a n_{c,a}r_{c,a}z_{c,a},\qquad
\widehat F_c=N^{-1}\sum_a n_{c,a}z_{c,a}z_{c,a}^\top,
\]

and returns the centered Moore-Penrose solution

\[
\widehat d_c=\operatorname{center}
\left(\widehat F_c^+\widehat g_c\right)
\]

with `rcond=1e-12`. No damping, clipping, baseline fit, state update, or exact
comparator call occurs in the sampled implementation. Because pseudoinversion is
nonlinear, ACL-005 does not assume `E[hat(F)^+ hat(g)] = F^+ g`; alignment of their
directions is the hypothesis under test.

The primary geometry is blockwise:

\[
\cos_{F_c}(x,y)=\frac{x^\top F_c y}
{\sqrt{x^\top F_c x}\sqrt{y^\top F_c y}}.
\]

The context-probability-weighted joint cosine is secondary and cannot rescue a failed
context. The transported quantity is the complete ACL-004 normalized rule: once every
disjoint-half block cosine is at least `0.98`, every regular target block must have
analytic cosine at least `0.99`. The thresholds, estimator schedule, and stopping rule
are copied without target fitting.
