# Finite-sample contextual-bandit NPG bridge

For context distribution `rho(c)`, categorical policy `pi(a|c)`, deterministic reward
`r(c,a)`, and centered logit parameters, the exact policy gradient block is

\[
g_c=\rho(c)\,\pi_c\odot(r_c-V_c),
\qquad V_c=\pi_c^T r_c.
\]

The analytic Fisher block is

\[
F_c=\rho(c)[\operatorname{diag}(\pi_c)-\pi_c\pi_c^T].
\]

In the centered-logit gauge, an exact natural direction is

\[
d_c=(r_c-V_c)-\operatorname{mean}_a(r_c-V_c).
\]

It satisfies `F_c d_c = g_c`.

## Finite-sample plug-in estimator

One shadow draws a fixed number `N` of independent joint context-action observations.
For each observation, the score is zero outside the observed context and

\[
s_{c,a}=e_a-\pi_c
\]

inside it. The plug-in estimator forms

\[
\widehat g=\frac1N\sum_n r_n s_n,
\qquad
\widehat F=\frac1N\sum_n s_ns_n^T,
\]

and returns the centered direction

\[
\widehat d=\operatorname{center}(\widehat F^+\widehat g),
\]

using an undamped Moore-Penrose pseudoinverse with frozen `rcond`.

Although `E[hat g]=g` and `E[hat F]=F`, in general

\[
E[\widehat F^+\widehat g]\ne F^+g.
\]

That nonlinear finite-sample inversion is the mechanism capable of falsifying
expected-direction transport from the Gaussian source class. Missing context blocks
remain explicit zero blocks; no damping or silent repair is permitted.

## Geometry and transport

Each context is a primary categorical-Fisher block. A joint Fisher cosine can hide a
poorly estimated context and is secondary. The eligible transported law from ACL-004
is dimensionless: after disjoint-half block convergence at `0.98`, every target block's
conditional-mean Fisher cosine with the independently analytic direction must be at
least `0.99`. Those values may not be tuned on control outcomes.
