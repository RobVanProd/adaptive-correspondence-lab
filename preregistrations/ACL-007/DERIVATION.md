# ACL-007 derivation

The exact particle-filter construction is derived in
[`docs/sequential_particle_filter_bias.md`](../../docs/sequential_particle_filter_bias.md).

## Exact target quantities

The finite bootstrap particle filter induces a Markov chain on

\[
\mathcal C_N=\{c\in\mathbb N^3:\mathbf1^\top c=N\}.
\]

For each observation step, grouped transition convolution followed by weighted
multinomial resampling gives an exact count kernel `K_t`. The initial count law is
multinomial under `b_0`, so exact finite matrix propagation gives terminal count
probability `pi_T(c)`. Hence

\[
m=\sum_c\pi_T(c)(c/N-b_0)
\]

and

\[
\Sigma=\sum_c\pi_T(c)
\left(c/N-E[\widehat b_T]\right)
\left(c/N-E[\widehat b_T]\right)^\top.
\]

The true-model comparator is independently computed by exact Bayes recursion and never
calls the particle update.

## Transported standardized score

ACL-006's source score in native metric `G` was

\[
Z_G=\frac{\|\bar v_L-m\|_G}
{\sqrt{\operatorname{tr}(G\Sigma)/L}}.
\]

ACL-007 changes only the declared native metric to centered Euclidean geometry,
`G=I`, yielding

\[
Z_2=\frac{\|\bar d_L-m\|_2}
{\sqrt{\operatorname{tr}(\Sigma)/L}}.
\]

The schedule and every threshold remain numerically unchanged.

## Angular envelope and dissociation

For `||x-m||_2<=r<||m||_2`,

\[
\left\|\frac{x}{\|x\|_2}-\frac{m}{\|m\|_2}\right\|_2
\le\frac{2r}{\|m\|_2-r}.
\]

This bounds the truth-cosine residual. Applying the same argument to both half means
gives the frozen lower bound

\[
\cos_2(\bar d^{(1)},\bar d^{(2)})\ge1-2a^2,
\quad
a=\frac{2r_{1/2}}{\|m\|_2-r_{1/2}}.
\]

These analytic quantities set strata before particle outcomes.

The strong-law converse is metric-agnostic in finite dimension: independent half means
converge to the same nonzero `m`, so their cosine converges to one regardless of
`cos_2(m,d)`. ACL-007 tests the copied finite-budget thresholds in sequential inference,
not merely the asymptotic theorem.
