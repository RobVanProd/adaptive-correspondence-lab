# ACL-004 frozen derivation

The full independent derivation is in
[`docs/gaussian_rank_mu_bridge.md`](../../docs/gaussian_rank_mu_bridge.md).

For `X=m+sigma*z`, linear ranking axis

\[
v=\frac{a\odot\sigma}{\|a\odot\sigma\|_2},
\]

and frozen conditional rank utility `h_lambda(t)`, ACL-004 predicts

\[
g_{m,\lambda}=\lambda(\sigma\odot v)E[h_\lambda(t)t]
\]

and

\[
g_{\log\sigma,\lambda}=\frac{\lambda}{2}(v\odot v)
E[h_\lambda(t)(t^2-1)].
\]

These expressions follow from the Gaussian score, inverse Fisher blocks, and the
conditional binomial rank law. The analytic implementation evaluates the two scalar
expectations directly. It never samples a rank-mu population and never calls the
sampled update implementation.

ACL-004 fixes `lambda=32`, `mu=16`, logarithmic normalized positive rank weights,
Gauss-Hermite order `160`, and a doubled-order `320` oracle. Mean and covariance
Fisher cosines are separate primary quantities.
