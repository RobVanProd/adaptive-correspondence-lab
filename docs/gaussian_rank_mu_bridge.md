# Finite-lambda Gaussian rank-mu comparator

This derivation defines the Gaussian bridge comparator independently of the sampled
rank-mu update.

Let

\[
X=m+\sigma\odot Z,\qquad Z\sim\mathcal N(0,I),
\]

with parameters `(m, log sigma)`, and rank samples by the nonzero linear objective
`a dot X`. Define the standardized objective axis

\[
v=\frac{a\odot\sigma}{\|a\odot\sigma\|_2},
\qquad t=v^T Z.
\]

For population size `lambda`, let the normalized positive weight assigned to rank
`r <= mu` be `w_r`; all other ranks receive zero. Conditional on one sample's `t`,
the number of the other `lambda-1` samples with a better objective is

\[
B\mid t\sim\operatorname{Binomial}(\lambda-1,1-\Phi(t)).
\]

Therefore its conditional expected assigned weight is

\[
h_\lambda(t)=
\sum_{k=0}^{\mu-1}
w_{k+1}{\lambda-1\choose k}
[1-\Phi(t)]^k\Phi(t)^{\lambda-1-k}.
\]

This is an order-statistic calculation, not a replay of the update.

## Score and Fisher map

The Gaussian scores and inverse-Fisher natural scores are

\[
\nabla_m\log p(X)=Z\oslash\sigma,
\quad
F_m^{-1}\nabla_m\log p(X)=\sigma\odot Z,
\]

and

\[
\nabla_{\log\sigma}\log p(X)=Z^2-1,
\quad
F_{\log\sigma}^{-1}\nabla_{\log\sigma}\log p(X)
=\frac12(Z^2-1).
\]

Exchangeability multiplies the single-sample expectation by `lambda`. Gaussian
conditioning gives

\[
E[Z\mid t]=vt,
\]

and, coordinatewise,

\[
E[Z_i^2-1\mid t]=v_i^2(t^2-1).
\]

Thus the independent finite-population comparator is

\[
\boxed{
g_{m,\lambda}^{\rm analytic}
=\lambda(\sigma\odot v)E[h_\lambda(t)t]
}
\]

and

\[
\boxed{
g_{\log\sigma,\lambda}^{\rm analytic}
=\frac{\lambda}{2}(v\odot v)
E[h_\lambda(t)(t^2-1)].
}
\]

The implementation evaluates only these one-dimensional expectations using
Gauss-Hermite quadrature. A separate tensor-product Gaussian score integral verifies
the result on toy fixtures.

## Scientific scope

This identity is specific to a diagonal Gaussian, linear ranking objective, frozen
finite population and rank weights, and the `(mean, log sigma)` parameterization. It
does not imply that a quadratic-objective natural gradient, an infinite-population
limit, CMA-ES with evolution paths, or another optimizer shares the same direction.

Mean and covariance/log-scale directions are evaluated separately in their Fisher
blocks. A joint cosine can hide block disagreement and is not a primary quantity.
