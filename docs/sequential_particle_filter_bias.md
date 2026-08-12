# Sequential Bayesian particle-filter bias

## Structurally distinct target class

ACL-007 moves from reward optimization and Fisher-natural estimators to sequential
belief inference. The state is a three-state posterior belief, the native comparison
metric is ordinary Euclidean geometry on the sum-zero belief tangent, and the realized
estimator is a finite bootstrap particle filter. No reward, objective gradient, Fisher
matrix, or natural-gradient update appears.

For initial belief `b_0`, true row-stochastic transition `P`, and a fixed sequence of
strictly positive observation likelihood vectors `l_t`, exact Bayes filtering is

\[
b_t=\operatorname{normalize}\left[(b_{t-1}P)\odot l_t\right].
\]

The ideal adaptive displacement is

\[
d=b_T-b_0.
\]

The particle filter may use a separately declared approximate transition `tilde(P)`
and likelihoods `tilde(l)_t`. Its terminal empirical belief and displacement are

\[
\widehat b_T=N^{-1}C_T,
\qquad
\widehat d=\widehat b_T-b_0.
\]

Thus finite-particle noise and structural model misspecification are explicit and
separable. Repeated shadows estimate the expectation of the declared approximate
filter; they do not change the true-model comparator.

## Exact count-state law

Let

\[
\mathcal C_N=\{c\in\mathbb N^3:\mathbf1^\top c=N\}.
\]

For source count `c`, particles in source state `i` independently transition according
to row `tilde(P)_{i:}`. The exact predicted-count distribution is the convolution

\[
T_c(z)=\prod_{i=1}^3
\left(\sum_{j=1}^3\widetilde P_{ij}z_j\right)^{c_i}.
\]

The coefficient of `z^u` is the probability of predicted count `u`. Conditional on
`u` and observation likelihood `tilde(l)_t`, multinomial resampling has probabilities

\[
w_j(u,t)=
\frac{u_j\widetilde l_{t,j}}
{\sum_k u_k\widetilde l_{t,k}}.
\]

Therefore the finite count-state transition kernel is

\[
K_t(c,c')=
\sum_{u\in\mathcal C_N}
P(u\mid c,\widetilde P)
\operatorname{Multinomial}(c';N,w(u,t)).
\]

Starting from the exact multinomial initial-count law, multiplication by the finite
kernels gives the complete terminal distribution. It yields the zero-fit quantities

\[
m=E[\widehat d],
\qquad
\Sigma=\operatorname{Cov}(\widehat d),
\qquad
\cos_2(m,d)
\]

and exact terminal support-loss probabilities.

## Independent simulation path

The sampled implementation uses labeled particles. Each particle consumes its own
transition uniform; each resampled particle consumes its own inverse-CDF uniform over
the labeled weighted population. This path never calls the count-state kernel and never
draws a count-level multinomial. Small fixtures compare it with brute-force labeled
path enumeration and the exact count oracle.

## Transported dimensionless law

ACL-006 established a metric-normalized stochastic-mean diagnostic. In abstract form,
for a native positive metric `G`, estimator expectation `m`, covariance `Sigma`, and
`L` independent shadows,

\[
Z_G=\frac{\|\bar v_L-m\|_G}
{\sqrt{\operatorname{tr}(G\Sigma)/L}}.
\]

It also confirmed the separate asymptotic statement that split-half cosine converges to
one around nonzero `m` even when `m` is biased relative to `d`. ACL-007 proposes to copy
the entire ACL-006 schedule and thresholds without target refitting, substituting only
the target class's native centered Euclidean metric `G=I`.

This does not assert that particle-filter support bias has the same detailed mechanism
as empirical-Fisher inversion. The hypothesis is narrower: the standardized mean-error
diagnostic and the variance-versus-bias dissociation survive a change in estimator,
geometry, semantics, and temporal structure.

## Failure meaning

A failure can mean that the ACL-006 finite benchmark thresholds do not transport under
sequential dependence or a changed estimator distribution. A pass is evidence for a
restricted reusable stochastic-adaptation diagnostic, not for a universal adaptive
process. The particle-filter exact count law itself remains theorem/software
reproduction; the hypothesis-bearing content is the no-refit cross-class rule.
