# ACL-002 row-vector sensitivity derivation

## Exact categorical map

Let `a_j = exp(eta (r_j - max(r)))`, `Z(p) = sum_k p_k a_k`, and

```text
F_j(p) = p_j a_j / Z(p).
```

Subtracting the reward maximum is only a numerical gauge; it does not change `F`.
Probabilities are row vectors, but components are written with lower indices.

## Row Jacobian

Differentiate output component `j` with respect to input component `i` in the ambient
coordinates:

```text
partial F_j / partial p_i
  = indicator(i=j) a_j/Z - p_j a_j a_i/Z^2
  = indicator(i=j) a_j/Z - (a_i/Z) F_j.
```

Define the row Jacobian with input index first and output index second:

```text
(J_F^R)_[i,j] = partial F_j / partial p_i
J_F^R = diag(a/Z) - outer(a/Z, F(p)).
```

Thus a row perturbation `h` propagates as `h J_F^R`. In conventional column notation
the Jacobian would be `(J_F^R)^T`; ACL-002 never uses that implicit transpose.

## Mutation sensitivity

Define

```text
G_epsilon(q) = (1-epsilon) F(q) + epsilon F(q) M
               = F(q) + epsilon F(q)(M-I).
```

Let `q_t(epsilon)` start from `q_0=p_0` and define the row sensitivity

```text
s_t = d q_t(epsilon) / d epsilon evaluated at epsilon=0.
```

At epsilon zero, `q_t(0)=p_t`. Applying the chain rule gives

```text
s_(t+1) = s_t J_F^R(p_t) + p_(t+1)(M-I),
s_0 = 0.
```

Because `F` maps the simplex to itself and `M` is row-stochastic, every sensitivity
sums to zero. The implementation treats failure of this tangent-mass invariant beyond
the frozen float64 tolerance as an error.

## L1 coefficient

For nonnegative epsilon,

```text
q_T(epsilon) = p_T + epsilon s_T + O(epsilon^2),
```

so continuity and positive homogeneity of the L1 norm give

```text
||q_T(epsilon)-p_T||_1
  = epsilon ||s_T||_1 + O(epsilon^2).
```

Therefore `C_analytic = ||s_T||_1`. At component sign changes or `C=0`, the one-sided
statement remains the preregistered object; special sensitivity strata avoid unstable
relative errors.

## Oriented KL coefficient

ACL-002 uses `D_KL(q_T || p_T)`. Since clean states remain strictly interior and
`sum_i s_i=0`, Taylor expansion around `q=p` yields

```text
D_KL(p + epsilon s || p)
  = sum_i (p_i+epsilon s_i) log((p_i+epsilon s_i)/p_i)
  = (epsilon^2/2) sum_i s_i^2/p_i + O(epsilon^3).
```

The linear term vanishes by tangent mass conservation. Hence

```text
K_analytic = 0.5 sum_i s_i^2/p_i.
```

The reverse KL shares this quadratic coefficient but differs at higher order and is not
an ACL-002 metric.
