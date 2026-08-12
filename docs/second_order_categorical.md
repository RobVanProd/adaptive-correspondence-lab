# Second-order categorical mutation sensitivity

This note derives the second epsilon derivative of the ACL-002 row-vector recurrence.
It is a mechanism derivation selected after ACL-002 and is therefore not confirmatory
evidence.

Let

\[
G_\epsilon(q)=F(q)(I+\epsilon B),\qquad B=M-I,
\]

where `F` is the normalized categorical selection map and all states are row vectors.
Write

\[
q_t(\epsilon)=p_t+\epsilon s_t+\frac{\epsilon^2}{2}u_t+O(\epsilon^3).
\]

Taylor expansion of `F` at the clean state gives

\[
F(q_t)=p_{t+1}+\epsilon a_t+
\frac{\epsilon^2}{2}\left(u_tJ_t+H_t\right)+O(\epsilon^3),
\]

with

\[
J_t=J_F^R(p_t),\qquad a_t=s_tJ_t,\qquad
H_t=D^2F(p_t)[s_t,s_t].
\]

Multiplication by `I+epsilon B` yields

\[
s_{t+1}=a_t+p_{t+1}B
\]

and

\[
\boxed{
u_{t+1}=u_tJ_t+H_t+2a_tB
}.
\]

The cross term contains the pre-mutation selected derivative `a_t`, not
`s_{t+1}`. Both derivatives start at zero.

For `F_j(p)=p_jd_j/(p\cdot d)`, where
`d=exp(eta*(r-max(r)))`, the numerator and denominator are linear in `p`.
For a row direction `s`, define

\[
\beta=\frac{s\cdot d}{p\cdot d}.
\]

Then

\[
\boxed{
D^2F(p)[s,s]=-2\beta\,sJ_F^R(p)
}.
\]

## Independent polynomial oracle

The terminal state also equals

\[
q_T(\epsilon)=\operatorname{normalize}
\left[p_0\{D(I+\epsilon B)\}^T\right].
\]

An independent implementation propagates the unnormalized coefficients

\[
v_t(\epsilon)=v_{0,t}+\epsilon v_{1,t}+\epsilon^2v_{2,t}+O(\epsilon^3)
\]

using polynomial multiplication, then differentiates the normalization formula. It
does not call the recurrence above. A separate five-point signed-epsilon stencil tests
both derivatives on toy fixtures.

## L1 nondifferentiability

At positive epsilon,

\[
\|q_T-p_T\|_1=
\sum_i\left|\epsilon s_i+\frac{\epsilon^2}{2}u_i\right|+O(\epsilon^3).
\]

If `s_i` is nonzero, its local quadratic contribution is
`sign(s_i)u_i/2`. If `s_i=0`, absolute value is not differentiable through the usual
sign formula and the leading contribution is `|u_i|/2`. Thus

\[
\boxed{
B_{L1}=\frac12\left[
\sum_{s_i\ne0}\operatorname{sign}(s_i)u_i+
\sum_{s_i=0}|u_i|
\right].
}
\]

The implementation freezes `|s_i| <= 2e-14` as the float64 zero-coordinate branch
and reports every such coordinate. For finite epsilon it evaluates the truncated
vector inside the absolute value directly, so a predicted sign crossing is not hidden
inside a scalar coefficient.
