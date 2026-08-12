# ACL-003 frozen derivation

The complete derivation is in [`docs/second_order_categorical.md`](../../docs/second_order_categorical.md).
This file fixes the equations consumed by ACL-003.

For row vectors and `B=M-I`,

\[
q_{t+1}=F(q_t)(I+\epsilon B).
\]

With

\[
q_t=p_t+\epsilon s_t+\frac12\epsilon^2u_t+O(\epsilon^3),
\]

define

\[
J_t=J_F^R(p_t),\qquad a_t=s_tJ_t.
\]

Then

\[
s_{t+1}=a_t+p_{t+1}B
\]

and

\[
\boxed{u_{t+1}=u_tJ_t+D^2F(p_t)[s_t,s_t]+2a_tB}.
\]

For `F_j(p)=p_jd_j/(p dot d)` and
`d=exp(eta*(r-max(r)))`,

\[
D^2F(p)[s,s]=
-2\frac{s\cdot d}{p\cdot d}\,sJ_F^R(p).
\]

The prediction used by the gate is the truncated vector norm

\[
\widehat\delta^{(2)}(\epsilon,T)=
\left\|\epsilon s_T+\frac12\epsilon^2u_T\right\|_1,
\]

not a fitted scalar correction.

The independent clean oracle propagates polynomial coefficients of

\[
p_0[D(I+\epsilon B)]^T
\]

before differentiating normalization. The outcome oracle independently evaluates the
full normalized matrix power and must agree with iterative dynamics within `5e-13`.
