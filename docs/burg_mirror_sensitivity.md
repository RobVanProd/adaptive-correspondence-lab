# Burg mirror-map mutation sensitivity

This derivation precedes every hypothesis-bearing ACL-008 target. The mirror potential
is `h(p)=-sum_i log(p_i)`, whose Hessian is `diag(1/p_i^2)`, not the Shannon entropy/
categorical-Fisher Hessian `diag(1/p_i)`.

## Constrained step and uniqueness

For linear reward `r`, step size `eta`, and interior row-vector state `p`, the KKT
equations for reward-ascent Burg mirror descent are

\[
q_i^{-1}=p_i^{-1}-\eta r_i+\nu,\qquad \sum_iq_i=1.
\]

The normalization residual is strictly decreasing above its largest pole, tends to
positive infinity at that pole, and tends to `-1` at positive infinity. The feasible
normalizer `nu` and interior update `F_B(p)=q` are therefore unique.

## Implicit derivatives

Along `p(z)=p+zv`, let `a_i(z)=p_i(z)^{-1}-eta*r_i+nu(z)` and `q_i=1/a_i`.
Writing `W=sum_i q_i^2`, mass conservation gives

\[
\nu'=\frac{\sum_iq_i^2v_i/p_i^2}{W},\qquad
q_i'=q_i^2(v_i/p_i^2-\nu').
\]

With `a_i'=-v_i/p_i^2+nu'`,

\[
\nu''=\frac{\sum_i(2q_i^3(a_i')^2-2q_i^2v_i^2/p_i^3)}{W},
\]

\[
q_i''=2q_i^3(a_i')^2-q_i^2(2v_i^2/p_i^3+\nu'').
\]

Both derivatives have zero coordinate sum.

## Post-step mutation

Let `B=M-I` and

\[
x_{t+1}(\epsilon)=F_B(x_t(\epsilon))(I+\epsilon B).
\]

For `x_t=p_t+epsilon*s_t+epsilon^2*u_t/2+O(epsilon^3)`,

\[
s_{t+1}=D F_B(p_t)[s_t]+F_B(p_t)B,
\]

\[
u_{t+1}=D F_B(p_t)[u_t]+D^2F_B(p_t)[s_t,s_t]
+2D F_B(p_t)[s_t]B.
\]

This is ACL-003's abstract chain/product-rule structure with entirely Burg-specific
native derivatives and clean dynamics. The zero-fit prediction is
`||epsilon*s_T+epsilon^2*u_T/2||_1`.

The independent oracle directly iterates signed-epsilon Burg trajectories and applies a
symmetric five-point stencil. It never calls the sensitivity recurrence. This is a
pre-outcome software check, not the scientific comparator.
