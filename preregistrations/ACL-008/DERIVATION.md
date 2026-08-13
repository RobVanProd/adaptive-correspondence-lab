# ACL-008 derivation

The full Burg KKT and implicit derivative derivation is frozen in
[`docs/burg_mirror_sensitivity.md`](../../docs/burg_mirror_sensitivity.md).

With `B=M-I` and row vectors, the zero-epsilon sensitivities satisfy

\[
s_{t+1}=D F_B(p_t)[s_t]+F_B(p_t)B,
\]

\[
u_{t+1}=D F_B(p_t)[u_t]+D^2F_B(p_t)[s_t,s_t]
+2D F_B(p_t)[s_t]B.
\]

The L1 norm is applied to the finite truncated vector rather than differentiated
coordinatewise, so zero first-derivative coordinates and sign crossings are explicit.

The primary path finds each dual shift by monotone bisection. The independent outcome
oracle constructs the constraint polynomial

\[
\prod_i(\nu+c_i)-\sum_i\prod_{j\ne i}(\nu+c_j)=0,
\qquad c_i=p_i^{-1}-\eta r_i,
\]

selects its unique root above every pole, and never calls the bisection normalizer.
