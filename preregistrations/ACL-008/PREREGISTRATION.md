# ACL-008 preregistration

## Status

This is an analytic-only checkpoint. `outcomes_generated` is false. No positive-epsilon
trajectory from the 17 frozen landscapes has been evaluated. Execution is forbidden
until an adversarial audit approves an exact public Git SHA.

## Question and source rule

Does ACL-003's practical zero-fit second-order mutation radius survive a genuine change
from Shannon-entropy/categorical-Fisher geometry to Burg log-barrier geometry?

ACL-008 copies without fitting: `eta=0.05`; horizons `1,5,20,50`; every epsilon and
region; the zero-fit vector truncation; maximum error across local epsilons; Type-7
median `<=0.10`; Q90 `<=0.20`; prediction floor `2e-12`; and identity-control logic.
The ACL-003 evidence/report identities and source results are locked in `manifest.json`.

## Target system

For `h(p)=-sum(log p_i)`, the exact constrained mirror step solves

\[
F_B(p)_i^{-1}=p_i^{-1}-\eta r_i+\nu,\qquad \sum_iF_B(p)_i=1.
\]

Mutation produces

\[
q_{t+1}=F_B(q_t)(I+\epsilon(M-I)).
\]

The target Hessian is `diag(1/p_i^2)` and the update is rational, not exponential. It
is not an algebraic reparameterization of the ACL-003 update.

## Frozen target benchmark

Sixteen deterministic hypothesis-bearing targets use eight new states, eight new reward
vectors, and six new mutation matrices. Numeric novelty against the byte-exact ACL-003
manifest is mandatory. `C01` is an identity-mutation software control. Target values
were selected using only clean states and zero-epsilon derivatives; no actual perturbed
target response was inspected.

## Primary prediction

At `T=20`, for each regular target `ell`,

\[
\widehat\delta_\ell^{(2)}(\epsilon)=
\|\epsilon s_{\ell,T}+\epsilon^2u_{\ell,T}/2\|_1.
\]

The landscape score is the maximum relative error over
`epsilon in {0.001,0.003,0.01}`. PASS requires Type-7 median at most `0.10` and Q90 at
most `0.20`. There is no alpha, coefficient fit, exclusion, or target calibration.

Numerical-control `{0.0001,0.0003}` and stress `{0.03,0.1}` results are reported but
cannot change the primary verdict. Secondary horizons cannot rescue failure.

## Meaning

PASS supports a geometry-general local retraction-sensitivity class spanning entropy/
Fisher and Burg mirror maps at the copied practical radius. FAIL shows that formal
second-order differentiability alone does not transport ACL-003's useful radius.
Neither result establishes one global adaptive law.

## Forbidden behavior

- no positive-epsilon target evaluation before exact-SHA approval;
- no source-gate, target, region, metric, or threshold change after outcomes;
- no target fitting or outcome-dependent exclusion;
- no shared normalizer between the iterative and polynomial target paths;
- no rerun, overwrite, regeneration, or replacement of the first artifact.
