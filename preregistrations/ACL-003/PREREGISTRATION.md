# ACL-003 preregistration

## Status

This is a preregistration-only checkpoint. `outcomes_generated` is false. No
epsilon-positive trajectory from this manifest may be evaluated until an adversarial
source/lock review approves an exact public Git SHA.

ACL-002 confirmed a first-order categorical mutation law on its frozen deterministic
benchmark. The second-order mechanism was selected after inspecting ACL-002 residuals;
its strong ACL-002 fit is exploratory and is not evidence for ACL-003.

## Scientific question

Does the analytic, zero-fit second-order categorical sensitivity remain quantitatively
predictive through epsilon `0.01` on entirely new state, reward, and mutation catalog
values, or was its ACL-002 performance catalog-local?

## Dynamics and prediction

All states are row vectors. With `B=M-I`,

\[
q_{t+1}=F(q_t)(I+\epsilon B),
\]

and

\[
q_t=p_t+\epsilon s_t+\frac{\epsilon^2}{2}u_t+O(\epsilon^3).
\]

The analytic recurrences are

\[
s_{t+1}=s_tJ_F^R(p_t)+p_{t+1}B
\]

and

\[
u_{t+1}=u_tJ_F^R(p_t)+D^2F(p_t)[s_t,s_t]
+2(s_tJ_F^R(p_t))B.
\]

The zero-fit finite-epsilon endpoint prediction is

\[
\widehat\delta^{(2)}_\ell(\epsilon,T)=
\left\|\epsilon s_{\ell,T}+\frac{\epsilon^2}{2}u_{\ell,T}\right\|_1.
\]

The absolute value is applied after vector truncation, so coordinates with zero first
derivative and finite-epsilon sign crossings are handled without assuming L1
differentiability. There is no fitted alpha or other calibration.

## Hypothesis-bearing units

The manifest contains 16 deterministic confirmatory-target landscapes `N01` through
`N16`. Landscapes—not epsilon cells—are the experimental units. They use eight state
vectors, eight reward vectors, and six mutation matrices. Every hypothesis-bearing
numeric catalog value must differ from every ACL-002 catalog value at absolute
tolerance `1e-15` with zero relative tolerance.

The novelty reference is the byte-exact ACL-002 manifest with SHA-256
`6a9e4e0a931277b1f5c464807d0bcacee3ccb684269843f8245a83ae88110741`.
Supplying a different reference file is an execution error, even if its contents would
also make the novelty comparison pass.

The benchmark is deterministic and is not a random sample from a population.
Median/Q90 are descriptive performance criteria, not confidence statements.

`C01` is an identity-mutation software control and is not hypothesis-bearing. Its
identity matrix is exempt from the new-value rule because zero response is its purpose.
If it exceeds the frozen absolute tolerance, the result is `INVALID`; a passing primary
gate is not reported as a scientific pass from an invalid instrument run.

## Regions and horizons

- Primary horizon: `T=20`.
- Secondary horizons: `T=1,5,50`.
- Full epsilon grid: `0,1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1`.
- Numerical-control region: `1e-4,3e-4`, reported and non-gating.
- Strict local confirmatory region: `1e-3,3e-3,1e-2`.
- Stress region: `3e-2,1e-1`, reported and non-gating.

`T=1` is an affine one-step theorem/software control: the second derivative of the
state response is zero. It cannot affect the scientific verdict.

## Primary estimand and verdict

For each regular confirmatory landscape `ell`, define

\[
\operatorname{score}_\ell=
\max_{\epsilon\in\{10^{-3},3\times10^{-3},10^{-2}\}}
\frac{|\delta_{\ell,\epsilon}-\widehat\delta^{(2)}_{\ell,\epsilon}|}
{\widehat\delta^{(2)}_{\ell,\epsilon}}.
\]

Using the frozen Hyndman-Fan Type-7 (`linear`) convention, ACL-003 passes if and only
if both

\[
\operatorname{median}_\ell(\operatorname{score}_\ell)\le 0.10
\]

and

\[
Q_{0.90,\ell}(\operatorname{score}_\ell)\le 0.20.
\]

These are the unchanged ACL-002 practical criteria, inherited rather than chosen from
ACL-002 target outcomes. Each local epsilon must behave acceptably within a landscape;
a failure at `0.01` cannot be hidden by the smaller epsilons.

If an analytic prediction is below `2e-12`, that landscape is frozen into a separate
low-sensitivity stratum before outcomes and receives an absolute-error report instead
of a relative-error gate. The current clean registry predeclares no such
hypothesis-bearing case.

## Secondary reporting

After the primary verdict is frozen, report:

- first-order versus second-order errors at every epsilon;
- the count of landscapes improved by second order at each epsilon;
- cumulative-prefix empirical radii at descriptive error levels `1%,5%,10%,20%`;
- endpoint and max-path L1 at `T=1,5,20,50`;
- numerical-control rows;
- all stress rows, including any catastrophic second-order failures;
- coordinate-zero classifications and all numerical/oracle diagnostics.

Secondary results cannot rescue, reverse, or redefine the primary verdict.

## Numerical and execution guards

- float64 throughout;
- endpoint prediction floor `2e-12`;
- iterative versus normalized matrix-power state tolerance `5e-13`;
- clean state derivative-oracle tolerance `5e-13`;
- clean first-derivative oracle tolerance `5e-11`;
- clean second-derivative oracle tolerance `2e-9`;
- exact approved Git SHA required;
- completely empty `git status --porcelain`, including untracked files;
- valid bundle hashes and exact analytic-registry recomputation;
- exact six-file lock membership and the frozen ACL-002 novelty-reference hash;
- previously nonexistent canonical output path
  `evidence/ACL-003-confirmatory-{approved_sha}.json`; no alternate output is allowed.

Any guard failure aborts before analysis and produces no valid scientific artifact.

## Frozen analysis sequence

1. Verify Git SHA, full worktree cleanliness, nonexistent output, lock, manifest,
   analytic registry, and numeric novelty.
2. Generate and retain every raw row exactly once.
3. Verify every iterative trajectory against the matrix-power oracle.
4. Evaluate the identity control and any predeclared low-sensitivity strata.
5. Compute the 16 primary landscape scores using only `T=20` and the three local
   confirmatory epsilons.
6. Compute Type-7 median and Q90 and freeze the conjunction verdict.
7. Only then report paired first/second comparisons, numerical controls, secondary
   horizons, max-path metrics, and stress behavior.

## Forbidden behavior and deviations

- No source or target fitting.
- No outcome-dependent landscape exclusion.
- No threshold, region, metric, or gate changes after execution.
- No rerun, overwrite, regeneration, or replacement of the first artifact.
- No claim of cross-class transport or population inference.
- No stress result may alter the primary gate.

If a defect is discovered after execution, preserve the original artifact, document
the defect, and design a separately preregistered experiment. Do not repair ACL-003 in
place.

## Possible conclusions

- **Pass:** evidence for a new-value, within-categorical-class second-order local law,
  with the stress boundary reported separately.
- **Fail:** the second-order ACL-002 success was catalog-local or the usable radius is
  narrower than `0.01` on new values.

Neither conclusion establishes a cross-class adaptive-system law.
