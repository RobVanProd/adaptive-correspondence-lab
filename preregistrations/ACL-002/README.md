# ACL-002 checkpoint contents

This directory is a preregistration-only bundle. It contains no confirmatory mutation
outcomes.

- `PREREGISTRATION.md` freezes hypotheses, estimators, strata, gates, reporting order,
  and deviation policy.
- `DERIVATION.md` fixes the row-vector Jacobian and derives the L1 and oriented-KL
  coefficients plus the independent matrix-power oracle.
- `ANALYSIS_PLAN.md` is the compact implementation contract.
- `manifest.json` fixes all source and target landscapes and experimental settings.
- `analytic_registry.json` contains clean, pre-outcome `C`/`K` calculations and frozen
  sensitivity strata. Its `outcomes_generated` field is `false`.
- `LOCK.json` hashes the preceding files and this README. The approved Git checkpoint
  additionally freezes all source and test code.

The strict-confirmatory region is `1e-4,3e-4,1e-3`. Within each regular target, its
score is the maximum of those three relative errors. `3e-3,1e-2` are extended-local
and non-gating; `3e-2,1e-1` are stress points and non-gating. The Type-7 median/Q0.90
criteria are applied across landscape scores independently for zero-fit and calibrated
predictions.

`acl acl002-validate` is safe before execution: it validates hashes, recomputes only
clean analytic sensitivities, and confirms that the locked registry matches. It never
evaluates an epsilon-positive confirmatory trajectory.

`acl acl002-run` is deliberately guarded and must not be invoked until a human approves
the exact public preregistration SHA. It also requires a completely clean worktree,
including no untracked files, and a
new output path.
