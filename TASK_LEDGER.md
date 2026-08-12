# Task ledger

## ACL-001 — Initial experimental substrate

- **Status:** complete
- **Observed behavior:** no implementation exists; only a research design is available.
- **Hypothesis:** under stationary shared rewards and an identity mapping on the
  interior of a finite simplex, exact replicator flow, multiplicative weights,
  and a categorical natural-gradient step produce the same next distribution.
  Controlled violations can be summarized as an epsilon-to-delta response curve.
- **Frozen semantics:** probabilities are float64 vectors on a closed simplex;
  analytic log/Fisher operations require strictly positive probabilities; rewards
  are maximized; exact replicator flow freezes the supplied reward over a step;
  categorical natural gradient uses centered logits; distances are L1 unless a
  command explicitly selects another metric; random streams are seeded PCG64
  generators; their complete state and a compact fingerprint are recorded before
  stochastic draws.
- **Allowed files:** all files created under `D:\adaptive-correspondence-lab`.
- **Acceptance tests:** analytic vector fields agree; exact three-way trajectories
  agree within a frozen float64 tolerance; Euler local error scales quadratically;
  accelerated batches match the reference path; stochastic runs replay from a seed;
  every record carries required instrumentation; Gaussian and contextual-bandit
  analytic checks pass; CLI produces machine-readable artifacts; lint and full tests pass.
- **Risks:** confusing exact flow with Euler discretization; singular categorical
  Fisher matrices; accidental RNG coupling across compared worlds; interpreting a
  numerical smoke test as a scientific result; memory growth in large seed sweeps.
- **Stop conditions:** an implementation requires an unfrozen semantic choice; the
  change crosses into neural policies or hardware acceleration; the baseline build
  is red; observed disagreement cannot be separated from a numerical defect.

### ACL-001 completion record

- **Implementation:** added three instrumented categorical worlds, nine named
  assumption violations, bounded NumPy batches, epsilon-to-delta and coefficient-
  transport protocols, analytic/rank-mu Gaussian optimization, exact/sampled
  contextual-bandit natural policy gradient, strict artifact serialization, and CLI.
- **Source checkpoint:** `48d3c5c5ef96c59fe18f6cd5b3d27a44d0fcccc6`.
- **Verification:** 47 tests passed; Ruff passed; wheel build passed; measured test
  coverage was 87%; `acl demo` and `examples/minimal.py` completed on the development
  host.
- **Immutable software evidence:** `evidence/software-verification.json` and
  `evidence/categorical-equivalence.json` both record the clean source checkpoint and
  pass. `evidence/mutation-stability.*` is a deterministic example curve, not a
  scientific claim.
- **Remaining uncertainty:** no independent scientific review has occurred; the
  transported-coefficient command demonstrates protocol plumbing within the same
  categorical correspondence and does not validate transport to a new system class;
  sample estimators have reproducibility tests but no statistical calibration claim.
- **Regression risks:** boundary states in finite populations cannot be represented
  by categorical logits; large Euler steps deliberately fail instead of projecting;
  long traces grow linearly because complete instrumentation and RNG states are kept.
- **Recommended next action:** preregister one perturbation, metric, epsilon grid,
  horizon, seed ensemble, and source/target split before treating any curve as research
  evidence.

## ACL-002 — Preregister categorical mutation stability (superseded checkpoint)

- **Status:** superseded by ACL-002A; commit `e90c097f58e4e4ab961272d3d50911d226eac25d`
  was never approved or executed. Its frozen semantics below are retained as historical
  evidence of the pre-outcome protocol change.
- **Observed behavior:** ACL-001 contains a deterministic mutation format example but
  no confirmatory manifest, analytic tangent propagation, frozen estimator, transport
  gate, or immutable preregistration bundle.
- **Hypotheses:** for the row-vector update
  `q_next = (1-epsilon) F(q) + epsilon F(q) M`, the endpoint L1 discrepancy has
  zero-fit coefficient `C = ||s_T||_1`, where
  `s_next = s J_F^R(p) + p_next (M-I)`. Separately, a landscape-balanced median source
  calibration may transport to held-out targets. For oriented
  `KL(q_T || p_T)`, the zero-fit quadratic coefficient is
  `K = 0.5 sum_i s_i^2 / p_i`.
- **Frozen semantics:** probabilities and sensitivities are row vectors;
  `(J_F^R)_[i,j] = partial F_j / partial p_i`; every `M` is row-stochastic and mutation
  is right multiplication; primary endpoint L1 uses `T=20`; secondary horizons are
  `1,5,50`; the epsilon grid is
  `0,1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1`; the five positive values through `1e-2`
  are confirmatory and the last two are non-gating stress points; source calibration
  is the median of per-landscape origin fits; empirical quantiles use Hyndman-Fan Type
  7 (`method="linear"`); analytic and calibrated predictions receive separate verdicts.
- **Sensitivity strata:** the inherited float64 verification tolerance is `2e-14`;
  the frozen safety multiplier is `100`, giving `delta_floor=2e-12`. Analytic zero is
  represented by `C<=2e-14`. A nonzero landscape is low-sensitivity when
  `C * 1e-2 < 2e-12`. These strata are determined without perturbed outcomes, excluded
  from relative-error gates, reported separately, and judged by absolute error against
  `delta_floor`.
- **Transport gate:** for each regular target landscape, take the Type-7 median of its
  five confirmatory relative errors. Across those landscape-level scores, success
  requires Type-7 median `<=0.10` and Type-7 0.90 quantile `<=0.20`. Apply this gate
  independently to `alpha=1` and to the frozen median source calibration. Stress points,
  secondary horizons, max-path L1, and KL cannot change either verdict.
- **Allowed files:** `TASK_LEDGER.md`, `README.md`, `docs/experiments.md`,
  `src/adaptive_correspondence/acl002.py`, `src/adaptive_correspondence/cli.py`,
  `tests/test_acl002.py`, and files under `preregistrations/ACL-002/`.
- **Acceptance tests:** toy fixtures (never the confirmatory manifest) check the row
  Jacobian against finite differences, tangent sensitivity against finite differences
  in epsilon, mass conservation, the KL coefficient, per-landscape alpha, median
  aggregation, Type-7 quantiles, strata, manifest/lock validation, separate gates, and
  execution guards. Full tests, Ruff, wheel build, and preregistration-only validation
  must pass.
- **Risks:** an implicit transpose; treating deterministic landscapes as seed
  replicates; allowing high-sensitivity landscapes to dominate calibration; unstable
  relative errors; reversing KL orientation; leaking stress results into the verdict;
  accidentally executing the confirmatory manifest while preparing the checkpoint.
- **Stop conditions:** any confirmatory mutation outcome is generated; a manifest
  choice changes after inspecting perturbed results; an unresolved metric or estimator
  choice appears; tests require the confirmatory landscapes; the baseline turns red.

### ACL-002 preregistration completion record

- **Implementation:** froze the narrative preregistration, row-vector derivation,
  analysis plan, 14-source/14-target manifest, clean analytic registry, SHA-256 bundle
  lock, landscape-balanced estimators, separate analytic/calibrated gates, special-
  stratum handling, complete raw reporting schema, and guarded future runner.
- **Regression sequence:** `tests/test_acl002.py` first failed because the ACL-002
  module did not exist; implementation then satisfied 13 ACL-002 toy-fixture tests.
- **Verification:** preregistration-only validation passed with 28 landscapes, 24
  regular cases, two analytic-zero cases, two low-sensitivity cases, and
  `outcomes_generated=false`; the complete suite passed 60 tests; Ruff passed; measured
  coverage was 84%; wheel build passed. A manifest-only audit found no duplicate
  landscape combinations, unused catalog entries, or exact source/target overlap.
- **Commands:** `python -m pytest tests/test_acl002.py -q`, `python -m ruff check .`,
  `python -m pytest -q`, `python -m pytest --cov=adaptive_correspondence
  --cov-report=term -q`, `python -m pip wheel . --no-deps --wheel-dir dist`, and
  `acl acl002-validate`.
- **Confirmatory status:** `acl acl002-run` was not invoked. No epsilon-positive
  trajectory from the ACL-002 manifest and no confirmatory outcome artifact exists.
- **Remaining uncertainty:** the manifest and derivation have not yet received their
  requested final human review; primary hypotheses may pass or fail once executed;
  secondary high-horizon sensitivities can be large and remain non-gating by design.
- **Regression risks:** changing any locked bundle file invalidates `LOCK.json`; future
  execution requires the exact approved full SHA, a clean tracked worktree, a matching
  clean analytic registry, and a previously nonexistent output path.
- **Recommended next action:** review the public manifest, derivation, analytic registry,
  and checkpoint SHA. If and only if accepted, explicitly approve that exact SHA for a
  later one-shot confirmatory execution.

## ACL-002A — Pre-outcome review amendments

- **Status:** replacement preregistration complete, not executed, and not yet approved
  for execution. Commit `e90c097f58e4e4ab961272d3d50911d226eac25d` remains
  superseded and unapproved.
- **Observed behavior:** clean analytic review showed that `epsilon=1e-2` predicts L1
  departures as large as roughly `0.4`; numerical guards were implemented but not all
  separately declared; execution ignored untracked files; the manifest recurrence had
  no independent matrix-power oracle; scope language overstated an additive-shift
  control and did not distinguish deterministic benchmark criteria from inference over
  a sampled population. The requested replacement for the permissive within-landscape
  median was not present in the review text after its counterexample.
- **Hypothesis:** restricting primary fitting/gating to `1e-4,3e-4,1e-3`, independently
  verifying iteration with the normalized matrix power, and freezing all numerical and
  execution guards will make the protocol more diagnostic without using outcomes.
- **Frozen semantics:** the full epsilon grid remains unchanged; strict confirmatory
  epsilons become `1e-4,3e-4,1e-3`; `3e-3,1e-2` become non-gating extended-local points;
  `3e-2,1e-1` remain non-gating stress points; the low-sensitivity rule uses the new
  maximum strict-confirmatory epsilon `1e-3`; tangent-mass tolerance is `2e-13`,
  perturbed-simplex absolute tolerance is `5e-13`, row-Jacobian mass tolerance remains
  `2e-14`, and the matrix-oracle maximum-absolute tolerance is `5e-13`; execution
  requires completely empty `git status --porcelain`, including untracked files; the
  oracle is `normalize(p0 @ matrix_power(D @ A_epsilon, T))`; for each regular target
  and prediction layer, the landscape score is the maximum relative error across the
  three strict-confirmatory epsilons; Type-7 median and Q0.90 are then applied across
  landscape scores with the existing `0.10/0.20` conjunction. The max-score rule is
  applied independently to zero-fit and calibrated layers.
- **Allowed files:** `TASK_LEDGER.md`, `README.md`, `docs/experiments.md`,
  `src/adaptive_correspondence/acl002.py`, `tests/test_acl002.py`, and files under
  `preregistrations/ACL-002/`.
- **Acceptance tests:** toy fixtures must first fail, then verify the exact region
  partition, separate numerical guards, matrix-power oracle parity at multiple epsilon
  values and horizons, oracle disagreement failure, completely clean worktree guard,
  locked manifest/registry recomputation, deterministic-benchmark scope metadata, and
  the confirmed maximum within-landscape reduction. Full tests, Ruff, wheel build,
  and preregistration-only validation must pass.
- **Risks:** treating extended-local points as gating; using the iterative function inside
  the oracle; silently weakening a numerical guard; claiming population inference from
  14 deterministic targets; finalizing an estimator that the reviewer did not state.
- **Stop conditions:** any ACL-002 confirmatory outcome is generated; `acl002-run` is
  invoked; outcome data influence an amendment; the baseline turns red.

### ACL-002A completion record

- **Implementation:** restricted strict confirmation to `1e-4,3e-4,1e-3`; made
  `3e-3,1e-2` extended-local and non-gating; replaced the within-landscape median with
  the confirmed maximum relative error; froze four separate numerical guards; required
  full porcelain worktree cleanliness; added the independent normalized matrix-power
  oracle, mismatch stop, and result provenance; removed the nonexistent additive-shift
  claim; and froze deterministic-benchmark, descriptive-criterion, and within-family
  transport scope.
- **Regression sequence:** the max-score test first failed because the prior median
  returned `0.02` instead of the required `0.03`; the new implementation returns the
  maximum and a `1%,2%,100%` fixture scores as `100%`. Matrix-oracle tests cover four
  epsilons and horizons `1,5,20,50`; an injected mismatch stops raw generation; an
  untracked Python file makes the execution worktree dirty.
- **Verification:** 79 tests passed; Ruff passed; measured coverage was 85%; wheel build
  passed; manifest and clean analytic registry recomputation matched; the registry
  retains `outcomes_generated=false`.
- **Commands:** `python -m pytest tests/test_acl002.py -q`, `python -m pytest -q`,
  `python -m ruff check .`, `python -m pytest --cov=adaptive_correspondence
  --cov-report=term -q`, `python -m pip wheel . --no-deps --wheel-dir dist`, and clean
  manifest/registry validation.
- **Confirmatory status:** `acl002-run` was not invoked. No ACL-002 raw rows,
  epsilon-positive confirmatory trajectory artifact, gate outcome, or confirmatory
  result exists.
- **Remaining uncertainty:** the replacement public SHA still requires final lock
  review and explicit approval before the one-shot run; the deterministic benchmark
  does not support population-confidence or cross-class transport claims.
- **Regression risks:** any locked-file change invalidates `LOCK.json`; an untracked
  file now blocks execution; oracle agreement verifies trajectory computation but does
  not determine whether either scientific hypothesis passes.
- **Recommended next action:** review the replacement public commit and, only if it is
  accepted, explicitly approve that exact full SHA for one-shot ACL-002 execution.

## ACL-002-PH1 — Deterministic post-confirmatory analysis

- **Status:** complete. This task derives analyses only from the immutable ACL-002
  evidence artifact; it is not a new experiment and cannot change ACL-002's frozen
  confirmatory verdict.
- **Observed behavior:** the approved preregistration
  `3f6a935942f43c7d3055582d123e58af5bf3f38b` was executed once and preserved at
  evidence commit `5caf47b510d70564415354f34ba729ff505f7ed4`. The artifact
  `evidence/ACL-002-confirmatory-3f6a935942f43c7d3055582d123e58af5bf3f38b.json`
  has SHA-256 `4d08e85b927a5d78a29078ff0d6549225d98069b20186b754629464739f29d74`,
  contains 896 deterministic rows over 28 landscapes and four horizons, and records
  passing zero-fit, calibrated-transport, special-stratum, and matrix-oracle checks.
- **Hypotheses explored, not confirmed:** the L1 residual may have a landscape-local
  second-order coefficient; dimensionless relative residuals may or may not reveal a
  shared structure; usable first-order radius may depend on horizon, reward spread,
  clean boundary proximity, mutation structure, or their interactions; oriented KL
  residuals may exhibit the predicted cubic remainder; source alpha near one may be a
  common finite-epsilon correction or an aggregate of heterogeneous errors.
- **Frozen analysis semantics:** read and hash-check the existing artifact; never call
  `acl002-run`, `generate_raw_rows`, `mutation_trajectory`, or any equivalent outcome
  generator. Preserve every raw row. Define L1 residual as
  `endpoint_l1 - C_endpoint_l1 * epsilon`, normalized residual as
  `endpoint_l1 / (C_endpoint_l1 * epsilon) - 1` only where the denominator is
  numerically valid, and quadratic scale as `residual / epsilon**2`. Define empirical
  radius at each descriptive level `1%,5%,10%,20%` as the largest tested positive
  epsilon for which every tested epsilon at or below it meets the level; this summary
  is exploratory. Analyze endpoint and max-path L1 at horizons `1,5,20,50`. Define KL
  residual as `kl_q_p - K_kl_q_p * epsilon**2`, and report its epsilon-squared and,
  when numerically stable, epsilon-cubed normalizations. Keep analytic-zero and
  low-sensitivity strata separate. Use the frozen source alpha values without refit.
  Any regression or model selection is descriptive/exploratory and must not be called
  population inference or confirmation.
- **Artifact-identifiability constraint:** raw rows store `p0`, `reward`, and
  `mutation` catalog IDs rather than their numeric arrays. To honor “use only the
  immutable artifact,” reward intensity is represented by the per-step clean
  log-odds spread inferred from stored clean terminal states (equal to `eta` times
  reward spread, so it preserves comparisons under the globally fixed eta). Mutation
  structure is represented as its frozen nominal catalog ID. Numeric mutation-matrix
  features are not identifiable from this artifact and will be reported as such; the
  analyzer must not load the external manifest to fill the gap.
- **Required outputs:** deterministic analysis code; regression tests; raw derived CSV
  tables; optional deterministic plots; a machine-readable summary; and
  `analysis/ACL-002-posthoc/ACL-002_POSTHOC.md` separating confirmed findings,
  exploratory observations, and new hypotheses. Every output records the source
  artifact path and SHA-256.
- **Allowed files:** `TASK_LEDGER.md`, `pyproject.toml` only if a required development
  dependency must be declared, new analysis code under `src/adaptive_correspondence/`,
  new tests under `tests/`, new files under `analysis/ACL-002-posthoc/`, and bridge
  ledger files if introduced without changing any prior evidence or preregistration.
- **Acceptance tests:** artifact identity and schema guards fail closed; toy fixtures
  verify residual signs/scales, cumulative empirical-radius semantics, KL
  normalizations, special-stratum separation, deterministic ordering, and summary
  classification. The generated tables recompute exactly from the immutable artifact;
  the evidence artifact hash remains unchanged; full pytest and Ruff pass.
- **Risks:** accidentally regenerating outcomes; treating posthoc patterns as
  confirmation; dividing by near-zero analytic coefficients; selecting only favorable
  epsilons; confusing endpoint with max-path quantities; allowing a flexible
  exploratory model to imply causal or population-level conclusions.
- **Stop conditions:** the evidence hash or approved SHA does not match; any requested
  statistic is ambiguous in a way that changes its interpretation; analysis requires
  modifying or recomputing the evidence artifact; a new confirmatory trajectory would
  be generated; or the baseline turns red.
- **Historical-record clarification:** the continuation prompt transcribed
  `alpha_source` as `0.9951356718983256`, while the byte-verified immutable artifact
  stores `0.9951356698171323`. The stored value exactly equals the median of the 12
  frozen per-source alphas; the absolute transcription discrepancy is
  `2.0811933287845363e-09`. Consistent with evidence-before-narrative, all analysis
  uses the artifact value and records the mismatch without altering either historical
  artifact or verdict.

### ACL-002-PH1 completion record

- **Implementation:** added a fail-closed, read-only posthoc analyzer that validates the
  source artifact bytes and evidence commit before deriving 784 positive-epsilon L1
  rows, 784 KL rows, grouped residual summaries, cumulative-prefix empirical radii,
  clean-state feature reconstructions, exploratory leave-one-landscape-out model
  comparisons, 10 CSV tables, three dependency-free SVG plots, a machine-readable
  summary, and `ACL-002_POSTHOC.md`. The analysis engine is frozen at
  `f85eb1bef950285ac80123eed89ed862a775d441`.
- **Evidence integrity:** the source artifact remains byte-identical at SHA-256
  `4d08e85b927a5d78a29078ff0d6549225d98069b20186b754629464739f29d74`.
  All 11 recorded derived-table/verification hashes and all three plot hashes
  recompute exactly. The posthoc `summary.json` SHA-256 is
  `116b8c6ec092dfdcff6a53e39f07a46fbbf8b75615d6f36e11bfed1abff14922`.
- **Confirmed findings:** only the immutable ACL-002 verdict is restated: analytic and
  calibrated primary gates pass, special strata pass, and the independent oracle
  passes. No confirmatory interpretation was added.
- **Exploratory findings:** at `T=20`, all 12 regular targets have negative first-order
  L1 residuals at every positive epsilon. Median target absolute relative error grows
  from `0.154%` in the pooled strict region to `2.626%` extended-local and `22.640%`
  stress. `T=1` has numerical-zero L1 remainder; at `T=5,20,50`, the strict-region
  quadratic scale is sign-coherent and locally stable but strongly landscape-specific.
  The additive horizon/reward-intensity/boundary/mutation-ID model has exploratory
  LOLO R-squared `0.770` for endpoint sensitivity; adding the fixed interactions lowers
  it to `0.747`. KL has a predominantly negative but more heterogeneous remainder.
  Eleven of 12 frozen source alphas are below one and one is above.
- **Regression sequence:** the first focused test failed at import because the posthoc
  module did not exist. Subsequent fixtures cover immutable identity, tamper failure,
  signed L1 scaling, zero-sensitivity guards, cumulative-prefix radius semantics, KL
  normalization, no-refit alpha summaries, artifact-only clean log-odds inference,
  and byte-deterministic full-package generation.
- **Verification:** nine focused posthoc tests pass; the last complete baseline before
  package publication passed 87 tests and Ruff, and the package integration test was
  then added. Final full-suite counts are recorded with the package commit. All three
  SVG files parse as XML and the generated Markdown passed a status-language audit.
- **No-new-outcome guarantee:** neither `acl002-run`, `generate_raw_rows`, nor any
  trajectory generator was called. Every output is a deterministic transformation of
  the preserved artifact.
- **Remaining uncertainty:** the artifact cannot identify numeric mutation-matrix
  covariates, and all second-order patterns are posthoc. A second-order recurrence must
  be derived and verified independently before deciding whether ACL-003 is warranted.
- **Recommended next action:** derive the row-vector second-order sensitivity,
  explicitly handle L1 coordinates with zero first derivative, and test it only on toy
  fixtures plus the already-exploratory ACL-002 rows before any new preregistration.

## ACL-CAT-S2 — Second-order categorical sensitivity mechanism

- **Status:** complete; derivation and posthoc mechanism evaluation only. No new
  confirmatory outcome was generated by this task.
- **Observed behavior:** ACL-002 posthoc analysis finds numerical-zero one-step L1
  remainder and a negative, locally quadratic remainder for every regular target at
  horizons `5,20,50` through the strict and extended-local regions. The scale is
  strongly landscape-dependent.
- **Hypothesis:** differentiating the frozen mutation recurrence twice produces a
  state-aware zero-fit second-order predictor that explains the existing exploratory
  residual and extends its descriptive epsilon radius better than any universal scalar
  correction.
- **Frozen semantics:** with row vectors, `B=M-I`, clean map `F`, row Jacobian `J`, and
  expansion `q_t=p_t+epsilon*s_t+(epsilon^2/2)*u_t+O(epsilon^3)`, define
  `a_t=s_t J_F^R(p_t)` and
  `H_t=D^2F(p_t)[s_t,s_t]`. The recurrence is
  `u_{t+1}=u_t J_F^R(p_t)+H_t+2*a_t*B`, with `u_0=0`. For the normalized linear
  categorical map, verify independently that
  `H_t=-2*((s_t dot d)/(p_t dot d))*a_t`, where
  `d=exp(eta*(r-max(r)))`. The endpoint finite-epsilon truncated-vector prediction is
  `sum_i |epsilon*s_i+(epsilon^2/2)*u_i|`. The asymptotic L1 quadratic coefficient is
  one half of `sum sign(s_i)*u_i` over nonzero first derivatives plus
  `sum |u_i|` over coordinates classified zero at the frozen absolute threshold
  `2e-14`. Such zero coordinates are reported explicitly because L1 is not
  differentiable there.
- **Independent oracles:** compare the recurrence to (1) coefficient propagation of
  the matrix polynomial `p0[D(I+epsilon B)]^T` followed by analytic normalization and
  (2) five-point signed-epsilon finite differences on toy fixtures. The matrix-
  polynomial path must not call the recurrence implementation.
- **Existing-data evaluation:** after software verification, apply the analytic
  second-order prediction to the already-exploratory ACL-002 raw rows. Do not generate
  trajectories or refit source/target coefficients. Compare first- and second-order
  cumulative-prefix radii at the fixed descriptive levels `1%,5%,10%,20%` and retain
  all failures. This evaluation remains posthoc and cannot confirm ACL-CAT-S2.
- **Frozen ACL-003 earning rule:** before evaluating the second-order predictor on
  stored ACL-002 outcomes, require all independent derivative oracles to pass; at
  target `T=20`, require the median discrete radius index to improve by at least one
  epsilon-grid step at both the `5%` and `10%` descriptive levels, no more than two of
  12 regular targets to lose radius at either level, median absolute relative error at
  `epsilon=0.01` to fall by at least `50%`, and strict-region Type-7 Q90 absolute
  relative error not to worsen. This rule only decides whether the mechanism is worth
  preregistering on new landscapes; it is not a confirmatory scientific verdict.
- **Allowed files:** `TASK_LEDGER.md`, new second-order source and tests, new derived
  files under `analysis/ACL-002-second-order/`, and bridge-ledger updates. Do not change
  the ACL-002 evidence artifact, preregistration bundle, or confirmatory runner.
- **Acceptance tests:** Hessian-vector formula matches finite differences; recurrence
  conserves first- and second-order tangent mass; recurrence matches the independent
  polynomial oracle over multiple dimensions/horizons; five-point finite differences
  recover both derivatives; L1 zero-coordinate fixtures use the nondifferentiable
  branch; `T=1` second-order L1 coefficient is zero up to tolerance; invalid inputs
  fail closed; full pytest and Ruff pass.
- **Risks:** losing the factor of two from the expansion convention; using
  `s_{t+1}B` instead of the pre-mutation `a_t B`; hiding an L1 sign crossing inside a
  scalar Taylor coefficient; treating posthoc radius improvement as confirmation;
  accidentally importing or calling an outcome generator.
- **Stop conditions:** either independent oracle disagrees beyond a frozen tolerance;
  coordinate-zero classification is unstable; existing-data evaluation requires
  target fitting; a new trajectory is generated; or baseline verification turns red.
- **Clean-only numerical amendment:** the first frozen mechanism commit
  `3f05fa663093163832bf33924d6167bf390534bf` stopped before reading stored outcome
  errors because the Hessian mass guard used a scale-blind `2e-13` absolute limit. On
  the high-curvature clean `edge/strong_ordered/uniform_mix` case, cancellation reached
  `9.1e-13` while the Hessian L1 norm reached about `1.9e4` (relative residual below
  `5e-17`); the independent polynomial path showed the same scaling issue. Freeze all
  derivative mass checks as `absolute_tolerance + 2e-15 * vector_L1_norm`, retaining
  absolute tolerances `2e-13` for first/Hessian and `2e-11` for second derivative. Add
  the clean high-curvature 50-step case as a regression. No ACL-002 outcome comparison
  or second-order package existed when this amendment was made.

### ACL-CAT-S2 completion record

- **Implementation:** derived and implemented
  `u_next=u J+D^2F[s,s]+2(s J)(M-I)`, the closed Hessian contraction, explicit L1
  zero-coordinate handling, finite-epsilon truncated-vector prediction, an independent
  unnormalized matrix-polynomial coefficient oracle, and a five-point signed-epsilon
  derivative test. The final analysis generator is commit
  `071373b9bc4dd9369f82f3bda9bb20363d3eabda`.
- **Oracle evidence:** across all 28 frozen clean landscapes through `T=50`, maximum
  recurrence/oracle errors are `4.44e-16` for state, `1.14e-13` for first derivative,
  and `5.82e-11` for second derivative, all below frozen tolerances.
- **Exploratory stored-row result:** at target `T=20`, second-order median/Q90 absolute
  relative error is `0.0001%/0.0050%` strict and `0.0453%/0.9441%` extended-local.
  The frozen ACL-003 earning rule passes: median radius index improves by one grid step
  at both `5%` and `10%`, no target loses radius, median error at `epsilon=0.01` falls
  by `97.39%`, and strict Q90 improves.
- **Mapped failure boundary:** second order is not uniformly stress-stable. At target
  `T=20`, stress Q90 worsens from `62.17%` first order to `77.66%` second order, and
  the worst second-order relative error is `778.42%`.
- **Artifact:** `analysis/ACL-002-second-order/summary.json` has SHA-256
  `d7533c3f3b5e0941e28cddcba58ce4106825c938f7244c24bd8f98c8e9403474` and records
  `outcomes_generated=false`, `target_refit=false`, and the frozen earning-rule result.
- **No-new-outcome guarantee:** the evaluator reads the immutable ACL-002 artifact and
  frozen numeric manifest, computes clean analytic derivatives, and compares them to
  stored rows. It never calls the mutation trajectory or confirmatory runner.
- **Interpretation:** the mechanism earns an ACL-003 preregistration on entirely new
  categorical catalog values. Its ACL-002 success remains exploratory and cannot be
  presented as confirmation.
- **Recommended next action:** construct a preregistration-only ACL-003 with new state,
  reward, and mutation catalog values; freeze second-order-versus-first-order local
  predictions and stress failure expectations; audit the checkpoint adversarially
  before any outcome generation.

## ACL-003 — New-value confirmation of the categorical second-order law

- **Status:** preregistration construction only; outcomes forbidden.
- **Observed behavior:** the no-fit analytic second-order truncation passed its frozen
  ACL-002 posthoc earning rule and failed visibly in stress. Because that mechanism was
  selected after ACL-002 outcomes, it requires an entirely new confirmatory benchmark.
- **Hypothesis:** on new categorical states, rewards, mutation matrices, and landscape
  combinations, the zero-fit truncated-vector prediction
  `||epsilon*s_T+(epsilon^2/2)*u_T||_1` remains quantitatively accurate through
  `epsilon=0.01` without any fitted source or target coefficient.
- **Scope:** deterministic three-state categorical systems; row-stochastic mutation;
  fixed reward and `eta=0.05`; primary endpoint `T=20`; float64; strictly interior
  clean states. This is a stronger within-class test, not cross-class transport.
- **Frozen experimental regions:** retain the grid
  `0,1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1`. The numerical-control region
  `1e-4,3e-4` is reported but non-gating. The strict local confirmatory region is
  `1e-3,3e-3,1e-2`. Stress `3e-2,1e-1` is reported and incapable of changing the
  confirmatory verdict. Primary horizon is `20`; `T=1,5,50` are secondary, with `T=1`
  serving as the affine one-step control.
- **Frozen primary estimand and gate:** for each of 16 regular new-value landscapes,
  compute the maximum second-order absolute relative endpoint-L1 error across the
  three local confirmatory epsilons. Apply the inherited ACL-002 Type-7 criteria
  unchanged: median landscape score at most `0.10` and Q90 at most `0.20`. Reusing
  these pre-outcome historical criteria avoids choosing thresholds from ACL-002 target
  outcomes. The conjunction is the ACL-003 verdict.
- **Paired secondary content:** report first-order and second-order errors, count the
  landscapes on which second order improves at each epsilon, and compare empirical
  `1%,5%,10%,20%` cumulative-prefix radii. These comparisons are secondary and cannot
  rescue or reverse the primary verdict.
- **Special handling:** include one identity-mutation software control outside the 16
  hypothesis-bearing landscapes. It is judged only by the inherited absolute
  numerical floor. Any other analytically near-zero case discovered before outcomes
  is frozen into a separate stratum. Relative errors are never evaluated when the
  analytic prediction is below the frozen floor.
- **New-value requirement:** every hypothesis-bearing state vector, reward vector, and
  mutation matrix must be numerically distinct from every ACL-002 catalog value;
  validation compares arrays, not names. Landscape triplets are new as a consequence.
  The identity control is exempt because its purpose is theorem/software validation.
- **No fitting:** ACL-003 has no source split, alpha, regression, or target calibration.
  Both `s_T` and `u_T` are computed from the analytic recurrence before perturbed
  outcomes. No ACL-002 outcome-derived coefficient enters the prediction.
- **Implementation contract:** freeze narrative preregistration, derivation reference,
  analysis plan, manifest, clean analytic registry, SHA-256 lock, guarded validator,
  and future one-shot runner. Execution requires an exact approved SHA, completely
  clean full porcelain worktree, valid locks/registry, and a nonexistent output path.
  Raw generation must compare iterative trajectories with the independent matrix-power
  oracle and abort above `5e-13`.
- **Allowed files:** `TASK_LEDGER.md`, `src/adaptive_correspondence/acl003.py`, CLI
  wiring, `tests/test_acl003.py`, and files under `preregistrations/ACL-003/`. Bridge
  ledgers may be updated only for preregistration status.
- **Acceptance tests:** fail first; then validate exact regions/gates, numeric novelty,
  analytic registry recomputation, identity control, second-order recurrence/oracle
  parity, full worktree/output guards, lock integrity, deterministic analysis order,
  special-stratum separation, and stress non-gating behavior. Full pytest, Ruff,
  coverage, wheel build, and preregistration-only validation must pass.
- **Risks:** tuning the new catalog using perturbed outcomes; accidentally inheriting
  an ACL-002 numeric value; allowing stress failures to alter the gate; silently
  turning a near-zero analytic case into a relative-error case; interpreting a pass as
  cross-class evidence.
- **Stop conditions:** any ACL-003 perturbed trajectory or outcome is generated before
  the lock audit; numeric novelty fails; the registry depends on outcomes; a threshold
  changes after outcomes; or the baseline is red.
- **First public checkpoint and adversarial hold:** commit
  `fecdd68809868280e3852d5bc23075db28ae2ff3` froze the intended design with no
  outcomes, but pre-outcome source audit found that the validator accepted an arbitrary
  caller-supplied novelty-reference manifest and did not require the lock to enumerate
  the exact six frozen files. It also reported the identity control separately without
  an explicit invalid-instrument verdict. That checkpoint is superseded before outcome
  generation.
- **Frozen audit amendments:** bind the numeric-novelty comparison to ACL-002 manifest
  SHA-256 `6a9e4e0a931277b1f5c464807d0bcacee3ccb684269843f8245a83ae88110741`;
  require the exact narrative/derivation/plan/readme/manifest/registry lock set; and
  report `INVALID` if the identity control fails even when the hypothesis gate passes.
- **Second public-checkpoint hold:** exact-SHA review of
  `eabdb7eca082c4f5d87e193d73edd892a9260d4b` found that the runner still accepted an
  arbitrary nonexistent output path. An out-of-repository first output would not dirty
  the worktree and therefore weakened the one-shot guard. Before any outcome, freeze
  the sole permitted destination as
  `evidence/ACL-003-confirmatory-{approved_sha}.json` resolved under the repository.

## ACL-003-POST — Artifact-only confirmatory report and categorical decision

- **Status:** post-confirmatory analysis; no trajectory generation permitted.
- **Immutable input:** evidence commit
  `b15d77600369d559cb586a3bb54924737758e038`, artifact
  `evidence/ACL-003-confirmatory-501464f3f6be07f6d813d94aefb818c461a3d5c7.json`,
  SHA-256 `1f80c3f5aba4089c67bbfec1ddd6eff53f7a6d42c658436dff0f7c82a1cf8c99`.
- **Observed frozen result:** valid instrument; primary PASS; median maximum local
  second-order relative error `0.0014843120912351297`; Type-7 Q90
  `0.007387117284289386`; no randomness or target refit.
- **Hypothesis:** no new confirmatory hypothesis. Reproduce the locked verdict and
  describe secondary/stress behavior solely from stored rows to decide whether the
  categorical stability question is sufficiently mapped to advance classes.
- **Frozen analysis:** verify artifact hash, approved SHA, row/region/horizon counts,
  identity control, oracle maximum, and exact stored primary gate. For each horizon and
  region, summarize first- and second-order absolute relative endpoint errors only
  where predictions exceed the frozen floor. At `T=20`, report paired improvement
  counts and cumulative-prefix empirical radii at `1%,5%,10%,20%`. Stress is
  exploratory and cannot change ACL-003's verdict. Never fit target rows.
- **Allowed files:** `TASK_LEDGER.md`, an artifact-only analysis module and tests,
  `analysis/ACL-003-confirmatory/`, and bridge-ledger status updates.
- **Acceptance tests:** corrupted hash/row count fails; stored primary scores are
  reproduced exactly; stress cannot affect the verdict; derived output is deterministic;
  full pytest, Ruff, and wheel build pass.
- **Risks:** silently recomputing trajectories; presenting stress summaries as
  preregistered gates; overclaiming new-value within-class confirmation as cross-class
  transport; altering the evidence file.
- **Stop conditions:** evidence hash changes; derived primary values disagree with the
  artifact; analysis requires refitting or a new outcome; categorical interpretation
  remains ambiguous after stored-row summaries.

## GNG-MECH — Independent finite-lambda Gaussian rank-mu comparator

- **Status:** clean mathematical/software mechanism only; no confirmatory landscapes or
  Gaussian scientific outcomes.
- **Observed behavior:** the existing Gaussian rung computes analytic quadratic natural
  gradient and finite-sample rank-mu updates, but its recorded analytic field is not the
  finite-lambda expected rank direction required by H2.
- **Hypothesis:** for a diagonal Gaussian and a nonzero linear objective, the exact
  finite-lambda conditional expected rank-mu tangent can be constructed independently
  from the Gaussian score, Fisher inverse, and conditional binomial rank law.
- **Frozen semantics:** parameterization `(mean, log_std)` in dimension at least two;
  linear objective `a dot x`; `lambda=32`, `mu=16`; logarithmic positive rank weights
  `log(mu+0.5)-log(rank)` normalized over selected ranks; no evolution paths, CSA,
  clipping, antithetic sampling, or state iteration. For standardized objective axis
  `v=(a*std)/||a*std||`, define conditional expected rank utility `h_lambda(t)`. The
  independent analytic blocks are `lambda*(std*v)*E[h(t)t]` and
  `(lambda/2)*(v^2)*E[h(t)(t^2-1)]`.
- **Geometry:** compare mean and covariance/log-scale blocks separately using the
  Gaussian Fisher metric. A joint cosine is secondary only.
- **Implementation:** evaluate the one-dimensional expectations by deterministic
  Gauss-Hermite integration under the standard-normal measure; verify convergence at
  frozen order `160` against doubled order `320` at relative tolerance `2e-9` and
  absolute tolerance `5e-12`, and compare with a separate tensor
  Gauss-Hermite score integral on toy
  fixtures. Sampling code returns raw independent shadow directions and does not use
  the analytic comparator.
- **Allowed files:** `TASK_LEDGER.md`, a new Gaussian bridge mechanism module, its
  tests, derivation documentation, and bridge-ledger mechanism status. No
  preregistration or evidence directory.
- **Acceptance tests:** rank weights and conditional utility normalize correctly;
  analytic blocks match independent multidimensional quadrature; large toy shadow mean
  has high separate-block Fisher cosine; invalid dimensions/scales/objective fail;
  full pytest, Ruff, and wheel build pass.
- **Risks:** accidentally comparing to the quadratic-objective natural gradient rather
  than finite-lambda rank expectation; replaying the update in the comparator; hiding
  covariance-block noise in a joint cosine; treating a clean mechanism check as
  scientific evidence.
- **Stop conditions:** the comparator requires sampled rank-mu outcomes; block direction
  is analytically zero for the proposed design; quadrature or independent oracle does
  not converge; implementation requires optimizer machinery outside the frozen scope.

## ACL-004 — Conditional-mean Gaussian finite-lambda bridge

- **Status:** preregistration construction only; confirmatory shadows forbidden until
  an exact public preregistration SHA passes adversarial review.
- **Scientific question:** at a frozen finite population, does the independently
  derived Gaussian score/Fisher comparator predict the conditional expected rank-mu
  tangent in both mean and covariance blocks, or is apparent alignment a joint-metric
  or infinite-population artifact?
- **Scope:** 12 deterministic three-dimensional diagonal-Gaussian landscapes with
  nonzero linear ranking objectives; parameterization `(mean, log_std)`; one state only
  per shadow; `lambda=32`, `mu=16`; normalized logarithmic positive rank weights;
  mean learning rate `0.2`, covariance/log-scale learning rate `0.1`; PCG64; float64;
  no evolution paths, CSA, clipping, antithetic pairs, common random numbers across
  landscapes, state iteration, or lambda scaling.
- **Analytic comparator:** finite-lambda block direction derived in
  `docs/gaussian_rank_mu_bridge.md` from conditional binomial rank utility, Gaussian
  score, and inverse Fisher. It may not call or replay sampled rank-mu code. Freeze
  Gauss-Hermite order `160`, doubled-order oracle `320`, relative tolerance `2e-9`,
  and absolute tolerance `5e-12`.
- **Shadow estimator:** independent shadows at the same frozen lambda. Check cumulative
  replication counts `4096,8192,16384,32768,65536`. At each checkpoint, compare the
  means of the first and second disjoint halves using separate Fisher cosines. Stop at
  the first checkpoint where both are at least `0.98`. Failure to converge by `65536`
  makes the H2 result `INCONCLUSIVE`, not PASS or FAIL.
- **Primary H2 estimands and gate:** for each converged landscape, compare its full
  stopped conditional-mean estimate with the independent analytic comparator. Report
  separate mean-block and covariance-block Fisher cosines. H2 passes only if all 12
  landscapes converge and the minimum landscape cosine is at least `0.99` in each
  block. Any converged value below its block threshold is FAIL. Joint cosine is
  secondary and cannot rescue a block.
- **H1 descriptive layer:** retain the first `2048` single-shadow mean/covariance
  cosines per landscape and report their quantiles/fraction positive. H1 cannot alter
  H2. Learning-rate scaling is recorded but block cosines are computed on raw tangents.
- **Evidence sufficiency:** retain for each fixed 2048-shadow chunk its count, tangent
  sum, and tangent outer-product sum, plus the raw H1 cosines. These sufficient
  statistics must exactly reproduce every stopped mean, half-mean convergence check,
  and uncertainty summary without retaining a massive per-shadow JSON array.
- **Units and inference:** landscapes are deterministic benchmark units, not a random
  population sample. Shadows estimate conditional expectations at frozen landscapes;
  they are not landscape replication. Minima are descriptive acceptance criteria, not
  confidence statements.
- **Execution guards:** exact approved SHA; full clean porcelain worktree; exact lock
  membership and hashes; analytic-registry recomputation; previously nonexistent sole
  canonical output `evidence/ACL-004-confirmatory-{approved_sha}.json`; independent
  landscape seeds; valid chunk arithmetic and finite outputs. One execution only.
- **Allowed files:** ACL-004 implementation and tests, CLI wiring, this ledger,
  `preregistrations/ACL-004/`, and Gaussian bridge preregistration status. No evidence
  file before approval.
- **Acceptance tests:** fail first; then exact design validation, independent analytic
  registry/oracle, stopping at first qualifying checkpoint, nonconvergence handling,
  block-specific gates, joint-cosine non-rescue, sufficient-statistic reproduction,
  canonical-path/worktree/lock guards, full pytest, Ruff, coverage, and wheel build.
- **Risks:** confirmatory-landscape pilot leakage; covariance signal too weak to meet
  the stopping rule; target-specific refitting; calling exact theorem reproduction a
  cross-class transported law; changing lambda to improve noise.
- **Stop conditions:** any proposed confirmatory landscape is sampled before its public
  lock; the analytic block is zero/near-zero; quadrature fails; a threshold changes
  after shadows; or the implementation introduces hidden Gaussian optimizer state.
- **First public checkpoint and audit hold:** commit
  `bf5df09c1c3d58c3e1892234cbdd3da2b921c66b` froze the full design with zero
  shadows. Exact-head audit held execution because the future evidence referenced but
  did not embed the locked analytic registry and bundle hashes, weakening standalone
  inspection. Before outcomes, embed both frozen objects and add an independent toy
  Monte Carlo check of the conditional binomial rank-utility calculation. The
  scientific design, landscapes, stopping rule, and gates do not change.

## ACL-004-POST — Artifact-only Gaussian bridge report

- **Status:** post-confirmatory analysis; no additional shadows or RNG calls.
- **Immutable input:** evidence commit
  `355dd97472da4230eff877b9a3c8c7c4626057cd`, artifact
  `evidence/ACL-004-confirmatory-3ba4be7ce1460a40c4ef0879018df58947c36edb.json`,
  SHA-256 `3f97f7c4debbd65014e6ee337d9a8990500bad38ce0dfef2e8ad3048c74cd91a`.
- **Frozen result:** H2 PASS; all 12 landscapes converged at 4096 shadows; minimum
  mean Fisher cosine `0.9999523237518517`; minimum covariance Fisher cosine
  `0.9995521999195026`; threshold `0.99`; target refit false; lambda scaling absent.
- **Analysis:** verify immutable bytes/envelope; reconstruct every stopped mean from
  stored chunk sums and every first/second half mean from disjoint chunks; reproduce
  all separate-block and joint cosines; summarize stored H1 Q10/median/Q90/fraction
  positive and coordinate standard errors; inspect no new lambda or objective class.
- **Interpretation rule:** confirmation supports the restricted Gaussian finite-lambda
  expected-direction bridge only. It is a second adaptive-system class with predictive
  content, but no categorical quantity was transported into it, so it does not yet
  satisfy cross-class no-refit transport.
- **Allowed files:** `TASK_LEDGER.md`, an artifact-only analyzer and tests,
  `analysis/ACL-004-confirmatory/`, and bridge-ledger status updates.
- **Acceptance tests:** wrong hash fails; chunk reconstruction is exact; block cosines
  and H2 minima reproduce; report is deterministic; full pytest, Ruff, and wheel pass.
- **Stop conditions:** evidence bytes change; reproduction disagrees; analysis requires
  new shadows/refitting; or Gaussian scope is overstated as cross-class transport.

## CTRL-MECH — Finite-sample contextual-bandit NPG mechanism

- **Status:** theorem/software mechanism only; no control confirmatory landscapes.
- **Question:** does a plug-in natural-policy-gradient estimator that pseudoinverts its
  empirical Fisher have a conditional mean aligned with the independently exact
  contextual-bandit NPG, or can finite-sample inversion bias rotate it?
- **Frozen estimator:** finite contexts/actions, exact categorical policy, deterministic
  reward matrix, and independent joint context-action draws. From one frozen batch,
  compute the empirical score gradient and empirical score outer-product Fisher, apply
  an undamped Moore-Penrose pseudoinverse with `rcond=1e-12`, and center each context's
  logit direction. No clipping, baseline fitting, damping, trajectories, or policy
  update.
- **Comparator:** exact expected policy gradient and analytic categorical-policy Fisher
  at the frozen state. Construct the centered exact natural direction directly from
  rewards and policy values; do not average or replay sampled plug-in updates.
- **Geometry:** context blocks are primary; each uses its analytic categorical Fisher.
  A context-probability-weighted joint Fisher cosine is secondary only.
- **Transport candidate:** the ACL-004 normalized expected-direction law and exact
  thresholds—disjoint-half block cosine `0.98` and target analytic block cosine
  `0.99`—are eligible for unchanged no-refit application in control. This mechanism
  task does not test them scientifically.
- **Implementation and evidence:** vectorize shadows through multinomial joint counts;
  retain counts-to-direction equivalence with a direct score-sum oracle on toy counts.
- **Allowed files:** `TASK_LEDGER.md`, a new control bridge mechanism, tests,
  derivation documentation, and theorem-reproduction ledger status. No control
  preregistration or evidence.
- **Acceptance tests:** fixed-count direction matches direct score/Fisher accumulation;
  analytic Fisher maps exact direction to policy gradient; a large toy conditional
  mean aligns in each context; empirical singular blocks remain explicit zero/pinv
  behavior; invalid policies/counts fail; full pytest, Ruff, and wheel pass.
- **Risks:** mistaking plug-in expectation for pseudoinverse of expected Fisher; hiding
  a failed context in a joint cosine; treating sample count or shadow count as a fit;
  reusing the product-of-simplexes coordinate map as proof of a new class.
- **Stop conditions:** comparator calls sampled code; empirical damping/clipping is
  introduced; toy oracles disagree; a transported threshold is modified for control.

## ACL-005 — Cross-class transport of the normalized expected-direction law

- **Status:** preregistration construction only; control target shadows forbidden.
- **Source law:** ACL-004 confirmed that, after every Fisher block's disjoint-half
  conditional-mean cosine reaches `0.98`, every source block's Fisher cosine with its
  independent analytic direction exceeds `0.99`. ACL-005 transports this dimensionless
  law, the thresholds, the replication schedule `4096,8192,16384,32768,65536`, chunk
  size `2048`, and H1 count `2048` unchanged into control. No control target fit or
  threshold adjustment is permitted.
- **Control system:** two-context, three-action exact contextual bandits; deterministic
  rewards; categorical policies in centered logits; fixed within-shadow interaction
  count `N=128`; independent PCG64 stream per landscape; empirical score gradient and
  empirical score Fisher; undamped pseudoinverse `rcond=1e-12`; one frozen state only.
- **Comparator:** exact centered natural policy direction from analytic advantage and
  categorical-policy Fisher. It must not call the plug-in estimator.
- **Landscapes and strata:** 10 regular confirmatory targets with pre-outcome minimum
  expected joint cell count `N*rho(c)*pi(a|c) >= 4`; four stress targets with minimum
  expected joint cell count `<=0.75`. Strata are computed from manifest-only analytic
  quantities. Stress is reported and non-gating.
- **Stopping:** per landscape, compare first/second disjoint-half conditional means in
  each context Fisher block. Stop at the first scheduled count where every context is
  at least `0.98`. A regular landscape not converged by `65536` makes the transported
  H2 verdict `INCONCLUSIVE`. Stress nonconvergence is descriptive only.
- **Primary transported gate:** for every stopped regular target, compute each context's
  Fisher cosine against the exact analytic NPG. PASS only if all 10 regular landscapes
  converge and the minimum over all 20 context blocks is at least `0.99`. A converged
  regular block below `0.99` is FAIL. Joint cosine cannot rescue a context.
- **Secondary/stress:** H1 first-2048 single-shadow context cosine quantiles and
  fraction positive; joint cosine; coordinate standard errors; stress context
  convergence/alignment. Stress cannot rescue or reverse primary transport.
- **Evidence:** fixed 2048-shadow chunk counts, direction sums, and outer-product sums
  reproduce stopped/half means; raw H1 cosines are retained. Embed manifest, analytic
  registry, lock, seeds, and terminal RNG states.
- **Scope:** this is a no-refit transport of a normalized score/Fisher expected-direction
  law from Gaussian rank-mu into finite-state control. It does not transport ACL-003's
  epsilon coefficient, claim product-of-simplexes is a new geometry, study sample-count
  scaling, or imply PPO/neural-policy behavior.
- **Execution guards:** exact approved SHA; full clean porcelain; exact lock and
  directory membership; analytic registry recomputation and stratum validation;
  previously nonexistent canonical
  `evidence/ACL-005-confirmatory-{approved_sha}.json`; one execution only.
- **Allowed files:** ACL-005 module/tests/CLI, this ledger, `README.md`,
  `preregistrations/ACL-005/`, and machine/human bridge-ledger preregistration status.
  No target evidence.
- **Acceptance tests:** fail first; exact transported constants; analytic strata;
  first qualifying stop; primary context-block gate; stress non-gating; nonconvergence
  semantics; sufficient-statistic reproduction; canonical/lock/worktree guards; full
  pytest, Ruff, coverage, wheel, and analytic-only validation.
- **Risks:** calling copied thresholds a fitted control result; target landscape pilots;
  hiding rare-context failure in joint geometry; changing N after outcomes; treating a
  control pass as evidence for neural or sequential RL.
- **Stop conditions:** any manifest target is sampled before public approval; regular or
  stress analytic stratum disagrees; source thresholds/schedule change; the comparator
  depends on sampled outcomes; or baseline turns red.
- **First public checkpoint and audit hold:** commit
  `677086f` froze the design with zero target shadows. Exact-head audit held execution
  because the runner accepted an arbitrary external self-consistent bundle, so the
  approved SHA did not uniquely bind the target manifest. Before outcomes, require the
  canonical in-repository bundle and directly verify the frozen ACL-004 evidence and
  report hashes. The hypothesis, target landscapes, thresholds, and schedule do not
  change; the first checkpoint is superseded pre-outcome.

## ACL-005-POST — Artifact-only cross-class transport report

- **Status:** post-confirmatory analysis only; no additional shadows or RNG calls.
- **Immutable input:** evidence commit
  `24d577f8a1d7bc6f4f45250f4bab3d5b2b925aeb`, artifact
  `evidence/ACL-005-confirmatory-c3ebc07a41e8dbb84a24c68cdbb4f75c36108c5b.json`,
  SHA-256 `5400a12392609f5cdf79a8b4b380f84ad11e68330f8ee93f653439129aa5db5b`.
- **Frozen result:** transport PASS; all 10 regular targets converged at 4096;
  minimum regular context Fisher cosine `0.9998416299085249` versus `0.99`;
  target refit false. Stress is non-gating and includes severe rotations.
- **Analysis:** verify immutable bytes and envelope; reconstruct stopped and half means
  from chunks; reproduce context/joint cosines, uncertainty, and H1 summaries; report
  regular transport and rare-cell stress separately; make no target fit.
- **Interpretation:** a PASS supports transport of the specific ACL-004 normalized
  conditional-mean diagnostic into this control class. Stress failures bound the law's
  scope and cannot be presented as a universal NPG result.
- **Allowed files:** this ledger, an ACL-005 artifact-only analyzer and tests,
  `analysis/ACL-005-confirmatory/`, bridge ledgers, README, and final synthesis.
- **Acceptance tests:** wrong hash fails; chunk/half/cosine/uncertainty/H1 reconstruction
  matches; regular and stress strata remain separate; report generation is deterministic;
  full pytest, Ruff, coverage, and wheel pass.
- **Stop conditions:** evidence bytes change; any new shadow/RNG call occurs; a target
  coefficient is fit; stress is allowed to change the primary verdict; or PASS is
  generalized to sequential/neural control.

## SYNTHESIS-001 — Program termination assessment

- **Question:** after ACL-005, does the evidence meet Outcome A, B, or C?
- **Frozen decision rule:** apply the user-specified termination criteria without
  adding a new experiment merely to strengthen a positive story.
- **Candidate:** Outcome A in restricted form, because categorical, Gaussian, and
  control classes now have explicit maps/scope; categorical quantitative degradation
  laws; preregistered within-class predictions; one preregistered Gaussian-to-control
  normalized-law transport without target refit; and categorical/control stress
  boundaries. This remains conditional on exact ACL-005 artifact reconstruction.
- **Output:** `FINAL_SYNTHESIS.md` with every preregistration/evidence hash, positive and
  negative findings, complete bridge ledger, falsified/surviving claims, exclusions,
  strongest supported statement, and open theorem.
- **Stop condition:** if artifact reconstruction fails or the cross-class quantity was
  not actually frozen before control outcomes, do not terminate under Outcome A.

### ACL-005 posthoc and synthesis completion record

- **Artifact reconstruction:** exact. Stored chunk statistics reproduce stopped means,
  disjoint-half means, context/joint cosines, uncertainty, and H1 summaries with zero
  maximum discrepancy. No new shadows or RNG calls were made.
- **Report:** `analysis/ACL-005-confirmatory/summary.json`, SHA-256
  `b5d310e9a32c059cb192e4f1001556b7a9d60ca98cc6319a1d67326246c13084`.
- **Confirmed transport:** PASS; all regular targets stopped at 4096; minimum of 20
  context cosines `0.9998416299085249`; target refit false.
- **Boundary:** all stress targets stopped at 4096, but 5/8 stress blocks were below
  `0.99` and the minimum was `0.05569937277874286`. Internal convergence is not a
  support-free certificate of analytic alignment.
- **Termination:** Outcome A, predictive unification in restricted form. The program
  has a successful preregistered cross-class transported diagnostic and explicit local
  and support-related breakdown regimes. `FINAL_SYNTHESIS.md` states the bounded claim.
- **Verification:** 157 tests passed; Ruff passed; total coverage 83%; wheel build
  passed. Immutable ACL-002 through ACL-005 evidence hashes remain unchanged.
- **Recommended next action:** no additional experiment is required for the original
  question. Future work should pursue the support-conditioned angular-bias theorem,
  and any new empirical extension must begin as a separate preregistered program.

## PHASE-II-AUDIT — Canonicalize and reclassify Phase I

- **Task ID:** `PHASE-II-AUDIT`.
- **Status:** complete; audit/protocol only, no new scientific outcomes.
- **Observed state:** Phase I ended on branch `agent/acl003-preregistration` at synthesis
  commit `358e18baccf29fd2959054be6bb25dd3869d8f77`; public `main` previously stopped at
  `50ad814138e783f036ca4acdf95ce9ebbf644a90`, before ACL-003 through ACL-005.
- **Hypothesis:** the public history, preregistration locks, immutable evidence bytes,
  and artifact-only reconstructions are internally consistent and support the narrower
  Phase-I classification required for Phase II.
- **Frozen classification:** ACL-002 confirms its amended strict first-order region;
  ACL-003 confirms the zero-fit second-order law on new categorical values and maps a
  nonuniform stress boundary; ACL-004 is primarily theorem/software positive control
  plus finite-population SNR evidence; ACL-005 is the load-bearing cross-estimator
  transport result and falsifies the support-free diagnostic in its frozen stress
  stratum.
- **Canonicalization:** merge reviewed PR #1 without rewriting history or publishing a
  release. Record the merge SHA as the canonical Phase-I public state.
- **Numerical semantics:** immutable artifacts and lock files retain byte-level SHA-256
  identity. Future analytic registries additionally use pinned CPython/NumPy and frozen
  absolute/relative numerical tolerances; independent environments need mathematical
  equivalence within tolerance, not cross-version bit identity.
- **Allowed files:** `TASK_LEDGER.md`, Phase-II audit/protocol/environment files,
  structural-distance ledgers, and README status. No evidence or old preregistration
  file may change.
- **Acceptance checks:** inspect complete history; verify each ACL-002–005 artifact hash,
  approved SHA, evidence commit bytes, and bundle lock; run all artifact-only verifiers;
  run full pytest, Ruff, coverage, and wheel build; confirm public merge and clean tree.
- **Risks:** promoting ACL-004 from positive control to independent unification evidence;
  conflating byte identity with numerical equivalence; silently changing an old claim;
  publishing a release contrary to repository rules.
- **Stop conditions:** any evidence hash or lock mismatch, public history divergence,
  failing baseline, or need to alter Phase-I evidence.

## ACL-006-MECH — Support-conditioned angular bias and consistency converse

- **Task ID:** `ACL-006-MECH`.
- **Status:** derivation/mechanism construction only; ACL-006 confirmatory targets and
  random streams are forbidden until a later locked public preregistration is audited.
- **Observed mechanism:** ACL-005 split-half means can converge to the estimator's own
  biased expectation. Five of eight rare-cell stress blocks missed truth alignment even
  though all stress landscapes satisfied the stopping rule.
- **Hypothesis:** for the frozen undamped empirical-Fisher plug-in estimator, finite
  categorical support permits an exact zero-fit decomposition of conditional-mean bias
  into missing-identifiable-subspace and observed-support perturbation components, and
  these quantities predict angular error across `N`, joint-cell support, Fisher
  conditioning, and reward geometry.
- **Frozen estimator semantics:** deterministic rewards; true-policy categorical scores;
  fixed-`N` joint multinomial counts; empirical score gradient and Fisher from the same
  counts; centered Moore-Penrose direction with `rcond=1e-12`; no damping, baseline fit,
  clipping, target fit, or state update.
- **Derivation target:** for support pattern `S` and Fisher-range projector `P_S`, use
  the exact identity `hat(d)-d = -(I-P_S)d + (hat(d)-P_S d)`, then average over exact
  multinomial support/count probabilities. Prove separately that two independent sample
  means converge in Fisher cosine to one whenever their common nonzero mean exists,
  whether or not that mean equals the analytic tangent.
- **Independent oracle:** direct count-table score/Fisher accumulation and exhaustive
  enumeration on small fixtures must agree with optimized support enumeration and
  closed-form support probabilities before any preregistration.
- **Allowed files:** this ledger, Phase-II theory/mechanism modules, tests, derivation
  documents, and non-outcome software fixtures. No `preregistrations/ACL-006/` manifest
  or evidence until the mechanism earns a frozen design.
- **Acceptance checks:** failing tests first; exact probability mass; exhaustive versus
  optimized expectation agreement; projector decomposition closure; analytic tangent
  identity; self-consistency converse fixtures; float64 guards; full repository gate.
- **Risks:** treating exact target enumeration as post-outcome fitting; defining support
  only by `Np_min` when context/action factorization matters; hiding reward-offset bias;
  producing a vacuous bound; confusing theorem reproduction with confirmatory breadth.
- **Stop conditions:** damping enters the estimator; an analytic comparator calls sampled
  code; a target outcome is generated; support decomposition fails to close; or a new
  semantic choice is needed after data exist.

### ACL-006 mechanism completion record

- **Outcome status:** theorem/mechanism checkpoint only. No ACL-006 manifest, target
  random stream, confirmatory row, or evidence artifact exists.
- **Exact finite law:** exhaustive four-cell multinomial enumeration computes the
  undamped plug-in direction's expectation, covariance, Fisher angular bias, support
  distribution, and two-term error decomposition without fitted constants.
- **Proved results:** exact support-pattern probabilities; Fisher-orthogonal support and
  observed-support error decomposition; the split-half self-consistency converse; and
  a reward-shift counterexample showing that `N p_min`, support probabilities, and the
  analytic Fisher spectrum do not determine plug-in angular bias by themselves.
- **Independent verification:** optimized tangent-coordinate enumeration agrees with
  direct count-table score/Fisher accumulation; enumerated support mass agrees with
  inclusion-exclusion; decomposition vectors and squared Fisher errors close at frozen
  numerical tolerances.
- **Classification:** the exact finite sum is a zero-fit law for this finite estimator
  family. A compact nonvacuous angular bound remains a candidate, not a proved theorem.
  Any subsequent stochastic execution is theorem/software verification plus a test of
  the preregistered dissociation and reductions, not independent unification evidence.
- **Design-only observations:** pre-preregistration fixtures show that equal `N p_min`
  can have materially different bias when rarity is allocated to context versus action,
  and additive reward shifts can rotate the plug-in expectation while preserving the
  analytic tangent. These observations constrain ACL-006 design but are not outcomes.
- **Verification:** 163 tests passed; Ruff passed; total coverage 84%; wheel build
  passed. ACL-002 through ACL-005 artifact hashes remain unchanged.
- **Recommended next action:** publish this mechanism checkpoint, then freeze new
  ACL-006 values that independently vary effective count, support factorization,
  conditioning, and reward offset before drawing any confirmatory shadows.
