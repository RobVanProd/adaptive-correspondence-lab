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

- **Status:** in progress. This task derives analyses only from the immutable ACL-002
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
