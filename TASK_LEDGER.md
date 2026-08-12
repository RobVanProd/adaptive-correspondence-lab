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
