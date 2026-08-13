# Adaptive Correspondence Lab

A deliberately small, CPU-first test bed for a falsifiable question:

> Does a restricted correspondence between adaptive dynamical systems continue to
> make quantitative predictions when its assumptions are violated by a known amount?

The project starts where the mathematics is inspectable. It does **not** claim that
evolution, learning, optimization, and control are generally the same process.

## The first correspondence

For probabilities `p` and a reward vector `r`, the reference experiment compares:

1. replicator dynamics, `dp_i/dt = p_i (r_i - <p,r>)`, using its exact flow while
   `r` is frozen over one step;
2. multiplicative weights, `p'_i ∝ p_i exp(eta r_i)`; and
3. categorical natural gradient, implemented in centered logits with the Fisher
   matrix exposed in the trace.

Under the frozen assumptions, all three next-state maps are the same. Explicit Euler
replicator integration is intentionally separate: its mismatch is a measurable
discretization effect, not an implementation of the exact correspondence.

The repository also includes a pure diagonal-Gaussian natural-gradient optimizer,
a finite-sample rank-mu estimator without evolution paths or hidden clamps, and an
exact/sampled contextual-bandit natural policy gradient. These are escalation rungs,
not part of the initial categorical equivalence claim.

## Quick start

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\python -m pytest
.venv\Scripts\acl demo --output-dir results\demo
```

On macOS/Linux, replace `.venv\Scripts\` with `.venv/bin/`.

Useful commands:

```text
acl equivalence --steps 50 --eta 0.05 --output results/equivalence.json
acl sweep --perturbation mutation --output results/mutation.csv
acl sweep --perturbation euler --output results/euler.csv
acl transport --perturbation reward-bias --output results/transport.json
acl gaussian --steps 30 --output results/gaussian.json
acl bandit --steps 30 --output results/bandit.json
acl verify
acl acl002-validate
```

`acl demo` uses 32 seeds and tiny arrays. Larger sweeps are processed in bounded
chunks, so the default workflow fits comfortably in 32 GiB RAM and needs no GPU.

## What gets recorded

Every step emits a common record containing canonical and native state, analytic
gradient/vector field, Fisher/Bregman geometry, realized update, step size,
stochastic error, potential/regret quantities, constraint checks, numerical guards,
the complete pre-draw RNG state, and a compact RNG-state fingerprint. Results also
contain the full run configuration and software/platform metadata.

See [docs/semantics.md](docs/semantics.md) for the exact contracts and
[docs/experiments.md](docs/experiments.md) for the epsilon-to-delta and transported-
prediction protocols.

## Scientific status

This is research software at version `0.1.0`. Passing tests establishes that the code
matches the frozen equations and deterministic fixtures. It does not establish a new
scientific law. The `evidence/` directory contains only labeled software verification
artifacts until a preregistered experiment is reviewed.

The bundled [software verification](evidence/software-verification.json) and
[categorical trace](evidence/categorical-equivalence.json) identify the exact clean
source commit that produced them. The [mutation curve](evidence/mutation-stability.csv)
is intentionally labeled as a deterministic format example rather than a research
result.

## Research program result

ACL-002 through ACL-008 have immutable confirmatory artifacts in `evidence/`.
Phase I established local categorical mutation laws, a finite-lambda Gaussian expected
direction, and no-refit transport of a normalized diagnostic into contextual-bandit
control, with rare-cell failures. ACL-006 then confirmed exact support-conditioned bias
and the split-consistency/truth dissociation while falsifying scalar support-only
reductions.

[FINAL_SYNTHESIS.md](FINAL_SYNTHESIS.md) gives the terminating claim, exact hashes,
negative results, exclusions, and open theorem. The machine-readable
[bridge ledger](BRIDGE_LEDGER.json) records every edge. No result supports a universal
identity among adaptive systems, sequential or neural RL, or support-free transport.
[PHASE_II_SYNTHESIS.md](PHASE_II_SYNTHESIS.md) is the terminal breadth synthesis;
[PHASE_II_OUTCOME.json](PHASE_II_OUTCOME.json) is its machine-readable provenance and
stopping record.
ACL-007 transported the finite-mean/dissociation diagnostic into sequential Euclidean
Bayesian particle filtering. ACL-008 transported ACL-003's zero-fit second-order local
mutation law into Burg log-barrier mirror geometry while finding a nonuniform stress
radius. Phase II therefore supports a correspondence lattice: distinct local-curvature
and stochastic-estimator subgraphs with explicit bridges and boundaries, not one
universal adaptive process.
