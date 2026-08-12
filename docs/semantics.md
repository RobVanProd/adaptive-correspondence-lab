# Frozen semantics

## Categorical state and mapping

The canonical state is a float64 probability vector on the simplex. The mapping among
the three initial domains is the identity on this vector. Natural-gradient native
state is a centered logit vector; the other native states are probabilities.

With reward `r` frozen during a step of length `eta`, exact replicator flow is

```text
p_i(eta) = p_i(0) exp(eta r_i) / sum_j p_j(0) exp(eta r_j).
```

This equals one multiplicative-weights update. For a categorical policy, the
Euclidean logit gradient is `g = p * (r - <p,r>)`, the Fisher matrix is
`F = diag(p) - p p^T`, and a centered natural direction is `r - mean(r)`. Updating
the logits by that direction yields the same probability map.

Exact log-domain updates require strictly positive probabilities. The explicit Euler
replicator step admits boundary states but rejects any step leaving the simplex.

## Perturbation meanings

- `euler`: target uses an explicit Euler replicator step; epsilon is the step size.
- `reward-bias`: epsilon times a fixed zero-mean direction is added only to the target
  reward.
- `noise`: epsilon is the target-only Gaussian reward standard deviation.
- `delay`: epsilon blends current target reward with the preceding reward; epsilon is
  in `[0,1]`.
- `frequency`: epsilon scales a frozen payoff matrix times the current target state.
- `mutation`: epsilon is the post-update probability of applying a frozen row-
  stochastic mutation kernel.
- `nonstationary`: epsilon scales a deterministic sinusoidal reward direction.
- `finite-population`: epsilon maps to population size `N = ceil(epsilon^-2)` and the
  target distribution is a multinomial frequency vector. Epsilon zero means infinite
  population and no sampling.
- `constraint`: epsilon is a lower probability floor and must be below `1/n`.

The baseline never receives a target-only perturbation. Shared-environment experiments
must state that choice separately because common noise can preserve pathwise equality.

`stochastic_error` is always recorded in canonical update coordinates: the stochastic
next state minus the conditional deterministic next state. It is not a reward-space or
parameter-direction residual.

## Metrics and tolerances

The default discrepancy is L1 distance in canonical probability space. KL divergence
is available only where the second argument is strictly positive. Exact deterministic
equivalence uses absolute tolerance `2e-14` and no relative tolerance in verification.
No stochastic acceptance criterion is inferred from that numerical tolerance.

## Randomness

Each replicate derives an independent seed from a NumPy `SeedSequence`. Compared
systems never share a mutable generator. A step record stores the full JSON-safe
bit-generator state and a compact fingerprint immediately before its stochastic draw,
allowing exact replay from the state or from the run seed and configuration.
