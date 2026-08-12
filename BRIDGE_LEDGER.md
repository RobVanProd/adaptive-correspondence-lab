# Adaptive correspondence bridge ledger

This ledger distinguishes coordinate maps from reusable predictive laws. The
machine-readable source is [`BRIDGE_LEDGER.json`](BRIDGE_LEDGER.json).

## Categorical simplex correspondence

- **Map:** an interior categorical probability vector is shared by multiplicative
  weights, exact replicator flow under fixed rewards, and categorical natural gradient.
  Reward, fitness, and the categorical objective gradient are the mapped driving
  quantities; negative-entropy mirror geometry matches categorical Fisher/Shahshahani
  geometry.
- **Scope:** fixed support, fixed reward, interior states where KL/Fisher expressions
  are used, and the exact exponential update or its explicitly stated continuous-time
  limit.
- **Content:** clean mapped trajectories must agree. Under the ACL-002 mutation
  perturbation, the analytic tangent must predict first-order endpoint L1 response.
- **Stability:** **preregistered confirmation within the frozen deterministic
  benchmark.** The zero-fit first-order target gate passed. Posthoc analysis finds a
  predominantly negative, locally quadratic, landscape-dependent remainder for
  horizons above one. ACL-003 then confirmed the analytic zero-fit second-order
  truncation on 16 entirely new state, reward, and mutation catalog values: median
  maximum-local error `0.148431%`, Type-7 Q90 `0.738712%` through epsilon `0.01`.
- **Transport:** **preregistered confirmation within family, without target refit.**
  The frozen source-median alpha `0.9951356698171323` passed on 12 held-out target
  combinations. ACL-003 separately passed a no-fit second-order prediction on new
  catalog values. Both are within the categorical class, not cross-class transport.
- **Failure boundary:** exploratory T=20 pooled target median absolute error grows from
  `0.154%` strict to `2.626%` extended-local and `22.640%` stress, with heterogeneous
  empirical radii. A state-aware second-order truncation sharply improves local errors
  and every target's 5%/10% radius, but worsens stress Q90 from `62.17%` to `77.66%`
  and reaches `778.42%` worst-case relative error on ACL-002. On ACL-003's new values,
  second order improves all 16 targets at epsilon `0.03` but only 13 at `0.1`; its
  T=20 stress median is `3.652%`, Q90 `36.875%`, and maximum `114.694%`.
- **Evidence:** ACL-002 artifact SHA-256
  `4d08e85b927a5d78a29078ff0d6549225d98069b20186b754629464739f29d74`;
  posthoc summary SHA-256
  `116b8c6ec092dfdcff6a53e39f07a46fbbf8b75615d6f36e11bfed1abff14922`;
  second-order summary SHA-256
  `d7533c3f3b5e0941e28cddcba58ce4106825c938f7244c24bd8f98c8e9403474`;
  ACL-003 evidence SHA-256
  `1f80c3f5aba4089c67bbfec1ddd6eff53f7a6d42c658436dff0f7c82a1cf8c99`.
- **Status:** theorem reproduction + preregistered first-order and second-order
  confirmation within the categorical class, with an explicit nonuniform stress
  boundary. Cross-class transport remains unresolved.

## Gaussian natural gradient ↔ finite-sample rank-mu

- **Map:** Gaussian mean/covariance parameters, score-function gradient, Gaussian
  Fisher metric, and separate mean/covariance rank-mu tangent blocks.
- **Scope:** pure Gaussian family; no evolution paths or CSA; frozen population,
  parents, weights, parameterization, and learning rates.
- **Content:** the independently constructed analytic Fisher direction should align
  with the converged conditional expected finite-lambda rank-mu update in each block.
- **Stability:** the finite-lambda conditional expected direction has now been derived
  independently from binomial rank utility, Gaussian score, and inverse Fisher blocks.
  It matches a tensor-product score integral on toy fixtures. This is theorem/software
  reproduction; confirmatory shadow alignment remains unresolved.
- **Transport:** unresolved; no quantity has yet been frozen for no-refit transfer.
- **Failure boundary:** unresolved.
- **Status:** unresolved. Existing software is an escalation rung, not scientific
  evidence for this edge.

## Categorical natural gradient ↔ finite-state control

- **Map:** categorical policy probabilities, expected contextual advantage, and the
  categorical policy Fisher tangent.
- **Scope:** exact finite contexts/actions, analytic policies, no PPO, and no neural
  function approximation.
- **Content:** this becomes a predictive bridge only if a quantity declared before
  control outcomes predicts a held-out control response without target refitting.
- **Stability:** unresolved.
- **Transport:** unresolved.
- **Failure boundary:** if no earlier-class quantity can be frozen before control
  outcomes, this remains a coordinate relation rather than a reusable law.
- **Status:** unresolved; control experimentation is deferred until the Gaussian edge
  is understood.
