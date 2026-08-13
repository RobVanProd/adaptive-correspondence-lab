# Theorem ledger

The machine-readable source is [`THEOREM_LEDGER.json`](THEOREM_LEDGER.json).

## Proved before ACL-006 outcomes

- **Support decomposition:** each count-table error splits into an analytically missing
  Fisher-subspace component and an observed-support perturbation component. The two are
  Fisher-orthogonal and their squared errors add.
- **Support-pattern law:** fixed-`N` multinomial support probabilities are exact by
  inclusion-exclusion; rank deficiency for three actions has a closed finite formula.
- **Self-consistency converse:** split-half mean alignment converges to one around any
  common nonzero estimator expectation, whether biased or not. Split consistency alone
  cannot certify truth.
- **Reward-shift necessity:** additive reward shifts preserve the analytic centered
  tangent but generally change the plug-in expectation. Support and Fisher conditioning
  alone cannot determine bias for the unbaselined estimator.

## Candidate, not yet a theorem

A compact nonvacuous angular bound combining support loss with whitened empirical-Fisher
perturbation remains open. ACL-006 confirmed the exact finite multinomial zero-fit law,
but falsified both `N p_min` alone and support plus analytic Fisher spectrum without
reward/baseline geometry. Any surviving compact theorem must distinguish context/action
support factorization and include reward geometry.

## ACL-007 sequential inference mechanism

For a finite three-state hidden Markov model and fixed particle count, the bootstrap
particle filter is an exact finite Markov chain over count compositions. Grouped
transition-polynomial convolution followed by likelihood-weighted multinomial
resampling produces each count kernel. Propagating the initial multinomial count law
therefore yields the exact terminal mean, covariance, and support-loss probabilities.
Small fixtures verify the construction against brute-force labeled paths and a separate
labeled-particle simulator.

## ACL-008 non-Fisher mirror mechanism

The constrained Burg log-barrier mirror step has a unique interior dual normalizer. Its
first and second directional derivatives follow by implicit differentiation of the
simplex constraint. Combining those derivatives with post-step mutation gives the same
abstract chain/product-rule recurrence as ACL-003, but with a non-Fisher Hessian and a
non-exponential update. This proves the local formal expansion; whether ACL-003's
practical finite-epsilon radius transports was the empirical question. ACL-008 answered
yes through epsilon `0.01` on 16 new Burg targets, while its epsilon-`0.1` stress error
remained nonuniform.

## Open after Phase II: typed local error atlas

For `theta+ = R_theta(eta[v+b+xi])`, derive a native-geometry bound separating
identifiability/support bias, sampling uncertainty, retraction curvature, and a
controlled remainder. It should reduce to the ACL-006 support decomposition and the
ACL-003/008 variational recurrence in their scopes, and prove when comparison is
impossible because the required covariance, perturbation family, or geometry map is
absent. This is a candidate theorem, not a supported global law.
