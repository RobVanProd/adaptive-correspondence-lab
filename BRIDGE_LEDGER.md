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
- **Stability:** **preregistered confirmation.** The finite-lambda conditional expected
  direction was derived independently from binomial rank utility, Gaussian score, and
  inverse Fisher blocks. ACL-004 passed on 12 landscapes: all converged at 4096
  shadows, with minimum mean/covariance Fisher cosines `0.999952324` and `0.999552200`
  against the frozen `0.99` gates.
- **Transport:** **preregistered cross-class confirmation.** ACL-005 carried the
  complete ACL-004 normalized rule—disjoint-half block cosine `0.98`, analytic block
  cosine `0.99`, and the same replication schedule—into finite-state control without
  target refitting. All 20 regular target blocks passed; the minimum was `0.999841630`.
- **Failure boundary:** H2 is an expected-direction result, not a guarantee for one
  finite population. Descriptive H1 single-shadow median cosine spans `0.953–0.957`
  for mean and `0.534–0.717` for covariance; covariance alignment is positive in
  `84.1%–87.9%` of shadows. Lambda scaling was not studied.
- **Evidence:** ACL-004 evidence SHA-256
  `3f97f7c4debbd65014e6ee337d9a8990500bad38ce0dfef2e8ad3048c74cd91a`;
  report SHA-256
  `b4d9864b6ab935aa39bc98ab1c144e13030ebad57931eaf4bc1cbcbaf3d2e019`.
- **Status:** theorem reproduction + preregistered finite-lambda conditional-mean
  confirmation + one successful restricted cross-class transported diagnostic.

## Categorical natural gradient ↔ finite-state control

- **Map:** categorical policy probabilities, expected contextual advantage, and the
  categorical policy Fisher tangent.
- **Scope:** exact finite contexts/actions, analytic policies, no PPO, and no neural
  function approximation.
- **Content:** ACL-005 declared before control outcomes that the ACL-004 blockwise
  stopping/alignment rule would predict held-out control conditional means without
  target refitting.
- **Stability:** **preregistered confirmation within the regular support scope.** All
  10 regular landscapes stopped at 4096 and all 20 context blocks exceeded `0.99`;
  the minimum was `0.999841630`.
- **Transport:** **PASS without target refit.** The source `0.98` convergence and
  `0.99` analytic-alignment rule was copied unchanged from ACL-004.
- **Failure boundary:** all four rare-cell stress landscapes also met the stopping
  rule, yet 5 of 8 stress context blocks fell below `0.99`; the minimum was
  `0.055699373`. Half-mean self-consistency is not enough when empirical Fisher blocks
  are support-deficient or nearly so.
- **Evidence:** ACL-005 evidence SHA-256
  `5400a12392609f5cdf79a8b4b380f84ad11e68330f8ee93f653439129aa5db5b`;
  report SHA-256
  `b5d310e9a32c059cb192e4f1001556b7a9d60ca98cc6319a1d67326246c13084`.
- **Status:** theorem reproduction + preregistered cross-class confirmation with a
  predeclared rare-cell boundary.

## Gaussian rank-mu → contextual-bandit normalized law

- **Map:** in both classes, compare the conditional mean of a finite-sample stochastic
  tangent with an independently constructed analytic Fisher-natural tangent. Normalize
  each independent Fisher block by Fisher cosine and diagnose estimator convergence by
  the cosine between disjoint-half means.
- **Scope:** ACL-004's pure diagonal-Gaussian rank-mu source; ACL-005's two-context,
  three-action categorical-policy target at `N=128`; undamped empirical-Fisher
  pseudoinverse; regular minimum expected joint-cell count at least 4; frozen schedule.
- **Content:** after every target block reaches disjoint-half cosine `0.98`, every
  regular analytic block cosine should reach `0.99`.
- **Stability:** all regular targets passed, with minimum `0.999841630`.
- **Transport:** the complete diagnostic and schedule moved from Gaussian to control
  without target refitting.
- **Failure boundary:** the same statement is false without adequate support. Five of
  eight rare-cell stress blocks failed despite apparent convergence.
- **Status:** preregistered cross-class confirmation, restricted by support coverage.

## ACL-006 support-conditioned plug-in bias

- **State map:** one interior three-action policy block plus an explicit outside-context
  sampling category.
- **Native geometry:** the analytic categorical Fisher metric on the centered-logit
  tangent.
- **Ideal tangent:** `d = F^+ g = r - mean(r) 1`, independently constructed from the
  analytic score gradient and Fisher metric.
- **Realized estimator:** `hat(d) = hat(F)^+ hat(g)` from one fixed-`N` multinomial
  count table, using the same samples for both empirical objects and no damping.
- **Scope:** deterministic rewards, true-policy scores, three interior actions, fixed
  context probability, no fitted baseline, clipping, state update, or target refit.
- **Transported prediction:** **preregistered confirmation.** Exact count-table moments
  freeze the conditional mean, covariance-normalized Fisher error, angular envelope,
  and split-half/truth dissociation before an independent PCG64 full-coordinate path.
  All three gates passed: maximum standardized direction score `1.707381`, full-score
  Type-7 median `0.609557`, and Q90 `1.272615`. Seven predeclared targets reached
  half-mean cosine at least `0.999995890` while truth cosine was as low as `0.483875705`.
- **Failure boundary:** proved pre-outcome counterexamples show that `N p_min` alone is
  insufficient across support factorizations and that support/Fisher spectrum without
  reward geometry is insufficient under additive reward shifts.
- **Falsifier:** a locked target outside its five-score exact-mean/angular envelope, or
  failure of the predeclared split-consistent but truth-misaligned stratum at the fixed
  budget.
- **Evidence:** preregistration SHA `a8b42042e397f1422866a0ca9496ee07abe0a42a`;
  evidence SHA-256
  `740c541bbd69db77f6d02327ded34765a37345f907b048f8d3f3a91aebc23918`.
- **Status:** theorem/software reproduction plus preregistered mechanism validation. It
  remains inside the ACL-005 estimator family and adds no independent breadth class.

## ACL-006 diagnostic → sequential Bayesian particle filtering

- **State map:** a frozen-state stochastic adaptive tangent maps to a terminal posterior-
  belief displacement from the initial belief.
- **Native geometry:** categorical Fisher in the source, centered Euclidean belief
  geometry in the target.
- **Ideal tangent:** source analytic natural direction versus target exact true-model
  Bayes posterior displacement.
- **Realized estimator:** empirical-Fisher pseudoinverse shadows versus a finite labeled-
  particle bootstrap-filter displacement.
- **Scope:** independent shadows, nonzero ideal and mean directions, finite native-metric
  second moments, independently exact mean/covariance, fixed schedule, and no target fit.
- **Transported prediction:** **preregistered confirmation.** ACL-007 copied ACL-006's
  complete standardized mean-error and split-consistency/truth-dissociation rule without
  changing thresholds, schedule, or contrast gap. Sixteen frozen targets span two new
  HMMs and correct, reversed-observation, flat-observation, and missing-observation
  particle filters. All three components passed; the maximum score was `1.815369`,
  Type-7 median `0.796407`, and Q90 `1.212732`.
- **Failure boundary:** stable estimator bias survives the diagnostic by design. Three
  targets had negative truth alignment, down to `-0.999797336`, while the locked
  dissociation cases reached half cosine at least `0.999994349`. The rule predicts
  finite-mean estimation and bias/variance dissociation, not truth alignment or a common
  detailed bias mechanism.
- **Falsifier:** any locked target outside the copied gates after exact target moments and
  the independent labeled-particle simulator have been verified.
- **Evidence:** preregistration SHA `0b807af1d0428340f1e5267b1e41f6e636b49d29`;
  evidence commit `c90954960b0fa099741ed9f35a61c5153b54c923`; artifact SHA-256
  `54793bcb3a40d914bce2b5a567f6d25e638a75edf4a55ef724e156a93d372133`.
- **Status:** preregistered no-refit cross-class confirmation outside the Fisher-natural
  family. It adds real breadth but is not alone a general theory of adaptive dynamics.

## Entropy/Fisher mirror sensitivity â†’ Burg mirror sensitivity

- **State map:** retain the interior probability simplex while changing the mirror
  potential from Shannon negative entropy to the Burg log barrier.
- **Native geometry:** `diag(1/p)` entropy/Fisher Hessian in the source versus
  `diag(1/p^2)` Burg Hessian in the target; the target update is not exponential.
- **Ideal vector field:** exact constrained linear-reward mirror step in each geometry.
- **Realized perturbation:** the same post-step row-stochastic mutation operator
  `I+epsilon(M-I)`.
- **Scope:** interior finite simplex, fixed reward and step size, smooth constrained
  mirror map, finite horizon, and no projection or clipping.
- **Transported prediction:** mechanism candidate, not yet preregistered. Copy ACL-003's
  zero-fit second-order L1 prediction, epsilon regions, maximum-within-landscape reducer,
  and 10%/20% Type-7 median/Q90 gates unchanged.
- **Failure boundary:** unresolved. Burg curvature and approach to its barrier may make
  ACL-003's epsilon `0.01` practical radius entropy-specific despite the formal Taylor
  recurrence.
- **Falsifier:** new-value Burg targets violate the copied gates or fail to improve over
  first order throughout the frozen local region.
- **Status:** proved local mechanism plus unresolved non-Fisher transport candidate.
