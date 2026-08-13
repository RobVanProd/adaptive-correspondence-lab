# Phase II synthesis: breadth, boundaries, and the correspondence lattice

## Question and answer

Phase II asked:

> Is the observed predictive structure an instance of a broad theory of adaptive
> dynamics, a sharply bounded Fisher/geometric island, or several correspondence
> islands with no useful global unification?

**Outcome U3: correspondence lattice.**

The evidence rejects both extremes. The original relations are more than coordinate
analogies: quantitative zero-fit or no-refit predictions survived in new geometries,
estimator families, semantics, and temporal structures. But no nontrivial single law
spans all tested systems. Instead the repository now contains two predictive subgraphs:

1. a local deterministic retraction-sensitivity class connecting entropy/Fisher
   categorical dynamics to Burg log-barrier mirror dynamics; and
2. a finite-sample conditional-mean diagnostic class linking Gaussian rank-mu,
   empirical-Fisher contextual control, and sequential particle-filter inference by
   pairwise, explicitly scoped bridges.

The subgraphs are not one law in disguise. The stochastic normalized-mean score is
undefined on deterministic exact dynamics because its covariance scale is zero. The
second-order epsilon-curvature law is undefined on the frozen-state estimator tests
because they do not supply a shared epsilon-parameterized dynamical perturbation. Making
either quantity global would require adding target-specific structure, not transporting
an established prediction.

No universal adaptive process is supported.

## Immutable experiment record

ACL-001 remains software/theorem reproduction only. ACL-002 through ACL-005 are the
immutable Phase-I foundation; ACL-006 through ACL-008 are Phase II.

| Experiment | Final preregistration SHA | Evidence commit | Evidence SHA-256 | Result and role |
|---|---|---|---|---|
| ACL-002 | `3f6a935942f43c7d3055582d123e58af5bf3f38b` | `5caf47b510d70564415354f34ba729ff505f7ed4` | `4d08e85b927a5d78a29078ff0d6549225d98069b20186b754629464739f29d74` | PASS: categorical first-order mutation law |
| ACL-003 | `501464f3f6be07f6d813d94aefb818c461a3d5c7` | `b15d77600369d559cb586a3bb54924737758e038` | `1f80c3f5aba4089c67bbfec1ddd6eff53f7a6d42c658436dff0f7c82a1cf8c99` | PASS: zero-fit second-order law on new categorical values |
| ACL-004 | `3ba4be7ce1460a40c4ef0879018df58947c36edb` | `355dd97472da4230eff877b9a3c8c7c4626057cd` | `3f97f7c4debbd65014e6ee337d9a8990500bad38ce0dfef2e8ad3048c74cd91a` | PASS: Gaussian finite-lambda expected-direction control |
| ACL-005 | `c3ebc07a41e8dbb84a24c68cdbb4f75c36108c5b` | `24d577f8a1d7bc6f4f45250f4bab3d5b2b925aeb` | `5400a12392609f5cdf79a8b4b380f84ad11e68330f8ee93f653439129aa5db5b` | PASS in regular support; support-free version falsified by frozen stress |
| ACL-006 | `a8b42042e397f1422866a0ca9496ee07abe0a42a` | `c94890dc8f361c0309802c0ef0173ec84e814d3d` | `740c541bbd69db77f6d02327ded34765a37345f907b048f8d3f3a91aebc23918` | PASS: exact support-bias mechanism and self-consistency converse |
| ACL-007 | `0b807af1d0428340f1e5267b1e41f6e636b49d29` | `c90954960b0fa099741ed9f35a61c5153b54c923` | `54793bcb3a40d914bce2b5a567f6d25e638a75edf4a55ef724e156a93d372133` | PASS: no-refit diagnostic transport into sequential inference |
| ACL-008 | `086c8187caa641a7699ee07cff540a7d8e77ba18` | `c972d886edddc2dd36d60bd8229640a8eec405db` | `856be5ff685d65e19e029fc243a2ef40170ddf64a8b035dd1b543b484e0eba4f` | PASS: no-refit second-order transport into Burg geometry |

Phase-II artifact reports are also immutable:

| Experiment | Report commit | Report summary SHA-256 |
|---|---|---|
| ACL-006 | `c8d599fe09887e22fe02f92a27bcc8c13ac4baf4` | `0748482b3796b861267fdb5781bab11605cfc82263e4a6fdbd206df4b96acd6c` |
| ACL-007 | `03424e3bb260ebe45e4fec09a2f3cf60fe5085b6` | `b03ba716bcfd8bfbb1e73a64202b11d605812d081c88d2d9077f334032d34166` |
| ACL-008 | `66dd4c1b102fe30024e33d1c5ef33bc0269a5980` | `403c69904842d0d09ef6d9091b3e5133d684e0a25a084748b5e415770a84b0a1` |

The machine-readable terminal record is
[`PHASE_II_OUTCOME.json`](PHASE_II_OUTCOME.json). The complete edge and structural
classifications remain in [`BRIDGE_LEDGER.json`](BRIDGE_LEDGER.json) and
[`STRUCTURAL_DISTANCE_LEDGER.json`](STRUCTURAL_DISTANCE_LEDGER.json).

## Phase-II experiments

### ACL-006: support is a mechanism, not a scalar

ACL-006 formalized the nonlinear finite-sample bias of

\[
\widehat d=\widehat F^+\widehat g
\]

in a three-action contextual-bandit block. Exact count-table enumeration fixed the
estimator mean, covariance, support patterns, and angular quantities before an
independent full-coordinate stochastic path.

All frozen gates passed. The maximum standardized full/half direction score was
`1.707381`; Type-7 full-score median and Q90 were `0.609557` and `1.272615`.
Seven predeclared targets reached split-half Fisher cosine at least `0.999995890` while
truth cosine was as low as `0.483875705`.

The important negative results were exact, not posthoc fits:

- `N p_min` alone cannot determine angular bias when context/action support is
  factorized differently;
- support probabilities plus analytic Fisher spectrum cannot determine bias without
  reward/baseline geometry; and
- split-half self-consistency converges to one around any common nonzero estimator mean,
  regardless of its alignment with truth.

ACL-006 did not add a structural class. It mapped the ACL-005 boundary.

### ACL-007: transport beyond optimization and Fisher geometry

ACL-007 copied ACL-006's complete native-metric standardized-mean, angular-envelope,
dissociation, schedule, and contrast rule without changing a threshold. The target was
a sequential bootstrap particle filter for a three-state hidden Markov model:

- estimator family changed from empirical-Fisher pseudoinverse to sequential particle
  propagation, likelihood weighting, and resampling;
- native geometry changed from Fisher to centered Euclidean belief geometry;
- semantics changed from reward optimization to Bayesian inference; and
- temporal structure changed from a frozen one-step estimator to repeated filtering.

The overall result passed. Maximum standardized score was `1.815369`; Type-7 median
and Q90 were `0.796407` and `1.212732`; all nine frozen contrasts passed. The three
dissociation targets reached split-half cosine at least `0.999994349` while their truth
cosines were only `0.344907` to `0.378970`.

The full benchmark included three negatively aligned estimator means, with minimum
truth cosine `-0.999797336`. This is not a contradiction. It is the point of the
transported diagnostic: it predicts estimation of an estimator's own expectation and
separates that fact from truth alignment. ACL-007 establishes breadth for a
bias/variance diagnostic, not a shared ideal vector field.

The runner was invoked exactly once. The shell harness reported a five-second timeout
after the atomic canonical artifact had completed. The complete file was preserved,
reconstructively validated, and committed untouched; no second invocation occurred.

### ACL-008: a mechanism-bearing cross-geometry law

ACL-008 changed the mirror potential from Shannon negative entropy to the Burg log
barrier. Its Hessian is `diag(1/p_i^2)`, and its constrained rational update is not
multiplicative weights. The state space, linear objective, and mutation operator were
retained deliberately to isolate geometry.

For a smooth Burg retraction `F_B` and `B=M-I`, the derived row-vector recurrences are

\[
s_{t+1}=D F_B(p_t)[s_t]+F_B(p_t)B,
\]

\[
u_{t+1}=D F_B(p_t)[u_t]+D^2F_B(p_t)[s_t,s_t]
+2D F_B(p_t)[s_t]B.
\]

ACL-008 copied ACL-003's entire practical rule without refitting: the finite vector
prediction `||epsilon*s_T+epsilon^2*u_T/2||_1`, local epsilon range through `0.01`,
maximum-within-landscape reducer, and Type-7 median/Q90 gates of `0.10/0.20`.

All 16 new-value targets passed. Median error was `0.001965748`; Q90 was
`0.005559435`; the independent polynomial-root oracle agreed to
`2.3314683517e-15`. Second order improved all targets through epsilon `0.03`.

The large-perturbation boundary was again nonuniform. At epsilon `0.1`, second order
improved only 13/16 targets; median relative error was `0.289224350` and the maximum
was `2.772975981` on `B05`. The transported claim is local, not global.

## Correspondence lattice

### Local retraction-sensitivity subgraph

The nodes are exact categorical entropy/Fisher dynamics and Burg mirror dynamics. The
edge carries a zero-fit second-order response to an affine post-step perturbation. Its
content is quantitative through a tested local radius, and its boundary is curvature-
and-state-dependent failure at larger epsilon.

The categorical node itself contains an exact coordinate island: multiplicative
weights, fixed-fitness exact replicator flow, and categorical natural gradient share
the same exponential update under their frozen scope. Burg is outside that exact map;
only the abstract derivative recurrence and practical local radius transport.

### Finite-sample conditional-mean subgraph

The nodes are Gaussian rank-mu, contextual-bandit empirical-Fisher NPG, and sequential
bootstrap particle filtering. Its edges carry two related but nonidentical diagnostics:

- Gaussian to control: disjoint-half Fisher cosine `>=0.98` followed by analytic
  Fisher cosine `>=0.99` under adequate support;
- control to particle filtering: native-metric standardized mean accuracy plus explicit
  split-consistency/truth dissociation.

The common structure is an ideal object `d`, estimator mean `m`, covariance `Sigma`,
and native metric. The data distinguish uncertainty about `m` from structural bias
`m-d`. The boundary is support, identifiability, and model/estimator bias—not merely
sample variance.

### Why this is U3 rather than U1

U1 required a shared no-refit law across at least three structurally distinct classes.
Phase II produced connected pairwise transport across more than three classes, but not
one shared quantitative law:

- ACL-004/005's Fisher-cosine implication is false without adequate support and is not
  defined in Burg geometry without inventing a Fisher structure;
- ACL-006/007's standardized score divides by an estimator covariance scale, which is
  zero for the deterministic exact categorical and Burg experiments;
- ACL-003/008's epsilon-curvature response requires a specified smooth perturbed
  trajectory, which the frozen-state Gaussian/control estimator tests do not contain.

Replacing these with the statement “bias plus variance plus curvature” would be a useful
bookkeeping identity, not a frozen quantitative prediction. The evidence therefore
supports an atlas of scoped laws, not a global normalization.

U2 is also too narrow: predictive transport did not remain confined to a Fisher island.
ACL-007 passed in Euclidean sequential inference, and ACL-008 passed under a non-Fisher
Burg geometry.

## Proved results and theorem candidates

Proved or constructively verified results include:

- exact categorical MWU/replicator/natural-gradient correspondence under fixed rewards;
- first- and second-order row-vector mutation sensitivity recurrences;
- ACL-006's Fisher-orthogonal support-loss/observed-support decomposition;
- exact multinomial support-pattern probabilities;
- the self-consistency converse for independent finite-moment means;
- the reward-shift counterexample showing reward geometry is necessary;
- the exact finite count-state Markov construction for the bootstrap particle filter;
- uniqueness and implicit first/second derivatives of the constrained Burg mirror step.

Narrowed theorem candidates are:

1. a support-conditioned plug-in angular-bias bound that includes support
   factorization, reward/baseline geometry, and empirical-Fisher perturbation;
2. a local retraction theorem with an explicit finite-horizon remainder bound in terms
   of derivative operator norms, third derivatives, boundary margin, and perturbation
   size; and
3. an atlas theorem specifying when bias, sampling, and curvature components can be
   compared invariantly across native geometries, and when such comparison is
   impossible without additional structure.

## Positive findings

- Local first- and second-order categorical perturbation laws predicted held-out and
  new-value targets without refitting.
- The second-order practical radius transported from entropy/Fisher to Burg mirror
  geometry with new catalog values and unchanged gates.
- Gaussian expected rank-mu directions reproduced independent analytic Fisher
  directions blockwise.
- A Fisher directional diagnostic transported to adequately supported contextual
  control targets.
- A native-metric standardized-mean/dissociation diagnostic transported from control
  into sequential Bayesian inference despite changed metric, semantics, and time.

## Negative findings and boundaries

- First- and second-order local expansions have no uniform tested stress radius.
- A finite Gaussian population, especially its covariance tangent, can be noisy even
  when its conditional mean is highly aligned.
- The support-free `0.98 -> 0.99` rule is false: ACL-005 rare-cell targets converged
  internally while truth alignment failed.
- `N p_min` is not a sufficient scalar law.
- Support and Fisher spectrum are insufficient without reward/baseline geometry.
- Split-half convergence cannot certify truth; ACL-007 includes nearly opposite stable
  estimator means.
- No existing quantitative estimand is nontrivially defined across every tested class.

No primary confirmatory experiment failed. The negative conclusions arise from
predeclared stress strata, exact counterexamples, or structural type boundaries—not
from changing criteria after outcomes.

## Strongest surviving claims

1. Smooth adaptive retractions under affine post-step perturbations share a calculable
   first/second variational structure. For the tested entropy and Burg mirrors, that
   zero-fit structure is quantitatively predictive through epsilon `0.01`, with a
   nonuniform larger-perturbation boundary.
2. For finite-moment stochastic adaptive estimators with independently known mean and
   covariance in a declared native metric, standardized mean error and split-half
   agreement quantify uncertainty about the estimator expectation, not truth. This
   diagnostic transported from empirical-Fisher control to sequential particle
   filtering without target refitting.
3. Within the studied Fisher-natural estimator subset, expected-direction alignment
   transports under adequate support; support-free truth certification is false.

These claims form a correspondence lattice. They are reusable local theories, not a
universal identity among learning, evolution, optimization, control, and inference.

## What the evidence does not justify

- one universal adaptive objective, geometry, vector field, or normalization;
- transport of a single quantitative law across every class in the ledger;
- population-level confidence beyond the deterministic benchmark units;
- uniform large-perturbation Taylor accuracy;
- truth alignment from estimator self-consistency;
- support laws based only on expected minimum count;
- damping, PPO, neural policies, continuous control, LLM fine-tuning, or GPU claims;
- carrying ACL-007's PASS into a claim that Bayesian filtering optimizes reward;
- carrying ACL-008's PASS into a claim that all mirror geometries share the same
  finite-radius constants.

The program has not earned an LLM fine-tuning experiment. It has no single pre-fine-
tune quantity with demonstrated transport across both correspondence subgraphs.

## Strongest mathematical statement supported

For the finite systems tested, adaptive correspondences have predictive content when
the transported object is typed by its mechanism:

\[
\text{smooth retraction} + \text{affine perturbation}
\Longrightarrow
\text{zero-fit local variational prediction},
\]

and

\[
\text{finite-moment estimator} + (m,\Sigma,G)
\Longrightarrow
\text{native-metric mean-uncertainty prediction distinct from bias }m-d.
\]

Both implications survived preregistered no-refit transport and both have observed
boundaries. The evidence does not support collapsing them into one process.

## Most important open theorem

The most valuable next theorem is a typed local error atlas. For

\[
\theta^+=R_\theta\!\left(\eta\,[v(\theta)+b(\theta)+\xi]\right),
\qquad E[\xi]=0,
\]

derive an explicit native-geometry bound separating

\[
\text{transport error}
\le
\text{identifiability/support bias}
+\text{sampling term}
+\text{retraction curvature term}
+\text{controlled remainder}.
\]

The theorem must state the data needed to compare these terms across geometries,
reduce to ACL-006's support decomposition and ACL-003/008's variational recurrence in
their respective scopes, and prove impossibility when those data are absent. A
nonvacuous bound of this form—not another equation-level analogy—would be the route from
the present lattice toward a broader theory.

## Stopping decision

Phase II terminates at **Outcome U3**. Additional small systems could refine the atlas,
but they are not required to answer the breadth question now: predictive correspondence
is real, restricted, mechanism-typed, and organized as a lattice. It is broader than a
Fisher-only island and narrower than a unified adaptive theory.
