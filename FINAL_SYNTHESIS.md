# Final synthesis: adaptive correspondence research program

## Original question

Do restricted mathematical correspondences between adaptive dynamical systems support
quantitative predictive laws that survive controlled violations of their assumptions
and transport to genuinely different system classes without target refitting, or are
they only local or coordinate-level equivalences?

## Answer

**Outcome A: predictive unification survives in a restricted form.**

The evidence supports neither a universal adaptive process nor a merely cosmetic
coordinate analogy. It supports a smaller reusable structure:

1. On a categorical simplex, exact mapped dynamics admit quantitative first- and
   second-order response laws under mutation. Those laws predict held-out systems
   locally without target fitting and fail nonuniformly at larger perturbations.
2. In a pure Gaussian rank-mu system, the conditional expected finite-population
   tangent aligns with an independently derived Gaussian Fisher-natural direction.
3. A dimensionless blockwise diagnostic from that Gaussian result transported,
   unchanged and without target refitting, to a finite-state contextual-bandit natural
   policy-gradient estimator when joint context-action support was adequate.
4. The transported diagnostic failed in predeclared rare-cell stress cases even after
   its internal half-mean convergence criterion passed. Adequate sampling support is
   therefore a substantive scope condition, not an implementation detail.

This satisfies the program's stopping rule: more computation is not needed to decide
whether any restricted cross-class predictive content exists. It does. Its present
boundary is also visible.

## Preregistered experiments and immutable evidence

ACL-001 was the software/theorem-reproduction substrate, source checkpoint
`48d3c5c5ef96c59fe18f6cd5b3d27a44d0fcccc6`; its artifacts are not scientific
confirmation.

| Experiment | Frozen claim | Final preregistration SHA | Evidence commit | Artifact SHA-256 | Result |
|---|---|---|---|---|---|
| ACL-002 | First-order categorical mutation response; zero-fit and frozen source calibration | `3f6a935942f43c7d3055582d123e58af5bf3f38b` | `5caf47b510d70564415354f34ba729ff505f7ed4` | `4d08e85b927a5d78a29078ff0d6549225d98069b20186b754629464739f29d74` | PASS |
| ACL-003 | Zero-fit second-order categorical mutation response on entirely new catalog values | `501464f3f6be07f6d813d94aefb818c461a3d5c7` | `b15d77600369d559cb586a3bb54924737758e038` | `1f80c3f5aba4089c67bbfec1ddd6eff53f7a6d42c658436dff0f7c82a1cf8c99` | PASS |
| ACL-004 | Finite-lambda Gaussian rank-mu conditional-mean alignment with an independent analytic Fisher direction | `3ba4be7ce1460a40c4ef0879018df58947c36edb` | `355dd97472da4230eff877b9a3c8c7c4626057cd` | `3f97f7c4debbd65014e6ee337d9a8990500bad38ce0dfef2e8ad3048c74cd91a` | PASS |
| ACL-005 | No-refit transport of ACL-004's complete blockwise diagnostic into contextual-bandit control | `c3ebc07a41e8dbb84a24c68cdbb4f75c36108c5b` | `24d577f8a1d7bc6f4f45250f4bab3d5b2b925aeb` | `5400a12392609f5cdf79a8b4b380f84ad11e68330f8ee93f653439129aa5db5b` | PASS |

Pre-outcome audits superseded, without execution, ACL-002 checkpoint
`e90c097f58e4e4ab961272d3d50911d226eac25d`; ACL-003 checkpoints
`fecdd68809868280e3852d5bc23075db28ae2ff3` and
`eabdb7eca082c4f5d87e193d73edd892a9260d4b`; ACL-004 checkpoint
`bf5df09c1c3d58c3e1892234cbdd3da2b921c66b`; and ACL-005 checkpoint
`677086fbb7fb6800880483c5ab883d93680c972b`. None produced confirmatory outcomes.

## Results

### Categorical stability

ACL-002 confirmed the first-order endpoint law

\[
q_T=p_T+\epsilon s_T+O(\epsilon^2),\qquad
\delta_{L1}=\epsilon\lVert s_T\rVert_1+O(\epsilon^2).
\]

On regular held-out targets, the zero-fit maximum-within-landscape relative-error gate
had median `0.4275%` and Type-7 Q90 `1.4730%`. The separately frozen calibrated layer
used artifact value `alpha_source = 0.9951356698171323` and had median `0.4677%` and
Q90 `0.9914%`. Both passed; special zero/low-sensitivity cases and the matrix-power
oracle also passed. This was within-family combinatorial transport, not cross-class
evidence.

The kickoff prompt transcribed alpha as `0.9951356718983256`; the immutable artifact's
value above differs by `2.0812e-9`. All analyses use the artifact value. This provenance
discrepancy is recorded and does not affect any gate.

Post-confirmatory residual analysis found a coherent local quadratic remainder and
earned the second-order model

\[
q_T=p_T+\epsilon s_T+\frac{\epsilon^2}{2}u_T+O(\epsilon^3).
\]

ACL-003 then tested the corresponding zero-fit L1 prediction on 16 landscapes using
new state, reward, and mutation values. Its maximum error over the confirmatory local
epsilon grid had median `0.1484%`, Q90 `0.7387%`, and maximum `1.0753%`, passing the
frozen `10%`/`20%` criteria without fitting.

The boundary is nonuniform. At epsilon `0.03`, second order improved all 16 ACL-003
landscapes; at `0.1`, it improved only 13. At horizon 20 in the stress region, its
relative error had median `3.652%`, Q90 `36.875%`, and maximum `114.694%`. The local
Taylor law is predictive, not globally reliable.

### Gaussian expected direction

For the frozen Gaussian rank-mu system, ACL-004 tested

\[
E[\Delta\theta_\lambda\mid\theta]\parallel
g_\lambda^{\mathrm{analytic}}(\theta)
\]

with separate mean and covariance Fisher blocks. All 12 landscapes met the disjoint-
half stopping rule at 4096 shadows. Minimum analytic Fisher cosines were
`0.999952324` for the mean block and `0.999552200` for covariance, above the frozen
`0.99` gate.

The practical one-population layer was much weaker, especially for covariance:
single-shadow median mean cosines spanned `0.9531–0.9570`, while covariance medians
spanned `0.5338–0.7169` and were positive in `84.13–87.94%` of shadows. ACL-004 is an
expected-direction result, not a guarantee that each finite population points well.

### Cross-class transport into control

ACL-005 copied this complete source rule without modification:

\[
\min_b\cos_F(\bar d_b^{(1)},\bar d_b^{(2)})\ge0.98
\quad\Longrightarrow\quad
\min_b\cos_F(\bar d_b,d_b^{\mathrm{analytic}})\ge0.99,
\]

using the same replication schedule. The source was Gaussian rank-mu; the target was
an empirical-Fisher plug-in natural policy gradient for a two-context, three-action
bandit at `N=128`. Nothing was fit in control.

All 10 regular target landscapes stopped at 4096. All 20 context blocks passed, with
minimum analytic Fisher cosine `0.999841630`. This is the program's successful
preregistered cross-class transport result.

The four non-gating stress landscapes also stopped at 4096, yet 5 of 8 context blocks
fell below `0.99`; the minimum was `0.055699373`. Their frozen minimum expected joint-
cell counts ranged from `0.1607` to `0.6863`, versus at least 4 for regular targets.
Thus agreement between two sample halves can certify a stable biased direction when
empirical Fisher support is poor.

Artifact-only ACL-005 reconstruction reproduced every stopped mean, checkpoint mean,
final cosine, uncertainty value, and H1 summary exactly. Its summary SHA-256 is
`b5d310e9a32c059cb192e4f1001556b7a9d60ca98cc6319a1d67326246c13084`.

## Bridge ledger

The full machine-readable ledger is [`BRIDGE_LEDGER.json`](BRIDGE_LEDGER.json); the
human-readable version is [`BRIDGE_LEDGER.md`](BRIDGE_LEDGER.md).

| Edge | Map and scope | Predictive content | Stability/transport | Failure boundary | Status |
|---|---|---|---|---|---|
| MWU ↔ replicator ↔ categorical NG | Interior simplex, fixed reward, entropy/Fisher geometry | Clean trajectories agree; mutation response follows analytic tangents | First- and second-order held-out local laws pass without target refit | Error becomes nonuniform at stress epsilon; no global radius established | Theorem reproduction + preregistered within-class confirmation |
| Gaussian NG ↔ finite-lambda rank-mu | Pure Gaussian, fixed rank weights and population, no CSA/evolution paths | Conditional expected mean/covariance tangents align with independent score/Fisher comparator | Both block gates pass | Individual finite-population covariance directions remain noisy | Preregistered within-class confirmation |
| Categorical NG ↔ finite-state control | Exact finite categorical policies, no PPO or neural approximation | Empirical-Fisher NPG conditional mean aligns blockwise with exact NPG under adequate support | Regular ACL-005 blocks pass the transported rule | Rare-cell pseudoinverse bias survives internal convergence | Preregistered cross-class confirmation with boundary |
| Gaussian rank-mu → control normalized law | Conditional stochastic mean, analytic Fisher tangent, block Fisher cosine | Source `0.98→0.99` rule predicts target alignment | PASS without target refit | 5/8 rare-cell stress blocks fail | Restricted reusable law |

A bridge with only a map and scope remains a coordinate relation. The final edge has
map, scope, falsifiable content, stability diagnostic, no-refit transport, and an
observed failure boundary.

## Positive and negative findings

Confirmed positive findings:

- exact categorical correspondences reproduce at numerical precision;
- first-order mutation sensitivity predicts frozen held-out targets locally;
- a derived second-order response predicts entirely new categorical values locally;
- the finite-lambda Gaussian conditional mean aligns in both Fisher blocks;
- the Gaussian blockwise diagnostic transports unchanged into regular control targets.

Confirmed or frozen negative/boundary findings:

- categorical first- and second-order truncations are not uniformly reliable at large
  epsilon;
- one finite Gaussian rank-mu population, especially its covariance block, is not a
  reliable proxy for its conditional mean;
- disjoint-half convergence is not sufficient for truth alignment under rare
  context-action support;
- the transported law is support-conditioned, not universal.

No primary confirmatory experiment failed. The negative evidence comes from
predeclared non-gating stress cells and descriptive H1 layers whose roles were frozen
before outcomes; it therefore maps boundaries without rewriting the PASS criteria.

## Falsified hypotheses and interpretations

- **Falsified:** a support-free version of the `0.98→0.99` expected-direction rule.
  ACL-005 stress targets converged internally but five blocks failed externally.
- **Falsified:** uniform usefulness of the categorical second-order truncation through
  the tested stress range. It worsened some epsilon-`0.1` cases and exceeded 100%
  relative error in the worst held-out case.
- **Ruled out as an interpretation:** equation-level equivalence alone guarantees a
  transferable scientific law. Predictive content required independent comparators,
  local radii or support conditions, and held-out tests.

## Surviving restricted claims

1. The categorical simplex correspondence has predictive local perturbation content,
   including a second-order response on new values, within a finite stability radius.
2. The studied finite-lambda Gaussian rank-mu conditional mean is accurately described
   by an independently derived Fisher-natural direction in separate tangent blocks.
3. Blockwise Fisher-cosine convergence/alignment is a reusable cross-class diagnostic
   for the studied Gaussian and regular finite-state control systems without target
   refitting.
4. Sampling support must enter the scope: internal estimator convergence alone does
   not protect against stable pseudoinverse bias.

## What the evidence does not justify

- that evolution, learning, optimization, and control are one universal process;
- a shared numerical epsilon-response coefficient across categorical, Gaussian, and
  control classes;
- population-level statistical confidence beyond the deterministic benchmark units;
- reliable individual stochastic updates merely because conditional means align;
- lambda or interaction-sample scaling laws;
- sequential MDP, PPO, neural-policy, continuous-control, GPU, or large-model claims;
- transport when Fisher blocks are singular, poorly supported, damped, clipped, or
  otherwise outside the frozen estimators;
- a globally valid categorical Taylor approximation.

## Strongest mathematical statement supported

For the finite systems and frozen benchmarks studied here, there is evidence for a
restricted class of Fisher-geometric adaptive correspondences with two reusable forms
of prediction:

\[
q_T(\epsilon)=p_T+\epsilon s_T+\frac{\epsilon^2}{2}u_T+O(\epsilon^3)
\]

predicts local categorical mutation response on held-out values; and, under adequate
sampling support,

\[
\cos_F(\text{disjoint-half means})\ge0.98
\quad\Rightarrow\quad
\cos_F(E[\text{finite-sample tangent}],\text{analytic natural tangent})\ge0.99
\]

transported from the studied Gaussian rank-mu class to the studied contextual-bandit
NPG class without target refitting. The second implication is empirical and scoped,
not a proved theorem; the rare-cell counterexamples show that its support condition
cannot be omitted.

## Most important open theorem

Derive a support-conditioned angular-bias bound for plug-in natural gradients. For
minimum joint-cell probability `p_min`, sample count `N`, bounded rewards, and an
undamped empirical Fisher pseudoinverse, seek an explicit function `Phi` such that

\[
1-\cos_F\!\left(E[\widehat F^+\widehat g],F^+g\right)
\le \Phi(Np_{\min},\text{reward geometry},\text{Fisher spectrum}),
\qquad \Phi(x,\cdot)\to0\text{ as }x\to\infty.
\]

The theorem should include the probability and angular effect of rank-deficient
empirical blocks and a converse showing when no half-mean self-consistency test can
detect stable bias. Such a result would explain both ACL-005's regular PASS and stress
failure, convert the observed support boundary into a predictive radius, and reveal
whether the Gaussian/control transport belongs to a broader exponential-family class.
