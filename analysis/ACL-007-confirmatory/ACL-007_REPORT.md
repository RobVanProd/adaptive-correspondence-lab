# ACL-007 confirmatory report

## Confirmed cross-class result

The frozen overall transport verdict is **PASS**. The standardized-mean, dissociation,
and all nine contrast components passed without target refitting. The maximum full/half
standardized direction score was `1.815369` against
`5`; Type-7 full-score median and Q90 were
`0.796407` and
`1.212732` against `1.5` and `2.5`. Every
truth-cosine residual lay within its frozen analytic envelope. Stored chunks reproduce
all 64 checkpoints with maximum vector discrepancy
`0`.

## Dissociation and adverse targets

All 3 predeclared dissociation targets reached
split-half Euclidean cosine at least
`0.999994349`, while observed truth alignment
within that stratum ranged from `0.344906786`
to `0.378970223`. Across the full benchmark,
the minimum truth cosine was `-0.999797336` and
3 targets were negatively aligned.
Thus the PASS does not hide target bias: it correctly predicts Monte Carlo convergence
around biased particle-filter expectations and preserves the contrast ordering.

## Scientific classification

ACL-007 is the first preregistered no-refit transport result outside the Fisher-natural
family. It changes estimator family, metric, reward optimization to inference, and
one-step sampling to repeated transition/weighting/resampling. Exact target moments are
theorem/software controls; the evidence-bearing result is the unchanged ACL-006
dimensionless diagnostic.

This does **not** yet establish a unified theory of adaptive dynamics. The transported
standardized score follows finite-mean Monte Carlo geometry and does not assert a common
detailed bias mechanism. Phase-II termination therefore requires one more structurally
distinct, mechanism-bearing test or a controlled failure that maps the boundary.

## Execution provenance

The runner was invoked exactly once. The shell harness reported its five-second timeout,
but the runner had already atomically completed the canonical artifact. The complete
file was preserved, parsed, reconstructively validated, and committed untouched; no
second invocation occurred.
