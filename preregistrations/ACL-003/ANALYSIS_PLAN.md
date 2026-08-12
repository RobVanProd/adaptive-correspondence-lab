# ACL-003 analysis plan

## Inputs

- `manifest.json`: 16 confirmatory targets plus identity control `C01`.
- `analytic_registry.json`: clean `s/u` values only; `outcomes_generated=false`.
- ACL-002 manifest: numeric novelty reference only, never an outcome source; its
  required SHA-256 is
  `6a9e4e0a931277b1f5c464807d0bcacee3ccb684269843f8245a83ae88110741`.
- First ACL-003 evidence artifact produced from an approved exact SHA.

## Primary filter

Use rows satisfying all of:

- role `confirmatory-target`;
- stratum `regular-sensitivity`;
- horizon `20`;
- epsilon in `0.001,0.003,0.01`.

For each landscape, compute the maximum absolute relative error of the zero-fit
second-order truncated-vector prediction. Across the 16 landscape scores, compute
Type-7 median and Q90. Pass only if median `<=0.10` and Q90 `<=0.20`.

No other row can affect this verdict.

## Special strata

- `C01`: maximum absolute endpoint L1 must be at most `2e-12`; report only as a
  software control. Failure makes the run `INVALID` regardless of the primary gate.
- Any predeclared low-sensitivity target: absolute error only, never relative error.
- Current clean registry expects 16 regular targets and one identity control.

## Reporting order

1. Execution and hash provenance.
2. Matrix and derivative oracle diagnostics.
3. Identity/low-sensitivity controls.
4. Primary per-landscape scores, Type-7 median/Q90, and conjunction verdict.
5. First/second paired descriptive comparisons.
6. Empirical radius summaries.
7. Secondary horizons and max-path L1.
8. Numerical-control region.
9. Stress region last, explicitly non-gating.

## Failure behavior

Abort before analysis for SHA/worktree/output/exact-lock-set/reference-hash/registry/
novelty/oracle failure. Never repair, filter, or rerun after the first outcome artifact
exists. A post-generation identity-control failure is retained and reported as
`INVALID`, not silently discarded.
