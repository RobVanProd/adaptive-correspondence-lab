# ACL-004 analysis plan

## Execution and convergence

For each of 12 landscapes, draw independent PCG64 population shadows at the frozen
`lambda=32`. At cumulative counts `4096,8192,16384,32768,65536`, compare first-half
and second-half conditional-mean directions using separate Fisher cosines for mean and
covariance/log-scale blocks.

Stop that landscape at the first checkpoint where both half cosines are at least
`0.98`. If any landscape does not converge by `65536`, report H2 `INCONCLUSIVE`.

## Primary H2

For every converged landscape, compare the stopped full conditional mean with the
independent finite-lambda analytic comparator. H2 passes only if the minimum across all
12 landscapes is at least `0.99` for the mean block and at least `0.99` for the
covariance block. A value below either threshold is FAIL. Joint Fisher cosine is
secondary and cannot rescue a block.

Landscapes are deterministic benchmark units. These minima and thresholds are
descriptive criteria, not population confidence statements.

## H1 and secondary reporting

Using the first 2048 shadows from each landscape, report separate single-shadow cosine
Q10, median, Q90, and fraction positive. Report stopped replication counts, checkpoint
histories, joint cosines, and covariance of shadow directions from stored chunk
sufficient statistics. H1 cannot change H2.

No lambda scaling, state iteration, or cross-class transport quantity is tested.

## Failure behavior

Abort before shadows for SHA, worktree, canonical-output, lock, manifest, analytic
registry, quadrature, seed, or numerical failure. After generation, retain and mark an
`INCONCLUSIVE` or `FAIL` result; never rerun, refit, change lambda, or alter thresholds.
