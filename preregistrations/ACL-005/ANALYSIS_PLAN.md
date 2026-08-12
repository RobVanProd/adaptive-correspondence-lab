# ACL-005 analysis plan

## Frozen source and transport claim

ACL-004 is the sole source domain. Its immutable artifact and report hashes are in
`manifest.json`. ACL-005 transports its normalized blockwise expected-direction rule,
thresholds, and replication schedule unchanged. No ACL-005 shadow can set or modify a
coefficient, threshold, exclusion, sample count, landscape, or stopping checkpoint.

## Pre-outcome strata

For every target, compute the minimum analytic joint-cell expectation

\[
m_\ell=128\min_{c,a}\rho_c\pi_\ell(a\mid c).
\]

The 10 regular targets must have `m_l >= 4`. The four stress targets must have
`m_l <= 0.75`. The analytic registry freezes these classifications before outcomes.
Stress targets are reported and never gate the transport verdict.

## Conditional-mean stopping

Generate independent shadows in fixed chunks of 2048. At cumulative counts
`4096,8192,16384,32768,65536`, split all retained chunks into disjoint first and second
halves. Stop a landscape at the first checkpoint where every context's Fisher cosine
between half means is at least `0.98`.

If a regular landscape does not converge by 65536, the transport verdict is
`INCONCLUSIVE`. Stress nonconvergence is descriptive and cannot alter the verdict.
An undefined half cosine from a zero Fisher-norm empirical mean does not satisfy the
stopping threshold. If any regular analytic cosine is undefined at the terminal
budget, the regular landscape is nonconverged and the verdict remains INCONCLUSIVE;
undefined stress quantities are retained as JSON `null`.

## Primary transported gate

For each stopped regular target compare the full conditional-mean estimate with the
independent exact NPG direction in each context Fisher block. PASS requires all 10
regular landscapes to converge and

\[
\min_{\ell\in R,c}\cos_{F_{\ell,c}}
(\overline{\widehat d}_{\ell,c},d_{\ell,c})\ge0.99.
\]

If all regular landscapes converge but any context is below `0.99`, the verdict is
FAIL. Joint cosine is non-gating. Stress results cannot rescue or reverse the result.
The criteria are deterministic benchmark rules, not population-confidence statements.

## Secondary reporting

For the first 2048 shadows, report per-context Q10, median, Q90, fraction positive
among defined cosines, and the number undefined because a sampled direction has zero
Fisher norm. Also report checkpoint histories, stopped replication counts, joint
cosines, coordinate standard errors, and all stress outcomes. No secondary result may
change the primary verdict.

## Evidence and failure behavior

Each chunk stores count, direction sum, and flattened direction outer-product sum.
These must reproduce the stopped and half means exactly. The artifact embeds the
manifest, analytic registry, bundle lock, seeds, terminal PCG64 states, and provenance.

Abort before shadows on SHA, dirty-worktree, output-path, pre-existing-artifact, lock,
directory-membership, analytic-registry, stratum, or finite-value failure. After any
shadow is generated, retain the first artifact and its FAIL/INCONCLUSIVE/PASS result;
never rerun ACL-005 or change the design in place.
