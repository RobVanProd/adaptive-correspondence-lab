# ACL-004 preregistration-only bundle

This directory contains no sampled Gaussian shadows or outcomes.

- `manifest.json` freezes 12 landscapes and all estimator constants.
- `DERIVATION.md` fixes the independent score/Fisher comparator.
- `ANALYSIS_PLAN.md` freezes stopping, separate-block gates, and reporting order.
- `analytic_registry.json` contains analytic directions and doubled-order checks only.
- `LOCK.json` hashes the exact six-file bundle and declares
  `outcomes_generated=false`.

Safe pre-outcome command:

```powershell
acl acl004-validate
```

The future `acl004-run` requires an approved exact public SHA, a fully clean worktree,
valid locks, and the SHA-derived canonical evidence path. It must not be invoked during
construction or audit.
