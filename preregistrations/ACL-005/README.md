# ACL-005 preregistration-only bundle

This directory contains no contextual-bandit target shadows or outcomes.

- `manifest.json` freezes the ACL-004 source evidence, copied normalized law, 10
  regular targets, four stress targets, and all estimator constants.
- `DERIVATION.md` fixes the independent score/Fisher comparator and plug-in estimator.
- `ANALYSIS_PLAN.md` freezes stopping, contextwise gates, and reporting order.
- `analytic_registry.json` contains analytic directions and pre-outcome strata only.
- `LOCK.json` hashes the exact six-file bundle and declares
  `outcomes_generated=false`.

Safe pre-outcome command:

```powershell
acl acl005-validate
```

The future `acl005-run` requires explicit approval of an exact public SHA, the canonical
in-repository bundle, a fully clean worktree, valid locks, matching ACL-004 source
artifact/report hashes, and the SHA-derived canonical evidence path. It must not be
invoked during construction or audit.
