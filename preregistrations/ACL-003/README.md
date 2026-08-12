# ACL-003 preregistration-only bundle

This directory contains no ACL-003 perturbed outcomes.

- `PREREGISTRATION.md` freezes the scientific question, units, regions, primary gate,
  reporting order, and forbidden behavior.
- `DERIVATION.md` freezes the row-vector second-order recurrence and prediction.
- `ANALYSIS_PLAN.md` is the executable analysis contract.
- `manifest.json` contains 16 hypothesis-bearing landscapes built from numeric catalog
  values not present in the byte-frozen ACL-002 reference manifest, plus one identity
  software control.
- `analytic_registry.json` contains clean first/second derivatives and oracle checks;
  it declares `outcomes_generated=false`.
- `LOCK.json` hashes this frozen bundle and also declares
  `outcomes_generated=false`.

Safe command before execution:

```powershell
acl acl003-validate
```

The future `acl acl003-run` command requires an explicitly approved exact public SHA,
a completely clean worktree including no untracked files, valid locks, and a new
SHA-derived canonical evidence path. It must not be invoked during preregistration
construction or audit.
