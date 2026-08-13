# ACL-007 locked bundle

This is the analytic-only preregistration bundle for no-refit transport from ACL-006
into sequential Bayesian particle filtering.

- `manifest.json` freezes source hashes, models, targets, seeds, schedule, and unchanged
  thresholds.
- `analytic_registry.json` contains exact count-state target moments and zero shadows.
- `DERIVATION.md`, `ANALYSIS_PLAN.md`, and `PREREGISTRATION.md` freeze mathematics,
  verdict order, scope, and forbidden behavior.
- `LOCK.json` hashes the six other files and is generated last.

The pre-outcome maintainer command `python scripts/freeze_acl007_bundle.py` regenerates
the exact registry and then the lock. It refuses to run after ACL-007 evidence exists.

Analytic-only validation:

```powershell
python -m adaptive_correspondence acl007-validate
```

Do not invoke `acl007-run` until an adversarial review explicitly approves the exact
public preregistration SHA.
