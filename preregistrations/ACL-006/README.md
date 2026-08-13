# ACL-006 locked bundle

This directory is the pre-outcome bundle for the exact support-conditioned plug-in bias
and self-consistency-dissociation experiment.

- `manifest.json` freezes all targets, contrasts, seeds, thresholds, environment, and
  execution budget.
- `analytic_registry.json` contains exact finite multinomial predictions and zero
  sampled shadows.
- `DERIVATION.md` freezes the mathematical quantities used by the analysis.
- `ANALYSIS_PLAN.md` freezes verdict order and failure behavior.
- `PREREGISTRATION.md` states scope, hypotheses, and exclusions.
- `LOCK.json` hashes the other six files and is generated last.

Before outcomes exist, the deterministic maintainer command
`python scripts/freeze_acl006_bundle.py` regenerates the analytic registry and then the
lock. It refuses to run after any ACL-006 evidence artifact exists.

Validation is analytic-only:

```powershell
python -m adaptive_correspondence acl006-validate
```

Do not invoke `acl006-run` until an adversarial review explicitly approves the exact
public preregistration SHA. The first confirmatory artifact must be
`evidence/ACL-006-confirmatory-{approved_sha}.json` and must never be regenerated.
