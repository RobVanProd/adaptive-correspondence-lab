# ACL-008 locked bundle

This directory freezes the analytic-only Burg mirror transport test. `manifest.json`
contains targets and copied source rules; `analytic_registry.json` contains only clean
states and zero-epsilon derivatives; the three protocol documents freeze interpretation
and order; `LOCK.json` hashes the six other files and is generated last.

Before outcomes:

```powershell
python scripts/freeze_acl008_bundle.py
python -m adaptive_correspondence acl008-validate
```

Do not invoke `acl008-run` before exact public-SHA approval.
