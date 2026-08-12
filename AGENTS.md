# Repository rules

This repository is an experimental instrument, not evidence for a broad claim.

- Keep the reference implementation direct and readable. Optimized paths must be
  checked against it before their outputs are used.
- Freeze semantics in `TASK_LEDGER.md` before behavior changes. Add a failing
  regression test first.
- Invalid probabilities, non-finite values, and dimension mismatches must fail
  before an update; never silently repair them.
- Use float64 in all scientific paths. Record seeds and configuration with results.
- Do not call two systems equivalent without naming the mapping, assumptions,
  metric, horizon, and tolerance.
- Generated results are local observations, not accepted scientific claims. Public
  claims require an immutable artifact under `evidence/` and an independently
  reviewable analysis.
- Run `python -m pytest` and `python -m ruff check .` before committing.
- Do not publish packages, releases, benchmark claims, or force-push history.
