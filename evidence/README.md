# Evidence policy

Artifacts here must say what they establish. Software verification can establish that
implementations agree with frozen equations or fixtures. It cannot by itself establish
a cross-domain scientific law. Do not add benchmark or research claims without the
exact configuration, seeds, software version, immutable source commit, and review status.
Generated verification artifacts should have `git_tracked_files_dirty: false`.

## Bundled artifacts

- `software-verification.json` checks exact three-way parity, batch/reference parity,
  and the expected second-order one-step Euler discrepancy.
- `categorical-equivalence.json` contains ten fully instrumented steps from each exact
  categorical world under the frozen assumptions.
- `mutation-stability.csv` and its metadata contain a one-seed deterministic example
  of an epsilon-to-delta curve. They demonstrate the file format, not a statistical or
  scientific result.

All bundled artifacts were generated from clean source checkpoint
`48d3c5c5ef96c59fe18f6cd5b3d27a44d0fcccc6`.
