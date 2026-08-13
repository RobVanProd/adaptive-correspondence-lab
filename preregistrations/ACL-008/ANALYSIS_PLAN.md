# ACL-008 analysis plan

1. Abort before any positive-epsilon path unless HEAD equals the approved public SHA,
   full porcelain is clean, the canonical bundle/lock/source hashes/environment pass,
   and the SHA-derived evidence path does not exist.
2. Generate every frozen landscape, epsilon, and horizon once. Compare the bisection
   path with the polynomial-root path before analysis.
3. Evaluate the identity control. Failure makes the run INVALID.
4. For all 16 regular targets compute the maximum second-order relative error across
   `0.001,0.003,0.01` at `T=20`.
5. Compute Type-7 median and Q90. PASS iff they are at most `0.10` and `0.20`.
6. Only after the verdict, report first-versus-second-order improvement, numerical-
   control, secondary-horizon, max-path, and stress summaries.

The deterministic target benchmark is not a population sample. Median and Q90 are
descriptive gates, not confidence statements. PASS, FAIL, and INVALID remain distinct.
