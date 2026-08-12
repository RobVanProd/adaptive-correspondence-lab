# Hardware envelope

Version 0.1 is CPU-only and uses float64. The default experiment has three categories,
32 seeds, 25 steps, and a short epsilon grid. Vectorized work is chunked; no command
allocates an array proportional to all requested seeds unless the caller explicitly
sets the chunk size that high.

The initial development host has 8 physical cores, 16 logical processors, and about
32 GiB RAM. A GPU is neither required nor used. Scaling should increase independent
replicates before state dimension, and every accelerated result must retain a parity
check against the reference implementation.
