# Architecture

The code has two deliberately separate numerical paths.

- `categorical.py`, `gaussian.py`, and `bandit.py` are reference implementations.
  Their equations are written in small steps and every state transition is inspected.
- `batch.py` implements bounded, vectorized categorical trajectories. It exists for
  seed ensembles and is tested against the reference path.
- `schema.py` defines the common step/trajectory record.
- `perturbations.py` gives each assumption violation a named, validated operation.
- `experiments.py` owns comparison protocols, metrics, coefficient estimation, and
  transported predictions. Worlds do not decide whether a discrepancy is meaningful.
- `cli.py` creates JSON/CSV artifacts and records provenance.

This split prevents optimization machinery from becoming part of the scientific
definition. JAX, GPUs, neural networks, and third-party optimizer implementations are
intentionally absent from v1.
