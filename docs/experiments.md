# Experiment protocols

## 1. Exact reproduction

Run all three categorical systems from the same strictly interior state with the same
stationary reward sequence. Compare every canonical state, not just the endpoint.
This is a software/theorem reproduction check.

## 2. Stability curve

Choose one named violation, a frozen epsilon grid, discrepancy metric, horizon, and
seed set. The baseline is multiplicative weights under the correspondence assumptions.
The target receives the violation. Record both per-seed discrepancies and their
aggregate; do not silently discard failed trajectories.

The output is the empirical map `epsilon -> delta(epsilon)`. A first-order coefficient
is estimated only from a caller-declared source window through an origin-constrained
least-squares fit. The raw curve remains the primary artifact.

## 3. Transported prediction

Fit a source coefficient on explicitly named source-domain data, freeze it, and apply
it to target epsilons without fitting target observations. Report predictions,
observations, residuals, and source/target identities. The bundled command is an
instrumentation demonstration, not evidence that the coefficient transports in a
broader system class.

## 4. Escalation rungs

The Gaussian optimizer is first checked on a quadratic objective with analytic expected
gradient and Fisher geometry. The contextual bandit is first checked against exact
expected policy gradients. Finite-sample estimators are compared with these reference
directions before any larger problem is admitted.

## ACL-002 mutation checkpoint

ACL-002 is frozen under `preregistrations/ACL-002/`. It uses deterministic landscapes,
not seed replication. Probabilities and sensitivities are row vectors, mutation is
right multiplication by a row-stochastic matrix, and the Jacobian stores input index
first and output index second.

The primary endpoint-L1 analysis reports two separate held-out verdicts: the derived
zero-fit prediction and a landscape-balanced median calibration fitted on source
landscapes. The secondary oriented divergence is exactly `KL(q_T || p_T)` and uses its
preregistered Fisher-curvature coefficient. Stress epsilons and secondary metrics are
structurally absent from primary gate calculation.

Preparing or validating this bundle may recompute clean trajectories and their analytic
tangents. It must not generate an epsilon-positive trajectory from the confirmatory
manifest until the public preregistration SHA is reviewed and explicitly approved.
