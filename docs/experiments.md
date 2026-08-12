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
