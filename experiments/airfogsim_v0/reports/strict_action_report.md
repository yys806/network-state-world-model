# Strict action log report v0

## Purpose

This experiment records scheduler decisions directly while AirFogSim is running.
It is stricter than `action_proxy_v0`, which inferred action-side variables from observable state logs.

## Recorded actions

- Offloading: task id, source task node, selected target node, target node type.
- Returning: task id, current node, selected return RSU.
- Communication: task id and allocated RB indices.
- Computation: task id and allocated CPU from the scheduler callback.
- UAV mobility: UAV id, speed, angle, phi, and target source.

## Tensor outputs

- `a_hist`: `(950, 8, 13)`
- `a_future`: `(950, 3, 13)`

## Key point

These tensors are now aligned with `dataset_multiseed_v0`, so future models can use historical states plus action variables and predict future node/link/task labels.

The resulting action-conditioned prediction interface can be written as:

$$
\hat{s}_{t+1:t+K}=f_{\theta}(s_{t-H+1:t}, a_{t-H+1:t+K})
$$

Here, `s` corresponds to node/link/task states and `a` corresponds to the strict scheduler actions recorded in this experiment.

## Per-seed action counts

| seed | offload | return | rb | cpu | uav |
|---:|---:|---:|---:|---:|---:|
| 0 | 131 | 2 | 131 | 371 | 400 |
| 1 | 75 | 1 | 75 | 206 | 400 |
| 2 | 109 | 2 | 109 | 290 | 400 |
| 3 | 46 | 2 | 46 | 100 | 400 |
| 4 | 80 | 1 | 80 | 198 | 400 |
