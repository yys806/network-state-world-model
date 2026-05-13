# Action proxy report v0

## Purpose

This file adds the first action-side interface for later action-conditioned world-model training.
The values are action proxies extracted from observable AirFogSim logs, not a complete reinforcement-learning action record.

## Action features

- `offload_decision_count`
- `offload_to_vehicle_count`
- `offload_to_uav_count`
- `offload_to_rsu_count`
- `offload_to_cloud_count`
- `offload_to_unknown_count`
- `active_link_count`
- `allocated_rb_total`
- `allocated_rb_mean_per_active_link`
- `cpu_progress_total`
- `computing_task_count`
- `uav_mean_speed`
- `uav_mean_displacement`

## Tensor shapes

- `a_hist`: `(950, 8, 13)`
- `a_future`: `(950, 3, 13)`

## Interpretation

`a_hist` aligns with the historical input window. `a_future` aligns with the future label window.
For strict action-conditioned rollout, the next version should log exact scheduler decisions before `env.step()`: offload route, RB indices, CPU allocation, and UAV mobility command.
