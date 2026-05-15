# Action-conditioned baseline report v0

## Goal

This experiment tests whether strict scheduler actions help cross-seed prediction.

## Split

- Train: seed 0, seed 1, seed 2
- Validation: seed 3
- Test: seed 4

## Compared inputs

- `persistence`: repeat the last observed future target.
- `state_only_ridge`: historical node/link/task states only.
- `state_action_ridge`: historical node/link/task states plus strict historical/future action tensors.

## Metrics

| split | model | all_mae | all_rmse | link_rate_by_type_mae | link_rate_by_type_rmse | task_state_mae | task_state_rmse |
|---|---|---:|---:|---:|---:|---:|---:|
| val_seed_3 | persistence | 0.516713 | 3.603568 | 1.207628 | 7.081746 | 0.286407 | 0.772801 |
| test_seed_4 | persistence | 0.516527 | 1.529625 | 0.687975 | 2.279793 | 0.459378 | 1.177787 |
| val_seed_3 | state_only_ridge | 0.875549 | 3.197282 | 2.209200 | 6.252461 | 0.430999 | 0.773988 |
| test_seed_4 | state_only_ridge | 1.296921 | 3.302883 | 3.303024 | 6.265132 | 0.628220 | 1.208893 |
| val_seed_3 | state_action_ridge | 0.706386 | 3.032503 | 2.091624 | 6.008933 | 0.244640 | 0.475050 |
| test_seed_4 | state_action_ridge | 1.003282 | 2.990303 | 2.868150 | 5.824143 | 0.381659 | 0.784644 |

## Interpretation

Adding strict actions improves the Ridge residual baseline on the held-out seed:

- Overall RMSE: `3.303 -> 2.990`
- Link-rate RMSE: `6.265 -> 5.824`
- Task-state RMSE: `1.209 -> 0.785`

This means scheduler actions contain useful transition information. However, the action-conditioned linear Ridge baseline still does not beat persistence on overall RMSE. The correct conclusion is: actions are useful, but the current linear compact baseline is not enough; the next step should be a more structured dual-graph or latent world-model architecture.

## Outputs

- metrics_csv: `experiments/airfogsim_v0/reports/action_conditioned_metrics.csv`
- bar_plot: `experiments/airfogsim_v0/figures/action_conditioned_rmse_bar.png`
- link_plot: `experiments/airfogsim_v0/figures/action_conditioned_link_predictions_seed4.png`
- task_plot: `experiments/airfogsim_v0/figures/action_conditioned_task_predictions_seed4.png`
