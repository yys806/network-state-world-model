# Cross-seed baseline report v0

## Goal

This experiment checks whether the current training sample format can support evaluation across different stochastic AirFogSim trajectories.

## Split

- Train: seed 0, seed 1, seed 2
- Validation: seed 3
- Test: seed 4
- Input: history window of node/link/task tensors
- Target: future link-rate statistics and task-state statistics

## Metrics

| split | model | all_mae | all_rmse | link_rate_by_type_mae | link_rate_by_type_rmse | task_state_mae | task_state_rmse |
|---|---|---:|---:|---:|---:|---:|---:|
| val_seed_3 | persistence | 0.516713 | 3.603568 | 1.207628 | 7.081746 | 0.286407 | 0.772801 |
| val_seed_3 | ridge_residual | 0.875549 | 3.197282 | 2.209200 | 6.252461 | 0.430999 | 0.773988 |
| test_seed_4 | persistence | 0.516527 | 1.529625 | 0.687975 | 2.279793 | 0.459378 | 1.177787 |
| test_seed_4 | ridge_residual | 1.296921 | 3.302883 | 3.303024 | 6.265132 | 0.628220 | 1.208893 |

## Interpretation

Ridge residual is worse than persistence on the held-out seed. This means the current compact baseline has limited cross-seed generalization, and a structured graph/world-model design is still necessary.

This result should be presented as a generalization check, not as the final world-model result.

## Outputs

- metrics_csv: `experiments/airfogsim_v0/reports/cross_seed_metrics.csv`
- bar_plot: `experiments/airfogsim_v0/figures/cross_seed_rmse_bar.png`
- link_plot: `experiments/airfogsim_v0/figures/cross_seed_link_rate_predictions_seed4.png`
- task_plot: `experiments/airfogsim_v0/figures/cross_seed_task_predictions_seed4.png`
