# AirFogSim v0 Experiments

This folder contains project-specific scripts, reports, datasets, and figures for the first AirFogSim-based baseline stage.

## Contents

- `scripts/`: experiment scripts copied from the local AirFogSim `examples/` workspace.
- `datasets/`: compact exported datasets that are small enough to track.
- `reports/`: dataset summaries, baseline reports, robustness reports, uncertainty reports, timing reports, mechanism analysis, and action logs.
- `figures/`: plots used for progress reporting and PPT material.

## Main Pipeline

Run these scripts inside an installed AirFogSim `examples/` directory:

```powershell
python export_dataset_demo.py
python build_dataset_v0.py
python visualize_dataset_v0.py
python train_baseline_v0.py
python run_robustness_v0.py
python run_uncertainty_v0.py
python run_timing_v0.py
python run_multiseed_v0.py
python export_multiseed_dataset_v0.py
python build_dataset_multiseed_v0.py
python run_cross_seed_baseline_v0.py
python build_action_proxy_v0.py
python export_strict_actions_v0.py
python make_airfogsim_analysis_v0.py
python make_weekly_summary_visuals_v0.py
```

## Interpretation

AirFogSim is used as a controllable data generator, not as the research contribution itself. The project contribution is the organization of raw `node/link/task` logs into joint physical-communication-task time-series samples, then using those samples for prediction, robustness evaluation, uncertainty estimation, cross-seed generalization checks, and action-conditioned world-model training.

The current stage is a baseline and analysis stage. It should not be overclaimed as a final world-model result.

## Timing v0

`run_timing_v0.py` compares AirFogSim's explicit `scheduleStep + env.step()` cost with Ridge residual baseline inference cost. In the current small scenario, AirFogSim takes about `5.96 ms/step`, so a `K=3` rollout is about `17.88 ms`; Ridge residual inference is about `0.0044 ms/sample`.

This result shows an online-inference acceleration opportunity, not that the current Ridge baseline is already a valid replacement for AirFogSim.

## Multi-Seed v0

`run_multiseed_v0.py` runs the same demo scene with seeds `[0, 1, 2, 3, 4]`. In the current 10-second scene, final vehicle counts range from `6` to `13`, final task counts range from `11` to `37`, and task success ratios range from about `0.829` to `0.969`.

`dataset_multiseed_v0` converts five seed trajectories into a unified history-window to future-label dataset:

- Total samples: `950`
- Samples per seed: `190`
- Node tensor sample shape: `(8, 37, 7)`
- Link tensor sample shape: `(8, 188, 5)`
- Task tensor sample shape: `(8, 9)`

## Cross-Seed Baseline v0

`run_cross_seed_baseline_v0.py` trains on seeds `0, 1, 2`, validates on seed `3`, and tests on seed `4`.

Held-out seed 4 result:

- Persistence RMSE: `1.530`
- Ridge residual RMSE: `3.303`

The result means the current compact residual baseline has limited cross-seed generalization. This is useful because it gives a concrete reason to move toward graph-structured and action-conditioned world-model methods.

## Action Proxy v0

`build_action_proxy_v0.py` extracts a first action-side proxy tensor from observable logs:

- `a_hist`: `(950, 8, 13)`
- `a_future`: `(950, 3, 13)`

The features include offload decision counts, offload target-type counts, active-link and RB allocation proxies, CPU progress proxies, and UAV movement proxies.

## Strict Action v0

`export_strict_actions_v0.py` records scheduler actions directly during AirFogSim execution:

- `a_hist`: `(950, 8, 13)`
- `a_future`: `(950, 3, 13)`
- Detail logs: offloading targets, return routes, RB indices, CPU allocation, and UAV mobility commands.
- Per-seed detail CSVs: `reports/strict_action_v0/seed_*/`.

This is the preferred action input for later action-conditioned world-model experiments.
