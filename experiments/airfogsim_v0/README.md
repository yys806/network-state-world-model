# AirFogSim v0 Experiments

This folder contains project-specific scripts, reports, and figures for the first AirFogSim-based dataset and baseline stage.

## Contents

- `scripts/`: experiment scripts copied from the local AirFogSim `examples/` workspace.
- `reports/`: dataset summary, field mapping, baseline report, robustness report, uncertainty report, and AirFogSim mechanism analysis.
- `figures/`: key plots used for progress reporting.

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
python make_airfogsim_analysis_v0.py
python make_weekly_summary_visuals_v0.py
```

## Interpretation

AirFogSim is used as a controllable data generator, not as the research contribution itself. The project contribution is the organization of raw `node/link/task` logs into joint physical-communication-task time-series samples, then using those samples for prediction, robustness evaluation, uncertainty estimation, and later action-conditioned world-model training.

The current stage is a baseline and analysis stage. It should not be overclaimed as a final world-model result.

## Timing v0

`run_timing_v0.py` compares AirFogSim's explicit `scheduleStep + env.step()` cost with Ridge residual baseline inference cost. In the current small scenario, AirFogSim takes about `5.96 ms/step`, so a `K=3` rollout is about `17.88 ms`; Ridge residual inference is about `0.0044 ms/sample`.

This result should be reported carefully: it shows an online-inference acceleration opportunity, not that the current Ridge baseline is already a valid replacement for AirFogSim.

## Multi-Seed v0

`run_multiseed_v0.py` runs the same demo scene with seeds `[0, 1, 2, 3, 4]`. In the current 10-second scene, final vehicle counts range from `6` to `13`, final task counts range from `11` to `37`, and task success ratios range from about `0.829` to `0.969`.

This supports the claim that AirFogSim can generate multiple stochastic trajectories. The current `dataset_v0` is still a single-seed dataset, so multi-seed logs should next be converted into `dataset_multiseed_v0`.
