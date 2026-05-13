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
python make_airfogsim_analysis_v0.py
python make_weekly_summary_visuals_v0.py
```

## Interpretation

AirFogSim is used as a controllable data generator, not as the research contribution itself. The project contribution is the organization of raw `node/link/task` logs into joint physical-communication-task time-series samples, then using those samples for prediction, robustness evaluation, uncertainty estimation, and later action-conditioned world-model training.

The current stage is a baseline and analysis stage. It should not be overclaimed as a final world-model result.
