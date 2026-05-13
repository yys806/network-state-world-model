# network-state-world-model

Research workspace for joint network-state prediction and action-conditioned world-model modeling in air-ground collaborative networks.

## Project Goal

This project studies how to organize air-ground network simulation logs into world-model training samples. The current scenario is smart-city air-ground collaborative sensing, where vehicles, UAVs, RSUs, and cloud nodes jointly generate, transmit, compute, and return sensing tasks.

The current technical line is:

1. Run AirFogSim to generate controllable node, link, and task logs.
2. Convert raw logs into a joint time-series dataset with physical states, communication states, and task states.
3. Build baseline predictors from historical windows to future labels.
4. Evaluate perturbation robustness and prediction uncertainty.
5. Extend the dataset toward action-conditioned world-model training.

## Repository Structure

```text
.
├── AGENTS.md
├── README.md
├── 本地计划表.xlsx
├── 课题介绍.md
├── 老师说明.md
├── 研究进展文档/
│   ├── research_progress_overview.tex
│   └── research_progress_overview.pdf
└── experiments/
    └── airfogsim_v0/
        ├── scripts/
        ├── reports/
        └── figures/
```

`code/AirFogSim/` is intentionally not tracked by this main repository because it is a third-party Git repository. The scripts under `experiments/airfogsim_v0/scripts/` are the project-specific experiment scripts copied from the local AirFogSim workspace.

## Current Dataset

The current dataset is `dataset_v0`, generated from one AirFogSim demo run.

Key settings:

- History length: `H = 8`
- Prediction horizon: `K = 3`
- Samples: `190`
- Nodes: `37`
- Candidate links: `188`

Main tensors:

- `x_node`: `(190, 8, 37, 7)`
- `x_link`: `(190, 8, 188, 5)`
- `x_task`: `(190, 8, 9)`
- `y_node`: `(190, 3, 37, 7)`
- `y_link`: `(190, 3, 188, 5)`
- `y_task`: `(190, 3, 9)`

## Current Results

The first baseline stage includes persistence, Ridge residual, and MLP residual models.

Main observations:

- Persistence is a strong short-horizon baseline because the current prediction horizon is only `0.3s`.
- Ridge residual improves task-state prediction but is weaker for link-rate prediction.
- MLP residual is unstable under the current small-sample, high-dimensional setting.
- Perturbation experiments show that the current residual baseline degrades under strong input noise.
- Residual-quantile uncertainty estimation provides preliminary 80% and 90% prediction intervals.

See:

- `experiments/airfogsim_v0/reports/weekly_result_summary_v0.md`
- `experiments/airfogsim_v0/reports/airfogsim_mechanism_report.md`
- `experiments/airfogsim_v0/figures/`

## Reproduction Notes

To reproduce the full local pipeline, first clone and install AirFogSim separately:

```powershell
git clone https://github.com/ZhiweiWei-NAMI/AirFogSim.git
cd AirFogSim
```

Then copy or place the scripts from:

```text
experiments/airfogsim_v0/scripts/
```

into AirFogSim's `examples/` directory and run from there.

Typical commands:

```powershell
conda activate airfogsim
cd D:\path\to\AirFogSim\examples
python export_dataset_demo.py
python build_dataset_v0.py
python train_baseline_v0.py
python run_robustness_v0.py
python run_uncertainty_v0.py
python make_airfogsim_analysis_v0.py
```

SUMO must be installed and available through `SUMO_HOME` for AirFogSim traffic simulation.

## Next Steps

- Add strict action variables: offloading, RB allocation, CPU allocation, and UAV movement.
- Run timing experiments comparing AirFogSim rollout and learned-model inference.
- Generate multi-seed and multi-scenario datasets.
- Add perturbation training instead of only clean-training/noisy-testing.
- Upgrade from simple baselines to dual-graph and action-conditioned latent world-model architectures.
