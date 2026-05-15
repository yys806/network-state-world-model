# network-state-world-model

Research workspace for joint network-state prediction and action-conditioned world-model modeling in air-ground collaborative networks.

## Project Goal

This project studies how to organize air-ground network simulation logs into world-model training samples. The current scenario is smart-city air-ground collaborative sensing, where vehicles, UAVs, RSUs, and cloud nodes jointly generate, transmit, compute, and return sensing tasks.

The current technical line is:

1. Run AirFogSim to generate controllable node, link, and task logs.
2. Convert raw logs into a joint time-series dataset with physical states, communication states, and task states.
3. Build baseline predictors from historical windows to future labels.
4. Evaluate perturbation robustness and prediction uncertainty.
5. Compare simulator rollout cost with lightweight learned-model inference.
6. Run multi-seed simulations and cross-seed evaluation.
7. Add strict scheduler action logs for action-conditioned world-model training.

## Repository Structure

```text
.
|-- AGENTS.md
|-- README.md
|-- local_plan.xlsx / 本地计划表.xlsx
|-- project background markdown files
|-- research progress documents
`-- experiments/
    `-- airfogsim_v0/
        |-- scripts/
        |-- reports/
        |-- figures/
        `-- datasets/
```

`code/AirFogSim/` is intentionally not tracked by this main repository because it is a third-party Git repository. The scripts under `experiments/airfogsim_v0/scripts/` are the project-specific experiment scripts copied from the local AirFogSim workspace.

## Current Datasets

`dataset_v0` is generated from one AirFogSim demo run:

- History length: `H = 8`
- Prediction horizon: `K = 3`
- Samples: `190`
- Nodes: `37`
- Candidate links: `188`

`dataset_multiseed_v0` is generated from seeds `[0, 1, 2, 3, 4]`:

- Samples: `950`
- Samples per seed: `190`
- `x_node`: `(950, 8, 37, 7)`
- `x_link`: `(950, 8, 188, 5)`
- `x_task`: `(950, 8, 9)`
- `y_node`: `(950, 3, 37, 7)`
- `y_link`: `(950, 3, 188, 5)`
- `y_task`: `(950, 3, 9)`

`action_proxy_v0` is the first action-side proxy interface:

- `a_hist`: `(950, 8, 13)`
- `a_future`: `(950, 3, 13)`
- Features include offload counts, RB allocation proxies, CPU progress proxies, and UAV mobility proxies.

`strict_action_v0` records scheduler decisions directly while AirFogSim is running:

- `a_hist`: `(950, 8, 13)`
- `a_future`: `(950, 3, 13)`
- Recorded actions include offloading targets, return routes, RB indices, CPU allocation, and UAV mobility commands.
- Per-seed detailed CSV logs are stored under `experiments/airfogsim_v0/reports/strict_action_v0/`.

## Current Results

The first baseline stage includes persistence, Ridge residual, and MLP residual models.

Main observations:

- Persistence is a strong short-horizon baseline because the current prediction horizon is only `0.3s`.
- Ridge residual improves task-state prediction in the single-seed setting but is weaker for link-rate prediction.
- MLP residual is unstable under the current small-sample, high-dimensional setting.
- Perturbation experiments show that the current residual baseline degrades under strong input noise.
- Residual-quantile uncertainty estimation provides preliminary 80% and 90% prediction intervals.
- Timing experiments show an online-inference acceleration opportunity: AirFogSim explicit rollout is much slower than Ridge baseline inference in the current small scenario. This does not mean the baseline can replace the simulator.
- Multi-seed simulations confirm that the same scenario can produce different vehicle counts, task loads, task success ratios, and link rates under different random seeds.
- Cross-seed evaluation has been added: train on seeds `0, 1, 2`, validate on seed `3`, and test on seed `4`. Current Ridge residual baseline does not generalize well to the held-out seed.
- Strict action logs are now aligned with `dataset_multiseed_v0`, so future models can use historical states plus action variables to predict future node/link/task labels.

See:

- `experiments/airfogsim_v0/reports/weekly_result_summary_v0.md`
- `experiments/airfogsim_v0/reports/airfogsim_mechanism_report.md`
- `experiments/airfogsim_v0/reports/cross_seed_report.md`
- `experiments/airfogsim_v0/reports/action_proxy_report.md`
- `experiments/airfogsim_v0/reports/strict_action_report.md`
- `experiments/airfogsim_v0/figures/`

## Reproduction Notes

Clone and install AirFogSim separately, then copy scripts from `experiments/airfogsim_v0/scripts/` into AirFogSim's `examples/` directory.

Typical commands:

```powershell
conda activate airfogsim
cd D:\path\to\AirFogSim\examples
python export_dataset_demo.py
python build_dataset_v0.py
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
```

SUMO must be installed and available through `SUMO_HOME` for AirFogSim traffic simulation.

## Next Steps

- Add perturbation training instead of only clean-training/noisy-testing.
- Upgrade from simple baselines to dual-graph and action-conditioned latent world-model architectures.
- Extend timing experiments to larger scenes and stronger models.
