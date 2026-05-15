# Repository Guidelines

## Project Structure & Module Organization
This repository is a research workspace for the network-group internship project.

- `课题介绍.md`, `老师说明.md`: project background and advisor requirements.
- `研究进展文档/`: rolling LaTeX progress document and compiled outputs.
- `开会/`: weekly meeting materials, PPTs, scripts, and supporting documents.
- `experiments/airfogsim_v0/`: tracked experiment scripts, reports, figures, and compact datasets.
- `code/AirFogSim/`: local third-party AirFogSim repository. Do not treat it as part of this GitHub repo.
- `本地计划表.xlsx`: primary local task/status table. Use this instead of Feishu unless the user explicitly asks to sync Feishu.

## Build, Test, and Development Commands
There is no repository-wide build pipeline. Use focused commands:

- `rg --files`: list files quickly.
- `git status`: inspect local changes before editing.
- `conda activate airfogsim`: enter the AirFogSim environment.
- `cd D:\shen\网络组\code\AirFogSim\examples`: run experiment scripts from the AirFogSim examples folder.
- `python train_baseline_v0.py`: run baseline prediction experiments.
- `python run_robustness_v0.py`: run perturbation/robustness experiments.
- `python run_uncertainty_v0.py`: run confidence-interval experiments.
- `python run_cross_seed_baseline_v0.py`: run cross-seed baseline generalization.
- `python build_action_proxy_v0.py`: build aligned action proxy tensors.
- `python export_strict_actions_v0.py`: export strict scheduler action logs and aligned action tensors.

For LaTeX progress documents, compile with XeLaTeX, not pdfLaTeX.

## Coding Style & Naming Conventions
- Prefer clear Markdown reports for experiment summaries.
- Use ASCII filenames for scripts when possible, for example `run_robustness_v0.py`.
- Keep generated outputs under the corresponding experiment folder, such as `outputs/dataset_multiseed_v0/cross_seed_baseline_v0/`.
- Preserve traceability: every plot or result table should have the script and input dataset recorded nearby.
- Do not claim a result is reproducible unless the command, input path, random seed, and output path are documented.

## Testing Guidelines
No formal test framework is defined yet. For new code:

- Prefer small, runnable scripts with explicit input/output paths.
- Add a short report file beside each experiment output.
- Validate shapes and key statistics before interpreting results.
- For stochastic experiments, record random seeds and scenario settings.

## Current Research Workflow
The current main line is: AirFogSim simulation logs -> `dataset_v0` / `dataset_multiseed_v0` construction -> baseline prediction -> perturbation robustness -> uncertainty estimation -> AirFogSim mechanism/complexity analysis -> cross-seed generalization -> strict action logs -> action-conditioned world-model interface.

Important current outputs:

- `D:\shen\网络组\code\AirFogSim\examples\outputs\dataset_v0_from_demo_run_20260507_190930\dataset_v0_samples.npz`
- `D:\shen\网络组\code\AirFogSim\examples\outputs\dataset_multiseed_v0\dataset_multiseed_v0_samples.npz`
- `D:\shen\网络组\code\AirFogSim\examples\outputs\strict_action_logs_v0\strict_action_v0_samples.npz`
- `D:\shen\网络组\experiments\airfogsim_v0\reports\weekly_result_summary_v0.md`

When a task is completed, update `D:\shen\网络组\本地计划表.xlsx`. Do not sync Feishu by default.

## Commit & Pull Request Guidelines
Use descriptive commit messages:

- `experiments: add cross-seed baseline v0`
- `experiments: add strict action logs v0`
- `docs: update AirFogSim mechanism summary`

Keep generated reports concise and aligned with actual outputs. Do not push the third-party AirFogSim repository to its upstream remote.
