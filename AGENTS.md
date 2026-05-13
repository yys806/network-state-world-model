# Repository Guidelines

## Project Structure & Module Organization
This repository is a research workspace for the network-group internship project.

- `课题介绍.md`, `老师说明.md`, `北大实习.md`: project background, advisor guidance, and working history.
- `论文笔记/` and `文献库/`: paper notes and source PDFs. Keep notes and PDFs traceable.
- `开会/`: weekly meeting materials, PPTs, scripts, and supporting documents.
- `code/AirFogSim/`: AirFogSim code, local experiments, exported simulation logs, `dataset_v0`, baseline scripts, robustness tests, and uncertainty tests.
- `研究进展文档/`: rolling LaTeX progress document and compiled outputs.
- `本地计划表.xlsx`: primary local task/status table. Use this instead of Feishu unless the user explicitly asks to sync Feishu.

## Build, Test, and Development Commands
There is no repository-wide build pipeline. Use focused commands:

- `rg --files`: list files quickly.
- `git status`: inspect local changes before editing.
- `conda activate airfogsim`: enter the AirFogSim environment.
- `cd D:\shen\网络组\code\AirFogSim\examples`: run experiment scripts from the examples folder.
- `python train_baseline_v0.py`: run baseline prediction experiments.
- `python run_robustness_v0.py`: run perturbation/robustness experiments.
- `python run_uncertainty_v0.py`: run confidence-interval experiments.

For LaTeX progress documents, compile with XeLaTeX, not pdfLaTeX.

## Coding Style & Naming Conventions
- Prefer clear Markdown reports for experiment summaries.
- Use ASCII filenames for scripts when possible, for example `run_robustness_v0.py`.
- Keep generated outputs under the corresponding experiment folder, such as `outputs/dataset_v0_from_demo_run_20260507_190930/baseline_v0/`.
- Preserve traceability: every plot or result table should have the script and input dataset recorded nearby.
- Do not claim a result is reproducible unless the command, input path, random seed, and output path are documented.

## Testing Guidelines
No formal test framework is defined yet. For new code:

- Prefer small, runnable scripts with explicit input/output paths.
- Add a short report file beside each experiment output.
- Validate shapes and key statistics before interpreting results.
- For stochastic experiments, record random seeds and scenario settings.

## Current Research Workflow
The current main line is: AirFogSim simulation logs -> `dataset_v0` construction -> baseline prediction -> perturbation robustness -> uncertainty estimation -> AirFogSim mechanism/complexity analysis -> action-conditioned world model.

Important current dataset:

- `D:\shen\网络组\code\AirFogSim\examples\outputs\dataset_v0_from_demo_run_20260507_190930\dataset_v0_samples.npz`

Important current analysis outputs:

- `baseline_v0/`: persistence, Ridge residual, and MLP residual baseline results.
- `robustness_v0/`: input-noise robustness results.
- `uncertainty_v0/`: residual-quantile prediction intervals.
- `airfogsim_analysis_v0/`: AirFogSim state transition, randomness, and complexity analysis.

When a task is completed, update `D:\shen\网络组\本地计划表.xlsx`. Do not sync Feishu by default.

## Commit & Pull Request Guidelines
If this workspace is committed manually, use descriptive commit messages:

- `code: add baseline prediction script`
- `results: add robustness analysis for dataset_v0`
- `docs: update AirFogSim mechanism summary`

Keep generated reports concise and aligned with actual outputs.
