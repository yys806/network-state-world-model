# Repository Guidelines

## Identity

This workspace is for **PI-JWM**: Physical-Information Joint World Model.

AirFogSim is only a reference simulator and data-generation tool. Do not describe it as the framework or the research main line.

## Structure

- `代码/src/pi_jwm/`: PI-JWM framework modules.
- `代码/scripts/`: runnable scripts.
- `代码/tests/`: tests.
- `代码/reference/`: third-party references and simulators.
- `代码/artifacts/`: data, reports, figures, and generated outputs.
- `文档/`: meeting materials, papers, research documents, and archives.
- `本地计划表.md`: the single local overview plan. Use this instead of Excel/Feishu unless the user asks otherwise.

## Common Commands

```powershell
cd D:\shen\网络组\代码\scripts
python run_world_model_v4_dual_graph_rollout.py
python run_world_model_metric_suite_v0.py
python -m unittest test_dual_graph_features.py test_v4_ablation_active_rate.py
```

For reference-simulator runs:

```powershell
conda activate airfogsim
cd D:\shen\网络组\代码\reference\AirFogSim\examples
```

For LaTeX progress documents, compile with XeLaTeX.

## Research Workflow

The main line is PI-JWM:

1. Build physical-network and information-network representations.
2. Train action-conditioned state prediction and rollout models.
3. Extend to physical-information dual-graph rollout.
4. Evaluate state prediction, link activity/rate, task evolution, robustness, uncertainty, and seed transfer.
5. Use decision/ranking diagnostics only after state rollout improves.

## Rules

- Keep reusable framework code in `代码/src/pi_jwm/`.
- Keep runnable scripts in `代码/scripts/`.
- Keep third-party code in `代码/reference/`.
- Keep generated outputs in `代码/artifacts/`.
- Do not create new top-level experiment/framework folders under `代码/`; `代码/` is the PI-JWM project root.
- New PI-JWM model code must be under `代码/src/pi_jwm/`, not inside AirFogSim or historical experiment folders.
- New validation or smoke-test scripts must be under `代码/scripts/`.
- New tests must be under `代码/tests/`.
- AirFogSim-related paths may be referenced only as simulator/data-source inputs through `代码/reference/AirFogSim/` or historical artifacts under `代码/artifacts/`.
- v5 selector/ranking work is a diagnostic interface. Do not present it as the main method unless the user explicitly asks for decision-interface diagnostics.
- Update `本地计划表.md` when the plan or task status changes.
- Advisor-facing documents should use PI-JWM as the framework name.
