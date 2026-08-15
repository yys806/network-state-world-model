# Repository Guidelines

## Identity

This workspace is for **PI-JWM**: Physical-Information Joint World Model.

AirFogSim is only a reference simulator and data-generation tool. Do not describe it as the framework or the research main line.

## Theory-Implementation-Evidence Consistency

This is a non-negotiable rule for all PI-JWM work:

- 这是最高优先级的永久约束：宁可放慢进度，也绝不允许理论一套、实现一套、结果表述再一套；做不到理论定义时，必须修改理论边界或给出可核验的证明与限制，绝不能用近似接口、代理量或换名糊弄。
- The theoretical definition, advisor-facing documents/PPT, code, runtime configuration, data fields, machine-readable artifacts, and experimental claims must agree item by item.
- Do not describe an interface, executable code path, loaded latent/belief, or short smoke test as a complete theoretical method. In particular, a policy that only consumes a world-model belief is not a "world-model candidate-rollout planner" unless it actually rolls out every candidate action with the world model and uses the predicted future state, task outcome, cost, and risk to select the action.
- Every method claim must point to the implementing code path, exact inputs/outputs, tests, machine-readable artifacts, and acceptance result. A matching name, tensor shape, imported module, or successful launch is not sufficient evidence.
- When theory and implementation differ, record the mismatch immediately and stop expanding the affected experiment. Either implement and verify the theory, or revise the theory and all public wording using data/interface/proof evidence. Never hide the mismatch with renaming, vague prose, masks, proxy metrics, or stage results.
- Anything that cannot yet be implemented or verified must be labeled `target definition`, `candidate method`, or `not implemented`, with its missing conditions and validation gate. A target flowchart is not evidence of current capability.
- Before long GPU training, formal baselines, locked-test access, or final method freezing, complete a theory-code-data-metric consistency audit. Any unresolved critical mismatch blocks the run.
- Progress speed is secondary to a truthful evidence chain. Never substitute an easier implementation for the stated method without explicitly changing the method definition.

## Structure

- `代码/src/pi_jwm/`: PI-JWM framework modules.
- `代码/scripts/`: runnable scripts.
- `代码/tests/`: tests.
- `代码/reference/`: third-party references and simulators.
- `代码/artifacts/`: data, reports, figures, and generated outputs.
- `文档/`: meeting materials, papers, research documents, and archives.
- `文档/知识库/`: repository-local authority documents; `PIJWM主文档.md` fixes theory/method boundaries and `8.12之后推进.md` records current progress.
- `本地计划表.md`: the single local overview plan. Use this instead of Excel/Feishu unless the user asks otherwise.

## Common Commands

```powershell
cd D:\shen\PKU\PIJWM
$env:PYTHONPATH='D:\shen\PKU\PIJWM\代码\src'
python .\代码\scripts\run_world_model_v4_dual_graph_rollout.py
python .\代码\scripts\run_world_model_metric_suite_v0.py
python -m compileall -q .\代码\src .\代码\scripts .\代码\tests
python -m unittest discover -s .\代码\tests -p 'test_*.py'
```

For reference-simulator runs:

```powershell
conda activate airfogsim
cd D:\shen\PKU\PIJWM\代码\reference\AirFogSim\examples
```

For LaTeX progress documents, compile with XeLaTeX.

## Local Literature Management

- `文档/文献/` is the authoritative PI-JWM literature library. Read `文档/文献/README.md` and `文档/文献/文献索引.csv` before adding or moving papers.
- Store each PDF in exactly one primary category directory. Preserve cross-category relationships in `文献索引.csv` instead of duplicating files.
- Deduplicate in this order: DOI, arXiv ID, normalized title plus author/year, then PDF SHA-256. Verify the `%PDF-` file signature before accepting a download.
- After adding a PDF, update `文献索引.csv`, `PDF_SHA256SUMS.txt`, `文献索引.md`, and `本地文献库状态.json`. Remove the matching entry from `需要手动下载.md` only after the file and metadata have been verified.
- The former Zotero PIJWM collection was retired on 2026-08-15. Its final metadata, collection structure, BibTeX, attachment audit, and download results are preserved under `文档/文献/`; do not recreate or write back to that collection unless the user explicitly asks.
- Historical Zotero keys remain provenance identifiers only. Do not treat a historical Zotero attachment flag as proof that a local PDF exists; use the local path and SHA-256 index.
- `D:\shen\PKU\RRM` and its Zotero collection remain a separate reference project. Never merge their papers or claims into PI-JWM without explicit provenance and independent PI-JWM evaluation.

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
- Update `文档/知识库/PIJWM主文档.md` for theory or method-boundary changes and `文档/知识库/8.12之后推进.md` for post-2026-08-12 progress; these files are no longer maintained in the former external knowledge-base directory.
- Advisor-facing documents should use PI-JWM as the framework name.
