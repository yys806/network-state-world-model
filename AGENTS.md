# Repository Guidelines

## Identity

This workspace is for **PI-JWM**: Physical-Information Joint World Model.

AirFogSim is only a reference simulator and data-generation tool. Do not describe it as the framework or the research main line.

## User-Facing Communication Standard (Highest Priority)

- Every user-facing answer about PI-JWM must use plain, easy-to-understand Chinese. Explain technical terms in ordinary words the first time they appear; do not assume the reader already knows the project.
- Each progress answer must clearly state: what was done, what the result means, what is still missing, what is blocked, and the single next action. Include GPU and `locked_test` boundaries when relevant.
- Give the conclusion first, then the necessary evidence and details. Do not hide an important limitation behind abbreviations, dense jargon, or a list of file paths.
- This communication rule is part of the project’s hard constraints. A technically correct result is not considered properly reported if a new project member cannot understand it from the answer.

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

- `code/src/pi_jwm/`: PI-JWM framework modules.
- `code/scripts/`: runnable scripts.
- `code/tests/`: tests.
- `code/reference/`: third-party references and simulators.
- `code/artifacts/`: data, reports, figures, and generated outputs.
- `记录/`: repository-local authority documents, plans, progress, handoffs, and migration records.
- `literature/`: authoritative local literature library.
- `paper/`: formal paper materials only.
- `meeting/`: meeting materials.
- `docs/`: templates, project notes, and miscellaneous documents.
- `记录/本地计划表.md`: the single local overview plan. Use this instead of Excel/Feishu unless the user asks otherwise.

## Common Commands

```powershell
cd D:\shen\PKU\PIJWM
$env:PYTHONPATH='D:\shen\PKU\PIJWM\code\src'
python .\code\scripts\run_world_model_v4_dual_graph_rollout.py
python .\code\scripts\run_world_model_metric_suite_v0.py
python -m compileall -q .\code\src .\code\scripts .\code\tests
python -m unittest discover -s .\code\tests -p 'test_*.py'
```

For reference-simulator runs:

```powershell
conda activate airfogsim
cd D:\shen\PKU\PIJWM\code\reference\AirFogSim\examples
```

For LaTeX progress documents, compile with XeLaTeX.

## Local Literature Management

- `literature/` is the authoritative PI-JWM literature library. Read `literature/README.md` and `literature/文献索引.csv` before adding or moving papers.
- Store each PDF in exactly one primary category directory. Preserve cross-category relationships in `文献索引.csv` instead of duplicating files.
- Deduplicate in this order: DOI, arXiv ID, normalized title plus author/year, then PDF SHA-256. Verify the `%PDF-` file signature before accepting a download.
- After adding a PDF, update `文献索引.csv`, `PDF_SHA256SUMS.txt`, `文献索引.md`, and `本地文献库状态.json`. Remove the matching entry from `需要手动下载.md` only after the file and metadata have been verified.
- The former Zotero PIJWM collection was retired on 2026-08-15. Its one-time JSON/BibTeX/attachment/migration exports were removed after the local library passed migration acceptance; do not recreate or write back to that collection unless the user explicitly asks.
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

- Keep reusable framework code in `code/src/pi_jwm/`.
- Keep runnable scripts in `code/scripts/`.
- Keep third-party code in `code/reference/`.
- Keep generated outputs in `code/artifacts/`.
- Do not create new top-level experiment/framework folders under `code/`; `code/` is the PI-JWM project root.
- New PI-JWM model code must be under `code/src/pi_jwm/`, not inside AirFogSim or historical experiment folders.
- New validation or smoke-test scripts must be under `code/scripts/`.
- New tests must be under `code/tests/`.
- AirFogSim-related paths may be referenced only as simulator/data-source inputs through `code/reference/AirFogSim/` or historical artifacts under `code/artifacts/`.
- v5 selector/ranking work is a diagnostic interface. Do not present it as the main method unless the user explicitly asks for decision-interface diagnostics.
- Update `记录/本地计划表.md` when the plan or task status changes.
- Update `记录/PIJWM主文档.md` for theory or method-boundary changes and `记录/8.12之后推进.md` for post-2026-08-12 progress; these files are the repository-local authority records.
- Advisor-facing documents should use PI-JWM as the framework name.

## Mainline Control and Scope Discipline

- PI-JWM execution follows one canonical coarse-grained roadmap: `P0 -> P1 -> P2 -> P4 -> P6 -> P7+`, as recorded in `记录/本地计划表.md`. Do not invent, insert, or promote extra top-level phases while the current gate is open.
- Before starting each task, read `task_plan.md`, this file, `记录/本地计划表.md`, `记录/PIJWM主文档.md`, `记录/8.12之后推进.md`, and `记录/文件树与证据分层_20260826.md`; then state the current gate, blocker, and single next deliverable. Recheck the plan after each completed gate so execution does not drift.
- If an unexpected result, theory mismatch, or apparent blocker appears, pause the current gate and first search prior records, machine-readable audits, and relevant local literature. Reuse verified interfaces and findings before designing new code or experiments, and record whether the issue is historical, already resolved, or genuinely new.
- `R0-R9`, `P1-A`, and `P2-A` are internal substeps or historical labels only. They must not replace the `P` roadmap in user-facing progress reports or become a reason to expand scope.
- Once a route is confirmed, execute it sequentially and keep unrelated planner, robustness, paper-baseline, or locked-test work out of the active gate. Historical GPU smoke or aggregate-baseline runs must remain labeled historical/exploratory unless the current formal training gate independently approves them.
- The current pre-GPU chain is fixed: freeze the selected aggregate training contract; freeze tensor contract and train-only statistics; close model/loss/metrics consistency; pass CPU micro-training and checkpoint reload; freeze the formal training protocol; complete an independent Go/No-Go audit. Only then may formal GPU training start.
- At every checkpoint, report only three things: completed evidence, remaining blocker, and the single next action. Do not multiply tasks, rename an unresolved mismatch, or claim completion beyond the recorded evidence.
- Every task must leave a short entry in `task_plan.md`, `progress.md`, and `findings.md`. Keep these root files as process records; only the authority files and approved manifests/artifacts can establish a final method or formal claim.

## Knowledge Index Maintenance

- `docs/PROJECT_INDEX.md`, `docs/ARCHITECTURE.md`, `docs/RESEARCH_STATUS.md`, `docs/EXPERIMENT_INDEX.md`, and `docs/RESULTS_INDEX.md` are navigation and status summaries. They never replace code, configuration, raw metrics, checkpoints, manifests, or audits as evidence.
- After adding or changing an important module, experiment, result, configuration, path, or research boundary, update the affected index and `docs/CHANGELOG.md` in the same task.
- When a result changes, trace it in both directions: research question → method → code → experiment → result, and result → experiment → configuration → data/model/code.
- Before claiming an index is current, recheck the underlying source and machine-readable artifact. If an index conflicts with newer evidence, mark the conflict and fix the index; do not alter research code or evidence merely to make the text look consistent.
- During an active remote run or file synchronization, do not move, rename, delete, overwrite, or bulk-rewrite `code/artifacts/experiments/`, `formal_tensor/`, `formal_data/`, `protocol/`, `protocols/`, live evidence, staging, checkpoints, predictions, runtime files, manifests, or transfer directories. Use read-only inspection and additive documentation only.
- Archiving is preferred to deletion. Any future move or archive requires a user-approved scope, a per-file mapping with hashes, a rollback path, and validation that no active process or synchronizer is writing the target.
