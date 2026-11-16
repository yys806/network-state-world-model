# Repository Guidelines

## Identity

This workspace is for **PI-JWM**: Physical-Information Joint World Model.

AirFogSim is only a reference simulator and data-generation tool. Do not describe it as the framework or the research main line.

## Project Memory and Context Recovery (2026-10-23)

- **Canonical research-notes root (read-only):** `D:\shen\OB\科研\PKU\PIJWM`. Obsidian vault root is `D:\shen\OB`; image attachments live in `D:\shen\OB\图片`. The older path `D:\shen\OB\科研\PIJWM` no longer exists. The file map, source priority, reading status and known conflicts are in `docs/PROJECT_MEMORY_AND_SOURCES.md` — read it when a task needs project context.
- **Current research scheme (highest priority for direction questions):** the 2026-10-23 meeting deck `PPT\10.23.pptx` and the 2026-10-21 meeting record `组会笔记\1021组会.md` (its script copy is `组会笔记\1028组会.md`), all under the canonical root. If the two conflict, record the conflict with evidence; never merge them or silently pick one.
- **The research scenario has switched.** The old scenario (AirFogSim UAV/vehicle network, task offloading, network-state world model) is **historical**. The current candidate scenario is vehicle–roadside cooperative perception with road-vehicle object detection. The old scenario is not the default object of study, and old modeling mechanisms may be reused only after fresh evaluation.
- **New-scenario status:** candidate Problem Formulation only. No new definition document, no new-scenario code, no authorized implementation Step. The deck itself calls it a "candidate scenario", and `1028组会.md` still lists open scenario questions. Do not describe it as an implemented or frozen method.
- **The `00–06` definitions in the notes root describe the OLD scenario** (written 2026-09-15~18). They remain the definition basis of the existing implementation but are not the new scenario's definition, and must not be presented as such.
- **Old-material boundary:** older meetings/decks, the `00–06` documents, existing code, checkpoints and historical Step records may be used to understand evolution, reuse verified mechanisms, and build comparisons. "Old code already implements X" never authorizes X in the new scenario.
- **Recovery order for any new PI-JWM conversation:** (1) classify the question; (2) read this file plus `docs/PROJECT_MEMORY_AND_SOURCES.md`; (3) for direction questions read the 10.23 deck and the 10.21 record first; (4) for engineering facts verify current source/config/experiment artifacts — `AI_CONTEXT` is navigation only; (5) for literature use the Zotero `PKU → PIJWM` collection and accessible full text, not paper-note memory; (6) label each claim as current implementation fact / historical design / unverified hypothesis / future proposal; (7) read only what the question needs instead of scanning the whole repository.
- **Cross-conversation persistence:** pi has no automatic chat memory. Persistence relies only on the agent-dir `AGENTS.md`, this file, and the documents they point to. Sessions under `~/.pi/agent/sessions/` load only on explicit resume. Never claim to remember an earlier conversation.
- **Source priority:** current source/config/experiment (engineering facts) > latest meeting material (direction and decision context) > `AI_CONTEXT` (navigation) > ordinary docs and paper notes > historical chat or memory. This file and `docs/PROJECT_MEMORY_AND_SOURCES.md` store the map, not the facts themselves.

## Research Engineer Role and Three-Party Boundary

- The project **Research Engineer** role is held by the executing agents: **研Dive** (local Hermes research profile, primary executor) and, when called, **Codex** (gpt-6-astra) and **Claude Code** (opus 5.5). **Pi** serves only as an independent third opinion and does not implement. All of them must understand the research context well enough to implement accurately, but none owns final research decisions.
- The researcher owns the research question, core algorithm, mathematical model, overall architecture, scientific loss design, hypothesis, experiment purpose, ablation variable, innovation claim, and final interpretation.
- 研Dive covers literature, method, experiment design and execution, code, drafting and result recording; Codex and Claude Code cover implementation, bulk refactoring, independent acceptance and review. Theory discussion and result interpretation are joint work between the researcher and the executors — there is no separate web-based agent in the loop.
- Any executor may proactively identify implementation conflicts, leakage, unfair comparisons, evaluator risks, or unsupported assumptions. It must report facts and evidence, then stop the affected scientific change when a researcher decision is required.

## User-Facing Communication Standard (Highest Priority)

- Every user-facing answer about PI-JWM must use plain, easy-to-understand Chinese. Explain technical terms in ordinary words the first time they appear; do not assume the reader already knows the project.
- Each progress answer must clearly state: what was done, what the result means, what is still missing, what is blocked, and the single next action. Include GPU and `locked_test` boundaries when relevant.
- Give the conclusion first, then the necessary evidence and details. Do not hide an important limitation behind abbreviations, dense jargon, or a list of file paths.
- This communication rule is part of the project’s hard constraints. A technically correct result is not considered properly reported if a new project member cannot understand it from the answer.

## Human-Driven Research Decisions

- The user owns the research problem, direction, key hypotheses, method choices, core experiment purpose, and final scientific interpretation. AI may organize evidence and propose alternatives, but must not silently turn an implementation convenience into a research decision.
- Before changing a hypothesis, model structure, loss, metric, protocol, threshold, or claim boundary, present the evidence, alternatives, and consequences and wait for the user's explicit decision.

## Independent AI Assessment

- Do not merely agree with a proposed idea. Identify its assumptions, possible counterexamples, confounding factors, simpler explanations, and whether the proposed experiment can actually distinguish the claimed mechanism.
- Do not oppose an idea for appearance. Every objection or alternative must be grounded in theory, mathematics, literature, code, data, or experiment evidence, with uncertainty stated plainly.

## Explanation Ladder

- Explain technical material in this order when relevant: problem being solved → intuition → why it is needed → concrete mechanism → mathematical form and every variable → implementing code path → expected experimental effect and falsification test.
- If one answer does not need every level, omit irrelevant levels but never replace an explanation with unexplained terminology or a bare formula.

## Evidence-Bounded Language

- Avoid empty claims such as “improves robustness” or “improves generalization.” State the affected object and perturbation/distribution, proposed mechanism, measured metric, comparison, evidence scope, and what remains unverified.
- Distinguish target definition, hypothesis, implementation, runtime execution, diagnostic result, accepted result, and final scientific conclusion. A later layer may not be inferred from an earlier one.
- Follow `docs/COLLABORATION_GUIDE.md` for the permanent user–AI division of work and answer format.

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
- `literature/`: retired on 2026-10-16; the PI-JWM literature library now lives in the Zotero `PKU → PIJWM` collection.
- `paper/`: formal paper materials only.
- `meeting/`: meeting materials.
- `docs/`: templates, project notes, and miscellaneous documents.
- `AI_CONTEXT/`: concise, stable context layer for ChatGPT Web; navigation only, never a replacement for source/config/experiment evidence.
- `记录/本地计划表.md`: the single local overview plan. Use this instead of Excel/Feishu unless the user asks otherwise.

## Common Commands

```powershell
cd D:\shen\PKU\PIJWM
$env:PYTHONPATH='D:\shen\PKU\PIJWM\code\src;D:\shen\PKU\PIJWM\code\scripts'
python -m compileall -q .\code\src .\code\scripts .\code\tests
python -m unittest discover -s .\code\tests -p 'test_*.py'
python .\code\scripts\build_project_knowledge_index_v1.py
python .\code\scripts\build_project_knowledge_index_v1.py --check
```

For reference-simulator runs:

```powershell
conda activate airfogsim
cd D:\shen\PKU\PIJWM\code\reference\AirFogSim\examples
```

For LaTeX progress documents, compile with XeLaTeX.

## Local Literature Management

- The PI-JWM literature library now lives **only in Zotero** — collection `PKU → PIJWM` (key `PXDR9FBZ`). The long-term primary-category system is fixed at ten collections: `01_系统场景与问题定义` (existing key `UUK8F8EQ`), `02_严格双图与图学习` (`283ARJS5`), `03_世界模型与状态预测` (`EV5VZHH3`), `04_任务卸载与资源优化` (`CI9HW9D9`), `05_数据集、测量与仿真` (`SXVPC5G4`), `06_评价、不确定性与决策诊断` (`8VKHS5YK`), `07_AI业务场景与任务模型` (`DQQG4EIB`), `08_Token与语义通信` (`VLA67XIK`), `09_网络QoS与任务质量映射` (`9ELTVQWN`), and `99_综述与奠基性文献` (`VXXZ7BJ4`). Do not rename, delete, or add a permanent eleventh category without researcher approval. Each paper has one primary collection; cross-category relationships live in tags instead of duplicated items. A paper is classified by its main contribution, verified from the paper or authoritative abstract, not by title keywords or by the method used as a tool.
- Each paper is one Zotero item with its full-text PDF attached, plus CCF / impact-factor / CAS-partition fields and (when generated) a bilingual reading copy.
- The local `literature/` directory was retired on 2026-10-16 (its CSV/SHA-256 index and notes are archived under `doc/literature-notes/`). Do not recreate a parallel PDF folder, and do not maintain a second literature source.
- Adding new literature: import the PDF into the `PIJWM` collection, deduplicate by DOI / arXiv ID / normalized title / SHA-256, run "Retrieve Metadata for PDFs", then refresh the CCF / impact-factor fields. Verify the `%PDF-` file signature before accepting a download.
- `D:\shen\PKU\RRM` and its Zotero collection remain a separate reference project. Never merge their papers or claims into PI-JWM without explicit provenance and independent PI-JWM evaluation.

## Active Research and Implementation Workflow (2026-09-18)

- Researcher-authorized target definitions are the seven `00–06` Markdown documents in `D:\shen\OB\科研\PKU\PIJWM\PIJWM` (moved there on 2026-10-23 from `D:\shen\OB\科研\PIJWM`). They describe the pre-2026-10-23 scenario; see "Project Memory and Context Recovery" above. They define intended behavior; only source/config/test/artifact can establish implementation facts. Do not confuse a target definition with an implemented or accepted method.
- Active definition order: `00 总体研究链路 → 01 仿真系统与轨迹 → 02 数据集与模型输入 → 03 Physical–Information 双图 → 04 世界模型 → 05 Training / Loss / Evaluation → 06 Planner / Closed Loop`.
- Engineering workspace: `D:\shen\PKU\PIJWM`. All code, configs, tests, experiments, documentation, trackers, records and Git work must stay here. Repository: `https://github.com/yys806/network-state-world-model`.
- Current implementation entrypoints: `docs/PIJWM_IMPLEMENTATION_TRACKER.md`, `docs/implementation_records/README.md`, and the authorized Step record. Definition chapter numbers are not automatic authorization to execute implementation Steps.
- Step 1 is **New Definition → Current Implementation Audit** only. Classify each item as `DIRECT_REUSE`, `MINOR_MODIFICATION`, `STRUCTURAL_CHANGE`, `MISSING`, `RESEARCHER_DECISION_REQUIRED`, or `HISTORICAL_ONLY`. No model restructuring, dataset rebuild, new planner, formal training, or automatic Step 2.
- Priority: **new-definition correctness > old-code reuse**. Reuse verified mechanisms when they fit; use minimal changes when sufficient; restructure only for a demonstrated semantic or structural conflict. Never change the target to preserve an old checkpoint.
- Each meaningful Step/Substep must be verified, recorded, diff-reviewed, committed and pushed, then reported to the researcher and **stopped**. A suggested next step is not permission to execute it. Never automatically complete multiple major Steps.
- Before proposing expensive training, require schema/tensor checks, unit tests, tiny forward/backward, finite gradients, tiny-data overfit, short smoke training, action sensitivity, rollout sanity, and learning-signal checks. These must exercise the new contract. Historical passes do not authorize long GPU, multi-seed, hyperparameter search or expensive remote jobs.

### Historical / Archived roadmap

The previous P0–P10/P4/P6 plans and older five-part research workflow are **Historical / Archived**, not the active execution sequence. Preserve their records, code, protocols, artifacts and checkpoints in place for provenance. Their observations remain valid only within their original definitions and acceptance scope; historical success does not establish the new method. This is a logical archive, not permission to move/delete evidence or resume deferred runs.

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
- Update `记录/本地计划表.md` when the plan or task status changes.
- Update `记录/PIJWM主文档.md` for theory or method-boundary changes and `记录/8.12之后推进.md` for post-2026-08-12 progress; these files are the repository-local authority records.
- Advisor-facing documents should use PI-JWM as the framework name.

## Mainline Control and Scope Discipline

- PI-JWM follows the active `00–06` definition chain and researcher-authorized implementation Step recorded in `docs/PIJWM_IMPLEMENTATION_TRACKER.md` and `记录/本地计划表.md`. Older P-stage gates are historical, never an automatic execution queue.
- Before starting each task, read `task_plan.md`, this file, `记录/本地计划表.md`, `记录/PIJWM主文档.md`, `记录/8.12之后推进.md`, and `记录/文件树与证据分层_20260826.md`; then state the current gate, blocker, and single next deliverable. (`task_plan.md`, `progress.md` and `findings.md` are currently absent from the working tree — see `docs/PROJECT_MEMORY_AND_SOURCES.md` §6 C7 — so record plan/progress in `记录/本地计划表.md` until the researcher restores or replaces them.) Recheck the plan after each completed gate so execution does not drift.
- If an unexpected result, theory mismatch, or apparent blocker appears, pause the current gate and first search prior records, machine-readable audits, and relevant local literature. Reuse verified interfaces and findings before designing new code or experiments, and record whether the issue is historical, already resolved, or genuinely new.
- Internal substeps or historical labels must not replace the canonical roadmap in user-facing progress reports or become a reason to expand scope.
- Once a route is confirmed, execute it sequentially and keep unrelated methods, planner work, robustness, baselines, or locked-test work out of the active gate. Historical smoke or diagnostic runs remain historical unless the current formal gate independently approves them.
- Before any long GPU training, freeze the method/data/tensor/loss/metric/runtime contract, pass CPU execution and reload checks, freeze the training protocol, and complete an independent Go/No-Go audit. The exact current chain belongs in the authority plan and `AI_CONTEXT`, not in this permanent rule file.
- During work, report completed evidence, remaining blocker and the single next action. At Step/Substep completion use the fixed Completion Report below, then stop. Do not rename an unresolved mismatch or claim completion beyond evidence.
- Every task must leave a short entry in `task_plan.md`, `progress.md`, and `findings.md`. Keep these root files as process records; only the authority files and approved manifests/artifacts can establish a final method or formal claim.

## AI_CONTEXT Maintenance and Context Consistency Check

- `AI_CONTEXT/00_PROJECT_STATE.md` is the first entrypoint for a new ChatGPT Web conversation. Keep it short and current; route details to `01`–`08` and then to real source/config/experiment paths.
- Project fact priority is: current source/config/experiment > `AI_CONTEXT/` > ordinary project documentation > historical chat or inference. Mark unverifiable statements `Unverified`; mark non-code motivation `Research Rationale`.
- After every effective task, run a **Context Consistency Check** across current state, research context, architecture, data flow, module map, experiments, researcher decisions, known issues, and important changes. Update only affected `AI_CONTEXT/` files.
- If code changes an important architecture, data flow, experiment state, or known issue, the matching `AI_CONTEXT/` update belongs in the same task and commit.
- `AI_CONTEXT/06_DECISIONS.md` may record a research decision only when the researcher explicitly made it. Codex analysis or implementation convenience is not a decision.

## Conflict Handling

- When documented intent, user instruction, `AI_CONTEXT/`, and actual source behavior materially conflict, do not silently rewrite either side to look consistent.
- Record `Documented Intent`, `Actual Implementation`, `Evidence`, `Affected Files`, `Conflict`, and `Status: Awaiting Researcher Decision`; stop the affected scientific modification and ask the researcher.

## Git Workflow

- Normal bounded tasks are developed directly on `main`; use a temporary branch only for higher-risk validation when useful.
- Each effective, independently understandable task must have a clear Conventional Commit. A larger task may use several independently reversible commits.
- After reasonable verification, perform **commit + push** so GitHub stays close to the latest trusted local state. The researcher has explicitly authorized routine pushes for this repository without repeated confirmation.
- Never push a known broken state to `main`. If new work cannot run or its required validation fails, keep it local, report the exact failure, and do not push merely to satisfy the workflow.

## Research Notes Read-Only Boundary

- The researcher explicitly authorized read-only access to `D:\shen\OB\科研\PIJWM` on 2026-09-18; that directory moved to `D:\shen\OB\科研\PKU\PIJWM` on 2026-10-23 and the read-only boundary follows it. Read/analyze the designated documents (now including the 10.23 deck and the 10.21 meeting record) and cite their file/section/hash as basis. Never modify, create, delete, rename, format or overwrite anything in that directory or in the vault's `图片` folder.
- Do not scan other private-note directories. Store implementation mappings and source fingerprints in the engineering repository; do not copy the entire private notebook into GitHub.
- Historical rule: **Private Notes Prohibition** previously prohibited all access. It is superseded only for the explicitly authorized directory and read-only purpose above; access elsewhere remains prohibited.

## AGENTS.md Change Control

- After the user-authorized 2026-09-10 governance update, modify this file **only when the user explicitly asks to modify AGENTS.md** or explicitly changes long-term collaboration rules. Codex must not alter its own authority autonomously.

## Completion Report

- Use `STEP X.X — 名称` followed by: `1. 本次做了什么`, `2. 得到了什么结果`, `3. 是否符合预期` (符合预期/部分符合/不符合), `4. 当前还存在的问题`, `5. 记录位置`, `6. Git 状态` (Commit hash/message, Push status, GitHub branch), `7. 建议下一步` (one minimal independently verifiable action). Include affected files, verification and AI_CONTEXT updates where relevant. Then stop and wait for researcher review.
- Each Step record must include Step Goal, Definition Basis, Initial State, Files Involved, Changes, Reuse, Validation (actual commands/outputs), Results, Expected vs Actual, Known Issues, Git, and Next Step. Keep implementation observations separate from scientific conclusions.

## Knowledge Index Maintenance

- `docs/PROJECT_INDEX.md`, `docs/ARCHITECTURE.md`, `docs/RESEARCH_STATUS.md`, `docs/EXPERIMENT_INDEX.md`, and `docs/RESULTS_INDEX.md` are navigation and status summaries. They never replace code, configuration, raw metrics, checkpoints, manifests, or audits as evidence.
- After adding or changing an important module, experiment, result, configuration, path, or research boundary, update the affected index and `docs/CHANGELOG.md` in the same task.
- When a result changes, trace it in both directions: research question → method → code → experiment → result, and result → experiment → configuration → data/model/code.
- Before claiming an index is current, recheck the underlying source and machine-readable artifact. If an index conflicts with newer evidence, mark the conflict and fix the index; do not alter research code or evidence merely to make the text look consistent.
- Keep `docs/registries/` synchronized with important file, dependency, experiment, result, authority, and deferred-work changes. After such a change, run `python code/scripts/build_project_knowledge_index_v1.py` and then rerun it with `--check`.
- When answering a project-history or provenance question, query `docs/registries/question_routes.json`, `experiment_registry.json`, `results_registry.json`, and `historical_method_registry.json` before broad repository scanning; then verify the selected claim against the listed original evidence.
- Every important experiment entry must use the complete registry schema. Unknown or not-applicable fields must be explicit `null` values with a reason; never omit them or invent a value. Formal result numbers must pass the acceptance-evidence comparison performed by the index builder.
- Treat `docs/registries/deferred_work.json` as a stop guard, not an execution queue. An item with `authorization_required=true` and `auto_start=false` must never be started by automation or inferred approval.
- During an active remote run or file synchronization, do not move, rename, delete, overwrite, or bulk-rewrite `code/artifacts/experiments/`, `formal_tensor/`, `formal_data/`, `protocol/`, `protocols/`, live evidence, staging, checkpoints, predictions, runtime files, manifests, or transfer directories. Use read-only inspection and additive documentation only.
- Archiving is preferred to deletion. Any future move or archive requires a user-approved scope, a per-file mapping with hashes, a rollback path, and validation that no active process or synchronizer is writing the target.

## 文献库与工具链（唯一入口：Zotero）

**唯一真相**：Zotero 集合 `PKU → PIJWM`。本地 `literature/` 已于 2026-10-16 退役，索引/说明类文件存档于 `doc/literature-notes/`。
**新增文献只在 Zotero 里加**，不要再在仓库里建 PDF 文件夹。本轮第二次补充前盘点基线为 **97 篇**；本轮新增 10 个 Zotero 条目，并复用 1 个原属于 RRM 的同论文条目加入 PIJWM，回读确认 PIJWM 当前为 **108 篇**（全库主条目为 244 条；附件记录为 243 条）。十个子集合及其 Key 以本节上方清单和 Zotero 回读为准；07–09 是固定入口，创建前必须先检查是否已经存在，创建后必须回读集合和条目数量。跨项目复用必须保留原项目集合关系，不得复制条目。

### 十类长期分类和核验口径
- `01_系统场景与问题定义`：系统场景、网络架构、系统模型、研究问题、物理—信息系统定义。
- `02_严格双图与图学习`：图神经网络、异构/动态图、物理—信息双图、消息传递、跨域耦合和图结构表示。
- `03_世界模型与状态预测`：World Model、RSSM、latent dynamics、动作条件预测和预测式环境建模。
- `04_任务卸载与资源优化`：任务卸载、通信/计算资源、调度、MPC、RL、CEM、联合优化和规划策略。
- `05_数据集、测量与仿真`：数据集、无线测量、仿真器、真实测试平台和 benchmark。
- `06_评价、不确定性与决策诊断`：预测误差、不确定性、校准、可靠性、规划偏差和评价/诊断方法。
- `07_AI业务场景与任务模型`：视觉检测/识别/跟踪、协同感知、边缘 AI 推理、LLM/多模态业务及具体任务模型。
- `08_Token与语义通信`：Token、语义特征、Token 重要性、选择/压缩、语义编码、特征传输和价值感知通信。
- `09_网络QoS与任务质量映射`：吞吐、时延、丢包、QoS、信息时效，以及网络交付与 AI 任务准确率/完成质量/任务效用之间的映射。
- `99_综述与奠基性文献`：跨领域综述或明显奠基工作；不能作为“暂时无法分类”的收容区。每篇论文只放一个主分类，跨方向用主题标签表达。
- 新论文流程必须是“确认正式来源 → DOI/arXiv/正式题名/SHA-256 查重 → 阅读摘要、引言、问题定义和主要贡献 → 选择主分类 → 入库 → 回读元数据、附件和集合归属 → 添加规范标签”。不得只按标题关键词分类；分类变化要记录原分类、新分类、主要贡献和核验依据，并将证据不足或主贡献有歧义的条目标为 `REVIEW_REQUIRED`。
- 批量创建集合、移动条目或修改元数据前，必须先确认可恢复的 Zotero 导出或同步快照，并能回读验证。只有空的/过期的 SQLite `.bak`、未验证的 WebDAV 或被锁定的数据库不构成恢复安全门；安全门未通过时只做只读审计，不做批量写入。

### 规范标签词表
- 继承历史标签，不批量删除；同一语义只新增一种规范写法。角色使用 `PIJWM/角色/核心`、`PIJWM/角色/支撑`、`PIJWM/角色/奠基`；证据使用 `PIJWM/证据/正式发表`、`PIJWM/证据/预印本`、`PIJWM/证据/破格`；全文状态使用 `PIJWM/全文/已核验PDF`、`PIJWM/全文/仅元数据`、`PIJWM/全文/预印本版本`。
- 主题标签使用稳定术语：`PIJWM/主题/系统场景`、`严格双图`、`图学习`、`世界模型`、`任务卸载`、`资源分配`、`数据与测量`、`评价诊断`、`AI业务`、`协同感知`、`Token与语义通信`、`任务感知QoS`、`网络QoS`、`网络可靠性`、`鲁棒性`、`信息时效`。主题标签表达跨分类关系，不代替主集合；不得把 Agent 自动读取标成“研究者已精读”。
- PDF 机器翻译必须作为独立附件保留，原文不能被替代；已有翻译附件则跳过。正式发表、预印本和仅附件记录必须分别按事实标记，不能由附件数量推断全文已核验。

### 新增一篇文献的标准流程
1. **建条目**：Zotero 里在该集合对应子分类下 → `Add Item by Identifier` 贴 DOI（最准）；只有 PDF 时拖入 PDF → 右键 `Retrieve Metadata for PDF`。
2. **核对五项元数据**：标题（第一列必须是正式题名，不能是文件名、作者行、页眉）、创建者、日期、期刊/会议名、刊名简称。
3. **补期刊标签/指标**（任一或都跑，幂等）：
   - `python D:\hermes\scripts\zotero_pipeline.py ranks --project PIJWM`
   - `python D:\hermes\scripts\complete_journal_metrics.py --project PIJWM`
   写入 Extra 的字段（与 Zotero「Green Frog/青蛙」插件同格式）：`影响因子 / 5年影响因子 / JCR分区 / 中科院分区基础版 / 中科院分区升级版 / CCF / EI / SSCI / AJG / JCI / ESI`。
   数据源＝easyScholar（与用户 Zotero 插件同源）+ CCF 第七版官方目录；**不得臆造**：会议论文与预印本没有期刊指标，就按事实写 `无（会议/预印本无期刊指标）`；不在 CCF 目录写 `无（不在 CCF 目录）`——**字段不留空**。指标必须保留来源年度/版本；字段状态只允许据实区分 `VERIFIED`、`NOT_APPLICABLE`、`UNVERIFIED`、`CONFLICT`。
4. **双语对照 PDF（左英右中，不外传）**：
   - 本地 PDF2zh 服务目录 `C:\Users\Lenovo\Zotero\zotero-pdf2zh\server`，端口 `8890`，引擎 `siliconflowfree`，启动命令：`uv run python server.py --port 8890 --env_tool uv --enable_venv true`
   - 批量/单篇：`set PIPELINE_PROJECT=PIJWM` 后跑 `python D:\hermes\scripts\zotero_translate_all.py`
   - 产物作为**新附件**挂回原条目（标题固定 `双语对照 PDF（机器翻译）`，标签 `translated:pdf2zh`），原文 PDF 不动；已存在则跳过（幂等，可反复重跑）。
5. **本地库已退役**：仓库里不再保留论文 PDF 文件夹；`doc/literature-notes/` 只放索引、说明、矩阵一类非论文文件。

### 期刊简称标准化（必做，别手工敲）
简称要**出版商官方简称**，不要自己拼。做法：
- 有 DOI 的条目：`python D:\hermes\scripts\standardize_abbrev3.py --project <PROJ> --apply`（按 DOI 从 Crossref 取官方简称与官方全称；顺带修正写错的刊名）
- 会议/预印本：用学界通用短名（ICLR / NeurIPS / ICML / arXiv / LNCS / IJCAI / GLOBECOM …），脚本已内置
- 机械缩写兜底：`Transactions on→Trans.`、`Journal of→J.`、`Communications→Commun.` 等
- ⚠️ **禁用「按题名反查 Crossref」**：实测会把书名/无关条目配成别的刊（曾把 "attention is all you need" 配成别的书）
- 拿不准就别填：宁可留空，也不要写一个错的简称

### 维护边界（重要）
- 本节是文献管理的**唯一操作口径**；后续接手本项目的 agent 按本节执行，不要另建本地文献库、不要把 PDF 塞回仓库。
- 只改本节，勿改本文件其他章节（除非用户明确要求）。
- Zotero 附件用**导入式附件 + 本地 API 写入**，不烧 zotero.org 云配额；Zotero 写入需 10.x + 本地 key（缓存在 `D:\hermes\cache\zotero-local-key.json`，失效时跑 `python D:\hermes\scripts\zotero_get_key.py` 重新授权）。


## WSL 长期使用规范（2026-10-24；PIJWM-WSL-CLEANUP-001）

- 研究者于 2026-10-24 明确允许今后按任务需要使用 Ubuntu WSL；PI-JWM 唯一允许的 WSL 项目工作目录为 `/home/shen/PIJWM`，Windows 入口为 `\\wsl.localhost\Ubuntu\home\shen\PIJWM`，仅包含该目录及其真实子目录。
- 项目代码、输入数据、项目专属虚拟环境/依赖安装、缓存、临时文件、日志和实验产物均须限定在上述目录内；运行前固定工作目录，将可控 TEMP/TMP/TMPDIR、包缓存及 Python 字节码缓存重定向到目录内，或禁用缓存写入，不永久修改系统环境变量。
- 不以其他 WSL 目录、其他项目、共享 Conda 环境或 `/mnt/c`、`/mnt/d` 为本项目工作区；不得向外扫描、复制、迁移、清理或通过符号链接/挂载点绕过边界。调用系统命令、系统 Python 及其正常运行必需的系统库，不构成扩展项目文件操作范围的授权。新增项目环境优先放在该目录的 `.venv` 中，不自动使用或改动其他目录的旧环境。
- 需要目录外项目文件操作、系统级安装、全局 WSL 配置或 VHDX 维护时，须另获研究者明确授权；过去的一次性清理/缩容/压实授权不自动延续。WSL 可用不等于已批准正式训练、GPU、大型实验或 locked_test 访问，原有研究门和审批边界继续有效。
- D 盘主项目仍为工程主目录；不得自动复制整个项目或大型正式数据集到 WSL。本规则仅保存在 PI-JWM 仓库内的项目记忆和使用规范，不更新全局、个人或跨项目记忆。

- Ubuntu WSL 的统一 PI-JWM 工作目录为 `/home/shen/PIJWM`；WSL 主要承担 Linux 专有功能和经批准的临时实验，不是 D 盘主项目的完整镜像。
- 不自动复制 D 盘大型正式数据集；代码、输入、依赖和输出须按具体批准的实验方案按需准备。不得覆盖 D 盘现有科研资产。
- Ubuntu 虚拟磁盘已于 2026-10-24 经研究者明确批准设为 20 GiB 最大容量（21474836480 B）；不是 RAM 配额。文件系统元数据及预留空间会减少普通用户可用量，实验前须按实际可用空间保留至少 5 GiB，不能仅用“20−15”保证余量。实际使用量、宿主物理占用和最大容量以最新环境记录分别为准，不得混为同一指标。
- 大文件、缓存、依赖和实验产物必须事先评估空间与恢复成本；大型实验须经研究者批准。
- 不自动删除唯一科研资产、checkpoint、研究证据、未推送提交、用户修改、未跟踪唯一文件或 locked_test；不得清理非 PIJWM 项目、私人文件、凭据及共享环境。
- 不擅自修改 Windows 全局 WSL 配置或 VHDX，不自动迁移、缩容、导出导入、重注册、注销或重装发行版。
- 不主动向 C 盘新增项目数据、备份或大型临时文件。WSL swap 的系统默认临时位置在 C 盘，尚未改动；迁移配置必须另行获得批准。
- WSL 环境可用性不是新的科研事实，不代表新场景、算法或实验已实现/验收。
- 使用前先读 `docs/WSL_ENVIRONMENT.md` 与 `docs/WSL_CLEANUP_20261024.json`。当前目录仅有说明、非 Git 仓库；系统 Python 3.12.3 无 NumPy/PyTorch/pytest，不得声称科学计算环境已就绪。

## Executor and Artifact Placement (2026-11-16, researcher-authorized)

This section was added with the researcher's explicit authorization on 2026-11-16. Only the role paragraph above and this section were changed; all other sections stand.

- **Artifacts live in this repository.** Experiment code, configs, generated outputs, checked-in evidence and figures belong under `code/`, `docs/`, `记录/` and `code/artifacts/`. Do not park experiment results in `D:\hermes-yan`; the researcher must be able to see, push and revert them from GitHub.
- **Records split in two layers.** Reproducible experiment evidence lives here; decision records, project understanding and retrospectives live in `D:\hermes-yan\projects\PIJWM\`.
- **Research notes stay read-only.** `D:\shen\OB\科研\PKU\PIJWM` remains read-only (see the boundary section above). No exception is granted by this section.
- **External agents** called for experiment work (Codex / Claude Code / Pi) must use this repository as their working directory. Pure text review may run in a scratch directory, but its output must not be written into this repository.
- **Git.** Routine commit + push after verification is authorized (see Git Workflow); never push a known-broken state. Before staging, review the full diff for unpublished manuscript text, raw datasets, credentials and machine-specific paths.
- **Large data boundary unchanged.** Raw datasets, checkpoints, weights, logs and caches stay local and out of Git (`code/artifacts/` holds reports and compact outputs only).
