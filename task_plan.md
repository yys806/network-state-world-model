# 2026-08-26 双约束实施计划 v2

## 2026-09-20 STEP 4.2A-PATCH — Comm Mask Semantics & Gap Reclassification

- 当前门：只解耦 wireless structural relation validity 与 CSI feature observability，并纠正 Task observer-source gap 分类；不新增 Raw 字段、不实现 stateful Flow 或 Graph Builder。
- [x] 核实 `airfogsim_full_dual_graph_observer_v1._physical_structure()` 与 `_extract_tasks()` 的真实源码语义。
- [x] TDD 固定 wireless endpoints present + CSI missing、wired valid/no-CSI、inactive relation feature mask、placeholder 与 validator negative tamper。
- [x] 最小修改 Sample/Tensor mask 语义、机器 validator 和 remaining-gap classification。
- [x] 重建 receipt/gap overlay/manifest，同步 Step 4.2A record/contract、Tracker、必要 AI_CONTEXT 和过程记录。
- [x] 运行 focused、Step 4.1/3.3/3.2/Raw 回归、deterministic rebuild/hash、round-trip、compileall、knowledge index write/check、diff checks；待 commit + push `main` 后停止。
- 边界：`graph_builder=false`、`training=false`、`gpu=false`、`locked_test=false`、`formal_dataset=false`；`past_outcome_flow_service` 继续禁止作为 current Flow state。
- 唯一下一动作：完成本 Patch；通过后 STEP 4.2A COMPLETE / FROZEN，等待研究者授权 STEP 4.2B audit。

## 2026-09-19 STEP 3.3F

- Scope: finalize JSON sample -> tensor semantic completeness only; no graph/model/training work.
- Completed: Past Outcome H-1 tensors, full Target facts/service, `allocated_cpu_per_s`, fixed vocabularies, causal entity type amendment, semantic receipt and regenerated development artifacts.
- Gate: `formal_dataset=false`, `training=false`, `gpu=false`, `locked_test=false`.
- Status: STEP 3.3 and definition 02 COMPLETE / FROZEN for the current minimal data contract.
- Next: researcher authorization for 03 object-field-relation mapping freeze; do not implement a full GNN automatically.

## 2026-09-19 STEP 3.2

- Scope: Raw -> trajectory-level split -> causal H=2/L=2 windows -> train-only masked preprocessing -> batch/load.
- Completed: seed 1/2 non-locked real trajectories, 3-trajectory bundle, 12 windows, deterministic manifest, stats, apply and round-trip.
- Gate: `formal_dataset=false`, `training=false`, `gpu=false`, `locked_test_accessed=false`; no Tensor/model/loss/planner work.
- Next: researcher review; do not automatically enter a later Step.

## 2026-09-19 STEP 3.2-PATCH

- Scope: finalize Dataset isolation evidence only: Raw provenance/lineage, real time-grid continuity, development future-reference batch audit, normalization units, presence padding counterfactual, computed validation checks.
- Result: 3 trajectories, 12 candidate/12 constructed windows, 0 unresolved references; observation-only and non-locked.
- Gate: formal_dataset=false, training=false, gpu=false, locked_test_accessed=false; no Tensor/model/loss/planner work.
- Next: researcher review STEP 3.2-PATCH; do not automatically enter a later Step.

## 2026-09-19 STEP 3.2-PATCH-RECEIPT

- Scope: receipt logic only; no Step 3.2 mechanism redesign or new data.
- Completed: required acceptance checks and explicit scope booleans are ANDed into top-level `passed`; negative fixture passes; sample contract version reuses frozen schema constant.
- Status: STEP 3.2 COMPLETE / FROZEN after final receipt regeneration.
- Next: researcher review; do not automatically enter a later Step.

## 2026-09-19 STEP 3.3

- Scope: frozen Step 3.1F JSON sample / Step 3.2 batch -> CPU fixed-shape tensor/collation only.
- Completed: stable ID/index slot mapping, independent presence/feature masks, four action axes, relation/DAG/target namespaces, explicit overflow rejection and NPZ round-trip; 12-sample development artifact.
- Gate: `formal_dataset=false`, `training=false`, `gpu=false`, `locked_test=false`; no dual graph/model/loss/planner work.
- Next: researcher review; do not automatically enter Step 03.

## 2026-09-19 STEP 3.1 execution note

- Scope: freeze minimal Raw Trajectory -> Model-ready Sample -> Tensor Contract only.
- Completed: H=2/L=2 causal windows, input/target index separation, four actions, masks, communication service/progress split, minimal real artifact and tests.
- Gate: no formal dataset, graph/model/loss/planner/training/GPU/locked_test.
- Next: researcher review and explicit STEP 3.2 authorization.

## 2026-09-19 STEP 3.1R correction note

- Scope: correct the reviewed sample contract only; STEP 3.2 remains unapproved.
- Deliverable: History `[1,2]`, Action/Target `[2,3]`, fixed physical/task History rows, typed target indices, causal DAG capture, relation endpoints, and strict Action reference resolution.
- Acceptance: real non-locked Raw v2, model-ready checks, focused tests, round-trip, source/artifact hashes, index check, commit and push.

## 2026-08-29 P4 节点位置误差聚焦处理计划

当前门：P4 最终精度门仍未关闭。GPU 三 seed 训练、回收和结构审计已完成，但验证集节点位置 MAE 相对 persistence 在三个 seed 均变差（`+0.4621/+0.8483/+1.3708 m`）。

本次只处理这一个问题，停止条件如下：

1. CPU 只读复用已有三个 GPU checkpoint，按 1--20 步拆解 node-x/node-y/node-z 误差、位置变化幅度、节点类型和节点有效性。
2. 只根据现有证据确认一个主因；未确认前不改模型、数据、tensor contract、训练协议或 GPU 配置。
3. 主因确认后，只提出一个最小修正，并先做 CPU 单变量验证；若没有明确主因，则停止并报告证据，不盲目修正。
4. 只有 CPU 三 seed 重新通过既有 Go/No-Go，才允许再次 GPU 三 seed；P4 通过前不得进入 P6，不访问 `locked_test`。

当前唯一执行动作：已完成 CPU 逐步位置误差诊断；下一步只做一个经批准的最小位置位移修正 CPU 单变量验证。

## 2026-08-29 P4 位置误差诊断结果

- 报告：`code/artifacts/audit/pi_jwm_p4_position_diagnosis_20260829/position_diagnosis.json`。
- 三个现有 GPU seed 均在相同 validation sample IDs 上表现为预测位移明显小于真实位移；第 20 步平均 x 真实位移约 `22.145 m`，预测位移约 `0.774/2.027/3.816 m`（seed `20260830/20260831/20260832`）。
- 主因收敛为 residual state head 输出幅度过小，导致 rollout 对节点运动严重欠预测；不是单一 seed 或单一节点类型问题。
- 仍不修改冻结 tensor/training contract、正式训练协议或 GPU 配置；不访问 `locked_test`。
- 唯一下一步：先做一个最小、单变量的 residual 位移幅度修正 CPU 验证；未通过前不启动 GPU、不进入 P6。

## Goal

把老师原话和 PI-JWM 自有理论—实现—证据主线合并为一套不发散的 P0–P10 执行计划，复用已完成证据，不重复训练或审计；在 planner 机制、方法适配理由、场景化调参和多源数据边界闭合前，不扩大 GPU、不访问 locked-test。

## Current State

- P0/P1/P2 及确定性规则层 CPU replay 门已有可复用证据。
- 规则启用 aggregate candidate 的 non-locked GPU smoke/固定数据三 seed 训练已完成，但 `formal_performance_claim_ready=false`。
- P5（候选动作 world-model rollout planner）代码审计为 `blocked`；CPU 原型为 `prototype_only`，不能称正式 planner。
- P4 机制门已通过，但当前复用 checkpoint 的训练 horizon 为 3，而冻结 tensor contract 为 20；因此 P4 最终精度门仍未完成。
- `locked_test_accessed=false`，本任务不访问 locked-test。

## Phases

| Phase | Status | Next acceptance |
|---|---|---|
| P0 双约束治理与全仓证据地图 | complete | 原话、`记录/双约束门矩阵_20260826.json`、当前状态矩阵和审计边界已写入权威记录 |
| P1 最小可靠信息边/统一契约 | complete | 仅 verify-only，漂移即停 |
| P2 数据/tensor/规则递推 | complete | 复用现有 manifest、守恒、lifecycle/DAG、CPU replay |
| P3 方法理论适配与多源数据方案 | complete_for_design_package | 三模块理由、限制、指标、数据角色和调参协议已写入 P3 文档；真实源核验仍待执行 |
| P4 动作条件世界模型闭合 | complete_for_mechanism_only | tensor contract、规则递推、未来信息不泄漏和 1/5/20 接口机制门已通过；h20 checkpoint 的正式 state/task/resource/uncertainty 精度仍待测量 |
| P5 候选 rollout planner/反馈重规划 | blocked | 逐候选调用 world model，以未来后果选择并重规划 |
| P6 公平场景化调参与模块选择 | pending | 预注册统一预算、多 seed、消融和理论理由共同通过 |
| P7 多源微调增强与扩大训练 | pending | 源覆盖、适配、迁移证据完整后再批准 GPU |
| P8 全指标与方法冻结 | pending | 一致性审计和 formal claim gate 通过 |
| P9 locked-test 与公平 baseline | pending | 仅 P8 后一次性执行 |
| P10 论文、创新证据、最终复现 | pending | 每项结论绑定代码、artifact、统计和局限 |

## Immediate Next Action

P4 机制门已完成，当前仍停留在 P4：需要明确批准后，使用冻结 horizon=20 contract 重训并评估同一 rule-enabled candidate，才能关闭 P4 最终精度门。不得重复 P0–P2、规则 replay 或既有 non-locked aggregate GPU 证据；在 P4 最终门和 P5 通过前不得进入 P6、扩大训练或访问 locked-test。

## P4 Closure Record (2026-08-26)

| Sub-gate | Status | Evidence and boundary |
|---|---|---|
| P4-1 frozen tensor contract | passed | `code/artifacts/formal_tensor/pi_jwm_v4_tensor_v4_rule_contract_h20_unlocked_20260826`; history=8, horizon=20, 54 unlocked seeds, 14,742 windows, train-only normalization |
| P4-2 CPU mechanism audit | passed | `code/artifacts/audit/pi_jwm_formal_p4_horizon_mechanism_audit_20260826/p4_horizon_mechanism_audit.json`; 1/5/20 outputs finite and correctly shaped, action perturbation changes prediction, target leakage=0 |
| P4-3 unified mechanism gate | passed | `code/artifacts/audit/pi_jwm_p4_world_model_gate_h20_20260826/p4_world_model_gate.json`; rule replay, consistency and non-locked artifact boundaries pass |
| P4-4 formal accuracy gate | pending | Current checkpoints are horizon=3; h20 state/task/resource/uncertainty accuracy has not been measured. This requires an explicit h20 retraining decision and remains outside this CPU-only closure |

The P4 reports are mechanism/interface evidence only. `formal_performance_claim_ready=false`, `gpu_execution=false` for this gate, and `locked_test_accessed=false` remain unchanged. P5 is not opened.

## Errors Encountered

| Error | Attempt | Resolution |
|---|---:|---|
| 旧计划仍将历史阶段写在当前入口 | 1 | 新增本 v2 入口，明确历史记录只作追溯 |
| 全仓文本扫描在大目录上超时 | 1 | 保留已完成清单/分类统计，后续使用分目录、排除第三方与生成物的 bounded audit，不重复无界扫描 |
| PowerShell 复合 `rg` 搜索的引号被解析为命令 | 1 | 改用 `Select-String`/单项搜索做同一内容核验；无项目文件变化 |
| 本次交接首次汇总 `run_summary.json` 时假设 reload/device 位于顶层，实际输出为空 | 1 | 不解释空值；先读取真实 JSON 字段结构，再按实际路径核验 |
| 本次直接展开完整 metrics JSON 导致工具输出截断 | 1 | 改为结构化定向提取所需字段，不再展开大型指标对象 |
| 本次追加过程记录的补丁因跨段上下文不匹配失败 | 1 | 重新读取真实段落并拆成最小补丁；失败时文件未被部分修改 |
| 样本哈希/CSV 汇总命令在 PowerShell 中把 `foreach` 直接接管道，触发 `An empty pipe element is not allowed` | 1 | 先把 `foreach` 输出收集到 `$rows`，再单独格式化输出 |
| 本轮历史 artifact 哈希汇总再次把 `foreach` 直接接管道，触发同一空管道错误 | 2 | 停止该写法；后续 PowerShell 一律先赋值 `$rows=foreach(...)` 再输出 |
| 诊断启动命令把临时 junction 创建、运行和 `Remove-Item` 清理放在一起，被进程策略以 `blocked by policy` 拒绝 | 1 | 命令未执行、无输出；改在项目 `code/artifacts/tmp/` 建立命名明确且保留的只读适配目录，不执行删除 |
| 首次诊断摘要读取不存在的 `negative_score_quantiles.q99`，空值被 PowerShell 转成 0 | 1 | 不使用该值；按报告真实 schema 改读 `q90/q100`，另从全体分数读取 `score_quantiles.q99` |

## 2026-08-27 P4 节点类型契约修复（进行中）

- 当前门：P4。已确认 h20 的 `V/U/I/C` 短码在张量化时被误写为 `-1`，使完整双图消息未生效；不进入 P5/P6。
- 已完成：先加入短码与 padding 的失败测试，旧实现按预期失败；随后只在 `airfogsim_tensor_v2.py` 的节点类型入口加入四种短码映射。张量、正式构建、采集适配器和世界模型相关回归共 30 项通过。
- 下一步：以原 h20 非锁定源数据、`--unlocked-only --history 8 --horizon 20` 生成一个全新 tensor 目录，随后做类型契约和 CPU 微训练/reload 验收。禁止 GPU 与 `locked_test`。
- 命令记录：一次 `python -m unittest code.tests...` 因 Python 内置 `code` 名称冲突报错，已改用项目标准 `unittest discover`，不影响测试结论。

---

# Historical CPU State-Error Diagnosis and Revised GPU Execution

## Goal

The CPU state-error diagnosis and rule-layer closure are complete. The current execution phase is to verify the revised rule-enabled aggregate candidate on non-locked GPU data, with the fixed contract and no locked-test access. Formal performance claims remain closed.

## Phases

| Phase | Status | Deliverable |
|---|---|---|
| 1. Baseline and contract review | complete | Confirm current gate failure, loss path, and frozen data protocol |
| 2. TDD runner parameterization | complete | Added tested `state_mae_weight` override; default remains `0.05` |
| 3. CPU single-variable diagnosis | complete | `zero_init_residual_state_heads=true` and `residual_state_scale=0.5` pass all three fixed-data seeds |
| 4. Interpret and update authority records | complete | Numerical gate is true; hardware CUDA gate remains blocked; no GPU claim made |
| 5. Verification | complete | Model 8/8, runner 7/7, gate 2/2, compileall, diff check, local/remote manifests pass; remote RTX 4090 GPU smoke passes |

## Historical CPU-diagnosis boundaries

- Tensor root: `code/artifacts/formal_tensor/pi_jwm_v4_tensor_v3_rb_v1_unlocked_20260821`
- Splits: train/validation/calibration only; locked-test remains sealed.
- Default protocol remains 256/128/128 windows, hidden=32, 3 epochs, batch=2, lr=3e-4, CPU-only.
- Existing gate remains unchanged: at least 3 seeds, validation link-F1 regression <=0.05, calibration link-F1 delta >=0.05, validation node-x MAE ratio <=1.25.
- The original CPU-only restriction applied before the revised rule-layer gate; it is superseded by Phase 11 below. Any result remains non-locked evidence, not a formal performance claim.

## Errors Encountered

| Error | Attempt | Resolution |
|---|---:|---|
| Direct unittest file invocation resolved the repository `code` package incorrectly | 1 | Use project-standard unittest discovery instead |
| `future_source_node_index` was sourced from label-time `task_node_index` | 1 | Treat as future leakage; tensorize source directly from action records and reject fallback in rule-enabled model |
| Initial rule layer treated action occurrence as full task-stage service | 1 | Replace with learned per-flow delivery plus deterministic work-conserving CPU service, remaining-work caps and lifecycle legality |
| Full predicted state was fed back as an additive latent correction | 1 | Stop the in-progress second CPU seed; add a failing hook test and feed back only masked `rule_state - learned_state` |
| Strict flow-task mapping at the shared tensor-v2 layer broke teacher-v3 compatibility | 1 | Preserve `-1` plus mask in the shared layer; enforce completeness through the formal 54-seed rule-contract audit |
| First revised GPU three-seed batch used `data_seed=seed`, causing multi-seed contract drift | 1 | Preserve the batch as failed audit evidence; rerun all three seeds with the frozen common `data_seed=20260823` in a new isolated directory |

## Decision

- The next GPU candidate configuration is explicit and auditable: residual model, zero-initialized residual state mean heads, residual state scale `0.5`, default loss weights, hidden=32, batch=2, lr=3e-4.
- This configuration is not made the silent default. It is authorized only after the numerical gate report and when CUDA is available.
- Remote smoke passed with `gpu_execution=true`, train/validation/calibration `2/1/1`, and `locked_test_accessed=false`; formal long training remains a separate execution decision.
- First formal GPU seed `20260824` completed on the remote RTX 4090 with the same explicit configuration and full 256/128/128 protocol; multi-seed GPU training is the remaining execution phase.

## Controlled GPU tuning screening (2026-08-23)

| Phase | Status | Deliverable |
|---|---|---|
| 1. Remote readiness and code identity | complete | RTX 4090/CUDA check, no existing training process, tensor present, local/remote runner hash comparison |
| 2. Frozen-grid execution | complete | 4 modules x 2 learning rates x 3 seeds, fixed non-locked data/budget |
| 3. Artifact recovery and manifest audit | complete | Recovered 6 run directories, 31 files per run, manifest mismatch count 0 |
| 4. Protocol-based selection | complete | `coupled_dual_gnn_residual + lr=3e-4` passes screening gates; final method freeze remains pending |
| 5. Selected-candidate downstream CPU audit | complete | Horizon rollout and input-perturbation robustness diagnostics; state-rollout gap remains |
| 6. CPU root-cause diagnosis | complete | Key-wise non-locked CPU perturbation audit identifies node-state and task-state sensitivity channels |
| 7. Consistency decision before new training | complete | Audit confirms residual/task/action/aggregate boundaries; blocks on missing per-step deterministic rule update |
| 8. Rule-layer resolution | complete | Added explicit window contract, physical-unit deterministic rule layer, per-step model feedback, CPU regression evidence; old GPU checkpoints require retraining with the enabled layer |
| 9. Rule-layer contract hardening | complete | Explicit action-record sources, flow-task mapping, slot duration, vectorized conservation rules, and new 54-seed unlocked tensor independently audited |
| 10. Rule-layer retraining gate | complete | Three-seed 256/128/128 CPU gate and 10-check consistency audit pass; revised candidate is eligible for non-locked GPU execution |
| 11. Revised-candidate GPU execution | complete | Rule-enabled smoke and corrected fixed-data three-seed GPU protocol passed recovery, manifest, consistency, and non-locked boundary audits; formal claims remain closed |

## Boundaries

- Revised rule-layer tensor root is `code/artifacts/formal_tensor/pi_jwm_v4_tensor_v4_rule_contract_unlocked_20260823`; splits remain train/validation/calibration.
- `locked_test` is forbidden; aggregate baseline remains distinct from the per-RB diagnostic sidecar.
- Do not expand the grid after observing results; do not select by a single seed.
- Key-wise diagnosis is complete in `code/artifacts/audit/pi_jwm_formal_tuning_keywise_robustness_cpu_20260823/`; next action is a theory-code consistency decision, not another GPU run.
- Rule-layer interface evidence is in `code/artifacts/audit/pi_jwm_formal_candidate_consistency_audit_20260823_rule_layer_interface/`; all four input-contract requirements are ready. The audit remains blocked only because the three selected 2026-08-23 checkpoints were trained before the rule layer and must be retrained.
- Superseding evidence: `pi_jwm_formal_rule_tensor_contract_audit_20260823`, `pi_jwm_formal_rule_layer_cpu_gpu_gate_v2_20260823`, and `pi_jwm_formal_candidate_consistency_audit_rule_layer_v2_20260823` all pass. The old pre-rule checkpoints remain historical evidence only.

## 2026-08-26 Rule-Rollout Replay Closure

| Phase | Status | Deliverable |
|---|---|---|
| 1. v4 source-flow conservation recomputation | complete | 54 unlocked trajectories; 155,875 active flow slots and 12,924 first-active slots; zero violations at `1e-5` |
| 2. Three-checkpoint CPU rule-rollout replay | complete | 3 x 128 validation windows; every seed captured 192/192 rule steps with no rule invariant violations |
| 3. Record the closure and verify the affected code | complete | Project records, focused tests, compilation, and whitespace check |

The replay uses `pi_jwm_v4_tensor_v4_rule_contract_unlocked_20260823`, the same v4 tensor root recorded by all three rule-enabled GPU checkpoints. It is CPU-only, non-locked, and aggregate-baseline only. The report is `code/artifacts/audit/pi_jwm_formal_rule_v2_cpu_rule_rollout_multiseed_20260826_created_flow_fixed/rule_rollout_multiseed_audit.json` (SHA-256 `ED52136A8711A25C5BE9E644EADBDBB59CAC83FA61063E16EDF2D873EBBD121B`). `formal_performance_claim_ready=false` remains unchanged.

## 2026-08-26 Candidate-Action Planner Consistency Audit

| Phase | Status | Deliverable |
|---|---|---|
| 1. Historical/current code evidence review | complete | R6 is direct belief/state-conditioned candidate scoring; formal model is action-conditioned prediction only |
| 2. Machine-checkable audit contract and focused tests | complete | Three focused tests enforce all candidate-rollout mechanism requirements |
| 3. Run non-locked code/artifact audit | complete | Report is `blocked`; no planner implementation or performance claim |
| 4. Record the boundary and verification | complete | Authority records updated; no GPU, training, or locked-test access |

The target is an independent mechanism audit, not a new planner implementation. An absent or incomplete mechanism blocks method freezing, locked-test evaluation, and any planner claim.

### Result

- Report: `code/artifacts/audit/pi_jwm_formal_candidate_rollout_planner_audit_20260826/candidate_rollout_planner_audit.json`, SHA-256 `7A71C78F78058F8F0C19EC0509816A2A31D972B27DA8CF1D8FABC8F7CC3B6637`.
- The audit is `blocked`: candidate generation, common-belief reuse, candidate-wise world-model calls/action injection, future state/task/cost/risk extraction, predicted-objective selection, and feedback/replanning are all absent from the audited mechanism.
- Existing evidence remains non-locked aggregate-baseline evidence. Planner GPU experiments, planner method freeze, locked-test access, and formal performance claims remain closed.

## 2026-08-26 Literature close reading gate

- Seven newly indexed arXiv preprints have been read and recorded in `literature/新增7篇精读笔记_20260826.jsonl` and `literature/新增7篇精读汇总_20260826.md`.
- Literature conclusions are inspiration/evaluation guidance only. They do not alter the frozen tensor contract or promote the current direct scorer to a candidate-rollout planner.
- Next work remains the non-locked planner mechanism gate, followed by CPU and GPU execution only if the theory-code-data-metric audit passes.

## 2026-08-26 File-tree and evidence-layer governance

| Item | Status | Deliverable |
|---|---|---|
| Read-order and scope rules | complete | `AGENTS.md`, root `README.md`, and `记录/文件树与证据分层_20260826.md` agree on the pre-task read order |
| Process/final distinction | complete | `记录/README.md` and `code/artifacts/README.md` define authority, process, historical, and candidate evidence boundaries |
| Current-session record | complete | This entry plus matching entries in `progress.md` and `findings.md` |

This was a non-destructive indexing pass. No files were moved, renamed, deleted, or promoted; no GPU, locked-test, or third-party AirFogSim path was accessed. The single next research action remains the P3 method-suitability and multi-source data-role package.

## 2026-08-27 P4 h20 precision-audit correction

| Item | Status | Evidence |
|---|---|---|
| h20 three-seed precision audit | complete, blocked | Seeds `20260824/20260825/20260826`, 128 validation windows each; manifest-verified h20 metrics are in `code/artifacts/audit/pi_jwm_p4_h20_precision_audit_20260826/` |
| Final P4 precision gate | blocked | At k=20, learned node-x MAE mean is `22.8591 m`, versus persistence `22.1450 m`; all three seeds are worse and node-x 95% coverage is `75.43%` |
| Next P4 deliverable | pending | CPU-only, one-variable diagnosis of the long-horizon node-x error; do not change the model, tensor contract, training protocol, GPU script, or access `locked_test` before a specific cause is verified |

The prior wording that h20 precision was pending is superseded. The h20 training and audit are complete, but `formal_performance_claim_ready=false`; P4 remains open and no later roadmap gate is entered.

## 2026-08-27 P4 node-type contract diagnosis

| Item | Status | Evidence and boundary |
|---|---|---|
| Root cause | confirmed | AirFogSim source data uses node-type short codes `V/U/I/C`; the collector preserves them, but tensorization only accepts full names. All 54 h20 unlocked trajectories therefore have every `node_kind_index=-1`. |
| Model impact | confirmed | The coupled model uses `node_kind_index >= 0` as the physical-node validity mask. This disables physical-graph messages, information-graph messages through attached agents, and their node-agent coupling; task-to-node and flow-edge paths are not equivalent substitutes. |
| P4 result boundary | blocked | The three h20 GPU runs remain reproducible non-locked aggregate results, but they are not valid evidence for the complete dual-graph method. `formal_performance_claim_ready=false` and `locked_test_accessed=false` remain unchanged. |
| Single next action | pending | Run CPU tensor/window acceptance on the repaired unlocked h20 tensor; GPU and `locked_test` remain closed. |

## 2026-08-27 P4 RB outcome reconstruction repair

| Item | Status | Evidence |
|---|---|---|
| Runtime RB reconstruction | complete | `build_runtime_rb_outcome_observations` joins RB actions and runtime transfer events by time/task/source/target; action supplies RB IDs, event supplies path and planned capacity |
| Missing runtime result boundary | complete | An action without an event remains an unobserved masked label; an event without an action or conflicting RB IDs is rejected |
| Unlocked h20 rebuild | complete | `code/artifacts/formal_tensor/pi_jwm_v4_tensor_v4_rule_contract_h20_rb_repair_unlocked_20260827`; 54 trajectories, 14,742 windows, `n_rb=50`, 48,447 observed RB labels, no invalid present-node types, no locked-test directory |
| Single next action | pending | Run CPU tensor/window acceptance and checkpoint reload against the repaired tensor; no GPU or `locked_test` |

This diagnosis is CPU-only and read-only. It does not change the model, tensor contract definition, training protocol, GPU configuration, or locked-test boundary.
## 2026-08-27 P4 repaired h20 CPU acceptance

- Current gate: P4 repaired h20 tensor acceptance.
- Completed: repaired unlocked tensor passed validation; focused tests passed RB 11/11, tensor build 8/8, world model 8/8, CPU smoke 2/2; existing CPU acceptance artifact contains train/validation/calibration 8/4/4 and horizon=20 checkpoint outputs.
- Full-suite boundary: 1457 tests were attempted; 1 repository-root expectation failure and 13 legacy AirFogSim/Windows fixture errors occurred outside this P4 path. They are recorded as environment/legacy test boundaries, not treated as P4 passes.
- Blocker: the repaired tensor has not passed the independent CPU baseline-vs-persistence Go/No-Go gate; `formal_performance_claim_ready=false` and GPU remains closed.
- Single next action: run the independent non-locked Go/No-Go audit on the repaired CPU evidence, without GPU or `locked_test`.

## 2026-08-27 P4 repaired h20 CPU Go/No-Go result

- Three independent CPU seeds are now available under the same repaired tensor and fixed configuration.
- Go/No-Go report: `code/artifacts/audit/pi_jwm_p4_h20_repair_cpu_go_no_go_20260827/cpu_to_gpu_gate.json`.
- Result: `gpu_allowed=false`. Position error ratio and operational metrics pass, but two seeds have validation link-F1 regression and the mean validation link-F1 is below persistence.
- Boundary: this is a CPU evidence gate only; no GPU was started, `locked_test` was not accessed, and `formal_performance_claim_ready=false` remains.
- Single next action: CPU-only diagnose the communication-activity F1 instability on the repaired contract; do not repeat the same GPU run or change the frozen contract before a cause is verified.

## 2026-08-27 P4 communication-activity F1 diagnosis

- Current gate: P4 remains blocked; this diagnosis was CPU-only and used only the three existing non-locked CPU checkpoints.
- Completed evidence: `code/artifacts/audit/pi_jwm_p4_link_activity_diagnosis_20260827/link_activity_diagnosis.json` records the same 4 validation and 4 calibration windows per seed, with validation positives=270/25,502 negatives and calibration positives=86/27,564 negatives.
- Result: seeds `20260827` and `20260829` retain strong validation ranking (ROC-AUC `0.9612` and `0.9580`), while seed `20260828` is inverted (ROC-AUC `0.3560`, average precision `0.0107`). Its calibration ROC-AUC is only `0.0586`, so the failure is not caused by threshold selection alone. The calibrated thresholds (`0.9`, `0.1`, `0.7`) are a consequence of the unstable score distributions.
- Interpretation: the 8-window/one-epoch CPU check is too small and too imbalanced (training link activity `669` positives vs `62,755` negatives) to establish seed-stable communication-activity ranking. This is a confirmed evidence limitation, not proof that the intended method is theoretically invalid.
- Execution boundary: `gpu_allowed=false`, `formal_performance_claim_ready=false`, `locked_test_accessed=false`; no model, tensor contract, or training protocol was changed.
- Single next action: run one expanded CPU-only stability check using the already frozen tensor and protocol definitions but a pre-registered larger CPU sample budget, then repeat the independent gate. Do not start GPU or access `locked_test` before that gate passes.

## 2026-08-27 P4 expanded CPU stability check

- Completed the pre-registered expanded CPU check with the frozen repaired h20 tensor and current candidate: three seeds (`20260830/20260831/20260832`), each using `256/128/128` train/validation/calibration windows, hidden=32, 3 epochs, batch=2, learning rate `3e-4`, deterministic rule layer, zero-init residual heads, and residual scale `0.5`.
- Independent Go/No-Go report: `code/artifacts/audit/pi_jwm_p4_h20_repair_expanded_cpu_go_no_go_20260827/cpu_to_gpu_gate.json`.
- Result remains `gpu_allowed=false`: validation link-F1 deltas `-0.0411/-0.1753/+0.0104`, mean `-0.0686`; validation node-x ratio `1.0849`; mean RB-occupancy error delta `+0.1132`; calibration link-F1 mean delta `+0.2404`.
- Interpretation: increasing the CPU sample budget removed the original single-seed inversion as the sole explanation, but the learned candidate still fails the pre-registered validation non-inferiority and RB non-regression gates. This is evidence against GPU authorization, not a theory-validity or final-method failure claim.
- Boundary: no GPU was started, `locked_test` remains sealed, and the frozen tensor/model/training contract was not changed.
- Single next action: pause P4 for human review of the failed validation/RB gates; do not start GPU, enter P6, or change the contract in this turn.

## 2026-08-27 P4 failure diagnosis: link threshold and RB metric boundary

- Added and ran the CPU-only read-only diagnostic `code/scripts/run_formal_p4_failure_diagnosis_v1.py` on the three expanded non-locked runs; report: `code/artifacts/audit/pi_jwm_p4_failure_diagnosis_20260827/p4_failure_diagnosis.json`.
- Communication activity: all three validation ROC-AUC values are about `0.991`; fixed validation F1 is substantially better at seed-stable thresholds (`0.5` or `0.7`) than some calibration-selected thresholds. This is a calibration/threshold-transfer problem under the current split imbalance, not evidence that the score ranking is broken.
- RB: action-derived occupancy matches the observed target almost exactly per edge (`99.41%`--`99.61%` exact fraction) and has total MAE `0.5949` (validation) / `0.7473` (calibration), better than persistence. The learned output is identical to the action-derived rule output.
- Critical metric boundary: `formal_world_model_metrics_v1.py` compares normalized model RB outputs and raw aggregate RB targets after multiplying both by `scale`; it does not inverse-normalize the prediction with `+ mean` and does not keep the target in the same physical unit. The reported `3.2023` RB MAE is therefore not comparable to the action-derived result and must not be used as a model failure claim until a separately approved metric fix is implemented and tested.
- Boundary remains `gpu_allowed=false`, `formal_performance_claim_ready=false`, `locked_test_accessed=false`; no GPU, locked-test, tensor-contract, model, or training-protocol change occurred.
- Single next action: review and approve one narrow follow-up consisting of (a) correcting RB metric de-normalization with a focused regression test and (b) a pre-registered threshold-transfer check; do not start GPU or enter P6 before those checks and an independent Go/No-Go rerun.

## 2026-08-28 P4 metric-repair re-evaluation

- Completed the approved CPU-only metric repair verification with `code/scripts/reevaluate_formal_cpu_runs_v1.py` and focused tests.
- Re-evaluated seeds `20260830/20260831/20260832` using the exact existing checkpoints/sample IDs and calibration-only threshold selection; historical runs were not overwritten.
- Independent report: `code/artifacts/audit/pi_jwm_p4_metric_repair_cpu_go_no_go_20260828/cpu_to_gpu_gate.json`.
- RB failure is removed: validation RB occupancy delta is `-0.6477 RB`. GPU remains blocked by validation link-F1 mean delta `-0.0686` and throughput MAE delta `+0.7101 Mbps`.
- Boundary: CPU only, no retraining, no GPU, no `locked_test`; `formal_performance_claim_ready=false` remains.
- Single next action: keep P4 blocked and obtain a focused decision on the remaining validation link-F1/throughput gates.

## 2026-08-28 P4 threshold protocol audit

- Completed a CPU-only, non-locked, read-only audit over the existing expanded runs `20260830/20260831/20260832`; no retraining, GPU, tensor-contract, or `locked_test` access occurred.
- New report: `code/artifacts/audit/pi_jwm_p4_threshold_protocol_audit_20260828/threshold_protocol_audit.json`; focused tests passed 2/2.
- Compared calibration-selected raw thresholds, fixed raw threshold `0.5`, and ordinary-probability correction for `pos_weight=50` (`p=s/(50-49s)`). Validation was evaluation-only and never used to choose thresholds.
- Fixed raw `0.5` is more stable than the historical calibration-selected threshold in some seeds, but validation F1 remains `0.6305/0.4924/0.6005`; one seed is still below persistence and the mean does not establish the pre-registered gate. Corrected ordinary-probability `0.5` is unsuitable because its equivalent raw threshold is `50/51` and recall collapses.
- Interpretation: threshold transfer contributes to the F1 gap but does not fully explain it. The throughput MAE regression (`+0.7101 Mbps`) remains unresolved. `gpu_allowed=false`, `formal_performance_claim_ready=false`, and `locked_test_accessed=false` remain unchanged.
- Single next action: keep P4 blocked and obtain a focused decision on the remaining link-F1/throughput failures; do not start GPU or enter P6.
## 2026-08-28 P4 throughput diagnosis

- Completed the CPU-only, read-only stepwise throughput diagnosis using the three existing expanded h20 non-locked checkpoints (`20260830/20260831/20260832`).
- Report: `code/artifacts/audit/pi_jwm_p4_throughput_diagnosis_20260828/throughput_diagnosis.json`.
- Evidence: validation learned-minus-persistence total-MAE deltas are `+0.0356`, `-0.1543`, and `+2.2492 Mbps`; seeds `20260830/20260831` become increasingly low-biased at later steps, while `20260832` is high-biased from step 1. This is a stepwise rollout calibration/seed-stability problem, not a single threshold or RB-unit issue.
- Boundary: `gpu_allowed=false`, `formal_performance_claim_ready=false`, `locked_test_accessed=false`; no training, model, tensor-contract, or protocol change.
- Single next action: human decision on one focused CPU-only remedy for the remaining link-F1/throughput gates; do not start GPU, enter P6, or access `locked_test`.

## 2026-08-28 P4 focused epoch-stability check

- Hypothesis: the remaining link-F1/throughput instability is partly under-training, because all three existing 3-epoch runs still had decreasing validation loss at epoch 3 and showed long-horizon rate bias.
- Single variable: increase CPU training epochs from `3` to `8`; keep the repaired h20 tensor, fixed `data_seed=20260823`, train/evaluation limits `256/128/128`, hidden=32, batch=2, `lr=3e-4`, zero-init residual heads, residual scale `0.5`, deterministic rule layer, calibration-only threshold selection, and all gate criteria unchanged.
- New outputs must be isolated from the historical 3-epoch runs. This is a CPU-only diagnostic, not GPU authorization and not a protocol rewrite.
- Acceptance: all three seeds complete, checkpoint reload is verified, then the existing independent Go/No-Go gate is rerun. `locked_test` remains sealed.

## 2026-08-28 P4 epoch-stability execution correction

- Seed `20260830` completed the intended single learned method; checkpoint reload was verified and the run remains non-locked CPU evidence only.
- A direct-function launch for seed `20260831` initially omitted `learned_methods`, so the function default began training multiple models and wrote an unintended `pooled_gru` checkpoint. The process was stopped before completion; this directory is failed process evidence only and must not enter the gate.
- Root cause: the command-line entrypoint enforces CUDA even for this module, while the direct function has a broader default method list. The corrected launch must pass `learned_methods=("coupled_dual_gnn_residual",)` explicitly.
- Single next action: rerun seed `20260831` in a fresh isolated output directory with the explicit one-method argument, then inspect its reload and metrics before starting seed `20260832`.

## 2026-08-28 P4 epoch-stability CPU gate result

- The corrected 8-epoch CPU runs for seeds `20260830`, `20260831`, and `20260832` completed with the single intended candidate and verified checkpoint reloads.
- Independent report: `code/artifacts/audit/pi_jwm_p4_h20_repair_epoch8_cpu_go_no_go_20260828/cpu_to_gpu_gate.json`.
- The pre-registered gate passed: `gpu_allowed=true`; mean validation link-F1 delta=`+0.0237`, node-x MAE ratio=`1.0463`, calibration link-F1 delta mean=`+0.2856`, and validation throughput/RB/task-delay MAE deltas were all negative versus persistence.
- Boundary: this only authorizes the next formal non-locked GPU run. It does not open `locked_test` or make a final performance claim; `formal_performance_claim_ready=false` and `locked_test_accessed=false` remain.
- Single next action: obtain confirmation before starting the formal GPU training; do not enter P6 or access `locked_test`.

## 2026-08-28 P4 formal GPU launch attempt

- User authorized the focused next action after the CPU Go/No-Go pass.
- Read-only SSH checks to the two recorded SeetaCloud endpoints (`14826` and `14507`) both returned `Connection refused`; no GPU process was started.
- Boundary remains `gpu_allowed=true` by CPU evidence, but execution is blocked by server reachability. `locked_test_accessed=false` and `formal_performance_claim_ready=false` remain.
- Single next action: retry the same read-only connectivity check only after the server is reachable, then launch the one approved non-locked GPU configuration.

## 2026-08-28 P4 reachability recheck

- Rechecked the two recorded SSH ports after the user requested continued execution; both remain unreachable (`14826=False`, `14507=False`).
- Local gate verification still reports `gpu_allowed=true`, three independent seeds, no failed criteria, `formal_performance_claim_ready=false`, and `locked_test_accessed=false`.
- Single next action remains unchanged: once the server accepts SSH, launch only the approved non-locked GPU configuration.

## 2026-08-28 P4 GPU launch on reachable server

- Server `connect.nmb1.seetacloud.com:14826` is reachable; RTX 4090 and CUDA are available, with no PI-JWM training process before launch.
- Uploaded only the fixed h20 repaired non-locked tensor and current PI-JWM code to the new remote directory `/root/autodl-tmp/pi_jwm_p4_h20_epoch8_gpu_20260828`; local/remote tensor manifest SHA-256 matches.
- The first seed launch exited before training because the remote process lacked `PYTHONPATH`; GPU utilization stayed at 0 and no run summary was written.
- Root cause is launch-environment propagation, not model/data/protocol behavior. The corrected launch must export `PYTHONPATH` remotely and pass `--learned-methods coupled_dual_gnn_residual` explicitly.
- Boundary remains non-locked only; `formal_performance_claim_ready=false`, `locked_test_accessed=false`.
- Single next action: restart seed `20260830` with the corrected remote environment, then verify its summary before seed `20260831`.

## 2026-08-28 P4 second GPU seed launch correction

- `20260830` completed with all required non-locked GPU markers: training complete, CUDA execution, checkpoint reload verified, and `locked_test_accessed=false`.
- The first background launch of `20260831` exited before training with `ModuleNotFoundError: No module named 'pi_jwm'`; no summary or valid result was produced and GPU remained idle.
- Root cause was confirmed by a minimal remote import check: the module is unavailable without `PYTHONPATH` and resolves when `/root/autodl-tmp/pi_jwm_p4_h20_epoch8_gpu_20260828/code/src` is explicitly set.
- Single next action: restart `20260831` with the explicit remote `PYTHONPATH`, then verify its summary before starting `20260832`.

## 2026-08-28 P4 second GPU seed completed

- Corrected seed `20260831` completed on CUDA with `training_run_complete=true`, `gpu_execution=true`, and `checkpoint_reload_verified=true`.
- Its summary records `locked_test_accessed=false`, train/validation/calibration counts `256/128/128`, and the fixed h20 contract.
- The earlier missing-`PYTHONPATH` attempt remains invalid process evidence only; this completed run uses the explicit environment and the single approved learned method.
- Single next action: start seed `20260832` with the same explicit environment and frozen configuration, then verify its summary.

## 2026-08-29 P4 third GPU seed completed

- Corrected seed `20260832` completed on CUDA with `training_run_complete=true`, `gpu_execution=true`, and `checkpoint_reload_verified=true`.
- Its summary records `locked_test_accessed=false`, train/validation/calibration counts `256/128/128`, and the fixed h20 contract.
- All three approved seeds now have complete non-locked GPU run summaries; formal performance remains closed pending local recovery and the independent multi-seed audit.
- Single next action: recover the three run directories, verify manifests and summaries locally, then run the prescribed independent GPU multi-seed audit.

## 2026-08-29 P4 GPU three-seed recovery and audit

- Recovered the three completed non-locked GPU runs from the RTX 4090 server. Archive SHA-256 is `373CBD41BA054512A5969B50F0F68845FC59130B69317DE06B02C50C2FBC0437`.
- Independent audit `code/artifacts/audit/pi_jwm_p4_h20_epoch8_gpu_multiseed_audit_20260829/gpu_multiseed_audit.json` passed its structural contract: three seeds, identical frozen configuration, zero manifest mismatches, calibration-only threshold selection, GPU execution true, and no `locked_test` access.
- Performance boundary remains open: validation node-x MAE deltas are `+0.4621/+0.8483/+1.3708 m` (all worse than persistence), while communication and operational deltas improve. The audit therefore does not authorize a formal performance claim or a roadmap phase change.
- Focused audit tests passed `2/2`.
- Single next action: keep P4 open and make no new training or tuning run until a separately approved, evidence-based handling of the remaining node-state error is defined; do not access `locked_test`.

## 2026-08-29 P4 residual 幅度三 seed CPU Go/No-Go

- 三个 non-locked CPU seed (`20260830/20260831/20260832`) 均以 `residual_state_scale=1.0`、8 epochs 和原冻结配置完成，checkpoint reload 均通过。
- 独立报告：`code/artifacts/audit/pi_jwm_p4_h20_repair_scale1_cpu_go_no_go_20260829/cpu_to_gpu_gate.json`；`failed_gates=[]`、`gpu_allowed=true`。
- 观察值：validation link-F1 delta 均值 `+0.0340`，node-x MAE ratio `1.0851`，calibration link-F1 delta 均值 `+0.3157`；throughput/RB/task-delay MAE 均值相对 persistence 改善。
- 该报告只开放同一配置的 non-locked GPU 执行，不开放 `locked_test`，也不代表最终性能声明；P4 仍需 GPU 结果和独立审计闭合。
- 唯一下一步：服务器可用后，按 `scale=1.0`、单一 `coupled_dual_gnn_residual`、256/128/128、8 epochs 的同一配置启动 non-locked GPU 三 seed；不扩大网格、不进入 P6。

## 2026-08-29 P4 residual 幅度单变量 CPU 验证

- 新建隔离 run `code/artifacts/experiments/pi_jwm_formal_p4_cpu_h20_repair_scale1_seed20260830`，只将 `residual_state_scale` 从 `0.5` 改为 `1.0`；数据、张量、seed、窗口数量、hidden、batch、学习率、epoch、规则层和 loss 全部保持不变。
- 训练完成并验证 checkpoint reload；仅使用 train/validation/calibration，`gpu_execution=false`、`locked_test_accessed=false`。
- 20 步节点位置诊断显示 seed `20260830` 的 node-x MAE 为 `22.9194 m`，persistence 为 `22.1450 m`，比例约 `1.035`，优于原 scale=0.5 的约 `1.077`，并满足原位置比例门 `<=1.25`；但仍不是全面优于 persistence。
- 结论：该单变量方向值得按预注册流程做三 seed CPU 复核，不改变冻结 tensor contract 或正式训练协议。
- 唯一下一步：以 `residual_state_scale=1.0`、其余配置完全相同，依次完成 seed `20260831`、`20260832`，再运行既有独立 Go/No-Go；未通过前不启动 GPU、不进入 P6、不访问 `locked_test`。

## 2026-08-29 P4 residual 幅度 GPU 三 seed 回收与独立审计

- 已完成并回收 `residual_state_scale=1.0` 的 non-locked GPU seeds `20260830/20260831/20260832`；三者均为 `training_run_complete=true`、`gpu_execution=true`、checkpoint reload verified、`locked_test_accessed=false`。
- 独立审计 `code/artifacts/audit/pi_jwm_p4_h20_scale1_gpu_multiseed_audit_20260829/gpu_multiseed_audit.json` 通过结构门：三 seed 合同一致、manifest mismatch=`0`、阈值只来自 calibration。
- 性能门仍未通过：validation link-F1 delta 均值 `+0.0059`（其中一个 seed 为 `-0.0337`），node-x MAE delta 均值 `+1.4841 m`；吞吐量、RB 占用、任务时延平均改善。
- 结论：P4 继续 blocked，`formal_performance_claim_ready=false`，不进入 P6、不访问 `locked_test`，不再重复同一 GPU 配置。
- 唯一下一步：依据该审计结果回到既有记录，形成一个有证据的节点状态误差处理决策；在决策前停止新增训练和扩大调参。

## 2026-08-29 P4 预测状态接力修复

- 只修改 `code/src/pi_jwm/formal_dual_graph_world_model_v1.py` 的 residual 状态基准：后续步使用上一轮最终预测状态，DAG 状态同步递推；数据、tensor contract、loss、训练协议均未改。
- 先加入两步递推失败测试，旧实现按预期失败；修复后 formal world-model 定向测试 `9/9`、compileall 和 `git diff --check` 通过。
- 新建隔离 CPU 运行目录 `code/artifacts/experiments/pi_jwm_p4_recursive_cpu_20260829/`。seed `20260830` 已完成 h20、256/128/128、8 epochs、hidden=32、batch=2、scale=1.0、确定性规则层训练和 reload；`gpu_execution=false`、`locked_test_accessed=false`。
- seed `20260831` 已按同一配置启动，但本轮因单 seed 长时间 CPU 运行未完成而停止，仅保留配置和样本清单，不作为性能证据；seed `20260832` 尚未启动。
- 当前结论：代码级状态接力修复已验证；三 seed CPU Go/No-Go 尚未完成，P4 仍 blocked，GPU 不开放，不进入 P6，不访问 `locked_test`。
- 唯一下一步：在独立运行窗口按同一配置完成 seed `20260831`、`20260832`，再运行既有 Go/No-Go；不叠加第二个修复。

## 2026-08-29 P4 预测状态接力修复三 seed CPU Go/No-Go

- 三个 non-locked CPU seed (`20260830/20260831/20260832`) 均已完成 h20、256/128/128、8 epochs、hidden=32、batch=2、`residual_state_scale=1.0`、确定性规则层训练与 checkpoint reload。
- 独立报告：`code/artifacts/audit/pi_jwm_p4_recursive_cpu_go_no_go_20260829/cpu_to_gpu_gate.json`。
- 结构和多数数值门通过：独立 seed 数量=3，node-x MAE ratio=`1.1869`，calibration link-F1 delta 均值=`0.3676`，throughput/RB/task-delay MAE 均值改善，阈值只来自 calibration。
- 唯一失败门：逐 seed validation link-F1 delta 为 `+0.1032/-0.0941/+0.2013`，其中一个 seed 低于允许的 `-0.05`；因此 `gpu_allowed=false`。
- 结论：状态接力修复的 CPU 三 seed 证据已闭合，但 P4 性能门仍 blocked；不启动 GPU、不进入 P6、不访问 `locked_test`，`formal_performance_claim_ready=false`。
- 唯一下一步：依据该单一失败门和既有记录，形成 link-F1 seed 不稳定性的证据化处理决策；不叠加第二个修复、不重复同一训练。

## 2026-08-29 P4 长期停滞根因分析

- 科研层：h20 长步 rollout 的 node-x 误差仍高于 persistence；scale=1.0 只改善了 CPU 放行比例，GPU 三 seed 的 node-x MAE 平均仍增加 `1.4841 m`，link-F1 也有 seed 回退。
- 数据/实现层：节点类型契约和规则层曾在 GPU 之后才发现/闭合，旧 GPU 数值只能降级为部分通路历史证据，导致 tensor 重建和重训。
- 指标/门层：RB occupancy 曾有单位/反归一化错误；CPU 放行门与最终 GPU 性能门不是同一个门，导致“可以上 GPU”被误读为“性能已通过”。
- 实验层：3->8 epoch、scale 0.5->1.0 等修正是在失败后逐个追加，首次失败时没有先分类根因并设置停机规则。
- 工程层：远端脚本路径、`PYTHONPATH`、默认 learned-method 列表多次造成无效运行，浪费等待和 GPU 费用。
- 治理层：追加式日志保留了大量旧的 pending/next-action 状态，缺少一个自动阻断 critical mismatch 的单一当前状态入口。
- 结论：长期卡住是“前置门不够硬 + 失败后继续试小修正 + 旧状态混杂 + 模型长步能力确实不足”的叠加，不是单个 bug。下一步应先做只读的当前 gate 清理和路线决策，不能再盲目重训。

## 2026-08-30 `_CONTEXT.md` 项目交接审阅（完成）

- 本次任务仅接收并核验 `PROJECT/_CONTEXT.md`，不修改代码、方法定义、实验配置或研究路线。
- 当前门暂按现有过程记录识别为 P4 blocked；GPU 与 `locked_test` 均不在本次范围内。
- 当前阻塞暂记为 P4 长步 rollout 的节点位置误差与 validation link-F1 跨 seed 不稳定；最终结论须由交接文件、权威记录、Git 状态和机器审计交叉确认。
- 唯一下一动作：完整读取交接文件及仓库规定的五份权威入口，核对当前 gate、已完成证据、冲突、未验证项和精确续接点。
- 路径核验：用户给出的 `PROJECT\_CONTEXT.md` 与当前文件树不一致；`PROJECT` 目录不存在，唯一匹配文件为根目录未跟踪的 `PROJECT_CONTEXT.md`，本次据此继续。
- 工作区边界：当前 `main@0630515` 存在大量用户已有修改和未跟踪文件；本次不回退、不整理、不提交这些内容。
- 交接文件给出的精确续接点：保持 P4，不再训练；先只读或 CPU-only 分析固定配置下 seed `20260831` 的 validation link-F1 回退，形成证据化判断。
- 交接文件同时记录长期位置证据缺口：最新 recursive CPU gate 只有合并预测步的 node-x MAE ratio，没有新的第 1/5/10/20 步分解，不能据此宣称原长期位置门通过。
- 三份运行配置已按真实 JSON 字段核验：固定 data seed/manifest、CPU、history=8、horizon=20、8 epochs、256/128/128，训练完成且 checkpoint reload verified；三份均未使用 GPU、未访问 `locked_test`、未开放正式性能声明。
- 代码核验：gate 源码以最差 seed 的 validation link-F1 delta 与 `-0.05` 比较；模型源码将每步最终 node/task/flow/edge 状态和 DAG 状态写回下一步 residual 基准，和交接描述一致。
- 完成证据：交接文件 532 行已完整通读；权威记录当前覆盖段、Git 状态、最新 gate JSON/哈希、三份 config/run summary/runtime/threshold selection/comparison 和相关源码均已交叉核验。
- fresh 验证：世界模型定向测试 `9/9`、CPU Go/No-Go gate 测试 `2/2`、`compileall`、`git diff --check` 均通过。
- 最终当前门：P4 blocked。剩余阻塞为 validation link-F1 单 seed 回退，以及原长期位置计划要求的第 1/5/10/20 步新证据尚未补齐；GPU 不开放，`locked_test` 不访问，不进入 P6。
- 单一下一动作：只读或 CPU-only 地分析相同 validation 样本下 seed `20260831` 的 link-F1 回退，形成证据化判断；不直接实施修复或新实验。
- 机器 gate 已核验：SHA-256 为 `E3CEC38987742A3D7527612FDB7CEDA0EE6B2B47ED262AE3A811902AB6B824DD`，`gpu_allowed=false`，唯一 `failed_gates` 项为 `maximum_per_seed_validation_link_f1_regression`，`gpu_started=false`、`locked_test_accessed=false`。

## 2026-08-30 P4 link-F1 证据复用与零重复实验计划（进行中）

### 固定目标与边界

- 当前门：P4 blocked；当前阻塞是 seed `20260831` 的 validation link-F1 回退及长期位置逐步证据缺口。
- 本轮交付物：一份基于现有 config、metrics、comparison、threshold、audit 和历史诊断的证据化处理判断。
- 禁止项：不重训、不重新运行已经完成的实验、不扩大参数搜索、不改模型/loss/tensor/阈值协议、不启动 GPU、不访问 `locked_test`、不进入 P6/planner。
- 保全项：已有通过的递推机制、数据合同、checkpoint reload、运营指标改善和历史失败证据全部保留；任何后续方案必须显式说明是否会使这些已验证效果回退。

### 执行阶段

| 阶段 | 状态 | 停止条件 |
| --- | --- | --- |
| A. 历史证据去重与可复用清单 | completed | 列清已完成的 link activity、threshold、failure diagnosis、recursive 三 seed 产物及其适用边界，不产生新实验 |
| B. 同样本三 seed 差异定位 | completed | 只比较已有分数/阈值/混淆统计/逐步指标，区分排序、校准迁移或训练随机性证据 |
| C. 理论--实现--指标一致性判断 | completed | 判断当前失败是历史已解释、已解决后复发，还是新的未闭合问题 |
| D. 单一处理决策 | completed | 只输出一个判断；若证据不足，明确缺失字段并停止，不用新训练补猜测 |

### 成本路由

- Luna：只读提取路径、字段、哈希、状态和已完成实验去重表，不作研究结论。
- Terra：在冻结口径下比较现有指标和历史诊断，列出证据支持/反对的解释，不修改文件。
- Sol：核验两份结果、处理冲突、决定是否达到“可判断”标准并写最终记录。
- 同时最多两个工作者；协调者验证原始文件，不把子代理结论直接当证据。

### 单一下一动作

复用现有 CPU link-activity diagnosis，对当前三个 checkpoint 和原 sample IDs 生成一次新的只读分数诊断报告。

### 阶段 A 结果

- 已完成：旧 link-activity 小样本诊断、expanded failure diagnosis、threshold protocol、RB metric repair 和 throughput diagnosis 均已登记；它们发生在状态接力修复前，只复用方法和边界，不继承数值结论。
- 最新 recursive 三 seed 只保存 metrics 与 calibration threshold，没有正负样本分数分位数和完整 validation 阈值网格，因此根因仍未确定。
- 进入阶段 B 的唯一动作：复用现有 CPU link-activity diagnosis 脚本，对当前三个 checkpoint 和原 sample IDs 做一次新 checkpoint 范围的只读诊断。
- 复用限制：旧脚本只匹配旧运行目录命名；使用系统临时目录的只读 junction 适配目录名，不修改脚本、不复制 checkpoint、不覆盖历史报告。
- 启动调整：系统临时目录加自动删除的命令被策略拒绝且未执行；改用 `code/artifacts/tmp/pi_jwm_p4_recursive_link_diag_adapter_20260830/` 保留只读 junction，作为非权威适配层并记录来源。

### 阶段 B--D 结果与下一步

- 新诊断报告：`code/artifacts/audit/pi_jwm_p4_recursive_link_activity_diagnosis_20260830/link_activity_diagnosis.json`，SHA-256=`BF94979D9A761EE440E0EC39EFCCAB6D9056F200B82455ECD11E8E38E9897F01`；仅 CPU、未启动 GPU、未访问 `locked_test`。
- seed `20260831` 在冻结阈值 `0.9` 下，validation 第 1/5/20 步的 TP/FP/FN 分别为 `0/0/364`、`85/42/309`、`353/1125/26`；calibration 第 20 步也有 `179/1460/24`。高置信假阳性主要随递推步数累积，不是第 1 步即出现。
- 全 20 步 validation 在共同阈值 `0.9` 下，seed `20260831` 的 FP=`6175`，明显高于另两 seed 的 `2186/1414`；其 overall score q99=`0.9229`，而 negative q90=`0.2823` 与另两 seed 同量级，说明问题集中在高分尾部，不是整个负样本分布整体抬升。
- 当前直接失败机制已经定位为“长步 recursive rollout 中负样本高分尾部放大，导致高置信假阳性累积”；阈值迁移有影响，但不能单独解释。造成尾部放大的底层模型/损失原因尚未闭合，因此不提出或实施修复。
- P4 仍 blocked，`gpu_allowed=false`、`formal_performance_claim_ready=false`、`locked_test_accessed=false`；既有递推机制、数据合同、checkpoint reload 和运营指标改善证据继续保留。
- 已完成只读底层审计：link head 直接读取逐步更新的 edge latent；第 2 步起 edge latent 同时经过 GRU 递推和规则状态反馈。三个 seed 的 class weight、loss weight、样本和训练协议相同，且训练记录只保存总损失。
- checkpoint 参数对比未发现 edge GRU 或 physical-edge feedback 权重范数异常；seed `20260831` 的 link-head bias=`+0.1450`，另两 seed为 `-0.1650/-0.1526`，但该差异不足以单独证明高置信尾部的来源。
- 单一下一动作：在获得新增只读诊断预算后，只对现有 checkpoint 保存逐 horizon 的负样本 logit 分位数及去除 link-head bias 后的对应分位数，用于区分“输出偏置”与“递推 edge latent 漂移”；不得训练或改方法。

## 2026-08-30 P4 严格收口总计划（当前）

- 详细实施计划：`记录/实施计划/2026-08-30-P4严格收口实施计划.md`；用户已确认，继续保持一个 P4 主线。
- 当前门：P4 blocked；未进入 P6，`formal_performance_claim_ready=false`、`locked_test_accessed=false`。
- 已完成 Task 1：冻结当前修复后 h20 tensor、三 seed sample IDs/配置、已通过门、失败门和历史适用边界。
- 实际模型路由：Luna 提取 P4 验收矩阵，Terra 设计 bias/latent 最小诊断，Sol 纠正旧 tensor 路径、冻结门和最终计划。
- CPU 新边界：只读诊断、单测、compile 和极小 checkpoint reload micro-smoke；不再做 CPU 三 seed 或 `256/128/128` 性能训练。
- GPU 新边界：任何真实性能训练前必须先向用户报告单一变更、冻结配置、预计 jobs、输出目录和停止条件；用户开启 GPU 后才执行。
- 当前单一下一动作：Task 2，先为 v2 bias/latent 诊断写失败测试；Task 4/5 完成前不提出修复，不需要开启 GPU。

## 2026-08-30 P4 v2 bias/latent 只读诊断完成

- 已按 TDD 新增 v2 只读诊断和 6 个定向测试；RED 为预期的模块不存在，GREEN 为 `6/6` 通过。相关世界模型、指标和 gate 测试共 `31/31` 通过，`compileall` 通过。
- 报告：`code/artifacts/audit/pi_jwm_p4_recursive_link_bias_latent_diagnosis_20260830/link_bias_latent_diagnosis.json`，SHA-256=`4fbbd9f41bd4e300200e4e276ea5f3401f63c21b4d58e8962641b50723df0783`。
- 3 seeds × validation/calibration × 20 horizons 齐全；v2 六组 raw FP 总数与 v1 完全一致；tensor manifest 仍为 `d5c9edb4200840ad085adaab970855555c19198ed0031234513ea1f078660781`。
- seed `20260831` validation 全步 raw/去 bias FP=`6175/5739`，h20=`1125/1086`；去 bias q99 在 h1/h5/h20=`-0.19/-0.34/4.96`。固定 bias 只能解释少量阈值穿越，长步 `w·edge_latent` 投影漂移仍然存在。
- 当前可证伪结论只到“edge latent 在 link head 方向发生长步漂移”，不能指认 GRU、图消息或规则反馈为具体根因。P4 仍 blocked，`gpu_allowed=false`、`formal_performance_claim_ready=false`、`locked_test_accessed=false`。
- 单一下一动作：只对 seed `20260831` validation 运行一个同 checkpoint、同样本、无训练的 direct physical-edge rule-feedback 置零干预；不同时测试其他分支，不改模型或阈值。

## 2026-08-30 P4 direct physical-edge rule-feedback 因果干预完成

- 新增严格受限的 intervention 脚本及 8 个测试；v2 回归 `6/6`、compileall 和 diff check 通过。真实运行只使用 seed `20260831` validation 和原 checkpoint/sample IDs。
- 报告：`code/artifacts/audit/pi_jwm_p4_direct_edge_rule_feedback_intervention_20260830/direct_edge_rule_feedback_intervention.json`，SHA-256=`6b89049612e97bcb50ad929662423f17630f5a220a6ac78b4bd31ee8f5ea1f2e`。
- 屏蔽 direct physical-edge rule-feedback 投影后，h20 去 bias FP `1086 -> 0`，pre-bias q99 `4.96 -> -0.16`；h1 全部统计不变。预注册判定为 `sufficient_to_explain`。
- 该结果只证明直接反馈路径足以造成失败 seed 的长步高分尾部；未统计真实正样本保护，不能把“永久关闭反馈”当修复，也不授权指认其他路径。
- 当前门：P4 blocked，`gpu_allowed=false`、`formal_performance_claim_ready=false`、`locked_test_accessed=false`。单一下一动作是由用户确认根因专属修复设计；确认前不改模型、不训练。

## 2026-08-30 P4 Task 6 物理边反馈输入路由完成

- 用户已确认方案 A；只把 physical-edge rule correction projection 从直接加到 edge hidden 改为加入 edge GRU input message，node/flow/task feedback 和其余合同未改。
- TDD RED 准确捕获旧行为：h2 hidden 48/48 元素不同，最大绝对差 `0.3548906`；GREEN 后八组定向回归 `70/70`，compileall、diff check 和两阶段 Sol 审查通过。
- Luna 核验发现旧 CPU smoke runner 不开启规则层且不执行 reload；计划已纠正为复用 `run_formal_training(..., device='cpu')`，未新增 runner。
- 隔离 CPU micro-smoke 仅运行一个 canonical 方法、`2/1/1`、1 epoch；新旧 checkpoint strict load，manifest 18 文件 `0 mismatch`，参数量 `83750`。
- 命令记录：一次 `rg` 正则因未闭合括号报错；已改用 `rg -F`，退出码 1 按“旧语句不存在”处理，未重复错误命令。
- 当前门：Task 6 接口门通过，但 P4 性能仍 blocked；`gpu_execution=false`、`formal_performance_claim_ready=false`、`locked_test_accessed=false`。单一下一动作是用户开启 GPU 后运行 sentinel seed `20260831`。

## 2026-08-31 P4 GPU sentinel 启动

- 用户已开启 `connect.nmb1.seetacloud.com:14826`；只读核验为 RTX 4090、PyTorch `2.8.0+cu128`、CUDA 可用，启动前 GPU 空闲。
- 已冻结机器可读协议 `code/artifacts/protocols/pi_jwm_p4_edge_feedback_gru_input_gpu_protocol_20260831/protocol.json`。模型文件、训练入口、协议和 tensor manifest 的本地/远端 SHA-256 均一致。
- 复用服务器上已验 canonical tensor，只复制到新隔离目录；未上传、生成或访问 `locked_test`，未复用旧 checkpoint/旧结果。
- 真实路由：Luna 提取历史远端流程，Terra 审计当前 CLI 与 sentinel 字段，Sol 纠正历史 residual scale `0.5` 为当前冻结 `1.0` 并完成最终启动审查。
- 仅启动 sentinel seed `20260831`，远端 PID=`2028`；另外两个 seed 未启动。当前单一下一动作是等待该进程完成并审计产物。

## 2026-08-31 P4 GPU sentinel No-Go

- sentinel 正常完成：`training_run_complete=true`、`gpu_execution=true`、strict reload 通过、18 个 manifest 文件 `0 mismatch`、`locked_test_accessed=false`。
- sample IDs 与冻结 seed `20260831` 内容完全相同；raw SHA 差异仅为 Windows CRLF 与 Linux LF，canonical JSON SHA 均为 `3ceec1827e65ca05f34d1e73ba3d6e73ac1b041e4860f85f3035bc9a28329021`。
- 保护项通过：node-x MAE ratio=`1.0988 <= 1.25`；throughput/RB/task-delay MAE delta=`-0.6611/-0.6477/-1.3395`，均未回退。
- 核心失败：candidate/persistence validation link-F1=`0.3724/0.5901`，delta=`-0.2177 < -0.05`。sentinel 判定 `no_go`。
- 审计 SHA-256=`8dddf747b19f6b06681e0e431270473696ed1a52adc8555ce737ad9ab2168d0a`。seeds `20260830/20260832` 未授权、未启动；GPU 当前空闲。
- 当前单一下一动作：只读判断新 checkpoint 的 link-F1 失败是否仍由长步高置信假阳性主导；不得训练、调阈值、增加第二个修复或进入 P6。

## 2026-08-31 P4 sentinel 阈值迁移只读判定完成

- 真实成本路由：Luna 机械提取候选表，Terra 复核 calibration-only 选择实现与证据缺口，Sol 回查原始 JSON/CSV、纠正默认阈值与事件阈值以及单步与汇总口径，并完成最终判定。
- 现有 sentinel 产物只保存 calibration 五候选和 validation 最终阈值，证据不足；因此仅复用既有 `run_formal_p4_link_activity_diagnosis_v1.py`，对同一 checkpoint、sample IDs 和有效 mask 做一次 CPU 只读 score replay。未训练、未启动 GPU、未访问 `locked_test`。
- 报告：`code/artifacts/audit/pi_jwm_p4_edge_feedback_gru_input_threshold_replay_20260831/link_activity_threshold_replay.json`，SHA-256=`756f63eead1847c94b9e53024c10af4d3499d25403bb34d333e6938b9a4a7f43`。
- validation 五候选 F1：`0.0721/0.2834/0.3715/0.4573/0.3724`；最优候选是 `0.7`，但仍低于 persistence `0.5901`，delta=`-0.1327 < -0.05`。
- 结论：calibration 选出的 `0.9` 确实造成 validation 低 recall，但不是 sentinel No-Go 的充分解释；在当前预注册候选集合内没有阈值超过 persistence，不能通过事后换阈值关闭 P4。
- 当前门：P4 blocked；`formal_performance_claim_ready=false`、`locked_test_accessed=false`，seeds `20260830/20260832` 保持未启动。
- 单一下一动作：停止训练和阈值试探，先形成并由用户确认一个理论--实现--指标一致的 link score/threshold 方法级决策；确认前不实施第二个修复、不改冻结协议、不进入 P6。

## 2026-08-31 P4 link score/threshold 方法决策计划完成

- 计划入口：`记录/实施计划/2026-08-31-P4-link-score-threshold方法级一致性决策计划.md`。
- 唯一目标：用已有理论、源码和当前 sentinel 证据，决定 link head raw sigmoid 是“事件概率”还是“cost-sensitive score”；只写 decision memo，不改方法。
- 实际路由：Luna 做带字段路径的机械提取，Terra 做验收矩阵审查，Sol 纠正 Luna 再次混淆顶层默认阈值 `0.5` 与 `thresholds.link_activity=0.9`，并冻结最终计划。
- 计划不产生训练、forward replay、阈值 sweep 或新 seed；GPU 不需要，`locked_test` 不访问，P4 保持 blocked。
- 唯一下一动作：按计划 Tasks 1--6 复用现有证据形成 decision memo，并在任何方法定义变化前提交用户确认。

## 2026-08-31 P4 link score/threshold 方法决策完成（待用户确认）

- decision memo：`记录/设计/2026-08-31-P4-link-score-threshold方法级一致性决策.md`。
- provenance、理论、loss/score/threshold 实现、当前 sentinel 和历史机制边界均已只读核验；没有训练、forward replay、GPU 或 `locked_test` 访问。
- 三态判定为 `protocol_theory_mismatch`：当前 No-Go 仍有效，但主文档要求事件概率/概率校准，而正式 runner 直接使用 `pos_weight=50.0` 的 raw weighted sigmoid 选阈值，correction utility 与 Brier/ECE 未进入正式链。
- 唯一推荐：保留 link event probability 理论目标和当前模型/loss/数据/方案 A；下一步只设计 post-training link event probability calibration boundary，不同时改 loss、模型或做 threshold sweep。
- Terra 固定清单审查已完成并纠正 overall 口径文字；Sol 已复核 top default=`0.5`、effective link threshold=`0.9`。
- 当前门：P4 blocked，Task 9 未开放，follow-up seeds 未启动，`formal_performance_claim_ready=false`、`locked_test_accessed=false`。
- 单一下一动作：由用户确认上述唯一方法方向；确认前不改代码、理论正文或协议。

## 2026-08-31 P4 link 事件概率校准边界方案 B 已批准

- 用户已批准方案 B：固定 `pos_weight` 解析反演后，仅在 calibration split 拟合一个 scalar temperature；不修改模型、weighted BCE、数据、方案 A 或 raw threshold candidates。
- 书面规格：`记录/设计/2026-08-31-P4-link事件概率校准边界设计.md`。raw score、event probability、mapped threshold、Brier/ECE/NLL 和 decision-equivalence 边界均已明确。
- 该设计只关闭 `protocol_theory_mismatch`，不会改变当前 sentinel TP/FP/FN/F1 或推翻 No-Go；因此不能单独宣称 P4 完成。
- 当前不改实现、不训练、不用 GPU、不访问 `locked_test`、不进入 P6；follow-up seeds 保持未启动。
- 单一下一动作：由用户复核书面规格；确认后才调用 `writing-plans` 制定 TDD 单变量实施计划。

## 2026-08-31 P4 link 事件概率校准边界实施计划已批准

- 用户已复核并通过方案 B 书面规格；设计状态已冻结为 `approved and frozen for implementation`。
- 实施计划：`记录/实施计划/2026-08-31-P4-link事件概率校准边界实施计划.md`，固定为数学工具 → metrics → 正式 runner → sentinel CPU audit → 独立验收 → 权威记录。
- 实施使用 TDD 和成本路由：Terra 执行边界明确的 RED/GREEN 小任务，Sol 复核每个 diff、测试和最终机器 artifact。
- 当前仍不需要 GPU，不做 CPU 大训练，不访问 `locked_test`，不启动 follow-up seeds，不进入 P6。
- 单一下一动作：执行 Task 1，先写失败测试并观察准确 RED。

## 2026-08-31 P4 方案 B Task 1 概率数学边界完成

- Terra 严格 TDD：首轮新增 API 先以 ImportError RED，审查补测再次准确捕获实例接口、极端温度、复数输入、非有限权重和极小权重反演问题。
- 最终只修改 `formal_binary_calibration_v1.py` 与对应测试；解析反演、scalar temperature LBFGS、固定 15-bin ECE、raw tie-break 和逐候选 decision equivalence 已实现。
- fresh 测试 `16/16`、compileall、路径级 diff check 通过；独立 Sol 规格审查 PASS，独立 Terra 质量审查 APPROVED。
- 未训练、未使用 GPU、未访问 `locked_test`、未修改 runner/metrics/model/loss/tensor。
- 单一下一动作：Task 2 先写 metrics RED，让 `FormalMetricAccumulator` 只对 link 使用正式事件概率并报告 NLL/Brier/ECE。

## 2026-08-31 P4 方案 B Task 2 metrics 接入完成

- fresh Terra 按 TDD 让 `FormalMetricAccumulator` 只在传入 calibration 时消费正式 link 概率；未校准路径的 NLL/Brier/ECE 明确为 `not_computable`。
- 审查中两次纠正阈值语义：最终调用方只传 legacy raw threshold，accumulator 内部用同一 calibration 对象精确映射；link/其他事件坐标分开记录。
- float32 等值点通过保留同 dtype/device 的 legacy raw decision 精确保护；正式概率仍用于 NLL/Brier/ECE/AUPRC。
- fresh `15/15` metrics、`16/16` calibration、compileall/diff check 通过；规格 PASS、质量 APPROVED。
- 质量审查发现的 throughput/RB 全 mask 计数问题为 Task 2 前既有问题，本轮按主线约束不扩张修复。
- 单一下一动作：Task 3 先写 runner RED，接入 calibration-only temperature、sidecar 和显式 raw/probability threshold 字段。

## 2026-09-01 P4 方案 B Task 3 正式 runner 接入完成

- fresh Terra 按 TDD 将 train-only link `pos_weight`、calibration-only temperature、raw candidate identity 和 mapped probability threshold 接入正式 runner。
- learned method 写独立 calibration sidecar；RULE baselines 不拟合温度、不写 sidecar或 probability threshold；comparison 明确 legacy raw 坐标。
- 审查新增硬门：空/全零/全一 calibration labels 均停止；输出先写同级唯一 staging，成功后一次发布，失败清理本次 staging且拒绝覆盖已有 target。
- fresh runner `10/10`、metrics `15/15`、calibration `16/16`、compileall/diff check 通过；规格 PASS、质量 APPROVED。
- 未运行真实训练、GPU、当前 sentinel inference 或 `locked_test`。
- 单一下一动作：Task 4 先写 audit core RED，再创建当前 sentinel CPU-only 校准审计入口。

## 2026-09-01 P4 方案 B Task 4 CPU-only 校准审计入口完成

- fresh Terra 按 TDD 新增 `run_formal_p4_link_probability_calibration_v1.py` 与对应测试；首轮 RED 为模块不存在，后续 RED 依次捕获 validation 概率退化、mask/provenance 漂移、float32 临界点、真实 tensor manifest 字段、strict reload 和发布竞态问题。
- audit core 固定 calibration 拟合、validation 只评价；正式分类显式复用 legacy raw decision，正式 probability threshold 严格复用共享 float64 映射，并把有限精度下的 direct comparison 结果单独如实记录。
- CLI 锁定 seed `20260831`、method、canonical tensor manifest SHA、train-only `pos_weight=50.0`、checkpoint/sample IDs/selected tensor 身份；strict reload、同一 physical-edge 与 aggregate mask、owned staging 和 non-locked 边界均有测试。
- Sol 规格审查 PASS、fresh Terra 质量审查 APPROVED；root fresh 复跑 Task 4 `11/11`、calibration `16/16`，compileall/diff check 通过。
- 未运行真实 sentinel inference、GPU、训练或 `locked_test`；P4 与 sentinel No-Go 尚未由新 audit 复核。
- 单一下一动作：Task 5 先跑四组完整定向回归，再对冻结 seed `20260831` 执行一次 CPU-only audit；任一机器门失败立即停止。

## 2026-09-01 P4 方案 B Task 5 真实 CPU audit 停止

- Task 5 前置回归全部通过：calibration `16/16`、metrics `15/15`、formal runner `10/10`、audit `11/11`，合计 `52/52`；全目录 compileall 通过。
- 随后只对冻结 seed `20260831`、canonical tensor、method `coupled_dual_gnn_residual` 执行一次 CPU-only audit；未训练、未使用 GPU、未访问 `locked_test`。
- 真实运行在 `build_link_probability_calibration_audit` 的 validation 非恶化门停止，精确异常为 `ValueError: validation nll regressed after temperature fitting`。
- 按冻结规则立即停止：未改 loss/model/threshold grid，未换 calibrator，未重跑，seeds `20260830/20260832` 仍未启动。
- 输出目录与 staging 均不存在，因此没有可接受的 audit JSON/manifest；不得写“方案 B 完成”或 `probability semantics gate = passed`。
- 当前状态：`probability semantics gate = failed`、`sentinel performance gate = no_go`、`P4 = blocked`、`formal_performance_claim_ready=false`、`locked_test_accessed=false`。
- 单一下一动作：由用户基于该失败确认是否另立新的单变量方法决策；确认前不再实现、训练或运行概率补救路线。

## 2026-09-04 P4低召回诊断与单次因果核验设计

- 冻结seed `20260831`真实CPU只读诊断已完成；candidate TP/FP/FN精确复现`1902/557/5855`，报告SHA-256=`ee03931c27ddccf4b8967247db3f06dcd3db7f2efe546917ca5e6d7bf1556e4a`。
- 主要漏报集中在持续活跃链路：该组candidate/persistence recall=`0.2648/0.7205`；2,806个persistence正确而candidate漏报样本中2,791个属于该组。
- 逐步recall呈h1=`0.0000`、h7=`0.5778`、h20=`0.0106`的“两端塌陷”；当前证据支持优先核验edge GRU更新，但尚未证明它就是根因。
- 用户已口头批准一次CPU-only edge GRU旁路干预；书面规格为`记录/设计/2026-09-04-P4-link低召回edge-GRU单次因果核验设计.md`，待用户复核后才实施。
- 当前仍为P4 blocked；不训练、不需要GPU、不启动follow-up seeds、不访问`locked_test`、不进入P6。
- 单一下一动作：用户复核书面规格；通过后用TDD实现唯一旁路干预，任何不支持结果立即停止。

## 2026-09-05 P4 edge GRU单次因果核验完成并停止

- 用户复核书面规格后，Terra按TDD完成唯一旁路干预；独立Sol规格审查发现并闭合固定输入SHA、关键集合、机器判定字段和逐元素mask指纹问题。最后一个mask展平缺口因Terra服务认证失败，由主Sol做最小修复。
- 主Solfresh回归：intervention `13/13`、recall `9/9`、formal model `9/9`、probability audit `11/11`，合计`42/42`；compileall通过。
- 唯一真实CPU干预运行一次。baseline精确复现TP/FP/FN=`1902/557/5855`；旁路edge GRU后为`0/0/7757`，原始2,791个关键漏报找回`0`个。
- 六项门只有FP上限通过，其余五项失败；机器判定=`not_sufficient_to_explain_dominant_low_recall`。该结果排除“edge GRU更新单独擦除活跃记忆”作为充分解释，不能指认下一模块。
- 报告：`code/artifacts/audit/pi_jwm_p4_edge_gru_bypass_intervention_20260904/edge_gru_bypass_intervention.json`，SHA-256=`61618d044960c77209cb8af4ebc33f16487e63928b2764d6b1090ca4a2fd9bf1`。
- 当前仍为P4 blocked；不训练、不需要GPU、不启动follow-up seeds、不访问`locked_test`、不进入P6。
- 单一下一动作：停止当前路线，由Sol形成下一份单变量只读机制设计并获得用户确认；确认前不尝试CFE、rule feedback、link head、loss或阈值干预。

## 2026-09-05 P4 edge GRU接口只读追踪设计待复核

- 用户已批准继续做一个不改变输出的edge GRU接口追踪；不再旁路CFE、rule feedback或其他模块。
- 书面规格：`记录/设计/2026-09-05-P4-edge-GRU接口只读追踪设计.md`。唯一问题是区分原始5855个FN在进入本步GRU前已经低于阈值，还是由本步更新向下跨阈值。
- 追踪只复制GRU input、hidden before/after并用同一link head读取；正式forward输出必须逐元素不变，GRU方程重算误差不超过`1e-6`。
- 预注册三态判定使用overall/h1/h20共同`0.80`覆盖门；任何结果都不自动授权修复或训练。
- 当前不写代码、不运行trace、不需要GPU、不访问`locked_test`；P4仍blocked。
- 单一下一动作：用户复核书面规格；通过后再制定TDD实施计划。

## 2026-09-05 P4 edge GRU接口只读追踪完成并停止

- 用户复核后严格执行`2026-09-05-P4-edge-GRU接口只读追踪实施计划.md`；只新增trace runner与测试，没有修改正式模型、loss、tensor、阈值或方案B。
- 成本路由真实执行：Terra负责首轮机械实现；两轮仍有覆盖缺口后按规则升级Sol；独立Sol规格复审PASS，fresh Sol质量复审APPROVED，主Sol完成最终验收。
- 主Solfresh回归：trace `22/22`、recall `9/9`、bypass `13/13`、formal model `20/20`、probability audit `11/11`，合计`75/75`；compileall退出码0。
- 唯一一次真实CPU trace成功发布到`code/artifacts/audit/pi_jwm_p4_edge_gru_interface_trace_20260905/`。报告SHA-256=`87b4fe4b17d06928a1a3583be45f0521b51aa90866d39ce43a95075d868027d9`，manifest SHA-256=`9fccddfbf43bc0832bdeea28dbe8d765963d2126c2016d852b78816d5e1147b4`。
- 冻结TP/FP/FN=`1902/557/5855`，hook calls=`1280/1280`，hook已移除，post/official与GRU重算最大误差均为`0`，三个输入SHA均匹配。
- 机器判定=`incoming_readout_below_threshold_dominant`：overall FN中`5559/5855=94.94%`在进入本步GRU前已低于阈值，h1为`364/364=100%`，h20为`370/375=98.67%`；本步更新向下跨阈值只占overall `296/5855=5.06%`。
- 该结果排除“本步edge GRU更新主导将正样本压低”，但不能单独指认history encoder、输入消息或link head。静态核对还确认`aggregate_link_activity` 由`active_task_count>0`定义，而`active_task_count`已在`physical_edge_state`中进入模型，不存在“活动信号字段完全缺失”。
- 当前仍为P4 blocked；不训练、不需要GPU、不启动seeds `20260830/20260832`、不访问`locked_test`、不进入P6。
- 单一下一动作：由Sol提交一份“link activity持久性残差状态转移”候选方法设计供用户确认；先闭合无target泄漏的递推定义、weighted-BCE/方案B概率语义和单机制边界，确认前不写修复代码、不训练。

## 2026-09-05 P4 link activity持久性残差候选设计待确认

- 成本路由已真实执行：Luna只盘点现有接口，Sol独立比较三种互斥方法并完成最终方法把关；没有训练、推理或代码实现。
- 唯一推荐候选为“上一活动状态的有限log-odds持久性基准 + 现有单输出head学习变化量”；hazard双输出因扩大参数和假设被暂不采用，简单skip/bias因概率与递推语义不闭合被否决。
- 唯一模型变化限定在link activity读出/递推适配器；双图、history encoder、edge GRU、消息与耦合、数据、loss、`pos_weight=50`、阈值候选、方案B、训练预算和seed均保持不变。
- h1只读历史最后一帧；h2-h20只递推模型自己的上一预测logit，未来target及其mask只能进入loss/metric，不能进入forward。
- 当前标记仍是`candidate method`和`not implemented`；P4继续blocked，GPU和follow-up seeds不开放，`locked_test_accessed=false`。
- 单一下一动作：等待用户只确认或否决该数学与范围定义；确认后才写正式设计文档和TDD实施计划，不并行尝试其他方法。

## 2026-09-05 P4 persistence residual Task 1 RED审查修订

- Sol审查发现并修正了一个关键坐标问题：已有`+20/-20`是加权raw logit，候选方法零残差时必须仍输出`+20/-20`；内部未加权`u`才是`raw-log(pos_weight)`。
- 同时修正测试夹具的最后一帧物理边mask，并补充同语义验证、缺失身份拒绝测试；没有改生产代码。
- 修订后的下一动作：Terra重新运行定向RED并在确认RED仍为接口缺失后，按已冻结范围实现最小GREEN。

## 2026-09-05 P4 persistence residual GREEN首轮问题

- Terra已开始最小实现，但首轮测试暴露两处契约问题，当前不进入回归或真实数据：配置内部机制名与外部方法身份需要明确分层；观测`±20`必须先减去`log(50)`作为内部`u`，再在输出端加回`log(50)`，否则会错误改变raw坐标。
- 测试夹具已补齐固定先验，下一次只验证这两个修正；不修改数据、GRU、loss或阈值。

## 2026-09-05 P4 persistence residual CPU门完成

- 已按TDD闭合内部`u`/正式`z`坐标、纯预测递推、train-only先验、外部/内部方法身份、旧checkpoint拒载和运行证据字段；新候选保持原`coupled_dual_gnn_residual`的其余状态头与方案A规则反馈。
- fresh定向与保护性回归共`88/88`通过，compileall和`git diff --check`通过。
- canonical h20 tensor上的一次CPU micro-smoke使用`2/1/1`、1 epoch、hidden=4；h1/h20有限，物理边规则反馈实际执行19次，checkpoint严格复载`0/0`，manifest 19项零不一致。
- 当前仍为`P4=blocked`和candidate method；本证据不说明性能提高。未使用GPU、未访问`locked_test`、未运行follow-up seeds。
- 单一下一动作：用户确认并提供可用GPU后，只运行seed `20260831`的冻结sentinel双门验收。

## 2026-09-05 P4 persistence residual GPU可用性检查

- CPU门通过后仅对用户已提供的`connect.nmb1.seetacloud.com:14826`执行一次只读SSH探测，握手前返回`Connection refused`。
- 未上传文件、未启动训练、未创建远端结果；这不是模型或数据门失败。
- P4仍blocked，`locked_test_accessed=false`。单一下一动作：GPU端点恢复后原样运行seed `20260831` sentinel，不用CPU替代、不启动其他seed。

## 2026-09-05 项目全局阅读与推进评估

- 已完成交接阅读、目录清点、关键代码/产物核查；报告：`记录/研究进展/2026-09-05-项目现状与主线推进评估.md`。
- 本轮重算CPU micro manifest 19项零不一致，checkpoint严格加载成功，模型契约测试5/5通过；未训练、未联网、未访问locked_test。
- 新发现：方案B独立审计入口拒绝新持久性残差方法，且硬锁旧0.9阈值；最小调用已复现，未创建产物。故旧“唯一阻塞是GPU”描述不完整。
- P4仍blocked；建议下一交付仅为新候选双门验收入口闭合，随后再执行原冻结单seed协议。建议尚未实施，不改变已确认模型/数据/loss/阈值集合。

## 2026-09-05 P4 新候选双门验收入口闭合

- 按TDD先加入新方法入口与calibration阈值解析测试，旧`coupled_dual_gnn_residual`仍固定raw threshold=`0.9`。
- 概率审计支持`link_activity_persistence_residual_v1`：动态checkpoint/模型推理身份、threshold-selection文件、run manifest绑定、calibration-only阈值和方法语义均已接通；不改变模型、数据、loss或候选集合。
- 定向概率审计`13/13`，持久性模型`5/5`，持久性runner`8/8`，formal模型`9/9`，compileall和diff check通过。
- GPU只读端口`connect.nmb1.seetacloud.com:14826`本轮TCP连接失败；补查历史端口`14507`也失败。未上传、未训练、未访问`locked_test`。P4仍blocked。
- 单一下一动作：GPU恢复后只运行新方法seed`20260831`冻结sentinel，先性能门，后独立方案B概率门；性能失败即停止。

## 2026-09-05 P4 GPU sentinel 启动参数偏差

- 首次远端启动命令遗漏`--train-limit 256 --evaluation-limit 128`，实际进入全量窗口准备，不能作为冻结sentinel证据。
- 已停止PID`1771`并保留远端隔离staging目录作追溯；未发布run、未进入概率门、未访问`locked_test`。
- 唯一下一动作：在新的隔离目录按冻结`256/128/128`参数重新启动seed`20260831`，完成后先做性能门。

## 2026-09-05 P4 persistence residual GPU sentinel No-Go

- [x] 冻结GPU sentinel seed `20260831`完成：样本`256/128/128`、8 epochs、hidden=32、history/horizon=`8/20`。
- [x] 正式run已归档；strict reload=`true`、manifest `0 mismatch`、`locked_test_accessed=false`。
- [x] 性能门`no_go`：validation link-F1=`0.4801978`，persistence=`0.5900904`，delta=`-0.1098926 < -0.05`；node-x ratio约`1.2163`，吞吐/RB/task-delay未回退。
- [x] 未运行方案B概率门、未启动其他seed；P4继续blocked，`formal_performance_claim_ready=false`。
- 当前单一下一步：保留失败证据并等待新的单变量机制设计确认，不补跑、不调参、不进入P6。

## 2026-09-05 完整 RSSM 论文对照与实现

- [x] 完成 PlaNet、Dreamer、Graph Network Simulator、TD-MPC2、RDR、latent ensemble、事件图模型的一手来源机制矩阵。
- [x] 审计旧 `_GraphRSSMBackend`：确认 context posterior/prior 同源、无逐步 posterior、KL 仅覆盖 context，不能称完整 RSSM。
- [x] 新增 `complete_graph_rssm_v1`：分离逐步 prior/posterior、prior-only 部署、balanced KL 和多步一致性；保留双图和现有输出契约。
- [x] 新增完整性测试 4/4，旧 R4 RSSM 测试 4/4 通过，compileall 通过。
- [x] 复用独立 R4 CPU preflight 完成新候选真实 canonical h1/h5/h20、strict checkpoint 和 manifest 证据；9 个窗口全部通过。
- [ ] 尚未进行性能实验；P4 仍 blocked，GPU 未使用，`locked_test_accessed=false`。
- 唯一下一动作：对新候选做一次理论—代码—数据—指标最终审查，确认后才决定是否进入单 seed sentinel。

## 2026-09-05 完整 RSSM teacher reconstruction 一致性审查

- 论文对照确认 posterior 不能只用于 KL；训练期还需通过观测/状态重构目标直接训练。
- `complete_graph_rssm_v1` 已增加逐步 target-conditioned posterior teacher 的显式状态/分类输出；R4 重构目标在训练模式使用 teacher，正式 `predicted_*` 保持 prior-only。
- 定向测试 `5/5`、R4 preflight 回归 `3/3`、canonical h1/h5/h20 共 9 窗口和 manifest 门通过；完整性证据不等于性能提升。
- P4 仍 blocked，GPU 未使用，`locked_test_accessed=false`；唯一下一动作：GPU 恢复后按冻结协议只跑 seed `20260831` sentinel，性能门失败即停。

## 2026-09-05 完整 RSSM prior/teacher 双重重构收口

- 保留自由 prior rollout 重构，并在 R4 objective 增加权重 `0.5` 的 posterior teacher 重构，避免训练目标只监督 teacher。
- 定向测试 `5/5`、CPU preflight `3/3`、GPU screening `6/6`、compileall 和 diff check 通过；最新 canonical preflight 9 窗口通过。
- 当前仍无性能提升证据，P4 blocked；下一动作只在 GPU 可用后运行 `20260831` sentinel，性能门失败即停。

## 2026-09-05 完整 RSSM R4 GPU 筛选回收与正式路径接入

- 当前阶段：P4，正式性能门仍 blocked；R4 结果仅为完整实现的单候选 GPU 执行证据。
- 已完成：回传并核验`pi_jwm_complete_rssm_r4_gpu_screening_20260905`；27 epoch 早停、最佳 score=`4.4865667891`、checkpoint复载一致、manifest 13/13匹配、`locked_test_accessed=false`。
- 阻塞：完整 RSSM 还没有进入 P4 正式模型/runner，R4 的 single-candidate score 不可作为 P4 性能结论。
- 单一下一动作：实现并审计 P4 正式路径中的 complete RSSM 语义，先完成 CPU consistency audit，再决定是否启动冻结 GPU sentinel。

## 2026-09-06 P4 正式完整 RSSM sentinel 收口

- 已在正式 P4 路径实现 `complete_rssm_dual_graph_v1`：动作条件 prior、观测条件 posterior teacher、prior-only 部署、balanced KL、teacher/prior 双重重构和 overshooting；连续状态及分类事件均由 RSSM latent 解码。
- TDD 暴露并修复两项理论--实现缺口：首版未让 latent 解码分类输出；第二版 RSSM 连续 residual heads 未遵守 `zero_init_residual_state_heads=true`。两次 GPU run 均降级为无效过程证据，不进入正式比较。
- 最终方法身份为 `formal_complete_rssm_v1_1`，额外绑定 `rssm_residual_head_initialization=zero_when_requested_v1`，旧 checkpoint 会被拒绝。
- fresh 回归 `36/36` 通过；canonical h20 CPU consistency audit 全项通过，manifest 无不一致，未来 target 不影响 prior-only 输出，prior/posterior/teacher/transition 梯度均非零。
- 最终冻结 GPU sentinel：seed=`20260831`、data seed=`20260823`、样本=`256/128/128`、8 epochs、hidden=32、horizon=20。strict reload 通过，manifest `19/19`，`locked_test_accessed=false`。
- 性能门 `no_go`：link-F1 delta=`-0.04113 >= -0.05`，throughput/RB/task-delay 均不回退；但 node-x MAE ratio=`1.28774 > 1.25`。h1/h5/h10/h20 node-x MAE=`2.205/6.948/13.763/30.005 m`，长期误差仍高于 persistence。
- 未启动 follow-up seeds，未运行独立概率门，`formal_performance_claim_ready=false`，P4 继续 blocked。
- 单一下一动作：仅用现有 v3 checkpoint 做 CPU 只读 node-x 分解，区分正式双图 base 与 RSSM correction 对长期位置误差的贡献；不训练、不调参、不访问 `locked_test`。

## 2026-09-06 P4 node-x 修正项只读诊断

- [x] 使用最终 v3 checkpoint 和同一组 128 个 validation sample IDs，完成 `final = formal base + RSSM correction` 的 CPU 只读分解；strict reload、样本复用和逐值重构均通过。
- [x] formal base 的 node-x MAE=`13.71936 m`、相对 persistence 比值=`1.15756 <= 1.25`；完整输出为 `15.26231 m`、比值=`1.28774 > 1.25`。因此基础双图预测本可通过位置门，RSSM 连续修正足以使其越过失败线。
- [x] RSSM 修正绝对值均值=`3.04245 m`，仅在 `23.79%` 的有效节点上降低误差；该证据定位了失败来源，但不证明任何修复有效。
- [x] 未训练、未调参、未使用 GPU、未访问 `locked_test`；P4 仍 blocked，`formal_performance_claim_ready=false`。
- 当前单一下一动作：评审候选 `node_x_residual_non_degradation_v1`，即只在训练期处罚 RSSM 修正使 node-x 差于 base 的部分；获准后先做 CPU 理论/梯度/泄漏审计，再决定是否允许一次新 sentinel。

## 2026-09-06 P4 node-x 非劣化约束 CPU 闭合

- [x] 新增独立候选 `complete_rssm_node_x_safe_dual_graph_v1`；旧 `complete_rssm_dual_graph_v1` 保持原损失合同，不覆盖、不重跑。
- [x] 实现唯一新增项 `node_x_residual_non_degradation_v1`，权重固定 `1.0`；只处罚 prior RSSM node-x correction 使误差差于 detached base 的部分，posterior teacher 不重复计算该约束。
- [x] RED 测试分别复现缺少权重、缺少显式 correction、缺少候选身份；GREEN 后相关回归 `39/39`，新增隔离梯度测试通过。
- [x] canonical h20 CPU micro train/validation/calibration=`1/1/1` 完成；strict reload、manifest、prior-only target invariance、posterior target sensitivity、动作条件和完整 RSSM 梯度均通过。
- [x] 隔离新损失=`0.00048227`，RSSM node correction gradient sum=`1.94717`，formal base gradient sum=`0.0`；证明约束只训练修正支路，不直接推动 base。
- [x] 新协议完全复用 v3 seed、data seed、`256/128/128`、8 epochs、hidden=32、h20、优化器和五个性能门；协议 SHA-256=`a93b1499a8decfdb6d015c5bd1eb2467d7045fc40a5c75bde8b5828bc63951f5`。
- 当前状态：协议门=`ready_when_gpu_available`，但 GPU 已关闭且未启动训练；P4 仍 blocked，其他 seed、概率门和 `locked_test` 继续关闭。
- 当前单一下一动作：等待 GPU 可用后，只运行该冻结协议的 seed `20260831` sentinel；性能门任一失败立即停止。

## 2026-09-06 P4 node-x 非劣化约束 GPU No-Go

- [x] 唯一冻结 seed `20260831` 已在 RTX 4090 完成；GPU run、strict checkpoint reload、相同 sample IDs 和 manifest `19/19` 通过，`locked_test_accessed=false`。
- [x] link-F1 delta=`-0.01241`，throughput/RB/task-delay delta=`-0.33768/-0.64604/-1.35343`，均通过保护项。
- [ ] node-x MAE=`18.02197 m`，persistence=`11.85200 m`，ratio=`1.52058 > 1.25`；正式 sentinel=`no_go`。
- [x] 按停止规则未运行其他 seed、独立概率门或 `locked_test`，`formal_performance_claim_ready=false`，P4 继续 blocked。
- [x] CPU 只读分解确认 correction 已压到 `0.07621 m`，但 base MAE 恶化至 `17.97475 m`、ratio=`1.51660`；旧、新初始化哈希相同，说明从头联合训练的非劣化约束改变了 base 优化轨迹。
- 当前单一下一动作：停止并保留该失败；仅评审“复用 v3 checkpoint、冻结已通过位置门的 base、只训练 RSSM correction”是否可作为新的单变量候选，确认前不实现、不训练。

## 2026-09-06 P4 长期停滞系统性根因复盘

- [x] 复用八月中旬以来的 gate、诊断、正式 run 和最新 checkpoint 分解，完成跨阶段根因复盘；未训练、未改模型、未访问 `locked_test`。
- [x] 确认当前 sentinel 仅用 `256/128/128` 窗口、8 epochs、hidden=32、h20；两次正式完整 RSSM validation loss 到 epoch 8 均仍下降，而独立 R4 最佳轮次为 22。
- [x] 确认 runner 按 aggregate validation loss 选 checkpoint，与 P4 五项独立性能门不一致；link activity 正例仅约 `0.918%`，正式位置输入没有 heading/速度向量/路线或机动动作。
- [x] 最新 node-x-safe 结果证明局部 correction 约束会改变共享 base 的联合优化轨迹，因此撤回“直接冻结 v3 base 再训 correction”作为立即下一训练的建议。
- 当前门：P4 blocked；GPU 不需要，`locked_test_accessed=false`，不启动新模型、seed 或概率门。
- 当前单一下一动作：CPU-only 完成信息充分性、共享目标梯度冲突、训练预算与 checkpoint 选择一致性审计，再据机器证据三选一决定数据、训练分解或 gate 合同调整。

## 2026-09-06 P4 相关文献对照与结构边界复核

- [x] 对照 RSSM/PlaNet、G-RSSM、R-SSM、Graph Dreamer、GNS、无线 latent dynamics、Trajectron++、PhyDNet、动态图与多任务优化文献；优先复用本地文献库，联网只核对官方论文页和方法细节。
- [x] 逐行复核正式 `formal_complete_rssm_v1_1`：prior/posterior、KL、teacher、overshooting 和 prior-only rollout 语义完整，但随机状态是单个 global aggregate latent。
- [x] 确认节点、边和任务先被 pooled，RSSM decoder 再把同一个同类 correction 广播给所有实体；当前方法应准确表述为“formal dual-graph base 上的 global aggregate RSSM adapter”，不能表述为实体级 Graph-RSSM。
- [x] 文献与代码共同支持先审计 latent 粒度，而不是继续调 KL、非劣化权重、阈值、seed 或训练轮数；该判断不证明实体级 RSSM 必然通过 P4。
- [x] 未训练、未修改模型、未使用 GPU、未访问 `locked_test`；P4 仍 blocked，`formal_performance_claim_ready=false`。
- 当前单一下一动作：复用 v3 checkpoint 和相同 validation sample IDs，做 CPU-only 实体级可表达性审计，量化 global broadcast correction 相对 entity-wise residual 的不可约误差与 link 排序不变性。

## 2026-09-06 P4 第一性原理实体级 RSSM 收口执行

- [x] 用户确认：正式性能使用全部 unlocked 数据收敛训练；允许从历史位置合法派生运动状态；P4 全部门槛保持不变。
- [ ] Task 1：统一 CPU 第一性原理审计。
- [ ] Task 2：因果运动 tensor v2 与数据合同验收。
- [ ] Task 3：`entity_aligned_dual_graph_rssm_v1` 实现。
- [ ] Task 4：CPU 完整性门和正式训练协议冻结。
- [ ] Task 5：等待 GPU 后运行执行 sentinel；此前不启动 GPU、不访问 `locked_test`。
- 当前单一下一动作：以 TDD 建立运动派生与第一性原理审计的可验证函数，再在 canonical unlocked tensor/v3 checkpoint 上生成审计证据。

### 2026-09-06 执行结果（当前覆盖）

- [x] Task 1：统一第一性原理审计完成；复用 128 个固定 validation 样本，确认旧 global correction 基本不能改变边间排序、历史 speed/acceleration 全零、历史位置可合法派生运动，共享目标存在量级差异和负梯度余弦。
- [x] Task 2：新 unlocked tensor `pi_jwm_v4_tensor_v5_causal_motion_h20_unlocked_20260906` 完成；窗口与 split 不变，train-only normalization，运动统计非零，manifest SHA-256=`85e50b5ecf1742cfaecb859522587efaed3412d03a74f1a00e322164ae8861f6`。
- [x] Task 3：`entity_aligned_dual_graph_rssm_v1` 已进入正式模型、loss、method registry、runner 和 checkpoint strict identity；逐节点/边/流/任务 latent、posterior teacher、prior-only、规则递推、逐边 link correction 和运动 proposal 已实现。
- [x] Task 4：真实 h20 CPU 两阶段 micro train 和独立一致性审计通过；动作改写影响四类 entity latent，future target 不影响 prior，各 RSSM 分支梯度非零，冻结 base 后梯度为零，run/tensor manifest 无 mismatch。
- [x] `p4_gate_aware_v1` 已实际重放一轮，证明逐 epoch 保存和字典序选择路径可运行；短预算指标只作执行观察。
- [x] 正式协议冻结在 `code/artifacts/protocol/pi_jwm_p4_entity_rssm_frozen_protocol_20260906_v2/`。
- [ ] Task 5：本地无 CUDA，远端 `23874` 当前拒绝连接；GPU batch probe、sentinel 和全量训练均未启动。
- 当前门：P4 仍 `blocked`，`formal_performance_claim_ready=false`、`locked_test_accessed=false`、P6 未开放。
- 单一下一动作：远端 GPU 恢复后运行无 optimizer step 的 `8/4/2/1` 显存探测，选出不超过 85% 显存的最大 batch；随后才运行固定 `256/128/128`、总 2 epoch execution sentinel。

### 2026-09-06 GPU 执行门与正式 seed 启动

- [x] RTX 4090 上完成无 optimizer step 的 `8/4/2/1` batch 探测；四档均有限，batch 8 峰值 `4,121,439,744` bytes、占总显存 `0.16322`，因此三个正式 seed 的冻结 batch size 为 `8`。
- [x] 完成 seed `20260831` 的 `256/128/128`、base/RSSM 各 1 epoch execution sentinel；训练完成、base 冻结成立、checkpoint strict reload 通过、峰值显存 `4,140,385,280` bytes、manifest `19/19` 一致。
- [x] batch probe 和 sentinel 已回传本地，分别位于 `code/artifacts/audit/pi_jwm_p4_entity_rssm_gpu_batch_probe_20260906_v1/` 与 `code/artifacts/experiments/pi_jwm_p4_entity_rssm_gpu_sentinel_20260906_v1/`。
- [x] sentinel 仅作为执行证据，短预算性能不用于接受或否决 P4；`locked_test_accessed=false`、`formal_performance_claim_ready=false`。
- [ ] 首个正式 seed `20260831` 已按冻结配置在远端启动：全量 `9828/3276/1638`、base 20 epoch、RSSM 最多 40 epoch、前 20 个 RSSM epoch 不早停、patience 10、逐 epoch checkpoint、`p4_gate_aware_v1` 选择。
- 当前门：P4 仍 `blocked`，P6 未开放。
- 单一下一动作：等待 seed `20260831` 完成并核验其原始 checkpoint/metrics/manifest；只有该 seed 全部数值门通过，才原样运行另外两个 seed。

### 2026-09-06 正式 seed 运行进展

- [x] 隐藏 staging 已保存 base epoch `001/002/003` checkpoint，说明全量训练正在按 epoch 推进。
- 当前 base 阶段进度为 `3/20`；已观察到每个全量 base epoch 约 26 分钟，RSSM 阶段尚未开始。
- 当前仍无正式性能结论；P4 blocked，另外两个 seed 未启动，`locked_test_accessed=false`。
- 单一下一动作：继续等待 base 阶段完成并核验每个 epoch 的 checkpoint 与进程状态。

### 2026-09-06 运行证据保全

- [x] 在不停止、不暂停、不修改远端训练进程的条件下，建立正式 seed `20260831` 的只读证据包：`code/artifacts/experiments/pi_jwm_p4_entity_rssm_gpu_formal_seed_20260831_live_evidence_20260906/`。
- [x] 已保存完整启动命令、冻结配置、tensor manifest、进程/GPU 状态、staging 文件清单及 SHA-256；已回传 base epoch `001–005` checkpoint 原样副本。
- [x] 证据包已追加 epoch `006` 和第二份时间戳快照；当前 manifest SHA-256=`c4ef2361e6ac153b37640981ba1c09bd6faba810c6044f6903523a1b4cc4a675`，快照标记 `process_untouched=true`、`locked_test_accessed=false`。
- [x] 训练继续推进至 base epoch `007`；已追加第三份时间戳快照和 epoch `007` checkpoint，证据包 manifest SHA-256=`30186b78bc7cf87b723047a405bca9071e7032dd715160693680f3d111150433`。
- [x] 训练继续推进至 base epoch `008`；已追加第四份时间戳快照和 epoch `008` checkpoint，证据包 manifest SHA-256=`96e855cbe40e805d5f0cfbea0d44ca389369f1fc3cd3cab5e4ef8e6149330c67`。

### 2026-09-07 正式训练进入 RSSM 阶段

- [x] deterministic base 已完成 `20/20`，不再训练；实体级 RSSM 已推进到 epoch `2/40`。
- [x] 已只读回传 base epoch `014–020` 与 RSSM epoch `001–002` checkpoint，并保存阶段切换快照。
- [x] 当前 live evidence manifest SHA-256=`26a0fd37ea84e580410af7f2fcd9344ae0015b58e8be1ed8a3233c1b3f5a35ac`；远端进程保持运行，`locked_test_accessed=false`。
- 单一下一动作：持续监控 RSSM epoch；至少完成 20 个 RSSM epoch 后才允许按冻结 patience 判定早停。

### 2026-09-07 RSSM epoch 7–14 中间门趋势

- [x] 只读检查逐 epoch checkpoint 内的 `p4_gate`；epoch 7–14 连续 8 个 checkpoint 均为 `failed_hard_gate_count=0`、`all_numeric_gates_passed=true`。
- [x] epoch 14 当前值：validation link-F1 delta=`+0.44123`，calibration link-F1 delta=`+0.85695`，node-x ratio=`0.75057`，h5/h10/h20 ratio=`0.75523/0.74395/0.75849`，throughput/RB/task-delay ratio=`0.93588/0.47263/0.01723`。
- [x] validation state NLL 从 epoch 7 的 `-3.10564` 持续改善到 epoch 14 的 `-3.10747`；当前不是单 epoch 偶然越线。
- [ ] 这些仍只是首 seed 的训练中间 checkpoint；最小 20 epoch、最终 strict reload/manifest、非数值机制门和另外两个 seed 均未完成。
- 单一下一动作：保持冻结配置训练到至少 RSSM epoch 20，再按预注册 patience 与 gate-aware 排名决定最终 checkpoint。
- [x] 监控规则已补充：每出现新 epoch 追加时间戳快照和新 checkpoint，不覆盖旧证据。
## 2026-09-07 组会 PPT 大纲任务

- [x] 复用2026-08-12以来权威推进记录、机器审计、冻结协议和正式训练中间证据，形成9月9日组会PPT详细大纲。
- [x] 主汇报按22页组织，覆盖一致性审计、信息边、CPU规则、采集器、正式数据、tensor、baseline、规则层、方法适配、P4失败诊断、完整RSSM、第一性原理审计、文献、新实体级方法、训练协议、验证、当前结果和后续验收。
- [x] 另设6页备份材料，保留完整门槛、字段、数据划分、RSSM目标、实验时间线和复现入口。
- 当前边界：只完成大纲，没有制作或修改PPT二进制文件；结果页使用seed `20260831` epoch 17中间证据，组会前必须按最新正式产物更新。
## 2026-09-07 组会PPT大纲精简重构

- [x] 按用户指定结构将22页主汇报重构为10页：封面、工作总览、新数据2页、双图编码2页、世界模型3页、当前结果与下一步1页。
- [x] 组会可见内容移除内部阶段编号；文献依据、旧方案对比、机制验证和训练可靠性均并入对应数据/双图/世界模型页面。
- [x] 双图统一使用 `Strict Physical–Information Coupled Dual-Graph Message Passing`；当前世界模型使用与代码身份对应的实体级动作条件耦合双图RSSM名称。
- 当前动态项仅为组会前刷新首seed及其余seed进度，不改变训练过程或冻结协议。

## 2026-09-07 组会PPT原生追加制作

- [x] 以第204页为封面模板、第205页为正文模板，在 `meeting/PI-JWM_组会汇报.pptx` 末尾追加第206–215页；第1–203页没有编辑、删除或重排。
- [x] 新增页面按“工作总览—新数据两页—双图编码两页—世界模型三页—当前结果与下一步”组织，组会可见页面不使用内部阶段编号。
- [x] 结果页刷新为正式 seed `20260831` 的 RSSM epoch 22 中间证据：全部当前单seed数值门通过，`locked_test_accessed=false`；跨seed验收仍未完成。
- [x] PowerPoint 原生重开通过，总页数215；新增页中文为楷体、英文为 Times New Roman、最小正文18磅，字体或页面边界违规数为0。
- [x] 第1–203页的203个 slide XML 与203个关系文件已和追加前备份逐字节核对，差异数为0；验收记录位于 `meeting/2026-09-09-PI-JWM组会汇报PPT验收.json`。
- 单一下一动作：继续完成首seed正式训练；PPT只在最终checkpoint或跨seed状态发生实质变化时更新结果页。

## 2026-09-07 组会PPT层级纠正

- [x] 根据用户反馈撤销第207–215页的卡片式主导布局，重新以第205页正文占位符的原生段落层级制作；第206页封面继续复用第204页。
- [x] 每个正文页固定保留11个原生段落：红色方框大标题、蓝色菱形小标题、4条浅蓝箭头正文、第二个蓝色菱形小标题、4条浅蓝箭头正文。
- [x] 双图统一改为“模块一：严格物理—信息耦合双图编码”，实体级RSSM统一改为“模块二：世界模型”，删除“方法一/方法二”的错误并列关系。
- [x] 重新验收第1–205页共410个slide XML/关系文件，差异数为0；新增页最小字号18磅、中文楷体、英文Times New Roman，层级、字体和溢出违规数均为0。
- [x] 验收记录已升级为 `PI-JWM-meeting-deck-append-acceptance-v2`，上一版追加页由本版替代。
- 单一下一动作：保持当前正式训练不变；只有最终checkpoint或跨seed证据变化时再定点更新结果页。

## 2026-09-07 组会PPT层级—图表折中版

- [x] 在不丢失原生层级的前提下恢复必要图表：每个正文页保留红方框大标题、两组蓝菱形小标题及其箭头正文，图表从正文层级下方开始。
- [x] 恢复本月流程图、字段合同图、数据划分与因果窗口、严格双图、逐步递推、全局广播瓶颈、实体级RSSM、两阶段训练、结果表和滚动规划链。
- [x] 结果页改用原生PowerPoint表格呈现门控数值；模型机制页使用原生形状和连接线，不以图片替代可编辑内容。
- [x] 第1–205页410个页面部件逐字节一致；新增文字、图中文字和表格最小18磅，层级、字体、越界和溢出违规数均为0。
- 单一下一动作：保持当前训练协议，等待正式checkpoint或跨seed证据更新结果页。

## 2026-09-07 组会PPT重点强化版

- [x] 按用户指定仅保留第207、210、212–215页的流程图、机制图或结果表，删除第208、209、211页的图形内容。
- [x] 纯文字页恢复11段原生层级，图表页保留7段原生层级；叙事改为“问题—数据—关系编码—时间动力学—验证—结果—后续决策”。
- [x] 移除轨迹划分数量等低价值细节，增强方法适配性、实现工作和当前证据的说明。
- [x] 第1–205页410个页面部件逐字节一致；新增页字体、字号、层级、溢出、重开和渲染验收通过。
- 单一下一动作：继续当前正式训练，只在最终checkpoint或跨seed状态发生实质变化时更新第215页。

## 2026-09-07 当前双图四类对象实现复核

- [x] 物理节点为车辆、UAV、RSU、边缘服务器和云节点，主状态为 7 维；物理边为有向通信链路，主状态为 `distance/csi_mean/rate_sum/active_task_count/allocated_rb_count` 5 维。
- [x] 信息节点为每个活动物理节点附着的通信/计算代理；当前没有独立观测张量，agent encoder 复用对应物理节点的 7 维历史状态形成代理 latent。
- [x] 信息边由 `information_edges` 中的任务输入流、结果回传流和显式依赖数据流构成，在正式 tensor 中以 `flow_state` 表示，包含 `total_data/remaining_data/delivered_cumulative/delivered_this_slot/age` 5 维，并保留端点、类型、任务映射和逐时隙物理承载关系。
- [x] 实体级 RSSM 为 node、physical_edge、flow、task 维护随机状态；agent 只在基础双图中作为确定性消息传递 latent，没有独立随机状态或直接监督头。
- 结论：当前 5 维链路指标应称为物理边特征；当前信息边是数据流，不能用是否存在 `information_edge_features` 键判断信息边是否实现。
- 单一下一动作：保持当前运行和证据不变；PPT 按“物理节点—物理通信边、信息代理—数据流信息边、两类跨图映射”解释当前实现。
## 2026-09-08 P4 实体级 RSSM 首个正式 seed 完成

- seed `20260831` 已完成 base 20 epoch + RSSM 40 epoch；按冻结的 `p4_gate_aware_v1` 规则选择 epoch 39。
- 本地正式产物 79 项 manifest 大小与 SHA-256 零差异；checkpoint 方法身份、实体级 prior/posterior 语义和 strict reload 通过。
- 从原始 validation/calibration metrics 独立重算 9 项单 seed 数值门全部通过，最大浮点差 `3.37e-09`；`locked_test_accessed=false`、`formal_performance_claim_ready=false`。
- seed `20260830` 已按同一冻结配置启动；P4 尚未关闭，P6 不开放。当前单一下一动作：监控并验收 seed `20260830`，通过后才运行 `20260832`。
## 2026-09-08 seed 20260830 后暂停边界

- [x] 用户明确要求当前 seed `20260830` 完成并验收后暂停。
- [ ] 不得自动启动 seed `20260832`；后续 seed 必须等待用户新的明确指令。
- 当前门：P4 仍等待跨 seed 验收，P6 不开放，`locked_test` 继续封存。
- 单一下一动作：保持冻结配置完成并验收 seed `20260830`，随后停止。

- 监控汇报格式已锁定：每个新 RSSM epoch 必须报告 9 项门状态、NLL、link-F1 增量、node-x 各 horizon ratio、三个运营指标、证据回传和 `locked_test` 边界。

## 2026-09-08 组会 PPT 实现一致性复核

- [x] 只读复核第 204–210 页与正式 tensor、双图编码、实体级 RSSM、首 seed 独立验收记录的一致性；未修改 PPT、训练进程或实验产物。
- [ ] 汇报前修正逐 RB 标签用途、旧运动字段口径、实体级 RSSM 过度结论和封存测试边界，并补充链路 persistence residual 与 h5/h10 结果。
- 当前门：首 seed `20260831` 通过，seed `20260830` 仍在冻结配置训练，`20260832` 不得自动启动；`locked_test` 继续封存。
- 单一下一动作：用户按复核清单修订第 205–210 页，训练侧继续等待并验收 seed `20260830`。

## 2026-09-08 项目知识入口重构第一阶段

- [x] 在不触碰远端训练、同步目录、checkpoint、tensor、协议和原始 artifact 的前提下完成只读盘点。
- [x] 新增 `docs/PROJECT_INDEX.md`、`ARCHITECTURE.md`、`RESEARCH_STATUS.md`、`EXPERIMENT_INDEX.md`、`RESULTS_INDEX.md`、`PROJECT_RESTRUCTURE_PLAN.md` 和 `CHANGELOG.md`。
- [x] 在 `README.md`、`docs/README.md` 和 `AGENTS.md` 增加导航及持续维护规则；未改变理论、模型、loss、指标或实验配置。
- [x] 发现当前 PPT 与旧验收 JSON 的页数和 SHA-256 不一致，已记录为待重新验收问题，未擅自改写 PPT。
- 当前门：P4 仍等待 seed `20260830` 的完成与独立验收；P6 不开放，`locked_test` 继续封存。
- 单一下一动作：保持当前训练和同步保护，待 seed `20260830` 完成后再进行机器化实验映射细化。
## 2026-09-09 组会追问高密度问答文档

- [x] 对照当前 tensor、双图/RSSM/规则层代码、冻结训练配置、第一性原理审计和首种子独立验收，生成 `meeting/2026-09-09-PI-JWM组会追问高密度问答.md`。
- [x] 文档共 56 个问答、19,164 字符；明确区分当前实现、已验证结果、目标闭环和图语义冲突，证据路径全部存在。
- [x] 未修改模型、数据、训练配置、checkpoint 或远端进程；`locked_test` 边界不变。
- 当前研究下一动作保持不变：完成并独立验收 seed `20260830` 后暂停，不自动启动 `20260832`。

## 2026-09-09 seed 20260830 完成并暂停

- [x] seed `20260830` 完成 base `20` + RSSM `40` epoch，冻结选择器选择 RSSM epoch `40`。
- [x] 正式产物已完整回传；run manifest `79` 项零缺失、零哈希差异，checkpoint strict reload 缺键/多键=`0/0`。
- [x] 从原始 validation/calibration metrics 独立重算 9 项单 seed 数值门，最大浮点差=`4.16e-09`，全部通过。
- [x] 远端训练进程已退出，GPU 显存占用=`0 MiB`；当前 GPU 可以释放。
- [x] 按用户边界暂停，不启动 seed `20260832`。P4 仍等待第三个 seed 和三 seed独立审计；P6、`locked_test`和正式性能声明继续关闭。
- 单一下一动作：等待用户是否明确授权运行最后一个冻结 seed `20260832`。

## 2026-09-09 全项目知识与工程结构重构

- [x] 用户明确调整执行优先级：seed `20260832`、远端训练和远端同步全部延后；先完成既定重构计划中尚未完成的部分，并保留后续续接接口。
- [x] 阶段 1 已复核：八个稳定知识入口已经存在，当前两枚 seed 状态和 P4/P6/`locked_test` 边界已经更新。
- [x] 阶段 2A：生成 tracked 文件清单、SHA-256、职责和生命周期状态。
- [x] 阶段 2B：生成 Python 模块依赖、脚本入口和测试覆盖映射，区分当前正式、支撑、原型和历史代码。
- [x] 阶段 2C：生成 artifact/实验/结果注册表，连接研究问题、方法、配置、数据、checkpoint、结果和 audit。
- [x] 阶段 2D：建立文档权威关系、已知冲突和用户重新学习项目的阅读路径。
- [x] 阶段 3：已依据依赖图和测试映射完成候选审查与逻辑归档；物理迁移因回归基线和 provenance 风险按停止门暂停。
- [x] 阶段 4：提供可重复生成和 `--check` 验证入口，并将维护义务接入测试与文档。
- [x] 延后接口：机器可读登记 seed `20260832` 和远端同步为 `deferred/user_authorization_required`，复用现有冻结 runner，不创建自动启动链路。
- 当前科研门：P4 仍缺第三 seed 和三 seed 审计；P6 不开放；`locked_test_accessed=false`、`formal_performance_claim_ready=false`。
- 当前重构阻塞：第二轮重构后全量基线为 1636 项中 0 failure/21 errors，包含缺少 AirFogSim `traci` 环境、历史 fixture 与当前严格 RB 合同冲突、历史 artifact 读权限和两个无有效 calibration link 的旧测试；物理迁移必须先消除或隔离这些风险。
- [x] 阶段 2A–2D、阶段 4 维护机制和延后接口已经完成；阶段 3 的逻辑归档已完成，物理迁移按停止门暂停。
- 单一下一动作：保持第三 seed、远端同步和物理归档停止；等待用户选择下一项明确授权的科研或迁移任务。

## 2026-09-09 项目长期协作与快速问答闭环

- [x] 用户确认继续第二轮无损重构，以后续协作和提问时更容易找到正确信息为最终标准。
- [x] 固化 Human-driven / AI-accelerated 的职责、独立质疑、通俗解释和证据表达规则。
- [x] 将重要实验登记升级为统一完整字段；所有未知/不适用项显式记录原因。
- [x] 建立重要历史方法语义注册表，连接尝试动机、结果、弃用原因、替代关系和原始证据。
- [x] 自动核对结果注册表与正式 acceptance JSON 的 seed、epoch、指标、状态和封存边界。
- [x] 建立并测试面向当前方法、历史尝试、结果来源和延期任务的只读问答检索入口。
- [x] 更新全部导航、权威记录和维护检查，并运行定向/正式/全量回归。
- 当前科研门：P4 仍缺 seed `20260832` 和三 seed 审计；P6 不开放；`locked_test_accessed=false`、`formal_performance_claim_ready=false`。
- 当前重构阻塞：物理归档仍受全量 21 个既有环境/fixture/权限错误阻断，但不阻止本阶段的增量文档、注册表和只读检索实现。
- 验收结果：统一生成器与 `--check` 通过；项目知识/结构 27/27、正式 P4 210/210、compileall 与 `git diff --check` 通过；全量 1636 项为 0 failure/21 个已登记环境或历史错误。
- 单一下一动作：从用户的下一个实际项目问题开始按新入口检索、回到原始证据核实并维护索引。

## 2026-09-10 ChatGPT–Codex 长期协作与 AI_CONTEXT

- [x] 读取用户的新长期协作要求，并复核当前 P4、第三 seed、GPU、同步和 `locked_test` 停止边界。
- [x] 只读检查 Git、现有知识注册表、当前模型/训练入口、冻结协议、正式 acceptance 和文件证据分层。
- [x] 先建立失败契约测试，固定 `AI_CONTEXT/` 九个文件、事实来源、冲突模板、维护检查和永久协作规则。
- [x] 创建并填写 `AI_CONTEXT/00_PROJECT_STATE.md` 至 `08_CHANGELOG.md`，只记录可由源码、配置、实验或用户决定支持的内容。
- [x] 按用户本次明确授权重构 `AGENTS.md`：Research Engineer 角色、三方边界、Context Consistency Check、Git 提交推送、私人笔记禁区和冲突处理。
- [x] 将 `AI_CONTEXT/` 接入文档权威注册、自然语言路由、项目导航、索引生成器和自动检查。
- [x] 执行 Context Consistency Check、定向/正式/全量测试、编译、索引防漂移和 Git 差异检查。
- [x] 验证达到合理完成状态后按 Conventional Commits 提交并推送 `main`；本次新增范围验证通过，未把全量既有错误误报为全绿。
- 当前科研门：P4 仍缺 seed `20260832` 和三 seed 审计；P6 不开放；`locked_test_accessed=false`、`formal_performance_claim_ready=false`。
- 当前工程阻塞：全量套件已有 21 个环境/历史错误，阻止物理迁移旧文件，但不阻止增量 `AI_CONTEXT/` 与治理层实现。
- 验证结果：AI_CONTEXT 6/6、项目知识/结构 28/28、正式 P4 210/210 通过；全量 1643 项为 0 assertion failure/17 个已登记环境或历史错误；当前 artifact 读取错误为 0。
- 单一下一动作：本任务提交推送后，科研主线仍等待研究者决定是否授权 seed `20260832`。

## 2026-09-16 最新 AirFogSim 源轨迹分享包（仅本地交付）

- 用户明确要求最新版，交付 v6 formal source（B层），不含六月A层CSV、训练npz/checkpoint或locked_test；没有重跑仿真。
- 源：code/artifacts/formal_data/pi_jwm_v4_formal_candidate_v6_rb_v1_unlocked_20260821；54条、6场景、每条300步、0.1秒、合计16200轨迹时点。
- 完成：code/artifacts/packages/AirFogSim_raw_data_share_20260916.zip；658389567字节；SHA256=e2b73fc877a3c5117767a20e23c2527281876c0036e42be16835fb99a08fcb59。
- 验证：486项轨迹文件及8项顶层历史清单匹配；54个时间网格通过；ZIP全部800文件逐项解压读取、大小和SHA256通过；所选数据零缺失。
- 限制：历史完整config哈希重建0/54匹配；生成时项目/AirFogSim commit无法确认；最新v6源与当前tensor摘要所指v4不是同一来源声明。现有场景不构成严格单变量对照。上述差异仅报告，没有改写源证据。
- Context Consistency Check只读完成；按用户要求未修改AI_CONTEXT、模型或训练代码；本任务不commit/push。科研P4、第三seed、GPU、同步和locked_test边界不变。
- 本次单一下一动作：用户直接将ZIP交给同学；若需逐值重现历史仿真，先恢复完整历史配置和版本，不能自行重跑。

## 2026-09-18 STEP 1 — New Definition → Current Implementation Audit

- 当前门：只执行新 `00–06` 定义与现有实现审计；旧 P4/P6/P0–P10 为 Historical / Archived。
- 已完成：只读定义指纹、源码/config/test/artifact 对照、复用分类、Tracker、实施记录框架和 Step 1 主报告。
- 阻塞：严格双图、四类动作、RSSM 动态边界、逐步规则反馈与完整在线闭环尚未实现；通信状态充分性、外生事件和 planner objective 等仍需研究者决定。
- 本 Step 不修改模型/数据/loss/planner/checkpoint，不训练，不使用 GPU，不访问 `locked_test`，不自动执行 Step 2。
- 唯一建议下一步：研究者审阅后，单独授权冻结一个决策步的原始轨迹字段与 Route/Comm/Comp/UAV 四类动作映射合同。

## 2026-09-18 STEP 2

- 状态：已完成合同冻结和低成本最小闭环验证，等待研究者检查。
- 范围：单决策步 Raw Trajectory；Route/Comm/Comp/UAV Mobility 四类 action；Execution/Outcome/下一 Decision 对齐。
- 证据：`docs/implementation_records/STEP_02_RAW_TRAJECTORY_ACTION_CONTRACT.md` 与 `code/artifacts/protocols/pi_jwm_raw_single_decision_step_contract_v1_20260918/`。
- 边界：未修改世界模型、双图、RSSM、Loss、Planner；未训练、未用 GPU、未访问 `locked_test`。
- 唯一下一步建议：单条非 locked 真实轨迹四类动作采集接线和 Outcome 对齐验收，需研究者另行检查后推进。
# 2026-09-19 Step 2.1

- [x] 真实 AirFogSim 单轨迹四类动作接线与 Outcome/next Decision 对齐
- [x] 修正 vehicle degree/UAV rad heading 合同，核对测试数量和 generated registry 临时文件
- [x] 写入 Step 2.1 记录、Tracker、计划/进度、AI_CONTEXT，待验证后 commit + push
- [ ] 研究者审阅后再授权跨决策步真实反馈闭环

## 2026-09-19 STEP 3.1F

- [x] 完成 History `O_{t-H+1:t}+A_{t-H+1:t-1}+Y_{t-H+1:t-1}`、History union index、past relation/DAG/flow 对齐和未来 reference observation audit。
- [x] 验收最小真实样本、late-entry/disappearing fixture、9 项 focused tests、23 项 sample checks、round-trip、compileall和 index check。
- [ ] 研究者审阅后再决定是否授权 STEP 3.2；本次不自动进入。

## 2026-09-19 Step 2.2

- 当前门：只验收一条真实、非 locked、6 个连续决策步的 Raw Trajectory；不进入 Dataset/Tensor、双图、World Model、Loss、Planner 或训练。
- [x] 每一轮从真实环境独立采集 Decision，禁止复用或 copy 上一 Outcome。
- [x] 验证 trajectory/frame/time、entity/task ID、lifecycle、四类动作与 empty/no-op 语义的跨步连续性。
- [x] 对比 AirFogSim acceleration 与速度有限差分，只记录实现语义，不修改仿真器或决定 Dataset 字段。
- [x] Step 2.1 v4 和 Step 2.2 v2 必要 JSON/manifest 已强制加入 Git 暂存，且 manifest SHA 与暂存 blob 一致。
- 唯一下一动作：完成最终验证、commit、push 后停止等待研究者审阅。

## 2026-09-19 Step 2.3

- 当前门：只完成 Raw Contract 因果和字段完整性收尾，不进入 Dataset/Tensor builder 或后续模型链。
- [x] 未来任务从 `O_t`、History 和 input-side Entity Index 隔离，保留 internal metadata。
- [x] 真实验收 Decision CSI/CPU capacity 与 Outcome delivered data/served CPU work。
- [x] 真实调用 return route，并冻结 raw/canonical acceleration 与缺历史 mask。
- [x] 17 项真实 checks 全部通过；Raw Trajectory Layer / 01 标记 COMPLETE / FROZEN。
- 唯一下一动作：完成记录、索引、最终验证、commit、push 后停止，等待研究者授权 Step 3。

## 2026-09-19 Step 2.4

- 当前门：只完成 Communication Outcome 语义收尾，不进入 Dataset/Tensor、双图、World Model、Loss、Planner 或训练。
- [x] 核对真实 wired 路径：`WiredNetworkManager.step` 返回逐 task slot service，随后 `Task.transmit_to_Node` 推进任务。
- [x] Outcome 拆分为 wireless/wired/total，并冻结 empty map 与 missing mask 的区别。
- [x] 真实 6 slot 非 locked 轨迹观察到 wireless→wired 两跳、task progress/lifecycle 和 14 项 checks 全通过。
- [x] 同步源码、测试、Raw Contract、机器证据、Tracker、authority records、AI_CONTEXT 和索引。
- 唯一下一动作：最终验证、commit、push 后停止，等待研究者单独授权 Step 3。

## 2026-09-19 STEP 3.1F-PATCH

- Scope: only Future Action index namespace, input-index policy wording, and future-reference audit provenance; no other research module.
- Current gate: patch validation before Batch Dataset; `locked_test=false`, `training=false`, `gpu=false`.
- Completed: anchor visibility separated from History-union numeric index; ID↔index validator and disappearing-object regression fixture added; sample/manifest/audit JSON refreshed.
- Next: researcher review; only then consider STEP 3.2 authorization.

## 2026-09-19 STEP 4.1 — Physical / Information Object–Field–Relation Mapping Freeze

- 当前门：只冻结定义 03 的对象、字段与关系映射；不实现 graph builder、encoder、GNN、message passing、GRU、跨图 coupling、World Model、Loss、Planner 或训练。
- [x] 读取定义 03、当前 Raw / Sample / Tensor 合同和旧双图代码，确认实际字段来源与旧实现冲突。
- [x] 先增加机器映射合同的 focused failure tests，再实现最小 schema、validator、artifact builder。
- [x] 形成 Current Data → Graph Role、Required Additive Data Extension、Forbidden Placement、Old Implementation Reuse/Conflict 四张机器表及可读文档。
- [x] 更新实施记录、Tracker、authority records、AI_CONTEXT、知识索引和过程记录。
- [x] 完成 focused tests、相关回归、compileall、知识索引 write/check、`git diff --check`；待 commit + push。
- 边界：`formal_dataset=false`、`training=false`、`gpu=false`、`locked_test=false`；任何定义 03 最小必需信息不足都记录为 gap，不在本 Step 伪造或实现图。
- 唯一下一动作：完成 STEP 4.1 映射证据并提交，等待研究者审阅；不自动执行 STEP 4.2。

## 2026-09-19 STEP 4.1-PATCH — Minimum Gap Semantic Correction

- 当前门：只修正 wired minimum relation 与 CPU capability/resource 分类；不重做 mapping，不修改 frozen 02，不实现 graph builder/GNN/model/training。
- [x] 核实 WiredNetworkManager topology/`hasLink`、Raw `environment.wired_edges`、FogProfile CPU 配置/使用路径和真实跨帧数值。
- [x] 先用 focused tests 固定：wired relation 与 optional numeric state 分离、no-CSI type+mask、CPU 四类语义分离、negative tamper 使顶层 receipt 失败。
- [x] 最小修改 mapping/validator/文档/artifact，并同步原 Step 4.1 record、Tracker、authority/process records 与必要 AI_CONTEXT。
- [x] 重跑 focused/Step 3.3 regression、deterministic rebuild/hash、compileall、knowledge index write/check、diff checks；待 commit + push 后停止。
- 边界：`graph_builder_implemented=false`、`training=false`、`gpu=false`、`locked_test=false`。
- 唯一下一动作：完成本 Patch；通过后 STEP 4.1 正式 COMPLETE / FROZEN，等待研究者另行授权最小 Data Contract Additive Extension。
# 2026-09-20 STEP 4.2A — Existing-Source Graph Input Additive Extension

- 当前门：只将 STEP 4.1 已确认 Raw/Simulator 已有或可因果推导的 graph minimum inputs，以新版本 additive contract 贯穿 Raw → Sample → Dataset/Normalization → Tensor；不实现 graph builder。
- [x] 恢复 `main@db000291563eb23ac8a0a577cc35fd84bb7b95c3`，工作树 clean；读取 02/03 只读定义、Step 3.1F/3.2/3.3、Step 4.1 mapping/receipt、Tracker 与 AI_CONTEXT。
- [x] 核实既有三条 development trajectory 的 position、wireless CSI、CPU capacity、Task current fields、wired topology 和 stable index 来源。
- [x] TDD 红灯固定 Raw wired decision rows、Sample typed Comm/static capability/Task-Agent、train-only normalization、Tensor shape/index/mask 与 12 项因果反事实。
- [x] 实现独立版本化 additive extension 与 builder，不覆盖旧 artifact，不改变 frozen Step 3 语义。
- [x] 生成最小 development artifact、Step 4.2A receipt/hash/provenance，回写 Step 4.1 resolved/blocked 状态但保留历史事实。
- [x] 运行 focused、Step 4.1/3.3/3.2/Raw 回归、deterministic rebuild、round-trip、compileall、knowledge index write/check、diff checks；待 commit + push `main` 后停止。
- 阻塞保持：stable stateful Flow total/rem/type/endpoints、return size/priority/deadline、dynamic available CPU、storage、wired queue/load/utilization；任何非零 unresolved future reference 仍为 `RESEARCHER_DECISION_REQUIRED`。
- 范围：`graph_builder=false`、`physical_topology=false`、`training=false`、`gpu=false`、`locked_test=false`、`formal_dataset=false`。
- 唯一下一动作：完成 STEP 4.2A；之后只建议审阅仍 Raw-insufficient 的 minimum graph gaps，特别是 stable stateful Flow。
