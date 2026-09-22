# Findings

## 2026-09-22 STEP 5.1B-PATCH

- 原实现把 `[B,L,S,F]` target 沿 horizon 求和为 `[B,S,D]`，导致 future posterior 可读取其他 horizon；Patch 保留 horizon，并验证篡改 horizon 2 不改变 horizon 1 embedding/q。
- 原 receipt 使用零 h、target==prediction 和硬编码 checks；Patch 真实调用 STEP 4.4 current latent/dynamics/priors/decoders，所有 required checks 由计算结果产生。
- 当前 5.1A target support 为 10 physical / 74 communication，STEP 4.4 development model support 为 8 / 44；receipt 使用显式固定支持子集，完整支持对齐仍是 blocker，不能因此宣称 COMPLETE/FROZEN。

## 2026-09-21 STEP 5.1A-PATCH

- Root cause 1: `extend_future_motion_csi_targets()` kept `history[-1]` as the position reference for every horizon, while STEP 4.4 recursively applies `state.position + vehicle_motion[..., :3]`; this produced cumulative anchor-to-future targets for horizon 2+ instead of local one-step targets.
- Root cause 2: Motion rows were enumerated from each future frame, and tensor rows were written by enumeration order. That made supervision depend on future row order/target index and allowed disappearance or future-only birth to change model slots.
- CSI values were already looked up by relation ID and RB ID, but the former receipt did not bind those rows to the actual current tensor relation slots used by STEP 4.3A/4.4. The patch adds and verifies the full slot/identity/endpoint/type/RB chain.
- The old STEP 5.1A receipt therefore did not prove multi-horizon Motion correctness or model-slot equality. The patched receipt requires all local-step, fixed-slot, current-model identity, normalization/mask, deterministic, real-development and scope checks to be true.

## 2026-09-21 STEP 5.1A

- Future Motion 的可核验实现是 Vehicle `[delta_x, delta_y, delta_z, next_speed]`；position delta 只除以冻结 position std，不减 position mean。
- Future CSI 必须由 target frame 对应 outcome 的 `channel_rows` 提取，不能读取 History CSI，也不能用 rate/service/outage 替代；输出只覆盖当前支持关系。
- 12 个真实 non-locked development samples 通过 focused contract、tensor round-trip、篡改检测和 receipt AND；这不是正式 Dataset、训练或性能结果。

## 2026-09-21 STEP 5.0 — Definition 05 audit

- The read-only Definition 05 note (SHA-256 `62eebb05e2eeefe9edb0038f12964f59915a7acce03a683d6c7e5c71f85684e9`) still specifies observation NLL, Event/Residual learning, and overshooting. The researcher's newer explicit ten decisions resolve this conflict: deterministic mean decoder + Motion/CSI MSE + family KL, Event/Residual heads absent in v1, and overshooting OFF.
- Current STEP 4.2A normalized samples contain future entity `position_m` only as raw value/mask/unit; it is neither normalized nor collated into the frozen tensor. `target_entity_features` contains only speed.
- Current target/sample/tensor namespace has no future per-RB CSI. History CSI and future communication service are not valid substitutes for CSI target.
- Current STEP 4.4 state adapter uses raw position while Comm CSI comes through the normalized graph path. STEP 5.1 needs an explicit normalized-loss/raw-rule bridge before training.
- Historical `_masked_mean`, analytic diagonal KL, horizon accumulation, seeding, AdamW, logging, checkpoint reload and manifest patterns are locally reusable. Old total loss, full-target posterior, overshooting tensors, staged encoder freezing, P4 gate selector, historical checkpoints and result numbers are not current Definition 05 evidence.
- No Definition 05 runtime was executed. Training remains blocked by target/data/loss implementation, not by GPU availability.
- Optional full-suite audit ran 1849 tests with 33 errors outside this documentation-only diff: Windows GBK output failure in AirFogSim import, absent historical artifact files, old teacher-tensor fixtures rejected by current RB-action validation, and clean-tree assumptions. Relevant Definition 05/STEP 4.4 audit regression remains 75/75 pass; do not report the whole suite as passing.

## 2026-09-21 STEP 4.4-PATCH3

- Frozen Task History features expose work/computed/transmitted/elapsed only; `return_size` is explicitly unavailable to the frozen Tensor/Graph contract. Therefore current-side no Return slot is epistemically unknown, not known-no-Return.
- The previous transition ignored `task_return_requirement_known`, so computation-finished unknown tasks could be falsely completed. PATCH3 adds a distinct unresolved side-state and keeps known-required/no-slot separate through `return_birth_required`.
- Previous DAG release counted invalid edges as predecessors. PATCH3 masks by `dag_validity` and machine-tests root, incomplete, completed, invalid, and multiple-predecessor cases.
- Previous terminal completion did not change `flow_status_index`. PATCH3 uses `FLOW_STATUS_VOCAB.index("COMPLETED")`; partial and intermediate-hop counterfactuals remain active.
- Builder acceptance now observes state changes rather than relying on field/module/policy-string existence. Focused 30/30, related regressions 82/82, formal receipt 92/92, compileall, and six-file independent hash/size equality pass; STEP 4.4 is COMPLETE / FROZEN.

## 2026-09-21 STEP 4.4-PATCH2

- 4.3A 已真实保留 Flow `task_index` 与 `flow_type_index`；此前 World Model adapter 丢弃前者并把所有 `return_flow_index` 初始化为 `-1`，这是已有 Return 无法绑定的直接根因。
- Return gate 必须由 typed structural identity 建立，不能依赖 slot 位置；完成的 Input Flow 不能替代 Return Flow。
- frozen Task tensor 没有 `return_size`，因此 future-only Return birth 不能从当前 support 可靠推断或创建；v1 必须显式声明不支持，并让外部已知的 `task_requires_return` side-state在缺少 slot 时阻止 final completion。
- Definition 05 应把跨越 unsupported Return birth 的 window mask/exclude/分类，不能把对象支持缺失当作普通预测误差。

## 2026-09-20 STEP 4.2A-PATCH

- `_physical_structure()` 先建立 V/U/I directed structural relation，再读取 CSI；`observed_mask=false` 只证明 CSI feature 缺失，不证明 relation 不存在。
- relation mask 与 feature mask 必须分层；否则 node-index 暂不可用会错误删除 Information Graph 的结构边。
- `_extract_tasks()` 已提供 return size、deadline、priority、task delay 的 simulator observer source；冻结 Raw 未透传不能写成 simulator 无可靠来源。
- 旧 `LogicalFlow`/`CarryingHop` 类型名和 past hop service 仍不足以证明定义 03 stateful Flow 的 identity、total/rem、multi-hop 与 route-revision 语义。

## 2026-09-20 STEP 4.2A

- wired topology 可以在动作执行前物化为 directed typed relation；CSI 缺失由 type + mask 表示，不能借用 wired service outcome。
- CPU capacity 应作为无 H 轴的静态 Agent capability；Comp allocation 和 actual service 的反事实变化都不改变该张量。
- Task progress/elapsed 与 Src/Host/Exec/Ret 可以从当前 Decision 因果构造；Future Route target 不得倒灌到当前关系。
- 已有来源字段贯穿 Tensor 后，stable stateful Flow 仍是独立的 Raw-insufficient blocker；不能用 past hop service 改名填补。

## 2026-09-19 STEP 4.1

- 当前数据缺口分层处理：position/无线 CSI/CPU static capacity 等是 Raw 已有但未暴露；wired 最小 relation 有 topology/`hasLink` 来源但未逐 Decision 物化；wired 可选动态 numeric state 与完整 stable stateful Flow 才是 Raw 本身不足，不能用同一种 patch 处理。
- `entity.getFogProfile()['cpu']` 在当前配置与真实轨迹中表示稳定的节点能力上限；它不能替代 `A_t^Comp` allocation、Outcome actual service 或尚无来源的 dynamic available CPU。
- `past_outcome_flow_service` 是过去某一 hop 的实际服务结果，不具备 current Flow 的 total/rem 语义；旧 `FLOW_FEATURES` 名称也不是当前数据证据。
- cloud 的 `[0,0,0]` 坐标不能证明它有独立空间建模意义；edge/cloud 的 Physical membership 必须由研究者决定。
- 旧 `EDGE_FEATURES` 把 distance、CSI、rate、active task 和 RB 混在一个 physical edge，和定义 03 冲突；仅通用 stable ID/mask 与无语义的 masked-index 算子可直接复用。

## 2026-09-19 STEP 3.3F

- fixed-shape 通过不代表语义完整；Past Outcome 和 Target future facts 必须作为独立 tensor 角色保存。
- category code 不能由 development batch 中“碰巧出现的值”决定；固定 vocab 才能保证 subset/reorder 稳定。
- past hop service event 不等于定义 03 的 current stateful Flow total/rem state；Gap Table 已明确区分。
- 实际 artifact 有 past offload Route，`max_route_hops=2`；旧记录中的“无 Route/max=0”与机器证据冲突，已纠正。

## 2026-09-19 STEP 3.3

- Tensor collation 必须使用 JSON sample 的 stable ID/index，不能按每帧可见集合重新编号。
- `presence=false` 与 `feature_mask=false` 均独立于 numeric value；padding 不进入有效 feature。
- target-only object 需要独立 target capacity；当前 development capacity 不是正式研究容量。
- 当前 batch 没有 route entry，因此 route hop capacity 为观测值 0；后续正式容量需研究者单独冻结。

## 2026-09-19 STEP 3.2 findings

- Existing Step 2.x Raw version directories were one seed-0 trajectory lineage, not independent split members; treating versions as trajectories would leak provenance.
- Fresh seed 1/2 non-locked collector runs passed all Raw checks without schema or simulator changes.
- A mask test exposed that object-level `feature_mask[field]=false` must override the field wrapper mask; the implementation now combines both masks and ignores masked extremes.
- The resulting bundle is observation-only validation evidence, not a formal Dataset or generalization result.

## 2026-09-19 STEP 3.1

- 新定义 `02数据集构建与模型输入.md` 要求已落实为最小合同；input-side index 不使用未来窗口 union，未来新对象在 target-side index 表达。
- STEP 3.1R 更正：真实 observer 已提供 DAG dependency rows；Raw v2 的 Decision 保存 2 条当前可见边，30 条含未来端点的边只在 internal metadata，model-ready input Static 不读取未来端点身份。
- Step 2.4 窗口的 target frames 没有 transfer event，故 flow rows 为 0；这是真实证据覆盖限制，不伪造服务记录。正式 batch 需要后续真实有服务窗口验收。

## 2026-08-26 双约束全仓审计与计划重构

### Teacher requirement (verbatim)

> “信息边特征可以不用这么多，就是可以考虑在消息不是那么完整的情况下面做预测；然后真实数据集可能没办法涵盖所有的内容，每一个真实的数据集可能各有侧重，所以可以考虑用不同的数据集对不同的方面进行微调训练增强；然后就是方法，不能只通过实验结果来说明每个模块选择哪个方法，而是要说明为什么这个方法适合我们的课题场景；然后就是不同的方法要针对我们的场景有调参这个过程”

### Audit scope and evidence

- 仓库清单按 `rg --files`/PowerShell inventory 复核，排除 `.git`、`.worktrees` 后约 725 个文件；文本/代码/配置/记录/JSON/JSONL 约 710 个、约 16.996 MB。核心目录 `code/src/pi_jwm`、`code/scripts`、`code/tests`、`记录`、`meeting`、`docs`、`literature`、`paper` 已逐目录核对；第三方 `code/reference/AirFogSim`、PDF/PPT/PNG 等二进制或外部资料按路径、manifest、哈希和元数据处理，不把二进制当普通文本解析。
- 权威入口为 `AGENTS.md`、`记录/本地计划表.md`、`记录/PIJWM主文档.md`、`记录/8.12之后推进.md`、`记录/接续记录/新对话接续说明_20260815.md` 及 planning files。早期“GPU未批准/P2未完成”文字均已由顶部最新状态覆盖，不能直接作为当前判断。
- 当前可复用证据：P0/P1/P2 contract、非 locked v4 tensor、确定性规则层、三 seed CPU rule replay、规则启用 aggregate candidate 的非 locked GPU smoke/固定数据训练；当前 `formal_performance_claim_ready=false`。
- 当前关键阻塞：candidate-action planner code-only audit 为 `blocked`；R6 是 direct belief/state-conditioned scorer，`formal_candidate_rollout_planner_v1.py` 仅 `prototype_only`。缺少合法候选生成、共同 belief 下逐候选 world-model 调用、未来 state/task/cost/risk 提取、预测后果选择和执行反馈重规划。

### Plan decisions

- 保留 P0–P10 编号，但以 2026-08-26 v2 入口重新定义依赖关系；P3 先补齐三模块理论适配与多源数据角色，P4 闭合 world model，P5 闭合 planner，P6 才做公平场景化调参，P7 才做多源微调/扩大训练，P8 后才锁方法，P9 最后访问 locked-test 与 baseline。
- 老师四项要求被固化为 T1 部分信息、T2 多源数据分工、T3 方法理论适配、T4 场景化调参；项目五项约束固化为 C1 一致性、C2 contract、C3 规则递推、C4 candidate rollout 闭环、C5 locked-test 最后。
- 已有结果只按其真实边界复用：aggregate/non-locked/diagnostic，不升级为逐 RB 方法、正式 planner、最终性能或 locked-test 结果。
- machine-readable 双约束门矩阵已落盘：`记录/双约束门矩阵_20260826.json`，包含 T1–T4、C1–C5、P0–P10 当前状态和 GPU/locked-test 边界。

## 2026-08-26 P3 方法适配与多源数据方案

- P3 设计包已完成：三个模块均写明“问题特点、方法、为什么适合、已知不足、检查指标”；说明使用尽量简单的语言，避免把设计目标写成已经完成的能力。
- 多源数据只按真实覆盖范围分工为仿真主线、无线遥测、边缘任务、移动/信道和 held-out transfer；尚未完成 source closure 的真实数据不得进入正式训练。
- 公平调参规则已预注册，但实际 sweep 仍属于 P6；planner 仍是 `prototype_only`/`blocked`，没有打开 GPU 或 locked-test。
- 下一门是 P4 的 CPU/non-locked 世界模型检查，不重复 P0–P2 和既有 aggregate GPU 证据。

## 2026-08-26 P4 世界模型机制门收口

- 冻结的 h20 tensor contract 已验收：`code/artifacts/formal_tensor/pi_jwm_v4_tensor_v4_rule_contract_h20_unlocked_20260826`，history=8、horizon=20、54 个 unlocked seed、14,742 个窗口，统计量只来自 train split。
- CPU 1/5/20 horizon 机制审计通过：输出长度和 finite 性正确，动作扰动会改变预测，target 改写不会改变预测；报告为 `code/artifacts/audit/pi_jwm_formal_p4_horizon_mechanism_audit_20260826/p4_horizon_mechanism_audit.json`。
- 统一 P4 机制门通过：规则 replay、候选一致性、非 locked GPU 历史 artifact 边界和 horizon 接口检查均通过；报告为 `code/artifacts/audit/pi_jwm_p4_world_model_gate_h20_20260826/p4_world_model_gate.json`。
- 关键限制：上述 probe 使用的既有 checkpoint 训练 horizon=3，h20 只用于输入/输出接口机制审计，不是 h20 重训后的精度结果。因此 P4 只能标记为 `complete_for_mechanism_only`；正式 state/task/resource/uncertainty 精度门仍 pending。
- 当前边界保持：不启动新的 GPU、不访问 `locked_test`、不进入 P5/P6，`formal_performance_claim_ready=false`。

## Rule-layer hardening superseding findings (2026-08-23)

- The first interface closure was insufficient: label-time endpoint sourcing, action-occurrence stage completion, and full predicted-state feedback were invalid. These paths are removed or corrected.
- Learned heads predict physical edge rate and a bounded flow-service fraction. Deterministic rules convert that fraction to capped delivered data, write RB occupancy, allocate CPU by capped equal sharing, and enforce lifecycle/DAG conservation.
- CPU is not a policy action in the revised path. Logged `cpu/cpu_allocated/cpu_fraction` values are zeroed before action encoding and are not read by the deterministic CPU rule.
- Node CPU is not overwritten as a fake remaining-resource state. Capacity is a constraint; per-task allocation and served work are emitted separately.
- Rule feedback is the masked correction `rule_state - learned_state`, not the full state, preventing double injection of unconstrained node predictions.
- New tensor and training audits are current authority. Earlier rule-interface audit and pre-rule GPU checkpoints are historical only.

## Deterministic rule-layer resolution (2026-08-23)

- The four former interface blockers are closed at the code/data boundary: train-only stats are passed through model configuration; action source endpoints are explicit in each formal window; service outcomes are produced by dedicated learned service heads and consumed by the deterministic layer; deterministic masks are explicit static tensors.
- Rule-governed updates are physical-unit operations: edge rate/RB occupancy, node CPU availability, flow remaining/cumulative/slot delivery, task transmitted/computed fields, and DAG unfinished-parent/release state. Outputs are converted back to the training-normalized space before loss/feedback.
- A first backward pass on a real validation window exposed and fixed an autograd-breaking in-place write. The replacement uses functional feature reconstruction and `scatter_add`; real forward/backward now succeeds.
- This is an implementation/interface closure, not a performance result. The selected 2026-08-23 GPU runs predate the rule layer; consistency audit therefore reports `candidate_checkpoint_requires_rule_layer_retraining`, and no GPU/locked-test claim is authorized.
- Pre-retraining hardening found that the first window-level `future_source_node_index` implementation copied label-time `task_node_index`. That is future-state leakage even though the field is called an action input. Source endpoints must instead be tensorized directly from offload/return/RB/CPU action records.
- The authority document fixes the split as: deterministic action/endpoint write, RB counts, CPU inner rule and remaining-work conservation; learned dynamics supplies channel/effective service outcomes. Therefore a rule-enabled model must not consume logged future CPU allocations as a core action. Required new tensor fields are `task_action_source_node_index`, `flow_task_index`, and `slot_seconds`.
- Task progress must be service-based: input transmission, compute, and return progress are capped by their current remaining quantities and lifecycle eligibility. An offload or CPU indicator alone is not evidence that the whole task stage completed.

## Key-wise input perturbation diagnosis (2026-08-23)

- Audit: `code/artifacts/audit/pi_jwm_formal_tuning_keywise_robustness_cpu_20260823/`, same three seeds and validation samples, four isolated history keys, noise `0/0.05/0.10/0.20`.
- `node_state` alone produced mean node-x MAE deltas `+10.129/+20.972/+42.697`; `task_state` alone produced mean task-delay deltas `+0.188/+0.423/+0.897`. `physical_edge_state` and `flow_state` produced zero node-x/task-delay deltas to reported precision. Link-F1 changes were negligible (maximum about `0.001`).
- The residual path directly anchors predicted component means to the last observed state (`formal_dual_graph_world_model_v1.py:278-282, 412-414`), explaining the node-x channel; task-state enters the task-history encoder and explains the task-delay channel. This is a confirmed sensitivity mechanism, not a fix or final robustness claim.
- Execution remains CPU-only and non-locked: `evaluation_devices=["cpu"]`, `gpu_execution=false`, `locked_test_accessed=false`, `formal_performance_claim_ready=false`. Next is a theory-code consistency decision before any new GPU run or method freeze.

## Candidate theory-code consistency audit (2026-08-23)

- Machine report: `code/artifacts/audit/pi_jwm_formal_candidate_consistency_audit_20260823/candidate_consistency_audit.json`.
- Passed checks: aggregate-baseline boundary, residual configuration (`zero_init=true`, `residual_state_scale=0.5`), task-history conditioning, and future-action conditioning. The three selected runs share one protocol; only the independent `seed` differs. The per-RB sidecar remains diagnostic-only and is not required for the aggregate baseline.
- Critical mismatch: `per_step_deterministic_rule_update_missing`. The current model loop updates latent states and sends them directly through learned heads; the audited source does not call a deterministic rule layer before explicit-state prediction. This conflicts with the PI-JWM theory requirement that rule-governed fields be updated at every rollout step.
- Gate: `status=blocked`, `cpu_training_allowed=false`, `gpu_allowed=false`, `formal_performance_claim_allowed=false`, `locked_test_allowed=false`. Do not start another training run until the rule-layer resolution is recorded and verified.
- A real validation-window contract check confirms four missing inputs for a truthful rule layer: train-only normalization statistics are outside the model batch; future actions lack an explicit source endpoint mapping; future service outcomes (rate/delivered data/CPU allocation) are absent; and deterministic-target masks are absent. These are interface blockers, not tunable hyperparameters.

- The fixed-data three-seed gate is blocked only by validation node-x MAE ratio `1.3647403366171358`; activity and operational criteria pass.
- The training runner constructs `FormalLossWeights()` internally, so state-loss diagnosis requires an explicit runner parameter rather than an ad hoc artifact edit.
- The intended first intervention is a single-variable increase of `state_mae`; all other protocol and data fields remain unchanged.
- The override is now implemented in `run_formal_dual_graph_gpu_train_v1.py`, recorded in `config.json`, validated for non-negative values, and covered by the runner test suite (`7/7`).
- Increasing `state_mae` to `0.5` worsened the screen run (validation node-x MAE `5.9431`, link F1 `0.5943`).
- Enabling `zero_init_residual_state_heads` with the frozen default loss improved the full-budget seed `20260824` run: validation link F1 `0.7371` vs persistence `0.6879`, node-x MAE `1.8530` vs `1.4939` (ratio about `1.2407`), calibration link F1 `0.3003` vs `0.2109`. This is promising but only one seed.
- Explicit residual damping `residual_state_scale=0.5` with zero-init passed the frozen three-seed numerical gate: validation node-x MAE ratio `1.13897603387634`, mean validation link-F1 delta `0.06478173263384901`, mean calibration link-F1 delta `0.10730500845124441`, and all operational deltas non-positive.
- Hardware remains unavailable: `torch.cuda.is_available() == False`, device count `0`. Numerical eligibility does not authorize a GPU run on this machine.
- The supplied remote server has RTX 4090 and Conda PyTorch `2.8.0+cu128`; the isolated current-code/v3-tensor GPU smoke completed with `gpu_execution=true`, peak device memory `75049984` bytes, and zero manifest mismatches.
- Formal GPU seed `20260824` completed with the gate-passing configuration, train/validation/calibration `256/128/128`, peak device memory `201894912` bytes, train time about `66.77` seconds, and zero manifest mismatches. This is a non-locked single-seed training result, not a final performance claim.

- Controlled GPU tuning screening completed in a new isolated remote directory: 6 runs covering 4 modules x 2 learning rates x 3 seeds, with 24 candidate records and zero manifest mismatches across all runs.
- Only `coupled_dual_gnn_residual` at learning rate `3e-4` passed the frozen screening gates: mean validation link-F1 delta `+0.0647817326`, minimum per-seed delta `+0.0494443949`, mean calibration delta `+0.1073050085`, mean validation node-x ratio `1.1391798404`, maximum ratio `1.1783511224`, and all three operational deltas non-positive.
- The selected combination is a screening candidate, not a final method freeze or final performance claim; all results remain aggregate-baseline and non-locked, and the model still does not consume the per-RB target sidecar.

- Downstream horizon audit for the selected candidate improved communication/throughput/RB metrics at `k=1/2/3`, but node-x MAE deltas remained positive (`+0.209790/+0.208241/+0.205736`) and learned node-x error growth averaged `3.16999x`.
- CPU input-perturbation audit for the selected candidate showed node-x MAE deltas of approximately `+10.13/+20.97/+42.70` at normalized noise `0.05/0.10/0.20`, while link-F1 changed little; this is sensitivity evidence, not a final robustness claim.
- The next defensible step is a CPU-only single-variable root-cause diagnosis for state error/input sensitivity. No new GPU run or locked-test access is justified by the current downstream evidence.

## Created-flow reset and replay closure (2026-08-26)

- The remaining `flow_conservation` replay failure was not in the v3/v4 source tensors. On the actual v4 rule-contract tensor root, `total_data - remaining_data - delivered_cumulative` was within `5.21540641784668e-08` for all 155,875 active flow slots and 12,924 first-active flow slots across 54 unlocked trajectories.
- Root cause was rule-state initialization for a newly created flow: a nonexistent previous slot can denormalize to the training mean, so its cumulative delivered value must not be inherited. The corrected rule starts a newly created flow's cumulative delivered data and age from physical zero, then applies only the current slot service/time.
- A fresh CPU replay of the three existing rule-enabled GPU checkpoints against their actual v4 tensor root passed. Each seed has 64 validation batches, 192 expected/observed rule calls, and empty flow/RB/CPU/lifecycle/DAG violation counts. The report remains a non-locked aggregate-baseline diagnostic; it does not establish final performance, per-RB consumption, candidate-action rollout planning, or locked-test evidence.

## Candidate-action planner mechanism review (2026-08-26, in progress)

- `r6_joint_policy.py` implements a direct candidate scorer: its forward path encodes the present explicit/latent state and `candidate_descriptors`, then produces logits. It does not accept a world model, call a world-model rollout, or expose predicted future state/task/cost/risk values.
- `r6_rollout.py` records executed policy transitions and computes GAE. Its use of the word rollout means an observed training trajectory, not candidate-wise world-model simulation before action selection.
- `formal_dual_graph_world_model_v1.py` rolls the logged `future_action` sequence in a batch. The next audit must distinguish that action-conditioned prediction capability from an action-selection mechanism which independently rolls out every legal candidate from a common history.

### Result

- The new static audit reports `blocked`: the formal model is action-conditioned, but no audited path generates legal candidate sequences, invokes the world model per candidate from a common history, extracts candidate-specific state/task/cost/risk outcomes, selects from those outcomes, or closes feedback/replanning. Artifact SHA-256: `7A71C78F78058F8F0C19EC0509816A2A31D972B27DA8CF1D8FABC8F7CC3B6637`.

## Seven-paper close reading (2026-08-26)

- The local close-reading bundle is `literature/新增7篇精读笔记_20260826.jsonl` plus `literature/新增7篇精读汇总_20260826.md`; all seven sources are public arXiv preprints.
- The literature reinforces three current boundaries: (1) simulator/world-model faithfulness must be decomposed into observable action-to-state and state-to-response links; (2) a candidate-action planner requires candidate-wise model rollouts from a common history, predicted outcomes, selection, and replanning; (3) task-aware uncertainty and paired decision effects should be measured separately from global prediction error.
- Physical priors, multi-modal tokens, shared state, and inverse-dynamics anti-collapse are candidate design ideas only. They do not change the frozen PI-JWM tensor contract, rule-layer replay result, planner audit status, or `formal_performance_claim_ready=false`.

## 2026-08-26 file-tree and evidence-layer governance

- The repository is intentionally not physically cleaned in this pass. Its large `code/artifacts/`, meeting archive, and historical log populations contain provenance and unique outputs; moving or deleting them without a reference manifest and explicit authorization would risk evidence loss.
- The reliable organization is an indexed reading order plus evidence labels: root planning files and work logs are process records; `记录/本地计划表.md`, `记录/PIJWM主文档.md`, `记录/8.12之后推进.md`, approved manifests, and P8-approved artifacts are authority/final candidates.
- A filename such as `formal`, `best`, or `latest` is not evidence of finality. State fields, blockers, hashes, split/seed closure, and the applicable phase gate decide the evidence level.

## 2026-08-27 P4 h20 precision finding

- This is a genuine new P4 precision result, not a missing-file problem: all three completed h20 seed artifacts are present and the recorded manifests match. The learned model loses to persistence for node-x at k=20 in every seed (mean `22.8591 m` versus `22.1450 m`), and its nominal 95% node-x interval covers only `75.43%` of targets.
- Therefore the correct state is `P4 precision blocked`, not `h20 pending` and not a final method failure claim. Preserve the frozen contract and diagnose one possible cause on CPU before considering any minimal change or retraining. `locked_test` remains unused.

## 2026-08-27 P4 node-type contract finding

- Code/data chain: `formal_airfogsim_collector_adapter_v2.py` stores AirFogSim `node_type` verbatim; the formal source bundles contain short codes `V/U/I/C`; `airfogsim_tensor_v2.py` recognizes only `vehicle/uav/rsu/edge_server/cloud` and writes `-1` otherwise.
- Full h20 scan: all 54 unlocked trajectory tensors contain 2,484/2,484 invalid `node_kind_index` entries. Their 290,076 present-node time slots consequently have zero valid type masks. This is not missing data: the source graph contains concrete node identities, positions, and short-code types.
- Mechanism consequence: `formal_dual_graph_world_model_v1.py` uses `node_kind_index >= 0` as the node validity mask. Physical edge-node message passing, information flow-agent message passing, and agent-node coupling are therefore zeroed on the h20 runs. Flow-edge coupling and task-to-node writes can still execute, so the path is partially active rather than a clean no-op.
- Interpretation: k=20 node-x MAE `22.8591 m` remains a real measured error, not a metric-mask artifact. But it is evidence for a malformed-input partial model, not for the intended complete dual-graph PI-JWM method. P4 remains blocked; first repair the four-code normalization and enforce the present-node type contract, then repeat unlocked CPU acceptance before deciding on retraining.

## 2026-08-27 P4 节点类型修复证据（进行中）

- 已用测试先固定规则：`V/U/I/C` 必须映射到冻结契约中的 `vehicle/uav/rsu/cloud`；所有实际出现节点必须有有效类型；padding 节点可保留 `-1`。
- 失败测试确认根因后，最小修复只改变张量化入口，不改历史源数据、世界模型、损失、训练配置或 GPU 脚本。
- 30 项直接相关测试均通过。尚未生成修复后的 h20 产物，因此还不能声称双图消息已在正式数据上重新启用；下一门是独立非锁定 tensor 重建和 CPU 验收。
# 2026-08-27 P4 RB outcome reconstruction finding

- Root cause: source bundles have `source_rb_actions` and `source_transfer_events`, but no `source_rb_observations` or top-level `n_rb`; reading only the absent observation list produced `n_rb=1` and lost direct RB labels.
- Verified rule: action `rb_indices` are RB identities; event `path` is the physical edge; rate is `planned_capacity / 0.1s`. Two unlocked trajectories have one action with no event; those remain masked/unobserved.
- Repair is limited to the RB reconstruction, tensor-contract inference, and formal graph entry, with focused tests. No model, loss, protocol, GPU script, or source data changed.
- Rebuilt artifact `code/artifacts/formal_tensor/pi_jwm_v4_tensor_v4_rule_contract_h20_rb_repair_unlocked_20260827` passes tensor readiness with 54 unlocked trajectories, 14,742 windows, `n_rb=50`, 48,447 observed RB labels, valid present-node types, and no `locked_test` output.
- P4 remains blocked pending CPU tensor/window acceptance and checkpoint reload; this is data-contract repair evidence, not a performance result or GPU authorization.
## 2026-08-27 P4 repaired h20 acceptance finding

- The repaired h20 tensor is structurally usable: `formal_tensor_ready=true`, history=8, horizon=20, 54 unlocked trajectories, `n_rb=50`, and no materialized `locked_test`.
- Focused P4 tests are green: RB reconstruction 11/11, tensor build 8/8, world-model interface 8/8, CPU smoke 2/2.
- The CPU acceptance artifact is a small interface/training check (8/4/4 windows), not a converged performance result. It proves data-to-model wiring, finite forward/backward behavior, and checkpoint artifact creation only.
- The independent training-protocol report says the contract is ready for review, but its GPU launch gate remains false because the repaired CPU baseline comparison has not yet been audited. Keep `formal_performance_claim_ready=false`.
- Full-suite failures are bounded outside this gate: one repository root markdown expectation and 13 legacy AirFogSim/Windows fixture errors. Do not use them to claim either P4 success or a new model defect.

## 2026-08-27 P4 repaired h20 Go/No-Go finding

- The repaired h20 tensor now has the required three independent CPU seeds.
- The gate still blocks GPU: validation link-F1 deltas versus persistence are `+0.0213`, `-0.4492`, and `-0.0079`; mean delta `-0.1453`. The two negative seeds are decisive.
- Other checks are not the blocker: validation node-x MAE ratio is `1.0071`, and throughput/RB/task-delay errors improve on average. Calibration link-F1 improves on average but cannot replace validation evidence.
- This does not prove the repaired model is fundamentally wrong; it proves the current small CPU protocol is unstable for communication-activity prediction. The only next work is CPU-only cause diagnosis.

## 2026-08-27 P4 communication-activity F1 diagnosis

- The three seeds were replayed on the same selected non-locked windows. Validation contains 270 positive and 25,502 negative link-activity labels; calibration contains 86 positive and 27,564 negative labels. The training subset contains 669 positives and 62,755 negatives.
- Seed `20260827` has validation ROC-AUC `0.9611749` and average precision `0.4358372`; seed `20260829` has ROC-AUC `0.9579971` and average precision `0.3066603`. These two seeds separate positives from negatives reasonably well before thresholding.
- Seed `20260828` has validation ROC-AUC `0.3559574` and average precision `0.0107389`; calibration ROC-AUC is `0.0586247` and average precision `0.0018521`. Its positive scores are generally lower than its negative scores, so this is a ranking failure, not only a bad threshold.
- Calibration-selected thresholds are `0.9`, `0.1`, and `0.7` for the three seeds. The threshold changes are a visible symptom of seed-dependent score location and ordering; they cannot be used to replace validation evidence.
- Root-cause status: confirmed evidence limitation is the combination of a very small CPU training sample/one epoch and extreme class imbalance, which makes the communication-activity head seed-unstable. A deeper architectural cause is not established by this diagnostic.
- Boundary: keep the frozen tensor contract and formal training protocol unchanged; do not repeat the same GPU run. `gpu_allowed=false`, `formal_performance_claim_ready=false`, `locked_test_accessed=false`.

## 2026-08-27 P4 expanded CPU stability finding

- Expanded CPU evidence now covers three independent seeds with the pre-registered `256/128/128` budget and the same repaired h20 contract.
- Gate report `code/artifacts/audit/pi_jwm_p4_h20_repair_expanded_cpu_go_no_go_20260827/cpu_to_gpu_gate.json` reports validation link-F1 deltas `-0.0411/-0.1753/+0.0104` (mean `-0.0686`), node-x MAE ratio `1.0849`, RB-occupancy delta `+0.1132`, and calibration link-F1 mean delta `+0.2404`.
- The larger budget improves evidence quality but does not pass the gate. Two of three seeds regress in validation F1 and the mean is below persistence; RB occupancy also regresses on average.
- Root-cause status: the earlier extreme seed inversion is not the only issue. The current candidate has not demonstrated validation non-inferiority under the expanded CPU protocol. No architectural or contract change is justified without a separate approved diagnosis.
- Boundary: `gpu_allowed=false`, `formal_performance_claim_ready=false`, `locked_test_accessed=false`; no GPU, locked-test, or frozen-contract modification occurred.

## 2026-08-27 P4 failure diagnosis: link activity and RB

- A CPU-only replay of the three expanded runs shows validation link-activity ROC-AUC values near `0.991` for all seeds. Fixed threshold F1 is much less variable than calibration-selected F1; the current failure is threshold transfer across calibration/validation class proportions, not a ranking inversion.
- Action-derived RB occupancy is almost aligned with the target: exact edge agreement is `99.41%`--`99.61%`, and total MAE is `0.5949` on validation / `0.7473` on calibration. The learned RB output is exactly the deterministic action-derived output, so it is not independently learning a different RB quantity in this candidate.
- The existing RB metric path is inconsistent with its own data units: normalized prediction and raw `aggregate_rb_occupancy` are both multiplied by the train scale; prediction mean restoration is missing and the target is not kept in the same physical unit. This explains why the gate reports `3.2023` despite action-derived MAE below `0.75`; the old RB regression gate is invalid until the metric is corrected and regression-tested.
- This is a diagnosis only. Do not silently change the metric or threshold protocol; require one approved, focused repair plus independent Go/No-Go rerun. GPU and `locked_test` remain closed.

## 2026-08-28 P4 metric-repair finding

- The approved minimal repair restored the prediction mean before comparison and kept prediction/target in the same physical unit. Re-evaluation confirms validation RB occupancy delta=`-0.6477 RB` versus persistence.
- This removes the old RB failure as an evidence item; it does not authorize GPU because the independent gate still reports validation link-F1 mean delta=`-0.0686` and throughput MAE delta=`+0.7101 Mbps`.
- The re-evaluation was read-only over existing non-locked CPU checkpoints and sample IDs. No model, tensor contract, training protocol, GPU state, or `locked_test` state changed.
- Current interpretation: P4 has one fewer failed gate, but the candidate still lacks validation non-inferiority. Keep `gpu_allowed=false`, `formal_performance_claim_ready=false`, and do not enter P6.
- Single next action: human decision on the remaining link-F1/throughput gate handling; no duplicate GPU run before that decision.

## 2026-08-28 P4 threshold protocol audit finding

- Added `code/scripts/run_formal_p4_threshold_protocol_audit_v1.py` and two focused tests. The script is read-only and uses only the three existing expanded CPU runs; report: `code/artifacts/audit/pi_jwm_p4_threshold_protocol_audit_20260828/threshold_protocol_audit.json`.
- With `pos_weight=50`, ordinary posterior correction is `p=s/(50-49s)`. Ordinary probability `0.5` corresponds to raw score `50/51`, which produces poor recall and is rejected for this gate.
- Fixed raw `0.5` gives validation F1 `0.6305/0.4924/0.6005` for seeds `20260830/20260831/20260832`, versus calibration-selected `0.5490/0.4148/0.6005`. Threshold transfer is a contributor, but seed `20260831` still fails against persistence and the mean remains below the pre-registered non-inferiority requirement.
- The remaining throughput MAE delta is `+0.7101 Mbps` after metric repair. The threshold audit does not authorize retraining, GPU, P6, or `locked_test`; it is a diagnosis, not a gate pass.
## 2026-08-28 P4 throughput diagnosis finding

- The read-only report `code/artifacts/audit/pi_jwm_p4_throughput_diagnosis_20260828/throughput_diagnosis.json` compares every one of 20 forecast steps for all three existing expanded CPU seeds.
- The failure is not uniform: seed `20260830` has validation bias about `-3.75 Mbps`, seed `20260831` about `-4.38 Mbps`, and seed `20260832` about `+2.03 Mbps`. For seeds `20260830/20260831`, the last five steps are more negative than the first five; seed `20260832` remains positive throughout.
- Therefore the remaining throughput gate is best described as unstable rollout magnitude/calibration across seeds, with error accumulation for two seeds. The diagnosis does not establish whether the cause is loss weighting, target scale, or model capacity; no such change is authorized yet.
- `gpu_allowed=false`, `formal_performance_claim_ready=false`, and `locked_test_accessed=false` remain unchanged. No training or code contract was changed.

## 2026-08-28 P4 epoch-stability hypothesis

- Existing seed `20260830` validation loss changed from `-0.2235` to `-1.1349` to `-1.7877` over epochs 1--3; the other seeds use the same 3-epoch budget. This is evidence that the short CPU check stopped while the objective was still moving, so under-training is a plausible single cause of the throughput and link-F1 instability.
- The next check changes only epochs (`3 -> 8`) in isolated CPU runs. It does not assume the hypothesis is true; the independent gate decides whether the evidence improves.

## 2026-08-28 P4 epoch-stability execution finding

- The first seed used the intended candidate only and completed with best epoch 6 and verified checkpoint reload.
- The initial second-seed direct-function invocation exposed a reproducibility hazard: the function default is all five V1 learned methods, whereas this diagnostic requires only `coupled_dual_gnn_residual`. The unintended run was stopped after a `pooled_gru` checkpoint appeared.
- This is a launch/configuration error, not a model result. The partial directory remains isolated and excluded from the independent gate. Future CPU invocations must pass the learned-method list explicitly.

## 2026-08-28 P4 epoch-stability gate finding

- The corrected three-seed 8-epoch CPU check passes the independent Go/No-Go gate. The report records `gpu_allowed=true` with no failed criteria.
- Validation link-F1 deltas are `+0.02496`, `-0.01173`, and `+0.05789` (mean `+0.02371`); node-x MAE ratio is `1.04629`; calibration link-F1 mean delta is `+0.28557`.
- Validation operational metrics improve over persistence: throughput MAE delta `-0.49943 Mbps`, RB occupancy MAE delta `-0.64768 RB`, and task-delay MAE delta `-1.34958`.
- The result opens only the next formal non-locked GPU execution decision. It does not open `locked_test` or `formal_performance_claim_ready`, and no GPU was started in this step.

## 2026-08-28 P4 GPU reachability finding

- The independent CPU Go/No-Go report authorizes a formal non-locked GPU run, but both recorded SSH ports (`14826`, `14507`) are currently unreachable and return `Connection refused`.
- This is an external server-availability block, not a model, data, metric, or protocol failure. No GPU training, upload, or `locked_test` access occurred.
- Do not switch endpoints, broaden the run, or retry repeatedly without a reachable server; resume with the same fixed configuration once connectivity is restored.

## 2026-08-28 P4 reachability recheck finding

- A second read-only check confirms the external block persists on both recorded ports; local evidence remains internally consistent and the P4 CPU gate remains passed.
- No new model or data diagnosis is justified while the approved GPU execution host is unavailable. Keep the mainline paused at the server boundary.
## 2026-08-28 P4 GPU launch finding

- The remote environment has CUDA, but the background launcher does not inherit the interactive precheck `PYTHONPATH` unless it is exported inside the command.
- Evidence: the precheck imported `pi_jwm` successfully with explicit `PYTHONPATH`; the first background run failed immediately at the first project import and used 0% GPU.
- No training result is valid from that attempt. Keep the directory as failed process evidence and restart the same seed with one explicit environment fix.

## 2026-08-28 P4 second GPU seed launch finding

- The first `20260831` background attempt did not train: it exited before model import with `ModuleNotFoundError: No module named 'pi_jwm'`, and GPU usage stayed at 0%.
- The failure is fully explained by missing environment propagation. Import succeeds with `PYTHONPATH=/root/autodl-tmp/pi_jwm_p4_h20_epoch8_gpu_20260828/code/src` and fails without it.
- This attempt is invalid process evidence only; no model/data/protocol conclusion changes, and `locked_test` remains untouched.

## 2026-08-28 P4 second GPU seed completion finding

- After explicitly exporting the remote source path, seed `20260831` completed without errors and passed the required completion, CUDA, checkpoint-reload, and non-locked boundary markers.
- This confirms the previous failure was limited to launcher environment propagation. No training parameter, tensor contract, or scope was changed.

## 2026-08-29 P4 third GPU seed completion finding

- Seed `20260832` completed with the same required markers as the first two seeds: training complete, CUDA execution, checkpoint reload verified, and `locked_test_accessed=false`.
- All three runs share the fixed data manifest and protocol. No formal performance claim is made until local manifest recovery and the independent multi-seed audit pass.

## 2026-08-29 P4 GPU three-seed audit finding

- Local recovery and `formal_gpu_multiseed_audit_v1` passed the reproducibility/boundary checks: three seeds share the same contract, all manifests have zero mismatches, GPU execution is true, and no `locked_test` content was used.
- The learned model improves link activity and operational errors, but node-x MAE is worse than persistence for all three seeds (`+0.4621`, `+0.8483`, `+1.3708 m`). This keeps the P4 performance gate closed and prevents any final-method claim.
- The correct next step is a focused decision about the remaining node-state error, not another unapproved training sweep or a phase jump.

## 2026-08-29 P4 节点位置误差聚焦处理

- 当前唯一性能阻塞是验证集 node-x MAE 相对 persistence 变差；三个 GPU seed 均变差。
- 现有仓库已有逐 horizon rollout 审计入口，可直接复用，避免重复实现。
- 待验证的单一诊断问题：位置误差是否主要在长 horizon 累积，还是由位置变化量/节点类型/归一化或反馈路径造成。当前不预设答案。

## 2026-08-29 P4 节点位置误差诊断发现

- 只读 CPU 回放报告：`code/artifacts/audit/pi_jwm_p4_position_diagnosis_20260829/position_diagnosis.json`。
- 三个 seed 在第 1 步到第 20 步均显示预测位移不足；第 20 步真实 x 位移均值约 `22.145 m`，预测 x 位移为 `0.774/2.027/3.816 m`。
- 车辆、UAV、RSU 分组均有同方向欠预测，静态节点误差不是唯一来源。
- 单一主因：residual state head 的位移更新幅度过小，造成长 horizon 位置 rollout 欠预测；该结论由三个独立 seed 的同方向证据支持。
- 风险：这仍是诊断结论，不是修复验证；P4 继续 blocked，不能进入 P6，不能访问 `locked_test`。
- 下一步只允许一个最小 residual 位移幅度修正的 CPU 单变量验证；若未通过，停止并报告，不扩大实验。

## 2026-08-29 P4 residual 幅度三 seed CPU 门发现

- 三 seed 的 scale=1.0 CPU 结果通过既有独立 Go/No-Go：`gpu_allowed=true`、`failed_gates=[]`。
- node-x MAE ratio=`1.0851`，验证 link-F1 delta=`+0.0476/-0.0400/+0.0945`（均值 `+0.0340`）；吞吐量、RB 占用和任务时延均值相对 persistence 改善。
- 这说明幅度修正方向在三 seed 上具有稳定的 CPU 证据，但它仍只是 GPU 放行证据，不是最终方法冻结或正式性能声明。
- 继续边界：只允许同一配置的 non-locked GPU 三 seed；不访问 `locked_test`，不进入 P6，不扩大参数搜索。

## 2026-08-29 P4 residual 幅度单变量发现

- `residual_state_scale=1.0` 的 seed `20260830` CPU 运行完成，配置和数据契约与 scale=0.5 仅一处不同。
- 第 20 步 node-x MAE 为 `22.9194 m`，persistence 为 `22.1450 m`，比 scale=0.5 的 `23.8567 m` 更接近真实移动；位置比例约 `1.035`，低于原门 `1.25`。
- 这只是一个 seed 的方向性证据，不能替代三 seed Go/No-Go；link-F1、吞吐量等完整门仍未重算。
- 风险边界：不要把 scale=1.0 直接写成定版参数；在三 seed CPU 复核前，P4 仍 blocked，GPU 和 `locked_test` 继续关闭。
- 唯一下一步：按同一配置运行 seed `20260831`，再运行 `20260832`，最后用既有独立审查脚本重算门。

## 2026-08-29 P4 scale=1.0 GPU 审计发现

- 三个 non-locked GPU seed 已完成；`gpu_execution=true`、checkpoint reload verified、`locked_test_accessed=false`，结构审计通过且 manifest mismatch 为 0。
- validation link-F1 delta 为 `-0.0010/-0.0337/+0.0524`，均值 `+0.0059`；node-x MAE delta 为 `+0.2504/+0.8316/+3.3702 m`，均值 `+1.4841 m`。
- 吞吐量 MAE、RB occupancy MAE、task-delay MAE 三项均值相对 persistence 改善，但不能抵消节点位置误差失败。
- 结论：residual scale=1.0 通过了 GPU 执行/可复现性门，没有通过 P4 性能门；不能把该配置写成最终方法或进入 P6。
- 下一步只允许：根据已有位置欠预测诊断，先定义并审核一个新的、最小且可证据检验的处理方案；不重复同一 GPU 配置，不访问 `locked_test`。

## 2026-08-29 为什么项目长期卡在 P4

- P4 的科学性能门从未真正闭合：当前 scale=1.0 GPU 结构审计通过，但 node-x 三 seed 全部变差，link-F1 平均提升很小且存在 seed 回退。
- 最大返工源头是顺序错误：数据契约、规则层和指标单位问题在较晚阶段才发现；旧 GPU 结果因此只能降级，必须重建或重训。
- 第二个返工源头是门定义不够严格：CPU gate 只判断有限统计条件，GPU audit 主要判断可复现性和边界，二者都不是最终性能门。
- 第三个返工源头是没有停机规则：每次失败都追加一个局部改动，却没有先判断是数据、实现、指标还是模型能力问题。
- 第四个返工源头是远端执行不稳定：路径、环境变量和默认方法列表没有在启动前硬性检查。
- 第五个返工源头是记录治理不够单一：旧状态和当前状态并存，容易重复做已经做过的工作。
- 改进原则：critical mismatch 未闭合不训练；执行通过和性能通过分开报告；同一失败最多做一个预注册最小验证，验证不支持就停止并重判路线。

## 2026-08-29 P4 状态接力修复发现

- 新失败测试证明旧 residual 逻辑在第二步仍使用历史最后状态；固定每步残差时，旧实现第二步只得到 `last + 1`，而不是应有的 `last + 2`。
- 最小修复后，两步 node state 和 task DAG state 都按上一轮预测值累积，规则层输出也会作为下一步递推状态；没有改数据、统计、损失或训练协议。
- 代码回归测试 `9/9` 通过，说明接口层行为符合预期；这不是模型性能门。
- 新 CPU seed `20260830` 已完成并 reload；seed `20260831` 因本轮运行时间过长被停止，未产生有效结果，seed `20260832` 未运行。
- 当前风险：长 horizon 真实数据三 seed 的位置、链路和业务指标尚未重新测量；P4 仍 blocked，GPU 和 `locked_test` 继续关闭。

## 2026-08-29 P4 状态接力修复三 seed CPU 门发现

- 三个固定 CPU seed 均已完成并通过 checkpoint reload，运行边界为 CPU-only、non-locked，未读取或生成 `locked_test`。
- 独立门审查报告为 `code/artifacts/audit/pi_jwm_p4_recursive_cpu_go_no_go_20260829/cpu_to_gpu_gate.json`；结构、位置、校准和运营指标门通过。
- 唯一失败是逐 seed validation link-F1 稳定性：`+0.1032/-0.0941/+0.2013`，其中 `20260831` 低于 `-0.05` 上限；均值虽为 `+0.0701`，不能掩盖单 seed 回退。
- 这说明状态接力修复已能完成三 seed CPU 运行，但链路活动预测仍有 seed 波动；不能放行 GPU、不能关闭 P4，也不能进入 P6。
- 下一步只允许基于已有证据分析该单一失败门，禁止继续叠加模型修复或扩大调参。

## 2026-08-30 `_CONTEXT.md` 交接审阅初始发现

- 现有过程记录的最新结论是 P4 仍 blocked；代码级状态接力修复已通过定向测试，但三 seed CPU Go/No-Go 因一个 seed 的 validation link-F1 回退而未放行 GPU。
- 该结论目前仅是交接审阅的起点，尚未与 `PROJECT/_CONTEXT.md`、权威记录、Git 工作区和机器可读审计完成交叉核验。
- 本次不启动 GPU、不读取 `locked_test`，也不把“执行链完成”写成“科研性能门通过”。
- 用户给出的 `PROJECT\_CONTEXT.md` 实际不存在，且 `PROJECT` 目录不存在；精确搜索仅找到根目录 `PROJECT_CONTEXT.md`，文件大小 33,221 bytes，修改时间为 2026-08-30 19:36:27，故将其作为本次交接源。
- Git 当前为 `main@0630515`，工作区有大量已修改和未跟踪的代码、测试、记录与脚本；它们是交接时必须保留的 active work，不能按干净提交状态理解。
- `PROJECT_CONTEXT.md` 明确把当前阶段限定为 P4：三 seed CPU 训练、验证、校准和 checkpoint reload 已完成，但最新 Go/No-Go 为 `gpu_allowed=false`；报告唯一失败项是逐 seed validation link-F1 回退。
- 直接阻塞数值是 seed `20260831` 的 validation link-F1 相对 persistence 为约 `-0.0941`，超过允许回退 `-0.05`；三 seed 平均提升不能掩盖单 seed 失败。
- 另一未闭合证据是逐步位置表现：当前 gate 的 node-x ratio `1.1869 <= 1.25` 是合并所有预测步的结果，不等于用户原计划要求的第 1/5/10/20 步长期位置门。
- 精确下一步只允许读取三 seed 指标、threshold selection、gate 定义和既有诊断，形成判断；如需改模型、loss、阈值协议、数据、预算或新实验，必须另行说明并确认。
- `记录/本地计划表.md` 第 1005--1010 行与 `记录/8.12之后推进.md` 第 681--687 行都把最新状态覆盖为 recursive CPU gate blocked；`记录/PIJWM主文档.md` 的 P4 进度只更新到 8 月 27 日，不能单独覆盖 8 月 29 日机器结果。
- `PIJWM主文档.md` 顶部当前统一命名明确禁止把现代码称为完整 RSSM；文档其他位置的 RSSM 历史审计语句不能越过该当前口径。
- 最新 gate JSON 的字段、数值、失败项和 SHA-256 与交接文件一致；这是当前状态最直接的机器证据。
- 三个 recursive CPU seed 的配置只有训练随机 seed 不同；共同 `data_seed=20260823`、manifest SHA-256 `d5c9edb4200840ad085adaab970855555c19198ed0031234513ea1f078660781`、CPU、h20、8 epochs、`residual_state_scale=1.0`、确定性规则层开启。
- 三个运行均为 `training_run_complete=true`、checkpoint reload verified、`gpu_execution=false`、`locked_test_accessed=false`、`formal_performance_claim_ready=false`。
- calibration 选择的 link threshold 为 `0.7/0.9/0.7`；该差异是后续诊断输入，当前不足以单独证明 seed 回退根因。
- gate 代码第 55--56 行对 `min(validation_deltas) < -0.05` 直接阻断，因此三 seed 平均为正不能覆盖 seed `20260831` 的失败。
- 当前模型代码中 `recursive_states` 和 `recursive_dag_state` 均在每步末尾更新；定向测试存在两步递推断言。这证明接力机制已实现，不证明真实长期精度通过。
- 三个 seed 的 `sample_ids.json` SHA-256 均为 `F9ED48D1AE1B532341FEEFA3E093AA44D0DF3FBFAECC94D4DA3EF0273744198C`，因此当前 link-F1 差异不是 validation 样本集合不同造成的。
- fresh 定向验证通过，但验证范围有限：世界模型 9 项、gate 2 项、compileall、diff check；未运行完整测试、AirFogSim、GPU、planner 全链或 `locked_test`。
- 交接结论：当前继续停在 P4。已完成证据是递推机制、三 seed CPU 运行/reload 和 gate 复算；阻塞是单 seed link-F1 回退及逐步位置证据缺口；唯一下一步是证据化诊断，不是训练。

## 2026-08-30 成本路由与零重复实验规则

- 用户要求后续大计划默认使用 `cost-aware-model-routing`，优先利用 Luna/Terra 的速度和成本优势，Sol 负责高判断任务和最终验收。
- 用户授权根据真实使用结果持续改进该技能，但改动必须由重复出现的路由问题或明确证据支持，不能因单次偶然快慢过拟合规则。
- 无技能压力测试暴露两类风险：一次安排 Terra 加两个 Luna 并行，容易过度委派；仅观察一次 Luna 更快就立即固化长期降级规则，存在单样本过拟合。
- PI-JWM 特定硬规则：先查历史证据和失败记录；无新增变量和新问题不得重跑；历史成功结果必须绑定配置/数据/seed/指标/哈希并受回归保护；历史失败也必须保留，不能通过覆盖或改名消失。
- 当前 P4 只允许复用现有证据诊断 link-F1 seed 不稳定性；任何模型、loss、threshold、数据、预算或实验变更都不在本轮授权内。
- 技能 GREEN 前向测试通过：读取修改后技能的 Terra 拒绝根据单次 40 秒结果改长期规则，改为记录候选路由证据并等待多个独立同类任务；项目门继续留在项目记录中。
- 技能经压缩后 534 词，`quick_validate.py` 返回 `Skill is valid!`；保留了实际模型委派、最多两个工作者、复用优先、证据驱动改进和协调者验收。
- 本轮实际路由：Luna 负责历史 artifact 去重清单；Terra 负责当前 recursive 三 seed 与历史诊断对比；两者均只读，Sol 不重复其机械扫描，只复核关键原始字段。

## 2026-08-30 P4 link-F1 去重清单与当前证据缺口

- Luna 去重确认：旧 link activity diagnosis 曾发现小样本 seed 排序反转；expanded failure diagnosis 后三 seed ROC-AUC 约 0.991，阈值迁移成为主要观察；RB 单位错误和 throughput 偏差也已分别诊断。这些都在状态接力修复前，不能直接解释最新 recursive checkpoint。
- Terra 对当前 JSON 的比较确认：三 seed 配置、样本、参数量和协议一致；`20260831` calibration 选阈值为 `0.9`，另两 seed 为 `0.7`；其 calibration F1/AUPRC 较弱，但 validation k=20 AUPRC 没有明显崩坏。
- Sol 复核原始字段得到 validation k=20 AUPRC `0.5601/0.6920/0.7136`、calibration k=20 AUPRC `0.2691/0.2340/0.3273`，与 Terra 一致。
- 当前目录没有 raw score/logit/prediction 文件，现有 JSON 不足以分离阈值迁移、排序和分数尺度的相对贡献。
- 现有 `run_formal_p4_link_activity_diagnosis_v1.py` 已实现 CPU-only 的 ROC-AUC、AP、正负分数分位数和阈值网格；但其 glob 只匹配旧目录名，当前可用临时 junction 复用而无需改代码。
- 修复后 checkpoint 的首次只读诊断已完成：`code/artifacts/audit/pi_jwm_p4_recursive_link_activity_diagnosis_20260830/link_activity_diagnosis.json`，SHA-256 `BF94979D9A761EE440E0EC39EFCCAB6D9056F200B82455ECD11E8E38E9897F01`；3 runs、CPU、GPU=false、locked-test=false。
- `20260831` validation 在冻结阈值网格中的最佳阈值仍为 `0.9`，最佳 F1=`0.4960`，低于 persistence `0.5901`；所以 calibration threshold 迁移不能单独解释失败。
- 在 threshold `0.9`，`20260831` validation 为 TP=`4594`、FP=`6175`、FN=`3163`；另两 seed 的 FP 为 `2186/1414`。当前直接失败机制是高置信假阳性显著增多，即顶部正负分离不稳定。

## 2026-08-30 P4 link-F1 逐步证据化判断

- validation 第 1/5/20 步的 `AUPRC/F1/precision/recall`：seed `20260830` 为 `0.9167/0/NA/0`、`0.9911/0.9498/0.9634/0.9365`、`0.5601/0.4525/0.2959/0.9604`；seed `20260831` 为 `0.8340/0/NA/0`、`0.7131/0.3263/0.6693/0.2157`、`0.6920/0.3802/0.2388/0.9314`；seed `20260832` 为 `0.8528/0.7235/0.9858/0.5714`、`0.9792/0.9050/0.9423/0.8706`、`0.7136/0.6673/0.5224/0.9235`。
- calibration 同口径下，seed `20260831` 第 1/5/20 步为 `0.2216/0/NA/0`、`0.4918/0.4201/0.4765/0.3757`、`0.2340/0.1944/0.1092/0.8818`。第 20 步 TP/FP/FN=`179/1460/24`，说明长步高置信假阳性在 calibration 内部也存在，不是 validation 阈值迁移单独造成。
- seed `20260831` validation 在同一冻结阈值 `0.9` 下，第 1/5/20 步 TP/FP/FN=`0/0/364`、`85/42/309`、`353/1125/26`：行为从短步漏报转为长步大量误报，直接指向 recursive rollout 过程中的分数尾部漂移。
- 全步诊断的 negative q90 为 `0.2420/0.2823/0.2992`，失败 seed 并不最高；overall q99 为 `0.7883/0.9229/0.6859`，threshold `0.9` 的 FP 为 `2186/6175/1414`。因此异常集中在极高分尾部，而不是整体负样本分数普遍抬升。禁止读取不存在的 `negative_score_quantiles.q99`；该字段不存在，PowerShell 空值转 `0` 不是证据。
- 直接失败机制已闭合到“长步递推放大高分负样本尾部”，但底层根因尚未闭合到具体的输入反馈、head、loss 或训练动态。按 systematic-debugging 规则，当前不提出模型、loss、阈值或训练修复。

## 2026-08-30 P4 link-F1 底层只读审计

- `link_activity_head` 直接读取每步的 physical-edge latent；该 latent 每步经过 edge GRU 更新，第 2 步起还接收 deterministic rule 产生的 physical-edge state feedback。因此“递推 edge latent 漂移”是与长步假阳性机制一致的候选路径，但当前仍是候选，不是已证根因。
- 三 seed 共用 train-only link `pos_weight=50`、`sparse_event=0.1` 和完全相同的样本；训练历史只记录 total train/validation loss，三个 seed 的 best epoch 均为 8，无法从现有曲线读取逐 horizon link loss。
- checkpoint 参数只读对比：seed `20260831` 的 link-head bias=`+0.1450`，另两 seed为 `-0.1650/-0.1526`；link-head weight L2=`0.5504/0.6372/0.7194`，edge GRU weight L2 约均为 `5.8--5.9`，physical-edge feedback weight L2=`3.3790/3.3568/3.4617`。没有参数范数爆炸证据。
- 正 bias 可能贡献分数上移，但约 `0.3` logit 的 seed 间差异不能单独解释阈值 `0.9` 以上 FP 的三倍差距。最小缺失证据是逐 horizon 负样本 logit 分位数，以及从 logit 中减去 head bias 后的对应分位数；获得前不能把根因写成 head bias 或 edge feedback。

## 2026-08-30 P4 严格收口计划发现

- Luna 的只读矩阵正确识别了 link-F1 和逐步位置两个剩余门，但误引用了 2026-08-26 旧 tensor 入口。Sol 用当前三个 run 的 `config.json` 纠正为 `pi_jwm_v4_tensor_v4_rule_contract_h20_rb_repair_unlocked_20260827`，manifest SHA-256=`d5c9edb4200840ad085adaab970855555c19198ed0031234513ea1f078660781`。
- Terra 的诊断设计确认代数边界：`raw_logit = w·edge_latent + bias`，固定 bias 不能制造 h1 -> h20 的增量；逐 horizon `raw_logit-bias` 能判断 latent 在 head 方向是否漂移，但不能进一步未经干预就指认 GRU、图消息或规则反馈。
- 旧计划中的 CPU 三 seed 性能训练不再作为下一轮执行方式。按用户新要求和 pre-GPU hard gate，CPU 只保留极小 reload micro-smoke；正式训练转为用户开启 GPU 后的 sentinel `20260831` -> 其余两 seed，不做 sweep。
- P4 最终验收必须同时关闭：每 seed validation link-F1 回退、calibration link 优势、聚合 node-x、1/5/10/20 位置、运营指标、uncertainty/mask、manifest/reload 和 non-locked 边界。不能只修 link-F1 就宣布 P4 完成。

## 2026-08-30 P4 v2 bias/latent 诊断发现

- Terra 在冻结规格内只新增诊断脚本和测试，TDD RED/GREEN 证据完整；Sol 两阶段复核未发现范围扩张或评价口径漂移。本次单次成功不足以修改长期模型路由规则，只作为后续路由样本保留。
- v2 与 v1 在固定 `0.9` 阈值下的 validation/calibration 六组 raw FP 总数完全相同：`2186/2634`、`6175/8579`、`1414/1878`，证明分 horizon 重放没有改变原评价对象。
- seed `20260831` 正 bias=`+0.1450`；去 bias 后 validation 总 FP 只减少 `436`，h20 只减少 `39`。因此 bias 是放大因素，但不能直接解释大部分失败。
- 同一 seed 去 bias 后 validation q99 在 h1/h5/h20 为 `-0.19/-0.34/4.96`，q95 为 `-0.57/-1.19/1.18`；尾部不是从第一步单调上升，而是在后段递推中急剧抬高。准确表述应是“长 horizon edge latent 在 link-head 方向漂移”，不是笼统的全分布上移。
- forward 代码证明 link logit 在每步由更新后的 `edge` 经过同一线性 head 产生；loss 对真实 mask 使用同一 aggregate activity 加权 BCE，评价阈值由 calibration 的固定网格选择。未发现 v2 与模型/loss/metric 的关键口径冲突。
- 目前仍不能区分 direct physical-edge rule feedback、edge GRU/physical message 或 cross-flow coupling。最小下一诊断只干预 direct physical-edge rule-feedback 投影，其他路径全部保持不变；结果只判定该直接路径的因果贡献，不外推到整个规则层。

## 2026-08-30 direct physical-edge rule-feedback 因果路径闭合

- 干预保持 checkpoint、样本、head bias、edge GRU、physical message、cross-flow coupling 及 node/flow/task feedback 不变，只把 `state_feedback['physical_edge'](...)` 的输出临时置零。
- h1 基线/干预完全一致，证明 hook 没有提前改变无 rule-feedback 的第一步；原 baseline 逐 horizon negative count/raw FP 与冻结 v2 报告完全一致。
- h20 去 bias FP `1086 -> 0`，pre-bias q95 `1.18 -> -0.16`、q99 `4.96 -> -0.16`；全 20 步 raw FP `6175 -> 0`。按预注册条件，该直接路径对观测到的高分尾部具有充分解释力。
- 理论仍要求规则修正后的最终状态进入下一步，故完全关闭这条反馈会造成理论--实现不一致，且本报告没有正样本/召回保护证据，不能作为修复。
- 最窄的理论一致候选是把 physical-edge correction projection 从“直接加到 recurrent hidden state”改为“作为 edge GRU 的输入消息”，让 GRU 门控接收规则修正，同时避免无门控式 hidden 注入。该候选尚未获用户确认、尚未实现或验证。

## 2026-08-30 physical-edge feedback 输入路由实现发现

- 用户已确认方案 A，候选已实现并通过接口门。精确语义是：规则 correction 仍进入下一步 edge transition，但作为 GRU input message，由 GRU 门控处理；它不再预先改写 recurrent hidden。
- RED 测试证明旧行为真实存在而非命名差异；GREEN 证明新行为的 h2 hidden 不受直接注入，normal/suppressed 的 GRU input 差精确等于非零 projection。
- 本次未改参数模块，原 seed `20260831` checkpoint strict load 后 missing/unexpected keys=`0/0`，参数量保持 `83750`。
- 旧 CPU smoke runner 无 deterministic rule layer 且无 strict reload，不能验证本次改动；复用正式训练模块的 CPU 内部接口才与理论和实现一致。
- `2/1/1`、1 epoch micro-smoke 只证明接口可训练、保存、strict reload 和指标 JSON 有限，不提供任何性能结论。P4 是否修复仍需 GPU sentinel 和固定门验证。

## 2026-08-31 GPU sentinel 启动发现

- 远端默认环境没有 `python` 命令，但 `/root/miniconda3/bin/python` 为 Python 3.12.3，PyTorch `2.8.0+cu128` 且 CUDA 可用；必须在命令中显式使用该路径。
- 历史 GPU 流程可复用的是远端隔离、显式 `PYTHONPATH`、单方法和 manifest 审计；历史 residual scale `0.5` 不适用于当前已冻结 `1.0` 协议。
- 远端已有相同 canonical tensor manifest，可服务器内复制以节省上传；旧 checkpoint、旧 run summary 和旧训练结果均不复用。
- 单 seed 不能运行现有三-seed gate；sentinel 完成后必须直接读取 config/summary/runtime/manifest/comparison/class_weights/threshold report 做预注册字段判断，通过才授权另外两个 seed。

## 2026-08-31 GPU sentinel 结果发现

- 方案 A 没有破坏 node-x、throughput、RB occupancy 或 task-delay 门，但没有解决失败 seed 的 validation link-F1：delta 从冻结门角度仍为 `-0.2177`，因此不能继续三 seed。
- calibration 选择的 link threshold 仍为 `0.9`；validation candidate F1=`0.3724`，低于 persistence `0.5901`。当前只知道“性能未修复”，尚不能从 aggregate F1 指认新旧具体机制。
- sample IDs 的 parsed JSON 完全相同；raw hash 差异是跨平台换行，不是样本漂移。未来跨 Windows/Linux 冻结样本身份应使用 canonical JSON hash或逐 ID 对比，不能只比较原始字节哈希。
- 下一诊断的唯一新增问题是：GRU-input routing 后，新 checkpoint 是否仍出现长步高置信假阳性尾部。既有旧 checkpoint 诊断不能直接替代该回答。

## 2026-08-31 新 checkpoint 阈值迁移判定

- 新 checkpoint 已修掉旧的长步高置信 FP 尾部，但冻结 `0.9` 下错误转为 FN 主导：validation `TP/FP/FN=1902/557/5855`，precision=`0.7735`、recall=`0.2452`、F1=`0.3724`。
- 只读候选回放显示 `0.7` 可把 recall 提到约 `0.9309`，但同时产生 `16600` 个 FP，precision 只有约 `0.3031`，F1=`0.4573`；因此简单降低阈值只是用大量 FP 换回 recall，仍打不过 persistence。
- persistence 在同一 validation 汇总上 `TP/FP/FN=8944/5856/6570`，precision=`0.6043`、recall=`0.5765`、F1=`0.5901`。候选最优 `0.7` 与它的 F1 delta=`-0.1327`，仍低于冻结允许值 `-0.05`。
- 准确边界：可以说 calibration 到 validation 存在阈值迁移影响；不能说 No-Go 只由 `0.9` 导致，也不能用 validation 事后选择 `0.7` 作为新正式阈值。
- 当前问题已从“旧反馈路径造成长步高置信 FP”转为“现有 link score 与冻结候选阈值之间没有满足 persistence 门的工作点”。这是方法/指标协议问题，下一步必须先做理论--实现--指标一致决策，不能直接叠加第二个代码修复。

## 2026-08-31 link score/threshold 计划审计发现

- 当前训练对 link activity 使用 train-only `pos_weight=50.0` 的 weighted BCE；正式 threshold selection 对 raw sigmoid score 直接在 calibration 上选 `0.1/0.3/0.5/0.7/0.9`。
- 代码库已有 weighted-score 到 unweighted posterior 的数学修正 utility 和测试，但当前正式 GPU runner 的 threshold path 未接入该 utility。
- 主文档同时要求 link activity 为离散事件概率、类别加权后仍报告概率校准、calibration split 只确定概率阈值。因此必须先判断 raw weighted score 的方法语义，不能一边称概率、一边按未校准 decision score 使用。
- 历史 threshold audit 只能证明 correction 机制和旧 checkpoint 现象，不能当作当前 sentinel 性能。
- 路由复核再次证明嵌套 metrics 必须读取 `thresholds.link_activity`，不能用顶层默认 `threshold` 替代；当前有效 link threshold 已 fresh 核验为 `0.9`。

## 2026-08-31 link score/threshold 方法判定

- `protocol_coherent_candidate_failed` 被排除：主文档没有把 raw weighted sigmoid 定义为纯 decision score，而是明确要求事件概率与概率校准。
- `evidence_incomplete` 被排除：checkpoint/sample IDs/tensor、mask、aggregate、effective threshold、overall F1 和 persistence 均可核验。
- 当前状态为 `protocol_theory_mismatch`：weighted BCE 合法，但 raw sigmoid 在正式链中既未做概率语义修正/校准，也未报告 Brier/ECE，不能称为已经校准的事件概率。
- 性能边界不变：正式 `0.9` F1=`0.3724`；只读候选最优 `0.7` F1=`0.4573`；persistence=`0.5901`。No-Go 继续成立。
- 唯一推荐不是立即接入某个 correction，而是先冻结一个 post-training probability calibration boundary 设计对象；具体数学与接口只能在用户确认后单独设计，不能同时试多条路线。

## 2026-08-31 方案 B 设计发现

- weighted BCE 的理论赔率偏移可用 `z-log(pos_weight)` 固定反演；随后只拟合一个正 scalar temperature，既保留类别不平衡训练，也符合主文档对 temperature scaling 的候选口径。
- 将原 raw thresholds 通过同一严格单调函数映射到 probability thresholds，可以逐元素保持全部分类决策；因此不会返工方案 A、checkpoint、tensor 或 sentinel，但也不会改善现有 F1。
- 方案 B 能关闭的是 link probability 语义、calibration split、Brier/ECE/NLL 的一致性缺口；当前 P4 仍被 sentinel link-F1 性能门阻止。
- 如果 validation 被用于拟合温度、挑选方案或重新使用数值 `0.1/0.3/0.5/0.7/0.9`，就会变成新协议或事后调参，必须拒绝。

## 2026-08-31 方案 B 实施分解发现

- 最小可靠实现需要把概率数学、metric 消费、runner split 所有权和当前 sentinel audit 分成四个清晰接口；若只接 utility 而不进入 runner，仍不能关闭理论--实现链。
- 当前 sentinel 只需 CPU 只读 inference；训练、GPU 和 follow-up seeds 都不是概率语义门的必要条件。
- 为保护历史消费者，runner 的 legacy raw threshold identity 与正式 probability threshold 必须分别保存，不能复用一个含糊的 `threshold` 字段。

## 2026-08-31 方案 B Task 1 实现发现

- 仅有 happy-path 数学测试不足以证明概率边界可靠；极端 log-temperature、复数 tensor、非有限 class weight 和极小正权重都会暴露静默污染风险。
- `correct_positive_weighted_probability` 的分母对 `score∈[0,1]`、`w>0` 理论上严格为正；旧 `clamp_min(1e-12)` 会错误改变极小正权重的精确反演，现已由回归测试保护。
- raw threshold 到 probability threshold 必须使用与概率预测完全相同的 torch float64 路径，否则临界点舍入可能破坏逐元素 decision equivalence。

## 2026-08-31 方案 B Task 2 实现发现

- 只在 metadata 写“event_probability”不够；实际 `thresholds.link_activity` 必须是 mapped 值，legacy raw identity 必须单独保留。
- 最可靠接口是让调用方只传 legacy raw threshold，由 accumulator 内部调用同一 calibration 对象映射，消除外部双阈值和 tolerance 分叉。
- 浮点等值点上，形式上单调的 float64 threshold 与 float32 probability 仍可能翻转；分类统计必须显式复用同 dtype/device 的 legacy decision，概率指标继续使用正式概率。
- 既有 throughput/RB 全 mask 计数问题与本任务无关；本轮未顺手修改，以免主线扩张。

## 2026-09-01 方案 B Task 3 实现发现

- learned calibration 缺少有效样本或只有单类标签时，拟合出的 temperature 可能极端但看似有效；正式 runner 必须在拟合前要求正负类同时存在。
- RULE baseline 没有概率校准，不能因为共用 threshold report schema 就写 `probability_threshold`；raw/probability 字段必须按方法身份分开。
- 正式 runner 若边训练边写最终目录，后段校准失败会留下可能污染下次 manifest 的半成品；同级 staging + 成功一次发布是必要的证据完整性门。

## 2026-09-01 方案 B Task 4 实现发现

- float32 阈值等值点证明“数学映射严格单调”不等于有限精度下直接 probability comparison 必然逐元素相同；artifact 必须区分 direct comparison 与正式分类复用 legacy decision，不能用改写 mapped threshold 制造等价。
- provenance 不能只记录非空 SHA 和正 mask 数；mask count 必须绑定实际有效元素，canonical tensor manifest、共同元文件和 calibration/validation 所选 tensor 都必须在读取前核验。
- strict reload 的成功路径不足以证明严格性；缺 key、unexpected key、错误 model contract、错误 seed 与 sample ID 漂移都需要独立失败测试。
- 当前只证明 Task 4 audit 入口达到规格和代码质量门；真实 temperature、NLL/Brier/ECE、TP/FP/FN、manifest 与 No-Go 保留情况仍须由 Task 5 的冻结 CPU audit 决定。

## 2026-09-01 方案 B Task 5 真实证据发现

- 单元测试和接口审查通过不等于真实概率门通过；冻结 sentinel 上，calibration-only scalar temperature 使 validation NLL 相对 `T=1` 解析反演基线恶化，停止门实际触发。
- 当前可证结论只到：方案 B 的实现和审计入口可执行，但冻结的“解析反演 + 单温度”方法未通过预注册的 validation 泛化要求，`probability semantics gate = failed`。
- 因运行在 artifact 发布前停止，不能事后补写 temperature、Brier/ECE、TP/FP/FN 或 manifest 数值，也不能把旧 sentinel 数值冒充本次 probability audit 产物。
- 不允许用 validation 重新选温度、换 calibrator、放宽非恶化门或重复运行制造通过；下一方法动作必须是用户批准的独立单变量决策。

## 2026-09-04 P4 link低召回诊断发现

- 低召回的主量来自持续活跃链路，而不是新激活链路：持续组占正样本`6050/7757`，且persistence正确而candidate漏报的`2791/2806`来自持续组。
- candidate逐步recall不是单调下降，而是h1为零、中段升高、h20再次接近零；这排除了“只看overall recall就直接换阈值”的解释。
- 同一个link head用于全部20步，horizon形状必须来自edge latent轨迹；h1又没有上一预测步rule feedback，因此rule feedback不能单独解释全部现象。
- 最小可证伪下一步是只旁路`edge_transition`，保留其他计算并检查历史edge记忆能否找回主要漏报；该干预不是正式修复，不能把结果当作模型性能。
- 真实诊断保持`gpu_execution=false`、`locked_test_accessed=false`、`formal_performance_claim_ready=false`，P4仍blocked。

## 2026-09-05 edge GRU旁路干预发现

- 让`edge_transition`直接返回进入GRU前的hidden state后，所有有效链路的raw score都未越过冻结阈值，candidate TP/FP/FN从`1902/557/5855`变为`0/0/7757`。
- 原始2,791个“持续活跃、persistence正确、candidate漏报”样本没有任何一个被找回；持续活跃recall从`0.2648`降为`0`，h1/h20均仍为`0`。
- 因此当前证据反对“GRU更新单独擦除了历史活跃记忆”这一解释；edge GRU更新反而是现checkpoint产生任何高于0.9链路分数的必要步骤。
- 该结果不证明edge GRU完全正确，也不证明CFE、rule feedback或link head错误；旁路同时阻断所有消息经GRU写入edge latent，只能否定本次充分原因假设，不能从失败结果跳到另一个模块结论。
- 后续若继续，应先设计一个更细的只读机制核验，把GRU门控/输入消息/hidden贡献拆成一个新的、单变量且可证伪的问题；未经用户确认不得执行。

## 2026-09-05 edge GRU接口只读追踪发现

- 冻结基线仍为TP/FP/FN=`1902/557/5855`。在5855个FN中，5559个是`pre_negative -> post_negative`，296个是`pre_positive -> post_negative`；即进入本步GRU前已偏低占`94.94%`，本步向下跨阈值占`5.06%`。
- h1的364个FN全部在更新前已低于阈值；h20的375个FN中370个如此。因此“本步edge GRU更新主导向下压分”被数据否定。
- h2以后的`hidden_before`是上一步`hidden_after`的递推结果，所以incoming主导不能被误读成history encoder单独失败；它只说明低分在当前步之前已形成并被递推带入。
- 数据--代码静态核对：`aggregate_link_activity` 精确定义为`physical_edge_state.active_task_count > 0`与mask的交；`active_task_count`在tensor contract的五个物理边特征中，而模型初始边编码消费完整`physical_edge_state`。因此不支持“训练输入没有链路活动信息”。
- 模型forward没有把`history.aggregate_link_activity`作为独立的二值递推状态，而是由edge hidden经link head直接预测绝对活动事件。这是一个可设计的候选方法边界，但当前证据尚不能宣称它已是根因或已验证修复。
- 下一步不再追个旁路旧模块，只允许先形成一份`candidate method`：用上一步链路活动作为持久性基准，学习action-conditioned变化量；必须先证明h2+只使用模型自身前一步预测而非真值target，并与weighted BCE和方案B概率语义一致。用户确认前不实现。

## 2026-09-05 link activity持久性残差候选设计发现

- 当前可复用的`last_persistence`会把历史最后一帧活动重复到20步，并用有限logit `+20/-20`表达确定性活动/非活动；因此候选可在零残差时精确复现该基线，而不另造标签或阈值。
- 唯一推荐参数化是：现有单路link head输出未加权事件log-odds变化量；h1以历史最后一帧的持久性logit为基准，h2-h20累加模型自身上一预测的未加权logit；正式输出再加`log(50)`供现有weighted BCE和legacy raw decision使用。
- 方案B的`weighted_logit-log(pos_weight)`会回到候选内部事件logit；temperature仍只能在训练后用calibration拟合，不能反馈到20步递推。
- 双路hazard虽有清晰转移语义，但新增输出参数且现证据不支持这份自由度；简单bias若使用硬阈值会把评价阈值塞进动力学，若改成软log-odds递推则等价于推荐方案。
- 旧checkpoint的参数形状可能仍能加载，但旧head是绝对logit、新head是变化量，语义不兼容；必须用方法/schema身份拒绝续训和正式评价，不能把`missing/unexpected keys=0/0`当兼容证据。
- 主要风险是20步log-odds变化量累积导致过度自信或漂移；只能由未来一次冻结sentinel证伪，不能失败后调持久性幅度或换第二方案补救。

## 2026-09-05 link activity持久性残差CPU实现发现

- 观测历史必须先从已有加权raw logit `+20/-20`减去`log(pos_weight)`得到内部`u0`；每步累加head输出的变化量，正式输出再加回`log(pos_weight)`。直接把`+20/-20`当内部`u`会破坏既有概率坐标。
- 新link机制必须挂在原`coupled_dual_gnn_residual`候选上；若注册为非残差模型，会在修link的同时改变其他状态头，违反单变量边界。
- 只核对`link_activity_method`不足以拒绝语义错误checkpoint；新方法还必须核对`mode=coupled_dual_gnn`和`residual_state_prediction=true`。
- canonical CPU micro证明接口、递推、规则反馈、保存和复载闭合，但不证明召回或概率质量改善；GPU sentinel仍是唯一性能证伪门。
- CPU门后提供的GPU端点在SSH握手前拒绝连接，因此本轮没有GPU执行事实；不能用历史GPU可用性或CPU micro代替新的sentinel。

## 2026-09-05 项目评估发现

- 当前方案B独立审计仍只支持旧`coupled_dual_gnn_residual`并硬锁raw threshold=0.9；新方法已注册/可复载不等于验收入口已支持。拒绝新身份的最小调用已复现，不涉及数据推理。
- 正式runner在阈值选择过程中拟合温度，再计算性能；需在后续入口闭合时对齐“先性能后概率”的书面顺序。没有据此发现validation参与拟合，也未改算法。
- 零delta复制persistence是条件契约；正式link head仍默认Linear初始化，zero-init残差状态头选项不等于link head置零。正负20起点可能阻碍状态切换，属于待sentinel检验的风险，不能表述为已证实失败。
- 当前持久性残差方向有针对性，应保持单变量；先补验收衔接，再做冻结单seed，不通过重复诊断、扩网格或新数据源掩盖P4未闭合。
- 详见`记录/研究进展/2026-09-05-项目现状与主线推进评估.md`；GPU/locked_test边界未改变。

## 2026-09-05 P4 验收入口修复发现

- 旧概率审计的硬编码已收缩为两种明确方法：旧`coupled_dual_gnn_residual`继续固定raw threshold=`0.9`；新`link_activity_persistence_residual_v1`从calibration threshold-selection产物读取冻结候选中的所选阈值。
- 新方法必须在run manifest中绑定threshold-selection文件，且推理使用对应方法身份；未知方法、错误坐标、非calibration选择、候选集外阈值和manifest漂移均拒绝。
- 这只是验收入口修复，不是概率门通过；GPU性能结果和新方法真实校准结果仍未知。
- GPU TCP检查失败，补查历史端口`14507`也失败；未上传、未训练、未访问locked_test，继续保持P4 blocked。

## 2026-09-05 P4 GPU sentinel 参数审计发现

- 启动后复核runner参数发现首次命令未包含样本上限，不能把全量窗口运行误写成`train/validation/calibration=256/128/128`。
- 证据边界已明确：偏差进程已停止，远端staging仅作失败追溯；不读取其指标、不运行方案B、不启动其他seed。
- 正确冻结命令必须显式传入`--train-limit 256 --evaluation-limit 128`；这仍是唯一下一动作，模型和实验协议不变。

## 2026-09-05 P4 persistence residual GPU sentinel evidence

- 远端retry按正确样本上限完成GPU训练；run摘要为`training_run_complete=true`、`gpu_execution=true`、`locked_test_accessed=false`，本地manifest核验`0 mismatch`。
- validation link-F1 candidate/persistence=`0.4801978088/0.5900903873`，delta=`-0.1098925785`，违反冻结下限`-0.05`。
- node-x candidate/persistence=`14.4161/11.8520`，ratio约`1.2163`；吞吐、RB occupancy、task delay未回退。
- 结论：sentinel performance gate=`no_go`，不执行概率门、不启动其他seed，P4继续blocked。

## 2026-09-05 完整 RSSM 论文与实现发现

- PlaNet 和 Dreamer 的关键可迁移条件不是“有一个随机层”，而是 deterministic state、action-conditioned prior、observation-conditioned posterior、训练 posterior 与部署 prior-only 的明确分工；PlaNet 还要求多步 latent supervision。
- 旧 `_GraphRSSMBackend` 的 context prior 和 posterior 都直接从同一个 `base_belief.joint` 计算；旧目标只计算 context KL，故名称可以保留为历史候选，但证据不足以支持完整 RSSM 结论。
- RDR 论文提示 teacher-forcing 与自由滚动分布不一致是长程误差来源；新候选把 prior rollout 与逐步 posterior 均值的一致性显式记录为辅助项。该项是 PI-JWM 适配，不等同于已证明 RDR 效果。
- 新 `complete_graph_rssm_v1` 使用 target 只构造训练 posterior；预测输出和 rollout prior 不读取 target。契约测试确认目标扰动不改变 prior-only 预测，且 KL 梯度到达 transition/prior/posterior。
- 完整性接口已闭合，但没有真实数据性能证据；不应把 4/4 测试、h20 finite 或论文机制直接写成性能提升。
- 当前 P4 仍 blocked；GPU 未使用，`locked_test_accessed=false`。下一步只做独立 CPU micro runner 和证据绑定，不扩展其它论文组件或参数网格。

## 2026-09-05 完整 RSSM canonical CPU preflight

- 首次运行将 target-conditioned posterior 辅助张量纳入整体输出等值比较，错误触发 `future_target_leakage_absent`；验收器已修正为只比较部署可见输出和 prior 分布，posterior 教师张量单独检查有限性与梯度。
- 修正后新候选 `complete_graph_rssm_v1` 在 train/validation/calibration 的 h1/h5/h20 共 9 个 canonical 窗口通过：训练步、有限梯度、非零梯度、全 rollout 有限、动作条件、target 泄漏、strict checkpoint roundtrip 均为 true。
- 产物：`code/artifacts/preflight/pi_jwm_complete_rssm_cpu_preflight_20260905_retry/`，`r4_cpu_preflight_ready=true`，`gpu_screening_ready=false`，`locked_test_accessed=false`。
- 该门只证明完整实现可在真实冻结数据上执行，不证明收敛、性能提升或最终方法选择；P4 仍 blocked，GPU 未启动。

## 2026-09-05 完整 RSSM teacher reconstruction 审查发现

- PlaNet/Dreamer/VGRNN 机制对照表明，posterior 除了 KL 还必须参与训练期观测或状态重构；否则只能称部分 variational 路径。
- 新增 `training_predicted_explicit/logits`，仅在 `model.train()` 生成并由 `compute_r4_objective` 用于重构；`predicted_*`、`rollout_prior_*` 和验证指标继续走 prior-only 路径。
- target 扰动测试确认部署可见输出不变，teacher 输出会随 target 改变；梯度、有限性、strict reload 和 manifest 均通过。
- 该修正补齐方法闭环但不证明性能改善；不启动 GPU 以外的额外变量，不调阈值，不访问 `locked_test`。

## 2026-09-05 完整 RSSM prior/teacher 双重重构结论

- 为避免训练目标丢失部署语义，最终保留 prior rollout 的 R3 重构，并额外加入权重 `0.5` 的 teacher 重构；KL、KL balancing 和 overshooting 保持不变。
- teacher 张量只在训练模式生成，验证/部署仍只读取 `predicted_*` prior 输出；target 泄漏测试保持通过。
- v2 canonical preflight 的 9 个 train/validation/calibration h1/h5/h20 窗口通过，artifact claim boundary 仍为 execution evidence only。
- 现在 GPU 才是必要的下一门：只运行 seed `20260831`，先性能门，失败即停；不访问 `locked_test`。

## 2026-09-05 完整 RSSM R4 GPU 筛选发现

- 新完整 RSSM 的 R4 单候选运行没有数值不稳定、保存或复载问题，27 epoch 内最佳 score出现在 epoch 22；这支持其作为可训练候选。
- 该 run 的训练 seed 为`20260803`，选择指标为R4 validation protocol score，且只有一个候选；不能外推到P4冻结 seed=`20260831`、正式link-F1门或多seed结论。
- 当前关键缺口不是再跑R4，而是将prior/posterior/teacher语义接入P4正式模型和正式指标路径，并以CPU一致性门证明训练、部署和评价定义一致。

## 2026-09-06 正式完整 RSSM sentinel 发现

- “完整 RSSM”必须让 stochastic latent 实际参与所有被正式评价的连续状态与事件输出；只补 prior/posterior/KL 接口而让分类头绕过 latent，仍属于不完整实现。
- `zero_init_residual_state_heads=true` 必须覆盖新增 RSSM 连续 correction heads。遗漏该契约会在 h1 直接破坏 persistence/residual 锚点；修复后 h1 node-x 从无效 v2 的 `14.239 m` 降到最终 v3 的 `2.205 m`。
- 最终 v3 证明完整 RSSM 能把 link-F1 控制在 persistence 允许回退范围内，并改善 throughput、RB 和 task delay；但 node-x ratio=`1.28774` 超出 `1.25`，P4 仍不能通过。
- v3 node-x 在 h1/h5/h10/h20 为 `2.205/6.948/13.763/30.005 m`，persistence 为 `1.664/5.332/11.047/22.145 m`。剩余问题是随 rollout 累积的位置误差，不再是初始随机 residual 偏移。
- 下一步应只读分解 `final prediction = formal dual-graph base + RSSM correction`，判断长期误差主要来自哪一项；在该证据前不应调 KL、decoder 权重、link 阈值或训练预算。

## 2026-09-06 node-x 修正项诊断发现

- 同一冻结 checkpoint、同一 validation 样本下，formal base ratio=`1.15756`，低于保护线 `1.25`；加入 RSSM correction 后 ratio=`1.28774`。故本轮 node-x No-Go 不是由 base 单独造成，而是 RSSM 连续修正把结果推过失败线。
- correction 平均幅度约 `3.04 m`，但仅 `23.79%` 的有效节点得到改善；从 h1 到 h20，完整输出的 MAE 均高于 base。这支持“修正可信度不足”的机制判断，不支持“多训练几个 epoch 就会好”的结论。
- 该分解是关联定位，不是因果修复实验。最小的新变量定义为训练期 `node_x_residual_non_degradation_v1`：只处罚修正后误差大于 base 的部分，同时保留 prior-only、posterior teacher、KL 和事件 decoder 语义；在 CPU 审计和新冻结协议之前不得启动下一次 GPU sentinel。

## 2026-09-06 node-x 非劣化约束实现发现

- 直接用完整输出和 base 计算约束会把新梯度同时传给 base，混入第二个变量；最终实现从显式 `rssm_node_state_correction` 重构 detached base，因此隔离损失只更新 correction 及其上游 RSSM 路径。
- posterior teacher reconstruction 必须把该新增权重置零，否则同一个安全约束会在 prior 与 teacher 两条路径重复出现；当前候选只约束部署实际使用的 prior correction。
- CPU 结果只证明公式、mask、梯度、checkpoint 身份和无未来泄漏成立。micro run 的任何 validation 数值均不用于性能判断。
- 协议比较确认除 `node_x_residual_non_degradation_v1` 外，v3 的数据、模型结构、训练预算、seed 和性能阈值均未变化。GPU 关闭期间没有必要开展其他实验。

## 2026-09-06 node-x 非劣化约束 GPU 发现

- 约束成功把 RSSM correction 平均绝对幅度从 `3.04245 m` 压到 `0.07621 m`，但这没有保护最终 node-x：base 自身 MAE 变为 `17.97475 m`，完整输出为 `18.02196 m`。
- 新损失对 base 的隔离直接梯度为零，不代表联合训练中 base 不会变化。原 state NLL/MAE 仍通过 `base + correction` 更新 base；当 correction 学习轨迹改变时，base 的最优轨迹也会改变。
- 旧、新 run 初始化哈希同为 `30efb3104c3de9b8ea4bc8fdfb9a321c6aa6036d6bf4bccd1c8ab4ed9d519095`，排除了初始化差异。当前证据反对继续增大非劣化权重或换 seed。
- 若继续主线，唯一有证据支撑的候选是复用旧 v3 已达到 ratio=`1.15756` 的 base 并冻结它，只训练 RSSM correction；这仍是待评审候选，不是已实施方法。

## 2026-09-06 P4 系统性根因发现

- “修正项过大”只解释了完整 RSSM v3 的一次失败，不能解释长期停滞。node-x-safe 已把 correction 压到 `0.07621 m`，base 仍恶化到 ratio=`1.51660`，说明共享优化轨迹是更底层问题。
- 正式 run 的 checkpoint 以 aggregate validation loss 最小为准，P4 却要求五项指标逐项过门；二者没有同一选择规则。局部总损失改善不能推出正式门改善。
- sentinel 的 `256` 个训练窗口只占全部 `14742` 个 unlocked 窗口的一小部分，且 8 epoch 结束时 validation loss 仍下降；当前证据不足以把 No-Go 全部归因于模型结构。
- 位置状态没有 heading、速度向量、路线或机动控制动作；link activity 正例约 `0.918%`。前者构成长步位置的信息充分性风险，后者使 F1 与校准高度敏感。
- “冻结 v3 base 再训 correction”只能隔离一条优化路径，无法排除输入、预算和目标冲突，故降级为待审候选。下一步必须先做现有数据/checkpoint 的 CPU-only 系统审计。

## 2026-09-06 文献对照与 latent 粒度发现

- 当前 `formal_complete_rssm_v1_1` 在“动作条件 prior、观测 posterior、KL、teacher reconstruction、overshooting、部署 prior-only”这些 RSSM 语义上是完整的；此前完整性结论在这一层仍成立。
- 新复核暴露了另一层边界：节点、边与任务先被 masked pooling，单个 stochastic latent 经过 decoder 后为每类实体广播同一 correction。它不能在同一步内为不同节点生成不同随机修正，也不能用单一 link offset 改变边之间的 logit 排序。
- G-RSSM、R-SSM、Graph Dreamer 和 GNS 都把动态状态保持在节点/对象粒度，再通过消息传播表达相互作用。由此得到的是一个有文献依据的结构候选，不是性能证明。
- 因此继续优化同一个 global broadcast correction、冻结 base 后只重训它，均不能消除其表达上限。当前更有信息量的动作是先计算这一上限，而不是立即重建模型。
- 位置运动信息、训练预算、checkpoint 选择和多任务梯度冲突仍是独立风险；不得与 latent 粒度在一次实验中同时修改。

## 2026-09-06 P4 实体级 RSSM 第一性原理结论

- P4 长期卡住不是单一超参数问题：旧随机状态是 global aggregate，link correction 无逐边排序能力；运动输入字段失真；共享目标量级和方向冲突；短预算仍未收敛；checkpoint 选择目标与 P4 门不一致。
- 旧 global node-x oracle 的 aggregate MAE ratio 约 `1.05346`，所以不能声称“全局修正一定过不了 aggregate node 门”；实体级方案的必要性主要来自逐边排序、实体差异和理论定义。
- 因果运动修复没有改变 split 或未来可见性，只把当前及更早位置差分成真实速度/加速度；它修复输入合同，不是加入未来标签。
- 实体级 RSSM 的 CPU 证据证明方法语义、训练可达性和正式 runner 接通，没有证明三 seed 性能。1-epoch gate selector 的失败数字不得作为 No-Go。
- 当前内部 CPU 链已经闭合，外部阻塞是远端 GPU 端口拒绝连接。恢复后必须从 frozen protocol v2 的 batch probe 开始，不得回退到旧 8-epoch sentinel、node-x safe loss 或 global RSSM。

## 2026-09-06 实体级 RSSM GPU 执行发现

- 4090 上 batch 8 的完整 forward/backward 峰值仅占总显存 `16.32%`，因此显存不是本方法正式训练的当前瓶颈；冻结协议仍按预注册候选顺序选 batch 8，不因余量临时扩大。
- batch probe 的 loss 随 batch 不同而变化是因为每档读取的样本集合不同，不能用来比较性能；四档共同证明 loss 和参与训练参数的梯度有限。
- 2 epoch sentinel 完成了真实 optimizer training，RSSM 阶段没有更新冻结 base，best checkpoint 已由 runner 严格重建并加载，说明此前 CPU 证明的分阶段合同在 CUDA 路径同样可执行。
- sentinel 的 validation loss 从 base 阶段 `0.77888656` 降至 RSSM 阶段 `0.76041028`，只属于执行观察；短预算结果不能接受或否决方法。
- 正式 seed `20260831` 已按唯一冻结配置运行。首 seed 结束前没有依据启动另外两个 seed、调整阈值、改变 loss 或访问 `locked_test`。

## 2026-09-06 正式训练证据保全发现

- 远端 runner 在隐藏 staging 中逐 epoch 写 checkpoint，正式目录在完成前为空；因此必须按 staging 做只读快照，不能只依赖最终发布目录。
- 已保存 base epoch `001–005` 的原始 checkpoint、配置、样本 ID、方法注册表、类别权重和进程/GPU 快照；本地证据包 manifest SHA-256=`e2ea66161b663cbbb1dc2807a946dbf5bda6d770d01be82ff96a165fc93b7ad7`。
- 复制和哈希查询未停止或暂停训练进程；当前运行仍由远端 PID `2307` 持有，`locked_test_accessed=false`。
- 监控期间检测到 epoch `006` 新增，立即完成第二次 staging 快照并回传该 checkpoint；这验证了中间证据保全可以与长时间训练并行，不需要中断或重启运行。
- 监控期间检测到 epoch `007` 新增，已按同一规则追加快照和哈希；历史证据没有覆盖。
- 监控期间检测到 epoch `008` 新增，已按同一规则追加快照和哈希；历史证据没有覆盖。

## 2026-09-07 正式训练阶段证据

- base 20 epoch 已完整结束，RSSM epoch 1–2 checkpoint 已生成；这证明 runner 已按冻结合同实际执行“先训练 base、再冻结 base 训练 RSSM”的阶段切换。
- 当前 RSSM 峰值显存约 5.0 GB，高于 base 阶段约 3.5 GB，符合实体 stochastic 分支进入训练后的资源变化；没有 OOM。
- 尚不能由前 2 个 RSSM epoch 判断 P4 性能或是否早停，最小训练轮数仍为 20。

## 2026-09-07 entity RSSM 中间性能发现

- epoch 7–14 连续八个 checkpoint 的单 seed 数值门均通过，说明当前改善不是某一个 checkpoint 的偶然波动。
- entity RSSM 当前同时改善链路排序、长期 node-x、throughput、RB occupancy 和 task delay；这与“逐实体 latent + 因果运动 proposal + 冻结 base”要解决的三个已证实根因方向一致。
- validation state NLL 从 `-3.10564` 单调改善到 `-3.10747`，gate-aware rank 仍在改善，因此现在不能提前停止或挑选 epoch 14 作为最终结果。
- calibration link-F1 delta 很大，但仍需最终独立审计确认阈值只来自 calibration、validation 没有参与选择；当前不据此声明 P4 已通过。

## 2026-09-06 正式 seed 全量运行时序

- runner 使用隐藏 staging 目录进行原子发布，正式输出目录在运行完成前保持为空；当前 staging 已有 base epoch `001/002/003` 三个 checkpoint。
- 实测全量 base epoch 约 26 分钟，当前瓶颈是完整窗口前向/反向和评估计算时间，不是显存容量或进程挂死。
- 当前配置仍是冻结协议：base 20 epoch 后冻结，再训练 RSSM 最多 40 epoch；没有修改 loss、阈值、seed 或数据。
## 2026-09-07 组会材料组织发现

- 这一个月的工作可以形成完整组会叙事：先关闭理论与数据缺口，再以正式P4门暴露模型问题，随后通过因果诊断和文献把方法收敛到实体级双图RSSM。
- 若逐项展示所有P4运行，汇报会变成实验流水账；按数据/指标错误、递推/链路错误、位置/共享训练错误三条失败链组织，可以保留每一步新增证据且不弱化工作量。
- 当前首seed中间结果足以支持“新方向获得正式unlocked数据的强正向证据”，尚不足以支持“P4已完成”或“最终方法已冻结”。
## 2026-09-07 组会PPT精简结构判断

- 10页已经能够覆盖本月所有关键模块；将采集器、字段合同和因果运动合并到两页数据，将图定义、消息传播和规则递推合并到两页双图编码，可以减少流水账且保留实际工作量。
- 文献依据最适合直接放在对应方法页：双图对应交互网络、无线GNN和edge-conditioned message passing；实体世界模型对应PlaNet/RSSM、GNS、Relational SSM、Trajectron++及节点级图世界模型。
- 面向组会时应使用论文式方法名称和自然语言阶段描述，内部阶段编号、脚本标签和gate名称只保留在证据记录中。

## 2026-09-07 组会PPT制作发现

- 直接用PowerPoint COM保存追加稿时，程序会自动重写少数未编辑旧页和一个版式XML；仅凭“没有选中旧页”不能证明第1–203页未变。最终稿在保留新增页面和全局页列表的同时恢复了原文件中的旧页、旧关系和共享版式，并对第1–203页共406个部件逐字节复核，差异为0。
- 第204/205页模板足以支持本次10页汇报：继承顶部标题、双线、徽标和字体体系，再用原生文本框、形状、连接线和表格表达流程与结构，无需把整页栅格化为图片。
- RSSM epoch 22 仍为单seed训练中间checkpoint。它继续同时通过链路、长期位置和运营指标数值门，但不能替代最小训练轮数后的严格复载、独立复算和另外两个seed验收。
- 组会页面对后续策略器的准确边界应写为：候选生成与合法性约束、逐候选世界模型推演、代价/风险选择、执行首动作和观测后重规划；当前只有CPU机制原型，尚未形成正式策略器。

## 2026-09-07 组会PPT层级纠正发现

- 继承页标题、徽标和配色并不足以称为“严格复用模板”；如果把正文改造成卡片和横向流程，原模板的红方框大标题、蓝菱形小标题和浅蓝箭头正文层级仍然会丢失。
- 本项目组会页的正确复用单位是第205页正文占位符及其段落结构，而不是只复用背景。最终版直接保留11个原生段落和Wingdings项目符号字符`113/118/216`，再替换段落文字和蓝色引导词。
- 双图编码与实体级RSSM属于串联的两个模型模块：前者生成关系感知的实体表示，后者在该表示上学习动作条件时间动力学。把它们写成“方法一/方法二”会错误暗示二选一或平行对比。
- 结构图只能作为某个小标题下的辅助证据，不能替代文字层级；本轮汇报采用纵向文字结构，优先保证老师能够按“问题—实现—依据—结果”顺序阅读。

## 2026-09-07 组会PPT图表折中发现

- 最合适的折中不是在“全图形”和“全正文”之间平均分配面积，而是先用原生层级给出判断与依据，再把图表放到对应小标题之后解释结构或比较数值。
- 流程、关系结构和递推机制适合用图；正式指标适合用表；定义、理论来源、能力边界和未完成条件仍应由层级正文承担。
- 最终折中版每页使用7个原生层级段落，图表全部位于正文下方；这样同时保留模板阅读顺序和模型结构的可解释性。
- 双图仍是模块一，RSSM仍是模块二；图形连接表达两者串联关系，没有恢复“方法一/方法二”的错误语义。

## 2026-09-07 组会PPT重点强化发现

- 数据页的轨迹划分数字不能解释方法的必要性，会分散对“字段是否可信、动作是否可追溯、样本是否因果合法”的注意力，因此从主汇报中移除。
- 图表的保留标准是能否承担关系结构、时间机制、训练阶段或数值对比；字段定义、适配理由和边界条件用原生层级文字说明更清楚。
- 本版形成单一叙事链：旧数据为什么不能直接学习 → 如何重建合法样本 → 双图如何编码关系 → 旧时间模型为什么失败 → 实体级RSSM如何修复 → 当前证据与后续决策闭环。

## 2026-09-07 当前双图四类对象的实现事实

- 物理节点主状态为 `x/y/z/speed/acceleration/cpu/storage`，并额外提供因果派生的三维速度和加速度给实体级运动 proposal。
- 物理边是物理设备之间的有向通信链路，特征为 `distance/csi_mean/rate_sum/active_task_count/allocated_rb_count`；因此这五项不能称为信息边特征。
- 信息节点是一物理节点一 agent 的附着代理。当前 agent 没有独立观测字段；模型用相同的 7 维 node history 初始化 agent latent，再通过信息图、任务和跨图消息更新。
- 信息边是 agent 之间的数据流，正式数组名为 `flow_state`，特征为 `total_data/remaining_data/delivered_cumulative/delivered_this_slot/age`，类型为任务输入、结果回传和有显式 payload 时的依赖数据流。
- `flow_endpoint_index`定义信息边端点，`flow_task_index`绑定任务，`flow_bearer_mask`逐时隙绑定承载该流的物理通信边。信息图消息沿 agent—flow—agent 传播，跨图消息沿 agent—physical-node 和 flow—physical-edge 传播。
- 实体级 RSSM 只为 node、physical_edge、flow、task 设置 prior/posterior 随机状态；agent latent 是基础双图里的确定性中间表示。这是当前实现边界，不能说四类对象都拥有独立观测和独立随机动力学。
## 2026-09-08 P4 首 seed 收敛发现

- 实体级双图 RSSM 在完整 unlocked 数据上并非只短暂过门：RSSM 40 个 epoch 均保存，冻结选择器最终选择 epoch 39，9 项单 seed 数值门全部通过。
- 最终检查点把旧 global complete RSSM 的 link-F1 delta `-0.04113`、node-x ratio `1.28774` 改善为 `+0.44415`、`0.75475`；当前证据支持因果运动合同、实体级 latent 和两阶段冻结训练这一组合方向。
- 该结论只覆盖 seed `20260831` 的 non-locked 数据。跨 seed 泛化仍未知，不能提前关闭 P4 或形成最终性能声明。
## 2026-09-08 用户锁定后续 seed 启动权限

- seed `20260830` 是当前唯一获授权继续运行的正式实验。
- seed `20260832` 的启动授权已明确收回到用户手动决定；现有通过结果不得被解释为自动续跑授权。
- seed `20260830` 完成后的正确动作是保全产物、独立验收、报告结果并暂停。
- 中间进度的有效证据是 checkpoint 内的完整 `p4_gate` 数值和哈希；单独报告轮次不能支持性能判断。

## 2026-09-08 组会 PPT 复核发现

- 当前 18 维历史记录应称为旧版通信边混合张量；本月工作不是简单从 18 维压到 5 维，而是把物理通信链路和业务数据流重新分成两类对象，各保留 5 维可信状态。
- `per_rb_target_sidecar` 仅为诊断保留，当前正式模型、损失和指标不消费；PPT 不能写成逐 RB 结果用于训练监督。
- 新链路头以上一时刻活动状态为 persistence 基线，再由逐物理边 latent 解码变化量；这是解释链路排序和相对 persistence 提升时不可省略的实现事实。
- 首 seed 结果支持当前结构方向，但跨 seed 泛化仍未知；汇报中必须把单 seed 验收、跨 seed 待验收和 `locked_test` 未访问分开表述。

## 2026-09-08 项目知识入口重构第一阶段发现

- 仓库在 2026-08-16 已完成顶层目录迁移；本次需求的缺口不是再次移动 `代码/文档`，而是缺少稳定的项目地图、架构、科研状态、实验和结果索引。
- 当前最安全的改造方式是新增导航文档并补充维护规则，不对正在同步的 `code/artifacts` 做批量整理。
- 当前 `code/artifacts` 同时存在正式 tensor、训练、audit、历史候选、tmp、transfer 和 live evidence；必须按证据层级索引，不能按 `final`/`best`/`latest` 文件名猜测当前结果。
- 当前 PPT 文件与对应验收 JSON 存在页数和 SHA-256 不一致，说明汇报材料需要独立版本验收，不能把旧验收记录自动套到当前文件。
- 保护边界有效：本轮未改变 `seed=20260830` 训练进程、同步文件、checkpoint、tensor、协议或 `locked_test` 状态。
## 2026-09-09 组会追问文档事实边界

- 当前实现的性能证据与图对象命名必须分开：`physical_edge_state` 承载通信链路、`flow_state` 承载任务数据流是代码事实，但主文档仍把通信关系定义为信息边；目前没有独立证据证明这次语义重分类更优。
- 首种子验收可以支持实体级 RSSM、逐边修正和两阶段训练的单种子可行性，不能支持跨种子、策略闭环或最终方法冻结。
- 截至本地最后可核验动态记录，seed `20260830` 已启动且至少到实体 RSSM 第 6 轮，本地尚无完成验收产物；文档没有推测远端后续状态。

## 2026-09-09 第二个正式 seed 收敛发现

- seed `20260830` 在同一冻结配置、完整 unlocked 数据和独立验收下再次通过 9 项单 seed 数值门，说明 seed `20260831` 的成功不是孤立的一次随机结果。
- 两个已完成 seed 的 link-F1、node-x 和运营指标方向一致；当前证据加强了实体级 latent、因果运动输入、逐边残差和两阶段冻结训练组合的合理性。
- 该证据仍不能替代预注册的三 seed 泛化门。seed `20260832` 未运行，三 seed均值和稳定性未知；P4不能关闭，`locked_test`不能开放。

## 2026-09-09 项目重构发现

- 当前问题的核心不是缺少顶层目录，而是 828 个项目文件、604 个 Python 节点和 802 个 artifact 目录缺少统一机器索引；第一阶段文档只能导航，不能回答反向依赖和批量实验定位。
- 211 个 Python 节点可按命名和证据边界标为历史；当前正式节点对它们的反向引用为 0，但 94 个仍有其他引用、93 个有直接测试，说明“当前主线隔离”与“历史可复现”可以同时成立。
- 物理移动历史模块当前没有可靠回归基线：全量套件包含 AirFogSim 环境缺失、过期 fixture、历史 artifact 权限和 calibration 样本问题。先逻辑归档比改变 import 路径更符合证据保全。
- 12 个所谓 Python 语法错误来自 UTF-8 BOM，而不是代码语法；生成器改为 `utf-8-sig` 后解析错误清零。14 个历史 manifest 仍因权限不可读，属于独立真实限制。
- 现有 GPU runner 已具备 seed 参数、输出目录和 locked-test 路径守卫；不需要创造新训练接口，只需把 `20260832` 的现有入口、冻结前提和用户授权门机器化登记。
- 重构后全量套件已消除根目录结构 failure，剩余 21 个 errors 没有来自本轮索引或当前 P4 核心；不能靠跳过严格合同或修改科研代码来换取表面全绿。

## 2026-09-09 长期协作闭环发现

- 单靠目录索引不能可靠回答“为什么旧方法不用了”；必须保存方法动机、真实结果、弃用原因、替代关系和证据路径。因此新增的历史方法注册表承担语义检索，artifact catalog 承担长尾精确定位。
- 正式数字写进摘要后仍可能随手工维护漂移；将注册表与 acceptance JSON 的文件哈希和 9 项指标逐项比较，才能让“容易找到”和“找到的是正确证据”同时成立。
- 中文口语提问不能只依赖英文关键词；CJK 字符片段匹配与人工问题路由结合后，“我们现在模型是啥”“第三个 seed 什么时候运行”等问法都能稳定落到正确入口。
- 当前达到的是无损的逻辑整理、语义检索和证据防漂移，不是历史文件的物理搬迁。21 个既有全量错误继续阻断物理归档，但不阻断日常问答和证据核实。

## 2026-09-10 AI_CONTEXT 重构发现

- 现有 `docs/` 和机器注册表已能支撑 Codex 检索，但缺少面向 ChatGPT 网页端的低上下文首入口；九文件 AI_CONTEXT 补的是角色化入口，不复制或替代既有证据。
- 静态文件无法可靠硬编码“包含自身的最新 commit hash”；`00_PROJECT_STATE.md` 因此记录创建基线 commit，并规定精确最新版本以 GitHub `main` 的 `HEAD` 为准，避免提交后立即自相矛盾。
- 2026-09-10 当前权限使原先 14 个 artifact 读取错误归零，全量错误由 21 降为 17；这是环境可读性变化，不是科研结果改善。
- `AGENTS.md` 中具体 P4/pre-GPU 临时步骤与“只放长期规则”冲突，已移入 AI_CONTEXT/计划层；永久文件仅保留通用冻结、审计和证据一致性门。
- 剩余 17 个全量 errors 仍阻断历史文件物理迁移，但不来自本次 AI_CONTEXT、注册表或当前 P4 正式路径。

## 2026-09-16 最新 AirFogSim 源轨迹分享包（仅本地交付）

- 用户明确要求最新版，交付 v6 formal source（B层），不含六月A层CSV、训练npz/checkpoint或locked_test；没有重跑仿真。
- 源：code/artifacts/formal_data/pi_jwm_v4_formal_candidate_v6_rb_v1_unlocked_20260821；54条、6场景、每条300步、0.1秒、合计16200轨迹时点。
- 完成：code/artifacts/packages/AirFogSim_raw_data_share_20260916.zip；658389567字节；SHA256=e2b73fc877a3c5117767a20e23c2527281876c0036e42be16835fb99a08fcb59。
- 验证：486项轨迹文件及8项顶层历史清单匹配；54个时间网格通过；ZIP全部800文件逐项解压读取、大小和SHA256通过；所选数据零缺失。
- 限制：历史完整config哈希重建0/54匹配；生成时项目/AirFogSim commit无法确认；最新v6源与当前tensor摘要所指v4不是同一来源声明。现有场景不构成严格单变量对照。上述差异仅报告，没有改写源证据。
- Context Consistency Check只读完成；按用户要求未修改AI_CONTEXT、模型或训练代码；本任务不commit/push。科研P4、第三seed、GPU、同步和locked_test边界不变。
- 本次单一下一动作：用户直接将ZIP交给同学；若需逐值重现历史仿真，先恢复完整历史配置和版本，不能自行重跑。

## 2026-09-18 新定义 Step 1 审计发现

- 当前 `physical_edge` 混合空间关系、CSI、rate、任务数和 RB，与目标严格 Physical/Information 分离不一致；独立 Agent state、Communication relation、Task-Agent typed relation 不完整。
- 旧 tensor/action 路径可复用 offload、RB、return 和 CPU 事件来源，但未闭合完整 Route、逐 RB Comm、CPU action 和 UAV Mobility 的采集、张量、执行及后果。
- 当前 RSSM 为 node/physical_edge/flow/task 都配置 `h,z`；新定义主要只对 Physical/Communication 未知动态设置随机状态，旧 checkpoint/layout 不能直接复用。
- base 内部有逐步规则递推，但实体 RSSM 修正在 base 整段推演后叠加；尚未实现“学习动态→规则更新→重构双图→下一步”的完整单步闭环。
- planner 有逐候选调用与首动作接口骨架，但合法候选、冻结 objective/risk/hard constraints/fallback、真实执行反馈和连续 replanning 仍缺失，只能标 `prototype_only`。
- 旧两 seed、旧 tensor/checkpoint 和旧 planner 结果全部保留为 Historical / Archived evidence，不能用于声称新 `00–06` 已实现或已有性能。

## 2026-09-18 STEP 2

- 已确认：`A_t=(A_t^Route,A_t^Comm,A_t^Comp,A_t^Mob)`，Mob 只控制 UAV，车辆由 SUMO 推进。
- 已确认：完整真实四类轨迹仍缺失；最小闭环不能替代真实场景验收。
- 已记录：本机缺少 `shapely/traci`，完整场景未运行；不影响 scheduler 源码 setter 的最小核验。
# 2026-09-19 Step 2.1 真实 AirFogSim 验收

- 真实非 locked AirFogSim 单轨迹通过四类 scheduler 与真实 `env.step()` 闭环；13 项 checks 全部为 true，未使用 `_MinimalEnv` 或手工 Outcome。
- vehicle traffic `angle` 是 degree，UAV `angle/phi` 按 rad；合同采用实体 `heading`，Mob action 保留 `azimuth_rad`。
- UAV 速度由 0 变 10 m/s 时 AirFogSim 真实 acceleration 为 `-100.0`，记录为仿真器现状，不在本 Step 修正。
- Step 2 专项测试真实为 6 项；generated registry 未纳入 `index-build.tmp.err`/`tmp.err`。

## 2026-09-19 Step 2.2 多步接口事实

- 下一 Decision 可以在下一循环对同一真实环境独立重采，并与上一 Outcome 严格对齐；不能用对象复制替代该证据。
- 6 步中 Route/Comm/Comp 均出现非空和空帧。空帧必须保留 action family 字段、空 entries 与原因；missing field 不等于 no-op。
- Decision 时刻固定 Comp 分配意味着同 slot 新完成传输的任务本 slot CPU 分配为 0，下一 Decision 才进入显式 Comp action；这保持 Decision 可见性边界。
- AirFogSim UAV acceleration 使用 `(last_speed-speed)/interval`，首次加速与常规前向差分符号相反；vehicle 本轨迹 6 行均匹配前向差分。
- `code/artifacts/*` 默认被忽略；仅在实施记录中写路径不能形成 GitHub 机器证据。Step 2.1 v3 和 Step 2.2 小型 JSON/manifest 需显式 force-add。

## 2026-09-19 Step 2.3 因果与字段发现

- `_to_generate_task_infos` 真实包含 `arrival_time_s > decision_time` 的未来任务。它是 simulator internal schedule，不能进入当前 `O_t`、History 或 input-side Entity Index。
- `channel_manager.getCSI` 可在 Decision 前得到 42 条逐 RB channel rows；同 slot transfer event 可给出真实 delivered data。
- `entity.getFogProfile()` 并非每个节点都有 `cpu` 键。合同必须允许 `null + mask + missing reason`，不能把缺失写成 0 capacity。
- `Task.getComputedSize()` 的 post-pre delta 可形成 slot served CPU work；`setTaskReturnRoute` 入队后由真实 env step 推进到 returning/done。
- AirFogSim raw acceleration 保留审计；PI-JWM canonical acceleration 使用只依赖当前/历史的 backward difference，首帧和新实体显式 mask。

## 2026-09-19 Step 2.4 通信 Outcome 发现

- AirFogSim wired service 的直接 slot 数据来自 `WiredNetworkManager.step(interval) -> {task_id: transmitted_bytes}`；`AirFogSimEnv._updateWiredCommunication` 随后调用 `Task.transmit_to_Node` 并在完成时推进 lifecycle。
- Step 2.3 的 `pi_jwm_transfer_events` 原先只覆盖 wireless；把 total 描述为 wireless+wired 与真实采集不一致，现已通过 transport split 修正。
- 已接线 transport 的无服务 slot 是空 map 且 mask=true；接口不可用必须是 null/mask=false/reason。只有两类分量均 observed 时，total 才可用。
- 真实两跳任务在一个 AirFogSim slot 内先完成 wireless、再完成 wired，post-step current node 为 cloud、lifecycle 为 computing；Action route 必须复制保存，否则 simulator 原地消费 route list 会污染记录。
- cloud 节点的现有 FogProfile 无 `cpu` 键；通信验收保持 Comp no-op，未改 Step 2.3 CPU missing 语义。

## 2026-09-19 STEP 3.1F-PATCH

- 发现：`994da0b` 的 Future Action 使用 anchor-only index，在 History union 包含已消失对象时会把合法对象重新编号。
- 证据：disappearing-object fixture 在修复前稳定复现 `History index=1` 被写成 `0`；修复后 validator 逐字段核对 ID 与 static index。
- 边界：audit 的 18 窗口/0 unresolved 仍只是当前非 locked Raw observation，不是正式 Dataset 可用率；STEP 3.2 未授权。

## 2026-09-19 STEP 3.2-PATCH findings

- 旧 batch provenance 只有 path/hash/trajectory/split/schema，无法完整证明 seed/config lineage 与 time-grid；本 Patch 从真实 Raw environment/execution 字段补齐，缺失字段保留 null，不伪造 metadata。
- Raw decisions 为 7 帧、steps 为 6 帧；冻结 slot duration 为 0.1 s。每个 step 的 execution start/end 与 decision/outcome 时间均通过逐项检查。
- Batch audit 必须重新按 Step 3.2 的三条 development trajectory 统计；当前为 12/12/0/0，不能引用 Step 3.1F 的 18-window audit 作为替代。
- task size 单位只能写 `AirFogSim data-unit`；现有源码和记录没有可靠 bit/byte 换算证据。
## 2026-09-20 STEP 4.2B

- `Task._transmitted_size` 每个 hop 完成后 reset；不能作为端到端 Flow remaining。
- `Task.getReturnedSize()` 是 return total requirement，不是 already-returned amount。
- `_task_dependencies` 是 DAG gating，不等于 DepData Flow；当前无 dependency payload/transfer source。
- 旧 `LogicalFlow`/`CarryingHop` 提供动作侧命名，但不是 simulator-issued stable Flow provenance。
- 机器 receipt：`FLOW_CONTRACT_NOT_YET_SUPPORTED`；停止进入 Graph Builder。

## 2026-09-20 STEP 4.2B-PATCH

- Flow verdict 现在由 identity/type/Task/端点/presence/total/remaining/causality/multi-hop/route evidence 计算；篡改 verdict 会被 validator 拒绝。
- dynamic available CPU、storage、wired queue/load/utilization 已移至 `other_information_graph_gaps`。
- simulator-issued Flow ID 缺失改为 implementation fact；logical identity 是否按 `task_id + input/return` 派生留给 researcher decision。
- DepData 当前无真实 transfer process，但不再作为 Input/Return readiness 的直接 blocker；DAG 仍禁止生成 fake Flow。
## 2026-09-20 STEP 4.2C-A-PATCH

- `Task._transmitted_size` remains hop-local, but it need not be the logical remaining source: logical-destination filtered real transfer events causally maintain E2E remaining.
- Final-destination delivery must be counted separately from intermediate hop service to avoid double counting.
- Future action changes do not alter replayed current ledger state; past outcome service remains non-current evidence.
- DepData has no audited transfer process; DAG gating cannot create a fake Flow.
## 2026-09-20 STEP 4.2C-B

- Flow identity is stable across carrying hops and same-destination reroute; only destination change at a clean boundary increments Epoch.
- Raw `O_t` exposes only Ledger state updated through the previous Outcome; same-slot delivery first appears in `O_{t+1}`.
- Real non-locked traces cover direct Input/Return and Input multi-hop. Return multi-hop, reroute, Epoch switch and local no-flow remain fixture-only observations.
- Legacy wireless Return event delivery may exceed observer return_size; frozen min-capping preserves logical conservation and the mismatch is retained as a real-trace limitation.

## 2026-09-20 STEP 4.2C-B-PATCH

- Root cause: Raw amendment used `entry.target_node_id` for logical destination, but ongoing actions expose the current carrying-hop target there.
- Proven Input source: `TaskManager.offloadTask` asserts `route[-1] == target_node_id`; `Task.offloadTo` stores that assigned target and remaining route; `transmit_to_Node` deletes only route element zero after each completed hop.
- Proven Return source: `Task.setToReturnRoute` stores its terminal in `_to_return_node_id`, and observer exposes it as `return_destination_id` independently from remaining route.
- Real `Task_1` service sequence is `UAV_0→RSU_0→cloudServer_4`; both events bind `flow::Task_1::Input::0`, Epoch 0 and destination `cloudServer_4`. Per-hop service sums to twice the payload, while E2E delivered counts only final-destination delivery once.
- Same-destination partial-hop reroute remains unresolved runtime evidence; no new research rule was invented.

## 2026-09-20 STEP 4.2C-C

- Sample/Tensor keeps logical Flow identity separate from hop carrying state: FlowID/Epoch/destination/E2E state come from C-B Raw, while stable history slots, target isolation, presence/mask, and route/holder arrays are collated independently.
- Only five continuous data-unit fields are fit with train-only, mask-aware normalization; identity/category/reference fields are not normalized and capacity overflow is rejected rather than truncated.
- The real cross-slot trace proves one Input Flow remains in one Tensor slot while intermediate hop service does not advance E2E delivery; it does not prove Return multi-hop, reroute runtime, formal capacity, Graph Builder, model, or training behavior.

## 2026-09-20 STEP 4.2C-C-PATCH

- Root cause of the first equality failure: the tensor was built from `apply_flow_normalization()`'s copy while the negative/coverage test passed the pre-normalization sample; equality now derives expected normalized values from the recorded stats when needed, while still comparing raw fields and masks.
- A second semantic gap was found: changing a logical destination/holder/route ID without changing its numeric index could pass numeric-only checks. Tensor metadata now preserves source ID/provenance projections and equality rejects that tamper.
- `target_carrying_*` arrays are a separate future target namespace. They are deterministic ground-truth transition state for this contract only; their presence does not imply a learned prediction head or Graph Builder input.
- Scope remains non-formal and CPU-only: no Graph Builder, model, Loss, Planner, training, GPU, or `locked_test`.

## 2026-09-20 STEP 4.3A

- The frozen current tensor already contains sufficient minimum fields to materialize typed Physical/Information graph objects without reading the simulator or target namespace.
- Physical membership can remain policy-driven: current presence plus valid XYZ admits a node without hard-coding entity classes. Development radius/kNN values are configuration evidence, not a research conclusion.
- Carrying hop endpoints cannot replace logical Flow endpoints. In the current real multihop frame the hop destination may equal the logical destination, so the negative fixture also checks the stage-local hop source.
- GeoComm is an endpoint-Physical dependency for wireless Comm and does not require an identical Physical edge; wired/no-spatial rows remain explicit but invalid.

## 2026-09-20 STEP 4.3A-CONTEXT-PATCH

- Stale current-state wording survived mainly in the `00_PROJECT_STATE` blocker, Tracker header/Step 4.2C-C summary, and generated PROJECT_INDEX/RESEARCH_STATUS summaries.
- Historical statements are still valid for their original Step, but require explicit “at that Step” wording so they cannot override current STEP 4.3A COMPLETE / FROZEN state.

## 2026-09-20 STEP 4.3B

- Current non-Flow Tensor arrays named normalized `*_features` are exactly equal to their raw counterparts; treating them as normalized would be an unsupported assumption. Fixed upstream train-only stats are therefore applied explicitly inside the encoder input path without fitting in forward.
- The stable Flow representation can exclude `e2e_delivered`, Epoch and Flow Index: total plus E2E remaining carry the minimum learned numeric state, while identity/provenance remain structural.
- Strong slot-permutation evidence must permute entity, task and flow rows and consistently remap every consumed reference. Passing only a batch permutation would not prove that numeric IDs are excluded.
- STEP 4.3B establishes encoder wiring and differentiability only. `Z_t^{PI,L_g}` remains distinct from `xi_t^Lat`; no representation-quality or prediction claim follows from deterministic untrained output.
## 2026-09-20 STEP 4.3B-PATCH

- Found and corrected two formula mismatches: P2A value had ignored Agent latent and P2C value had ignored Comm relation latent, despite their gates using joint context.
- Found and corrected incomplete `Z_t^{PI,L_g}` structural output: all eleven STEP 4.3A blocks are now preserved as side information, with semantic equality and tamper rejection checks.
- Removed the Comm CSI width source constant `50`; actual width is read from tensor contract `n_comm_rb` and mismatch fails explicitly.
- Evidence remains untrained CPU development wiring only; no World Model/RSSM/dynamics/training/GPU/locked-test/formal Dataset.
## 2026-09-21 STEP 4.4 communication service sufficiency finding

- `ChannelManagerCP.computeRate` computes nominal per-RB rate from signal/interference/noise and bandwidth, but independently samples Rayleigh outage with `random.rand` and zeroes rate on sampled outage.
- The frozen Decision observer exposes per-RB CSI including fast fading; the actual outage realization is only available in the execution collector with `temporal_role=outcome_only_not_same_frame_decision_input`.
- Therefore `CSI + A^Comm + known parameters` does not uniquely determine actual service. Promoting the outcome outage to input would be future leakage.
- Wired service needs configured capacity and active-flow count. Both exist in simulator state but are not frozen Raw/Tensor inputs, so they are additive gaps rather than unobservable residual evidence.
- Required verdict: `SERVICE_RESIDUAL_RESEARCH_DECISION_REQUIRED`; residual target/architecture remains unselected and STEP 4.4 implementation is stopped.

## 2026-09-21 STEP 4.4 resolved service boundary and model finding

- The researcher selected outage as a conditional known stochastic transition, not Comm latent and not learned residual. The earlier audit verdict is preserved as pre-decision evidence, not current blocker state.
- Complete RB allocation and predicted CSI feed channel-type power/interference/noise SINR. AirFogSim nominal-rate and Rayleigh-outage formulas are explicit; `sample` requires a generator and `expectation` is marked approximate.
- Existing Flow Carrying state exactly derives wired active membership/count in the real manager equality fixture; no extra membership tensor is needed. Capacity itself is read from the causal trajectory config and remains an explicit World Model state input.
- The implemented artifact proves untrained mechanism only. Accuracy, calibration, loss, training, planning and performance remain unknown.
## 2026-09-21 STEP 4.4-PATCH structural closure

- Root cause: canonical recursive receipt reused a route action after step one completed its Flow; the existing absent-index rejection was correct. The builder now uses a contract-valid negative-index no-op for step two while retaining complete RB allocation.
- The final receipt contains 87 required checks: 50 original, 21 service-transition, and 16 structural/rule checks. It passed with zero failures. Task lifecycle acceptance reads the frozen `LIFECYCLE_VOCAB` instead of a numeric literal.
- Evidence remains untrained CPU development only. No Raw/Tensor/graph schema, Loss, Training, Planner, GPU, `locked_test`, or formal Dataset was changed or used.
# 2026-09-21 STEP 4.4-PATCH2

- Root causes were weak hop cap, pre-route rebinding, scalar route revision input, unused Flow categorical embeddings, and static-only DAG receipt.
- Existing Return Flow is not fabricated or omitted: the adapter now binds frozen 4.3A `task_index + flow_type_index=Return`; future-only Return birth remains unsupported and missing required support is an explicit blocking side-state.
- Evidence remains mechanism-level, CPU-only, untrained, and non-formal.
# 2026-09-21 STEP 5.1B

- 当前实现只允许对应 family 的 normalized target 和 mask 进入 target encoder；prior predictor 的接口不接受 target，tampering invariance test 通过。
- family loss 在各自有效 mask 内归一化，空 mask 返回零和零计数；KL 显式返回 raw、free-bit adjusted 和 eligible count，未实现 balancing/overshooting。
- 12-sample CPU receipt 证明原语可执行，不构成训练收敛、性能或正式数据集证据。
