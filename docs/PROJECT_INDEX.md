## 2026-10-03 STEP 6.4A 闭环前就绪审计（当前）

STEP 6.4A审计完成（审计证据PASS），CLOSED_LOOP_READINESS=BLOCKED。SEARCH_METHOD=MH-CEM仍只为pure-search backbone，非最终hybrid。局部链路7 IMPLEMENTED/3 PARTIAL/5 MISSING；缺live history-only输入/belief/sidecar、winner动作序列/首动作及真实ID执行桥、研究者fallback和latency决定、两轮真实反馈MH证据。MH B1024搜索median90.506/P90 204.916/P95 284.152/max421.701秒；仿真slot0.1秒不是墙钟deadline，DECISION_LATENCY_REQUIREMENT=UNRESOLVED，尚无在线执行证据。Stage B用途由研究者冻结为online-budget trade-off，不能重选算法；A约30.46–41.85h、B NEW PROPOSAL约10.26–14.06h、C NEW PROPOSAL约1.54–2.11h，仅成本情景。4090_MIGRATION_NOT_YET_JUSTIFIED。Stage B=NOT_STARTED，GPU=NOT_USED，locked_test=false。唯一下一动作研究者确定fallback及decision latency模式，再授权最小接口集成；不自动运行实验。



记录：`docs/implementation_records/STEP_06_4A_CLOSED_LOOP_READINESS_STAGE_B_PURPOSE_FREEZE.md`；机器审计：`code/artifacts/protocols/pi_jwm_step6_4a_readiness_v1_20261003/06_go_no_go_receipt.json`。下方为历史状态，不能用Stage A PASS替代闭环readiness。

---

## 2026-10-03 STEP 6.3D Stage A 最终验收（当前）

STEP_6_3D_VALIDATION_STAGE_A=PASS：960/960 原始结果完成并独立验身份、选法、配对统计及 SHA 归档；名义/实际一步转移均983,040。冻结规则选择 SEARCH_METHOD=MH-CEM，仅表示 Planner v1 structured-search backbone / pure-search baseline，不是最终 hybrid PI-JWM planner 冻结。三方法各160/320可评分H4（50%）；MH 对 S 为78胜/198平/44负，anchor-cluster 95% CI=[0.03125,0.184375]。正式运行40.6074小时、23.6410 cases/hour；GPU RTX3080Ti/CUDA/FP32/batch16 已硬停止。Stage B=NOT_STARTED，locked_test=false。future Return-birth 固定支持限制保留；无未解决执行阻塞。唯一下一动作是研究者审阅本次结果，禁止自动扩展实验。

实施记录：`docs/implementation_records/STEP_06_3D_FORMAL_VALIDATION_STAGE_A.md`；独立验收：`code/artifacts/protocols/pi_jwm_step6_3d_validation_stage_a_v1_20261001/13_stage_a_acceptance_receipt.json`；归档 SHA：`code/artifacts/protocols/pi_jwm_step6_3d_validation_stage_a_v1_20261001/14_stage_a_archive_sha_manifest.json`。本机 D: 保留960 raw和ZIP，远端原始结果保留；Git仅保存小型证据。

---

以下均为历史记录；RUNNING / NOT_SELECTED 等较早状态不代表当前验收状态。

## 2026-10-01 STEP 6.3D FORMAL VALIDATION STAGE A（当前运行）

研究者已单独授权并启动 Formal Validation Stage A（仅 B1024）：64锚点×5 seeds×3方法=960 cases，名义预算983,040。2026-10-01北京时间19:26在RTX3080Ti/CUDA/FP32/batch16启动，启动前源码/768 TRAIN/771文件SHA/64锚点/checkpoint/执行身份全部PASS，Validation起点0。首批2个case已通过身份检查并持久备份；这是运行快照，不是最终效果结论。STEP_6_3D_VALIDATION_STAGE_A=RUNNING，VALIDATION_COMPARISON=RUNNING，SEARCH_METHOD=NOT_SELECTED，Stage B=NOT_STARTED，locked_test=false；future Return-birth限制保留。唯一下一动作是监控并备份Stage A，960完成后独立验收并停止。选择范围仅Planner v1 structured-search backbone/pure-search baseline，不是最终hybrid planner冻结。

记录：`docs/implementation_records/STEP_06_3D_FORMAL_VALIDATION_STAGE_A.md`；启动前机器收据：`code/artifacts/protocols/pi_jwm_step6_3d_validation_stage_a_v1_20261001/01_launch_preflight_receipt.json`；动态监控与本地备份在该运行目录。下方均为较早阶段快照，关于未授权/Validation0的表述不能代表当前运行。

# PI-JWM 项目索引

> 2026-10-01 当前：PRE_VALIDATION_BLOCKER_CLOSURE=PASS，PRE_VALIDATION_AUDIT=PASS，VALIDATION_GO。研究者已明确CEM更新门统计当前轮实际完成的不同可评分四步候选；历史重采样可计入，retained-only不计入，原solver无改动。runner已分primary/diagnostic，1024完成后硬停止，256/512另行授权；六类配对统计及Return/scorer诊断完整。旧TRAIN 768 raw/log/05/06及原执行身份保持，TRAIN_REUSE_VALID=true、TRAIN_RERUN_REQUIRED=false，两种CEM仍(4,0.1)。新3080Ti Validation身份单独冻结。VALIDATION_RESULT_COUNT=0，VALIDATION_COMPARISON=NOT_STARTED，SEARCH_METHOD=NOT_SELECTED，locked_test=false；future Return-birth限制保留。唯一下一动作是研究者审阅并单独授权Stage A；本次不启动GPU或Validation。 入口：`docs/implementation_records/STEP_06_3D_PRE_VALIDATION_BLOCKER_CLOSURE.md` / `code/artifacts/protocols/pi_jwm_step6_3d_pre_validation_blocker_closure_v1_20261001/08_final_pre_validation_acceptance.json`。

> 2026-10-01 当前：STEP 6.3D Validation 前审计完成：PRE_VALIDATION_AUDIT=BLOCKED、VALIDATION_NO_GO。数学、预算/缓存、batch16四步可完成性、随机/顺序独立性及统计 oracle 通过；CEM 旧候选重放触发更新的“newly”定义待研究者确认，runner 缺六类配对/Return/scorer 汇总与 Stage A 停止门。TRAIN MH-vs-S=26胜/58平/12负，仅诊断。TRAIN closure PASS 和两种 (4,0.1) 配置保留；Validation=0、SEARCH_METHOD=NOT_SELECTED、locked_test=false，future Return birth 限制不变。唯一下一动作是明确语义并授权必要修补后再审计；本次不启动 GPU/Validation。 入口：`docs/implementation_records/STEP_06_3D_PRE_VALIDATION_SCIENTIFIC_IMPLEMENTATION_AUDIT.md` / `code/artifacts/protocols/pi_jwm_step6_3d_pre_validation_audit_v1_20261001/09_pre_validation_audit_receipt.json`。

> 2026-10-01 当前关口：`STEP_6_3D_FORMAL_TRAIN_TUNING_CLOSURE=PASS`。768 份 TRAIN 原始结果和本机 SHA 归档已验收，两种 CEM 的参数各为 `(4,0.1)`；Validation 未开始、搜索方法未选。入口：`docs/implementation_records/STEP_06_3D_FORMAL_TRAIN_TUNING_CLOSURE.md` 与 `code/artifacts/protocols/pi_jwm_step6_3d_fixed_budget_search_v1_20260929/train_tuning_closure_acceptance.json`。future Return birth 支持边界保留。

> 2026-09-30 当前关口：STEP 6.3D-PREFLIGHT-PATCH 的静态 Objective 资格、TRAIN-only H4 支持诊断和 CPU batch 等价已通过；正式 6.3D 搜索方法比较暂停，方法未选。入口：`docs/implementation_records/STEP_06_3D_PREFLIGHT_PATCH_OBJECTIVE_H4_BATCH.md` 与 `code/artifacts/protocols/pi_jwm_step6_3d_fixed_budget_search_v1_20260929/25_preflight_patch_acceptance.json`。

> 2026-09-29 STEP 6.3B/6.3C-PATCH：研究者批准 Comm 在当前已有无线 Flow 唯一绑定任务中按 TRAIN 通信结构条件数量选子集；Route 依赖的历史 Comm 行只在投影审计中排除，原始 TRAIN 不变。当前代码、合同和机器入口见 `implementation_records/STEP_06_3BC_PATCH_COMM_TASK_SELECTION_TRAIN_SELF_REPLAY_AUDIT.md`、`code/artifacts/protocols/pi_jwm_step6_3c_candidate_search_protocol_v1_20260929/07_step6_3bc_patch_acceptance.json`。不含优化器或闭环。

> 2026-09-29 STEP 6.3B（当前）：Planner v1 候选语法和 Formal TRAIN 支持准入见 `contracts/PIJWM_STEP_06_3B_STRUCTURED_CANDIDATE_GRAMMAR_V1.md`、`implementation_records/STEP_06_3B_STRUCTURED_CANDIDATE_GRAMMAR_AND_SUPPORT_POLICY.md`；源码 `code/src/pi_jwm/step6_3b_candidate_grammar_v1.py` 与 `step6_3b_candidate_support_v1.py`。仅语法/准入，不含优化器、性能或闭环。

> 2026-09-28 STEP 6.3A（当前）：`STEP_6_3A=PASS` 仅表示 Formal TRAIN 候选支持审计完成；Comm/Comp/Mob 联合结构不支持独立因子化，Candidate Method 仍待研究者决定。记录与收据：`implementation_records/STEP_06_3A_CANDIDATE_SUPPORT_AND_SEARCH_ARCHITECTURE_AUDIT.md`、`code/artifacts/protocols/pi_jwm_step6_3a_candidate_support_audit_v1_20260928/`。

> 2026-09-28 STEP 6.2B-PATCH（当前）：Planner v1 Route 仅显式 no-op；`STEP_6_2B=PASS` 仅为 CPU Objective scorer/比较器合同。入口：`implementation_records/STEP_06_2B_PATCH_ROUTE_NOOP_CLOSURE.md`、`code/src/pi_jwm/step6_0c_planner_action_domain_v1.py`、`code/artifacts/protocols/pi_jwm_step6_2b_patch_route_noop_v1_20260928/`。下方 6.2B BLOCKED 为历史状态。

> 2026-09-28 STEP 6.2B：Objective scorer/comparator 的固定支持 CPU 机制已实现，Comm effort RB 分母已核正；pending Route 与同路径 Host 语义冲突使验收 `BLOCKED_ON_OBJECTIVE_SEMANTICS`。见 `implementation_records/STEP_06_2B_PLANNER_OBJECTIVE_SCORER_AND_COMPARATOR.md` 与对应 13 份机器 receipt。无 Planner 性能或闭环结果。

> 2026-09-28 STEP 6.1：冻结训练模型上的 H4 候选推演机制入口见 `implementation_records/STEP_06_1_TRAINED_WORLD_MODEL_CANDIDATE_ROLLOUT_PREFLIGHT.md`；源码 `code/src/pi_jwm/step6_1_trained_candidate_rollout_v1.py`、CPU 运行脚本与七份 `code/artifacts/protocols/pi_jwm_step6_1_trained_candidate_rollout_preflight_v1_20260928/` 收据。无 objective/winner/闭环性能结论。

> 2026-09-28 STEP 6.2A-CLOSURE：no-retrain checkpoint acceptance、Planner v1 single-hop Route domain 与 `STEP_6_2B_READINESS=READY_FOR_SINGLE_HOP_SCORER_IMPLEMENTATION` 收口入口见 `implementation_records/STEP_06_2A_CLOSURE_NO_RETRAIN_SINGLE_HOP_PLANNER_V1.md` 和 `code/artifacts/protocols/pi_jwm_step6_2a_closure_single_hop_v1_20260928/`；无 scorer、ranking、baseline、closed-loop 或性能结论。

> 2026-09-26 STEP 6.0C：Planner v1 静态 CPU 预算、UAV 六档操作域与 CPU 合同验收见 `contracts/PIJWM_STEP_06_0C_PLANNER_ACTION_DOMAIN_V1.md`、`implementation_records/STEP_06_0C_PLANNER_ACTION_DOMAIN_V1.md`；五份机器凭证在 `code/artifacts/protocols/pi_jwm_step6_0c_planner_action_domain_v1_20260926/`。无模型候选 rollout 或性能结果。

> 2026-09-26 STEP 6.0B：CPU/UAV 可行性来源审计见 `implementation_records/STEP_06_0B_PLANNER_ACTION_FEASIBILITY_SOURCE_AUDIT.md`；五份机器凭证在 `code/artifacts/protocols/pi_jwm_step6_0b_planner_action_feasibility_audit_v1_20260926/`。静态容量不等于动态可用量，UAV 配置/数据范围不等于硬边界，两项 UNKNOWN 保留。

> 2026-09-26 当前增量：STEP 6.0A CPU 候选生成静态合同与机器验收见 `contracts/PIJWM_STEP_06_0A_UNIFIED_CANDIDATE_GENERATION_CONTRACT_V1.md`、`implementation_records/STEP_06_0A_UNIFIED_CANDIDATE_GENERATION_CONTRACT_CPU.md`。5.6B 独立远端训练未接触；下文较早的“未正式训练/GPU”是历史快照。

> 这是项目的导航入口，不替代代码、配置、原始实验产物或机器可读验收文件。
> 生成于 2026-09-08，状态更新于 2026-09-23。STEP 5.5 Formal Dataset v1 已接受；原 H4 interface smoke 只读 runtime 1+1，STEP 5.5-PATCH 另行验证 full-shard 4416/1104 CPU 消费路径与 Return/lifecycle 审计。full/formal training、GPU execution、Planner 与 locked-test 仍关闭，旧训练与结果继续逻辑归档。

## 1. 进入项目的最短路径

ChatGPT 网页端先读 [`AI_CONTEXT/00_PROJECT_STATE.md`](../AI_CONTEXT/00_PROJECT_STATE.md)，再按 `01`–`08` 定位；Codex 开始工程任务时先读 `AGENTS.md`。完整顺序如下：

1. [`AGENTS.md`](../AGENTS.md)：永久治理规则、证据口径和安全边界。
2. [`AI_CONTEXT/00_PROJECT_STATE.md`](../AI_CONTEXT/00_PROJECT_STATE.md)：ChatGPT 当前快照和继续读取入口。
3. [`docs/PIJWM_IMPLEMENTATION_TRACKER.md`](PIJWM_IMPLEMENTATION_TRACKER.md)：新定义实施总表、复用分类和当前 Step。
4. [`docs/implementation_records/STEP_01_AUDIT.md`](implementation_records/STEP_01_AUDIT.md)：Step 1 定义—实现审计主记录。
5. [`docs/implementation_records/STEP_02_4_COMMUNICATION_OUTCOME_SEMANTICS.md`](implementation_records/STEP_02_4_COMMUNICATION_OUTCOME_SEMANTICS.md)：Raw 通信 Outcome 最终真实验收与冻结记录。
6. [`docs/implementation_records/STEP_04_1_PI_GRAPH_OBJECT_FIELD_RELATION_MAPPING.md`](implementation_records/STEP_04_1_PI_GRAPH_OBJECT_FIELD_RELATION_MAPPING.md)：当前对象—字段—关系 mapping、数据 gap 与旧实现冲突。
7. [`docs/contracts_PIJWM_DEFINITION_05_LOSS_TRAINING_EVALUATION_V1.md`](contracts_PIJWM_DEFINITION_05_LOSS_TRAINING_EVALUATION_V1.md)：Definition 05 最新冻结决策、评价口径和禁止项。
8. [`docs/implementation_records/STEP_05_0_DEFINITION_05_DECISION_FREEZE.md`](implementation_records/STEP_05_0_DEFINITION_05_DECISION_FREEZE.md)：STEP 5.0 定向审计、复用分类、缺口和证据边界。
9. [`docs/implementation_records/STEP_05_5_PATCH_FULL_CONSUMPTION_FIXED_SUPPORT_CLOSURE.md`](implementation_records/STEP_05_5_PATCH_FULL_CONSUMPTION_FIXED_SUPPORT_CLOSURE.md)：Formal Dataset full-shard CPU 消费、future Return 与 lifecycle repair 补丁证据。
10. [`AI_CONTEXT/04_MODULE_MAP.md`](../AI_CONTEXT/04_MODULE_MAP.md)：问题到真实源码的最短路由。
11. [`docs/COLLABORATION_GUIDE.md`](COLLABORATION_GUIDE.md)：用户与 AI 的职责、独立判断、解释和问答规则。
12. [`docs/RESTRUCTURE_ACCEPTANCE.md`](RESTRUCTURE_ACCEPTANCE.md)：重构目标、可观察验收和剩余安全限制。
13. [`task_plan.md`](../task_plan.md)：当前任务、停止门和唯一下一动作。
13. [`记录/文件树与证据分层_20260826.md`](../记录/文件树与证据分层_20260826.md)：文件职责和证据等级。
14. [`记录/本地计划表.md`](../记录/本地计划表.md)：项目粗粒度路线和阶段边界。
15. [`记录/PIJWM主文档.md`](../记录/PIJWM主文档.md)：理论、数据、方法和评价定义。
16. [`记录/8.12之后推进.md`](../记录/8.12之后推进.md)：最新推进、失败和阻塞。
17. [`docs/RESEARCH_STATUS.md`](RESEARCH_STATUS.md)：面向人和 AI 的当前研究状态摘要。
18. [`docs/CODE_INDEX.md`](CODE_INDEX.md) 与 [`docs/RETRIEVAL_GUIDE.md`](RETRIEVAL_GUIDE.md)：代码状态和固定检索路线。
19. [`docs/registries/`](registries/)：文件、依赖、实验、结果、历史方法、问题路由、文档权威和延后任务的机器入口。
20. `code/artifacts/` 中与问题直接对应的 manifest、audit、runtime 和原始结果：最终判断必须回到这里核实。

根目录 `PROJECT_CONTEXT.md` 是最近一次交接快照；它用于补充上下文，但如果与更新的过程记录或机器产物冲突，以更新证据为准。

## 2. 当前状态卡片

- 项目：PI-JWM（Physical-Information Joint World Model）。
- AirFogSim：只作为仿真器和数据生成工具，不是 PI-JWM 框架主体。
- 当前路线：研究者最新 `00–06` 目标定义 → 经授权的 Implementation Step。
- 当前 Step：STEP 5.2 COMPLETE / FROZEN FOR CPU DEVELOPMENT TRAINING-LOOP INTEGRATION；STEP 5.3 尚未执行。
- 新定义实现：typed graph、aligned `Z_t^{PI,L_g}` encoder、Structured RSSM prior rollout、5.1B loss/posterior/KL/metric primitives 与 5.2 CPU training loop 已实现。现有旧 `entity_aligned_dual_graph_rssm_v1`、P4/P6、两个 seed 和 checkpoint 仍是 Historical / Archived evidence。
- 当前主要缺口：STEP 5.2 只有少量 CPU optimizer smoke，尚无 tiny-data overfit、full training、性能或校准结果；正式候选生成和真实反馈重规划仍未实现。Route/Comp non-empty development coverage=0；真实 Return multi-hop 与 same-destination partial-hop reroute 仍缺更强 runtime evidence；formal capacities 未冻结。
- `locked_test`：继续封存；`formal_performance_claim_ready=false`。

## 3. 顶层目录地图

| 路径 | 主要职责 | 使用口径 |
| --- | --- | --- |
| `code/src/pi_jwm/` | 可复用 PI-JWM 实现 | 说明代码实际实现了什么，不能单独证明方法已验收 |
| `code/scripts/` | 数据构建、训练、审计、评估入口 | 脚本启动成功不等于实验通过 |
| `code/tests/` | 单元、契约、机制、回归测试 | 只证明测试覆盖的断言 |
| `code/reference/AirFogSim/` | 第三方仿真器 | 只作为数据源/执行环境，不改作框架主体 |
| `code/artifacts/` | 数据、tensor、checkpoint、报告、哈希和机器证据 | 读取 manifest、状态和来源后才能作结论 |
| `记录/` | 理论、计划、进展、交接、迁移证据 | 权威记录与过程/历史记录分开读取 |
| `literature/` | 本地权威文献库 | 文献事实来源，不等于项目实现证据 |
| `meeting/` | 组会材料和汇报资料 | 面向老师的解释材料，不能单独证明实现 |
| `paper/` | 正式论文和论文归档 | 方法冻结后使用，草稿不替代实验证据 |
| `docs/` | 导航、架构、状态、索引、模板和杂项 | 帮助检索，不替代原始证据 |
| `AI_CONTEXT/` | ChatGPT 网页端的精简状态、架构、数据、模块、实验、决策和问题入口 | 快速恢复上下文，不替代源码/config/experiment |

## 4. 当前代码入口

| 研究环节 | 主要入口 | 作用 |
| --- | --- | --- |
| 信息边合同 | `code/src/pi_jwm/information_edge_contract_v4.py` | 定义可审计的信息流字段和缺失语义 |
| 双图采集合同 | `code/src/pi_jwm/full_dual_graph_collector_contract_v1.py` | 分离决策、执行和结果，检查动作合法性 |
| 新 PI graph mapping | `code/src/pi_jwm/step4_1_pi_graph_mapping_v1.py` | 当前数据到严格 Physical/Information 语义、gap 与 forbidden placement；不构图 |
| 正式窗口数据 | `code/src/pi_jwm/formal_airfogsim_window_v1.py` | 读取正式 split、窗口、mask 和静态映射 |
| 正式图编码 | `code/src/pi_jwm/formal_dual_graph_world_model_v1.py` | 物理图、信息图、任务图和跨图消息传播 |
| 确定性规则层 | `code/src/pi_jwm/formal_deterministic_rule_layer_v1.py` | 更新容量、交付量、工作量、生命周期和 DAG 状态 |
| 实体级 RSSM | `code/src/pi_jwm/formal_entity_aligned_rssm_world_model_v1.py` | 节点、物理边、数据流、任务的实体级 prior/posterior |
| 当前 Structured RSSM | `code/src/pi_jwm/step4_4_structured_rssm_world_model_v1.py` | 五类 aligned h、Vehicle/Comm z、Action routing、known stochastic outage、rule feedback 与 dynamic graph rollout；未训练 |
| 损失与指标 | `code/src/pi_jwm/formal_world_model_loss_v1.py`、`formal_world_model_metrics_v1.py` | 训练目标、状态预测和系统指标 |
| P4 门控 | `code/src/pi_jwm/formal_p4_gate_v1.py` | 计算正式单 seed 数值门 |
| 正式 GPU seed 入口 | `code/scripts/run_formal_p4_entity_rssm_gpu_v1.py` | 校验 seed、冻结前提、输出目录和 `locked_test` 边界后调用底层训练器 |
| 底层 GPU 训练器 | `code/scripts/run_formal_dual_graph_gpu_train_v1.py` | 两阶段 base/RSSM 训练和 checkpoint 选择 |
| 项目知识检索 | `code/scripts/query_project_knowledge_v1.py` | 只读查询当前方法、实验、结果、历史方案和延期任务 |
| 规划器原型 | `code/src/pi_jwm/formal_candidate_rollout_planner_v1.py` | 逐候选 world-model rollout 机制原型，尚未开放 P6 |

## 5. 当前证据链

```text
AirFogSim 场景/轨迹
        ↓
严格采集合同：决策 → 执行 → 结果
        ↓
正式 tensor：过去 8 步 + 未来 20 步动作/目标
        ↓
物理图 + 信息图 + 任务图 + 跨图附着/承载关系
        ↓
确定性双图 base
        ↓ 冻结 base
实体级 RSSM prior/posterior
        ↓
checkpoint / 原始 metrics / manifest / 独立验收
        ↓
跨 seed P4 审计
        ↓（尚未开放）
P6 逐候选推演、选首动作、执行后滚动重规划
```

## 6. 历史代码如何定位

- `formal_*`：当前 P4 及正式合同相关实现，但仍需以最新 artifact 判断状态。
- `airfogsim_*`、`full_dual_graph_*`、`information_edge_*`：数据、采集器和双图合同演进。
- `r3_*` 至 `r6_*`：历史阶段实验和策略路径；不能覆盖当前 P4 边界。
- `v6_*` 至 `v11_*`：更早的世界模型、selector 和决策诊断；主要用于追溯失败原因。
- `code/artifacts/audit/`、`experiments/`、`tmp/`：按 evidence guide 读取，不能按 `final`、`best` 或 `latest` 文件名提升证据等级。

## 7. 常见问题的检索路线

| 问题 | 先看 | 再核实 |
| --- | --- | --- |
| 当前用了什么方法 | `RESEARCH_STATUS.md` | 当前模型、冻结协议、run manifest |
| 某个模块在哪里 | 本页入口表 | `code/src/pi_jwm/` 和对应测试 |
| 某个数字从哪里来 | `RESULTS_INDEX.md` | 原始 metrics、checkpoint、manifest、audit |
| 实验做过没有 | `EXPERIMENT_INDEX.md` | 具体实验目录和 run summary |
| 旧方法为什么弃用 | `registries/historical_method_registry.json` | 对应历史实验、失败门和诊断 artifact |
| 能不能进入 P6 | `RESEARCH_STATUS.md` | 最新 P4 gate 和计划记录 |
| 某文件是否当前正式代码 | `CODE_INDEX.md` | `registries/generated/python_dependency_map.json` |
| 第三个 seed 如何续接 | `registries/deferred_work.json` | 冻结协议和两枚 acceptance JSON |
| 文档或历史状态冲突 | `KNOWN_CONFLICTS.md` | `registries/document_authority.json` 和原始证据 |

索引只负责导航。任何影响科研结论的回答，必须回到代码、配置、原始结果和机器可读证据验证。
- Step 3.1F finalized model-ready History and its bounded index/provenance patch: `docs/contracts_PIJWM_MODEL_READY_SAMPLE_TENSOR_CONTRACT_V1.md`, `docs/implementation_records/STEP_03_1F_MODEL_READY_HISTORY_CONTRACT_FINALIZATION.md`, sample artifact and Git-tracked observation-only future-reference audit under `code/artifacts/protocols/pi_jwm_model_ready_sample_v1_20260919/`.

常见问题可以先运行 `$env:PYTHONUTF8='1'; python code/scripts/query_project_knowledge_v1.py --query "问题"`。该入口只读注册表并给出候选路径，不访问 `locked_test`，也不会自动执行实验。

## 8. 训练与同步保护

以下位置在训练或同步期间禁止移动、重命名、删除、覆盖和批量整理：

- `code/artifacts/experiments/`；
- `code/artifacts/formal_tensor/`、`formal_data/`、`protocol/`、`protocols/`；
- 当前 live evidence、staging、checkpoint、predictions、runtime、manifest；
- 远端同步产生的传输目录；
- 任何正在被 Python/GPU 进程写入的文件。

第一阶段只新增导航文件。当前训练虽然已经结束，但后续如果需要归档或移动，仍必须先确认同步静止，并准备逐文件路径映射、哈希和回滚方案。
## Current Planner objective PATCH (2026-09-28)

STEP 6.2A-PATCH reconciles the current 4.2C Flow Ledger and adds a Planner-only causal side-state. STEP 6.2B remains BLOCKED by the 4.2C-C/4.4 route-index mismatch. Start with `docs/implementation_records/STEP_06_2A_PATCH_PLANNER_OBJECTIVE_READINESS_RECONCILIATION.md`, then `code/artifacts/protocols/pi_jwm_step6_2a_patch_objective_readiness_v1_20260928/12_step6_2b_readiness_recomputed.json`.
2026-09-30 STEP 6.3D 3080 Ti 迁移与执行资格：入口为 `docs/implementation_records/STEP_06_3D_3080TI_MIGRATION_QUALIFICATION.md`；机器证据在 `code/artifacts/protocols/pi_jwm_step6_3d_3080ti_migration_v1_20260930/`。正式调参/Validation 比较未运行。
