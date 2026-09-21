# PI-JWM 项目索引

> 这是项目的导航入口，不替代代码、配置、原始实验产物或机器可读验收文件。
> 生成于 2026-09-08，状态更新于 2026-09-21。定义 01、当前最小定义 02、STEP 4.3A Builder、STEP 4.3B Encoder 与 STEP 4.4 Structured RSSM World Model Contract 已完成并冻结。STEP 5.0 已冻结 Definition 05 的研究决策，但 Loss/Training 实现未开始；旧训练与结果已逻辑归档。

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
9. [`AI_CONTEXT/04_MODULE_MAP.md`](../AI_CONTEXT/04_MODULE_MAP.md)：问题到真实源码的最短路由。
10. [`docs/COLLABORATION_GUIDE.md`](COLLABORATION_GUIDE.md)：用户与 AI 的职责、独立判断、解释和问答规则。
11. [`docs/RESTRUCTURE_ACCEPTANCE.md`](RESTRUCTURE_ACCEPTANCE.md)：重构目标、可观察验收和剩余安全限制。
12. [`task_plan.md`](../task_plan.md)：当前任务、停止门和唯一下一动作。
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
- 当前 Step：STEP 5.0 COMPLETE / FROZEN（决策合同与审计）；Definition 05 实现尚未开始。
- 新定义实现：typed graph、aligned `Z_t^{PI,L_g}` encoder 与未训练 Structured RSSM prior rollout 已实现。现有旧 `entity_aligned_dual_graph_rssm_v1`、P4/P6、两个 seed 和 checkpoint 仍是 Historical / Archived evidence。
- 当前主要缺口：future per-RB CSI target 缺失、future Motion position 未归一化/张量化、Definition 05 loss/posterior/curriculum/training/evaluation 未实现，以及正式候选生成和真实反馈重规划。真实 Return multi-hop 与 same-destination partial-hop reroute 仍缺更强 runtime evidence；formal capacities 未冻结。
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
