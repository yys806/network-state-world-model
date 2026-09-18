# PI-JWM Implementation Tracker

更新时间：2026-09-18。当前 **STEP 1 — New Definition → Current Implementation Audit**；新定义实现尚未开始。审计和工程框架交付后停止，等待研究者检查。

## 当前依据与执行边界

- 目标研究定义：`D:\shen\OB\科研\PIJWM` 中七个 `00–06` Markdown 文件，只读。文件名、大小、行数和 SHA-256 见 `code/artifacts/audit/pi_jwm_new_definition_step01_20260918/initial_snapshot.json`。
- 实现事实：本仓库源码、配置、测试和原始 artifact。笔记中“当前代码已经……”的描述也必须核对。
- 工程工作区：`D:\shen\PKU\PIJWM`；旧 P4/P6/P0–P10 工作流为 **Historical / Archived**，不再是 active workflow。旧结果保留原验收含义，不变成新定义结果。
- 本轮不改变模型、数据、loss、协议、planner 或 checkpoint；不训练、不使用 GPU、不访问 `locked_test`、不自动执行 Step 2。
- 主报告：[STEP_01_AUDIT.md](implementation_records/STEP_01_AUDIT.md)；数据附件：[STEP_01_DATA_GRAPH_AUDIT.md](implementation_records/STEP_01_DATA_GRAPH_AUDIT.md)。下表中的 00–06 对应上述源文件章节；详细定位在报告中。

## 总体实施状态

Status 表示工程进度；Reuse 表示与目标定义的匹配类别。`DIRECT_REUSE` 只对该行明确划定的局部机制成立，不等于整个模块符合新定义。

| 模块 | Definition Source | Current Code | Status | Reuse | Main Gap | Next Action |
| --- | --- | --- | --- | --- | --- | --- |
| 总体研究链路 | 00 四–六 | formal model / planner / r6 历史链 | AUDITED / NOT_IMPLEMENTED | STRUCTURAL_CHANGE | 各接口存在不等于新定义端到端闭环 | 先冻结一步数据合同 |
| AirFogSim trajectory | 01 原始轨迹与时间语义 | `airfogsim_full_dual_graph_collector_v2.py`、`formal_airfogsim_collector_adapter_v2.py` | AUDITED / MIGRATION_NOT_STARTED | MINOR_MODIFICATION | 既有采集器可保留；新字段和 UAV 指令尚未闭合 | 字段可获得性表 |
| Decision / Execution / Outcome | 01；02 | 同上；`formal_airfogsim_window_v1.py` | AUDITED | MINOR_MODIFICATION | 采集顺序可复用；窗口逐字段可见时刻需重新冻结 | 一步索引对照 |
| Dataset / split / masks | 02 | `formal_airfogsim_dataset_v1.py` | AUDITED | MINOR_MODIFICATION | split/mask隔离复用；对象和字段变更后重新构建 | 新 schema 验收后适配 |
| tensor contract | 02；03 | `airfogsim_tensor_v2.py`、v5 tensor contract | AUDITED / NOT_STARTED | STRUCTURAL_CHANGE | Agent/Comm/四类动作与新关系不存在于当前完整合同 | 字段和关系合同 |
| Physical / Information 双图 | 03 | `formal_graph_ops_v1.py`、`formal_dual_graph_world_model_v1.py` | AUDITED / NOT_STARTED | STRUCTURAL_CHANGE | 通信仍混在physical_edge；Agent复用node输入；旧跨图路径不同 | 依合同重构，尚未授权 |
| entity alignment 局部工具 | 02；03 | `airfogsim_tensor_v2.py`、`formal_graph_ops_v1.py` | AUDITED | DIRECT_REUSE | ID/index/mask原则可复用；新增对象映射需扩展 | 保留身份稳定性检查 |
| Route action | 06 §2.1；04 §3.3 | 旧offload/return事件、task索引 | AUDITED / NOT_STARTED | MINOR_MODIFICATION | 旧端点机制可用，完整route语义需映射 | 明确事件→Route |
| Comm action | 06 §2.1；04 §2.3 | RB事件、规则层聚合RB计数 | AUDITED / NOT_STARTED | STRUCTURAL_CHANGE | 逐RB动作未完整进入模型动态 | 明确RB身份与资源约束 |
| Comp action | 06 §2.1；04 §2.3 | 旧CPU内层确定性分配 | AUDITED / NOT_STARTED | STRUCTURAL_CHANGE | CPU是新决策量，旧路径忽略动作尾部并按规则分配 | 定义CPU动作与执行一致性 |
| UAV Mobility action | 06 §2.1/5.1 | AirFogSim mobility API；formal action无对应字段 | AUDITED / NOT_STARTED | MISSING | 仿真器有接口不等于PI-JWM有采集/模型/执行通路 | UAV动作合同；车辆仍外生 |
| action routing | 04 §3.3 | `FormalDualGraphWorldModel.forward` | AUDITED / NOT_STARTED | STRUCTURAL_CHANGE | task→node/agent/task不能覆盖新Phy/Comm/Comp/Mob路径 | 建四类对象路由表 |
| RSSM prior/posterior 局部机制 | 04 §3.1/4.1 | `formal_entity_aligned_rssm_world_model_v1.py` | AUDITED | MINOR_MODIFICATION | 可复用分工与实体维度，不可复用全部布局/解码头 | 新layout完成后迁移 |
| latent layout | 04 §3.2 | 同上 | AUDITED / NOT_STARTED | STRUCTURAL_CHANGE | 旧node/edge/flow/task均有z；新Flow/Task不设独立z | 按Phy/Comm未知动态划分 |
| learned / deterministic boundary | 04 §2；05 §1 | `formal_world_model_loss_v1.py`、model heads | AUDITED / NOT_STARTED | STRUCTURAL_CHANGE | 学习大量可规则恢复的状态/事件和派生指标 | 目标字段→生成方式表 |
| 通信状态充分性 | 04 §4.2 第一个边界 | channel manager / `DeterministicRuleLayer` | AUDITED / OPEN | RESEARCHER_DECISION_REQUIRED | CSI均值+RB计数未证明可还原实际服务；不能默认另加rate头 | 给出所需状态证据再决定 |
| 外生事件 | 04 §2.3/4.2 | 固定entity slot/mask；presence head | AUDITED / OPEN | RESEARCHER_DECISION_REQUIRED | 已知未来场景还是随机到达过程未冻结 | 研究者定观测与生成边界 |
| deterministic rule feedback | 04 §3.3–3.4 | base.forward + RSSM.forward/_decode | AUDITED / NOT_STARTED | STRUCTURAL_CHANGE | base逐步规则存在；RSSM修正在整段规则后，无完整反馈 | 新单步transition统一接入 |
| 动态重构图 | 04 §3.4/4.2 | static endpoints / bearer mappings | AUDITED / NOT_STARTED | MISSING | 未基于预测位置/状态重构新图再进入下一步 | 新图更新接口，尚未实现 |
| training | 05 §2 | `run_formal_dual_graph_gpu_train_v1.py` | AUDITED / NOT_STARTED | STRUCTURAL_CHANGE | base→freeze→RSSM不同于posterior teacher→prior训练 | 新训练阶段合同 |
| loss / overshooting | 05 §1.2 | `formal_world_model_loss_v1.py`、RSSM末尾切片 | AUDITED / NOT_STARTED | STRUCTURAL_CHANGE | 多任务目标与新边界不同；切片KL不等于多起点多距离公式 | 冻结公式→计算图映射 |
| checkpoint selection | 05 §2.2 | `p4_gate_aware_v1`、frozen v2 | AUDITED / NOT_STARTED | STRUCTURAL_CHANGE | 旧门控字典序不同于新validation Pred loss | 新协议单独批准 |
| prediction evaluation | 05 §3 | `formal_world_model_metrics_v1.py` | AUDITED / NOT_STARTED | MINOR_MODIFICATION | MAE/RMSE/NLL/coverage/F1/AUPRC及逐步工具可用，评价对象需改 | 区分未知动态/规则检查 |
| learned candidate generation | 06 §4.3 | planner的callback | AUDITED / NOT_STARTED | MISSING | 尚无新四类动作proposal模型/训练 | 后续单独授权 |
| legality / fallback / warm start | 06 §2.2/3.3/4.3 | CandidateAction.legal、空集raise | AUDITED / NOT_STARTED | MISSING | bool过滤不等于合法性规则；无安全fallback与warm start | 冻结场景约束 |
| candidate rollout | 06 §3.1 | `FormalCandidateRolloutPlanner.plan` | AUDITED / PROTOTYPE_ONLY | MINOR_MODIFICATION | 逐候选调用可复用；共同latent快照/随机评估/新动作未接通 | 保留原型，后续适配 |
| planner objective | 06 §3.2 | prediction_extractor/objective回调 | AUDITED / OPEN | RESEARCHER_DECISION_REQUIRED | 权重、风险形式、未来硬约束、proposal训练目标未定 | 决策前提交备选证据 |
| execute-first-action | 06 §3.3/5.1 | selected_first_action切片 | AUDITED / PROTOTYPE_ONLY | MINOR_MODIFICATION | 只返回第一步，未接真实四类动作执行器 | 后续接入AirFogSim |
| real feedback | 06 §5.2 | 旧R6运行器；formal replan接口 | AUDITED / NOT_STARTED | MISSING | 新planner无真实history/state/latent更新通路 | 后续真实一步反馈验收 |
| replanning / 完整closed loop | 06 §5 | replan(updated_batch) | AUDITED / NOT_STARTED | MISSING | 人工传入batch不等于真实环境反馈闭环 | 后续两决策步端到端验收 |
| 旧结果与checkpoint | 旧P4协议；新00–06 | 两seed验收、v5 tensor、旧实验 | ARCHIVED_IN_PLACE | HISTORICAL_ONLY | 新语义/布局/动作/损失不同，不能外推 | 保留原结果作历史回归 |

## 已完成、未开始与待决策

- 本轮完成范围：Step 1 审计矩阵、只读权限与新工作流、实施记录框架、历史逻辑归档、导航与注册表同步；实际验证见 Step 1 报告。
- 新方案的数据重构、模型、loss、训练、candidate proposal和在线闭环均未开始；不把审计完成写成实现完成。
- 研究者已明确目标：严格Physical/Information划分，四类动作含CPU与UAV，结构化RSSM，只学习未知动态，混合proposal+世界模型选择。无需再次确认这些方向。
- 仍待决定：通信状态不足时的补充字段/必要residual；未知未来到达与离开；proposal训练方式；objective权重/风险/硬约束/fallback；新合同下实验预算与门槛。

## Checkpoint 与结果复用边界

| 变更 | 对旧checkpoint/tensor的影响 | 允许复用 |
| --- | --- | --- |
| Physical/Agent/Comm分拆、特征顺序/单位更改 | 编码器输入语义/尺寸改变；即使同shape也不兼容 | 通用算子、ID/mask工具；不得直接strict-load新模型 |
| Flow/Task去随机状态、Phy/Comm新latent | 参数键和含义改变 | prior/posterior/KL的实现思想；权重迁移需独立证明 |
| 四类动作与逐RB/UAV通路 | action encoder与训练分布改变 | 仿真器接口、旧事件记录作来源证据 |
| 单步规则反馈、动态图、loss/selector | 即使部分权重可加载，旧精度结论也不再适用 | 历史checkpoint复现和回归，不作新验收 |
| 旧两seed单seed验收 | 原数字与原协议保留；并非新00–06性能证据 | 有边界的历史对照；不自动补第三seed |

## 唯一建议的 Step 2（NOT_STARTED）

**冻结一个决策步的原始轨迹字段与四类动作映射合同。** 只对01/02/06建立可核验的 Decision→Action→Execution→Outcome→下一Decision 映射：列字段、单位、可见时刻、缺失语义、实体ID和Route/Comm/Comp/UAV的真实采集/执行入口，使用已有非locked片段或合成样例作低成本检查。输出 schema/映射表与断言，不重构模型、不生成大数据、不训练；通信/外生未决项显式留空并说明，不自行选科学方案。等待研究者审阅授权后才开始。
