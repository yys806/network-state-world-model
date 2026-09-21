# PI-JWM Implementation Tracker

更新时间：2026-09-21。**Raw Trajectory Layer / 定义 01、当前最小 Dataset/Tensor / 定义 02、STEP 4.2C-B/C-C、STEP 4.3A Typed Dual-Graph Builder 与 STEP 4.3B Dual-Graph Encoder 均已完成并冻结**。STEP 4.4 已完成强制通信 service sufficiency audit，但因 actual wireless service 含 Decision 时不可见的随机 outage realization，机器 verdict=`SERVICE_RESIDUAL_RESEARCH_DECISION_REQUIRED`，World Model 实现尚未开始。Loss、Planner 与 Training 均为 NOT STARTED。

## 当前依据与执行边界

- 目标研究定义：`D:\shen\OB\科研\PIJWM` 中七个 `00–06` Markdown 文件，只读。文件名、大小、行数和 SHA-256 见 `code/artifacts/audit/pi_jwm_new_definition_step01_20260918/initial_snapshot.json`。
- 实现事实：本仓库源码、配置、测试和原始 artifact。笔记中“当前代码已经……”的描述也必须核对。
- 工程工作区：`D:\shen\PKU\PIJWM`；旧 P4/P6/P0–P10 工作流为 **Historical / Archived**，不再是 active workflow。旧结果保留原验收含义，不变成新定义结果。
- 本轮不生成正式大规模数据集，不改变双图、World Model、loss、planner 或 checkpoint；不训练、不使用 GPU、不访问 `locked_test`。
- 主报告：[STEP_01_AUDIT.md](implementation_records/STEP_01_AUDIT.md)；数据附件：[STEP_01_DATA_GRAPH_AUDIT.md](implementation_records/STEP_01_DATA_GRAPH_AUDIT.md)。下表中的 00–06 对应上述源文件章节；详细定位在报告中。

## 总体实施状态

Status 表示工程进度；Reuse 表示与目标定义的匹配类别。`DIRECT_REUSE` 只对该行明确划定的局部机制成立，不等于整个模块符合新定义。

| 模块 | Definition Source | Current Code | Status | Reuse | Main Gap | Next Action |
| --- | --- | --- | --- | --- | --- | --- |
| 总体研究链路 | 00 四–六 | formal model / planner / r6 历史链 | AUDITED / NOT_IMPLEMENTED | STRUCTURAL_CHANGE | 各接口存在不等于新定义端到端闭环 | 先冻结一步数据合同 |
| AirFogSim trajectory | 01 原始轨迹与时间语义 | Raw contract、Step 2.1–2.4 runner/artifact | COMPLETE / FROZEN | MINOR_MODIFICATION | Raw 层已闭合；尚未映射到新 Dataset/Tensor | Step 3 冻结 Dataset/Tensor Contract |
| Decision / Execution / Outcome | 01；02 | Raw contract、causal helper、Step 2.3/2.4 真实 artifact | COMPLETE / FROZEN | MINOR_MODIFICATION | future schedule 已与 `O_t`/History/input index 隔离；wireless/wired slot outcome 已拆分并实测 | Step 3 保持同一因果边界 |
| Dataset / split / masks | 02 | `step3_2_batch_preprocessing_v1.py`、Step 3.2 bundle | COMPLETE / FROZEN FOR CURRENT MINIMAL DATA CONTRACT | MINOR_MODIFICATION | trajectory split、lineage、time-grid、train-only normalization、mask/presence counterfactual 已验收；不是正式大规模 Dataset | 03 所需新增字段只能走 additive extension |
| Model-ready sample / tensor contract | 02 | `model_ready_sample_contract_v1.py`、Step 3.1F artifact、Step 3.2 bundle | COMPLETE / FROZEN FOR CURRENT MINIMAL DATA CONTRACT | MINOR_MODIFICATION | History past A/Y、entity type、union input index、typed target、batch/split/preprocessing 已验收；不是正式大规模 Dataset | 进入 03 前先冻结对象-字段-关系映射 |
| tensor contract | 02；03 | `step3_3_model_input_tensor_v1.py`、`build_step3_3_model_input_tensor_v1.py` | COMPLETE / FROZEN FOR CURRENT MINIMAL DATA CONTRACT | MINOR_MODIFICATION | Past Outcome、完整 Target facts、固定 vocab、Comp 正式字段与 semantic receipt 已验收；03/04 feature selection 和正式容量未决定 | 单独授权双图字段映射 |
| Physical / Information 双图 | 03 | Step 4.1 mapping；Step 4.2A additive Sample/Tensor；Step 4.2C-B Ledger/Raw；Step 4.2C-C Flow Sample/Tensor；Step 4.3A typed builder；Step 4.3B encoder | BUILDER + ENCODER COMPLETE / FROZEN | STRUCTURAL_CHANGE | History temporal encoding、typed directed propagation、P2A/P2C 与 aligned `Z_t^{PI,L_g}` 已实现；topology/encoder config 仅为 development、未研究冻结；这不是 World Model latent | 仅建议 Definition 04 World Model Representation / Dynamics Contract；不自动执行 |
| entity alignment 局部工具 | 02；03 | `airfogsim_tensor_v2.py`、`formal_graph_ops_v1.py` | AUDITED | DIRECT_REUSE | ID/index/mask原则可复用；新增对象映射需扩展 | 保留身份稳定性检查 |
| Route action | 06 §2.1；04 §3.3 | Step 2 Raw + Step 3.3 past/future route tensors | INPUT TENSOR COMPLETE / FROZEN | MINOR_MODIFICATION | route kind/target/task node/hops 与 mask 已映射；尚未接新 graph/model | 后续按新对象路由，未授权 |
| Comm action | 06 §2.1；04 §2.3 | Step 2 Raw + Step 3.3 per-RB action tensors；Step 4.2A per-RB CSI additive tensor；Step 4.3A Comm relation | INPUT TENSOR + CURRENT COMM GRAPH COMPLETE / FROZEN | MINOR_MODIFICATION | CSI/typed relation 已进入 current Graph Builder；动作尚未接 Graph Encoder/model | 后续按独立授权接 Encoder/model，当前不执行 |
| Comp action | 06 §2.1；04 §2.3 | Step 2 Raw + Step 3.3 node/allocated CPU tensors | INPUT TENSOR COMPLETE / FROZEN | STRUCTURAL_CHANGE | `allocated_cpu_per_s` 已张量化；新 graph/model 执行语义尚未接入 | 后续单独授权模型路由 |
| UAV Mobility action | 06 §2.1/5.1 | Step 2 Raw + Step 3.3 UAV index/azimuth/elevation/speed tensors | INPUT TENSOR COMPLETE / FROZEN | MINOR_MODIFICATION | UAV action 已张量化、vehicle motion 仍为 SUMO external；尚未接新 graph/model | 后续单独授权模型路由 |
| action routing | 04 §3.3 | `FormalDualGraphWorldModel.forward` | AUDITED / NOT_STARTED | STRUCTURAL_CHANGE | task→node/agent/task不能覆盖新Phy/Comm/Comp/Mob路径 | 建四类对象路由表 |
| RSSM prior/posterior 局部机制 | 04 §3.1/4.1 | `formal_entity_aligned_rssm_world_model_v1.py` | AUDITED | MINOR_MODIFICATION | 可复用分工与实体维度，不可复用全部布局/解码头 | 新layout完成后迁移 |
| latent layout | 04 §3.2 | 同上 | AUDITED / NOT_STARTED | STRUCTURAL_CHANGE | 旧node/edge/flow/task均有z；新Flow/Task不设独立z | 按Phy/Comm未知动态划分 |
| learned / deterministic boundary | 04 §2；05 §1 | `formal_world_model_loss_v1.py`、model heads | AUDITED / NOT_STARTED | STRUCTURAL_CHANGE | 学习大量可规则恢复的状态/事件和派生指标 | 目标字段→生成方式表 |
| 通信状态充分性 | 04 §4.2 第一个边界 | STEP 4.4 communication service audit / ChannelManagerCP / WiredNetworkManager | AUDITED / BLOCKED | RESEARCHER_DECISION_REQUIRED | nominal pre-outage rate 可由 CSI/RB/已知参数恢复；actual rate 还受随机 per-RB outage realization 置零，且该值只在 Outcome 出现；wired capacity/active-flow count 可走 additive extension | 研究者冻结 outage/effective-service stochastic target 或 residual 边界后，才可恢复 STEP 4.4 |
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
- 新方案的 Raw→Dataset/Tensor→Typed Graph Builder→Dual-Graph Encoder 已按当前最小合同完成并冻结；World Model、loss、训练、candidate proposal 和在线闭环仍未开始。不得把 `Z_t^{PI,L_g}` 外推为 `xi_t^Lat` 或模型实现完成。
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

## Raw Trajectory Layer / 01 已完成并冻结

Step 2.1 v4 完成真实单步验收；Step 2.2 完成真实多步独立重采集与 no-op 连续性；Step 2.3 用 8 个连续真实步完成因果和字段收尾；Step 2.4 在真实 `RSU_0 ↔ cloudServer_4` 有线链路上补齐 Communication Outcome。AirFogSim 未来 schedule 只保留为 internal metadata，不进入 `O_t`、History 或 input-side Entity Index；Decision CSI、CPU capacity/missing mask、wireless/wired split slot service、total 聚合、`setTaskReturnRoute` 和 return lifecycle 均有真实证据。AirFogSim raw acceleration 与 PI-JWM canonical backward difference 使用不同字段，缺历史时为 `null + mask=false`。最终机器证据位于 `code/artifacts/protocols/pi_jwm_raw_contract_causal_complete_v2_20260919/` 和 `code/artifacts/protocols/pi_jwm_communication_outcome_semantics_v1_20260919/`，并要求纳入 Git。

## STEP 3.1F Model-ready History Contract Finalization

最小真实证据位于 `code/artifacts/protocols/pi_jwm_model_ready_sample_v1_20260919/`。History frame `1,2`中明确保留 `O_1+A_1+Y_1+O_2`，Future Action/Target frame `2,3`；input-side index 是整个 History 对象并集，machine policy 为 `history_causal_observable_object_union`，不包含 target-only `Task_7/Task_8`。Future Action 先做 anchor visibility 检查，再使用同一 History-union static index；四类 action、历史 flow/relation/DAG 对齐、固定 presence/mask、ID↔index 校验和 round-trip 均有机器检查。该结果冻结最小 schema 语义，不等于正式数据集完成。Future-reference 观察扫描 JSON 已纳入 artifact provenance，不作自动丢弃或研究决策。

## 当前 Step 3.2 结果

STEP 3.2 已完成最小 non-locked validation：3 条独立 development trajectory、12 个 H=2/L=2 windows，trajectory-level split，train-only mask-aware normalization，deterministic rebuild 和 serialize/load 均有机器证据。该结果不是正式 Dataset、训练或泛化结论。

STEP 3.2-PATCH 已补齐 Dataset isolation evidence：Raw provenance 含真实 trajectory/seed/source SHA/config lineage/time range/slot duration/sample contract；time-grid 与 step start/end 对齐在 window construction 前检查；train/validation trajectory 无交集；batch future-reference audit 独立保存并统计 12/12/0/0，仍为 observation-only；normalization 单位为 `m/s`、`m/s^2`、`AirFogSim data-unit`。

STEP 3.2-PATCH-RECEIPT 已修正最终机器验收：顶层 `passed` 现在是全部 required checks 与 `locked_test/training/gpu/formal_dataset=false` scope checks 的逻辑 AND；负向 fixture 已证明单项失败会使 `passed=false`；provenance contract version 直接复用冻结的 `model_ready_sample_contract_v1.SCHEMA_VERSION`。STEP 3.2 现正式 COMPLETE / FROZEN。

## 当前 STEP 3.3 结果

STEP 3.3F 已补齐 JSON→Tensor 语义：Past Outcome 使用独立 `H-1` 轴，Target 保留 entity/task/flow/service future facts，Comp 读取 `allocated_cpu_per_s`，entity/lifecycle/route/transport 使用固定 vocab，Static entity type 来自 causal History。semantic validation receipt 由 required checks 实际 AND。artifact 位于 `code/artifacts/protocols/pi_jwm_step3_3_model_input_tensor_v1_20260919/`。因此 STEP 3.3 正式 COMPLETE / FROZEN，02 数据集构建与模型输入已按当前最小数据合同冻结；这不等于正式大规模 Dataset 已生成，也不决定 03/04 最终 feature selection。

## 当前 STEP 4.1 结果

STEP 4.1 当时完成 Physical / Information object-field-relation mapping，并以 `DO_NOT_IMPLEMENT_GRAPH_BUILDER_IN_STEP_4.1` 停止；该历史 readiness 已由后续 4.2A/4.2C 数据闭合和 STEP 4.3A builder 实施覆盖。其 CPU capacity/allocation/service/available resource 分离及禁止旧 mixed `physical_edge_state` 的语义仍有效。

## 当前 STEP 4.2A 结果

Step 4.1 中已有可靠来源的 position、wireless per-RB CSI、wired typed relation、CPU static capability、Task demand/progress/elapsed 与 Src/Host/Exec/Ret 已通过独立版本链贯穿 Raw amendment → Sample → train-only preprocessing → Tensor。PATCH 已把 wireless structural validity 与 CSI observability 解耦：CSI missing 不删除 relation，mask=false 的 numeric placeholder 为 0；wired valid/no-CSI 保持合法。三条 development trajectory 共形成 12 个样本；这是机器合同证据，不是正式 Dataset 或容量结论。return size/priority/deadline 已核实存在 simulator observer source，但冻结 Raw 未透传；stable stateful Flow 等仍为 Raw-insufficient。Physical topology 与 graph object 均未实现。机器 artifact 位于 `code/artifacts/protocols/pi_jwm_step4_2a_graph_input_extension_v1_20260920/`。
Step 4.1 中已有可靠来源的 position、wireless per-RB CSI、wired typed relation、CPU static capability、Task demand/progress/elapsed 与 Src/Host/Exec/Ret 已通过独立版本链贯穿 Raw amendment → Sample → train-only preprocessing → Tensor。PATCH 已把 wireless structural validity 与 CSI observability 解耦：CSI missing 不删除 relation，mask=false 的 numeric placeholder 为 0；wired valid/no-CSI 保持合法。三条 development trajectory 共形成 12 个样本；这是机器合同证据，不是正式 Dataset 或容量结论。return size/priority/deadline 已核实存在 simulator observer source，但冻结 Raw 未透传；stable stateful Flow 等仍为 Raw-insufficient。Physical topology 与 graph object 均未实现。机器 artifact 位于 `code/artifacts/protocols/pi_jwm_step4_2a_graph_input_extension_v1_20260920/`。

## 当前 STEP 4.2C-C 结果

STEP 4.2C-C 已完成并冻结 Raw Flow → Model-ready Sample → CPU Tensor additive extension。新增独立 `logical_flow` History-union/target namespace，Flow 与 Carrying state 分离，completed/superseded 与 padding 分离，五个连续字段仅在 `dev_train` 且 `presence=true AND feature_mask=true AND value!=null` 上标准化，capacity overflow 显式拒绝；Sample/Tensor 不重新解释 C-B Raw。随后 `STEP 4.2C-C-PATCH` 补齐 presence-aware stats、History/target Logical/Carrying 四组全字段 semantic equality 和完整 target carrying namespace，并由 receipt 实际 AND 子检查、target namespace、future Epoch、placeholder/bounds/route mask 与 normalization policy。23 项 focused tests、4.2B/4.2A/3.3 回归、真实低 wired capacity 跨时隙 trace、deterministic rebuild/hash、serialize/load 和 receipt/semantic tamper 均通过。artifact 位于 `code/artifacts/protocols/pi_jwm_step4_2c_c_flow_sample_tensor_v1_20260920/`；真实 Return multi-hop、reroute runtime 与 formal capacity 仍未声称。

这是 STEP 4.2C-C 当时的历史范围：`graph_builder=false`、`information_graph=false`、`physical_topology=false`。之后 STEP 4.3A 已完成 Graph Builder；`training=false`、`gpu=false`、`locked_test=false`、`formal_dataset=false` 继续有效。

## 当前 STEP 4.3B 结果

STEP 4.3B 使用完整冻结 History Tensor 编码 Physical/Agent/Task/Flow 的对象级时间状态，并使用 STEP 4.3A current typed graph 完成 Physical/Comm/Flow/Task-Agent/DAG 分族有向传播、relation-wise masked mean、P2A Align 与 wireless-only P2C GeoComm。Logical Flow 与 Carrying 分支独立编码后融合，最终只产生一个 Flow relation latent。输出保持 entity/relation alignment，只到 `Z_t^{PI,L_g}`。artifact 是 `UNTRAINED_DEVELOPMENT_ENCODER_EVIDENCE`；不代表 `xi_t^Lat`、World Model、预测或性能。数值配置与 Physical topology 均为 development-only，`research_frozen=false`。

## 下一步边界

STEP 4.3A Typed Graph Builder 与 STEP 4.3B Dual-Graph Encoder（含 patch）正式 COMPLETE / FROZEN。STEP 4.4 在 communication service gate 暂停，未实现 `xi_t^Lat` 或 rollout。当前唯一下一动作是研究者决定 outage/effective-service 的随机状态或 residual target；决定前不得继续 World Model、Definition 05、训练、GPU 或 `locked_test`。
