# 当前代码架构与新定义差异

> 下列 1–5 节描述被 Step 1 审计的现有旧协议实现，不表示它符合 2026-09-18 的新 `00–06` 目标。总体差异见 `docs/PIJWM_IMPLEMENTATION_TRACKER.md` 和 `docs/implementation_records/STEP_01_AUDIT.md`。

## 总体结构

```text
历史实体状态 + 静态拓扑 + 未来动作
→ FormalDualGraphWorldModel（确定性双图底座）
→ node / physical_edge / flow / task 的实体级 h,z
→ prior 逐步 rollout（验证与部署）
→ 多任务预测头 + 可选确定性规则递推
→ 20 步联合状态与事件预测
```

Source of truth：以下实现事实来自所列源码与冻结协议；运行采用情况还需核对具体实验 `config.json` 和 checkpoint manifest。

## 1. 确定性双图底座

- 文件：`code/src/pi_jwm/formal_dual_graph_world_model_v1.py`
- 配置：`FormalWorldModelConfig`
- 模型：`FormalDualGraphWorldModel`
- 主入口：`FormalDualGraphWorldModel.forward()`
- latent 追踪：`FormalDualGraphWorldModel.forward_with_latent_trace()`
- 实体状态宽度：node=7、physical_edge=5、flow=5、task=8，定义于 `COMPONENT_FEATURES`。
- 图操作：`code/src/pi_jwm/formal_graph_ops_v1.py`，包含物理图、信息图、DAG 与跨图耦合。
- 规则层：`code/src/pi_jwm/formal_deterministic_rule_layer_v1.py::DeterministicRuleLayer.forward()`。

## 2. 实体对齐 RSSM

- 文件：`code/src/pi_jwm/formal_entity_aligned_rssm_world_model_v1.py`
- 配置：`FormalEntityAlignedRSSMConfig`
- 模型：`FormalEntityAlignedRSSMWorldModel`
- 主入口：`FormalEntityAlignedRSSMWorldModel.forward()`
- 当前冻结维度：hidden=32、stochastic=16、history=8、horizon=20、overshooting distance=5。
- entity latent：node、physical_edge、flow、task 各有独立确定性状态 `h` 和随机状态 `z`；agent 是双图底座中的确定性中间 latent，没有独立随机状态。
- `priors` 只依赖已知历史、动作和递推状态；`posteriors` 在训练期结合目标观测。
- prior 与 teacher 各有状态、存在性、DAG、链路、生命周期和能耗输出头。
- 验证/部署路径固定为 prior-only；训练固定使用 posterior teacher。

## 3. 确定性规则层

`DeterministicRuleLayer` 在物理单位中处理动作端点、RB、flow 守恒、任务阶段、CPU 服务、DAG release 等确定性更新，再返回归一化空间反馈下一步。它不应被描述为学习模型自动发现的规律。

## 4. Loss

- 文件：`code/src/pi_jwm/formal_world_model_loss_v1.py`
- 入口：`formal_world_model_loss()`
- 配置结构：`FormalLossWeights`
- 组成：masked Gaussian NLL/MAE、presence 与稀疏事件 BCE、生命周期、DAG、吞吐量/RB/时延等任务项，以及 RSSM KL、teacher reconstruction 和 overshooting。
- 正式权重以冻结协议为准：KL=0.1、teacher reconstruction=0.5、overshooting=0.1、KL balance=0.8。

## 5. 训练结构

- 正式包装入口：`code/scripts/run_formal_p4_entity_rssm_gpu_v1.py::main()`。
- 通用训练器：`code/scripts/run_formal_dual_graph_gpu_train_v1.py::run_gpu_training()` / `run_formal_training()`。
- 两阶段：deterministic base 训练 20 epochs；随后冻结 base，训练 RSSM 最多 40 epochs。
- checkpoint：逐 epoch 保存，使用 `p4_gate_aware_v1` 字典序选择；RSSM 最少 20 epochs，patience=10。
- 正式配置事实：`code/artifacts/protocol/pi_jwm_p4_entity_rssm_frozen_protocol_20260906_v2/protocol.json`。

## 6. 新定义下的实现状态

- STEP 4.1 已冻结目标 mapping：Physical 只包含实体空间/运动与空间关系；Information Nodes 为 Agent/Task，Relations 为 Comm、Src/Host/Exec/Ret、Flow、DAG。
- 现有 `physical_edge` 混合空间、CSI、rate、任务数和 RB，不符合目标严格双图划分，禁止按原语义继续使用。
- STEP 4.2C-C（含 PATCH）已冻结 Flow Sample/Tensor additive extension：C-B Raw logical Flow/Carrying rows 进入独立 `logical_flow` History-union/target namespace；Flow 与 carrying 分离，History/target Logical/Carrying 四组语义逐字段核对；target carrying 明确是 future ground-truth/deterministic-transition state，不是 learned prediction head。该层在其完成时仍不是 Graph Builder；后续 STEP 4.3A 已完成 current typed graph materialization。
- STEP 4.3A 已冻结 current typed dual-graph representation：Physical nodes/relations 与 Agent/Task/Comm/Task-Agent/Flow/DAG 严格分离，Carrying 保持 side state，Align/GeoComm 仅为 structural cross-domain references。builder 不包含编码、传播、聚合或 latent。
- 独立 Agent/Communication/Task-Agent/current stateful Flow graph 尚未实现；四类动作 tensor 已冻结但尚未接入新图或模型。
- 旧 entity RSSM 对 node/physical_edge/flow/task 均维护随机状态，与新定义的未知动态边界不同。
- base 的逐步规则和 RSSM 修正尚未组成“预测→规则→重构图→下一步”的完整闭环。
- 因此现有模型、tensor、checkpoint 和结果为 Historical / Archived；新定义模型尚未实现。

STEP 4.1 source of truth：`docs/contracts_PIJWM_PI_GRAPH_OBJECT_FIELD_RELATION_MAPPING_V1.md` 与 `code/artifacts/protocols/pi_jwm_step4_1_pi_graph_mapping_v1_20260919/`。其中 `DO_NOT_IMPLEMENT_GRAPH_BUILDER_IN_STEP_4.1` 是该 Step 当时的停止门；后续 STEP 4.3A 已在数据合同闭合后完成 builder。

## 7. 当前不属于新定义正式架构的内容

- `code/src/pi_jwm/formal_candidate_rollout_planner_v1.py` 是 P6 CPU 原型，不是已开放的正式规划器。
- `formal_complete_rssm_world_model_v1.py` 是历史 global RSSM，不是当前候选模型。
- v11 selector/ranking 是历史诊断，不是当前 PI-JWM 主线。

Unverified：任何只由旧 PPT、文件名或历史聊天提出但没有当前代码/config/experiment 支持的架构声明。
