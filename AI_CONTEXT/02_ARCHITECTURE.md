# 当前真实模型架构

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

## 6. 当前不属于正式架构的内容

- `code/src/pi_jwm/formal_candidate_rollout_planner_v1.py` 是 P6 CPU 原型，不是已开放的正式规划器。
- `formal_complete_rssm_world_model_v1.py` 是历史 global RSSM，不是当前候选模型。
- v11 selector/ranking 是历史诊断，不是当前 PI-JWM 主线。

Unverified：任何只由旧 PPT、文件名或历史聊天提出但没有当前代码/config/experiment 支持的架构声明。
