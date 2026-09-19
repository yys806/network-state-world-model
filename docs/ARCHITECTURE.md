# PI-JWM 架构说明

> 本文解释当前代码和证据中的架构，不重新设计研究方法。若本文与代码、冻结协议或机器产物冲突，以后者为准。
> 2026-09-18：下文描述的是 Step 1 被审计的旧协议实现。最新目标来自只读 `00–06`；现有架构与目标的逐项差异见 `PIJWM_IMPLEMENTATION_TRACKER.md`，不能把下文直接称为新定义已实现。

## 1. 要解决的问题

PI-JWM 研究的是一个同时包含物理网络和信息网络的动态系统。设备会移动，通信链路会变化，任务数据会传输和计算，资源动作会反过来影响后续状态。因此目标不是只预测一个信号，而是学习：

> 在过去状态和未来已知动作条件下，物理实体、数据流和任务如何共同演化。

最终的规划目标是让候选动作分别经过世界模型推演，再比较未来任务结果、代价和风险。当前 active workflow 已切换为新定义 Step 1 审计；旧 P4 只保留历史证据含义。

## 2. 数据对象

### 当前已冻结的 Raw 边界

Step 2.4 已冻结 `Decision -> Action -> Execution -> Outcome -> next Decision` 的真实采集层。通信 Outcome 分为 wireless、wired 和按 task 聚合 total delivered data；wired 直接来自真实 `WiredNetworkManager.step` slot result。空 map 表示已观测但无服务，null 加 missing mask/reason 表示不可恢复。未来到达 task 只保留 internal metadata，不进入当前观察或输入索引；Decision CSI/CPU capacity 和 slot delivered-data/served-CPU 已有真实来源。AirFogSim raw acceleration 与 PI-JWM canonical backward-difference acceleration 使用不同字段和 mask。该事实尚未进入 Dataset/Tensor 或模型架构。

### 物理图

- 物理节点：车辆、无人机、RSU、边缘服务器、云节点等。
- 物理边：有向通信链路。
- 当前物理边 5 维状态：`distance`、`csi_mean`、`rate_sum`、`active_task_count`、`allocated_rb_count`。
- 节点主状态包含位置、速度/加速度、CPU、存储等字段；实体级运动合同另外使用由历史位置因果计算的三维速度和加速度。

### 信息图

- 信息节点：附着在活动物理节点上的通信/计算代理。
- 信息边：任务输入流、结果回传流和有显式 payload 的依赖数据流。
- 正式张量中以 `flow_state` 表示，5 维状态为 `total_data`、`remaining_data`、`delivered_cumulative`、`delivered_this_slot`、`age`。
- 信息流保存端点、任务索引和逐时隙承载它的物理边；它不是物理边的别名。

### 任务和动作

- 任务状态包含任务规模、回传规模、计算量、剩余 deadline、优先级、传输/计算/时延等字段。
- 动作独立记录卸载、RB、回传和 RB 数量/比例；CPU 分配由统一规则层执行。
- 采集器先检查端点、任务生命周期、DAG、RB 范围、冲突和有线/无线规则，再调用仿真器执行。

## 3. 时间和预测

- 历史窗口：8 个时间步。
- 预测窗口：20 个时间步。
- 未来动作作为条件输入。
- 运动特征使用因果 backward difference，不读取未来位置。
- train、validation、calibration 使用正式非锁定数据；`locked_test` 尚未访问。

## 4. 模型链路

### 4.1 双图确定性底座

`formal_dual_graph_world_model_v1.py` 分别编码节点、物理边、信息代理、数据流和任务的历史表示，并通过以下关系传递消息：

- 节点—物理边；
- agent—flow；
- 任务 DAG；
- agent—物理节点附着；
- flow—物理边承载；
- 任务—执行节点。

未来动作被分发到受影响的任务、节点和代理。规则层负责容量、服务量、剩余数据、任务生命周期和 DAG 释放等可解释递推。

### 4.2 实体级 RSSM

`formal_entity_aligned_rssm_world_model_v1.py` 在确定性双图表示上维护实体级随机动力学：

- node：节点状态和运动残差；
- physical edge：逐边链路修正；
- flow：数据流变化；
- task：任务生命周期和 DAG 变化。

训练时 posterior 可以读取目标实体；部署和正式评价只能使用 prior。这样可以检查训练目标是否泄漏到预测，同时避免把所有实体压成一个全局随机向量。

### 4.3 训练目标

`formal_world_model_loss_v1.py` 组合状态 NLL、状态 MAE、存在性/事件、生命周期/DAG、系统指标、KL、teacher reconstruction 和 overshooting。当前正式协议采用两阶段训练：

1. 训练确定性 base 20 个 epoch；
2. 冻结 base，只训练实体级 RSSM，最多 40 个 epoch，按 `p4_gate_aware_v1` 选择 checkpoint。

## 5. 旧协议训练和验收入口（Historical / Archived）

| 层 | 文件 |
| --- | --- |
| 正式 tensor | `code/artifacts/formal_tensor/pi_jwm_v4_tensor_v5_causal_motion_h20_unlocked_20260906/` |
| 冻结协议 | `code/artifacts/protocol/pi_jwm_p4_entity_rssm_frozen_protocol_20260906_v2/protocol.json` |
| GPU runner | `code/scripts/run_formal_dual_graph_gpu_train_v1.py` |
| 一致性审计 | `code/scripts/run_formal_entity_aligned_rssm_consistency_audit_v1.py` |
| batch probe | `code/scripts/run_formal_p4_entity_rssm_gpu_batch_probe_v1.py` |
| 第一枚单 seed 验收 | `code/artifacts/audit/pi_jwm_p4_entity_rssm_seed_20260831_acceptance_20260908/single_seed_acceptance.json` |
| 第二枚单 seed 验收 | `code/artifacts/audit/pi_jwm_p4_entity_rssm_seed_20260830_acceptance_20260909/single_seed_acceptance.json` |
| 延后训练接口 | `docs/registries/deferred_work.json`；只登记，不授权自动启动 |

## 6. 旧 P6 规划器的真实边界

`formal_candidate_rollout_planner_v1.py` 已实现“每个合法候选都送入动作条件世界模型、提取未来状态/任务/代价/风险并选择首动作”的机制骨架，但目前缺少冻结的正式候选生成器、领域合法性集合、风险目标、最终 RSSM 闭环接入和真实执行反馈。因此它是机制原型，不是已完成策略器，也不能写成 P6 已完成。

## 7. 当前不应做的架构解释

- 不把 AirFogSim 叫作 PI-JWM 框架。
- 不把信息边写成物理边的别名。
- 不把单 seed 通过写成最终性能结论。
- 不把执行 sentinel、短 smoke 或 checkpoint 名称写成方法验收。
- 不把只读取 belief 的策略叫作候选动作世界模型规划器。
