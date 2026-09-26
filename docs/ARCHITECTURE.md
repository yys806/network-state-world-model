# PI-JWM 架构说明

> 2026-09-26：STEP 6.0A 新增 CPU 静态 Candidate 结构与编译器，包装正式 Trainer 的 `build_action`。Search/Learned/Hybrid 仅有统一接口和开发 stub；没有 World Model 候选 rollout/评价。5.6B 远端正式训练是独立线路。

> 本文解释当前代码和证据中的架构，不重新设计研究方法。若本文与代码、冻结协议或机器产物冲突，以后者为准。
> 2026-09-23：下文保留 Step 1/历史协议说明；STEP 5.5 已用 60 条真实 trajectory 接通 H2/L4 五类 formal package 与 CPU H4 training-interface smoke。该证据不能称为正式训练、GPU runtime 或性能结果。

## 当前 Definition 05 训练边界（STEP 5.0）

研究者已冻结目标架构；STEP 5.2 已把 STEP 4.3B encoder、STEP 4.4 RSSM/decoder 与 family-specific training-only target encoder 接入 CPU development loop。Motion/CSI 使用独立 component-mask-normalized MSE，Vehicle/Comm latent 使用分族解析 KL；不使用 learned observation variance、latent overshooting、KL balancing 或 Flow/Task/DAG 辅助 loss。训练计划由 posterior warmup 转到 prior-dominant 的 `1→2→4→L` horizon，验证始终 prior-only。

STEP 5.5-PATCH 只扩展数据消费架构：完整 Formal Dataset index 选择 60 个 trajectory shard 中 batch 所需的文件，并按同一 sample ID 切取 Sample/Tensor/Graph/Target；原 Encoder/RSSM/loss 结构不变。旧 `runtime/` 1+1 mini smoke 与 full-shard CPU 路径分别记录。Future Return birth 只进入 target-side fixed-support accounting，不进入 current graph/input。

STEP 5.1A-PATCH 已在 additive target namespace 中把 Motion 修正为 multi-horizon local one-step delta，并固定到 current physical input slots；future per-RB CSI 显式绑定 current model relation slots。5.1B/5.1D 已闭合 loss、posterior teacher、KL、metric 与 paired integration；5.2 已实现配置化 curriculum、optimizer/checkpoint 和 prior-only validation。当前仍仅为 8/4 development CPU smoke，不是正式训练；旧训练器的 staged base-freeze 与旧综合 loss 不属于当前架构。

### STEP 5.1A Future Target 边界

`code/src/pi_jwm/step5_1a_motion_csi_target_contract_v1.py` 从真实 non-locked development sample 与 raw future outcome 构造 `[p_(t+k)-p_(t+k-1), next_speed]` Motion target，以及按 current model communication slot 和 RB identity 对齐的 future CSI target。Motion entity 轴固定使用 `input_entity_index["physical"]`；position delta 只除以冻结 position std，speed/CSI 复用冻结 train-only mean/std。wired、缺失 CSI、invalid relation 和 unsupported future structure 保留身份并使用独立 mask/side metadata。该 target 已由 5.1B/5.1D 的 posterior/loss 路径读取；12 个样本和 receipt 仍只是 development 合同证据，不是正式 Dataset 或性能结果。

## 1. 要解决的问题

PI-JWM 研究的是一个同时包含物理网络和信息网络的动态系统。设备会移动，通信链路会变化，任务数据会传输和计算，资源动作会反过来影响后续状态。因此目标不是只预测一个信号，而是学习：

> 在过去状态和未来已知动作条件下，物理实体、数据流和任务如何共同演化。

最终的规划目标是让候选动作分别经过世界模型推演，再比较未来任务结果、代价和风险。当前 active workflow 已切换为新定义 Step 1 审计；旧 P4 只保留历史证据含义。

## 2. 数据对象

### 当前已冻结的 Raw 边界

Step 2.4 已冻结 `Decision -> Action -> Execution -> Outcome -> next Decision` 的真实采集层；Step 3.2/3.3 已冻结当前最小 Dataset/Tensor。STEP 4.2A 已以独立版本链输入化 position、wireless CSI、wired typed relation、CPU static capability、部分 Task current state 和 typed Task-Agent relation；PATCH 明确 relation validity 独立于 CSI feature observability。STEP 4.2C-B 已在独立 Raw amendment 中实现 stateful Flow：Input logical destination 来自 established route terminal，Return 来自 `return_destination_id`；`target_node_id` 仍是当前 action/carrying-hop target，不能充当 end-to-end destination。真实 Input 两跳已证明普通 hop advancement 不改变 FlowID/Epoch/RouteRevision。STEP 4.2C-C（含 PATCH）已将 C-B Flow/Carrying 贯穿独立 Sample/Tensor，并冻结 presence-aware normalization、四组全字段 semantic equality 与 target carrying future-ground-truth namespace；Physical topology 与 graph builder 尚未实现。CPU capacity 不等于 allocation、actual service 或 dynamic available CPU。

### 新定义 03 的已冻结映射（尚未实现）

- Physical Node：现实实体的 position/motion；Physical relation：relative position/distance/motion。
- Information Nodes：Agent + Task；Relations：directed Comm、Task→Agent 的 Src/Host/Exec/Ret、directed multiedge Flow、Task→Task DAG。
- vehicle/UAV/RSU 同时具有 Physical + Info identity；edge/cloud 的 Physical membership 待研究者决定。
- CSI/rate/RB/task activity 不得作为 Physical relation feature；过去 hop service 不得作为 current Flow remaining state。
- Source of truth：`docs/contracts_PIJWM_PI_GRAPH_OBJECT_FIELD_RELATION_MAPPING_V1.md`。当前不具备直接实现 graph builder 的数据条件。

### 旧协议物理图（Historical / Archived）

- 物理节点：车辆、无人机、RSU、边缘服务器、云节点等。
- 物理边：有向通信链路。
- 当前物理边 5 维状态：`distance`、`csi_mean`、`rate_sum`、`active_task_count`、`allocated_rb_count`。
- 节点主状态包含位置、速度/加速度、CPU、存储等字段；实体级运动合同另外使用由历史位置因果计算的三维速度和加速度。

### 旧协议信息图（Historical / Archived）

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
