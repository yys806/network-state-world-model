# PI-JWM P2 单步非训练采集器契约设计

**日期：** 2026-08-13  
**状态：** 已确认设计，待按计划实施  
**范围：** 单步真实 AirFogSim 集成与机器可读证据；不包含正式 v4 数据集生成、模型训练、GPU、locked test 或最终方法冻结。

## 1. 设计目标

本步骤要证明一件有限且可核验的事实：给定两个仅在卸载目标或 RB 分配上不同的候选动作，动作被真实写入 AirFogSim 后，环境按真实 `step()` 顺序完成通信、计算和能耗更新，且 CPU 内层规则在候选通信后形成的实际计算任务集合上被调用。

它不证明模型预测、候选世界模型 Rollout、跨轨迹泛化或正式数据训练资格。

## 2. 复用边界与禁止复用

允许复用：

- AirFogSim 的外部 scheduler/callback 接口，不修改 `reference/AirFogSim` 第三方源码；
- `TaskScheduler.setTaskOffloading` 和 `CommunicationScheduler.setCommunicationWithRB` 的动作入口；
- `WirelessTransferEventRecorder` 的 direct per-RB transfer event 记录逻辑；
- `task_resource_conservation_audit.py` 的 task/CPU/energy ledger 及守恒验证；
- `information_edge_contract_v4.py` 的 field registry、Mask/MissingReason、动作 COO、link/RB outcome 验证器；
- 现有严格双图构建和 manifest 哈希工具，但输出必须使用新的版本化目录和 schema。

禁止复用为正式 CPU 规则：

- `CpuPolicyAllocator` 的 deadline/random/max-task 策略；
- `capacity_safe_cpu_allocations` 的最多三任务一次均分；
- 旧 v3 teacher tensor 中的 18 槽语义或旧 `csi_mean/rate_sum` 字段名。

## 3. 输入动作契约

单步采集器接收一个候选动作对象：

```text
CandidateAction {
  candidate_id: string,
  offloads: [{task_node_id, task_id, target_node_id, route_nodes}],
  rb_assignments: [{time_index, flow_index, information_edge_index, rb_index}]
}
```

动作入口要求：

- `candidate_id` 非空且在本次 bundle 内唯一；
- offload 任务、源节点、目标节点和 route 必须能在当前环境查到；
- RB COO 必须是整数 `[record,4]`，无重复，且在 `(time, flow, edge, RB)` 容量内；
- 在调用 AirFogSim setter 前完成 `validate_assignment_coo`；不得依赖 `setCommunicationWithRB` 对越界 RB 的取模行为；
- CPU 不出现在动作对象中。

单步测试至少运行两个候选：相同初始 seed/配置和初始环境状态，只改变一个动作因素。候选动作导致的差异必须可在 transfer event、计算任务集合或 CPU 分配中观测；若差异没有形成，样例应判为未通过而不是补造差异。

实现自审后的规范化夹具固定为：两个候选都选择真实无线远端目标，使用完全相同的全 RB COO，只改变 `target_node_id`。最初考虑的“本地目标/远端目标”组合会使 RB 从不适用变为适用，实质同时改变卸载目标和条件性 RB 动作，不能作为严格单因素候选对；该方案不用于正式 P2 单步证据。两次独立环境还必须具有相同的分叉前环境快照以及 Python/NumPy RNG 状态哈希。

## 4. 真实时序

每个候选执行以下顺序：

1. 在调度阶段写入卸载和 RB 动作；记录 action ledger/COO；
2. 调用真实 `env.step()`；AirFogSim 内部顺序为任务更新、无线通信、有线通信、计算、存储、能耗和时间推进；
3. 无线通信 wrapper 记录每个激活 link/RB 的 CSI 衰减、rate、outage（若直接可读）、planned capacity、remaining-before 和 delivered-data；
4. 计算 callback 将候选通信后的 computing task set 交给 `PIJWM-CPU-Inner-Rule-v1`，记录逐任务 allocation 和 compute before/after；
5. 能耗 wrapper 记录 UAV energy before/after、通信事件输入和能耗方程残差；
6. 用更新后的任务快照、物理节点/边快照和 transfer event 构建单步严格双图；
7. 用 v4 验证器检查字段 Mask/MissingReason、动作 COO、link/RB 守恒和 CPU/energy ledger；任一关键门失败则拒绝发布成功 bundle。

## 5. 信息边最小可靠集

强制写入并验证：

- E0：edge present/type/endpoint/CEP 结构；
- E1：当前动作前逐 RB channel attenuation mean/std、上一槽 active flow count、上一槽 effective rate、上一槽 served data；首帧 prior outcome 使用 `no_history`；
- 当前 action：offload action 与稀疏 RB COO；
- 当前 link outcome：active flow count、effective rate、served data，并满足 assigned-RB rate 与 delivered-data 守恒；
- task/CPU/energy ledger：真实 before/after 与冻结 CPU 规则版本。

E2/E3 可选增强：

- 只有 AirFogSim 在相同 `(edge,RB)` 身份上直接暴露 SINR、interference-plus-noise、rate、outage 或 attenuation 时才写 `valid_mask=true`；
- 不能从 rate 反推 SINR/干扰，不能从衰减冒称 SINR；
- 无直接来源时写零值并配正确 `MissingReason`（如 `not_collected`、`source_absent` 或 `not_applicable`），不得进入训练资格；
- 单步通过不等于 E2/E3 全部可用。

## 6. 输出与证据边界

建议新目录：

```text
代码/artifacts/preflight/pi_jwm_p2_single_step_collector_v1/
  candidate_comparison.json
  action_ledger.json
  transfer_events.json
  single_step_graph.json
  resource_bundle.json
  field_mask_audit.json
  validation_report.json
  summary.json
  manifest.json
```

manifest 必须绑定设计输入、实现代码、测试、AirFogSim 源码和所有输出哈希，并声明：

- `single_step_real_airfogsim_executed=true` 仅表示本 bundle 的一步真实环境执行；
- `v4_collector_implemented=false`、`v4_dataset_complete=false`；
- `model_training_started=false`、`gpu_started=false`、`locked_test_accessed=false`；
- `candidate_rollout_planner_complete=false`、`final_method_frozen=false`；
- 所有样例均为 `contract_fixture` 或 `single_step_nontraining`，`training_eligible=false`。

## 7. 验收门

- 同一初始条件下两个候选均真实执行，且候选差异可在 direct event、计算集合或 CPU 输出中观测；
- 两候选 RB COO 完全一致、只改变远端目标；两者均有正的 direct transfer，同槽进入对应目标计算集合，CPU 实际执行节点等于候选目标；
- CPU callback 与纯函数在每个候选上完全一致，并且 CPU 不是候选无关常量；
- 真实环境时序证据存在：通信事件时间先于同槽 compute before/after，energy after 在同槽更新后记录；
- v4 E0/E1、action、link outcome、ledger 验证全部通过；E2/E3 缺失被正确标记；
- 非法重复/越界 RB 动作在 AirFogSim setter 之前被拒绝；
- 原子发布、失败不发布成功目录、manifest 哈希全部通过；
- 不启动 GPU、不访问 locked test、不生成正式训练数据。
