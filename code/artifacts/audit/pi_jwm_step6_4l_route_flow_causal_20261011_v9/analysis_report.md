# STEP 6.4L Route–Flow–World Model–Scorer 因果一致性报告

## 状态

- `ROUTE_FLOW_CAUSALITY=PASS_REAL_SETTER_AND_OFFICIAL_LEDGER`
- `FROZEN_WM_H4_COMPATIBILITY=NOT_ESTABLISHED_FOR_TASK5`
- `REAL_SIMULATOR_ALIGNMENT=PASS_ONE_STEP_CAUSAL_OBSERVATION`
- `SCORER_SCOREABILITY=BLOCKED_BEFORE_FORMAL_SCORE`
- `INPUT_ROUTE_SUPPORT_BOUNDARY=SOURCE_AUDIT_ONLY`
- `RETRAIN_REQUIRED=UNDETERMINED`

## 实测证据

真实 AirFogSim 使用 `formal-v1-sim-2026092318-policy-2026092418::anchor-0006`，frame 6，Task_5 为 `waiting_to_return`，精确计算完成，源 `UAV_0`，目的 `UAV_1`，required return size `0.07610754782944204`。

A 保持 Route NOOP。一步后 Task_5 仍为 `waiting_to_return`，无 transfer event。

B 在同一因果前缀中只调用一次真实 `TaskScheduler.setTaskReturnRoute(env, "Task_5", ["UAV_1"])`。setter 返回并写入 `env.task_return_routes`；官方 Ledger 重放生成 `flow::Task_5::Return::0`，epoch 0、slot/index 1、source `UAV_0`、destination `UAV_1`、size `0.07610754782944204`、remaining 相同、active。真实一步没有 transfer event，Task_5 进入 `failed`，因此 Route setter 本身不等于 Return 完成。

A/B 的 normalized causal prefix hash 一致；literal hash 仅因诊断 trajectory id 不同而不同。r39 精确等价仍为 `UNPROVEN`。

冻结模型旁路使用未改变的 checkpoint、normalization、World Model 和原 Scorer。A 在 H4 rollout 的后续 Planner v1 domain admission 处抛出 `ValueError: Planner v1 domain admission required`。B 在生成 Shadow Flow 前因 `SHADOW_ROUTE_COMM_SUPPORT_MISSING` 停止：当前 live model state 没有 `UAV_0→UAV_1` 的可用 communication support。因此没有任何 Shadow 分数或正式分数被记录为 SCOREABLE。

## 证据分类

- **Implementation Fact**：AirFogSim setter 只写入 Route 缓冲；官方 Ledger 只在显式 `route_kind=return` action 处理时创建 Return Flow；正式 Live bridge 仍拒绝非空 Route。
- **Observation**：真实 B Route birth 成功，但一步没有 transfer；模型 B 注入在通信支持检查处 fail-closed；A 后续 rollout 未通过 Planner domain admission。
- **Inference**：Route birth、通信资源调度、World Model Flow slot 支持是三个独立门槛；仅新增 Return Flow 不能保证真实传输或 Scorer 可评分。
- **Hypothesis**：Task_5 的当前阻塞更接近“Route action + communication support + dynamic Flow state”合同缺口，而不是单一 Scorer bug。
- **Research Proposal**：STEP 6.4M 应先定义 Route/Comm/Flow birth 的顺序和状态合同，再做正式候选资格验证；本轮不实施。

## 限制

本轮只执行一个真实 root、A/B 两个 Route 条件、每条件一个真实 step 和有限 CPU 旁路。没有运行正式 CEM、B512、GPU、训练或 locked_test，不能外推 r39 的 3744 个 H4 事件比例，也不能据此判断是否需要重训。
