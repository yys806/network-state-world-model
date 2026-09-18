# PI-JWM 单决策步 Raw Trajectory 与四类 Action Contract v1

更新时间：2026-09-18。本文冻结 Step 2 的工程数据边界，不代表世界模型、双图、RSSM、Loss 或 Planner 已实现。

## 1. 冻结范围

单步必须按以下顺序记录：

`Decision_t -> Action_t -> Execution_t -> Outcome_t -> Decision_{t+1}`

时间单位统一为秒，空间位置为米，速度为米/秒，加速度为米/秒平方，角度为弧度。`trajectory_id` 在一条轨迹内不变，`frame_index` 每个已完成环境步加一，实体和任务用稳定字符串 ID 对齐；缺失对象保留固定槽位并用 `present=false` 或显式 mask 表示。

## 2. Decision_t 可见字段

| 字段 | 来源 | 单位/语义 | 可见时刻 |
| --- | --- | --- | --- |
| trajectory_id, frame_index | collector context/frame counter | 轨迹 ID、决策步序号 | 决策前 |
| decision_time_s, slot_duration_s | `AirFogSimEnv.simulation_time/simulation_interval` | s | 决策前 |
| entities[].entity_id/type/present | AirFogSim entity collections | 稳定 ID、枚举、存在标志 | 决策前 |
| entities[].position/speed/acceleration/azimuth/elevation | traffic manager current state | m, m/s, m/s^2, rad | 决策前 |
| vehicle_route_id | SUMO/traffic manager vehicle state | SUMO route ID | 决策前，车辆可为空 |
| channel_rows | `channel_manager.getCSI` | 每 RB 的 dB，按有向实体对和 channel type 对齐 | 决策前 |
| tasks[] | task manager lifecycle collections and Task getters | task ID、Task node、当前节点、生命周期、route、arrival、data/CPU units、已计算量 | 决策前 |
| node_cpu_capacity_per_s | `entity.getFogProfile()['cpu']` | AirFogSim CPU work unit/s | 决策前 |
| n_rb | `channel_manager.n_RB` | RB 数 | 决策前 |

Outcome 字段、同一 slot 的执行结果、未来信息不得进入 `source_phases=decision`。

## 3. 四类正式动作与真实入口

动作空间固定为：`A_t=(A_t^Route,A_t^Comm,A_t^Comp,A_t^Mob)`。

| 动作 | 正式字段 | AirFogSim 入口 | 约束 |
| --- | --- | --- | --- |
| Route | `task_id, task_node_id, route_kind(offload/return), target_node_id, route_node_ids` | `TaskScheduler.setTaskOffloading` / `setTaskReturnRoute` | 任务和节点 ID 必须存在，route 终点等于 target |
| Comm | `task_id, rb_indices` | `CommunicationScheduler.setCommunicationWithRB` | RB 不重复且在 `[0,n_rb)` |
| Comp | `task_id, node_id, allocated_cpu_per_s` | `ComputationScheduler.setComputingCallBack` | 节点存在，单节点总分配不超过 Decision 时刻容量 |
| Mob | `uav_id, azimuth_rad, elevation_rad, speed_mps` | `TrafficScheduler.setUAVMobilityPatterns`，映射为 `angle/phi/speed` | 只允许 UAV；每个 present UAV 必须有显式动作。车辆运动由 SUMO 推进，不属于 planner action |

动作必须携带与 Decision 相同的 `trajectory_id/frame_index/decision_time_s`，不能跨轨迹或跨步复用。

## 4. Execution_t 与 Outcome_t

Execution 至少记录 setter kind、subject ID、是否成功、是否调用并完成 `env.step`，以及起止时间。Outcome 至少记录 task 的 delivered data、served CPU work、执行后实体/任务快照。执行结束时间和 Outcome 时间必须等于 `decision_time_s + slot_duration_s`；Outcome 快照必须逐字段等于下一决策快照，下一决策的 `frame_index=frame_index+1`。

## 5. 证据边界

机器可读合同和最小闭环位于 `code/artifacts/protocols/pi_jwm_raw_single_decision_step_contract_v1_20260918/`。该闭环加载仓库真实 AirFogSim scheduler 源码，在最小环境中实际调用四类 setter 并完成一步。它不是完整 AirFogSim 场景验收。

`historical_real_evidence.json` 指向既有非 locked 真实轨迹中的 accepted `cpu_callback+offload+rb+env_step` 记录；该历史证据没有 UAV mobility setter，不能替代新四类真实轨迹。

## 6. 当前未冻结的科学选择

通信状态是否足以规则计算实际 service、外生到达/离开过程、以及未来 planner 的 objective/risk/fallback 仍属于后续研究决策；本 Step 不自行补充这些定义。
