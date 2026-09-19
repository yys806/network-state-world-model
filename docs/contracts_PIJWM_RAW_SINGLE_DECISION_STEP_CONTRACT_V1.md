# PI-JWM 单决策步 Raw Trajectory 与四类 Action Contract v1

更新时间：2026-09-19。本文冻结 Step 2/2.3 的 Raw Trajectory 工程数据边界，不代表 Dataset/Tensor、世界模型、双图、RSSM、Loss 或 Planner 已实现。

## 1. 冻结范围

单步必须按以下顺序记录：

`Decision_t -> Action_t -> Execution_t -> Outcome_t -> Decision_{t+1}`

时间单位统一为秒，空间位置为米，速度为米/秒，加速度为米/秒平方。实体观测使用 `heading`：车辆来自 SUMO 的 `angle`，单位为 degree；UAV 来自 AirFogSim 的 `angle`，单位为 rad。UAV 的动作字段仍使用 `azimuth_rad`，因为 `setUAVMobilityPatterns` 的 `angle` 按弧度参与运动计算。`trajectory_id` 在一条轨迹内不变，`frame_index` 每个已完成环境步加一，实体和任务用稳定字符串 ID 对齐；缺失对象保留固定槽位并用 `present=false` 或显式 mask 表示。

加速度分成两个不同字段：`raw_simulator_acceleration_mps2` 原样保存 AirFogSim 报告值，只用于 raw simulator observation / audit；`canonical_acceleration_mps2=(v_t-v_{t-1})/delta_t` 是 PI-JWM 的 canonical physical acceleration，只使用当前与历史速度。轨迹首个观测或实体缺少历史速度时，canonical 值必须为 `null`、`observed_mask=false`，并记录 `NO_PREVIOUS_SPEED_IN_TRAJECTORY`，不得填 0。

## 2. Decision_t 可见字段

| 字段 | 来源 | 单位/语义 | 可见时刻 |
| --- | --- | --- | --- |
| trajectory_id, frame_index | collector context/frame counter | 轨迹 ID、决策步序号 | 决策前 |
| decision_time_s, slot_duration_s | `AirFogSimEnv.simulation_time/simulation_interval` | s | 决策前 |
| entities[].entity_id/type/present | AirFogSim entity collections | 稳定 ID、枚举、存在标志 | 决策前 |
| entities[].position/speed/heading/elevation | traffic manager current state | m, m/s, vehicle degree or UAV rad; UAV elevation rad | 决策前 |
| entities[].raw_simulator_acceleration_mps2 | traffic manager current state | m/s^2；AirFogSim 原始审计观测 | 决策前 |
| entities[].canonical_acceleration_mps2/mask/missing_reason | 当前和上一 Decision 的 speed | m/s^2；后向差分，缺历史时 `null + false + reason` | 决策前 |
| vehicle_route_id | SUMO/traffic manager vehicle state | SUMO route ID | 决策前，车辆可为空 |
| channel_rows | `channel_manager.getCSI` | 每 RB 的 dB，按有向实体对和 channel type 对齐 | 决策前 |
| dag_edges | `airfogsim_full_dual_graph_observer_v1._extract_dag_edges` | Decision-time task dependency rows from `task_manager._task_dependencies`; future-only endpoints remain raw/internal and are excluded from input-side index | 决策前 |
| tasks[] | task manager lifecycle collections and Task getters | 只含 `arrival_time_s <= decision_time_s` 的 task ID、Task node、当前节点、生命周期、route、arrival、data/CPU units、已计算量 | 决策前 |
| internal_metadata.future_task_schedule | task manager `_to_generate_task_infos` | `arrival_time_s > decision_time_s`；仅 raw/internal metadata，禁止进入 `O_t`、History 和 input-side Entity Index | 决策前内部审计 |
| node_cpu_capacity_per_s / observation rows | `entity.getFogProfile()['cpu']` | AirFogSim CPU work unit/s；未暴露 `cpu` 的节点使用 `null + observed_mask=false + CPU_NOT_EXPOSED_IN_FOG_PROFILE` | 决策前 |
| n_rb | `channel_manager.n_RB` | RB 数 | 决策前 |

Outcome 字段和同一 slot 的执行结果不得进入 `source_phases=decision`。AirFogSim 的未来任务 schedule 可以保留在 internal metadata，但其 task ID 和字段不得进入当前 `O_t`、History 或 input-side Entity Index。

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

Execution 至少记录 setter kind、subject ID、是否成功、是否调用并完成 `env.step`，以及起止时间。通信 Outcome 显式分成 `wireless_delivered_data_by_task` 和 `wired_delivered_data_by_task`：前者来自 fast fading 后、真实 wireless transfer 前记录的事件；后者来自 `WiredNetworkManager.step(simulation_interval)` 返回的 `{task_id: transmitted_bytes}`。两者单位都是 AirFogSim native data unit/slot；`delivered_data_by_task` 只有在两类 transport 都有可靠观测时才按 task 求和，否则为 `null + observed_mask=false + missing_reason`，不得把不可观测 transport 当作空 map。已接线 transport 在本 slot 没有服务时使用空 map 且 `observed_mask=true`。`served_cpu_work_by_task` 来自同一 Task 的 `getComputedSize()` post-step 减 pre-step，单位为 AirFogSim CPU-work-unit/slot。Outcome 还记录执行后实体/任务快照。执行结束时间和 Outcome 时间必须等于 `decision_time_s + slot_duration_s`；Outcome 快照必须逐字段等于下一决策快照，下一决策的 `frame_index=frame_index+1`。

多决策步轨迹中，`Decision_{t+1}` 必须在下一轮循环重新调用真实 collector，不能复制 `Outcome_t` 对象。四类 action 字段每步都必须存在；该步没有 Route、Comm 或 Comp 时使用空 `entries` 和明确 `no_op_reason`，不能省略字段。Route 的 offload 和 return 必须分别映射到 `setTaskOffloading` 与 `setTaskReturnRoute`。

## 5. 证据边界

Step 2.3 的最终真实验收证据位于 `code/artifacts/protocols/pi_jwm_raw_contract_causal_complete_v2_20260919/`；通信 Outcome 的最终拆分和真实 wired 接线以 Step 2.4 证据 `code/artifacts/protocols/pi_jwm_communication_outcome_semantics_v1_20260919/` 为准。Step 2.4 在非 locked 的真实 AirFogSim 中配置 `RSU_0 ↔ cloudServer_4` 有线链路，观察到 wireless 与 wired 两类真实事件，验证 task progress、lifecycle、空 map 与 missing mask、以及 total 按 task 求和。Step 2.1/2.2 证据保留为前序接口与连续性证据。

## 6. 当前未冻结的科学选择

Dataset/Tensor 如何组织这些已冻结 Raw 字段，以及未来 planner 的 objective/risk/fallback，属于后续 Step；本 Step 不自行补充这些定义。
