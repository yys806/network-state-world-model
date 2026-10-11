# STEP 6.4L-R1 Route–Flow 因果与冻结模型补充报告

## 结论先行

本轮排除了两个诊断工具误判，并锁定了 Task_5 的真实失败分支。AirFogSim 不是在 deadline 到期时失败，而是在 Route 被消费后的同一个 `_updateTask()` 检查中，以旧的 `last_transmission_time=1.2` 触发 `TASK_FAIL_OUT_OF_TTI`；此时还没有新的 RB 请求，因此没有 Return 传输。`UAV_0→UAV_1` 在该 root 的 Raw channel、Tensor/Graph 和 frozen state 中均存在且有效，主要分类是 `TEMPORARILY_NO_SERVICE`，不是物理不可支持或模型通信边缺失。

修正 A 候选资格后，A 完成真实四步递归，原 Scorer 首拒绝为 `UNSUPPORTED_FUTURE_RETURN_BIRTH:Task_5:H1`，`H_eff=0`。修正 B 旁路的字符串/整数实体索引接线后，B 的隔离 Shadow Flow 能完成 H1–H4，原 Scorer 返回 `SCOREABLE/H_eff=4`。B 的 Flow 在模型 anchor 前注入，且 Route action 不是正式 Planner 候选，因此该结果只能说明冻结模型和 Scorer 对该 Shadow 状态可执行，不能证明真实 Route 候选已经可评分，也不能与 A 做 Objective 排序。

## 证据与身份

- 真实 root：`anchor-0006 / Task_5 / frame=6 / simulation_time=1.8`，`LIVE_ROOT_CAPTURE_VALID=true`；与 r39 旧 root 的精确等价仍为 `UNPROVEN`。
- checkpoint：`best.pt` SHA-256 `941ee94131d406de914a79aeda43e929c92727631263d85221615422146a32c9`。
- TRAIN normalization：`stats.json` SHA-256 `1e86bbac6e6ee2d1f1d24e572c5433ed56b1e00c0423bd468fb5150378e6fed5`。
- AirFogSim 实际源文件 SHA 及逐文件证据见 `04_route_comm_support_matrix.json`。
- R1 v1 保留了旧旁路的 B 类型接线失败证据；R1 v2 是修正后的独立 namespace，没有覆盖 v1。

## Phase A：A 分支资格修正

Implementation Fact：原 6.4L 脚本复制 `BoundStructuredStep.action` 后直接构造 `CandidateActionSequence`，没有调用正式 `annotate_domain()`，因此 `score_candidate_set()` 的 `planner_action_domain == "V1"` 门禁抛出 `Planner v1 domain admission required`。R1 调用 `annotate_domain(candidate, context, operational)`，未改动 CandidateDomain、rollout、Objective 或 Scorer 语义。

Observation：R1 A 的 `domain_count=6`，四个 horizon 均完成，Scorer status=`OBJECTIVE_UNSCOREABLE`，`H_eff=0`，唯一记录的首拒绝是 `UNSUPPORTED_FUTURE_RETURN_BIRTH:Task_5:H1`。因此旧 A 异常属于诊断候选缺少官方资格元数据；修正后剩余拒绝是正式 fixed-support 语义。

## Phase B：Task_5 真实失败调用链

源码事实：

1. `TaskScheduler.setTaskReturnRoute()`（`code/reference/AirFogSim/airfogsim/scheduler/task_sched.py:305`）只写入 `env.task_return_routes`。
2. `AirFogSimEnv.step()`（`airfogsim_env.py:219`）每个内部时间步先调用 `_updateTask()`，再调用 `_updateWirelessCommunication()`。
3. `_updateTask()`（`airfogsim_env.py:756`）先消费 `task_return_routes`，调用 `TaskManager.setTaskReturnRouteAndStartReturn()`（约 765 行），然后立即调用 `checkTasks()`。
4. `setTaskReturnRouteAndStartReturn()`（`task_manager.py:1021`）把 Task_5 从 `_waiting_to_return_tasks` 移到 `_returning_tasks`，并设置 `last_return_time=1.8`。
5. `checkTasks()`（`task_manager.py:835`）对 returning/offloading task 使用 `getLastTransmissionTime()`；约 868 行以 `cur_time - last_transmission_time > tti_threshold` 调用 `failOffloadingTask(... TASK_FAIL_OUT_OF_TTI)`。Task_5 的 `last_transmission_time=1.2`，所以在 `cur_time=1.8` 已满足 0.6 > 0.5。
6. `_updateWirelessCommunication()`（`airfogsim_env.py:485`）随后才进入；本次 `activated_offloading_tasks_with_RB_Nos={}`，没有 Task_5 的 RB assignment，也没有新的 transfer event。

Runtime Observation（`RUNTIME/03_runtime_events.json`）：Route setter 后任务进入 returning；随后 `failOffloadingTask` 参数为 `1`，结果 true，失败描述为 `Task fails due to transmission timeout.`；失败发生在 wireless update 之前。Task deadline 为 `arrival 0.6 + deadline 8.598286... = 9.198286...`，远晚于失败时刻 1.8/1.9。

因此 `TASK5_FAILURE_ROOT_CAUSE=PASS`，精确根因是“return route 消费时未同步新的 TTI 计时且当前决策没有 RB assignment，先触发传输超时检查”。这不是 setter 拒绝，也不是当前证据支持的 deadline failure。

## Phase C：UAV_0→UAV_1 支持矩阵

`04_route_comm_support_matrix.json` 给出逐层结果：

| 层 | 证据 | 结论 |
|---|---|---|
| AirFogSim 节点 | `UAV_0/UAV_1` 均为 `U`，类型为 `U2U` | 合法节点类型 |
| Raw channel | `physical::UAV_0::UAV_1::U2U`，`observed_mask=true` | 当前 root 可观测 |
| AirFogSim 当前服务 | Route setter 后没有 RB 请求，失败在 wireless 前 | 本步暂时无服务 |
| Tensor/Graph | relation index `38`，source index `4`、target index `5`，presence/validity/wireless_mask 均 true | 表示存在 |
| Frozen Shadow | B 使用整数模型实体索引后建立 Return slot | 旁路可执行 |

分类：`ROUTE_COMM_SUPPORT_CLASSIFICATION=PASS / TEMPORARILY_NO_SERVICE`。`PHYSICALLY_UNSUPPORTED_HOP=false`、`RAW_OR_GRAPH_SUPPORT_MISSING=false`、`MODEL_REPRESENTATION_MISMATCH=false`。这不表示所有时刻都有服务，只表示该 root 的物理/表示层支持存在，当前失败由 Route→RB→TTI 顺序阻塞。

## Phase D：冻结模型与 Scorer

- A：H1–H4 递归完成，`H_eff=0`，首拒绝 `UNSUPPORTED_FUTURE_RETURN_BIRTH:Task_5:H1`。
- B：使用合法当前 U2U relation 的 Shadow Flow，Flow ID=`flow::Task_5::Return::0`，slot=`32`，source index=`4`，destination index=`5`；H1–H4 递归完成，finite 和 recursive link 检查通过，Scorer=`SCOREABLE`，`H_eff=4`，cohort size=`3`。
- B 的新增 slot 使用 `ZERO_COLD_START_SHADOW_PROPOSAL`；这意味着结构能递归不等于已证明预测可靠。
- B 的正式分数没有提升；A/B 的 anchor 不同（B 在 rollout 前加入 Shadow slot），且 Route B 没有经过正式 Planner domain，因此 `CANDIDATE_OBJECTIVE_COMPARABILITY=NOT_ESTABLISHED`。

`REAL_RETURN_FLOW_H4_COMPATIBILITY=PASS_SHADOW_ONLY`；不是正式 Route candidate PASS。`SCORER_SCOREABILITY=BLOCKED_FOR_FORMAL_A_B_COMPARISON`：A 可诊断但被固定支持拒绝，B Shadow 可评分但不具有正式候选可比身份。

## 研究归类

- **Implementation Fact**：A 漏了 `annotate_domain()`；B 旧脚本把字符串实体 ID 传给整数索引接口；AirFogSim TTI 检查使用旧 `last_transmission_time`。
- **Observation**：A 真实 fixed-support 拒绝；B Shadow H4/Scorer 通过；真实 B 没有新传输并以 TTI 失败。
- **Inference**：当前主要缺口是 Route、通信资源 assignment、TTI/lifecycle 顺序和动态 Flow birth 之间的状态合同，不是“模型完全没有 U2U 边”。
- **Hypothesis**：若研究者希望 Route B 成为正式可评价动作，需要同时定义 Route setter、RB assignment、return start/TTI 计时和新增 Flow 的同一步因果顺序；只修一个层面不足以建立公平排序。
- **Research Proposal**：STEP 6.4M 先冻结上述顺序和候选资格合同，再设计同一 pre-route anchor 的 A/B 正式配对；不要把 Shadow 分数写回正式结果。

## 验收状态

- `A_DIAGNOSTIC_QUALIFICATION=PASS`：官方 domain annotation 后四步递归完成。
- `TASK5_FAILURE_ROOT_CAUSE=PASS`：真实 failure code=1，deadline 未到期，失败早于 wireless。
- `ROUTE_COMM_SUPPORT_CLASSIFICATION=PASS`：primary=`TEMPORARILY_NO_SERVICE`，Raw/Model relation 存在。
- `REAL_RETURN_FLOW_H4_COMPATIBILITY=PASS_SHADOW_ONLY`：B Shadow H4 完成；正式 Route 仍未建立。
- `SCORER_SCOREABILITY=BLOCKED`：A fixed-support 拒绝；B Shadow SCOREABLE 但未提升为正式分数。
- `CANDIDATE_OBJECTIVE_COMPARABILITY=NOT_ESTABLISHED`：不同 anchor/非正式 Route candidate。
- `RETRAIN_REQUIRED=UNDETERMINED`：本轮没有预测误差或训练失效证据。

范围：CPU-only；1 个真实 root、A/B 两个隔离条件；无 GPU、训练、B512、正式 CEM、locked_test。没有新增 env.step 之外的正式策略动作，没有进入 STEP 6.4M。
