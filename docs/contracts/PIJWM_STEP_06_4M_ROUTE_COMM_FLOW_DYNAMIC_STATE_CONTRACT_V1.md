# PI-JWM STEP 6.4M — Route–Comm–Flow Dynamic State Contract v1

状态：`DIAGNOSTIC_CONTRACT_ONLY / RESEARCHER_DECISION_REQUIRED_FOR_FORMAL_INTEGRATION`

本文件是 6.4M 的跨层诊断合同，不改变正式 Planner、World Model、Objective、Scorer、CandidateDomain、Formal Dataset、checkpoint 或 AirFogSim。它把当前源码事实、可执行的单跳参考状态机和仍需研究者决定的正式语义分开。

## 1. 解决的问题和范围

6.4L/R1 已证明三件不同的事情：AirFogSim 可以接受 Return Route；Task_5 在 Route 被消费后先触发 `TASK_FAIL_OUT_OF_TTI`；冻结 World Model 可以在预注入 Shadow Flow 后递归 H1–H4。尚未证明的是：同一个 pre-Route anchor 上，Route action 是否能同时产生合法 Flow、通信服务和可比较的正式 Scorer 输入。

v1 首轮只支持当前因果信息可证明的单跳 Input/Return。多跳、reroute、destination-change epoch 和正式新 Flow 学习分布均属于边界或决策门。

## 2. 源码确认的真实事件顺序

| 阶段 | 真实事件 | 时间语义 | 模型/诊断对应 |
|---|---|---|---|
| 决策前 | collector callback 读取当前 Raw、History、Task bucket、channel relation | 当前 `simulation_time`；禁止未来字段 | 共同 pre-Route anchor |
| setter | `TaskScheduler.setTaskReturnRoute()` 把 route 写入 `env.task_return_routes` | 仍在当前决策 callback；Task 尚未移动 bucket | Route intent，尚不是 Flow/服务 |
| setter | `CommunicationScheduler.setCommunicationWithRB()` 写入 `activated_offloading_tasks_with_RB_Nos` | 当前 callback，为下一次 `env.step()` 准备 | Comm assignment；必须有当前可执行 Task |
| `env.step()` 前半 | `_updateTask()` 消费 route，调用 `setTaskReturnRouteAndStartReturn()` | step 内部；waiting→returning；设置 `last_return_time` | Route effect / Task lifecycle |
| `env.step()` 同一内部 tick | `checkTasks()` | wireless 之前 | TTI 和 deadline fail-fast；Task_5 此处使用旧 `last_transmission_time` |
| `env.step()` 后半 | `_updateWirelessCommunication()` → `_allocate_communication_RBs()` → rate → transfer | 同一内部 tick | 只有已经进入可枚举 transmitting bucket 且有 RB assignment 才可能服务 |
| transfer 后 | `transmit_to_Node()`、`finishOffloadingTask()` 或 `failOffloadingTask()` | 同一 tick | Task/Flow outcome；Ledger 只能记录已建立、已观察的事件 |
| tick 结束 | computation/storage/energy、simulation time 增加、最终 `checkTasks()` | 下一次 decision 前 | fresh observation / causal prefix |
| Ledger | `amend_raw_with_causal_flow_ledger()` 在 decision 的 route action 已存在时建立逻辑 Flow | decision-before-action 的当前因果记录；不是模型未来预测 | Flow identity、epoch、route revision 和守恒 |

直接源码依据：`airfogsim_env.py:219–260,485–490,624–675,756–767`；`task_manager.py:835–885,1021–1035`；`task.py:153–180,182–195,259–290,494–502`；`communication_sched.py:58–69`；`step4_2c_b_causal_flow_ledger_raw_v1.py:206–226,238–301,418–473`。

### 2.1 已确认的时序边界

1. Return setter 不直接创建 AirFogSim transmission；它只暂存 route。
2. `startToReturn()` 设置 `last_return_time`，但 `checkTasks()` 的 TTI guard 使用 `getLastTransmissionTime()`；因此新生 Return 可能继承旧的 transmission clock。
3. wireless execution 中对 returning task 的 rate 执行使用 `last_return_time`，这和前面的 TTI guard 不是同一个时间字段。
4. collector 当前在 callback 中先尝试 Route，再按已经位于 offloading/returning bucket 的 task 设置 RB。`waiting_to_return` 的 Task_5 在该时刻不在 transmitting 列表，所以不会获得同一 callback 的 RB assignment。
5. Raw channel/Graph relation 存在只证明 hop 可表达和可观测，不证明本 tick 已获得 RB/service。

## 3. v1 状态合同

### 3.1 共同状态和动作时序

```text
pre_route state
  → Route / Comm / Comp / Mob action declaration
  → ID、slot、endpoint、route、support、deadline 合法性检查
  → action setter / isolated transition
  → Flow birth or existing Flow update
  → Task lifecycle and TTI check
  → communication assignment and service
  → transfer / completion / failure
  → next causal observation
```

在当前 AirFogSim compatibility mode 中，`TTI check` 先于 wireless service；参考状态机默认重现这个顺序，不重置 `last_transmission_time`。若研究者选择把新 Return 的 TTI 起点改为 route-consumption time，这是新的业务语义，必须进入 Decision Gate，不能由诊断工具隐式修复。

### 3.2 Task 合同

必需字段：`task_id`、`lifecycle`、`current_node_id`、`return_destination_id`、`required_returned_size`、`task_cpu_work`、`computed_cpu_work`、`last_transmission_time`、`last_return_time`、`arrival_time`、`deadline`。

- 计算完成是 `computed_cpu_work >= task_cpu_work` 的精确业务判断；FP64/FP32 shadow 只能作为诊断，不可改变正式状态。
- `required_returned_size == 0` 或计算宿主等于 Return destination 时不创建 Return Flow。
- `waiting_to_return` 是已计算但尚未消费 Return Route 的状态；`returning` 是 route 已消费、正在传输的状态；两者不能混称。
- `failed` 必须保留 failure code 和触发阶段；不能将 TTI failure 改写为 deadline failure。

### 3.3 Flow 合同

`FlowID = flow::<TaskID>::<FlowType>::<Epoch>`；`slot` 是当前 tensor support 中的稳定整数位置。Input 和 Return 共用以下不变量：

- birth 必须有当前 Task、合法 source/destination、非空单跳 route、当前 Raw/Graph communication relation；缺任一项 fail closed。
- 同一个 `(TaskID, FlowType, Epoch)` 不能重复 birth；slot 不能覆盖已有对象；容量不足返回 `FLOW_SLOT_CAPACITY_EXHAUSTED`。
- `known` 表示身份/结构已进入当前 support，`presence` 表示 Flow 仍有正剩余，`active/carrying_active` 表示当前 hop 可服务，`COMPLETED` 表示 logical Flow 已完成；这些字段不可互换。
- `route_revision` 只在正式允许的 clean-boundary reroute 改变；普通 hop advancement 不增加 revision。
- `e2e_delivered + e2e_remaining == total_data`；中间 hop 的服务不增加 logical E2E delivered，只有 logical destination hop 增加 E2E delivered。
- destination change 必须产生新 epoch，并以旧 Flow 剩余量为新 Flow total；v1 不把它当成普通 route update。
- `waiting_to_return` 没有 Flow 是可表达的当前状态；它不是数据缺失的自动证明，也不代表未来 birth 已获固定 support。

### 3.4 World Model 动态对象接口

隔离接口采用以下顺序：

1. 业务 ID 映射到当前 `physical/entity/task` 整数 index；字符串 ID 不能直接写入张量。
2. 在不覆盖旧 slot 的前提下分配新 Flow slot；同步 `flow_known/presence/task_index/source/destination/type/status/identity/route_revision/holder/hop/remaining/route mask`。
3. 复用当前 `comm_presence & comm_validity & comm_wireless_mask`；缺通信 relation 时拒绝 birth，不能虚构边。
4. rebuild Graph，使 Flow relation、Task relation、Comm endpoint 和 masks 对齐。
5. 新 latent hidden 的 zero cold start 只能标记为 `SHADOW_PROPOSAL`；结构可递归不等于预测可靠，也不自动代表训练分布内。
6. H1–H4 后续动作必须重新通过当前 predicted state 的 ID、support 和 domain 检查；不得读取未来 target 或真实执行结果补字段。

正式模型当前仍声明 `future_return_birth_supported=false`；因此本接口只提供隔离参考，不接入正式 `one_step()` 或 Planner。

### 3.5 Planner/Objective 合同

- A/B 必须从同一份 pre-Route state、graph、latent、cohort、side-state 和时间开始。
- A 的 Route-NOOP 不改变 anchor；B 的 Flow birth 只能由 B 的 Route action effect 触发，不能预先向 B anchor 注入 Flow。
- Route/Comm/Comp/Mob 的五项目标仍为 `(N_DDL,A_DDL,J_Delay,J_Burden,J_Effort)`；v1 不修改分母、mask、排序或 `H_eff` 定义。
- 若新 Flow 尚未进入 anchor-frozen support，原 Scorer 必须返回明确 rejection，而不是用 Shadow 分数替代正式结果。
- TTI failure 作为未来 Task 状态和 failure evidence 进入诊断；若要改变为新 Return clock，需要研究者决定。
- Route effort 当前没有已冻结的独立代价来源；不能擅自加入第六个目标或隐藏处罚。

## 4. 隔离参考状态机接口

实现位于 `code/src/pi_jwm/step6_4m_route_comm_flow_contract_v1.py`，只依赖标准库和已有 Ledger identity helpers：

- `ContractState.route_noop()`：返回同一 pre-Route 状态的明确 A 分支。
- `ContractState.apply_route()`：只在 B action effect 时创建 Input/Return Flow。
- `ContractState.assign_comm()`：检查当前 Flow、当前 hop、通信 relation 和全局 RB 范围。
- `ContractState.advance_tick()`：按 compatibility order 先做 TTI check，再应用显式 service event。
- `ContractState.apply_transfer()`：做 holder、hop、route revision、E2E 守恒和 Task completion 检查。
- `common_anchor_fingerprint()`：拒绝不同 pre-Route anchor 的候选比较。
- `bind_new_flow_to_model_state()`：只读校验业务 ID→整数 index、slot、Comm relation 和 shape；不调用正式模型。

该状态机不会调用 AirFogSim、不会修改 checkpoint、不会改变正式 Scorer，也不会自动创建 Route 或 RB。

## 5. 模块影响矩阵

| 层 | 当前实现 | 6.4M 诊断动作 | 正式集成状态 |
|---|---|---|---|
| AirFogSim | setter、TTI、RB、wireless 已实现 | 只读 trace/compatibility replay | 不修改 |
| Causal Ledger | Input/Return birth、epoch、守恒已实现 | 复用 identity/transition 规则 | 不修改 |
| Raw/Tensor/Graph | fixed current support | 只校验 mapping/mask/slot | 不扩正式容量 |
| World Model | route rule 不创建 future Flow | Shadow/参考 adapter | `RESEARCH_DECISION_REQUIRED` |
| CandidateDomain | Route 当前受 fixed support 限制 | 只测试 admission 负例 | 不修改 |
| Scorer | fixed-support `H_eff` 和五目标冻结 | 原样调用、记录拒绝 | `RESEARCH_DECISION_REQUIRED` |
| Planner | `Route=EXPLICIT_NOOP_ONLY` | A/B 只在隔离 harness | 不解除 |

## 6. 集中 Researcher Decision Gate

| 决策项 | 源码事实 | 候选方案 | 推荐 | 不决定时的阻塞 |
|---|---|---|---|---|
| 新 Return TTI 起点 | `checkTasks()` 使用旧 `last_transmission_time` | 保持 AirFogSim compatibility；或新 Return clock | 先保持 compatibility，单独做语义实验 | 无法公平定义 Route B 是否获服务 |
| 同一时隙 Route→Comm | waiting task 在 callback 不在 transmitting bucket | 下一时隙服务；或增加显式 pending-route comm adapter | 先定义 pending-route adapter 的正式边界 | Route action 与 RB assignment 不可绑定 |
| Flow birth 所在层 | Ledger 可记录当前 route，WM 不支持 future slot | Planner-side dynamic slot；重新训练；保持 fixed support | 先做 Planner-side contract shadow | Scorer 仍拒绝 future birth |
| cold-start hidden | Shadow 可 zero-init 递归，可靠性未知 | zero-init development；统计 prior；重训 | 先只做 distribution audit | 不能宣称预测可靠 |
| Objective 对新 Flow | 现有 burden mask anchor-frozen | 保持拒绝；扩展 mask；重定义分母 | 研究者冻结前保持拒绝 | A/B 不能比较 |
| Route effort | 现有五目标没有 Route 独立来源 | 不计；映射 Comm/Comp；新增目标 | 暂不新增目标 | 无法声称 Route cost 已定义 |
| Input Route | Ledger 有 Input birth，但联合 Planner 未验证 | 只做接口对照；同样动态机制；暂缓 | 本轮只做不变量测试 | Input/Return 合同不对称 |

## 7. 状态与证据边界

- `SOURCE_EVENT_ORDER_VERIFIED=PASS`：真实 AirFogSim 调用顺序已由源码和 R1 runtime receipt 双重确认。
- `ROUTE_COMM_TTI_CONTRACT=RESEARCH_DECISION_REQUIRED`：compatibility 顺序已可执行，但是否改变 TTI 起点和同隙 Comm 仍需决定。
- `INPUT_RETURN_FLOW_STATE_CONTRACT=PASS（v1 单跳参考）`：身份、slot、epoch、守恒和失败边界可执行；不是正式模型合同。
- `DYNAMIC_STATE_INTERFACE_READINESS=PASS（isolated reference）`：状态机和 binding 可验证；正式 WM dynamic birth 未接入。
- `COMMON_ANCHOR_CAUSALITY=PASS（reference A/B）`：参考状态机从同一 anchor 派生 A/B；R1 的旧 B 结果仍不可比。
- `FROZEN_WM_H4_COMPATIBILITY=NOT_ESTABLISHED（formal Route）`：R1 只有 pre-injected Shadow H4 PASS。
- `SCORER_COMPARABILITY_READINESS=BLOCKED`：原 Scorer 对 future birth 保持拒绝，五目标语义未被修改。
- `FORMAL_INTEGRATION_READINESS=RESEARCH_DECISION_REQUIRED`。
- `RETRAIN_REQUIRED=UNDETERMINED`。
