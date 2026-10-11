# STEP 6.4M — Route–Comm–Flow 统一状态合同与 CPU 机制验证

## 证据范围

本报告只使用 AirFogSim 源码、STEP 6.4L-R1 的真实 Task_5 runtime trace、官方 Causal Flow Ledger 快照、R1 身份回执，以及本目录的隔离合同状态机。没有新增 `env.step`、World Model transition、GPU、训练、正式 CEM、B512 或 `locked_test`。冻结 checkpoint SHA 为 `941ee94131d406de914a79aeda43e929c92727631263d85221615422146a32c9`，TRAIN normalization SHA 为 `1e86bbac6e6ee2d1f1d24e572c5433ed56b1e00c0423bd468fb5150378e6fed5`。

## 1. 真实事件顺序（Implementation Fact）

在正式 collector 的 decision callback 中，Route setter 先写入 route buffer；Task_5 此时仍在 waiting-to-return bucket，因此同一 callback 的 Comm assignment 条件不成立。`AirFogSimEnv._updateTask()` 在内部 tick 消费 route，调用 `setTaskReturnRouteAndStartReturn()`，把任务移入 returning 并设置 `last_return_time`，随后 `checkTasks()` 先使用旧的 `last_transmission_time` 检查 TTI。R1 trace 在 1.8s 记录 `TASK_FAIL_OUT_OF_TTI`，无线 RB allocation 之前已经失败；没有新增 Task_5 transfer，deadline 尚未到期。该顺序由 `01_source_event_order.json` 和 `07_independent_acceptance.json` 的 16/16 独立复算通过。

`_updateWirelessCommunication()` 只在前置任务检查通过后分配 RB。因而“通信关系存在”“Return Flow 已建立”“这一 tick 获得无线服务”是三个不同条件。

## 2. 统一参考合同（Implementation Fact / Proposal）

`PIJWM_STEP_06_4M_ROUTE_COMM_FLOW_DYNAMIC_STATE_CONTRACT_V1.md` 固定了 v1 单跳参考范围：

```text
同一 pre-Route state/graph/latent/cohort
  → Route/Comm/Comp/Mob action
  → 合法性检查
  → Route effect / Flow birth
  → TTI check
  → explicit service / transfer
  → time advance / next state
```

其中 Flow identity 为 `flow::<TaskID>::<FlowType>::<Epoch>`；Flow 的 known、presence、active、completed、holder、hop、route revision 与 E2E 守恒分开记录。Input/Return 共用身份和守恒不变量；source 等于 destination 时不创建 Return Flow；多跳、reroute、destination-change epoch 只定义边界，不声称正式支持。

参考实现 `step6_4m_route_comm_flow_contract_v1.py` 只做 fail-closed 状态检查，不调用 AirFogSim、正式 Planner、World Model 或 Scorer，也不自动创建 Route/RB。

## 3. 同根 A/B 与官方 Ledger（Observation）

隔离 harness 从同一个 Task_5 pre-Route anchor 派生：A 是 Route-NOOP，B 仅在显式 Route action effect 中创建 `flow::Task_5::Return::0`。A/B 的 pre-Route fingerprint 相同；B 不预注入 Flow。B 使用兼容时序在 1.8s 先得到 TTI failure，服务事件被跳过，任务变为 failed，且没有 transfer。官方 `ReturnShadow` Ledger 生成相同 Flow ID；slot、source、destination、route 和 epoch 可对齐。

模型绑定检查发现该新 Flow 对现有 anchor 需要 append slot 32，并使用已有 relation 38（source index 4、destination index 5）。本轮没有修改 tensor、latent 或 checkpoint，也没有执行 cold-start hidden。因此这里只证明接口字段可绑定，不能证明冻结模型对正式动态 birth 的预测可靠性。

## 4. 结果状态

| 状态 | 结论 | 证据边界 |
|---|---|---|
| `SOURCE_EVENT_ORDER_VERIFIED` | `PASS` | 源码与 R1 runtime trace、独立 16 项重算 |
| `ROUTE_COMM_TTI_CONTRACT` | `RESEARCH_DECISION_REQUIRED` | AirFogSim compatibility 顺序可复现；是否同隙服务、是否重置 Return TTI 尚未冻结 |
| `INPUT_RETURN_FLOW_STATE_CONTRACT` | `PASS`（v1 单跳参考） | identity、slot、epoch、守恒和负例测试；不是正式模型合同 |
| `DYNAMIC_STATE_INTERFACE_READINESS` | `PASS`（隔离参考） | ID→index、slot append、relation binding 可验证；未接入正式 WM |
| `COMMON_ANCHOR_CAUSALITY` | `PASS`（隔离 A/B） | 同一 pre-Route fingerprint；旧 R1 Shadow B 仍不可用于正式比较 |
| `FROZEN_WM_H4_COMPATIBILITY` | `NOT_ESTABLISHED_FORMAL_ROUTE_SHADOW_ONLY` | 仅复用 6.4K/R1 的预注入或 Shadow H4 证据；本轮未把 Route birth 接入正式 one-step |
| `SCORER_COMPARABILITY_READINESS` | `BLOCKED` | 原 Scorer 固定支持仍拒绝 future Return birth；未伪造正式分数 |
| `FORMAL_INTEGRATION_READINESS` | `RESEARCH_DECISION_REQUIRED` | 需要先冻结 Route→Comm→Flow→TTI、dynamic slot、burden mask 和 Route cost 合同 |
| `RETRAIN_REQUIRED` | `UNDETERMINED` | 没有重训必要性的直接证据；cold-start/分布外风险未通过验证 |

## 5. 研究者 Decision Gate

1. **Return TTI 起点**：保持 AirFogSim 旧 `last_transmission_time` compatibility，还是定义独立 Return clock。推荐先保持 compatibility，并把新语义作为单独实验。
2. **Route→Comm 时序**：Route action 同一 tick 是否允许 pending-route 进入 Comm/RB，还是下一 tick 才能服务。当前 R1 真实行为属于后者的 failure path。
3. **Flow birth 层**：由 Planner-side dynamic slot 处理，还是扩大正式 World Model 的结构状态。Shadow 可递归不等于训练分布内可靠。
4. **新 Flow 的 cold-start hidden**：zero-init 只能作为开发参考；正式使用前需要分布审计，是否重训取决于该审计。
5. **Objective/Scorer mask**：anchor-frozen burden/cohort 是否扩展到候选新生 Flow；在决定前维持拒绝，不比较不可比候选。
6. **Route cost**：现有五目标没有独立 Route effort 来源，不新增第六目标或隐藏惩罚。
7. **Input Route 对称性**：本轮只验证 identity/守恒接口；联合 Input Route 规划需要独立合同。

## 6. 独立验收与历史缺陷

`07_independent_acceptance.json` 从源 trace、post-step、v2 artifacts 和 R1 identity 重新计算 16 项检查，16/16 通过，状态为 `PASS_WITH_DECISION_GATES`。v1 的外层状态白名单错误保留在 `..._v1/05_acceptance_receipt.json`，没有被覆盖；v2 只修正收据 schema/allowlist，不改变科学结果。

本轮新增 focused contract tests 共 14 项，覆盖同根 A/B、TTI-before-service、Flow identity、重复 birth、非法 Route、Comm/RB、slot、Input/Return、Task_15 FP 边界、模型索引和独立验收回执。GPU、训练、正式搜索、locked_test 均为 false。

## 7. 不能从本轮推出的结论

不能把一次 Task_5 真实 failure 解释成所有状态的普遍规则；不能把隔离状态机的 H4 结构可递归解释成冻结模型正式 H4 预测可靠；不能把 R1 的 Shadow score 提升为正式 Planner score；不能据此决定是否重训或直接集成动态 Return Flow。
