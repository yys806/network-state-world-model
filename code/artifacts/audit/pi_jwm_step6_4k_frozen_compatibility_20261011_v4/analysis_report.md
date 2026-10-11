# STEP 6.4K 冻结 World Model 兼容性报告

## 结论状态

- `RULE_ONLY_PASS`：Causal Ledger 规则原型通过正负例、重复 birth、epoch/Flow identity、多跳守恒和失败闭门测试。
- `FROZEN_WM_RECURSION_PASS`：已有 Return Flow 的真实 Formal anchor 完成 4 步递归；新增 Shadow Return Flow 也完成 4 步只读递归，全部 finite，递归输入等于前一步输出。
- `FORMAL_PLANNER_SCOREABILITY_PASS`：未测试。正式 CandidateDomain、Route NOOP 和原 Scorer 未被 Shadow 状态绕过。
- `RETRAIN_REQUIRED`：`UNDETERMINED`。虽然冻结模型能执行本次 4 步 Shadow 输入，但新 Flow slot/zero hidden 是分布外接口，不能据此宣布无需重训。

## 真实规则与 Shadow 规则

AirFogSim 完成计算后进入 `waiting_to_return`，Return Route 是后续显式业务动作；现有 Ledger 的 Return Flow birth 只发生在 `route_kind=return` 的决策动作。Shadow 保留这条因果边界，不因 `task_work_remaining=0` 自动 birth。Flow ID 使用 `flow::<task>::Return::<epoch>`，重复 birth 幂等，Route 变更只在 clean hop boundary 处理，多跳中间 hop 不增加 E2E delivered，终点才增加。

## Validation 修正

完整重放全部 12 条 `dev_validation` Raw：1164 decision roots、48078 task records、1237 条缺少 Return requirement 字段的 mapping error（主要是 terminal/failed Raw 记录，已 fail-closed）。在字段完整记录中：710 条 waiting-to-return 无 Return Flow，8460 条已知 Return Flow，5675 条 Return 仍 active，2785 条 Return completed，713 条 FP32 视角潜在 fixed-support block，710 条 FP64 视角 block，3 条 FP64/FP32 完成冲突，157 个独立 Flow ID。旧 721/724 分别是上一帧 route-action 近似口径；Ledger 复核后的 waiting 缺失为 710，FP32 潜在阻塞为 713，不能把旧数字当作真实缺失 Flow 数量。

## 冻结模型影响

已有 Return Flow anchor：4 步 H1–H4，Flow presence 维持 8，Task completed 轨迹 finite，递归指纹检查通过。Shadow route：Task_112，Flow `flow::Task_112::Return::0`，追加 slot 26，Flow slot 由 26 扩为 27；冻结模型 4 步递归通过，参数 fingerprint 未改变。该结果证明当前实现的规则和一次分布外状态接口可执行，不证明正式 Planner 能生成 Route，也不证明 checkpoint 对大范围动态 Flow 泛化。

## 未完成与边界

正式 Scorer 没有把 Shadow 分数计入任何结果；正式 Planner Route 仍为 NOOP；没有运行新 env.step、B512、S-CEM 或 GPU。历史 r39 仍无逐候选 trace，不能还原其 3744 完整 H4 的原因分布。

## 建议

先把 State/Flow 合同和 FP64/FP32 完成标志固定为研究者决定，再设计正式 Route 动作和动态 Flow capacity；随后必须重新做 CandidateDomain、Objective/Scorer 资格验证。Shadow 成功本身不足以支持“不重训”结论。
