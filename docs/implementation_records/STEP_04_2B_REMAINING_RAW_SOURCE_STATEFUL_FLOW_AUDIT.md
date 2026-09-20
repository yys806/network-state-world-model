# STEP 4.2B — Remaining Raw Source & Stateful Flow Contract Audit

状态：审计完成；`FLOW_CONTRACT_NOT_YET_SUPPORTED`；不进入 Graph Builder。

## Step Goal

用定义 03、真实 AirFogSim 源码和机器 receipt 审计 Input/Return/DepData Flow 是否具备稳定 identity、类型、端点、total、remaining 和 presence 的因果来源。

## Definition Basis

只读定义 03；Step 4.1 mapping 与 Step 4.2A contract/patch；AirFogSim `Task`、`TaskManager`、`airfogsim_env.py`、`WiredNetworkManager`、observer 和 frame-builder 源码。

## Changes

- 新增 `step4_2b_stateful_flow_source_audit_v1.py`、构建脚本和 focused tests。
- 生成 Input/Return/DepData 分开证据、Task Progress ↔ Flow Progress 矩阵、Flow/Hop 区分、route revision、remaining-source matrix 和源码 provenance。
- 明确 `transmitted_size` 是当前阶段/hop 累计量且 hop 完成后 reset；不得把它或 `past_outcome_flow_service` 改名为 current Flow remaining。

## Results

Input logical end-to-end 与 hop-local 候选均只能部分构造；Return 只能部分构造。DepData 当前没有真实传输 process，但这是事实与研究决策边界，不单独作为 Input/Return blocker。综合 verdict 由 Flow-specific required evidence 实际计算为 `FLOW_CONTRACT_NOT_YET_SUPPORTED`。这是阻止 Graph Builder 的审计结果，不是失败的实现测试。

## Validation

- focused audit：7/7 通过，包含 verdict tamper、resource-gap 解耦、identity decision boundary 和 DepData 分类。
- artifact builder：生成 deterministic JSON/manifest，保存源码 SHA-256、source symbol、semantic claim 和 symbol-level anchor。
- scope：graph builder、formal dataset、training、GPU、locked_test 全部 false。
- 其余 Step 4.2A/4.1/3.3/3.2/Raw 回归待本轮最终验收命令统一执行。

## Known Issues

当前真正的 Flow blocker 是 Input/Return 跨 multi-hop 的动作前 current remaining 缺少 source-of-truth，以及 identity/type/端点/route semantics 尚未完成研究决策。simulator-issued stable Flow ID 缺失本身不是唯一 blocker；logical identity 可能在研究者选择语义后因果派生。DepData 仍无真实 process，不能凭 DAG 虚构。dynamic available CPU、storage、wired queue/load/utilization 属于其他 graph input gaps。

## Git / Next Step

本记录随本 Step commit。唯一建议：研究者审阅 Flow source audit 后，单独授权最小 Raw additive source contract；不要自动实现 Graph Builder。
