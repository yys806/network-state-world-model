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

Input logical end-to-end 与 hop-local 候选均只能部分构造；Return 只能部分构造；DepData 无真实传输来源。综合 verdict 为 `FLOW_CONTRACT_NOT_YET_SUPPORTED`。这是阻止 Graph Builder 的审计结果，不是失败的实现测试。

## Validation

- focused audit：5/5 通过。
- artifact builder：生成 deterministic JSON/manifest，保存源码 SHA-256。
- scope：graph builder、formal dataset、training、GPU、locked_test 全部 false。
- 其余 Step 4.2A/4.1/3.3/3.2/Raw 回归待本轮最终验收命令统一执行。

## Known Issues

缺少 simulator-issued stable Flow identity、跨 hop current remaining、Return 独立 identity/remain、DepData payload/transfer，以及 dynamic available CPU、storage、wired queue/load/utilization。不得通过旧 `LogicalFlow`/`CarryingHop` 或 DAG 名称绕过这些缺口。

## Git / Next Step

本记录随本 Step commit。唯一建议：研究者审阅 Flow source audit 后，单独授权最小 Raw additive source contract；不要自动实现 Graph Builder。
