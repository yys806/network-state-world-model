# STEP 6.4L — Route–Flow–World Model–Scorer 因果一致性

## Step Goal

在不改正式算法的前提下，验证真实 AirFogSim Route setter、官方 Causal Flow Ledger、冻结 World Model 和 Scorer 的最小因果链。

## Definition Basis / Initial State

基于 6.4K Shadow Return Flow 证据。正式 Route 仍为 NOOP；checkpoint、normalization、Objective、Scorer 和 Planner 均冻结。r39 精确等价不宣称。

## Files Involved

- `code/scripts/run_step6_4l_airfogsim_route_v1.py`
- `code/scripts/run_step6_4l_frozen_wm_scorer_v1.py`
- `code/scripts/audit_step6_4l_route_flow_ledger_v1.py`
- `code/artifacts/audit/pi_jwm_step6_4l_route_flow_causal_20261011_v7/`
- `code/artifacts/audit/pi_jwm_step6_4l_route_flow_causal_20261011_v8/`
- `code/artifacts/audit/pi_jwm_step6_4l_route_flow_causal_20261011_v9/`

## Changes

增加诊断专用 AirFogSim A/B gate，防止 collector 的隐式随机 Route 污染 A/B；增加冻结模型旁路、官方 Ledger 独立审计、Input Route 源码审计和报告。正式采集在未设置 gate 时保持原行为。

## Validation

- AirFogSim A/B：同 seed/root prefix，A 无 Route，B 真实 setter 成功，各推进 1 个真实 step。
- 官方 Ledger：A 无 Return birth；B 建立 `flow::Task_5::Return::0`，独立审计 PASS。
- Frozen WM/Scorer：A `Planner v1 domain admission required`；B `SHADOW_ROUTE_COMM_SUPPORT_MISSING`；未晋升任何分数。
- GPU、训练、B512、正式 CEM、locked_test：未运行。

## Results / Expected vs Actual

Route setter 与 Ledger birth 符合预期；真实传输和正式 Scorer 可评分未达成。World Model 的一般 Shadow H4 兼容性沿用 6.4K 证据，但 Task_5 真实 Route 后兼容性未证实。

## Known Issues / Next Step

需要研究者决定正式 Route/Comm/Flow 状态合同，随后再做最小候选资格验证。不得自动集成 Shadow 或进入 STEP 6.4M。
