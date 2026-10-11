# STEP 6.4K — Deterministic Return Flow Shadow & Frozen World Model Compatibility

## Step Goal

在不修改正式 World Model、Objective、Scorer、CandidateDomain 或 Planner 的前提下，纠正完整 Validation Return Flow 统计，实现确定性 Return Flow Shadow 原型，并验证冻结模型的已有 Flow 和 Shadow 新 Flow 递归边界。

## Definition Basis

依据 AirFogSim 真实源码、Step 4.2C-B Ledger 合同、现有 Step 4.4/6.1/6.2B 实现和研究者本轮授权。所有 Shadow 状态标记为隔离研究原型。

## Initial State

基线 `bde348d26dc21e4e235f42c972f79da3ad12425d`；r39 冻结 `0586cde90c8cc244e9f5091fd51e7c77217b817a`。旧 Validation 统计 721/724 是上一帧 Route action 近似口径。

## Files Involved

- `code/src/pi_jwm/step6_4k_return_shadow_v1.py`
- `code/scripts/run_step6_4k_validation_flow_audit_v1.py`
- `code/scripts/run_step6_4k_frozen_compatibility_v1.py`
- `code/scripts/audit_step6_4k_independent_v1.py`
- `code/tests/test_step6_4k_return_shadow_v1.py`
- `code/artifacts/audit/pi_jwm_step6_4k_return_shadow_20261011_v4/`
- `code/artifacts/audit/pi_jwm_step6_4k_frozen_compatibility_20261011_v4/`

## Changes and Reuse

新增独立 Shadow 模块，复用 CausalFlowLedger 的 Flow identity、epoch、conservation、hop transition 和 route revision 语义；新增全量 Raw Ledger replay、独立审计和冻结模型 compatibility probe。没有改动正式模型、checkpoint、Scorer、CandidateDomain、Planner 或 r39 证据。

## Validation

- Validation replay：12 trajectory、1164 roots、48078 task records；独立审计 PASS。
- Shadow tests：8/8 PASS，覆盖 waiting、合法 birth、重复 birth、multi-hop、reroute、无 Return、非法 route、守恒、numeric boundary 和 fail-closed。
- Frozen existing Return Flow：真实 Formal anchor H1–H4 recursion PASS，4 transitions。
- Frozen Shadow Return Flow：Task_112，Flow `flow::Task_112::Return::0`，新增 slot 26，4 transitions PASS，finite，recursive links PASS，参数 fingerprint unchanged。
- Formal Planner scoreability：NOT_TESTED；Shadow 分数未计入正式结果。
- GPU/training/B512/S-CEM/locked_test：未运行。

## Results

`RULE_ONLY_PASS`；`FROZEN_WM_RECURSION_PASS`（包括一个真实已有 Flow anchor 和一个隔离 Shadow 新 Flow）；`FORMAL_PLANNER_SCOREABILITY_PASS` 未测试；`RETRAIN_REQUIRED=UNDETERMINED`。

完整 Ledger 统计：710 waiting-to-return 无 Return Flow、8460 Return known、5675 active Return、2785 completed Return、713 FP32 potential fixed-support blocks、710 FP64 blocks、3 FP64/FP32 conflicts、1237 missing-field mapping errors（fail-closed）。这些是离线 Raw/Flow 状态，不是 r39 candidate rejection 比例。

## Expected vs Actual

预期是同时区分真实 Flow 缺失和旧近似统计，并验证冻结模型是否能接收 Shadow 新 Flow。实际达成：规则和冻结递归均通过；正式 Planner/Scorer 仍保持未测试，不能宣布正式可评分或无需重训。

## Known Issues

Shadow 使用追加 slot 与 zero hidden 初始化，属于分布外接口；未验证正式 normalization、训练分布、CandidateDomain Route 合同和 scorer scoreability。旧 r39 缺少逐候选 trace。

## Git

待测试、索引和 diff check 完成后 commit/push。

## Next Step

研究者决定是否正式定义 Return Route 动作与动态 Flow state contract；Codex 不自动合并 Shadow 语义。
