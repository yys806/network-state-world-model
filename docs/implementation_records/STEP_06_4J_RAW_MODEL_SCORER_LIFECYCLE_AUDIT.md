# STEP 6.4J — Raw–Model–Scorer 生命周期与 Return Flow 全链路审计

## Step Goal

在不运行 GPU、训练、B512、正式闭环或 locked_test 的前提下，解释两个已捕获 Live Root 的 Task_5/Task_15 生命周期与 `UNSUPPORTED_FUTURE_RETURN_BIRTH`，并核对现有 Validation Raw 快照。

## Definition Basis

本记录只依据源码、已保存 Live Root/v1 H1 证据和冻结 `dev_validation` Raw 快照。r39 精确候选重放未声称完成。

## Initial State

HEAD/origin/main=`7ba949146108eafc8cc426a37a12bf8784f7e13e`；r39=`0586cde90c8cc244e9f5091fd51e7c77217b817a`。已有两个真实 Live Root 和 18 条 H1 动作诊断结果。

## Files Involved

- 新脚本：`code/scripts/audit_step6_4j_raw_model_scorer_lifecycle_v1.py`
- 新测试：`code/tests/test_step6_4j_raw_model_scorer_lifecycle.py`
- 证据：`code/artifacts/audit/pi_jwm_step6_4j_raw_model_scorer_lifecycle_20261011_v1/`
- AirFogSim：`code/reference/AirFogSim/airfogsim/entities/task.py`、`manager/task_manager.py`
- Ledger/World Model/Objective/Scorer：对应 `code/src/pi_jwm/` 源码

## Changes and Reuse

新增只读审计脚本和边界单测；复用既有 v1 Live Root、既有 Raw 快照、原始 Objective/Scorer 和源码，不修改模型预测、CandidateDomain、Objective、Scorer、fallback 或 Planner 语义。

## Validation

已执行审计脚本，生成 identity、Live Root task ledger、FP64/FP32 shadow boundary、Validation Raw-only stats、acceptance receipt 和报告。统计范围为全部 12 条冻结 `dev_validation` trajectory：1164 decision roots、48078 task rows。已核对 `conda run -n airfogsim` 实际导入项目参考 AirFogSim 源码；环境中没有独立 AirFogSim distribution。新增单测覆盖精确完成、FP32 zero rounding 和 Return-birth 判定。后续收口还需执行 compileall、完整 unittest、知识索引 builder 与 `--check`。

## Results

AirFogSim `Task.compute()` 采用精确 `computed_size >= task_cpu`，TaskManager 完成后先进入 waiting-to-return；Return route setter 是后续独立步骤。Causal Ledger 仅在显式 `route_kind=return` 决策动作时创建 Return Flow。World Model route 规则不创建新 logical Flow identity。Task_5 是等待回传但无 Return Flow 的固定支持边界；Task_15 的 Raw FP64 剩余为 `1.1102230246251565e-16`，而 FP32 shadow 和模型剩余为 0，表现为数值阈值与生命周期观察边界差异。

## Expected vs Actual

预期得到可复核的生命周期责任边界；实际完成了源码、两个 Live Root 和全部 dev_validation Raw 快照的只读审计。没有把静态潜在阻塞比例写成 r39 候选拒绝比例。

## Known Issues

Validation 统计是 Raw-only：没有为 1164 roots 运行 World Model 或 Scorer，因此潜在 Return-birth 计数不是实际拒绝计数。r39 缺少逐候选 trace，`R39_EXACT_EQUIVALENCE=UNPROVEN`。Task_15 的差异不能仅凭该审计判定为 FP32 实现 bug，仍需研究者决定状态合同是否应调整。

## Git

待本轮测试、索引检查通过后提交并推送。

## Next Step

研究者只需在“状态/数值合同修复”“确定性动态 Return Flow/Route 能力”之间选择最小区分实验；本记录不自动实施科学改动。
