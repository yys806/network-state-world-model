# Step 6.4J Raw–Model–Scorer 生命周期审计

本目录是只读审计证据。AirFogSim 的计算完成使用 `computed_size >= task_cpu`，完成后先进入 `waiting_to_return`；项目 Ledger 只有显式 `route_kind=return` 决策动作才创建 Return Flow。两个 Live Root 的结果分别支持 Task_5 的等待回传/固定支持边界，以及 Task_15 的 Raw 生命周期与模型零剩余边界差异。r39 精确候选原因分布仍未被证明。详见 JSON ledger、数值边界和 Validation 原始统计。
