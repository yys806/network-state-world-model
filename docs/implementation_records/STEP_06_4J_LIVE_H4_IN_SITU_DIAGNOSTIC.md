# STEP 6.4J Live H4可评分性原位诊断实施记录

## Step Goal
在真实 AirFogSim decision callback 内捕获新的 live root，并以不改变 Scorer/Objective/World Model 语义的最小 H4 CPU probe 获取机制证据。

## Definition Basis
用户授权的 STEP 6.4J 原位诊断定义；r39 protocol/manifest；正式 live planner helper path。

## Initial State
r39只保存了工程级统计和 root摘要，没有完整 causal prefix/candidate ledger；此前静态 prepared anchor 路径被判定为不可用于精确重放。

## Files Involved
- `code/scripts/run_step6_4j_live_root_capture_diagnostic_v4.py`
- `code/scripts/audit_step6_4j_live_root_capture_v1.py`
- `code/tests/test_step6_4j_live_root_capture.py`
- `code/artifacts/audit/pi_jwm_step6_4j_live_root_capture_20261011_v4/`

## Changes
新增原位 callback 诊断脚本：在目标 frame、任何 policy command/RNG/env.step 之前持久化 causal prefix；复用 live tensor/state/context/domain/objective/root 构建；对最多两个候选记录 rollout/scorer旁路结果。新增独立审计与三项回执测试。未修改 World Model、Objective、Scorer、CandidateDomain 合法性、fallback 或搜索算法语义。

## Validation
- `airfogsim` 环境依赖检查：`osmnx,traci` 导入通过。
- 两个真实 callback root capture：frame 6 / capture-012，frame 29 / capture-058。
- 两 root `LIVE_ROOT_CAPTURE_VALID`；domain count 6 / 17。
- 各 1 个合法 H4 完成并进入原 Scorer；Task_5 / Task_15 在 H1 触发 `UNSUPPORTED_FUTURE_RETURN_BIRTH`；H_eff=0；0 scoreable H4；0 scorer exception。
- 两个第二候选分别触发 mobility domain rollout ValueError，独立分类。
- 新 WM transitions=8，总量约束通过；GPU未使用，locked_test=false。
- 独立审计 PASS；新增测试 3/3 PASS；compileall 与 diff check 待最终收口时重跑。

## Results
本次真实 live root 上，fixed-support future Return-birth 是直接造成 H4 不可评分的已观察机制。该结果是 NEW_CPU_DIAGNOSTIC，不是 r39 3744事件的精确比例。

## Known Issues
`R39_EXACT_EQUIVALENCE=UNPROVEN`；没有观察 H_eff=1/2/3、合法 scoreable H4 或 scorer exception。第二候选枚举存在mobility domain错误，已保留为独立异常，不改变科学语义。

## Git
Pending final commit/push.

## Next Step
研究者与 ChatGPT 决定是否改变 support 语义并设计最小区分实验；Codex不自动改变科学定义。
