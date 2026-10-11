# STEP 6.4M — Route–Comm–Flow 统一状态合同与 CPU 机制验证

## Step Goal

在不修改正式 World Model、Planner、CandidateDomain、Objective、Scorer、AirFogSim 或 checkpoint 的前提下，依据真实 AirFogSim/R1 证据冻结一份可复核的 Route–Comm–Flow–Task 状态合同，并用隔离 CPU 状态机验证同根 Route-NOOP/Route birth、TTI 失败、Flow identity、模型对象绑定和负例边界。

## Definition Basis

目标定义来自研究者授权的 STEP 6.4M；源码事实来自 `airfogsim_env.py`、`task_manager.py`、`task.py`、`communication_sched.py`、`step4_2c_b_causal_flow_ledger_raw_v1.py`、`step6_2a_route_rule_metadata_v1.py` 和 `step6_2b_planner_objective_scorer_v1.py`。合同正文见 `docs/contracts/PIJWM_STEP_06_4M_ROUTE_COMM_FLOW_DYNAMIC_STATE_CONTRACT_V1.md`。

## Initial State

STEP 6.4L-R1 已确认 Task_5 的 Return setter 成功，但 1.8s 在 wireless allocation 前以 `TASK_FAIL_OUT_OF_TTI` 失败；R1 的 Shadow Flow H4/Scorer 结果不能作为正式 Route 候选。正式 Planner 仍为 Route explicit NOOP，future Return birth 仍不在冻结模型/Scorer fixed support 内。

## Files Involved

- `code/src/pi_jwm/step6_4m_route_comm_flow_contract_v1.py`
- `code/scripts/run_step6_4m_contract_cpu_v1.py`
- `code/scripts/run_step6_4m_contract_cpu_v2.py`
- `code/scripts/audit_step6_4m_contract_acceptance_v1.py`
- `code/tests/test_step6_4m_route_comm_flow_contract_v1.py`
- `docs/contracts/PIJWM_STEP_06_4M_ROUTE_COMM_FLOW_DYNAMIC_STATE_CONTRACT_V1.md`
- `code/artifacts/audit/pi_jwm_step6_4m_20261011_v1/`（保留的 v1 收据缺陷）
- `code/artifacts/audit/pi_jwm_step6_4m_20261011_v2/`（canonical v2 证据）

## Changes and Reuse

1. 新增标准库隔离参考状态机：Flow identity/epoch/slot、single-hop Route、Comm eligibility、TTI-before-service、transfer conservation、future-truth/missing-field/duplicate/capacity fail-closed checks。
2. 复用 R1 的真实 runtime event、post-step、官方 Ledger identity、模型 relation 38 与 checkpoint/normalization identity；没有新增 AirFogSim step 或 WM transition。
3. 新增 v2 CPU receipt，修复 v1 外层状态 allowlist 导致的误报；v1 原始 receipt 未被覆盖。
4. 新增独立验收器，从源回执重算事件顺序、TTI failure、无 transfer、同根 anchor、Ledger ID、模型 binding、身份 SHA、资源边界和历史 v1 缺陷。
5. 新增 contract tests 和分析报告；未把 Shadow 结果提升为正式 Scorer 结果。

## Validation

- focused contract test：14/14 通过（含 `07_independent_acceptance.json` 回执检查）。
- independent acceptance：16/16 checks 通过，`PASS_WITH_DECISION_GATES`。
- `python -m py_compile code/src/pi_jwm/step6_4m_route_comm_flow_contract_v1.py code/scripts/run_step6_4m_contract_cpu_v2.py code/scripts/audit_step6_4m_contract_acceptance_v1.py`：通过。
- `python -m compileall -q code/src code/scripts code/tests`：通过。
- GPU=false，locked_test=false，new AirFogSim steps=0，new WM transitions=0，formal semantics modified=false。

## Results and Expected vs Actual

预期是先验证合同和阻塞位置，而不是获得正式 Route score。实际结果与预期一致：真实顺序和同根隔离合同通过；Task_5 仍按真实 compatibility path 在无线服务前 TTI failure；Flow identity 可与官方 Ledger 对齐；动态模型接口只能做到只读 binding/slot append，正式冻结模型 H4 与 Scorer 可比性仍未建立。

## Known Issues / Decision Boundary

- `ROUTE_COMM_TTI_CONTRACT=RESEARCH_DECISION_REQUIRED`：同隙 Route→Comm 和 Return TTI 起点未冻结。
- `FROZEN_WM_H4_COMPATIBILITY=NOT_ESTABLISHED_FORMAL_ROUTE_SHADOW_ONLY`：未接入正式 one-step，未验证 zero cold-start hidden 的预测可靠性。
- `SCORER_COMPARABILITY_READINESS=BLOCKED`：fixed-support future Return birth 和 anchor-frozen burden/cohort 仍拒绝或不可比。
- `FORMAL_INTEGRATION_READINESS=RESEARCH_DECISION_REQUIRED`；`RETRAIN_REQUIRED=UNDETERMINED`。

这些是需要研究者冻结的科学/接口合同，不由本轮实现便利隐式决定。

## Git

本记录随 6.4M 代码、测试、合同文档、审计回执和上下文索引提交。第一版收口提交为 `8b2047e`（`contract: verify route comm flow state boundary`），已 push 到 `origin/main`；本次记录字段补丁随后单独提交并 push。

## Next Step

唯一下一步：研究者审阅并冻结 Route→Comm→Flow→TTI 合同（包括是否同隙服务、Return TTI 起点、dynamic slot/cold-start 与 Scorer mask）；在该决定前不进入正式 Route 联合规划或 6.4M 后续集成。
