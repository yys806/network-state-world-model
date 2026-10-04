# STEP 6.4E — 预算冻结、真实 Comm 与 fallback 闭合审计

## Step Goal / Definition Basis

落实本轮研究者明确决定：Planner v1 pure-search 默认 B512；补真实 non-empty Comm 一步证据，统一 fallback 执行桥并给出 A/B/C 实现审计，最终策略留待研究者决定。用户本轮授权与冻结 CandidateDomain/grammar/live contract 为实现依据，不写私人研究笔记。

## Initial State

fetch 后 HEAD=origin/main=`9c9cdfc9845ac5485ca07493aa3f332e43b3aa50`，tracked clean。6.4D 完整320配对 PASS，6.4B 真实两轮机制 PASS但 Comm 空；GPU已由研究者关机，未联系或启动。三个旧 untracked 保留。

## Files Involved / Changes / Reuse

新增 `step6_4e_fallback_v1.py`，统一 reason/显式候选 dispatch、lazy canonical A、受准入限制的 Route-only 投影 B、终止 C。新增 CPU tests、真实一步审计 runner、只读闭合 auditor 与合同。完整复用 live causal builder、冻结 TRAIN normalization、typed graph/state、CandidateDomain、grammar准入、winner action bridge 三原生 setters。未改变搜索、World Model、Objective、CandidateDomain、Return 或 H4科学定义，没有 checkpoint/历史 raw 重写。

## Validation（实际执行）

1. 新测试先 RED（模块缺失），实现后通过；缺失 behavior action offer 的负测试先出现 KeyError，再补接口失败捕获，7/7 PASS。
2. `D:/miniconda/envs/airfogsim/python.exe code/scripts/run_step6_4e_comm_fallback_audit_v1.py --freeze`：只读 TRAIN 当前前缀选择/准入与 poison 不变 PASS，固定 fixture/动作；不看新执行结果选样本。
3. `--execute` 导入 AirFogSim 时 GBK 输出 UnicodeEncodeError，环境尚未创建。保留03a；设 PYTHONIOENCODING=utf-8、使用显式 `--continue-pre-environment-import`，01b绑定工程控制代码新SHA；fixture动作不变。不是科学失败后重跑。
4. 首次真实 episode 在冻结时刻重现，live sample SHA相同；`validate_command` 拒绝已 failed 的 Task_1，`ACTION_BRIDGE_REJECTED`。0本轮setter，0本轮env.step，停止，第二个预注册场景不启动，不换 fixture/动作。没有自动恢复这次失败。
5. `python code/scripts/audit_step6_4e_closure_v1.py`：CPU 重建冲突、候选/负路径、预算/保护源码；科学 verdict BLOCKED。
6. focused tests：新6.4E 7/7、旧6.4B 7/7、grammar 13/13、Domain 6/6，共33 PASS；compileall PASS。knowledge-index write/check、diff与Context检查见11_closure_checks_receipt（完成后填写真实结果）。

## Results / Expected vs Actual

**STEP_6_4E=BLOCKED，CLOSED_LOOP_PRE_FORMAL_READINESS=BLOCKED。**

预算冻结与CPU接口/负路径符合预期；真实 Comm 消费与非终止 fallback 真实一步证据未取得，因此整体不符合完成目标，不能报 READY_FOR_FALLBACK_DECISION。按要求先失败终止，未为 PASS 放宽语义。

真实 fixture `formal-v1-sim-2026092300-policy-2026092400::anchor-0041`，t=4.5s。Domain 无线任务为 Task_1、Task_6、Task_24，前两者 failed，后者 offloading。Domain/grammar 首个非空动作 Task_1/RB0 合法，但当前真实执行桥不允许 failed task。保护源 `_wireless_bindings` 未检生命周期，而活跃 Flow 掩码仍保留失败任务绑定。这是当前支持与真实执行条件冲突；不声称所有 non-empty Comm 都不可执行，也不自行修正 frozen scientific Domain。

A：当前 replay 的规范首动作（Comm/Comp空、两个UAV合法当前动作）能准备且确定性，不能证明所有状态可执行，本轮真实一步未取得。B：已有 behavior 写 Route与其他动作、共享控制/RNG，未提供独立纯 current offer；投影 adapter 有CPU测试，真实 provider MISSING。C：无动作终止及空域/缺字段/bridge/setter失败路径 PASS，不具备性能证据。

## Known Issues / Conflict / Next Step

Documented Intent：Domain当前合法候选应映射到真实可执行任务。Actual Implementation：当前 Flow活跃绑定可能包括 failed task，bridge拒绝。Evidence：04/06回执、真实当前观测、grammar/live源码。Affected Files：既有 `_wireless_bindings`、causal Flow/state构建、live action eligibility。Status：**Awaiting Researcher Decision**。唯一下一动作：研究者审阅一致性冲突，明确支持/执行契约和另行授权后的修复/复验范围；本轮不改变 Domain或重试。

PLANNER_V1_CLOSED_LOOP_B_WM=512；Stage B=NOT_STARTED、DEFERRED / NOT_REQUIRED_FOR_CURRENT_MAINLINE；FINAL_FALLBACK_POLICY=RESEARCHER_DECISION_PENDING；FORMAL_CLOSED_LOOP_PERFORMANCE=NOT_STARTED；HYBRID_PLANNER=NOT_FROZEN；GPU=NOT_USED；locked_test=false。future Return-birth fixed-support limitation保持。

## Git / AI_CONTEXT / Evidence

AI_CONTEXT00/02/03/04/05/06/07/08、authority/process/registry/index同步本轮预算决定与 BLOCKED事实；不得覆盖6.4D历史预算待决定状态或旧运行身份。最终 commit/push main 以本记录所属 Git提交定位；只提交已通过CPU检查的接口与真实阻塞证据，不提交虚假PASS。收口后停止。

机器证据：`code/artifacts/protocols/pi_jwm_step6_4e_comm_fallback_v1_20261004/10_pre_formal_readiness_receipt.json`；合同 `docs/contracts/PIJWM_STEP_06_4E_FALLBACK_INTERFACE_V1.md`。
