# STEP 6.4N — Route–Comm–Flow 动态机制集成与 CPU 验证

## Step Goal / Definition Basis

实现研究者本轮明确授权的隔离 v2，不改变正式 v1 默认行为。定义依据为研究者 STEP 6.4N 的 A–D 方案及合同 `docs/contracts/PIJWM_STEP_06_4N_DYNAMIC_RETURN_V2.md`。私有定义目录 `D:/shen/OB/科研/PIJWM` 在本次主机上不存在，未读取或修改，不虚构文件哈希。此前 6.4L/R1/6.4M 源码、合同和真实回执已读取。远端 `git fetch origin` 成功，起点 main/origin/main=`26a51fa4b4fd27300bcb460fcd306c5f94fdd7ad`；gh 未安装，Git 远端直接核对。

## Initial State

已有未提交 runtime adapter、四分支 harness 和5项测试；v1 namespace 在执行前因 RB 检查失败。动态 WM/Scorer 与最终验收尚无。本轮保留旧失败证据，没有覆盖历史 receipt。

## Files Involved / Changes / Reuse

- `code/src/pi_jwm/step6_4n_runtime_v2.py`：instance-scoped Return 时钟、native Task 传输旁路、joint setter 前置检查/写回核验/部分失败隔离。Input、deadline、TTI 阈值与真实 radio 规则不变。
- `code/src/pi_jwm/step6_4n_dynamic_v2.py`：共同 anchor 去尾部无主填充、append Flow、DynamicScorerV2。
- `code/src/pi_jwm/step6_4n_rollout_v2.py`：官方 Ledger snapshot/Route action birth、ID→index、Graph/hidden 同步、当步进入冻结 one_step、递归与阶段 clock。
- `code/scripts/run_step6_4n_real_cpu_v2.py`、`run_step6_4n_wm_cpu_v2.py`：真实/模型有界实验。
- `code/scripts/audit_step6_4n_gate1_v2.py`、`audit_step6_4n_acceptance_v2.py`：独立原始落盘重建和评分重算，无模型选择或搜索。
- `code/tests/test_step6_4n_*.py`：23项机制/负例测试。
- `code/artifacts/manifests/pi_jwm_step6_4n_v2_20261012/config.json`：版本显式 opt-in。

复用官方 CausalFlowLedger、真实 AirFogSim setters/step、冻结 live history builder、Graph rebuild、ZERO_COLD_START append helper、冻结 one_step、v1 effort 公式；未修改上述 v1 文件。

## Validation (actual commands / outputs)

1. `D:/miniconda/envs/airfogsim/python.exe code/scripts/run_step6_4n_real_cpu_v2.py --mode <mode>`：四模式均 COMPLETED/各4步，最后 source-bound namespace=code/artifacts/audit/pi_jwm_step6_4n_20261012_v5。
2. `python code/scripts/audit_step6_4n_gate1_v2.py`：9项 PASS；相同 prefix/current/RNG/time。
3. `python code/scripts/run_step6_4n_wm_cpu_v2.py`：两个候选各 H4，8 transitions；原参数 SHA 不变、Future poison PASS；source-bound namespace=code/artifacts/audit/pi_jwm_step6_4n_20261012_wm_v4。
4. `python code/scripts/audit_step6_4n_acceptance_v2.py`：Gate 1/2/3 独立重建 PASS，原始逐 Task/Horizon 评分独立重算一致。
5. `python -m unittest discover -s code/tests -p 'test_step6_4n_*.py' -v`：23项通过。真实 TaskManager 测试使用 airfogsim 环境及 PYTHONUTF8=1。
6. v1 focused regression：`test_step6_2b_planner_objective_scorer_v1 test_step6_4f_comm_eligibility_v1 test_step6_3b_candidate_grammar_v1 test_step6_1_trained_candidate_rollout_v1 test_step6_4m_route_comm_flow_contract_v1 test_step6_4k_return_shadow_v1 test_step6_4b_live_bridge_v1`：61项通过。
7. compileall、知识索引 build/--check、diff 与 Context Consistency Check 在收口时执行；实际结果见最终 verification manifest。

## Results / Expected vs Actual

STEP 6.4N 的隔离 opt-in v2 CPU 机制已实现：Return 独立活动时钟、真实同隙 Route+RB、首步动作触发官方 Ledger Flow birth、共同 anchor 动态 slot/Graph/zero hidden、冻结 WM H1–H4 和 DynamicScorerV2。正式 v1 Route=EXPLICIT_NOOP_ONLY、旧 Scorer/Grammar/Domain/bridge、checkpoint 与 normalization 保持不变。Task_5 同根四分支 Gate 1 和双候选 Gate 2/3 通过；只是一条真实 root 的机制证据。预测可靠性 NOT_ESTABLISHED，正式 Route 集成 BLOCKED，是否需要重训 NOT_ESTABLISHED。GPU/训练/B512/正式S-CEM/baseline/locked_test/6.4O 均未启动。

真实1.8秒：旧 Route 被 TTI code1 拒绝；v2 无RB不继承Input时钟、Return clock保持1.8；v2 Route+RB0 同隙真实完成 Task_5，有效Return交付=0.07610754782944204；NOOP仍 waiting。无线 Task 原始传输参数=2.2081797374222076，但旧 collector 事件 delivered=Input大小0.556973122843206，存在已证实的历史采集字段错误。v2 直接记录 native方法参数、阶段需求与完成返回值，保留旧事件并单独解释，不修改历史数据或替真实结果打补丁。

模型共同 anchor 保留一个官方 Input Flow index0；移除31个无主填充后，B新Return index1与Ledger一致，而旧Shadow index32不是正式动态slot证据。出生在首步Route内部、learned dynamics前；Route effect不重复执行。A H4保持未满足Return，B首步完成。评分共同cohort=Task_5/6/7，分母Comm=50/Comp=52/Mob=30，H_eff=4。A=(0,0,1,1,0)，B=(0,0,2/3,2/3,1/600)。这是v2目标下两个指定动作的机制排序，不是搜索或正式性能结论。

## Known Issues / Status

本轮保留修复失败证据，未覆盖namespace：v1执行前RB检查失败；v2/v3真实step后的Shadow/collector schema冲突；WM v2缺wired配置；WM v3完成后relation清除触发Scorer拒绝。最终source-bound v5=16真实诊断步、WM v4=8转移；本轮累计50真实诊断步及24模型转移，始终同一指定Root/H≤4/两个固定模型候选，无搜索扩容或按未来结果选样。两项独立验收负例能拒绝score/fingerprint篡改与receipt缺失。

delivery_manifest保存82份实际AirFogSim Python/许可/依赖源文件ZIP与逐SHA、SUMO静态输入SHA。reference目录无独立Git元数据，不虚称上游commit；v2来自版本管理内adapter，未改AirFogSim源文件，完整运行依赖仍需安装。

{
  "RETURN_TTI_V2_REAL_SIM_PASS": "PASS",
  "SAME_SLOT_ROUTE_COMM_EXECUTION": "PASS",
  "DYNAMIC_FLOW_BIRTH_CAUSALITY": "PASS",
  "FROZEN_WM_H4_COMPATIBILITY": "PASS",
  "PREDICTION_ALIGNMENT": "NOT_ESTABLISHED",
  "SCORER_V2_COMPARABILITY": "PASS",
  "LEGACY_V1_REGRESSION": "PASS",
  "FORMAL_ROUTE_INTEGRATION_READINESS": "BLOCKED",
  "RETRAIN_REQUIRED": "NOT_ESTABLISHED"
}

预测剩余量/完成事件在本 Task_5 H1–H4 对齐，不能证明 learned CSI 准确；第一隙小需求已饱和，错误速率也可能给相同完成结果。误差报告单独记录CSI/速率差和任务规则误差。最多2 Root是上限而非必须数量；Task_5已完成合法传输，故未根据结果另挑Root。

当前 v2 harness 候选为Route NOOP与Task_5 Return+RB0，并非可替代完整正式 CandidateDomain 的搜索域。完整 Route+Comm+Comp+Mob 批量联合资源 prevalidation / pre-existing returning实时bridge / 输入阶段零服务活动clock / 长 horizon 终止封装 尚未建立正式验收；部分setter失败会隔离并禁止step/retry。H≤4本root不会触发新Return TTI，但规则测试覆盖阈值前后与服务刷新。正式集成 readiness=BLOCKED，不能给出“不需重训”。Input优化/future birth/多跳/reroute/Epoch变化仍fail closed。

## Minimum Formal Integration Changes / Historical Revalidation Scope

1. 把v2动态对象映射、当前returning资格和批量联合资源事务接入版本化 CandidateDomain/Grammar/live bridge；先做CPU零副作用与所有setter失败门。
2. 预注册最多第二Root/未饱和服务条件，核验冻结模型CSI→速率→有效进展误差，研究者据证据决定是否重训；不可按未来成功结果选Root。
3. 历史6.4F–J以Route NOOP/fixed-support/v1评分为界保留，不删除；新Domain/Scorer会改变支持和排序，不能沿用6.4G方法选法、6.4H预算资格或6.4J性能身份。若正式集成获授权，至少重新核验支持catalog/候选域、搜索选法/预算、fallback、closed-loop协议、日志验收与指标分母。

## Context Consistency Check / Git / Next Step

AI_CONTEXT 00/02/03/04/05/06/07/08 增量同步，仅06记录研究者在本轮明确授权的v2研究方案；主文档、计划表、推进、tracker、索引和registry同步。Git commit/push见最终完成报告；未纳入无关未跟踪文件。唯一建议下一动作：研究者审阅预测可靠性未建立与正式集成缺口。完成本Step后停止，不启用正式Route搜索/6.4O/训练。
