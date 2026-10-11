# STEP 6.4L-R1 — Route–Flow 因果与冻结模型支持边界补充验证

## Step Goal

在不改变正式 World Model、CandidateDomain、Objective、Scorer 或 Planner 语义的前提下，追踪真实 `anchor-0006 / Task_5` 的 Return Route 失败原因，修正 6.4L 诊断资格/索引接线，并验证现有 U2U 支持上的 Shadow Flow 是否可被冻结模型递归。

## Definition Basis

本记录只实现研究者授权的 6.4L-R1 诊断。正式 Route 仍是 Planner v1 的显式 NOOP；Shadow 结果不进入正式候选或正式目标比较。r39 精确重放仍为 `UNPROVEN`。

## Initial State

6.4L 的 A 诊断因遗漏 `annotate_domain()` 抛出 domain admission 异常；B 诊断把字符串节点 ID 传给整数模型索引接口，报告了不可直接采信的 `SHADOW_ROUTE_COMM_SUPPORT_MISSING`。真实 AirFogSim B 已观察到 Task_5 failed，但失败分支未有 failure code。

## Files Involved

- `code/scripts/run_step6_4l_r1_task5_runtime_trace_v1.py`：真实 AirFogSim 有界运行时调用顺序和 failure code 旁路。
- `code/scripts/run_step6_4l_r1_frozen_wm_scorer_v1.py`：A 资格修正及保留的 v1 失败 namespace。
- `code/scripts/run_step6_4l_r1_frozen_wm_scorer_v2.py`：整数实体索引修正后的独立 Shadow namespace。
- `code/scripts/audit_step6_4l_r1_route_comm_matrix_v1.py`：Raw channel、Tensor/Graph relation 和运行时服务矩阵。
- `code/scripts/audit_step6_4l_r1_acceptance_v1.py`：独立计数、状态和 SHA 审计。
- `code/tests/test_step6_4l_r1_route_flow_boundary.py`：R1 focused regression。
- `code/artifacts/audit/pi_jwm_step6_4l_r1_20261011_v1/`：未覆盖的 v1 旁路结果。
- `code/artifacts/audit/pi_jwm_step6_4l_r1_20261011_v2/`：修正后的 R1 结果、矩阵、独立验收和报告。

## Changes

1. 真实 runtime wrapper 记录 `setTaskReturnRouteAndStartReturn`、`failOffloadingTask`、`_updateTask`、`_allocate_communication_RBs` 和 `_updateWirelessCommunication` 的顺序及任务桶。
2. A 使用正式 `annotate_domain()`；没有手工写入资格标志。
3. B 将 RouteShadow 的字符串业务 ID 与模型张量所需的整数实体索引分开；没有放宽通信支持检查。
4. 新增 Raw/Model/Scorer 证据矩阵、独立验收回执和 focused test。

## Reuse and Non-Changes

复用原始 live root、AirFogSim collector、冻结 checkpoint、TRAIN normalization、原 `rollout_candidate_sequential()` 和原 `score_candidate_set()`。未修改 World Model、Objective、Scorer、CandidateDomain、Planner Route NOOP 政策、checkpoint、正式数据或历史 receipts。

## Validation

- AirFogSim：`D:\miniconda\envs\airfogsim\python.exe code/scripts/run_step6_4l_r1_task5_runtime_trace_v1.py`，真实 root 捕获和 bounded step 完成。
- Frozen WM v1：A qualification fixed，B 类型接线失败保留在 R1 v1 namespace。
- Frozen WM v2：A/B CPU 旁路完成；A 4 horizons/H_eff 0，B 4 horizons/H_eff 4。
- Matrix：`python code/scripts/audit_step6_4l_r1_route_comm_matrix_v1.py`。
- Independent audit：`python code/scripts/audit_step6_4l_r1_acceptance_v1.py`，status `PASS`。
- Focused test：`python -m unittest code.tests.test_step6_4l_r1_route_flow_boundary`。
- 另外运行项目相关回归、compileall、Context Consistency Check 和知识索引 `--check`。

## Results

Task_5 失败码为 `TASK_FAIL_OUT_OF_TTI`（code 1），发生在 `_updateTask()` 的 `checkTasks()`，早于 wireless RB allocation，deadline 尚未到期。Raw channel 和 model relation 都支持 UAV0→UAV1，但本次没有新 RB request，故分类为 `TEMPORARILY_NO_SERVICE`。A 的正式 fixed-support 拒绝为 `UNSUPPORTED_FUTURE_RETURN_BIRTH:Task_5:H1`。B 在 Shadow Flow 提前注入并使用整数索引后，冻结模型完成 H1–H4，原 Scorer 返回 `SCOREABLE/H_eff=4`；该分数没有正式提升。

## Expected vs Actual

预期是区分真实 runtime failure、诊断工具缺陷和模型支持边界。实际发现三者均存在：真实失败是 TTI 顺序问题；A 是资格元数据缺失；B 的旧拒绝是字符串/整数索引接线缺陷，修正后证明当前 U2U relation 存在。真实 Route B 仍未形成同一 pre-route anchor 的正式可比候选。

## Known Issues / Research Boundary

尚未定义正式 Route→Comm assignment→Return Flow birth→TTI 计时的统一动作合同；也未证明 zero cold-start Flow 的预测可靠性。A/B Objective 可比性 `NOT_ESTABLISHED`，r39 精确等价 `UNPROVEN`，`RETRAIN_REQUIRED=UNDETERMINED`。这些属于研究者需要决定的状态/动作语义，未在本 Step 修改。

## Git

本记录与 R1 诊断脚本、测试、证据和 AI_CONTEXT 同一提交收口；详见最终报告中的 commit/push 状态。

## Next Step

研究者只需决定 STEP 6.4M 是否采用统一的 Route、RB assignment、Return Flow birth 和 TTI/lifecycle 顺序合同；在该决定前不启动正式 Route 候选实验。

## Full regression note

完整 `python -m unittest discover -s code/tests -p 'test_*.py'` 运行了 2172 tests，R1 focused 与相关 Step 回归均通过；全量结果为 1 个历史 receipt 重建 failure、39 个历史环境/fixture errors（包括缺少默认 `traci`、缺失旧 artifact/protocol、旧 tensor fixture 与命令行参数门），没有 R1 新测试 failure。没有为修复这些越界历史问题修改科学源码；AirFogSim R1 运行使用 `D:\miniconda\envs\airfogsim\python.exe` 并独立通过。
