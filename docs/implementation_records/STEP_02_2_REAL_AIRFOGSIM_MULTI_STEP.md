# STEP 2.2 — 真实 AirFogSim 多决策步 Raw Trajectory 验收

## Step Goal

在非 locked 的真实 AirFogSim 环境运行一条 6 个连续决策步的短轨迹，验证 `Decision_t -> Action_t -> Outcome_t -> Decision_{t+1}` 的跨步连续性。每个下一 Decision 必须在下一轮循环重新读取真实环境；本 Step 不进入 Dataset/Tensor、双图、World Model、Loss、Planner 或训练。

## Definition Basis

- `docs/contracts_PIJWM_RAW_SINGLE_DECISION_STEP_CONTRACT_V1.md`
- `docs/PIJWM_IMPLEMENTATION_TRACKER.md`
- Step 2.1 已确认的真实接口：Route/Comm/Comp/UAV Mobility、vehicle heading degree、UAV heading rad、Mob 只控制 UAV。

## Initial State

Step 2.1 已通过一个真实环境步，但其 `real_single_step.json` 与 `manifest.json` 因 `.gitignore` 的 `code/artifacts/*` 规则未进入 commit `0a11d41`。跨步独立重新采集、lifecycle 连续性、显式 empty/no-op 和 acceleration 差分语义尚未验收。

## Files Involved

- `code/scripts/run_step2_2_real_airfogsim_multi_step_v1.py`
- `code/scripts/run_step2_1_real_airfogsim_single_step_v1.py`
- `code/artifacts/protocols/pi_jwm_raw_multi_decision_step_real_airfogsim_v2_20260919/`
- `code/artifacts/protocols/pi_jwm_raw_single_decision_step_real_airfogsim_v4_20260919/`
- 合同、Tracker、实施记录、计划/进度/发现和 AI_CONTEXT。

## Changes

- 新增真实 6 步 runner，共采集 7 个 Decision 和 6 个 Outcome。
- 下一 Decision 在下一轮循环重新调用真实 collector；不从上一 Outcome 复制构造。`capture_event_id` 独立，且重新采集的实体、任务、lifecycle 与上一 Outcome 对齐。
- 每步四类 action 字段都存在。空动作使用 `entries=[]`、`empty=true` 和明确 `no_op_reason`；缺字段会使验收失败。
- Comp 在 Decision 时刻固定当前 computing task 的显式 `task_id/node_id/allocated_cpu_per_s`，真实 callback 返回该分配；同 slot 新进入 computing 的任务到下一 Decision 才获得 Comp 动作。
- 逐实体记录 AirFogSim acceleration 与 `(v_t-v_{t-1})/delta_t` 的比较，只做实现语义观察。
- Step 2.1 改为追加式 v4 真实重跑，使 manifest 中 runner SHA-256 与当前源码一致；Step 2.1 v4 和 Step 2.2 v2 的 JSON/manifest 使用 `git add -f` 纳入版本控制。两个 runner 均以 LF 写 JSON，使 manifest SHA 与 Git blob 字节一致。

## Reuse

复用项目已有真实环境 `_build_environment`、真实 observer、CPU inner rule、AirFogSim 四类 scheduler、task manager 和 traffic manager。未修改第三方 AirFogSim。

## Validation

真实命令：

```powershell
conda run -n airfogsim python .\code\scripts\run_step2_2_real_airfogsim_multi_step_v1.py
conda run -n airfogsim python .\code\scripts\run_step2_1_real_airfogsim_single_step_v1.py
```

Step 2.2 真实运行从 `2.4 s` 到 `3.0 s`，slot 为 `0.1 s`。18 项 checks 全部为 `true`。Step 2.1 v4 的 13 项 checks 全部为 `true`，且 manifest runner hash 与当前文件一致。

收尾验证还包括 `compileall`、Step 2 合同专项测试、artifact/manifest SHA 校验、Git tracked 检查、`git diff --check`、知识索引写入与 `--check`；以本 Step 最终命令输出为准。

## Results

- 7 个 Decision 的 frame 为 `0..6`，时间为 `2.4..3.0 s`；6 次 Outcome 均与下一轮独立 Decision 的实体 ID、task ID、lifecycle 和状态对齐。
- Route/Comm/Comp 都同时出现非空和空动作帧；Mobility 每步覆盖两个真实 UAV，车辆由 SUMO 推进。
- `Task_1` 的真实 lifecycle 为 `waiting_to_offload -> computing -> computing -> waiting_to_return`，之后保持 `waiting_to_return`。
- acceleration 对比共 18 行：vehicle 6 行匹配常规前向差分；UAV 10 行在恒速时同时等于正负零差分，归为 forward match；两个 UAV 在 `0 -> 10 m/s` 时有限差分约 `+100 m/s^2`，AirFogSim 报告 `-100 m/s^2`，匹配源码的反号公式。
- `gpu=false`、`training=false`、`locked_test=false`。

## Expected vs Actual

预期的跨步独立采集、ID/时间/lifecycle 连续性、四类动作、显式 no-op 和真实车辆/UAV 运动均通过。实际环境由项目 observer 包装类 `ObservedAirFogSimEnv` 承载，其基类是仓库真实 `airfogsim.airfogsim_env.AirFogSimEnv`；验收按真实 MRO 和接口判断，不按包装类名称猜测。

## Known Issues

- AirFogSim UAV acceleration 的符号与常规 `(v_t-v_{t-1})/delta_t` 相反；本 Step 不修改 simulator，也不决定 Dataset 最终采用哪个字段。
- 本 Step 是 Raw Trajectory 接口验收，不证明 Dataset/Tensor、模型或 planner 已实现。
- 第一次运行因 scheduler 导入早于环境 loader 而失败；修正导入顺序后复现。第二次运行仅因对包装类名的硬编码检查失败；读取真实 MRO 后改为验证 `AirFogSimEnv` 基类，全部其他跨步检查当时已通过。

## Git

待最终验证后记录本 Step 的 Conventional Commit 和 push 状态。

## Next Step

研究者审阅本 Step 2.2 的真实轨迹和加速度语义后，再决定是否授权进入 Dataset/Tensor 合同设计。
