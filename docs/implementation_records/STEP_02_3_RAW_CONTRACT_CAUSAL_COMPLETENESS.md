# STEP 2.3 — Raw Contract Causal Completeness & Finalization

## Step Goal

在非 locked 的真实 AirFogSim 短轨迹中收齐 Raw Trajectory 的最后边界：隔离未来任务 schedule，验收 Decision CSI/CPU capacity 与 slot Outcome 服务量，真实执行 return route，并冻结 raw/canonical acceleration 的不同语义。本 Step 不进入 Dataset/Tensor builder、双图、World Model、Loss、Planner 或训练。

## Definition Basis

- 研究者 2026-09-19 明确要求：未来 task 不得进入 `O_t`、History 或 input-side Entity Index；canonical acceleration 固定为 `(v_t-v_{t-1})/delta_t`，缺历史必须 mask。
- `docs/contracts_PIJWM_RAW_SINGLE_DECISION_STEP_CONTRACT_V1.md`
- `docs/PIJWM_IMPLEMENTATION_TRACKER.md`
- Step 2.1/2.2 的真实单步和多步接口证据。

## Initial State

Step 2.2 已通过 6 个真实决策步，但 observer 会把 `_to_generate_task_infos` 中未来到达任务放入 snapshot；Decision `channel_rows`、CPU capacity 和 slot Outcome 两个 map 未在最终 Raw 验收中同时闭合；Route 只验收了 offload；AirFogSim acceleration 与速度后向差分尚未形成不同字段和缺历史 mask。

## Files Involved

- `code/src/pi_jwm/raw_trajectory_causal_contract_v1.py`
- `code/src/pi_jwm/raw_single_decision_step_contract_v1.py`
- `code/scripts/run_step2_3_real_airfogsim_raw_contract_finalization_v1.py`
- `code/tests/test_raw_trajectory_causal_contract_v1.py`
- `code/tests/test_raw_single_decision_step_contract_v1.py`
- `code/artifacts/protocols/pi_jwm_raw_contract_causal_complete_v2_20260919/`
- Raw contract、Tracker、authority records、AI_CONTEXT 和知识索引。

## Changes

- 新增 Raw 因果 helper：按 `arrival_time_s <= decision_time_s` 划分可观测任务与 internal future schedule。
- input-side Entity Index 只从当前可观测实体/任务生成；future task ID 逐 Decision 验证不相交。
- Decision 真实采集 `channel_manager.getCSI` 的逐 RB `channel_rows`。
- CPU capacity 读取 `entity.getFogProfile()['cpu']`；真实环境中部分节点没有 `cpu` 键，因此增加 observation rows，以 `null + observed_mask=false + CPU_NOT_EXPOSED_IN_FOG_PROFILE` 表示，未伪造 0。
- Outcome 的 `delivered_data_by_task` 聚合当前 slot 的真实 transfer events；`served_cpu_work_by_task` 使用同一 Task 的 post-step 与 pre-step `getComputedSize()` 差。
- 真实调用 `TaskScheduler.setTaskReturnRoute`，并验收任务进入 `returning/done`。
- 将 acceleration 分为 `raw_simulator_acceleration_mps2` 与 `canonical_acceleration_mps2`；后者只用当前/历史 speed，首帧或新实体缺历史时为 `null + mask=false`。
- 单决策步 validator 现在拒绝未来任务进入 Decision。

## Reuse

复用 Step 2.1 的真实 `ObservedAirFogSimEnv` transfer hook、Step 2.2 的环境创建/调度模式、现有 AirFogSim observer、CPU inner rule 和四类 scheduler。未修改第三方 AirFogSim。

## Validation

测试驱动首个预期失败：

```powershell
python -m unittest discover -s .\code\tests -p 'test_raw_trajectory_causal_contract_v1.py' -v
```

新增模块前结果为 1 个 ImportError；实现后 3/3 通过。单步合同专项测试为 7/7 通过。

真实命令：

```powershell
conda run -n airfogsim python .\code\scripts\run_step2_3_real_airfogsim_raw_contract_finalization_v1.py
```

真实轨迹从 `2.4 s` 到 `3.2 s`，8 个 execution slot、9 个独立 Decision；17 项机器 checks 全部为 `true`。最终 compileall、专项测试、artifact hash、Git tracked、知识索引、`git diff --check` 以本 Step 收尾输出为准。

## Results

- 首个 Decision 有 9 个 future schedule task、5 个当前可观测 task；逐 Decision 均确认 future task 不进入 `O_t` 或 input-side Entity Index。
- 首个 Decision 有 42 条真实 channel row；CPU observation 有 7 个已观测值和 1 个明确 missing，不用 0 代替。
- 8 个 slot 中，6 个有真实 delivered-data map，6 个有真实 served-CPU map。
- `setTaskReturnRoute` 实际调用 3 次：`Task_1`（frame 3）、`Task_4`（frame 6）、`Task_2`（frame 7），setter 入队和后续 lifecycle 均通过。
- 首个 Decision 的全部 canonical acceleration 均为 `null + mask=false`；后续已有历史的实体使用 backward difference，新出现实体继续显式 mask。
- `gpu=false`、`training=false`、`locked_test=false`。

## Expected vs Actual

因果可观测、四个缺口字段、return route 与双 acceleration 语义全部符合预期。实际环境额外显示部分 FogProfile 不提供 `cpu`，合同按真实缺失值语义修正，没有扩大为 Dataset 设计或修改 simulator。

## Known Issues

- AirFogSim raw acceleration 仍按其源码行为保留，可能与 canonical backward difference 符号相反；该差异是审计信息，不修改第三方。
- 没有 `cpu` 键的节点保持显式 missing；后续 Dataset/Tensor 必须保留 mask，不能用 0 冒充容量。
- 本 Step 只证明 Raw Trajectory Layer；不证明 Dataset/Tensor、双图、模型、loss、planner 或性能结果。

## Git

本 Step 使用 Conventional Commit `feat(pijwm): finalize causal raw trajectory contract`，完成最终验证后推送 GitHub `main`；精确 hash 以 `git log` 和 Completion Report 为准。

## Next Step

研究者审阅本 Step 后，单独授权 **STEP 3 — Dataset / Tensor Contract**。
