# STEP 2.1 — 真实 AirFogSim 单轨迹四类动作接线与 Outcome 对齐验收

## Step Goal

在非 locked 范围用真实 AirFogSim 验证 `Decision_t -> Action_t -> Execution_t -> Outcome_t -> Decision_{t+1}`。本记录不覆盖 Dataset/Tensor、双图、World Model、Loss、Planner 或训练。

## Definition Basis

- `docs/contracts_PIJWM_RAW_SINGLE_DECISION_STEP_CONTRACT_V1.md`
- `docs/PIJWM_IMPLEMENTATION_TRACKER.md`
- 研究者确定的动作空间：`A_t=(A_t^Route,A_t^Comm,A_t^Comp,A_t^Mob)`；Mob 只控制 UAV，车辆由 SUMO 推进。

## Initial State

Step 2 已有合同和最小环境四类 setter 测试，但没有完整真实四类 AirFogSim 轨迹证据；Step 2 记录把专项测试误记为 5 项，实际为 6 项。

## Files Involved

- `code/src/pi_jwm/raw_single_decision_step_contract_v1.py`
- `code/scripts/run_step2_1_real_airfogsim_single_step_v1.py`
- `code/artifacts/protocols/pi_jwm_raw_single_decision_step_real_airfogsim_v4_20260919/`
- `docs/contracts_PIJWM_RAW_SINGLE_DECISION_STEP_CONTRACT_V1.md`
- `docs/implementation_records/STEP_02_RAW_TRAJECTORY_ACTION_CONTRACT.md`
- 本记录、Tracker、计划/进度/发现记录和 AI_CONTEXT。

## Changes

- 使用项目已有 `_build_environment` 启动真实 `AirFogSimEnv`，warm-up 后选择一个真实 waiting task。
- 实际调用 Route、Comm、Comp callback、UAV Mobility 四类 scheduler，并调用真实 `env.step()`。
- Outcome 和下一 Decision 均直接从同一个真实 post-step 环境读取；没有 `_MinimalEnv`，没有手工构造 Outcome。
- 将实体观测字段从统一 `azimuth_rad` 修正为 `heading`：vehicle 为 degree，UAV 为 rad；UAV Mobility 动作继续使用 `azimuth_rad`。
- 修正 Step 2 专项测试记录为 6 项。

## Reuse

复用 `run_p2_single_step_collector_preflight_v1._build_environment`、`SingleStepRecorder`、真实 AirFogSim scheduler 和 traffic/task manager；未改模型、数据、loss、planner 或训练配置。

## Validation

命令：

```powershell
$env:PYTHONPATH='D:\shen\PKU\PIJWM\code\src;D:\shen\PKU\PIJWM\code\scripts'
python -m unittest discover -s .\code\tests -p 'test_raw_single_decision_step_contract_v1.py' -v
conda run -n airfogsim python .\code\scripts\run_step2_1_real_airfogsim_single_step_v1.py
```

结果：专项测试 `Ran 6 tests ... OK`；真实运行 warm-up 24 步，决策时间 2.4 s，真实 slot 0.1 s，Outcome/next Decision 时间 2.5 s。artifact 的 13 项 checks 全部为 `true`：真实环境、非 `_MinimalEnv`、真实 Outcome、四类 setter、UAV-only mobility、SUMO 车辆继续推进、动作身份、时间推进、Outcome/next Decision 身份与时间、traffic fields、UAV phi、task ID 对齐。

## Results

- Task：`Task_1`；Route：`UAV_0 -> vehicle_0`；Comm：RB `[0]`。
- Comp 使用真实 CPU callback，Task_1 computed size 从 `0.0` 增加 `0.2`。
- Mobility 实际作用于 `UAV_0`，speed `10.0`、angle `0.2`、phi `0.0`；vehicle 位置由 SUMO 推进并发生变化。
- `trajectory_id=step2.1-real-seed0` 保持不变，frame `0 -> 1`，Outcome 快照等于下一 Decision 快照。
- 环境边界：`gpu=false`、`training=false`、`locked_test=false`。

## Expected vs Actual

预期是真实四类动作都进入 scheduler，并以真实 post-step 状态闭合到下一 Decision；实际全部通过。真实接口暴露一个必须修正的合同事实：vehicle traffic `angle` 是 degree，而 UAV `angle`/`phi` 按 rad 参与运动，因此合同不再把实体角度统一标为 `azimuth_rad`。

## Known Issues

- AirFogSim 当前 `acceleration=(last_speed-speed)/traffic_interval`；本次 UAV 速度从 0 到 10 m/s 时真实记录为 `-100.0`。这是仿真器观察事实，本 Step 不自行修正。
- 只验收一个决策步；跨多个决策步的反馈闭环仍未验证。
- generated registry 检查未发现 `index-build.tmp.err` 或 `tmp.err` 被纳入。

## Git

Step 2.1 代码和记录由 commit `0a11d41` 推送，但 v2 JSON 受 `.gitignore` 影响未进入该 commit。Step 2.2 追加真实 v4 重跑并将 v4 JSON/manifest 强制纳入版本控制，补齐 GitHub 可追溯性；JSON 使用 LF 写入，manifest SHA 与 Git blob 字节一致。

## Next Step

研究者审阅本记录和真实 artifact 后，单独授权一个跨决策步真实反馈闭环验收。
