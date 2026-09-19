# STEP 2.4 — Communication Outcome Semantics Finalization

## Step Goal

消除 Raw Contract 中“源码声称 wireless+wired、Step 2.3 实际只采 wireless”的矛盾。核对真实 AirFogSim wired 路径，冻结 wireless、wired 和 total 三个 Outcome 字段及空 map/missing 语义，并用一条非 locked 真实短轨迹验收。本 Step 不进入 Dataset/Tensor、双图、World Model、Loss、Planner 或训练。

## Definition Basis

- 研究者 2026-09-19 明确授权 Step 2.4，并要求优先拆分 `wireless_delivered_data_by_task`、`wired_delivered_data_by_task` 和 total。
- `docs/contracts_PIJWM_RAW_SINGLE_DECISION_STEP_CONTRACT_V1.md`
- `docs/PIJWM_IMPLEMENTATION_TRACKER.md`
- Step 2.3 的真实 causal Raw 证据。

## Initial State

`raw_single_decision_step_contract_v1.py` 将 total 写成 direct wireless/wired execution rows，但 `ObservedAirFogSimEnv.pi_jwm_transfer_events` 只在 `_updateWirelessCommunication()` 追加 event；`_updateWiredCommunication()` 仅调用父类。Step 2.3 的 `delivered_data_by_task` 因此实际只是 wireless 聚合，合同、文档和运行事实不一致。

## Files Involved

- `code/scripts/run_p2_single_step_collector_preflight_v1.py`
- `code/src/pi_jwm/raw_trajectory_causal_contract_v1.py`
- `code/src/pi_jwm/raw_single_decision_step_contract_v1.py`
- `code/scripts/run_step2_3_real_airfogsim_raw_contract_finalization_v1.py`
- `code/scripts/run_step2_4_real_airfogsim_communication_outcome_semantics_v1.py`
- 两个 Raw contract 测试文件。
- `code/artifacts/protocols/pi_jwm_communication_outcome_semantics_v1_20260919/`
- Raw Contract、Tracker、authority records、AI_CONTEXT 和知识索引。

## Changes

- 核对真实 wired 路径：`AirFogSimEnv.step()` 在 wireless 后调用 `_updateWiredCommunication()`；该方法将 I/C 链路任务放入 `WiredNetworkManager`，`step(interval)` 返回 `{task_id: transmitted_bytes}`，随后 `Task.transmit_to_Node()` 和 `TaskManager.finishOffloadingTask()` 推进进度/lifecycle。
- 不修改第三方 AirFogSim；在 `ObservedAirFogSimEnv` 中临时包装真实 `wired_manager.step`，把其直接返回值写为 `transport=wired` event。
- wireless event 显式增加 `transport=wireless`，禁止靠空 RB 或 channel name 推断 transport。
- Outcome 拆分为 `wireless_delivered_data_by_task`、`wired_delivered_data_by_task` 和两者按 task 求和的 `delivered_data_by_task`。
- 每类 transport 及 total 都记录 `observed_mask/missing_reason`：hook 可用但本 slot 无服务为 `{}` 且 mask=true；hook 不可用为 `null + mask=false + reason`；任一分量 missing 时 total 也必须 missing。
- 单步 validator 检查 map 非负、mask/reason 一致和 total 等于 wireless+wired。
- Step 2.3 runner 的字段说明同步到最终 split 语义；未改 future task、acceleration 和 CPU helper。

## Reuse

复用 Step 2.3 的真实环境、Decision/Outcome 对齐、四类动作和 causal helper；复用已有 full collector 中包装 `wired_manager.step` 的已验证机制。第三方 simulator 源码保持不变。

## Validation

真实命令：

```powershell
conda run -n airfogsim python .\code\scripts\run_step2_4_real_airfogsim_communication_outcome_semantics_v1.py
```

真实非 locked 轨迹使用 seed 0、6 个 execution slot、7 个独立 Decision，并在运行配置中加入 `RSU_0 ↔ cloudServer_4` 的 100 Mbps 双向 wired link。首 slot 的同一真实任务 `Task_1` 依次完成 `UAV_0 → RSU_0` wireless 和 `RSU_0 → cloudServer_4` wired；post-step task current node 为 `cloudServer_4`、lifecycle 为 `computing`。14 项机器 checks 全部为 true。

专项验证：

```powershell
python -m unittest discover -s .\code\tests -p 'test_raw_trajectory_causal_contract_v1.py' -v
python -m unittest discover -s .\code\tests -p 'test_raw_single_decision_step_contract_v1.py' -v
python -m unittest discover -s .\code\tests -p 'test_run_p2_single_step_collector_preflight_v1.py' -v
python -m compileall -q .\code\src .\code\scripts .\code\tests
```

结果分别为 4/4、9/9、9/9 和 compileall exit 0。最终 artifact/source hash、知识索引、diff 和 Git tracked 状态以本 Step 收尾输出为准。

## Results

- 真实轨迹观察到 1 条 wireless event 和 1 条 wired event；wired source method 为 `wired_manager.step`。
- 首 slot wireless map 与 wired map 均为 `Task_1: 0.28065475894249814`，total 为 `0.5613095178849963`，逐 task 加法一致。
- 后续无通信服务 slot 的三个 map 均为 `{}`，同时 transport/total 的 `observed_mask=true`，明确不是 missing。
- 单元测试另外验证：wired hook unavailable 时 wired/total 为 `null + mask=false + reason`，不能冒充空 map或 0。
- Route 记录保留原始 `['RSU_0', 'cloudServer_4']` 副本；真实 simulator 在执行中修改自己的 route list，不再反向污染 Action 记录。
- `gpu=false`、`training=false`、`locked_test=false`。

## Expected vs Actual

三类通信 Outcome 字段、真实 wired 采集、total 聚合和 empty/missing 区分符合预期。首次真实运行额外遇到 cloud 节点 FogProfile 没有 `cpu`，runner 继续遵守 Step 2.3 的 missing CPU 合同，对该节点保持显式 Comp no-op；没有修改 CPU 机制或伪造容量。

## Known Issues

- AirFogSim 当前配置键使用 `fog_profile.cloud`，实例化 cloud 时读取 `fog_profile.cloud_server`，因此该真实 cloud 节点没有 CPU capacity；这是既有 simulator/config 事实，不属于本通信 Step 的修复范围。
- 当前真实证据覆盖一条 wireless→wired 两跳任务和空服务 slot；它冻结字段语义，不证明吞吐性能或 Dataset/Tensor 设计。
- 本 Step 不证明双图、World Model、Loss、Planner 或训练能力。

## Git

本 Step 使用 Conventional Commit `fix(pijwm): finalize communication outcome semantics`，完成最终验证后推送 GitHub `main`；精确 hash 以 `git log` 和 Completion Report 为准。

## Next Step

研究者审阅本 Step 后，单独授权 **STEP 3 — Dataset / Tensor Contract**。
