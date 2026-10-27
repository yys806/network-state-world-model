# STEP R2-05B — V2U4Real 感知接口与 AirFogSim 软件接入验证

日期：2026-10-27
状态：**软件链路完成；真实无线交付阻塞；等待研究者审阅**

## Step Goal / 目标

在不训练模型、不使用 GPU/`locked_test`、不修改原始数据和第三方仿真器源码的前提下，验证：

```text
固定真实 PCD/YAML → 严格读取 → MockPerception 输入/输出 → payload → AirFogSim 应用任务记录
```

本 Step 不验证真实检测精度、物理坐标对齐、真实无线传输、特征包服务或完整 PI-JWM 候选场景。

## Definition Basis / 定义依据

- R1：`docs/implementation_records/STEP_V2U4REAL_AIRFOGSIM_R1.md`。R1 已明确真实数据接入和特征包服务未通过，AirFogSim 只是参考仿真器。
- R2-04C：`docs/implementation_records/STEP_R2_04C_V2U4REAL_LABEL_MERGE_POLICY.md` 与 `docs/contracts/PIJWM_V2U4REAL_LABEL_MERGE_POLICY_V1.md`。方案 B 已冻结：车辆先写入，同 ID 由 UAV 覆盖，独有对象保留。
- R2-05A：`docs/implementation_records/STEP_R2_05A_V2U4REAL_BASELINE_DATA_EVAL_AUDIT.md` 与 `docs/contracts/PIJWM_PERCEPTION_BASELINE_DATA_EVAL_CONTRACT_DRAFT_V1.md`。固定 LiDAR 样本可用于接口核验，但物理对齐和完整候选场景仍未建立。
- 固定输入：数据版本 `36f64423a639db7c5bbc8ac8575b1ee74d7b8e9a`，`train/2025-07-18-12-53_2`，帧键 `1752814590.2`，agent `1/2`，传感器 `ouster`。

## Initial State / 初始状态

已有 `v2u4real_interface_v1.py` 只能做严格 PCD/YAML 读取、字段审计、显式刚体工具和隔离回放；没有感知输入/检测输出合同，也没有 V2U4Real payload 到 AirFogSim 应用任务的独立适配器。R2-04C 的合并脚本和 R1 的 AirFogSim 回放辅助函数可以复用。工作区在任务开始时已有用户修改、删除和未跟踪文件；本 Step 未清理、覆盖或混入它们。

## Files Involved / 文件

新增：

- `code/src/pi_jwm/v2u4real_perception_interface_v1.py`：`PerceptionFrame`、`DetectionObject`、`DetectionResult`、严格回读校验、固定样本读取和确定性 `MockPerception`。
- `code/src/pi_jwm/v2u4real_airfogsim_adapter_v1.py`：payload 序列化、TaskManager waiting-to-offload 接入、分层回执。
- `code/scripts/run_v2u4real_r2_05b_interface_v1.py`：CPU 入口、真实帧读取、R2-04C 只读证据复用、AirFogSim 双环境探针和机器结果落盘。
- `code/tests/test_v2u4real_r2_05b_interface_v1.py`：7 项接口/确定性/非法输入/真实单帧/应用接入测试。
- `docs/contracts/PIJWM_V2U4REAL_PERCEPTION_INTERFACE_CONTRACT_DRAFT_V1.md`：未冻结的接口合同草案。
- `code/artifacts/audit/v2u4real_r2_05b_20261027/`：真实数据、标签追溯、感知输出、日志、基础环境阻塞和 AirFogSim 环境 CPU 冒烟证据。

同任务增量更新：`AI_CONTEXT/00_PROJECT_STATE.md`、`AI_CONTEXT/03_DATA_FLOW.md`、`AI_CONTEXT/04_MODULE_MAP.md`、`AI_CONTEXT/07_KNOWN_ISSUES.md`、`AI_CONTEXT/08_CHANGELOG.md`、`记录/本地计划表.md`、`记录/8.12之后推进.md`、`docs/CHANGELOG.md`、项目索引和本记录。

## Changes / 变化

### 1. 真实数据读取

复用 `pi_jwm.v2u4real_interface_v1.read_pcd_xyz` 与 `UniqueKeyLoader`。实际读取了四个原始文件：

- 车辆 PCD：141,177 个 XYZ 点，2,259,016 bytes，SHA-256 `ed22da92e681b36d2cbaadcde02d1e47d04a96bb9b999df80a583af5d8891ae6`。
- 车辆 YAML：3,592 bytes，SHA-256 `b31f030d4cf28284e9b53105a7210fc56578639416fff6db5d1ea3bc629c7449`。
- UAV PCD：70,532 个 XYZ 点，1,128,694 bytes，SHA-256 `3113cbedaf517d72d51d2019871e77671f38b1413ef96253dc8067ecf74874d5`。
- UAV YAML：1,390 bytes，SHA-256 `9f0142f2ae4c153578f8ccf7c07ad29f3d395bf79bd01ddf5d8a91e56689a7db`。

PCD 头部为 `x y z rgb`、F4、count1、binary；XYZ 均有限。帧键原样保留并标为 `identifier_only_not_physical_timestamp`。四个文件在执行前后 SHA/字节数均未改变。坐标状态仍为 `dataset_native_unverified`，物理对齐仍为 `NOT_ESTABLISHED`。

### 2. 方案 B 标签追溯

复用 R2-04C `audit_frame` 和官方固定源码函数，仅读取/内存合并，不写回 YAML。固定帧车辆 ID 为 `1,2,3,4,5,6`，UAV ID 为 `1,2`；共有 ID `1,2` 均由 UAV 覆盖，车辆独有 `3,4,5,6` 保留，输出并集共 6 个对象，集合检查全部通过。MockPerception 明确不读取这些标签；`labels_used_by_mock_perception=false`。

### 3. 感知输入/输出合同

输入 `PerceptionFrame` 强制非空有限 `N×3` 点云、帧键、传感器、坐标系/状态和来源元数据。输出 `DetectionResult` 强制对象列表、类别、有限置信度、`center_xyz_length_width_height_yaw` 框格式、维度顺序、帧键、传感器、坐标声明和来源。

`MockPerception.geometry_envelope_v1` 只用点云 XYZ 外包围盒生成固定模拟对象，结果标明 `prediction_semantics=predictions_are_not_ground_truth`。标签变更不会改变输出；重复运行的结果和 payload SHA 完全一致。本轮没有精度、召回率、AP 或 mAP。

### 4. AirFogSim 应用层接入

`PerceptionPayloadRecord` 对检测结果做规范 JSON/UTF-8 序列化并记录 payload SHA、字节数和来源。适配器使用已有 `TaskManager.generateTaskInfoOfMission`、`getWaitingToOffloadTasksByNodeId`、`setWaitingToOffloadTasksByNodeId`，为 `vehicle_0` 和 `UAV_0` 各放入一个 `perception_payload` waiting-to-offload 任务。

在 `airfogsim` conda 环境实际执行两次相同 CPU smoke：每次一个 `AirFogSimEnv.step()`，仿真时间从 0 到 0.1；两端任务入队前后均为 1；总计和分信道 `data_size` 均保持 0，只有时钟字段前进；两次落盘状态一致。没有调用 `offloadTask`、RB setter、速率计算或通信完成路径。

AirFogSim 原生 `Task.to_dict()` 在零计算/零回传任务上会把 `task_lifecycle_state` 显示成 `finished`。这不是本 Step 的交付证据。适配器另写入 `application_task_state=waiting_to_offload` 和 `native_lifecycle_interpretation=not_used_as_wireless_delivery_evidence`；验收依据是 waiting 队列和分层回执。

## Reuse / 复用

- R1 严格 PCD/YAML 读取器和 R1 `make_absolute_replay_manager`、`prepare_config`、`snapshots`。
- R2-04C 固定官方函数执行和方案 B 来源追溯。
- vendored AirFogSim 原生 `AirFogSimEnv`、`TaskManager` 和任务队列接口；未修改 `code/reference/AirFogSim/`。

## Validation / 实际验证

### 入口与机器结果

```powershell
$env:PYTHONUTF8='1'
$env:PYTHONPATH='D:\shen\PKU\PIJWM\code\src;D:\shen\PKU\PIJWM\code\scripts'
python code/scripts/run_v2u4real_r2_05b_interface_v1.py
# 基础环境 exit 2：data/label/perception PASS；AirFogSim BLOCKED_DEPENDENCY（No module named 'traci'）

conda run --no-capture-output -n airfogsim python code/scripts/run_v2u4real_r2_05b_interface_v1.py `
  --output code/artifacts/audit/v2u4real_r2_05b_20261027/airfogsim_cpu_smoke_20261027 `
  --label-merge-receipt code/artifacts/audit/v2u4real_r2_05b_20261027/label_merge.json
# exit 0：overall PASS_SOFTWARE_CHAIN_WITH_WIRELESS_BLOCK
```

最终软件链机器回执为：

| 层 | 状态 |
|---|---|
| 数据接入 | `PASS` |
| 方案 B 标签追溯 | `PASS`（复用已通过的只读回执） |
| 感知接口/Mock | `PASS` |
| AirFogSim 软件接入 | `PASS`（实际 `AirFogSimEnv` CPU） |
| 仿真任务记录 | `PASS`（waiting-to-offload 队列） |
| 真实无线交付 | `BLOCKED_NOT_ATTEMPTED` |

### 专项和回归测试

| 测试 | 结果 |
|---|---|
| `test_v2u4real_r2_05b_interface_v1.py` | 7/7 `OK` |
| `test_v2u4real_r2_05a_baseline_data_eval_v1.py` | 19/19 `OK` |
| `test_v2u4real_r2_04c_label_merge_policy_v1.py` | 16/16 `OK` |
| `test_v2u4real_r2_04b_multiframe_coordinate_consistency_v1.py` | 15/15 `OK` |
| `test_v2u4real_r2_04a_single_frame_geometry_v1.py` | 17/17 `OK` |
| `test_v2u4real_full_sample_v1.py` | 9/9 `OK`；因工作区 `.tmp` ACL，使用本任务输出目录重定向临时目录 |
| `test_v2u4real_preparation_v1.py` | 8/8 `OK`；同上 |
| R1 旧接口 `GeometryTests + DatasetAuditTests` | 12/12 `OK` |
| R1 `AirFogSimTests`（基础环境） | 0 tests，`setUpClass` 因缺 `traci` 报错；不计为通过 |

变更文件做了无写入编译检查；新增测试和脚本均可导入。工作区 `.tmp` 直接 `tempfile.mkdtemp` 会卡住，未清理或修改该目录。

## Results / Expected vs Actual / 结果

预期是固定真实帧能读、接口能拒绝非法输入、模拟输出可重复、检测结果能进入实际应用层，同时不把队列或仿真时钟当成无线成功。实际四个软件层均取得对应证据；基础环境依赖缺口和 conda 环境可运行但缺 SciPy 的差异已分别记录；最终通过的 AirFogSim smoke 复用了已通过的 R2-04C 只读标签回执，没有新的科学决策。

**符合本轮软件接口目标，但整体状态为部分符合：真实无线交付、物理坐标对齐、真实感知模型和完整 PI-JWM 新场景仍未实现或未验证。**

## Known Issues / 当前问题

1. R1 的特征包服务、无线资源分配、实际速率、累计字节、完成事件和截止时间语义仍未通过；本轮 `BLOCKED_NOT_ATTEMPTED`。
2. V2U4Real 原始坐标、轴向、单位、物理时间和跨端对齐仍未建立。
3. 基础 Python 缺 `traci`，`airfogsim` 环境缺 `scipy`；本轮不修改环境依赖，使用只读回执跨环境复用。
4. AirFogSim 原生任务生命周期字段与 waiting 队列语义不一致，已通过自有状态字段和回执隔离；后续若要把 payload 送入无线路径，必须先由研究者确认任务语义和通信合同。
5. 当前候选研究场景仍是车路协同感知 Problem Formulation；本 Step 不宣称已实现该新场景。

## Git / Next Step

本记录和代码只包含 R2-05B 相关增量；任务开始前的 `AGENTS.md` 修改、历史资料删除、旧脚本和索引差异未纳入。`python code/scripts/build_project_knowledge_index_v1.py --check` 实际返回 exit 1，既有五项生成索引不同步：`tracked_file_inventory.csv`、`python_dependency_map.json`、`artifact_catalog.csv`、`archive_candidate_registry.csv`、`registry_summary.json`；本轮没有写入式重建，避免覆盖既有工作区状态，因此该验证门仍阻塞推送。

Git：代码与接口独立提交 `525f3c0`（`feat(v2u4real): add perception and AirFogSim ingress interface`）；本次记录修订另作文档提交。由于知识索引 `--check` 的既有五项不同步仍未解决，本轮不推送。任务开始前的工作区修改、删除和未跟踪脚本保持原样，未纳入本提交。

唯一下一步：研究者审阅本 Step 的接口合同、AirFogSim 原生生命周期语义差异和 `BLOCKED_NOT_ATTEMPTED` 无线边界，决定是否另行授权通信语义/特征包服务核验；本轮停止，不自动进入真实感知模型或训练。
