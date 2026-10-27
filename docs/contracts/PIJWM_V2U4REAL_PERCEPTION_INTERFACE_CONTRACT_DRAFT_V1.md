# PI-JWM V2U4Real 感知接口与 AirFogSim 接入合同草案 V1

**DRAFT — 仅用于 R2-05B 工程接口验证，等待研究者审阅。**

日期：2026-10-27。范围：固定 `train/2025-07-18-12-53_2` 的单帧车辆/UAV Ouster 读取、模拟感知结果格式、AirFogSim 应用层任务记录接入。本文不冻结新场景、真实检测模型、类别/指标、坐标对齐、通信协议或训练方案。

## 1. 分层结论

本轮把四个状态分开保存：

| 层 | 本轮可验证内容 | 状态口径 |
|---|---|---|
| 数据接入 | PCD/YAML 存在、字段可读、XYZ 有限、固定 SHA 不变、原始帧键保留 | `PASS`（文件层） |
| 感知接口 | `PerceptionFrame` 输入、`DetectionResult` 输出、框格式/维度/坐标声明和非法值拒绝 | `PASS`（MockPerception 软件合同） |
| 应用层 | 检测结果序列化为 payload，进入 AirFogSim `TaskManager` 的 waiting-to-offload 队列 | 依赖实际 AirFogSim 环境；未连接时 `BLOCKED` |
| 无线交付 | 真实速率、RB、字节累计、服务完成、截止时间 | `BLOCKED_NOT_ATTEMPTED`，不得从队列写入推断成功 |

AirFogSim 在本项目中仍是参考仿真器和数据/接口验证工具，不是 PI-JWM 新场景的完整实现。V2U4Real 的 LiDAR 车—UAV 样本也不等于候选车—RSU 视觉协同感知场景。

## 2. 输入合同

实现：`code/src/pi_jwm/v2u4real_perception_interface_v1.py::PerceptionFrame`。

| 字段 | 类型/约束 | 语义边界 |
|---|---|---|
| `frame_key` | 非空字符串 | 原始发布文件键；`1752814590.2` 不能直接称为物理秒 |
| `sensor_id` | 非空字符串 | 例如 `agent-1/ouster` |
| `points` | 有限、非空 `N×3` 数组 | 只保留读取出的 XYZ，不推断单位、轴向或跨端对齐 |
| `coordinate_frame` | 非空声明 | 例如 `v2u4real::agent-1::ouster::native` |
| `coordinate_status` | 非空状态 | 固定样本为 `dataset_native_unverified` |
| `source_metadata` | 映射 | 路径、PCD/YAML SHA、字段头、点数和来源版本 |
| `annotation_metadata` | 可选映射 | 只记录原始标签可追溯性；`MockPerception` 不读取它 |

固定真实样本读取复用 `v2u4real_interface_v1.read_pcd_xyz` 和 `UniqueKeyLoader`。读取器拒绝空点云、重复 YAML 键、非有限 XYZ、字段/字节数不匹配和不支持的 PCD 压缩格式。

## 3. 输出合同

实现：`DetectionObject`、`DetectionResult`。

- 一个结果对应一个 `frame_key + sensor_id`；结果对象的帧、传感器和坐标声明必须与结果头一致。
- 三维框格式固定写成 `center_xyz_length_width_height_yaw`，维度顺序为 `(x, y, z, length, width, height, yaw)`。长度、宽度和高度必须为正；7 个数必须有限。
- 每个对象必须带 `category`、`box3d`、`confidence`、`frame_key`、`sensor_id`、`coordinate_frame`、`coordinate_status`、`result_source` 和 `prediction_semantics`。
- `confidence` 只允许有限 `[0,1]` 数值；结果来源必须显式写出。
- 序列化回读拒绝缺字段、未知字段、错误框格式、非有限值和帧/坐标声明冲突。

`MockPerception.geometry_envelope_v1` 只根据输入点云的 XYZ 包围范围生成一个确定性模拟框，输出明确标成模拟预测，不能当作真实标签、检测模型或精度结果。本轮不计算 Precision、Recall、AP、mAP 或任何正式性能。

## 4. 方案 B 标签边界

标签合并只在内存审计中执行，复用 R2-04C 固定官方转换器行为：车辆先写入，同 ID 再由 UAV 覆盖；车辆独有和 UAV 独有对象都保留；ID 不做字符串/整数归一化。每个对象保存最终来源、共有/独有关系和覆盖记录。

合并结果是来源追溯证据，不能注入 `MockPerception` 的预测，也不能写回原始 YAML 或生成正式训练标签。坐标和物理对齐仍为 `NOT_ESTABLISHED`。

## 5. AirFogSim 应用层边界

实现：`code/src/pi_jwm/v2u4real_airfogsim_adapter_v1.py`。

1. `PerceptionPayloadRecord.from_result` 对检测结果做规范 JSON/UTF-8 序列化，记录 payload SHA-256、字节数、帧键、传感器和坐标声明。
2. `submit_payload_to_task_manager` 调用 AirFogSim 现有 `TaskManager.generateTaskInfoOfMission`、`getWaitingToOffloadTasksByNodeId` 和 `setWaitingToOffloadTasksByNodeId`，把 payload 作为 `perception_payload` 任务记录放入 waiting-to-offload 队列。
3. 适配器不调用 `offloadTask`，不设置 `activated_offloading_tasks_with_RB_Nos`，不调用信道速率，不递增通信字节，不声明任务完成。
4. 回执分别记录 `software_interface_status`、`simulation_task_record_status` 和 `wireless_delivery_status`。无线状态固定为 `NOT_ATTEMPTED_BLOCKED`，直到 R1 通信语义和特征包服务另行通过验证。

AirFogSim 原生 `Task.to_dict()` 的 `task_lifecycle_state` 在“零计算、零回传、尚未 offload”的任务上可能显示为 `finished`（这是该参考实现的属性推导方式）。本合同不把这个字段当作交付证据；适配器另外写入 `application_task_state=waiting_to_offload` 和 `native_lifecycle_interpretation=not_used_as_wireless_delivery_evidence`，并以 waiting-to-offload 队列计数和回执为应用层证据。`perception_payload_bytes` 是 payload 序列化字节数，保留在自定义应用字段中，不把 AirFogSim 的 `task_size` 计算字段改名后冒充无线字节交付。

真实 AirFogSim 环境缺少依赖时，入口必须写 `BLOCKED_DEPENDENCY`；不能用 fake queue、人工速率或成功启动替代实际接入。

## 6. 复跑与验收入口

```powershell
$env:PYTHONUTF8='1'
$env:PYTHONPATH='D:\shen\PKU\PIJWM\code\src;D:\shen\PKU\PIJWM\code\scripts'
python code/scripts/run_v2u4real_r2_05b_interface_v1.py
python -m unittest discover -s code/tests -p 'test_v2u4real_r2_05b_interface_v1.py'
```

机器结果、逐文件哈希、日志和状态分层保存在 `code/artifacts/audit/v2u4real_r2_05b_20261027/`。本草案的合同通过不代表真实无线、物理坐标对齐或完整 PI-JWM 候选场景已实现。
