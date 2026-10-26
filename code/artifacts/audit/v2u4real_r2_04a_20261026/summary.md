# STEP R2-04A — 真实单帧官方变换几何诊断

任务执行：**COMPLETED**。数学检查通过也不能证明真实物理对齐。
固定训练场景 `2025-07-18-12-53_2`、发布键 `1752814590.2`；agent 1 车辆、agent 2 UAV，均 Ouster。

- `input_integrity_status`：`PASS`
- `official_source_status`：`PASS`
- `software_transform_status`：`PASS`
- `annotation_diagnostic_status`：`official_code_interpretation_only`
- `physical_alignment_status`：`NOT_ESTABLISHED`
- `task_execution_status`：`COMPLETED`

## 输入证据（完整字段与原文见 result.json）

- vehicle：141177 点；位姿 `[162.4558620813, -195.5474155993, -3.0822680281, -179.1582303045, 128.6537437347, 0.4917033974]`；标注 6 个。
  - `code/artifacts/data/v2u4real_r2_01/selected_samples/train/2025-07-18-12-53_2/1/ouster/1752814590.2.pcd`：2259016 bytes，SHA-256 `ed22da92e681b36d2cbaadcde02d1e47d04a96bb9b999df80a583af5d8891ae6`，前后不变 `True`。
  - `code/artifacts/data/v2u4real_r2_01/selected_samples/train/2025-07-18-12-53_2/1/yaml/ouster/1752814590.2.yaml`：3592 bytes，SHA-256 `b31f030d4cf28284e9b53105a7210fc56578639416fff6db5d1ea3bc629c7449`，前后不变 `True`。
- uav：70532 点；位姿 `[146.0642330580825, -178.5034348014285, -51.65063285498502, -179.0519502518403, 131.9188901918, 0.9285541649206536]`；标注 2 个。
  - `code/artifacts/data/v2u4real_r2_01/selected_samples/train/2025-07-18-12-53_2/2/ouster/1752814590.2.pcd`：1128694 bytes，SHA-256 `3113cbedaf517d72d51d2019871e77671f38b1413ef96253dc8067ecf74874d5`，前后不变 `True`。
  - `code/artifacts/data/v2u4real_r2_01/selected_samples/train/2025-07-18-12-53_2/2/yaml/ouster/1752814590.2.yaml`：1390 bytes，SHA-256 `9f0142f2ae4c153578f8ccf7c07ad29f3d395bf79bd01ddf5d8a91e56689a7db`，前后不变 `True`。

## 数学结果

实际调用固定官方 `x1_to_x2(uav_pose, vehicle_pose)` list/list 分支。
矩阵将 UAV 局部列向量映射到车辆局部：`inv(T_world_from_vehicle) @ T_world_from_uav`。车辆点云保持原坐标。

```text
[[ 9.98347855e-01  5.68211050e-02 -8.53942550e-03  2.39645883e+01]
 [-5.68308746e-02  9.98383409e-01 -9.05597609e-04 -1.44415272e+00]
 [ 8.47416368e-03  1.38940445e-03  9.99963128e-01  4.83909266e+01]
 [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]]
```

独立 solve 的矩阵最大差：3.5527136788005e-14；全部 UAV 点正逆最大误差 4.2632564145606e-14，坐标 RMSE 5.3668473728457e-15。
确定性 128 点的 8128 对距离最大差 5.6843418860808e-14。容差固定见 config，未自动放宽。

## 标注诊断

执行哈希验证后提取的官方函数原体；未执行 converter main，也没有合并或覆盖同 ID 框。
`location+center`、`angle` 按代码原顺序、`extent/2` 生成角点，再执行官方 corner_to_center；保存完整角点及最终简化框。
同 ID 中心间距（代码单位，仅观察）：`{'1': 0.2331088987578514, '2': 0.7801747224527404}`。
框只标记 `official_code_interpretation_only`；不把同 ID 当作已验证匹配，不验收跨端框对齐。

## 图与解释边界

`bev_overlay.png` 是鸟瞰图（从上方查看 XY 平面）；每端最多 20,000 个确定性点，一致范围、等比例坐标，完整框角点投影。
图像只能帮助发现异常。真实单位、轴向、公共原点、跨端同步及配准修正未建立；不得据此手调矩阵或宣布标定通过。
点变换复用 float64 工具；官方转换器的 torch/float32 投影没有执行，不能宣称完整导出器逐位复现。

## 未解决事项

- Same published key is a file association, not measured physical synchronization.
- Physical length unit, local axes, common origin and applied predefined axes transform are unverified.
- RTK/GICP/manual registration corrections and whether already applied remain unverified.
- Annotation angle order in paper differs from code; location/center semantics remain unresolved.
- list/matrix x1_to_x2 branch uses its matrix directly despite its docstring; only identity is passed there for boxes.
- Boxes are official_code_interpretation_only, not independently verified ground truth.
- No correction, ICP/GICP, physical acceptance, GPU, training, replay or locked_test.

## 错误

[]

CPU-only；GPU、locked_test、AirFogSim、无线服务与训练均未使用。独立测试见 validation.json；实际命令和输出见 run.log。
本任务到此停止，等待研究者审阅；不自动进入 R2-04B。

## 图像审阅观察

车辆与变换后 UAV 在部分区域重叠，但覆盖形状和范围明显不同；UAV 在图上主要覆盖狭长区域，车辆更分散，远处车辆框可延伸到当前点云范围外。没有可靠的跨端点/地标对应或物理真值，不能据此量化配准误差或判定覆盖差异成因。审阅与图像哈希见 visual_review.json。
