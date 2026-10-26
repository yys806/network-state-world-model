# STEP R2-04A — V2U4Real 单帧官方坐标变换与几何诊断

日期：2026-10-26（Asia/Shanghai）。**有限诊断完成，单帧官方变换的数学一致性通过；真实物理空间对齐仍 NOT_ESTABLISHED。** 工程推送因既存知识索引门未通过而停止。本任务不代表完整 R2-04 坐标与标定验收。

## Step Goal / Definition Basis

依据研究者本轮完整 R2-04A 指令，只对固定的一组真实训练文件执行 CPU 官方变换与几何诊断。R2-03 曾阻止未经说明的真实坐标验收；本轮新授权明确允许 **软件数学检查及 official_code_interpretation_only 标注诊断**，没有解除物理同步、轴向、标定和标注合同阻塞。

输入数据版本 `36f64423a639db7c5bbc8ac8575b1ee74d7b8e9a`，划分 `train`，场景 `2025-07-18-12-53_2`，文件键 `1752814590.2`，车辆 agent `1`、UAV agent `2`，均 Ouster。固定键仅为发布文件关联；没有推断其物理时间、测量配对偏差或同步通过。

官方仓库 `https://github.com/VjiaLi/V2U4Real`，commit `e099a3a746c6092a614e18b74d1f5b2f3c8351e8`。来源为本地已固定的部分源码快照，不是完整安装，也不是本轮重新从远程认证。核对 `V2U4Real.SOURCE.json` 中的仓库、commit、全部 9 个文件 SHA-256 和 Git blob，并额外在诊断脚本中固定下列 3 个关键 SHA。实验完成时再次验证通过。

| 文件 | SHA-256 |
|---|---|
| `opencood/utils/transformation_utils.py` | `2c198a94ffaf9ac3087396ee364b866a817fb2a075ac7c37c618cb03091ebd80` |
| `opencood/utils/uav_to_single.py` | `cab7b48fe7c343cbb356db131cd1b87f64249f9cfcb7b4073fdeac18ed51ec4b` |
| `opencood/utils/box_utils.py` | `820ac498be2fa586ab2c797344d55dbd3681b6e3b21536ea00412990a9777657` |

本轮未重新阅读论文全文，沿用 R2-03 已记录的论文与代码合同冲突作为未解决事项；没有把它升级为本轮已验证的论文事实。

## Initial State

初始分支 `main`，HEAD `d842cbacd16c3a9c33bda290b2e9f71d1dae0361`。任务前已存在 AGENTS、AI_CONTEXT/00、计划、索引等修改，以及论文/文献目录和三个过程文件的未提交删除、若干未跟踪脚本。完整初始清单在结果目录 `initial_workspace.json`。不 reset/clean，不恢复删除，不更改 AGENTS，不提交这些既有变更。

`task_plan.md`、`progress.md`、`findings.md` 仍按项目记忆 C7 保持缺失；本次计划、进度、发现统一记录到 `记录/本地计划表.md` 和本文，不擅自重建这三个文件。

已有索引问题：R2-03 记录 3 个生成文件不同步。本轮只读 `--check` 启动于诊断源码新增前、结束于编辑期间，报告 5 项不同步；不能把全部 5 项都声称为任务前已存在。本轮没有执行写入式 builder，避免重写既有删除/修改对应的无关索引。

## Files Involved / Changes / Reuse

- 新脚本：`code/scripts/audit_v2u4real_r2_04a_single_frame_geometry_v1.py`。
- 新测试：`code/tests/test_v2u4real_r2_04a_single_frame_geometry_v1.py`。
- 结果：`code/artifacts/audit/v2u4real_r2_04a_20261026/` 中 `result.json`、`summary.md`、`bev_overlay.png`、`config.json`、`transform_inputs.json`、`run.log`、`validation.json`、`source_identity.json` 及上下文/索引检查回执。
- 复用 `pi_jwm.v2u4real_interface_v1` 的 `UniqueKeyLoader`、`read_pcd_xyz`、`sha256`、`rigid_matrix`、`transform_points`；未更改该模块。
- 更新实施/实验/研究/结果导航、人工实验注册表和问题路由，以及受影响的 AI_CONTEXT。生成索引没有重建；原始数据、第三方源码、模型/双图/Loss/Planner/Scorer/训练配置均无修改。

官方部分快照缺少完整 OpenCOOD 依赖。因此在 SHA 核验后，以 Python 抽象语法树（保留函数原体，只选择需要执行的函数）加载官方 `x_to_world`、`x1_to_x2`、`create_bbx`、`corner_to_center`、`load_annotation_objects` 以及原 `CLASS_MAP`。没有改写函数、伪造导入模块或替代官方数学公式。实际文件、起止行号、SHA、`ast_body_modified=false` 均写入 `result.json.official_source.execution`。

点坐标应用使用已有 NumPy float64 `transform_points`。官方完整导出器使用 `pcd_to_np`（`pcd_utils.py:40` 明确返回 float32）和 `project_points_by_matrix_torch`；本轮没有执行此读取/torch 投影或 converter main，故**只证明官方矩阵及 float64 几何计算，不能宣称完整导出器逐位复现**。独立测试另以齐次列向量、`solve(T_world_from_vehicle, T_world_from_uav @ points)` 核对全部点，未仅复用受测函数自证。

## 输入实测与完整性

共同目录前缀：`code/artifacts/data/v2u4real_r2_01/selected_samples/train/2025-07-18-12-53_2/`。四文件实验前 SHA 均与 R2-03 对应条目一致；实验后 SHA 全部不变。点云解码 XYZ SHA 同时与 R2-03 对照通过。

| 相对上述前缀的路径 | bytes | SHA-256 |
|---|---:|---|
| `1/ouster/1752814590.2.pcd` | 2,259,016 | `ed22da92e681b36d2cbaadcde02d1e47d04a96bb9b999df80a583af5d8891ae6` |
| `1/yaml/ouster/1752814590.2.yaml` | 3,592 | `b31f030d4cf28284e9b53105a7210fc56578639416fff6db5d1ea3bc629c7449` |
| `2/ouster/1752814590.2.pcd` | 1,128,694 | `3113cbedaf517d72d51d2019871e77671f38b1413ef96253dc8067ecf74874d5` |
| `2/yaml/ouster/1752814590.2.yaml` | 1,390 | `9f0142f2ae4c153578f8ccf7c07ad29f3d395bf79bd01ddf5d8a91e56689a7db` |

车辆 141,177 点、UAV 70,532 点。两端头字段均为 `x y z rgb`、4 字节浮点、binary、HEIGHT=1，点数/宽高/负载大小严格匹配，XYZ 全有限。车辆字节与解码 XYZ 均不属于 R2-03 两组已知重复内容；这不是整个数据集的全局唯一性证明。

YAML 无重复键；两端 `lidar_pose` 均为有限数字 `list[6]`，顺序保持原样。根字段类型、位姿字段元素类型/长度/有限性、完整 YAML 文本、全部原始 vehicles 与对象 ID 保留在结果文件中。车辆 6 个对象（ID 1–6），UAV 2 个（ID 1、2）；两端 `center` 本样本均为 `[0,0,0]`，只作字段观察，不据此定义语义。

```text
车辆原始位姿 [x,y,z,roll,yaw,pitch]
[162.4558620813, -195.5474155993, -3.0822680281,
 -179.1582303045, 128.6537437347, 0.4917033974]
UAV 原始位姿 [x,y,z,roll,yaw,pitch]
[146.0642330580825, -178.5034348014285, -51.65063285498502,
 -179.0519502518403, 131.9188901918, 0.9285541649206536]
```

## 变换路径与数值结果

官方函数按 `[x,y,z,roll,yaw,pitch]`、度、`Rotation.from_euler('ZYX',[yaw,pitch,roll])` 消费输入。

`T_world_from_vehicle`、`T_world_from_uav` 分别将车辆/UAV 局部坐标映射到代码公共世界参考系。实际调用 **`x1_to_x2(uav_pose_list, vehicle_pose_list)`**，触发官方 list/list 分支，得到：

`T_vehicle_from_uav = inv(T_world_from_vehicle) @ T_world_from_uav`。

其源是 UAV 局部，目标是车辆局部；使用齐次列向量左乘。车辆点云保持原坐标。执行点变换前写出原始两端位姿及三份矩阵到 `transform_inputs.json`。

```text
T_vehicle_from_uav（展示舍入，完整 float64 值见 result.json）
[[ 0.998347855328,  0.056821105009, -0.008539425496, 23.964588335125],
 [-0.056830874638,  0.998383409107, -0.000905597609, -1.444152724453],
 [ 0.008474163682,  0.001389404450,  0.999963128373, 48.390926569625],
 [ 0,               0,               0,               1              ]]
```

| 检查 | 实测 | 预设软件容差 | 结果 |
|---|---:|---:|---|
| 三份矩阵形状/有限性/齐次末行 | 4×4、有限、末行 [0,0,0,1] | 严格结构 | PASS |
| 相对旋转正交误差最大值 | 6.661338147750939e-16 | 1e-10 | PASS |
| 相对旋转行列式 | 0.9999999999999998 | 与 +1 相差 ≤1e-10 | PASS |
| 官方 inv+dot 与独立 solve 矩阵最大差 | 3.552713678800501e-14 | 1e-12 | PASS |
| 正/逆两个乘法顺序离单位阵最大差 | 3.552713678800501e-15 | 1e-12 | PASS |
| 全部 70,532 点正逆最大坐标绝对误差 | 4.263256414560601e-14 | 1e-8 | PASS |
| 正逆每坐标均方根误差 RMSE | 5.366847372845697e-15 | 记录项 | — |
| 正逆每点距离均方根误差 | 9.295652326236296e-15 | 记录项 | — |
| 128 点全部 8,128 对距离最大变化 | 5.684341886080802e-14 | 1e-8 | PASS |
| 变换后质心与直接变换质心最大差 | 4.618527782440651e-14 | 1e-8 | PASS |

确定性子集采用 `unique(linspace(0,N-1,count,dtype=int64))`；没有人工挑点。float64、绝对容差、rtol=0、子集大小和分位数位置均写入运行前的 `config.json`，运行后没有放宽。

UAV 变换前/后/逆变换点数均 70,532、XYZ 全有限。变换前坐标范围约 `[-41.5020,-107.3336,-54.2620]` 至 `[24.0311,98.0709,-40.9708]`，变换后约 `[-21.2264,-109.4382,-5.7026]` 至 `[52.5915,97.1703,7.4886]`；质心从 `[-4.93375,-0.90644,-50.94081]` 变为 `[19.42249,-2.02261,-2.59107]`。这些是代码长度单位，不是独立核实的米。逐轴 min/max、标准差或分位数会随旋转改变，不应要求它们数值不变；完整统计在 `result.json.point_checks`。

## 标注与可视化诊断

官方转换器原体对车辆 6 个、UAV 2 个对象执行：`location+center` 组成位置、原 `angle` 按代码 pose 顺序组成姿态，`x1_to_x2(object_pose,np.eye(4))` 后对 UAV 左乘相对矩阵；`create_bbx(extent)` 使用每轴 `extent/2` 构造 8 角点，最后由 `corner_to_center(order='lwh')` 得到中心/尺寸/yaw。保存原始类别、字段、物体矩阵、完整 8 角点、官方最终输出，并核对暴露的中间角点与转换器最终值完全一致（最大差 0）。

没有依据注释重新解释 extent 顺序，也没有混用论文角度顺序。最终简化框只保留 yaw，不能等同于所有倾斜 8 角点；图中使用实际完整角点的 12 条边投影，而不是再从简化输出重造框。车辆/UAV 框独立保存，没有执行官方 main 的同 ID UAV 优先覆盖。

**状态 `official_code_interpretation_only`。** 代码解释下，同 ID 1、2 的两端框中心距离分别为 0.2331088987578514、0.7801747224527404 个代码长度单位。没有把同 ID 认定为已独立验证的跨端匹配，没有预设物理误差门槛或宣布框对齐通过。

`bev_overlay.png` 鸟瞰图每端使用 20,000 个确定性下采样点（数值验收仍使用全部点），蓝色车辆原点云，橙色变换后 UAV；框颜色对应来源、UAV 框为虚线。使用共同完整 XY 范围（含全部框）、等比例轴；图中注明单位、物理轴向和同步未核验，物理对齐 NOT ESTABLISHED。没有 ICP/GICP 或手动校正。

**实际观察**：两端在部分区域有重叠，但覆盖形状和范围明显不同；图上 UAV 主要覆盖狭长区域，车辆覆盖更分散，远处车辆框可延伸到当前点云范围外。此图没有建立点对点、地标或可靠真值对应，因此不能量化物理配准误差，也不能把覆盖差异解释为轴错、姿态错或标注错；这些成因保持未判定。没有新发现可由当前证据独立确认的源码/论文冲突。

## Validation（实际命令与输出）

PowerShell，cwd=`D:\shen\PKU\PIJWM`，`PYTHONUTF8=1`，`PYTHONPATH=code/src;code/scripts;code/tests`（实际环境为绝对路径）。完整执行输出、包括早期验证和最终源码复验，保留在 `run.log`；最终验收分组见 `validation.json`。

```powershell
python code/scripts/audit_v2u4real_r2_04a_single_frame_geometry_v1.py
# exit 0; input/source/software PASS; annotation official_code_interpretation_only
# physical_alignment NOT_ESTABLISHED; task_execution COMPLETED
python -m unittest test_v2u4real_r2_04a_single_frame_geometry_v1.SyntheticGeometryTests -v
# 12 tests OK
python -m unittest test_v2u4real_r2_04a_single_frame_geometry_v1.RealSampleTests -v
# 5 tests OK
python -m unittest test_v2u4real_full_sample_v1 test_v2u4real_preparation_v1 test_v2u4real_interface_v1.DatasetAuditTests test_v2u4real_interface_v1.GeometryTests -v
# 29 tests OK
python -m compileall -q code/src code/scripts code/tests
# exit 0
python code/scripts/build_project_knowledge_index_v1.py --check
# exit 1; generated registry mismatch, no write-mode builder run
```

人工测试涵盖已知旋转/平移与方向、非平面正逆/距离、非有限点、畸形/非有限/反射/缩放矩阵、坏位姿、重复 YAML 键、源码篡改拒绝、官方框路径、原始数据目录写入保护及机器证据的固定 LF 换行。真实测试单独验证固定输入、全点正逆、独立齐次变换、官方框与字段保留、缺失固定样本时不自动换样本。合计 **46/46 测试通过，失败 0、跳过 0**；人工测试不冒充真实数据证据。暂存审查发现 Windows 自动换行可能改变结果文件 SHA，因此输出固定为 LF 后重新运行全部相关验证；未放宽数值容差，未改原始数据。

没有执行需要初始化 AirFogSim 的测试类或全仓库运行测试，因为本任务禁止启动回放和无线环境；既存相关的纯解析/几何测试已通过。编译检查只检查语法，不等于完整运行测试。

## Results / Expected vs Actual

| 分层状态 | 实测 |
|---|---|
| input_integrity_status | PASS |
| official_source_status | PASS |
| software_transform_status | PASS |
| annotation_diagnostic_status | official_code_interpretation_only |
| physical_alignment_status | NOT_ESTABLISHED |
| task_execution_status | COMPLETED（仅有限诊断） |

**诊断符合预期；工程收口部分符合**（本地独立提交，推送仍被知识索引门阻止）。没有把数学可逆性或 BEV 外观提升成真实同步/轴向/标定的通过证据。

## Known Issues / Conflict record

沿用 R2-03 已核实的代码路径和已记录的论文差异：

- **Documented Intent**：R2-03 引述论文的标注 Euler `(roll,pitch,yaw)` 及预定义局部轴变换、RTK/GICP/人工调整流程。
- **Actual Implementation**：固定源码按 `[roll,yaw,pitch]` 消费原 angle，位置使用 `location+center`；本次 list/list 相对矩阵未额外读取轴变换或注册修正。list/matrix 分支仍直接乘入矩阵，本次框路径只传恒等矩阵，不将其 docstring 推广到一般矩阵。
- **Evidence**：R2-03 记录；固定 3 份源码 SHA 与函数行号；原始两份 YAML；本轮 result/图/数值回执。
- **Affected Files**：固定 transformation_utils/uav_to_single/box_utils 和本样本 PCD/YAML；不改其中任何原始文件。
- **Conflict**：角度排列与 location/center 精确语义、轴变换是否已应用、跨端修正来源、真实时钟/最近帧配对仍缺独立证据。
- **Status: Awaiting Researcher Decision / authoritative contract evidence**。不是本轮新增且已解决的问题；暂停受影响的物理结论。

## Context Consistency Check

核对当前状态、研究背景、架构、数据流、模块、实验、决策、已知问题及变更。更新 AI_CONTEXT/00、03、04、05、07、08；01/02/06 保持原研究背景/核心架构/科学决策。03 仅补充隔离诊断支路，不宣称真实数据进入仿真/服务/训练。没有新增科研决策。人工实验注册表使用完整字段及 null 原因；问题路由直接定位代码和 result；本轮不是正式性能实验，不向正式结果注册表加入性能数字。

记录 context 检查与相关路径/哈希到 `context_consistency_check.json`。三个根过程文件的既有删除、无关生成索引、AGENTS 及未跟踪脚本保持原状。不能声称整个知识索引已更新；既存门仍阻止推送。

## Git

选择性提交 R2-04A 代码、测试、本文、审阅过的诊断证据和导航增量；dirty 文档只暂存本轮新增前缀，保留其原未提交内容。初始快照和前缀暂存辅助文件只留本地，不提交。实际 commit hash 与分支由完成汇报和 Git 历史给出，commit message：`feat(v2u4real): verify R2-04A single-frame geometry`。

**Push：未执行。** 必需索引只读检查仍失败；按研究者指令不重建无关索引、不绕过验证门。提交不是物理验收通过，也不是下一任务授权。

## Next Step

只建议研究者审阅本次诊断，优先确认发布 PCD/位姿的轴向及预定义轴变换是否已应用所需的权威证据。到此停止，不自动进入 R2-04B、标定验收、连续回放、协同感知服务、GPU、locked_test 或训练。
