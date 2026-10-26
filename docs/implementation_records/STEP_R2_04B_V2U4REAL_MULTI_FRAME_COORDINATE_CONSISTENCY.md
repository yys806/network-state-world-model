# STEP R2-04B — V2U4Real 发布坐标约定与五帧内部几何审计

日期：2026-10-26（Asia/Shanghai）。**五帧有限审计完成：20 文件身份与读取通过，官方变换数学检查 5/5 通过；发布数据的直接官方用法获得有限内部支持，真实物理对齐仍 NOT_ESTABLISHED。** 不构成完整标定验收或 R2-05 启动授权。知识索引门仍阻止推送。

## Step Goal / Definition Basis

本轮依据研究者 R2-04B 明确授权，检查论文 §3.1、§3.2、§4.1，固定官方源码、五组已下载 PCD/YAML 以及 R2-03/R2-04A 证据。只评估候选使用方式：按论文的发布坐标声明解释原 XYZ，以官方 pose/list 相对矩阵变换 UAV，不额外添加未知轴变换。它不是对原始传感器标定、同步或注册修正的独立证明。

数据版本 `36f64423a639db7c5bbc8ac8575b1ee74d7b8e9a`，`train/2025-07-18-12-53_2`，车辆 `1/ouster`、UAV `2/ouster`。固定键为 `1752814590.2`、`1752814590.4`、`1752814590.6`、`1752814591.2`、`1752814591.6`。这些是发布文件关联键，不是本轮核实的物理时刻；不计算物理采样间隔、速度或角速度。

## Initial State

分支 `main`，初始 HEAD `fe7d2fe1a65d0ac98f60902059fc1a133144f855`，即 R2-04A 本地独立提交。既有 AGENTS/上下文/计划/索引修改、文献和论文文件删除、未跟踪脚本保留。没有 amend/reset/clean/force push，也没有恢复三个已删除的根过程文件；过程记录按项目记忆 C7 留在 `记录/本地计划表.md` 和本文。完整初始状态与脏文档字节快照保存在本轮目录；快照辅助文件不提交。

R2-04A 推送曾受知识索引检查阻止。本轮只运行只读 `--check`，同样报告 5 项生成文件不同步；不执行写入式 builder，不顺手修复无关索引或现有删除。索引检查失败不冒充本轮数学检查失败，但继续阻止推送。

## Files Involved / Changes / Reuse

- 新增 `code/scripts/audit_v2u4real_r2_04b_multiframe_coordinate_consistency_v1.py` 和 `code/tests/test_v2u4real_r2_04b_multiframe_coordinate_consistency_v1.py`。
- 结果目录 `code/artifacts/audit/v2u4real_r2_04b_20261026/`：`result.json`、`config.json`、`summary.md`、`multiframe_bev.png`、`run.log`、`validation.json`、来源/检查/独立复核回执。
- 复用 R2-04A 的官方函数加载器、SHA、严格 PCD/YAML 读取、刚体矩阵检查、float64 点变换、全点正逆/128 点距离检查、统计与确定性下采样。保持 R2-04A 源码、测试、结果和提交不变。
- 每次运行校验 R2-04A 的源身份清单全部文件及原配置；本轮数学配置与 R2-04A 完全相同：刚体误差 1e-10、矩阵差/逆乘积 1e-12、全点正逆与距离最大绝对误差 1e-8，rtol=0。没有物理中心误差阈值。
- 不修改任何第三方代码、PIJWM 核心模块、训练配置或原始数据；没有下载数据、运行 GPU、训练、仿真/无线环境、ICP/GICP 或 locked_test。

## 来源核验与坐标合同：A/B/C/D 分层

### A. 论文声明

通过 Zotero 集合 `05_数据集、测量与仿真`（`SXVPC5G4`）条目 `4SKU5M6X`、原文附件 `JHB4X5KT` 重读原文，并以本地 PDF 核对第 4–6 页及渲染图。版本为 arXiv `2603.25275v1`，15 页、26,393,888 bytes，SHA-256 `f3257de22a9e438c3e755237bef901dd0b889b0d58b4a76a1040e3faa8a6350c`，与 R2-03 原文身份相同。未宣称已比对最终发表版。论文全文/PDF/页面图不另建文献库、不提交到 Git。

- §3.1，第 4 页：论文描述 GPS/IMU 初始定位、RTK 初始化、GICP 精化及人工调整的注册流程；不是本场景公开矩阵或逐帧测量。
- §3.2，第 4 页：LiDAR 局部坐标经预定义变换统一为 X 前、Y 左、Z 上；世界系遵循 NED，即北、东、下。框分别在各 LiDAR 局部系标注，GPS 提供逐帧初始相对位姿。
- §3.2，第 4–5 页：标注先变换到 ego 系，再按与类别有关的框交并比（IoU，框相交体积占并集体积的比例）门限重新统一 ID，并补充部分遗漏标注。因此同 ID 中心接近可能部分依赖该处理过程，不能作为独立标定真值。
- §3.2 的框角度文字顺序为 `(roll,pitch,yaw)`；代码消费顺序是 `[roll,yaw,pitch]`，差异保留。
- §4.1，第 6 页：为单端检测训练/评估，把 UAV 点云与框转到 ego 局部系。**§3.2 的传感器局部轴统一与 §4.1 的 UAV→ego 任务坐标转换是两层描述，不可认定为同一份已知矩阵。**

上述章节没有给出当前场景轴变换数值、精确导出阶段或 RTK/GICP/人工修正的应用回执。论文合同与发布文件的来源链之间仍有缺口。

### B. 固定官方运行时代码

仓库 `https://github.com/VjiaLi/V2U4Real`，固定 commit `e099a3a746c6092a614e18b74d1f5b2f3c8351e8`。实验前后核验本地来源清单的全部 9 文件 SHA-256/Git blob。关键源身份如下：

| 源码 | SHA-256 |
|---|---|
| `pcd_utils.py` | `3a823f538315531b0feca98ea4351e1225088a5173bd36345c96bd92dea611df` |
| `transformation_utils.py` | `2c198a94ffaf9ac3087396ee364b866a817fb2a075ac7c37c618cb03091ebd80` |
| `uav_to_single.py` | `cab7b48fe7c343cbb356db131cd1b87f64249f9cfcb7b4073fdeac18ed51ec4b` |
| `box_utils.py` | `820ac498be2fa586ab2c797344d55dbd3681b6e3b21536ea00412990a9777657` |

`pcd_utils.pcd_to_np:31–40` 直接读取 Open3D 点坐标并返回 float32；此函数没有旋转 XYZ 或另乘轴矩阵。`x_to_world:13–40` 消费 `[x,y,z,roll,yaw,pitch]`，角度默认度，通过 `Rotation.from_euler('ZYX',[yaw,pitch,roll])` 构造旋转，平移原样写入。`x1_to_x2:61–66` 的 list/list 分支是 `inv(T_W_from_V) @ T_W_from_U`。

`uav_to_single:177` 直接调用 `x1_to_x2(uav_pose,ego_pose)`，`:185–186` 对点执行该矩阵投影。**这一检查过的执行路径没有独立固定轴变换因子。** 本轮同 R2-04A，运行固定官方函数原体与 NumPy float64 点应用，没有启动完整 converter main 或 torch 投影，不宣称完整导出器逐位复现。

另通过只读 GitHub 工具取得固定版本完整递归文件树：227 项、`truncated=false`，与本地 9 份源码的 Git blob 一致。路径清单未见明确独立的传感器轴向/标定/注册矩阵文件；额外审读 `pose_utils.py`、`uav_to_kitti.py`、`vehicle_to_kitti.py`、`coalign_fuse.py`、`base_preprocessor.py`。pose_utils 的噪声/平面化工具、CoAlign 的特征坐标 warp 和常规预处理，不能当作本场景传感器轴标定合同。**只完成完整路径清单及相关源码检查，没有逐行审读全仓库，不能声称所有文件内都没有隐含常数或作者私有采集处理工具。** 清单和固定 URL/blob 见 `official_repository_tree.json`、`repository_inspection.json`。

### C. 五帧可直接核验的内容

20 文件均存在，字节 SHA 与 R2-03 逐项一致，严格解析通过；PCD 字段为 `x y z rgb`、binary，数据体结构与点数匹配、XYZ 全有限；YAML 无重复键，pose 为有限数字 list[6]，原始对象位置/center/extent/angle 均通过字段检查，extent 有限且正。完整 SHA/字节数/头/根字段/位姿/原始字段与类型清单见 `result.json.frames[*].inputs`。

五帧车辆与 UAV 各自没有字节重复或解码 XYZ 重复，也未命中 R2-03 已知车辆重复组；这是当前五帧及已知组的核验，不是整个数据集唯一性证明。20 文件实验前后 SHA 全部不变。

### D. 仍缺权威依据的内容

传感器轴变换具体系数、预处理与导出的精确阶段；发布 PCD 与 pose 的物理单位/参考点/公共原点；原始轴与发布轴的样本级独立绑定；RTK/GICP/人工修正值和是否已应用；逐帧物理时钟与最近帧配对；标注 angle/location/center 的完整数据字典。没有额外估计或假设未知矩阵来补齐这些缺口。

## 五帧数学检查结果

源为 UAV 局部、目标为各帧车辆局部。每帧均实际调用原始 list/list 位姿分支；完整相对矩阵、两端世界矩阵、原位姿和统计保存在 result。每帧三份矩阵都是有限 4×4，齐次末行正确，旋转正交性和行列式检查通过；独立 `solve` 对照通过。对全部 UAV 点正逆变换，并对确定性 128 点的全部 8,128 对距离检查。

| 文件键 | 车辆点数 | UAV点数 | 矩阵独立计算最大差 | 全点正逆最大差 | 距离最大变化 |
|---|---:|---:|---:|---:|---:|
| 1752814590.2 | 141177 | 70532 | 3.5527e-14 | 4.2633e-14 | 5.6843e-14 |
| 1752814590.4 | 142866 | 70751 | 1.7764e-14 | 2.8422e-14 | 5.6843e-14 |
| 1752814590.6 | 143252 | 70732 | 1.0658e-14 | 2.8422e-14 | 5.6843e-14 |
| 1752814591.2 | 144406 | 70579 | 1.1768e-14 | 2.8422e-14 | 8.5265e-14 |
| 1752814591.6 | 144002 | 70597 | 4.6185e-14 | 2.8422e-14 | 2.8422e-14 |

全部容差沿用 R2-04A，不按结果改变。原始/变换后/恢复点数相同，XYZ 有限；逐轴范围、质心、标准差和分位数保留，未要求旋转前后每轴范围或分位数相同。

五帧车辆原始 Z 中位数约 -2.140～-2.034，UAV 原始约 -50.910～-50.775，UAV 转车辆后约 -2.485～-2.355，均为代码长度单位。**条件性解释**：在接受论文发布轴约定的前提下，这些数值与高处 UAV 向地面观测、变换后进入车辆附近高度范围相容；五图覆盖形态重复出现，也未见图上突然翻转。它只支持候选直接用法，没有独立地理/地标参照，不能证明实际前/左/上方向、单位或真实配准精度。

## 同 ID 中心、尺寸与标注集合

按官方解释直接取 `location+center`，UAV 中心乘同一相对矩阵，与车辆中心比较。没有再经过 yaw-only 框简化，没有设置物理对齐门限、计算最终 3D IoU/航向误差或合并同 ID 标签。

| 文件键 | 共同 ID | ID 1 中心距离 | ID 2 中心距离 | 两侧集合情况 |
|---|---|---:|---:|---|
| 1752814590.2 | 1、2 | 0.233108899 | 0.780174722 | 车辆6，UAV2 |
| 1752814590.4 | 1、2 | 0.563238541 | 1.113238070 | 车辆5，UAV2 |
| 1752814590.6 | 1、2 | 0.223962044 | 0.599422712 | 车辆5，UAV2 |
| 1752814591.2 | 2 | — | 0.128557591 | 车辆6，UAV1 |
| 1752814591.6 | 2 | — | 0.164980363 | 车辆6，UAV1 |

8 次比较均值 0.475835368、中位数 0.398173720、最小 0.128557591、最大 1.113238070。最大项为 `.4/ID2`，列入排序审阅项；没有依据把它自动判为物理异常或失败。后两帧 UAV 侧不含 ID1，车辆侧仍有 ID1；仅记录标注集合差异，不自动认定漏标。五个邻近发布键与重复对象 ID 不是 8 个独立标定样本。

原始 extent 在各侧对同 ID 的已选帧内保持相同，但跨端数值不同：

| ID | 车辆 extent | UAV extent | UAV−车辆 |
|---|---|---|---|
| 1 | [4.506838,1.578579,2.321422] | [4.983179,2.067932,2.215566] | [+0.476341,+0.489353,-0.105856] |
| 2 | [1.694557,0.821663,1.567604] | [1.727786,0.837228,1.500413] | [+0.033229,+0.015565,-0.067191] |

ID1 的第二个 extent 字段跨端差约 31%，不能概括为两侧框尺寸相同；本轮不将字段重新命名或交换轴。各角度原值保留，论文/代码顺序冲突仍未解决。中心接近、固定尺寸重复以及同 ID 关系均可能继承标注处理流程，不能作为独立真值。

## 几何图与解释边界

`multiframe_bev.png` 为五帧鸟瞰图和说明面板。每端每帧最多 20,000 点、确定性 linspace 下采样；统一完整 XY 范围、等比例轴，蓝色车辆原点云、橙色变换 UAV，+ / x 是对应来源的共同 ID 中心。数值验收仍使用全部点。

图上各帧重复出现部分重叠及不同视角覆盖形态，未见突然整体翻转；不能凭外观确认地标重合或量化注册误差。各图是各自帧的车辆局部系；统一绘图范围不意味着共享静止世界原点。图中注明单位、物理轴向、ID依赖关系和真实对齐未建立，没有配准或手调矩阵。

## Validation（实际命令与输出）

PowerShell，cwd=`D:\shen\PKU\PIJWM`；`PYTHONUTF8=1`；PYTHONPATH 为本仓库 `code/src;code/scripts;code/tests` 的绝对路径。

```powershell
python code/scripts/audit_v2u4real_r2_04b_multiframe_coordinate_consistency_v1.py
# exit 0; five-frame limited audit COMPLETED
python -m unittest test_v2u4real_r2_04b_multiframe_coordinate_consistency_v1.SyntheticTests -v
# 11 tests OK
python -m unittest test_v2u4real_r2_04b_multiframe_coordinate_consistency_v1.RealFiveFrameTests -v
# 4 tests OK (each real check covers the fixed five frames)
python -m unittest test_v2u4real_r2_04a_single_frame_geometry_v1 test_v2u4real_full_sample_v1 test_v2u4real_preparation_v1 test_v2u4real_interface_v1.DatasetAuditTests test_v2u4real_interface_v1.GeometryTests -v
# 46 tests OK
python -m compileall -q code/src code/scripts code/tests
# exit 0
python code/scripts/build_project_knowledge_index_v1.py --check
# exit 1; five generated files mismatch, no write-mode rebuild
```

合计 **61/61 测试通过，失败 0、跳过 0**。人工测试涵盖中心构造与方向、原尺寸差异、集合差异/ID类型不偷换、非法数值拒绝、字节/解码重复检测、缺样本不替换、沿用容差及输出目录保护。真实测试对全部五帧用独立齐次列向量公式和 solve 核对全部点，并单独计算共同中心，确认共同对象数 [2,2,2,1,1]。人工结果与真实证据分组保存在 validation。没有运行需初始化 AirFogSim 的测试类。

实际完整命令和输出在 `run.log`；进一步独立重读的矩阵/点/中心/输入 SHA 核验见 `acceptance.json`。source_identity、来源清单和结果 SHA 固定到本次最终执行代码；本任务文本输出使用 LF，确保结果文件入库后的字节身份一致。

## Results / Expected vs Actual

| 状态 | 结果 |
|---|---|
| source_contract_status | DOCUMENTED_WITH_UNRESOLVED_FIELDS |
| input_integrity_status | PASS |
| software_transform_status | PASS（5/5） |
| multiframe_internal_consistency_status | INTERNAL_EVIDENCE_RECORDED_WITH_LIMITATIONS |
| physical_alignment_status | NOT_ESTABLISHED |
| task_execution_status | COMPLETED（有限审计） |

**符合本轮有限审计预期；工程收口部分符合（推送受阻）。** 文件、官方软件数学、多帧数值与图像关系支持继续把直接发布数据+官方位姿作为候选输入方式研究；不足以证明物理坐标正确、全数据集有效、独立框对齐或真实同步通过。

## Known Issues / 新发现的冲突

沿用的未决项：预定义轴变换系数/应用阶段、单位/参考点/公共原点、跨端修正、物理时钟与标注字段合同。

本轮额外读到同页 §4.2 的合并规则，与固定官方 converter 存在新记录的明确差异：

- **Documented Intent**：论文第 6 页 §4.2 在同对象两端框有偏移时保留 ego/车辆框。
- **Actual Implementation**：`uav_to_single.py:193–199` 先保存车辆对象，再用 UAV 同 ID 覆盖。
- **Evidence**：固定 SHA 的本地源码与 Zotero v1 第 6 页；paper_contract_evidence 最后一项。
- **Affected Files**：后续任何合并标签、协同感知 baseline 或与论文对照的评价实现；本轮不产生合并标签。
- **Conflict**：论文 ego 优先，converter UAV 优先；不能把两者描述为完全相同的数据合同。
- **Status: Awaiting Researcher Decision**。备选口径为“按论文保留车辆框”或“复现 converter 的 UAV 优先并明确差异”，本轮不代研究者选择或修改代码。

## Context Consistency Check / Git

核对九份 AI_CONTEXT 的研究背景、架构、数据流、模块、实验、决策、已知问题和变更；仅更新受影响的 00/03/04/05/07/08，01/02/06 保持原背景、核心架构与研究决策。记录新冲突不等于新增科研决策。同步任务计划、进展、实施导航、实验/结果导航、完整字段人工实验注册表及问题路由。没有为本轮诊断添加正式性能数字。

只暂存本任务新增文件和文档增量；已脏文档只提交本轮前缀，保留旧内容，原始数据/论文 PDF/第三方快照/无关索引与删除不纳入。独立 Conventional Commit：`feat(v2u4real): audit R2-04B multi-frame coordinates`；实际 hash 见完成汇报/Git 历史，分支 `main`。R2-04A 提交保持原样。

**Push 未执行**：只读知识索引检查仍有 5 项不同步，不重建无关生成索引，不绕过验证门。

## Next Step / R2-05 条件

当前具备五帧严格读取和官方变换的 CPU 工程基础，但**不具备直接启动 R2-05 的完整准入条件**：尚需研究者冻结标签角度/字段解释、同 ID 真值保留优先级以及固定基线/评价合同，并另行授权。不能把本轮内部一致性审计当作独立物理标定通过。

唯一建议下一步：研究者审阅并冻结 R2-05 的标签合同，优先裁决上述 ego/UAV 保留规则。本轮到此停止，不执行 R2-05、更多数据、物理注册、仿真接入、服务或训练。
