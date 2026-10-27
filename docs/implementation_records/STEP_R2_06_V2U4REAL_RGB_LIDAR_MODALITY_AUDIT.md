# STEP R2-06 — V2U4Real RGB/LiDAR 数据与模态适用性审计

日期：2026-10-27。**审计完成；实际 RGB 样本为 0，RGB 单模态和 RGB–LiDAR 融合均 BLOCKED。现有 train LiDAR 可用于有限输入/接口验证，正式真实感知基线仍需标签、坐标和评价合同准入。** 文件可读、数学变换通过，不代表物理同步或真实检测效果通过。

## Step Goal / Definition Basis

完成最新版《新场景下训练前任务书.md》任务 1，只补 RGB 模态和关联证据，继承 R2-01～R2-05B。第一阶段仍为 LiDAR 3D；车辆/UAV 都是信息源，UAV 采用预设轨迹，RSU 承担协同感知。本轮不选择模型、不修改方案、不启动任务 2～8。

| 最新依据（私有笔记只读，未复制全文） | SHA-256 | 作用 |
|---|---|---|
| D:/shen/OB/科研/PKU/PIJWM/PIJWM/新场景下训练前任务书.md | 9df6fe58b9fb563f87bded2744cfe2243f67b7f0c57d0dd1e7a24bdbdefafa40 | 任务1、阶段准入和LiDAR第一阶段范围 |
| 同笔记根 组会笔记/1028组会.md | 5399a4b4519b7b5b1b6427a057e1837e622208a614fb9ca33c14163906fa3144 | 当前场景与真实/仿真边界；文件名不证明未来会议已发生 |
| 当前用户R2-06消息、AGENTS、协作规则、项目记忆 | 初始工作区指纹与本轮授权 | train-only、CPU审计、不改数据、不自动继续 |

最新会议文件SHA与R2-05A记录不同，不能用旧快照替代最新版。早期AGENTS/项目记忆“尚无新场景代码/Step”的日期性摘要不覆盖当前用户授权和本地R2成果；未修改AGENTS。

数据固定版本 `36f64423a639db7c5bbc8ac8575b1ee74d7b8e9a`，场景仅 `train/2025-07-18-12-53_2`；官方源码固定 `e099a3a746c6092a614e18b74d1f5b2f3c8351e8`。论文沿用R2-03/04B已核验Zotero原文 arXiv:2603.25275v1，PDF SHA `f3257de22a9e438c3e755237bef901dd0b889b0d58b4a76a1040e3faa8a6350c`；不冒充已比对最终发表版。

## Initial State / 当前门

起始分支main，HEAD `90c3690605fd885aa052b5abf4fa3e372f03e9f7`，比origin/main多7个提交，包含R2-03～R2-05B。本轮以本地代码/证据为准，不以远端缺失否定成果。

起始已有AGENTS、AI_CONTEXT/00、CHANGELOG、PROJECT_INDEX、两份生成索引修改，历史资料/论文/三个过程文件删除，以及未跟踪旧脚本；完整列表与SHA在baseline.json。未清理、恢复或混入提交。task_plan/progress/findings原已删除，按项目记忆规则使用记录/本地计划表.md和本实施记录，不重建。

当前门=R2-06有限审计。数据阻塞=无RGB、物理时间/标定依据不足；工程阻塞=既有知识索引五项不同步。唯一交付=本报告和本轮机器回执。

## Files Involved / Changes / Reuse

新增审计脚本 `code/scripts/audit_v2u4real_r2_06_modality_v1.py`、17项测试 `code/tests/test_v2u4real_r2_06_modality_v1.py`、本报告及 `code/artifacts/audit/v2u4real_r2_06_20261027/`；仅增量更新任务记录、受影响AI_CONTEXT、导航与人工注册表。

没有重新解码PCD，没有重跑五帧坐标、B合并或Mock/AirFogSim。只重算两压缩包及64份已选原始文件SHA；复用R2-02完整包目录和32份相机元数据，另读2份首键原始YAML的相机字段核对锚点。相机数值检查是RGB必要核验，不是重跑完整旧字段审计。

已提交旧证据与Git HEAD逐字节或仅CRLF/LF换行差异核对，分别保留工作区原始SHA和Git内容SHA，不改写原件。R2-05B conda CPU最终回执在本机被忽略、尚未入Git；单独标记LOCAL_EVIDENCE_NOT_IN_GIT，并保存原字节副本和SHA，不伪称远端已有。

## 1. RGB清单、范围与完整性

复用已下载的两个完整官方训练包，没有新数据下载：

| agent | 包大小bytes | 本轮SHA（与官方固定版本一致） | 继承完整目录 |
|---|---:|---|---|
| 车辆1 | 1,070,684,502 | 34e03eb7a34c896aab55b44f32797a1e66a62386a5a7c4e58074abfb6741fa75 | 1,314文件：m1/ouster/ruby及各自YAML，各219份 |
| UAV2 | 172,308,593 | 64ae659bca67144c8713d5a689ce310474328a3c4ffd7a6f35948358b72b61f3 | 438文件：ouster与yaml/ouster，各219份 |

合计1,242,993,095 bytes，完整目录共1,752项，无图像或camera目录成员，无重名。包SHA不变使旧完整目录可复用，不再解压全包。只对既有selected_samples/train和partial_real/train下该场景查找图片，实际也为0。

| 官方读取器声明的相机 | 预期相对路径 | 实际图像 | 格式/分辨率/解码/EXIF |
|---|---|---:|---|
| 车辆左/中/右三路 | 1/camera/{left,middle,right}/<key>.jpg | 各0 | 未实测；jpg仅是代码预期扩展名 |
| UAV向下一路 | 2/camera/<key>.jpg | 0 | 未实测 |

已选16键 `1752814590.2`～`1752814591.7`。官方加载器规则产生64条预期图像引用，均无发布成员。这不是64张损坏图，也不是全数据集缺失率。无原图可检查解码、重复图、曝光或坏帧，分别记BLOCKED_NO_SAMPLE / NOT_ASSESSABLE_NO_SAMPLE，不能报0损坏率或100%解码成功。

2026-10-27经GitHub只读接口重新读取作者issue #7评论5031674615（OWNER，原回复日2026-07-21）：当时仅部分发布，图像未公开。见official_owner_reply.json。与固定训练包清单交叉核对，不声称审计了全部train场景或排除了以后新发布图像。没有联系作者。

PCD名为rgb的F4字段是点记录字段，**不是RGB图像文件**；实际着色/生成来源未知，不能证明相机图像存在或已同步。

## 2. 时间、标定、关联与标注证据表

| 对象 | 通过了什么有限检查 / 直接证据 | 缺口与状态 |
|---|---|---|
| 图像存在 | 完整包身份与目录核验，RGB=0 | 原图缺失，BLOCKED |
| 相机内参K | 32份继承Ouster YAML共64组有限3×3；2份原始相机字段与旧记录一致 | 无原图尺寸、缩放、裁剪、畸变/去畸变说明；不能由主点反推分辨率 |
| 相机外参E | 64组有限4×4，末行[0,0,0,1]；旋转行列式近1，最大正交误差约1.1641e-11 | 只记数值诊断，不新增物理阈值；源/目标、方向、长度单位、LiDAR轴变换是否包含均BLOCKED |
| 相机名称绑定 | YAML车辆camera0/1/2、UAV camera0；加载器按left/middle/right返回路径 | 列表次序不能独立证明槽位到实体相机绑定；不外推M1/Ruby |
| 标定适用范围 | 16键车辆各K/E固定；UAV K固定，**E有16个不同值** | UAV不能复用首帧全局外参；逐帧变化原因和物理含义未知 |
| RGB时间 | 继承元数据无time/stamp原始字段，无原图曝光元数据 | timestamp_raw/unit/origin=null；文件键生成代码、曝光/扫描区间、逐帧偏差缺失 |
| LiDAR/位姿/标签 | 固定agent、Ouster、原始key可定位PCD/YAML，64文件SHA不变 | 文件层追溯通过，不等于物理同时采集 |
| 车辆/UAV时间 | 继承论文`3.1/C.3：名义10Hz、GPS/PTP/PPS、最近LiDAR配对、报告≤20ms | 无本场景设备时钟/配对表/偏差日志；相同key和0.1文本差不能当真实秒或Unix时间 |
| 标签形式 | 同帧vehicles含ID、location、center、extent、angle、obj_type、speed，为3D框参数记录 | 无核验到的图像2D框，未建立RGB投影3D标签 |
| 3D真值合同 | 继承冻结B：UAV覆盖共有ID，独有保留 | 角度次序、字段含义、跨端尺寸、原生路径/B差异未解决 |

固定源码 `code/reference/V2U4Real/opencood/data_utils/datasets/basedataset.py:163–203` 按YAML stem构造路径，526–580返回相机文件名；224–288的主读取函数实际加载YAML和LiDAR，没有解码相机或建立曝光同步。预留路径不代表多模态训练路径已运行。

相机E来源为真实YAML及论文标定流程说明，未绑定可靠的T_camera_from_lidar或反向矩阵；不擅选方向、偏移或单位。逐项K/E、原YAML SHA和跨键变化见result.json.camera_calibration及calibration_summary.json。数值正常不能证明物理正确。

## 3. 最小几何诊断及补样方案

**真实点云向真实图像投影=BLOCKED，执行0次。** 同时缺原图、有向标定和逐模态物理时间，不投影到人工画布后宣称通过，不凭外观选方向。只做矩阵形状/有限性/旋转/跨键变化检查，不能证明像素重合、框覆盖、曝光同步或物理配准精度。

未来若另行授权，最小补样建议：同一官方train场景3个原发布键，每键车辆三路+UAV一路，合计12张原始图像；附Ouster/YAML明确对应、曝光/扫描时间和配对偏差、命名源/目标及方向的标定、畸变/去畸变、标定版本/处理来源。格式/大小未知，字节预算null，不擅定大规模下载量。本轮未获取、未联系他人；依据充分后才做真实投影。

## 4. 七类用途判断

SUPPORTED=必要条件证据充分；PARTIAL=部分条件满足；BLOCKED=缺关键数据/证据；NOT_APPLICABLE=原生数据没有完整任务所需实体/功能。判断限固定版本和本地样本。

| 用途 | 状态 | 依据与使用边界 |
|---|---|---|
| 1. 单车LiDAR 3D感知 | PARTIAL | 16个真实Ouster/YAML候选输入、R2-05A原生路径静态审计；标签角度/尺寸、类别/ROI/评价与真实检测路径未准入 |
| 2. 单UAV LiDAR 3D感知 | PARTIAL | 16个UAV输入、R2-04B官方UAV→ego软件数学通过；原生/ego标签与独立物理依据仍缺 |
| 3. 车辆+UAV协同3D感知 | PARTIAL | 双源文件关联、五帧软件变换、B来源规则有证据；真实同步/物理配准、B适配和RSU/ROI仍缺 |
| 4. RGB单模态 | BLOCKED | 只有元数据/路径，无原图、解码和适用图像监督验证 |
| 5. RGB+LiDAR融合 | BLOCKED | K/E存在，但无原图、可靠方向/时间；仅未来扩展 |
| 6. 双源数据+AirFogSim | PARTIAL | R2-05B实际CPU应用队列接入通过；无线字节为0，物理坐标/轨迹/时钟/RSU/服务未验收 |
| 7. 多车辆、区域任务、通信调度完整实验 | NOT_APPLICABLE（原生完整场景） | 仅1车传感来源+1UAV，无实测RSU/CSI/任务队列；其他实体/服务须仿真补充；标注多目标车不是多传感来源 |

没有一项升级为完整PIJWM系统已支持。第7项不否定未来明确标注的仿真组合，只否定原生真实数据覆盖完整网络场景。

## 5. 既有LiDAR问题最新状态（不重跑旧审计）

| 问题 | 最新证据与本轮状态 |
|---|---|
| 文件来源/读取 | R2-02的64原始文件、R2-05A/B字段/接口通过；本轮只核SHA，PCD解码0次 |
| 真实时间 | R2-03原始时钟/单位/原点缺失，仍BLOCKED；RGB没有补入任何新测量 |
| 软件坐标/方向 | R2-04A/B pose=[x,y,z,roll,yaw,pitch]、度、ZYX，UAV→vehicle=inv(T_W_from_V)@T_W_from_U，software_transform=PASS；此方向结论不适用于相机E |
| 独立物理语义/注册 | 世界原点、单位、参考点、轴变换、RTK/GICP/人工修正发布来源不足，physical_alignment=NOT_ESTABLISHED；无人工修正 |
| 重复PCD | R2-03车辆1752814590.7/.8与1752814590.9/1752814591.0两组字节/XYZ重复；相应YAML不同。原因仍未知；不算4次独立扫描，不删/替换；R2-04B五键未命中不代表问题消失 |
| B标签 | R2-04C研究者已冻结；R2-05A原生early/intermediate真值路径不等同B。保留决定，不重合并或输出正式标签 |
| 类别/评价 | R2-05A原生Car-only，XY/BEV面积IoU非体积3D IoU；排序、空GT等规则待定，不改阈值 |
| 仿真接入 | R2-05B CPU软件链PASS，真实无线BLOCKED_NOT_ATTEMPTED；不重跑Mock/仿真 |

重复组SHA分别为 `054aebfbbd28685a665c2cb3c215293f26baa93b198bffa92aba88f8dd5939df`、`1e07de7bbe3494ba7b237e4136617165f4169eac56c423812ad835639e1eaebd`；逐文件来源继承在result.json.inherited_lidar，本轮SHA未变。

## 6. 正式数据使用边界与冲突

**现在允许**：上述train点云/YAML和已有证据用于读取、映射、感知接口、来源追溯及明示限制的软件几何诊断。重复点云/相机矩阵只作辅助诊断；未经政策裁决不去重、不当独立物理样本。

**仿真补充**：额外车辆传感来源，预设UAV/车辆轨迹与时钟，RSU位置和覆盖，区域/任务/截止时间，CSI/资源块/队列/服务量。不能用重复真实帧伪装同一时空多车采集；未来组合必须标明合成来源、坐标/时间/评价边界。本轮不设参数。

**RGB仅未来扩展，当前数据和证据均阻塞；不纳入第一阶段。**

**真实感知基线条件**：已有真实train输入，可准备有界LiDAR CPU输入路径验证；正式模型/监督标签/性能对照/训练尚不准入。需要先冻结B适配、标签字段、坐标/ROI、类别、评价维度/排序/空集、重复样本政策及物理证据适用范围。RGB缺失不必阻塞纯LiDAR，但不能借此跳过LiDAR未决项。

| Documented Intent | Actual Evidence | Conflict / Affected Scope | Status |
|---|---|---|---|
| 会议调研称数据集含RGB、统一时间、同步双源 | 作者部分发布声明+固定包无图像；同键非逐帧时间实测 | 采集描述不能提升为本地模态/同步验收；影响RGB、时效、协同声明 | Awaiting Researcher Decision / official evidence；不改方案或私有笔记 |
| YAML有extrinsic | 数值正常但方向/绑定未知，UAV逐帧变化 | 不能套常见方向或首帧矩阵；影响融合 | BLOCKED |
| 3D任务与B真值 | 既有BEV/Car-only/首次优先路径差异 | 保留R2-05A冲突，不改算法/指标求表面一致 | Awaiting Researcher Decision |

## 7. Validation / 复现

- 审计入口最终exit 0，输出COMPLETED_AUDIT_ONLY、RGB=0、相机64组、geometry=BLOCKED、formal_baseline=BLOCKED；不把成功退出当数据准入。前置两次停止分别为旧文件CRLF/LF与本地回执未入Git；查证后显式分类，未改变证据原字节。
- `python -m unittest -v test_v2u4real_r2_06_modality_v1`：17/17 OK，失败/跳过0。仅本轮用例，无完整旧审计重跑。
- `python -m compileall -q code/src code/scripts code/tests`：exit 1，四处__pycache__临时输出权限错误；未改权限或清缓存。随后无落盘compile(source_bytes,filename,'exec')对964份源码语法检查，错误0；历史三条无效转义警告保留，不声称原compileall通过。
- 知识索引--check：exit 1，既有5项生成文件不同步：tracked_file_inventory.csv、python_dependency_map.json、artifact_catalog.csv、archive_candidate_registry.csv、registry_summary.json；不覆盖重建，阻塞推送。
- Context Consistency Check、原始数据/旧证据/起始工作区保全、人工注册表和暂存差异检查，见同目录专项回执；九份AI_CONTEXT核对，只增量更新受影响项，未新增AI自定研究决策。

复现环境：PowerShell，PYTHONUTF8=1，PYTHONPATH为仓库code/src;code/scripts;code/tests。命令 `python code/scripts/audit_v2u4real_r2_06_modality_v1.py --output code/artifacts/audit/v2u4real_r2_06_<new_id>`。脚本拒绝覆盖已有result.json，需本地原始包/前序回执，不会自动联网补数据。

主回执result.json SHA-256：`12e06bf61c63386b213b6cd96d3eaace3589aaca4d5775f7efbe864e4de097af`。source_identity.json记录包/64原始文件/旧证据SHA；supplemental_source_identity.json记录本轮源码、官方读取器、任务书/会议和作者回复。逐命令退出码/stdout见validation.json及日志。

## Results / Expected vs Actual / Known Issues

符合有限审计目标；实际图像/投影因缺样本和物理依据未执行，不能记通过；正式基线条件只部分具备。GPU、训练、模型推理、正式检测性能、validation/test/locked_test、数据下载、原始标签/坐标/标定/时间写回均未执行。核心模型、checkpoint、训练协议和既定方向未改。

## Git

仅独立提交本任务 `feat(v2u4real): audit R2-06 RGB LiDAR modality applicability`；实际SHA见Git历史和完成汇报，避免自引用。脏文档仅暂存本轮前缀，不带入起始改动/删除/旧脚本/生成索引。知识索引未通过，因此不推送，不绕过门。

## Next Step / 唯一建议

研究者审阅R2-05A草案并冻结一个**仅LiDAR的最小真实感知基线数据与评价合同**，明确B真值、坐标/ROI、类别/评价规则和物理证据限制。本轮停止，不自动开始后续任务。
