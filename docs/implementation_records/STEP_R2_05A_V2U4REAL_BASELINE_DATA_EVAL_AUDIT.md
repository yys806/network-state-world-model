# STEP R2-05A — V2U4Real 感知基线适配性、字段与评价合同审计

2026-10-27，Asia/Shanghai。**有限审计完成；完整候选系统存在关键适配缺口，正式准入待研究者决定，训练BLOCKED。** 无训练、模型实例化、推理服务、GPU、val/test/locked_test样本读取或正式标签输出。

## Step Goal / Definition Basis

按本轮授权核对可支持任务、字段与评价规则，形成待审草案，不冻结场景、模型或指标。依据当前研究资料、论文v1、固定官方commit和R2-03/04A/B/C；B只解决来源优先级。

研究资料只读：10.23 PPT第7–9页倾向多车视觉到RSU、UAV备选中继；1021记录另有先单视角的建议。保留候选分歧，不自行裁决。只存指纹、位置和概括，不复制私人正文。

## Initial State

分支main，HEAD d2f3c92c529097725b5f82fe4a4e7605f51065e1（R2-04C未推送）。精确状态见initial_workspace.json。既有AGENTS/上下文/导航/计划/生成索引修改、文献/论文/process删除及未跟踪文件全部保留；不reset/clean/amend/force push。process根文件原先已删除，不恢复，过程记录加到本地计划表。

官方仓库https://github.com/VjiaLi/V2U4Real，commit e099a3a746c6092a614e18b74d1f5b2f3c8351e8。原清单9文件先核验；缺少的源码/配置从同一commit补取到本轮audit/official_snapshot，不改第三方目录。75份来源SHA256和Git blob与固定tree一致，这是身份数量，不是逐行审阅或完整运行数量。仅补源码/配置，没有更多数据或权重下载。

## Evidence Map / 实际审阅位置

source_locations.json给出函数/方法精确行号。原9文件位于code/reference/V2U4Real/，补充文件位于本轮official_snapshot/；实际路径逐项见supplemental_source_manifest.json。

| 证据 | 位置 | 支撑事实 |
|---|---|---|
| E1论文 | Zotero条目4SKU5M6X/附件JHB4X5KT，arXiv2603.25275v1，15页；§3.1/3.2第4–5页、§4.1/4.2第6页、§5.1第7页 | 多模态、车辆/UAV、坐标/标注文字、类别、范围、3D AP表述、ego优先、4:1:1划分 |
| E2研究资料 | 私人10.23.pptx第7–9页；1021组会.md:708–749/762–867；项目来源地图 | 视觉/RSU候选和单视角建议分歧 |
| E3读点/坐标 | pcd_utils.py:14–40；transformation_utils.x_to_world/x1_to_x2 | XYZ+颜色通道/补0；pose/度/ZYX/变换方向 |
| E4框/converter | uav_to_single.py:61–116,190–199；vehicle_to_single.py:61–84；box_utils.py:145–191,419–471及create_bbx/corner_to_center | 原字段到角点/框中心、lwh/hwl差异、单端converter差异、B覆盖 |
| E5读取/划分 | basedataset.py:61–213,224–288,393–424,452–523；datasets/__init__.py:19–53 | 实际路径/ego、模拟延迟、范围不是划分 |
| E6原生GT | early_fusion_dataset.py:38–193；intermediate的unique_indices；base_postprocessor.py:46–212 | ego首次去重、Car-only、hwl、输出ID转int |
| E7预处理/模型 | sp_voxel_preprocessor.py:19–91；point_pillar.py:15–54；voxel_postprocessor.py:29–199,237–403；对应early/late/intermediate YAML | voxel、目标分配、objectness/残差、解码、阈值/过滤 |
| E8评价 | common_utils.py:189–229；eval_utils.py:16–157；inference.py:23–84,136起；box_utils.py:572–617 | BEV IoU、匹配/AP、排序、场景汇总、NMS |
| E9样本 | 五帧20输入及R2-04B/C结果、B合同 | 字段有限性、SHA、来源冲突，不是检测性能 |

PDF SHA256 f3257de22a9e438c3e755237bef901dd0b889b0d58b4a76a1040e3faa8a6350c；PPT SHA256 b24faa313aebdb53f4b62fe8dda282f742afccba6cff9e6720ae620386bab8ce；会议记录SHA256 8095edcd9a2800ab9b5650cb594d5d859d852d27892b76e8f96e8fe2bf548db5。未复制PDF/笔记至仓库，未声称比对最终发表版。

## Results — 适配性与字段

完整数据集含LiDAR和RGB，不能称无相机数据集；本轮仅PCD/YAML，主线模型是LiDAR。车辆/UAV不等于多车/RSU；特征融合不证明视觉压缩、真实无线多轮传输与截止时间质量。结论：**存在关键适配阻塞；限定范围后可用于LiDAR三维感知模块，不能直接开展完整候选系统基线。**

固定revision 36f64423a639db7c5bbc8ac8575b1ee74d7b8e9a，train/2025-07-18-12-53_2，agent1/2、Ouster。全部binary PCD含x/y/z/rgb各F4 count1，XYZ有限；YAML根结构、原pose/对象及SHA见field_schema.json。

| 键 | 车辆/UAV点数 | 车辆/UAV标注数 | 原生Car+范围保留数 | B来源冲突ID（筛类前） |
|---|---:|---:|---:|---|
|1752814590.2|141177 / 70532|6 / 2|3 / 1|1,2|
|1752814590.4|142866 / 70751|5 / 2|3 / 1|1,2|
|1752814590.6|143252 / 70732|5 / 2|3 / 1|1,2|
|1752814591.2|144406 / 70579|6 / 1|3 / 0|2|
|1752814591.6|144002 / 70597|6 / 1|3 / 0|2|

原生车辆Car均为1/3/4；UAV前三帧仅1、后两帧无Car。8次共享来源差异中，5次Cyclist被筛除，3次Car仍有来源冲突。20份SHA与R2-04B相同、前后不变；不外推全数据集分布。

代码pose为[x,y,z,roll,yaw,pitch]、度、ZYX；UAV→vehicle=inv(T_W_from_V) @ T_W_from_U，无额外未知轴变换。extent按完整跨度生成角点；converter为lwh，原生为hwl。论文/代码角顺序、完整物理语义/单位、跨端尺寸差异、物理标定/同步未解决。单端converter忽略center，不能一般等同UAV角点路线。

调用链与三层字段解释见 [合同草案](../contracts/PIJWM_PERCEPTION_BASELINE_DATA_EVAL_CONTRACT_DRAFT_V1.md)。本轮未实际执行Open3D reader、spconv完整预处理或模型forward，不能宣称基线跑通。

## Results — 评价与人工复现

评价取前4角点XY面积IoU，忽略高度；帧内分数降序贪心匹配剩余GT，IoU>=0.25/0.5/0.7为TP，GT只匹配一次，无类别参数。AP为VOC2010精度包络积分。默认global_sort_detections=False，只帧内排序再按数据次序串接；开关改变结果。输出总AP和各场景AP，无类别mAP、场景均值mAP或正式FN/F1输出；未定义写UNSPECIFIED。

PointPillar三种参考配置score>0.2，旋转BEV NMS阈值0.15、严格>抑制、top1000；其他配置不同。另有XY跨度/z/GT_RANGE过滤。train/val和inference范围不同，train=False仍读validate_dir；不能把mode=test当测试划分证据。训练anchor的轴对齐overlap正负阈值0.6/0.45不是评价阈值。

下表仅是**人工输入对未修改官方函数的CPU探针，不是真实检测性能**：

| 探针 | 实测 | 解释 |
|---|---|---|
| 相同XY，z中心相差100代码单位 | IoU=1，TP=1 | 忽略高度，不能称体积3D IoU |
| 重复预测一个GT | TP=[1,0]、FP=[0,1] | GT仅用一次 |
| 两帧分数次序 | 默认AP=0.5，全局排分AP=1 | 排序影响结果 |
| 同状态重复AP | 2/3→0.75 | tp/fp原地累计，不能复用 |
| 有预测而总GT=0 | ZeroDivisionError | 空集规则缺口，不静默补定义 |

## Conflicts / Known Issues

| Documented Intent | Actual Implementation | Evidence / Affected Files | Conflict / Status |
|---|---|---|---|
| 论文称3D AP | XY面积IoU | E1/E8；eval/common/inference | 指标维度和命名冲突；Awaiting Researcher Decision |
| 论文Vehicle/Cyclist | 原生Car-only；converter另映射 | E1/E4/E6 | 不可声称复现多类别；Awaiting Researcher Decision |
| 冻结B（UAV覆盖） | 原生early/intermediate/GT首次优先 | E6/E9 | B保留，未来适配未实现；Awaiting Researcher Decision（不重选B） |
| 论文ego优先、IoU统一/补标 | converter UAV覆盖 | E1/E4和R2-04C | 保留既存冲突，B只是实现行为复现 |
| 视觉/RSU/截止时间候选 | LiDAR/车-UAV/数据集AP | E1/E2/E5/E7/E8 | 完整系统适配不足；Awaiting Researcher Decision |

LateFusionDataset训练随机agent、评价多端，不是现成vehicle-only。静态链还显示单端点云已投影ego，却保留相对矩阵给后处理，有重复变换疑点；未运行整个late链，故仅记待核对代码风险，不当已运行证实的故障。

候选J_per=mean Q_k(D_k)需任务/帧/ROI/类别、GT/预测/匹配、IoU/score、空集、最后有效结果和可靠时间。AP不能直接代替Q_k；无完整任务评价器，不伪造物理时间。

## Baseline Candidates / Admission

| 候选 | 官方依据 | 差距与结论 |
|---|---|---|
| vehicle-only PointPillar | 模型/late配置/vehicle converter组件 | 无独立ego-only配置/入口；未来需输入限制和共用B真值适配，不标现成完整可复现 |
| early PointPillar | early配置、dataset、模型/后处理/inference存在 | 原生B/类别/评价口径冲突；完整依赖未跑。官方路径存在不等于PI-JWM准入 |
| intermediate参考 | 配置与LiDAR特征融合 | 不验证视觉Token/真实通信/多轮调度，不扩大本轮实现 |

公平比较共用B最终真值、类别/ROI/split/评价器及排序阈值；训练GT范围/不可见对象也待获批。本轮不选模型、不写生产适配、不生成正式标签。

状态：scenario_alignment=AUDITED_CRITICAL_ADAPTATION_GAPS；source_audit=PASS_WITH_DOCUMENTED_CONFLICTS；field_schema=PASS_CODE_CONSUMPTION_ONLY；label_contract=B_PRESERVED_NATIVE_PATH_CONFLICT；evaluation_audit=PASS_WITH_DOCUMENTED_CONFLICTS；baseline_readiness=NEEDS_RESEARCHER_DECISION；formal_training_admission=BLOCKED；physical_alignment=NOT_ESTABLISHED；task_execution=COMPLETED_AUDIT_ONLY。

## Files Involved / Changes / Reuse

新增只读脚本、19项专项测试、合同草案、本记录和独立audit。复用严格YAML/XYZ/SHA、R2-04A来源与官方函数加载、R2-04C五帧/B纯函数。只执行来源核验后未修改AST函数体，CPU NumPy/PyTorch/Shapely，不调用检测入口。上下文/权威记录/导航注册表只加本轮增量，不改历史结论。

## Validation

实际命令/stdout/退出码/计数见run.log和validation.json；源码身份source_identity.json，保全回执acceptance.json。人工探针与真实字段分开记录，不报告检测性能。

    python code/scripts/audit_v2u4real_r2_05a_baseline_data_eval_v1.py
    # exit 0；20份原始输入不变，75份固定来源身份通过
    python -m unittest -v test_v2u4real_r2_05a_baseline_data_eval_v1 test_v2u4real_r2_04c_label_merge_policy_v1 test_v2u4real_r2_04b_multiframe_coordinate_consistency_v1 test_v2u4real_r2_04a_single_frame_geometry_v1 test_v2u4real_full_sample_v1 test_v2u4real_preparation_v1 test_v2u4real_interface_v1.DatasetAuditTests test_v2u4real_interface_v1.GeometryTests
    # 96 tests OK；新增19、相关77；失败/错误/跳过均0
    python -m compileall -q code/src code/scripts code/tests
    # exit 0
    python code/scripts/build_project_knowledge_index_v1.py --check
    # exit 1；原有5项生成文件不同步；未执行写入式重建

Context Consistency Check：9份AI_CONTEXT均通过结构校验，研究上下文/架构/数据/模块/实验/决策/问题/变更逐项核对，生产架构未改；实验注册表和问题路由校验通过。40份此前R2-04A/B/C已跟踪审计产物逐字节不变；任务前18份已存在文件的修改逐字节保留。

索引不同步文件为docs/registries/generated/下的tracked_file_inventory.csv、python_dependency_map.json、artifact_catalog.csv、archive_candidate_registry.csv、registry_summary.json。该阻塞不是审计测试失败；仍阻止推送，不绕过验证门。

差异审查：本轮编写文件的git diff --cached --check退出0；包含固定官方快照时退出2，告警全部来自原样保存的上游空白样式，详见upstream_snapshot_whitespace_review.json。未为消除告警改写任何官方字节；这是源码保全记录，不隐瞒为全量空白检查通过。

## Expected vs Actual / Git / Next Step

符合有限审计目标，未达到正式基线准入：发现并保留类别、来源、评价维度/排序和系统适配冲突。原始输入及旧审计不变。知识索引只读--check的既存5项不同步仍在，因此保留独立本地提交，不推送、不重建或绕门。最终commit从Git日志定位本记录，避免自引用哈希；父提交保持d2f3c92。

唯一下一步：研究者先决定是否将V2U4Real用途限定为LiDAR三维感知模块验证。本轮完成后停止，不启动R2-05B/训练/GPU/locked_test。
