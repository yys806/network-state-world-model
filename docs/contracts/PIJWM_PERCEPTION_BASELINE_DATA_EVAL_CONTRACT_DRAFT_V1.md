# PI-JWM 感知基线数据与评价合同草案 V1

**DRAFT — AWAITING RESEARCHER REVIEW**

日期：2026-10-27。授权范围：STEP R2-05A 的只读审计、字段核验与人工评价函数探针。本草案不冻结新场景、模型、类别、指标或正式实验协议，不授权训练或后续任务。

## 1. 结论与适用范围

**V2U4Real 与当前完整候选系统存在关键适配阻塞；研究者明确限定范围后，可作为 LiDAR 三维感知模块的验证来源。不能据此直接开展完整的视觉协同、通信调度与截止时间任务基线。**

证据以 [R2-05A实施审计](../implementation_records/STEP_R2_05A_V2U4REAL_BASELINE_DATA_EVAL_AUDIT.md) 和 code/artifacts/audit/v2u4real_r2_05a_20261027/ 为准。源码限固定commit e099a3a746c6092a614e18b74d1f5b2f3c8351e8，数据限revision 36f64423a639db7c5bbc8ac8575b1ee74d7b8e9a。75份来源文件已做身份核验，不表示每份均逐行审查或完整运行。

| 维度 | 证据和适配边界 | 能力分类 |
|---|---|---|
| 模态 | 论文完整集包含LiDAR和RGB相机；本地五帧仅核验Ouster PCD/YAML，所审主线是LiDAR检测 | 三维点云字段可直接核验；视觉输入/特征需另行取证与适配 |
| 主体 | 采集端为车辆/UAV；候选PPT为多车视觉到RSU，UAV为备选中继；会议记录仍有单视角起步建议 | 检测对象中的多辆车不等于多个传感agent；需研究者裁决 |
| 任务 | 数据是原始传感输入及三维标注，非现成语义特征；论文基线是三维检测 | LiDAR检测模块可借鉴；不能替代视觉任务定义 |
| 通信 | early拼接点云，intermediate融合LiDAR特征；异步模式使用公式/索引回取 | 特征传输需适配；不能直接验证真实无线、多轮交付 |
| 时间 | 文件键只作关联，物理时钟和同步偏差未独立测量 | 不支持真实截止时间质量；需可靠时间证据或获授权的仿真 |
| 几何 | R2-04A/B刚体数学一致性及有限内部证据通过，物理对齐未建立 | 可复用官方软件变换；不能声称标定真值 |

私人资料只读，工程记录只保存指纹、位置和概括，不复制正文。当前候选场景的分歧保留，不由本草案裁决。

## 2. 输入字段：声明、代码事实、未确认语义

样本固定为train/2025-07-18-12-53_2，agent1/2，Ouster，键1752814590.2/.4/.6、1752814591.2/.6。20份输入路径、大小、SHA、原始位姿/标注见field_schema.json；不能外推全数据集分布。

| 字段 | 论文声明 / 官方代码消费事实 | 仍未确认或限制 |
|---|---|---|
| PCD | binary，x/y/z/rgb各F4、count1；严格读取为有限N×3 XYZ | 未执行官方Open3D reader，不能等同完整模型输入测试 |
| 第四通道 | pcd_to_np直接取Open3D XYZ；有颜色取colors第0通道，否则补0；输出float32 N×4 | 不是经本轮核验的原始反射强度；不能因变量名intensity就认定物理意义 |
| lidar_pose | 代码[x,y,z,roll,yaw,pitch]，默认度，ZYX使用[yaw,pitch,roll] | 论文文字角顺序不同；代码顺序不构成物理标定证明 |
| 相对矩阵 | x1_to_x2(uav_pose,vehicle_pose)=inv(T_world_from_vehicle) @ T_world_from_uav，作用于列齐次坐标 | 车辆保持原始坐标；无额外未知轴矩阵。轴向、长度单位、公共参考系、标定修正未独立建立 |
| vehicles | YAML mapping；五帧键为字符串，obj_type为字符串；location/center/extent/angle各3个有限数 | 原键无损保留，不转整数/近邻合并，不把ID匹配当独立真值 |
| location/center | UAV converter和原生框投影以location+center构造对象位置，投影角点后再求中心 | 不自行判定具体物理参考点；五帧center全零不足以证明一般语义 |
| extent | create_bbx用每方向±extent/2，代码消费为三轴完整跨度 | 不依其他数据集惯例改成半长；物理轴/单位仍待权威定义 |
| angle | 原三元素按原顺序放入对象pose；角点处理后生成弧度yaw | 保留论文(roll,pitch,yaw)与代码(roll,yaw,pitch)差异，不自行重排 |
| 框格式 | UAV converter为[x,y,z,l,w,h,yaw]；原生VoxelPostprocessor为[x,y,z,h,w,l,yaw]；评价[N,8,3] | 不可直接串接不同顺序；KITTI-style注释不证明任意KITTI接口兼容 |
| 类别 | 论文Car+Truck→Vehicle及Cyclist；converter Car/Truck/Bus→Vehicle，Pedestrian/Cyclist分列，未知默认Vehicle；原生只接受精确Car | 路径不同；不能声称复现论文多类别/mAP |

## 3. 已冻结来源规则与原生链冲突

唯一已冻结部分仍为 [R2-04C方案B](PIJWM_V2U4REAL_LABEL_MERGE_POLICY_V1.md)：先车辆后UAV；共有原始ID采用UAV，两端独有保留；不规范化ID、不平均、不重新仲裁。UAV优先不代表标定或标签更准确。

- Documented Intent：研究者选择converter的implementation-faithful policy（实现行为复现策略），未采用论文§4.2的ego优先。
- Actual Implementation：原生early/intermediate的list.index保留首次对象，ego先堆入；GT后处理也首次优先，带ID输出再转int；原生Car-only与converter类别映射不同。
- Evidence / Affected Files：early_fusion_dataset.py的unique_indices，intermediate同段，base_postprocessor.py:46–212；field_schema.json。
- Conflict：方案B不等于原生训练/评价已遵守B；五帧共享ID潜在来源冲突8次，其中5次Cyclist被原生筛除、3次Car仍有冲突，不把8次都说成最终Car评价冲突。
- Status: Awaiting Researcher Decision：B本身保持冻结；采用哪条基线和如何统一类别/真值需另行授权，本轮不修改官方或生产代码。

单端converter直接使用location/extent与angle[1]，忽略center，与UAV完整角点路线不一般等价。不能为两种方法选不同且更有利的标签路径。

## 4. 官方调用链与模型输入

配置→build_dataset→BaseDataset读取/reform_param→PCD/YAML→EarlyFusionDataset.get_item_single_car→原生框投影和点云变换→堆叠/增强/范围过滤→SpVoxelPreprocessor→PointPillar→VoxelPostprocessor→eval_utils。

PointPillar参考配置：4通道点、voxel[0.4,0.4,8]、每voxel最多32点，训练/非训练voxel上限32000/70000。输入为voxel_features/voxel_coords/voxel_num_points；输出psm/rm是anchor存在分数和7维框残差，不是独立多类别logits。点云随机shuffle不等于全管线确定性；本轮未实例化模型或运行spconv完整链。

目标生成以XY轴对齐外接框overlap分配anchor：正例>0.6，负例对所有GT<0.45，每GT最高overlap anchor若overlap>0也强制正例；回归为归一化坐标、尺寸对数比、yaw差。该规则不同于评价匹配。

## 5. 划分与泄漏防护

论文§5.1报告train/val/test=4:1:1，正文报val、test在补充材料。配置root_dir=v2u4real/train、validate_dir=v2u4real/val；README的validation_dir不是实际代码键。train=False仍读validate_dir；mode=test只选择GT_RANGE，不自动选择测试目录。

本轮未读val/test/locked_test样本或分布。未来须明确实际split/path/清单，依获批划分隔离，不能把模式名当划分证明。五帧train只做字段与工程核验，不能报告泛化性能。

## 6. 官方评价事实（非已选定的PI-JWM指标）

| 项目 | 固定代码规则 |
|---|---|
| IoU（交并比） | 前4角点XY多边形面积交/并，忽略高度，是BEV俯视面积IoU，不是三维体积IoU |
| 匹配 | 帧内分数降序，贪心找剩余GT最大IoU，IoU>=阈值算TP；命中后删GT，重复预测FP；无类别匹配参数 |
| 阈值 | inference累计0.25/0.5/0.7 |
| AP（精度—召回曲线面积） | VOC2010单调精度包络，所有召回跳点积分；不是11点/KITTI40点 |
| 排序 | global_sort_detections默认False：帧内排分后依数据次序串接；开启才全数据排分，结果可能不同 |
| 汇总 | 所有加载样本总AP、各场景AP；无场景均值或类别mAP实现 |
| 其他指标 | 中间Precision=累计TP/(累计TP+累计FP)，Recall=累计TP/总GT；未冻结单一工作点。FN/F1和类别mAP正式输出为UNSPECIFIED，不补成官方标准 |
| 后处理 | PointPillar early/late/intermediate score严格>0.20；旋转BEV NMS阈值0.15，严格>才抑制，pre_max_size1000；其他配置不同 |
| 预测过滤 | XY外接跨度≤6，z_min≥−3、z_max≤1，NMS后全部8角点在GT_RANGE内。大框过滤将z_len作布尔量，没有明确高度上限，不替其修正 |
| 范围 | train/val GT[-100.8,-80,-4,100.8,80,4]；inference/test GT[-15,-80,-4,100,80,4]；PointPillar输入同前者。点严格不等式，框含边界且全部角点 |
| 状态安全 | 默认AP原地累计tp/fp，重复复用会改变结果；有预测且总GT=0会除零。未来须新状态/深拷贝及获批空集规则，不静默修改官方 |

论文表格称3D AP，代码用BEV IoU；论文按Vehicle/Cyclist分阈值，原生只Car。不能命名为已复现论文三维AP或多类别mAP。此项Documented Intent/Actual Implementation冲突证据为论文§4和common_utils/eval_utils/inference；Status: Awaiting Researcher Decision，本轮不换评价器。

## 7. 模型层与任务层

Precision用于固定工作点预测可靠程度；Recall观察漏检；F1为两者调和平均，须先定类别/匹配/分数阈值；AP跨分数排序衡量检测质量，mAP还需类别平均规则。它们不自动等于截止时间目标。

候选J_per=(1/K)∑Q_k(D_k)：K为任务数，D_k为任务k截止时间，Q_k(D_k)为此前最后有效检测质量，超时按候选定义计0。单帧预测尚不产生唯一Q_k：需任务与帧/ROI/类别关联、GT/预测和匹配记录、固定IoU/分数阈值、空集规则、最后有效结果及可靠交付时间。官方无该任务级评价器。不能把数据集AP逐任务赋值，不能用文件键代替物理时间。

## 8. 最小候选（未选定）

| 候选 | 官方依据 | 差距与当前准入 |
|---|---|---|
| vehicle-only PointPillar | 模型、late配置、单端converter组件存在 | 未发现独立vehicle-only配置/推理选项；late训练随机agent、评价多端。需获批输入限制和共用B真值适配；不是现成完整可复现 |
| vehicle+UAV early PointPillar | early配置、dataset、模型、后处理、inference分支存在 | 原生B/类别不一致，完整依赖与端到端未执行；官方路径存在，PI-JWM准入BLOCKED |
| intermediate参考 | intermediate配置和特征融合实现 | LiDAR特征不等于视觉Token/真实通信；B/类别/指标仍需适配，仅参考 |

公平比较必须同一最终B真值、类别、ROI、split、评价器/排序/阈值，主要差别为可用传感输入。训练是否共用真值、如何处理单端不可见对象，也须写入获批协议。不能单端用ego真值、协同用union真值后直接比较。

## 9. 准入与待决事项

baseline_readiness_status=NEEDS_RESEARCHER_DECISION；formal_training_admission=BLOCKED；physical_alignment_status=NOT_ESTABLISHED。测试通过证明审计可运行，不解除科学/合同阻塞。

待研究者决定：是否限定LiDAR模块；场景/主体/模态；目标类别与统一真值适配；BEV复现或另立体积指标及命名；AP排序/ROI/阈值/空集；Q_k及时间证据；模型、数据规模与训练协议。B保持冻结。

唯一优先下一步：**研究者先决定是否将V2U4Real用途明确限定为LiDAR三维感知模块验证。** 本轮停止，不自动实施适配、R2-05B或训练。
