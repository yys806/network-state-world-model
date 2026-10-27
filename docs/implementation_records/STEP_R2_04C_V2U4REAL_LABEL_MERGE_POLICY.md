# STEP R2-04C — V2U4Real 官方标签合并规则冻结与真实样本核验

日期：2026-10-27（Asia/Shanghai）。**方案 B 已按研究者明确选择冻结：同 ID 冲突时 UAV 覆盖车辆，单端对象保留。五帧内存模拟通过，原始 YAML SHA 前后不变；本轮不启动 R2-05。**

## Step Goal / Definition Basis

本轮只解决标签来源优先级。研究者明确选择“方案 B：以官方转换器实际行为为准”。因此本轮形成的是 **implementation-faithful policy（实现行为复现策略）**：复现固定版本官方转换器行为，不声称与论文所有文字描述完全一致。

适用版本：仓库 `https://github.com/VjiaLi/V2U4Real`，commit `e099a3a746c6092a614e18b74d1f5b2f3c8351e8`，固定文件 `opencood/utils/uav_to_single.py`，SHA-256 `cab7b48fe7c343cbb356db131cd1b87f64249f9cfcb7b4073fdeac18ed51ec4b`。

固定真实样本：数据版本 `36f64423a639db7c5bbc8ac8575b1ee74d7b8e9a`，`train/2025-07-18-12-53_2`，车辆 agent `1`、UAV agent `2`，均 Ouster；键为 `1752814590.2`、`.4`、`.6`、`1752814591.2`、`.6`。文件键仅作发布文件关联，不作物理时间戳。

## Initial State

初始分支 `main`，HEAD `39656105117b39a9618558e792757510d2c1dabc`，即 R2-04B 独立提交。工作区已有 AGENTS、AI_CONTEXT/00、导航文件修改、文献/论文删除和未跟踪脚本；全部保留。本轮不 reset、clean、amend、force push，不修改 R2-04A/R2-04B 提交。

R2-04B 的既有报告和结果是本轮上下文依据；官方源码身份在本轮再次从来源清单和实际文件核验。知识索引只读检查仍是既存阻塞，本轮不执行写入式重建。

## Official Behavior and Frozen Policy

固定源码第 190–199 行的行为是：创建空 `merged_objs`；读取车辆对象并 `merged_objs.update(ego_objs)`；读取 UAV 对象；遍历 UAV 对象执行 `merged_objs[oid] = obj`。因此：

- 同 ID 同时出现：UAV 覆盖车辆；
- 仅车辆出现：保留车辆；
- 仅 UAV 出现：保留 UAV；
- 两端无对象：输出空集合；
- ID 使用 Python/YAML 原始键相等语义，不转整数、不去前导零、不做空间近邻匹配；
- 不平均中心/尺寸、不重新仲裁来源、不因点云覆盖或中心距离改变优先级。

详细正式合同见 [PIJWM_V2U4REAL_LABEL_MERGE_POLICY_V1.md](../contracts/PIJWM_V2U4REAL_LABEL_MERGE_POLICY_V1.md)。合同明确记录了研究者方案 B、论文 §4.2 的 ego 优先描述与官方 UAV 覆盖实现之间的差异，以及未来修改必须重新获得研究者授权。

## 五帧真实样本核验

脚本 `code/scripts/audit_v2u4real_r2_04c_label_merge_policy_v1.py` 重新读取五帧十份 YAML，使用严格重复键检查和原始 SHA 核对，复用 R2-04A 的 official_functions()，核验整个固定来源清单后加载未改动的官方函数体。实际执行 x1_to_x2(uav_pose, vehicle_pose) 和 load_annotation_objects：车辆不叠加相对变换，UAV 转换到车辆系，再在内存中按官方更新顺序合并。转换后的最终对象及两端转换结果均标记 official_code_interpretation_only，原始输入另行保留。每个最终对象记录最终来源、车辆输入、UAV 输入以及被覆盖的车辆原始对象；没有写回任何数据。

| 文件键 | 车辆 IDs | UAV IDs | 共有 IDs | 车辆独有 | UAV 独有 | 被覆盖车辆数 | 合并后对象数 |
|---|---|---|---|---|---|---:|---:|
| `1752814590.2` | 1,2,3,4,5,6 | 1,2 | 1,2 | 3,4,5,6 | — | 2 | 6 |
| `1752814590.4` | 1,2,3,4,6 | 1,2 | 1,2 | 3,4,6 | — | 2 | 5 |
| `1752814590.6` | 1,2,3,4,6 | 1,2 | 1,2 | 3,4,6 | — | 2 | 5 |
| `1752814591.2` | 1,2,3,4,5,6 | 2 | 2 | 1,3,4,5,6 | — | 1 | 6 |
| `1752814591.6` | 1,2,3,4,5,6 | 2 | 2 | 1,3,4,5,6 | — | 1 | 6 |

五帧累计覆盖车辆对象 **8 个**，最终保留对象 **28 个**；最终来源统计为车辆 **20 个**、UAV **8 个**。所有帧均满足 `I_out = I_vehicle ∪ I_UAV`，所有共有 ID 的最终来源均为 UAV，车辆独有对象均保留。本样本没有 UAV 独有 ID；规则本身仍通过合成测试覆盖 UAV 独有输入。

十份 YAML 的 SHA、大小、根字段类型、对象 ID、逐对象来源和原始输入均保存在 `code/artifacts/audit/v2u4real_r2_04c_20261027/result.json`。模拟前后 SHA 全部一致。

## 验收结果与边界

| 状态 | 结果 |
|---|---|
| `policy_status` | `FROZEN_BY_RESEARCHER_CHOICE` |
| `source_contract_status` | `PASS` |
| `input_integrity_status` | `PASS` |
| `label_merge_status` | `PASS` |
| `physical_alignment_status` | `NOT_ESTABLISHED` |
| `task_execution_status` | `COMPLETED` |

本轮仍未解决：论文/代码角度顺序差异，`location/center/extent/angle` 的完整语义和单位，跨端尺寸不一致原因，论文 IoU/补标流程与转换器差异，以及发布点云/位姿的物理轴向、标定修正和实际时间同步。UAV 优先只表示标签来源优先级，不表示 UAV 几何或标定天然更准确。

本轮不把同 ID 中心距离当作独立物理真值，不生成正式训练标签，不启动 R2-05、检测器、GPU、`locked_test`、训练、仿真或感知服务。

## Validation

```powershell
python code/scripts/audit_v2u4real_r2_04c_label_merge_policy_v1.py
# exit 0; policy/input/merge PASS; physical_alignment NOT_ESTABLISHED
python -m unittest test_v2u4real_r2_04c_label_merge_policy_v1.SyntheticMergeTests -v
# 10 tests OK
python -m unittest test_v2u4real_r2_04c_label_merge_policy_v1.RealFiveFrameTests -v
# 6 tests OK
python -m unittest test_v2u4real_r2_04b_multiframe_coordinate_consistency_v1 test_v2u4real_r2_04a_single_frame_geometry_v1 test_v2u4real_full_sample_v1 test_v2u4real_preparation_v1 test_v2u4real_interface_v1.DatasetAuditTests test_v2u4real_interface_v1.GeometryTests -v
# 61 tests OK
python -m compileall -q code/src code/scripts code/tests
# exit 0
python code/scripts/build_project_knowledge_index_v1.py --check
# exit 1; five existing generated index files mismatch; no write-mode rebuild
```

合计 **77/77 测试通过，失败 0、跳过 0**。专项测试覆盖同 ID UAV 覆盖、车辆独有/UAV 独有保留、空输入、键不规范化、集合并集、来源追溯、固定五帧和输入 SHA 不变。

## Files Involved / Changes / Reuse

新增独立审计脚本、专项测试、正式标签合同及本实施记录；产物在 code/artifacts/audit/v2u4real_r2_04c_20261027/。复用 UniqueKeyLoader、sha256、R2-04A 的 strict_vector、matrix_stats、verify_source 和 official_functions，不新增生产数据接口，不接入训练或推理。必要上下文、权威进度、实验注册表和问题路由只添加本轮增量，所有既有未提交内容保留。

## Expected vs Actual / Known Issues

符合本轮来源优先级冻结目标：五帧输出均为原始 ID 并集，共有 ID 均来自 UAV，单端 ID 均保留。真实五帧没有 UAV 独有对象，该分支由人工测试核验，不能声称已由真实样本验证。字段缺失、非有限输入、异常 ID 类型和来源不匹配会停止受影响的核验，不自动修复。来源核验失败生成 BLOCKED 回执，不执行官方函数。物理对齐仍为 NOT_ESTABLISHED；知识索引五项不一致阻止推送，未扩展修复范围。

## Git / Next Step

本轮只提交合同、脚本、测试、五帧审计结果和必要上下文/注册表增量；不提交原始数据、论文 PDF、官方源码快照或既有无关变更。知识索引检查失败，因此不推送。

唯一建议下一步：研究者审阅冻结合同后，另行授权并冻结 R2-05 感知基线的评价与字段解释；本轮到此停止。
