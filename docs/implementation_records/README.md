# PI-JWM 实施记录

当前工作以研究者 2026-09-18 授权的只读 `00–06` 定义为目标，以仓库 source/config/test/artifact 为实现事实。总体状态见 [Implementation Tracker](../PIJWM_IMPLEMENTATION_TRACKER.md)。

## 记录目录

| Step | 记录 | 范围 | 状态 |
| --- | --- | --- | --- |
| STEP 1 | [STEP_01_AUDIT.md](STEP_01_AUDIT.md) | 新定义与现有实现审计、治理与导航同步 | 见记录中的验证和 Git 状态 |
| STEP 1 数据/双图附件 | [STEP_01_DATA_GRAPH_AUDIT.md](STEP_01_DATA_GRAPH_AUDIT.md) | 01–03 定义、时间、张量、实体和动作映射 | 支撑证据，不是下一 Step |
| STEP 2 | [STEP_02_RAW_TRAJECTORY_ACTION_CONTRACT.md](STEP_02_RAW_TRAJECTORY_ACTION_CONTRACT.md) | 单决策步 Raw Trajectory 与四类动作合同 | COMPLETE |
| STEP 2.1 | [STEP_02_1_REAL_AIRFOGSIM_SINGLE_STEP.md](STEP_02_1_REAL_AIRFOGSIM_SINGLE_STEP.md) | 真实 AirFogSim 单步接线与对齐 | COMPLETE |
| STEP 2.2 | [STEP_02_2_REAL_AIRFOGSIM_MULTI_STEP.md](STEP_02_2_REAL_AIRFOGSIM_MULTI_STEP.md) | 真实 AirFogSim 6 步 Raw Trajectory | COMPLETE |
| STEP 2.3 | [STEP_02_3_RAW_CONTRACT_CAUSAL_COMPLETENESS.md](STEP_02_3_RAW_CONTRACT_CAUSAL_COMPLETENESS.md) | Raw 因果可观测、真实字段、return route 与 acceleration 最终冻结 | COMPLETE；Raw Trajectory Layer / 01 FROZEN |
| STEP 2.4 | [STEP_02_4_COMMUNICATION_OUTCOME_SEMANTICS.md](STEP_02_4_COMMUNICATION_OUTCOME_SEMANTICS.md) | wireless/wired Communication Outcome 拆分、total 与 empty/missing 语义 | COMPLETE；Raw Trajectory Layer / 01 FROZEN |
| STEP 3.1 | [STEP_03_1_MODEL_READY_SAMPLE_TENSOR_CONTRACT.md](STEP_03_1_MODEL_READY_SAMPLE_TENSOR_CONTRACT.md) | Model-ready sample、时间窗口、index/mask、四类 action 与最小真实 tensor contract | COMPLETE；经 STEP 3.1R 修正；正式大数据集未开始 |
| STEP 3.1R | [STEP_03_1R_MODEL_READY_SAMPLE_CONTRACT_CORRECTION.md](STEP_03_1R_MODEL_READY_SAMPLE_CONTRACT_CORRECTION.md) | 修正 History、固定 index/presence、真实 DAG、typed target index、relation 和 Action reference | COMPLETE；STEP 3.2 未开始 |

不预建貌似已经执行的后续 Step 文件。`00–06` 是研究定义章节，不是可以自动执行的七个工程 Step。

## 每步记录要求

每个 Step/Substep 记录必须包含：Step Goal、Definition Basis、Initial State、Files Involved、Changes、Reuse、Validation、Results、Expected vs Actual、Known Issues、Git、Next Step。

- Definition Basis 写清文件、节号和目标；源文件哈希保存在对应 audit 中。笔记只读，工程映射留在本仓库，不整库复制笔记。
- 实现判断只用 `DIRECT_REUSE / MINOR_MODIFICATION / STRUCTURAL_CHANGE / MISSING / RESEARCHER_DECISION_REQUIRED / HISTORICAL_ONLY`。执行进度（如 NOT_STARTED）另列，不能与复用判断混用。
- 验证写实际命令、退出码、测试数量及日志路径；旧测试通过不等于新方法通过。
- 不移动或改写历史数据、实验、协议和 checkpoint；逻辑归档为 Historical / Archived。
- 每一步完成：验证 → Tracker/记录/AI_CONTEXT → diff → commit → push → 固定格式汇报 → 停止。不得自动进入建议下一步。
- Git hash 无法写入包含自身的同一 commit。正文用唯一 commit message 定位主提交，后续 Git 回执记录真实 hash 与 push 结果；回执提交自身用 Git log 定位。

## 成本与边界

正式训练之前先完成新合同下的 schema/tensor、单测、tiny forward/backward、finite gradient、tiny-data overfit、短 smoke、动作敏感性、rollout sanity 和 learning signal 验证，再向研究者提出训练请求。记录目录不是自动训练队列。`locked_test` 保持封存。
