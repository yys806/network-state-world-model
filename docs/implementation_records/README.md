# PI-JWM 实施记录

当前工作以研究者 2026-09-18 授权的只读 `00–06` 定义为目标，以仓库 source/config/test/artifact 为实现事实。总体状态见 [Implementation Tracker](../PIJWM_IMPLEMENTATION_TRACKER.md)。

## 记录目录

| Step | 记录 | 范围 | 状态 |
| --- | --- | --- | --- |
| STEP 5.1A | [STEP_05_1A_MOTION_CSI_TARGET_CONTRACT.md](STEP_05_1A_MOTION_CSI_TARGET_CONTRACT.md) | Future Motion/CSI target、mask、stable support alignment、normalization 与 raw-rule bridge | COMPLETE；non-locked development evidence，不是正式 Dataset |
| STEP 5.2 | [STEP_05_2_TRAINING_LOOP_CURRICULUM_JOINT_TRAINING.md](STEP_05_2_TRAINING_LOOP_CURRICULUM_JOINT_TRAINING.md) | CPU training loop、posterior warm-up、prior recursive curriculum、KL schedule、validation、checkpoint/resume | COMPLETE；CPU development smoke，不是 formal training/performance |
| STEP 5.3 | [STEP_05_3_CPU_TRAINING_PREFLIGHT_TINY_OVERFIT_GO_NO_GO.md](STEP_05_3_CPU_TRAINING_PREFLIGHT_TINY_OVERFIT_GO_NO_GO.md) | 固定 dev_train tiny subset、Stage 1/2 capacity、H1/H2 recursive learning、learning-signal、resume 与 Go/No-Go | GO；development-only CPU preflight，不是 formal training/performance |
| STEP 5.3E | [STEP_05_3E_CSI_TRAIN_MEAN_BIAS_FORMALIZATION_TINY_OVERFIT_ACCEPTANCE.md](STEP_05_3E_CSI_TRAIN_MEAN_BIAS_FORMALIZATION_TINY_OVERFIT_ACCEPTANCE.md) | raw CSI decoder train-only mean bias formalization、checkpoint identity、固定 tiny-overfit acceptance | FORMALIZATION_PASS / TINY_OVERFIT_GO；CPU development evidence，不是 formal training/performance |
| STEP 5.4 | [STEP_05_4_GPU_TRAINING_READINESS_FORMAL_TRAINING_PREPARATION.md](STEP_05_4_GPU_TRAINING_READINESS_FORMAL_TRAINING_PREPARATION.md) | manifest-driven formal interface、formal data/action coverage、config/checkpoint/device readiness | BLOCKED；formal Dataset 与研究者参数未决，GPU 未执行 |
| STEP 5.5 | [STEP_05_5_FORMAL_DATASET_V1_BUILD_ACCEPTANCE.md](STEP_05_5_FORMAL_DATASET_V1_BUILD_ACCEPTANCE.md) | 60 条真实 trajectory、H2/L4 五类 package、四动作 coverage、deterministic rebuild 与 CPU trainer smoke | COMPLETE；FORMAL_DATASET_READY，formal training 仍等待 5.6A 配置冻结/GPU smoke |
| STEP 5.6A | [STEP_05_6A_GPU_SMOKE_FORMAL_CONFIG_EVIDENCE.md](STEP_05_6A_GPU_SMOKE_FORMAL_CONFIG_EVIDENCE.md) / [STEP_05_6A_CONFIG_FREEZE.md](STEP_05_6A_CONFIG_FREEZE.md) | Formal Dataset H4 CUDA smoke、全量 prior-only validation、trajectory-aware sampler、Formal Training Config v1 与 availability bookkeeping | GPU smoke/full validation PASS；config FROZEN；bookkeeping correction CPU-only；formal training 未开始 |
| STEP 5.6B | [STEP_05_6B_FORMAL_GPU_TRAINING_LAUNCH.md](STEP_05_6B_FORMAL_GPU_TRAINING_LAUNCH.md) | 冻结配置正式 runner、独立 Go/No-Go、日志/心跳/checkpoint 与 detached GPU launch | source 提交时为预启动；实时状态以远端 run manifest/heartbeat 为准 |
| STEP 1 | [STEP_01_AUDIT.md](STEP_01_AUDIT.md) | 新定义与现有实现审计、治理与导航同步 | 见记录中的验证和 Git 状态 |
| STEP 1 数据/双图附件 | [STEP_01_DATA_GRAPH_AUDIT.md](STEP_01_DATA_GRAPH_AUDIT.md) | 01–03 定义、时间、张量、实体和动作映射 | 支撑证据，不是下一 Step |
| STEP 2 | [STEP_02_RAW_TRAJECTORY_ACTION_CONTRACT.md](STEP_02_RAW_TRAJECTORY_ACTION_CONTRACT.md) | 单决策步 Raw Trajectory 与四类动作合同 | COMPLETE |
| STEP 2.1 | [STEP_02_1_REAL_AIRFOGSIM_SINGLE_STEP.md](STEP_02_1_REAL_AIRFOGSIM_SINGLE_STEP.md) | 真实 AirFogSim 单步接线与对齐 | COMPLETE |
| STEP 2.2 | [STEP_02_2_REAL_AIRFOGSIM_MULTI_STEP.md](STEP_02_2_REAL_AIRFOGSIM_MULTI_STEP.md) | 真实 AirFogSim 6 步 Raw Trajectory | COMPLETE |
| STEP 2.3 | [STEP_02_3_RAW_CONTRACT_CAUSAL_COMPLETENESS.md](STEP_02_3_RAW_CONTRACT_CAUSAL_COMPLETENESS.md) | Raw 因果可观测、真实字段、return route 与 acceleration 最终冻结 | COMPLETE；Raw Trajectory Layer / 01 FROZEN |
| STEP 2.4 | [STEP_02_4_COMMUNICATION_OUTCOME_SEMANTICS.md](STEP_02_4_COMMUNICATION_OUTCOME_SEMANTICS.md) | wireless/wired Communication Outcome 拆分、total 与 empty/missing 语义 | COMPLETE；Raw Trajectory Layer / 01 FROZEN |
| STEP 3.1 | [STEP_03_1_MODEL_READY_SAMPLE_TENSOR_CONTRACT.md](STEP_03_1_MODEL_READY_SAMPLE_TENSOR_CONTRACT.md) | Model-ready sample、时间窗口、index/mask、四类 action 与最小真实 tensor contract | COMPLETE；经 STEP 3.1R 修正；正式大数据集未开始 |
| STEP 3.1R | [STEP_03_1R_MODEL_READY_SAMPLE_CONTRACT_CORRECTION.md](STEP_03_1R_MODEL_READY_SAMPLE_CONTRACT_CORRECTION.md) | 修正 History、固定 index/presence、真实 DAG、typed target index、relation 和 Action reference | COMPLETE；STEP 3.2 未开始 |
| STEP 3.1F | [STEP_03_1F_MODEL_READY_HISTORY_CONTRACT_FINALIZATION.md](STEP_03_1F_MODEL_READY_HISTORY_CONTRACT_FINALIZATION.md) | 收尾 History 中 past Action/Outcome、History union index、历史 relation/DAG/Flow 对齐、Future Action namespace 修正和 future-reference 观察审计 | COMPLETE；含 3.1F-PATCH；STEP 3.2 未开始 |
| STEP 3.2 | [STEP_03_2_RAW_TO_DATASET_BATCH_SPLIT_PREPROCESSING.md](STEP_03_2_RAW_TO_DATASET_BATCH_SPLIT_PREPROCESSING.md) | Raw-to-Dataset 批量、trajectory-level split、causal windows、train-only preprocessing、batch/load 验证 | COMPLETE；non-locked validation bundle；正式 Dataset/训练未开始 |
| STEP 3.3 | [STEP_03_3_MODEL_INPUT_TENSOR_COLLATION_CONTRACT.md](STEP_03_3_MODEL_INPUT_TENSOR_COLLATION_CONTRACT.md) | fixed-shape CPU tensor/collation、Past Outcome、Target、四类 action、stable vocab/mask | COMPLETE / FROZEN；双图/模型未开始 |
| STEP 4.1 | [STEP_04_1_PI_GRAPH_OBJECT_FIELD_RELATION_MAPPING.md](STEP_04_1_PI_GRAPH_OBJECT_FIELD_RELATION_MAPPING.md) | Physical / Information 对象—字段—关系映射、数据缺口和旧实现冲突 | COMPLETE / FROZEN mapping；graph builder 未开始 |
| STEP 4.2A | [STEP_04_2A_EXISTING_SOURCE_GRAPH_INPUT_ADDITIVE_EXTENSION.md](STEP_04_2A_EXISTING_SOURCE_GRAPH_INPUT_ADDITIVE_EXTENSION.md) | 已有来源的 graph minimum inputs 贯穿 Raw amendment、Sample、preprocessing 与 Tensor | COMPLETE；graph builder 未开始 |
| STEP 4.4 | [STEP_04_4_STRUCTURED_RSSM_WORLD_MODEL.md](STEP_04_4_STRUCTURED_RSSM_WORLD_MODEL.md) | Structured RSSM、known stochastic outage、deterministic transition 与 dynamic graph rollout | COMPLETE / FROZEN；PATCH3 92/92；untrained CPU model evidence；5.1B/5.2 consumes it |
| STEP 5.0 | [STEP_05_0_DEFINITION_05_DECISION_FREEZE.md](STEP_05_0_DEFINITION_05_DECISION_FREEZE.md) | Definition 05 十项研究决定冻结、旧 Loss/Training/Evaluation 定向复用审计 | DECISION FROZEN；5.1B/5.2 implementation follows；无 full training/GPU/locked-test |

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
