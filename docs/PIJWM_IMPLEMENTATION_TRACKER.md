# PI-JWM Implementation Tracker

更新时间：2026-09-22。**Raw / 最小 Dataset-Tensor / Stateful Flow / Typed Dual-Graph Builder / Dual-Graph Encoder / STEP 4.4 Structured RSSM World Model Contract 均已完成并冻结；Definition 05 的 10 项 Researcher Decision 已由 STEP 5.0 冻结；STEP 5.1A-PATCH、STEP 5.1C-PATCH、STEP 5.1D 与 STEP 5.2 已分别闭合 Future target、统一 identity lineage、dev_train normalization provenance、paired model-chain CPU integration 和 CPU training-loop integration**。Full training、GPU training、formal Dataset、Planner 与 locked_test 仍未开始。

## 当前依据与执行边界

- 目标研究定义：`D:\shen\OB\科研\PIJWM` 中七个 `00–06` Markdown 文件，只读。文件名、大小、行数和 SHA-256 见 `code/artifacts/audit/pi_jwm_new_definition_step01_20260918/initial_snapshot.json`。
- 实现事实：本仓库源码、配置、测试和原始 artifact。笔记中“当前代码已经……”的描述也必须核对。
- 工程工作区：`D:\shen\PKU\PIJWM`；旧 P4/P6/P0–P10 工作流为 **Historical / Archived**，不再是 active workflow。旧结果保留原验收含义，不变成新定义结果。
- 本轮不生成正式大规模数据集，不改变双图、World Model、loss、planner 或 checkpoint；不训练、不使用 GPU、不访问 `locked_test`。
- 主报告：[STEP_01_AUDIT.md](implementation_records/STEP_01_AUDIT.md)；数据附件：[STEP_01_DATA_GRAPH_AUDIT.md](implementation_records/STEP_01_DATA_GRAPH_AUDIT.md)。下表中的 00–06 对应上述源文件章节；详细定位在报告中。

## 总体实施状态

Status 表示工程进度；Reuse 表示与目标定义的匹配类别。`DIRECT_REUSE` 只对该行明确划定的局部机制成立，不等于整个模块符合新定义。

| 模块 | Definition Source | Current Code | Status | Reuse | Main Gap | Next Action |
| --- | --- | --- | --- | --- | --- | --- |
| 总体研究链路 | 00 四–六 | formal model / planner / r6 历史链 | AUDITED / NOT_IMPLEMENTED | STRUCTURAL_CHANGE | 各接口存在不等于新定义端到端闭环 | 先冻结一步数据合同 |
| AirFogSim trajectory | 01 原始轨迹与时间语义 | Raw contract、Step 2.1–2.4 runner/artifact | COMPLETE / FROZEN | MINOR_MODIFICATION | Raw 层已闭合；尚未映射到新 Dataset/Tensor | Step 3 冻结 Dataset/Tensor Contract |
| Decision / Execution / Outcome | 01；02 | Raw contract、causal helper、Step 2.3/2.4 真实 artifact | COMPLETE / FROZEN | MINOR_MODIFICATION | future schedule 已与 `O_t`/History/input index 隔离；wireless/wired slot outcome 已拆分并实测 | Step 3 保持同一因果边界 |
| Dataset / split / masks | 02 | `step3_2_batch_preprocessing_v1.py`、Step 3.2 bundle | COMPLETE / FROZEN FOR CURRENT MINIMAL DATA CONTRACT | MINOR_MODIFICATION | trajectory split、lineage、time-grid、train-only normalization、mask/presence counterfactual 已验收；不是正式大规模 Dataset | 03 所需新增字段只能走 additive extension |
| Model-ready sample / tensor contract | 02 | `model_ready_sample_contract_v1.py`、Step 3.1F artifact、Step 3.2 bundle | COMPLETE / FROZEN FOR CURRENT MINIMAL DATA CONTRACT | MINOR_MODIFICATION | History past A/Y、entity type、union input index、typed target、batch/split/preprocessing 已验收；不是正式大规模 Dataset | 进入 03 前先冻结对象-字段-关系映射 |
| tensor contract | 02；03 | `step3_3_model_input_tensor_v1.py`、`build_step3_3_model_input_tensor_v1.py` | COMPLETE / FROZEN FOR CURRENT MINIMAL DATA CONTRACT | MINOR_MODIFICATION | Past Outcome、完整 Target facts、固定 vocab、Comp 正式字段与 semantic receipt 已验收；03/04 feature selection 和正式容量未决定 | 单独授权双图字段映射 |
| Physical / Information 双图 | 03 | Step 4.1 mapping；Step 4.2A additive Sample/Tensor；Step 4.2C-B Ledger/Raw；Step 4.2C-C Flow Sample/Tensor；Step 4.3A typed builder；Step 4.3B encoder | BUILDER + ENCODER COMPLETE / FROZEN | STRUCTURAL_CHANGE | History temporal encoding、typed directed propagation、P2A/P2C 与 aligned `Z_t^{PI,L_g}` 已实现；topology/encoder config 仅为 development、未研究冻结；这不是 World Model latent | 仅建议 Definition 04 World Model Representation / Dynamics Contract；不自动执行 |
| entity alignment 局部工具 | 02；03 | `airfogsim_tensor_v2.py`、`formal_graph_ops_v1.py` | AUDITED | DIRECT_REUSE | ID/index/mask原则可复用；新增对象映射需扩展 | 保留身份稳定性检查 |
| Route action | 06 §2.1；04 §3.3 | Step 2 Raw + Step 3.3 past/future route tensors | INPUT TENSOR COMPLETE / FROZEN | MINOR_MODIFICATION | route kind/target/task node/hops 与 mask 已映射；尚未接新 graph/model | 后续按新对象路由，未授权 |
| Comm action | 06 §2.1；04 §2.3 | Step 2 Raw + Step 3.3 per-RB action tensors；Step 4.2A per-RB CSI additive tensor；Step 4.3A Comm relation | INPUT TENSOR + CURRENT COMM GRAPH COMPLETE / FROZEN | MINOR_MODIFICATION | CSI/typed relation 已进入 current Graph Builder；动作尚未接 Graph Encoder/model | 后续按独立授权接 Encoder/model，当前不执行 |
| Comp action | 06 §2.1；04 §2.3 | Step 2 Raw + Step 3.3 node/allocated CPU tensors | INPUT TENSOR COMPLETE / FROZEN | STRUCTURAL_CHANGE | `allocated_cpu_per_s` 已张量化；新 graph/model 执行语义尚未接入 | 后续单独授权模型路由 |
| UAV Mobility action | 06 §2.1/5.1 | Step 2 Raw + Step 3.3 UAV index/azimuth/elevation/speed tensors | INPUT TENSOR COMPLETE / FROZEN | MINOR_MODIFICATION | UAV action 已张量化、vehicle motion 仍为 SUMO external；尚未接新 graph/model | 后续单独授权模型路由 |
| action routing | 04 §3.3 | `step4_4_structured_rssm_world_model_v1.py` | COMPLETE / FROZEN | STRUCTURAL_CHANGE | 四类动作按 stable slot 局部路由并拒绝无效引用；尚未用于训练/Planner | Definition 05 单独冻结 loss/training |
| RSSM prior/posterior 局部机制 | 04 §3.1/4.1 | `step4_4_structured_rssm_world_model_v1.py` | COMPLETE / FROZEN | MINOR_MODIFICATION | Phy/Comm diagonal Gaussian prior/posterior、mean/sample 与 prior-only rollout 已实现；未训练 | Definition 05 单独冻结 loss/training |
| latent layout | 04 §3.2 | 同上 | COMPLETE / FROZEN | STRUCTURAL_CHANGE | 五类独立 h；仅 Vehicle Physical 与 Comm 有 z；slot 对齐且无 global pooling | 保持 v1 合同 |
| learned / deterministic boundary | 04 §2；05 §1 | Step 4.4 model/rule transition | COMPLETE / FROZEN FOR DEFINITION 04 CONTRACT | STRUCTURAL_CHANGE | 仅 vehicle motion/CSI 为 learned head；outage 为 known stochastic event；Flow/Task/CPU/UAV 为规则 | Definition 05 冻结监督与 Loss |
| 通信状态充分性 | 04 §4.2 第一个边界 | STEP 4.4 audit + model / ChannelManagerCP / WiredNetworkManager | COMPLETE / FROZEN | MINOR_MODIFICATION | 研究者已选择独立 known stochastic outage；wired capacity 从真实 config 读入，Carrying-derived membership 与 simulator equality 通过；无 learned residual | 保持 `learned_service_residual=false` |
| 外生事件 | 04 §2.3/4.2 | 固定entity slot/mask；presence head | AUDITED / OPEN | RESEARCHER_DECISION_REQUIRED | 已知未来场景还是随机到达过程未冻结 | 研究者定观测与生成边界 |
| deterministic rule feedback | 04 §3.3–3.4 | Step 4.4 `deterministic_transition` | COMPLETE / FROZEN | STRUCTURAL_CHANGE | 每步 learned dynamics 后执行 known stochastic + deterministic rules，并反馈下一 latent | Definition 05 仅定义训练监督 |
| 动态重构图 | 04 §3.4/4.2 | Step 4.4 `rebuild_graph` | COMPLETE / FROZEN | MISSING→IMPLEMENTED | 每步由 predicted state 更新 Physical/Comm/Flow/Task/DAG/Align/GeoComm；topology config 仍 development-only | 不擅自研究冻结 topology |
| Definition 05 decision contract | 05；研究者 STEP 5.0 明确决定 | `contracts_PIJWM_DEFINITION_05_LOSS_TRAINING_EVALUATION_V1.md` | DECISION FROZEN / 5.2 CPU LOOP CLOSED | STRUCTURAL_CHANGE | 10 项决策已冻结；5.1B primitives、5.1D paired integration 与 5.2 CPU loop evidence 已闭合；formal training 仍未开始 | STEP 5.3 CPU Preflight |
| Future Motion / CSI target contract | 05 §1；STEP 5.1A-PATCH 授权 | `code/src/pi_jwm/step5_1a_motion_csi_target_contract_v1.py`、`code/scripts/build_step5_1a_motion_csi_target_contract_v1.py` | COMPLETE / FROZEN FOR TARGET CONTRACT | MINOR_MODIFICATION | local one-step Motion；current physical/model comm slots；12 个 non-locked development samples；不是正式 Dataset；5.1D 只在 posterior/loss 路径读取 target | 复用 paired integration；不得自动进入 STEP 5.2 |
| STEP 5.1C unified identity/normalization lineage | 05 §1；4.2A/4.2C-B/C | `build_step5_1c_unified_development_bundle_v1.py`、unified bundle artifact | COMPLETE / FROZEN FOR DEVELOPMENT BUNDLE | MINOR_MODIFICATION | 12 paired samples 的 Physical/Communication slot identity、10/74 capacity、no-prefix 和 dev_train-only stats 已机器闭合；不是 formal Dataset | 复用 5.1D receipt；不得自动进入 STEP 5.2 |
| STEP 5.1D unified model-chain paired acceptance | 05 §1；4.3A/4.3B/4.4 | `build_step5_1d_unified_model_chain_v1.py`、`STEP_05_1D_UNIFIED_MODEL_CHAIN_PAIRED_ACCEPTANCE.md`、tracked evidence artifact | COMPLETE / FROZEN FOR CPU DEVELOPMENT INTEGRATION | STRUCTURAL_CHANGE | 47/47 checks、12/12 pairing、exact upstream train lineage、runtime prior isolation、actual prior/posterior/decoder、gradient 和 deterministic rebuild 已通过；模型仍 untrained；Route/Comp non-empty coverage=0 | 已由 STEP 5.2 复用；下一步 STEP 5.3 |
| STEP 5.2 training loop | 05 §2 | `code/src/pi_jwm/step5_2_training_loop_v1.py`、`run_step5_2_training_loop_smoke_v1.py`、`STEP_05_2_TRAINING_LOOP_CURRICULUM_JOINT_TRAINING.md` | COMPLETE / FROZEN FOR CPU DEVELOPMENT TRAINING-LOOP INTEGRATION | STRUCTURAL_CHANGE | Stage 1 future teacher、current-observation posterior initialization、Stage 2 prior recursive `1→2` curriculum、KL warm-up/free bits、joint optimizer groups、global mask-normalized prior-only validation、identity-safe checkpoint/resume；26/26 receipt checks | 仅 8/4 development bundle CPU smoke；full training/GPU/formal performance 未开始 | STEP 5.3 CPU Preflight |
| STEP 5.3 CPU preflight | 05 §2 | `code/scripts/step5_3_cpu_training_preflight_v1.py`、`code/tests/test_step5_3_cpu_training_preflight_v1.py`、`STEP_05_3_CPU_TRAINING_PREFLIGHT_TINY_OVERFIT_GO_NO_GO.md`、`code/artifacts/protocols/pi_jwm_step5_3_cpu_training_preflight_v1_20260922/` | LEARNING_SIGNAL_GO / TINY_OVERFIT_NO_GO | MINOR_MODIFICATION | 修正真实 Phase A pre/post、Phase C H=1→H=2 + KL warm-up、raw Motion/CSI bridge、phase-specific gradients、独立 fresh-run 和 resume；receipt passed=true 但 tiny-overfit 两项 false | 仅 bounded CPU learnability evidence；不可称 tiny-data overfit；full training/GPU/formal Dataset/performance 未开始；Route/Comp non-empty=0/0 | STEP 5.3D CSI scale diagnosis |
| STEP 5.3D CSI scale diagnosis | 05 §2 | `code/scripts/step5_3d_csi_scale_optimization_diagnosis_v1.py`、`code/tests/test_step5_3d_csi_scale_optimization_diagnosis_v1.py`、`code/artifacts/protocols/pi_jwm_step5_3d_csi_scale_optimization_diagnosis_v1_20260922/` | COMPLETE / DIAGNOSTIC-ONLY | MINOR_MODIFICATION | H1/H2 raw bridge exact match；baseline 200-step CSI remains above stronger gate；mean-bias diagnostic reaches normalized CSI MSE <1；formal initialization/decoder bridge not selected | CPU-only observation; no GPU/formal training/locked_test; researcher decision required between raw-head mean-bias and normalized-output bridge | researcher decision |
| STEP 5.3E CSI train-mean bias formalization | 05 §2 | `code/scripts/step5_3e_csi_train_mean_bias_formalization_v1.py`、`docs/implementation_records/STEP_05_3E_CSI_TRAIN_MEAN_BIAS_FORMALIZATION_TINY_OVERFIT_ACCEPTANCE.md`、tracked evidence artifact | FORMALIZATION_PASS / TINY_OVERFIT_GO | MINOR_MODIFICATION | Researcher-selected raw-dB decoder with train-only CSI mean bias formally initialized before optimizer; fixed [0,1] CPU run passes H1/H2 stronger gate; checkpoint contract guards and reproducibility pass | development tiny-data evidence only; full training/GPU/formal Dataset/locked_test/performance remain closed; Route/Comp non-empty=0/0 | researcher review before STEP 5.4 |
| STEP 5.4 GPU training readiness / formal preparation | 05 §2/5 | `code/src/pi_jwm/step5_4_formal_training_readiness_v1.py`、`code/scripts/step5_4_gpu_training_readiness_v1.py`、`docs/implementation_records/STEP_05_4_GPU_TRAINING_READINESS_FORMAL_TRAINING_PREPARATION.md`、readiness artifact | TRAINING_STACK_PASS / FORMAL_DATASET_NOT_READY / GPU_CODEPATH_PREPARED / FORMAL_TRAINING_BLOCKED | MINOR_MODIFICATION | Manifest-driven interface, CPU-only readiness audit, config/checkpoint schema and action coverage | Formal Dataset absent; Route/Comp non-empty=0/0; formal L/topology/training budget require researcher decision; CUDA not executed | researcher decides/formalizes Dataset before next Step |
| training | 05 §2 | STEP 5.2 training loop；历史 `run_formal_dual_graph_gpu_train_v1.py` | IMPLEMENTED FOR CPU DEVELOPMENT / FULL TRAINING NOT STARTED | STRUCTURAL_CHANGE | 旧 staged base-freeze/P4 gate 不复用；当前 loop 只执行少量 CPU optimizer smoke | STEP 5.3 CPU Preflight |
| loss / posterior / KL | 05 §1.2；STEP 5.0 决策 | `code/src/pi_jwm/step5_1b_posterior_loss_metric_v1.py` + 5.1D/5.2 evidence | COMPLETE / FROZEN FOR CPU DEVELOPMENT PRIMITIVES + LOOP | STRUCTURAL_CHANGE | paired integration 与 CPU loop 已通过；无正式性能结论 | STEP 5.3 诊断 |
| checkpoint selection | 05 §2.2；STEP 5.0 决策 | `Step52Trainer.update_validation_state()` | COMPLETE / FROZEN FOR DEVELOPMENT | STRUCTURAL_CHANGE | selector/early stopping 只依据 prior-only horizon-mean `L_Val`；KL 只作 diagnostic | STEP 5.3 CPU Preflight |
| prediction evaluation | 05 §3；STEP 5.0 决策 | `code/src/pi_jwm/step5_1b_posterior_loss_metric_v1.py` + 5.2 validation smoke | CPU DEVELOPMENT EVIDENCE / NOT PERFORMANCE FROZEN | MINOR_MODIFICATION | 逐 horizon raw-unit Motion/CSI metrics 和 prior-only `L_Val` 可执行；未形成性能结果 | STEP 5.3 后再做 learning-signal/overfit 判断 |
| learned candidate generation | 06 §4.3 | planner的callback | AUDITED / NOT_STARTED | MISSING | 尚无新四类动作proposal模型/训练 | 后续单独授权 |
| legality / fallback / warm start | 06 §2.2/3.3/4.3 | CandidateAction.legal、空集raise | AUDITED / NOT_STARTED | MISSING | bool过滤不等于合法性规则；无安全fallback与warm start | 冻结场景约束 |
| candidate rollout | 06 §3.1 | `FormalCandidateRolloutPlanner.plan` | AUDITED / PROTOTYPE_ONLY | MINOR_MODIFICATION | 逐候选调用可复用；共同latent快照/随机评估/新动作未接通 | 保留原型，后续适配 |
| planner objective | 06 §3.2 | prediction_extractor/objective回调 | AUDITED / OPEN | RESEARCHER_DECISION_REQUIRED | 权重、风险形式、未来硬约束、proposal训练目标未定 | 决策前提交备选证据 |
| execute-first-action | 06 §3.3/5.1 | selected_first_action切片 | AUDITED / PROTOTYPE_ONLY | MINOR_MODIFICATION | 只返回第一步，未接真实四类动作执行器 | 后续接入AirFogSim |
| real feedback | 06 §5.2 | 旧R6运行器；formal replan接口 | AUDITED / NOT_STARTED | MISSING | 新planner无真实history/state/latent更新通路 | 后续真实一步反馈验收 |
| replanning / 完整closed loop | 06 §5 | replan(updated_batch) | AUDITED / NOT_STARTED | MISSING | 人工传入batch不等于真实环境反馈闭环 | 后续两决策步端到端验收 |
| 旧结果与checkpoint | 旧P4协议；新00–06 | 两seed验收、v5 tensor、旧实验 | ARCHIVED_IN_PLACE | HISTORICAL_ONLY | 新语义/布局/动作/损失不同，不能外推 | 保留原结果作历史回归 |

## 已完成、未开始与待决策

- 本轮完成范围：Step 1 审计矩阵、只读权限与新工作流、实施记录框架、历史逻辑归档、导航与注册表同步；实际验证见 Step 1 报告。
- 新方案的 Raw→Dataset/Tensor→Typed Graph Builder→Dual-Graph Encoder→Structured RSSM World Model Contract 已按当前最小合同完成并冻结；World Model 仅有未训练 CPU 机制证据，loss、训练、candidate proposal 和在线闭环仍未开始。
- 研究者已明确目标：严格Physical/Information划分，四类动作含CPU与UAV，结构化RSSM，只学习未知动态，混合proposal+世界模型选择。无需再次确认这些方向。
- Definition 05 的 distribution、loss、posterior、KL、overshooting、training stage、fixed-support mask、rule-state supervision、trainable modules 和 validation/evaluation 已决定；具体超参数、正式 Dataset资格和训练预算仍待后续独立 Step 冻结。未知未来到达与离开、proposal训练方式、objective权重/风险/硬约束/fallback仍待后续处理。当前通信 residual 明确关闭。

## 当前 STEP 4.4 结果

STEP 4.4 已将冻结 `Z_t^{PI,L_g}` 接入五类 entity/relation-aligned deterministic state，并只为 Vehicle Physical 与 Communication 建立 stochastic state。PATCH3 明确：冻结 current-side 输入没有 `Task.return_size`，所以无 Return slot 表示 requirement unknown，而不是 no-return；真实 adapter 的 unknown 状态在 computation finished 后输出 unresolved/blocking side-state，Existing Return 仍只按 `(task_index, flow_type_index=Return)` 绑定，Future Target 不改变 object support。DAG release 只计算有效前驱，并要求所有有效前驱完成；terminal Flow completion 使用冻结 `FLOW_STATUS_VOCAB` 同步 remaining/presence/carrying/status，partial/intermediate 不会错误 completed。focused tests 30/30、正式 receipt 92/92、6 文件独立重建 hash/size equality 均通过，STEP 4.4 = COMPLETE / FROZEN。证据始终仅为 untrained CPU development，`training=false`、`gpu=false`、`locked_test=false`、`formal_dataset=false`。

## Checkpoint 与结果复用边界

| 变更 | 对旧checkpoint/tensor的影响 | 允许复用 |
| --- | --- | --- |
| Physical/Agent/Comm分拆、特征顺序/单位更改 | 编码器输入语义/尺寸改变；即使同shape也不兼容 | 通用算子、ID/mask工具；不得直接strict-load新模型 |
| Flow/Task去随机状态、Phy/Comm新latent | 参数键和含义改变 | prior/posterior/KL的实现思想；权重迁移需独立证明 |
| 四类动作与逐RB/UAV通路 | action encoder与训练分布改变 | 仿真器接口、旧事件记录作来源证据 |
| 单步规则反馈、动态图、loss/selector | 即使部分权重可加载，旧精度结论也不再适用 | 历史checkpoint复现和回归，不作新验收 |
| 旧两seed单seed验收 | 原数字与原协议保留；并非新00–06性能证据 | 有边界的历史对照；不自动补第三seed |

## Raw Trajectory Layer / 01 已完成并冻结

Step 2.1 v4 完成真实单步验收；Step 2.2 完成真实多步独立重采集与 no-op 连续性；Step 2.3 用 8 个连续真实步完成因果和字段收尾；Step 2.4 在真实 `RSU_0 ↔ cloudServer_4` 有线链路上补齐 Communication Outcome。AirFogSim 未来 schedule 只保留为 internal metadata，不进入 `O_t`、History 或 input-side Entity Index；Decision CSI、CPU capacity/missing mask、wireless/wired split slot service、total 聚合、`setTaskReturnRoute` 和 return lifecycle 均有真实证据。AirFogSim raw acceleration 与 PI-JWM canonical backward difference 使用不同字段，缺历史时为 `null + mask=false`。最终机器证据位于 `code/artifacts/protocols/pi_jwm_raw_contract_causal_complete_v2_20260919/` 和 `code/artifacts/protocols/pi_jwm_communication_outcome_semantics_v1_20260919/`，并要求纳入 Git。

## STEP 3.1F Model-ready History Contract Finalization

最小真实证据位于 `code/artifacts/protocols/pi_jwm_model_ready_sample_v1_20260919/`。History frame `1,2`中明确保留 `O_1+A_1+Y_1+O_2`，Future Action/Target frame `2,3`；input-side index 是整个 History 对象并集，machine policy 为 `history_causal_observable_object_union`，不包含 target-only `Task_7/Task_8`。Future Action 先做 anchor visibility 检查，再使用同一 History-union static index；四类 action、历史 flow/relation/DAG 对齐、固定 presence/mask、ID↔index 校验和 round-trip 均有机器检查。该结果冻结最小 schema 语义，不等于正式数据集完成。Future-reference 观察扫描 JSON 已纳入 artifact provenance，不作自动丢弃或研究决策。

## 当前 Step 3.2 结果

STEP 3.2 已完成最小 non-locked validation：3 条独立 development trajectory、12 个 H=2/L=2 windows，trajectory-level split，train-only mask-aware normalization，deterministic rebuild 和 serialize/load 均有机器证据。该结果不是正式 Dataset、训练或泛化结论。

STEP 3.2-PATCH 已补齐 Dataset isolation evidence：Raw provenance 含真实 trajectory/seed/source SHA/config lineage/time range/slot duration/sample contract；time-grid 与 step start/end 对齐在 window construction 前检查；train/validation trajectory 无交集；batch future-reference audit 独立保存并统计 12/12/0/0，仍为 observation-only；normalization 单位为 `m/s`、`m/s^2`、`AirFogSim data-unit`。

STEP 3.2-PATCH-RECEIPT 已修正最终机器验收：顶层 `passed` 现在是全部 required checks 与 `locked_test/training/gpu/formal_dataset=false` scope checks 的逻辑 AND；负向 fixture 已证明单项失败会使 `passed=false`；provenance contract version 直接复用冻结的 `model_ready_sample_contract_v1.SCHEMA_VERSION`。STEP 3.2 现正式 COMPLETE / FROZEN。

## 当前 STEP 3.3 结果

STEP 3.3F 已补齐 JSON→Tensor 语义：Past Outcome 使用独立 `H-1` 轴，Target 保留 entity/task/flow/service future facts，Comp 读取 `allocated_cpu_per_s`，entity/lifecycle/route/transport 使用固定 vocab，Static entity type 来自 causal History。semantic validation receipt 由 required checks 实际 AND。artifact 位于 `code/artifacts/protocols/pi_jwm_step3_3_model_input_tensor_v1_20260919/`。因此 STEP 3.3 正式 COMPLETE / FROZEN，02 数据集构建与模型输入已按当前最小数据合同冻结；这不等于正式大规模 Dataset 已生成，也不决定 03/04 最终 feature selection。

## 当前 STEP 4.1 结果

STEP 4.1 当时完成 Physical / Information object-field-relation mapping，并以 `DO_NOT_IMPLEMENT_GRAPH_BUILDER_IN_STEP_4.1` 停止；该历史 readiness 已由后续 4.2A/4.2C 数据闭合和 STEP 4.3A builder 实施覆盖。其 CPU capacity/allocation/service/available resource 分离及禁止旧 mixed `physical_edge_state` 的语义仍有效。

## 当前 STEP 4.2A 结果

Step 4.1 中已有可靠来源的 position、wireless per-RB CSI、wired typed relation、CPU static capability、Task demand/progress/elapsed 与 Src/Host/Exec/Ret 已通过独立版本链贯穿 Raw amendment → Sample → train-only preprocessing → Tensor。PATCH 已把 wireless structural validity 与 CSI observability 解耦：CSI missing 不删除 relation，mask=false 的 numeric placeholder 为 0；wired valid/no-CSI 保持合法。三条 development trajectory 共形成 12 个样本；这是机器合同证据，不是正式 Dataset 或容量结论。return size/priority/deadline 已核实存在 simulator observer source，但冻结 Raw 未透传；stable stateful Flow 等仍为 Raw-insufficient。该 4.2A Step 当时尚无 Physical topology 与 graph object，之后已由 4.3A/4.3B/4.4 完成。机器 artifact 位于 `code/artifacts/protocols/pi_jwm_step4_2a_graph_input_extension_v1_20260920/`。
Step 4.1 中已有可靠来源的 position、wireless per-RB CSI、wired typed relation、CPU static capability、Task demand/progress/elapsed 与 Src/Host/Exec/Ret 已通过独立版本链贯穿 Raw amendment → Sample → train-only preprocessing → Tensor。PATCH 已把 wireless structural validity 与 CSI observability 解耦：CSI missing 不删除 relation，mask=false 的 numeric placeholder 为 0；wired valid/no-CSI 保持合法。三条 development trajectory 共形成 12 个样本；这是机器合同证据，不是正式 Dataset 或容量结论。return size/priority/deadline 已核实存在 simulator observer source，但冻结 Raw 未透传；stable stateful Flow 等仍为 Raw-insufficient。Physical topology 与 graph object 均未实现。机器 artifact 位于 `code/artifacts/protocols/pi_jwm_step4_2a_graph_input_extension_v1_20260920/`。

## 当前 STEP 4.2C-C 结果

STEP 4.2C-C 已完成并冻结 Raw Flow → Model-ready Sample → CPU Tensor additive extension。新增独立 `logical_flow` History-union/target namespace，Flow 与 Carrying state 分离，completed/superseded 与 padding 分离，五个连续字段仅在 `dev_train` 且 `presence=true AND feature_mask=true AND value!=null` 上标准化，capacity overflow 显式拒绝；Sample/Tensor 不重新解释 C-B Raw。随后 `STEP 4.2C-C-PATCH` 补齐 presence-aware stats、History/target Logical/Carrying 四组全字段 semantic equality 和完整 target carrying namespace，并由 receipt 实际 AND 子检查、target namespace、future Epoch、placeholder/bounds/route mask 与 normalization policy。23 项 focused tests、4.2B/4.2A/3.3 回归、真实低 wired capacity 跨时隙 trace、deterministic rebuild/hash、serialize/load 和 receipt/semantic tamper 均通过。artifact 位于 `code/artifacts/protocols/pi_jwm_step4_2c_c_flow_sample_tensor_v1_20260920/`；真实 Return multi-hop、reroute runtime 与 formal capacity 仍未声称。

这是 STEP 4.2C-C 当时的历史范围：`graph_builder=false`、`information_graph=false`、`physical_topology=false`。之后 STEP 4.3A 已完成 Graph Builder；`training=false`、`gpu=false`、`locked_test=false`、`formal_dataset=false` 继续有效。

## 当前 STEP 4.3B 结果

STEP 4.3B 使用完整冻结 History Tensor 编码 Physical/Agent/Task/Flow 的对象级时间状态，并使用 STEP 4.3A current typed graph 完成 Physical/Comm/Flow/Task-Agent/DAG 分族有向传播、relation-wise masked mean、P2A Align 与 wireless-only P2C GeoComm。Logical Flow 与 Carrying 分支独立编码后融合，最终只产生一个 Flow relation latent。输出保持 entity/relation alignment，只到 `Z_t^{PI,L_g}`。artifact 是 `UNTRAINED_DEVELOPMENT_ENCODER_EVIDENCE`；不代表 `xi_t^Lat`、World Model、预测或性能。数值配置与 Physical topology 均为 development-only，`research_frozen=false`。

## 下一步边界

STEP 4.3A、STEP 4.3B 与 STEP 4.4 均正式 COMPLETE / FROZEN；STEP 5.0 已冻结 Definition 05 决策和复用审计。STEP 5.1A-PATCH 已冻结 local one-step Motion、current physical slots 与 current model CSI slots。STEP 5.1B-PATCH、STEP 5.1D-PATCH 与 STEP 5.2 已 COMPLETE/FROZEN FOR CPU DEVELOPMENT PRIMITIVES + PAIRED/TRAINING-LOOP INTEGRATION；full training、GPU、`locked_test`、formal Dataset、baseline 与 Planner 仍未开始。
