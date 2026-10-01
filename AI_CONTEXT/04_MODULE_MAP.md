## 2026-10-01 Stage A 运行监控

冻结formal runner不变，正在执行validation/primary。运行管理与只读备份工具位于`code/artifacts/protocols/pi_jwm_step6_3d_validation_stage_a_v1_20261001/`；`backup_snapshot.py`交互输入SSH凭据，验科学source和结果identity后做原子本地备份，新增工具不属于冻结算法source集合。入口：`docs/implementation_records/STEP_06_3D_FORMAL_VALIDATION_STAGE_A.md`。没有新增科研方法模块。

## 2026-10-01 STEP 6.3D BLOCKER CLOSURE

新增 `code/scripts/close_step6_3d_pre_validation_blockers_v1.py` 负责B1语义、历史TRAIN完整性、身份桥、旧新runner等价及final Go/No-Go；`run_step6_3d_formal_cpu_matrix_v1.py`新增阶段/summary/provenance检查。旧 `audit_step6_3d_pre_validation_v1.py` CLI绑定旧source与NO_GO历史快照，不对新source直接重跑；本次新审计复用其未改的局部oracle。 依据：`docs/contracts/PIJWM_STEP_06_3D_VALIDATION_STAGES_AND_CEM_UPDATE_SEMANTICS_V1.md`、`docs/implementation_records/STEP_06_3D_PRE_VALIDATION_BLOCKER_CLOSURE.md`。

## 2026-10-01 Validation 前审计工具

`code/scripts/audit_step6_3d_pre_validation_v1.py` 只用合成 TRAIN-only CPU fixture 与现有 TRAIN raw，生成概率/预算/随机顺序/统计 oracle 和 NO_GO 收据；不调用 formal runner、不读取 Validation outcome。证据入口 `code/artifacts/protocols/pi_jwm_step6_3d_pre_validation_audit_v1_20261001/09_pre_validation_audit_receipt.json`；原搜索源码未改。

# 模块导航地图

## 2026-09-30 STEP 6.3D-PREFLIGHT-PATCH 增量

`step6_3d_objective_anchor_patch_v1.py` 静态同层哈希替换零 cohort；对应 builder、replacement sidecar replay 和 eligibility audit 在 `code/scripts/`。`step6_1_trained_candidate_rollout_v1.py::rollout_one_step_batch` 与串行共用转移准备；`step6_3d_fixed_budget_search_v1.py` 可选批量 wavefront，继续使用 6.3C 的预算/cache。真实 TRAIN 检查脚本分别生成 21 一步等价与吞吐、22 H4 支持、23 四步等价；25 接受收据没有方法选择含义。

## 2026-09-29 STEP 6.3D（方法比较进行中）

- `code/src/pi_jwm/step6_3d_anchor_selection_v1.py`：split 内候选数对数四分位、Comp-base 与 sample-id hash 锚点选择。
- `code/src/pi_jwm/step6_3d_structured_proposal_v1.py`：同一五层动态 mask 提议分布；HRS/S-CEM/MH-CEM 仅更新范围不同。
- `code/src/pi_jwm/step6_3d_fixed_budget_search_v1.py`：共享 H4 搜索、现有预算/cache/rollout/Objective 接口。
- `code/src/pi_jwm/step6_3d_method_selection_v1.py`：TRAIN CEM 配置规则、Validation paired outcome 与 anchor-cluster bootstrap。
- `code/scripts/run_step6_3d_formal_cpu_matrix_v1.py`：可恢复的正式矩阵 runner；目前仅首条 TRAIN solve 已完成，不能用其选择方法。

## 2026-09-29 STEP 6.3B/6.3C-PATCH

- `code/src/pi_jwm/step6_3b_candidate_grammar_v1.py`：当前已有无线 Flow 的唯一 Task 绑定、TRAIN 结构条件下的选中任务数准入。
- `code/src/pi_jwm/step6_3c_candidate_domain_v1.py`：同一条件支持的符号计数和 lazy 子集枚举，含有 eligible Task 时的 Comm NOOP。
- `code/scripts/run_step6_3bc_train_self_replay_v1.py`：逐锚点保存 raw historical action、Planner-v1 projected action、被排除的 Comm/Route 行及拒绝原因。
- `code/scripts/build_step6_3bc_patch_acceptance_v1.py`：验证全量审计、残余分类和前后候选域收据。它不运行搜索器。

## 2026-09-29 STEP 6.3C

- `code/src/pi_jwm/step6_3c_candidate_domain_v1.py`: lazy state-conditioned CandidateDomain, structural-mode filtering, canonical Comm multiset counting, and binding through the frozen 6.3B admission path.
- `code/src/pi_jwm/step6_3c_search_protocol_v1.py`: search node/prefix contract and `B_WM` transition/cache accountant; no optimizer.
- `code/scripts/run_step6_3c_candidate_domain_audit_v1.py`: complete TRAIN static feasibility and descriptive Validation audit.
- `code/scripts/run_step6_3c_recursive_mechanism_smoke_v1.py`: bounded interleaved H1-H4 mechanism smoke and 6.1 equivalence receipt.

## 2026-09-29 STEP 6.3B

`step6_3b_candidate_grammar_v1.py` binds current causal/predicted state to Comm/Comp/shared-Mob action rows and validates a supplied `CandidateActionSequence`; `step6_3b_candidate_support_v1.py` labels TRAIN family/joint/temporal/causal/fixed-support axes from the frozen TRAIN catalog. `run_step6_3b_comm_support_boundary_audit_v1.py` creates the catalog and a separate descriptive validation receipt. The 6.0A adapter remains shared; no search optimizer exists in these modules.

## STEP 6.3A audit

- `code/scripts/run_step6_3a_candidate_support_audit_v1.py` scans Formal Raw and aligns temporal signatures using the accepted sample index; Comp reconstruction calls the existing deterministic CPU allocation rule.
- `code/artifacts/protocols/pi_jwm_step6_3a_candidate_support_audit_v1_20260928/` contains Comm/Comp/Mob, joint/temporal support, anchor search-size illustrations, H_sup boundary, security and acceptance receipts.
- `STEP_6_3A=PASS` means support evidence was measured. Independent family factorization is `NOT_SUPPORTED`; no optimizer or candidate method is implemented/selected.

## STEP 6.2B-PATCH 当前入口

- `code/src/pi_jwm/step6_0c_planner_action_domain_v1.py`：每 horizon 非空 Route 统一拒绝 `OUTSIDE_PLANNER_ROUTE_NOOP_ONLY_V1`；Comm/Comp/Mob 准入沿用原逻辑。
- `code/src/pi_jwm/step6_2b_planner_objective_scorer_v1.py`：五项目标和严格字典序，无公式更改；当前 `STEP_6_2B=PASS` 仅为 CPU 合同验收。
- `code/scripts/build_step6_2b_patch_route_noop_receipts_v1.py`：门控、Route 缺席编译、有效自由度、Comm 分母和冻结 checkpoint 的新机器收据。
- `code/tests/test_step6_0c_planner_action_domain_v1.py`、`test_step6_2b_planner_objective_scorer_v1.py`：Route no-op 和三种剩余动作族回归。下文 STEP 6.2B 条目是历史阻塞时的导航。

## STEP 6.2B 历史入口

- `code/src/pi_jwm/step6_2b_planner_objective_scorer_v1.py`：已给定 rollout 的 common `H_eff`、逐 Task/Horizon 五项目标与严格字典序；pending Flow Route 拒绝静默评分。验收状态为 `BLOCKED_ON_OBJECTIVE_SEMANTICS`。
- `code/scripts/run_step6_2b_objective_scorer_cpu_v1.py`、`build_step6_2b_objective_receipts_v1.py`：单非锁定 validation anchor 的冻结 checkpoint CPU 机制证据与收据。
- `code/tests/test_step6_2b_planner_objective_scorer_v1.py`：合同测试；Route admission/4.4 Host 冲突见实施记录。

## STEP 6.2A-CLOSURE

- Route v1 admission: `code/src/pi_jwm/step6_0c_planner_action_domain_v1.py::_validate_single_hop_route`
- Burden/readiness semantics: `code/src/pi_jwm/step6_2a_planner_objective_side_state_v1.py::transmission_burden`
- Closure receipts: `code/scripts/build_step6_2a_closure_receipts_v1.py`

## STEP 6.1 当前入口

- `code/src/pi_jwm/step6_1_trained_candidate_rollout_v1.py`：共同当前 belief、串行/批量 H4 递归推演、指纹和 action-response 摘要。
- `code/src/pi_jwm/step6_0a_candidate_generation_v1.py::compile_candidate_step`：在每步预测状态上包装未改的正式 `build_action`。
- `code/scripts/run_step6_1_trained_candidate_rollout_preflight_v1.py`：冻结 checkpoint、validation/Raw 身份、四族机制与 CPU 诊断的运行入口；机器收据在 `code/artifacts/protocols/pi_jwm_step6_1_trained_candidate_rollout_preflight_v1_20260928/`。
- `code/tests/test_step6_1_trained_candidate_rollout_v1.py`：合成合同测试；真实机制证据需查机器收据。

## STEP 6.0C 新入口

`code/src/pi_jwm/step6_0c_planner_action_domain_v1.py`：静态 CPU 预算证据、UAV 当前控制侧状态、六档域验证、联合支持分类、显式 HOLD fallback。`code/tests/test_step6_0c_planner_action_domain_v1.py`：合成 CPU 合同与原正式 adapter 11 tensor 等价。`code/scripts/build_step6_0c_planner_action_domain_v1.py`：五份可复核机器合同/收据。`step6_0a_candidate_generation_v1.py::compile_candidate` 只加 6.0C 域验证入口，训练 `build_action` 未改。

## 2026-09-26 STEP 6.0B 审计入口

`code/scripts/build_step6_0b_planner_action_feasibility_audit_v1.py` 和 `docs/implementation_records/STEP_06_0B_PLANNER_ACTION_FEASIBILITY_SOURCE_AUDIT.md` 定位 CPU/UAV 来源；机器凭证在 `code/artifacts/protocols/pi_jwm_step6_0b_planner_action_feasibility_audit_v1_20260926/`。AirFogSim 本地 `code/reference/AirFogSim/` 只读、无独立 Git 元数据；源码路径与逐文件哈希在凭证中，不能将 PI-JWM SHA 当成 AirFogSim SHA。

## 2026-09-26 新入口

STEP 6.0A：`code/src/pi_jwm/step6_0a_candidate_generation_v1.py`（CPU 候选结构、三态约束、固定支持、三后端接口、池、暖启动、正式动作编译器）；`code/scripts/step6_0a_candidate_contract_acceptance_v1.py`（合成合同 receipt）；正式 Trainer 仍通过 `step5_2_training_loop_v1.py` 导入 `build_step5_1d_unified_model_chain_v1.py::build_action`。旧 P6 Planner 不是当前接口。

> 当前实施入口先读 `docs/PIJWM_IMPLEMENTATION_TRACKER.md` 和 `docs/implementation_records/STEP_02_3_RAW_CONTRACT_CAUSAL_COMPLETENESS.md`。下表中的模型、训练和 P4 gate 是被审计的旧协议路径；它们不能自动代表新定义实现。

Source of truth：文件存在性、依赖和反向引用可查 `docs/registries/generated/python_dependency_map.json`；生命周期可查 `docs/CODE_INDEX.md`。本表用于决定下一步读哪些源码。

| 问题 | 当前入口 | 核心对象 | 分类 |
| --- | --- | --- | --- |
| 新定义实施状态 | `docs/PIJWM_IMPLEMENTATION_TRACKER.md` | Step/模块矩阵 | audit/current |
| STEP 5.6A CUDA smoke / validation | `code/scripts/run_step5_6a_gpu_smoke_v1.py`、`code/scripts/accept_step5_6a_gpu_readiness_v1.py`、`code/src/pi_jwm/step5_5_full_sharded_loader_v1.py` | H4 few-step GPU、trajectory-aware sampler、1104-window prior-only validation | current/GPU smoke and full validation passed；formal config pending；formal training false |
| STEP 5.5-PATCH full-shard consumption | `code/src/pi_jwm/step5_5_full_sharded_loader_v1.py`、`code/scripts/accept_step5_5_patch_cpu_v1.py` | 4416/1104 index、按需 shard/batch、H4 CPU Trainer/checkpoint | current/CPU acceptance；no formal training/GPU |
| STEP 5.5-PATCH Return/lifecycle audit | `code/src/pi_jwm/step5_5_fixed_support_audit_v1.py`、`code/src/pi_jwm/step5_5_lifecycle_repair_v1.py`、`code/scripts/audit_step5_5_patch_v1.py` | target-only Return birth 检测、Raw 213 次同对象集合修复审计 | current/audit |
| STEP 5.5 Formal Dataset collector/builder | `code/scripts/collect_step5_5_formal_raw_v1.py`、`code/scripts/build_step5_5_formal_dataset_v1.py` | 60×96 Raw、H2/L4 sharded five-package build、coverage/hash/acceptance | current/accepted formal dataset |
| STEP 5.5 CPU interface acceptance | `code/scripts/step5_5_formal_dataset_cpu_acceptance_v1.py` | FormalTrainingInterface → `runtime/` 1+1 subset → Step52Trainer → H4 optimizer/validation/checkpoint | historical mini smoke；no full-shard proof |
| Step 1 详细证据 | `docs/implementation_records/STEP_01_AUDIT.md` | 定义—实现—验证 | audit/current |
| Raw 因果合同 | `code/src/pi_jwm/raw_trajectory_causal_contract_v1.py` | future-task 分区、canonical acceleration、slot outcome 聚合 | current/frozen raw |
| Step 2.3 真实 runner | `code/scripts/run_step2_3_real_airfogsim_raw_contract_finalization_v1.py` | 真实字段、return route 与因果验收 | current/evidence |
| Step 2.4 通信 runner/observer | `code/scripts/run_step2_4_real_airfogsim_communication_outcome_semantics_v1.py`, `code/scripts/run_p2_single_step_collector_preflight_v1.py` | wired/wireless split、total 与 empty/missing 验收 | current/evidence |
| STEP 4.1 PI graph mapping | `code/src/pi_jwm/step4_1_pi_graph_mapping_v1.py`、mapping artifact | 对象/字段/关系、gap、forbidden placement、旧实现冲突 | current/frozen mapping |
| STEP 4.2A graph input extension | `code/src/pi_jwm/step4_2a_graph_input_extension_v1.py`、Step 4.2A artifact | existing-source Raw amendment、Sample/preprocessing/Tensor | current/complete；no graph object |
| STEP 4.2B Flow source audit | `code/src/pi_jwm/step4_2b_stateful_flow_source_audit_v1.py`、Step 4.2B artifact | source evidence / gap verdict only | current/audit complete；`FLOW_CONTRACT_NOT_YET_SUPPORTED`; no graph object |
| STEP 4.2C-C Flow Sample/Tensor | `code/src/pi_jwm/step4_2c_c_flow_sample_tensor_v1.py`、`code/scripts/build_step4_2c_c_flow_sample_tensor_v1.py`、Step 4.2C-C artifact | C-B Raw logical Flow/Carrying → Sample/Tensor, presence-aware masks, train-only normalization, full History/target semantic equality | current/frozen additive extension；target carrying is future ground truth only；no graph object/model |
| STEP 4.3A Typed Dual-Graph Builder | `code/src/pi_jwm/step4_3a_typed_dual_graph_builder_v1.py`、`code/scripts/build_step4_3a_typed_dual_graph_builder_v1.py`、Step 4.3A artifact | frozen current Tensor → 11 typed Physical/Information/cross-domain blocks | current/frozen representation builder；development topology not research-frozen；no encoder/model |
| STEP 4.3B Dual-Graph Encoder | `code/src/pi_jwm/step4_3b_dual_graph_encoder_v1.py`、`code/scripts/build_step4_3b_dual_graph_encoder_v1.py`、Step 4.3B artifact | frozen History + current typed graph → aligned `Z_t^{PI,L_g}` | current/frozen untrained encoder；not `xi_t^Lat`/World Model |
| STEP 4.4 Structured RSSM World Model | `code/src/pi_jwm/step4_4_structured_rssm_world_model_v1.py`、`code/scripts/build_step4_4_structured_rssm_world_model_v1.py`、Step 4.4 artifact | structured prior/posterior、local Action routing、vehicle/CSI dynamics、known stochastic outage、Return tri-state、valid-predecessor DAG、terminal Flow status、dynamic graph recursive rollout、prior-only latent initialization | COMPLETE / FROZEN；untrained CPU model evidence；5.1B/5.2 consumes it；no formal training/Planner |
| STEP 5.0 Definition 05 Decision Contract | `docs/contracts_PIJWM_DEFINITION_05_LOSS_TRAINING_EVALUATION_V1.md`、`docs/implementation_records/STEP_05_0_DEFINITION_05_DECISION_FREEZE.md` | 10 项 researcher decisions、旧实现复用审计、STEP 5.1 target/mask/unit前置条件 | DECISION FROZEN |
| STEP 5.1A-PATCH Future Motion / CSI Target Contract | `code/src/pi_jwm/step5_1a_motion_csi_target_contract_v1.py`、`code/scripts/build_step5_1a_motion_csi_target_contract_v1.py`、`code/artifacts/protocols/pi_jwm_step5_1a_motion_csi_target_contract_v1_20260921/` | local one-step Vehicle Motion、current physical input-slot alignment、future outcome CSI、current model relation-slot/identity/endpoint/type/RB alignment、component masks、frozen normalization、raw bridge、NPZ/receipt | COMPLETE / FROZEN FOR TARGET CONTRACT；non-locked development evidence；当前模型未读取 |
| STEP 5.1B-PATCH posterior/loss/metric integration | `code/src/pi_jwm/step5_1b_posterior_loss_metric_v1.py`、`code/scripts/build_step5_1b_posterior_loss_metric_receipt_v1.py`、`code/artifacts/protocols/pi_jwm_step5_1b_posterior_loss_metric_v1_20260921/` | per-horizon target encoders、mask evidence、family-specific future teachers、真实 STEP 4.4 prior/decoder、per-dim KL/free bits、raw-unit metrics、12-sample receipt | COMPLETE / FROZEN FOR CPU DEVELOPMENT PRIMITIVES + PAIRED INTEGRATION；5.2 CPU loop consumes it；full training/GPU NOT STARTED |
| STEP 5.2 training loop | `code/src/pi_jwm/step5_2_training_loop_v1.py`、`code/scripts/run_step5_2_training_loop_smoke_v1.py`、`code/tests/test_step5_2_training_loop_v1.py`、`code/artifacts/protocols/pi_jwm_step5_2_training_loop_v1_20260922/` | Stage 1 future teacher、current-observation posterior initialization、Stage 2 prior recursive curriculum、KL schedule、joint optimizer audit、global mask-normalized prior-only validation、identity-safe checkpoint/resume、receipt | COMPLETE / FROZEN FOR CPU DEVELOPMENT TRAINING-LOOP INTEGRATION；full training/GPU/formal Dataset/Planner/locked_test NOT STARTED |
| Historical Loss / Metrics / GPU Runner | `formal_world_model_loss_v1.py`、`formal_world_model_metrics_v1.py`、`run_formal_dual_graph_gpu_train_v1.py` | 局部 masked reduction/KL/metric/logging/checkpoint模式可参考 | HISTORICAL_ONLY as complete pipeline；不得直接运行新 Definition 05 |
| 当前模型如何递推 | `code/src/pi_jwm/formal_entity_aligned_rssm_world_model_v1.py` | `FormalEntityAlignedRSSMWorldModel.forward()` | model/current |
| 双图如何传播 | `code/src/pi_jwm/formal_dual_graph_world_model_v1.py` | `FormalDualGraphWorldModel.forward()` | model/current |
| 图与跨图算子 | `code/src/pi_jwm/formal_graph_ops_v1.py` | physical/information/coupling functions | model/support |
| 规则如何更新 | `code/src/pi_jwm/formal_deterministic_rule_layer_v1.py` | `DeterministicRuleLayer.forward()` | model/current |
| 运动输入如何生成 | `code/src/pi_jwm/formal_motion_state_v1.py` | `derive_causal_node_motion()` | data/current |
| 数据 split 如何封存 | `code/src/pi_jwm/formal_airfogsim_dataset_v1.py` | `require_split_access()` | data/current |
| window 如何组装 | `code/src/pi_jwm/formal_airfogsim_window_v1.py` | `FormalAirFogSimWindowDataset` | data/current |
| loss 如何计算 | `code/src/pi_jwm/formal_world_model_loss_v1.py` | `formal_world_model_loss()` | training/current |
| metrics 如何定义 | `code/src/pi_jwm/formal_world_model_metrics_v1.py` | `metric_registry()`, `FormalMetricAccumulator` | evaluation/current |
| P4 gate 如何判断 | `code/src/pi_jwm/formal_p4_gate_v1.py` | `evaluate_p4_checkpoint_gates()` | evaluation/current |
| 正式训练如何启动 | `code/scripts/run_formal_p4_entity_rssm_gpu_v1.py` | `validate_prerequisites()`, `main()` | experiment/current |
| 通用训练循环 | `code/scripts/run_formal_dual_graph_gpu_train_v1.py` | `run_formal_training()` | training/current |
| CPU 机制门 | `code/scripts/run_formal_entity_aligned_rssm_consistency_audit_v1.py` | audit entry | audit/current |
| GPU batch 门 | `code/scripts/run_formal_p4_entity_rssm_gpu_batch_probe_v1.py` | probe entry | audit/current |
| 后续规划原型 | `code/src/pi_jwm/formal_candidate_rollout_planner_v1.py` | `FormalCandidateRolloutPlanner` | prototype/P6 closed |
| 项目知识查询 | `code/scripts/query_project_knowledge_v1.py` | CLI `--query` | support/current |
| 知识索引生成 | `code/scripts/build_project_knowledge_index_v1.py` | `build_outputs()`, `--check` | support/current |

## 目录职责

- `code/src/pi_jwm/`：可复用框架实现。
- `code/scripts/`：运行、构建、训练和审计入口。
- `code/tests/`：单元、合同、回归和机制测试。
- `code/reference/AirFogSim/`：第三方参考仿真器，不是 PI-JWM 主体。
- `code/artifacts/`：数据、实验、checkpoint 和 audit；只有状态、manifest 与证据闭合后才能支持结果声明。
- `docs/registries/`：导航注册表，不建立正式结果。
- Model-ready sample contract: `code/src/pi_jwm/model_ready_sample_contract_v1.py`; STEP 3.1F schema/checks include past action/outcome, History union index, anchor visibility plus unified Future Action index namespace, history relation/DAG/flow alignment and round-trip.
- Future action reference audit: `code/scripts/audit_step3_1f_future_action_references_v1.py`; observation-only scan, does not drop windows or decide future-object representation.
- STEP 3.3F CPU tensor collation: `code/src/pi_jwm/step3_3_model_input_tensor_v1.py`; JSON sample -> semantically complete fixed-shape Observation/Past Outcome/Action/Target arrays with stable index, fixed vocab, mask/padding and validation receipt. Not consumed by a graph/model yet.
- STEP 4.1 mapping builder/validator: `code/scripts/build_step4_1_pi_graph_mapping_v1.py` and `pi_jwm.step4_1_pi_graph_mapping_v1.validate_mapping_checks`; audit-only, no graph construction.
- STEP 4.2A builder/validator: `code/scripts/build_step4_2a_graph_input_extension_v1.py` and `pi_jwm.step4_2a_graph_input_extension_v1`; versioned existing-source input extension, no graph construction/model read path.
- STEP 4.2C-B Ledger/Raw amendment: `code/src/pi_jwm/step4_2c_b_causal_flow_ledger_raw_v1.py` and `code/scripts/build_step4_2c_b_causal_flow_ledger_raw_v1.py`; logical destination provenance, single-Flow multi-hop continuity, E2E conservation and acceptance receipt. No Sample/Tensor or graph construction.
- STEP 4.2C-C Flow Sample/Tensor: `code/src/pi_jwm/step4_2c_c_flow_sample_tensor_v1.py` and `code/scripts/build_step4_2c_c_flow_sample_tensor_v1.py`; independent logical Flow namespace, carrying state, target isolation, presence-aware train-only normalization, full four-way Sample→Tensor semantic equality, overflow/round-trip/tamper receipt. No graph construction or model read path.
- Raw DAG capture amendment: `code/scripts/run_step2_3_real_airfogsim_raw_contract_finalization_v1.py::_capture`; source `airfogsim_full_dual_graph_observer_v1._extract_dag_edges`.

- STEP 5.1C unified lineage builder: `code/scripts/build_step5_1c_unified_development_bundle_v1.py`; emits slot-wise identity audit, no-prefix capacity proof, unified dev_train normalization stats, tensor contract, deterministic/serialization receipt. It is COMPLETE/FROZEN FOR DEVELOPMENT and is consumed by the 5.1D CPU paired graph/encoder/world-model path.
- STEP 5.2 CPU training loop: `code/src/pi_jwm/step5_2_training_loop_v1.py`; consumes the 5.1D unified state/action adapter and 5.1B loss primitives. `run_step5_2_training_loop_smoke_v1.py` writes the 8/4 development CPU evidence. It does not perform tiny-data overfit, full training, GPU, formal Dataset, baseline, Planner or locked_test.
- STEP 5.4 formal interface/readiness: `code/src/pi_jwm/step5_4_formal_training_readiness_v1.py` and `code/scripts/step5_4_gpu_training_readiness_v1.py`; manifest-driven split/provenance/action audit and CPU-only receipts. It does not build formal data or execute CUDA.

## 历史代码定位

不要按文件名猜是否弃用。先查 `docs/registries/historical_method_registry.json`，再查 `docs/registries/generated/archive_candidate_registry.csv` 和依赖图。211 个历史候选当前只做逻辑归档，未物理移动。

Unverified：未在依赖图、当前 runner、config 或实验 manifest 中出现的模块，不得仅凭名称说成当前执行路径。
# STEP 5.1D module integration（2026-09-22）

`code/scripts/build_step5_1d_unified_model_chain_v1.py` 复用 4.2C package API、4.3A builder、4.3B encoder、4.4 Structured RSSM 与 5.1B primitives；新增 focused test 覆盖 package roundtrip、pairing、action mapping、hardcoded-index/optimizer guard 和 receipt tamper。
# STEP 6.2A — objective source audit

- Audit runner: `code/scripts/run_step6_2a_planner_objective_semantics_audit_v1.py`
- Focused test: `code/tests/test_step6_2a_planner_objective_audit_v1.py`
- Objective target contract: `docs/contracts/PIJWM_STEP_06_2_PLANNER_OBJECTIVE_CONTRACT_V1.md`
- Baseline metric interface: `docs/contracts/PIJWM_BASELINE_SYSTEM_METRIC_INTERFACE_V1.md`
- Evidence bundle: `code/artifacts/protocols/pi_jwm_step6_2a_planner_objective_semantics_audit_v1_20260928/`

These files audit and record semantics; there is no candidate scorer or ranking implementation in this Step.
# 2026-09-28 STEP 6.2A-PATCH modules

`code/src/pi_jwm/step6_2a_planner_objective_side_state_v1.py`：Planner-only Task/Route 元数据、当前对齐、时间推进、support 边界和路线分歧检测；无 objective scorer。`step6_2a_throughput_metric_v1.py`：未来 PI-JWM/Baseline 共用真实 Flow/Outcome 吞吐提取器。`code/scripts/replay_step6_2a_patch_deadline_sidecar_v1.py`：单条非锁定 Formal Validation 因果重放；`build_step6_2a_patch_readiness_v1.py`：机器回执与就绪矩阵。4.2C-C 路线数组与 4.4 跨跳规则冲突详见 PATCH 实施记录。
2026-09-30：`code/scripts/run_step6_3d_one_cpu_solve_v1.py` 保留 CPU 路径并接入 CUDA FP32、主机搜索状态存储和 execution identity；`run_step6_3d_formal_cpu_matrix_v1.py` 接收冻结 GPU config 并严格校验 resume。迁移/身份、吞吐、小规模 smoke、接受收据分别见 `verify_step6_3d_3080ti_migration_v1.py`、`run_step6_3d_gpu_throughput_v1.py`、`run_step6_3d_3080ti_formal_gpu_smoke_v1.py`、`build_step6_3d_3080ti_acceptance_v1.py`。
