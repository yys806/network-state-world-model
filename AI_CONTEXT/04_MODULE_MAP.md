# 模块导航地图

> 当前实施入口先读 `docs/PIJWM_IMPLEMENTATION_TRACKER.md` 和 `docs/implementation_records/STEP_02_3_RAW_CONTRACT_CAUSAL_COMPLETENESS.md`。下表中的模型、训练和 P4 gate 是被审计的旧协议路径；它们不能自动代表新定义实现。

Source of truth：文件存在性、依赖和反向引用可查 `docs/registries/generated/python_dependency_map.json`；生命周期可查 `docs/CODE_INDEX.md`。本表用于决定下一步读哪些源码。

| 问题 | 当前入口 | 核心对象 | 分类 |
| --- | --- | --- | --- |
| 新定义实施状态 | `docs/PIJWM_IMPLEMENTATION_TRACKER.md` | Step/模块矩阵 | audit/current |
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
| STEP 4.4 Structured RSSM World Model | `code/src/pi_jwm/step4_4_structured_rssm_world_model_v1.py`、`code/scripts/build_step4_4_structured_rssm_world_model_v1.py`、Step 4.4 artifact | structured prior/posterior、local Action routing、vehicle/CSI dynamics、known stochastic outage、Return tri-state、valid-predecessor DAG、terminal Flow status、dynamic graph recursive rollout | COMPLETE / FROZEN；untrained CPU evidence；no Loss/Training/Planner |
| STEP 5.0 Definition 05 Decision Contract | `docs/contracts_PIJWM_DEFINITION_05_LOSS_TRAINING_EVALUATION_V1.md`、`docs/implementation_records/STEP_05_0_DEFINITION_05_DECISION_FREEZE.md` | 10 项 researcher decisions、旧实现复用审计、STEP 5.1 target/mask/unit前置条件 | DECISION FROZEN；IMPLEMENTATION NOT STARTED |
| STEP 5.1A-PATCH Future Motion / CSI Target Contract | `code/src/pi_jwm/step5_1a_motion_csi_target_contract_v1.py`、`code/scripts/build_step5_1a_motion_csi_target_contract_v1.py`、`code/artifacts/protocols/pi_jwm_step5_1a_motion_csi_target_contract_v1_20260921/` | local one-step Vehicle Motion、current physical input-slot alignment、future outcome CSI、current model relation-slot/identity/endpoint/type/RB alignment、component masks、frozen normalization、raw bridge、NPZ/receipt | COMPLETE / FROZEN FOR TARGET CONTRACT；non-locked development evidence；当前模型未读取 |
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

## 历史代码定位

不要按文件名猜是否弃用。先查 `docs/registries/historical_method_registry.json`，再查 `docs/registries/generated/archive_candidate_registry.csv` 和依赖图。211 个历史候选当前只做逻辑归档，未物理移动。

Unverified：未在依赖图、当前 runner、config 或实验 manifest 中出现的模块，不得仅凭名称说成当前执行路径。
