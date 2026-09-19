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

## 历史代码定位

不要按文件名猜是否弃用。先查 `docs/registries/historical_method_registry.json`，再查 `docs/registries/generated/archive_candidate_registry.csv` 和依赖图。211 个历史候选当前只做逻辑归档，未物理移动。

Unverified：未在依赖图、当前 runner、config 或实验 manifest 中出现的模块，不得仅凭名称说成当前执行路径。
