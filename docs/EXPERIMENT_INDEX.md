# PI-JWM 实验索引

STEP 5.5-PATCH 是正式数据消费路径和语义审计，不是训练/性能实验；机器凭证见 `code/artifacts/audit/pi_jwm_step5_5_patch_20260923/`。原 STEP 5.5 H4 smoke 只消费 `runtime/` 的 1+1 mini subset，PATCH 另行验证 4416/1104 full-shard 可索引及跨轨迹 CPU batch。

> 本索引负责回答“实验做过没有、它回答什么问题、结果在哪里”。具体结论必须回到原始实验目录和机器可读产物。

重要实验的结构化记录见 `docs/registries/experiment_registry.json`；全部 artifact 一级目录的自动目录见 `docs/registries/generated/artifact_catalog.csv`。注册表 v2 对每条重要实验使用统一字段；未知或不适用的内容必须写成 `null` 并在 `field_notes` 解释，不能直接省略。

STEP 5.5 Formal Dataset v1 acceptance 不是性能实验：60 条真实 trajectory、H2/L4、四动作覆盖、五类 package、确定性重建和 CPU H=4 interface smoke 的可追踪证据位于 `code/artifacts/manifests/pi_jwm_step5_5_formal_dataset_v1_20260923/`。

> 2026-09-19：新 `00–06` 已完成 Raw Trajectory Layer 的真实接口验收，但没有新训练实验。下列 P4/P6 条目全部按旧定义作 Historical / Archived evidence。

Step 2.4 真实接口验收不属于训练实验：机器证据位于 `code/artifacts/protocols/pi_jwm_communication_outcome_semantics_v1_20260919/`，覆盖 6 个 execution slot、7 个独立 Decision 和 14 项检查。

Step 3.1F 最小样本与 History 修正也不属于训练实验：证据位于 `code/artifacts/protocols/pi_jwm_model_ready_sample_v1_20260919/`，Raw source 为真实 v2 communication artifact，覆盖 `H=2/L=2`、24 项合同检查、12 项 focused tests、past action/outcome、History union index、Future Action ID↔index 对齐、history relation/DAG/flow 和 round-trip；`future_action_reference_audit.json` 对 4 个非 locked Raw artifact 的 18 个窗口做 observation-only 扫描，sample manifest 保存其 SHA-256 provenance；`locked_test=false`、training/gpu=false。

## 1. 旧协议正式实验（Historical / Archived）

| 实验 | 目的 | 入口/配置 | 结果和状态 |
| --- | --- | --- | --- |
| P4 entity RSSM seed 20260831 | 验证实体级双图 RSSM 在正式 unlocked tensor 上的单 seed 表现 | `code/scripts/run_formal_dual_graph_gpu_train_v1.py`；冻结协议 `code/artifacts/protocol/pi_jwm_p4_entity_rssm_frozen_protocol_20260906_v2/protocol.json` | `code/artifacts/experiments/pi_jwm_p4_entity_rssm_gpu_formal_seed_20260831_v1/`；单 seed 验收通过 |
| P4 entity RSSM seed 20260830 | 在相同冻结配置下检查另一 seed 的稳定性 | 同上；正式 run `code/artifacts/experiments/pi_jwm_p4_entity_rssm_gpu_formal_seed_20260830_v1/` | 已完成；单 seed 验收通过，audit 位于 `code/artifacts/audit/pi_jwm_p4_entity_rssm_seed_20260830_acceptance_20260909/` |
| P4 entity RSSM seed 20260832 | 第三个 seed 的跨 seed 证据 | 同上 | 未授权，不得自动启动 |

`20260832` 和后续远端同步已经登记在 `docs/registries/deferred_work.json`，状态为 `deferred`、`authorization_required=true`、`auto_start=false`。接口预留不等于运行许可。

## 2. 正式前置和机制实验

| 实验 | 所回答的问题 | 证据位置 |
| --- | --- | --- |
| 第一性原理审计 | 旧 global RSSM 是否能表达实体差异、运动和链路排序 | `code/artifacts/audit/pi_jwm_p4_first_principles_audit_20260906_v1/` |
| CPU 一致性审计 | 实体级 prior/posterior、运动合同、梯度和 strict reload 是否真实接通 | `code/artifacts/audit/pi_jwm_p4_entity_rssm_cpu_consistency_20260906_v2/` |
| GPU batch probe | batch 8 是否在显存边界内可执行 | `code/artifacts/audit/pi_jwm_p4_entity_rssm_gpu_batch_probe_20260906_v1/` |
| GPU execution sentinel | 正式 runner 是否能在 CUDA 执行两阶段训练 | 对应 `code/artifacts/experiments/pi_jwm_p4_entity_rssm_gpu_sentinel_20260906_v1/` 和 audit |
| 候选规划器 audit | 逐候选 rollout 机制是否具备可审计骨架 | `code/artifacts/audit/pi_jwm_formal_candidate_rollout_planner_audit_20260826/`；当前 blocked/原型边界 |

## 3. P4 历史候选族

这些实验保留用于解释失败原因，不得自动提升为当前方法：

重要历史方法的“为什么尝试—实际结果—为什么不再使用—被谁替代—原始证据”见 `docs/registries/historical_method_registry.json`。

- global complete RSSM：`code/src/pi_jwm/formal_complete_rssm_world_model_v1.py`，对应 `code/artifacts/experiments/pi_jwm_p4_complete_rssm_*`。
- node-x safe correction：对应 `code/artifacts/experiments/pi_jwm_p4_complete_rssm_node_x_safe_*`。
- edge feedback GRU：对应 `code/artifacts/experiments/pi_jwm_p4_edge_feedback_gru_input_*`。
- persistence residual：对应 `code/artifacts/experiments/pi_jwm_p4_link_persistence_residual_*`。
- 旧概率校准 Scheme B：对应 2026-09-01 的校准审计和失败记录。
- h20、threshold、throughput、position、link recall 等诊断：对应 `code/artifacts/audit/pi_jwm_p4_*diagnosis*` 和相关 `记录/研究进展/`。

## 4. 早期阶段实验族

| 阶段/前缀 | 主要内容 | 当前口径 |
| --- | --- | --- |
| P0/P1/P2 | 理论一致性、信息边合同、采集器、Attempt/Reject Ledger、正式数据 | 当前数据和合同的历史来源，按最新冻结版本读取 |
| R3/R4/R5 | 世界模型候选筛选、模块预检和 GPU screening | 历史候选证据，不覆盖 P4 |
| R6 | belief-conditioned direct policy 和闭环策略预检 | 不是候选动作世界模型规划器 |
| v6/v7/v8/v11 | 旧双图、active-rate、selector、收益可辨识性和策略实验 | 历史研究线，失败结果必须保留 provenance |

## 5. 新实验登记要求

新增重要实验时，至少登记：

- 实验 ID 和日期；
- 它回答的研究问题；
- 方法/模型身份；
- 代码入口和代码版本；
- 配置、数据/tensor manifest、split 和 seed；
- checkpoint、原始 metrics、audit 和 SHA-256；
- 结果状态：`running`、`passed`、`failed`、`blocked`、`historical`；
- 当前结论、证据范围和不能推出的结论；
- 是否访问 `locked_test`。

新增实验只能在用户批准研究目的和方法变量后执行。索引登记不等于实验通过。
