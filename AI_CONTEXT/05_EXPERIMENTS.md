# 当前与历史实验

## 2026-09-21 STEP 5.1A-PATCH target-contract validation

- 这不是训练实验：脚本从真实 STEP 4.2A non-locked development trajectory 和 STEP 4.3B frozen train-only stats 构造 12 个 Future Target samples/tensor。
- 19/19 focused contract tests 与 102/102 related regression 通过；receipt 的 local H1/H2 Motion、current physical/model comm slots、sample/tensor、deterministic rebuild、real trajectory、formal_dataset/training/gpu/locked_test scope checks全部为 true，顶层 `passed=true`。artifact digest 为 `dc6c5b0b0d957f0e1e57ee09c19d2b54e7b2ecd0bb631a9612a8200078ab579a`。
- 证据覆盖 local one-step Motion/next-speed、相邻帧 component masks、future entity/order/birth/disappearance、未来 outcome per-RB CSI、History CSI 隔离、current model slot identity、wired/missing masks、future-only isolation、unsupported side metadata、normalization round-trip、serialization 和 tamper rejection。它不是正式 Dataset、Loss/Posterior/Metric、训练或性能结论。

本文件只记录客观状态，不自动解释科研意义。Source of truth：完整字段见 `docs/registries/experiment_registry.json`，正式数字见 `results_registry.json` 和对应 acceptance JSON。

> 2026-09-19：新的 active workflow 已冻结 Raw Trajectory Layer / 01。Step 2.4 是非 locked 真实通信接口验收，不是训练实验；下列 P4/P6 实验保留为旧定义下的 Historical / Archived evidence。未启动 GPU，未访问 `locked_test`。

Step 2.4 机器证据：`code/artifacts/protocols/pi_jwm_communication_outcome_semantics_v1_20260919/`，真实 6 slot / 7 independently recaptured Decisions / 14 checks；wireless、wired、total transmitted progress 与 task lifecycle 对齐。

## 旧 P4 正式实验（Historical / Archived）

| ID | Seed | 数据/协议 | 状态 | 客观边界 |
| --- | ---: | --- | --- | --- |
| `P4-EARSSM-SEED-20260831` | 20260831 | v5 causal-motion h20 / frozen v2 | 单 seed 通过，best epoch 39 | unlocked 单 seed，不是跨 seed 结论 |
| `P4-EARSSM-SEED-20260830` | 20260830 | 同上 | 单 seed 通过，best epoch 40 | unlocked 单 seed，不是跨 seed 结论 |
| `P4-EARSSM-SEED-20260832` | 20260832 | 同上 | deferred，未运行 | 没有结果；需要用户明确授权 |

共同训练入口：`code/scripts/run_formal_p4_entity_rssm_gpu_v1.py`。共同 tensor：`code/artifacts/formal_tensor/pi_jwm_v4_tensor_v5_causal_motion_h20_unlocked_20260906/`。冻结协议：`code/artifacts/protocol/pi_jwm_p4_entity_rssm_frozen_protocol_20260906_v2/protocol.json`。

## 正式验收证据

- seed 20260831：`code/artifacts/audit/pi_jwm_p4_entity_rssm_seed_20260831_acceptance_20260908/single_seed_acceptance.json`
- seed 20260830：`code/artifacts/audit/pi_jwm_p4_entity_rssm_seed_20260830_acceptance_20260909/single_seed_acceptance.json`
- 两份报告均为 `status=passed`、`locked_test_accessed=false`、`formal_performance_claim_ready=false`。
- 完整 9 项指标保存在 `docs/registries/results_registry.json`，并由索引生成器自动与 acceptance 核对。

## 前置门

- `P4-EARSSM-CPU-CONSISTENCY`：机制、梯度和 strict reload 证据通过；不证明性能。
- `P4-EARSSM-GPU-BATCH-PROBE`：选择 batch=8，未执行 optimizer step；不证明性能。
- `P4-FIRST-PRINCIPLES-20260906`：历史 global RSSM 表达限制的只读诊断；动机证据，不是新方法性能。

## 重要历史方法

以下均不是当前方法，详情与原始路径见 `docs/registries/historical_method_registry.json`：

- aggregate dual-graph residual baseline：通信/资源诊断有部分通过，node 位置和逐 RB/实体级边界未闭合。
- physical-edge feedback GRU：冻结 sentinel 的 validation link-F1 门失败。
- link persistence residual：单独修链路头不足以闭合完整 P4。
- global complete RSSM：补齐 prior/posterior/KL 语义，但全局池化/广播不能区分实体，且 sentinel node-x 门失败。
- node-x non-degradation loss：修正幅度下降但共享 base 变差，sentinel No-Go。
- v11 selector/ranking：历史决策诊断，不是逐候选世界模型 rollout planner。

## 旧 P6 实验边界

`P6-CANDIDATE-ROLLOUT-AUDIT-20260826` 状态为 `blocked_prototype_only`。CPU 原型存在，但 P6 未开放，不得运行正式 planner GPU 或给出规划收益结论。

Unverified：第三 seed 结果、三 seed 均值/方差、locked test、最终泛化和正式 planner 收益均不存在。

## 2026-09-22 STEP 5.1B-PATCH Posterior / Loss / KL / Metric

修正原 5.1B 的跨 horizon target aggregation、mask evidence 缺失、独立 prior、zero-h/identity loss、batch-level free bits、raw-unit metric 和硬编码 receipt。当前 `TargetEncoder` 输出保留 `[B,L,S,D]`，future teacher 按 Physical/Communication 分离；receipt 真实调用 STEP 4.4 current latent/dynamics/priors/decoders，并验证 temporal isolation、mask evidence、per-dim free bits、gradient、raw-unit metric、prior target isolation。

截至 STEP 5.1B-PATCH 的历史快照：Focused 5/5，STEP 4.4 regression 30/30，12-sample non-locked development receipt `passed=true`。随后 5.1C/5.1D 已统一 support 为 10/74 并完成 paired integration；5.2 仅增加 CPU development optimizer smoke，当前仍明确 `full_training=false`、`gpu=false`、`formal_dataset=false`、`locked_test_accessed=false`、`performance_claim=false`。

## 当前新定义实验状态

## 2026-09-22 STEP 5.2 Training Loop / Curriculum / Joint Training

- 这是 CPU development implementation smoke，不是正式训练实验。`Step52Trainer` 接入 8 个 `dev_train` 与 4 个 `dev_validation` unified samples；Stage 1 使用 family-specific posterior teacher，Stage 2 使用 prior-only recursive rollout，当前 `L=2` 的配置化 curriculum 为 `1→2`。
- receipt `code/artifacts/protocols/pi_jwm_step5_2_training_loop_v1_20260922/acceptance_receipt.json` 的 20/20 required checks 为 true：KL warm-up/free bits、joint optimizer groups、known-rule 参数排除、两步 CPU optimizer update、prior-only validation、`argmin L_Val` selector、checkpoint/reload/resume、causal leakage negative checks 和 reproducibility 均通过。
- smoke diagnostic `L_Val=168.31609344482422` 不是性能结果；没有 tiny-data overfit、full training、GPU、formal Dataset、baseline、Planner 或 locked-test。Route/Comp non-empty coverage=0，Comm=1，Mobility=48，Route/Comp 仍是 future formal training/data coverage gate。

- `STEP 1` 只有只读实现审计和 49 项旧 synthetic CPU contract 回归；它们不是新定义性能实验。
- 新定义正式 Dataset、full training、性能、planner 实验均为 `NOT_STARTED`；5.1B/5.1D primitives 与 5.2 CPU development loop 不是正式性能实验。
- STEP 4.2C-C 是非训练的 CPU/non-locked contract validation：真实 direct Input/Return、真实两-hop和低 wired capacity cross-slot trace进入 additive Flow Sample/Tensor artifact；`passed=true`，不构成 formal Dataset、模型或性能实验。
- STEP 4.3A 是非训练的 CPU/non-locked representation validation：复用冻结的五个 development Tensor samples，生成 typed graph artifact；24 项 required checks 与 20 项 negative/counterfactual 均通过。这不是图编码器实验、正式 Dataset 或性能结论。
- STEP 4.3B 是 `UNTRAINED_DEVELOPMENT_ENCODER_EVIDENCE`：复用同一批 frozen development inputs，在 CPU 上验证结构接线、History 因果、mask、方向、P2A/P2C、置换等变、序列化、确定性与 backward。5.2 之后该 encoder 可在 CPU development loop 中参与 joint optimizer smoke，但没有正式训练或性能结论。
- 下一实验步骤尚未授权；Step 2 建议仅冻结一步轨迹和四类动作合同，不训练。
- STEP 3.1 原 10 项/4 tests 记录已由 STEP 3.1R 修正证据取代，不再作为当前合同验收。
- STEP 3.1F 不是训练实验：最小样本通过 24 项合同 checks、12 项 focused tests、round-trip；History 为 `O_1+A_1+Y_1+O_2`，Action/Target `[2,3]`，History union index、Future Action ID↔index 对齐、history relation/DAG/flow 对齐已验收。未来 reference audit 扫描 4 个非 locked Raw artifact、18 个窗口，0 个 unresolved reference；locked/training/gpu 均为 false。
# 2026-09-19 STEP 3.2 validation

- `code/artifacts/protocols/pi_jwm_step3_2_raw_to_dataset_batch_v1_20260919/` is an observation-only development bundle: 3 independent trajectories, 12 causal windows, `dev_train=8`, `dev_validation=4`.

- STEP 3.2-PATCH finalized isolation evidence in the same bundle: provenance records real seed/source SHA/config lineage/time ranges/slot duration and 3.1F contract version; time-grid and execution timing checks run before windowing; future-reference audit is a separate Git-tracked observation artifact with 12 candidate/12 constructed/0 unresolved windows; normalization units are `m/s`, `m/s^2`, and `AirFogSim data-unit`.
- STEP 3.2-PATCH-RECEIPT finalized the machine receipt: top-level `passed` is the AND of computed required checks plus explicit non-locked scope checks; negative fixture verified failure propagation; sample contract provenance reuses the frozen schema constant. STEP 3.2 is COMPLETE / FROZEN.
- The split is trajectory-level; normalization is fit only on train valid masked values for speed, canonical acceleration and task size. The bundle is not a formal Dataset and has no training/GPU/locked-test evidence.
# 2026-09-21 STEP 4.4 Communication Service Source Audit（历史前置门）

- Evidence class: `SOURCE_AUDIT_ONLY_NO_WORLD_MODEL_IMPLEMENTATION`.
- Verdict: `SERVICE_RESIDUAL_RESEARCH_DECISION_REQUIRED`.

# 2026-09-21 STEP 4.4 Structured RSSM acceptance

- Researcher resolved the audit gate by selecting an independent known stochastic outage event and closing learned service residual.
- PATCH3 focused tests: 30/30. Formal machine receipt: 92/92 required checks (50 original + 21 service + 21 structural), including the canonical real-adapter unknown Return path, typed Existing Return binding, Future-Target support isolation, valid/invalid/multiple-predecessor DAG changes, and partial/intermediate/terminal Flow status transitions. Six formal artifact files match an independent rebuild by SHA-256 and size; evidence remains untrained CPU development only.
- Evidence class: `UNTRAINED_DEVELOPMENT_WORLD_MODEL_EVIDENCE`. No prediction/calibration/planning/performance claim; `training=false`, `gpu=false`, `locked_test=false`, `formal_dataset=false`.
- This is source/contract evidence only, not training, prediction accuracy, stochastic quality, or performance evidence.
# 2026-09-21 STEP 5.0 Definition 05 decision/audit

- Documentation and source audit only; no forward/backward, optimizer, training, GPU, or metric result was produced.
- Historical pre-5.1A audit: STEP 4.2A sample future position was raw-only and frozen tensor lacked future CSI target; the additive 5.1A target contract has since closed this prerequisite.
- Historical P4 checkpoints/results remain Historical Reference only and cannot be compared with the new method without matching dataset/split/history/horizon/target/normalization/metrics/seed policy.
- Scope: `implementation=false`, `training=false`, `gpu=false`, `formal_dataset=false`, `locked_test=false`, `performance_claim=false`.
# STEP 5.1D-PATCH acceptance（2026-09-22）

receipt 为 47/47 checks true，12 samples，capacity 10/74，prior/posterior/decoder calls 24/24/24，gradient probe finite/non-zero，unified stats source IDs、4.2A frozen batch recovery、exact upstream train lineage 与 runtime prior-target isolation 通过，deterministic rebuild identical；仅为 non-locked CPU development integration evidence。真实 action coverage 为 Mobility=48、Comm=1、Route=0、Comp=0，Route/Comp 仅 explicit no-op。
