# STEP 5.1D - Unified Graph / Encoder / World-Model Rebuild & Paired 5.1B Acceptance

日期：2026-09-22
状态：COMPLETE / CPU DEVELOPMENT INTEGRATION ONLY
范围：non-locked unified development bundle；未训练、未用 GPU、未访问 locked_test。

## Step Goal

将同一个 12-sample unified Tensor 逐样本贯穿 Typed Dual Graph、Dual-Graph Encoder、STEP 4.4 state/dynamics、真实 Future Action、recursive hidden state、Prior/Posterior、Decoder、Prediction Loss、KL、Raw-unit Metric 和 backward probe。

## Definition Basis

任务依据为 `TASK/Step5.1D.md`、Definition 03/04/05 的已冻结工程合同、STEP 5.1A-PATCH/5.1B-PATCH/5.1C-PATCH records。没有改变研究定义、latent layout、规则边界、normalization 规则或 action semantics。

## Initial State

STEP 5.1C 已提供 12 个 paired development windows，容量 `max_entity=10`、`max_comm_relation=74`，并证明 identity、no-prefix/no-crop 和 unified dev_train-only Flow stats。旧 5.1B receipt 使用旧 carrier，不能作为本 Step 的 paired acceptance。

## Files Involved

- `code/scripts/build_step5_1d_unified_model_chain_v1.py`
- `code/tests/test_step5_1d_unified_model_chain_v1.py`
- `code/artifacts/protocols/pi_jwm_step5_1d_unified_model_chain_v1_20260922/`
- `docs/PIJWM_IMPLEMENTATION_TRACKER.md`
- `AI_CONTEXT/00_PROJECT_STATE.md`, `03_DATA_FLOW.md`, `04_MODULE_MAP.md`, `05_EXPERIMENTS.md`, `07_KNOWN_ISSUES.md`, `08_CHANGELOG.md`

## Changes

1. 以 `save_flow_tensor_batch/load_flow_tensor_batch` 写入并恢复完整 package：contract、sample_ids、sample_static、sample_metadata、base Step 3.3 checks、Flow normalization stats 和全部 ndarray。
2. 从 unified Tensor 重建 4.3A graph（`radius_knn`, `radius_m=1000`, `k=2`, no self-loop；仍为 development-only），再重建 4.3B encoder，并复用 upstream non-Flow stats、统一 bundle Flow stats 和 4.3A relation stats。
3. 每个 sample 独立恢复 STEP 4.4 state/dynamic graph；Future Action 从 sample 自身轨迹恢复。Mobility 48 条映射，Comm 1 条唯一映射，Comp/Route 为空动作均显式 no-op。
4. 对每个 sample 真实执行两个 horizon 的 `one_step` 递推，接入真实 prior、逐 horizon target encoder/posterior、同一 vehicle/CSI decoder、冻结 target normalization、family MSE、analytic KL、per-dimension free bits 和 raw-unit metrics。
5. 添加 focused tests，固定 CPU evidence seed=5101；receipt 增加完整 package checks、receipt tamper negative、梯度组检查和 deterministic output verification。

## Reuse

复用现有 `save/load_flow_tensor_batch`、4.3A builder、4.3B encoder、4.4 Structured RSSM、5.1A target contract 和 5.1B primitives；没有新增科研接口或改变冻结架构。

## Validation

- 初始 5.1D receipt：42/42 checks true；本 Patch 更新后的最终 receipt：47/47 checks true，`passed=true`；12 samples；prior/posterior/decoder 路径各 24 次；full Tensor package round-trip true；gradient probe finite/non-zero；required-check 与 forbidden-scope tamper negative true。
- capacities：Physical 10，Communication 74；pairing 12/12；unsupported/unresolved/fixed-support blocked counts 均为 0。
- focused tests：5/5。
- regression tests：4.3A 15/15、4.3B 21/21、4.4 30/30、5.1A 19/19、5.1B 5/5、4.4 communication audit 7/7。
- builder receipts：5.1C 12-sample rebuild passed；5.1A/5.1B/4.3A/4.3B/4.4 temporary regression rebuilds passed。
- deterministic rebuild：同一脚本、同一 seed 的两次独立输出逐文件 SHA-256 相同。

## Results and Boundary

这证明 unified development Tensor→Graph→Encoder→paired STEP 4.4→Loss/KL/Metric 的 CPU integration path 已真实贯通，且 sample identity、causal isolation、mask/padding、normalization provenance 和 gradient signal 有机器证据。它不证明模型已训练、收敛、泛化、性能优于 baseline、正式 Dataset 合格或 Planner 可用。

`training=false`, `optimizer_step=false`, `gpu=false`, `formal_dataset=false`, `locked_test_accessed=false`, `performance_claim=false`, `planner=false`。模型权重仍为 untrained development evidence；STEP 5.2 未开始。

## STEP 5.1D-PATCH — Evidence / Context / Action-Coverage Closure

### Goal and Changes

本 Patch 不改变模型或科研设计，只闭合共享证据、上下文状态、动作覆盖和 prior-target 隔离。receipt 现在区分 `executed_scope` 与 `forbidden_scope`；小型 JSON evidence 被明确列入 GitHub tracked manifest。unified Flow stats 的 8 个 `source_sample_ids` 与 bundle `dev_train` lineage 精确一致；旧 4.2A stats 没有逐 sample IDs，审计从冻结 `batch.json` 的 samples/provenance 恢复 fit source，并与 4.2A `normalized_samples.json` 的 `sample_id/trajectory_id/anchor_decision_frame/split` 逐项、同序比较。对真实 paired sample 改变 Motion/CSI target 后，actual `phy_prior`/`comm_prior` 的 mean/log_std 保持完全一致，posterior 发生变化。

### Validation

- receipt：47/47 required checks true，`passed=true`。
- exact upstream train lineage：true，8/8 exact order。
- runtime prior-target isolation：true；posterior target sensitivity：true。
- action coverage：Mobility=48，Comm=1，Route=0，Comp=0；Route/Comp 仅 explicit no-op，未伪造非空动作。
- executed scope：encoder/graph/message passing/world model/posterior/loss/metric=true。
- forbidden scope：training/optimizer_step/GPU/formal_dataset/locked_test/Planner/performance_claim 全为 false；required-check tamper 与 forbidden-scope tamper 均使计算结果失败。
- GitHub tracked evidence：`acceptance_receipt.json`、`manifest.json`、pairing/identity/action audits、gradient/recursive/metric summaries、normalization lineage audit；巨大 full audit 与 binary package 保留 local-only。

### Boundary

这是 non-locked CPU development integration evidence；模型权重仍未训练。未开始 Training Loop、optimizer、tiny-data overfit、GPU、formal Dataset、locked_test、Planner、baseline 或 performance claim。Route/Comp non-empty coverage 是未来 formal training/data coverage gate，不在本 Patch 作科研决定。

## Git

本 Patch 完成最终 verification 后提交并推送到 `main`。

## Next Step

研究者审阅本 Step 后，唯一建议下一步为另行授权 `STEP 5.2 — Training Loop / Curriculum / Joint Training Implementation`；本记录不自动执行。
