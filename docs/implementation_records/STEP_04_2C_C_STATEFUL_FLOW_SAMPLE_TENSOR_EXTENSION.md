# STEP 4.2C-C — Stateful Flow Sample/Tensor Additive Extension

状态：**COMPLETE / FROZEN（本 Step 范围）**
日期：2026-09-20
基线：`9090985b034d8b26f444b20283e0df0d1a406f0e`（STEP 4.2C-C 基线；本 Patch 在其上完成）

## Step Goal

将已经冻结的 STEP 4.2C-B Raw logical Flow / Carrying rows，按 STEP 3.1F 因果边界贯穿到独立版本化的 Model-ready Sample 与 CPU fixed-shape Tensor。只做 index、align、mask、连续字段 train-only normalization 和 collation。

## Definition Basis

依据：

- `docs/contracts_PIJWM_STEP_04_2C_B_CAUSAL_FLOW_LEDGER_RAW_V1.md`
- `docs/contracts_PIJWM_STEP_04_2C_C_STATEFUL_FLOW_SAMPLE_TENSOR_V1.md`
- `code/src/pi_jwm/step4_2c_b_causal_flow_ledger_raw_v1.py`
- STEP 3.1F/3.3 History、stable index、target isolation 和 mask 合同。

## Initial State

STEP 4.2C-B 已有真实 Input/Return 和两-hop Raw Ledger，但 Sample/Tensor 尚未接入；旧 Step 3 hop-service Flow 数组不能直接充当 logical Flow tensor。基线工作树 clean，未使用 GPU、training、locked_test 或 formal Dataset。

## Files Involved

- `code/src/pi_jwm/step4_2c_c_flow_sample_tensor_v1.py`
- `code/scripts/build_step4_2c_c_flow_sample_tensor_v1.py`
- `code/scripts/run_step4_2c_c_real_multihop_cross_slot_v1.py`
- `code/tests/test_step4_2c_c_flow_sample_tensor_v1.py`
- `docs/contracts_PIJWM_STEP_04_2C_C_STATEFUL_FLOW_SAMPLE_TENSOR_V1.md`
- `code/artifacts/protocols/pi_jwm_step4_2c_c_flow_sample_tensor_v1_20260920/`
- `code/artifacts/protocols/pi_jwm_step4_2c_c_real_multihop_cross_slot_v1_20260920/`

## Changes and Reuse

- 新增 `logical_flow` History union 与 target-side namespace；Epoch、FlowID、logical destination、E2E state 和 Carrying state 原样消费 C-B Raw，不在 Sample/Tensor 重算。
- Logical Flow 与 Carrying State 分离；route 用 bounded node-index + mask 表示；known inactive 与 padding 分开。
- 仅对五个连续字段做 train-only mask-aware normalization，categorical/identity/reference 不标准化；显式 development capacity 超限拒绝。
- 复用 STEP 4.2A 的 base Sample/Tensor collation、STEP 3.3 的 fixed-shape conventions 和 C-B Raw amendment；未修改 AirFogSim、旧 Raw、旧 Step 3 semantic arrays 或 graph/model code。
- 新增低 wired-capacity 的真实 CPU/non-locked 跨时隙 runner，使同一 Flow 在多个 Decision 中保留 carrying progress；这只是观察证据，不是正式 Dataset。

## STEP 4.2C-C-PATCH — Presence-aware Normalization & Full Sample→Tensor Flow Semantic Coverage

本 Patch 只修正 Sample/Tensor completeness 与 normalization consistency，不改变 4.2C-B Flow 语义，也不进入 Graph Builder。

- `_iter_numeric()` 及 stats contract 现在明确使用 `known=true AND presence=true AND feature_mask=true AND value!=null AND split=dev_train`；`mask_policy` 精确写入 `presence=true AND feature_mask=true AND value!=null; train split only`。重复的 `presence=false` completed/superseded lineage 不会改变 train-only stats，并有 negative fixture。
- Sample→Tensor equality 分别检查 History Logical、History Carrying、target Logical、target Carrying 的全部数值、mask、index、状态、route 与 ID/provenance；raw sample 与 normalization copy 均可被严格核对，字符串 ID 篡改即使 numeric index 不变也会失败。
- Tensor 新增完整 target carrying namespace（known/active/RouteRevision/hop/holder/source/destination/raw/normalized/mask/route），并明确其是 future ground-truth/deterministic-transition state，不是 learned prediction head。
- validator/receipt 实际 AND semantic 子检查、target namespace、future Epoch、placeholder、History/target bounds、route mask、normalization policy、round-trip、deterministic rebuild 与 scope；negative receipt/semantic tamper fixture 证明失败会传递到顶层。

本 Patch 的版本：normalization=`PI-JWM-Step-4.2C-C-Train-Only-Flow-Normalization-v2-presence-aware`，tensor=`PI-JWM-Model-Input-Tensor-Collation-v5-step4.2C-C-PATCH`，acceptance=`PI-JWM-Step-4.2C-C-PATCH-Acceptance-v1`。

## Validation

已通过：

- `python -m unittest discover -s code/tests -p 'test_step4_2c_c_flow_sample_tensor_v1.py'`：23/23；
- Step 4.2C-B focused：25/25；Step 4.2A focused：17/17；Step 3.3 focused：8/8；
- builder：`python code/scripts/build_step4_2c_c_flow_sample_tensor_v1.py --refresh-existing`；最终 `passed=true`；
- real cross-slot runner：AirFogSim environment、wired manager source、transport split、progress/lifecycle 和 scope checks 全部 true；
- artifact 内 deterministic semantic digest、NPZ serialize/load、Sample/Tensor equality、tamper、future epoch isolation、overflow rejection 全部通过。

最终 Patch artifact 的 `acceptance.json` 机器 required checks 全部 true，顶层 `passed=true` 且 validator 的 `declared_matches_expected=true`；scope 为：`graph_builder=false`、`information_graph=false`、`physical_topology=false`、`training=false`、`gpu=false`、`locked_test=false`、`formal_dataset=false`。

## Results / Evidence Boundary

真实跨时隙 trace 观察到 `flow::Task_1::Input::0` 保持单一 FlowID/Epoch、固定 logical destination `cloudServer_4` 和同一 Tensor slot；低容量 wired hop 在连续 Decision 中推进，最终 delivery 才完成。direct Input/Return 真实 source 保持 Input/Return 类别分离。DepData vocabulary 保留但 runtime=0。

这不等于 formal Dataset、Graph Builder、Physical/Information graph、模型、训练或性能结论。Return multi-hop、same-destination partial-hop reroute 和 formal capacity 仍未声称为真实证据。

## Expected vs Actual

预期的 Raw→Sample→Tensor additive extension、stable namespace、mask/presence、train-only normalization、round-trip 和 negative guards 均已实现并通过。与范围一致，Graph Builder 与后续模型链仍未开始。

## Known Issues / Stop Boundary

- 旧 direct trace 的 wired structural relation 需要显式 contract fixture 才能满足既有 Step 4.2A base validator；该 fixture 不改变 Flow semantics。
- formal Flow capacity、Return multi-hop 和 reroute runtime semantics 仍未冻结。
- 不得把本 Step artifact 外推为正式 Dataset 或模型可用性/性能结论。

## Git / Next Step

本记录与代码、测试、force-added artifact/manifest 一并提交；最终 commit、push 和分支状态以本次 Completion Report 为准。唯一下一建议：研究者审阅后，另行授权 **Definition 03 Graph Builder Contract**；本 Step 不自动进入 Graph Builder。
