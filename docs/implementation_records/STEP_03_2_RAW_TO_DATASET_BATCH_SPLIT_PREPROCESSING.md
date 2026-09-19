# STEP 3.2-PATCH - Dataset Isolation Evidence Finalization

本 Patch 只补齐 Dataset-level isolation 的机器合同和证据，不重做核心流水、不增加正式 Dataset，也不进入 Tensor、模型、训练或 GPU。

## Patch Changes

- provenance 从每条 Raw 实际读取 `trajectory_id`、`seed`、`source_path`、`source_sha256`、`split`、`schema_version`、`config_hash`（缺失则保留 `null`）、decision/step frame 与 simulation-time range、`slot_duration_s`、Step 3.1F sample contract version 和 seed/config lineage。
- 在 window construction 前检查 frame 连续性、decision time-grid、每个 step 的 execution start/end 与 decision/outcome 时间关系；中间 time gap 或跨 reset 不得构造 window。
- 增加 trajectory-id 唯一性、train/validation 无交集、source SHA 可追溯和相同 seed/config lineage 不得跨 split 的机器检查。
- 生成独立 `future_action_reference_audit.json`，统计本 Step development trajectories 的 candidate/constructed/unresolved windows、action family、object kind 和 affected rate；结果是 observation-only。
- normalization metadata 明确 `m/s`、`m/s^2`、`AirFogSim data-unit`，并增加 presence=false 极端 padding counterfactual 测试。
- validation report 的 split、causal、train-only、source SHA 和 lineage 检查均从最终 bundle/provenance/stats 计算，保留 evidence source，不使用未经计算的常量 True。

## Step Goal

验证多条独立 development Raw trajectory 能在冻结的 STEP 3.1F contract 下批量构造 causal window、按 trajectory 切分、只用 train 拟合连续字段统计并加载 batch。该 Step 不是正式 Dataset、训练或科研结论。

## Definition Basis

- `D:\shen\OB\科研\PIJWM\02数据集构建与模型输入.md`：History、Static、Future Action、Target、稳定 Entity Index、Presence/Mask 与输入不读取 future-only object。
- STEP 3.1F：`H=2, L=2`；History union input index；Future Action 先 anchor visibility 再引用同一 static index。
- Raw Trajectory Layer / 定义 01：Step 2.4 frozen Raw causal and communication outcome semantics。

## Initial State

已有 Raw artifact 的不同 schema/version 实际只有同一 `seed=0` trajectory，不能直接作为独立 split。旧 formal dataset protocol 不复用其 trajectory 数量、场景比例或 CPU policy。

## Files Involved

- `code/src/pi_jwm/step3_2_batch_preprocessing_v1.py`
- `code/scripts/build_step3_2_raw_to_dataset_batch_v1.py`
- `code/scripts/run_step2_4_real_airfogsim_communication_outcome_semantics_v1.py`（仅增加 seed/output 参数）
- `code/tests/test_step3_2_batch_preprocessing_v1.py`
- `code/artifacts/protocols/pi_jwm_step3_2_raw_dev_seed1_20260919/`
- `code/artifacts/protocols/pi_jwm_step3_2_raw_dev_seed2_20260919/`
- `code/artifacts/protocols/pi_jwm_step3_2_raw_to_dataset_batch_v1_20260919/`

## Changes

- 复用 frozen Step 2.4 collector 生成 seed 1/2 两条 non-locked development trajectory；不修改 Raw schema 或 AirFogSim。
- 新 batch builder 先检查 scope、trajectory_id、schema/version provenance、连续 frame，再按 trajectory 分配 `dev_train`/`dev_validation`，最后在 split 内构造 3.1F windows。
- 每个 sample 写入可反查的 `sample_id`、trajectory_id、anchor frame 和 split；拒绝 duplicate trajectory_id，即使来源 version 不同。
- 新 train-only mask-aware normalization 只处理 History 中 `speed_mps`、canonical acceleration、task size；忽略 padding、presence=false、feature_mask=false 和 null；保存 count/mean/std/unit policy/source/mask policy。
- 提供 validation apply、同 split collate、bundle manifest hash 和 serialize/load round-trip。

## Reuse

复用 `model_ready_sample_contract_v1.build_sample/validate_sample`、Step 2.4 collector、Raw causal/Outcome 语义和 JSON manifest 习惯。未修改 Raw Layer、Flow/DAG 科研语义、双图、World Model、Loss、Planner、训练或 locked_test。

## Validation

- `test_step3_2_batch_preprocessing_v1.py`: 5/5 PASS。
- `test_model_ready_sample_contract_v1.py`: 12/12 PASS；History past A/Y、History union index、Future Action index 对齐和 disappearing-object fixture 保持通过。
- seed 1/2 collector：真实 AirFogSim、wireless/wired/total、lifecycle、scope checks 全部 true；`training=false, gpu=false, locked_test=false`。
- batch build：3 trajectories、12 windows；`dev_train=8, dev_validation=4`；deterministic rebuild=true；round-trip loader 成功。

## Results

得到的是一个可复现的 non-locked batch/preprocessing validation bundle，不是正式研究 Dataset。trajectory-level split、因果窗口、train-only 统计和 batch load 证据均已保存。

## Expected vs Actual

符合预期。独立 trajectory 不足的问题通过最小 seed 1/2 development 采集解决；未扩大 Raw 语义或研究范围。

## Known Issues

- 仅 3 条短 development trajectory，不能支持正式 split 比例、scenario 覆盖或泛化结论。
- 只验证 JSON-native batch/collate；尚未进入 Tensor、模型、训练或 GPU。
- future-reference audit 仍为 STEP 3.1F observation，不外推 Dataset 结论。

## Patch Validation

- Step 3.2 focused tests: 11/11 PASS，包含 provenance、time-gap rejection、train-only validation counterfactual、mask=false 与 presence=false padding counterfactual、batch future-reference audit、negative `passed=false` fixture 和 frozen schema consistency。
- development batch audit: 12 candidate windows，12 successfully constructed，0 unresolved-reference windows，0 unresolved references，affected rate `0.0`；四类 action family 与 task/physical_entity 均为 0；结果仅为 Observation。
- 三条 development trajectory 的 provenance 均有真实 seed/config/source SHA/time-grid；train/validation trajectory-id 无交集。
- normalization units: entity speed `m/s`，canonical acceleration `m/s^2`，task size `AirFogSim data-unit`。
- validation receipt 顶层 `passed` 为所有 required checks 与四项 scope boolean 的逻辑 AND；负向 fixture 已证明 required check 失败时 `passed=false`。
- `model_ready_sample_contract_version` 直接复用 `model_ready_sample_contract_v1.SCHEMA_VERSION`，避免手写漂移。

## Git

待本 Step 全部验收后使用独立 Conventional Commit 推送 `main`；最终 hash 以 Git 回执为准。

## Final Status

STEP 3.2 正式 COMPLETE / FROZEN。后续任何 Step 均需研究者单独授权。
