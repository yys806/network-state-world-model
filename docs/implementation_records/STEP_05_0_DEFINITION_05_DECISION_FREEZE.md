# STEP 5.0 — Definition 05 Decision Freeze / Context Sync

## Step Goal

将研究者明确确认的 Definition 05 Loss / Training / Evaluation 决策正式落盘，对当前数据、STEP 4.4 与历史 Loss/Training/Evaluation 实现做定向复用审计，并在不实现 STEP 5.1、不训练、不使用 GPU、不访问 `locked_test` 的边界内同步项目上下文。

## Definition Basis

- 研究者 2026-09-21 明确确认的 10 项 Definition 05 决策，是本 Step 的最高科研决策依据。
- 只读定义源：`D:\shen\OB\科研\PIJWM\05模型训练、Loss与评价.md`，SHA-256 `62eebb05e2eeefe9edb0038f12964f59915a7acce03a683d6c7e5c71f85684e9`，20,020 bytes。
- 新决定与只读文件的冲突已显式记录：observation NLL → mean decoder + MSE；Event/Residual supervision → v1 only Motion/CSI/KL；overshooting → OFF。私人文件未被修改。
- STEP 4.4-PATCH3 commit `faf3b1bc6f40c17599138cae19a0fa9b6f0d96e1` 是当前 World Model 基线。

## Initial State

- STEP 4.4 已冻结 untrained CPU World Model mechanism，Loss、optimizer、Training 和 Definition 05 target posterior 尚未实现。
- 当前 Tensor 的 future entity target 只有 speed；STEP 4.2A sample 额外保存 future position raw value，但未归一化、未 collate 为 target tensor。
- 当前 target namespace 没有 future per-RB CSI。History CSI 不能替代 future CSI target。
- 历史 formal loss/runner/metrics 服务于旧 aggregate/entity-aligned P4 定义，包含 observation NLL、downstream rule-state losses、KL balancing、overshooting 或旧 checkpoint gates，不能直接当作新 Definition 05 实现。

## Files Involved

- `docs/contracts_PIJWM_DEFINITION_05_LOSS_TRAINING_EVALUATION_V1.md`
- `docs/implementation_records/STEP_05_0_DEFINITION_05_DECISION_FREEZE.md`
- `code/src/pi_jwm/step4_4_structured_rssm_world_model_v1.py`
- `code/src/pi_jwm/step4_3b_dual_graph_encoder_v1.py`
- `code/src/pi_jwm/step4_2a_graph_input_extension_v1.py`
- `code/src/pi_jwm/step3_3_model_input_tensor_v1.py`
- `code/src/pi_jwm/formal_world_model_loss_v1.py`
- `code/src/pi_jwm/formal_world_model_metrics_v1.py`
- `code/src/pi_jwm/formal_entity_aligned_rssm_world_model_v1.py`
- `code/src/pi_jwm/formal_training_protocol_audit_v1.py`
- `code/scripts/run_formal_dual_graph_gpu_train_v1.py`

## Changes

- 冻结 10 项研究决定及其公式、mask、leakage、training stage、trainable parameter、validation/checkpoint/evaluation 边界。
- 将旧只读 Definition 05 与新显式决定的冲突记录为“研究者已解决”，而不是偷偷沿用旧 NLL/overshooting。
- 建立 STEP 5.1 所需 target/normalization/unit bridge 和 negative-test 前置合同。
- 更新 tracker、authority records、AI_CONTEXT、process records 和 knowledge index。

## Reuse Audit

| 现有对象 | 分类 | 可复用内容 | 不可直接复用 / 所需变化 |
| --- | --- | --- | --- |
| STEP 4.4 structured RSSM prior-only rollout、mean vehicle/CSI decoder、fixed-support rule transition | DIRECT_REUSE | learned/rule boundary、stochastic family、prior target isolation、recursive rollout、规则状态恢复 | training-only target encoder、family loss/KL、normalized/raw bridge 尚未实现 |
| STEP 4.3B Dual-Graph Encoder architecture/interface | DIRECT_REUSE | 结构和接口；v1 joint training 时权重可训练 | 不能把 FROZEN 误写为 weights frozen；optimizer wiring 留到 STEP 5.2 |
| STEP 4.2A future position source与 train-only position stats | MINOR_MODIFICATION | future raw position和既有 train-only stats可追溯复用 | target position 未归一化、未 tensorize；必须 additive collate，禁止 refit |
| Future per-RB CSI target | MISSING | Raw future decision observation与现有 `_communication_rows`/CSI normalization helper可能作为来源复用 | 当前 target/sample/tensor均未提供；STEP 5.1 必须建立 causal target path 和 masks |
| STEP 3.3 target speed/mask、horizon、namespace isolation | MINOR_MODIFICATION | horizon axis、target index、speed target/mask、Future Target 隔离机制 | 需要与 future position/CSI target 和 current-support alignment 明确拼接 |
| `formal_world_model_loss_v1._masked_mean` | DIRECT_REUSE | 空 mask 安全的 mask-normalized reduction 思路/实现 | 必须由 STEP 5.1 tests 独立验证 family denominator |
| `formal_world_model_loss_v1._normal_kl` | MINOR_MODIFICATION | diagonal-Gaussian analytic KL 公式 | 改为 Phy/Comm 独立 mask、free bits、无 balancing、无 overshooting |
| 旧 `formal_world_model_loss` 总目标 | HISTORICAL_ONLY | 仅作反例与回归来源 | NLL、presence/event/lifecycle/DAG/system loss、balanced KL、overshooting 与新合同冲突，不能调用为 v1 loss |
| `formal_world_model_metrics_v1` horizon bucket / denormalization bookkeeping | MINOR_MODIFICATION | per-horizon accumulation、MAE/RMSE、raw-unit reporting模式 | 指标对象必须缩到 Motion/CSI；旧 NLL、event、system混合指标不能作为 v1 selector |
| `formal_entity_aligned_rssm_world_model_v1` | HISTORICAL_ONLY | prior/posterior separation和 prior-only部署经验 | posterior读取多类 full target，且显式产生 overshooting tensors；不能作为当前模型实现 |
| `run_formal_dual_graph_gpu_train_v1.py` | STRUCTURAL_CHANGE | seed、AdamW、日志、checkpoint save/reload、manifest等工程模式 | 旧 staged base-freeze、旧 loss、P4 gate selector和历史 dataset均不符合 joint curriculum；不可直接运行 |
| `formal_training_protocol_audit_v1.py` | MINOR_MODIFICATION | schema/split/hash/target completeness审计模式 | 旧 aggregate/per-RB target集合与 gate需换成 Definition 05 的 Motion/CSI/KL合同 |
| 历史 P4 checkpoints / metrics | HISTORICAL_ONLY | 仅作历史参考和工程回归 | 未证明 dataset/split/horizon/target/normalization/metric公平，禁止声称新旧优劣 |

## Validation

- 核验 `HEAD == origin/main == faf3b1bc6f40c17599138cae19a0fa9b6f0d96e1`，起始 worktree clean。
- 对 9 个当前/历史核心源码记录 SHA-256，并逐项读取实际 target、posterior、loss、KL、overshooting、validation 和 checkpoint 路径。
- 读取当前 STEP 4.2C-C tensor keys/shapes：没有 target position tensor、future CSI target 或图扩展 normalization stats内嵌字段；`target_entity_features` 仅一维 speed。
- 读取 STEP 4.2A normalized sample：future `position_m` 只有 raw `value/mask/unit`，没有 `normalized_value`；History position 已有 train-only normalized value。
- Definition 05 contract machine check：10/10 decision semantics、5/5 reuse classes 与 scope flags 通过；只读 Definition 05 SHA-256 保持 `62eebb05e2eeefe9edb0038f12964f59915a7acce03a683d6c7e5c71f85684e9`。
- 相关回归：STEP 4.4 30/30、历史 loss 11/11、metrics 15/15、protocol audit 4/4、training runner 15/15，共 75/75 通过。`compileall` 通过。
- 全量仓库测试额外运行 1849 项，出现 33 个既有环境/历史依赖 error：GBK 无法输出 AirFogSim emoji、已移除历史 artifact 缺失、旧 teacher-tensor fixture 不满足现行 RB-action 校验，以及 clean-tree 测试在本次文档 worktree 中不成立。本 Step 未改源码/测试；这些失败不作为 Definition 05 runtime 证据，也不被掩盖为通过。
- knowledge index write/check 与 `git diff --check` 通过；最终 Git 同步检查在提交后执行。

## Results

10 项科研决定已形成工程侧唯一 active Definition 05 contract。旧实现存在可复用数学/工程小部件，但没有任何旧 Loss、旧 runner 或旧 checkpoint 可以整体直接升级为当前实现。STEP 5.1 的首要工程任务不是套用旧 loss，而是先补齐 Motion/CSI target、mask、normalization lineage 和 posterior whitelist，再实现 loss/KL/metric。

## Expected vs Actual

符合 STEP 5.0 预期：科研决定已足够明确，不需要新增研究选择。实际审计比旧 tracker 更具体地暴露了 future CSI target 缺失和 Motion target normalized/raw bridge 未闭合；这些是 STEP 5.1 的工程缺口，不是本 Step 可跳过的细节。

## Known Issues

- STEP 5.1 尚未实现，当前没有 Loss/Posterior/Metric runtime acceptance。
- current sample/tensor target path 不满足 Motion/CSI training contract；不得开始训练。
- 当前 Dataset 仍为 development / non-locked，`formal_dataset=false`。
- `training=false`、`gpu=false`、`locked_test_accessed=false`；没有预测结果或性能结论。
- 外生 arrival/departure 的完整建模仍是更后续边界，但 v1 Definition 05 明确不增加 Event supervision/head。
- 仓库全量 suite 当前有 33 个与本次文档变更无关的环境/历史依赖 error；本 Step 只以 75/75 定向回归、合同检查、compile 和索引一致性作为验收，不宣称全量 suite 通过。

## Git

本记录与合同、上下文和审计矩阵在同一 STEP 5.0 commit 中提交并推送到 `main`；准确 SHA 在 Completion Report 中给出。

## Next Step

仅建议研究者确认后授权 **STEP 5.1 — Definition 05 Loss / Posterior / Metric Implementation**。不得自动实现 STEP 5.1，不得训练，不得使用 GPU。
