# STEP 5.1B-PATCH — Per-Horizon Posterior / Real World-Model Integration / KL-Metric Correction

## Step Goal

修正原 5.1B 实现的逐 horizon teacher leakage、独立 prior、fake h/identity loss、KL reduction、raw-unit metric 与弱 receipt；接入真实 STEP 4.4 CPU world-model path。不实现训练循环或优化器。

## Definition Basis

依据 `docs/contracts_PIJWM_DEFINITION_05_LOSS_TRAINING_EVALUATION_V1.md`、STEP 5.0 决策和 STEP 5.1A-PATCH 的 12-sample target tensor。Future target 只允许进入 training-only posterior target encoder；prior 不接收 target。

## Initial State

原 5.1B 已提供但语义不完整：target encoder 聚合 horizon，receipt 使用 zero h 和 target==prediction，prior/decoder 未接入真实 STEP 4.4，KL 与 metric 不能作为冻结合同证据。

## Files Involved

- `code/src/pi_jwm/step5_1b_posterior_loss_metric_v1.py`
- `code/tests/test_step5_1b_posterior_loss_metric_v1.py`
- `code/scripts/build_step5_1b_posterior_loss_metric_receipt_v1.py`

## Changes

- Target Encoder 保留 `[B,L,S,D]` horizon，并把 component/RB mask 作为网络输入 evidence；空 slot 输出零。
- 新增参数分离的 Physical/Communication future posterior teacher；正式 KL 使用真实 STEP 4.4 `phy_prior/comm_prior`，decoder 使用真实 `vehicle_decoder/csi_decoder`。
- KL 先逐 latent dimension 应用 free bits，再按 horizon/slot eligibility 归约；新增完整 `L_Mot/L_CSI/L_Pred/L_KL/L_Total` primitive 与 counts/availability。
- Motion raw-unit 指标分离 `delta_x/y/z (m)` 与 `next_speed (m/s)`；CSI 指标明确 `dB` 与 valid count。
- 未新增 Event、outage、rate、service residual、Flow、Task、DAG、lifecycle 或 completion loss/head。

## Reuse

仅复用旧实现的 masked reduction 与 analytic KL 数学形式；未调用旧总 loss、NLL、overshooting、KL balancing 或旧训练 runner。

## Validation

- focused 5/5；STEP 4.4 regression 30/30；`compileall` 通过。
- 真实 12-sample non-locked CPU receipt：`code/artifacts/protocols/pi_jwm_step5_1b_posterior_loss_metric_v1_20260921/acceptance_receipt.json`，逐 horizon、mask evidence、真实 prior/decoder、逐维 free bits、梯度、raw-unit metric 和 target isolation 均由机器检查计算。
- Scope receipt 明确 `training=false`、`optimizer_step=false`、`gpu=false`、`formal_dataset=false`、`locked_test_accessed=false`、`performance_claim=false`。
- `compileall`、serialization/reload 和 gradient finite check 在最终验收命令中执行；无 optimizer step。

## Results / Expected vs Actual

Patch 的 CPU 原语和真实 STEP 4.4 接线符合当前检查；但当前 target support 为 10/74，而真实 STEP 4.4 development model support 为 8/44，receipt 采用显式固定支持子集并记录该对齐限制。因此尚不能宣称 5.1B COMPLETE/FROZEN。

## Known Issues

模型支持与 5.1A target support 的完整 10/74 对齐仍需研究者确认；训练 curriculum、optimizer、正式 Dataset、prior recursive validation runner 和性能 claim 仍未实现。

## Git

Patch 待文档/索引/回归最终验证后提交并推送 `origin/main`。

## Next Step

完成 Patch 验收后由研究者决定是否授权 STEP 5.2；本 Patch 不自动进入训练。
