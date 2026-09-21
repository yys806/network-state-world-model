# STEP 5.1B — Definition 05 Posterior / Loss / KL / Metric Implementation

## Step Goal

实现 Definition 05 的 additive posterior、Motion/CSI family loss、analytic KL、验证损失和逐 horizon 指标原语；不实现训练循环或优化器。

## Definition Basis

依据 `docs/contracts_PIJWM_DEFINITION_05_LOSS_TRAINING_EVALUATION_V1.md`、STEP 5.0 决策和 STEP 5.1A-PATCH 的 12-sample target tensor。Future target 只允许进入 training-only posterior target encoder；prior 不接收 target。

## Initial State

5.1A 已提供 normalized/raw Motion 与 future per-RB CSI target、固定 current support 和 component masks；Loss/Posterior/Metric runtime 尚不存在。

## Files Involved

- `code/src/pi_jwm/step5_1b_posterior_loss_metric_v1.py`
- `code/tests/test_step5_1b_posterior_loss_metric_v1.py`
- `code/scripts/build_step5_1b_posterior_loss_metric_receipt_v1.py`

## Changes

- 新增独立 Motion/CSI target encoders、posterior teacher mean/reparameterized sample 和 target-free prior predictor。
- 新增 family-wise mask-normalized MSE、冻结统计量 raw→normalized bridge、Physical/Communication diagonal Gaussian KL（raw/adjusted free bits/count）、`L_Val` 聚合。
- 新增 per-horizon Motion component/xyz MAE/RMSE 与 CSI dB MAE/RMSE。
- 未新增 Event、outage、rate、service residual、Flow、Task、DAG、lifecycle 或 completion loss/head。

## Reuse

仅复用旧实现的 masked reduction 与 analytic KL 数学形式；未调用旧总 loss、NLL、overshooting、KL balancing 或旧训练 runner。

## Validation

- focused 5/5；STEP 5.1A regression 19/19；STEP 4.3B regression 21/21；STEP 4.4 regression 37/37。
- 12-sample non-locked tensor CPU receipt：`code/artifacts/protocols/pi_jwm_step5_1b_posterior_loss_metric_v1_20260921/acceptance_receipt.json`，target encoder→posterior→prediction bridge→loss→KL→metric finite forward 通过。
- Scope receipt 明确 `training=false`、`optimizer_step=false`、`gpu=false`、`formal_dataset=false`、`locked_test_accessed=false`、`performance_claim=false`。
- `compileall`、serialization/reload 和 gradient finite check 在最终验收命令中执行；无 optimizer step。

## Results / Expected vs Actual

实现结果符合当前 Definition 05 primitive contract。该 receipt 只证明 CPU 原语和开发 artifact 的可执行性，不证明训练收敛、GPU readiness、正式 Dataset 或预测性能。

## Known Issues

训练 curriculum、optimizer、正式 Dataset、prior recursive validation runner 和性能 claim 仍未实现，属于后续 STEP 5.2 或更后授权范围。

## Git

待最终验证后提交并推送 `origin/main`。

## Next Step

研究者审阅本 Step；下一动作仅为单独授权后的 STEP 5.2 training contract implementation。
