# STEP 5.3D — CSI Scale / Optimization Diagnosis

日期：2026-09-22

状态：**COMPLETE / DIAGNOSTIC-ONLY；等待研究者决定，不进入 STEP 5.4**

## Step Goal

只诊断固定 `[0,1]` `dev_train` subset 上 CSI raw decoder 输出的尺度与优化行为。不修改 Definition 05、loss 权重、模型默认初始化、decoder bridge、GPU 或 formal training。

## Definition Basis and Boundaries

- 复用 STEP 5.2 正式路径：Stage 1→Stage 2、H=1→H=2、KL warm-up/free-bits、`L_Total=L_Pred+beta_KL L_KL`、`0.5/0.5` family weighting。
- CPU only，固定 seed `5303`，最大 `200` optimizer steps，每 `20` steps 记录；不访问 validation 参数更新、GPU、formal Dataset、`locked_test`、Planner 或正式 baseline。
- mean-bias 仅在内存诊断 trainer 中把 CSI decoder 最后一层 bias 设置为 train CSI mean；没有写回正式模型默认配置。

## Files Involved

- `code/scripts/step5_3d_csi_scale_optimization_diagnosis_v1.py`
- `code/tests/test_step5_3d_csi_scale_optimization_diagnosis_v1.py`
- `code/artifacts/protocols/pi_jwm_step5_3d_csi_scale_optimization_diagnosis_v1_20260922/`

## Observation

- Train CSI normalization：mean=`98.34974797 dB`，std=`5.44766825 dB`。
- 初始 baseline H1/H2 actual normalized CSI MSE：`329.3314819` / `332.8831482`。
- Expected bridge MSE 与 actual MSE：H1=`329.3314819` / `329.3314819`，H2=`332.8831482` / `332.8831482`，差值均为 `0`。
- Baseline 200-step：H1 `329.3314819 -> 117.6610107`，H2 `332.8831482 -> 119.7403946`；relative drop=`64.27%` / `64.03%`，但最终 MSE 未达到 `<=1.0`。
- Mean-bias diagnostic 200-step：H1 `1.0934864 -> 0.1405199`，H2 `1.1040019 -> 0.1633186`；relative drop=`87.15%` / `85.21%`，达到诊断 gate。
- Baseline 初始 raw prediction 约 `0.02 dB`，target 约 `98.72–99.26 dB`；raw bridge 与 normalized loss 数学一致。
- Motion 分量、CSI dB raw metrics、KL、gradient audit 与每 20 step trajectory 均保存在 artifact。

## Interpretation

当前证据更支持 **raw-output initialization / conditioning bottleneck**，而不是 normalization bridge bug。该解释不是研究者决策，也不等于正式方法选择。

待研究者在以下候选之间明确决定：

1. raw CSI decoder 保持不变，仅采用 train-mean bias initialization；
2. decoder 在 normalized CSI space 输出，再在进入 `F^Trans` 前反归一化。

第二项涉及更大的 normalized/raw bridge 变化，本 Step 未实现。

## Verdict

- Scale bridge：`PASS`。
- Baseline：`TINY_OVERFIT_NO_GO`。
- Mean-bias diagnostic：`TINY_OVERFIT_GO`，仅为诊断证据，不是正式方案验收。

## Validation

- 5.3D focused：`2/2`。
- 5.3 focused：`4/4`。
- 5.2：`12/12`。
- 5.1D：`5/5`。
- 5.1B：`5/5`。
- 4.4：`30/30`。
- compileall、knowledge index write/check、`git diff --check`：通过。

## Git and Next Step

本记录与脚本、测试、artifact、context 同一提交。下一步只等待研究者选择上述两个候选之一；不自动进入 STEP 5.4，不启动 GPU。
