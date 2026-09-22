# STEP 5.3 — CPU Training Preflight / Tiny-Data Overfit / Go-No-Go

日期：2026-09-22

状态：**GO / DEVELOPMENT-ONLY CPU PREFLIGHT**

## Step Goal

在冻结 Definition 04/05、5.1D 和 5.2 语义不变的前提下，用固定 `dev_train` tiny subset 检查基本可学习性、Motion/CSI 双 family learning signal、prior recursive H1/H2、数值稳定性和 checkpoint resume。该 Step 不产生泛化、正式性能或 GPU 结论。

## Definition Basis and Boundaries

- Definition 05：`L_Total=L_Pred+beta_KL L_KL`，`L_Pred=0.5 L_Mot+0.5 L_CSI`，family-wise mask MSE、analytic KL/free bits、Stage 1 posterior teacher、Stage 2 prior recursive、validation prior-only。
- 使用真实 unified development bundle 的 `dev_train=8`，固定 tiny subset：Phase A/B 使用 index `0`，Phase C 使用 indices `0,1`；validation 只作 prior-only diagnostic，不参与更新。
- 预注册 development-only gate：family loss relative drop `>=0.5%`；prior H1/H2 relative `L_Pred` drop `>=0.5%`；最大步数 A=30、B=30、C=40、resume=20。

## Files Involved

- `code/scripts/step5_3_cpu_training_preflight_v1.py`
- `code/tests/test_step5_3_cpu_training_preflight_v1.py`
- `code/artifacts/protocols/pi_jwm_step5_3_cpu_training_preflight_v1_20260922/`
- `code/src/pi_jwm/step5_2_training_loop_v1.py`（仅增加 module-level learning-signal audit）

## Results

### Phase A — one-sample Stage 1 capacity

固定 sample index `0`：`step2.4-real-communication-seed0::anchor-0001`。在 `beta_KL=0` 的 posterior-assisted Stage 1 诊断中，首步→末步：Motion `0.0005155009 → 0.0000088083`（相对下降 `98.29%`）；CSI `324.4448853 → 320.2017517`（相对下降 `1.31%`）。两个 family 都达到预注册阈值。

### Phase B — one-sample prior learning

同一 sample，`beta_KL=0`、prior-dominant recursive path，H1 `L_Pred 162.1999054 → 160.2557373`（`1.20%`），H2 `164.6582031 → 162.6972046`（`1.19%`）。H2 保持 prior-only recursive state feedback，Future GT state 未作为下一步输入。

### Phase C — two-sample tiny overfit

固定 samples `0,1`：

- `step2.4-real-communication-seed0::anchor-0001`
- `step2.4-real-communication-seed0::anchor-0002`

正常 `1→2` curriculum、KL warm-up/free bits 和 `L_Total` 下，prior-only diagnostic：H1 Motion `0.0014188 → 0.0003041`（`78.57%`），H1 CSI `329.3315 → 323.7831`（`1.68%`）；H2 Motion `0.0029511 → 0.0004518`（`84.69%`），H2 CSI `332.8831 → 327.2408`（`1.69%`）。

### Learning-signal / KL / balance

Encoder、RSSM dynamics、current-observation posterior、两类 prior、两类 future posterior、两类 target encoder、Motion decoder、CSI decoder 均出现 finite gradient 并发生参数更新；receipt 同时记录每组 `min/max gradient norm` 与 `zero_gradient_steps`。部分 teacher 组有零梯度步骤，但没有持续全程饥饿。

Motion normalized target std=`0.1732458`，CSI normalized target std=`1.0431176`（Phase A；Phase C 为 `0.1613350/1.0372258`）。CSI 数值损失高于 Motion 与其 normalized target 分布和当前表达难度一致；本 Step 未发现 normalization 实现错误，不修改冻结的 `0.5/0.5` family weighting。

KL raw/adjusted Physical 与 Communication、free-bits 活动、beta、prior/posterior 轨迹均写入 Phase C JSON；无 NaN/Inf，H2 数值有限且未爆炸。

### Reproducibility / resume / validation

同 seed、同 config、同 subset 的 uninterrupted 20 steps 与 checkpoint→resume 后续 10 steps 轨迹完全一致。4 个 `dev_validation` sample 仅运行 prior-only `eval()+no_grad()` diagnostic，无参数更新，Future posterior teacher 未调用；不用于 tiny-overfit gate。

## Go / No-Go

`GO`。机器 receipt 的所有 required checks 为 true：finite/no-NaN、module learning signal、no persistent gradient starvation、Phase A 双 family、Phase B H1/H2、Phase C 双 sample 双 family、recursive H2、resume trajectory、leakage isolation、validation prior-only、locked_test 未访问。

GO 仅表示当前 CPU development bundle 上的 bounded capacity/optimization preflight 通过；不表示正式训练已完成或模型具有泛化/性能优势。

## Known Issues / Scope

仍保持：`full_training=false`、`gpu=false`、`formal_dataset=false`、`locked_test=false`、`baseline=false`、`planner=false`、`performance_claim=false`。Route non-empty 与 Comp non-empty coverage 仍为 `0/0`，是未来 formal training/data coverage gate。Tiny-data overfit 只覆盖固定 development samples，不替代正式数据评估。

## Validation / Git

Focused 5.3、5.2 regression、5.1D、5.1B、4.4、必要 upstream、compileall、knowledge index write/check、git diff --check 均在提交前执行并记录；receipt/manifest 位于 `code/artifacts/protocols/pi_jwm_step5_3_cpu_training_preflight_v1_20260922/`。下一步只建议研究者另行授权 `STEP 5.4 — GPU Training Readiness / Formal Training Preparation`，本 Step 完成后停止。
