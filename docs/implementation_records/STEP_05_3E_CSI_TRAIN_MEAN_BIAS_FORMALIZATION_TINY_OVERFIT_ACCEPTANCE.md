# STEP 5.3E — CSI Train-Mean Bias Formalization / Tiny-Overfit Acceptance

日期：2026-09-22

状态：**FORMALIZATION_PASS / TINY_OVERFIT_GO；COMPLETE FOR CPU DEVELOPMENT EVIDENCE**

## Step Goal

将研究者确认的 `raw CSI decoder + train-only CSI mean bias initialization` 接入正常 STEP 5.2 Trainer，并在固定 `[0,1]` `dev_train` tiny subset 上完成受限 CPU tiny-overfit 验收。

## Definition Basis and Boundary

- Definition 05 Decision 1/2/6/9/10：raw decoder、normalized-space family MSE、Stage 1→Stage 2、1→2 curriculum、KL warm-up/free-bits、joint training、prior-only recursive evaluation。
- CPU only；固定 samples `[0,1]`；200 steps；不访问 validation 做参数更新、GPU、formal Dataset、locked_test、baseline、Planner 或性能声明。

## Changes

- `Step52TrainingConfig` 增加并校验 `csi_decoder_output_space="raw_db"` 与 `csi_decoder_bias_init_policy="train_only_csi_mean"`。
- `Step52Trainer` 在 optimizer 创建前从 frozen `dev_train` target normalization stats 初始化 CSI decoder 最后一层全部 RB bias；不改变 Motion decoder 初始化。
- checkpoint 保存并严格校验初始化 contract、resolved mean 和 provenance；compatible reload 覆盖初始化权重，wrong contract 拒绝恢复。
- Structured RSSM contract 显式声明 `csi_decoder_output_space=raw_db`。

## Evidence

Artifact：`code/artifacts/protocols/pi_jwm_step5_3e_csi_train_mean_bias_formalization_v1_20260922/`。

- resolved CSI train mean：`98.34974797337962 dB`；source split=`dev_train`；validation/Future Target/locked_test 均未参与。
- H1 Motion：`0.0014187975 → 0.0002258055`，relative drop `84.08%`；H1 CSI：`1.0934864 → 0.1405199`，relative drop `87.15%`。
- H2 Motion：`0.0029511414 → 0.0002239104`，relative drop `92.41%`；H2 CSI：`1.1040019 → 0.1633186`，relative drop `85.21%`。
- final normalized family MSE 均小于 `1.0`；raw-unit Motion 分量与 CSI dB MAE/RMSE、KL、gradient audit、leakage、reproducibility、checkpoint audit 均写入 JSON。

## Validation

- STEP 5.3E focused：3/3；STEP 5.2：12/12；STEP 5.3：4/4；STEP 5.3D：2/2；STEP 5.1D：5/5；STEP 5.1B：5/5；STEP 4.4：30/30。
- `compileall`、knowledge index write/check、`git diff --check` 与 Context Consistency Check 在提交前执行。

## Results and Limits

`FORMALIZATION_PASS` 与 `TINY_OVERFIT_GO` 仅证明当前 development tiny-data 的初始化契约、CPU 优化路径和容量诊断通过；不证明 formal training、泛化、benchmark performance、最终模型质量或完整 Route/Comp 动作族训练。Route/Comp non-empty coverage 仍为 `0/0`，是未来 formal data coverage gate。

## Git and Next Step

本记录与源码、tests、artifact、context 同一提交。`full_training=false`、`gpu=false`、`formal_dataset=false`、`locked_test=false`、`baseline=false`、`planner=false`、`performance_claim=false`。下一步仅建议研究者审阅后决定是否授权 STEP 5.4；本 Step 不自动进入。
