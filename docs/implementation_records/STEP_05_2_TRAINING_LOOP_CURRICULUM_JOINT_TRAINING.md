# STEP 5.2 — Training Loop / Curriculum / Joint Training Implementation

日期：2026-09-22
状态：**COMPLETE / FROZEN FOR CPU DEVELOPMENT TRAINING-LOOP INTEGRATION**

## Step Goal

把已经验收的 5.1A Future Motion/CSI target、5.1B posterior/loss/KL/metric、5.1C unified development bundle、4.3A/4.3B graph/encoder 和 4.4 Structured RSSM 接成可执行的 CPU 训练循环。范围只包括少量 optimizer smoke、prior-only validation、checkpoint save/load/resume 和机器证据；不包括正式训练、GPU、baseline、Planner、locked_test 或 formal Dataset。

## Definition Basis

- `docs/contracts_PIJWM_DEFINITION_05_LOSS_TRAINING_EVALUATION_V1.md`：Decision 1–10，尤其是 `L_Total=L_Pred+beta_KL*L_KL`、family mask-normalized MSE、analytic diagonal Gaussian KL/free bits、posterior-assisted warm-up、prior-dominant recursive curriculum、prior-only validation、`argmin L_Val` selector 和 joint trainable modules。
- 只读研究定义 `D:\shen\OB\科研\PIJWM\05模型训练、Loss与评价.md` 仅作为目标依据，本任务未修改私人目录。
- 当前数据仅为 12-sample unified non-locked development bundle：`dev_train=8`、`dev_validation=4`；其 normalization statistics 仍由 `dev_train` lineage 提供。

## Initial State

- `HEAD=17697ee1904a96faee2862c2787e5e8b4442bc18`，branch=`main`，工作区只有用户已有未跟踪 `TASK/`。
- 5.1D 已完成 CPU paired integration，但 source 中没有 joint optimizer loop、curriculum scheduler、prior-only validation selector 或 checkpoint/resume implementation。
- Route/Comp non-empty development coverage 均为 0；Comm=1、Mobility=48；Route/Comp 只允许显式 no-op。

## Files Involved

- 新增 `code/src/pi_jwm/step5_2_training_loop_v1.py`：配置化 curriculum/KL schedule、unified bundle loader、Stage 1/Stage 2 rollout、Definition 05 loss/KL、optimizer audit、prior-only validation、L_Val selector/early stopping state、checkpoint/resume、receipt validator。
- 新增 `code/scripts/run_step5_2_training_loop_smoke_v1.py`：两步 CPU smoke 和 machine-readable receipt/artifact builder。
- 新增 `code/tests/test_step5_2_training_loop_v1.py`：9 个 focused tests。
- 修改 `code/src/pi_jwm/step4_4_structured_rssm_world_model_v1.py`：增加 `initialize_latent(..., posterior_mode="prior")`，使 Stage 2/validation 的初始 stochastic state 不评估 posterior。
- 新增 artifact：`code/artifacts/protocols/pi_jwm_step5_2_training_loop_v1_20260922/`。checkpoint 为 local-only binary；compact JSON evidence 纳入 Git。

## Changes

### Stage 1 — posterior-assisted warm-up

`Future Motion Target → motion TargetEncoder → q^Phy`，`Future CSI Target → CSI TargetEncoder → q^Comm`。q 的 mean 用于 prediction，q 与 actual prior 做 KL；Future full graph、Flow、Task、DAG、lifecycle 未进入 teacher。target 不进入 prior path。

### Stage 2 — prior-dominant recursive curriculum

`initialize_latent(posterior_mode="prior")` 只读取当前 `Z_t` 和 prior；每一步执行 `one_step → learned head → deterministic/rule transition → rebuild graph`，下一 horizon 使用上一步 predicted state。Future GT 只进入 Motion/CSI loss 与 training-only KL teacher，不生成 rollout state。配置的 curriculum 为 `1→2→4`，当前 `L=2` 自动得到 `1→2`。

### Loss / KL / schedule

- Motion 和 CSI 各自先做 mask-normalized MSE，再按 `0.5/0.5` 组合。
- KL 使用现有 5.1B analytic diagonal-Gaussian primitive、per-dimension free bits；Physical 只计有效 Vehicle stochastic slots，Comm 只计 valid/present/wireless/CSI-valid relations，wired 不计入。
- `beta_KL`、warm-up steps 和 free-bits 在 `KLSchedule` 中配置，loss primitive 不读取 epoch 偷改；每个 smoke step 记录 beta、raw/adjusted Phy/Comm KL 字段。

### Joint optimizer / fixed rules

optimizer groups 机器审计覆盖 Encoder、RSSM deterministic dynamics、`phy_prior`、`comm_prior`、future Physical/Communication posterior、Motion/CSI target encoders、Vehicle Motion/CSI decoders。normalization stats、mask/vocab/identity contract、UAV rule、SINR/rate/outage、Flow/Task/DAG/lifecycle transitions 没有 learnable optimizer parameters。

### Validation / checkpoint

Validation 用 `eval()+no_grad()` 和 prior recursive rollout；posterior teacher 不被用于 rollout。每个 horizon 计算 `L_Mot,k`、`L_CSI,k`、`L_Pred,k`，`L_Val` 只对有有效两族 target 的 horizon 做均值；selector 与 early stopping 只读取 `L_Val`，KL 仅作 diagnostic。checkpoint 保存 model/encoder/teacher、optimizer、stage/step/epoch、beta/curriculum、best `L_Val`、early-stopping state、config、RNG、data identity、git commit 和 normalization provenance。

## Reuse

- 复用 4.3A graph builder、4.3B encoder、4.4 `one_step`/rule feedback、5.1A target tensor 和 5.1B `TargetEncoder`/`FuturePosterior`/KL/MSE 原语。
- 未复用历史 P4/R6/v8 training runner 的 NLL、KL balancing、overshooting、staged-freeze、旧数据或 GPU 入口。
- `5.1D build_state/build_action` 是当前 unified paired action/state adapter 的唯一来源；本 Step 未伪造 Route/Comp action。

## Validation

实际执行：

- `python -m unittest discover -s code/tests -p 'test_step5_2_training_loop_v1.py'` → **9/9 OK**。
- `python code/scripts/run_step5_2_training_loop_smoke_v1.py` → receipt `passed=true`；Stage 1/2 各一 CPU optimizer step，validation 4 samples prior-only，`L_Val=168.31609344482422`（仅 smoke diagnostic）。
- STEP 5.1D 5/5、5.1B 5/5、5.1A 19/19、4.4 30/30、4.3B 21/21、4.3A 15/15。
- upstream：4.2C-C 23/23、4.2A 17/17、3.3 8/8、3.2 11/11。
- `python -m compileall -q .\code\src .\code\scripts .\code\tests` → exit 0。
- receipt required-check tamper 与 forbidden-scope tamper 均使 validator 返回 `passed=false`。
- full historical suite：`1887 tests, 33 errors`；错误来自既有 AirFogSim GBK 输出、缺失 archived artifact/fixture drift 和旧 runner 参数边界，未出现在上述 STEP 5.2 或相关 current regressions；不声明全量通过。

## Results

machine receipt：`code/artifacts/protocols/pi_jwm_step5_2_training_loop_v1_20260922/acceptance_receipt.json`，20/20 required checks true；manifest、config、optimizer audit、validation audit、checkpoint audit、prior-target isolation audit 均列在同目录并作为 compact GitHub evidence。实际结果包括：

- Stage 1 posterior teacher：通过；Stage 2 prior recursive：通过。
- curriculum：`1→2`；KL beta smoke 为 `0.0→0.1`（target=1.0、warm-up=10）。
- optimizer：required groups 全部存在，known-rule parameter count=0；两步后 trainable parameters 均有真实更新且 gradient finite。
- validation：4 个 validation samples，horizon 1/2 均有有效 Motion/CSI target，prior-only，`parameter_changed_count=0`，selector metric=`L_Val`。
- checkpoint：save→reload 同输入 forward digest 一致；step、epoch、beta、curriculum、best `L_Val` 和 early-stopping counter 保留。
- prior-target isolation：改变 Future target 不改变 prior mean/log_std，改变对应 posterior mean；Future GT state 未注入 rollout。
- Route non-empty=0、Comp non-empty=0、Comm=1、Mobility=48，Route/Comp 仍是 future formal training/data coverage gate。

## Expected vs Actual

预期是获得可配置、可验证的 CPU training-loop implementation，而不是训练收敛或性能结果。实际完全达到该范围：loop、optimizer smoke、validation、checkpoint/resume 和 machine evidence 均通过；没有进行 GPU、formal training、tiny-data overfit、baseline、Planner 或 locked_test。

## Known Issues / Boundaries

- 本 Step 的两步 optimizer smoke 不是 tiny-data overfit、收敛、泛化或预测性能证明；STEP 5.3 才做 CPU Preflight/overfit/Go-No-Go。
- Route/Comp non-empty adapter 尚未由真实 development sample 覆盖；不得声称四动作族完整正式训练已验证。
- full training、GPU、formal Dataset、locked_test、baseline、Planner 和 performance claim 均保持 false。
- 当前 graph topology、model dimensions 和 12-sample bundle 仍是 development scope；不据此开放正式训练预算。

## Git

本记录与代码、tests、receipt、AI_CONTEXT 和 tracker 一起提交；提交前必须重新运行 focused/regression、compileall、knowledge-index write/check、diff check，确认 `HEAD == origin/main` 且除研究者已有 `TASK/` 外工作区干净。

## Next Step

唯一建议：`STEP 5.3 — CPU Training Preflight / Tiny-Data Overfit / Go-No-Go`。本记录完成后停止，不自动执行 5.3、GPU 或 formal training。
