# STEP 5.2-PATCH — Initial-Latent / Validation / Resume Semantic Closure

日期：2026-09-22

状态：**COMPLETE / FROZEN FOR CPU DEVELOPMENT TRAINING-LOOP INTEGRATION**

## Step Goal

在不改变 Definition 04/05 科研设计的前提下，闭合 STEP 5.2 的当前时刻 latent 初始化、验证集损失聚合、validation posterior 隔离和 checkpoint 身份安全恢复。

## Definition Basis

- Definition 04 contract：当前 `Z_t^{PI,L_g}` 经过 current-observation posterior 得到 `z_t`；未来递推使用 prior；posterior-compatible initialization 与 prior-only future rollout 同时保留。
- Definition 05 Decision 3/6/8：Future Target 只进入 family-specific training posterior；Stage 2 future state prior recursive；Validation prior-only；`L_Val` 作为唯一 checkpoint/early-stopping selector。
- 本 Patch 未修改 latent layout、decoder、loss/KL/free-bits、deterministic rule 或数据合同。

## Initial State

STEP 5.2 原实现将 Stage 2/Validation 初始 latent 设为 `posterior_mode="prior"`，冻结 `phy_posterior/comm_posterior`，validation 以 sample-level normalized loss 平均替代全验证集 numerator/count 聚合，且 checkpoint load 未核验 data/normalization/architecture identity。

## Changes

- Stage 2/Validation 入口改为 `initialize_latent(..., posterior_mode="mean")`，即当前观测 posterior `q(z_t | h_t, Z_t)`；每个未来 step 仍只从 prior 和上一步 predicted state 递归产生。
- current-observation `phy_posterior/comm_posterior` 纳入 optimizer group 与 trainability audit；Future Target posterior 仍是独立的 training-only teacher。
- 增加 current posterior、future teacher、Future Target Encoder 的 runtime invocation counters；validation 强制 `future_posterior_teacher_calls=0`、`future_target_encoder_calls=0`，并记录允许的 current posterior calls。
- Validation 每个 horizon 分别累计 Motion/CSI squared-error numerator 与 valid-element count，再独立归一化后计算 `L_Pred,k`，最后取 horizon mean `L_Val`。
- Checkpoint 保存 architecture identity；load 时拒绝 schema、data identity、normalization provenance 或 architecture-critical config 不匹配。training-only 超参数（学习率、weight decay、batch size、clip、epoch/step、KL schedule、curriculum state）不作为架构恢复依据；stage/curriculum/beta/best `L_Val`/early-stop state 仍从 checkpoint state 恢复。
- 增加 current-posterior target-isolation probe 与实际 validation isolation audit；Future Target mutation 不改变 current posterior 或 future prior。

## Files Involved

- `code/src/pi_jwm/step5_2_training_loop_v1.py`
- `code/src/pi_jwm/step4_4_structured_rssm_world_model_v1.py`
- `code/tests/test_step5_2_training_loop_v1.py`
- `code/scripts/run_step5_2_training_loop_smoke_v1.py`
- `code/artifacts/protocols/pi_jwm_step5_2_training_loop_v1_20260922/`
- this record and synchronized context/tracker/process notes

## Validation

- Focused: `python -m unittest discover -s code/tests -p 'test_step5_2_training_loop_v1.py'` → **12/12 OK**.
- CPU smoke: `python code/scripts/run_step5_2_training_loop_smoke_v1.py` → **26/26 required checks, passed=true**.
- Runtime validation evidence: current-observation posterior calls `4`; future posterior teacher calls `0`; Future Target Encoder calls `0`; per-horizon Motion/CSI numerator/count and `L_Mot`, `L_CSI`, `L_Pred`, `L_Val` recorded.
- Checkpoint negative evidence: wrong data identity and wrong normalization provenance both rejected; compatible save→reload forward digest equal.
- Additional required regressions, compileall, knowledge-index write/check and diff check are run before commit and recorded in the completion report.

## Results / Boundaries

This closes the CPU development-loop semantic gaps. It does not establish tiny-data overfit, convergence, formal Dataset validity, performance, GPU readiness, Planner capability or `locked_test` access. Route/Comp non-empty development coverage remains `0/0`; this remains a future formal training/data coverage gate.

## Git / Next Step

Commit and push this bounded Patch only after all listed regressions and context consistency checks pass. The only suggested next action is researcher-authorized `STEP 5.3 — CPU Training Preflight / Tiny-Data Overfit / Go-No-Go`; do not execute it automatically.
