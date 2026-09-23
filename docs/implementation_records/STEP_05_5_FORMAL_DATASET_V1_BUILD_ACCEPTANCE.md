# STEP 5.5 — Formal Dataset v1 Build & Acceptance

## Step Goal

在 CPU 范围内完成真实 AirFogSim Raw trajectory 采集、H=2/L=4 Sample/Tensor/Graph/Target/Normalization package、四动作覆盖、可移植 manifest/hash、机器验收和 `FormalTrainingInterface → Step52Trainer` 单步 smoke。明确不执行 GPU、正式训练、baseline、Planner 或 `locked_test`。

## Definition Basis

- 研究者在 STEP 5.5 指令中冻结：H=2、L=4、60 条完整 trajectory、每条 96 transitions、48/12 trajectory split、split seed 20260923、`radius_knn(1000m,k=2)`、因果 coverage-oriented 四动作采集、无 locked test。
- 延续已冻结的 History/Future Action/Future Target、Causal Flow Ledger、Typed Dual Graph、Structured RSSM、Definition 05 loss/validation 和 STEP 5.4-PATCH2 device/action adapter 合同。

## Initial State

- `main` 起点：`36a66a8f7cf8f82b428d23b1c38e79121d806afd`。
- STEP 5.4-PATCH2 的训练接口和四动作 adapter 已通过 development/synthetic 证据，但正式 Dataset 不存在，development Route/Comp coverage 为 0/0，L>2 只到接口准备层。

## Files Involved

- Raw/Package builder：`code/scripts/collect_step5_5_formal_raw_v1.py`、`build_step5_5_formal_dataset_v1.py`。
- Acceptance：`finalize_step5_5_existing_package_v1.py`、`verify_step5_5_behavior_policy_determinism_v1.py`、`verify_step5_5_deterministic_rebuild_v1.py`、`step5_5_formal_dataset_cpu_acceptance_v1.py`。
- Generic H/L 与 runtime：`code/src/pi_jwm/model_ready_sample_contract_v1.py`、Step 3.2/4.2A/4.2C/5.2/5.4 modules。
- Tests：`code/tests/test_step5_5_formal_dataset_v1.py` 及相关回归。
- Git 可追踪证据：`code/artifacts/manifests/pi_jwm_step5_5_formal_dataset_v1_20260923/`。

## Changes and Reuse

- `MINOR_MODIFICATION`：把原 H=2/L=2 minimum contract 泛化为正整数 H/L，development default 仍是 2/2。
- `STRUCTURAL_CHANGE`：正式 package 按 trajectory shard 流式构建，五类 package 与 runtime smoke package 分别具有 mandatory SHA-256；路径使用 manifest-relative/repo-relative identity。
- `DIRECT_REUSE`：复用已冻结 Raw collector、sample/tensor/Flow/graph/target/trainer 语义；行为策略只调用真实 setter/scheduler/callback，不伪造动作行。
- `MINOR_MODIFICATION`：target-only future namespace 用自身容量校验，未将未来对象加入 input index；NPZ 使用固定 member 顺序与时间戳以获得 byte-level deterministic rebuild。

## Validation and Results

- Raw：60/60 accepted，0 rejected/replacement；每条 96 transitions / 97 Decisions，连续时间网格。
- Split/window：48 train / 12 validation；4416 / 1104 / 5520 windows；trajectory/simulator seed/policy seed 跨 split 隔离。
- Coverage：Route/Comm/Comp/Mobility 的 train eligible intervention rate 分别为 38.60%/42.78%/38.52%/40.08%，validation 为 42.23%/44.87%/41.76%/41.15%；每族 train/validation intervention trajectory coverage 为 48/12。
- Package：Samples/Tensor/Graph/Target/Normalization 五类真实 package 均存在且 SHA-256 exact match；missing/wrong hash negative fixture 均拒绝；train-only normalization 排除 validation 与 Future Target。
- Acceptance：25/25 machine checks 为 true；H1-H4 Motion/CSI 均有有效监督；Future Action unresolved=0；Future Target 不进入 input；identity、serialize/reload、unsupported accounting、locked-test boundary 通过。
- Determinism：同 Raw、同 frozen split 的第二次完整 build，五类 package SHA-256 全部一致。
- Runtime：CPU one optimizer step、真实 H=4 prior-only validation、checkpoint compatible reload、错误 dataset/architecture identity rejection 均通过，`L_gt_2_runtime_verified=true`。
- Fresh final verification：STEP 5.5 focused 11/11、相关 Raw/3.x/4.x/5.x upstream regression 221/221、formal finalizer、deterministic rebuild、CPU H4 smoke 与 behavior-policy determinism 均通过；compileall、knowledge index write/check、JSON/Context Consistency 和 `git diff --check` 通过。

## Expected vs Actual

- 符合预期：正式 Dataset v1 和 CPU interface acceptance 均闭合。
- 边界保持：这不是正式训练或性能结果；GPU 只保持静态 prepared，未执行 CUDA。

## Known Issues / Blockers

- Engineering：本 Step 内无剩余 Dataset/CPU interface blocker。
- Data：Formal Dataset v1 已 READY；没有 locked test split。
- Research：正式 training seed、batch、epoch/max_steps、patience、预算仍未冻结。
- GPU Verification：未执行 GPU smoke；`GPU_TRAINING_VERIFIED=false`。

## Git

- 计划提交信息：`feat(pi-jwm): build and accept formal dataset v1`。
- commit/push hash 在提交后由 `git log` 与 `origin/main` 确认；本文不预写自身 commit hash。

## Next Step

唯一建议：研究者另行授权 **STEP 5.6A — GPU Smoke + Formal Training Config Freeze**。本 Step 不自动执行。
