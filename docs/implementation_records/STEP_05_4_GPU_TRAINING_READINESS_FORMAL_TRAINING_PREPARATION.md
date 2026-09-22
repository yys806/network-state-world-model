# STEP 5.4 — GPU Training Readiness / Formal Training Preparation

日期：2026-09-22

状态：**FORMAL TRAINING BLOCKED；GPU_CODEPATH_PREPARED；CPU-ONLY**

## 1. Step Goal

审计正式训练前置条件，并把训练入口从固定 development bundle 解耦为 manifest-driven interface。此 Step 不生成正式 Dataset、不启动 CUDA、不访问 `locked_test`。

## 2. Definition Basis and Initial State

- Definition 05 只冻结 `1 → 2 → 4 → L`，正式 `L` 尚未由研究者决定。
- 5.3E 已确认 raw CSI decoder + train-only CSI mean bias；证据仍是 CPU development tiny-overfit。
- 当前 development bundle 是 12 samples / `dev_train=8` / `dev_validation=4`，manifest 明确 `formal_dataset=false`。

## 3. Changes and Reuse

- 新增 `FormalTrainingInterface`：从 manifest 读取 sample package、split、contract、provenance 和 manifest hash；不要求 12 samples 或 8/4 split。
- 新增 CPU-only readiness script 和九类 machine-readable receipts。
- 保留 `DevelopmentBundle` 与 5.2 adapter 不变，避免破坏 development regression。
- 没有生成正式数据，没有改变理论、loss、horizon、topology 或研究参数。

## 4. Evidence

Artifact：`code/artifacts/protocols/pi_jwm_step5_4_gpu_training_readiness_v1_20260922/`。

- `readiness_receipt.json`：training stack `PASS`，formal Dataset `NOT_READY`，GPU codepath `PREPARED`，formal training `BLOCKED`。
- `action_coverage_audit.json`：Mobility non-empty=24，Comm=1，Route=0，Comp=0；Route/Comp 没有伪造补齐。
- `horizon_readiness.json`：development horizon=2，formal horizon=`null`，research decision required。
- `topology_readiness.json`：`radius_knn(radius=1000m,k=2)` 仅 development，`research_frozen=false`。
- `formal_training_config_schema.json`：manifest/split/normalization、model、optimizer、curriculum/KL、validation、runtime/checkpoint 均有字段，但未决正式值保持 `null`/`research_frozen=false`。
- `checkpoint_readiness.json`：列出 weights/optimizer/trainer/RNG/config/provenance/topology/CSI contract 等必需字段；当前 development checkpoint guard 可复用，formal resume 尚未 ready。

## 5. Validation

- `python code/scripts/step5_4_gpu_training_readiness_v1.py`：CPU dry-run 通过。
- STEP 5.4 focused：1/1；STEP 5.3E：3/3；STEP 5.2：12/12。
- `compileall`、knowledge index write/check、`git diff --check`：见提交前命令记录。

## 6. Verdict and Boundaries

- `TRAINING_STACK_READINESS=PASS`
- `FORMAL_DATASET_READINESS=NOT_READY`
- `GPU_CODEPATH_READINESS=PREPARED`（未执行 CUDA，不能写 GPU_TRAINING_VERIFIED）
- `FORMAL_TRAINING_READINESS=BLOCKED`
- `formal_dataset=false, full_training=false, gpu=false, locked_test=false, baseline=false, planner=false, performance_claim=false`

## 7. Researcher Decision Required

1. 正式 horizon `L`。
2. 正式 Physical topology 参数（当前 1000m/k=2 仅 development）。
3. 正式 Dataset 规模、seed、model/training numeric budget。
4. Route/Comp action coverage 的正式数据策略。

## 8. Known Issues / Next Step

Data blocker 是 formal Dataset 缺失且 Route/Comp non-empty coverage=0/0；research blocker 是 `L`、topology 和训练预算未冻结；GPU blocker 是尚未运行 CUDA。唯一下一动作：研究者先决定并授权 Formal Dataset 构建/冻结；本 Step 到此停止。

## STEP 5.4-PATCH Closure

- Generic interface loads and hashes samples, tensor, graph, target and normalization packages.
- `Step52Trainer.from_formal_interface()` executes the real CPU train step, prior-only validation, checkpoint reload and wrong-identity rejection.
- Device contract supports `cpu|cuda`; this patch only verifies CPU. Readiness is computed from machine checks; L=4 is a config fixture only (`L_gt_2_runtime_verified=false`).

## STEP 5.4-PATCH2 Closure

- Optimizer creation follows final-device model/data migration; encoder, state, action and padding tensors share the configured device.
- Frozen Route/Comp fields map to route task/flow/node tensors and comp agent/task/value tensors; existing no-op paths remain valid.
- Package hashes are mandatory; missing/wrong hashes fail. Four-action adapter support is separate from development coverage (Route=0, Comp=0).
