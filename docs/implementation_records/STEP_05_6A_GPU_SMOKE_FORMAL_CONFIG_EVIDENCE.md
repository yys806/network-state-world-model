# STEP 5.6A — GPU Smoke + Formal Training Config Evidence/Freeze

> 本记录保留 GPU smoke/full validation 收口时的历史配置待决快照。数值配置已在后续 `STEP 5.6A-CONFIG-FREEZE` 记录中由研究者明确冻结；当前正式配置以 `code/artifacts/manifests/pi_jwm_step5_6a_formal_config_v1_20260924/` 为准。

## Step Goal and Definition Basis

Use the accepted Formal Dataset v1 for a bounded CUDA smoke and full prior-only validation. Verify the deterministic trajectory sampler and distinguish frozen Definition 05 method choices from development numerical defaults. This step does not authorize formal training. Basis: `AI_CONTEXT/06_DECISIONS.md` STEP 5.0 and STEP 5.5 decisions, Formal Dataset manifest SHA-256 `6392a08b31340812463be9e3f5f78891d39933c54d50c71cca1984b3bf9448bc`, and STEP 5.5-PATCH audit/readiness receipts.

## Initial State

`main@a8c5dfacd6d1ea0ef5414934e5cbf4e8a2665725`; original user-owned untracked `TASK/` and `code/scripts/plot_step5_3e_tiny_overfit.py` were left untouched. Formal Dataset H=2/L=4, 60 trajectories, train/validation 48/12, windows 4416/1104. Full-shard CPU path was accepted; GPU execution and formal config were open.

## Files and Reuse

- Reuse `FormalTrainingInterface`, `FullFormalShardDataset`, `FullFormalTrainer`, Step 5.2 training loop and Definition 05 loss/metric semantics.
- `step5_5_full_sharded_loader_v1.py`: deterministic trajectory-aware exact-once train sampler. Only batch-needed shards are materialized; resume order derives from seed and global step.
- `build_step5_1d_unified_model_chain_v1.py`: preserve fixed current Flow support when a smoke-model prediction prematurely marks a Flow inactive; future-only Return Comm rows without a current slot become explicit blocked components. No future target enters model input and no Flow is created.
- `step5_2_training_loop_v1.py`: restore CPU byte tensors for CPU/CUDA RNG state after CUDA checkpoint load.
- `run_step5_6a_gpu_smoke_v1.py`: bounded H=4 CUDA optimizer probes, checkpoint identity checks, and complete prior-only GPU validation with raw Motion/CSI metrics. The 12 validation trajectories can be partitioned into four disjoint groups on the same GPU; aggregation reuses the frozen per-family numerator/count rule and rejects duplicate/missing sample IDs.
- `audit_step5_6a_formal_fixed_support_v1.py`: read-only 5520-window structural cross-check. The earlier STEP 5.5-PATCH audit remains authoritative for the known 8828 events.

The accepted Dataset packages, split, normalization and model/loss definitions were not modified. A local Raw-only audit checks all 60 frozen wired edges against the source constant used for portable Training Bundle consumption.

## Validation and Results

Machine receipts: `code/artifacts/audit/pi_jwm_step5_6a_20260923/`. Remote source is an isolated snapshot of `main@a8c5dfa` plus the listed local code changes. The initial remote environment receipt records the smoke source SHA-256 match, GPU, PyTorch/CUDA and disk space. `remote_source_final_receipt.json` separately verifies all four final synced source files after diagnostic/merge/reload additions; validation groups ran the same scientific validation function with successive diagnostic script revisions.

- RTX 4090, PyTorch 2.8.0+cu128, CUDA available. Training Bundle package hashes and manifest identity match the accepted Formal Dataset. No Raw was sent to the server.
- CUDA H=4 batch 1/2/4/8 forward, backward and optimizer step passed with finite loss/gradients and changed parameters. Cross-trajectory batch, all optimizer groups on CUDA, checkpoint save/load, wrong Dataset/config rejection passed. The smoke checkpoint is a transient few-step artifact, not a formally trained model.
- Peak allocated VRAM for batch 1/2/4/8: 33,851,392 / 52,718,080 / 85,287,936 / 148,032,512 bytes. Step wall time: 7.57 / 6.68 / 13.11 / 25.03 s; first shard load 3.75 s, warm batch-8 shard load 0.054 s. CUDA event intervals: 3.81 / 6.65 / 13.07 / 24.95 s. These intervals span host dispatch and kernels; pure kernel time was not established. A one-step PyTorch CPU/CUDA profiler probe exceeded 4 minutes at one busy CPU core and was stopped without a receipt; it is excluded from acceptance and no kernel-time claim is made. Low sampled GPU utilization and only 1.02% validation time in shard loading identify CPU dispatch as a likely bottleneck, not a measured kernel/CPU decomposition. More batch capacity alone is not proof of a scientifically preferable batch.
- Deterministic sampler: train-only 4416 indices, each once per epoch, trajectory/window shuffle, 48 bounded trajectory groups, seed + global-step reproducibility. Validation indices are excluded.
- Full GPU validation: **PASS**. Four disjoint groups each covered 276 windows/3 trajectories; merged receipt verifies all 1104 unique validation windows and 12 trajectories against the frozen split. Prior-only H1–H4, 0 future-posterior/target-encoder calls, no parameter change. The untrained smoke model's `L_Val=0.8297511641582647` is a runtime diagnostic, not a performance result. H1–H4 Motion raw MAE: 1.7783/1.8754/1.9561/2.0146; Motion raw RMSE: 3.8468/3.9210/3.9940/4.0599. CSI raw MAE: 6.0480/6.0296/6.0204/6.0177; CSI raw RMSE: 7.5473/7.5225/7.5111/7.5062. Motion valid elements: 20,600 per horizon; CSI: 6,179,400 per horizon. Four groups ran serially on one GPU: worker wall sums to 7103.09 s (0.1554 windows/s); data loading sums to 72.15 s (1.02%). Largest validation worker peak allocation was 96,548,352 bytes. The elapsed sum excludes the user's overnight pause and transfer time.
- A separate H4 two-trajectory checkpoint forward check reloaded the exact same smoke checkpoint twice. Parameter digest and state matched exactly; predictions matched at `rtol=atol=1e-6`. GPU floating-point results were not bitwise identical: maximum absolute Motion delta 8.20e-8 and CSI delta 7.63e-6. No future teacher was called. This supports numerical reload compatibility, not bitwise CUDA determinism.
- Read-only future Return re-audit: 8828 unsupported/fixed-support component events, 0 unresolved, 2901 affected windows across 5520; matches STEP 5.5-PATCH and does not change the accepted packages.

Final local verification used `PYTHONPATH=code/src;code/scripts` and `python -m unittest discover -s code/tests -p ...` for Step 5.2 (12/12), 5.4 (5/5), 5.5 (11/11), 5.5-PATCH (10/10), 5.1D (5/5), and 5.6A merge (1/1). The merge test first hit an external Windows Temp-directory disappearance during its slow second merge; its fixture was moved to the repository's ignored audit temp directory and the fresh rerun passed. `python -m compileall -q code/src code/scripts code/tests`, knowledge index write and `--check` (`mismatches=[]`, five outputs), `git diff --check`, and a manual Context Consistency Check across `AI_CONTEXT/00–08`, source receipts, tracker and registries passed. The readiness acceptance command reported all CUDA/validation checks true, `SAMPLER_FORMAL_PATH=VERIFIED`, `DATASET_IDENTITY=MATCH`, and `passed=true`.

## Expected vs Actual and Known Issues

The full-shard GPU smoke path works after two engineering closures revealed by real data: portable wired capacity and action mapping when prior rollout support differs from the recorded continuation. The latter remains bounded to current typed support or explicit future Return fixed-support blocking. The Dataset itself remains unchanged. Full validation has passed; the remaining boundary is the researcher's numerical formal-training configuration decision.

Definition 05 freezes the method, loss, curriculum order, prior-only validation and `argmin L_Val` selector. It does not freeze numerical training seed, batch, optimizer settings, duration, stage/curriculum timing, KL numbers, validation/checkpoint interval or patience. `formal_training_config_decision_receipt.json` therefore states `AWAITING_RESEARCHER_DECISION`; development defaults must not be used as formal values.

| Parameter requiring a formal value | Development default | GPU evidence | Engineering candidate / decision |
| --- | --- | --- | --- |
| Training and sampler seed | 5201 | Seed + epoch sampler reproduces all 4416 train windows; no multi-seed training evidence | Researcher chooses one seed; sampler uses that seed. |
| Physical batch size | 1 | CUDA H4 batch 1/2/4/8 all run; batch 8 peak allocated 148,032,512 bytes, step 25.03 s on the smoke sample | 8 is a compatible engineering candidate, not a frozen research choice. |
| Learning rate / weight decay / gradient clip | 0.001 / 0 / 1.0 | Few optimizer steps are finite; no convergence or stability run | Researcher chooses formal values. |
| Max epochs or steps, Stage-1 length, 1→2→4 start points | 1 epoch / 2 steps / 1 step / [0,1,2] | Only H4 few-step smoke; no curriculum-length evidence | Researcher chooses training budget and boundaries. |
| KL target beta / warm-up / free bits | 1.0 / 10 steps / 0.1 | Frozen KL implementation runs on CUDA; no formal schedule comparison | Researcher chooses numbers, preserving the frozen analytic family KL rule. |
| Validation interval / checkpoint patience / save interval and resume | no interval / 3 / unset | Prior-only full validation wall time is measured below; checkpoint reload and deterministic sampler resume are verified | Researcher chooses intervals and patience; selector stays frozen as `argmin L_Val`. |

## Git and Next Step

GPU acceptance receipt: `readiness_receipt.json` reports `GPU_AVAILABLE`, `CUDA_FORWARD`, `CUDA_BACKWARD`, `CUDA_OPTIMIZER_STEP`, `CUDA_CHECKPOINT_RELOAD`, `H4_GPU_RUNTIME`, `FULL_1104_GPU_VALIDATION` as true; `SAMPLER_FORMAL_PATH=VERIFIED` and `DATASET_IDENTITY=MATCH`. `FORMAL_TRAINING_READINESS=BLOCKED_BY_CONFIG_DECISION`. Git commit/push evidence is completed in the Git log; this record does not hard-code its own commit SHA. The sole next action is a short researcher decision on the remaining numerical training config; do not start STEP 5.6B without it. Boundaries: `formal_training=false`, `locked_test_accessed=false`, `baseline=false`, `planner=false`, `performance_claim=false`.
