# STEP 5.6A-CONFIG-FREEZE — Formal Training Config v1

## Goal and boundary

The researcher explicitly fixed the numerical Formal Training Config v1. This CPU/static step materializes that decision and repairs validation availability bookkeeping. It does not access GPU/SSH, start formal training, rebuild data, change Dataset identity/split, change model architecture or Definition 05 loss semantics, access `locked_test`, run a baseline or run Planner.

## Frozen config

Source artifact: `code/artifacts/manifests/pi_jwm_step5_6a_formal_config_v1_20260924/formal_training_config_v1.json`; freeze receipt: `config_freeze_receipt.json`.

- Dataset: H=2, L=4; 4416 train windows; 1104 validation windows; manifest SHA-256 `6392a08b31340812463be9e3f5f78891d39933c54d50c71cca1984b3bf9448bc`.
- AdamW: seed 5601, batch 8, 552 steps/epoch, learning rate 3e-4, constant schedule, weight decay 0, betas (0.9, 0.999), eps 1e-8, gradient clip 1.0.
- Budget: 10 epochs, 5520 steps; Stage 1 steps 0–551 is posterior-assisted H1, from step 552 prior-dominant; curriculum H1 at 0, H2 at 1104, H4 at 2208.
- KL: target beta 1.0, linear warmup 1104 steps, free bits 0.1 per latent dimension, overshooting off.
- Validation: full 1104 windows, prior-only H1–H4, interval 1104 steps; selector `argmin L_Val`.
- Checkpoint: latest every 552 completed steps; best on strict `L_Val` improvement; patience 3 full validations; resume restores model, optimizer, RNG, progress, selector state and curriculum state; sampler order derives from seed/global step. FP32 only, no AMP.

The independent source constructs the existing `Step52TrainingConfig` without changing its development defaults or historical checkpoint schema. PyTorch's effective AdamW defaults are queried and recorded, with no optimizer step.

## Bookkeeping correction

The old merged GPU receipt remains unchanged at `code/artifacts/audit/pi_jwm_step5_6a_20260923/full_gpu_validation_receipt.json`. Its `available_sample_count` was zero because batch rows only emitted `available` and the merge stage re-aggregated already aggregated rows. The corrected implementation carries `available_sample_count` and sums it without changing official numerators, denominators, `L_Val`, or checkpoint selection.

CPU recomputation from the frozen validation target shards uses the required definition: for each horizon, count validation windows with at least one valid Motion target and at least one valid CSI target. Corrected counts are `[1104, 1104, 1104, 1104]`; original counts were `[0, 0, 0, 0]`; `L_Val` remains `0.8297511641582647`. Receipt: `full_gpu_validation_bookkeeping_correction_receipt.json`. It records `no_gpu_rerun=true` and `bookkeeping_only=true`.

## Verification

CPU config tests: 3/3; Step 5.2 tests after the aggregation change: 13/13. The new merge fixture covers batch-style aggregated counts, re-aggregation, and unchanged official loss. No GPU process or SSH was used in this step.

## Readiness

`FORMAL_TRAINING_CONFIG=FROZEN`; `FORMAL_TRAINING_READINESS=READY_TO_START`; `formal_training=false`; `gpu_training_verified=false`; `locked_test_accessed=false`; `baseline=false`; `planner=false`; `performance_claim=false`. The only next action is separately authorized **STEP 5.6B — Formal GPU Training**.
