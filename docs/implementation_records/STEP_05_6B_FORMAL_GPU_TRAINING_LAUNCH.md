# STEP 5.6B — Formal GPU Training Launch

## Step Goal and Definition Basis

The researcher authorized one detached RTX 4090 formal run using the already frozen Formal Training Config v1, with a stop immediately after sustained launch evidence. The scientific basis is the approved Definition 05 decisions in `AI_CONTEXT/06_DECISIONS.md`, accepted Formal Dataset v1 manifest SHA-256 `6392a08b31340812463be9e3f5f78891d39933c54d50c71cca1984b3bf9448bc`, and the frozen config in `code/artifacts/manifests/pi_jwm_step5_6a_formal_config_v1_20260924/`. The read-only target Definition 05 remains unchanged.

## Initial State and Files Involved

Initial Git HEAD was `10d2aa5cd828d627e72ad3017c4ee93447c98bc9`. Formal Dataset 4416/1104, full shard loader, RTX 4090 CUDA smoke, complete 1104-window prior-only validation, and formal numeric config had already passed; formal training had not begun. The user-owned untracked `TASK/` and `code/scripts/plot_step5_3e_tiny_overfit.py` remain outside this Step.

Added `code/src/pi_jwm/step5_6b_formal_runner_v1.py`, the thin CLI `code/scripts/run_step5_6b_formal_training_v1.py`, and focused CPU bookkeeping tests. The runner reuses `FormalTrainingInterface → FullFormalShardDataset → FullFormalTrainer` and the frozen Step 5.2 `train_step`, validation batch computation, loss aggregation, checkpoint and resume methods. No Dataset package, model architecture, loss or split changed.

## Changes and Reuse

- Verify the frozen config JSON against the programmatic config, Dataset manifest and all five package hashes before constructing a trainer.
- Require the exact config JSON byte SHA from the accepted freeze receipt. A Git archive normalizes that tracked JSON's CRLF bytes to LF; field equivalence alone is insufficient for the frozen artifact identity.
- Use the frozen 5601/8/552/5520 schedule. Each step records the actual stage, horizon, KL beta, family/KL losses, gradient norm, learning rate, elapsed time and ETA. The sampler remains the accepted exact-once trajectory sampler.
- Run complete 1104-window H1–H4 prior-only validation every 1104 completed steps, reuse the official per-family loss aggregation, and record raw Motion/CSI MAE/RMSE. Reject duplicate/missing windows, future posterior teacher calls, nonfinite `L_Val` or parameter changes.
- Save `latest.pt` every 552 completed steps and after validation; save `best.pt` only on strict `L_Val` improvement. Existing checkpoint schema carries model, optimizer, RNG, progress and architecture/data identity; additive runner state binds formal-config SHA and source Git SHA. Resume requires the same run directory and matching identities.
- Atomically replace `heartbeat.json` and `progress.json` every optimizer step and validation batch. Runtime logs and checkpoints are remote local-only; none are committed.

## ETA and Go/No-Go

MEASURED in STEP 5.6A: H4 batch-8 optimizer step `25.026449 s`; complete 1104-window validation `7103.09 s`. ESTIMATED: scale H1/H2 training linearly with horizon because no corresponding full formal timing was measured. This gives training `103609.5 s` (28.78 h), up to five validations `35515.45 s` (9.87 h), and total `139124.95 s` (38.65 h), excluding startup, checkpoint I/O and interruptions. Early stopping may shorten the run. These estimates are operational planning, not results.

An independent launch Go/No-Go checks the accepted dataset identity/package hashes, frozen config, 5.6A CUDA/full validation receipts, focused regression/compile/index checks, remote RTX 4090, free disk and source archive identity before launch. The separate receipt under `code/artifacts/manifests/pi_jwm_step5_6b_launch_20260924/` is the prelaunch evidence. A source commit/push precedes the run; the remote `run_manifest.json` and live heartbeat establish launch status, not this prelaunch document.

## Validation, Results, Expected vs Actual, Known Issues, Git, Next Step

Focused CPU tests: new runner 2/2; formal config 3/3; Step 5.2 13/13; Step 5.4 5/5; Step 5.5 11/11; Step 5.5-PATCH 10/10; Step 5.6A validation merge 1/1. `python -m compileall -q code/src code/scripts code/tests` passed. A mistaken filename pattern found 0 tests, then the real `test_step5_5_patch_full_consumption_v1.py` suite was run and passed 10/10. Knowledge index write/check and `git diff --check` are final commit gates. At the source commit stage, launch is intentionally pending. Once detached, the run must show a live PID, GPU process, `RUNNING` heartbeat and at least 2–3 recorded optimizer steps; then Codex stops. No final performance result, `locked_test`, baseline or Planner is authorized. Source commit hash and push are read from Git; the runtime manifest binds that exact SHA.

The only next action after confirmed launch is waiting and monitoring this run. No new Step starts automatically.

## Prelaunch identity incident

The first detached attempt used the exact committed code archive but its Git-normalized config JSON had SHA `7c4358df233daa345cfc2a3b2959a83175aff175e8c993372e997d176dab356d`, differing from frozen receipt SHA `a806c320f1d238a3997a94ca74af5641169a512a9551af7f3ff04c0ee2447660`. The fields agreed, but the byte identity did not. The attempt was terminated after 1 optimizer step and marked `FAILED`; it must not be resumed or presented as the formal sustained run. The original frozen config bytes were transferred separately and verified on the server. The runner now enforces the receipt byte hash before creating a Trainer. A new source commit and new run ID are required for the accepted launch.

## Interim process visualization (2026-09-25)

The accepted active run has ID `pi_jwm_formal_train_v1_seed5601_20260924T112424Z` and source Git SHA `6e15ec2da0e3a6e0561dc821d0aaef90696a2387`. A read-only copy of its live JSONL logs at 13:35 UTC contained 2646 consecutive optimizer steps and full prior-only validations at steps 1104 and 2208. `code/scripts/plot_step5_6b_training_progress_v1.py` renders training losses and validation loss/raw-error curves; `docs/figures/step5_6b_live_progress_20260925/` stores the PNGs and source/plot SHA receipt. The training process and original logs were not modified. This is a running-process diagnostic, not final result acceptance; future snapshot updates must use fresh log copies and retain this provenance.
