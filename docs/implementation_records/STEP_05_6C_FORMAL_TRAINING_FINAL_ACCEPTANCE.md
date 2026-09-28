# STEP 5.6C — Formal Training Final Acceptance & Best Checkpoint Freeze

## Step Goal and Definition Basis

Accept the single completed Formal Training v1 run from local evidence and freeze the strict `argmin L_Val` checkpoint identity. Definition basis: read-only `D:/shen/OB/科研/PIJWM/05模型训练、Loss与评价.md` (SHA-256 `62eebb05e2eeefe9edb0038f12964f59915a7acce03a683d6c7e5c71f85684e9`) and researcher-approved `formal_training_config_v1.json`. This Step performs CPU-only verification and one bounded inference; it does not retrain or evaluate a Planner, baseline, or locked test.

## Initial State and Files Involved

Git start: `main=origin/main=486dec20cf4436311ea7c2cf6c2f250ff75ae921`. The run source is the earlier Git commit `6e15ec2da0e3a6e0561dc821d0aaef90696a2387`, an ancestor of current main; the relevant 5.2/5.4/5.5/5.6A/5.6B training modules are unchanged between the two commits. Local-only run directory: `code/artifacts/formal_training/pi_jwm_formal_train_v1_seed5601_20260924T112424Z/`. Source Dataset manifest SHA-256 `6392a08b31340812463be9e3f5f78891d39933c54d50c71cca1984b3bf9448bc`; frozen config byte SHA-256 `a806c320f1d238a3997a94ca74af5641169a512a9551af7f3ff04c0ee2447660`.

The user-owned untracked `TASK/` and `code/scripts/plot_step5_3e_tiny_overfit.py` were not modified or staged. No SSH or GPU was used in this Step.

## Changes and Reuse

- Added `code/scripts/accept_step5_6c_formal_training_v1.py` with exact run/Dataset/config identity checks, all 5520 ordered training rows, frozen stage/curriculum/KL/LR checks, five complete prior-only validation rows, finite metrics, strict `argmin L_Val`, and best/latest checkpoint identity and tensor comparison.
- Reused `FormalTrainingInterface`, `FullFormalTrainer` and the accepted training config to reload the best checkpoint on CPU. The frozen checkpoint's `device=cuda` is verified unchanged; only the bounded replay Trainer device is translated to CPU. The existing base checkpoint loader still validates architecture, Dataset and normalization identities. One validation window is evaluated through H1–H4 with no parameter update or Future Posterior teacher.
- Added focused negative fixtures for missing training step, wrong curriculum, NaN, incomplete/leaking validation, wrong `L_Val`, wrong checkpoint config/Dataset identity, and changed model tensor.
- Produced tracked `final_checkpoint_manifest.json`, `formal_validation_observation.json`, and `final_acceptance_receipt.json` under `code/artifacts/manifests/pi_jwm_step5_6c_final_acceptance_20260928/`. The large run metrics and checkpoint files remain local-only.

## Validation and Results

The local heartbeat and progress agree: `COMPLETED`, process not alive, `5520/5520`, five validations, no NaN/Inf and no failure receipt. Exactly 5520 sequential log rows have global steps `0..5519` and completed steps `1..5520`. Complete 1104-window prior-only validations occurred at steps `1104, 2208, 3312, 4416, 5520`; each reports zero future teacher/target-encoder calls and no parameter update. The logged `L_Val` recomputes from H1–H4 `L_Pred` for each pass.

| Completed step | `L_Val` (Formal Validation Observation) |
| ---: | ---: |
| 1104 | 0.17661245681400606 |
| 2208 | 0.08010126911119236 |
| 3312 | 0.07754632086844497 |
| 4416 | 0.07646087923042731 |
| 5520 | **0.07431338784170399** |

Final step 5520 is the unique strict minimum. Both `best.pt` and `latest.pt` contain `global_step=5520`, `epoch=10`, `validation_count=5`, `best_l_val=0.07431338784170399`, the same Dataset/config/source identities, and 428 model tensors equal element by element. Their complete serialized file hashes differ, as expected for separate saves. `best.pt` SHA-256 is `941ee94131d406de914a79aeda43e929c92727631263d85221615422146a32c9`; `latest.pt` SHA-256 is `a642b4654feb267caf9f711f234830416cfed30b10aa181adea15e0e7932d53f`.

The bounded CPU smoke reloaded best, evaluated one validation sample at H1–H4 prior-only, produced finite rows, made no Future Posterior teacher/target-encoder calls, and left model parameters unchanged. All five per-horizon `L_Pred`, Motion raw MAE/RMSE and CSI raw MAE/RMSE rows are in `formal_validation_observation.json`. Motion raw aggregate mixes physical units; CSI raw errors are dB.

## Expected vs Actual, Known Issues, Git, Next Step

Expected and actual agree: `STEP 5.6B=COMPLETE`, `FORMAL_BEST_CHECKPOINT=FROZEN`, and local final acceptance passes. This is **Formal Validation Observation**, not test performance, a SOTA result, a generality claim, or a closed-loop system claim. There is one formal seed, no baseline and no locked-test result. Existing STEP 6.0A–C CPU Planner contracts remain separate; no Planner rollout or execution was performed here. The tracked SHA manifest freezes local checkpoint identity; local-only checkpoint bytes must be retained with it.

Validation commands and outputs: `python code/scripts/accept_step5_6c_formal_training_v1.py` returned `passed=true`; `python -m unittest discover -s code/tests -p 'test_step5_6*.py'` passed 10/10 CPU/static tests; `python -m compileall -q code/src code/scripts code/tests` passed; `python code/scripts/build_project_knowledge_index_v1.py` and `--check` both returned `passed=true` with zero mismatches. Context Consistency Check compared current state with experiments, decisions, known issues, data flow, architecture and module map. Historical 6.0A–C status text in `AI_CONTEXT/00_PROJECT_STATE.md` is explicitly marked as historical. The first index attempt collided with a temporary test directory; after tests ended, the next attempt identified that completed STEP 5.6B cannot remain in the deferred-work registry. It was removed from that queue and recorded in the experiment registry, then index write/check passed. Git diff check and commit/push are the final gate. The single next action after this Step is researcher review of the accepted checkpoint and validation evidence; no Planner or baseline starts automatically.
