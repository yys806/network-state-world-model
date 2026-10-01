# STEP 6.3D — FORMAL TRAIN TUNING CLOSURE

Status: `STEP_6_3D_FORMAL_TRAIN_TUNING_CLOSURE=PASS`; `VALIDATION_COMPARISON=NOT_STARTED`; `SEARCH_METHOD=NOT_SELECTED`.

## Step Goal and Definition Basis

Accept, independently recompute, back up and report the completed TRAIN-only CEM tuning. The researcher froze this closure scope on 2026-10-01. It uses the already authorized STEP 6.3D fixed-budget protocol and read-only target definition `06策略器与候选动作规划.md` §3.1 (SHA-256 `f20294bd8708076ae7583be679a8182a0990ecf6777e05df378ae4c2855431f1`). No new GPU search or Validation is part of this Step.

## Initial State and Files Involved

`HEAD=origin/main=5b48ad665489e3891250b4b5439e5ef3759addbd` after `git fetch`; tracked worktree clean. Existing untracked ZIP, `TASK/` and plotting script were preserved. The RTX 3080 Ti TRAIN runner had exited after creating 768 raw result JSONs, the formal log, `05_train_hyperparameter_tuning_receipt.json` and `06_frozen_selected_cem_configs.json`. Validation had zero results.

Source/config: `code/scripts/run_step6_3d_formal_cpu_matrix_v1.py`, `run_step6_3d_one_cpu_solve_v1.py`, `code/src/pi_jwm/step6_3d_method_selection_v1.py`, and the frozen 3080 Ti execution config. Closure code: `code/scripts/fetch_step6_3d_train_evidence_v1.py`, `close_step6_3d_formal_train_tuning_v1.py`. Local evidence root: `code/artifacts/protocols/pi_jwm_step6_3d_fixed_budget_search_v1_20260929/`.

## Changes and Reuse

Read-only SFTP checked all 16 source/manifest SHA values on both machines against the frozen execution config before transfer. A remote SHA inventory and tar stream copied 768 raw results, the complete log and both receipts. Each local file was checked against the independently computed remote SHA and size; remote originals were preserved. Local raw results remain under `solve_results/train/`, with `train_remote_stream.tar` and `train_tuning_local_archive.zip` as additional local backups. No credential was written to a file, receipt, command log or Git.

The closure analyzer reconstructs the exact 32 anchors × 3 seeds × 2 methods × 4 configurations from raw JSON, checks every execution/source/checkpoint/sidecar identity, independently calls the frozen `tune_cem_config()` rule, and compares the entire rebuilt ranking with both runner receipts. It writes per-method/config and per-anchor/seed diagnostics plus an acceptance receipt. The local ZIP contains 774 files: 768 raw JSON, log, runner receipts, SHA inventory, diagnostic, acceptance. The large raw results, tar and ZIP stay local; only the small manifest/diagnostic/acceptance are Git tracked.

## Validation: Actual Evidence

- Grid: 768/768 unique identities; exactly 96 cases/config, methods `{S-CEM,MH-CEM}`, `K={3,4}`, `rho={0.1,0.2}`, 32 selected TRAIN anchors, seeds `{6301,6302,6303}`. No duplicate, missing or out-of-grid result.
- Every result: CUDA, `NVIDIA GeForce RTX 3080 Ti`, FP32, batch 16, execution config ID `3834e6b2e76c93e66dc92f0b3d0c56ad7e753c07be9ef0b8cc022a2596b7b680`, checkpoint SHA `941ee94131d406de914a79aeda43e929c92727631263d85221615422146a32c9`, frozen source hash map, aligned sidecar, `locked_test=false`. No CPU/4090/mixed-config result.
- Nominal `B_WM`: `768×512=393216` candidate one-step transitions. Actual `N_unique_transition_evals=393216`; cache hits `94159` are separately counted. Proposed/admitted `519556/519556`, rejected `0`.
- Independent TRAIN selection reproduced complete runner ranking and frozen `(K=4,rho=0.1)` for both methods. The selection order is H4 success count, paired objective net outcome, smaller K, smaller rho.
- Selected-anchor static empty/cohort-zero residuals both `0`, from the accepted input-eligibility receipt. Validation raw results and receipts absent; no method-selection receipt. No scorer exception in raw `score_residuals`.
- Remote-to-local file inventory: 771/771 SHA/size matches. Inventory SHA `deecd4a13d9d44db189cb185862a18611b3d5a92eadb345ebaefdd0c49428957`; tar SHA `52bb0a87e2fda2a20f15ad195353d85df54b95e2d3e45fff99389b79b76cd553`; complete ZIP SHA `13a610b1a3f33dd66d2f47465fc34dc117b3d97e8a679fa2eb86c11cd799bded`, ZIP integrity test PASS. Machine manifest and acceptance remain in the evidence root.
- Local `python -m unittest discover -s code/tests -p 'test_step6_3d*.py'`: 15/15 PASS; `python -m compileall -q code/src code/scripts code/tests`: exit 0. `python code/scripts/build_project_knowledge_index_v1.py` and `--check`: `passed=true`, `mismatches=[]`. `git diff --check`: exit 0. Context Consistency Check covered 00–08: 00/05/07/08 updated; 01/02/03/04/06 have no new research, architecture, data-flow, module or researcher-decision change. Existing old entries are historical snapshots under the new current state.

## TRAIN-only Results

Each row has 96 cases and 49,152 actual unique one-step transitions. `H4 scored` counts distinct candidate fingerprints; `H4 unscored` counts completion attempts, while `complete` counts all completed paths. Repeated scoreable paths mean these three columns do not form an additive partition.

| Method | K | rho | Scoreable cases | Paired net within method | Complete H4 | H4 scored | H4 unscored | Return boundary events | Grammar dead ends |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| S-CEM | 3 | 0.1 | 48/96 | +5 | 10,300 | 3,484 | 6,768 | 6,768 | 4,954 |
| S-CEM | 3 | 0.2 | 48/96 | -40 | 10,234 | 3,450 | 6,779 | 6,779 | 4,898 |
| S-CEM | 4 | 0.1 | 48/96 | +18 | 12,046 | 4,233 | 7,638 | 7,638 | 4,648 |
| S-CEM | 4 | 0.2 | 48/96 | +17 | 11,946 | 4,211 | 7,648 | 7,648 | 4,481 |
| MH-CEM | 3 | 0.1 | 48/96 | -16 | 10,599 | 3,819 | 6,735 | 6,735 | 4,957 |
| MH-CEM | 3 | 0.2 | 48/96 | -38 | 10,522 | 3,733 | 6,776 | 6,776 | 4,863 |
| MH-CEM | 4 | 0.1 | 48/96 | +53 | 12,447 | 4,387 | 7,745 | 7,745 | 4,671 |
| MH-CEM | 4 | 0.2 | 48/96 | +1 | 12,169 | 4,350 | 7,708 | 7,708 | 4,468 |

Totals: 384/768 scoreable cases, 90,263 completed H4 paths, 31,667 distinct scoreable H4 candidates, 57,797 unscoreable H4 completion attempts, 57,797 future Return-birth boundary events, 37,940 Grammar dead ends. All 16 zero-scoreable anchors remain in the grid; all 48 corresponding anchor/seed pairs have no scoreable result across eight configs. The Return count is repeated scorer events across configurations and seeds, not independent Task births. `H4_UNSCOREABLE:SCOREABLE` appears 6,217 times: the scorer status alone was scoreable but the strict H4/H_sup=4 gate still rejected the path; every unscoreable completion has a recorded Return-birth boundary event. Scorer exceptions (`ScorerStateInconsistency`/`BurdenSemanticsBlocked`) are zero. Full per-anchor/seed data, proposal/admission/rejection/cache/budget and wall-clock totals are in `train_tuning_diagnostic_summary.json`.

The formal log was created 2026-09-30 18:30:26.597 CST (`stat -c %w`); last result was written 2026-10-01 15:53:14 CST. Elapsed wall time is 76,967.4 s = 21.38 h, 35.92 cases/h and 5.109 actual unique transitions/s including loading and orchestration. Summed in-solve timers are 50,941.6 s and cannot substitute for elapsed wall time. Validation nominal budget is 1,720,320 transitions; at the TRAIN effective rate this is about 93.54 h, an approximate resource-planning value only. HRS and B_WM 256/1024 may differ; 1024 host-storage probes were slower. No Validation runtime or result is claimed.

## Expected vs Actual, Known Issues, and Decision Boundary

The 768-case TRAIN grid and source identity passed, and the frozen rule selected both expected configs. The 48/96 case success for every config does **not** imply equal method quality or that K/rho have no effect. It also does not compare either CEM method with HRS; that requires the separately authorized Validation matrix. The fixed-support future Return-birth limitation remains; `H4_SEARCH_COMPARISON_READINESS=SUPPORTED_WITH_MODEL_OBJECTIVE_SUPPORT_LIMITATION`. No anchor was removed, no scorer/grammar/objective/model/budget rule changed, and no new GPU search or `locked_test` was run.

## Git and Next Step

Commit/push and post-commit verification are recorded in Git history for this closure. The local workstation retains 60 Formal Raw files, the Formal Dataset and a `best.pt` whose SHA matches `941ee94131d406de914a79aeda43e929c92727631263d85221615422146a32c9`, in addition to the independently SHA-verified TRAIN backup. Thus the 3080 Ti instance can be stopped if Validation will not start soon; reusing an ephemeral instance later would require restoring the data/checkpoint and rechecking identities. This Step does not stop it. The sole next research action is researcher review and separate authorization of Validation comparison; it is not started here.
