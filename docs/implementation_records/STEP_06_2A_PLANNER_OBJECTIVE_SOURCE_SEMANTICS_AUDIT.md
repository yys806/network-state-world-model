# STEP 6.2A — Planner Objective Source & Semantics Audit

## Step goal

Audit source semantics and freeze a machine-readable protocol foundation for Planner Objective v1. No candidate ranking, winner selection, baseline execution, closed-loop run, checkpoint load, or dataset modification was performed.

## Researcher decisions received

The researcher-specified target tuple is `(N_DDL, A_DDL, J_Delay, J_Burden, J_Effort)` with lexicographic minimization. Throughput is diagnostic/final metric only; Energy and Fairness are outside Planner v1; Priority is inactive; Risk is defined but inactive; primary rollout is mean prior plus expectation service; common support-aware horizon is mandatory; unsupported future Return is a model support boundary.

The ordinary Definition 06 note contains a tendency toward Hybrid in places, but current `AI_CONTEXT/06_DECISIONS.md` keeps `CANDIDATE_METHOD_SELECTION=RESEARCH_PENDING`. The current frozen decision wins; this Step does not select Hybrid.

## Source and provenance

Audited AirFogSim symbols include `Task.getTaskDeadline`, `getTaskArrivalTime`, `getTaskPriority`, `getTaskSize`, `getTaskCPU`, `getReturnedSize`, `Task.wait_to_ddl`, `TaskManager.addToComputeTask`, `TaskManager.checkTasks`, `generateAndCheckTasks`, and `AirFogSimEnv.step`. Current PI-JWM sources include the Full Dual Graph observer, tensor builder, Structured RSSM rule transition, fixed-support detector, stateful Flow audit, and metrics module. Exact SHA256 values are in `01_source_semantics_receipt.json`; local AirFogSim has no independent Git identity.

Read-only researcher definitions and their SHA256 values were recorded in the same receipt. The private directory was not modified.

## Findings

AirFogSim deadline is a relative duration in simulation-time units. The absolute due time is arrival plus duration. The completion path accepts equality; the later hard-deadline sweep fails only when elapsed time is strictly greater. The check runs after environment updates. The runtime can distinguish DONE, FAILED, computation-finished, release, presence, and Return requirement, but current World Model rollout does not derive deadline failure because deadline is not in its causal state.

Intermediate Flow service reduces hop remaining and advances the hop while preserving end-to-end remaining. Terminal service reduces Flow remaining. The target transmission burden formula is only partially source-supported because the current causal audit does not universally prove end-to-end remaining across completed hops and stable route/epoch identity. Computation rule and Return-support flags are source-backed; future Return birth remains unsupported and imposes the `H_sup=u_k-1` boundary.

The full field-by-field Source-to-Metric matrix is machine-readable in `10_objective_field_provenance_matrix.json`. Deadline, arrival, priority, return size and task failure are marked source-available but not accepted Planner-side exposure; no Future Target or failed label is used to reconstruct them.

## Readiness verdict

`DEADLINE_SOURCE_READINESS=BLOCKED`; `DELAY_SOURCE_READINESS=PARTIAL`; `TX_BURDEN_SOURCE_READINESS=PARTIAL`; `COMP_BURDEN_SOURCE_READINESS=PARTIAL`; `EFFORT_SOURCE_READINESS=PARTIAL`; `SUPPORT_BOUNDARY_READINESS=PARTIAL`; `FINAL_METRIC_INTERFACE=RECORDED`; `BASELINE_SYNC_INTERFACE=RECORDED`.
`STEP_6_2B_READINESS=BLOCKED`

The minimum blocker is a Planner-only causal side-state additive exposure for deadline/arrival/priority/return support, plus source-backed end-to-end Flow remaining and a candidate-independent Route effort denominator. This Step does not implement that repair.

## Provenance correction and boundary details

The first draft of the provenance table used an unsupported `CAUSALLY_DERIVABLE` label. It was removed. The final matrix distinguishes runtime source, observer availability, Formal Raw, Sample, Tensor, Graph, World Model state, and Planner-side exposure. A source existing in AirFogSim or an observer is not treated as Planner input until the field is actually carried by the frozen causal contract.

Deadline semantics are recorded in two separate runtime paths: completion accepts `delay <= deadline`, while the later active-task hard-deadline sweep fails only when `delay > deadline`; the small `1e-5` tolerance belongs to the no-return completion path. Completion handling precedes the later sweep. This distinction is preserved in `02_deadline_lifecycle_audit.json` and is not collapsed into a single generic deadline comparison.

Energy was audited from the AirFogSim EnergyManager paths (flight, hover, sensing, transmit and receive). The current Formal Raw/Tensor/Planner contract does not expose an accepted causal energy state, so Energy remains a final closed-loop metric boundary and is not activated in Planner Objective v1. Task priority has a runtime source, but v1 keeps `w_q=1` and does not use priority weighting.

The final machine verdict is `PASS_WITH_READINESS_BLOCKERS`. The unresolved blockers are: Planner-only causal exposure for deadline, aligned arrival/elapsed, priority and return-support state; incomplete source-backed cross-hop E2E remaining/route identity evidence for `B_Tx`; and no candidate-independent normalized Route-effort denominator. Consequently `STEP_6_2B_READINESS=BLOCKED`. This Step does not modify Formal Dataset, checkpoint, model architecture, action domain or AirFogSim source.

## Validation record

The audit runner was executed on CPU with the frozen repository state. The focused test passed (`1/1`). The generated bundle contains the source receipt, deadline lifecycle audit, cohort, Flow/compute/support/throughput/effort audits, provenance matrix, objective contract, baseline synchronization contract, readiness receipt and manifest. The full focused regression and index gates are recorded in the final validation section below.

## Validation and boundaries

The audit runner is `code/scripts/run_step6_2a_planner_objective_semantics_audit_v1.py`. It generated the 13 receipts and manifest under `code/artifacts/protocols/pi_jwm_step6_2a_planner_objective_semantics_audit_v1_20260928/`. It is CPU-only and read-only with respect to Dataset, checkpoint, model architecture, AirFogSim source, and researcher definitions.

No performance claim, baseline, locked-test, GPU, candidate selection, Search/Learned/Hybrid choice, or closed-loop experiment is implied.

## Final validation

Validation commands and results:

- `python -m unittest discover -s code/tests -p 'test_step6_2a*.py'`: 1/1 passed.
- Focused regressions: STEP 6.0C 12/12; STEP 6.1 2/2; STEP 4.4 37/37; STEP 4.2C-B Flow ledger 25/25; STEP 5.6C checkpoint acceptance 3/3; STEP 5.5 Formal Dataset 21/21; STEP 6.0A adapter/fixed-support 6/6 passed.
- `python -m compileall -q code/src code/scripts code/tests`: passed.
- `python code/scripts/build_project_knowledge_index_v1.py` and `--check`: passed, 5 generated outputs, zero mismatches.
- `git diff --check`: passed; only line-ending normalization warnings were emitted.
- Contract/protocol cross-check: the Markdown objective and baseline interfaces agree with the corresponding machine-readable JSON contracts on tuple/order, inactive dimensions, metric separation, relative-improvement direction, and readiness blockers.

The evidence bundle is ignored by the repository-wide `code/artifacts/*` rule, so only this Step-specific receipt directory is force-added. No other ignored artifact, Formal Dataset, checkpoint, or user-untracked file is included.

## Git

The verified audit commit `fa7c369` (`audit(pi-jwm): freeze planner objective source semantics`) was pushed to `origin/main`. A follow-up documentation-only closure commit records the completed Git gate and is reported in the final completion report. `TASK/` and `code/scripts/plot_step5_3e_tiny_overfit.py` remain untracked and were not added.
