# STEP 6.2A-PATCH — Planner Objective Readiness Reconciliation

## Step Goal and Definition Basis

Rejudge STEP 6.2B prerequisites from current source and accepted artifacts; add only Planner-side causal metadata. The researcher keeps the lexicographic objective `(N_DDL,A_DDL,J_Delay,J_Burden,J_Effort)` and explicitly removes Route effort, priority weighting, active risk, energy and fairness. Main throughput is E2E useful delivery; all-hop service is diagnostic. The read-only Definition 06 remains a target, not proof of implementation. This Patch does not implement a scorer or alter the trained model.

## Initial State and Why the Patch Was Needed

At `6ba5b483037615156b4d14499660a3bc3b13fd93`, local `main` equaled fetched `origin/main`. Existing user-untracked `TASK/` and `code/scripts/plot_step5_3e_tiny_overfit.py` were preserved. STEP 6.2A had incorrectly carried the earlier 4.2B no-E2E-source conclusion past the later 4.2C-B/C implementation. It also treated missing Priority and a Route-effort denominator as prerequisites even though the researcher excluded both from v1.

## Historical and Current Flow Facts

**HISTORICAL FACT:** `step4_2b_stateful_flow_source_audit_v1.py` predates the causal Ledger; raw hop events alone did not expose conserved E2E remaining. **CURRENT IMPLEMENTATION FACT:** `step4_2c_b_causal_flow_ledger_raw_v1.py` builds `FlowID=(TaskID,FlowType,Epoch)`, records logical source/destination, total/E2E delivered/E2E remaining, holder, hop progress/remaining, route revision, presence/status and lineage. `step4_2c_c_flow_sample_tensor_v1.py` preserves these in Sample/Tensor; typed Graph and 4.4 state carry them. **CURRENT VERDICT:** E2E Flow state exists and the 4.2B missing-field verdict is historical only.

Real non-locked evidence proves direct Input/Return, two-hop Input, intermediate service without E2E decrement, final service with E2E decrement, and no double count. Return multi-hop, same-destination reroute and destination-change Epoch remain contract fixtures. DepData runtime count is zero (`NOT_APPLICABLE_V1`). Exact checks and source hashes are in receipts 01–02.

## Route and B_Tx Conflict

The target is `B_Tx=R_hop+(N_hop-1-current_hop_index)*R_e2e`. The current model can provide `R_hop` and `R_e2e`. However, 4.2C-C stores a route as remaining destinations excluding its holder. The frozen 4.4 rule reads `route[next_hop]` as the next holder and `route[next_hop+1]` as the next target. An executed deterministic test gave a destination-only `[next,final]` route and completed the intermediate hop: positive delivered bytes, `hop_remaining=0`, but `current_hop_index=0`, holder unchanged and E2E remaining unchanged. This reproduces a current semantic conflict even without a Route action.

The Route action tensor contains `[source,destination,kind,hop_count]`; the deterministic rule consumes only source/destination. It changes carrying endpoints and may increment RouteRevision, but it does not write the full `route_node_indices`/mask, logical destination or Epoch. The new `PlannerRouteCausalSideState` can carry a candidate route and reject state divergence; it cannot make the frozen model advance along that route. Consequently single current hop is definable, multi-hop Input is blocked, direct existing Return has contract support with runtime coverage limitations, reroute is blocked, destination change is blocked and DepData is not applicable. The unresolved mismatch blocks STEP 6.2B; no model or dataset bytes were changed to conceal it.

## Task Causal Side-State, Return and Deadline

The replay script executed exactly one accepted non-locked Formal Validation trajectory (`simulator_seed=2026092302`, `policy_seed=2026092402`) using the same collector and previously executed action prefix. Frame 1/sample 4416 matches original trajectory/frame/time/capture ID, current task rows, entity rows, physical observations and action prefix. It captured `Task.getTaskDeadline()` before the action for six current tasks. The accepted Raw and current Sample/Model state independently align those six Task IDs to slots 0–5; one active Flow also aligns. Source time equals anchor time. The sidecar is additive and did not rewrite Formal Raw, Dataset manifest, normalization, Tensor or checkpoint.

`PlannerTaskCausalSideState` carries Task identity, arrival/elapsed/deadline/absolute deadline, returned size and Return destination. Deadline failure is derived by the Planner; 4.4 has no deadline learned output or lifecycle transition. Relative deadline uses simulation seconds; active failure is strict `elapsed>deadline`, Return completion accepts equality, and no-Return completion has `1e-5` tolerance. The completion path runs before the later failure sweep. The support boundary is `H_sup=u-1` for first unsupported state `u`, and `H_eff=min H_sup`; zero is unscoreable.

The current accepted Formal Raw already stores `required_returned_size`, contrary to the old 6.2A blanket missing-return statement. Source audit found that positive size alone is **not** equivalent to actual `Task.requireReturn()`: if the assigned compute host equals the Return destination, ordinary Task execution can complete without a Return Flow. The side-state therefore preserves the Return destination and evaluates Return birth against the predicted compute host at completion. A future Return birth is a support boundary, never a fabricated Flow. The selected Formal anchor has six positive return sizes; zero-size behavior is covered by source and focused fixture, not a real zero-size anchor claim.

## Effort, Throughput and Baseline Sync

`J_Effort` is the mean of anchor-applicable normalized Comm, Comp and Mob components. Denominators are frozen from current valid RB support, observed raw static CPU capacity and 15 m/s times present UAV count. The 15 m/s is a Formal Dataset profile bound, not a physical limit. RouteRevision has no accepted resource-cost map, so Route effort is absent. Priority weights remain one and are no readiness prerequisite.

The shared Flow/Outcome extractor reports real E2E useful throughput from decreases in conserved E2E remaining and all-hop network-service throughput from carried bytes, both over real elapsed time. It accepts wired and wireless terminal delivery and prevents intermediate-hop useful double count. A real two-hop Input Ledger trace provides 0.28065 MB useful versus 0.56131 MB hop service; a three-hop fixture tests 1 MB useful versus about 3 MB service. Future PI-JWM and baseline executions must call the same extractor. No baseline was executed here. Objective, final metric, predicted diagnostic and acceptance gate remain separate.

## Files, Reuse and Validation

New source: `code/src/pi_jwm/step6_2a_planner_objective_side_state_v1.py`, `step6_2a_throughput_metric_v1.py`. New scripts: `code/scripts/replay_step6_2a_patch_deadline_sidecar_v1.py`, `build_step6_2a_patch_readiness_v1.py`. Tests: `code/tests/test_step6_2a_patch_side_state_v1.py`. Reused the STEP 5.5 collector and Formal packages, 4.2C Flow Ledger, typed Graph, 4.4 deterministic transition, and 6.1 anchor identity. The receipt bundle is `code/artifacts/protocols/pi_jwm_step6_2a_patch_objective_readiness_v1_20260928/`, with 01–12, separate baseline contract, exact-aligned sidecar and manifest.

Focused PATCH tests: 9/9 pass, including actual 4.4 cross-hop and Route-action array/hop-count probes, deadline/tolerance, Return host condition, support off-by-one, stale route rejection, E2E throughput and future-input rejection. Replay alignment and anchor preparation: PASS. Additional regression and index outcomes are recorded in the completion report/process logs. This is CPU-only, no optimizer step, GPU, SSH, `locked_test`, baseline, ranking, winner, closed loop or performance claim.

Final focused regressions with `PYTHONPATH=code/src;code/scripts`: 4.2C-B 25/25, 4.2C-C 23/23, 4.4 37/37, 5.5 21/21, 6.0A 6/6, 6.0C 12/12, 6.1 2/2 and 6.2A including PATCH 9/9. The initial parallel regression attempt omitted `PYTHONPATH` for older test modules and failed on imports only; the documented project environment was then used and all required suites passed. `compileall`, `git diff --check`, knowledge-index write and `--check` passed. This Patch's builder re-read current Raw and source after validation and regenerated the machine receipts.

## Expected vs Actual, Known Issues, Git, Next Step

Expected: remove superseded blockers and, only if all current conditions close, mark 6.2B ready. Actual: deadline source, Return source, Flow E2E fields, effort and throughput definitions close for the selected anchor, but 4.2C-C/4.4 route semantics conflict. `STEP_6_2B_READINESS=BLOCKED`. The side-state is a foundation and divergence guard, not a repair of model prediction. The remaining scientific boundary is whether to change and revalidate the trained model's route transition or restrict the Planner v1 action/support domain; Codex makes neither decision in this Patch. Git commit/push SHA is reported at completion. Single next action: researcher reviews the route semantic conflict and authorizes a bounded repair route before 6.2B.
