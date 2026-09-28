# STEP 6.2B-PATCH — Planner v1 Route No-Op Closure

## Step Goal and Definition Basis

Close the two Route admission blockers found by STEP 6.2B and recalculate scorer acceptance. The 2026-09-28 researcher decision freezes `PLANNER_V1_ROUTE_POLICY=EXPLICIT_NOOP_ONLY`. The existing five-part Objective and lexicographic comparator remain frozen. Source basis: the 6.0C action-domain gate, 6.1 candidate rollout, 6.2B scorer, formal action adapter, 4.4 Route encoder/rule, and prior 6.2B machine receipts. This patch does not modify AirFogSim, 4.4, training or checkpoint bytes.

## Initial State and Conflict

Start: `HEAD=origin/main=3ac4470080b11906d16c3a4c33c3a3306f7f31c4`. Planner v1 accepted direct single-hop Route. A pending/no-current-Flow Route mapped to `flow_index=-1` and did not create a Flow. An existing same-path Route changed Task-Agent Host and learned latent, so it was not a no-op. The original STEP 6.2B record therefore correctly said `BLOCKED_ON_OBJECTIVE_SEMANTICS` at that time.

## Changes and Reuse

`step6_0c_planner_action_domain_v1.py` now rejects every nonempty Route family at admission with `VIOLATED: OUTSIDE_PLANNER_ROUTE_NOOP_ONLY_V1`; an empty Route family passes. This applies to every horizon and to pending, existing same-path, multi-hop and destination-change rows. Comm, Comp and Mob validation is unchanged. The existing formal adapter compiles empty Route to negative Task/Flow indices; 4.4 skips negative Route rows before its learned Route encoder and deterministic Route rule. The generic Route interface, learned Route encoder, repaired multi-hop code and all 11 action tensors remain present. The scorer's pending Route guard stays as an additional defensive check. This is a Planner v1 learned-support restriction, not a physical legality claim.

The original 6.2B blocked-receipt builder now stops with an explicit historical-snapshot message under the new Route policy, before writing any old receipt. The old observations remain immutable; the new Patch builder is the active machine-evidence entrypoint.

The existing 6.2B frozen-checkpoint CPU integration was reused with its historical nonempty-Route audit disabled. It scores two legal H1–H4 candidates from one non-locked validation anchor; both have empty Route, and differ in Comp. This bounded fixture is mechanism evidence, not an online candidate-generation method or performance comparison. The independent effective-freedom audit uses bounded causal fixtures: pending, existing same-path, multi-hop and destination-change Route rows all fail admission; explicit empty Route passes. Thus legal state-changing Route freedom in v1 is `NONE`.

## Validation and Results

The focused 6.0C tests first failed on the old direct-route acceptance, then passed after the gate change. The Route no-op compiled indices are all negative, and the HOLD candidate has zero routed Task/Flow learned action. The same frozen checkpoint strictly loads on CPU; its SHA-256 remains `941ee94131d406de914a79aeda43e929c92727631263d85221615422146a32c9`, parameter digest is unchanged, mean-prior/expected-service H1–H4 states and scorer values are finite. At the selected anchor the Comm effort denominator remains 50 global RB IDs, not 242 communication relation rows. The scorer's deadline, delay, burden, effort, shared `H_eff`, order and leakage tests remain in force.

Actual CPU commands: `python -m unittest discover -s code/tests -p <pattern>` for 6.0A (6), 6.0C (15), 6.1 (2), 6.2A (1), 6.2A-PATCH (11), 6.2A-ROUTE-RECOVERY (6), 6.2B (11): 52/52 PASS. `python code/scripts/run_cross_layer_rule_semantics_gate_v1.py`: PASS, 115 tests; its rewrite of an older historical receipt was restored. `python code/scripts/build_step6_2b_patch_route_noop_receipts_v1.py`: PASS, eight receipts plus manifest. `python -m compileall -q code/src code/scripts code/tests`, `git diff --check`, and knowledge-index write/check passed. All commands ran on CPU without `locked_test`.

Expected and actual: Planner v1 admits only explicit Route no-op; pending and existing Route blockers cannot enter its rollout; the five-part scorer and strict comparator need no mathematical change. `STEP_6_2B=PASS`, `PLANNER_OBJECTIVE_SCORER=IMPLEMENTED_AND_CPU_CONTRACT_VERIFIED`, and `MPC_OBJECTIVE=FROZEN_AND_IMPLEMENTED` are CPU contract verdicts only.

## Known Issues, Git and Next Step

4.4 Route-to-Task-Agent Host behavior and learned Route response remain future Route-expansion semantics to revisit. Multi-hop code is retained, while formal multi-hop training and validation coverage are zero. The selected deadline sidecar has one-anchor source-alignment evidence; broad runtime coverage, candidate method, ranking quality and closed-loop performance are not established. `CANDIDATE_METHOD_SELECTION=RESEARCH_PENDING`, `CLOSED_LOOP_READINESS=NOT_READY`, `MULTIHOP_PLANNER_READINESS=NOT_IN_V1_DOMAIN`, `ROUTE_OPTIMIZATION_ACTIVE_V1=false`.

No GPU, retraining, optimizer step, checkpoint replacement, Formal Dataset rebuild, `locked_test`, baseline or closed loop. Commit/push identity is reported at closure. The single next action is researcher review of this bounded acceptance; any candidate-method work requires separate authorization.
