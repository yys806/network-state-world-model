# STEP 6.2A-CLOSURE — No-Retrain and Single-Hop Planner v1

## Step Goal and Definition Basis

Close the recovered checkpoint decision, freeze Planner v1 Route to Formal Dataset single-hop support, and recalculate the narrowly scoped STEP 6.2B scorer readiness. Researcher decisions are stated in the 2026-09-28 task authorization. No training, objective scoring, ranking, closed loop, baseline, GPU, or locked test is included.

## Initial State

Start `ec0eeedc947fc4190050661b3d0d0e9f37402f84`, equal to fetched `origin/main`. The frozen best checkpoint SHA is `941ee94131d406de914a79aeda43e929c92727631263d85221615422146a32c9`. Existing untracked `TASK/` and `code/scripts/plot_step5_3e_tiny_overfit.py` were preserved.

## Changes

The candidate admission gate keeps Route enabled while rejecting any nonempty path whose `route_node_indices` length is not exactly one, and checks an existing Flow target against its frozen logical destination. Empty Route remains the contract's no-op. This gate does not modify the repaired 4.4 rule, Flow ledger, route metadata, 11 learned action tensors, or candidate fingerprint format.

The v1 legal route is `[current logical destination]`; no relay, destination change, new Flow, or Epoch creation is admitted. Existing Flow destination change remains `UNSUPPORTED_BY_FIXED_OBJECT_SUPPORT`. Multi-hop semantics are repaired and cross-layer tested in code, but outside Planner v1 and reserved for later expansion/ablation. This is a restriction to Formal Dataset support, not a simulator physical-legality claim.

## Reuse and Evidence

The route-recovery receipts report 4416 train windows / 48 trajectories and 1104 validation windows / 12 trajectories. Route width greater than one, active multi-hop windows/flows, and legacy intermediate completion are all zero in both splits. Nonempty Route action counts are 5671 train and 1594 validation; multi-hop Route action counts are zero. Existing-Flow Route overlap is 38 train and 16 validation windows. On those 54 windows, paired H1-H4 legacy/patched state, graph, prior, Motion and CSI are exactly equal.

The accepted recovery receipt records 428 checkpoint tensors strict-loaded, unchanged parameter digest and unchanged checkpoint bytes. `LVal=0.07431338784170399` remains the original accepted-run observation. Patched full validation was not executed and is not represented as a patched result. No-retrain is now an explicit researcher decision, not an engineering inference; no formal multi-hop performance claim is made.

## Objective and Readiness

The frozen objective remains `(N_DDL, A_DDL, J_Delay, J_Burden, J_Effort)`, lexicographic minimize. Route effort stays excluded. The general `B_Tx = R_hop + (N_hop - 1 - current_hop_index) * R_e2e` contract remains; under legal v1 conditions `N_hop=1,index=0`, so `B_Tx=R_hop`. The focused side-state test verifies zero remaining hops, partial-service decrease, zero terminal/completed burden, and retains a general multi-hop formula case.

Deadline side-state API is generic for any current Decision with an exactly aligned current-time deadline sidecar; available deadline evidence remains selected-anchor only. Return need derives from required size, predicted compute host and causal return destination. Future Return birth remains the `H_sup/H_eff` support boundary and does not create a Flow. `H_eff=0` is `OBJECTIVE_UNSCOREABLE`.

Current verdict: `STEP_6_2B_READINESS=READY_FOR_SINGLE_HOP_SCORER_IMPLEMENTATION`. This permits only scorer/comparator implementation and CPU contract tests. `CLOSED_LOOP_READINESS=NOT_READY`; candidate method remains `RESEARCH_PENDING`; multi-hop is `NOT_IN_V1_DOMAIN`; performance is not established.

## Validation

Focused action-domain tests: 14/14 pass. Planner objective side-state tests: 10/10 pass. The current cross-layer gate passes 114 tests (the recovery receipt's historical run had 113 before this focused burden test). Closure machine receipts are generated under `code/artifacts/protocols/pi_jwm_step6_2a_closure_single_hop_v1_20260928/`. Final regression, compile, index, consistency, checkpoint identity and Git outcomes are recorded in this implementation record's committed revision.

## Known Issues and Boundaries

Full patched validation remains unexecuted. Formal learned support contains no multi-hop examples, so no multi-hop learned-performance claim is made. Deadline evidence remains selected-anchor coverage despite generic API behavior. Scorer, ranking, candidate-method choice, baseline, closed loop, GPU, retraining and locked test remain unopened. Future formal training requires `CROSS_LAYER_RULE_SEMANTICS_GATE=PASS`.

## Expected vs Actual

Expected: preserve the accepted checkpoint and support only formal single-hop Planner v1 Routes. Actual: recovery identity receipts and paired invariance support no-retrain acceptance; the domain gate rejects relay paths and mutable existing destinations; readiness is limited to scorer implementation.

## Git

Start SHA: `ec0eeedc947fc4190050661b3d0d0e9f37402f84`. Final commit and push are recorded after verification.

## Next Step

Stop for researcher review. No subsequent step is started automatically.
