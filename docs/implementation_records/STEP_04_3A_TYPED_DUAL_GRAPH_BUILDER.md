# STEP 4.3A — Definition 03 Typed Dual-Graph Builder Contract

## Step Goal

Map the frozen STEP 4.2A/4.2C-C Tensor into typed current Physical and Information graph objects plus Align/GeoComm structural references, without implementing an encoder or changing upstream semantics.

## Definition Basis

- Read-only authority: `D:\shen\OB\科研\PIJWM\03物理-信息双图建模.md`
- SHA-256: `6f3e17b0691aa81c60c2ea1e76a1031d2f7f622abf4b40884e9ae4380953c15e`
- Relevant sections: Physical graph lines 88–147; Information graph lines 151–338; Align/GeoComm and forbidden shortcuts lines 548–650.
- Frozen engineering baseline: `fd20e86a99b20f71a5dfd031fa673bd862a4f2bf`.

## Initial State

STEP 4.2C-C Tensor already carried current position/motion, typed Comm, Task-Agent, DAG, logical Flow, and Carrying state. No current typed graph object, physical topology contract, Align, GeoComm, or Step 4.3A receipt existed.

## Files Involved

- `code/src/pi_jwm/step4_3a_typed_dual_graph_builder_v1.py`
- `code/scripts/build_step4_3a_typed_dual_graph_builder_v1.py`
- `code/tests/test_step4_3a_typed_dual_graph_builder_v1.py`
- `docs/contracts_PIJWM_STEP_04_3A_TYPED_DUAL_GRAPH_BUILDER_V1.md`
- `code/artifacts/protocols/pi_jwm_step4_3a_typed_dual_graph_builder_v1_20260920/`
- Tracker, authority/process records, AI_CONTEXT, indexes, and changelog.

## Changes and Reuse

- Reused frozen Tensor indices, masks, and all state values without simulator reads or semantic recomputation.
- Added eleven independent typed fixed-capacity blocks.
- Built Physical membership from current presence plus valid XYZ, and spatial relations from explicit development topology config only.
- Kept Comm validity separate from CSI observation; kept logical Flow separate from Carrying hop state; retained parallel Flow rows.
- Added identity Align and wireless endpoint GeoComm references without requiring matching Physical edges.
- Added actual-AND validator/receipt, serialization, deterministic digest, and twenty negative/counterfactual fixtures.

## Validation

- Focused: `python -m unittest discover -s code/tests -p 'test_step4_3a_typed_dual_graph_builder_v1.py'` → 15/15 passed.
- Frozen regressions: STEP 4.2C-C 23/23; STEP 4.2A 17/17; STEP 4.1 7/7; STEP 3.3 8/8; STEP 3.2 11/11 passed.
- Artifact builder: 24/24 required checks and 20/20 negative/counterfactual fixtures passed.
- Deterministic rebuild: NPZ SHA-256 remained `6e0f6d68bb7214babf706f5af94fd004a664d61d714bbcb8b502867f45ec2bc6`; semantic digest `d488c127c2ba5d8d5ddea106016bc38a002af861e85b670968824bc4fb811282`.
- Serialize/load plus manifest hashes passed; `compileall`, knowledge-index build/check, and `git diff --check` passed after the final code edit.
- Scope receipt: `training=false`, `gpu=false`, `locked_test=false`, `formal_dataset=false`, and all model/encoder/planner scopes false.

## Results

The frozen Tensor now deterministically maps to `G_t^Phy`, `G_t^Info`, and `R_t^PI` at `history[-1]`. The artifact distinguishes real trace inputs, derived graph structure, development topology config, and contract fixtures. The topology values are not a frozen research choice.

## Expected vs Actual

Expected: preserve strict Physical/Information semantics, logical Flow identity, masks, causal current-time isolation, and explicit cross-domain references. Actual: implementation and machine checks match this boundary; no upstream data or research semantics were modified.

## Known Issues

- Physical topology mode/radius/k remain a development config, not a research decision.
- Return multi-hop, same-destination reroute runtime, and formal graph capacities remain outside current evidence.
- No graph encoder, message passing, temporal aggregation, model, loss, planner, or training has been implemented.

## Git

Implementation commit `e91881abba5ebfcc4761b9c422f0c0187c4fc0b6` (`feat(graph): freeze typed dual-graph builder contract`) was pushed to `origin/main`. A documentation-only closure commit records the post-push process state.

## Next Step

Only recommend `STEP 4.3B — Definition 03 Dual-Graph Encoder Contract`; do not execute automatically.
