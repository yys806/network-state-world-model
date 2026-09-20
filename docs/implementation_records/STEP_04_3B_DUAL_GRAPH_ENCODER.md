# STEP 4.3B — Definition 03 Dual-Graph Encoder Contract

## Step Goal

Implement and machine-validate `History + G_t^{PI} → Z_t^{PI,L_g}` only, stopping before world-model state or dynamics.

## Definition Basis

- Read-only authority: `D:\shen\OB\科研\PIJWM\03物理-信息双图建模.md`.
- SHA-256: `6f3e17b0691aa81c60c2ea1e76a1031d2f7f622abf4b40884e9ae4380953c15e`.
- Relevant sections: §5.1–§5.4 and §6.1.
- Frozen engineering baseline: `7d9b015169188dd26eadef02beebd24ced922ccb`.

## Initial State

STEP 4.3A already materialized eleven typed current graph blocks, but had no feature encoder, temporal aggregation, message passing, cross coupling, or learned latent. Frozen non-Flow Tensor arrays named `*_features` were byte-equal to raw values, so they could not truthfully be assumed pre-normalized.

## Files Involved

- `code/src/pi_jwm/step4_3b_dual_graph_encoder_v1.py`
- `code/scripts/build_step4_3b_dual_graph_encoder_v1.py`
- `code/tests/test_step4_3b_dual_graph_encoder_v1.py`
- `docs/contracts_PIJWM_STEP_04_3B_DUAL_GRAPH_ENCODER_V1.md`
- `code/artifacts/protocols/pi_jwm_step4_3b_dual_graph_encoder_v1_20260920/`
- Tracker, authority/process records, AI_CONTEXT, indexes, and changelog.

## Changes and Reuse

- Reused frozen History slots, typed current graph blocks, stable indices/masks, and frozen train-only statistics without changing upstream schemas.
- Added mask-explicit type-specific MLPs, independent Physical/Agent/Task/Flow presence-gated GRUs, and current-only Physical/Comm relation encoders.
- Kept Logical Flow and Carrying branches independent until FlowFuse; excluded delivered/Epoch/indices from learned numeric input and retained Carrying structural state.
- Added five typed directed relation processors, direction embeddings, relation-wise masked means, independent node updates, P2A Align gates, and wireless-only P2C GeoComm gates.
- Added additive train-only statistics only for STEP 4.3A-derived Physical relation features.
- Produced a deterministic CPU state dict and aligned output solely as untrained development evidence.

## Validation

- Focused: `python -m unittest discover -s code/tests -p 'test_step4_3b_dual_graph_encoder_v1.py'` → 19/19 passed.
- Frozen regressions: STEP 4.3A 15/15, STEP 4.2C-C 23/23, STEP 4.2A 17/17, STEP 3.3 8/8 passed.
- Artifact builder: 37/37 required checks and 17/17 negative/counterfactual checks passed. The receipt top-level `passed` is the logical AND of input audit, contract/architecture checks, fixtures, required checks, and explicit scope.
- Deterministic double rebuild: manifest SHA-256 remained `c24c26b314a38b96c959264e06087001ed708f104f108642879363ab4eda3b2d`; acceptance SHA-256 remained `a5305387b90e73723166c67930e9e4bb15929ffbd4cf8f8b035b90ff7ad016d0`.
- Source lineage: 4.3A manifest tensor SHA equals the actual 4.2C-C tensor SHA `adf6ab22fcdc2d6338bec0ab5ef7db11e595981fe4d3f1e260c68d4e5756155f`.
- Serialize/load reproduces the semantic output digest; CPU `latent.sum().backward()` produces finite gradients with no optimizer.
- Final `compileall`, knowledge-index write/check, stale-state scan, and `git diff --check` are run after the final documentation edit.

## Results

The machine evidence establishes the requested History + current typed graph → aligned `Z_t^{PI,L_g}` path. All 37 required receipt checks and all 17 negative/counterfactual fixtures pass. The artifact is explicitly untrained development evidence.

## Expected vs Actual

Expected: an entity/relation-aligned `Z_t^{PI,L_g}` with typed temporal and graph encoding, one-way Physical→Information coupling, and no future/dynamics leakage. Actual: implementation and machine checks match that boundary; no world-model or performance claim is made.

## Known Issues

- Physical topology `radius_knn/radius=1000m/k=2` remains a deterministic development config, not a frozen research choice.
- Agent temporal input is limited to frozen static CPU capability, type, presence, and masks; dynamic available CPU/storage/queue load are unavailable and not fabricated.
- Task return size/priority/deadline and physical heading/elevation/posture are not in the frozen Tensor and are not consumed.
- Return multi-hop, same-destination partial-hop reroute runtime, and formal graph capacities retain their prior evidence boundaries.
- The encoder is untrained; representation quality and downstream prediction/performance are untested.

## Git

Implementation commit: `9c45b32` (`feat(graph): freeze dual-graph encoder contract`). A documentation-only closure commit records the post-commit status; both are pushed to `origin/main`.

## Next Step

Only recommend **Definition 04 — World Model Representation / Dynamics Contract**. The researcher will choose the exact Step name; do not execute automatically.

## 2026-09-20 STEP 4.3B-PATCH — Cross-Processor Formula & Z_PI Structural Interface Closure

### Scope

This patch closes the Definition 03 encoder formula and output-interface gaps only. It does not enter World Model, RSSM, dynamics, prediction, Loss, Planner, Training, GPU, locked_test, or formal Dataset.

### Changes

- P2A value now consumes `[aligned_physical, aligned_agent]`; P2C value now consumes `[geo_source, geo_target, comm_latent]`. Gate and value processors remain parameter-independent.
- The output structural interface now preserves all eleven STEP 4.3A blocks, including physical/agent/task nodes, all five relation families, Flow Carrying state, Align, and GeoComm. It is copied side information, not learned numeric input.
- Structural endpoint/index/presence/validity equality is checked against the source graph. Structural tampering is a negative fixture.
- `n_comm_rb` is read from the frozen tensor contract and checked against actual CSI width; no `50` source constant remains.
- Semantic digest and serialize/load acceptance now cover latent, structural, contract, and diagnostics output.

### Validation

- Focused 4.3B tests: 21/21 passed.
- Rebuilt artifact: all required checks and negative/counterfactual checks passed; `passed=true`.
- Evidence class remains `UNTRAINED_DEVELOPMENT_ENCODER_EVIDENCE`; CPU autograd only, no optimizer.

### Result and boundary

The implementation now matches the frozen P2A/P2C joint-context formulas and exposes a complete aligned `Z_t^{PI,L_g}` structural interface. This is wiring and semantic acceptance evidence, not representation-quality or performance evidence. `training=false`, `gpu=false`, `locked_test=false`, and `formal_dataset=false`.

### Git

This patch is pending final regression, index, diff, commit, and push in the current task.
