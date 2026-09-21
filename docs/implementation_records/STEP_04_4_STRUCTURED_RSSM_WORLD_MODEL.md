# STEP 4.4 — Structured RSSM World Model Contract

## Step Goal

Implement and freeze the untrained structured RSSM mechanism from frozen `Z_t^{PI,L_g}` through action-conditioned recursive prior rollout, known stochastic communication events, deterministic system transition, and dynamic graph rebuild.

## Definition Basis

- Read-only authority: `D:\shen\OB\科研\PIJWM\04世界模型预测边界与当前模型.md`, SHA-256 `ef49cd0802a163886ae324879c01e9fbbd3f3f2dc1ec4079e4022b2c4f7a46c5`.
- Researcher authorization: original STEP 4.4 plus the 2026-09-21 decision freezing wireless outage as an independent known stochastic service event, `learned_service_residual=false`, and permitting minimal causal wired capacity/membership evidence.
- Pre-gate audit commit: `30e34cb2867207ecfd636eb84097413cf81bb6dc`; its stop verdict was resolved by the explicit researcher decision, not silently reinterpreted.

## Initial State

STEP 4.3B produced aligned `Z_t^{PI,L_g}`. The communication audit proved nominal wireless rate recoverable but stopped because the future outage draw was not a Decision input. No current-definition `xi_t^Lat`, action routing, recursive prior transition, or predicted-state graph rebuild existed.

## Files Involved

- `code/src/pi_jwm/step4_4_structured_rssm_world_model_v1.py`
- `code/scripts/build_step4_4_structured_rssm_world_model_v1.py`
- `code/tests/test_step4_4_structured_rssm_world_model_v1.py`
- `code/artifacts/protocols/pi_jwm_step4_4_structured_rssm_world_model_v1_20260921/`
- `docs/contracts_PIJWM_STEP_04_4_STRUCTURED_RSSM_WORLD_MODEL_V1.md`
- Existing Step 4.4 communication audit source/artifact remain preserved as pre-decision evidence.

## Changes

- Added five aligned deterministic recurrent states and stochastic Physical-vehicle/Communication states with diagonal-Gaussian prior/posterior and clamped log standard deviation.
- Added local validated routing for Route, Comm, Comp, and Mobility; no action is globally broadcast.
- Added independent future dynamics processors for Physical, Comm, Flow, Task-Agent, DAG, P2A, and P2C plus explicit predicted-state feedback.
- Limited learned decoders to vehicle delta/next speed and per-RB CSI.
- Implemented the audited complete-allocation SINR, exact nominal-rate and outage-probability formulas, explicit seeded Bernoulli event, and marked deterministic expectation mode.
- Read wired capacity from the real trajectory simulator config; proved Flow Carrying-derived active membership/count equal to a real `WiredNetworkManager` instance; retained exact Mbps-to-bytes conversion.
- Added vehicle/UAV/static physical rules, Flow cap and terminal-hop conservation, CPU/Task rules, and per-step Physical/Comm/Flow/Task/DAG/Align/GeoComm graph rebuild.
- Added a two-step prior-only rollout whose second step consumes its own first predicted state/graph. Future Target is accepted only as a counterfactual argument and is never read.

## Reuse

Reused the frozen STEP 4.3B encoder package and verified diagonal-Gaussian/GRU implementation patterns. Historical latent layout, prediction heads, static graph rollout, mixed physical-edge semantics, and old task-only action route were not reused.

## Validation

- TDD red: focused test failed with `ModuleNotFoundError` before the new module existed.
- Focused tests: 20/20 passed after implementation, covering formulas, explicit RNG, complete Comm allocation write, local routing, invalid index rejection, posterior/prior separation, recursive feedback, dynamic graph, route/identity semantics, intermediate-hop conservation, serialization, negative receipt logic, absent-relation CSI-mask protection, graph depth, carrying state, and hop advancement.
- Artifact builder: 87/87 required checks (50 original + 21 service + 16 structural/rule checks), zero failed; top-level `passed` is validated from the logical AND plus explicit scope booleans. The recursive receipt uses a contract-valid negative-index no-op on step two when step one has completed a Flow, so absent Flow references remain rejected.
- Frozen real Encoder output is used for latent initialization; the canonical wired capacity `0.00001 Mbps` is read from its real non-locked source trajectory config.
- Real `WiredNetworkManager` equality fixture: derived and simulator active memberships/counts match.
- Deterministic expectation rollout, seeded sampled outage replay, state-dict reload, CPU autograd, immutable input, and machine provenance are included in the artifact.
- Independent double rebuild produced identical files; final rollout file SHA-256 `892de87e79326f4ec02cd5a461667f80252304bbddefd348a4b2ebc97b8a16bb`, package SHA-256 `dfead8977bb95b9d082dd03762a9a6040bc562403bfb401b6abf290bba997684`, manifest SHA-256 `327d186b9ac64c4a0c0dcc2f8de4e0e227979b3b750a0e474ac0fc866a44e221`.
- Structural closure checks typed categorical embeddings, effective graph depth, 4.3A physical topology reuse/no self edges, carrying/hop state and advancement, dynamic Flow-Comm rebinding, completion/presence synchronization, lifecycle/DAG/endpoint validity rules, and a state-changing recursive counterfactual.
- Final focused/regression tests, compileall, deterministic double rebuild/hash, knowledge-index write/check, and `git diff --check` are recorded after final documentation synchronization.

## Results

The untrained mechanism now executes `Z_t^{PI,L_g} + A_{t:t+L-1} -> predicted S_{t+1:t+L}` with structured prior/posterior state, locally routed actions, learned unknown dynamics, a known stochastic outage event, deterministic service/Flow/Task rules, predicted-state feedback, and dynamic graph rebuild. No outage/rate/service residual head exists.

This is architecture and causal-mechanism evidence only. It does not establish learned prediction accuracy, calibration, stochastic quality, planning quality, or performance.

## Expected vs Actual

Expected and actual agree for the authorized v1 boundary. The earlier audit blocker was closed by the researcher-selected known-stochastic-event design. Active wired membership required no new upstream membership field because Carrying-derived membership equaled the simulator; wired capacity remained a minimal causal config extension in the World Model state adapter.

## Known Issues

- Development topology `radius_knn/radius=1000m/k=2` and all model sizes are not research-frozen.
- The canonical artifact is non-locked, CPU, untrained development evidence; `formal_dataset=false`.
- Return multi-hop and same-destination partial-hop reroute runtime evidence remain at their previously documented strength.
- Definition 05 loss, optimization, training protocol, uncertainty calibration, and performance remain unimplemented.

## Git

Pending final validation, commit, and push for STEP 4.4-PATCH. The final completion receipt records the resulting commit and branch.

## Next Step

Only recommend **Definition 05 — World Model Loss / Training Contract**. Do not execute it automatically.
