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
- Focused tests: PATCH3 suite 30/30 passed, including the canonical real adapter unknown-Return path, Return tri-state, valid/invalid/multiple-predecessor DAG behavior, and partial/intermediate/terminal Flow status transitions.
- Artifact builder: PATCH3 formal receipt 92/92 required checks (50 original + 21 service + 21 structural/rule checks), zero failed; top-level `passed` is validated from computed semantic counterfactuals plus explicit scope booleans. The recursive receipt uses a contract-valid negative-index no-op on step two when step one has completed a Flow, so absent Flow references remain rejected.
- Frozen real Encoder output is used for latent initialization; the canonical wired capacity `0.00001 Mbps` is read from its real non-locked source trajectory config.
- Real `WiredNetworkManager` equality fixture: derived and simulator active memberships/counts match.
- Deterministic expectation rollout, seeded sampled outage replay, state-dict reload, CPU autograd, immutable input, and machine provenance are included in the artifact.
- Independent rebuild produced six byte-count/SHA-256-identical files. Final hashes include rollout `dd8fb89e6a38535b171e83c994f12edfdb03b5acdf57f150c15124a9cd129868`, package `c015a169dc82322417bbc01653df4fa695d4d2ab044b9820e39f0eb895161db1`, and manifest `6dd93eddd87b8f169a396b193b0ea056e9b14f7b9df04e2ca736616bc4f13c79`.
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

STEP 4.4-PATCH3 is COMPLETE / FROZEN after focused 30/30, related regression 82/82, compileall, formal receipt 92/92, deterministic six-file comparison, registry, diff, and Git closure. The exact final commit and push synchronization are reported in the completion response.

## Next Step

Only recommend **Definition 05 — World Model Loss / Training Contract**. Do not execute it automatically.

## STEP 4.4-PATCH2 — Carrying, Route Rebinding & Task-DAG Transition Closure

- Goal: close hop-local carrying progress, post-rule Flow→Comm rebinding, typed Flow semantics, Task completion/Return boundary, and dynamic DAG release without changing Raw, Tensor, graph schema, or training.
- Changes: service is capped by `min(raw_service, hop_remaining, flow_remaining)`; partial progress accumulates and resets only on intermediate hop advancement; terminal completion synchronizes Flow presence and carrying state. Rebinding occurs after hop/route rules and validates source, destination, presence, and validity. Route revision increments only on an actual hop change; it remains structural bookkeeping and is not a learned scalar. Flow type/status embeddings now enter Flow state features. Task completion uses the frozen lifecycle vocabulary and optional Return Flow gate; dynamic predecessor completion produces `task_released`/`dag_satisfied` state.
- Validation: focused tests 26/26; builder receipt 91/91 (50 original + 21 service + 20 structural); compileall, deterministic artifact rebuild, and scope remain CPU-only with `training=false`, `gpu=false`, `locked_test=false`, `formal_dataset=false`.
- Existing Return boundary: the adapter now carries frozen 4.3A `task_index` and `flow_type_index`, and binds only the unique current-support `(TaskIndex, Return)` Flow. Input and another Task's Return cannot substitute. A mapped Return must complete under the frozen Flow semantics before final Task completion.
- Future birth boundary: `future_return_birth_supported=false`. Future Target/route never creates a slot. When explicit current side state requires Return but no current-support Return exists, `return_birth_required` and `final_completion_blocked_by_fixed_support` are set while `task_completed` remains false and the frozen lifecycle vocabulary is unchanged. Definition 05 must mask, exclude, or explicitly classify such windows; Loss is not implemented here.

## STEP 4.4-PATCH3 — Return Requirement, Dynamic DAG & Flow Completion Status Final Closure

- Goal: close the three remaining source/contract mismatches without modifying Raw, STEP 4.2C-C Tensor, STEP 4.3A Graph, stochastic boundary, learned heads, or training scope.
- Return source fact: frozen current-side inputs do not expose `Task.return_size`. The real adapter therefore uses a tri-state side contract: an existing typed Return proves `known=true/requires=true`; no typed slot is `known=false`, not known-no-Return. Computation-finished unknown state is conservatively unresolved and blocked, while known-required/no-slot separately requests unsupported Return birth.
- DAG rule: release and satisfaction are recomputed from valid incoming edges only. Machine cases prove root release, incomplete predecessor blocking, completed predecessor release, invalid-edge isolation, and the all-valid-predecessors requirement.
- Flow completion: `FLOW_STATUS_VOCAB` supplies the `COMPLETED` index. Only terminal logical completion synchronizes remaining, presence, carrying activity, and status; partial and intermediate hop cases remain non-completed.
- Acceptance strength: the builder now computes real state changes for Return, DAG, Flow status, RouteRevision, categorical embedding, and raw-index exclusion checks rather than accepting field/module/policy-string existence.
- Validation: focused 30/30, related regression 82/82, compileall, formal builder receipt 92/92, and six-file independent hash/size equality pass. No Definition 05 work was started.
