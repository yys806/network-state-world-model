# STEP 4.4 — Structured RSSM World Model Contract

## Step Goal

Implement the authorized structured RSSM only after the mandatory communication-service sufficiency gate permits implementation.

## Definition Basis

- Read-only authority: `D:\shen\OB\科研\PIJWM\04世界模型预测边界与当前模型.md`.
- SHA-256: `ef49cd0802a163886ae324879c01e9fbbd3f3f2dc1ec4079e4022b2c4f7a46c5`.
- Authorized task text: STEP 4.4 sections 19–22 and 45 require `Rule → Add Missing Causal State → Learn Residual`, with an immediate stop on residual evidence.

## Initial State

STEP 4.3B produces aligned `Z_t^{PI,L_g}`. No current-definition World Model, action-conditioned structured latent transition, deterministic feedback loop, or dynamic future graph rebuild existed. Historical RSSM code remains archived reuse evidence only.

## Files Involved

- `code/src/pi_jwm/step4_4_communication_service_audit_v1.py`
- `code/scripts/build_step4_4_communication_service_audit_v1.py`
- `code/tests/test_step4_4_communication_service_audit_v1.py`
- `code/artifacts/protocols/pi_jwm_step4_4_communication_service_audit_v1_20260921/`
- `docs/contracts_PIJWM_STEP_04_4_COMMUNICATION_SERVICE_AUDIT_V1.md`
- Tracker, authority/process records, AI_CONTEXT, registries, and indexes.

## Changes

- Added a machine-readable dependency matrix for CSI, RB allocation, bandwidth, transmit power, interference, noise, fast fading, outage, wired capacity, wired active-flow count, slot duration, Flow remaining, and transfer activation state.
- Added source-hash and symbol-level provenance for the exact AirFogSim rate, outage, execution, wired service, Decision observer, and Outcome collector paths.
- Added a computed three-way verdict and validator. Verdict tamper and outcome-to-input promotion are negative fixtures.
- No World Model source or learned residual was implemented because the mandatory stop gate fired.

## Reuse

Reused only verified source-audit patterns and existing Raw/Outcome causal boundaries. Historical RSSM layout, heads, static rollout graph, and task action routing were not reused.

## Validation

- TDD red: focused test initially failed with `ModuleNotFoundError` before the audit module existed.
- Focused audit tests: 7/7 passed after implementation.
- Artifact builder returned `passed=true` and `SERVICE_RESIDUAL_RESEARCH_DECISION_REQUIRED`.
- STEP 4.3B regression: 21/21 passed; Raw causal regression: 4/4 passed.
- Deterministic double rebuild: audit SHA-256 `af1ce2ca5ac8586e30e328717e8367ce5d090695791a475fcabbce0e4c97b9b7`; manifest SHA-256 `25c315525fb1e2c4280af5455a2c45b745a18a9cd6a9cd0b1d404881fd0313d1`.
- Final compileall, relevant regressions, deterministic rebuild/hash, knowledge-index check, and diff check are recorded after final documentation edits.

## Results

Nominal pre-outage wireless rate is rule-recoverable. Actual wireless service is not uniquely recoverable because AirFogSim samples a per-RB outage realization after the Decision input; that realization zeroes rate and is only recorded as outcome evidence. Wired capacity and active-flow competition are simulator-available but frozen Raw/Tensor does not expose them.

## Expected vs Actual

Expected: continue only for `SERVICE_RULE_SUFFICIENT` or after a minimal additive extension. Actual: the audit found a true unobservable actual-service factor and produced the authorized stop verdict.

## Known Issues / Blocker

Researcher decision required: choose and freeze one of (a) include outage/effective channel state in the communication stochastic target, (b) model a separate stochastic service event, or (c) define a learned residual target and architecture. Codex has not selected among them.

## Git

Pending final validation, commit, and push in this task.

## Next Step

Researcher decision on the communication service uncertainty boundary. Do not resume STEP 4.4 implementation or enter Definition 05 automatically.
