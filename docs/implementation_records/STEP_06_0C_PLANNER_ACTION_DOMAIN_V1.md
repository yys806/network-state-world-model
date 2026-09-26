# STEP 6.0C — Planner Action Domain v1

## Research question, basis and initial state

Can the researcher freeze an executable first-rollout action domain without pretending that unavailable simulator constraints exist? Definition basis is the authorized Definition 06 Planner chain and this Step's explicit researcher decision. Initial Git HEAD after fetch: `1c24fc4b4f919b06352ad86f94174dc0020682cb` = `origin/main`; the pre-existing untracked `TASK/` and plot script were left alone. STEP 6.0A provided generic candidates and a formal adapter but blocked Comp/Mob on UNKNOWN; STEP 6.0B recorded `STATIC_CAPACITY_ONLY`, native CPU oversubscription, and no UAV hard numeric bound. AirFogSim's local independent Git SHA remains unavailable; the 6.0B source receipt carries relevant file hashes. This Step consumes those facts, does not rewrite them.

## Researcher decisions and causal sources

Comp now uses **STATIC_PER_SLOT_BUDGET_V1**: per node and slot, nonnegative raw-unit allocations sum to no more than observed static capacity. Missing capacity plus positive request is rejected. This is Planner operational policy, not native AirFogSim behavior or dynamic available CPU. Raw current `node_cpu_capacity_observation_rows` is the runtime source; causal Sample `agent_static_capability.value` is cross-checked; `agent_cpu_capacity_raw`/mask and Graph `agent_nodes.cpu_capacity`/mask are corresponding representations. Normalized CPU is excluded.

Mobility now uses **FORMAL_DATASET_MOBILITY_CORE_DOMAIN_V1**: the six exact marginal profiles and explicit HOLD in the [contract](../contracts/PIJWM_STEP_06_0C_PLANNER_ACTION_DOMAIN_V1.md). Current Raw UAV heading must explicitly be radians; elevation must be observed. The state stays Planner-only, outside frozen Model input. The collection source is `code/scripts/collect_step5_5_formal_raw_v1.py`; formal behavior support became operational domain only through this researcher decision. Different UAVs may choose different supported profiles but receive the marginal-composition label. No wrap/clamp or spatial/geofence rule was invented.

## Implementation and reuse

New `code/src/pi_jwm/step6_0c_planner_action_domain_v1.py` contains raw-unit evidence, current UAV control side-state, H1–H4 domain validator, provenance annotation, cross-backend pool admission with explicit rejection records, and explicit HOLD `RULE_FALLBACK`. A small `compile_candidate` extension in the generic 6.0A module revalidates the supplied domain context before compiling and keeps explicit UNKNOWNs blocked. The actual `build_step5_1d_unified_model_chain_v1.py::build_action` and its 11 tensor semantics are unchanged. No Formal Dataset, normalization, Encoder, RSSM, loss, training, config or AirFogSim source was modified. No World Model prior or decoder is imported.

## Validation, expected versus actual

`code/tests/test_step6_0c_planner_action_domain_v1.py` uses **SYNTHETIC_CONTRACT_EVIDENCE**. Its 12 tests cover budget under/equal/over capacity, per-node sums, missing capacity, no-op, negative request, raw-unit and Raw frame/time enforcement, six profiles, unsupported speed/angle/elevation, missing side-state, UAV rad-unit gate, multiple-UAV support labels, no wrapping, H4 control update, explicit fallback, backend pool admission and exact four-family 11-tensor equality against the current adapter. No simulator performance or Planner rollout is inferred. The STEP 6.0B test was adjusted to verify its candidate source hash against the frozen `1c24fc4` Git snapshot; the historical 6.0B receipt was not changed. Exact commands/results and receipt build/check are recorded in `progress.md` and machine acceptance receipt. Expected: a frozen operational domain with no fabricated native constraints. Actual: that scope was implemented; simulator dynamic availability remains absent, spatial/geofence remain pending, and exact AirFogSim Git identity remains limited to local source hashes.

Actual CPU commands/results: `python -m unittest discover -s code/tests -p 'test_step6_0*.py'` (22/22), `test_step5_1d*.py` (5/5), `test_step4_4*.py` (37/37), `test_step5_5*.py` (21/21), all with repository `code/src;code/scripts` on `PYTHONPATH`. `python code/scripts/build_step6_0c_planner_action_domain_v1.py` and `--check` generated/rechecked 5/5 receipts. `python -m compileall -q code/src code/scripts code/tests`, `git diff --check`, and the project knowledge index write/`--check` passed (5 outputs, no mismatches). Context Consistency Check: `00` current state, `02` architecture, `03` data flow, `04` module map, `06` researcher decisions, `07` known issues, and `08` changelog were reconciled with this contract and the 6.0B source audit. The only pre-existing untracked files remain outside the commit.

## Known limits, pending research and isolation

This domain only validates current-support actions and command-side evolution. It has no future physical feasibility or safety proof and no claim of optimality. Search/Learned/Hybrid method, expansion outside six profiles, nonzero elevation, workspace/geofence, OOD policy, objective/risk and final fallback safety remain researcher decisions. STEP 5.6B remote was not contacted; no SSH, GPU, checkpoint, formal-training edit, World Model candidate rollout, objective, proposal training, baseline, closed-loop run or `locked_test` occurred. The next Step requires separate authorization after the formal best checkpoint is determined.

## Git and next step

Commit/push identity is given by `git log -1` after validation. Stop after this Step; wait for the formal World Model best checkpoint and researcher authorization before model-dependent Planner work.
