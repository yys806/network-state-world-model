# STEP 6.0C — Planner Feasible Action Domain v1

**Status: researcher-frozen Planner v1 operational domain.** This is the conservative first formal-rollout action domain, not the final optimal action space, AirFogSim native legality, or a safety certification. No World Model rollout or Planner objective has run.

## Why this domain exists

STEP 6.0A represented actions but rightly blocked nonempty Comp/Mob compilation on two UNKNOWNs. STEP 6.0B established source facts: only static CPU capability is causally observed; native AirFogSim computation can oversubscribe; direct UAV control has no numerical hard action bounds. The researcher now chooses an explicit operational domain based on current causal inputs and formal training action support. These decisions do **not** change the simulator facts.

## Comp: `STATIC_PER_SLOT_BUDGET_V1`

For every current node `n` and decision slot `t`, each request `f_{i,n,t} >= 0` is in raw `AirFogSim CPU-work-unit/s`, and `sum_i f_{i,n,t} <= C_static(n)`. The sum is over all Comp rows for that node in that action step; nodes and steps are independent. `C_static` comes from current Raw `node_cpu_capacity_observation_rows` with `observed_mask=true`. The implementation cross-checks causal Sample `static.agent_static_capability[].cpu_capacity_per_s.value` when present. Tensor `agent_cpu_capacity_raw`/mask and Graph `agent_nodes.cpu_capacity`/mask carry the same static capability; normalized `agent_cpu_capacity` is never a physical budget. The source is `entity.getFogProfile()['cpu']`.

If the current capacity is missing, a positive allocation is `VIOLATED: STATIC_CPU_CAPACITY_UNOBSERVED`; Comp no-op remains allowed. No default, past service, other-node estimate, or normalized placeholder is used. `dynamic_available_cpu_available=false` and `planner_requires_dynamic_available_cpu=false`: the unavailable simulator variable was not resolved; Planner v1 simply does not use it for legality. The static budget is a **PI-JWM Planner operational constraint**, not an AirFogSim native check or measured available CPU.

## Mobility: `FORMAL_DATASET_MOBILITY_CORE_DOMAIN_V1`

The current Raw decision provides each UAV's `heading` with unit `rad` and `elevation_rad`; this is Planner-only causal control side-state, not a learned World Model feature or Future Target. Vehicle headings can use degrees and are never accepted as UAV controls. Missing heading/elevation yields UNKNOWN and blocks execution. No silent unit conversion occurs.

| Profile | Delta azimuth (rad) | Delta elevation (rad) | Speed (m/s) |
| --- | ---: | ---: | ---: |
| PROFILE_HOLD | 0 | 0 | 0 |
| PROFILE_1 | -0.2 | 0 | 5 |
| PROFILE_2 | -0.1 | 0 | 8 |
| PROFILE_3 | +0.05 | 0 | 10 |
| PROFILE_4 | +0.1 | 0 | 12 |
| PROFILE_5 | +0.2 | 0 | 15 |

Each action is absolute: `azimuth_action = current_heading + delta_azimuth`, `elevation_action = current_elevation`, and speed is the listed value. The values come from `collect_step5_5_formal_raw_v1.py`: intervention profile `(frame + policy_seed) % 5`, offsets/speeds above, unchanged `phi`; nonintervention sends current heading/elevation and speed zero. These values were originally **dataset behavior support**. Their promotion to Planner v1 operational domain is this explicit researcher decision. No interpolation, nonzero elevation change, angle wrap, clamp or clipping is allowed. A command outside the six profiles is `VIOLATED: OUTSIDE_PLANNER_MOBILITY_CORE_DOMAIN_V1`, meaning outside Planner v1 support, not physically illegal.

After a command the next **control side-state only** is `heading_next = azimuth_action`, `elevation_next = elevation_action`. This permits H1–H4 static sequence validation; it predicts no position, CSI, task, Flow or objective. Every current UAV needs an explicit command row. `MOBILITY_HOLD_COMMAND` is a row with current heading/elevation and speed zero; an empty mobility family is `NO_MOBILITY_COMMAND`, never HOLD. Formal collection used one profile index across all UAVs. Planner v1 permits independently supported per-UAV profiles and labels each step `EXACT_COLLECTION_SHARED_PROFILE` or `PER_UAV_MARGINAL_SUPPORT_COMPOSITION`; the second label does not claim observed exact joint dataset support.

## Integration, fallback and limits

The 6.0A generic Candidate contract and fixed-support/Route/Comm checks remain. A supplied 6.0C domain context validates before the unchanged formal `build_action` mapping to 11 tensors; unresolved explicit constraints still block compilation. The domain pool-admission gate checks candidates from any backend, records rejected seeds, and adds explicit HOLD fallback; it performs no scoring or ranking. The generic 6.0A Search fixture's empty mobility is rejected when a UAV is present. `RULE_FALLBACK` v1 has Route/Comm/Comp no-op and one explicit PROFILE_HOLD row per present UAV, requiring observed current control side-state. Its metadata states `safe=false`. It has no spatial/geofence, future-feasibility or performance claim.

Research Pending: Search/Learned/Hybrid selection; continuous/interpolated mobility and nonzero elevation; spatial workspace/geofence policy; treatment of outside-core actions; Planner objective/risk; fallback safety; final action-space ablation. The domain does not authorize model-dependent rollout, optimizer, proposal training, baseline, closed loop or `locked_test`.
