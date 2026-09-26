# STEP 6.0B — Planner Action Feasibility Source Audit

## 1. Research Question and STEP 6.0A stop reason

Determine whether a Planner can causally know dynamic available CPU at a decision and which UAV numerical control bounds AirFogSim actually enforces. STEP 6.0A intentionally marks both `dynamic_available_cpu` and `mobility_numeric_bounds` UNKNOWN. Removing either mark without source evidence would change the action-feasibility method.

## 2. Exact source identity and initial state

PI-JWM start: `2a9c06aa7737d112722b0d1f53d239ce828f9107`, equal to freshly fetched `origin/main`. Existing untracked `TASK/` and `code/scripts/plot_step5_3e_tiny_overfit.py` were left alone. `code/reference/AirFogSim/` exists, but **has no independent `.git`**. `git -C code/reference/AirFogSim rev-parse HEAD` and `remote -v` walk upward and report the PI-JWM repository, so those values are *not* AirFogSim identity. Branch/tag and clean/dirty status for AirFogSim cannot be asserted. Relevant local source files are identified by SHA-256 in the five machine receipts. Official upstream [AirFogSim](https://github.com/ZhiweiWei-NAMI/AirFogSim) commit `76c0edb4...` matches only 3 of 5 sampled local Git blobs; local `task_manager.py` and `airfogsim_env.py` differ. This commit is **not** claimed as local source SHA. `airfogsim_source_sha=null` records the missing Git provenance, while `airfogsim_local_relevant_source_sha256` fingerprints inspected bytes.

## 3. CPU source chain and four distinct meanings

| Meaning | Source and current PI-JWM representation |
| --- | --- |
| Static capacity `C_static` | Scenario `examples/config.yaml::fog_profile.*.cpu`; `FogNode.getFogProfile()['cpu']`, runtime per-node when key exists. Raw `decision.node_cpu_capacity_observation_rows[].capacity_per_s` includes mask/missing reason. Sample `static.agent_static_capability[].cpu_capacity_per_s`; Tensor `agent_cpu_capacity_raw/agent_cpu_capacity/agent_cpu_capacity_mask`; Graph `agent_nodes.cpu_capacity/cpu_capacity_mask`. Unit: AirFogSim CPU work units/s. |
| Comp request `A_t^Comp` | PI-JWM action row `task_id,node_id,allocated_cpu_per_s`; collector installs a callback through `ComputationScheduler.setComputingCallBack`. It is a request, not remaining capacity. |
| Actual service | `AirFogSimEnv._updateComputation → TaskManager.computeTasks → Task.compute`; each nonzero callback allocation yields `allocated_cpu * simulation_interval`, capped at task's remaining total work. Raw Outcome/progress reflects past execution, not current available capacity. World Model's deterministic rule similarly subtracts `max(Comp,0)*slot_duration` from `task_work_remaining`. |
| Dynamic available `C_available(t)` | Absent from inspected simulator decision state and PI-JWM Raw/Sample/Tensor/Graph/World Model state. The callback can be chosen at execution, with no decision-time reservation/busy/remaining CPU field. No strict causal formula from currently represented variables is established. |

The simulator's `TaskManager.computeTasks` does **not** sum allocations by node against `FogProfile.cpu`, nor reject, clip, normalize or queue an over-capacity callback. It passes per-task values directly to `Task.compute`. **Oversubscription behavior: `ALLOW_OVERSUBSCRIPTION`**, subject only to per-task work completion cap. The PI-JWM formal collector separately scales/validates its own allocations against static capacity in `_capacity_respecting_comp`; its rule is not a simulator enforcement rule and does not establish dynamic availability. The older PI-JWM Raw action validator also applies a static-capacity sum check, likewise distinct from native simulator behavior.

**CPU verdict: `STATIC_CAPACITY_ONLY`.** No `CPUFeasibilityEvidence` or candidate code change is justified. `dynamic_available_cpu=UNKNOWN` remains. A static capacity number, an action request and prior actual service cannot be substituted for `C_available(t)`.

## 4. UAV source chain, units and bounds

`TrafficScheduler.setUAVMobilityPatterns(env, {uav_id: {angle,phi,speed}})` stores the control; `AirFogSimEnv._updateTraffics` passes it to `TrafficManager.updateUAVMobilityPatterns`; `TrafficManager.stepSimulation` updates position. In executable code, `angle` is azimuth from +x toward +y in radians; `phi` is elevation from the xy plane in radians; `speed` multiplies `traffic_interval` to produce displacement, so operational unit is distance/s. The example config comment says “distance unit per timeslot”, which conflicts with executable multiplication by seconds; the code and PI-JWM `speed_mps` mapping are the implementation evidence. Current example uses `traffic_interval=simulation_interval=0.1s`.

The direct setter and stepSimulation contain no numerical reject, clamp or wrap for azimuth, elevation, speed, acceleration, altitude or x/y position. Out-of-grid x/y merely loses grid membership in `_update_map_by_grid`; it does not stop motion. Example `UAV_speed_range=[10,30]` is used by generation helpers, not direct setter validation. `UAV_z_range=[100,200]` initializes positions; x/y map ranges initialize/build the spatial grid. `nonfly_zone_coordinates` is used by target selection helper, not as a direct mobility-action gate. These are **scenario/config settings**, not universal simulator hard action bounds. No candidate hard constraint was added.

Formal Dataset behavior policy in `collect_step5_5_formal_raw_v1.py` uses speed choices `0,5,8,10,12,15` m/s, azimuth `current angle + {-0.2,-0.1,0.05,0.1,0.2}` on interventions, and unchanged `phi`. A read-only audit of 60/60 accepted Raw trajectories checked all listed SHA-256 values (0 mismatches) and 11,520 recorded UAV action rows: observed azimuth `[-1.25, 2.0500000000000007]` rad, elevation `0` rad, speed `[0,15]` m/s with six distinct values. This is **`DATASET_BEHAVIOR_SUPPORT_ONLY`**, not simulator legality or chosen Planner domain.

Verdicts: azimuth `DATASET_SUPPORT_BOUND_ONLY`; elevation `DATASET_SUPPORT_BOUND_ONLY`; speed `CONFIG_BOUND_FOUND` (generation range, not enforced); spatial/geofence `CONFIG_BOUND_FOUND` (initialization/grid/target helper, not enforced). No `SIMULATOR_HARD_BOUND_FOUND` for any variable. All numerical Planner bounds remain `UNKNOWN`/Research Pending.

## 5. World Model vs simulator mobility semantics

Simulator and Structured RSSM positions both use `dt * speed * [cos(angle)cos(phi), sin(angle)cos(phi), sin(phi)]`; angle mapping and current 0.1-s step agree. Neither path clamps the numeric control. Simulator raw UAV acceleration computes `(old_speed - new_speed)/dt`; PI-JWM deliberately defines a separate **canonical** acceleration `(new_speed - old_speed)/dt` in `raw_trajectory_causal_contract_v1.py`, and the World Model uses this canonical sign. Thus raw simulator acceleration and model canonical acceleration differ, but the Raw contract already labels them separately. No World Model edit was made. Future Planner action bounds and operational spatial safety still require researcher definition.

## 6. Implementation, evidence and acceptance

No Candidate/Constraint, World Model, training, config, dataset or simulator source changed. Added read-only audit builder `code/scripts/build_step6_0b_planner_action_feasibility_audit_v1.py`, focused receipt tests, this record and five machine receipts under `code/artifacts/protocols/pi_jwm_step6_0b_planner_action_feasibility_audit_v1_20260926/`. The builder checks source symbol anchors and hashes; `--check` reproduces receipts. Runtime probe: **false**, because the setter, CPU execution and UAV update are explicit in source. These are local simulator mechanism/source facts, not PI-JWM Planner performance evidence. The missing independent Git metadata remains a provenance limitation and prevents an exact AirFogSim Git SHA claim.

Actual verification: `python code/scripts/build_step6_0b_planner_action_feasibility_audit_v1.py --check` passed 5/5 receipts; focused 6.0B unittest 4/4, 6.0A 6/6, STEP 4.4 30/30, STEP 5.5 11/11, STEP 5.4 5/5 and STEP 5.1D 5/5 passed. `python -m compileall -q code/src code/scripts code/tests`, `git diff --check`, and knowledge-index write/`--check` passed. Expected-versus-actual: source facts were determined, neither UNKNOWN was closed, and exact local AirFogSim Git identity remained unavailable. The machine acceptance status is `SOURCE_FACTS_COMPLETE_GIT_PROVENANCE_LIMITED`, not a frozen feasibility contract.

## 7. Research Pending and STEP 5.6B isolation

Researcher decisions still required: dynamic CPU feasibility policy or new causal observable, Planner UAV speed/angle/spatial domain, behavior-support/OOD treatment, and any safety/geofence claim. Search/Learned/Hybrid remains pending. STEP 5.6B remote was not contacted; no SSH, GPU, checkpoint, candidate rollout, objective, baseline, closed loop or `locked_test`. No formal training file or AirFogSim file changed. Next minimal action after this audit: resolve source Git provenance if an exact historical AirFogSim commit is required; otherwise wait for 5.6B best checkpoint and a separate researcher authorization before World Model-dependent Planner work.
