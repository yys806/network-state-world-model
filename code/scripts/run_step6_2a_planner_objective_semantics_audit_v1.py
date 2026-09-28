"""Read-only source and contract audit for Planner Objective v1.

This script does not load a checkpoint, build a dataset, rank candidates, or
run AirFogSim.  It records source-backed semantics and readiness blockers.
"""
from __future__ import annotations

import hashlib
import json
import gzip
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "code" / "artifacts" / "protocols" / "pi_jwm_step6_2a_planner_objective_semantics_audit_v1_20260928"
AIR = ROOT / "code" / "reference" / "AirFogSim" / "airfogsim"
FORMAL = ROOT / "code" / "artifacts" / "formal_dataset" / "pi_jwm_formal_dataset_v1_h2_l4_20260923_causalfix1"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name: str, value: object) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    raw_rel = Path("raw") / "formal-v1-sim-2026092302-policy-2026092402.json.gz"
    raw_path = FORMAL / raw_rel
    raw_doc = json.loads(gzip.open(raw_path, "rt", encoding="utf-8").read())
    raw_decision = raw_doc["decisions"][1]
    raw_tasks = raw_decision["tasks"]
    raw_task_keys = sorted(set().union(*(set(row) for row in raw_tasks))) if raw_tasks else []
    raw_entities = raw_decision["entities"]
    raw_entity_keys = sorted(set().union(*(set(row) for row in raw_entities))) if raw_entities else []
    raw_anchor = {
        "path": str(raw_rel).replace("\\", "/"),
        "sha256": sha(raw_path),
        "trajectory_id": raw_decision["trajectory_id"],
        "capture_event_id": raw_decision["capture_event_id"],
        "frame_index": raw_decision["frame_index"],
        "simulation_time_s": raw_decision["simulation_time_s"],
        "task_fields": raw_task_keys,
        "entity_fields": raw_entity_keys,
        "deadline_field_present": "deadline" in raw_task_keys or "task_deadline" in raw_task_keys,
        "priority_field_present": "priority" in raw_task_keys or "task_priority" in raw_task_keys,
        "arrival_time_field_present": "arrival_time_s" in raw_task_keys,
        "return_size_field_present": "required_returned_size" in raw_task_keys,
        "task_sample_count": len(raw_tasks),
        "scope": "one first legal STEP 6.1 validation anchor Raw decision only; no future frames/targets read",
    }
    source_files = {
        "task": AIR / "entities" / "task.py",
        "task_manager": AIR / "manager" / "task_manager.py",
        "environment": AIR / "airfogsim_env.py",
        "observer": ROOT / "code" / "src" / "pi_jwm" / "airfogsim_full_dual_graph_observer_v1.py",
        "tensor": ROOT / "code" / "src" / "pi_jwm" / "airfogsim_tensor_v2.py",
        "world_model": ROOT / "code" / "src" / "pi_jwm" / "step4_4_structured_rssm_world_model_v1.py",
        "flow_audit": ROOT / "code" / "src" / "pi_jwm" / "step4_2b_stateful_flow_source_audit_v1.py",
        "support_audit": ROOT / "code" / "src" / "pi_jwm" / "step5_5_fixed_support_audit_v1.py",
        "metrics": ROOT / "code" / "src" / "pi_jwm" / "airfogsim_metrics_v2.py",
    }
    hashes = {name: sha(path) for name, path in source_files.items()}
    write("01_source_semantics_receipt.json", {
        "schema_version": "PI-JWM-Step6.2A-Source-Semantics-v1",
        "status": "PASS_WITH_READINESS_BLOCKERS",
        "airfogsim_git_identity": None,
        "airfogsim_git_identity_note": "The reference directory has no independent Git identity; file SHA256 is used.",
        "source_sha256": hashes,
        "formal_raw_anchor": raw_anchor,
        "scope": {"cpu_only": True, "checkpoint_loaded": False, "training": False, "optimizer_step": False,
                  "formal_dataset_modified": False, "locked_test_accessed": False, "ranking": False,
                  "closed_loop": False, "performance_claim": False},
        "definition_sha256": {
            "00": "AC651586CADD2638C7CB8F68E80144C5BF087B62B2E1923C6D6625C5DE2B82F0",
            "01": "01478D0C61065ADCCC50CD58747CB28CDF554A5B963E07EF154B8AFCBAD43AC1",
            "03": "6F3E17B0691AA81C60C2EA1E76A1031D2F7F622ABF4B40884E9AE4380953C15E",
            "04": "EF49CD0802A163886AE324879C01E9FBBD3F3F2DC1EC4079E4022B2C4F7A46C5",
            "05": "62EEBB05E2EEEFE9EDB0038F12964F59915A7ACCE03A683D6C7E5C71F85684E9",
            "06": "F20294BD8708076AE7583BE679A8182A0990ECF6777E05DF378AE4C2855431F1",
        },
    })
    write("02_deadline_lifecycle_audit.json", {
        "status": "PARTIAL",
        "deadline_kind": "relative_allowed_latency_duration",
        "unit": "seconds in the formal scenario (simulation_interval is configured in seconds); deadline is the same elapsed-time unit",
        "absolute_deadline": "arrival_time + task_deadline",
        "violation_operator": "> in TaskManager.checkTasks hard deadline sweep; Task.wait_to_ddl uses <= as equivalent predicate",
        "completion_boundary": "returning completion in removeOffloadingTaskByNodeIdAndTaskId accepts task_delay <= task_deadline; no-return completion in checkTasks accepts task_delay <= task_deadline + 1e-5. These are distinct completion predicates, and both run before the later active-task hard-deadline sweep.",
        "check_stage": "TaskManager.generateAndCheckTasks calls checkTasks after generation; AirFogSimEnv.step calls task_manager.checkTasks after simulator updates and before clearing decisions",
        "same_step_tie": "Completion handling runs before the later active-task hard-deadline sweep. Returning completion accepts delay <= deadline. No-return completion accepts delay <= deadline + 1e-5. Any task still active after completion handling fails only when elapsed time is strictly greater than deadline. DONE wins at equality for both paths; no-return completion may also win within the explicit 1e-5 tolerance.",
        "failed_lifecycle": "removed from active collection, appended to _out_of_ddl_tasks, failure code set for the hard deadline path; FAILED is terminal observation-side collection state",
        "task_delay": "observer computes max(env.simulation_time - task arrival_time, 0); Formal Raw anchor has arrival_time_s but no elapsed_time_s, Formal Sample stores arrival_time_s and elapsed_time_s; the trained Tensor/4.4 Planner path does not expose deadline/elapsed as objective side-state",
        "lifecycle_scope": "deadline covers the Task lifetime, including offload, computation, and Return when Return is required",
        "source_anchors": {
            "Task.wait_to_ddl": "code/reference/AirFogSim/airfogsim/entities/task.py:240-249",
            "TaskManager.addToComputeTask": "code/reference/AirFogSim/airfogsim/manager/task_manager.py:235-257",
            "TaskManager.generateAndCheckTasks": "code/reference/AirFogSim/airfogsim/manager/task_manager.py:797-815",
            "TaskManager.checkTasks": "code/reference/AirFogSim/airfogsim/manager/task_manager.py:835-891",
            "AirFogSimEnv.step": "code/reference/AirFogSim/airfogsim/airfogsim_env.py:257",
        },
        "planner_causal_exposure": "BLOCKED",
        "planner_exposure_reason": "Runtime Task/observer has the fields, but accepted Formal Raw omits deadline/priority; Formal Sample preserves arrival_time_s and elapsed_time_s history, while the STEP 5.6C training Tensor/4.4 state does not provide deadline or priority to Planner scoring. The current Planner side-state contract is absent.",
        "required_next_change": "Planner-only causal side-state additive exposure; do not reconstruct from Future Target or FAILED labels.",
    })
    write("03_task_objective_cohort_audit.json", {
        "status": "PARTIAL",
        "cohort_definition": "current causally visible tasks with presence=true and lifecycle not terminal DONE or FAILED",
        "terminal_done": "World Model task_completed/final_done is true only when computation is finished and no unresolved required Return remains; lifecycle index is changed to completed index.",
        "terminal_failed": "Current anchor FAILED is represented by the frozen lifecycle vocabulary/index and can be excluded; future deadline-driven transition to FAILED is absent from one_step and cannot be forecast without causal deadline side-state/rule.",
        "computation_finished": "task_work_remaining <= 0; this is not final completion when return is required or unsupported.",
        "released": "task_released is dependency/DAG availability and is distinct from presence and completion.",
        "delay_surrogate": "u(q,h)=1 iff q is not successfully final-complete at predicted h; FAILED is not successful completion.",
        "anchor_terminal_exclusion": "DONE and FAILED are excluded from the current scoring cohort; presence alone is insufficient.",
        "future_birth_exclusion": "future-only tasks and future-only Return flows are excluded from cohort and denominator.",
        "readiness": "PARTIAL: cohort rule is defined, but deadline failure and some task fields are not planner-causally exposed.",
    })
    write("04_flow_multihop_burden_audit.json", {
        "status": "PARTIALLY_SUPPORTED",
        "source_anchors": {
            "one_step_service": "code/src/pi_jwm/step4_4_structured_rssm_world_model_v1.py:568-616",
            "hop_advance": "code/src/pi_jwm/step4_4_structured_rssm_world_model_v1.py:581-614",
            "route_revision": "code/src/pi_jwm/step4_4_structured_rssm_world_model_v1.py:626-636",
            "flow_source_audit": "code/src/pi_jwm/step4_2b_stateful_flow_source_audit_v1.py",
        },
        "verified_semantics": {
            "intermediate_hop": "hop_remaining decreases and current_hop_index advances; flow_remaining does not decrease",
            "terminal_hop": "e2e delivered reduces flow_remaining; completed Flow loses presence/carrying",
            "identity": "flow identity, FlowType, Epoch/RouteRevision and current holder are required for a stable burden calculation; current route revision is model state/action-side revision",
            "completed_or_inactive": "exclude from active burden mask",
            "unknown_padded": "exclude; never use padding as service support",
            "reroute": "recompute from predicted route/hop state only when identity and route revision remain supported",
        },
        "target_formula": "B_Tx = R_hop + (N_hop - 1 - current_hop_index) * R_e2e",
        "formula_verdict": "PARTIALLY_SUPPORTED",
        "formula_limitations": [
            "Current causal flow audit does not prove end-to-end remaining across completed hops for all Input/Return flows.",
            "Route revision and superseded Epoch identity are not fully runtime-source-backed.",
            "DepData transfer is absent in the audited simulator runtime.",
        ],
        "common_support_rule": "A candidate whose first unsupported Return state is u_k has H_sup=u_k-1; H_eff=min_k H_sup(k), and H_eff=0 is OBJECTIVE_UNSCOREABLE.",
    })
    write("05_compute_burden_audit.json", {
        "status": "PARTIAL",
        "source_anchors": {
            "cpu_rule": "code/src/pi_jwm/step4_4_structured_rssm_world_model_v1.py:616-621",
            "progress_rule": "code/src/pi_jwm/step4_4_structured_rssm_world_model_v1.py:639",
            "completion_rule": "code/src/pi_jwm/step4_4_structured_rssm_world_model_v1.py:647-657",
            "planner_domain": "docs/contracts/PIJWM_STEP_06_0C_PLANNER_ACTION_DOMAIN_V1.md",
        },
        "verified_rule": "cpu_service = allocated_cpu_per_s * slot_duration_s; task_work_remaining_next=max(current-cpu_service,0); task_progress=1-next_remaining/task_work_total",
        "target_formula": "b_Comp(q,h)=W_rem(q,h)/max(W_rem(q,0), epsilon)",
        "zero_denominator": "If W_rem(q,0)=0, the task is already computation-finished and contributes zero compute burden; no arbitrary epsilon value may be used to manufacture positive burden.",
        "pre_computing": "remaining work is a causal state quantity even before COMPUTING, but applicability must be anchor-frozen by lifecycle/support mask.",
        "return_boundary": "computation-finished is not final completion when required Return is unresolved or unsupported.",
        "mask_semantics": "anchor-defined task applicability mask; candidate cannot change denominator",
        "readiness": "PARTIAL: rule is source-backed, but deadline/complete cohort and causal capacity exposure need a Planner-only side-state contract.",
    })
    write("06_support_boundary_audit.json", {
        "status": "PARTIAL",
        "future_return_birth_supported": False,
        "trigger": "task_return_requirement_known AND task_requires_return AND return_flow_index<0 AND task_presence",
        "blocked_flags": "final_completion_blocked_by_fixed_support is set when computation_finished and return_birth_required or unresolved_return_requirement",
        "first_unsupported_state_rule": "If first unsupported state is u_k, score only H1..H_(u_k-1); the unsupported state itself is not scored.",
        "common_support": "H_eff=min candidate H_sup; no candidate may use a longer horizon; H_eff=0 => OBJECTIVE_UNSCOREABLE.",
        "source_anchors": {
            "world_model": "code/src/pi_jwm/step4_4_structured_rssm_world_model_v1.py:641-657",
            "detector": "code/src/pi_jwm/step5_5_fixed_support_audit_v1.py:7-38",
            "formal_audit": "code/scripts/audit_step5_5_patch_v1.py",
        },
        "limitation": "The detector proves fixed-support absence/blocked status; it does not create a new Return slot or make future Return prediction safe.",
    })
    write("07_throughput_metric_semantics_audit.json", {
        "status": "PARTIAL",
        "planner_role": "Diagnostic only; no independent weighted Throughput term in Planner Objective v1 because delivered service already drives hop/Flow progress and J_Burden.",
        "fields": ["nominal_rate", "outage_probability", "actual_rate", "wireless_delivered_data", "wired_delivered_data", "hop_service", "e2e_flow_progress"],
        "final_metric_target": "Q_Thr = real successfully transmitted data / real runtime",
        "metric_split": "NETWORK_SERVICE_THROUGHPUT versus END_TO_END_USEFUL_THROUGHPUT remains METRIC_SEMANTICS_PENDING when multi-hop bytes are counted at each hop.",
        "baseline_requirement": "Report which byte convention is used; do not mix carried network traffic with useful end-to-end completion.",
        "source_anchors": {"world_model_service": "code/src/pi_jwm/step4_4_structured_rssm_world_model_v1.py:568-616", "metrics": "code/src/pi_jwm/airfogsim_metrics_v2.py"},
    })
    write("08_effort_source_audit.json", {
        "status": "PARTIAL",
        "components": {
            "Comp": {"target": "sum allocated CPU / sum applicable observed static capacity", "status": "SOURCE_AVAILABLE", "unit": "CPU units per second / CPU units per second", "limitation": "dynamic available CPU unavailable"},
            "Comm": {"target": "allocated RB fraction", "status": "SOURCE_AVAILABLE", "unit": "dimensionless", "limitation": "N_RB and duplicate/multi-relation aggregation must remain anchor-frozen"},
            "Mob": {"target": "speed / 15 m/s", "status": "PARTIAL", "unit": "dimensionless", "limitation": "15 m/s is formal mobility core domain maximum, not physical or safety bound"},
            "Route": {"target": "route revision/change effort", "status": "BLOCKED", "unit": None, "limitation": "no source-backed candidate-independent normalized denominator for route changes"},
        },
        "planner_role": "J_Effort is last lexicographic tie-break; unavailable components cannot be silently replaced by proxies.",
    })
    write("09_energy_priority_fairness_boundary.json", {
        "energy_in_planner_objective_v1": False,
        "energy_status": "FINAL_CLOSED_LOOP_METRIC_ONLY; AirFogSim EnergyManager models UAV movement/sensing/receive/send energy, but Formal Raw/Tensor/Planner causal energy state is absent; independent Energy Source/Formula Audit and causal derivation contract required before planner use",
        "energy_source": "code/reference/AirFogSim/airfogsim/manager/energy_manager.py:EnergyManager.updateEnergyPattern/updateEnergy; env._updateEnergy",
        "energy_terms": ["flight_or_hover_per_timeslot", "sensing_per_timeslot", "receive_per_data_unit", "send_per_data_unit"],
        "computation_energy_source": "not present in the audited EnergyManager accounting path",
        "planner_energy_availability": "UNAVAILABLE",
        "priority_weighting_v1": False,
        "priority_status": "TaskManager supports Uniform(low,high) or Normal(mean,std) generation and Task.getTaskPriority; audited scheduling/lifecycle source does not use priority for deadline checks; Formal Raw omits it. v1 uses w_q=1 and does not use priority.",
        "priority_range": "configuration-dependent; no single formal range asserted",
        "priority_generation": "TaskManager._generatePriority",
        "priority_scheduling_use": "no use found in audited TaskManager scheduling/deadline path",
        "priority_deadline_coupling": "none in audited deadline predicates",
        "fairness_in_objective_v1": False,
        "fairness_status": "Final metric only; Agent-based versus Task-based Jain support quantity remains METRIC_DEFINITION_PENDING.",
        "risk_mode_v1": "DEFINED_BUT_INACTIVE",
        "risk_status": "mean prior + expectation service is primary; stochastic latent/outage exists, but no calibration, chance guarantee, CVaR or tail-risk claim",
        "epistemic_aleatoric_separate": True,
    })
    fields = [
        ("arrival_time", "Formal Raw decisions[].tasks[].arrival_time_s; observer _extract_tasks", "s", "CAUSALLY_EXPOSED", "PLANNER_SIDE_STATE_REQUIRED"),
        ("deadline", "AirFogSim Task.getTaskDeadline -> observer TaskSnapshot.deadline; absent from Formal Raw task row", "s", "SOURCE_AVAILABLE_BUT_NOT_EXPOSED", "PLANNER_SIDE_STATE_REQUIRED"),
        ("elapsed/task_delay", "observer derives simulation_time_s - arrival_time_s; derivable from current Raw", "s", "CAUSALLY_EXPOSED", "PLANNER_SIDE_STATE_REQUIRED"),
        ("priority", "Task.getTaskPriority -> observer", "dimensionless", "SOURCE_AVAILABLE_BUT_NOT_EXPOSED", "inactive in v1"),
        ("task_size", "Task.getTaskSize -> observer/tensor", "data units", "CAUSALLY_EXPOSED", "Tx denominator"),
        ("task_cpu", "Task.getTaskCPU -> observer/tensor", "CPU work units", "CAUSALLY_EXPOSED", "Comp denominator"),
        ("return_size", "Formal Raw decisions[].tasks[].required_returned_size; observer TaskSnapshot.return_size", "data units", "CAUSALLY_EXPOSED", "PLANNER_SIDE_STATE_REQUIRED for fixed-support objective logic"),
        ("task_completed", "WorldModel one_step final_done", "bool", "MODEL_STATE_AVAILABLE", "cohort/delay"),
        ("task_failed", "observer lifecycle collection -> task_lifecycle_index; future transition in TaskManager.checkTasks", "bool", "MODEL_STATE_AVAILABLE", "exclude anchor terminal; future deadline failure unavailable"),
        ("task_released", "WorldModel DAG release rule", "bool", "MODEL_STATE_AVAILABLE", "not completion"),
        ("task_work_total", "WorldModel state", "CPU work units", "MODEL_STATE_AVAILABLE", "Comp burden"),
        ("task_work_remaining", "WorldModel one_step", "CPU work units", "MODEL_STATE_AVAILABLE", "Comp burden"),
        ("task_progress", "WorldModel one_step", "fraction", "MODEL_STATE_AVAILABLE", "diagnostic"),
        ("flow_total", "WorldModel state", "data units", "MODEL_STATE_AVAILABLE", "Tx burden"),
        ("flow_remaining", "WorldModel terminal service", "data units", "MODEL_STATE_AVAILABLE", "Tx burden"),
        ("hop_progress", "WorldModel state", "fraction", "MODEL_STATE_AVAILABLE", "diagnostic"),
        ("hop_remaining", "WorldModel intermediate/terminal service", "data units", "MODEL_STATE_AVAILABLE", "Tx burden"),
        ("current_hop_index", "WorldModel hop advancement", "index", "MODEL_STATE_AVAILABLE", "Tx burden"),
        ("route", "WorldModel route_node_indices/mask", "node indices", "MODEL_STATE_AVAILABLE", "Tx burden"),
        ("route_revision", "WorldModel flow_route_revision", "integer", "MODEL_STATE_AVAILABLE", "identity limitation"),
        ("flow_status", "WorldModel flow_status_index", "enum", "MODEL_STATE_AVAILABLE", "mask"),
        ("flow_type", "WorldModel flow_type_index", "enum", "MODEL_STATE_AVAILABLE", "Input/Return"),
        ("return_birth_required", "WorldModel one_step", "bool", "MODEL_STATE_AVAILABLE", "support boundary"),
        ("fixed_support_blocked", "WorldModel one_step", "bool", "MODEL_STATE_AVAILABLE", "support boundary"),
        ("nominal_rate", "WorldModel service trace", "data/time", "MODEL_STATE_AVAILABLE", "diagnostic"),
        ("outage_probability", "WorldModel known stochastic wireless service input", "probability", "MODEL_STATE_AVAILABLE", "diagnostic"),
        ("actual_rate", "WorldModel service trace", "data/time", "MODEL_STATE_AVAILABLE", "diagnostic"),
        ("wireless_service", "WorldModel wireless relation service trace", "data", "MODEL_STATE_AVAILABLE", "burden progress"),
        ("wired_service", "WorldModel wired relation service trace", "data", "MODEL_STATE_AVAILABLE", "burden progress"),
        ("delivered_data", "WorldModel total service trace", "data", "MODEL_STATE_AVAILABLE", "burden progress"),
        ("total_delivered_data", "WorldModel summed service trace", "data", "MODEL_STATE_AVAILABLE", "diagnostic"),
        ("static_cpu_capacity", "6.0C current Raw side-state", "CPU/time", "CAUSALLY_EXPOSED", "Comp effort"),
        ("comp_allocation", "6.0A formal action", "CPU/time", "CAUSALLY_EXPOSED", "Effort"),
        ("n_RB", "6.0C current relation support", "count", "CAUSALLY_EXPOSED", "Comm effort"),
        ("rb_allocation", "6.0A formal action", "count", "CAUSALLY_EXPOSED", "Effort"),
        ("uav speed/profile", "6.0C Raw/action domain", "m/s/profile", "CAUSALLY_EXPOSED", "Mob effort"),
        ("energy fields", "no accepted causal Planner source", None, "UNAVAILABLE", "final metric only"),
    ]
    source_file_by_field = {
        "arrival_time": "code/artifacts/formal_dataset/.../raw/*.json.gz; code/src/pi_jwm/airfogsim_full_dual_graph_observer_v1.py",
        "deadline": "code/reference/AirFogSim/airfogsim/entities/task.py; code/src/pi_jwm/airfogsim_full_dual_graph_observer_v1.py",
        "elapsed/task_delay": "code/artifacts/formal_dataset/.../raw/*.json.gz; code/src/pi_jwm/airfogsim_full_dual_graph_observer_v1.py",
        "priority": "code/reference/AirFogSim/airfogsim/entities/task.py; code/src/pi_jwm/airfogsim_full_dual_graph_observer_v1.py",
        "task_size": "code/artifacts/formal_dataset/.../raw/*.json.gz; code/src/pi_jwm/airfogsim_tensor_v2.py",
        "task_cpu": "code/artifacts/formal_dataset/.../raw/*.json.gz; code/src/pi_jwm/airfogsim_tensor_v2.py",
        "return_size": "code/artifacts/formal_dataset/.../raw/*.json.gz; code/src/pi_jwm/airfogsim_full_dual_graph_observer_v1.py",
        "task_completed": "code/src/pi_jwm/step4_4_structured_rssm_world_model_v1.py",
        "task_failed": "code/reference/AirFogSim/airfogsim/manager/task_manager.py",
        "task_released": "code/src/pi_jwm/step4_4_structured_rssm_world_model_v1.py",
        "task_work_total": "code/src/pi_jwm/step4_4_structured_rssm_world_model_v1.py",
        "task_work_remaining": "code/src/pi_jwm/step4_4_structured_rssm_world_model_v1.py",
        "task_progress": "code/src/pi_jwm/step4_4_structured_rssm_world_model_v1.py",
        "flow_total": "code/src/pi_jwm/step4_4_structured_rssm_world_model_v1.py",
        "flow_remaining": "code/src/pi_jwm/step4_4_structured_rssm_world_model_v1.py",
        "hop_progress": "code/src/pi_jwm/step4_4_structured_rssm_world_model_v1.py",
        "hop_remaining": "code/src/pi_jwm/step4_4_structured_rssm_world_model_v1.py",
        "current_hop_index": "code/src/pi_jwm/step4_4_structured_rssm_world_model_v1.py",
        "route": "code/src/pi_jwm/step4_4_structured_rssm_world_model_v1.py",
        "route_revision": "code/src/pi_jwm/step4_4_structured_rssm_world_model_v1.py",
        "flow_status": "code/src/pi_jwm/step4_4_structured_rssm_world_model_v1.py",
        "flow_type": "code/src/pi_jwm/step4_4_structured_rssm_world_model_v1.py",
        "return_birth_required": "code/src/pi_jwm/step4_4_structured_rssm_world_model_v1.py",
        "fixed_support_blocked": "code/src/pi_jwm/step4_4_structured_rssm_world_model_v1.py",
        "nominal_rate": "code/src/pi_jwm/step4_4_structured_rssm_world_model_v1.py",
        "actual_rate": "code/src/pi_jwm/step4_4_structured_rssm_world_model_v1.py",
        "delivered_data": "code/src/pi_jwm/step4_4_structured_rssm_world_model_v1.py",
        "static_cpu_capacity": "docs/contracts/PIJWM_STEP_06_0C_PLANNER_ACTION_DOMAIN_V1.md",
        "comp_allocation": "code/src/pi_jwm/step6_0a_candidate_generation_v1.py",
        "n_RB": "code/src/pi_jwm/step6_0c_planner_action_domain_v1.py",
        "rb_allocation": "code/src/pi_jwm/step6_0a_candidate_generation_v1.py",
        "uav speed/profile": "code/src/pi_jwm/step6_0c_planner_action_domain_v1.py",
        "energy fields": "code/src/pi_jwm/airfogsim_metrics_v2.py",
        "outage_probability": "code/src/pi_jwm/step4_4_structured_rssm_world_model_v1.py",
        "wireless_service": "code/src/pi_jwm/step4_4_structured_rssm_world_model_v1.py",
        "wired_service": "code/src/pi_jwm/step4_4_structured_rssm_world_model_v1.py",
        "total_delivered_data": "code/src/pi_jwm/step4_4_structured_rssm_world_model_v1.py",
    }
    raw_display = f"{FORMAL.relative_to(ROOT).as_posix()}/{raw_rel.as_posix()}"
    layer_overrides = {
        "arrival_time": {"runtime_source": "SOURCE_AVAILABLE", "observer": "SOURCE_AVAILABLE", "formal_raw": "CAUSALLY_EXPOSED", "sample": "CAUSALLY_EXPOSED", "tensor": "UNAVAILABLE", "graph": "UNAVAILABLE", "world_model": "UNAVAILABLE", "planner": "PLANNER_SIDE_STATE_REQUIRED"},
        "deadline": {"runtime_source": "SOURCE_AVAILABLE", "observer": "SOURCE_AVAILABLE", "formal_raw": "SOURCE_AVAILABLE_BUT_NOT_EXPOSED", "sample": "UNAVAILABLE", "tensor": "UNAVAILABLE", "graph": "UNAVAILABLE", "world_model": "UNAVAILABLE", "planner": "PLANNER_SIDE_STATE_REQUIRED"},
        "elapsed/task_delay": {"runtime_source": "SOURCE_AVAILABLE", "observer": "SOURCE_AVAILABLE", "formal_raw": "SOURCE_AVAILABLE_BUT_NOT_EXPOSED", "sample": "CAUSALLY_EXPOSED", "tensor": "UNAVAILABLE", "graph": "UNAVAILABLE", "world_model": "UNAVAILABLE", "planner": "PLANNER_SIDE_STATE_REQUIRED"},
        "priority": {"runtime_source": "SOURCE_AVAILABLE", "observer": "SOURCE_AVAILABLE", "formal_raw": "SOURCE_AVAILABLE_BUT_NOT_EXPOSED", "sample": "UNAVAILABLE", "tensor": "UNAVAILABLE", "graph": "UNAVAILABLE", "world_model": "UNAVAILABLE", "planner": "UNAVAILABLE"},
        "return_size": {"runtime_source": "SOURCE_AVAILABLE", "observer": "SOURCE_AVAILABLE", "formal_raw": "CAUSALLY_EXPOSED", "sample": "SOURCE_AVAILABLE_BUT_NOT_EXPOSED", "tensor": "SOURCE_AVAILABLE_BUT_NOT_EXPOSED", "graph": "UNAVAILABLE", "world_model": "UNAVAILABLE", "planner": "PLANNER_SIDE_STATE_REQUIRED"},
        "task_size": {"runtime_source": "SOURCE_AVAILABLE", "observer": "SOURCE_AVAILABLE", "formal_raw": "CAUSALLY_EXPOSED", "sample": "CAUSALLY_EXPOSED", "tensor": "CAUSALLY_EXPOSED", "graph": "MODEL_STATE_AVAILABLE", "world_model": "MODEL_STATE_AVAILABLE", "planner": "MODEL_STATE_AVAILABLE"},
        "task_cpu": {"runtime_source": "SOURCE_AVAILABLE", "observer": "SOURCE_AVAILABLE", "formal_raw": "CAUSALLY_EXPOSED", "sample": "CAUSALLY_EXPOSED", "tensor": "CAUSALLY_EXPOSED", "graph": "MODEL_STATE_AVAILABLE", "world_model": "MODEL_STATE_AVAILABLE", "planner": "MODEL_STATE_AVAILABLE"},
        "task_completed": {"runtime_source": "SOURCE_AVAILABLE", "observer": "SOURCE_AVAILABLE", "formal_raw": "CAUSALLY_EXPOSED", "sample": "CAUSALLY_EXPOSED", "tensor": "CAUSALLY_EXPOSED", "graph": "MODEL_STATE_AVAILABLE", "world_model": "MODEL_STATE_AVAILABLE", "planner": "MODEL_STATE_AVAILABLE"},
        "task_failed": {"runtime_source": "SOURCE_AVAILABLE", "observer": "SOURCE_AVAILABLE", "formal_raw": "CAUSALLY_EXPOSED", "sample": "CAUSALLY_EXPOSED", "tensor": "CAUSALLY_EXPOSED", "graph": "MODEL_STATE_AVAILABLE", "world_model": "MODEL_STATE_AVAILABLE", "planner": "PLANNER_SIDE_STATE_REQUIRED"},
        "static_cpu_capacity": {"runtime_source": "SOURCE_AVAILABLE", "observer": "SOURCE_AVAILABLE", "formal_raw": "CAUSALLY_EXPOSED", "sample": "CAUSALLY_EXPOSED", "tensor": "CAUSALLY_EXPOSED", "graph": "MODEL_STATE_AVAILABLE", "world_model": "MODEL_STATE_AVAILABLE", "planner": "CAUSALLY_EXPOSED"},
        "comp_allocation": {"runtime_source": "SOURCE_AVAILABLE", "observer": "SOURCE_AVAILABLE", "formal_raw": "CAUSALLY_EXPOSED", "sample": "CAUSALLY_EXPOSED", "tensor": "CAUSALLY_EXPOSED", "graph": "MODEL_STATE_AVAILABLE", "world_model": "MODEL_STATE_AVAILABLE", "planner": "CAUSALLY_EXPOSED"},
        "n_RB": {"runtime_source": "SOURCE_AVAILABLE", "observer": "SOURCE_AVAILABLE", "formal_raw": "CAUSALLY_EXPOSED", "sample": "CAUSALLY_EXPOSED", "tensor": "CAUSALLY_EXPOSED", "graph": "MODEL_STATE_AVAILABLE", "world_model": "MODEL_STATE_AVAILABLE", "planner": "CAUSALLY_EXPOSED"},
        "rb_allocation": {"runtime_source": "SOURCE_AVAILABLE", "observer": "SOURCE_AVAILABLE", "formal_raw": "CAUSALLY_EXPOSED", "sample": "CAUSALLY_EXPOSED", "tensor": "CAUSALLY_EXPOSED", "graph": "MODEL_STATE_AVAILABLE", "world_model": "MODEL_STATE_AVAILABLE", "planner": "CAUSALLY_EXPOSED"},
        "uav speed/profile": {"runtime_source": "SOURCE_AVAILABLE", "observer": "SOURCE_AVAILABLE", "formal_raw": "CAUSALLY_EXPOSED", "sample": "CAUSALLY_EXPOSED", "tensor": "CAUSALLY_EXPOSED", "graph": "MODEL_STATE_AVAILABLE", "world_model": "MODEL_STATE_AVAILABLE", "planner": "CAUSALLY_EXPOSED"},
    }
    matrix = []
    allowed_states = {"SOURCE_AVAILABLE", "SOURCE_AVAILABLE_BUT_NOT_EXPOSED", "CAUSALLY_EXPOSED", "MODEL_STATE_AVAILABLE", "PLANNER_SIDE_STATE_REQUIRED", "UNAVAILABLE"}
    for f, symbol, unit, availability, usage in fields:
        chain = layer_overrides.get(f)
        if chain is None:
            stage = "MODEL_STATE_AVAILABLE" if availability == "MODEL_STATE_AVAILABLE" else "CAUSALLY_EXPOSED" if availability == "CAUSALLY_EXPOSED" else "UNAVAILABLE"
            chain = {"runtime_source": "SOURCE_AVAILABLE" if f not in {"energy fields"} else "UNAVAILABLE", "observer": "UNAVAILABLE", "formal_raw": "UNAVAILABLE", "sample": "UNAVAILABLE", "tensor": "UNAVAILABLE", "graph": "UNAVAILABLE", "world_model": stage, "planner": usage if usage == "PLANNER_SIDE_STATE_REQUIRED" else stage}
        if not set(chain.values()).issubset(allowed_states):
            raise ValueError(f"invalid provenance status for {f}: {chain}")
        source_file = source_file_by_field.get(f, "code/src/pi_jwm/step4_4_structured_rssm_world_model_v1.py")
        if f in {"arrival_time", "elapsed/task_delay", "task_size", "task_cpu", "return_size"}:
            source_file = f"{raw_display}; code/src/pi_jwm/airfogsim_full_dual_graph_observer_v1.py"
        matrix.append({"field": f, "source_file": source_file, "source_symbol": symbol, "layer_status": chain,
                       "unit": unit, "planner_usage": usage, "mask": "anchor-defined fixed support; candidate-independent",
                       "final_evaluation_usage": "see baseline contract", "baseline_usage": "same semantic field if available",
                       "support_status": "SOURCE_AUDITED", "known_limitation": "; ".join(k for k, v in chain.items() if v in {"SOURCE_AVAILABLE_BUT_NOT_EXPOSED", "PLANNER_SIDE_STATE_REQUIRED", "UNAVAILABLE"}) or None})
    write("10_objective_field_provenance_matrix.json", {"schema_version": "PI-JWM-Step6.2A-Provenance-Matrix-v2", "allowed_layer_statuses": sorted(allowed_states), "rows": matrix})
    write("11_planner_objective_contract_v1.json", {
        "status": "RESEARCHER_FROZEN_TARGET_CONTRACT_SOURCE_AUDITED_IMPLEMENTATION_NOT_STARTED",
        "scope": "generic computation tasks",
        "tuple": ["N_DDL", "A_DDL", "J_Delay", "J_Burden", "J_Effort"],
        "order": "lexicographic_minimize",
        "target_formulas": {
            "v(q,h)": "deadline violation at predicted t+h and not timely completed under source semantics",
            "N_DDL": "count current cohort tasks with any violation over H_eff",
            "A_DDL": "sum(v(q,h))/(|Q_t|*H_eff)",
            "u(q,h)": "1 iff task is not successfully final-complete at t+h; FAILED is not successful completion",
            "J_Delay": "mean(u(q,h)) over anchor cohort and common horizon; finite-horizon surrogate, not realized delay percentile/mean",
            "b_Tx": "R_hop + (N_hop - 1 - current_hop_index)*R_e2e; partially source-supported",
            "b_Comp": "W_rem(q,h)/max(W_rem(q,0),epsilon); if W_rem(q,0)=0 then contribution is zero",
            "b_task": "(m_Tx*b_Tx + m_Comp*b_Comp)/(m_Tx+m_Comp); masks frozen at anchor; completed task burden is zero",
            "J_Burden": "mean b_task over anchor cohort and common horizon",
            "J_Effort": "last tie-break over source-backed normalized Comp/Comm/Mob/Route components; Route currently unavailable",
        },
        "deadline_comparison": {"return_completion": "delay <= deadline", "no_return_completion": "delay <= deadline + 1e-5", "active_failure": "delay > deadline", "unit": "scenario simulation-time seconds"},
        "score_readiness": "target definition only; objective execution blocked by missing Planner causal deadline side-state, partially supported Tx burden and unavailable Route effort denominator",
        "hard_constraint_principle": "only source-backed constraints; unsupported Return is model support boundary, not candidate illegality",
        "h_eff": "min candidate H_sup; common H1..H_eff only; H_eff=0 OBJECTIVE_UNSCOREABLE",
        "deadline": "source semantics <= at equality, but Planner readiness BLOCKED until causal side-state exposure",
        "delay": "unfinished-task time area over current nonterminal cohort; FAILED is not successful completion",
        "burden": "masked mean of anchor-supported normalized Tx and Comp burden",
        "effort": "last tie-break; no invented proxy for unavailable Route component",
        "throughput": "diagnostic/final metric, no independent weighted Planner term",
        "risk": "defined but inactive; primary mean/expectation",
        "priority": False, "energy": False, "fairness": False,
        "leakage": "no future task schedule, Future Target, failed label reconstruction, or candidate-dependent denominator",
        "implementation_status": "NOT_STARTED",
    })
    write("12_baseline_sync_contract_v1.json", {
        "status": "RECORDED",
        "planner_semantics": {"tuple": ["N_DDL", "A_DDL", "J_Delay", "J_Burden", "J_Effort"], "order": "lexicographic_minimize", "h_eff": "common support-aware horizon", "risk": "inactive", "priority": "inactive", "energy": "not in Planner v1", "fairness": "not in Planner v1"},
        "final_metric_families": ["Task Completion Rate", "Deadline Violation Rate", "Throughput", "Average Task Delay", "P95 Task Delay", "P99 Task Delay", "Energy Consumption", "Resource Utilization", "Fairness/Jain when applicable"],
        "separation": ["Planner Objective != Final Evaluation Metric", "Predicted metric != Real closed-loop metric", "Evaluation Metric != Acceptance Gate"],
        "same_protocol": ["simulator scenario", "task generation", "random seed protocol", "decision interval", "action execution semantics", "metric definitions", "episode horizon", "warm-up rule", "failure/deadline semantics"],
        "relative_improvement": {"higher_is_better": "(PIJWM-Baseline)/Baseline*100%", "lower_is_better": "(Baseline-PIJWM)/Baseline*100%", "positive_means": "PI-JWM improvement", "required_reporting": ["absolute", "relative", "cross-seed statistics"], "zero_baseline": "METRIC_SEMANTICS_PENDING; specify denominator policy before reporting"},
        "throughput_semantics": "NETWORK_SERVICE_THROUGHPUT versus END_TO_END_USEFUL_THROUGHPUT remains pending until byte counting is frozen",
        "baseline_method_selection": "NOT_SELECTED",
    })
    readiness = {"DEADLINE_SOURCE_READINESS": "BLOCKED", "DELAY_SOURCE_READINESS": "PARTIAL", "TX_BURDEN_SOURCE_READINESS": "PARTIAL", "COMP_BURDEN_SOURCE_READINESS": "PARTIAL", "EFFORT_SOURCE_READINESS": "PARTIAL", "SUPPORT_BOUNDARY_READINESS": "PARTIAL", "FINAL_METRIC_INTERFACE": "RECORDED", "BASELINE_SYNC_INTERFACE": "RECORDED", "STEP_6_2B_READINESS": "BLOCKED", "minimum_blocker": "Additive Planner-only causal side-state for deadline/arrival/priority/return support, plus source-backed end-to-end Flow remaining and candidate-independent Route effort denominator; do not reconstruct from future data."}
    write("13_step6_2b_readiness.json", readiness)
    write("manifest.json", {"schema_version": "PI-JWM-Step6.2A-Manifest-v1", "status": "PARTIAL_BLOCKED_FOR_6_2B", "artifact_files": [f"{i:02d}_{name}.json" for i, name in [(1, "source_semantics_receipt"), (2, "deadline_lifecycle_audit"), (3, "task_objective_cohort_audit"), (4, "flow_multihop_burden_audit"), (5, "compute_burden_audit"), (6, "support_boundary_audit"), (7, "throughput_metric_semantics_audit"), (8, "effort_source_audit"), (9, "energy_priority_fairness_boundary"), (10, "objective_field_provenance_matrix"), (11, "planner_objective_contract_v1"), (12, "baseline_sync_contract_v1"), (13, "step6_2b_readiness")]] + ["manifest.json"], "future_task_schedule_used_by_objective": False, "future_only_task_in_objective_cohort": False, "locked_test_accessed": False, "candidate_ranking": False, "winner_selection": False, "baseline_executed": False, "gpu": False, "checkpoint_modified": False, "formal_dataset_modified": False})
    print(json.dumps({"status": "PASS_WITH_READINESS_BLOCKERS", "output": str(OUT), "readiness": readiness}, ensure_ascii=False))


if __name__ == "__main__":
    main()
