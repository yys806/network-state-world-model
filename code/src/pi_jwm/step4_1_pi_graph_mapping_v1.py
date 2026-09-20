"""Machine-readable Step 4.1 PI graph semantic mapping.

This module is an audit contract, not a graph builder.  It maps the frozen
Raw / model-ready / tensor facts to the object, field, and relation semantics
defined for the Physical / Information graph.  Missing inputs stay explicit;
no value is synthesized to make the graph appear ready.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "PI-JWM-Step-4.1-PI-Graph-Object-Field-Relation-Mapping-v1"

AVAILABILITY = {
    "AVAILABLE_NOW",
    "DERIVABLE_CAUSALLY",
    "RAW_AVAILABLE_BUT_NOT_EXPOSED",
    "RAW_INSUFFICIENT",
    "RESEARCHER_DECISION_REQUIRED",
}
MAPPING_CLASSES = {
    "DIRECT",
    "REINTERPRET",
    "DERIVED",
    "MISSING",
    "NOT_A_FEATURE",
    "RESEARCHER_DECISION_REQUIRED",
}


def _field(
    field_id: str,
    graph_role: str,
    *,
    raw_source: str | None,
    sample_source: str | None,
    tensor_source: str | None,
    temporal_role: str,
    mapping_class: str,
    availability: str,
    mask_policy: str,
    note: str,
) -> dict[str, Any]:
    return {
        "field_id": field_id,
        "graph_role": graph_role,
        "raw_source": raw_source,
        "sample_source": sample_source,
        "tensor_source": tensor_source,
        "temporal_role": temporal_role,
        "mapping_class": mapping_class,
        "availability": availability,
        "mask_policy": mask_policy,
        "note": note,
    }


def build_mapping() -> dict[str, Any]:
    """Return the frozen mapping as JSON-native data."""

    fields = [
        _field("physical.position_m", "physical_node.dynamic_feature", raw_source="decisions[].entities[].position_m", sample_source=None, tensor_source=None, temporal_role="dynamic", mapping_class="DIRECT", availability="RAW_AVAILABLE_BUT_NOT_EXPOSED", mask_policy="presence plus per-coordinate observed mask required on exposure", note="Definition 03 minimum spatial state; additive input extension required."),
        _field("physical.speed_mps", "physical_node.dynamic_feature", raw_source="decisions[].entities[].speed_mps", sample_source="history[].entities[].speed_mps", tensor_source="entity_raw_features/entity_features[..., speed_mps]", temporal_role="dynamic", mapping_class="DIRECT", availability="AVAILABLE_NOW", mask_policy="entity_presence and entity_feature_mask", note="Scalar speed; it is not an Information Agent resource."),
        _field("physical.motion_direction", "physical_node.dynamic_feature", raw_source="decisions[].entities[].heading + heading_unit + elevation_rad", sample_source=None, tensor_source=None, temporal_role="dynamic", mapping_class="REINTERPRET", availability="RAW_AVAILABLE_BUT_NOT_EXPOSED", mask_policy="explicit observed mask; null is not zero direction", note="Raw heading units differ by entity type and must be canonicalized before exposure."),
        _field("physical.canonical_acceleration_mps2", "physical_node.dynamic_feature", raw_source="decisions[].entities[].canonical_acceleration_mps2", sample_source="history[].entities[].canonical_acceleration_mps2", tensor_source="entity_raw_features/entity_features[..., canonical_acceleration_mps2]", temporal_role="dynamic", mapping_class="DERIVED", availability="AVAILABLE_NOW", mask_policy="canonical_acceleration_observed_mask propagated to entity_feature_mask", note="Causal backward speed difference; simulator acceleration remains audit-only."),
        _field("physical.relative_position", "physical_relation.dynamic_feature", raw_source="derived from endpoint position_m", sample_source=None, tensor_source=None, temporal_role="dynamic", mapping_class="DERIVED", availability="DERIVABLE_CAUSALLY", mask_policy="both endpoints present and position masks true", note="Construction requires additive position exposure."),
        _field("physical.distance", "physical_relation.dynamic_feature", raw_source="derived from endpoint position_m", sample_source=None, tensor_source=None, temporal_role="dynamic", mapping_class="DERIVED", availability="DERIVABLE_CAUSALLY", mask_policy="both endpoints present and position masks true", note="Radius/kNN/radius+kNN is deliberately not decided in Step 4.1."),
        _field("physical.relative_motion", "physical_relation.dynamic_feature", raw_source="derived from endpoint speed/direction", sample_source=None, tensor_source=None, temporal_role="dynamic", mapping_class="DERIVED", availability="DERIVABLE_CAUSALLY", mask_policy="both endpoint motion masks true", note="Requires canonical motion direction exposure."),
        _field("agent.entity_type", "information_agent.static_type", raw_source="decisions[].entities[].entity_type", sample_source="static.input_entity_type_by_index", tensor_source="entity_type_index", temporal_role="static", mapping_class="DIRECT", availability="AVAILABLE_NOW", mask_policy="agent presence; categorical padding is distinct", note="Store once as type/static information, not as repeated continuous state."),
        _field("agent.cpu_capacity_per_s", "information_agent.static_capability", raw_source="decisions[].node_cpu_capacity_observation_rows[].capacity_per_s", sample_source=None, tensor_source=None, temporal_role="configured static capability repeated in decision observations", mapping_class="DIRECT", availability="RAW_AVAILABLE_BUT_NOT_EXPOSED", mask_policy="observed_mask plus missing_reason", note="Source is entity.getFogProfile()['cpu']; current configuration and audited decisions keep it constant. It is capacity, not current available CPU, allocation, or actual service."),
        _field("agent.available_cpu", "information_agent.dynamic_resource", raw_source=None, sample_source=None, tensor_source=None, temporal_role="dynamic", mapping_class="MISSING", availability="RAW_INSUFFICIENT", mask_policy="required if introduced", note="Capacity is not the same as currently available CPU."),
        _field("agent.storage", "information_agent.dynamic_resource", raw_source=None, sample_source=None, tensor_source=None, temporal_role="dynamic", mapping_class="MISSING", availability="RAW_INSUFFICIENT", mask_policy="required if introduced", note="Legacy zero fill is not evidence of storage state."),
        _field("agent.service_load", "information_agent.dynamic_resource", raw_source=None, sample_source=None, tensor_source=None, temporal_role="dynamic", mapping_class="RESEARCHER_DECISION_REQUIRED", availability="RESEARCHER_DECISION_REQUIRED", mask_policy="only if a non-derivable source is approved", note="Do not duplicate a load that can be aggregated from explicit Task/Flow objects."),
        _field("task.data_size", "task_node.demand", raw_source="decisions[].tasks[].task_size", sample_source="history[].tasks[].task_size", tensor_source="task_raw_features/task_features[..., task_size]", temporal_role="task-static demand", mapping_class="DIRECT", availability="AVAILABLE_NOW", mask_policy="task_presence and task_feature_mask", note="AirFogSim data-unit; current unit remains the frozen Raw unit."),
        _field("task.return_size", "task_node.demand", raw_source=None, sample_source=None, tensor_source=None, temporal_role="task-static demand", mapping_class="MISSING", availability="RAW_INSUFFICIENT", mask_policy="required if exposed", note="Simulator getter exists in legacy observer, but the frozen current Raw decision row does not contain it."),
        _field("task.computation_demand", "task_node.demand", raw_source="decisions[].tasks[].task_cpu_work", sample_source=None, tensor_source=None, temporal_role="task-static demand", mapping_class="DIRECT", availability="RAW_AVAILABLE_BUT_NOT_EXPOSED", mask_policy="task presence plus feature mask", note="Do not substitute actual served CPU work."),
        _field("task.priority", "task_node.demand", raw_source=None, sample_source=None, tensor_source=None, temporal_role="task-static demand", mapping_class="MISSING", availability="RAW_INSUFFICIENT", mask_policy="required if exposed", note="Available in simulator/legacy observer, absent from the frozen current Raw row."),
        _field("task.transmission_progress", "task_node.progress", raw_source="decisions[].tasks[].transmitted_size", sample_source="past outcome/target only; absent from current History observation", tensor_source="past_outcome_task_features/target_task_features; absent from current task_features", temporal_role="dynamic", mapping_class="REINTERPRET", availability="RAW_AVAILABLE_BUT_NOT_EXPOSED", mask_policy="task presence and current-state feature mask", note="Past outcome and target labels cannot replace current decision-time input state."),
        _field("task.computing_progress", "task_node.progress", raw_source="decisions[].tasks[].computed_cpu_work", sample_source=None, tensor_source=None, temporal_role="dynamic", mapping_class="DIRECT", availability="RAW_AVAILABLE_BUT_NOT_EXPOSED", mask_policy="task presence and feature mask", note="Actual per-slot CPU service is an outcome, not this cumulative current state."),
        _field("task.deadline_remaining", "task_node.time_state", raw_source=None, sample_source=None, tensor_source=None, temporal_role="dynamic derived", mapping_class="MISSING", availability="RAW_INSUFFICIENT", mask_policy="required if exposed", note="Frozen current Raw omits deadline; it therefore cannot derive remaining time."),
        _field("task.elapsed_time", "task_node.time_state", raw_source="simulation_time_s - decisions[].tasks[].arrival_time_s", sample_source=None, tensor_source=None, temporal_role="dynamic derived", mapping_class="DERIVED", availability="DERIVABLE_CAUSALLY", mask_policy="task presence and finite time fields", note="Causal at decision time but not exposed by v4 sample/v2 tensor."),
        _field("task.lifecycle", "task_node.lifecycle", raw_source="decisions[].tasks[].lifecycle", sample_source="history[].tasks[].lifecycle", tensor_source="task_lifecycle_index", temporal_role="dynamic categorical", mapping_class="DIRECT", availability="AVAILABLE_NOW", mask_policy="task presence; padding/unknown distinct", note="Current lifecycle vocabulary is frozen in Step 3.3."),
        _field("comm.csi", "information_comm_relation.dynamic_feature", raw_source="decisions[].channel_rows[].channel_attenuation_db", sample_source="relation endpoints only", tensor_source=None, temporal_role="decision-time dynamic", mapping_class="DIRECT", availability="RAW_AVAILABLE_BUT_NOT_EXPOSED", mask_policy="channel row observed_mask plus missing_reason and RB mask", note="Wireless directed per-RB attenuation; it is not a Physical relation feature."),
        _field("comm.type", "information_comm_relation.static_or_categorical", raw_source="decisions[].channel_rows[].channel_type for wireless", sample_source=None, tensor_source=None, temporal_role="relation categorical", mapping_class="DIRECT", availability="RAW_AVAILABLE_BUT_NOT_EXPOSED", mask_policy="relation presence and type mask", note="Current Raw channel rows cover directed wireless types; wired decision-time rows are absent."),
        _field("flow.total_data", "information_flow_relation.dynamic_feature", raw_source=None, sample_source=None, tensor_source=None, temporal_role="dynamic", mapping_class="MISSING", availability="RAW_INSUFFICIENT", mask_policy="flow presence and feature mask", note="No explicit current stateful Flow total exists in frozen Raw."),
        _field("flow.remaining_data", "information_flow_relation.dynamic_feature", raw_source=None, sample_source=None, tensor_source=None, temporal_role="dynamic", mapping_class="MISSING", availability="RAW_INSUFFICIENT", mask_policy="flow presence and feature mask", note="past_outcome_flow_service is a hop event and cannot be reused as remaining state."),
        _field("dag.direction", "information_dag_relation.structure", raw_source="decisions[].dag_edges[].source_task_id -> target_task_id", sample_source="history[].dag_relations", tensor_source="dag_edges + dag_mask", temporal_role="current directed structure", mapping_class="DIRECT", availability="AVAILABLE_NOW", mask_policy="dag_mask and endpoint task presence", note="j -> k means k depends on j; DAG has no minimum continuous feature."),
    ]

    extensions = [
        {"item": "physical.position_m", "availability": "RAW_AVAILABLE_BUT_NOT_EXPOSED", "minimum_for_03": True, "required_change": "additive Sample/Tensor input exposure with mask and unit"},
        {"item": "physical.spatial_relations", "availability": "DERIVABLE_CAUSALLY", "minimum_for_03": True, "required_change": "derive only after position/motion exposure; topology policy remains undecided"},
        {"item": "comm.wireless_csi", "availability": "RAW_AVAILABLE_BUT_NOT_EXPOSED", "minimum_for_03": True, "required_change": "add directed per-RB CSI/type/mask to Sample/Tensor"},
        {"item": "comm.wired_relation", "availability": "RAW_AVAILABLE_BUT_NOT_EXPOSED", "minimum_for_03": True, "required_change": "materialize decision-time source/target/direction/relation_type=wired/presence from environment.wired_edges and WiredNetworkManager.hasLink; expose null CSI with feature mask false"},
        {"item": "comm.wired_optional_dynamic_state", "availability": "RAW_INSUFFICIENT", "minimum_for_03": False, "required_change": "optional only: collect genuine live wired queue/load/utilization if later selected; configured capacity_mbps and prop_ms are static link capabilities, and service outcome is not current state"},
        {"item": "agent.cpu_capacity_per_s", "availability": "RAW_AVAILABLE_BUT_NOT_EXPOSED", "minimum_for_03": True, "required_change": "additive Sample/Tensor static capability exposure with observed mask"},
        {"item": "agent.available_cpu", "availability": "RAW_INSUFFICIENT", "minimum_for_03": False, "required_change": "only add if a genuine decision-time remaining/available CPU source is established; do not infer it from capacity, allocation, or service"},
        {"item": "agent.storage", "availability": "RAW_INSUFFICIENT", "minimum_for_03": False, "required_change": "only add if selected and genuinely observed; do not zero-fill"},
        {"item": "task.demand_progress_time", "availability": "RAW_AVAILABLE_BUT_NOT_EXPOSED", "minimum_for_03": True, "required_change": "expose CPU demand and current progress; extend Raw for return size/priority/deadline"},
        {"item": "task_agent.typed_relations", "availability": "DERIVABLE_CAUSALLY", "minimum_for_03": True, "required_change": "materialize Src/Host/Exec/Ret with lifecycle-conditioned validity and stable entity index"},
        {"item": "flow.current_state", "availability": "RAW_INSUFFICIENT", "minimum_for_03": True, "required_change": "collect stable Flow ID/type/endpoints/presence/total/remaining; include Input/Return/DepData"},
        {"item": "physical_membership.edge_cloud", "availability": "RESEARCHER_DECISION_REQUIRED", "minimum_for_03": False, "required_change": "decide whether simulator coordinates carry independent spatial meaning; Info Agent remains allowed"},
    ]

    relations = [
        {"relation": "physical_spatial", "source": "physical_node", "target": "physical_node", "directed": "construction policy not frozen", "endpoint_index_namespace": "static.input_entity_index.physical filtered by approved physical membership", "state_source_status": "DERIVABLE_CAUSALLY", "allowed_current_state_sources": ["position_m", "speed_mps", "motion_direction", "canonical_acceleration_mps2"]},
        {"relation": "comm", "source": "agent", "target": "agent", "directed": True, "endpoint_index_namespace": "static.input_entity_index.physical reused as agent identity namespace", "state_source_status": "RAW_AVAILABLE_BUT_NOT_EXPOSED", "minimum_relation_fields": ["source_agent_id", "target_agent_id", "direction", "relation_type", "presence"], "allowed_current_state_sources": ["decision channel_rows channel_attenuation_db", "channel_type", "observed_mask", "environment.wired_edges / WiredNetworkManager.hasLink"], "type_contracts": {"wireless": {"relation_source": "decisions[].channel_rows", "relation_type": "wireless channel_type", "csi_value": "channel_attenuation_db", "csi_feature_mask": "observed_mask"}, "wired": {"relation_source": "environment.wired_edges / WiredNetworkManager.hasLink", "relation_type": "wired", "csi_value": None, "csi_feature_mask": False, "minimum_relation_fields_complete_without_csi": True, "presence_semantics": "decision-time hasLink(source,target); never post-action service outcome"}}},
        {"relation": "task_agent_src", "source": "task", "target": "agent", "directed": True, "endpoint_index_namespace": "task + physical/agent input namespaces", "state_source_status": "DERIVABLE_CAUSALLY", "raw_source": "tasks[].task_node_id", "validity": "task present and referenced agent present"},
        {"relation": "task_agent_host", "source": "task", "target": "agent", "directed": True, "endpoint_index_namespace": "task + physical/agent input namespaces", "state_source_status": "DERIVABLE_CAUSALLY", "raw_source": "tasks[].current_node_id", "validity": "current decision-time location/host only; never a newly selected A_t^Route target"},
        {"relation": "task_agent_exec", "source": "task", "target": "agent", "directed": True, "endpoint_index_namespace": "task + physical/agent input namespaces", "state_source_status": "DERIVABLE_CAUSALLY", "raw_source": "tasks[].current_node_id conditioned on computing lifecycle", "validity": "only when current lifecycle establishes execution"},
        {"relation": "task_agent_ret", "source": "task", "target": "agent", "directed": True, "endpoint_index_namespace": "task + physical/agent input namespaces", "state_source_status": "DERIVABLE_CAUSALLY", "raw_source": "tasks[].return_destination_id", "validity": "non-null current decision field and endpoint present"},
        {"relation": "flow", "source": "agent", "target": "agent", "directed": True, "multiedge": True, "associated_object": "task", "types": ["Input", "Return", "DepData"], "endpoint_index_namespace": "physical/agent input namespace", "state_source_status": "RAW_INSUFFICIENT", "allowed_current_state_sources": [], "explicitly_not_equivalent": ["slot_transfer_events", "past_outcome_flow_service", "delivered_data_by_task"]},
        {"relation": "dag", "source": "task", "target": "task", "directed": True, "endpoint_index_namespace": "static.input_entity_index.task", "state_source_status": "AVAILABLE_NOW", "allowed_current_state_sources": ["dag_edges source_task_id/target_task_id"], "direction_semantics": "j -> k means task k depends on task j", "continuous_feature": None},
    ]

    forbidden = [
        {"fact": "CSI", "forbidden_role": "physical_edge_feature", "required_role": "information_comm_relation_state"},
        {"fact": "rate_or_actual_service", "forbidden_role": "physical_edge_feature", "required_role": "execution_outcome"},
        {"fact": "RB_allocation", "forbidden_role": "physical_edge_feature", "required_role": "comm_action"},
        {"fact": "CPU_allocation", "forbidden_role": "information_agent_state", "required_role": "comp_action"},
        {"fact": "actual_CPU_service", "forbidden_role": "comp_action", "required_role": "execution_outcome"},
        {"fact": "CPU_capacity", "forbidden_role": "information_agent.dynamic_available_resource", "required_role": "information_agent.static_capability"},
        {"fact": "source_host_exec_ret", "forbidden_role": "task_continuous_feature", "required_role": "typed_task_agent_relation"},
        {"fact": "past_hop_service", "forbidden_role": "current_flow_remaining_state", "required_role": "past_execution_outcome"},
        {"fact": "DAG", "forbidden_role": "dependency_data_flow", "required_role": "task_dependency_relation"},
        {"fact": "position_or_speed", "forbidden_role": "information_agent_resource_feature", "required_role": "physical_node_feature"},
        {"fact": "future_action_result", "forbidden_role": "current_graph_state", "required_role": "future_target_or_execution_outcome"},
        {"fact": "active_task_count", "forbidden_role": "physical_edge_feature", "required_role": "derived_information_load_if_approved"},
        {"fact": "allocated_RB_count", "forbidden_role": "physical_edge_feature", "required_role": "comm_action_or_outcome_audit"},
    ]

    old = [
        {"component": "formal_graph_ops_v1.masked_index_mean", "classification": "DIRECT_REUSE", "reason": "generic padded-index masked aggregation; no field semantics"},
        {"component": "formal_graph_ops_v1.dag_message_pass", "classification": "MINOR_MODIFICATION", "reason": "direction matches parent j -> child k, but message passing is outside Step 4.1"},
        {"component": "formal_graph_ops_v1.couple_agent_physical", "classification": "MINOR_MODIFICATION", "reason": "identity attachment idea reusable; coupling network is not authorized"},
        {"component": "formal_graph_ops_v1.information_message_pass", "classification": "STRUCTURAL_CHANGE", "reason": "models only Agent/Flow endpoint exchange and does not cover typed Comm/Task-Agent/DAG relation semantics"},
        {"component": "formal_graph_ops_v1.couple_flow_bearer", "classification": "DO_NOT_REUSE_AS_CURRENT_MAPPING", "reason": "binds Flow to old physical/channel edge and violates the strict semantic split"},
        {"component": "airfogsim_tensor_v2.NODE_FEATURES", "classification": "STRUCTURAL_CHANGE", "reason": "mixes physical position/motion with CPU/storage in one node vector"},
        {"component": "airfogsim_tensor_v2.EDGE_FEATURES", "classification": "DO_NOT_REUSE_AS_PHYSICAL_EDGE", "reason": "mixes distance with CSI/rate/task/RB facts"},
        {"component": "airfogsim_tensor_v2.FLOW_FEATURES", "classification": "EVIDENCE_ONLY", "reason": "names target stateful fields, but current frozen Raw does not substantiate them"},
        {"component": "airfogsim_tensor_v2.TASK_ENDPOINT_FIELDS", "classification": "REINTERPRET", "reason": "source/host/exec/ret must become typed Task->Agent relations, not Task features"},
        {"component": "formal_dual_graph_world_model_v1", "classification": "STRUCTURAL_CHANGE", "reason": "old physical_edge_state/flow path and coupling consume mixed semantics; no model reuse is authorized here"},
        {"component": "step3_3_model_input_tensor_v1 stable IDs/masks", "classification": "DIRECT_REUSE", "reason": "stable history-union identity, presence, masks, DAG endpoints and action references remain authoritative"},
    ]

    return {
        "schema_version": SCHEMA_VERSION,
        "scope": {
            "mapping_only": True,
            "graph_builder_implemented": False,
            "encoder_implemented": False,
            "message_passing_implemented": False,
            "cross_graph_coupling_implemented": False,
            "training": False,
            "gpu": False,
            "locked_test": False,
            "formal_dataset": False,
        },
        "definition_basis": {
            "document": r"D:\shen\OB\科研\PIJWM\03物理-信息双图建模.md",
            "sha256": "6f3e17b0691aa81c60c2ea1e76a1031d2f7f622abf4b40884e9ae4380953c15e",
            "read_only": True,
        },
        "identity_contract": {
            "real_entity_id_source": "frozen Step 02 stable entity ID",
            "physical_index_source": "static.input_entity_index.physical",
            "agent_index_source": "static.input_entity_index.physical",
            "task_index_source": "static.input_entity_index.task",
            "flow_index_requirement": "new stable stateful Flow index; current event-derived flow index is not sufficient",
            "physical_agent_alignment": "same real entity ID and same stable numeric slot where both representations exist",
            "presence_rule": "Physical and Agent presence are separate masks even when identity index is shared",
        },
        "cpu_semantic_contract": {
            "capacity": {"source": "decisions[].node_cpu_capacity_observation_rows[].capacity_per_s from entity.getFogProfile()['cpu']", "graph_role": "information_agent.static_capability", "availability": "RAW_AVAILABLE_BUT_NOT_EXPOSED"},
            "allocation": {"source": "A_t^Comp.allocated_cpu_per_s", "graph_role": "comp_action", "availability": "AVAILABLE_NOW"},
            "actual_service": {"source": "Outcome.served_cpu_work_by_task", "graph_role": "execution_outcome", "availability": "AVAILABLE_NOW"},
            "available_cpu": {"source": None, "graph_role": "information_agent.dynamic_resource", "availability": "RAW_INSUFFICIENT"},
        },
        "entity_membership": [
            {"entity_type": "vehicle", "physical": True, "information_agent": True, "status": "FROZEN"},
            {"entity_type": "uav", "physical": True, "information_agent": True, "status": "FROZEN"},
            {"entity_type": "rsu", "physical": True, "information_agent": True, "status": "FROZEN"},
            {"entity_type": "edge", "physical": None, "information_agent": True, "status": "RESEARCHER_DECISION_REQUIRED", "reason": "physical only if coordinates have independent deployment meaning"},
            {"entity_type": "cloud", "physical": None, "information_agent": True, "status": "RESEARCHER_DECISION_REQUIRED", "reason": "current [0,0,0] simulator position is not proof of spatial meaning"},
        ],
        "graph_contract": {
            "physical": {"node_types": ["physical_entity"], "relation_types": ["physical_spatial"], "edge_construction_invariant": "depends on current physical spatial state, never communication/task/RB activity", "topology_policy": "RESEARCHER_DECISION_REQUIRED: radius vs kNN vs radius+kNN"},
            "information": {"node_types": ["agent", "task"], "relation_types": ["comm", "task_agent_src", "task_agent_host", "task_agent_exec", "task_agent_ret", "flow", "dag"], "parallel_relation_invariant": "Comm and Flow remain distinct even with identical Agent endpoints; multiple Flow edges may coexist"},
        },
        "field_mappings": fields,
        "relations": relations,
        "current_data_to_graph_role": [
            {"current_fact": "entity ID/type/presence", "graph_role": "Physical/Agent identity and static type", "disposition": "DIRECT"},
            {"current_fact": "speed + canonical acceleration", "graph_role": "Physical node motion", "disposition": "DIRECT"},
            {"current_fact": "position/heading/elevation in Raw", "graph_role": "Physical node spatial/motion input", "disposition": "ADDITIVE_EXPOSURE_REQUIRED"},
            {"current_fact": "channel_rows endpoints + per-RB attenuation", "graph_role": "directed wireless Comm relation", "disposition": "REINTERPRET_AND_EXPOSE"},
            {"current_fact": "environment.wired_edges / WiredNetworkManager.hasLink", "graph_role": "directed wired Comm relation with type and presence; CSI null and masked", "disposition": "MATERIALIZE_AND_EXPOSE"},
            {"current_fact": "FogProfile cpu capacity", "graph_role": "Information Agent static capability", "disposition": "ADDITIVE_EXPOSURE_REQUIRED"},
            {"current_fact": "task size/lifecycle", "graph_role": "Task demand/lifecycle", "disposition": "DIRECT"},
            {"current_fact": "task_node/current_node/return_destination", "graph_role": "typed Task->Agent relation candidates", "disposition": "DERIVE_WITH_VALIDITY"},
            {"current_fact": "slot_transfer_events / past_outcome_flow_service", "graph_role": "past service outcome only", "disposition": "NOT_CURRENT_FLOW_STATE"},
            {"current_fact": "dag_edges", "graph_role": "Task->Task DAG", "disposition": "DIRECT"},
            {"current_fact": "future Route/Comm/Comp/Mobility tensors", "graph_role": "future action conditioning", "disposition": "NOT_CURRENT_GRAPH_STATE"},
        ],
        "required_additive_data_extensions": extensions,
        "forbidden_wrong_placements": forbidden,
        "old_implementation_reuse_conflict": old,
        "graph_readiness": {
            "minimum_definition_supported_by_current_tensor": False,
            "blocking_gaps": ["physical.position_m", "comm.wireless_csi", "comm.wired_relation", "agent.cpu_capacity_per_s", "task.demand_progress_time", "task_agent.typed_relations", "flow.current_state"],
            "rationale": "Minimum graph remains blocked by missing additive exposure/materialization, including wired relation identity/presence. Optional wired numeric dynamic state is not a minimum blocker because wired is represented by relation type with CSI masked absent.",
            "decision": "DO_NOT_IMPLEMENT_GRAPH_BUILDER_IN_STEP_4.1",
        },
    }


def validate_mapping_checks(
    mapping: Mapping[str, Any],
    raw_artifact: Mapping[str, Any],
    tensor_schema: Mapping[str, Any],
) -> dict[str, bool]:
    """Compute the Step 4.1 acceptance receipt from mapping and real evidence."""

    decisions = list(raw_artifact.get("decisions", []))
    entity_rows = [row for decision in decisions for row in decision.get("entities", [])]
    task_rows = [row for decision in decisions for row in decision.get("tasks", [])]
    channel_rows = [row for decision in decisions for row in decision.get("channel_rows", [])]
    cpu_rows = [row for decision in decisions for row in decision.get("node_cpu_capacity_observation_rows", [])]
    dag_rows = [row for decision in decisions for row in decision.get("dag_edges", [])]
    fields = list(mapping.get("field_mappings", []))
    relations = list(mapping.get("relations", []))
    extensions = list(mapping.get("required_additive_data_extensions", []))
    forbidden = list(mapping.get("forbidden_wrong_placements", []))
    scope = dict(mapping.get("scope", {}))

    field_ids = [row.get("field_id") for row in fields]
    required_relations = {"physical_spatial", "comm", "task_agent_src", "task_agent_host", "task_agent_exec", "task_agent_ret", "flow", "dag"}
    relation_names = {row.get("relation") for row in relations}
    forbidden_pairs = {(row.get("fact"), row.get("forbidden_role")) for row in forbidden}
    csi_rows = [row for row in fields if row.get("field_id") == "comm.csi"]
    cpu_capacity_rows = [row for row in fields if row.get("field_id") == "agent.cpu_capacity_per_s"]
    comm_rows = [row for row in relations if row.get("relation") == "comm"]
    flow_rows = [row for row in relations if row.get("relation") == "flow"]
    dag_contract = [row for row in relations if row.get("relation") == "dag"]
    required_gap_items = {
        "physical.position_m", "physical.spatial_relations", "comm.wireless_csi",
        "comm.wired_relation", "comm.wired_optional_dynamic_state", "agent.cpu_capacity_per_s", "task.demand_progress_time",
        "task_agent.typed_relations", "flow.current_state",
    }
    extension_by_item = {row.get("item"): row for row in extensions}
    blocking_gaps = set(mapping.get("graph_readiness", {}).get("blocking_gaps", []))
    wired_contract = comm_rows[0].get("type_contracts", {}).get("wired", {}) if comm_rows else {}
    cpu_contract = dict(mapping.get("cpu_semantic_contract", {}))
    cpu_roles = [row.get("graph_role") for row in cpu_contract.values()]
    observed_cpu_by_node: dict[str, set[float]] = {}
    for row in cpu_rows:
        if row.get("observed_mask") is True and row.get("capacity_per_s") is not None:
            observed_cpu_by_node.setdefault(str(row.get("node_id")), set()).add(float(row["capacity_per_s"]))
    wired_edges = list(raw_artifact.get("environment", {}).get("wired_edges", []))

    raw_entity_required = {"entity_id", "entity_type", "position_m", "speed_mps", "heading", "heading_unit", "elevation_rad", "canonical_acceleration_mps2"}
    raw_task_required = {"task_id", "task_node_id", "current_node_id", "return_destination_id", "arrival_time_s", "task_size", "task_cpu_work", "computed_cpu_work", "transmitted_size", "lifecycle"}
    raw_channel_required = {"source_id", "target_id", "channel_type", "rb_indices", "channel_attenuation_db", "observed_mask", "missing_reason"}

    checks = {
        "schema_version": mapping.get("schema_version") == SCHEMA_VERSION,
        "scope_is_mapping_only": scope.get("mapping_only") is True and all(scope.get(key) is False for key in ("graph_builder_implemented", "encoder_implemented", "message_passing_implemented", "cross_graph_coupling_implemented", "training", "gpu", "locked_test", "formal_dataset")),
        "field_ids_unique": len(field_ids) == len(set(field_ids)) and all(field_ids),
        "classification_vocabularies": all(row.get("availability") in AVAILABILITY and row.get("mapping_class") in MAPPING_CLASSES for row in fields) and all(row.get("availability") in AVAILABILITY for row in extensions),
        "required_relation_types_complete": required_relations == relation_names,
        "identity_namespace_frozen": mapping.get("identity_contract", {}).get("physical_index_source") == "static.input_entity_index.physical" and mapping.get("identity_contract", {}).get("agent_index_source") == "static.input_entity_index.physical" and mapping.get("identity_contract", {}).get("task_index_source") == "static.input_entity_index.task",
        "real_raw_entity_sources_present": bool(entity_rows) and all(raw_entity_required.issubset(row) for row in entity_rows),
        "real_raw_task_sources_present": bool(task_rows) and all(raw_task_required.issubset(row) for row in task_rows),
        "wireless_csi_direction_and_mask_present": bool(channel_rows) and all(raw_channel_required.issubset(row) and row.get("source_id") != row.get("target_id") for row in channel_rows),
        "wired_relation_source_available_but_not_exposed": bool(wired_edges) and not any(str(row.get("channel_type")).lower() == "wired" for row in channel_rows) and extension_by_item.get("comm.wired_relation", {}).get("availability") == "RAW_AVAILABLE_BUT_NOT_EXPOSED" and extension_by_item.get("comm.wired_relation", {}).get("minimum_for_03") is True,
        "wired_no_csi_type_mask_contract": wired_contract.get("relation_source") == "environment.wired_edges / WiredNetworkManager.hasLink" and wired_contract.get("relation_type") == "wired" and wired_contract.get("csi_value") is None and wired_contract.get("csi_feature_mask") is False and wired_contract.get("minimum_relation_fields_complete_without_csi") is True,
        "wired_optional_numeric_state_not_minimum": extension_by_item.get("comm.wired_optional_dynamic_state", {}).get("availability") == "RAW_INSUFFICIENT" and extension_by_item.get("comm.wired_optional_dynamic_state", {}).get("minimum_for_03") is False and "comm.wired_optional_dynamic_state" not in blocking_gaps,
        "cpu_capacity_raw_but_tensor_missing": all("node_cpu_capacity_observation_rows" in decision for decision in decisions) and "agent.cpu_capacity_per_s" in field_ids and not any("cpu" in str(name).lower() for name in tensor_schema.get("entity_feature_order", [])),
        "cpu_capacity_static_capability_semantics": bool(cpu_capacity_rows) and cpu_capacity_rows[0].get("graph_role") == "information_agent.static_capability" and "static capability" in str(cpu_capacity_rows[0].get("temporal_role")) and bool(observed_cpu_by_node) and all(len(values) == 1 for values in observed_cpu_by_node.values()) and all(row.get("source_method") == "entity.getFogProfile()['cpu']" for row in cpu_rows),
        "cpu_semantics_distinct": bool(cpu_capacity_rows) and cpu_capacity_rows[0].get("graph_role") == cpu_contract.get("capacity", {}).get("graph_role") == "information_agent.static_capability" and cpu_contract.get("allocation", {}).get("graph_role") == "comp_action" and cpu_contract.get("actual_service", {}).get("graph_role") == "execution_outcome" and cpu_contract.get("available_cpu", {}).get("graph_role") == "information_agent.dynamic_resource" and cpu_contract.get("available_cpu", {}).get("availability") == "RAW_INSUFFICIENT" and len(cpu_roles) == 4 and len(set(cpu_roles)) == 4,
        "position_raw_but_tensor_missing": all("position_m" in row for row in entity_rows) and "position_m" not in tensor_schema.get("entity_feature_order", []),
        "flow_current_state_not_fabricated": bool(flow_rows) and flow_rows[0].get("state_source_status") == "RAW_INSUFFICIENT" and not flow_rows[0].get("allowed_current_state_sources") and not any("flows" in decision for decision in decisions),
        "dag_direct_reuse_direction": bool(dag_rows) and bool(dag_contract) and dag_contract[0].get("direction_semantics") == "j -> k means task k depends on task j" and tensor_schema.get("dag_direction") == "source_task_j -> target_task_k; k depends on j",
        "required_gap_table_complete": required_gap_items.issubset({row.get("item") for row in extensions}),
        "forbidden_table_complete": {("CSI", "physical_edge_feature"), ("past_hop_service", "current_flow_remaining_state"), ("position_or_speed", "information_agent_resource_feature"), ("future_action_result", "current_graph_state")}.issubset(forbidden_pairs),
        "forbidden_placement_absent": bool(csi_rows) and csi_rows[0].get("graph_role") == "information_comm_relation.dynamic_feature" and not any(row.get("graph_role") == "physical_edge.dynamic_feature" and row.get("field_id") in {"comm.csi", "comm.rate", "comm.rb", "comm.service"} for row in fields),
        "old_implementation_audited": {row.get("component") for row in mapping.get("old_implementation_reuse_conflict", [])}.issuperset({"airfogsim_tensor_v2.EDGE_FEATURES", "formal_graph_ops_v1.couple_flow_bearer", "formal_dual_graph_world_model_v1"}),
        "graph_blocked_by_real_gaps": mapping.get("graph_readiness", {}).get("minimum_definition_supported_by_current_tensor") is False and mapping.get("graph_readiness", {}).get("decision") == "DO_NOT_IMPLEMENT_GRAPH_BUILDER_IN_STEP_4.1" and "comm.wired_relation" in blocking_gaps and "comm.wired_optional_dynamic_state" not in blocking_gaps,
    }
    checks["passed"] = bool(all(checks.values()))
    return checks


def save_mapping(mapping: Mapping[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_mapping(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("mapping must be a JSON object")
    return value


__all__ = [
    "AVAILABILITY",
    "MAPPING_CLASSES",
    "SCHEMA_VERSION",
    "build_mapping",
    "load_mapping",
    "save_mapping",
    "validate_mapping_checks",
]
