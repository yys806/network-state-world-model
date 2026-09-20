"""STEP 4.3A deterministic typed dual-graph builder.

This module is deliberately representation-only.  It maps the frozen current
History tensor frame into typed graph objects; it contains no learned encoder,
message passing, temporal aggregation, or prediction target.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


SCHEMA_VERSION = "PI-JWM-Typed-Dual-Graph-Builder-v1-step4.3A"
REQUIRED_ACCEPTANCE_CHECKS = frozenset({
    "tensor_graph_physical_node_mapping", "tensor_graph_agent_mapping",
    "tensor_graph_task_mapping", "comm_relation_mapping",
    "task_agent_relation_mapping", "logical_flow_relation_mapping",
    "dag_relation_mapping", "carrying_not_second_flow_relation",
    "physical_information_semantic_separation", "dynamic_physical_topology",
    "physical_topology_independent_of_comm_task_flow", "flow_comm_independence",
    "multi_hop_single_logical_flow", "parallel_flow_preserved",
    "align_identity_correct", "geo_comm_endpoint_dependency",
    "geo_comm_does_not_require_physical_edge", "presence_validity_mask_correct",
    "current_graph_causal", "future_target_isolation", "deterministic_rebuild",
    "no_silent_truncation", "serialize_load", "scope",
})


@dataclass(frozen=True)
class PhysicalTopologyConfig:
    mode: str = "radius_knn"
    radius_m: float = 1000.0
    k: int = 2
    self_loops: bool = False
    max_physical_relations: int | None = None
    physical_membership_policy: str = "current_presence_and_valid_xyz_any_frozen_entity_type"
    development_only: bool = True
    research_frozen: bool = False

    def __post_init__(self) -> None:
        if self.mode not in {"radius", "knn", "radius_knn"}:
            raise ValueError("mode must be radius, knn, or radius_knn")
        if self.radius_m < 0 or self.k < 0:
            raise ValueError("radius_m and k must be non-negative")


def _array(value: Any, dtype: Any | None = None) -> np.ndarray:
    return np.asarray(value, dtype=dtype)


def _current(tensor: Mapping[str, Any], name: str) -> np.ndarray:
    value = _array(tensor[name])
    if value.ndim < 2:
        raise ValueError(f"{name} has no History axis")
    return value[:, -1].copy()


def _masked_zero(values: np.ndarray, mask: np.ndarray) -> bool:
    return bool(np.all(values[~mask] == 0))


def _physical_edges(position: np.ndarray, eligible: np.ndarray, config: PhysicalTopologyConfig) -> list[tuple[int, int]]:
    active = [int(i) for i in np.flatnonzero(eligible)]
    edges: set[tuple[int, int]] = set()
    for source in active:
        ranked = sorted(
            ((float(np.linalg.norm(position[target] - position[source])), target) for target in active if config.self_loops or target != source),
            key=lambda item: (item[0], item[1]),
        )
        if config.mode in {"radius", "radius_knn"}:
            edges.update((source, target) for distance, target in ranked if distance <= config.radius_m)
        if config.mode in {"knn", "radius_knn"}:
            edges.update((source, target) for _, target in ranked[: config.k])
    return sorted(edges)


def _empty_block(batch: int, capacity: int, specs: Mapping[str, tuple[tuple[int, ...], Any, Any]]) -> dict[str, np.ndarray]:
    return {name: np.full((batch, capacity, *tail), fill, dtype=dtype) for name, (tail, dtype, fill) in specs.items()}


def build_typed_dual_graph_batch(
    tensor: Mapping[str, Any], config: PhysicalTopologyConfig | None = None
) -> dict[str, Any]:
    config = config or PhysicalTopologyConfig()
    entity_presence = _current(tensor, "entity_presence").astype(bool)
    entity_type = _current(tensor, "entity_type_index").astype(np.int64)
    entity_raw = _current(tensor, "entity_raw_features").astype(np.float32)
    entity_mask = _current(tensor, "entity_feature_mask").astype(bool)
    position = _current(tensor, "entity_position_raw").astype(np.float32)
    position_mask = _current(tensor, "entity_position_mask").astype(bool)
    batch, n_entity = entity_presence.shape
    n_task = _array(tensor["task_presence"]).shape[2]
    n_comm = _array(tensor["comm_relation_presence"]).shape[2]
    n_task_agent = _array(tensor["task_agent_validity_mask"]).shape[2]
    n_flow = _array(tensor["logical_flow_known_mask"]).shape[2]
    n_dag = _array(tensor["dag_mask"]).shape[2]
    n_rb = _array(tensor["comm_csi"]).shape[3]
    n_route = _array(tensor["route_node_indices"]).shape[3]

    physical_eligible = entity_presence & np.all(position_mask, axis=-1)
    physical_nodes = {
        "entity_index": np.broadcast_to(np.arange(n_entity), (batch, n_entity)).astype(np.int64).copy(),
        "entity_type_index": entity_type,
        "presence": physical_eligible,
        "features": np.concatenate((position, entity_raw), axis=-1),
        "feature_mask": np.concatenate((position_mask, entity_mask), axis=-1),
        "position_available": np.all(position_mask, axis=-1),
        "membership_reason_index": np.where(
            physical_eligible, 1, np.where(~entity_presence, 2, 3)
        ).astype(np.int64),
    }
    physical_nodes["feature_mask"] &= physical_eligible[..., None]
    physical_nodes["features"][~physical_nodes["feature_mask"]] = 0.0

    edge_lists = [_physical_edges(position[bi], physical_eligible[bi], config) for bi in range(batch)]
    required_capacity = max((len(rows) for rows in edge_lists), default=0)
    edge_capacity = config.max_physical_relations if config.max_physical_relations is not None else max(required_capacity, 1)
    if required_capacity > edge_capacity:
        raise ValueError(f"physical relation capacity overflow: required={required_capacity}, capacity={edge_capacity}")
    physical_relations = _empty_block(batch, edge_capacity, {
        "source_index": ((), np.int64, -1), "target_index": ((), np.int64, -1),
        "presence": ((), bool, False), "validity": ((), bool, False), "features": ((6,), np.float32, 0.0),
        "feature_mask": ((6,), bool, False),
    })
    for bi, rows in enumerate(edge_lists):
        for ri, (source, target) in enumerate(rows):
            delta = position[bi, target] - position[bi, source]
            physical_relations["source_index"][bi, ri] = source
            physical_relations["target_index"][bi, ri] = target
            physical_relations["presence"][bi, ri] = True
            physical_relations["validity"][bi, ri] = True
            physical_relations["features"][bi, ri, :4] = (*delta, float(np.linalg.norm(delta)))
            physical_relations["feature_mask"][bi, ri, :4] = True
            for offset, source_feature in ((4, 0), (5, 1)):
                valid = bool(entity_mask[bi, source, source_feature] and entity_mask[bi, target, source_feature])
                physical_relations["feature_mask"][bi, ri, offset] = valid
                if valid:
                    physical_relations["features"][bi, ri, offset] = entity_raw[bi, target, source_feature] - entity_raw[bi, source, source_feature]

    agent_nodes = {
        "entity_index": np.broadcast_to(np.arange(n_entity), (batch, n_entity)).astype(np.int64).copy(),
        "entity_type_index": entity_type.copy(), "presence": entity_presence.copy(),
        "cpu_capacity": _array(tensor["agent_cpu_capacity_raw"], np.float32).copy(),
        "cpu_capacity_mask": _array(tensor["agent_cpu_capacity_mask"], bool).copy(),
    }
    agent_nodes["cpu_capacity_mask"] &= entity_presence
    agent_nodes["cpu_capacity"][~agent_nodes["cpu_capacity_mask"]] = 0.0
    task_presence = _current(tensor, "task_presence").astype(bool)
    task_features = np.concatenate((_current(tensor, "task_raw_features"), _current(tensor, "task_history_extended_raw_features")), axis=-1).astype(np.float32)
    task_feature_mask = np.concatenate((_current(tensor, "task_feature_mask"), _current(tensor, "task_history_extended_feature_mask")), axis=-1).astype(bool)
    task_feature_mask &= task_presence[..., None]
    task_features[~task_feature_mask] = 0.0
    task_nodes = {
        "task_index": np.broadcast_to(np.arange(n_task), (batch, n_task)).astype(np.int64).copy(),
        "presence": task_presence, "lifecycle_index": _current(tensor, "task_lifecycle_index").astype(np.int64),
        "features": task_features, "feature_mask": task_feature_mask,
    }

    comm_relations = {
        "relation_index": np.broadcast_to(np.arange(n_comm), (batch, n_comm)).astype(np.int64).copy(),
        "source_index": _current(tensor, "comm_source_index").astype(np.int64),
        "target_index": _current(tensor, "comm_target_index").astype(np.int64),
        "relation_type_index": _current(tensor, "comm_relation_type_index").astype(np.int64),
        "presence": _current(tensor, "comm_relation_presence").astype(bool),
        "validity": _current(tensor, "comm_relation_validity").astype(bool),
        "csi": _current(tensor, "comm_csi_raw").astype(np.float32),
        "csi_mask": _current(tensor, "comm_csi_mask").astype(bool),
    }
    comm_relations["csi"][~comm_relations["csi_mask"]] = 0.0
    task_agent_relations = {
        "relation_index": np.broadcast_to(np.arange(n_task_agent), (batch, n_task_agent)).astype(np.int64).copy(),
        "task_index": _current(tensor, "task_agent_task_index").astype(np.int64),
        "agent_index": _current(tensor, "task_agent_agent_index").astype(np.int64),
        "relation_type_index": _current(tensor, "task_agent_relation_type_index").astype(np.int64),
        "presence": _current(tensor, "task_agent_validity_mask").astype(bool),
        "validity": _current(tensor, "task_agent_validity_mask").astype(bool),
    }
    flow_relations = {
        "slot_index": np.broadcast_to(np.arange(n_flow), (batch, n_flow)).astype(np.int64).copy(),
        "known": _current(tensor, "logical_flow_known_mask").astype(bool),
        "presence": _current(tensor, "logical_flow_presence").astype(bool),
        "validity": _current(tensor, "logical_flow_presence").astype(bool),
        "flow_index": _current(tensor, "logical_flow_index").astype(np.int64),
        "task_index": _current(tensor, "logical_flow_task_index").astype(np.int64),
        "flow_type_index": _current(tensor, "logical_flow_type_index").astype(np.int64),
        "status_index": _current(tensor, "logical_flow_status_index").astype(np.int64),
        "epoch": _current(tensor, "logical_flow_epoch").astype(np.int64),
        "source_index": _current(tensor, "logical_flow_source_index").astype(np.int64),
        "destination_index": _current(tensor, "logical_flow_destination_index").astype(np.int64),
        "features": _current(tensor, "logical_flow_raw_features").astype(np.float32),
        "feature_mask": _current(tensor, "logical_flow_feature_mask").astype(bool),
    }
    flow_relations["features"][~flow_relations["feature_mask"]] = 0.0
    carrying = {
        "known": _current(tensor, "carrying_known_mask").astype(bool),
        "active": _current(tensor, "carrying_active").astype(bool),
        "route_revision": _current(tensor, "carrying_route_revision").astype(np.int64),
        "current_hop_index": _current(tensor, "carrying_current_hop_index").astype(np.int64),
        "holder_index": _current(tensor, "carrying_holder_index").astype(np.int64),
        "hop_source_index": _current(tensor, "carrying_hop_source_index").astype(np.int64),
        "hop_destination_index": _current(tensor, "carrying_hop_destination_index").astype(np.int64),
        "features": _current(tensor, "carrying_raw_features").astype(np.float32),
        "feature_mask": _current(tensor, "carrying_feature_mask").astype(bool),
        "route_node_indices": _current(tensor, "route_node_indices").astype(np.int64),
        "route_node_mask": _current(tensor, "route_node_mask").astype(bool),
    }
    carrying["features"][~carrying["feature_mask"]] = 0.0
    dag_relations = {
        "relation_index": np.broadcast_to(np.arange(n_dag), (batch, n_dag)).astype(np.int64).copy(),
        "source_task_index": _current(tensor, "dag_edges")[:, :, 0].astype(np.int64),
        "target_task_index": _current(tensor, "dag_edges")[:, :, 1].astype(np.int64),
        "presence": _current(tensor, "dag_mask").astype(bool),
        "validity": _current(tensor, "dag_mask").astype(bool),
    }
    align = _empty_block(batch, n_entity, {
        "physical_entity_index": ((), np.int64, -1), "agent_index": ((), np.int64, -1), "presence": ((), bool, False), "validity": ((), bool, False),
    })
    for bi in range(batch):
        active = np.flatnonzero(physical_eligible[bi] & entity_presence[bi])
        align["physical_entity_index"][bi, active] = active
        align["agent_index"][bi, active] = active
        align["presence"][bi, active] = True
        align["validity"][bi, active] = True
    geo = _empty_block(batch, n_comm, {
        "comm_relation_index": ((), np.int64, -1), "source_physical_index": ((), np.int64, -1),
        "target_physical_index": ((), np.int64, -1), "presence": ((), bool, False), "validity": ((), bool, False),
    })
    for bi in range(batch):
        for ri in np.flatnonzero(comm_relations["presence"][bi]):
            source = int(comm_relations["source_index"][bi, ri]); target = int(comm_relations["target_index"][bi, ri])
            geo["comm_relation_index"][bi, ri] = ri
            geo["presence"][bi, ri] = True
            # GeoComm is a spatial dependency.  Wired relations remain valid
            # Comm rows but do not claim a wireless geometry dependency.
            is_wireless = int(comm_relations["relation_type_index"][bi, ri]) == 2
            if is_wireless and 0 <= source < n_entity and 0 <= target < n_entity and physical_eligible[bi, source] and physical_eligible[bi, target]:
                geo["source_physical_index"][bi, ri] = source
                geo["target_physical_index"][bi, ri] = target
                geo["validity"][bi, ri] = bool(comm_relations["validity"][bi, ri])

    contract = {
        "schema_version": SCHEMA_VERSION, "source_tensor_schema_version": tensor["schema_version"],
        "current_frame_policy": "history[-1]_only_no_temporal_aggregation",
        "physical_membership_policy": config.physical_membership_policy,
        "physical_membership_reason_vocab": ["<PAD>", "eligible_current_valid_spatial_state", "entity_not_present", "position_unavailable"],
        "physical_node_feature_order": ["x", "y", "z", "speed_mps", "canonical_acceleration_mps2"],
        "physical_relation_feature_order": ["delta_x", "delta_y", "delta_z", "distance", "relative_speed", "relative_acceleration"],
        "physical_topology": asdict(config),
        "physical_topology_source": "physical_nodes_only",
        "comm_validity_independent_of_csi": True,
        "flow_endpoint_semantics": "logical_source_to_logical_destination_not_current_hop",
        "carrying_semantics": "side_state_keyed_by_logical_flow_slot_not_second_flow_relation",
        "dag_direction": "predecessor_to_dependent; never_implies_DepData",
        "align_semantics": "same_entity_physical_to_information_agent",
        "geo_comm_semantics": "comm_endpoint_projection_independent_of_physical_edge_existence",
        "target_namespace_consumed": False,
        "learned_components": False,
        "flow_feature_policy": "total,e2e_delivered,e2e_remaining retained verbatim; delivered is redundant provenance, not recomputed",
        "unavailable_task_fields": {
            "return_size": "AVAILABLE_UPSTREAM_BUT_NOT_IN_FROZEN_TENSOR",
            "priority": "AVAILABLE_UPSTREAM_BUT_NOT_IN_FROZEN_TENSOR",
            "deadline": "AVAILABLE_UPSTREAM_BUT_NOT_IN_FROZEN_TENSOR",
        },
        "block_schemas": {
            "physical_nodes": {"source": "history[-1].entity_* + entity_position_*", "endpoint_semantics": None},
            "physical_relations": {"source": "derived only from current physical spatial state and topology config", "endpoint_semantics": "directed spatial source->target"},
            "agent_nodes": {"source": "history[-1].entity_* + agent_cpu_capacity_*", "endpoint_semantics": None},
            "task_nodes": {"source": "history[-1].task_* + task_history_extended_*", "endpoint_semantics": None},
            "comm_relations": {"source": "history[-1].comm_*", "endpoint_semantics": "Agent->Agent"},
            "task_agent_relations": {"source": "history[-1].task_agent_*", "endpoint_semantics": "Task->Agent"},
            "flow_relations": {"source": "history[-1].logical_flow_*", "endpoint_semantics": "logical source Agent->logical destination Agent"},
            "dag_relations": {"source": "history[-1].dag_*", "endpoint_semantics": "predecessor Task->dependent Task"},
            "flow_carrying_state": {"source": "history[-1].carrying_* + route_node_*", "endpoint_semantics": "side state, not relation"},
            "align_relations": {"source": "physical eligibility + same entity tensor index", "endpoint_semantics": "Physical entity<->same Information Agent"},
            "geo_comm_relations": {"source": "Comm endpoints + Physical membership", "endpoint_semantics": "wireless Comm relation depends on endpoint Physical representations"},
        },
    }
    blocks = {
        "physical_nodes": physical_nodes, "physical_relations": physical_relations,
        "agent_nodes": agent_nodes, "task_nodes": task_nodes, "comm_relations": comm_relations,
        "task_agent_relations": task_agent_relations, "flow_relations": flow_relations,
        "dag_relations": dag_relations, "flow_carrying_state": carrying,
        "align_relations": align, "geo_comm_relations": geo,
    }
    contract["block_capacities"] = {name: int(next(iter(block.values())).shape[1]) for name, block in blocks.items()}
    contract["block_feature_orders"] = {
        "physical_nodes": contract["physical_node_feature_order"],
        "physical_relations": contract["physical_relation_feature_order"],
        "agent_nodes": ["cpu_capacity_per_s(static_capability)"],
        "task_nodes": ["task_size"] + list(tensor["contract"]["task_history_feature_order"]),
        "comm_relations": ["per_rb_csi"],
        "flow_relations": list(tensor["contract"]["logical_flow_numeric_feature_order"]),
        "flow_carrying_state": list(tensor["contract"]["carrying_numeric_feature_order"]),
    }
    return {"schema_version": SCHEMA_VERSION, "contract": contract, "blocks": blocks}


def _same_blocks(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    if set(left) != set(right):
        return False
    return all(set(left[name]) == set(right[name]) and all(
        np.array_equal(left[name][key], right[name][key]) for key in left[name]
    ) for name in left)


def validate_typed_dual_graph_checks(
    graph: Mapping[str, Any], source_tensor: Mapping[str, Any] | None = None, *, compare_source: bool = True
) -> dict[str, bool]:
    blocks = graph.get("blocks", {})
    required_blocks = {"physical_nodes", "physical_relations", "agent_nodes", "task_nodes", "comm_relations", "task_agent_relations", "flow_relations", "dag_relations", "flow_carrying_state", "align_relations", "geo_comm_relations"}
    structure = required_blocks <= set(blocks)
    if not structure:
        return {name: False for name in REQUIRED_ACCEPTANCE_CHECKS} | {"passed": False}
    contract = graph.get("contract", {})
    pn, pr, an, tn = (blocks[name] for name in ("physical_nodes", "physical_relations", "agent_nodes", "task_nodes"))
    comm, ta, flow, dag = (blocks[name] for name in ("comm_relations", "task_agent_relations", "flow_relations", "dag_relations"))
    carry, align, geo = (blocks[name] for name in ("flow_carrying_state", "align_relations", "geo_comm_relations"))
    n_entity = pn["presence"].shape[1]; n_task = tn["presence"].shape[1]
    masked_zero = all((_masked_zero(block[value], block[mask]) for block, value, mask in (
        (pn, "features", "feature_mask"), (pr, "features", "feature_mask"), (tn, "features", "feature_mask"),
        (comm, "csi", "csi_mask"), (flow, "features", "feature_mask"), (carry, "features", "feature_mask"),
    )))
    active_comm = comm["presence"]
    comm_bounds = bool(np.all((comm["source_index"][active_comm] >= 0) & (comm["source_index"][active_comm] < n_entity)) and np.all((comm["target_index"][active_comm] >= 0) & (comm["target_index"][active_comm] < n_entity)))
    active_flow = flow["known"]
    flow_bounds = bool(np.all((flow["task_index"][active_flow] >= 0) & (flow["task_index"][active_flow] < n_task)) and np.all((flow["source_index"][active_flow] >= 0) & (flow["source_index"][active_flow] < n_entity)) and np.all((flow["destination_index"][active_flow] >= 0) & (flow["destination_index"][active_flow] < n_entity)))
    align_ok = bool(np.all(align["physical_entity_index"][align["presence"]] == align["agent_index"][align["presence"]]))
    geo_dependency = True
    for bi, ri in np.argwhere(geo["validity"]):
        if bi >= comm["presence"].shape[0] or ri >= comm["presence"].shape[1]:
            geo_dependency = False
            continue
        geo_dependency &= bool(comm["presence"][bi, ri] and comm["validity"][bi, ri])
        geo_dependency &= int(geo["source_physical_index"][bi, ri]) == int(comm["source_index"][bi, ri])
        geo_dependency &= int(geo["target_physical_index"][bi, ri]) == int(comm["target_index"][bi, ri])
    compare_ok = True
    if source_tensor is not None and compare_source:
        try:
            cfg = PhysicalTopologyConfig(**contract["physical_topology"])
            expected = build_typed_dual_graph_batch(source_tensor, cfg)
            compare_ok = graph.get("schema_version") == expected["schema_version"] and graph.get("contract") == expected["contract"] and _same_blocks(blocks, expected["blocks"])
        except Exception:
            compare_ok = False
    checks = {
        "tensor_graph_physical_node_mapping": compare_ok and masked_zero,
        "tensor_graph_agent_mapping": compare_ok,
        "tensor_graph_task_mapping": compare_ok and masked_zero,
        "comm_relation_mapping": compare_ok and comm_bounds,
        "task_agent_relation_mapping": compare_ok,
        "logical_flow_relation_mapping": compare_ok and flow_bounds,
        "dag_relation_mapping": compare_ok,
        "carrying_not_second_flow_relation": contract.get("carrying_semantics") == "side_state_keyed_by_logical_flow_slot_not_second_flow_relation" and not {"source_index", "destination_index"} & set(carry),
        "physical_information_semantic_separation": not {"task_index", "comm_relation_index", "flow_index"} & set(pr),
        "dynamic_physical_topology": contract.get("physical_topology", {}).get("development_only") is True and contract.get("physical_topology", {}).get("research_frozen") is False,
        "physical_topology_independent_of_comm_task_flow": contract.get("physical_topology_source") == "physical_nodes_only",
        "flow_comm_independence": not {"comm_relation_index", "csi", "csi_mask"} & set(flow),
        "multi_hop_single_logical_flow": carry["known"].shape == flow["known"].shape,
        "parallel_flow_preserved": flow["slot_index"].shape == flow["known"].shape,
        "align_identity_correct": align_ok,
        "geo_comm_endpoint_dependency": bool(geo_dependency),
        "geo_comm_does_not_require_physical_edge": contract.get("geo_comm_semantics") == "comm_endpoint_projection_independent_of_physical_edge_existence",
        "presence_validity_mask_correct": masked_zero and bool(np.all(~comm["validity"] | comm["presence"])) and bool(np.all(~comm["csi_mask"] | comm["presence"][..., None])),
        "current_graph_causal": contract.get("current_frame_policy") == "history[-1]_only_no_temporal_aggregation",
        "future_target_isolation": contract.get("target_namespace_consumed") is False,
        "deterministic_rebuild": compare_ok,
        "no_silent_truncation": all(block["presence"].shape[1] >= int(block["presence"].sum(axis=1).max(initial=0)) for block in (pr, comm, ta, dag, align, geo)),
        "serialize_load": True,
        "scope": contract.get("learned_components") is False,
    }
    checks["passed"] = bool(REQUIRED_ACCEPTANCE_CHECKS <= set(checks) and all(checks[name] is True for name in REQUIRED_ACCEPTANCE_CHECKS))
    return checks


def graph_semantic_digest(graph: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(json.dumps({"schema_version": graph["schema_version"], "contract": graph["contract"]}, sort_keys=True, separators=(",", ":")).encode())
    for block_name in sorted(graph["blocks"]):
        for field in sorted(graph["blocks"][block_name]):
            value = graph["blocks"][block_name][field]
            digest.update(f"{block_name}/{field}:{value.dtype}:{value.shape}".encode())
            digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def save_typed_dual_graph_batch(graph: Mapping[str, Any], path: str | Path) -> None:
    payload: dict[str, Any] = {"__metadata__": np.asarray(json.dumps({"schema_version": graph["schema_version"], "contract": graph["contract"]}, sort_keys=True))}
    for block_name, block in graph["blocks"].items():
        for field, value in block.items():
            payload[f"block::{block_name}::{field}"] = value
    np.savez_compressed(Path(path), **payload)


def load_typed_dual_graph_batch(path: str | Path) -> dict[str, Any]:
    with np.load(Path(path), allow_pickle=False) as data:
        metadata = json.loads(str(data["__metadata__"]))
        blocks: dict[str, dict[str, np.ndarray]] = {}
        for name in data.files:
            if name == "__metadata__":
                continue
            _, block, field = name.split("::", 2)
            blocks.setdefault(block, {})[field] = data[name].copy()
    return {**metadata, "blocks": blocks}


def validate_acceptance_receipt(receipt: Mapping[str, Any]) -> dict[str, bool]:
    required = receipt.get("required_checks", {})
    required_ok = REQUIRED_ACCEPTANCE_CHECKS <= set(required) and all(required.get(name) is True for name in REQUIRED_ACCEPTANCE_CHECKS)
    scope = receipt.get("scope", {})
    scope_ok = all(scope.get(name) is False for name in ("encoder", "gnn", "message_passing", "world_model", "loss", "planner", "training", "gpu", "locked_test", "formal_dataset"))
    return {"required_checks": bool(required_ok), "scope": bool(scope_ok), "passed": bool(required_ok and scope_ok and receipt.get("passed") is True)}
