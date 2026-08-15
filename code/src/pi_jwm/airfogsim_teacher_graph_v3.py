"""Teacher-aligned physical-information graph contract for AirFogSim records."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from math import sqrt
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "PIJWM-DG-Contract-v3"
PHYSICAL_EDGE_RULE = "complete_directed_spatial_relation"
FORBIDDEN_PHYSICAL_FIELDS = {
    "csi",
    "csi_mean",
    "channel_gain",
    "path_loss",
    "noise",
    "interference",
    "sinr",
    "allocated_rb_count",
    "rb",
    "rate",
    "rate_sum",
    "throughput",
    "active_task_count",
}
OPTIONAL_INFORMATION_FIELDS = (
    "channel_gain",
    "path_loss",
    "noise",
    "interference",
    "sinr",
    "tx_power",
    "mcs",
    "outage",
    "throughput",
    "served_data",
)


def _physical_edge_id(source: str, target: str) -> str:
    return f"physical_edge::{source}::{target}"


def _information_edge_id(source: str, target: str, kind: str) -> str:
    return f"information_edge::{source}::{target}::{kind}"


def _agent_id(node_id: str) -> str:
    return f"agent::{node_id}"


def _node_from_agent(agent_id: str) -> str:
    return str(agent_id).removeprefix("agent::")


def _position(row: Mapping[str, Any]) -> tuple[float, float, float]:
    value = list(row.get("position", []))
    value.extend([0.0] * (3 - len(value)))
    return float(value[0]), float(value[1]), float(value[2])


def _time_key(value: Any) -> float:
    return round(float(value), 9)


def _group_by_time(rows: Iterable[Mapping[str, Any]], field: str) -> dict[float, list[dict[str, Any]]]:
    grouped: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_time_key(row[field])].append(dict(row))
    return dict(grouped)


def _make_information_snapshot(row: Mapping[str, Any]) -> dict[str, Any]:
    source = str(row["src"])
    target = str(row["dst"])
    kind = str(row.get("kind", "unknown"))
    csi_valid = row.get("csi_mean") is not None
    rb_valid = row.get("allocated_rb_count") is not None
    activity_valid = row.get("active_task_count") is not None
    rate_valid = row.get("rate_sum") is not None
    return {
        "id": _information_edge_id(source, target, kind),
        "src": _agent_id(source),
        "dst": _agent_id(target),
        "kind": kind,
        "observed_time": float(row["observed_time"]),
        "pre": {
            "interface_available": 1.0,
            "csi_mean": float(row.get("csi_mean", 0.0) or 0.0),
            "channel_gain": 0.0,
            "path_loss": 0.0,
            "noise": 0.0,
            "historical_interference": 0.0,
            "historical_sinr": 0.0,
            "historical_rate": 0.0,
        },
        "action": {
            "allocated_rb_count": float(row.get("allocated_rb_count", 0.0) or 0.0),
            "tx_power": 0.0,
            "mcs": 0.0,
        },
        "outcome": {
            "active_task_count": float(row.get("active_task_count", 0.0) or 0.0),
            "rate_sum": float(row.get("rate_sum", 0.0) or 0.0),
            "actual_interference": 0.0,
            "actual_sinr": 0.0,
            "outage": 0.0,
            "throughput": 0.0,
            "served_data": 0.0,
        },
        "feature_mask": {
            "pre": {
                "interface_available": True,
                "csi_mean": csi_valid,
                "channel_gain": False,
                "path_loss": False,
                "noise": False,
                "historical_interference": False,
                "historical_sinr": False,
                "historical_rate": False,
            },
            "action": {
                "allocated_rb_count": rb_valid,
                "tx_power": False,
                "mcs": False,
            },
            "outcome": {
                "active_task_count": activity_valid,
                "rate_sum": rate_valid,
                "actual_interference": False,
                "actual_sinr": False,
                "outage": False,
                "throughput": False,
                "served_data": False,
            },
        },
        "source_interface": "AirFogSim channel manager",
        "evidence": "direct_channel_snapshot_reclassified_as_information_edge",
    }


def _make_physical_snapshots(
    node_snapshots: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for time, rows in sorted(_group_by_time(node_snapshots, "observed_time").items()):
        by_id = {str(row["id"]): row for row in rows}
        for source in sorted(by_id):
            for target in sorted(by_id):
                if source == target:
                    continue
                source_row = by_id[source]
                target_row = by_id[target]
                sx, sy, sz = _position(source_row)
                tx, ty, tz = _position(target_row)
                dx, dy, dz = tx - sx, ty - sy, tz - sz
                result.append(
                    {
                        "id": _physical_edge_id(source, target),
                        "src": source,
                        "dst": target,
                        "observed_time": time,
                        "delta_position": [dx, dy, dz],
                        "distance": sqrt(dx * dx + dy * dy + dz * dz),
                        "relative_speed": float(target_row.get("speed", 0.0) or 0.0)
                        - float(source_row.get("speed", 0.0) or 0.0),
                        "los": 0.0,
                        "blocked": 0.0,
                        "feature_mask": {
                            "delta_position": True,
                            "distance": True,
                            "relative_speed": True,
                            "los": False,
                            "blocked": False,
                        },
                        "edge_rule": PHYSICAL_EDGE_RULE,
                        "evidence": "derived_from_same_slot_device_positions",
                    }
                )
    return result


def _task_endpoint(row: Mapping[str, Any], lifecycle: str) -> str | None:
    if lifecycle == "to_offload":
        return str(row.get("source") or row.get("host") or "") or None
    if lifecycle in {"computing", "returning"}:
        return str(row.get("exec") or row.get("host") or row.get("source") or "") or None
    return str(row.get("host") or row.get("source") or "") or None


def _make_information_node_snapshots(
    node_snapshots: Iterable[Mapping[str, Any]],
    task_snapshots: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    nodes_by_time = _group_by_time(node_snapshots, "observed_time")
    tasks_by_time = _group_by_time(task_snapshots, "observed_time")
    result: list[dict[str, Any]] = []
    for time, node_rows in sorted(nodes_by_time.items()):
        accum: dict[str, dict[str, Any]] = {
            str(row["id"]): {
                "unassigned_queue_count": 0.0,
                "tx_queue_count": 0.0,
                "cpu_backlog": 0.0,
                "running_count": 0.0,
                "return_queue_count": 0.0,
                "deadlines": [],
                "priorities": [],
            }
            for row in node_rows
        }
        for task in tasks_by_time.get(time, []):
            lifecycle = str(task.get("lifecycle_state", ""))
            if lifecycle in {"to_generate", "finished", "failed"}:
                continue
            endpoint = _task_endpoint(task, lifecycle)
            if endpoint not in accum:
                continue
            state = accum[endpoint]
            if lifecycle == "to_offload":
                if task.get("exec"):
                    state["tx_queue_count"] += 1.0
                else:
                    state["unassigned_queue_count"] += 1.0
            elif lifecycle == "computing":
                state["running_count"] += 1.0
                state["cpu_backlog"] += max(
                    float(task.get("task_cpu", 0.0) or 0.0)
                    - float(task.get("computed_size", 0.0) or 0.0),
                    0.0,
                )
            elif lifecycle == "returning":
                state["return_queue_count"] += 1.0
            deadline_time = task.get("deadline_time")
            if deadline_time is not None:
                state["deadlines"].append(max(float(deadline_time) - time, 0.0))
            if task.get("priority") is not None:
                state["priorities"].append(float(task["priority"]))
        for node_id in sorted(accum):
            state = accum[node_id]
            result.append(
                {
                    "id": _agent_id(node_id),
                    "physical_node_id": node_id,
                    "observed_time": time,
                    "unassigned_queue_count": state["unassigned_queue_count"],
                    "tx_queue_count": state["tx_queue_count"],
                    "cpu_backlog": state["cpu_backlog"],
                    "running_count": state["running_count"],
                    "return_queue_count": state["return_queue_count"],
                    "deadline_min": min(state["deadlines"]) if state["deadlines"] else 0.0,
                    "priority_mean": (
                        sum(state["priorities"]) / len(state["priorities"])
                        if state["priorities"]
                        else 0.0
                    ),
                    "feature_mask": {
                        "unassigned_queue_count": True,
                        "tx_queue_count": True,
                        "cpu_backlog": True,
                        "running_count": True,
                        "return_queue_count": True,
                        "deadline_min": bool(state["deadlines"]),
                        "priority_mean": bool(state["priorities"]),
                    },
                    "evidence": "derived_from_same_slot_task_lifecycle",
                }
            )
    return result


def audit_v3_source_fields(source: Mapping[str, Any]) -> dict[str, Any]:
    """Audit whether saved records are sufficient for deterministic v3 remapping."""

    required_missing: list[str] = []
    required_collections = (
        "physical_nodes",
        "source_physical_node_snapshots",
        "source_physical_edge_snapshots",
        "information_edges",
        "source_task_snapshots",
    )
    for name in required_collections:
        if name not in source:
            required_missing.append(name)
    node_rows = list(source.get("source_physical_node_snapshots", []))
    for field in ("id", "kind", "observed_time", "position"):
        if node_rows and any(field not in row for row in node_rows):
            required_missing.append(f"source_physical_node_snapshots.{field}")
    channel_rows = list(source.get("source_physical_edge_snapshots", []))
    for field in ("src", "dst", "kind", "observed_time"):
        if channel_rows and any(field not in row for row in channel_rows):
            required_missing.append(f"source_physical_edge_snapshots.{field}")
    available = set().union(*(set(row) for row in channel_rows)) if channel_rows else set()
    optional_missing = [field for field in OPTIONAL_INFORMATION_FIELDS if field not in available]
    return {
        "schema_version": "PIJWM-DG-Contract-v3-source-audit",
        "required_missing": sorted(set(required_missing)),
        "optional_missing": optional_missing,
        "airfogsim_rerun_required": bool(required_missing),
        "source_trajectory_id": str(source.get("trajectory_id", "")),
    }


def remap_teacher_aligned_graph(source: Mapping[str, Any]) -> dict[str, Any]:
    """Create a teacher-aligned graph view without changing AirFogSim records."""

    audit = audit_v3_source_fields(source)
    if audit["required_missing"]:
        raise ValueError(f"missing v3 required fields: {audit['required_missing']}")
    node_ids = sorted({str(row["id"]) for row in source.get("physical_nodes", [])})
    physical_edges = [
        {
            "id": _physical_edge_id(source_id, target_id),
            "src": source_id,
            "dst": target_id,
            "edge_rule": PHYSICAL_EDGE_RULE,
            "evidence": "PI-JWM deterministic spatial relation",
        }
        for source_id in node_ids
        for target_id in node_ids
        if source_id != target_id
    ]
    channel_rows = list(source.get("source_physical_edge_snapshots", []))
    information_keys = sorted(
        {
            (str(row["src"]), str(row["dst"]), str(row.get("kind", "unknown")))
            for row in channel_rows
        }
    )
    information_edges = [
        {
            "id": _information_edge_id(source_id, target_id, kind),
            "src": _agent_id(source_id),
            "dst": _agent_id(target_id),
            "kind": kind,
            "evidence": "AirFogSim communication interface endpoint pair",
        }
        for source_id, target_id, kind in information_keys
    ]
    information_nodes = [
        {
            "id": _agent_id(node_id),
            "physical_node_id": node_id,
            "agent_type": "composite_communication_computation_service_agent",
            "evidence": "PI-JWM one composite agent per physical device",
        }
        for node_id in node_ids
    ]
    cip_relations = [
        {
            "id": f"cip::{_agent_id(node_id)}::{node_id}",
            "information_node_id": _agent_id(node_id),
            "physical_node_id": node_id,
            "value": 1,
        }
        for node_id in node_ids
    ]
    cep_relations = [
        {
            "id": f"cep::{edge['id']}::{_physical_edge_id(_node_from_agent(edge['src']), _node_from_agent(edge['dst']))}",
            "information_edge_id": edge["id"],
            "physical_edge_id": _physical_edge_id(
                _node_from_agent(edge["src"]), _node_from_agent(edge["dst"])
            ),
            "value": 1,
        }
        for edge in information_edges
    ]
    link_by_endpoints: dict[tuple[str, str], list[str]] = defaultdict(list)
    for edge in information_edges:
        link_by_endpoints[(_node_from_agent(edge["src"]), _node_from_agent(edge["dst"]))].append(
            str(edge["id"])
        )
    data_flows = [dict(row) for row in source.get("information_edges", [])]
    cfl_relations: list[dict[str, Any]] = []
    for flow in data_flows:
        source_id = _node_from_agent(str(flow.get("src", "")))
        target_id = _node_from_agent(str(flow.get("dst", "")))
        candidates = sorted(link_by_endpoints.get((source_id, target_id), []))
        if len(candidates) != 1:
            raise ValueError(
                f"flow {flow.get('id')} requires exactly one information edge, found {len(candidates)}"
            )
        cfl_relations.append(
            {
                "id": f"cfl::{flow['id']}::{candidates[0]}",
                "flow_id": str(flow["id"]),
                "information_edge_id": candidates[0],
                "value": 1,
                "evidence": "direct endpoint-compatible communication link",
            }
        )
    result = {
        "schema_version": SCHEMA_VERSION,
        "trajectory_id": str(source.get("trajectory_id", "")),
        "physical_edge_rule": PHYSICAL_EDGE_RULE,
        "physical_nodes": [dict(row) for row in source.get("physical_nodes", [])],
        "physical_edges": physical_edges,
        "information_nodes": information_nodes,
        "information_edges": information_edges,
        "data_flows": data_flows,
        "cip_relations": cip_relations,
        "cep_relations": cep_relations,
        "cfl_relations": cfl_relations,
        "task_nodes": deepcopy(list(source.get("task_nodes", []))),
        "task_dag_edges": deepcopy(list(source.get("task_dag_edges", []))),
        "source_physical_node_snapshots": deepcopy(
            list(source.get("source_physical_node_snapshots", []))
        ),
        "source_physical_edge_snapshots": _make_physical_snapshots(
            source.get("source_physical_node_snapshots", [])
        ),
        "source_information_node_snapshots": _make_information_node_snapshots(
            source.get("source_physical_node_snapshots", []),
            source.get("source_task_snapshots", []),
        ),
        "source_information_edge_snapshots": [
            _make_information_snapshot(row) for row in channel_rows
        ],
        "source_task_snapshots": deepcopy(list(source.get("source_task_snapshots", []))),
        "source_transfer_events": deepcopy(list(source.get("source_transfer_events", []))),
        "source_offload_actions": deepcopy(list(source.get("source_offload_actions", []))),
        "source_return_actions": deepcopy(list(source.get("source_return_actions", []))),
        "source_rb_actions": deepcopy(list(source.get("source_rb_actions", []))),
        "source_cpu_actions": deepcopy(list(source.get("source_cpu_actions", []))),
        "source_audit": audit,
    }
    validate_teacher_aligned_graph(result)
    return result


def _assert_dag_acyclic(rows: Iterable[Mapping[str, Any]]) -> None:
    outgoing: dict[str, list[str]] = defaultdict(list)
    nodes: set[str] = set()
    for row in rows:
        source, target = str(row.get("src", "")), str(row.get("dst", ""))
        outgoing[source].append(target)
        nodes.update((source, target))
    state: dict[str, int] = {}

    def visit(node: str) -> None:
        if state.get(node) == 1:
            raise ValueError("task DAG contains a cycle")
        if state.get(node) == 2:
            return
        state[node] = 1
        for target in outgoing.get(node, []):
            visit(target)
        state[node] = 2

    for node in nodes:
        visit(node)


def validate_teacher_aligned_graph(graph: Mapping[str, Any]) -> dict[str, Any]:
    """Reject ontology leakage and inconsistent teacher-aligned relations."""

    if graph.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("teacher-aligned graph schema mismatch")
    physical_nodes = {str(row["id"]) for row in graph.get("physical_nodes", [])}
    physical_edges = {str(row["id"]): row for row in graph.get("physical_edges", [])}
    for row in list(graph.get("physical_edges", [])) + list(
        graph.get("source_physical_edge_snapshots", [])
    ):
        leaked = FORBIDDEN_PHYSICAL_FIELDS & set(row)
        if leaked:
            raise ValueError(f"wireless fields leaked into physical edge: {sorted(leaked)}")
        if str(row.get("src")) not in physical_nodes or str(row.get("dst")) not in physical_nodes:
            raise ValueError("physical edge endpoint is unknown")
    information_nodes = {str(row["id"]): row for row in graph.get("information_nodes", [])}
    cip_by_agent: dict[str, list[str]] = defaultdict(list)
    for row in graph.get("cip_relations", []):
        cip_by_agent[str(row["information_node_id"])].append(str(row["physical_node_id"]))
    if set(cip_by_agent) != set(information_nodes) or any(len(values) != 1 for values in cip_by_agent.values()):
        raise ValueError("each information agent must have exactly one CIP attachment")
    if any(values[0] not in physical_nodes for values in cip_by_agent.values()):
        raise ValueError("CIP physical endpoint is unknown")
    information_edges = {str(row["id"]): row for row in graph.get("information_edges", [])}
    cep_by_edge: dict[str, list[str]] = defaultdict(list)
    for row in graph.get("cep_relations", []):
        cep_by_edge[str(row["information_edge_id"])].append(str(row["physical_edge_id"]))
    if set(cep_by_edge) != set(information_edges) or any(len(values) != 1 for values in cep_by_edge.values()):
        raise ValueError("each information edge must have exactly one CEP relation")
    for edge_id, edge in information_edges.items():
        if str(edge.get("src")) not in information_nodes or str(edge.get("dst")) not in information_nodes:
            raise ValueError("information edge endpoint is unknown")
        expected = _physical_edge_id(
            cip_by_agent[str(edge["src"])][0], cip_by_agent[str(edge["dst"])][0]
        )
        actual = cep_by_edge[edge_id][0]
        if actual != expected:
            raise ValueError(f"CEP endpoint mismatch for {edge_id}: {actual} != {expected}")
        if actual not in physical_edges:
            raise ValueError("CEP physical edge is unknown")
    flows = {str(row["id"]): row for row in graph.get("data_flows", [])}
    cfl_by_flow: dict[str, list[str]] = defaultdict(list)
    for row in graph.get("cfl_relations", []):
        cfl_by_flow[str(row["flow_id"])].append(str(row["information_edge_id"]))
    if set(cfl_by_flow) != set(flows) or any(len(values) != 1 for values in cfl_by_flow.values()):
        raise ValueError("each data flow must have exactly one CFL relation")
    for flow_id, flow in flows.items():
        edge = information_edges.get(cfl_by_flow[flow_id][0])
        if edge is None:
            raise ValueError("CFL information edge is unknown")
        if str(flow.get("src")) != str(edge.get("src")) or str(flow.get("dst")) != str(edge.get("dst")):
            raise ValueError("CFL endpoint mismatch")
    _assert_dag_acyclic(graph.get("task_dag_edges", []))
    checks = {
        "physical_edges_are_spatial_only": True,
        "cip_unique": True,
        "cep_unique_and_endpoint_consistent": True,
        "cfl_unique_and_endpoint_consistent": True,
        "task_dag_acyclic": True,
    }
    return {
        "schema_version": "PIJWM-DG-Contract-v3-validation",
        "teacher_aligned_graph_valid": all(checks.values()),
        "checks": checks,
        "counts": {
            "physical_nodes": len(physical_nodes),
            "physical_edges": len(physical_edges),
            "information_nodes": len(information_nodes),
            "information_edges": len(information_edges),
            "data_flows": len(flows),
            "cip_relations": sum(map(len, cip_by_agent.values())),
            "cep_relations": sum(map(len, cep_by_edge.values())),
            "cfl_relations": sum(map(len, cfl_by_flow.values())),
            "task_dag_edges": len(list(graph.get("task_dag_edges", []))),
        },
    }


__all__ = [
    "OPTIONAL_INFORMATION_FIELDS",
    "PHYSICAL_EDGE_RULE",
    "SCHEMA_VERSION",
    "audit_v3_source_fields",
    "remap_teacher_aligned_graph",
    "validate_teacher_aligned_graph",
]
