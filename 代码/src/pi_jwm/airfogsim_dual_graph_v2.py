"""PI-JWM v2 dual-graph reconstruction from AirFogSim evidence rows."""

from __future__ import annotations

import copy
import math
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping
from typing import Any


SCHEMA_VERSION = "PI-JWM-AirFogSim-dual-graph-v2"
FLOW_TYPES = {"task_input", "result_return", "dependency_data"}
PHASE_TO_FLOW_TYPE = {
    "offload": "task_input",
    "return": "result_return",
    "dependency": "dependency_data",
}
NETWORK_ATTACHED_ONLY_KINDS = {"cloud", "edge_server"}


def _agent_id(physical_node_id: str) -> str:
    return f"agent::{physical_node_id}"


def _flow_id(task_id: str, flow_type: str, source: str, target: str) -> str:
    return f"flow::{task_id}::{flow_type}::{source}::{target}"


def _finite_nonnegative(value: Any, *, field: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{field} must be finite and nonnegative")
    return result


def _deduplicate(rows: Iterable[Mapping[str, Any]], *, key: str = "id") -> list[dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for row in rows:
        row_copy = copy.deepcopy(dict(row))
        row_id = str(row_copy[key])
        row_copy[key] = row_id
        selected[row_id] = row_copy
    return [selected[row_id] for row_id in sorted(selected)]


def _filter_physical_nodes(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    incident_node_ids = {
        str(edge[field])
        for edge in edges
        for field in ("src", "dst")
    }
    return [
        node
        for node in nodes
        if str(node.get("kind", "")).lower() not in NETWORK_ATTACHED_ONLY_KINDS
        or str(node["id"]) in incident_node_ids
    ]


def _build_agents(
    trajectory_id: str,
    physical_nodes: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    agents: list[dict[str, Any]] = []
    attachments: list[dict[str, Any]] = []
    for node in physical_nodes:
        physical_id = str(node["id"])
        agent_id = _agent_id(physical_id)
        agents.append(
            {
                "trajectory_id": trajectory_id,
                "id": agent_id,
                "physical_node_id": physical_id,
                "agent_type": f"{str(node.get('kind', 'unknown')).lower()}_agent",
                "evidence": "pi_jwm_one_agent_per_active_physical_node",
            }
        )
        attachments.append(
            {
                "trajectory_id": trajectory_id,
                "id": f"cip::{agent_id}::{physical_id}",
                "agent_id": agent_id,
                "physical_node_id": physical_id,
                "value": 1,
            }
        )
    return agents, attachments


def _event_flows(
    trajectory_id: str,
    task_records: list[dict[str, Any]],
    transfer_events: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tasks = {str(row["id"]): row for row in task_records}
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for event in transfer_events:
        phase = str(event.get("phase", ""))
        flow_type = PHASE_TO_FLOW_TYPE.get(phase)
        if flow_type is None:
            continue
        task_id = str(event["task_id"])
        source = str(event["source"])
        target = str(event["target"])
        grouped[(task_id, flow_type, source, target)].append(copy.deepcopy(event))

    flows: list[dict[str, Any]] = []
    bearers: list[dict[str, Any]] = []
    for (task_id, flow_type, source, target), events in sorted(grouped.items()):
        ordered = sorted(
            events,
            key=lambda row: (
                float(row.get("time", 0.0)),
                str(row.get("event_id", "")),
            ),
        )
        task = tasks.get(task_id, {})
        field = "task_size" if flow_type == "task_input" else "return_size"
        fallback_total = float(ordered[0].get("remaining_before", 0.0))
        total_data = _finite_nonnegative(task.get(field, fallback_total), field=field)
        delivered_data = sum(
            _finite_nonnegative(row.get("delivered_data", 0.0), field="delivered_data")
            for row in ordered
        )
        last = ordered[-1]
        remaining_data = max(
            _finite_nonnegative(last.get("remaining_before", 0.0), field="remaining_before")
            - _finite_nonnegative(last.get("delivered_data", 0.0), field="delivered_data"),
            0.0,
        )
        flow_id = _flow_id(task_id, flow_type, source, target)
        flows.append(
            {
                "trajectory_id": trajectory_id,
                "id": flow_id,
                "src": _agent_id(source),
                "dst": _agent_id(target),
                "flow_type": flow_type,
                "task_id": task_id,
                "total_data": total_data,
                "remaining_data": remaining_data,
                "delivered_data": delivered_data,
                "status": "completed" if bool(last.get("flow_completed")) else "active",
                "event_ids": [str(row.get("event_id", "")) for row in ordered],
                "first_time": float(ordered[0].get("time", 0.0)),
                "last_time": float(last.get("time", 0.0)),
                "evidence": "direct_runtime_channel_event",
            }
        )
        bearer_events: dict[str, list[str]] = defaultdict(list)
        for event in ordered:
            for physical_edge_id in event.get("path", []):
                bearer_events[str(physical_edge_id)].append(str(event.get("event_id", "")))
        for physical_edge_id, event_ids in sorted(bearer_events.items()):
            bearers.append(
                {
                    "trajectory_id": trajectory_id,
                    "id": f"cfe::{flow_id}::{physical_edge_id}",
                    "flow_id": flow_id,
                    "physical_edge_id": physical_edge_id,
                    "value": 1,
                    "event_ids": event_ids,
                    "evidence": "direct_runtime_channel_event",
                }
            )
    return flows, bearers


def _dependency_flows(
    trajectory_id: str,
    task_records: list[dict[str, Any]],
    dag_edges: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    tasks = {str(row["id"]): row for row in task_records}
    flows: list[dict[str, Any]] = []
    for edge in dag_edges:
        payload = edge.get("data_mb")
        if payload is None:
            edge.setdefault("payload_status", "not_modeled")
            continue
        payload_value = _finite_nonnegative(payload, field="dependency data_mb")
        if payload_value == 0.0:
            continue
        parent_id = str(edge["src"])
        child_id = str(edge["dst"])
        parent_exec = tasks.get(parent_id, {}).get("exec")
        child_exec = tasks.get(child_id, {}).get("exec")
        if parent_exec in (None, "") or child_exec in (None, ""):
            continue
        source = str(parent_exec)
        target = str(child_exec)
        if source == target:
            continue
        flow_id = _flow_id(f"{parent_id}->{child_id}", "dependency_data", source, target)
        flows.append(
            {
                "trajectory_id": trajectory_id,
                "id": flow_id,
                "src": _agent_id(source),
                "dst": _agent_id(target),
                "flow_type": "dependency_data",
                "task_id": child_id,
                "parent_task_id": parent_id,
                "dag_edge_id": str(edge["id"]),
                "total_data": payload_value,
                "remaining_data": payload_value,
                "delivered_data": 0.0,
                "status": "pending",
                "event_ids": [],
                "evidence": str(edge.get("payload_status", "pi_jwm_explicit")),
            }
        )
    return flows


def build_dual_graph_v2_bundle(
    *,
    trajectory_id: str,
    physical_nodes: Iterable[Mapping[str, Any]],
    physical_edges: Iterable[Mapping[str, Any]],
    task_records: Iterable[Mapping[str, Any]],
    dag_edges: Iterable[Mapping[str, Any]],
    transfer_events: Iterable[Mapping[str, Any]],
    task_snapshots: Iterable[Mapping[str, Any]] = (),
    offload_actions: Iterable[Mapping[str, Any]] = (),
    return_actions: Iterable[Mapping[str, Any]] = (),
    rb_actions: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Build the v2 graph without treating tasks or DAG edges as information topology."""

    edges = _deduplicate(physical_edges)
    nodes = _filter_physical_nodes(_deduplicate(physical_nodes), edges)
    tasks = _deduplicate(task_records)
    dags = _deduplicate(dag_edges)
    events = _deduplicate(transfer_events, key="event_id")
    agents, attachments = _build_agents(str(trajectory_id), nodes)
    event_flows, bearers = _event_flows(str(trajectory_id), tasks, events)
    dependency_flows = _dependency_flows(str(trajectory_id), tasks, dags)
    information_edges = sorted(
        event_flows + dependency_flows,
        key=lambda row: str(row["id"]),
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "trajectory_id": str(trajectory_id),
        "physical_nodes": nodes,
        "physical_edges": edges,
        "information_nodes": agents,
        "information_edges": information_edges,
        "task_nodes": tasks,
        "task_dag_edges": dags,
        "agent_attachments": attachments,
        "flow_bearers": bearers,
        "source_transfer_events": events,
        "source_task_snapshots": [copy.deepcopy(dict(row)) for row in task_snapshots],
        "source_offload_actions": [copy.deepcopy(dict(row)) for row in offload_actions],
        "source_return_actions": [copy.deepcopy(dict(row)) for row in return_actions],
        "source_rb_actions": [copy.deepcopy(dict(row)) for row in rb_actions],
        "evidence_boundary": {
            "task_dag": "airfogsim_precedence",
            "dependency_payload": "not_modeled_unless_data_mb_is_explicit",
            "information_flows": "runtime_transfer_events_or_explicit_dependency_payload",
        },
    }


def _is_dag(task_ids: set[str], dag_edges: list[dict[str, Any]]) -> bool:
    indegree = {task_id: 0 for task_id in task_ids}
    children: dict[str, list[str]] = defaultdict(list)
    for edge in dag_edges:
        source = str(edge.get("src"))
        target = str(edge.get("dst"))
        if source not in indegree or target not in indegree:
            return False
        children[source].append(target)
        indegree[target] += 1
    queue = deque(sorted(task_id for task_id, degree in indegree.items() if degree == 0))
    visited = 0
    while queue:
        source = queue.popleft()
        visited += 1
        for target in children[source]:
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    return visited == len(task_ids)


def validate_dual_graph_v2_bundle(bundle: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the semantic separation and the two direct coupling relations."""

    physical_nodes = list(bundle.get("physical_nodes", []))
    physical_edges = list(bundle.get("physical_edges", []))
    information_nodes = list(bundle.get("information_nodes", []))
    information_edges = list(bundle.get("information_edges", []))
    task_nodes = list(bundle.get("task_nodes", []))
    dag_edges = list(bundle.get("task_dag_edges", []))
    attachments = list(bundle.get("agent_attachments", []))
    bearers = list(bundle.get("flow_bearers", []))

    physical_ids = [str(row.get("id")) for row in physical_nodes]
    physical_id_set = set(physical_ids)
    physical_edge_ids = [str(row.get("id")) for row in physical_edges]
    physical_edge_id_set = set(physical_edge_ids)
    agent_ids = [str(row.get("id")) for row in information_nodes]
    agent_id_set = set(agent_ids)
    flow_ids = [str(row.get("id")) for row in information_edges]
    flow_id_set = set(flow_ids)
    task_ids = [str(row.get("id")) for row in task_nodes]
    task_id_set = set(task_ids)
    dag_ids = [str(row.get("id")) for row in dag_edges]

    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    check(
        "physical_identifiers_unique",
        len(physical_ids) == len(physical_id_set)
        and len(physical_edge_ids) == len(physical_edge_id_set),
        "Physical node and edge identifiers must be unique.",
    )
    check(
        "physical_edge_endpoints_exist",
        all(
            str(row.get("src")) in physical_id_set and str(row.get("dst")) in physical_id_set
            for row in physical_edges
        ),
        "Every physical edge endpoint must be a physical node.",
    )
    check(
        "information_nodes_are_agents",
        len(agent_ids) == len(agent_id_set)
        and not (agent_id_set & task_id_set)
        and all(
            str(row.get("id", "")).startswith("agent::")
            and str(row.get("physical_node_id")) in physical_id_set
            for row in information_nodes
        ),
        "Information nodes must be device-attached agents, never tasks.",
    )
    attachment_counts: dict[str, int] = defaultdict(int)
    attachment_pairs: list[tuple[str, str]] = []
    for row in attachments:
        agent_id = str(row.get("agent_id"))
        physical_id = str(row.get("physical_node_id"))
        attachment_counts[agent_id] += 1
        attachment_pairs.append((agent_id, physical_id))
    check(
        "unique_agent_attachment",
        set(attachment_counts) == agent_id_set
        and all(attachment_counts[agent_id] == 1 for agent_id in agent_id_set)
        and len(attachment_pairs) == len(set(attachment_pairs))
        and all(
            agent_id in agent_id_set and physical_id in physical_id_set
            for agent_id, physical_id in attachment_pairs
        ),
        "Every information agent must attach to exactly one physical node.",
    )
    check(
        "information_flow_semantics",
        len(flow_ids) == len(flow_id_set)
        and all(
            str(row.get("src")) in agent_id_set
            and str(row.get("dst")) in agent_id_set
            and str(row.get("flow_type")) in FLOW_TYPES
            and float(row.get("total_data", -1.0)) >= float(row.get("remaining_data", -1.0)) >= 0.0
            for row in information_edges
        ),
        "Information edges must be typed agent-to-agent flows with valid amounts.",
    )
    check(
        "dag_is_auxiliary",
        not (set(dag_ids) & flow_id_set)
        and all(str(row.get("src")) in task_id_set and str(row.get("dst")) in task_id_set for row in dag_edges),
        "Task DAG edges must stay outside the information graph.",
    )
    check(
        "task_dag_acyclic",
        _is_dag(task_id_set, dag_edges),
        "The auxiliary task graph must be acyclic.",
    )
    check(
        "bearer_edges_exist",
        all(
            str(row.get("flow_id")) in flow_id_set
            and str(row.get("physical_edge_id")) in physical_edge_id_set
            for row in bearers
        ),
        "Every CFE relation must reference an existing information flow and physical edge.",
    )
    failed = [row["name"] for row in checks if not row["passed"]]
    return {
        "schema_version": str(bundle.get("schema_version", "")),
        "dual_graph_v2_ready": not failed,
        "failed_checks": failed,
        "checks": checks,
        "counts": {
            "physical_nodes": len(physical_nodes),
            "physical_edges": len(physical_edges),
            "information_nodes": len(information_nodes),
            "information_edges": len(information_edges),
            "task_nodes": len(task_nodes),
            "task_dag_edges": len(dag_edges),
            "agent_attachments": len(attachments),
            "flow_bearers": len(bearers),
        },
    }
