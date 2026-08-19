"""Adapt ledger-bound full-collector frames into formal PI-JWM bundles."""

from __future__ import annotations

import copy
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any


class CollectorBundleContractError(ValueError):
    """Raised when a collector payload cannot support the formal bundle contract."""


def build_formal_bundles(
    frames: Sequence[Mapping[str, Any]],
    *,
    task_records: Sequence[Mapping[str, Any]],
    n_rb: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not frames:
        raise CollectorBundleContractError("at least one complete frame is required")
    if isinstance(n_rb, bool) or not isinstance(n_rb, int) or n_rb <= 0:
        raise CollectorBundleContractError("n_rb must be a positive integer")
    trajectory_ids = {str(frame.get("trajectory_id", "")) for frame in frames}
    if len(trajectory_ids) != 1 or "" in trajectory_ids:
        raise CollectorBundleContractError("all frames must share one trajectory_id")
    trajectory_id = next(iter(trajectory_ids))

    physical_nodes: dict[str, dict[str, Any]] = {}
    physical_edges: dict[str, dict[str, Any]] = {}
    node_snapshots: list[dict[str, Any]] = []
    edge_snapshots: list[dict[str, Any]] = []
    task_snapshots: list[dict[str, Any]] = []
    dag_edges: dict[str, dict[str, Any]] = {}
    transfer_events: list[dict[str, Any]] = []
    offload_actions: list[dict[str, Any]] = []
    return_actions: list[dict[str, Any]] = []
    rb_actions: list[dict[str, Any]] = []

    for frame in frames:
        frame_index = _required_int(frame, "frame_index")
        decision = _required_mapping(frame, "decision_snapshot")
        execution = _required_mapping(frame, "execution_snapshot")
        outcome = _required_mapping(frame, "outcome_snapshot")
        action = _required_mapping(frame, "action")
        _append_snapshot_rows(
            frame=frame,
            snapshot=decision,
            trajectory_id=trajectory_id,
            node_snapshots=node_snapshots,
            edge_snapshots=edge_snapshots,
            physical_nodes=physical_nodes,
            physical_edges=physical_edges,
            dag_edges=dag_edges,
        )
        _append_snapshot_rows(
            frame=frame,
            snapshot=execution,
            trajectory_id=trajectory_id,
            node_snapshots=node_snapshots,
            edge_snapshots=edge_snapshots,
            physical_nodes=physical_nodes,
            physical_edges=physical_edges,
            dag_edges=dag_edges,
        )
        _append_snapshot_rows(
            frame=frame,
            snapshot=outcome,
            trajectory_id=trajectory_id,
            node_snapshots=node_snapshots,
            edge_snapshots=edge_snapshots,
            physical_nodes=physical_nodes,
            physical_edges=physical_edges,
            dag_edges=dag_edges,
            include_tasks=True,
            task_snapshots=task_snapshots,
        )

        flow_by_id = {
            str(row["flow_id"]): row
            for row in _required_sequence_mapping(action, "flows")
        }
        hop_by_id = {
            str(row["hop_id"]): row
            for row in _required_sequence_mapping(action, "hops")
        }
        task_by_id = {
            str(row["task_id"]): row
            for row in _snapshot_tasks(decision)
        }
        for row in _required_sequence_mapping(action, "decisions"):
            if not bool(row.get("selected")):
                continue
            phase = _phase_for_decision(row)
            if phase is None:
                continue
            action_row = {
                "trajectory_id": trajectory_id,
                "task_id": str(row["task_id"]),
                "source_node_id": str(task_by_id.get(str(row["task_id"]), {}).get("current_node_id", "")),
                "target_node_id": str(row.get("target_node_id")),
                "route_nodes": list(row.get("route_nodes", [])),
                "time": _snapshot_time(decision),
                "frame_index": frame_index,
                "evidence": "PIJWM full collector action decision",
            }
            (offload_actions if phase == "offload" else return_actions).append(action_row)
        for allocation in _required_sequence_mapping(action, "rb_allocations"):
            flow = flow_by_id.get(str(allocation.get("flow_id")))
            hop = hop_by_id.get(str(allocation.get("hop_id")))
            if flow is None or hop is None:
                raise CollectorBundleContractError("RB allocation references unknown flow/hop")
            rb_actions.append(
                {
                    "trajectory_id": trajectory_id,
                    "task_id": str(flow["task_id"]),
                    "current_node_id": str(hop["source_id"]),
                    "assigned_to": str(hop["target_id"]),
                    "rb_indices": [int(allocation["rb_index"])],
                    "n_rb": n_rb,
                    "time": _snapshot_time(decision),
                    "frame_index": frame_index,
                    "evidence": "PIJWM full collector RB action",
                }
            )
        transfer_events.extend(
            _transfer_events_for_frame(
                frame,
                trajectory_id=trajectory_id,
                frame_index=frame_index,
                flow_by_id=flow_by_id,
                hop_by_id=hop_by_id,
            )
        )

    records = [copy.deepcopy(dict(row)) for row in task_records]
    if not records and task_snapshots:
        raise CollectorBundleContractError(
            "task snapshots exist but no direct AirFogSim task records were supplied"
        )
    _validate_task_records(records)
    source = {
        "schema_version": "PIJWM-AirFogSim-formal-source-v2",
        "trajectory_id": trajectory_id,
        "physical_nodes": list(physical_nodes.values()),
        "physical_edges": list(physical_edges.values()),
        "physical_node_snapshots": node_snapshots,
        "physical_edge_snapshots": edge_snapshots,
        "information_nodes": records,
        "information_edges": [
            {
                "trajectory_id": trajectory_id,
                "id": str(row["dag_edge_id"]),
                "src": str(row["source_task_id"]),
                "dst": str(row["target_task_id"]),
                "data_mb": None,
                "semantic": "precedence_only",
                "payload_status": "not_modeled",
            }
            for row in dag_edges.values()
        ],
        "task_snapshots": task_snapshots,
        "offload_actions": offload_actions,
        "return_actions": return_actions,
        "rb_actions": rb_actions,
        "transfer_events": transfer_events,
        "dependency_flows": [],
        "ep_relations": [],
    }
    resource = _resource_bundle(frames, transfer_events, n_rb=n_rb)
    return source, resource


def _required_mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    row = value.get(key)
    if not isinstance(row, Mapping):
        raise CollectorBundleContractError(f"{key} must be a mapping")
    return row


def _required_sequence_mapping(value: Mapping[str, Any], key: str) -> list[Mapping[str, Any]]:
    rows = value.get(key)
    if not isinstance(rows, list):
        raise CollectorBundleContractError(f"{key} must be a list")
    if any(not isinstance(row, Mapping) for row in rows):
        raise CollectorBundleContractError(f"{key} contains a non-mapping row")
    return rows


def _required_int(value: Mapping[str, Any], key: str) -> int:
    raw = value.get(key)
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
        raise CollectorBundleContractError(f"{key} must be a nonnegative integer")
    return raw


def _snapshot_time(snapshot: Mapping[str, Any]) -> float:
    raw = snapshot.get("simulation_time")
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise CollectorBundleContractError("snapshot simulation_time is required")
    return float(raw)


def _snapshot_tasks(snapshot: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = snapshot.get("tasks")
    if not isinstance(rows, list) or any(not isinstance(row, Mapping) for row in rows):
        raise CollectorBundleContractError("snapshot tasks must be a list of mappings")
    return rows


def _append_snapshot_rows(
    *,
    frame: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    trajectory_id: str,
    node_snapshots: list[dict[str, Any]],
    edge_snapshots: list[dict[str, Any]],
    physical_nodes: dict[str, dict[str, Any]],
    physical_edges: dict[str, dict[str, Any]],
    dag_edges: dict[str, dict[str, Any]],
    include_tasks: bool = False,
    task_snapshots: list[dict[str, Any]] | None = None,
) -> None:
    time_value = _snapshot_time(snapshot)
    nodes = snapshot.get("nodes")
    edges = snapshot.get("physical_edges")
    dags = snapshot.get("dag_edges")
    if not isinstance(nodes, list) or not isinstance(edges, list) or not isinstance(dags, list):
        raise CollectorBundleContractError("snapshot nodes/physical_edges/dag_edges must be lists")
    for row in nodes:
        if not isinstance(row, Mapping) or not str(row.get("node_id", "")):
            raise CollectorBundleContractError("physical node snapshot lacks node_id")
        node_id = str(row["node_id"])
        converted = {
            "trajectory_id": trajectory_id,
            "id": node_id,
            "kind": str(row.get("node_type", "unknown")),
            "position": list(row["position"]) if row.get("position") is not None else None,
            "cpu": row.get("cpu"),
            "observed_time": time_value,
            "present": bool(row.get("present", False)),
            "evidence": "direct AirFogSim snapshot",
        }
        node_snapshots.append(converted)
        physical_nodes[node_id] = dict(converted)
    for row in edges:
        if not isinstance(row, Mapping):
            raise CollectorBundleContractError("physical edge snapshot row is invalid")
        edge_id = str(row.get("edge_id", ""))
        source = str(row.get("source_id", ""))
        target = str(row.get("target_id", ""))
        if not edge_id or not source or not target:
            raise CollectorBundleContractError("physical edge snapshot lacks identity")
        converted = {
            "trajectory_id": trajectory_id,
            "id": edge_id,
            "src": source,
            "dst": target,
            "kind": str(row.get("edge_type", "unknown")),
            "observed_time": time_value,
            "evidence": "direct AirFogSim snapshot",
        }
        edge_snapshots.append(converted)
        physical_edges[edge_id] = dict(converted)
    for row in dags:
        if not isinstance(row, Mapping):
            raise CollectorBundleContractError("DAG snapshot row is invalid")
        dag_id = str(row.get("dag_edge_id", ""))
        if not dag_id:
            raise CollectorBundleContractError("DAG snapshot lacks dag_edge_id")
        dag_edges[dag_id] = {
            "dag_edge_id": dag_id,
            "source_task_id": str(row.get("source_task_id", "")),
            "target_task_id": str(row.get("target_task_id", "")),
        }
    if include_tasks:
        assert task_snapshots is not None
        for row in _snapshot_tasks(snapshot):
            task_id = str(row.get("task_id", ""))
            if not task_id:
                raise CollectorBundleContractError("task snapshot lacks task_id")
            task_snapshots.append(
                {
                    "trajectory_id": trajectory_id,
                    "id": task_id,
                    "lifecycle": str(row.get("lifecycle", "")),
                    "current_node_id": str(row.get("current_node_id", "")),
                    "route_nodes": list(row.get("route_nodes", [])),
                    "return_destination_id": row.get("return_destination_id"),
                    "arrival_time": float(row.get("arrival_time", 0.0)),
                    "observed_time": time_value,
                    "evidence": "direct AirFogSim snapshot",
                }
            )


def _phase_for_decision(row: Mapping[str, Any]) -> str | None:
    lifecycle = str(row.get("lifecycle", ""))
    if lifecycle in {"waiting_to_offload", "offloading"}:
        return "offload"
    if lifecycle in {"waiting_to_return", "returning"}:
        return "return"
    return None


def _transfer_events_for_frame(
    frame: Mapping[str, Any],
    *,
    trajectory_id: str,
    frame_index: int,
    flow_by_id: Mapping[str, Mapping[str, Any]],
    hop_by_id: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = frame.get("transfer_rows")
    if not isinstance(rows, list) or any(not isinstance(row, Mapping) for row in rows):
        raise CollectorBundleContractError("transfer_rows must be a list of mappings")
    events: list[dict[str, Any]] = []
    for ordinal, row in enumerate(rows):
        if row.get("observed_mask") is False:
            continue
        flow = flow_by_id.get(str(row.get("flow_id")))
        hop = hop_by_id.get(str(row.get("hop_id")))
        if flow is None or hop is None:
            raise CollectorBundleContractError("transfer row references unknown flow/hop")
        required = ("time", "remaining_before", "planned_capacity", "delivered_data")
        if any(key not in row for key in required):
            raise CollectorBundleContractError(
                f"transfer row lacks direct ledger fields: {', '.join(required)}"
            )
        rb_index = row.get("rb_index")
        event = {
            "event_id": f"event::{trajectory_id}::{frame_index}::{ordinal}",
            "trajectory_id": trajectory_id,
            "task_id": str(flow["task_id"]),
            "phase": str(flow["phase"]),
            "source": str(hop["source_id"]),
            "target": str(hop["target_id"]),
            "path": [str(hop["physical_edge_id"])],
            "rb_indices": [] if rb_index is None else [int(rb_index)],
            "channel_type": row.get("channel_type"),
            "planned_capacity": float(row["planned_capacity"]),
            "remaining_before": float(row["remaining_before"]),
            "delivered_data": float(row["delivered_data"]),
            "flow_completed": bool(row.get("flow_completed", False)),
            "time": float(row["time"]),
            "evidence": "direct full collector transfer row",
        }
        events.append(event)
    return events


def _resource_bundle(
    frames: Sequence[Mapping[str, Any]],
    transfer_events: Sequence[Mapping[str, Any]],
    *,
    n_rb: int,
) -> dict[str, Any]:
    task_ledger: list[dict[str, Any]] = []
    rb_ledger: list[dict[str, Any]] = []
    for event in transfer_events:
        delivered = float(event["delivered_data"])
        before = float(event["remaining_before"])
        task_ledger.append(
            {
                "record_id": str(event["event_id"]),
                "kind": "communication",
                "phase": str(event["phase"]),
                "time": float(event["time"]),
                "task_id": str(event["task_id"]),
                "planned_capacity": float(event.get("planned_capacity", 0.0)),
                "remaining_before": before,
                "delivered_data": delivered,
                "remaining_after": max(before - delivered, 0.0),
                "evidence": "direct full collector transfer row",
            }
        )
        rb_ledger.append(
            {
                "record_id": f"rb::{event['event_id']}",
                "time": float(event["time"]),
                "task_id": str(event["task_id"]),
                "rb_indices": list(event.get("rb_indices", [])),
                "n_rb": n_rb,
                "evidence": "direct full collector RB row",
            }
        )
    cpu_ledger: list[dict[str, Any]] = []
    energy_ledger: list[dict[str, Any]] = []
    policy_callbacks: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    callback_counts: dict[tuple[float, str], int] = defaultdict(int)
    for frame in frames:
        for row in frame.get("cpu_rows", []):
            if not isinstance(row, Mapping):
                raise CollectorBundleContractError("cpu_rows contains a non-mapping row")
            policy_rows = row.get("policy_rows")
            if not isinstance(policy_rows, list):
                if row:
                    raise CollectorBundleContractError(
                        "CPU rows without policy_rows cannot establish formal policy provenance"
                    )
                continue
            for policy_row in policy_rows:
                if not isinstance(policy_row, Mapping):
                    raise CollectorBundleContractError("policy_rows contains a non-mapping row")
                node_id = str(policy_row["node_id"])
                base_time = float(row.get("time", 0.0))
                callback_counts[(base_time, node_id)] += 1
                policy_callbacks.append((row, policy_row))
        for energy_row in frame.get("energy_rows", []):
            if not isinstance(energy_row, Mapping):
                raise CollectorBundleContractError("energy_rows contains a non-mapping row")
            energy_ledger.append(copy.deepcopy(dict(energy_row)))
    callback_seen: dict[tuple[float, str], int] = defaultdict(int)
    for row, policy_row in policy_callbacks:
        node_id = str(policy_row["node_id"])
        base_time = float(row.get("time", 0.0))
        callback_key = (base_time, node_id)
        callback_index = callback_seen[callback_key]
        callback_seen[callback_key] += 1
        task_id = str(policy_row.get("task_id", ""))
        allocated = float(policy_row.get("allocated_cpu", 0.0))
        computed_before = row.get("computed_before", {})
        computed_after = row.get("computed_after", {})
        before = float(computed_before.get(task_id, 0.0))
        after = float(computed_after.get(task_id, before))
        task_cpu = float(policy_row.get("task_cpu", after))
        cpu_ledger.append(
            {
                "record_id": f"cpu::{task_id}::{base_time:.6f}::{callback_index}",
                "kind": "compute",
                "time": base_time + callback_index * 1e-9,
                "node_id": node_id,
                "task_id": task_id,
                "allocated_cpu": allocated,
                "node_cpu_capacity": float(policy_row["node_cpu_capacity"]),
                "allocated_fraction": float(policy_row.get("allocated_fraction", 0.0)),
                "policy_id": str(policy_row.get("policy_id", "")),
                "policy_weight": float(policy_row.get("policy_weight", 0.0)),
                "deadline_remaining": float(policy_row.get("deadline_remaining", 0.0)),
                "queue_size": int(policy_row.get("queue_size", 0)),
                "dt": float(row.get("slot_seconds", 0.0)),
                "computed_before": before,
                "computed_after": after,
                "task_cpu": task_cpu,
                "remaining_before": max(task_cpu - before, 0.0),
                "delivered_data": max(after - before, 0.0),
                "remaining_after": max(task_cpu - after, 0.0),
                "evidence": "direct full collector CPU policy row",
            }
        )
    task_ledger.extend(copy.deepcopy(cpu_ledger))
    return {
        "task_ledger": task_ledger,
        "dependency_ledger": [],
        "rb_ledger": rb_ledger,
        "cpu_ledger": cpu_ledger,
        "uav_energy_ledger": energy_ledger,
    }


def _validate_task_records(rows: Sequence[Mapping[str, Any]]) -> None:
    for row in rows:
        if not str(row.get("id", "")):
            raise CollectorBundleContractError("task record lacks id")
