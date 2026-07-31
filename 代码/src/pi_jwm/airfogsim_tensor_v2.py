"""Semantic tensorization for the PI-JWM AirFogSim dual-graph v2 contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "PI-JWM-AirFogSim-tensor-v2"
NODE_FEATURES = ("x", "y", "z", "speed", "acceleration", "cpu", "storage")
EDGE_FEATURES = ("distance", "csi_mean", "rate_sum", "active_task_count", "allocated_rb_count")
FLOW_FEATURES = ("total_data", "remaining_data", "delivered_cumulative", "delivered_this_slot", "age")
TASK_FEATURES = (
    "task_size",
    "return_size",
    "task_cpu",
    "deadline_remaining",
    "priority",
    "transmitted",
    "computed",
    "delay",
)
ACTION_FEATURES = ("offload", "rb", "return", "rb_count", "rb_fraction")
TASK_ENDPOINT_FIELDS = ("source", "host", "exec", "ret")
LIFECYCLE_TYPES = ("to_offload", "computing", "returning", "finished", "failed")
FLOW_TYPES = ("task_input", "result_return", "dependency_data")
NODE_TYPES = ("vehicle", "uav", "rsu", "edge_server", "cloud")
PHYSICAL_EDGE_TYPES = ("V2V", "V2U", "V2I", "U2V", "U2U", "U2I", "I2V", "I2U", "I2I", "wired")


def natural_id_key(value: Any) -> tuple[str, int | str]:
    text = str(value)
    prefix, separator, suffix = text.rpartition("_")
    return prefix if separator else text, int(suffix) if separator and suffix.isdigit() else suffix


@dataclass(frozen=True)
class TensorContract:
    max_nodes: int
    max_physical_edges: int
    max_flows: int
    max_tasks: int
    max_dag_edges: int
    history_steps: int = 8
    horizon_steps: int = 3
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "max_nodes": self.max_nodes,
            "max_physical_edges": self.max_physical_edges,
            "max_flows": self.max_flows,
            "max_tasks": self.max_tasks,
            "max_dag_edges": self.max_dag_edges,
            "history_steps": self.history_steps,
            "horizon_steps": self.horizon_steps,
            "node_features": list(NODE_FEATURES),
            "physical_edge_features": list(EDGE_FEATURES),
            "flow_features": list(FLOW_FEATURES),
            "task_features": list(TASK_FEATURES),
            "action_features": list(ACTION_FEATURES),
            "lifecycle_types": list(LIFECYCLE_TYPES),
            "flow_types": list(FLOW_TYPES),
            "node_types": list(NODE_TYPES),
            "physical_edge_types": list(PHYSICAL_EDGE_TYPES),
        }


def infer_tensor_contract(
    graphs: Sequence[Mapping[str, Any]],
    *,
    history_steps: int = 8,
    horizon_steps: int = 3,
) -> TensorContract:
    if not graphs:
        raise ValueError("at least one graph is required")
    if int(history_steps) <= 0 or int(horizon_steps) <= 0:
        raise ValueError("history_steps and horizon_steps must be positive")
    return TensorContract(
        max_nodes=max(len(graph.get("physical_nodes", [])) for graph in graphs),
        max_physical_edges=max(len(graph.get("physical_edges", [])) for graph in graphs),
        max_flows=max(
            sum(row.get("flow_type") in FLOW_TYPES for row in graph.get("information_edges", []))
            for graph in graphs
        ),
        max_tasks=max(
            len({str(row.get("id")) for row in graph.get("task_nodes", [])}
                | {str(row.get("id")) for row in graph.get("source_task_snapshots", [])})
            for graph in graphs
        ),
        max_dag_edges=max(len(graph.get("task_dag_edges", [])) for graph in graphs),
        history_steps=int(history_steps),
        horizon_steps=int(horizon_steps),
    )


def contract_from_dict(value: Mapping[str, Any]) -> TensorContract:
    fields = (
        "max_nodes",
        "max_physical_edges",
        "max_flows",
        "max_tasks",
        "max_dag_edges",
        "history_steps",
        "horizon_steps",
    )
    return TensorContract(**{field: int(value[field]) for field in fields})


def _time_grid(graph: Mapping[str, Any]) -> list[float]:
    rows = []
    for key in ("source_physical_node_snapshots", "source_physical_edge_snapshots", "source_task_snapshots"):
        rows.extend(row for row in graph.get(key, []) if row.get("observed_time") is not None)
    times = sorted({round(float(row["observed_time"]), 9) for row in rows})
    if len(times) < 2:
        raise ValueError("at least two observed times are required")
    interval = times[1] - times[0]
    if interval <= 0 or any(not np.isclose(times[i] - times[i - 1], interval, atol=1e-8) for i in range(2, len(times))):
        raise ValueError("observed times must form a uniform grid")
    return times


def _index_map(values: Iterable[Any]) -> dict[str, int]:
    return {str(value): index for index, value in enumerate(values)}


def _node_ids(graph: Mapping[str, Any]) -> list[str]:
    return sorted({str(row["id"]) for row in graph.get("physical_nodes", [])}, key=natural_id_key)


def _task_ids(graph: Mapping[str, Any]) -> list[str]:
    ids = {str(row["id"]) for row in graph.get("task_nodes", [])}
    ids.update(str(row["id"]) for row in graph.get("source_task_snapshots", []))
    ids.update(str(row["task_id"]) for row in graph.get("source_offload_actions", []))
    ids.update(str(row["task_id"]) for row in graph.get("source_return_actions", []))
    ids.update(str(row["task_id"]) for row in graph.get("source_rb_actions", []))
    return sorted(ids, key=natural_id_key)


def _flow_rows(graph: Mapping[str, Any]) -> list[dict[str, Any]]:
    return sorted(
        [dict(row) for row in graph.get("information_edges", []) if row.get("flow_type") in FLOW_TYPES],
        key=lambda row: (
            float(row.get("first_time", 0.0)),
            FLOW_TYPES.index(str(row.get("flow_type"))),
            natural_id_key(row.get("task_id", "")),
            str(row.get("src", "")),
            str(row.get("dst", "")),
        ),
    )


def _node_feature(row: Mapping[str, Any]) -> list[float]:
    position = list(row.get("position", [0.0, 0.0, 0.0]))
    position.extend([0.0] * (3 - len(position)))
    return [
        float(position[0]),
        float(position[1]),
        float(position[2]),
        float(row.get("speed", 0.0)),
        float(row.get("acceleration", 0.0)),
        float(row.get("cpu", 0.0) or 0.0),
        float(row.get("storage", 0.0) or 0.0),
    ]


def _edge_feature(row: Mapping[str, Any]) -> list[float]:
    return [float(row.get(field, 0.0) or 0.0) for field in EDGE_FEATURES]


def _event_matches_flow(event: Mapping[str, Any], flow: Mapping[str, Any], node_ids: set[str]) -> bool:
    phase_type = "task_input" if event.get("phase") == "offload" else "result_return" if event.get("phase") == "return" else None
    if phase_type != flow.get("flow_type") or str(event.get("task_id")) != str(flow.get("task_id")):
        return False
    source = str(flow.get("src", "")).removeprefix("agent::")
    target = str(flow.get("dst", "")).removeprefix("agent::")
    return str(event.get("source")) in node_ids and str(event.get("target")) in node_ids and str(event.get("source")) == source and str(event.get("target")) == target


def _action_time_index(action: Mapping[str, Any], time_index: Mapping[float, int]) -> int:
    time = round(float(action["time"]), 9)
    if time not in time_index:
        raise ValueError(f"action time {time} is not on the observed time grid")
    return time_index[time]


def tensorize_seed_graph(
    graph: Mapping[str, Any],
    contract: TensorContract,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    times = _time_grid(graph)
    time_index = {time: index for index, time in enumerate(times)}
    node_vocab = _node_ids(graph)
    edge_vocab = sorted(
        [dict(row) for row in graph.get("physical_edges", [])],
        key=lambda row: (str(row.get("kind", "")), natural_id_key(row.get("src", "")), natural_id_key(row.get("dst", ""))),
    )
    flow_vocab = _flow_rows(graph)
    task_vocab = _task_ids(graph)
    dag_vocab = sorted(
        [dict(row) for row in graph.get("task_dag_edges", [])],
        key=lambda row: (natural_id_key(row.get("src", "")), natural_id_key(row.get("dst", ""))),
    )
    if len(node_vocab) > contract.max_nodes or len(edge_vocab) > contract.max_physical_edges or len(flow_vocab) > contract.max_flows or len(task_vocab) > contract.max_tasks or len(dag_vocab) > contract.max_dag_edges:
        raise ValueError("seed graph exceeds frozen tensor capacity")
    node_index = _index_map(node_vocab)
    edge_index = _index_map(row["id"] for row in edge_vocab)
    task_index = _index_map(task_vocab)
    node_count, edge_count, flow_count, task_count, dag_count = (
        contract.max_nodes,
        contract.max_physical_edges,
        contract.max_flows,
        contract.max_tasks,
        contract.max_dag_edges,
    )
    step_count = len(times)
    arrays: dict[str, np.ndarray] = {
        "time": np.asarray(times, dtype=np.float32),
        "node_state": np.zeros((step_count, node_count, len(NODE_FEATURES)), dtype=np.float32),
        "node_present": np.zeros((step_count, node_count), dtype=bool),
        "node_kind_index": np.full((node_count,), -1, dtype=np.int16),
        "physical_edge_state": np.zeros((step_count, edge_count, len(EDGE_FEATURES)), dtype=np.float32),
        "physical_edge_present": np.zeros((step_count, edge_count), dtype=bool),
        "physical_edge_endpoint_index": np.full((edge_count, 2), -1, dtype=np.int32),
        "physical_edge_kind_index": np.full((edge_count,), -1, dtype=np.int16),
        "flow_state": np.zeros((step_count, flow_count, len(FLOW_FEATURES)), dtype=np.float32),
        "flow_present": np.zeros((step_count, flow_count), dtype=bool),
        "flow_completed": np.zeros((step_count, flow_count), dtype=bool),
        "flow_valid": np.zeros((flow_count,), dtype=bool),
        "flow_endpoint_index": np.full((flow_count, 2), -1, dtype=np.int32),
        "flow_type_index": np.full((flow_count,), -1, dtype=np.int16),
        "flow_bearer_mask": np.zeros((step_count, flow_count, edge_count), dtype=bool),
        "flow_bearer_edge_index": np.full((step_count, flow_count), -1, dtype=np.int32),
        "task_state": np.zeros((step_count, task_count, len(TASK_FEATURES)), dtype=np.float32),
        "task_present": np.zeros((step_count, task_count), dtype=bool),
        "task_lifecycle_index": np.full((step_count, task_count), -1, dtype=np.int16),
        "task_node_index": np.full((step_count, task_count, len(TASK_ENDPOINT_FIELDS)), -1, dtype=np.int32),
        "task_action": np.zeros((step_count, task_count, len(ACTION_FEATURES)), dtype=np.float32),
        "task_action_present": np.zeros((step_count, task_count), dtype=bool),
        "task_action_node_index": np.full((step_count, task_count, 3), -1, dtype=np.int32),
        "dag_edge_index": np.full((2, dag_count), -1, dtype=np.int32),
        "dag_edge_valid": np.zeros((dag_count,), dtype=bool),
        "agent_node_index": np.full((node_count,), -1, dtype=np.int32),
    }
    for index, node in enumerate(node_vocab):
        static = next((row for row in graph.get("physical_nodes", []) if str(row["id"]) == node), {})
        arrays["node_kind_index"][index] = NODE_TYPES.index(str(static.get("kind", "")).lower()) if str(static.get("kind", "")).lower() in NODE_TYPES else -1
    for index, edge in enumerate(edge_vocab):
        arrays["physical_edge_endpoint_index"][index] = [node_index[str(edge["src"])], node_index[str(edge["dst"])]]
        kind = str(edge.get("kind", ""))
        arrays["physical_edge_kind_index"][index] = PHYSICAL_EDGE_TYPES.index(kind) if kind in PHYSICAL_EDGE_TYPES else -1
    for index, agent in enumerate(graph.get("agent_attachments", [])):
        agent_node = str(agent.get("physical_node_id", ""))
        if agent_node in node_index:
            arrays["agent_node_index"][index] = node_index[agent_node]
    for time, rows in _group_unique_rows(graph.get("source_physical_node_snapshots", []), "id"):
        ti = time_index[time]
        for row in rows:
            ni = node_index.get(str(row["id"]))
            if ni is not None:
                arrays["node_state"][ti, ni] = _node_feature(row)
                arrays["node_present"][ti, ni] = True
    for time, rows in _group_unique_rows(graph.get("source_physical_edge_snapshots", []), "id"):
        ti = time_index[time]
        for row in rows:
            ei = edge_index.get(str(row["id"]))
            if ei is not None:
                arrays["physical_edge_state"][ti, ei] = _edge_feature(row)
                arrays["physical_edge_present"][ti, ei] = True
    task_rows_by_key: dict[tuple[float, str], Mapping[str, Any]] = {}
    for row in graph.get("source_task_snapshots", []):
        key = (round(float(row["observed_time"]), 9), str(row["id"]))
        if key in task_rows_by_key:
            raise ValueError(f"duplicate task snapshot {key}")
        task_rows_by_key[key] = row
    for ti, time in enumerate(times):
        for task_id, qi in task_index.items():
            row = task_rows_by_key.get((time, task_id))
            if row is None or float(row.get("arrival_time", time)) > time or str(row.get("lifecycle_state")) == "to_generate":
                continue
            arrays["task_present"][ti, qi] = True
            deadline_time = float(row.get("deadline_time", float(row.get("arrival_time", time)) + float(row.get("deadline", 0.0))))
            delay = row.get("task_delay")
            arrays["task_state"][ti, qi] = [
                float(row.get("task_size", 0.0) or 0.0),
                float(row.get("return_size", 0.0) or 0.0),
                float(row.get("task_cpu", 0.0) or 0.0),
                max(deadline_time - time, 0.0),
                float(row.get("priority", 0.0) or 0.0),
                float(row.get("in_stage_transmitted_size", 0.0) or 0.0),
                float(row.get("computed_size", 0.0) or 0.0),
                float(delay or 0.0),
            ]
            lifecycle = str(row.get("lifecycle_state", ""))
            if lifecycle in LIFECYCLE_TYPES:
                arrays["task_lifecycle_index"][ti, qi] = LIFECYCLE_TYPES.index(lifecycle)
            for endpoint_index, field in enumerate(TASK_ENDPOINT_FIELDS):
                endpoint = row.get(field)
                if endpoint in node_index:
                    arrays["task_node_index"][ti, qi, endpoint_index] = node_index[str(endpoint)]
    for qi, task_id in enumerate(task_vocab):
        task_row = next((row for row in graph.get("task_nodes", []) if str(row.get("id")) == task_id), {})
        for endpoint_index, field in enumerate(TASK_ENDPOINT_FIELDS):
            endpoint = task_row.get(field)
            if endpoint not in node_index:
                continue
            arrays["task_node_index"][:, qi, endpoint_index] = np.where(
                arrays["task_present"][:, qi], node_index[str(endpoint)], -1
            )
    action_time_index = time_index
    max_rb_count = max([int(row.get("rb_count", 0) or 0) for row in graph.get("source_rb_actions", [])] + [1])
    for action in graph.get("source_offload_actions", []):
        qi = task_index.get(str(action.get("task_id")))
        target = str(action.get("target_node_id", ""))
        if qi is None or target not in node_index:
            raise ValueError(f"unknown offload action reference {action}")
        ti = _action_time_index(action, action_time_index)
        arrays["task_action"][ti, qi, 0] = 1.0
        arrays["task_action_node_index"][ti, qi, 0] = node_index[target]
        arrays["task_action_present"][ti, qi] = True
    for action in graph.get("source_return_actions", []):
        qi = task_index.get(str(action.get("task_id")))
        target = str(action.get("return_target_id", ""))
        if qi is None or target not in node_index:
            raise ValueError(f"unknown return action reference {action}")
        ti = _action_time_index(action, action_time_index)
        arrays["task_action"][ti, qi, 2] = 1.0
        arrays["task_action_node_index"][ti, qi, 1] = node_index[target]
        arrays["task_action_present"][ti, qi] = True
    for action in graph.get("source_rb_actions", []):
        qi = task_index.get(str(action.get("task_id")))
        assigned = str(action.get("assigned_to", ""))
        current = str(action.get("current_node_id", ""))
        if qi is None or assigned not in node_index or current not in node_index:
            raise ValueError(f"unknown RB action reference {action}")
        ti = _action_time_index(action, action_time_index)
        rb_count = float(action.get("rb_count", len(str(action.get("rb_indices", "")).split())))
        arrays["task_action"][ti, qi, 1] = 1.0
        arrays["task_action"][ti, qi, 3] = rb_count
        arrays["task_action"][ti, qi, 4] = rb_count / max_rb_count
        arrays["task_action_node_index"][ti, qi, 2] = node_index[assigned]
        arrays["task_action_present"][ti, qi] = True
    edge_event_rows = {str(row["id"]): row for row in graph.get("physical_edges", [])}
    events = list(graph.get("source_transfer_events", []))
    node_id_set = set(node_vocab)
    flow_fallback_count = 0
    for fi, flow in enumerate(flow_vocab):
        arrays["flow_valid"][fi] = True
        source = str(flow.get("src", "")).removeprefix("agent::")
        target = str(flow.get("dst", "")).removeprefix("agent::")
        if source not in node_index or target not in node_index:
            raise ValueError(f"unknown flow endpoint {flow.get('id')}")
        arrays["flow_endpoint_index"][fi] = [node_index[source], node_index[target]]
        arrays["flow_type_index"][fi] = FLOW_TYPES.index(str(flow["flow_type"]))
        matching_events = [event for event in events if _event_matches_flow(event, flow, node_id_set)]
        action_phase = "offload" if flow.get("flow_type") == "task_input" else "return"
        matching_actions = [
            action for action in (
                graph.get("source_offload_actions", []) if action_phase == "offload" else graph.get("source_return_actions", [])
            ) if str(action.get("task_id")) == str(flow.get("task_id"))
        ]
        creation_times = [float(action["time"]) for action in matching_actions]
        creation_times.extend(float(event["time"]) for event in matching_events)
        if not creation_times and flow.get("first_time") is not None:
            creation_times = [float(flow["first_time"])]
            flow_fallback_count += 1
        creation_time = min(creation_times) if creation_times else float(times[0])
        total = max(float(flow.get("total_data", 0.0) or 0.0), 0.0)
        for ti, time in enumerate(times):
            delivered_here = sum(float(event.get("delivered_data", 0.0) or 0.0) for event in matching_events if np.isclose(float(event["time"]), time, atol=1e-8))
            delivered_cumulative = sum(float(event.get("delivered_data", 0.0) or 0.0) for event in matching_events if float(event["time"]) <= time + 1e-8)
            completed = any(bool(event.get("flow_completed")) and float(event["time"]) <= time + 1e-8 for event in matching_events)
            completed_before_slot = any(
                bool(event.get("flow_completed")) and float(event["time"]) < time - 1e-8
                for event in matching_events
            )
            present = time + 1e-8 >= creation_time and not completed_before_slot
            arrays["flow_present"][ti, fi] = present
            arrays["flow_completed"][ti, fi] = completed
            if present:
                arrays["flow_state"][ti, fi] = [
                    total,
                    max(total - delivered_cumulative, 0.0),
                    delivered_cumulative,
                    delivered_here,
                    max(time - creation_time, 0.0),
                ]
            for event in matching_events:
                if np.isclose(float(event["time"]), time, atol=1e-8):
                    for physical_edge_id in event.get("path", []):
                        ei = edge_index.get(str(physical_edge_id))
                        if ei is not None:
                            arrays["flow_bearer_mask"][ti, fi, ei] = True
            bearer_indices = np.flatnonzero(arrays["flow_bearer_mask"][ti, fi])
            if bearer_indices.size:
                arrays["flow_bearer_edge_index"][ti, fi] = int(bearer_indices[0])
    for di, dag in enumerate(dag_vocab):
        source = str(dag.get("src"))
        target = str(dag.get("dst"))
        if source not in task_index or target not in task_index:
            raise ValueError(f"unknown DAG endpoint {dag.get('id')}")
        arrays["dag_edge_index"][:, di] = [task_index[source], task_index[target]]
        arrays["dag_edge_valid"][di] = True
    report = {
        "schema_version": SCHEMA_VERSION,
        "time_count": step_count,
        "time_start": times[0],
        "time_end": times[-1],
        "node_vocab": node_vocab,
        "edge_vocab": [str(row["id"]) for row in edge_vocab],
        "flow_vocab": [str(row["id"]) for row in flow_vocab],
        "task_vocab": task_vocab,
        "dag_vocab": [str(row["id"]) for row in dag_vocab],
        "flow_creation_fallback_count": flow_fallback_count,
        "counts": {
            "nodes": len(node_vocab),
            "physical_edges": len(edge_vocab),
            "flows": len(flow_vocab),
            "tasks": len(task_vocab),
            "dag_edges": len(dag_vocab),
        },
    }
    arrays["flow_delivered_this_slot"] = arrays["flow_state"][..., FLOW_FEATURES.index("delivered_this_slot")].astype(np.float64, copy=True)
    validate_seed_tensors(arrays, contract)
    return arrays, report


def _group_unique_rows(rows: Iterable[Mapping[str, Any]], key: str):
    grouped: dict[float, list[Mapping[str, Any]]] = {}
    seen: set[tuple[float, str]] = set()
    for row in rows:
        time = round(float(row["observed_time"]), 9)
        row_key = (time, str(row[key]))
        if row_key in seen:
            raise ValueError(f"duplicate snapshot {row_key}")
        seen.add(row_key)
        grouped.setdefault(time, []).append(row)
    return sorted(grouped.items())


def validate_seed_tensors(arrays: Mapping[str, np.ndarray], contract: TensorContract) -> dict[str, Any]:
    required = ("node_state", "node_present", "physical_edge_state", "flow_state", "task_state", "task_present")
    missing = [name for name in required if name not in arrays]
    if missing:
        raise ValueError(f"missing tensor arrays: {missing}")
    if any(not np.isfinite(value).all() for name, value in arrays.items() if np.issubdtype(value.dtype, np.floating)):
        raise ValueError("tensor arrays contain non-finite values")
    if np.any(arrays["node_state"][~arrays["node_present"]] != 0.0):
        raise ValueError("node padding must be zero")
    if np.any(arrays["task_state"][~arrays["task_present"]] != 0.0):
        raise ValueError("task padding must be zero")
    if np.any(arrays["physical_edge_state"][~arrays["physical_edge_present"]] != 0.0):
        raise ValueError("physical-edge padding must be zero")
    if np.any(arrays["flow_state"][~arrays["flow_present"]] != 0.0):
        raise ValueError("flow padding must be zero")
    for name in ("physical_edge_endpoint_index", "flow_endpoint_index", "task_node_index", "task_action_node_index", "dag_edge_index", "flow_bearer_edge_index"):
        value = arrays.get(name)
        if value is not None and np.any(value < -1):
            raise ValueError(f"{name} contains an invalid negative index")
    if np.any(arrays["flow_state"][..., 1] < -1e-7):
        raise ValueError("flow remaining data must be nonnegative")
    return {
        "tensor_valid": True,
        "schema_version": contract.schema_version,
        "time_count": int(arrays["time"].shape[0]),
        "node_capacity": contract.max_nodes,
        "physical_edge_capacity": contract.max_physical_edges,
        "flow_capacity": contract.max_flows,
        "task_capacity": contract.max_tasks,
        "dag_edge_capacity": contract.max_dag_edges,
        "present_counts": {
            "nodes": int(arrays["node_present"].sum()),
            "physical_edges": int(arrays["physical_edge_present"].sum()),
            "flows": int(arrays["flow_present"].sum()),
            "tasks": int(arrays["task_present"].sum()),
        },
    }
