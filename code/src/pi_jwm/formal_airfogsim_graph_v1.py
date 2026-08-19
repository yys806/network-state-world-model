"""Formal-v1 graph boundary and tensor extensions for PI-JWM."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping

import numpy as np

from .airfogsim_tensor_v2 import (
    ACTION_FEATURES,
    LIFECYCLE_TYPES,
    TensorContract,
    natural_id_key,
    tensorize_seed_graph,
)


SCHEMA_VERSION = "PI-JWM-AirFogSim-formal-tensor-v1"
FORMAL_ACTION_FEATURES = (*ACTION_FEATURES, "cpu", "cpu_allocated", "cpu_fraction")
FORMAL_DAG_STATE_FEATURES = (
    "parent_count",
    "unfinished_parent_count",
    "release_ready",
)


def validate_formal_graph_boundary(graph: Mapping[str, Any]) -> None:
    """Enforce precedence-only DAG semantics for formal dataset v1."""

    for edge in graph.get("task_dag_edges", []):
        if edge.get("data_mb") is not None:
            raise ValueError("formal v1 DAG payload must remain unmodelled")
    dependency_flows = [
        row
        for row in graph.get("information_edges", [])
        if row.get("flow_type") == "dependency_data"
    ]
    if dependency_flows:
        raise ValueError("formal v1 forbids synthetic DAG dependency information flows")


def _matched_time_index(arrays: Mapping[str, np.ndarray], value: float) -> int:
    matches = np.flatnonzero(
        np.isclose(
            arrays["time"].astype(np.float64),
            float(value),
            rtol=0.0,
            atol=1e-5,
        )
    )
    if matches.size != 1:
        raise ValueError(f"CPU action time {value} is not on the observed grid")
    return int(matches[0])


def _tensorize_cpu_actions(
    graph: Mapping[str, Any],
    arrays: dict[str, np.ndarray],
    report: Mapping[str, Any],
) -> None:
    base_action = arrays["task_action"]
    extended = np.zeros(
        (*base_action.shape[:-1], len(FORMAL_ACTION_FEATURES)),
        dtype=base_action.dtype,
    )
    extended[..., : base_action.shape[-1]] = base_action
    arrays["task_action"] = extended

    base_node_index = arrays["task_action_node_index"]
    extended_node_index = np.full(
        (*base_node_index.shape[:-1], base_node_index.shape[-1] + 1),
        -1,
        dtype=base_node_index.dtype,
    )
    extended_node_index[..., : base_node_index.shape[-1]] = base_node_index
    arrays["task_action_node_index"] = extended_node_index

    task_index = {task_id: index for index, task_id in enumerate(report["task_vocab"])}
    node_index = {node_id: index for index, node_id in enumerate(report["node_vocab"])}
    seen: set[tuple[int, int]] = set()
    for action in graph.get("source_cpu_actions", []):
        time_key = round(float(action["time"]), 6)
        task_id = str(action.get("task_id", ""))
        node_id = str(action.get("node_id", ""))
        if task_id not in task_index or node_id not in node_index:
            raise ValueError(f"unknown CPU action reference {action}")
        ti = _matched_time_index(arrays, time_key)
        qi = task_index[task_id]
        key = (ti, qi)
        if key in seen:
            raise ValueError(f"duplicate CPU action for {task_id} at {time_key}")
        seen.add(key)
        allocated = float(action.get("allocated_cpu", 0.0))
        capacity = float(action.get("node_cpu_capacity", 0.0))
        fraction = (
            float(action["allocated_fraction"])
            if action.get("allocated_fraction") is not None
            else allocated / capacity if capacity > 0.0 else 0.0
        )
        arrays["task_action"][ti, qi, len(ACTION_FEATURES)] = 1.0
        arrays["task_action"][ti, qi, len(ACTION_FEATURES) + 1] = allocated
        arrays["task_action"][ti, qi, len(ACTION_FEATURES) + 2] = fraction
        arrays["task_action_node_index"][ti, qi, -1] = node_index[node_id]
        arrays["task_action_present"][ti, qi] = True


def _tensorize_dag_state(
    graph: Mapping[str, Any],
    arrays: dict[str, np.ndarray],
    report: Mapping[str, Any],
    contract: TensorContract,
) -> None:
    task_index = {task_id: index for index, task_id in enumerate(report["task_vocab"])}
    dags = sorted(
        [dict(row) for row in graph.get("task_dag_edges", [])],
        key=lambda row: (
            natural_id_key(row.get("src", "")),
            natural_id_key(row.get("dst", "")),
        ),
    )
    step_count = arrays["time"].shape[0]
    task_count = contract.max_tasks
    dag_count = contract.max_dag_edges
    arrays["dag_edge_present"] = np.zeros((step_count, dag_count), dtype=bool)
    arrays["task_dag_state"] = np.zeros(
        (step_count, task_count, len(FORMAL_DAG_STATE_FEATURES)),
        dtype=np.float32,
    )
    arrays["task_dag_state_present"] = np.zeros((step_count, task_count), dtype=bool)
    incoming_by_child: dict[str, list[tuple[int, str]]] = defaultdict(list)
    times = [round(float(value), 6) for value in arrays["time"].tolist()]
    for edge_index, edge in enumerate(dags):
        source = str(edge.get("src", ""))
        target = str(edge.get("dst", ""))
        observed_time = round(float(edge.get("time", times[0])), 6)
        for ti, time in enumerate(times):
            if time + 1e-8 >= observed_time:
                arrays["dag_edge_present"][ti, edge_index] = True
        incoming_by_child[target].append((edge_index, source))

    finished_index = LIFECYCLE_TYPES.index("finished")
    for ti in range(step_count):
        for task_id, qi in task_index.items():
            observed_parents = [
                parent_id
                for edge_index, parent_id in incoming_by_child.get(task_id, [])
                if arrays["dag_edge_present"][ti, edge_index]
            ]
            if not observed_parents:
                if arrays["task_present"][ti, qi]:
                    arrays["task_dag_state"][ti, qi] = [0.0, 0.0, 1.0]
                    arrays["task_dag_state_present"][ti, qi] = True
                continue
            unfinished = 0
            for parent_id in observed_parents:
                parent_index = task_index[parent_id]
                parent_finished = bool(arrays["task_present"][ti, parent_index]) and int(
                    arrays["task_lifecycle_index"][ti, parent_index]
                ) == finished_index
                unfinished += int(not parent_finished)
            arrays["task_dag_state"][ti, qi] = [
                float(len(observed_parents)),
                float(unfinished),
                float(unfinished == 0),
            ]
            arrays["task_dag_state_present"][ti, qi] = True


def tensorize_formal_graph(
    graph: Mapping[str, Any],
    contract: TensorContract,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Extend the existing v2 tensors with formal CPU and DAG state arrays."""

    validate_formal_graph_boundary(graph)
    # v3 source bundles predate the return-action field rename. Normalize that
    # historical spelling explicitly while keeping the audited source intact.
    normalized_graph = dict(graph)
    legacy_return_actions = 0
    normalized_returns = []
    for action in graph.get("source_return_actions", []):
        row = dict(action)
        if "return_target_id" not in row and "target_node_id" in row:
            row["return_target_id"] = row.pop("target_node_id")
            legacy_return_actions += 1
        normalized_returns.append(row)
    normalized_graph["source_return_actions"] = normalized_returns
    arrays, base_report = tensorize_seed_graph(normalized_graph, contract)
    report = dict(base_report)
    _tensorize_cpu_actions(normalized_graph, arrays, report)
    _tensorize_dag_state(normalized_graph, arrays, report, contract)
    report.update(
        {
            "schema_version": SCHEMA_VERSION,
            "action_features": list(FORMAL_ACTION_FEATURES),
            "dag_state_features": list(FORMAL_DAG_STATE_FEATURES),
            "cpu_action_count": len(normalized_graph.get("source_cpu_actions", [])),
            "legacy_return_action_field_count": legacy_return_actions,
        }
    )
    return arrays, report
