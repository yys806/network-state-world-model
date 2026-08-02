"""Real system-outcome targets aligned to formal PI-JWM trajectory tensors."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


SCHEMA_VERSION = "PI-JWM-formal-system-targets-v1"


def _normalized_time(value: Any) -> float:
    return round(float(value), 5)


def _time_lookup(time_values: np.ndarray) -> tuple[dict[float, int], float]:
    values = np.asarray(time_values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.all(np.isfinite(values)):
        raise ValueError("time grid must be a finite one-dimensional array with at least two steps")
    normalized_values = np.asarray([_normalized_time(value) for value in values])
    differences = np.diff(normalized_values)
    if np.any(differences <= 0) or not np.allclose(differences, differences[0], atol=1e-8):
        raise ValueError("time grid must be strictly increasing and uniformly spaced")
    lookup = {float(value): index for index, value in enumerate(normalized_values)}
    if len(lookup) != len(values):
        raise ValueError("time grid contains duplicate normalized values")
    return lookup, float(differences[0])


def _require_time(value: Any, lookup: Mapping[float, int]) -> int:
    normalized = _normalized_time(value)
    if normalized not in lookup:
        raise ValueError(f"event time {normalized} is not on the tensor time grid")
    return int(lookup[normalized])


def build_system_target_arrays(
    *,
    time_values: np.ndarray,
    node_vocab: Sequence[str],
    task_vocab: Sequence[str],
    task_snapshots: Sequence[Mapping[str, Any]],
    energy_rows: Sequence[Mapping[str, Any]],
    transfer_events: Sequence[Mapping[str, Any]],
) -> dict[str, np.ndarray]:
    """Build causal per-step outcome labels without inventing unavailable quantities."""

    time_values = np.asarray(time_values, dtype=np.float64)
    time_index, interval = _time_lookup(time_values)
    node_index = {str(value): index for index, value in enumerate(node_vocab)}
    task_index = {str(value): index for index, value in enumerate(task_vocab)}
    if len(node_index) != len(node_vocab) or len(task_index) != len(task_vocab):
        raise ValueError("node and task vocabularies must contain unique IDs")

    step_count = len(time_values)
    node_count = len(node_vocab)
    task_count = len(task_vocab)
    arrays = {
        "time": time_values.astype(np.float32),
        "task_completion_event": np.zeros((step_count, task_count), dtype=bool),
        "completed_task_delay": np.zeros((step_count, task_count), dtype=np.float32),
        "completed_task_delay_valid": np.zeros((step_count, task_count), dtype=bool),
        "uav_energy_delta": np.zeros((step_count, node_count), dtype=np.float32),
        "uav_energy_valid": np.zeros((step_count, node_count), dtype=bool),
        "source_service_delta": np.zeros((step_count, node_count), dtype=np.float32),
        "source_population_valid": np.zeros(node_count, dtype=bool),
        "delivered_data_total": np.zeros(step_count, dtype=np.float64),
    }

    completed_by_task: dict[str, Mapping[str, Any]] = {}
    for row in task_snapshots:
        source = str(row.get("source"))
        if source not in node_index:
            raise ValueError(f"unknown source node in task snapshot: {source}")
        arrays["source_population_valid"][node_index[source]] = True
        if str(row.get("lifecycle_state")) != "finished" or row.get("completion_time") is None:
            continue
        task_id = str(row.get("id"))
        if task_id not in task_index:
            raise ValueError(f"unknown task ID in completion snapshot: {task_id}")
        previous = completed_by_task.get(task_id)
        if previous is None or float(row["completion_time"]) < float(previous["completion_time"]):
            completed_by_task[task_id] = row

    for task_id, row in completed_by_task.items():
        ti = _require_time(row["completion_time"], time_index)
        qi = task_index[task_id]
        delay = row.get("task_delay")
        if delay is None or not np.isfinite(float(delay)) or float(delay) < 0:
            raise ValueError(f"completed task {task_id} lacks a valid direct task_delay")
        source = str(row.get("source"))
        if source not in node_index:
            raise ValueError(f"unknown source node for completed task {task_id}: {source}")
        arrays["task_completion_event"][ti, qi] = True
        arrays["completed_task_delay"][ti, qi] = float(delay)
        arrays["completed_task_delay_valid"][ti, qi] = True
        arrays["source_service_delta"][ti, node_index[source]] += 1.0

    for row in energy_rows:
        node_id = str(row.get("uav_id"))
        if node_id not in node_index:
            raise ValueError(f"unknown UAV ID in energy row: {node_id}")
        consumed = float(row.get("energy_before", 0.0)) - float(row.get("energy_after", 0.0))
        if not np.isfinite(consumed) or consumed < -1e-8:
            raise ValueError(f"negative energy consumption for {node_id}")
        ti = _require_time(float(row["time"]) + interval, time_index)
        ni = node_index[node_id]
        arrays["uav_energy_delta"][ti, ni] += max(consumed, 0.0)
        arrays["uav_energy_valid"][ti, ni] = True

    for row in transfer_events:
        delivered = float(row.get("delivered_data", 0.0) or 0.0)
        if not np.isfinite(delivered) or delivered < 0:
            raise ValueError("delivered data must be finite and non-negative")
        ti = _require_time(row["time"], time_index)
        arrays["delivered_data_total"][ti] += delivered

    if int(arrays["task_completion_event"].sum()) != len(completed_by_task):
        raise RuntimeError("each completed task must produce exactly one completion event")
    return arrays


__all__ = ["SCHEMA_VERSION", "build_system_target_arrays"]
