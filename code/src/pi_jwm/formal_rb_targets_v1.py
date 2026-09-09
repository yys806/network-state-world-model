"""Direct per-RB outcome labels for the formal PI-JWM tensor contract."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


SCHEMA_VERSION = "PI-JWM-formal-per-rb-targets-v1"
OUTCOME_TEMPORAL_ROLE = "outcome_only_not_same_frame_decision_input"
RATE_UNIT = "AirFogSim data-unit/s"


def _runtime_pair_key(
    row: Mapping[str, Any], *, source_key: str, target_key: str
) -> tuple[float, str, str, str]:
    try:
        time = round(float(row["time"]), 5)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("RB runtime rows require a numeric time") from error
    task_id = str(row.get("task_id", ""))
    source = str(row.get(source_key, ""))
    target = str(row.get(target_key, ""))
    if not task_id or not source or not target:
        raise ValueError("RB runtime rows require task, source, and target IDs")
    return time, task_id, source, target


def _rb_indices(row: Mapping[str, Any]) -> list[int]:
    values = row.get("rb_indices")
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or not values:
        raise ValueError("RB runtime rows require a non-empty rb_indices list")
    indices = list(values)
    if any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) for value in indices):
        raise ValueError("RB runtime rows require integer RB indices")
    if len(set(indices)) != len(indices) or any(int(value) < 0 for value in indices):
        raise ValueError("RB runtime rows require unique non-negative RB indices")
    return [int(value) for value in indices]


def build_runtime_rb_outcome_observations(
    graph: Mapping[str, Any], *, slot_seconds: float
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Rebuild direct per-RB outcome labels from logged actions and transfers.

    Older formal source bundles do not contain ``source_rb_observations``.  In
    those bundles, each RB action has exactly one runtime transfer event with
    the same time, task, source, and target.  The action supplies RB identity;
    the runtime event supplies the observed physical edge and slot capacity.
    """

    if not np.isfinite(slot_seconds) or float(slot_seconds) <= 0.0:
        raise ValueError("slot_seconds must be finite and positive")
    actions = list(graph.get("source_rb_actions", []))
    events = list(graph.get("source_transfer_events", []))
    if not actions and not events:
        return [], {
            "source": "no_runtime_rb_records",
            "action_count": 0,
            "event_count": 0,
            "observation_count": 0,
        }

    actions_by_key: dict[tuple[float, str, str, str], Mapping[str, Any]] = {}
    for action in actions:
        key = _runtime_pair_key(
            action, source_key="current_node_id", target_key="assigned_to"
        )
        if key in actions_by_key:
            raise ValueError(f"duplicate RB action for runtime key {key}")
        actions_by_key[key] = action

    events_by_key: dict[tuple[float, str, str, str], Mapping[str, Any]] = {}
    for event in events:
        key = _runtime_pair_key(event, source_key="source", target_key="target")
        if key in events_by_key:
            raise ValueError(f"duplicate transfer event for runtime key {key}")
        events_by_key[key] = event

    missing_actions = sorted(set(events_by_key) - set(actions_by_key))
    if missing_actions:
        raise ValueError(
            "every transfer event requires one matching RB action"
        )

    observations: list[dict[str, Any]] = []
    for key in sorted(actions_by_key):
        action = actions_by_key[key]
        action_indices = _rb_indices(action)
        event = events_by_key.get(key)
        edge_id = next(
            (
                str(edge.get("id"))
                for edge in graph.get("physical_edges", [])
                if str(edge.get("src")) == key[2] and str(edge.get("dst")) == key[3]
            ),
            "",
        )
        planned_capacity = None
        if event is not None:
            if action_indices != _rb_indices(event):
                raise ValueError(f"RB action and transfer event disagree on RB indices for {key}")
            path = event.get("path")
            if not isinstance(path, Sequence) or isinstance(path, (str, bytes)) or len(path) != 1:
                raise ValueError(f"transfer event requires exactly one physical path edge for {key}")
            edge_id = str(path[0])
            if not edge_id:
                raise ValueError(f"transfer event requires a physical path edge for {key}")
            try:
                planned_capacity = float(event["planned_capacity"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"transfer event requires finite planned_capacity for {key}") from error
            if not np.isfinite(planned_capacity) or planned_capacity < 0.0:
                raise ValueError(f"transfer event requires finite planned_capacity for {key}")
        if not edge_id:
            raise ValueError(f"cannot resolve physical edge for RB runtime key {key}")
        for rb_index in action_indices:
            observations.append(
                {
                    "time": key[0],
                    "physical_edge_id": edge_id,
                    "rb_index": rb_index,
                    "rate_per_s": None if planned_capacity is None else planned_capacity / float(slot_seconds),
                    "observed_mask": planned_capacity is not None,
                    "missing_reason": None if planned_capacity is not None else "runtime_transfer_event_unavailable",
                    "temporal_role": OUTCOME_TEMPORAL_ROLE,
                    "source_method": "AirFogSim_RB_action_plus_runtime_transfer_event",
                    "flow_id": None if event is None else event.get("event_id"),
                }
            )
    return observations, {
        "source": "derived_from_action_and_runtime_event",
        "action_count": len(actions),
        "event_count": len(events),
        "observation_count": len(observations),
        "unobserved_count": sum(row["observed_mask"] is not True for row in observations),
        "slot_seconds": float(slot_seconds),
    }


def _time_lookup(time_values: np.ndarray) -> dict[float, int]:
    values = np.asarray(time_values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.all(np.isfinite(values)):
        raise ValueError("time grid must be a finite one-dimensional array with at least two steps")
    # Tensor storage is float32, so values such as 16.2 may be represented as
    # 16.2000007629. Five decimal places keeps slot identity stable across the
    # source JSON and float32 tensor boundary without changing the time grid.
    normalized = [round(float(value), 5) for value in values]
    if len(set(normalized)) != len(normalized):
        raise ValueError("time grid contains duplicate values")
    if any(right <= left for left, right in zip(normalized, normalized[1:])):
        raise ValueError("time grid must be strictly increasing")
    return {value: index for index, value in enumerate(normalized)}


def validate_label_window_alignment(
    *, history_times: np.ndarray, label_times: np.ndarray
) -> None:
    """Require every label time to be strictly after the observed history."""

    history = np.asarray(history_times, dtype=np.float64)
    labels = np.asarray(label_times, dtype=np.float64)
    if history.ndim != 1 or labels.ndim != 1 or not len(history) or not len(labels):
        raise ValueError("history and label times must be non-empty one-dimensional arrays")
    if not np.all(labels > history[-1] + 1e-8):
        raise ValueError("label times must be strictly after the history")


def _require_observation_row(row: Mapping[str, Any], edge_index: Mapping[str, int], time_index: Mapping[float, int], n_rb: int) -> tuple[int, int, bool]:
    edge_id = str(row.get("physical_edge_id", ""))
    if edge_id not in edge_index:
        raise ValueError(f"unknown physical edge in RB observation: {edge_id}")
    try:
        time = round(float(row["time"]), 5)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("RB observation requires a numeric time") from error
    if time not in time_index:
        raise ValueError(f"RB observation time {time} is not on the tensor grid")
    rb_index = row.get("rb_index")
    if isinstance(rb_index, bool) or not isinstance(rb_index, (int, np.integer)):
        raise ValueError("RB observation requires an integer rb_index")
    if not 0 <= int(rb_index) < n_rb:
        raise ValueError(f"RB index {rb_index} is outside [0, {n_rb})")
    if row.get("temporal_role") not in {
        OUTCOME_TEMPORAL_ROLE,
        "runtime_outcome_missing",
    }:
        raise ValueError("per-RB labels require an outcome-only temporal role")
    observed = row.get("observed_mask") is True
    if observed:
        rate = row.get("rate_per_s")
        if rate is None or not np.isfinite(float(rate)) or float(rate) < 0.0:
            raise ValueError("observed RB rate must be finite and non-negative")
        if row.get("missing_reason") is not None:
            raise ValueError("observed RB labels cannot carry a missing reason")
    elif row.get("rate_per_s") is not None:
        raise ValueError("unobserved RB rate must remain null")
    return time_index[time], edge_index[edge_id], observed


def build_per_rb_target_arrays(
    *,
    time_values: np.ndarray,
    edge_vocab: Sequence[str],
    n_rb: int,
    observations: Sequence[Mapping[str, Any]],
    edge_capacity: int | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Materialize direct `(time, physical_edge, RB)` outcome labels.

    Activity is defined from the directly observed per-RB runtime rate.  An
    unobserved RB remains zero-valued only as tensor storage and is excluded by
    both masks; no inactive label is fabricated for it.
    """

    if isinstance(n_rb, bool) or not isinstance(n_rb, int) or n_rb <= 0:
        raise ValueError("n_rb must be a positive integer")
    edges = [str(value) for value in edge_vocab]
    if len(set(edges)) != len(edges):
        raise ValueError("edge_vocab must contain unique IDs")
    capacity = len(edges) if edge_capacity is None else int(edge_capacity)
    if capacity < len(edges):
        raise ValueError("edge_capacity cannot be smaller than edge_vocab")
    time_index = _time_lookup(np.asarray(time_values))
    edge_index = {edge_id: index for index, edge_id in enumerate(edges)}
    shape = (len(time_index), capacity, n_rb)
    arrays = {
        "time": np.asarray(time_values, dtype=np.float32),
        "link_activity": np.zeros(shape, dtype=bool),
        "link_activity_mask": np.zeros(shape, dtype=bool),
        "link_rate_by_rb": np.zeros(shape, dtype=np.float32),
        "link_rate_by_rb_mask": np.zeros(shape, dtype=bool),
    }
    seen: dict[tuple[int, int, int], Mapping[str, Any]] = {}
    observed_count = 0
    collision_count = 0
    for row in observations:
        ti, ei, observed = _require_observation_row(row, edge_index, time_index, n_rb)
        key = (ti, ei, int(row["rb_index"]))
        if key in seen:
            previous = seen[key]
            same_flow = row.get("flow_id") is None or previous.get("flow_id") is None or str(row.get("flow_id")) == str(previous.get("flow_id"))
            if same_flow:
                raise ValueError(f"duplicate RB observation {key}")
            previous_observed = previous.get("observed_mask") is True
            previous_rate = None if previous.get("rate_per_s") is None else float(previous["rate_per_s"])
            current_rate = None if row.get("rate_per_s") is None else float(row["rate_per_s"])
            if previous_observed and observed and not np.isclose(previous_rate, current_rate, rtol=0.0, atol=1e-7):
                raise ValueError(f"conflicting cross-flow RB observation {key}")
            if observed and not previous_observed:
                seen[key] = row
            collision_count += 1
            continue
        seen[key] = row
    for key, row in seen.items():
        observed = row.get("observed_mask") is True
        if not observed:
            continue
        rate = float(row["rate_per_s"])
        arrays["link_rate_by_rb"][key] = rate
        arrays["link_activity"][key] = rate > 0.0
        arrays["link_rate_by_rb_mask"][key] = True
        arrays["link_activity_mask"][key] = True
        observed_count += 1
    return arrays, {
        "schema_version": SCHEMA_VERSION,
        "edge_count": len(edges),
        "edge_capacity": capacity,
        "n_rb": n_rb,
        "observed_label_count": observed_count,
        "unobserved_slot_count": int(np.prod(shape) - observed_count),
        "collision_count": collision_count,
        "rate_unit": RATE_UNIT,
        "activity_definition": "direct_observed_rate_per_s_gt_zero",
        "source_temporal_role": OUTCOME_TEMPORAL_ROLE,
    }


__all__ = [
    "OUTCOME_TEMPORAL_ROLE",
    "RATE_UNIT",
    "SCHEMA_VERSION",
    "build_per_rb_target_arrays",
    "build_runtime_rb_outcome_observations",
    "validate_label_window_alignment",
]
