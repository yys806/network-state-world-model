"""Causal raw-trajectory helpers for the PI-JWM/AirFogSim boundary.

The helpers in this module operate before Dataset/Tensor construction.  They
separate simulator-internal future schedules from the observable decision,
derive canonical acceleration from present and past speed only, and aggregate
real slot execution ledgers by stable task ID.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any


FUTURE_TASK_TOLERANCE_S = 1e-9
NO_PREVIOUS_SPEED = "NO_PREVIOUS_SPEED_IN_TRAJECTORY"


def partition_tasks_at_decision(
    tasks: Iterable[Mapping[str, Any]],
    decision_time_s: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split task rows into causal observation and internal future schedule."""

    decision_time = float(decision_time_s)
    if not math.isfinite(decision_time):
        raise ValueError("decision_time_s must be finite")
    observable: list[dict[str, Any]] = []
    internal_future: list[dict[str, Any]] = []
    for source in tasks:
        row = dict(source)
        arrival_time = float(row["arrival_time_s"])
        if not math.isfinite(arrival_time):
            raise ValueError("arrival_time_s must be finite")
        if arrival_time <= decision_time + FUTURE_TASK_TOLERANCE_S:
            row["visibility"] = "decision_observable"
            observable.append(row)
        else:
            row["visibility"] = "internal_future_schedule"
            internal_future.append(row)
    key = lambda row: str(row["task_id"])
    return sorted(observable, key=key), sorted(internal_future, key=key)


def attach_canonical_acceleration(
    entities: Iterable[Mapping[str, Any]],
    *,
    previous_speed_by_entity: Mapping[str, float] | None,
    delta_t_s: float | None,
) -> list[dict[str, Any]]:
    """Attach backward finite-difference acceleration and an explicit mask."""

    valid_delta = delta_t_s is not None and math.isfinite(float(delta_t_s)) and float(delta_t_s) > 0
    result: list[dict[str, Any]] = []
    for source in entities:
        row = dict(source)
        entity_id = str(row["entity_id"])
        current_speed = float(row["speed_mps"])
        previous_speed = (
            None
            if previous_speed_by_entity is None
            else previous_speed_by_entity.get(entity_id)
        )
        if previous_speed is None or not valid_delta:
            row["canonical_acceleration_mps2"] = None
            row["canonical_acceleration_observed_mask"] = False
            row["canonical_acceleration_missing_reason"] = NO_PREVIOUS_SPEED
        else:
            row["canonical_acceleration_mps2"] = (
                current_speed - float(previous_speed)
            ) / float(delta_t_s)
            row["canonical_acceleration_observed_mask"] = True
            row["canonical_acceleration_missing_reason"] = None
        row["canonical_acceleration_source"] = "backward_speed_difference_current_and_history"
        result.append(row)
    return result


def aggregate_slot_outcomes(
    *,
    transfer_events: Iterable[Mapping[str, Any]],
    computed_before: Mapping[str, float],
    computed_after: Mapping[str, float],
) -> dict[str, dict[str, float]]:
    """Aggregate real communication events and Task computed-size deltas."""

    delivered: dict[str, float] = {}
    for event in transfer_events:
        task_id = str(event["task_id"])
        amount = float(event["delivered_data"])
        delivered[task_id] = delivered.get(task_id, 0.0) + amount

    served: dict[str, float] = {}
    for task_id in sorted(set(computed_before) | set(computed_after)):
        before = float(computed_before.get(task_id, 0.0))
        after = float(computed_after.get(task_id, before))
        delta = after - before
        if delta < -1e-9:
            raise ValueError(f"computed work decreased for {task_id}")
        if delta > 1e-12:
            served[str(task_id)] = delta
    return {
        "delivered_data_by_task": dict(sorted(delivered.items())),
        "served_cpu_work_by_task": served,
    }
