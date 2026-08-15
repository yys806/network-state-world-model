"""Auditable PI-JWM adapters for known AirFogSim contract boundaries.

The functions in this module do not modify AirFogSim source code.  They expose
the PI-JWM semantics used when generating new trajectories so the resulting
CPU and communication-energy labels can be checked independently.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any


ADAPTER_VERSION = "pi-jwm-airfogsim-contract-v1"
CPU_EVIDENCE = "pi_jwm_capacity_safe_adapter"
TRANSMISSION_EVIDENCE = "pi_jwm_direct_event_accounting_adapter"


def _plain_float(value: Any) -> float:
    if hasattr(value, "get"):
        value = value.get()
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


def capacity_safe_cpu_allocations(
    env: Any,
    computing_tasks: Mapping[str, Iterable[Any]],
    *,
    max_tasks_per_node: int = 3,
) -> dict[str, float]:
    """Allocate each node's CPU only among tasks assigned to that same node."""

    if isinstance(max_tasks_per_node, bool) or int(max_tasks_per_node) <= 0:
        raise ValueError("max_tasks_per_node must be a positive integer")
    limit = int(max_tasks_per_node)
    allocations: dict[str, float] = {}
    for node_id, task_values in computing_tasks.items():
        node = env._getNodeById(node_id)
        if node is None:
            continue
        capacity = float(node.getFogProfile().get("cpu", 0.0))
        if not math.isfinite(capacity):
            raise ValueError(f"non-finite CPU capacity for {node_id}")
        if capacity <= 0.0:
            continue
        selected_tasks = list(task_values)[:limit]
        if not selected_tasks:
            continue
        share = capacity / len(selected_tasks)
        for task in selected_tasks:
            allocations[str(task.getTaskId())] = share
    return allocations


def direct_transmission_totals(
    events: Iterable[Mapping[str, Any]],
    *,
    amount_field: str = "planned_capacity",
) -> tuple[dict[str, float], dict[str, float]]:
    """Aggregate direct communication capacity by source and target node."""

    sending: dict[str, float] = defaultdict(float)
    receiving: dict[str, float] = defaultdict(float)
    for event in events:
        amount = float(event[amount_field])
        if not math.isfinite(amount) or amount < 0.0:
            raise ValueError(f"invalid transmission amount: {amount}")
        sending[str(event["source"])] += amount
        receiving[str(event["target"])] += amount
    return dict(sending), dict(receiving)


def activated_transmission_events(
    env: Any,
    activated_profiles: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Project activated AirFogSim profiles into direct per-timeslot events."""

    events: list[dict[str, Any]] = []
    for profile in activated_profiles.values():
        task = profile["task"]
        route = list(task.getToOffloadRoute())
        if not route:
            continue
        rb_indices = [int(value) for value in profile.get("RB_Nos", [])]
        rates = env.channel_manager.getRateByChannelType(
            profile["tx_idx"],
            profile["rx_idx"],
            profile["channel_type"],
            rb_indices,
        )
        planned_capacity = sum(_plain_float(value) for value in rates) * float(
            env.simulation_interval
        )
        events.append(
            {
                "source": str(task.getCurrentNodeId()),
                "target": str(route[0]),
                "planned_capacity": planned_capacity,
            }
        )
    return events


def apply_transmission_totals(
    channel_manager: Any,
    sending: Mapping[str, float],
    receiving: Mapping[str, float],
) -> None:
    """Replace the current AirFogSim channel-energy boundary inputs."""

    channel_manager.setThisTimeslotTransSize(dict(sending), dict(receiving))


def encode_optional_value(value: Any, *, status: str) -> dict[str, Any]:
    """Encode availability without conflating a real zero with missing data."""

    normalized_status = str(status)
    if normalized_status in {"not_modeled", "not_modelled", "missing"}:
        if value is not None:
            raise ValueError("an unmodelled value must use None, never a numeric placeholder")
        return {"value": None, "observed_mask": 0, "status": normalized_status}
    if value is None:
        raise ValueError("an observed value cannot be None")
    return {"value": value, "observed_mask": 1, "status": normalized_status}
