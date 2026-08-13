"""Deterministic CPU inner rule for PI-JWM candidate rollouts.

CPU is not an action dimension in this contract. The caller must invoke this
rule with each candidate's post-communication compute-task set at every rollout
step.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass


CPU_INNER_RULE_VERSION = "PIJWM-CPU-Inner-Rule-v1"
_ABS_TOLERANCE = 1e-12
_REL_TOLERANCE = 1e-12


@dataclass(frozen=True)
class CpuTaskDemand:
    task_id: str
    node_id: str
    remaining_work: float


@dataclass(frozen=True)
class CpuTaskAllocation:
    task_id: str
    node_id: str
    remaining_work: float
    demand_rate: float
    allocated_cpu: float
    served_work: float


@dataclass(frozen=True)
class CpuNodeSummary:
    node_id: str
    capacity: float
    total_demand_rate: float
    total_allocated_cpu: float
    water_level: float
    task_count: int


@dataclass(frozen=True)
class CpuRuleDecision:
    rule_version: str
    slot_seconds: float
    allocations: tuple[CpuTaskAllocation, ...]
    node_summaries: tuple[CpuNodeSummary, ...]

    def as_allocation_dict(self) -> dict[str, float]:
        return {row.task_id: row.allocated_cpu for row in self.allocations}


def _finite_nonnegative(value: object, *, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite nonnegative number") from exc
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{field} must be a finite nonnegative number")
    return result


def _identifier(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _tolerance(scale: float) -> float:
    return max(_ABS_TOLERANCE, abs(scale) * _REL_TOLERANCE)


def _capped_equal_share(
    demand_rates: Mapping[str, float],
    capacity: float,
) -> tuple[dict[str, float], float]:
    task_ids = sorted(demand_rates)
    total_demand = math.fsum(demand_rates.values())
    target = min(capacity, total_demand)
    if not task_ids or target == 0.0:
        return {task_id: 0.0 for task_id in task_ids}, 0.0
    if total_demand <= capacity:
        return dict(demand_rates), max(demand_rates.values(), default=0.0)

    allocations: dict[str, float] = {}
    ordered_by_demand = sorted(task_ids, key=lambda task_id: (demand_rates[task_id], task_id))
    remaining_capacity = target
    water_level = 0.0
    for index, task_id in enumerate(ordered_by_demand):
        active_count = len(ordered_by_demand) - index
        equal_share = remaining_capacity / active_count
        demand = demand_rates[task_id]
        if demand <= equal_share:
            allocations[task_id] = demand
            remaining_capacity -= demand
            continue
        water_level = equal_share
        for active_task_id in ordered_by_demand[index:]:
            allocations[active_task_id] = min(demand_rates[active_task_id], water_level)
        remaining_capacity = 0.0
        break

    for task_id in task_ids:
        allocations.setdefault(task_id, demand_rates[task_id])

    allocated_total = math.fsum(allocations.values())
    residual = target - allocated_total
    tolerance = _tolerance(target)
    if abs(residual) > tolerance:
        raise ArithmeticError(
            f"capped equal sharing residual {residual} exceeds tolerance {tolerance}"
        )
    if residual > 0.0:
        for task_id in task_ids:
            headroom = demand_rates[task_id] - allocations[task_id]
            correction = min(headroom, residual)
            if correction > 0.0:
                allocations[task_id] += correction
                residual -= correction
            if residual <= 0.0:
                break
    elif residual < 0.0:
        for task_id in reversed(task_ids):
            correction = min(allocations[task_id], -residual)
            if correction > 0.0:
                allocations[task_id] -= correction
                residual += correction
            if residual >= 0.0:
                break

    return allocations, water_level


def allocate_work_conserving_cpu(
    tasks: Iterable[CpuTaskDemand],
    node_capacities: Mapping[str, float],
    slot_seconds: float,
) -> CpuRuleDecision:
    """Allocate CPU for one candidate's post-communication compute-task set."""

    slot = _finite_nonnegative(slot_seconds, field="slot_seconds")
    if slot == 0.0:
        raise ValueError("slot_seconds must be greater than zero")

    capacities: dict[str, float] = {}
    for raw_node_id, raw_capacity in node_capacities.items():
        node_id = _identifier(raw_node_id, field="capacity node_id")
        capacities[node_id] = _finite_nonnegative(
            raw_capacity,
            field=f"CPU capacity for {node_id}",
        )

    normalized_tasks: list[CpuTaskDemand] = []
    seen_task_ids: set[str] = set()
    for task in tasks:
        if not isinstance(task, CpuTaskDemand):
            raise TypeError("tasks must contain CpuTaskDemand values")
        task_id = _identifier(task.task_id, field="task_id")
        node_id = _identifier(task.node_id, field=f"node_id for {task_id}")
        if task_id in seen_task_ids:
            raise ValueError(f"duplicate task_id: {task_id}")
        seen_task_ids.add(task_id)
        remaining_work = _finite_nonnegative(
            task.remaining_work,
            field=f"remaining_work for {task_id}",
        )
        if node_id not in capacities:
            raise ValueError(f"missing CPU capacity for node_id: {node_id}")
        normalized_tasks.append(CpuTaskDemand(task_id, node_id, remaining_work))

    grouped: dict[str, list[CpuTaskDemand]] = defaultdict(list)
    for task in normalized_tasks:
        grouped[task.node_id].append(task)

    allocation_rows: list[CpuTaskAllocation] = []
    node_summaries: list[CpuNodeSummary] = []
    for node_id in sorted(grouped):
        node_tasks = sorted(grouped[node_id], key=lambda task: task.task_id)
        demand_rates: dict[str, float] = {}
        for task in node_tasks:
            demand_rate = task.remaining_work / slot
            if not math.isfinite(demand_rate):
                raise ValueError(f"demand_rate must be finite for task_id: {task.task_id}")
            demand_rates[task.task_id] = demand_rate
        capacity = capacities[node_id]
        try:
            total_demand = math.fsum(demand_rates.values())
        except OverflowError as exc:
            raise ValueError(
                f"total demand_rate must be finite for node_id: {node_id}"
            ) from exc
        if not math.isfinite(total_demand):
            raise ValueError(f"total demand_rate must be finite for node_id: {node_id}")
        allocations, water_level = _capped_equal_share(demand_rates, capacity)
        total_allocated = math.fsum(allocations.values())
        target = min(capacity, total_demand)
        tolerance = _tolerance(max(capacity, total_demand, 1.0))

        if total_allocated > capacity + tolerance:
            raise ArithmeticError(f"CPU capacity exceeded for node_id: {node_id}")
        if abs(total_allocated - target) > tolerance:
            raise ArithmeticError(f"CPU work conservation failed for node_id: {node_id}")

        for task in node_tasks:
            allocated_cpu = allocations[task.task_id]
            demand_rate = demand_rates[task.task_id]
            if allocated_cpu < 0.0 or allocated_cpu > demand_rate + tolerance:
                raise ArithmeticError(f"task demand bound failed for task_id: {task.task_id}")
            served_work = min(task.remaining_work, allocated_cpu * slot)
            allocation_rows.append(
                CpuTaskAllocation(
                    task_id=task.task_id,
                    node_id=node_id,
                    remaining_work=task.remaining_work,
                    demand_rate=demand_rate,
                    allocated_cpu=allocated_cpu,
                    served_work=served_work,
                )
            )

        node_summaries.append(
            CpuNodeSummary(
                node_id=node_id,
                capacity=capacity,
                total_demand_rate=total_demand,
                total_allocated_cpu=total_allocated,
                water_level=water_level,
                task_count=len(node_tasks),
            )
        )

    return CpuRuleDecision(
        rule_version=CPU_INNER_RULE_VERSION,
        slot_seconds=slot,
        allocations=tuple(allocation_rows),
        node_summaries=tuple(node_summaries),
    )
