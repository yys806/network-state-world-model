"""Auditable CPU behavior policies for PI-JWM AirFogSim trajectories."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from .formal_airfogsim_dataset_v1 import CPU_POLICY_IDS


@dataclass(frozen=True)
class CpuAllocationDecision:
    allocations: dict[str, float]
    rows: list[dict[str, Any]]


def _finite(value: Any, *, field: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


class CpuPolicyAllocator:
    """Allocate each physical node's CPU using one reproducible policy."""

    def __init__(
        self,
        policy_id: str,
        *,
        seed: int,
        max_tasks_per_node: int = 3,
    ) -> None:
        normalized = str(policy_id)
        if normalized not in CPU_POLICY_IDS:
            raise ValueError(f"unknown CPU policy: {normalized}")
        if isinstance(max_tasks_per_node, bool) or int(max_tasks_per_node) <= 0:
            raise ValueError("max_tasks_per_node must be a positive integer")
        self.policy_id = normalized
        self.max_tasks_per_node = int(max_tasks_per_node)
        self.rng = np.random.default_rng(int(seed))

    def _task_profile(self, task: Any, current_time: float) -> dict[str, Any]:
        task_id = str(task.getTaskId())
        arrival = _finite(task.getTaskArrivalTime(), field=f"{task_id} arrival")
        deadline = _finite(task.getTaskDeadline(), field=f"{task_id} deadline")
        task_cpu = _finite(task.getTaskCPU(), field=f"{task_id} task_cpu")
        computed = _finite(task.getComputedSize(), field=f"{task_id} computed")
        deadline_remaining = max(arrival + deadline - current_time, 0.0)
        return {
            "task": task,
            "task_id": task_id,
            "deadline_remaining": deadline_remaining,
            "remaining_cpu": max(task_cpu - computed, 0.0),
        }

    def _weight(self, profile: Mapping[str, Any]) -> float:
        if self.policy_id == "equal_share":
            return 1.0
        urgency = 1.0 / max(float(profile["deadline_remaining"]), 0.1)
        if self.policy_id == "deadline_aware":
            return urgency
        return urgency * float(self.rng.uniform(0.5, 1.5))

    def allocate(
        self,
        env: Any,
        computing_tasks: Mapping[str, Iterable[Any]],
    ) -> CpuAllocationDecision:
        allocations: dict[str, float] = {}
        rows: list[dict[str, Any]] = []
        current_time = _finite(getattr(env, "simulation_time", 0.0), field="simulation_time")

        for node_id in sorted(str(value) for value in computing_tasks):
            node = env._getNodeById(node_id)
            if node is None:
                continue
            capacity = _finite(
                node.getFogProfile().get("cpu", 0.0),
                field=f"{node_id} CPU capacity",
            )
            if capacity <= 0.0:
                continue
            profiles = [
                self._task_profile(task, current_time)
                for task in computing_tasks[node_id]
            ]
            profiles = [row for row in profiles if row["remaining_cpu"] > 0.0]
            queue_size = len(profiles)
            if not profiles:
                continue
            for profile in profiles:
                profile["policy_weight"] = self._weight(profile)
            if self.policy_id == "equal_share":
                profiles.sort(key=lambda row: row["task_id"])
            else:
                profiles.sort(
                    key=lambda row: (-float(row["policy_weight"]), row["task_id"])
                )
            selected = profiles[: self.max_tasks_per_node]
            weight_total = sum(float(row["policy_weight"]) for row in selected)
            if not math.isfinite(weight_total) or weight_total <= 0.0:
                raise ValueError(f"invalid CPU policy weights for {node_id}")
            for profile in selected:
                weight = float(profile["policy_weight"])
                allocated = capacity * weight / weight_total
                task_id = str(profile["task_id"])
                allocations[task_id] = allocated
                rows.append(
                    {
                        "task_id": task_id,
                        "node_id": node_id,
                        "policy_id": self.policy_id,
                        "policy_weight": weight,
                        "deadline_remaining": float(profile["deadline_remaining"]),
                        "queue_size": queue_size,
                        "allocated_cpu": allocated,
                        "node_cpu_capacity": capacity,
                        "allocated_fraction": allocated / capacity,
                    }
                )

        return CpuAllocationDecision(allocations=allocations, rows=rows)
