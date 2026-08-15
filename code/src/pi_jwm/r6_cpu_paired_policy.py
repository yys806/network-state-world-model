"""Deterministic CPU policies used by the R6 paired closed-loop baseline."""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any


PAIRED_CPU_POLICY_IDS = (
    "equal_share",
    "deadline_aware",
    "feasible_exploration",
    "local_search",
)


@dataclass(frozen=True)
class PairedCpuAllocationDecision:
    allocations: dict[str, float]
    rows: list[dict[str, Any]]


def _finite(value: Any, *, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def project_cpu_allocations(weights: Mapping[str, Any], capacity: Any) -> dict[str, float]:
    """Project positive finite weights onto a node's CPU capacity."""

    limit = _finite(capacity, field="CPU capacity")
    if limit < 0.0:
        raise ValueError("CPU capacity cannot be negative")
    if not isinstance(weights, Mapping):
        raise ValueError("weights must be a mapping")
    clean: dict[str, float] = {}
    for task_id, value in weights.items():
        weight = _finite(value, field=f"weight {task_id}")
        if weight < 0.0:
            raise ValueError(f"weight {task_id} cannot be negative")
        if weight > 0.0:
            clean[str(task_id)] = weight
    total = sum(clean.values())
    if total <= 0.0 or limit <= 0.0:
        return {task_id: 0.0 for task_id in clean}
    scale = limit / total
    return {task_id: float(weight * scale) for task_id, weight in clean.items()}


class PairedCpuPolicyAllocator:
    """CPU-only policy allocator with a fixed, auditable local-search heuristic."""

    def __init__(self, policy_id: str, *, seed: int, max_tasks_per_node: int = 3) -> None:
        policy = str(policy_id)
        if policy not in PAIRED_CPU_POLICY_IDS:
            raise ValueError(f"unknown paired CPU policy: {policy}")
        if isinstance(max_tasks_per_node, bool) or int(max_tasks_per_node) <= 0:
            raise ValueError("max_tasks_per_node must be positive")
        self.policy_id = policy
        self.seed = int(seed)
        self.max_tasks_per_node = int(max_tasks_per_node)
        self.rng = random.Random(self.seed)

    @staticmethod
    def _profile(task: Any, current_time: float) -> dict[str, Any]:
        task_id = str(task.getTaskId())
        arrival = _finite(task.getTaskArrivalTime(), field=f"{task_id} arrival")
        deadline = _finite(task.getTaskDeadline(), field=f"{task_id} deadline")
        requested_cpu = _finite(task.getTaskCPU(), field=f"{task_id} task_cpu")
        computed = _finite(task.getComputedSize(), field=f"{task_id} computed")
        remaining = max(requested_cpu - computed, 0.0)
        return {
            "task_id": task_id,
            "deadline_remaining": max(arrival + deadline - current_time, 0.0),
            "remaining_cpu": remaining,
        }

    def _weight(self, profile: Mapping[str, Any], rank: int) -> tuple[float, str]:
        if self.policy_id == "equal_share":
            return 1.0, "equal"
        urgency = 1.0 / max(float(profile["deadline_remaining"]), 0.1)
        if self.policy_id == "deadline_aware":
            return urgency, "deadline_inverse"
        if self.policy_id == "feasible_exploration":
            return urgency * self.rng.uniform(0.5, 1.5), "seeded_deadline_perturbation"
        # Fixed local neighborhood around deadline-aware weights.  The rank and
        # multipliers are part of the protocol, not learned from outcomes.
        multiplier = (1.25, 1.0, 0.75)[min(rank, 2)]
        return urgency * multiplier, "deadline_local_search"

    def allocate(
        self,
        env: Any,
        computing_tasks: Mapping[str, Iterable[Any]],
    ) -> PairedCpuAllocationDecision:
        current_time = _finite(getattr(env, "simulation_time", 0.0), field="simulation_time")
        allocations: dict[str, float] = {}
        rows: list[dict[str, Any]] = []
        for node_id in sorted(str(value) for value in computing_tasks):
            node = env._getNodeById(node_id)
            if node is None:
                raise ValueError(f"CPU node is missing: {node_id}")
            capacity = _finite(node.getFogProfile().get("cpu", 0.0), field=f"{node_id} CPU capacity")
            profiles = [self._profile(task, current_time) for task in computing_tasks[node_id]]
            profiles = [profile for profile in profiles if profile["remaining_cpu"] > 0.0]
            if capacity <= 0.0 or not profiles:
                continue
            for profile in profiles:
                profile["policy_weight"] = 1.0
            if self.policy_id == "equal_share":
                profiles.sort(key=lambda profile: profile["task_id"])
            else:
                # First sort establishes a stable rank for local-search and
                # avoids using future simulator outcomes.
                profiles.sort(key=lambda profile: (float(profile["deadline_remaining"]), profile["task_id"]))
            selected = profiles[: self.max_tasks_per_node]
            weights: dict[str, float] = {}
            for rank, profile in enumerate(selected):
                weight, mechanism = self._weight(profile, rank)
                weights[profile["task_id"]] = weight
                profile["policy_weight"] = weight
                profile["mechanism"] = mechanism
                profile["search_rank"] = rank
            projected = project_cpu_allocations(weights, capacity)
            for profile in selected:
                task_id = profile["task_id"]
                allocated = projected.get(task_id, 0.0)
                allocations[task_id] = allocated
                rows.append(
                    {
                        "task_id": task_id,
                        "node_id": node_id,
                        "policy_id": self.policy_id,
                        "policy_weight": float(profile["policy_weight"]),
                        "deadline_remaining": float(profile["deadline_remaining"]),
                        "queue_size": len(profiles),
                        "allocated_cpu": allocated,
                        "node_cpu_capacity": capacity,
                        "allocated_fraction": allocated / capacity if capacity else 0.0,
                        "mechanism": profile["mechanism"],
                        "search_rank": profile["search_rank"],
                    }
                )
        return PairedCpuAllocationDecision(allocations=allocations, rows=rows)
