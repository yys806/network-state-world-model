"""Pure contracts for the PI-JWM P2 real single-step collector.

This module defines action validation and status wording only. It does not
execute AirFogSim, generate training data, or implement candidate rollouts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .information_edge_contract_v4 import validate_assignment_coo


COLLECTOR_CONTRACT_VERSION = "PIJWM-P2-Single-Step-Collector-v1"


@dataclass(frozen=True)
class OffloadAction:
    task_node_id: str
    task_id: str
    target_node_id: str
    route_nodes: tuple[str, ...]


@dataclass(frozen=True)
class RbAssignment:
    time_index: int
    flow_index: int
    information_edge_index: int
    rb_index: int

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (
            self.time_index,
            self.flow_index,
            self.information_edge_index,
            self.rb_index,
        )


@dataclass(frozen=True)
class CandidateAction:
    candidate_id: str
    offloads: tuple[OffloadAction, ...]
    rb_assignments: tuple[RbAssignment, ...]


def _identifier(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _capacity(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer")
    return value


def validate_candidate_action(
    action: CandidateAction,
    *,
    task_ids: Iterable[str],
    edge_count: int,
    flow_count: int,
    n_rb: int,
    node_ids: Iterable[str] | None = None,
) -> CandidateAction:
    """Validate a candidate before any AirFogSim action setter is called."""

    if not isinstance(action, CandidateAction):
        raise TypeError("action must be CandidateAction")
    _identifier(action.candidate_id, field="candidate_id")
    known_tasks = set(task_ids)
    known_nodes = None if node_ids is None else set(node_ids)
    seen_offload_tasks: set[str] = set()

    for offload in action.offloads:
        if not isinstance(offload, OffloadAction):
            raise TypeError("offloads must contain OffloadAction values")
        task_node_id = _identifier(offload.task_node_id, field="task_node_id")
        task_id = _identifier(offload.task_id, field="task_id")
        target_node_id = _identifier(offload.target_node_id, field="target_node_id")
        if task_id not in known_tasks:
            raise ValueError(f"unknown task_id: {task_id}")
        if task_id in seen_offload_tasks:
            raise ValueError(f"duplicate offload task_id: {task_id}")
        seen_offload_tasks.add(task_id)
        if not isinstance(offload.route_nodes, tuple) or not offload.route_nodes:
            raise ValueError(f"route_nodes must be a non-empty tuple for task_id: {task_id}")
        for route_node in offload.route_nodes:
            _identifier(route_node, field=f"route node for {task_id}")
        if offload.route_nodes[-1] != target_node_id:
            raise ValueError(f"route must end at target_node_id for task_id: {task_id}")
        if known_nodes is not None:
            referenced_nodes = {task_node_id, target_node_id, *offload.route_nodes}
            unknown = sorted(referenced_nodes - known_nodes)
            if unknown:
                raise ValueError(f"unknown node_id values for {task_id}: {unknown}")

    capacities = (
        1,
        _capacity(flow_count, field="flow_count"),
        _capacity(edge_count, field="edge_count"),
        _capacity(n_rb, field="n_rb"),
    )
    rows: list[tuple[int, int, int, int]] = []
    for assignment in action.rb_assignments:
        if not isinstance(assignment, RbAssignment):
            raise TypeError("rb_assignments must contain RbAssignment values")
        values = assignment.as_tuple()
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise ValueError("RB assignment indices must be integers")
        rows.append(values)
    coo = np.asarray(rows, dtype=np.int64).reshape((-1, 4))
    validate_assignment_coo(coo, capacities)
    return action


def build_single_step_status_flags() -> dict[str, bool]:
    """Return the conservative status boundary before a successful real run."""

    return {
        "single_step_real_airfogsim_executed": False,
        "v4_collector_implemented": False,
        "v4_dataset_complete": False,
        "model_training_started": False,
        "gpu_started": False,
        "locked_test_accessed": False,
        "candidate_rollout_planner_complete": False,
        "final_method_frozen": False,
        "training_eligible": False,
    }
