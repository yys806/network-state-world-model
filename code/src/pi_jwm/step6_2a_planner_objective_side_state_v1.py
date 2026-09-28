"""Causal objective metadata carried beside, never inside, the trained model.

This module validates inputs and support boundaries. It does not score or rank
candidate actions and does not create Flow slots or alter model tensors.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Mapping, Sequence


FORBIDDEN = frozenset({"future_target", "future_schedule", "future_action", "future_outcome", "failed_label", "target_tensor"})


def _number(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class PlannerTaskCausalSideState:
    task_id: str
    task_index: int
    arrival_time_s: float
    elapsed_time_s: float
    deadline_s: float
    absolute_deadline_s: float
    required_returned_size: float
    return_destination_index: int
    requires_return: bool | None
    terminal_at_anchor: bool
    source_time_s: float


@dataclass(frozen=True)
class PlannerRouteCausalSideState:
    flow_id: str
    flow_index: int
    task_id: str
    flow_type: str
    epoch: int
    logical_destination_index: int
    route_node_indices: tuple[int, ...]
    current_hop_index: int
    route_revision: int
    source_time_s: float

    @property
    def remaining_hops_after_current(self) -> int:
        return len(self.route_node_indices) - 2 - self.current_hop_index


@dataclass(frozen=True)
class PlannerObjectiveCausalSideState:
    anchor_time_s: float
    current_time_s: float
    tasks: tuple[PlannerTaskCausalSideState, ...]
    routes: tuple[PlannerRouteCausalSideState, ...]
    effort_component_mask: tuple[bool, bool, bool]  # Comm, Comp, Mob
    effort_denominators: tuple[float | None, float | None, float | None]
    source: str


def prepare_objective_side_state(
    *, decision: Mapping[str, Any], deadline_sidecar: Mapping[str, Any],
    task_slots: Mapping[str, int], physical_slots: Mapping[str, int],
    routes: Sequence[PlannerRouteCausalSideState],
    model_state: Mapping[str, Any],
) -> PlannerObjectiveCausalSideState:
    """Join one current Raw decision with a verified current-time deadline sidecar."""
    if FORBIDDEN.intersection(decision) or FORBIDDEN.intersection(deadline_sidecar):
        raise ValueError("future or target input forbidden")
    if deadline_sidecar.get("alignment_passed") is not True:
        raise ValueError("deadline sidecar lacks exact replay alignment")
    time = _number(decision["simulation_time_s"], "decision time")
    for field in ("trajectory_id", "frame_index", "capture_event_id"):
        if decision[field] != deadline_sidecar[field]:
            raise ValueError(f"sidecar {field} mismatch")
    if not math.isclose(time, _number(deadline_sidecar["simulation_time_s"], "sidecar time"), abs_tol=1e-9):
        raise ValueError("sidecar time mismatch")
    if _number(deadline_sidecar["source_time_s"], "source time") > time + 1e-9:
        raise ValueError("future sidecar source")
    task_rows = {str(row["task_id"]): row for row in decision["tasks"]}
    if len(task_rows) != len(decision["tasks"]):
        raise ValueError("duplicate current task ID")
    deadline_rows = {str(row["task_id"]): row for row in deadline_sidecar["tasks"]}
    if len(deadline_rows) != len(deadline_sidecar["tasks"]):
        raise ValueError("duplicate sidecar task ID")
    tasks = []
    for task_id, slot in sorted(task_slots.items(), key=lambda item: item[1]):
        if task_id not in task_rows or task_id not in deadline_rows:
            raise ValueError("current task missing from Raw or sidecar")
        raw, side = task_rows[task_id], deadline_rows[task_id]
        if not bool(model_state["task_presence"][0, slot]):
            raise ValueError("model task slot not present")
        arrival = _number(raw["arrival_time_s"], "arrival")
        if not math.isclose(arrival, _number(side["arrival_time_s"], "sidecar arrival"), abs_tol=1e-9):
            raise ValueError("arrival mismatch")
        deadline = _number(side["deadline_s"], "deadline")
        returned = _number(raw["required_returned_size"], "required return size")
        if deadline < 0 or returned < 0 or arrival > time + 1e-9:
            raise ValueError("invalid current task timing or return size")
        destination_id = raw.get("return_destination_id")
        if destination_id is None or str(destination_id) not in physical_slots:
            raise ValueError("current Return destination must resolve in causal physical support")
        destination_index = int(physical_slots[str(destination_id)])
        lifecycle = str(raw["lifecycle"]).lower()
        host_id = raw.get("current_node_id") if lifecycle in {"computing", "waiting_to_return", "returning"} else None
        requires_return = None if host_id is None else returned > 0 and str(host_id) != str(destination_id)
        tasks.append(PlannerTaskCausalSideState(task_id, int(slot), arrival, time-arrival,
            deadline, arrival+deadline, returned, destination_index, requires_return,
            str(raw["lifecycle"]).lower() in {"done", "failed", "completed"}, time))
    for route in routes:
        check_route_model_alignment(route, model_state)
        if route.source_time_s > time + 1e-9:
            raise ValueError("future route source")
    n_rb = sum(bool(v) for v in model_state["rb_active_mask"][0].tolist())
    cpu = sum(_number(row["capacity_per_s"], "CPU capacity") for row in decision["node_cpu_capacity_observation_rows"]
              if row.get("observed_mask") and row.get("capacity_per_s") is not None)
    uavs = sum(str(row.get("entity_type", "")).lower() == "uav" for row in decision["entities"])
    mask = (n_rb > 0, cpu > 0, uavs > 0)
    denominators = (float(n_rb) if mask[0] else None, cpu if mask[1] else None,
                    15.0 * uavs if mask[2] else None)
    return PlannerObjectiveCausalSideState(time, time, tuple(tasks), tuple(routes), mask,
        denominators, "current Formal Raw + aligned decision-before-action replay")


def check_route_model_alignment(route: PlannerRouteCausalSideState, state: Mapping[str, Any]) -> None:
    f = route.flow_index
    nodes = route.route_node_indices
    i = route.current_hop_index
    if not (len(nodes) >= 2 and 0 <= i < len(nodes)-1 and nodes[-1] == route.logical_destination_index):
        raise ValueError("invalid causal route suffix")
    checks = (
        int(state["flow_identity_index"][0, f]) == f,
        int(state["flow_destination_index"][0, f]) == route.logical_destination_index,
        int(state["flow_route_revision"][0, f]) == route.route_revision,
        int(state["current_hop_index"][0, f]) == i,
        int(state["current_holder_index"][0, f]) == nodes[i],
        int(state["carrying_hop_source_index"][0, f]) == nodes[i],
        int(state["carrying_hop_destination_index"][0, f]) == nodes[i+1],
    )
    if not all(checks):
        raise ValueError("Planner route and predicted model state diverged")


def transmission_burden(route: PlannerRouteCausalSideState, state: Mapping[str, Any]) -> float:
    check_route_model_alignment(route, state)
    f = route.flow_index
    hop = _number(state["hop_remaining"][0, f], "hop remaining")
    e2e = _number(state["flow_remaining"][0, f], "E2E remaining")
    return hop + route.remaining_hops_after_current * e2e


def advance_objective_side_state(
    side: PlannerObjectiveCausalSideState, *, predicted_state: Mapping[str, Any],
    slot_duration_s: float, route_actions: Sequence[Mapping[str, Any]] = (),
) -> PlannerObjectiveCausalSideState:
    """Advance from planned controls and predicted current state only.

    A route edit must be expressible by the frozen model endpoint rule. Changes
    to logical destination/Epoch and unsupported Flow births are rejected.
    """
    dt = _number(slot_duration_s, "slot duration")
    if dt <= 0:
        raise ValueError("slot duration must be positive")
    now = side.current_time_s + dt
    routes = {row.flow_index: row for row in side.routes}
    for action in route_actions:
        if FORBIDDEN.intersection(action):
            raise ValueError("future route truth forbidden")
        f = int(action["flow_index"])
        if f not in routes:
            raise ValueError("Route cannot create a Flow slot")
        old = routes[f]
        nodes = tuple(int(x) for x in action["route_node_indices"])
        if not nodes or nodes[0] != int(predicted_state["current_holder_index"][0, f]) or nodes[-1] != old.logical_destination_index:
            raise ValueError("route must start at predicted holder and retain logical destination")
        routes[f] = replace(old, route_node_indices=nodes, current_hop_index=0,
                            route_revision=int(predicted_state["flow_route_revision"][0, f]), source_time_s=side.current_time_s)
    aligned = []
    for route in routes.values():
        f = route.flow_index
        model_i = int(predicted_state["current_hop_index"][0, f])
        if f not in {int(x["flow_index"]) for x in route_actions}:
            route = replace(route, current_hop_index=model_i)
        check_route_model_alignment(route, predicted_state)
        aligned.append(route)
    return replace(side, current_time_s=now, routes=tuple(sorted(aligned, key=lambda x: x.flow_index)))


def deadline_violation(task: PlannerTaskCausalSideState, current_time_s: float, *,
                       final_completion_time_s: float | None = None,
                       no_return_completion: bool = False) -> bool:
    tolerance = 1e-5 if no_return_completion else 0.0
    observed_time = current_time_s if final_completion_time_s is None else _number(final_completion_time_s, "completion time")
    if observed_time > current_time_s + 1e-9:
        raise ValueError("future completion truth forbidden")
    return observed_time > task.absolute_deadline_s + tolerance


def support_horizon(first_unsupported_state: int | None, maximum_horizon: int) -> int:
    if maximum_horizon < 1:
        raise ValueError("positive horizon required")
    return maximum_horizon if first_unsupported_state is None else max(0, first_unsupported_state-1)


def common_support_horizon(horizons: Sequence[int]) -> int:
    if not horizons:
        raise ValueError("candidate horizons required")
    result = min(horizons)
    if result == 0:
        raise ValueError("OBJECTIVE_UNSCOREABLE")
    return result


def planner_derived_return_birth_required(task: PlannerTaskCausalSideState,
                                           predicted_computation_finished: bool,
                                           existing_return_slot: bool,
                                           predicted_compute_host_index: int | None) -> bool:
    if not predicted_computation_finished or existing_return_slot:
        return False
    if predicted_compute_host_index is None:
        raise ValueError("predicted computation host required to resolve Return")
    return (task.required_returned_size > 0 and
            int(predicted_compute_host_index) != task.return_destination_index)
