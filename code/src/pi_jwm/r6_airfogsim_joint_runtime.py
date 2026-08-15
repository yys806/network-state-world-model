"""Thin, audited AirFogSim adapter for PI-JWM R6 joint action candidates."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Sequence

import numpy as np

from .r6_joint_action import (
    ComputeTaskContext,
    JointActionCandidate,
    JointActionCandidateSet,
    JointActionContext,
    OffloadTaskContext,
    TargetContext,
    generate_joint_candidates,
    validate_joint_candidate,
)


@dataclass(frozen=True)
class _PreScheduleCapture:
    scenario_id: str
    seed: int
    slot: int
    split: str
    offload_tasks: tuple[OffloadTaskContext, ...]


@dataclass(frozen=True)
class PreparedJointActionStep:
    context: JointActionContext
    candidates: JointActionCandidateSet
    simulation_time: float
    default_schedule_called: bool

    def with_candidates(
        self,
        candidates: Sequence[JointActionCandidate],
    ) -> "PreparedJointActionStep":
        return replace(self, candidates=JointActionCandidateSet.create(candidates))


@dataclass(frozen=True)
class JointActionExecutionRecord:
    candidate_id: str
    template_id: str
    context_fingerprint: str
    offload_applied_count: int
    rb_task_count: int
    cpu_task_count: int
    offload_changed: bool
    rb_changed: bool
    cpu_changed: bool
    hard_violation_count: int


def _task_value(info: dict[str, Any], name: str, default: float = 0.0) -> float:
    value = info.get(name, default)
    return float(default if value is None else value)


def _rate_proxy(algorithm: Any, env: Any, source: str, target: str) -> float:
    matrix, _ = algorithm.commScheduler.getEstimatedRateBetweenNodeIds(
        env,
        [source],
        [target],
    )
    values = np.asarray(matrix, dtype=np.float64)
    if values.shape != (1, 1) or not np.isfinite(values).all():
        raise ValueError("AirFogSim estimated-rate API returned an invalid pair matrix")
    return max(float(values[0, 0]), 0.0)


def _capture_offload_tasks(env: Any, algorithm: Any) -> tuple[OffloadTaskContext, ...]:
    task_infos = algorithm.taskScheduler.getAllToOffloadTaskInfos(
        env,
        check_dependency=True,
    )
    rows: list[OffloadTaskContext] = []
    current_time = float(env.simulation_time)
    for info in sorted(task_infos, key=lambda row: (str(row["task_node_id"]), str(row["task_id"]))):
        task_id = str(info["task_id"])
        source = str(info["task_node_id"])
        neighbors = algorithm.entityScheduler.getNeighborNodeInfosById(
            env,
            source,
            sorted_by="distance",
            max_num=10,
        )
        targets: list[TargetContext] = []
        for neighbor in neighbors:
            target = str(neighbor.get("id", ""))
            if not target:
                continue
            distance = float(
                algorithm.entityScheduler.getDistanceBetweenNodes(env, source, target)
            )
            fog = dict(neighbor.get("fog_profile") or {})
            targets.append(
                TargetContext(
                    node_id=target,
                    distance=max(distance, 0.0),
                    available_cpu=max(float(fog.get("cpu", 0.0)), 0.0),
                    rate_proxy=_rate_proxy(algorithm, env, source, target),
                    energy_proxy=max(distance, 0.0),
                )
            )
        if not targets:
            continue
        arrival = _task_value(info, "task_arrival_time")
        deadline = _task_value(info, "task_deadline")
        rows.append(
            OffloadTaskContext(
                task_id=task_id,
                source_node_id=source,
                legal_targets=tuple(targets),
                deadline_remaining=max(arrival + deadline - current_time, 0.0),
                priority=max(_task_value(info, "task_priority", 1.0), 0.0),
                input_mb=max(_task_value(info, "task_size"), 0.0),
            )
        )
    return tuple(rows)


def _capture_compute_tasks(env: Any, algorithm: Any) -> tuple[ComputeTaskContext, ...]:
    rows: list[ComputeTaskContext] = []
    current_time = float(env.simulation_time)
    for node_id, tasks in sorted(env.task_manager.getComputingTasks().items()):
        node_info = algorithm.entityScheduler.getNodeInfoById(env, node_id)
        capacity = max(float(dict(node_info.get("fog_profile") or {}).get("cpu", 0.0)), 0.0)
        for task in sorted(tasks, key=lambda value: str(value.getTaskId())):
            if task.getAssignedTo() != node_id or task.getCurrentNodeId() != node_id:
                continue
            rows.append(
                ComputeTaskContext(
                    task_id=str(task.getTaskId()),
                    node_id=str(node_id),
                    node_capacity=capacity,
                    deadline_remaining=max(
                        float(task.getTaskArrivalTime())
                        + float(task.getTaskDeadline())
                        - current_time,
                        0.0,
                    ),
                    priority=max(float(task.getTaskPriority()), 0.0),
                )
            )
    return tuple(rows)


def _capture_before_schedule(
    env: Any,
    algorithm: Any,
    *,
    scenario_id: str,
    seed: int,
    slot: int,
    split: str,
) -> _PreScheduleCapture:
    return _PreScheduleCapture(
        scenario_id=str(scenario_id),
        seed=int(seed),
        slot=int(slot),
        split=str(split),
        offload_tasks=_capture_offload_tasks(env, algorithm),
    )


def prepare_joint_action_step(
    env: Any,
    algorithm: Any,
    *,
    scenario_id: str,
    seed: int,
    slot: int,
    split: str,
    max_candidates: int = 6,
) -> PreparedJointActionStep:
    """Capture DAG-released tasks, run default scheduling once, then form candidates."""

    capture = _capture_before_schedule(
        env,
        algorithm,
        scenario_id=scenario_id,
        seed=seed,
        slot=slot,
        split=split,
    )
    simulation_time = float(env.simulation_time)
    algorithm.scheduleStep(env)
    default_rb = {
        str(task_id): tuple(int(rb) for rb in rb_ids)
        for task_id, rb_ids in env.activated_offloading_tasks_with_RB_Nos.items()
    }
    context = JointActionContext.create(
        scenario_id=capture.scenario_id,
        seed=capture.seed,
        slot=capture.slot,
        split=capture.split,
        offload_tasks=capture.offload_tasks,
        compute_tasks=_capture_compute_tasks(env, algorithm),
        default_rb_plan=default_rb,
        rb_capacity=int(algorithm.commScheduler.getNumberOfRB(env)),
    )
    return PreparedJointActionStep(
        context=context,
        candidates=generate_joint_candidates(context, max_candidates=max_candidates),
        simulation_time=simulation_time,
        default_schedule_called=True,
    )


def _apply_offload(env: Any, candidate: JointActionCandidate) -> tuple[int, bool]:
    applied = 0
    changed = False
    for row in candidate.offload:
        task = env.task_manager.getTaskByTaskId(row.task_id)
        if task is None or task.isComputing() or task.isComputed():
            raise ValueError(f"offload task changed stage before execution: {row.task_id}")
        assigned = task.getAssignedTo()
        task.changeOffloadTo(
            row.target_node_id,
            [row.target_node_id],
            float(env.simulation_time),
        )
        applied += 1
        changed = changed or assigned != row.target_node_id
    return applied, changed


def _apply_rb(env: Any, algorithm: Any, candidate: JointActionCandidate) -> bool:
    before = {
        str(task_id): tuple(int(rb) for rb in rb_ids)
        for task_id, rb_ids in env.activated_offloading_tasks_with_RB_Nos.items()
    }
    env.activated_offloading_tasks_with_RB_Nos = {}
    for row in candidate.rb:
        algorithm.commScheduler.setCommunicationWithRB(env, row.task_id, list(row.rb_ids))
    after = {
        str(task_id): tuple(int(rb) for rb in rb_ids)
        for task_id, rb_ids in env.activated_offloading_tasks_with_RB_Nos.items()
    }
    return before != after


def _apply_cpu(env: Any, algorithm: Any, candidate: JointActionCandidate) -> bool:
    if not candidate.cpu:
        return False
    requested = {row.task_id: float(row.amount) for row in candidate.cpu}
    original = env.alloc_cpu_callback

    def callback(computing_tasks: dict[str, list[Any]], **kwargs: Any) -> dict[str, float]:
        allocation = {} if original is None else dict(original(computing_tasks, **kwargs))
        current = {
            str(task.getTaskId())
            for node_id, tasks in computing_tasks.items()
            for task in tasks
            if task.getAssignedTo() == node_id and task.getCurrentNodeId() == node_id
        }
        allocation.update(
            {task_id: amount for task_id, amount in requested.items() if task_id in current}
        )
        return allocation

    algorithm.compScheduler.setComputingCallBack(env, callback)
    # Do not call the baseline allocator here.  Some formal CPU policies are
    # stochastic, so an eager comparison would advance their RNG before the
    # environment executes the action.
    return True


def _effect_time(env: Any) -> float:
    return round(
        float(env.simulation_time) + float(getattr(env, "simulation_interval", 0.1)),
        6,
    )


def _replace_selected_action_rows(
    env: Any,
    algorithm: Any,
    candidate: JointActionCandidate,
) -> None:
    """Replace default scheduler logs with the action that will be executed."""

    effect_time = _effect_time(env)
    seed = int(getattr(algorithm, "seed", 0))
    if candidate.template_id != "default" and hasattr(algorithm, "offload_rows"):
        touched = {row.task_id for row in candidate.offload}
        algorithm.offload_rows[:] = [
            row
            for row in algorithm.offload_rows
            if not (
                math.isclose(float(row.get("time", -1.0)), effect_time, abs_tol=1e-7)
                and str(row.get("task_id")) in touched
            )
        ]
        for row in candidate.offload:
            algorithm.offload_rows.append(
                {
                    "seed": seed,
                    "time": effect_time,
                    "task_id": row.task_id,
                    "task_node_id": row.source_node_id,
                    "source_node_id": row.source_node_id,
                    "target_node_id": row.target_node_id,
                    "route_nodes": [row.target_node_id],
                    "template_id": candidate.template_id,
                    "action_source": "selected_joint_candidate",
                }
            )
    if hasattr(algorithm, "rb_rows"):
        touched = {row.task_id for row in candidate.rb}
        algorithm.rb_rows[:] = [
            row
            for row in algorithm.rb_rows
            if not (
                math.isclose(float(row.get("time", -1.0)), effect_time, abs_tol=1e-7)
                and str(row.get("task_id")) in touched
            )
        ]
        offload_targets = {row.task_id: row.target_node_id for row in candidate.offload}
        for row in candidate.rb:
            task = env.task_manager.getTaskByTaskId(row.task_id)
            current = "" if task is None else str(task.getCurrentNodeId())
            assigned = offload_targets.get(
                row.task_id,
                "" if task is None or task.getAssignedTo() is None else str(task.getAssignedTo()),
            )
            algorithm.rb_rows.append(
                {
                    "seed": seed,
                    "time": effect_time,
                    "task_id": row.task_id,
                    "current_node_id": current,
                    "assigned_to": assigned,
                    "rb_count": len(row.rb_ids),
                    "rb_indices": " ".join(str(value) for value in row.rb_ids),
                    "template_id": candidate.template_id,
                    "action_source": "selected_joint_candidate",
                }
            )


def _install_cpu_action_recorder(
    env: Any,
    algorithm: Any,
    candidate: JointActionCandidate,
) -> None:
    """Log the final allocation when AirFogSim invokes the CPU callback."""

    original = env.alloc_cpu_callback
    effect_time = _effect_time(env)
    seed = int(getattr(algorithm, "seed", 0))
    if not hasattr(algorithm, "cpu_rows"):
        algorithm.cpu_rows = []

    def callback(computing_tasks: dict[str, list[Any]], **kwargs: Any) -> dict[str, float]:
        allocation = {} if original is None else dict(original(computing_tasks, **kwargs))
        algorithm.cpu_rows[:] = [
            row
            for row in algorithm.cpu_rows
            if not math.isclose(float(row.get("time", -1.0)), effect_time, abs_tol=1e-7)
        ]
        for node_id, tasks in sorted(computing_tasks.items()):
            node = env._getNodeById(node_id)
            capacity = 0.0 if node is None else float(node.getFogProfile().get("cpu", 0.0))
            for task in sorted(tasks, key=lambda value: str(value.getTaskId())):
                task_id = str(task.getTaskId())
                if task_id not in allocation:
                    continue
                amount = float(allocation[task_id])
                algorithm.cpu_rows.append(
                    {
                        "seed": seed,
                        "time": effect_time,
                        "task_id": task_id,
                        "node_id": str(node_id),
                        "allocated_cpu": amount,
                        "node_cpu_capacity": capacity,
                        "allocated_fraction": amount / capacity if capacity > 0.0 else 0.0,
                        "template_id": candidate.template_id,
                        "action_source": "selected_joint_candidate",
                    }
                )
        return allocation

    algorithm.compScheduler.setComputingCallBack(env, callback)


def apply_prepared_candidate(
    env: Any,
    algorithm: Any,
    prepared: PreparedJointActionStep,
    *,
    candidate_index: int,
) -> JointActionExecutionRecord:
    if not prepared.default_schedule_called:
        raise ValueError("default AirFogSim schedule must run before joint action overlay")
    index = int(candidate_index)
    if not 0 <= index < len(prepared.candidates.candidates):
        raise ValueError("candidate_index is outside the current candidate set")
    candidate = prepared.candidates.candidates[index]
    validation = validate_joint_candidate(prepared.context, candidate)
    if not validation.valid:
        raise ValueError("; ".join(validation.reasons))
    offload_applied, offload_changed = _apply_offload(env, candidate)
    rb_changed = _apply_rb(env, algorithm, candidate)
    cpu_changed = _apply_cpu(env, algorithm, candidate)
    _replace_selected_action_rows(env, algorithm, candidate)
    _install_cpu_action_recorder(env, algorithm, candidate)
    return JointActionExecutionRecord(
        candidate_id=candidate.candidate_id,
        template_id=candidate.template_id,
        context_fingerprint=prepared.context.protocol_fingerprint(),
        offload_applied_count=offload_applied,
        rb_task_count=len(candidate.rb),
        cpu_task_count=len(candidate.cpu),
        offload_changed=offload_changed,
        rb_changed=rb_changed,
        cpu_changed=cpu_changed,
        hard_violation_count=0,
    )
