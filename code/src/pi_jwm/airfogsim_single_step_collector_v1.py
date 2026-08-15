"""AirFogSim boundary adapter for one truthful PI-JWM candidate step."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from .airfogsim_cpu_inner_rule_v1 import allocate_airfogsim_cpu
from .single_step_collector_contract_v1 import (
    CandidateAction,
    validate_candidate_action,
)


@dataclass
class SingleStepRecorder:
    """Collect only the callback ledger at the simulator boundary."""

    env: Any
    candidate_id: str
    cpu_rows: list[dict[str, Any]] = field(default_factory=list)
    _task_refs: dict[str, Any] = field(default_factory=dict, repr=False)

    def install_cpu_callback(self, computation_scheduler: Any) -> None:
        def callback(computing_tasks: Mapping[str, list[Any]]) -> dict[str, float]:
            task_ids = [
                str(task.getTaskId())
                for node_tasks in computing_tasks.values()
                for task in node_tasks
            ]
            before = {
                str(task.getTaskId()): float(task.getComputedSize())
                for node_tasks in computing_tasks.values()
                for task in node_tasks
            }
            self._task_refs.update(
                {
                    str(task.getTaskId()): task
                    for node_tasks in computing_tasks.values()
                    for task in node_tasks
                }
            )
            decision = allocate_airfogsim_cpu(self.env, computing_tasks)
            self.cpu_rows.append(
                {
                    "candidate_id": self.candidate_id,
                    "rule_version": decision.decision.rule_version,
                    "slot_seconds": decision.decision.slot_seconds,
                    "task_ids": sorted(task_ids),
                    "computed_before": before,
                    "allocations": dict(decision.allocations),
                    "served_work": {
                        row.task_id: row.served_work for row in decision.decision.allocations
                    },
                    "node_summaries": [
                        {
                            "node_id": summary.node_id,
                            "capacity": summary.capacity,
                            "total_demand_rate": summary.total_demand_rate,
                            "total_allocated_cpu": summary.total_allocated_cpu,
                            "water_level": summary.water_level,
                            "task_count": summary.task_count,
                        }
                        for summary in decision.decision.node_summaries
                    ],
                }
            )
            return decision.allocations

        callback.__name__ = "pi_jwm_single_step_cpu_callback_v1"
        computation_scheduler.setComputingCallBack(self.env, callback)

    def finalize_after_step(self) -> None:
        for row in self.cpu_rows:
            row["computed_after"] = {
                task_id: float(self._task_refs[task_id].getComputedSize())
                for task_id in row["task_ids"]
            }


@dataclass(frozen=True)
class SingleStepExecutionResult:
    candidate_id: str
    cpu_rows: tuple[dict[str, Any], ...]
    simulator_order: tuple[str, ...]
    pre_action_observation: Any
    temporal_trace: tuple[str, ...]
    stepped: bool


def _default_scheduler_classes() -> tuple[Any, Any, Any]:
    from airfogsim.scheduler import CommunicationScheduler, ComputationScheduler, TaskScheduler

    return TaskScheduler, CommunicationScheduler, ComputationScheduler


def execute_candidate(
    env: Any,
    action: CandidateAction,
    *,
    task_ids: tuple[str, ...],
    node_ids: tuple[str, ...] | None = None,
    edge_count: int,
    flow_count: int,
    n_rb: int,
    task_scheduler: Any | None = None,
    communication_scheduler: Any | None = None,
    computation_scheduler: Any | None = None,
    pre_action_observer: Callable[[], Any] | None = None,
) -> SingleStepExecutionResult:
    """Apply one validated candidate and execute exactly one simulator step."""

    validate_candidate_action(
        action,
        task_ids=task_ids,
        edge_count=edge_count,
        flow_count=flow_count,
        n_rb=n_rb,
        node_ids=node_ids,
    )
    temporal_trace = ["action_validated"]
    if pre_action_observer is None:
        pre_action_observation = None
        temporal_trace.append("decision_time_observation_skipped")
    else:
        pre_action_observation = pre_action_observer()
        temporal_trace.append("decision_time_observation_captured")
    if task_scheduler is None or communication_scheduler is None or computation_scheduler is None:
        task_scheduler, communication_scheduler, computation_scheduler = _default_scheduler_classes()

    recorder = SingleStepRecorder(env, action.candidate_id)
    recorder.install_cpu_callback(computation_scheduler)
    temporal_trace.append("cpu_callback_installed")

    # Apply offload decisions before resource assignments, matching the public
    # scheduler boundary used by AirFogSim algorithms.
    for offload in action.offloads:
        ok = task_scheduler.setTaskOffloading(
            env,
            offload.task_node_id,
            offload.task_id,
            offload.target_node_id,
            route=list(offload.route_nodes),
        )
        if not ok:
            raise RuntimeError(f"AirFogSim rejected offload action: {offload.task_id}")

    rb_by_task: dict[str, list[int]] = {}
    for assignment in action.rb_assignments:
        if assignment.flow_index >= len(task_ids):
            raise ValueError(f"flow index has no task vocabulary entry: {assignment.flow_index}")
        task_id = task_ids[assignment.flow_index]
        rb_by_task.setdefault(task_id, []).append(assignment.rb_index)
    for task_id, rb_nos in sorted(rb_by_task.items()):
        communication_scheduler.setCommunicationWithRB(env, task_id, rb_nos)

    temporal_trace.append("action_setters_called")
    temporal_trace.append("env_step_started")
    env.step()
    temporal_trace.append("env_step_finished")
    recorder.finalize_after_step()
    return SingleStepExecutionResult(
        candidate_id=action.candidate_id,
        cpu_rows=tuple(recorder.cpu_rows),
        simulator_order=(
            "update_task",
            "wireless_communication",
            "wired_communication",
            "computation",
            "storage",
            "energy",
            "time_advance",
        ),
        pre_action_observation=pre_action_observation,
        temporal_trace=tuple(temporal_trace),
        stepped=True,
    )
