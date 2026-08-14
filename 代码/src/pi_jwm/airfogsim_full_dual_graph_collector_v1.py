"""Validated one-step AirFogSim executor for the full dual-graph collector."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .airfogsim_contract_adapter import (
    apply_transmission_totals,
    direct_transmission_totals,
)
from .airfogsim_full_dual_graph_frame_builder_v1 import BuiltFrameDecision
from .airfogsim_full_dual_graph_observer_v1 import (
    AirFogSimSnapshot,
    capture_execution_snapshot,
    observe_airfogsim_snapshot,
)
from .airfogsim_single_step_collector_v1 import SingleStepRecorder
from .full_dual_graph_collector_contract_v1 import (
    JointFrameAction,
    SnapshotPhase,
    TaskLifecycle,
    validate_joint_frame_action,
)


FULL_COLLECTOR_VERSION = "PIJWM-AirFogSim-Full-Collector-v1"


@dataclass(frozen=True)
class FullCollectorStepResult:
    trajectory_id: str
    frame_index: int
    decision_snapshot: AirFogSimSnapshot
    execution_snapshot: AirFogSimSnapshot | None
    outcome_snapshot: AirFogSimSnapshot | None
    action: JointFrameAction | None
    lifecycle_rows: tuple[dict[str, object], ...]
    transfer_rows: tuple[dict[str, object], ...]
    cpu_rows: tuple[dict[str, object], ...]
    energy_rows: tuple[dict[str, object], ...]
    temporal_trace: tuple[str, ...]
    quarantined: bool
    quarantine_reason: str | None
    stepped: bool
    training_eligible: bool


def _plain(value: object) -> float:
    if hasattr(value, "get"):
        value = value.get()
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


def _energy_snapshot(env) -> dict[str, dict[str, object]]:
    manager = getattr(env, "energy_manager", None)
    if manager is None:
        return {}
    current = copy.deepcopy(getattr(manager, "_UAVs_energy_info", {}))
    removed = copy.deepcopy(getattr(manager, "_removed_UAVs_energy_info", {}))
    rows: dict[str, dict[str, object]] = {}
    for status, collection in (("present", current), ("removed", removed)):
        for node_id, values in collection.items():
            rows[str(node_id)] = {
                "status": status,
                **{str(key): _plain(value) for key, value in values.items()},
            }
    return rows


def _energy_rows(
    before: dict[str, dict[str, object]], after: dict[str, dict[str, object]]
) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for node_id in sorted(set(before) | set(after)):
        before_energy = before.get(node_id, {}).get("energy")
        after_energy = after.get(node_id, {}).get("energy")
        observed = before_energy is not None and after_energy is not None
        rows.append(
            {
                "node_id": node_id,
                "energy_before": before_energy,
                "energy_after": after_energy,
                "energy_consumed": (
                    float(before_energy) - float(after_energy) if observed else None
                ),
                "sending_data_size": after.get(node_id, {}).get("sending_data_size"),
                "receiving_data_size": after.get(node_id, {}).get("receiving_data_size"),
                "observed_mask": observed,
                "missing_reason": None if observed else "energy_endpoint_unavailable",
                "source": "energy_manager_direct_state",
            }
        )
    return tuple(rows)


def _task_by_id(env, task_id: str):
    task = env.task_manager.getTaskByTaskId(task_id)
    if task is None:
        raise RuntimeError(f"AirFogSim task disappeared before setter: {task_id}")
    return task


def _flow_task_ids(action: JointFrameAction) -> dict[str, str]:
    return {flow.flow_id: flow.task_id for flow in action.flows}


def _install_lifecycle_alias_guard(manager):
    method_name = "getOffloadingTasksWithNumber"
    instance_attributes = vars(manager)
    had_instance_override = method_name in instance_attributes
    previous_instance_value = instance_attributes.get(method_name)

    def copied_lookup():
        rows = {
            node_id: list(tasks)
            for node_id, tasks in manager._offloading_tasks.items()
        }
        for node_id, tasks in manager._returning_tasks.items():
            rows.setdefault(node_id, []).extend(list(tasks))
        return rows, sum(len(tasks) for tasks in rows.values())

    setattr(manager, method_name, copied_lookup)

    def restore():
        if had_instance_override:
            setattr(manager, method_name, previous_instance_value)
        else:
            delattr(manager, method_name)

    return restore


def _allocate_communication_rbs_without_lifecycle_alias(env, rb_assignments):
    """Call AirFogSim allocation through copied lifecycle lists."""

    restore = _install_lifecycle_alias_guard(env.task_manager)
    try:
        return env._allocate_communication_RBs(rb_assignments)
    finally:
        restore()


def _capture_wireless_profile_rows(
    env,
    activated_profiles: dict[str, dict[str, object]],
    action: JointFrameAction,
) -> list[dict[str, object]]:
    flow_task = _flow_task_ids(action)
    flow_by_task = {task_id: flow_id for flow_id, task_id in flow_task.items()}
    hop_by_flow = {hop.flow_id: hop for hop in action.hops}
    rows: list[dict[str, object]] = []
    for task_id, profile in sorted(activated_profiles.items(), key=lambda item: str(item[0])):
        task_id = str(task_id)
        flow_id = flow_by_task.get(task_id)
        if flow_id is None or flow_id not in hop_by_flow:
            continue
        hop = hop_by_flow[flow_id]
        channel_type = str(profile["channel_type"])
        tx_idx = int(profile["tx_idx"])
        rx_idx = int(profile["rx_idx"])
        rb_indices = tuple(int(value) for value in profile["RB_Nos"])
        rates = env.channel_manager.getRateByChannelType(
            tx_idx, rx_idx, channel_type, list(rb_indices)
        )
        tx_type, rx_type = channel_type.split("2")
        csi = env.channel_manager.getCSI(tx_idx, rx_idx, tx_type, rx_type)
        interference = getattr(env.channel_manager, f"{channel_type}_Interference")
        sinr = getattr(env.channel_manager, f"{channel_type}_SINR")
        outage = getattr(env.channel_manager, f"is_{channel_type}_outage")
        task = profile["task"]
        required_size = (
            float(task.getReturnedSize())
            if task.isReturning()
            else float(task.getTaskSize())
        )
        remaining = max(required_size - float(task.getTransmittedSize()), 0.0)
        for offset, rb_index in enumerate(rb_indices):
            rate = _plain(rates[offset])
            planned = rate * float(env.simulation_interval)
            delivered = min(planned, remaining)
            remaining -= delivered
            rows.append(
                {
                    "task_id": task_id,
                    "flow_id": flow_id,
                    "hop_id": hop.hop_id,
                    "source_id": hop.source_id,
                    "target_id": hop.target_id,
                    "physical_edge_id": hop.physical_edge_id,
                    "transport": "wireless",
                    "rb_index": rb_index,
                    "channel_type": channel_type,
                    "channel_attenuation_db": _plain(csi[rb_index]),
                    "interference_plus_noise_mw": _plain(
                        interference[tx_idx, rx_idx, rb_index]
                    ),
                    "sinr_db": _plain(sinr[tx_idx, rx_idx, rb_index]),
                    "outage": bool(_plain(outage[tx_idx, rx_idx, rb_index])),
                    "rate_per_s": rate,
                    "planned_capacity": planned,
                    "delivered_data": delivered,
                    "observed_mask": True,
                    "missing_reason": None,
                    "capture_phase": "after_fast_fading_before_transfer",
                    "temporal_role": "outcome_only_not_same_frame_decision_input",
                    "source_method": "AirFogSim_direct_per_rb_runtime_arrays",
                }
            )
    return rows


def _missing_wireless_rows(
    action: JointFrameAction,
    execution_snapshot: AirFogSimSnapshot,
    existing: list[dict[str, object]],
) -> list[dict[str, object]]:
    present_nodes = {node.node_id for node in execution_snapshot.nodes if node.present}
    existing_keys = {
        (str(row["flow_id"]), str(row["hop_id"]), int(row["rb_index"]))
        for row in existing
        if row.get("rb_index") is not None
    }
    flow_task = _flow_task_ids(action)
    hop_by_id = {hop.hop_id: hop for hop in action.hops}
    rows: list[dict[str, object]] = []
    for allocation in action.rb_allocations:
        key = (allocation.flow_id, allocation.hop_id, allocation.rb_index)
        if key in existing_keys:
            continue
        hop = hop_by_id[allocation.hop_id]
        reason = (
            "endpoint_absent_at_execution"
            if hop.source_id not in present_nodes or hop.target_id not in present_nodes
            else "runtime_channel_row_unavailable"
        )
        rows.append(
            {
                "task_id": flow_task[allocation.flow_id],
                "flow_id": allocation.flow_id,
                "hop_id": allocation.hop_id,
                "source_id": hop.source_id,
                "target_id": hop.target_id,
                "physical_edge_id": hop.physical_edge_id,
                "transport": "wireless",
                "rb_index": allocation.rb_index,
                "channel_type": None,
                "channel_attenuation_db": None,
                "interference_plus_noise_mw": None,
                "sinr_db": None,
                "outage": None,
                "rate_per_s": None,
                "planned_capacity": None,
                "delivered_data": None,
                "observed_mask": False,
                "missing_reason": reason,
                "capture_phase": SnapshotPhase.EXECUTION.value,
                "temporal_role": "runtime_outcome_missing",
                "source_method": None,
            }
        )
    return rows


def _wireless_totals(rows: list[dict[str, object]]):
    events = [
        {
            "source": row["source_id"],
            "target": row["target_id"],
            "planned_capacity": row["planned_capacity"],
        }
        for row in rows
        if row.get("observed_mask") and row.get("planned_capacity") is not None
    ]
    return direct_transmission_totals(events)


def _quarantined_result(
    *,
    trajectory_id: str,
    built: BuiltFrameDecision,
    decision_snapshot: AirFogSimSnapshot,
    trace: list[str],
    reason_detail: str,
) -> FullCollectorStepResult:
    trace.append("quarantined_after_partial_setter_failure")
    trace.append(reason_detail)
    return FullCollectorStepResult(
        trajectory_id=trajectory_id,
        frame_index=built.action.frame_index,
        decision_snapshot=decision_snapshot,
        execution_snapshot=None,
        outcome_snapshot=None,
        action=None,
        lifecycle_rows=built.lifecycle_rows,
        transfer_rows=(),
        cpu_rows=(),
        energy_rows=(),
        temporal_trace=tuple(trace),
        quarantined=True,
        quarantine_reason="quarantined_after_partial_setter_failure",
        stepped=False,
        training_eligible=False,
    )


def execute_full_collector_step(
    env,
    built: BuiltFrameDecision,
    *,
    trajectory_id: str,
    task_scheduler,
    communication_scheduler,
    computation_scheduler,
    observer: Callable[..., AirFogSimSnapshot] = observe_airfogsim_snapshot,
) -> FullCollectorStepResult:
    """Apply one fully validated joint action and execute exactly one real step."""

    trace: list[str] = []
    decision_snapshot = observer(env, phase=SnapshotPhase.DECISION)
    trace.append("decision_snapshot_captured")
    validate_joint_frame_action(
        built.action,
        phase=decision_snapshot.phase,
        nodes=decision_snapshot.nodes,
        physical_edges=decision_snapshot.physical_edges,
        tasks=decision_snapshot.tasks,
        dag_edges=decision_snapshot.dag_edges,
        n_rb=int(env.channel_manager.n_RB),
        input_source_phases={
            "nodes": SnapshotPhase.DECISION,
            "tasks": SnapshotPhase.DECISION,
            "channel": SnapshotPhase.DECISION,
        },
    )
    trace.append("action_validated")

    recorder = SingleStepRecorder(
        env, f"{trajectory_id}::frame::{built.action.frame_index}"
    )
    lifecycle_by_task = {
        str(row["task_id"]): row for row in built.lifecycle_rows
    }
    selected = tuple(row for row in built.action.decisions if row.selected)
    flow_task = _flow_task_ids(built.action)
    task_by_flow = dict(flow_task)
    hop_by_id = {hop.hop_id: hop for hop in built.action.hops}
    rb_by_task: dict[str, list[int]] = {}
    for allocation in built.action.rb_allocations:
        hop = hop_by_id[allocation.hop_id]
        task_id = task_by_flow[allocation.flow_id]
        decision = next(row for row in selected if row.task_id == task_id)
        if decision.hop_id != hop.hop_id or decision.route_nodes[0] != hop.target_id:
            raise RuntimeError("validated RB allocation no longer matches carrying hop")
        rb_by_task.setdefault(task_id, []).append(allocation.rb_index)

    try:
        recorder.install_cpu_callback(computation_scheduler)
        trace.append("cpu_callback_installed")
        offload_called = False
        for decision in selected:
            lifecycle = decision.lifecycle
            requires_setter = bool(
                lifecycle_by_task.get(decision.task_id, {}).get(
                    "requires_route_setter", False
                )
            )
            if lifecycle == TaskLifecycle.WAITING_TO_OFFLOAD and requires_setter:
                task_snapshot = next(
                    row
                    for row in decision_snapshot.tasks
                    if row.task_id == decision.task_id
                )
                ok = task_scheduler.setTaskOffloading(
                    env,
                    task_snapshot.task_node_id,
                    decision.task_id,
                    decision.target_node_id,
                    route=list(decision.route_nodes),
                )
                if not ok:
                    raise RuntimeError(
                        f"AirFogSim rejected offload setter: {decision.task_id}"
                    )
                task = _task_by_id(env, decision.task_id)
                route = tuple(str(node_id) for node_id in task.getToOffloadRoute())
                if route != decision.route_nodes:
                    raise RuntimeError(
                        f"offload setter route mismatch: {decision.task_id}"
                    )
                offload_called = True
        if offload_called:
            trace.append("offload_setters_called")

        return_called = False
        for decision in selected:
            requires_setter = bool(
                lifecycle_by_task.get(decision.task_id, {}).get(
                    "requires_route_setter", False
                )
            )
            if decision.lifecycle == TaskLifecycle.WAITING_TO_RETURN and requires_setter:
                task_scheduler.setTaskReturnRoute(
                    env, decision.task_id, list(decision.route_nodes)
                )
                queued_route = tuple(
                    str(node_id)
                    for node_id in env.task_return_routes.get(decision.task_id, ())
                )
                if queued_route != decision.route_nodes:
                    raise RuntimeError(
                        f"return setter route mismatch: {decision.task_id}"
                    )
                return_called = True
        if return_called:
            trace.append("return_route_setters_called")

        for task_id, rb_indices in sorted(rb_by_task.items()):
            task = _task_by_id(env, task_id)
            decision = next(row for row in selected if row.task_id == task_id)
            if decision.lifecycle not in {TaskLifecycle.WAITING_TO_RETURN}:
                route = tuple(str(node_id) for node_id in task.getToOffloadRoute())
                if not route or route[0] != decision.route_nodes[0]:
                    raise RuntimeError(
                        f"runtime route first hop mismatch before RB setter: {task_id}"
                    )
            communication_scheduler.setCommunicationWithRB(
                env, task_id, sorted(rb_indices)
            )
        if rb_by_task:
            trace.append("rb_setters_called")
    except Exception as error:
        return _quarantined_result(
            trajectory_id=trajectory_id,
            built=built,
            decision_snapshot=decision_snapshot,
            trace=trace,
            reason_detail=f"setter_error::{type(error).__name__}::{error}",
        )

    energy_before = _energy_snapshot(env)
    wireless_rows: list[dict[str, object]] = []
    wired_results: dict[str, float] = {}
    original_wireless = getattr(env, "_updateWirelessCommunication", None)
    original_wired_step = getattr(getattr(env, "wired_manager", None), "step", None)

    if callable(original_wireless) and all(
        hasattr(env, name)
        for name in (
            "_allocate_communication_RBs",
            "_compute_communication_rate",
            "_execute_communication",
        )
    ):
        def observed_wireless():
            activated = _allocate_communication_rbs_without_lifecycle_alias(
                env,
                env.activated_offloading_tasks_with_RB_Nos
            )
            env._compute_communication_rate(activated)
            rows = _capture_wireless_profile_rows(env, activated, built.action)
            wireless_rows.extend(rows)
            env._execute_communication(activated)
            sending, receiving = _wireless_totals(rows)
            apply_transmission_totals(env.channel_manager, sending, receiving)

        env._updateWirelessCommunication = observed_wireless
        trace.append("airfogsim_lifecycle_alias_guard_installed")

    if callable(original_wired_step):
        def observed_wired_step(interval):
            result = original_wired_step(interval)
            wired_results.update(
                {str(task_id): float(value) for task_id, value in result.items()}
            )
            return result

        env.wired_manager.step = observed_wired_step

    execution_snapshot: AirFogSimSnapshot | None = None
    outcome_snapshot: AirFogSimSnapshot | None = None
    restore_lifecycle_lookup = _install_lifecycle_alias_guard(env.task_manager)
    try:
        def execution_observer():
            snapshot = observer(env, phase=SnapshotPhase.EXECUTION)
            trace.append("execution_snapshot_captured")
            return snapshot

        trace.append("env_step_started")
        with capture_execution_snapshot(env, execution_observer) as captures:
            env.step()
        if len(captures) != 1:
            raise RuntimeError("execution snapshot was not captured exactly once")
        execution_snapshot = captures[0]
        trace.append("env_step_finished")
        recorder.finalize_after_step()
        outcome_snapshot = observer(env, phase=SnapshotPhase.OUTCOME)
        trace.append("outcome_snapshot_captured")
    finally:
        restore_lifecycle_lookup()
        if callable(original_wireless):
            env._updateWirelessCommunication = original_wireless
        if callable(original_wired_step):
            env.wired_manager.step = original_wired_step

    assert execution_snapshot is not None
    wireless_rows.extend(
        _missing_wireless_rows(built.action, execution_snapshot, wireless_rows)
    )
    wired_hop_by_task = {
        flow_task[hop.flow_id]: hop
        for hop in built.action.hops
        if hop.transport == "wired"
    }
    wired_rows = [
        {
            "task_id": task_id,
            "flow_id": hop.flow_id,
            "hop_id": hop.hop_id,
            "source_id": hop.source_id,
            "target_id": hop.target_id,
            "physical_edge_id": hop.physical_edge_id,
            "transport": "wired",
            "rb_index": None,
            "rate_per_s": delivered / float(env.simulation_interval),
            "planned_capacity": delivered,
            "delivered_data": delivered,
            "observed_mask": True,
            "missing_reason": None,
            "capture_phase": SnapshotPhase.OUTCOME.value,
            "temporal_role": "outcome_only_not_same_frame_decision_input",
            "source_method": "wired_manager.step_direct_result",
        }
        for task_id, delivered in sorted(wired_results.items())
        if (hop := wired_hop_by_task.get(task_id)) is not None
    ]
    transfer_rows = tuple(
        sorted(
            [*wireless_rows, *wired_rows],
            key=lambda row: (
                str(row["flow_id"]),
                str(row["hop_id"]),
                -1 if row["rb_index"] is None else int(row["rb_index"]),
            ),
        )
    )
    return FullCollectorStepResult(
        trajectory_id=trajectory_id,
        frame_index=built.action.frame_index,
        decision_snapshot=decision_snapshot,
        execution_snapshot=execution_snapshot,
        outcome_snapshot=outcome_snapshot,
        action=built.action,
        lifecycle_rows=built.lifecycle_rows,
        transfer_rows=transfer_rows,
        cpu_rows=tuple(recorder.cpu_rows),
        energy_rows=_energy_rows(energy_before, _energy_snapshot(env)),
        temporal_trace=tuple(trace),
        quarantined=False,
        quarantine_reason=None,
        stepped=True,
        training_eligible=False,
    )
