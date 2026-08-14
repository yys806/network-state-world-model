"""Pure natural-frame action builder for the AirFogSim v4 collector."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

from .airfogsim_full_dual_graph_observer_v1 import AirFogSimSnapshot
from .full_dual_graph_collector_contract_v1 import (
    CarryingHop,
    DecisionRow,
    JointFrameAction,
    LogicalFlow,
    RbAllocation,
    SnapshotPhase,
    TaskLifecycle,
    TaskSnapshot,
    validate_joint_frame_action,
)
from .full_dual_graph_coverage_v1 import (
    TargetCandidate,
    WirelessFlowRequest,
    allocate_rb_coverage,
    choose_resource_arm,
    choose_target_family,
    target_family_for_ordinal,
)
from .full_dual_graph_vocabulary_v1 import (
    FullTrajectoryVocabulary,
    RouteRevisionLedger,
)


FRAME_BUILDER_VERSION = "PIJWM-AirFogSim-Natural-Frame-Builder-v1"


@dataclass(frozen=True)
class BuiltFrameDecision:
    action: JointFrameAction
    lifecycle_rows: tuple[dict[str, object], ...]
    resource_policy: str


_ACTIONABLE = frozenset(
    {
        TaskLifecycle.WAITING_TO_OFFLOAD,
        TaskLifecycle.OFFLOADING,
        TaskLifecycle.WAITING_TO_RETURN,
        TaskLifecycle.RETURNING,
    }
)


def build_logical_flow_id(
    trajectory_id: str, task_id: str, phase: str, route_revision: int
) -> str:
    if not all(isinstance(value, str) and value.strip() for value in (trajectory_id, task_id)):
        raise ValueError("trajectory_id and task_id must be non-empty")
    if phase not in {"offload", "return"}:
        raise ValueError("phase must be offload or return")
    if isinstance(route_revision, bool) or not isinstance(route_revision, int) or route_revision < 0:
        raise ValueError("route_revision must be a nonnegative integer")
    return f"flow::{trajectory_id}::{task_id}::{phase}::{route_revision}"


def build_carrying_hop_id(
    flow_id: str,
    hop_index: int,
    source_id: str,
    target_id: str,
) -> str:
    if not all(
        isinstance(value, str) and value.strip()
        for value in (flow_id, source_id, target_id)
    ):
        raise ValueError("flow and endpoint identifiers must be non-empty")
    if isinstance(hop_index, bool) or not isinstance(hop_index, int) or hop_index < 0:
        raise ValueError("hop_index must be a nonnegative integer")
    return f"hop::{flow_id}::{hop_index}::{source_id}::{target_id}"


def _preview_revision(
    route_revisions: RouteRevisionLedger,
    trajectory_id: str,
    task_id: str,
    phase: str,
    route: tuple[str, ...],
) -> int:
    previous = route_revisions.snapshot().get((trajectory_id, task_id, phase))
    if previous is None:
        return 0
    revision, previous_route = previous
    return revision if previous_route == route else revision + 1


def _task_order(task: TaskSnapshot) -> tuple[object, ...]:
    phase = (
        "offload"
        if task.lifecycle in {TaskLifecycle.WAITING_TO_OFFLOAD, TaskLifecycle.OFFLOADING}
        else "return"
        if task.lifecycle in {TaskLifecycle.WAITING_TO_RETURN, TaskLifecycle.RETURNING}
        else "none"
    )
    return (task.arrival_time, task.task_node_id, task.task_id, phase)


def _edge_for(
    snapshot: AirFogSimSnapshot, source_id: str, target_id: str
):
    matches = tuple(
        edge
        for edge in snapshot.physical_edges
        if edge.present
        and edge.source_id == source_id
        and edge.target_id == target_id
    )
    if len(matches) > 1:
        raise ValueError(f"multiple physical edges bind endpoints {source_id}->{target_id}")
    return matches[0] if matches else None


def _initial_decision(
    task: TaskSnapshot,
    *,
    snapshot: AirFogSimSnapshot,
    trajectory_id: str,
    requested_family: str | None,
    route_revisions: RouteRevisionLedger,
    node_cpu: Mapping[str, float],
    node_distance: Mapping[tuple[str, str], float],
) -> tuple[DecisionRow, LogicalFlow | None, CarryingHop | None, bool]:
    phase = (
        "offload"
        if task.lifecycle in {TaskLifecycle.WAITING_TO_OFFLOAD, TaskLifecycle.OFFLOADING}
        else "return"
    )
    route: tuple[str, ...]
    target_id: str | None
    reason: str
    executed_family: str | None = None
    fallback = False
    requires_route_setter = False

    present_nodes = {node.node_id for node in snapshot.nodes if node.present}
    if task.lifecycle == TaskLifecycle.WAITING_TO_OFFLOAD:
        candidates: list[TargetCandidate] = []
        for node_id in sorted(present_nodes):
            if node_id not in node_cpu or float(node_cpu[node_id]) < 0:
                continue
            is_local = node_id == task.current_node_id
            if not is_local and _edge_for(snapshot, task.current_node_id, node_id) is None:
                continue
            distance = 0.0 if is_local else node_distance.get((task.current_node_id, node_id))
            if distance is None or float(distance) < 0:
                continue
            candidates.append(
                TargetCandidate(
                    node_id=node_id,
                    is_local=is_local,
                    distance=float(distance),
                    available_cpu=float(node_cpu[node_id]),
                )
            )
        choice = choose_target_family(
            trajectory_id=trajectory_id,
            task_id=task.task_id,
            route_revision=0,
            candidates=candidates,
            requested_family=requested_family,
        )
        if choice.target_node_id is None:
            return (
                DecisionRow(
                    task.task_id,
                    task.lifecycle,
                    False,
                    choice.reason,
                    None,
                    (),
                    None,
                    None,
                    choice.requested_family,
                    choice.executed_family,
                    choice.fallback,
                ),
                None,
                None,
                False,
            )
        target_id = choice.target_node_id
        route = (target_id,)
        reason = choice.reason
        executed_family = choice.executed_family
        fallback = choice.fallback
        requires_route_setter = True
    elif task.lifecycle == TaskLifecycle.WAITING_TO_RETURN:
        target_id = task.return_destination_id
        if target_id is None or target_id not in present_nodes:
            return (
                DecisionRow(
                    task.task_id,
                    task.lifecycle,
                    False,
                    "return_destination_absent",
                    None,
                    (),
                    None,
                    None,
                    None,
                    None,
                    False,
                ),
                None,
                None,
                False,
            )
        route = (target_id,)
        reason = "selected_return_route"
        requires_route_setter = True
    else:
        route = tuple(task.route_nodes)
        if not route:
            return (
                DecisionRow(
                    task.task_id,
                    task.lifecycle,
                    False,
                    "current_route_unavailable",
                    None,
                    (),
                    None,
                    None,
                    None,
                    None,
                    False,
                ),
                None,
                None,
                False,
            )
        target_id = route[0]
        reason = "continue_current_route"

    if target_id == task.current_node_id:
        return (
            DecisionRow(
                task.task_id,
                task.lifecycle,
                True,
                reason,
                target_id,
                route,
                None,
                None,
                requested_family,
                executed_family,
                fallback,
            ),
            None,
            None,
            requires_route_setter,
        )

    edge = _edge_for(snapshot, task.current_node_id, target_id)
    if edge is None:
        return (
            DecisionRow(
                task.task_id,
                task.lifecycle,
                False,
                "physical_edge_absent",
                None,
                (),
                None,
                None,
                requested_family,
                executed_family,
                fallback,
            ),
            None,
            None,
            False,
        )
    revision = _preview_revision(
        route_revisions, trajectory_id, task.task_id, phase, route
    )
    flow_id = build_logical_flow_id(trajectory_id, task.task_id, phase, revision)
    hop_id = build_carrying_hop_id(
        flow_id, 0, task.current_node_id, target_id
    )
    flow = LogicalFlow(flow_id, trajectory_id, task.task_id, phase, revision)
    hop = CarryingHop(
        hop_id=hop_id,
        flow_id=flow_id,
        hop_index=0,
        source_id=task.current_node_id,
        target_id=target_id,
        physical_edge_id=edge.edge_id,
        transport="wired" if edge.edge_type == "wired" else "wireless",
    )
    return (
        DecisionRow(
            task.task_id,
            task.lifecycle,
            True,
            reason,
            target_id,
            route,
            flow_id,
            hop_id,
            requested_family,
            executed_family,
            fallback,
        ),
        flow,
        hop,
        requires_route_setter,
    )


def build_frame_decision(
    snapshot: AirFogSimSnapshot,
    *,
    trajectory_id: str,
    frame_index: int,
    seed: int,
    n_rb: int,
    vocabulary: FullTrajectoryVocabulary,
    route_revisions: RouteRevisionLedger,
    node_cpu: Mapping[str, float],
    node_distance: Mapping[tuple[str, str], float],
) -> BuiltFrameDecision:
    """Build and validate one complete natural collector action without mutation."""

    if snapshot.phase != SnapshotPhase.DECISION:
        raise ValueError("frame builder requires a decision snapshot")
    if not isinstance(vocabulary, FullTrajectoryVocabulary):
        raise TypeError("vocabulary must be FullTrajectoryVocabulary")
    if not isinstance(route_revisions, RouteRevisionLedger):
        raise TypeError("route_revisions must be RouteRevisionLedger")

    resource_policy = choose_resource_arm(trajectory_id, seed)
    ordered_tasks = tuple(sorted(snapshot.tasks, key=_task_order))
    decisions: list[DecisionRow] = []
    flows_by_id: dict[str, LogicalFlow] = {}
    hops_by_id: dict[str, CarryingHop] = {}
    setter_by_task: dict[str, bool] = {}
    task_lifecycle_by_id = {task.task_id: task.lifecycle for task in ordered_tasks}
    predecessors_by_task: dict[str, set[str]] = {}
    for dag_edge in snapshot.dag_edges:
        predecessors_by_task.setdefault(dag_edge.target_task_id, set()).add(
            dag_edge.source_task_id
        )
    waiting_ordinal = 0
    for row in ordered_tasks:
        if row.lifecycle not in _ACTIONABLE:
            continue
        requested_family = None
        if row.lifecycle == TaskLifecycle.WAITING_TO_OFFLOAD:
            requested_family = target_family_for_ordinal(frame_index + waiting_ordinal)
            waiting_ordinal += 1
            predecessor_states = tuple(
                task_lifecycle_by_id.get(task_id)
                for task_id in sorted(predecessors_by_task.get(row.task_id, ()))
            )
            if any(state == TaskLifecycle.FAILED for state in predecessor_states):
                decisions.append(
                    DecisionRow(
                        row.task_id,
                        row.lifecycle,
                        False,
                        "dependency_failed",
                        None,
                        (),
                        None,
                        None,
                        requested_family,
                        None,
                        False,
                    )
                )
                setter_by_task[row.task_id] = False
                continue
            if any(state != TaskLifecycle.DONE for state in predecessor_states):
                decisions.append(
                    DecisionRow(
                        row.task_id,
                        row.lifecycle,
                        False,
                        "dependency_not_satisfied",
                        None,
                        (),
                        None,
                        None,
                        requested_family,
                        None,
                        False,
                    )
                )
                setter_by_task[row.task_id] = False
                continue
        decision, flow, hop, requires_route_setter = _initial_decision(
            row,
            snapshot=snapshot,
            trajectory_id=trajectory_id,
            requested_family=requested_family,
            route_revisions=route_revisions,
            node_cpu=node_cpu,
            node_distance=node_distance,
        )
        decisions.append(decision)
        setter_by_task[row.task_id] = requires_route_setter
        if flow is not None and hop is not None:
            flows_by_id[flow.flow_id] = flow
            hops_by_id[hop.hop_id] = hop

    wireless_requests = tuple(
        WirelessFlowRequest(
            flow_id=hop.flow_id,
            hop_id=hop.hop_id,
            transmitter_id=hop.source_id,
            receiver_id=hop.target_id,
        )
        for hop in hops_by_id.values()
        if hop.transport == "wireless"
    )
    resource_decisions = allocate_rb_coverage(
        wireless_requests, n_rb=n_rb, arm=resource_policy
    )
    resource_by_hop = {row.hop_id: row for row in resource_decisions}
    rejected_flow_ids: set[str] = set()
    final_decisions: list[DecisionRow] = []
    for decision in decisions:
        if decision.hop_id is None:
            final_decisions.append(decision)
            continue
        hop = hops_by_id[decision.hop_id]
        resource = resource_by_hop.get(decision.hop_id)
        if hop.transport == "wireless" and resource is not None and not resource.selected:
            rejected_flow_ids.add(hop.flow_id)
            setter_by_task[decision.task_id] = False
            final_decisions.append(
                DecisionRow(
                    task_id=decision.task_id,
                    lifecycle=decision.lifecycle,
                    selected=False,
                    reason=resource.reason,
                    target_node_id=None,
                    route_nodes=(),
                    flow_id=None,
                    hop_id=None,
                    requested_target_family=decision.requested_target_family,
                    executed_target_family=decision.executed_target_family,
                    target_family_fallback=decision.target_family_fallback,
                )
            )
        else:
            final_decisions.append(decision)

    flows = tuple(
        sorted(
            (flow for flow_id, flow in flows_by_id.items() if flow_id not in rejected_flow_ids),
            key=lambda row: row.flow_id,
        )
    )
    retained_flow_ids = {flow.flow_id for flow in flows}
    hops = tuple(
        sorted(
            (hop for hop in hops_by_id.values() if hop.flow_id in retained_flow_ids),
            key=lambda row: row.hop_id,
        )
    )
    rb_allocations = tuple(
        RbAllocation(resource.flow_id, resource.hop_id, rb_index)
        for resource in resource_decisions
        if resource.selected and resource.flow_id in retained_flow_ids
        for rb_index in resource.rb_indices
    )
    action = JointFrameAction(
        frame_index=frame_index,
        decisions=tuple(sorted(final_decisions, key=lambda row: row.task_id)),
        flows=flows,
        hops=hops,
        rb_allocations=tuple(
            sorted(
                rb_allocations,
                key=lambda row: (row.flow_id, row.hop_id, row.rb_index),
            )
        ),
    )
    validate_joint_frame_action(
        action,
        phase=snapshot.phase,
        nodes=snapshot.nodes,
        physical_edges=snapshot.physical_edges,
        tasks=snapshot.tasks,
        dag_edges=snapshot.dag_edges,
        n_rb=n_rb,
        input_source_phases={
            "nodes": SnapshotPhase.DECISION,
            "tasks": SnapshotPhase.DECISION,
            "channel": SnapshotPhase.DECISION,
        },
    )
    vocabulary.observe(
        nodes=snapshot.nodes,
        physical_edges=snapshot.physical_edges,
        tasks=snapshot.tasks,
        dag_edges=snapshot.dag_edges,
        flows=action.flows,
        hops=action.hops,
    )
    task_by_id = {task.task_id: task for task in snapshot.tasks}
    for flow in action.flows:
        decision = next(row for row in action.decisions if row.flow_id == flow.flow_id)
        assigned = route_revisions.assign(
            trajectory_id,
            flow.task_id,
            flow.phase,
            decision.route_nodes,
        )
        if assigned != flow.route_revision:
            raise RuntimeError("route revision changed after validated preview")

    decision_by_task = {row.task_id: row for row in action.decisions}
    lifecycle_rows = tuple(
        {
            "task_id": row.task_id,
            "task_node_id": row.task_node_id,
            "lifecycle": row.lifecycle,
            "actionable": row.lifecycle in _ACTIONABLE,
            "decision_selected": (
                decision_by_task[row.task_id].selected
                if row.task_id in decision_by_task
                else False
            ),
            "decision_reason": (
                decision_by_task[row.task_id].reason
                if row.task_id in decision_by_task
                else "not_actionable"
            ),
            "requires_route_setter": setter_by_task.get(row.task_id, False),
        }
        for row in ordered_tasks
    )
    del task_by_id
    return BuiltFrameDecision(action, lifecycle_rows, resource_policy)
