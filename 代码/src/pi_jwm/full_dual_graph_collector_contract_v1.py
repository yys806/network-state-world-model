"""Pure contracts for one PI-JWM v4 full dual-graph collector frame.

The module intentionally has no AirFogSim import and performs no runtime action.
It is the setter-pre boundary for a fully specified joint frame action.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Mapping, Sequence


COLLECTOR_CONTRACT_VERSION = "PIJWM-P2-Full-Dual-Graph-Collector-v1"


class SnapshotPhase(str, Enum):
    DECISION = "decision"
    EXECUTION = "execution"
    OUTCOME = "outcome"


class TaskLifecycle(str, Enum):
    WAITING_TO_OFFLOAD = "waiting_to_offload"
    OFFLOADING = "offloading"
    COMPUTING = "computing"
    WAITING_TO_RETURN = "waiting_to_return"
    RETURNING = "returning"
    DONE = "done"
    FAILED = "failed"
    TO_GENERATE = "to_generate"


class CollectorContractError(ValueError):
    """A deterministic action rejection with a stable machine-readable code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


@dataclass(frozen=True)
class PhysicalNode:
    node_id: str
    node_type: str
    present: bool


@dataclass(frozen=True)
class PhysicalEdge:
    edge_id: str
    source_id: str
    target_id: str
    edge_type: str
    present: bool


@dataclass(frozen=True)
class DagEdge:
    dag_edge_id: str
    source_task_id: str
    target_task_id: str
    communication_mapping: str = "not_modeled"


@dataclass(frozen=True)
class TaskSnapshot:
    task_id: str
    task_node_id: str
    lifecycle: TaskLifecycle
    current_node_id: str
    route_nodes: tuple[str, ...]
    return_destination_id: str | None
    arrival_time: float


@dataclass(frozen=True)
class LogicalFlow:
    flow_id: str
    trajectory_id: str
    task_id: str
    phase: str
    route_revision: int


@dataclass(frozen=True)
class CarryingHop:
    hop_id: str
    flow_id: str
    hop_index: int
    source_id: str
    target_id: str
    physical_edge_id: str
    transport: str


@dataclass(frozen=True)
class RbAllocation:
    flow_id: str
    hop_id: str
    rb_index: int


@dataclass(frozen=True)
class DecisionRow:
    task_id: str
    lifecycle: TaskLifecycle
    selected: bool
    reason: str
    target_node_id: str | None
    route_nodes: tuple[str, ...]
    flow_id: str | None
    hop_id: str | None
    requested_target_family: str | None
    executed_target_family: str | None
    target_family_fallback: bool


@dataclass(frozen=True)
class JointFrameAction:
    frame_index: int
    decisions: tuple[DecisionRow, ...]
    flows: tuple[LogicalFlow, ...]
    hops: tuple[CarryingHop, ...]
    rb_allocations: tuple[RbAllocation, ...]


_ACTIONABLE = frozenset(
    {
        TaskLifecycle.WAITING_TO_OFFLOAD,
        TaskLifecycle.OFFLOADING,
        TaskLifecycle.WAITING_TO_RETURN,
        TaskLifecycle.RETURNING,
    }
)
_NEW_ROUTE_LIFECYCLES = frozenset(
    {TaskLifecycle.WAITING_TO_OFFLOAD, TaskLifecycle.WAITING_TO_RETURN}
)
_WIRELESS_TYPES = frozenset(
    {
        "V2V",
        "V2U",
        "V2I",
        "U2V",
        "U2U",
        "U2I",
        "I2V",
        "I2U",
        "I2I",
    }
)


def _require_identifier(value: object, *, field: str, code: str = "unknown_identity") -> str:
    if not isinstance(value, str) or not value.strip():
        raise CollectorContractError(code, f"{field} must be a non-empty string")
    return value


def _indexed(rows: Sequence[object], *, attr: str, kind: str) -> dict[str, object]:
    indexed: dict[str, object] = {}
    for row in rows:
        identifier = _require_identifier(getattr(row, attr, None), field=f"{kind} id")
        if identifier in indexed:
            raise CollectorContractError("unknown_identity", f"duplicate {kind} id: {identifier}")
        indexed[identifier] = row
    return indexed


def _require_nonnegative_integer(value: object, *, field: str, code: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CollectorContractError(code, f"{field} must be a nonnegative integer")
    return value


def _require_task_snapshot(task: TaskSnapshot) -> None:
    _require_identifier(task.task_id, field="task id")
    _require_identifier(task.task_node_id, field="task node id")
    _require_identifier(task.current_node_id, field="current node id")
    if not isinstance(task.lifecycle, TaskLifecycle):
        raise CollectorContractError("unknown_identity", "task lifecycle is invalid")
    if task.return_destination_id is not None:
        _require_identifier(task.return_destination_id, field="return destination id")
    if not isinstance(task.route_nodes, tuple):
        raise CollectorContractError("route_first_hop_mismatch", "task route must be a tuple")
    if any(not isinstance(node_id, str) or not node_id for node_id in task.route_nodes):
        raise CollectorContractError("route_first_hop_mismatch", "task route contains invalid node id")
    if (
        isinstance(task.arrival_time, bool)
        or not isinstance(task.arrival_time, (int, float))
        or not isfinite(float(task.arrival_time))
    ):
        raise CollectorContractError("unknown_identity", "task arrival time must be finite")


def _expected_phase(lifecycle: TaskLifecycle) -> str | None:
    if lifecycle in {TaskLifecycle.WAITING_TO_OFFLOAD, TaskLifecycle.OFFLOADING}:
        return "offload"
    if lifecycle in {TaskLifecycle.WAITING_TO_RETURN, TaskLifecycle.RETURNING}:
        return "return"
    return None


def _validate_source_timing(input_source_phases: Mapping[str, SnapshotPhase] | None) -> None:
    if input_source_phases is None:
        return
    for field, source_phase in input_source_phases.items():
        _require_identifier(field, field="input field")
        if source_phase == SnapshotPhase.OUTCOME:
            raise CollectorContractError(
                "same_slot_outcome_leak",
                f"decision input {field} has outcome source phase",
            )
        if not isinstance(source_phase, SnapshotPhase):
            raise CollectorContractError("unknown_identity", f"invalid source phase for {field}")


def validate_joint_frame_action(
    action: JointFrameAction,
    *,
    phase: SnapshotPhase,
    nodes: Sequence[PhysicalNode],
    physical_edges: Sequence[PhysicalEdge],
    tasks: Sequence[TaskSnapshot],
    dag_edges: Sequence[DagEdge],
    n_rb: int,
    input_source_phases: Mapping[str, SnapshotPhase] | None = None,
) -> JointFrameAction:
    """Validate a full action before any AirFogSim setter is called.

    The validator performs no mutation. A caller must treat any raised
    :class:`CollectorContractError` as a no-setter/no-step outcome.
    """

    if not isinstance(action, JointFrameAction):
        raise TypeError("action must be JointFrameAction")
    if phase != SnapshotPhase.DECISION:
        raise CollectorContractError("unknown_identity", "joint actions require decision phase")
    _require_nonnegative_integer(action.frame_index, field="frame index", code="unknown_identity")
    _require_nonnegative_integer(n_rb, field="n_rb", code="rb_out_of_range")
    _validate_source_timing(input_source_phases)

    node_by_id = _indexed(nodes, attr="node_id", kind="node")
    edge_by_id = _indexed(physical_edges, attr="edge_id", kind="physical edge")
    task_by_id = _indexed(tasks, attr="task_id", kind="task")
    flow_by_id = _indexed(action.flows, attr="flow_id", kind="flow")
    hop_by_id = _indexed(action.hops, attr="hop_id", kind="hop")
    dag_by_id = _indexed(dag_edges, attr="dag_edge_id", kind="DAG edge")

    for node in nodes:
        _require_identifier(node.node_type, field="node type")
        if not isinstance(node.present, bool):
            raise CollectorContractError("unknown_identity", "node presence must be bool")
    for edge in physical_edges:
        _require_identifier(edge.source_id, field="physical edge source")
        _require_identifier(edge.target_id, field="physical edge target")
        _require_identifier(edge.edge_type, field="physical edge type")
        if edge.source_id not in node_by_id or edge.target_id not in node_by_id:
            raise CollectorContractError("unknown_identity", "physical edge references unknown node")
        if not isinstance(edge.present, bool):
            raise CollectorContractError("unknown_identity", "physical edge presence must be bool")
    for task in tasks:
        _require_task_snapshot(task)
    for dag_edge in dag_edges:
        _require_identifier(dag_edge.source_task_id, field="DAG source task")
        _require_identifier(dag_edge.target_task_id, field="DAG target task")
        if dag_edge.communication_mapping != "not_modeled":
            raise CollectorContractError(
                "dag_edge_used_as_communication_hop",
                "precedence DAG edge cannot carry communication mapping",
            )
    for hop_id in hop_by_id:
        if hop_id in dag_by_id:
            raise CollectorContractError(
                "dag_edge_used_as_communication_hop",
                f"hop id collides with DAG edge id: {hop_id}",
            )

    decision_by_task: dict[str, DecisionRow] = {}
    for decision in action.decisions:
        _require_identifier(decision.task_id, field="decision task id")
        if decision.task_id in decision_by_task:
            raise CollectorContractError(
                "duplicate_task_decision", f"duplicate decision for {decision.task_id}"
            )
        decision_by_task[decision.task_id] = decision
    actionable_ids = {
        task.task_id for task in tasks if task.lifecycle in _ACTIONABLE
    }
    missing_decisions = sorted(actionable_ids - set(decision_by_task))
    if missing_decisions:
        raise CollectorContractError(
            "missing_task_decision", f"missing decisions for {missing_decisions}"
        )
    unknown_decisions = sorted(set(decision_by_task) - set(task_by_id))
    if unknown_decisions:
        raise CollectorContractError("unknown_identity", f"unknown decision tasks: {unknown_decisions}")

    selected_flow_ids: set[str] = set()
    selected_hop_ids: set[str] = set()
    for task_id, decision in decision_by_task.items():
        task = task_by_id[task_id]
        if not isinstance(decision.lifecycle, TaskLifecycle) or decision.lifecycle != task.lifecycle:
            raise CollectorContractError(
                "offload_wrong_lifecycle", f"lifecycle mismatch for {task_id}"
            )
        _require_identifier(decision.reason, field="decision reason")
        if not isinstance(decision.selected, bool):
            raise CollectorContractError("unknown_identity", "decision selection must be bool")
        if not isinstance(decision.route_nodes, tuple):
            raise CollectorContractError("route_first_hop_mismatch", "decision route must be tuple")

        if not decision.selected:
            if any(
                value is not None
                for value in (decision.target_node_id, decision.flow_id, decision.hop_id)
            ) or decision.route_nodes:
                raise CollectorContractError(
                    "unknown_identity", "unselected decision contains executable action"
                )
            continue

        if task.lifecycle not in _ACTIONABLE:
            raise CollectorContractError(
                "offload_wrong_lifecycle", f"selected task is not actionable: {task_id}"
            )
        _require_identifier(decision.target_node_id, field="decision target")
        if decision.target_node_id not in node_by_id:
            raise CollectorContractError("unknown_identity", "decision target is unknown")
        target_node = node_by_id[decision.target_node_id]
        if not target_node.present:
            raise CollectorContractError(
                "node_absent_at_decision", f"decision target absent: {decision.target_node_id}"
            )
        if task.current_node_id not in node_by_id or not node_by_id[task.current_node_id].present:
            raise CollectorContractError(
                "node_absent_at_decision", f"task source absent: {task.current_node_id}"
            )

        is_local = decision.target_node_id == task.current_node_id
        if is_local:
            if decision.flow_id is not None or decision.hop_id is not None:
                raise CollectorContractError("unknown_identity", "local action has communication identity")
            if decision.route_nodes != (decision.target_node_id,):
                raise CollectorContractError("route_first_hop_mismatch", "local route is invalid")
            continue

        if decision.flow_id is None or decision.hop_id is None:
            raise CollectorContractError("unknown_identity", "remote action lacks flow or hop")
        if decision.flow_id not in flow_by_id or decision.hop_id not in hop_by_id:
            raise CollectorContractError("unknown_identity", "decision references unknown flow or hop")
        flow = flow_by_id[decision.flow_id]
        hop = hop_by_id[decision.hop_id]
        if flow.task_id != task_id or hop.flow_id != flow.flow_id:
            raise CollectorContractError("unknown_identity", "flow and hop are not bound to decision task")
        expected_phase = _expected_phase(task.lifecycle)
        if flow.phase != expected_phase:
            raise CollectorContractError("offload_wrong_lifecycle", "flow phase mismatches task lifecycle")
        if not decision.route_nodes or decision.route_nodes[0] != hop.target_id:
            raise CollectorContractError("route_first_hop_mismatch", "route first hop mismatches carrying hop")
        if task.lifecycle == TaskLifecycle.WAITING_TO_RETURN:
            if task.return_destination_id is None or decision.route_nodes[-1] != task.return_destination_id:
                raise CollectorContractError(
                    "return_destination_mismatch", "return route ends at wrong destination"
                )
        if hop.source_id != task.current_node_id or hop.target_id != decision.target_node_id:
            raise CollectorContractError("route_first_hop_mismatch", "hop endpoints mismatch task route")
        if hop.physical_edge_id not in edge_by_id:
            raise CollectorContractError("unknown_identity", "hop references unknown physical edge")
        physical_edge = edge_by_id[hop.physical_edge_id]
        if (physical_edge.source_id, physical_edge.target_id) != (hop.source_id, hop.target_id):
            raise CollectorContractError("cep_endpoint_mismatch", "CEP endpoints mismatch")
        if not physical_edge.present:
            raise CollectorContractError("cep_endpoint_mismatch", "CEP physical edge is absent")
        if hop.transport == "wireless" and physical_edge.edge_type not in _WIRELESS_TYPES:
            raise CollectorContractError("cep_endpoint_mismatch", "wireless hop uses nonwireless edge")
        if hop.transport == "wired" and physical_edge.edge_type != "wired":
            raise CollectorContractError("cep_endpoint_mismatch", "wired hop uses nonwired edge")
        if hop.transport not in {"wireless", "wired"}:
            raise CollectorContractError("unknown_identity", "unsupported carrying-hop transport")
        selected_flow_ids.add(flow.flow_id)
        selected_hop_ids.add(hop.hop_id)

    for flow in action.flows:
        _require_identifier(flow.trajectory_id, field="flow trajectory id")
        _require_identifier(flow.task_id, field="flow task id")
        if flow.task_id not in task_by_id:
            raise CollectorContractError("unknown_identity", "flow references unknown task")
        if flow.phase not in {"offload", "return"}:
            raise CollectorContractError("unknown_identity", "flow phase is invalid")
        _require_nonnegative_integer(flow.route_revision, field="route revision", code="unknown_identity")
        if flow.flow_id not in selected_flow_ids:
            raise CollectorContractError("unknown_identity", "unreferenced flow is not allowed")
    for hop in action.hops:
        _require_nonnegative_integer(hop.hop_index, field="hop index", code="unknown_identity")
        if hop.hop_id not in selected_hop_ids:
            raise CollectorContractError("unknown_identity", "unreferenced hop is not allowed")

    allocation_keys: set[tuple[str, str, int]] = set()
    allocations_by_hop: dict[str, list[RbAllocation]] = {}
    transmitter_rb_pairs: set[tuple[str, int]] = set()
    for allocation in action.rb_allocations:
        if allocation.flow_id not in flow_by_id or allocation.hop_id not in hop_by_id:
            raise CollectorContractError("unknown_identity", "RB allocation references unknown flow or hop")
        hop = hop_by_id[allocation.hop_id]
        if allocation.flow_id != hop.flow_id:
            raise CollectorContractError("unknown_identity", "RB allocation flow and hop mismatch")
        if isinstance(allocation.rb_index, bool) or not isinstance(allocation.rb_index, int):
            raise CollectorContractError("rb_out_of_range", "RB index must be an integer")
        if allocation.rb_index < 0 or allocation.rb_index >= n_rb:
            raise CollectorContractError("rb_out_of_range", "RB index outside configured range")
        allocation_key = (allocation.flow_id, allocation.hop_id, allocation.rb_index)
        if allocation_key in allocation_keys:
            raise CollectorContractError("duplicate_rb_allocation", "duplicate wireless COO allocation")
        allocation_keys.add(allocation_key)
        if hop.transport != "wireless":
            raise CollectorContractError("nonwireless_flow_has_rb", "nonwireless hop has RB allocation")
        transmitter_key = (hop.source_id, allocation.rb_index)
        if transmitter_key in transmitter_rb_pairs:
            raise CollectorContractError(
                "same_transmitter_rb_conflict", "transmitter uses RB for multiple flows"
            )
        transmitter_rb_pairs.add(transmitter_key)
        allocations_by_hop.setdefault(hop.hop_id, []).append(allocation)

    for hop_id in selected_hop_ids:
        hop = hop_by_id[hop_id]
        has_rb = bool(allocations_by_hop.get(hop_id))
        if hop.transport == "wireless" and not has_rb:
            raise CollectorContractError("wireless_flow_without_rb", "wireless hop lacks RB allocation")
        if hop.transport == "wired" and has_rb:
            raise CollectorContractError("nonwireless_flow_has_rb", "wired hop has RB allocation")

    return action
