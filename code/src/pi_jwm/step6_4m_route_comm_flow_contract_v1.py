"""Isolated STEP 6.4M Route--Comm--Flow--Task contract reference.

This module is deliberately outside the formal planner and World Model.  It
is a small deterministic state machine used to test the causal contract and
its failure boundaries.  It does not call AirFogSim, allocate real RBs, or
promote a candidate to the formal Scorer.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import copy
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

from .step4_2c_b_causal_flow_ledger_raw_v1 import build_flow_id


CONTRACT_SCHEMA = "PI-JWM-STEP-6.4M-ROUTE-COMM-FLOW-CONTRACT-v1"
SUPPORTED_FLOW_TYPES = frozenset({"Input", "Return"})
SUPPORTED_LIFECYCLES = frozenset({
    "waiting_to_offload", "offloading", "computing", "waiting_to_return",
    "returning", "done", "failed",
})


class ContractError(ValueError):
    """A fail-closed reference-contract violation."""


def _finite(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ContractError(f"{name}:NONFINITE")
    return result


def _nonnegative(value: Any, name: str) -> float:
    result = _finite(value, name)
    if result < 0:
        raise ContractError(f"{name}:NEGATIVE")
    return result


def _time_value(value: Any, name: str) -> float:
    result = _finite(value, name)
    # AirFogSim uses -1 as the unset sentinel for last transmission/return.
    if result < -1.0:
        raise ContractError(f"{name}:INVALID_SENTINEL")
    return result


def _stable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _stable(value[k]) for k in sorted(value, key=str)}
    if isinstance(value, (tuple, list)):
        return [_stable(x) for x in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ContractError("FINGERPRINT_NONFINITE")
        return value
    return value


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(_stable(value), sort_keys=True,
                                     separators=(",", ":"), default=str).encode()).hexdigest()


@dataclass
class TaskContract:
    task_id: str
    lifecycle: str
    current_node_id: str
    return_destination_id: str | None
    required_returned_size: float
    task_cpu_work: float
    computed_cpu_work: float
    last_transmission_time: float
    last_return_time: float
    arrival_time: float
    deadline: float
    task_size: float = 0.0

    def __post_init__(self) -> None:
        if self.lifecycle not in SUPPORTED_LIFECYCLES:
            raise ContractError(f"TASK_LIFECYCLE_UNSUPPORTED:{self.lifecycle}")
        for name in ("required_returned_size", "task_cpu_work", "computed_cpu_work",
                     "arrival_time", "deadline", "task_size"):
            setattr(self, name, _nonnegative(getattr(self, name), name))
        self.last_transmission_time = _time_value(self.last_transmission_time,
                                                  "last_transmission_time")
        self.last_return_time = _time_value(self.last_return_time, "last_return_time")
        if self.computed_cpu_work > self.task_cpu_work and self.task_cpu_work >= 0:
            raise ContractError("COMPUTED_WORK_EXCEEDS_TOTAL")

    @property
    def completed_exact(self) -> bool:
        return self.computed_cpu_work >= self.task_cpu_work

    @property
    def requires_return(self) -> bool:
        return (self.required_returned_size > 0 and
                self.return_destination_id is not None and
                self.current_node_id != str(self.return_destination_id))


@dataclass
class FlowContract:
    flow_id: str
    flow_type: str
    epoch: int
    slot: int
    task_id: str
    source_id: str
    destination_id: str
    route: tuple[str, ...]
    route_revision: int
    holder_id: str
    current_hop_index: int
    total_data: float
    e2e_delivered: float
    e2e_remaining: float
    known: bool = True
    presence: bool = True
    active: bool = True
    status: str = "ACTIVE"
    hop_progress: float = 0.0
    hop_remaining: float = 0.0

    def __post_init__(self) -> None:
        if self.flow_type not in SUPPORTED_FLOW_TYPES:
            raise ContractError(f"FLOW_TYPE_UNSUPPORTED:{self.flow_type}")
        if self.epoch < 0 or self.slot < 0 or self.route_revision < 0:
            raise ContractError("FLOW_IDENTITY_NEGATIVE")
        if not self.route or self.route[-1] != self.destination_id:
            raise ContractError("FLOW_ROUTE_DESTINATION_MISMATCH")
        if len(self.route) != 1:
            raise ContractError("MULTIHOP_OUTSIDE_V1_SINGLE_HOP_DOMAIN")
        self.total_data = _nonnegative(self.total_data, "total_data")
        self.e2e_delivered = _nonnegative(self.e2e_delivered, "e2e_delivered")
        self.e2e_remaining = _nonnegative(self.e2e_remaining, "e2e_remaining")
        self.hop_progress = _nonnegative(self.hop_progress, "hop_progress")
        self.hop_remaining = _nonnegative(self.hop_remaining, "hop_remaining")
        if not math.isclose(self.e2e_delivered + self.e2e_remaining,
                            self.total_data, abs_tol=1e-9):
            raise ContractError("FLOW_CONSERVATION_FAILED")
        if self.status == "COMPLETED" and (self.presence or self.active):
            raise ContractError("COMPLETED_FLOW_STILL_ACTIVE")


@dataclass
class ContractState:
    now_s: float
    tasks: dict[str, TaskContract]
    links: frozenset[tuple[str, str]]
    rb_count: int
    slot_capacity: int
    tti_threshold: float = 0.5
    flows: dict[str, FlowContract] = field(default_factory=dict)
    rb_assignments: dict[str, tuple[int, ...]] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.now_s = _nonnegative(self.now_s, "now_s")
        if self.rb_count <= 0 or self.slot_capacity <= 0:
            raise ContractError("INVALID_CAPACITY")
        self.tti_threshold = _nonnegative(self.tti_threshold, "tti_threshold")
        if any(str(k) != v.task_id for k, v in self.tasks.items()):
            raise ContractError("TASK_ID_KEY_MISMATCH")
        if len(self.flows) > self.slot_capacity:
            raise ContractError("FLOW_SLOT_CAPACITY_EXHAUSTED")
        self._validate_flow_identity()

    def clone(self) -> "ContractState":
        return copy.deepcopy(self)

    def anchor_fingerprint(self) -> str:
        """Fingerprint pre-Route causal state, excluding event log/assignments."""
        payload = {
            "schema": CONTRACT_SCHEMA,
            "now_s": self.now_s,
            "tasks": {k: asdict(v) for k, v in sorted(self.tasks.items())},
            "links": sorted(self.links), "rb_count": self.rb_count,
            "slot_capacity": self.slot_capacity, "tti_threshold": self.tti_threshold,
            "flows": {k: asdict(v) for k, v in sorted(self.flows.items())},
        }
        return _digest(payload)

    def state_fingerprint(self) -> str:
        payload = {"anchor": self.anchor_fingerprint(),
                   "assignments": self.rb_assignments,
                   "events": self.events}
        return _digest(payload)

    def route_noop(self) -> "ContractState":
        result = self.clone()
        result.events.append({"event": "ROUTE_NOOP", "time_s": self.now_s})
        return result

    def _validate_flow_identity(self) -> None:
        ids = set()
        slots = set()
        for flow in self.flows.values():
            if flow.flow_id in ids or flow.slot in slots:
                raise ContractError("DUPLICATE_FLOW_IDENTITY")
            ids.add(flow.flow_id); slots.add(flow.slot)
            expected = build_flow_id(flow.task_id, flow.flow_type, flow.epoch)
            if flow.flow_id != expected:
                raise ContractError("FLOW_IDENTITY_MISMATCH")

    def _next_slot(self) -> int:
        used = {f.slot for f in self.flows.values()}
        for slot in range(self.slot_capacity):
            if slot not in used:
                return slot
        raise ContractError("FLOW_SLOT_CAPACITY_EXHAUSTED")

    def _existing_active(self, task_id: str, flow_type: str) -> FlowContract | None:
        rows = [f for f in self.flows.values()
                if f.task_id == task_id and f.flow_type == flow_type and f.status == "ACTIVE"]
        if len(rows) > 1:
            raise ContractError("DUPLICATE_ACTIVE_FLOW")
        return rows[0] if rows else None

    def _validate_route(self, source: str, destination: str, route: Sequence[str]) -> tuple[str, ...]:
        route_tuple = tuple(str(x) for x in route)
        if not route_tuple or route_tuple[-1] != destination:
            raise ContractError("INVALID_ROUTE_DESTINATION")
        if source in route_tuple or len(set(route_tuple)) != len(route_tuple):
            raise ContractError("INVALID_ROUTE_CYCLE")
        if len(route_tuple) != 1:
            raise ContractError("MULTIHOP_OUTSIDE_V1_SINGLE_HOP_DOMAIN")
        if (source, route_tuple[0]) not in self.links:
            raise ContractError("ROUTE_LINK_UNAVAILABLE")
        return route_tuple

    def apply_route(self, task_id: str, route: Sequence[str], *,
                    flow_type: str = "Return", logical_destination: str | None = None,
                    total_data: float | None = None) -> str | None:
        """Apply one explicit Route action to a cloned pre-Route state."""
        if task_id not in self.tasks:
            raise ContractError("TASK_ID_UNKNOWN")
        if flow_type not in SUPPORTED_FLOW_TYPES:
            raise ContractError("FLOW_TYPE_UNSUPPORTED")
        if self._existing_active(task_id, flow_type) is not None:
            raise ContractError("DUPLICATE_FLOW_BIRTH")
        task = self.tasks[task_id]
        if flow_type == "Return":
            if task.lifecycle != "waiting_to_return" or not task.completed_exact:
                raise ContractError("TASK_NOT_LEGALLY_WAITING_RETURN")
            destination = str(task.return_destination_id) if task.return_destination_id is not None else None
            if destination is None:
                raise ContractError("RETURN_DESTINATION_MISSING")
            if task.required_returned_size <= 0 or task.current_node_id == destination:
                self.events.append({"event": "NO_RETURN_REQUIRED", "task_id": task_id,
                                    "time_s": self.now_s})
                return None
            amount = task.required_returned_size if total_data is None else _nonnegative(total_data, "total_data")
            if not math.isclose(amount, task.required_returned_size, abs_tol=1e-12):
                raise ContractError("RETURN_SIZE_MISMATCH")
            source = task.current_node_id
        else:
            if task.lifecycle not in {"waiting_to_offload", "offloading"}:
                raise ContractError("TASK_NOT_LEGALLY_INPUT_OFFLOAD")
            destination = str(logical_destination) if logical_destination is not None else None
            if destination is None:
                raise ContractError("INPUT_DESTINATION_MISSING")
            amount = task.task_size if total_data is None else _nonnegative(total_data, "total_data")
            source = task.current_node_id
        route_tuple = self._validate_route(source, destination, route)
        epoch = max((f.epoch for f in self.flows.values()
                     if f.task_id == task_id and f.flow_type == flow_type), default=-1) + 1
        flow_id = build_flow_id(task_id, flow_type, epoch)
        slot = self._next_slot()
        flow = FlowContract(flow_id, flow_type, epoch, slot, task_id, source,
                            destination, route_tuple, 0, source, 0, amount,
                            0.0, amount, hop_progress=0.0,
                            hop_remaining=amount)
        self.flows[flow_id] = flow
        if flow_type == "Return":
            task.lifecycle = "returning"
            task.last_return_time = self.now_s
        else:
            task.lifecycle = "offloading"
        self.events.append({"event": "FLOW_BIRTH", "flow_id": flow_id,
                            "slot": slot, "task_id": task_id,
                            "flow_type": flow_type, "time_s": self.now_s,
                            "source": source, "destination": destination})
        return flow_id

    def assign_comm(self, task_id: str, rb_indices: Sequence[int]) -> None:
        flow = self._flow_for_task(task_id)
        if flow is None or not flow.active:
            raise ContractError("COMM_FLOW_NOT_ACTIVE")
        if (flow.holder_id, flow.route[flow.current_hop_index]) not in self.links:
            raise ContractError("COMM_LINK_UNAVAILABLE")
        values = tuple(int(x) for x in rb_indices)
        if not values or len(set(values)) != len(values):
            raise ContractError("COMM_RB_LIST_INVALID")
        if any(x < 0 or x >= self.rb_count for x in values):
            raise ContractError("COMM_RB_OUT_OF_RANGE")
        self.rb_assignments[task_id] = values
        self.events.append({"event": "COMM_ASSIGNMENT", "task_id": task_id,
                            "rb_indices": values, "time_s": self.now_s})

    def _flow_for_task(self, task_id: str) -> FlowContract | None:
        rows = [f for f in self.flows.values() if f.task_id == task_id and f.status == "ACTIVE"]
        if len(rows) > 1:
            raise ContractError("AMBIGUOUS_ACTIVE_FLOW")
        return rows[0] if rows else None

    def advance_tick(self, dt_s: float, *, service_events: Sequence[Mapping[str, Any]] = ()) -> list[dict[str, Any]]:
        """Compatibility order: TTI check first, then explicit service events."""
        dt = _finite(dt_s, "dt_s")
        if dt <= 0:
            raise ContractError("DT_NOT_POSITIVE")
        emitted: list[dict[str, Any]] = []
        # This intentionally uses last_transmission_time, matching the actual
        # AirFogSim checkTasks() path.  It does not reset the clock on birth.
        for task in self.tasks.values():
            if task.lifecycle in {"offloading", "returning"}:
                if self.now_s - task.last_transmission_time > self.tti_threshold:
                    task.lifecycle = "failed"
                    event = {"event": "TASK_FAIL_OUT_OF_TTI", "task_id": task.task_id,
                             "time_s": self.now_s, "service_skipped": True}
                    self.events.append(event); emitted.append(event)
        if not emitted:
            for event in service_events:
                self.apply_transfer(event)
                emitted.append(dict(event))
        self.now_s += dt
        return emitted

    def apply_transfer(self, event: Mapping[str, Any]) -> dict[str, Any]:
        required = {"task_id", "source_id", "target_id", "delivered_data", "stage_or_hop_completed"}
        missing = sorted(required - set(event))
        if missing:
            raise ContractError(f"TRANSFER_FIELDS_MISSING:{','.join(missing)}")
        if set(event) & {"future_target", "future_outcome", "target_tensor"}:
            raise ContractError("FUTURE_TRUTH_FORBIDDEN")
        task_id = str(event["task_id"])
        task = self.tasks.get(task_id)
        flow = self._flow_for_task(task_id)
        if task is None or flow is None:
            raise ContractError("TRANSFER_FLOW_UNKNOWN")
        if task.lifecycle in {"failed", "done"}:
            raise ContractError("TRANSFER_TASK_TERMINAL")
        source, target = str(event["source_id"]), str(event["target_id"])
        expected_target = flow.route[flow.current_hop_index]
        if source != flow.holder_id or target != expected_target:
            raise ContractError("TRANSFER_HOLDER_OR_TARGET_MISMATCH")
        amount = _nonnegative(event["delivered_data"], "delivered_data")
        if amount > flow.hop_remaining + 1e-12:
            raise ContractError("TRANSFER_EXCEEDS_HOP_REMAINING")
        completed = bool(event["stage_or_hop_completed"])
        if completed != math.isclose(amount, flow.hop_remaining, abs_tol=1e-12):
            raise ContractError("HOP_COMPLETION_INCONSISTENT")
        before = asdict(flow)
        flow.hop_progress += amount
        flow.hop_remaining -= amount
        flow.holder_id = target if completed else source
        if target == flow.destination_id:
            flow.e2e_delivered = min(flow.total_data, flow.e2e_delivered + amount)
            flow.e2e_remaining = flow.total_data - flow.e2e_delivered
        if completed:
            flow.current_hop_index += 1
            if flow.current_hop_index >= len(flow.route):
                flow.active = False; flow.presence = False; flow.status = "COMPLETED"
                flow.hop_remaining = 0.0; flow.e2e_remaining = 0.0
                flow.e2e_delivered = flow.total_data
                task.lifecycle = "done"; task.current_node_id = target
            else:
                flow.hop_progress = 0.0
                flow.hop_remaining = flow.e2e_remaining
        if not math.isclose(flow.e2e_delivered + flow.e2e_remaining,
                            flow.total_data, abs_tol=1e-9):
            raise ContractError("FLOW_CONSERVATION_FAILED")
        result = {"event": "TRANSFER", "before": before, "after": asdict(flow),
                  "task_lifecycle": task.lifecycle}
        self.events.append(result)
        return result


def assert_common_anchor(left: ContractState, right: ContractState) -> None:
    if left.anchor_fingerprint() != right.anchor_fingerprint():
        raise ContractError("COMMON_PRE_ROUTE_ANCHOR_MISMATCH")


def bind_new_flow_to_model_state(*, state: Mapping[str, Any], source_index: int,
                                 destination_index: int, flow_slot: int,
                                 task_index: int, route: Sequence[int]) -> dict[str, Any]:
    """Read-only dynamic-slot binding check; no tensor or model mutation."""
    required = ("flow_presence", "flow_known", "comm_presence", "comm_validity",
                "comm_source_index", "comm_target_index")
    missing = [key for key in required if key not in state]
    if missing:
        raise ContractError(f"MODEL_STATE_FIELDS_MISSING:{','.join(missing)}")
    if not isinstance(source_index, int) or not isinstance(destination_index, int):
        raise ContractError("MODEL_ENTITY_INDEX_MUST_BE_INT")
    width = int(state["flow_presence"].shape[1])
    if flow_slot < 0 or flow_slot > width:
        raise ContractError("MODEL_FLOW_SLOT_OUT_OF_RANGE")
    append_required = flow_slot == width
    if not append_required and (bool(state["flow_known"][0, flow_slot]) or bool(state["flow_presence"][0, flow_slot])):
        raise ContractError("MODEL_FLOW_SLOT_ALREADY_USED")
    if not route or int(route[-1]) != destination_index:
        raise ContractError("MODEL_ROUTE_DESTINATION_MISMATCH")
    hits = ((state["comm_presence"][0] & state["comm_validity"][0]) &
            (state["comm_source_index"][0] == source_index) &
            (state["comm_target_index"][0] == int(route[0])))
    if not bool(hits.any()):
        raise ContractError("MODEL_COMM_SUPPORT_MISSING")
    if task_index < 0:
        raise ContractError("MODEL_TASK_INDEX_INVALID")
    return {"flow_slot": flow_slot, "task_index": task_index,
            "source_index": source_index, "destination_index": destination_index,
            "route": tuple(int(x) for x in route),
            "relation_index": int(hits.nonzero(as_tuple=False)[0]),
            "append_required": append_required}


def contract_metadata() -> dict[str, Any]:
    return {"schema": CONTRACT_SCHEMA, "formal_semantics_modified": False,
            "single_hop_only": True, "future_truth_used": False,
            "tti_mode": "AIRFOGSIM_COMPAT_LAST_TRANSMISSION_TIME",
            "formal_planner_integration": False}
