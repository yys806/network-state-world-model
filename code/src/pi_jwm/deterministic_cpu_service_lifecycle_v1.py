"""Deterministic CPU service and perception-task lifecycle for R2-09A.

This module is deliberately independent of AirFogSim's wireless implementation.
It consumes already validated R2-08 records and explicit test rates.  A service
interval produces two separate quantities:

* ``theoretical_capacity_bytes`` is the rate/time calculation; and
* ``delivered_bytes`` is an integer simulated delivery obtained from a
  fractional-byte credit carried by that packet (or request).

The latter is an engineering fixture receipt.  It is never marked as verified
physical wireless service.  Feature packets are keyed by ``packet_id`` so a
  future spatial-block representation can use the same lifecycle engine.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Any, Callable, Mapping, Sequence

from .scenario_contract_v1 import record, validate_record, validate_scene_config
from .v2u4real_interface_v1 import InterfaceBlocked


VERSION = "PIJWM-R2-09A-Deterministic-CPU-Lifecycle-v1"
TOL = 1e-9

REQUEST_NOT_DELIVERED = "REQUEST_NOT_DELIVERED"
FEATURE_PACKET_NOT_DELIVERED = "FEATURE_PACKET_NOT_DELIVERED"
RSU_PROCESSING_TIMEOUT = "RSU_PROCESSING_TIMEOUT"
DETECTION_FAILED = "DETECTION_FAILED"
DETECTION_NOT_EXECUTED = "DETECTION_NOT_EXECUTED"
COMPLETED = "COMPLETED"

_TERMINAL_REASONS = {
    REQUEST_NOT_DELIVERED,
    FEATURE_PACKET_NOT_DELIVERED,
    RSU_PROCESSING_TIMEOUT,
    DETECTION_FAILED,
    DETECTION_NOT_EXECUTED,
    COMPLETED,
}


def _finite(value: Any, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InterfaceBlocked(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value) or (minimum is not None and value < minimum):
        raise InterfaceBlocked(f"{name} must be a finite number >= {minimum}")
    return value


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise InterfaceBlocked(f"{name} must be an integer >= {minimum}")
    return value


def _same(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=0.0, abs_tol=TOL)


@dataclass(frozen=True)
class DeterministicLifecyclePolicy:
    """Explicit engineering-only timing and service policy.

    ``deadline_boundary`` and ``same_time_event_order`` are intentionally
    constructor arguments.  They are test strategies, not frozen research
    decisions.  ``processing_duration_s`` models fixed RSU processing only;
    it is not a measured detector latency.
    """

    rb_count: int
    processing_duration_s: float
    rate_unit: str = "Mbps"
    deadline_boundary: str = "inclusive"
    same_time_event_order: str = "processing_before_deadline"
    rsu_id: str = "RSU_0"

    def __post_init__(self) -> None:
        _integer(self.rb_count, "rb_count", minimum=1)
        _finite(self.processing_duration_s, "processing_duration_s", minimum=0.0)
        if self.rate_unit not in ("Mbps", "bit/s"):
            raise InterfaceBlocked("rate_unit must be Mbps or bit/s")
        if self.deadline_boundary not in ("inclusive", "exclusive"):
            raise InterfaceBlocked("deadline_boundary must be inclusive or exclusive")
        if self.same_time_event_order not in (
            "processing_before_deadline",
            "deadline_before_processing",
        ):
            raise InterfaceBlocked(
                "same_time_event_order must be processing_before_deadline or deadline_before_processing"
            )
        if not isinstance(self.rsu_id, str) or not self.rsu_id:
            raise InterfaceBlocked("rsu_id must be a non-empty string")

    @classmethod
    def from_scene_config(
        cls,
        config: Mapping[str, Any],
        *,
        processing_duration_s: float,
        rate_unit: str = "Mbps",
        deadline_boundary: str = "inclusive",
        same_time_event_order: str = "processing_before_deadline",
    ) -> "DeterministicLifecyclePolicy":
        validate_scene_config(config, require_complete=True)
        return cls(
            rb_count=config["parameters"]["rb_count"],
            processing_duration_s=processing_duration_s,
            rate_unit=rate_unit,
            deadline_boundary=deadline_boundary,
            same_time_event_order=same_time_event_order,
        )


@dataclass(frozen=True)
class DetectionOutcome:
    """Controlled detector stub result; it is never a real metric result."""

    status: str
    output: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.status not in ("success", "failed", "not_executed"):
            raise InterfaceBlocked("detector status must be success, failed, or not_executed")


Detector = Callable[[str, tuple[str, ...]], DetectionOutcome | bool | Mapping[str, Any] | None]


@dataclass
class _Transfer:
    target_id: str
    total_bytes: int
    delivered_bytes: int = 0
    fractional_credit_bytes: float = 0.0
    first_service_at_s: float | None = None
    completed_at_s: float | None = None
    intervals: list[tuple[float, float, str]] | None = None

    def __post_init__(self) -> None:
        if self.intervals is None:
            self.intervals = []

    @property
    def remaining_bytes(self) -> int:
        return self.total_bytes - self.delivered_bytes


class DeterministicCpuServiceLifecycle:
    """CPU-only multi-slot service engine and task lifecycle.

    The engine is intentionally explicit: callers register records, submit a
    request transfer, submit packet service intervals, and call ``advance_to``
    to process RSU completion/deadlines.  It does not read an environment or
    infer future service from a prediction.
    """

    def __init__(
        self,
        config: Mapping[str, Any],
        policy: DeterministicLifecyclePolicy,
        *,
        detector: Detector | None = None,
    ) -> None:
        validate_scene_config(config, require_complete=True)
        self.config = deepcopy(dict(config))
        self.policy = policy
        self.detector = detector
        self._tasks: dict[str, dict[str, Any]] = {}
        self._packets: dict[str, dict[str, Any]] = {}
        self._request_transfers: dict[str, _Transfer] = {}
        self._packet_transfers: dict[str, _Transfer] = {}
        self._used_event_ids: set[str] = set()
        self._used_service_ids: set[str] = set()
        self._resource_intervals: list[tuple[float, float, frozenset[int], str]] = []
        self._service_records: list[dict[str, Any]] = []
        self._request_receipts: list[dict[str, Any]] = []
        self._perception_results: list[dict[str, Any]] = []
        self._events: list[dict[str, Any]] = []
        self._clock_s = 0.0

    # ---------- registration and inspection ----------
    def register_task(self, task: Mapping[str, Any], *, request_byte_count: int) -> None:
        row = validate_record(deepcopy(dict(task)))
        if row["record_type"] != "PerceptionTask":
            raise InterfaceBlocked("register_task expects a PerceptionTask record")
        task_id = row["task_id"]
        if task_id in self._tasks:
            raise InterfaceBlocked("duplicate task_id")
        total = _integer(request_byte_count, "request_byte_count")
        generated = _finite(row["generated_at_s"], "generated_at_s", minimum=0.0)
        if generated < self._clock_s - TOL:
            raise InterfaceBlocked("task is generated before the lifecycle clock")
        packet_ids = list(row["feature_packet_ids"])
        transfer = _Transfer(f"request:{task_id}", total)
        if total == 0:
            transfer.completed_at_s = generated
        self._request_transfers[task_id] = transfer
        self._tasks[task_id] = {
            "record": row,
            "request_byte_count": total,
            "request_status": "delivered" if total == 0 else "pending",
            "request_delivered_at_s": generated if total == 0 else None,
            "packet_ids": packet_ids,
            "packet_status": {},
            "processing_started_at_s": None,
            "processing_completed_at_s": None,
            "detector_called": False,
            "terminal_reason": None,
            "terminal_time_s": None,
            "result_id": None,
        }
        self._emit(generated, task_id, "task_generated", request_byte_count=total)

    def register_packet(self, packet: Mapping[str, Any]) -> None:
        row = validate_record(deepcopy(dict(packet)))
        if row["record_type"] != "FeaturePacket":
            raise InterfaceBlocked("register_packet expects a FeaturePacket record")
        packet_id = row["packet_id"]
        task_id = row["task_id"]
        if packet_id in self._packets:
            raise InterfaceBlocked("duplicate packet_id")
        if task_id not in self._tasks:
            raise InterfaceBlocked("packet references an unknown task")
        task = self._tasks[task_id]
        if task["terminal_reason"] is not None:
            raise InterfaceBlocked("cannot register a packet after task termination")
        if task["packet_ids"] and packet_id not in task["packet_ids"]:
            raise InterfaceBlocked("packet_id is not declared by its task")
        if packet_id not in task["packet_ids"]:
            task["packet_ids"].append(packet_id)
            task["record"]["feature_packet_ids"] = list(task["packet_ids"])
        self._packets[packet_id] = row
        total = row["byte_count"]
        self._packet_transfers[packet_id] = _Transfer(packet_id, total)
        task["packet_status"][packet_id] = "delivered" if total == 0 else "pending"
        if total == 0:
            self._packet_transfers[packet_id].completed_at_s = row["ready_at_s"]
        self._emit(row["ready_at_s"], task_id, "packet_ready", packet_id=packet_id, byte_count=total)

    def task_state(self, task_id: str) -> dict[str, Any]:
        if task_id not in self._tasks:
            raise InterfaceBlocked("unknown task_id")
        return deepcopy(self._task_snapshot(task_id))

    def packet_state(self, packet_id: str) -> dict[str, Any]:
        if packet_id not in self._packet_transfers:
            raise InterfaceBlocked("unknown packet_id")
        transfer = self._packet_transfers[packet_id]
        return {
            "packet_id": packet_id,
            "byte_count": transfer.total_bytes,
            "delivered_bytes": transfer.delivered_bytes,
            "remaining_bytes": transfer.remaining_bytes,
            "fractional_credit_bytes": transfer.fractional_credit_bytes,
            "status": "delivered" if transfer.completed_at_s is not None else (
                "partial" if transfer.delivered_bytes else "pending"
            ),
            "completed_at_s": transfer.completed_at_s,
        }

    def results(self) -> dict[str, Any]:
        return {
            "schema_version": VERSION,
            "service_records": deepcopy(self._service_records),
            "request_receipts": deepcopy(self._request_receipts),
            "perception_results": deepcopy(self._perception_results),
            "events": deepcopy(self._events),
            "tasks": {task_id: self._task_snapshot(task_id) for task_id in sorted(self._tasks)},
            "policy": {
                "rb_count": self.policy.rb_count,
                "processing_duration_s": self.policy.processing_duration_s,
                "rate_unit": self.policy.rate_unit,
                "deadline_boundary": self.policy.deadline_boundary,
                "same_time_event_order": self.policy.same_time_event_order,
                "rsu_id": self.policy.rsu_id,
            },
            "clock_s": self._clock_s,
            "physical_service_status": "SYNTHETIC_ONLY",
            "locked_test": False,
        }

    # ---------- deterministic service ----------
    def service_request(
        self,
        task_id: str,
        *,
        event_id: str,
        interval_start_s: float,
        interval_end_s: float,
        effective_rbs: Sequence[int],
        rate_values: Sequence[float],
        rate_unit: str | None = None,
    ) -> dict[str, Any]:
        self._require_task_active(task_id)
        task = self._tasks[task_id]
        transfer = self._request_transfers[task_id]
        if transfer.completed_at_s is not None:
            raise InterfaceBlocked("task request is already delivered")
        start, end, rbs, rates, capacity = self._prepare_interval(
            task_id, event_id, interval_start_s, interval_end_s, effective_rbs, rate_values,
            rate_unit=rate_unit, transfer=transfer,
        )
        delivered = self._apply_bytes(transfer, capacity, end)
        receipt = {
            "schema_version": VERSION,
            "record_type": "RequestServiceReceipt",
            "service_id": f"service:{event_id}",
            "event_id": event_id,
            "task_id": task_id,
            "interval_start_s": start,
            "interval_end_s": end,
            "effective_rbs": list(rbs),
            "rate_values": list(rates),
            "rate_unit": rate_unit or self.policy.rate_unit,
            "theoretical_capacity_bytes": capacity,
            "delivered_bytes": delivered,
            "remaining_bytes": transfer.remaining_bytes,
            "delivery_status": "complete" if transfer.completed_at_s is not None else (
                "partial" if delivered else "no_delivery"
            ),
            "physical_service_status": "SYNTHETIC_ONLY",
        }
        self._request_receipts.append(receipt)
        self._emit(end, task_id, "request_service", **{k: receipt[k] for k in (
            "service_id", "delivered_bytes", "remaining_bytes", "delivery_status"
        )})
        if transfer.completed_at_s is not None:
            task["request_status"] = "delivered"
            task["request_delivered_at_s"] = transfer.completed_at_s
            self._emit(end, task_id, "request_delivered", delivered_at_s=transfer.completed_at_s)
        return deepcopy(receipt)

    def service_packet(
        self,
        packet_id: str,
        *,
        event_id: str,
        interval_start_s: float,
        interval_end_s: float,
        effective_rbs: Sequence[int],
        rate_values: Sequence[float],
        action_id: str | None = None,
        rate_unit: str | None = None,
    ) -> dict[str, Any]:
        if packet_id not in self._packets:
            raise InterfaceBlocked("unknown packet_id")
        packet = self._packets[packet_id]
        task_id = packet["task_id"]
        self._require_task_active(task_id)
        task = self._tasks[task_id]
        transfer = self._packet_transfers[packet_id]
        if task["request_delivered_at_s"] is None:
            raise InterfaceBlocked("feature service requires delivered task request")
        start_value = _finite(interval_start_s, "interval_start_s", minimum=0.0)
        if start_value < task["request_delivered_at_s"] - TOL:
            raise InterfaceBlocked("feature service starts before request delivery")
        if start_value < packet["ready_at_s"] - TOL:
            raise InterfaceBlocked("feature service starts before packet ready time")
        if transfer.completed_at_s is not None:
            raise InterfaceBlocked("packet is already completely delivered")
        start, end, rbs, rates, capacity = self._prepare_interval(
            packet_id, event_id, interval_start_s, interval_end_s, effective_rbs, rate_values,
            rate_unit=rate_unit, transfer=transfer, task_id=task_id,
        )
        delivered = self._apply_bytes(transfer, capacity, end)
        service_id = f"service:{event_id}"
        provenance = deepcopy(packet["provenance"])
        service = record(
            "ServiceResult",
            provenance,
            service_id=service_id,
            link_service_id=service_id,
            action_id=action_id or f"action:{event_id}",
            packet_id=packet_id,
            source_id=packet["source_id"],
            rsu_id=self.policy.rsu_id,
            interval_start_s=start,
            interval_end_s=end,
            available_at_s=end,
            effective_rbs=list(rbs),
            rate_values=list(rates),
            rate_unit=rate_unit or self.policy.rate_unit,
            rate_semantics="fixture_constant",
            result_kind="delivery_receipt",
            theoretical_capacity_bytes=capacity,
            delivered_bytes=delivered,
            delivery_evidence_ref=f"{VERSION}:{service_id}",
            physical_service_status="SYNTHETIC_ONLY",
        )
        self._service_records.append(service)
        status = "complete" if transfer.completed_at_s is not None else ("partial" if delivered else "no_delivery")
        self._emit(end, task_id, "packet_service", service_id=service_id, packet_id=packet_id,
                   delivered_bytes=delivered, remaining_bytes=transfer.remaining_bytes,
                   delivery_status=status)
        if transfer.completed_at_s is not None:
            task["packet_status"][packet_id] = "delivered"
            self._emit(end, task_id, "packet_delivered", packet_id=packet_id,
                       delivered_at_s=transfer.completed_at_s)
        elif delivered:
            task["packet_status"][packet_id] = "partial"
        return deepcopy(service)

    def advance_to(self, time_s: float) -> dict[str, Any]:
        target = _finite(time_s, "time_s", minimum=0.0)
        if target < self._clock_s - TOL:
            raise InterfaceBlocked("lifecycle clock cannot move backwards")
        self._clock_s = max(self._clock_s, target)
        for task_id in sorted(self._tasks):
            self._evaluate_task(task_id, target)
        return {task_id: self._task_snapshot(task_id) for task_id in sorted(self._tasks)}

    # ---------- internal validation and state transitions ----------
    def _prepare_interval(
        self,
        target_id: str,
        event_id: str,
        interval_start_s: float,
        interval_end_s: float,
        effective_rbs: Sequence[int],
        rate_values: Sequence[float],
        *,
        rate_unit: str | None,
        transfer: _Transfer,
        task_id: str | None = None,
    ) -> tuple[float, float, tuple[int, ...], tuple[float, ...], float]:
        if not isinstance(event_id, str) or not event_id:
            raise InterfaceBlocked("event_id must be a non-empty string")
        if event_id in self._used_event_ids:
            raise InterfaceBlocked("duplicate service event")
        selected_unit = rate_unit or self.policy.rate_unit
        if selected_unit not in ("Mbps", "bit/s"):
            raise InterfaceBlocked("rate_unit must be Mbps or bit/s")
        start = _finite(interval_start_s, "interval_start_s", minimum=0.0)
        requested_end = _finite(interval_end_s, "interval_end_s", minimum=0.0)
        if requested_end <= start + TOL:
            raise InterfaceBlocked("service interval must have positive duration")
        if start < self._clock_s - TOL:
            raise InterfaceBlocked("service interval starts before the lifecycle clock")
        if len(effective_rbs) != len(rate_values):
            raise InterfaceBlocked("one rate value is required per effective RB")
        rbs: list[int] = []
        for rb in effective_rbs:
            _integer(rb, "RB index")
            if rb >= self.policy.rb_count:
                raise InterfaceBlocked("RB index exceeds configured budget")
            if rb in rbs:
                raise InterfaceBlocked("duplicate RB in one service event")
            rbs.append(rb)
        rates: list[float] = []
        for rate in rate_values:
            rates.append(_finite(rate, "rate", minimum=0.0))
        task_key = task_id or (target_id if target_id in self._tasks else None)
        if task_key is None:
            raise InterfaceBlocked("service target is not attached to a task")
        deadline = self._tasks[task_key]["record"]["absolute_deadline_s"]
        if start >= deadline - TOL:
            raise InterfaceBlocked("service starts at or after the task deadline")
        end = min(requested_end, deadline)
        if end <= start + TOL:
            raise InterfaceBlocked("service interval has no time before deadline")
        self._check_target_interval(transfer, start, end, event_id)
        self._check_resource_overlap(start, end, rbs, event_id)
        factor = 1e6 if selected_unit == "Mbps" else 1.0
        capacity = sum(rates) * factor * (end - start) / 8.0
        if not math.isfinite(capacity) or capacity < -TOL:
            raise InterfaceBlocked("service capacity is invalid")
        self._used_event_ids.add(event_id)
        service_id = f"service:{event_id}"
        if service_id in self._used_service_ids:
            raise InterfaceBlocked("duplicate service id")
        self._used_service_ids.add(service_id)
        transfer.intervals.append((start, end, event_id))
        self._resource_intervals.append((start, end, frozenset(rbs), event_id))
        if transfer.first_service_at_s is None:
            transfer.first_service_at_s = start
        return start, end, tuple(rbs), tuple(rates), capacity

    @staticmethod
    def _check_target_interval(transfer: _Transfer, start: float, end: float, event_id: str) -> None:
        for old_start, old_end, _ in transfer.intervals or []:
            if old_start < end - TOL and start < old_end - TOL:
                raise InterfaceBlocked(f"overlapping service intervals for {transfer.target_id}: {event_id}")

    def _check_resource_overlap(self, start: float, end: float, rbs: Sequence[int], event_id: str) -> None:
        new = frozenset(rbs)
        for old_start, old_end, old_rbs, old_event in self._resource_intervals:
            if old_start < end - TOL and start < old_end - TOL and new.intersection(old_rbs):
                raise InterfaceBlocked(f"RB reused by overlapping service events {old_event} and {event_id}")

    @staticmethod
    def _apply_bytes(transfer: _Transfer, capacity: float, completed_at_s: float) -> int:
        if transfer.remaining_bytes <= 0:
            raise InterfaceBlocked("service would exceed an already complete transfer")
        available = transfer.fractional_credit_bytes + capacity
        if available < -TOL or not math.isfinite(available):
            raise InterfaceBlocked("non-finite accumulated service")
        delivered = min(transfer.remaining_bytes, max(0, int(math.floor(available + 1e-12))))
        transfer.delivered_bytes += delivered
        transfer.fractional_credit_bytes = max(0.0, available - delivered)
        if transfer.delivered_bytes > transfer.total_bytes:
            raise InterfaceBlocked("service delivered more than packet/request size")
        if transfer.delivered_bytes == transfer.total_bytes:
            transfer.completed_at_s = completed_at_s
            transfer.fractional_credit_bytes = 0.0
        return delivered

    def _evaluate_task(self, task_id: str, now: float) -> None:
        task = self._tasks[task_id]
        if task["terminal_reason"] is not None:
            return
        deadline = float(task["record"]["absolute_deadline_s"])
        request = self._request_transfers[task_id]
        if request.completed_at_s is None:
            if now >= deadline - TOL:
                self._terminate(task_id, REQUEST_NOT_DELIVERED, deadline)
            return
        packet_ids = task["packet_ids"]
        if not packet_ids or any(
            pid not in self._packet_transfers or self._packet_transfers[pid].completed_at_s is None
            for pid in packet_ids
        ):
            if now >= deadline - TOL:
                self._terminate(task_id, FEATURE_PACKET_NOT_DELIVERED, deadline)
            return
        transfers = [self._packet_transfers[pid] for pid in packet_ids]
        latest_packet_time = max(
            [request.completed_at_s or task["record"]["generated_at_s"]]
            + [transfer.completed_at_s for transfer in transfers if transfer.completed_at_s is not None]
        )
        if task["processing_started_at_s"] is None:
            if latest_packet_time > now + TOL:
                return
            task["processing_started_at_s"] = latest_packet_time
            task["record"]["status"] = "detecting"
            task["record"]["actual_completion_time_s"] = None
            processing_end = latest_packet_time + self.policy.processing_duration_s
            task["processing_completed_at_s"] = processing_end
            self._emit(latest_packet_time, task_id, "rsu_processing_started",
                       processing_completed_at_s=processing_end)
        processing_end = task["processing_completed_at_s"]
        if processing_end is None or now + TOL < processing_end:
            if now >= deadline - TOL and (
                processing_end is None or processing_end > deadline + TOL or
                not self._completion_allowed(processing_end, deadline)
            ):
                self._terminate(task_id, RSU_PROCESSING_TIMEOUT, deadline)
            return
        same_time_deadline = _same(processing_end, deadline)
        if same_time_deadline and self.policy.same_time_event_order == "deadline_before_processing":
            self._terminate(task_id, RSU_PROCESSING_TIMEOUT, deadline)
            return
        if not self._completion_allowed(processing_end, deadline):
            self._terminate(task_id, RSU_PROCESSING_TIMEOUT, deadline)
            return
        self._run_detection(task_id, processing_end)

    def _completion_allowed(self, time_s: float, deadline: float) -> bool:
        if self.policy.deadline_boundary == "inclusive":
            return time_s <= deadline + TOL
        return time_s < deadline - TOL

    def _run_detection(self, task_id: str, output_at_s: float) -> None:
        task = self._tasks[task_id]
        if task["detector_called"]:
            return
        task["detector_called"] = True
        packet_ids = tuple(task["packet_ids"])
        outcome = self._normalize_detector(self.detector(task_id, packet_ids) if self.detector else None)
        provenance = deepcopy(task["record"]["provenance"])
        provenance["kind"] = "mock_output" if outcome.status != "not_executed" else "synthetic_fixture"
        receipts = [
            {
                "packet_id": pid,
                "received_byte_count": self._packet_transfers[pid].total_bytes,
                "available_at_s": self._packet_transfers[pid].completed_at_s,
                "integrity_status": "complete",
                "evidence_ref": f"{VERSION}:packet:{pid}",
                "evidence_kind": "runtime_receipt",
            }
            for pid in packet_ids
        ]
        result_id = f"result:{task_id}"
        result = record(
            "PerceptionResult",
            provenance,
            result_id=result_id,
            task_id=task_id,
            used_packet_ids=list(packet_ids),
            output_at_s=output_at_s,
            actual_completion_time_s=output_at_s,
            detection_output=None,
            evaluation_result=None,
            output_role="synthetic_contract_fixture",
            input_receipts=receipts,
        )
        self._perception_results.append(result)
        task["result_id"] = result_id
        if outcome.status == "success":
            task["record"]["status"] = "completed"
            task["record"]["actual_completion_time_s"] = output_at_s
            self._terminate(task_id, COMPLETED, output_at_s, preserve_completion=True)
        elif outcome.status == "failed":
            task["record"]["status"] = "detecting"
            task["record"]["actual_completion_time_s"] = None
            self._terminate(task_id, DETECTION_FAILED, output_at_s, preserve_completion=True)
        else:
            task["record"]["status"] = "detecting"
            task["record"]["actual_completion_time_s"] = None
            self._terminate(task_id, DETECTION_NOT_EXECUTED, output_at_s, preserve_completion=True)
        self._emit(output_at_s, task_id, "detection_result", result_id=result_id,
                   detection_status=outcome.status, simulated=True)

    @staticmethod
    def _normalize_detector(value: DetectionOutcome | bool | Mapping[str, Any] | None) -> DetectionOutcome:
        if isinstance(value, DetectionOutcome):
            return value
        if value is None:
            return DetectionOutcome("not_executed")
        if isinstance(value, bool):
            return DetectionOutcome("success" if value else "failed")
        if isinstance(value, Mapping):
            status = value.get("status", value.get("detection_status"))
            if status in ("success", "failed", "not_executed"):
                return DetectionOutcome(status, value.get("output"))
        raise InterfaceBlocked("detector must return DetectionOutcome, bool, mapping, or None")

    def _terminate(self, task_id: str, reason: str, time_s: float, *, preserve_completion: bool = False) -> None:
        if reason not in _TERMINAL_REASONS:
            raise InterfaceBlocked("unknown lifecycle terminal reason")
        task = self._tasks[task_id]
        if task["terminal_reason"] is not None:
            return
        task["terminal_reason"] = reason
        task["terminal_time_s"] = time_s
        if reason in (REQUEST_NOT_DELIVERED, FEATURE_PACKET_NOT_DELIVERED, RSU_PROCESSING_TIMEOUT):
            task["record"]["status"] = "timed_out"
            task["record"]["actual_completion_time_s"] = None
        elif reason == COMPLETED:
            task["record"]["status"] = "completed"
            if not preserve_completion:
                task["record"]["actual_completion_time_s"] = time_s
        self._emit(time_s, task_id, "task_terminal", terminal_reason=reason)

    def _require_task_active(self, task_id: str) -> None:
        if task_id not in self._tasks:
            raise InterfaceBlocked("unknown task_id")
        if self._tasks[task_id]["terminal_reason"] is not None:
            raise InterfaceBlocked("task is already terminal")

    def _emit(self, time_s: float, task_id: str, kind: str, **fields: Any) -> None:
        event = {"event_id": f"event:{len(self._events):06d}", "time_s": float(time_s),
                 "task_id": task_id, "kind": kind, **fields}
        self._events.append(event)

    def _task_snapshot(self, task_id: str) -> dict[str, Any]:
        task = self._tasks[task_id]
        packets = {pid: self.packet_state(pid) for pid in task["packet_ids"] if pid in self._packet_transfers}
        row = deepcopy(task["record"])
        return {
            "task_id": task_id,
            "record": row,
            "request": {
                "byte_count": task["request_byte_count"],
                "delivered_bytes": self._request_transfers[task_id].delivered_bytes,
                "remaining_bytes": self._request_transfers[task_id].remaining_bytes,
                "status": task["request_status"],
                "delivered_at_s": task["request_delivered_at_s"],
            },
            "packets": packets,
            "processing_started_at_s": task["processing_started_at_s"],
            "processing_completed_at_s": task["processing_completed_at_s"],
            "terminal_reason": task["terminal_reason"],
            "terminal_time_s": task["terminal_time_s"],
            "result_id": task["result_id"],
        }


__all__ = [
    "COMPLETED",
    "DETECTION_FAILED",
    "DETECTION_NOT_EXECUTED",
    "FEATURE_PACKET_NOT_DELIVERED",
    "REQUEST_NOT_DELIVERED",
    "RSU_PROCESSING_TIMEOUT",
    "DetectionOutcome",
    "DeterministicCpuServiceLifecycle",
    "DeterministicLifecyclePolicy",
    "VERSION",
]
