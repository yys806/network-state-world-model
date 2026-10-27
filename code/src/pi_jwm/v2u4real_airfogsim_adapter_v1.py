"""Application-layer adapter for V2U4Real perception payloads.

The adapter stops at AirFogSim's task-manager ingress.  It creates a normal
AirFogSim ``Task`` record in the waiting-to-offload queue, but it does not call
the offload setter, reserve resource blocks, compute a rate, or claim delivery.
This separation is required while the R1 communication semantic contract is
still blocked.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping

from .v2u4real_perception_interface_v1 import (
    DetectionResult,
    canonical_result_bytes,
    validate_detection_result,
)
from .v2u4real_interface_v1 import InterfaceBlocked


ADAPTER_VERSION = "PIJWM-V2U4Real-AirFogSim-Perception-Adapter-v1"
WIRELESS_BLOCK_REASON = "R1 communication semantic contract and feature-packet service remain unverified"


@dataclass(frozen=True)
class PerceptionPayloadRecord:
    payload_id: str
    frame_key: str
    sensor_id: str
    byte_count: int
    detection_count: int
    coordinate_frame: str
    coordinate_status: str
    result_source: str
    serialization: str = "canonical_detection_result_json_utf8"
    wireless_delivery_status: str = "NOT_ATTEMPTED_BLOCKED"

    @classmethod
    def from_result(cls, result: DetectionResult | Mapping[str, Any]) -> "PerceptionPayloadRecord":
        validated = validate_detection_result(result)
        payload_bytes = canonical_result_bytes(validated)
        return cls(
            payload_id=hashlib.sha256(payload_bytes).hexdigest(),
            frame_key=validated.frame_key,
            sensor_id=validated.sensor_id,
            byte_count=len(payload_bytes),
            detection_count=len(validated.objects),
            coordinate_frame=validated.coordinate_frame,
            coordinate_status=validated.coordinate_status,
            result_source=validated.result_source,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter_version": ADAPTER_VERSION,
            "payload_id": self.payload_id,
            "frame_key": self.frame_key,
            "frame_key_semantics": "identifier_only_not_physical_timestamp",
            "sensor_id": self.sensor_id,
            "byte_count": self.byte_count,
            "detection_count": self.detection_count,
            "coordinate_frame": self.coordinate_frame,
            "coordinate_status": self.coordinate_status,
            "result_source": self.result_source,
            "serialization": self.serialization,
            "wireless_delivery_status": self.wireless_delivery_status,
        }


@dataclass(frozen=True)
class AirFogSimIngressReceipt:
    task_id: str
    source_node_id: str
    target_node_id: str
    simulation_time: float
    payload_id: str
    software_interface_status: str
    simulation_task_record_status: str
    wireless_delivery_status: str
    wireless_block_reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter_version": ADAPTER_VERSION,
            "task_id": self.task_id,
            "source_node_id": self.source_node_id,
            "target_node_id": self.target_node_id,
            "simulation_time": self.simulation_time,
            "payload_id": self.payload_id,
            "software_interface_status": self.software_interface_status,
            "simulation_task_record_status": self.simulation_task_record_status,
            "wireless_delivery_status": self.wireless_delivery_status,
            "wireless_block_reason": self.wireless_block_reason,
        }


def _require_nonempty(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InterfaceBlocked(f"{field} must be a non-empty string")
    return value


def submit_payload_to_task_manager(
    task_manager: Any,
    payload: PerceptionPayloadRecord,
    *,
    source_node_id: str,
    target_node_id: str,
    simulation_time: float,
    deadline: float = 1.0,
) -> tuple[Any, AirFogSimIngressReceipt]:
    """Register a payload as an AirFogSim application task.

    The existing public task-manager methods are used for generation and queue
    insertion.  No ``offloadTask`` or communication decision is made here.
    """

    if not isinstance(payload, PerceptionPayloadRecord):
        raise InterfaceBlocked("payload must be PerceptionPayloadRecord")
    source_node_id = _require_nonempty(source_node_id, "source_node_id")
    target_node_id = _require_nonempty(target_node_id, "target_node_id")
    try:
        now = float(simulation_time)
        ddl = float(deadline)
    except (TypeError, ValueError) as exc:
        raise InterfaceBlocked("simulation_time and deadline must be numeric") from exc
    if now < 0 or ddl <= 0:
        raise InterfaceBlocked("simulation_time must be nonnegative and deadline positive")
    required = ("generateTaskInfoOfMission", "getWaitingToOffloadTasksByNodeId", "setWaitingToOffloadTasksByNodeId")
    if any(not hasattr(task_manager, name) for name in required):
        raise InterfaceBlocked("task manager does not expose the required application ingress methods")

    task = task_manager.generateTaskInfoOfMission(
        source_node_id,
        ddl,
        now,
        0,
        to_return_node_id=source_node_id,
        return_lazy_set=False,
    )
    if task is None or not hasattr(task, "setAttribute") or not hasattr(task, "getTaskId"):
        raise InterfaceBlocked("task manager returned an invalid Task object")
    task.setAttribute("task_kind", "perception_payload")
    task.setAttribute("application_ingress", ADAPTER_VERSION)
    task.setAttribute("application_task_state", "waiting_to_offload")
    task.setAttribute("native_lifecycle_interpretation", "not_used_as_wireless_delivery_evidence")
    task.setAttribute("perception_payload_id", payload.payload_id)
    task.setAttribute("perception_payload_bytes", payload.byte_count)
    task.setAttribute("perception_frame_key", payload.frame_key)
    task.setAttribute("perception_sensor_id", payload.sensor_id)
    task.setAttribute("perception_result_source", payload.result_source)
    task.setAttribute("perception_coordinate_frame", payload.coordinate_frame)
    task.setAttribute("perception_coordinate_status", payload.coordinate_status)
    task.setAttribute("requested_target_node_id", target_node_id)
    task.setAttribute("wireless_delivery_status", "NOT_ATTEMPTED_BLOCKED")

    waiting = list(task_manager.getWaitingToOffloadTasksByNodeId(source_node_id))
    if any(getattr(item, "getTaskId", lambda: None)() == task.getTaskId() for item in waiting):
        raise InterfaceBlocked("duplicate task id at application ingress")
    waiting.append(task)
    task_manager.setWaitingToOffloadTasksByNodeId(source_node_id, waiting)
    receipt = AirFogSimIngressReceipt(
        task_id=str(task.getTaskId()), source_node_id=source_node_id,
        target_node_id=target_node_id, simulation_time=now,
        payload_id=payload.payload_id, software_interface_status="PASS",
        simulation_task_record_status="PASS_WAITING_TO_OFFLOAD",
        wireless_delivery_status="NOT_ATTEMPTED_BLOCKED",
        wireless_block_reason=WIRELESS_BLOCK_REASON,
    )
    return task, receipt


def submit_payload_to_env(
    env: Any,
    payload: PerceptionPayloadRecord,
    *,
    source_node_id: str,
    target_node_id: str,
    deadline: float = 1.0,
) -> tuple[Any, AirFogSimIngressReceipt]:
    """Submit through the actual AirFogSim environment's task manager."""

    if not hasattr(env, "task_manager") or not hasattr(env, "simulation_time"):
        raise InterfaceBlocked("AirFogSim environment lacks task manager/simulation time")
    return submit_payload_to_task_manager(
        env.task_manager, payload, source_node_id=source_node_id,
        target_node_id=target_node_id, simulation_time=env.simulation_time,
        deadline=deadline,
    )


def task_record_dict(task: Any) -> dict[str, Any]:
    """Return a JSON-safe task record without implying transmission."""

    if not hasattr(task, "to_dict"):
        raise InterfaceBlocked("task object lacks to_dict")
    value = task.to_dict()
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise InterfaceBlocked("task record is not JSON-safe") from exc
    return value
