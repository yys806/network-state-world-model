"""V2U4Real perception input/output contract and deterministic mock detector.

This module is deliberately an interface layer.  It reads one already-audited
V2U4Real PCD/YAML pair, keeps the raw frame key and native coordinate
declaration, and exposes a strict detection-result schema.  ``MockPerception``
uses only the point cloud; labels are never used to construct a prediction.
The native coordinate frame is explicitly unverified and is never transformed
or interpreted here.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from .v2u4real_interface_v1 import (
    InterfaceBlocked,
    UniqueKeyLoader,
    read_pcd_xyz,
    sha256,
)


PERCEPTION_INTERFACE_VERSION = "PIJWM-V2U4Real-Perception-Interface-v1"
DETECTION_RESULT_VERSION = "PIJWM-V2U4Real-DetectionResult-v1"
BOX_FORMAT = "center_xyz_length_width_height_yaw"
BOX_DIMENSION_ORDER = ("x", "y", "z", "length", "width", "height", "yaw")
MOCK_SOURCE = "MockPerception.geometry_envelope_v1"


def _finite_points(value: Any) -> np.ndarray:
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] <= 0:
        raise InterfaceBlocked("point cloud must be a nonempty Nx3 array")
    if not np.isfinite(points).all():
        raise InterfaceBlocked("point cloud contains non-finite values")
    return points


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InterfaceBlocked(f"{field} must be a non-empty string")
    return value


def _strict_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise InterfaceBlocked(f"{field} must be a mapping")
    return value


@dataclass(frozen=True)
class PerceptionFrame:
    """One sensor frame supplied to a perception implementation.

    ``frame_key`` is the original V2U4Real file key.  It is an identifier only;
    this class intentionally does not call it a physical timestamp.
    """

    frame_key: str
    sensor_id: str
    points: np.ndarray
    coordinate_frame: str
    coordinate_status: str
    source_metadata: Mapping[str, Any]
    annotation_metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame_key", _text(self.frame_key, "frame_key"))
        object.__setattr__(self, "sensor_id", _text(self.sensor_id, "sensor_id"))
        object.__setattr__(self, "coordinate_frame", _text(self.coordinate_frame, "coordinate_frame"))
        object.__setattr__(self, "coordinate_status", _text(self.coordinate_status, "coordinate_status"))
        object.__setattr__(self, "points", _finite_points(self.points))
        _strict_mapping(self.source_metadata, "source_metadata")
        if self.annotation_metadata is not None:
            _strict_mapping(self.annotation_metadata, "annotation_metadata")

    @property
    def point_count(self) -> int:
        return int(self.points.shape[0])

    def contract_dict(self) -> dict[str, Any]:
        points_bytes = np.ascontiguousarray(self.points, dtype=np.float64).tobytes()
        return {
            "schema_version": PERCEPTION_INTERFACE_VERSION,
            "frame_key": self.frame_key,
            "frame_key_semantics": "identifier_only_not_physical_timestamp",
            "sensor_id": self.sensor_id,
            "point_shape": list(self.points.shape),
            "point_dtype": str(self.points.dtype),
            "point_count": self.point_count,
            "point_sha256": hashlib.sha256(points_bytes).hexdigest(),
            "coordinate_frame": self.coordinate_frame,
            "coordinate_status": self.coordinate_status,
            "source_metadata": copy.deepcopy(dict(self.source_metadata)),
            "annotation_metadata": copy.deepcopy(dict(self.annotation_metadata or {})),
        }


@dataclass(frozen=True)
class DetectionObject:
    """A single 3-D detection in an explicitly declared coordinate frame."""

    category: str
    box3d: tuple[float, float, float, float, float, float, float]
    confidence: float
    frame_key: str
    sensor_id: str
    coordinate_frame: str
    coordinate_status: str
    result_source: str
    prediction_semantics: str = "prediction"

    def __post_init__(self) -> None:
        object.__setattr__(self, "category", _text(self.category, "category"))
        object.__setattr__(self, "frame_key", _text(self.frame_key, "frame_key"))
        object.__setattr__(self, "sensor_id", _text(self.sensor_id, "sensor_id"))
        object.__setattr__(self, "coordinate_frame", _text(self.coordinate_frame, "coordinate_frame"))
        object.__setattr__(self, "coordinate_status", _text(self.coordinate_status, "coordinate_status"))
        object.__setattr__(self, "result_source", _text(self.result_source, "result_source"))
        object.__setattr__(self, "prediction_semantics", _text(self.prediction_semantics, "prediction_semantics"))
        try:
            box = tuple(float(x) for x in self.box3d)
        except (TypeError, ValueError) as exc:
            raise InterfaceBlocked("box3d must contain seven numeric values") from exc
        if len(box) != 7 or not np.isfinite(np.asarray(box, dtype=np.float64)).all():
            raise InterfaceBlocked("box3d must be finite and have seven values")
        if any(x <= 0 for x in box[3:6]):
            raise InterfaceBlocked("box3d length/width/height must be positive")
        object.__setattr__(self, "box3d", box)
        try:
            score = float(self.confidence)
        except (TypeError, ValueError) as exc:
            raise InterfaceBlocked("confidence must be a finite number in [0, 1]") from exc
        if not np.isfinite(score) or not 0.0 <= score <= 1.0:
            raise InterfaceBlocked("confidence must be a finite number in [0, 1]")
        object.__setattr__(self, "confidence", score)

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "box_format": BOX_FORMAT,
            "box_dimension_order": list(BOX_DIMENSION_ORDER),
            "box3d": list(self.box3d),
            "confidence": self.confidence,
            "frame_key": self.frame_key,
            "sensor_id": self.sensor_id,
            "coordinate_frame": self.coordinate_frame,
            "coordinate_status": self.coordinate_status,
            "result_source": self.result_source,
            "prediction_semantics": self.prediction_semantics,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DetectionObject":
        required = {
            "category", "box_format", "box_dimension_order", "box3d", "confidence",
            "frame_key", "sensor_id", "coordinate_frame", "coordinate_status",
            "result_source", "prediction_semantics",
        }
        _strict_mapping(value, "detection object")
        if set(value) != required:
            raise InterfaceBlocked("detection object fields do not match the contract")
        if value["box_format"] != BOX_FORMAT or tuple(value["box_dimension_order"]) != BOX_DIMENSION_ORDER:
            raise InterfaceBlocked("unsupported box format or dimension order")
        return cls(
            category=value["category"], box3d=tuple(value["box3d"]),
            confidence=value["confidence"], frame_key=value["frame_key"],
            sensor_id=value["sensor_id"], coordinate_frame=value["coordinate_frame"],
            coordinate_status=value["coordinate_status"], result_source=value["result_source"],
            prediction_semantics=value["prediction_semantics"],
        )


@dataclass(frozen=True)
class DetectionResult:
    """Strict output of a perception module for one sensor frame."""

    frame_key: str
    sensor_id: str
    coordinate_frame: str
    coordinate_status: str
    result_source: str
    objects: tuple[DetectionObject, ...]
    prediction_semantics: str = "predictions_are_not_ground_truth"

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame_key", _text(self.frame_key, "frame_key"))
        object.__setattr__(self, "sensor_id", _text(self.sensor_id, "sensor_id"))
        object.__setattr__(self, "coordinate_frame", _text(self.coordinate_frame, "coordinate_frame"))
        object.__setattr__(self, "coordinate_status", _text(self.coordinate_status, "coordinate_status"))
        object.__setattr__(self, "result_source", _text(self.result_source, "result_source"))
        object.__setattr__(self, "prediction_semantics", _text(self.prediction_semantics, "prediction_semantics"))
        objects = tuple(self.objects)
        if any(not isinstance(item, DetectionObject) for item in objects):
            raise InterfaceBlocked("objects must contain DetectionObject values")
        for item in objects:
            if (item.frame_key, item.sensor_id, item.coordinate_frame, item.coordinate_status) != (
                self.frame_key, self.sensor_id, self.coordinate_frame, self.coordinate_status
            ):
                raise InterfaceBlocked("detection object frame/sensor/coordinate declaration mismatch")
        object.__setattr__(self, "objects", objects)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": DETECTION_RESULT_VERSION,
            "frame_key": self.frame_key,
            "frame_key_semantics": "identifier_only_not_physical_timestamp",
            "sensor_id": self.sensor_id,
            "coordinate_frame": self.coordinate_frame,
            "coordinate_status": self.coordinate_status,
            "result_source": self.result_source,
            "prediction_semantics": self.prediction_semantics,
            "box_format": BOX_FORMAT,
            "box_dimension_order": list(BOX_DIMENSION_ORDER),
            "objects": [item.to_dict() for item in self.objects],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DetectionResult":
        required = {
            "schema_version", "frame_key", "frame_key_semantics", "sensor_id",
            "coordinate_frame", "coordinate_status", "result_source",
            "prediction_semantics", "box_format", "box_dimension_order", "objects",
        }
        _strict_mapping(value, "detection result")
        if set(value) != required:
            raise InterfaceBlocked("detection result fields do not match the contract")
        if value["schema_version"] != DETECTION_RESULT_VERSION:
            raise InterfaceBlocked("unsupported detection result schema")
        if value["frame_key_semantics"] != "identifier_only_not_physical_timestamp":
            raise InterfaceBlocked("frame key semantics must remain identifier-only")
        if value["box_format"] != BOX_FORMAT or tuple(value["box_dimension_order"]) != BOX_DIMENSION_ORDER:
            raise InterfaceBlocked("unsupported result box format")
        objects = tuple(DetectionObject.from_dict(item) for item in value["objects"])
        return cls(
            frame_key=value["frame_key"], sensor_id=value["sensor_id"],
            coordinate_frame=value["coordinate_frame"], coordinate_status=value["coordinate_status"],
            result_source=value["result_source"], objects=objects,
            prediction_semantics=value["prediction_semantics"],
        )


class MockPerception:
    """Deterministic point-cloud-only smoke implementation.

    The box is the finite XYZ envelope of the input cloud.  It is a software
    fixture, not a detector and not a ground-truth conversion.  Its output
    must never be used to report accuracy.
    """

    source = MOCK_SOURCE

    def predict(self, frame: PerceptionFrame) -> DetectionResult:
        if not isinstance(frame, PerceptionFrame):
            raise InterfaceBlocked("MockPerception expects PerceptionFrame")
        lower = frame.points.min(axis=0)
        upper = frame.points.max(axis=0)
        center = (lower + upper) / 2.0
        dimensions = np.maximum(upper - lower, 1e-6)
        box = tuple(float(x) for x in (*center, *dimensions, 0.0))
        obj = DetectionObject(
            category="mock_vehicle",
            box3d=box,
            confidence=0.5,
            frame_key=frame.frame_key,
            sensor_id=frame.sensor_id,
            coordinate_frame=frame.coordinate_frame,
            coordinate_status=frame.coordinate_status,
            result_source=self.source,
            prediction_semantics="deterministic_point_cloud_envelope_fixture_not_ground_truth",
        )
        return DetectionResult(
            frame_key=frame.frame_key,
            sensor_id=frame.sensor_id,
            coordinate_frame=frame.coordinate_frame,
            coordinate_status=frame.coordinate_status,
            result_source=self.source,
            objects=(obj,),
        )


@dataclass(frozen=True)
class V2U4RealFrameBundle:
    frame: PerceptionFrame
    annotations: Mapping[str, Any]
    file_metadata: Mapping[str, Any]


def read_v2u4real_frame(
    root: str | Path,
    scene: str,
    frame_key: str,
    agent: str,
    sensor: str = "ouster",
) -> V2U4RealFrameBundle:
    """Read one fixed PCD/YAML pair without changing its bytes or semantics."""

    root = Path(root).resolve()
    scene = _text(scene, "scene")
    frame_key = _text(frame_key, "frame_key")
    agent = _text(agent, "agent")
    sensor = _text(sensor, "sensor")
    scene_root = root / "train" / scene
    yaml_path = scene_root / agent / "yaml" / sensor / f"{frame_key}.yaml"
    pcd_path = scene_root / agent / sensor / f"{frame_key}.pcd"
    for path in (yaml_path, pcd_path):
        resolved = path.resolve()
        if not resolved.is_relative_to(scene_root.resolve()) or not path.is_file():
            raise InterfaceBlocked(f"missing or escaping V2U4Real frame file: {path}")
    raw = yaml.load(yaml_path.read_text(encoding="utf-8-sig"), Loader=UniqueKeyLoader)
    if not isinstance(raw, dict) or not isinstance(raw.get("vehicles"), dict):
        raise InterfaceBlocked("V2U4Real YAML requires a mapping root and vehicles mapping")
    points, header = read_pcd_xyz(pcd_path)
    metadata = {
        "dataset": "VJiaLi/V2U4Real",
        "split": "train",
        "scene": scene,
        "agent": agent,
        "sensor": sensor,
        "frame_key": frame_key,
        "frame_key_semantics": "identifier_only_not_physical_timestamp",
        "pcd_path": pcd_path.relative_to(root).as_posix(),
        "yaml_path": yaml_path.relative_to(root).as_posix(),
        "pcd_bytes": pcd_path.stat().st_size,
        "yaml_bytes": yaml_path.stat().st_size,
        "pcd_sha256": sha256(pcd_path),
        "yaml_sha256": sha256(yaml_path),
        "pcd_header": header,
        "point_count": int(points.shape[0]),
        "xyz_finite": bool(np.isfinite(points).all()),
        "annotation_count": len(raw["vehicles"]),
        "annotation_object_ids": [str(x) for x in raw["vehicles"].keys()],
        "coordinate_status": "dataset_native_unverified",
    }
    frame = PerceptionFrame(
        frame_key=frame_key,
        sensor_id=f"agent-{agent}/{sensor}",
        points=points,
        coordinate_frame=f"v2u4real::agent-{agent}::{sensor}::native",
        coordinate_status="dataset_native_unverified",
        source_metadata=metadata,
        annotation_metadata={
            "raw_yaml_sha256": metadata["yaml_sha256"],
            "labels_available": True,
            "labels_used_by_mock_perception": False,
        },
    )
    return V2U4RealFrameBundle(frame=frame, annotations=copy.deepcopy(raw["vehicles"]), file_metadata=metadata)


def validate_detection_result(value: DetectionResult | Mapping[str, Any]) -> DetectionResult:
    """Validate an in-memory result or strict serialized result."""

    if isinstance(value, DetectionResult):
        return value
    return DetectionResult.from_dict(value)


def canonical_result_bytes(result: DetectionResult | Mapping[str, Any]) -> bytes:
    validated = validate_detection_result(result)
    return json.dumps(validated.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
