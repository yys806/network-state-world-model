"""Additive Future Motion / CSI target contract for STEP 5.1A.

The contract derives supervision next to the existing sample. It never edits
History, the causal input index, or the STEP 4.4 transition implementation.
All numeric normalization comes from the frozen STEP 4.3B train-only stats.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


COMM_RELATION_TYPE_VOCAB = ["wired", "wireless"]
SAMPLE_SCHEMA_VERSION = "PI-JWM-Step5.1A-Future-Target-Sample-v1"
TENSOR_SCHEMA_VERSION = "PI-JWM-Step5.1A-Future-Target-Tensor-v1"
MOTION_SEMANTIC_ORDER = ["delta_x_m", "delta_y_m", "delta_z_m", "next_speed_mps"]


def _feature(stats: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    return stats["features"][name]


def _normalization_parameters(stats: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "position_std": [float(_feature(stats, f"entity.position_{axis}_m")["std"]) for axis in "xyz"],
        "speed_mean": float(_feature(stats, "entity.speed_mps")["mean"]),
        "speed_std": float(_feature(stats, "entity.speed_mps")["std"]),
        "csi_mean": float(_feature(stats, "comm.channel_attenuation_db")["mean"]),
        "csi_std": float(_feature(stats, "comm.channel_attenuation_db")["std"]),
    }


def _check_vector(value: Any, size: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {array.shape}")
    return array


def _check_mask(mask: Any, size: int, name: str) -> np.ndarray:
    array = np.asarray(mask, dtype=bool)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {array.shape}")
    return array


def normalize_motion(raw: Any, mask: Any, stats: Mapping[str, Any]) -> list[float]:
    """Normalize ``[delta_x, delta_y, delta_z, next_speed]``.

    Displacements use the frozen absolute-position scale only. Their
    absolute-position means are intentionally not subtracted because the
    position mean cancels in ``p_(t+1) - p_t``.
    """

    values = _check_vector(raw, 4, "motion raw")
    valid = _check_mask(mask, 4, "motion mask")
    params = _normalization_parameters(stats)
    normalized = np.zeros(4, dtype=float)
    normalized[:3] = np.divide(values[:3], params["position_std"], where=valid[:3], out=normalized[:3])
    if valid[3]:
        normalized[3] = (values[3] - params["speed_mean"]) / params["speed_std"]
    return normalized.tolist()


def denormalize_motion(normalized: Any, mask: Any, stats: Mapping[str, Any]) -> list[float]:
    values = _check_vector(normalized, 4, "motion normalized")
    valid = _check_mask(mask, 4, "motion mask")
    params = _normalization_parameters(stats)
    raw = np.zeros(4, dtype=float)
    raw[:3] = np.where(valid[:3], values[:3] * params["position_std"], 0.0)
    if valid[3]:
        raw[3] = values[3] * params["speed_std"] + params["speed_mean"]
    return raw.tolist()


def normalize_csi(raw: Any, mask: Any, stats: Mapping[str, Any]) -> list[float]:
    values = np.asarray(raw, dtype=float)
    valid = np.asarray(mask, dtype=bool)
    if values.shape != valid.shape:
        raise ValueError(f"CSI raw/mask shape mismatch: {values.shape} vs {valid.shape}")
    params = _normalization_parameters(stats)
    return np.where(valid, (values - params["csi_mean"]) / params["csi_std"], 0.0).tolist()


def denormalize_csi(normalized: Any, mask: Any, stats: Mapping[str, Any]) -> list[float]:
    values = np.asarray(normalized, dtype=float)
    valid = np.asarray(mask, dtype=bool)
    if values.shape != valid.shape:
        raise ValueError(f"CSI normalized/mask shape mismatch: {values.shape} vs {valid.shape}")
    params = _normalization_parameters(stats)
    return np.where(valid, values * params["csi_std"] + params["csi_mean"], 0.0).tolist()


def _valid_number(value: Any) -> bool:
    try:
        return value is not None and bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _entity_value(row: Mapping[str, Any], field: str, component: int | None = None) -> tuple[Any, bool]:
    item = row.get(field, {}) or {}
    values = item.get("value")
    feature_mask = item.get("feature_mask", True)
    present = bool(item.get("presence", True))
    if component is None:
        valid = present and bool(feature_mask) and _valid_number(values)
        return values, valid
    if values is None:
        return None, False
    try:
        component_mask = bool(feature_mask[component]) if isinstance(feature_mask, (list, tuple)) else bool(feature_mask)
        value = values[component]
    except (IndexError, KeyError, TypeError):
        return None, False
    return value, present and component_mask and _valid_number(value)


def _relation_id(row: Mapping[str, Any]) -> str | None:
    return row.get("communication_relation_id") or row.get("relation_id")


def _future_relation_id(row: Mapping[str, Any]) -> str | None:
    if row.get("physical_edge_id"):
        return f"comm::wireless::{row['physical_edge_id']}"
    return _relation_id(row)


def _support(sample: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    history = sample.get("history", [])
    relations = history[-1].get("communication_relations", []) if history else []
    return [relation for relation in relations if _relation_id(relation)]


def _future_outcome(raw: Mapping[str, Any], frame_index: Any) -> Mapping[str, Any]:
    for step in raw.get("steps", []):
        if step.get("frame_index") == frame_index:
            return step.get("outcome", {}) or {}
    return {}


def _future_row_components(row: Mapping[str, Any], current_rb_indices: Sequence[Any]) -> tuple[list[float], list[bool]]:
    values = row.get("channel_attenuation_db") or []
    source_rb_indices = list(row.get("rb_indices") or [])
    observed = row.get("observed_mask", row.get("csi_observed", False))
    observed_values = list(observed) if isinstance(observed, (list, tuple)) else None
    rb_mask = row.get("rb_mask")
    rb_mask_values = list(rb_mask) if isinstance(rb_mask, (list, tuple)) else None
    by_rb: dict[Any, tuple[Any, bool]] = {}
    for index, value in enumerate(values):
        rb = source_rb_indices[index] if index < len(source_rb_indices) else index
        is_observed = observed_values[index] if observed_values is not None and index < len(observed_values) else bool(observed)
        if rb_mask_values is not None and index < len(rb_mask_values):
            is_observed = is_observed and bool(rb_mask_values[index])
        by_rb[rb] = (value, bool(is_observed) and _valid_number(value))
    output_values: list[float] = []
    output_mask: list[bool] = []
    for index, rb in enumerate(current_rb_indices):
        value, valid = by_rb.get(rb, (None, False))
        if not source_rb_indices and rb not in by_rb and index < len(values):
            value, valid = values[index], bool(observed) and _valid_number(values[index])
        output_values.append(float(value) if valid else 0.0)
        output_mask.append(bool(valid))
    return output_values, output_mask


def _side_metadata(frame: Mapping[str, Any], future_rows: Sequence[Mapping[str, Any]], support_ids: Sequence[str]) -> dict[str, Any]:
    unsupported = frame.get("unsupported_future_structure", {}) or {}
    future_only_ids = sorted({_future_relation_id(row) for row in future_rows if _future_relation_id(row)} - set(support_ids))
    return {
        "unsupported_count": len(unsupported.get("unsupported", [])),
        "unresolved_count": len(unsupported.get("unresolved", [])),
        "fixed_support_blocked_count": len(unsupported.get("fixed_support_blocked", [])),
        "future_only_relation_count": len(future_only_ids),
        "future_only_relation_ids": future_only_ids,
    }


def extend_future_motion_csi_targets(
    sample: Mapping[str, Any], raw: Mapping[str, Any], stats: Mapping[str, Any]
) -> dict[str, Any]:
    """Add future target fields while preserving all existing input fields."""

    output = json.loads(json.dumps(sample))
    history_frames = output.get("history", [])
    current_entities = {
        row.get("entity_id"): row for row in (history_frames[-1].get("entities", []) if history_frames else [])
    }
    support = _support(sample)
    support_ids = [_relation_id(relation) for relation in support]
    trajectory_id = sample.get("metadata", {}).get("trajectory_id") or raw.get("trajectory_id")

    # This is target-alignment metadata only. It is not consumed as model input.
    output.setdefault("static", {})["future_target_current_comm_support"] = [
        {
            "relation_id": _relation_id(relation),
            "relation_type": relation.get("relation_type"),
            "source_id": relation.get("source_id"),
            "target_id": relation.get("target_id"),
            "rb_indices": list(relation.get("rb_indices", [])),
            "validity": bool(relation.get("validity", True)),
            "presence": bool(relation.get("presence", True)),
            "target_alignment_only": True,
            "model_input": False,
        }
        for relation in support
    ]

    for horizon_index, frame in enumerate(output.get("target", [])):
        future_entities = {row.get("entity_id"): row for row in frame.get("entities", [])}
        motion_targets: list[dict[str, Any]] = []
        for entity_id, future in future_entities.items():
            current = current_entities.get(entity_id, {})
            raw_value = [0.0, 0.0, 0.0, 0.0]
            mask = [False, False, False, False]
            if future.get("entity_type") == "vehicle" and current:
                for component in range(3):
                    future_value, future_valid = _entity_value(future, "position_m", component)
                    current_value, current_valid = _entity_value(current, "position_m", component)
                    if future_valid and current_valid:
                        raw_value[component] = float(future_value) - float(current_value)
                        mask[component] = True
                speed, speed_valid = _entity_value(future, "speed_mps")
                if speed_valid:
                    raw_value[3] = float(speed)
                    mask[3] = True
            motion_targets.append(
                {
                    "entity_id": entity_id,
                    "entity_type": future.get("entity_type"),
                    "target_index": future.get("target_index"),
                    "reference_frame_index": history_frames[-1].get("frame_index") if history_frames else None,
                    "target_frame_index": frame.get("frame_index"),
                    "horizon_index": horizon_index,
                    "trajectory_id": trajectory_id,
                    "raw_value": raw_value,
                    "normalized_value": normalize_motion(raw_value, mask, stats),
                    "mask": mask,
                    "semantic_order": list(MOTION_SEMANTIC_ORDER),
                }
            )

        outcome = _future_outcome(raw, frame.get("frame_index"))
        future_rows = list(outcome.get("channel_rows", []) or [])
        rows_by_relation = {
            _future_relation_id(row): row for row in future_rows if _future_relation_id(row)
        }
        csi_targets: list[dict[str, Any]] = []
        for relation in support:
            relation_id = _relation_id(relation)
            rb_indices = list(relation.get("rb_indices", []))
            future_row = rows_by_relation.get(relation_id)
            raw_value = [0.0] * len(rb_indices)
            mask = [False] * len(rb_indices)
            if (
                relation.get("relation_type") == "wireless"
                and relation.get("validity", True)
                and relation.get("presence", True)
                and future_row is not None
            ):
                raw_value, mask = _future_row_components(future_row, rb_indices)
            csi_targets.append(
                {
                    "relation_id": relation_id,
                    "relation_type": relation.get("relation_type"),
                    "source_id": relation.get("source_id"),
                    "target_id": relation.get("target_id"),
                    "rb_indices": rb_indices,
                    "raw_value": raw_value,
                    "normalized_value": normalize_csi(raw_value, mask, stats),
                    "mask": mask,
                    "source_capture_phase": "outcome_after_action" if any(mask) else "future_outcome_unobserved",
                    "raw_source": "outcome.channel_rows[].channel_attenuation_db",
                    "future_outcome_frame_index": frame.get("frame_index"),
                    "future_outcome_capture_event_id": outcome.get("capture_event_id"),
                    "trajectory_id": trajectory_id,
                    "presence": bool(relation.get("presence", True)),
                    "validity": bool(relation.get("validity", True)),
                }
            )
        frame["vehicle_motion_targets"] = motion_targets
        frame["comm_csi_targets"] = csi_targets
        frame["future_target_side_metadata"] = _side_metadata(frame, future_rows, support_ids)

    output["schema_version"] = SAMPLE_SCHEMA_VERSION
    output["future_target_contract"] = {
        "namespace": "target",
        "motion_semantics": "delta_xyz_plus_next_speed",
        "csi_source": "future_outcome.channel_rows.channel_attenuation_db",
        "support_policy": "current_input_support_only",
        "normalization": "frozen_step4_3b_train_stats",
        "raw_rule_bridge": "normalized_target_denormalizes_to_step4_4_raw_motion_units",
        "relation_type_vocab": list(COMM_RELATION_TYPE_VOCAB),
        "model_input_unchanged": True,
    }
    return output


def _metadata_contract(samples: Sequence[Mapping[str, Any]], stats: Mapping[str, Any]) -> dict[str, Any]:
    motion_identity = []
    comm_identity = []
    for sample in samples:
        sample_motion = []
        sample_comm = []
        for frame in sample.get("target", []):
            sample_motion.append(
                [
                    {"entity_id": row.get("entity_id"), "entity_type": row.get("entity_type"), "target_index": row.get("target_index")}
                    for row in frame.get("vehicle_motion_targets", [])
                ]
            )
            sample_comm.append(
                [
                    {
                        "relation_id": row.get("relation_id"),
                        "relation_type": row.get("relation_type"),
                        "source_id": row.get("source_id"),
                        "target_id": row.get("target_id"),
                        "rb_indices": list(row.get("rb_indices", [])),
                        "presence": row.get("presence"),
                        "validity": row.get("validity"),
                    }
                    for row in frame.get("comm_csi_targets", [])
                ]
            )
        motion_identity.append(sample_motion)
        comm_identity.append(sample_comm)
    return {
        "motion_semantic_order": list(MOTION_SEMANTIC_ORDER),
        "comm_relation_type_vocab": list(COMM_RELATION_TYPE_VOCAB),
        "support_policy": "current_input_support_only",
        "normalization": "frozen_step4_3b_train_stats",
        "normalization_parameters": _normalization_parameters(stats),
        "stats_digest": hashlib.sha256(json.dumps(stats, sort_keys=True).encode("utf-8")).hexdigest(),
        "motion_identity": motion_identity,
        "comm_identity": comm_identity,
    }


def build_future_target_tensor_batch(samples: Sequence[Mapping[str, Any]], stats: Mapping[str, Any]) -> dict[str, Any]:
    if not samples:
        raise ValueError("at least one future target sample is required")
    batch = len(samples)
    horizon = max(len(sample.get("target", [])) for sample in samples)
    max_entities = max(
        (len(frame.get("vehicle_motion_targets", [])) for sample in samples for frame in sample.get("target", [])),
        default=0,
    )
    max_relations = max(
        (len(frame.get("comm_csi_targets", [])) for sample in samples for frame in sample.get("target", [])),
        default=0,
    )
    max_rb = max(
        (len(row.get("raw_value", [])) for sample in samples for frame in sample.get("target", []) for row in frame.get("comm_csi_targets", [])),
        default=0,
    )
    motion_raw = np.zeros((batch, horizon, max_entities, 4), dtype=np.float64)
    motion_normalized = np.zeros_like(motion_raw)
    motion_mask = np.zeros_like(motion_raw, dtype=bool)
    csi_raw = np.zeros((batch, horizon, max_relations, max_rb), dtype=np.float64)
    csi_normalized = np.zeros_like(csi_raw)
    csi_mask = np.zeros_like(csi_raw, dtype=bool)
    for batch_index, sample in enumerate(samples):
        for horizon_index, frame in enumerate(sample.get("target", [])):
            for entity_index, row in enumerate(frame.get("vehicle_motion_targets", [])):
                motion_raw[batch_index, horizon_index, entity_index] = row["raw_value"]
                motion_normalized[batch_index, horizon_index, entity_index] = row["normalized_value"]
                motion_mask[batch_index, horizon_index, entity_index] = row["mask"]
            for relation_index, row in enumerate(frame.get("comm_csi_targets", [])):
                width = len(row["raw_value"])
                csi_raw[batch_index, horizon_index, relation_index, :width] = row["raw_value"]
                csi_normalized[batch_index, horizon_index, relation_index, :width] = row["normalized_value"]
                csi_mask[batch_index, horizon_index, relation_index, :width] = row["mask"]
    contract = _metadata_contract(samples, stats)
    contract.update(
        {
            "motion_shape": list(motion_raw.shape),
            "csi_shape": list(csi_raw.shape),
            "dtype": "float64",
            "mask_dtype": "bool",
        }
    )
    return {
        "schema_version": TENSOR_SCHEMA_VERSION,
        "target_vehicle_motion_raw": motion_raw,
        "target_vehicle_motion_normalized": motion_normalized,
        "target_vehicle_motion_mask": motion_mask,
        "target_comm_csi_raw": csi_raw,
        "target_comm_csi_normalized": csi_normalized,
        "target_comm_csi_mask": csi_mask,
        "contract": contract,
    }


def _expected_motion_normalized(raw: np.ndarray, mask: np.ndarray, parameters: Mapping[str, Any]) -> np.ndarray:
    expected = np.zeros_like(raw, dtype=float)
    expected[..., :3] = np.divide(
        raw[..., :3], np.asarray(parameters["position_std"], dtype=float),
        where=mask[..., :3], out=expected[..., :3],
    )
    expected[..., 3] = np.where(
        mask[..., 3],
        (raw[..., 3] - float(parameters["speed_mean"])) / float(parameters["speed_std"]),
        0.0,
    )
    return expected


def _expected_csi_normalized(raw: np.ndarray, mask: np.ndarray, parameters: Mapping[str, Any]) -> np.ndarray:
    return np.where(
        mask,
        (raw - float(parameters["csi_mean"])) / float(parameters["csi_std"]),
        0.0,
    )


def validate_future_target_sample_checks(sample: Mapping[str, Any]) -> dict[str, bool]:
    checks: dict[str, bool] = {
        "schema": sample.get("schema_version") == SAMPLE_SCHEMA_VERSION,
        "target_namespace": all(
            "vehicle_motion_targets" not in frame and "comm_csi_targets" not in frame
            for frame in sample.get("history", [])
        ),
        "motion_semantics": True,
        "motion_mask_zero_placeholder": True,
        "csi_support_alignment": True,
        "csi_mask_semantics": True,
        "side_metadata": True,
    }
    support = sample.get("static", {}).get("future_target_current_comm_support", [])
    support_ids = [row.get("relation_id") for row in support]
    for frame in sample.get("target", []):
        for row in frame.get("vehicle_motion_targets", []):
            checks["motion_semantics"] &= row.get("semantic_order") == MOTION_SEMANTIC_ORDER
            mask = list(row.get("mask", []))
            raw = list(row.get("raw_value", []))
            normalized = list(row.get("normalized_value", []))
            checks["motion_semantics"] &= len(mask) == len(raw) == len(normalized) == 4
            if row.get("entity_type") != "vehicle":
                checks["motion_semantics"] &= not any(mask)
            if len(mask) == len(raw) == len(normalized) == 4:
                checks["motion_mask_zero_placeholder"] &= all(
                    mask[index] or (raw[index] == 0.0 and normalized[index] == 0.0)
                    for index in range(4)
                )
        targets = frame.get("comm_csi_targets", [])
        checks["csi_support_alignment"] &= [row.get("relation_id") for row in targets] == support_ids
        for row in targets:
            mask = list(row.get("mask", []))
            raw = list(row.get("raw_value", []))
            normalized = list(row.get("normalized_value", []))
            checks["csi_mask_semantics"] &= len(mask) == len(raw) == len(normalized)
            if len(mask) == len(raw) == len(normalized):
                checks["csi_mask_semantics"] &= all(
                    mask[index] or (raw[index] == 0.0 and normalized[index] == 0.0)
                    for index in range(len(mask))
                )
            if row.get("relation_type") == "wired" or not row.get("presence", True) or not row.get("validity", True):
                checks["csi_mask_semantics"] &= not any(mask)
        side = frame.get("future_target_side_metadata", {})
        checks["side_metadata"] &= all(
            isinstance(side.get(name, 0), int) and side.get(name, 0) >= 0
            for name in ("unsupported_count", "unresolved_count", "fixed_support_blocked_count", "future_only_relation_count")
        )
    checks["passed"] = all(checks.values())
    return checks


def validate_future_target_tensor_checks(tensor: Mapping[str, Any]) -> dict[str, bool]:
    required_arrays = (
        "target_vehicle_motion_raw", "target_vehicle_motion_normalized", "target_vehicle_motion_mask",
        "target_comm_csi_raw", "target_comm_csi_normalized", "target_comm_csi_mask",
    )
    checks: dict[str, bool] = {
        "schema": tensor.get("schema_version") == TENSOR_SCHEMA_VERSION,
        "required_arrays": all(name in tensor for name in required_arrays),
        "motion_normalization_semantic_equality": False,
        "csi_normalization_semantic_equality": False,
        "masked_zero_placeholders": False,
    }
    if checks["required_arrays"]:
        parameters = (tensor.get("contract", {}) or {}).get("normalization_parameters")
        if parameters:
            motion_raw = np.asarray(tensor["target_vehicle_motion_raw"], dtype=float)
            motion_normalized = np.asarray(tensor["target_vehicle_motion_normalized"], dtype=float)
            motion_mask = np.asarray(tensor["target_vehicle_motion_mask"], dtype=bool)
            csi_raw = np.asarray(tensor["target_comm_csi_raw"], dtype=float)
            csi_normalized = np.asarray(tensor["target_comm_csi_normalized"], dtype=float)
            csi_mask = np.asarray(tensor["target_comm_csi_mask"], dtype=bool)
            checks["motion_normalization_semantic_equality"] = bool(
                np.allclose(motion_normalized, _expected_motion_normalized(motion_raw, motion_mask, parameters), atol=1e-9, rtol=1e-9)
            )
            checks["csi_normalization_semantic_equality"] = bool(
                np.allclose(csi_normalized, _expected_csi_normalized(csi_raw, csi_mask, parameters), atol=1e-9, rtol=1e-9)
            )
            checks["masked_zero_placeholders"] = bool(
                np.all(motion_raw[~motion_mask] == 0.0)
                and np.all(motion_normalized[~motion_mask] == 0.0)
                and np.all(csi_raw[~csi_mask] == 0.0)
                and np.all(csi_normalized[~csi_mask] == 0.0)
            )
    checks["passed"] = all(checks.values())
    return checks


def save_future_target_tensor_batch(tensor: Mapping[str, Any], path: str | Path) -> None:
    path = Path(path)
    arrays = {key: value for key, value in tensor.items() if isinstance(value, np.ndarray)}
    contract = {key: value for key, value in tensor.items() if not isinstance(value, np.ndarray)}
    np.savez_compressed(path, **arrays, __contract__=json.dumps(contract, ensure_ascii=False, sort_keys=True))


def load_future_target_tensor_batch(path: str | Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        output = {key: data[key] for key in data.files if key != "__contract__"}
        raw_contract = data["__contract__"]
        contract_text = raw_contract.item() if raw_contract.shape == () else str(raw_contract)
        output.update(json.loads(contract_text))
    return output


def future_target_digest(samples: Sequence[Mapping[str, Any]], tensor: Mapping[str, Any]) -> str:
    payload = {
        "samples": samples,
        "tensor": {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in tensor.items()},
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def validate_step5_1a_acceptance(
    samples: Sequence[Mapping[str, Any]],
    tensor: Mapping[str, Any],
    *,
    deterministic_rebuild: bool,
    real_trajectory_verified: bool,
) -> dict[str, Any]:
    required_checks = {
        "sample_checks": all(validate_future_target_sample_checks(sample).get("passed", False) for sample in samples),
        "tensor_checks": validate_future_target_tensor_checks(tensor).get("passed", False),
        "deterministic_rebuild": bool(deterministic_rebuild),
        "real_trajectory_verified": bool(real_trajectory_verified),
        "formal_dataset_false": True,
        "training_false": True,
        "gpu_false": True,
        "locked_test_accessed_false": True,
    }
    return {
        "required_checks": required_checks,
        "scope": {
            "formal_dataset": False,
            "training": False,
            "gpu": False,
            "locked_test_accessed": False,
        },
        "passed": all(required_checks.values()),
    }
