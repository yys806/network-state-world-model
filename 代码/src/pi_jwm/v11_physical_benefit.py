"""Leakage-safe physical-benefit supervision for the PI-JWM v11 selector."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np


SUPPORTED_PHYSICAL_FAMILIES = {
    "default": "identity_control",
    "rb_count": "rb_repair",
    "mixed_offload_rb": "offload_rb",
    "cpu_scale": "compute_cpu",
    "return_route": "return_route",
}

COMMON_DESCRIPTOR_NAMES = (
    "rb_total_sum",
    "rb_action_count",
    "cpu_total_sum",
    "cpu_action_count",
    "offload_action_count",
    "return_action_count",
    "intervention_start_step",
    "pattern_persistent",
    "pattern_decayed",
    "family_rb",
    "family_offload",
    "family_compute",
    "family_return",
    "family_control",
)


def normalize_physical_family(name: str) -> str | None:
    """Map simulator candidate families to the shared selector vocabulary."""

    return SUPPORTED_PHYSICAL_FAMILIES.get(str(name).strip().lower())


def _time_key(value: Any, tolerance: float) -> float:
    tol = float(tolerance)
    if not math.isfinite(tol) or tol <= 0.0:
        raise ValueError("time alignment tolerance must be finite and positive")
    digits = max(0, int(math.ceil(-math.log10(tol))))
    return round(float(value), digits)


def align_decision_points(
    points: Sequence[Mapping[str, Any]],
    sample_index_rows: Sequence[Mapping[str, Any]],
    tolerance: float = 1e-8,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Attach sample IDs only when seed and input-end time match exactly."""

    by_key: dict[tuple[int, float], int] = {}
    for source in sample_index_rows:
        key = (
            int(source["seed"]),
            _time_key(source["input_end_time"], tolerance),
        )
        if key in by_key:
            raise ValueError(f"duplicate sample time for seed={key[0]} time={key[1]}")
        by_key[key] = int(source["sample_id"])

    aligned: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for source in points:
        row = dict(source)
        key = (
            int(row["seed"]),
            _time_key(row["decision_time"], tolerance),
        )
        sample_id = by_key.get(key)
        if sample_id is None:
            rejected.append({**row, "reason": "no_exact_input_end_time"})
        else:
            aligned.append({**row, "sample_id": sample_id})
    return aligned, rejected


def physical_action_descriptor(
    row: Mapping[str, Any],
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Build the common action-only descriptor from an AirFogSim result row."""

    family = normalize_physical_family(str(row.get("action_family", "")))
    if family is None:
        raise ValueError(
            f"unsupported physical action family: {row.get('action_family', '')}"
        )
    pattern = str(row.get("temporal_pattern", "persistent")).strip().lower()
    if pattern not in {"persistent", "decayed"}:
        raise ValueError(f"unsupported temporal pattern: {pattern}")

    values = np.asarray(
        [
            float(row.get("rb_total", row.get("total_rb", 0.0))),
            float(row.get("num_rb_tasks", 0.0)),
            float(row.get("cpu_total", row.get("total_cpu", 0.0))),
            float(row.get("num_cpu_overrides", 0.0)),
            float(row.get("num_offload_overrides", 0.0)),
            float(row.get("num_return_route_overrides", 0.0)),
            float(row.get("intervention_start_step", 1.0)),
            float(pattern == "persistent"),
            float(pattern == "decayed"),
            float(family == "rb_repair"),
            float(family == "offload_rb"),
            float(family == "compute_cpu"),
            float(family == "return_route"),
            float(family == "identity_control"),
        ],
        dtype=np.float32,
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("physical action descriptor must be finite")
    return values, COMMON_DESCRIPTOR_NAMES


def selector_action_descriptors(batch: Any) -> tuple[np.ndarray, tuple[str, ...]]:
    """Project selector candidates onto the shared action-only descriptor."""

    source_names = tuple(str(name) for name in batch.feature_names)
    source = np.asarray(batch.candidate_features, dtype=np.float32)
    required = (
        "rb_total_sum",
        "rb_action_count",
        "cpu_total_sum",
        "cpu_action_count",
        "offload_action_count",
        "return_action_count",
        "action_family_identity",
        "action_family_rb",
        "action_family_offload",
        "action_family_compute",
        "action_family_return",
        "action_family_historical",
    )
    missing = [name for name in required if name not in source_names]
    if missing:
        raise ValueError(f"selector batch is missing common descriptor fields: {missing}")

    def field(name: str) -> np.ndarray:
        return source[:, :, source_names.index(name)]

    candidate_names = tuple(str(name).lower() for name in batch.candidate_names)
    if not candidate_names:
        candidate_names = tuple("" for _ in range(source.shape[1]))
    decayed = np.asarray(
        ["decayed" in name for name in candidate_names], dtype=np.float32
    )
    decayed = np.broadcast_to(decayed[None, :], source.shape[:2])
    persistent = 1.0 - decayed
    family_rb = np.maximum(field("action_family_rb"), field("action_family_historical"))

    result = np.stack(
        (
            field("rb_total_sum"),
            field("rb_action_count"),
            field("cpu_total_sum"),
            field("cpu_action_count"),
            field("offload_action_count"),
            field("return_action_count"),
            np.ones(source.shape[:2], dtype=np.float32),
            persistent,
            decayed,
            family_rb,
            field("action_family_offload"),
            field("action_family_compute"),
            field("action_family_return"),
            field("action_family_identity"),
        ),
        axis=2,
    ).astype(np.float32)
    if not np.all(np.isfinite(result)):
        raise ValueError("selector action descriptors must be finite")
    return result, COMMON_DESCRIPTOR_NAMES
