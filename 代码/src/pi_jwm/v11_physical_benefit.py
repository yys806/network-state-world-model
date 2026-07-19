"""Leakage-safe physical-benefit supervision for the PI-JWM v11 selector."""

from __future__ import annotations

import math
from dataclasses import dataclass
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


@dataclass(frozen=True)
class PhysicalBenefitTrainingBatch:
    """Sparse paired physical outcomes aligned to observable selector context."""

    features: np.ndarray
    feature_names: tuple[str, ...]
    task_delta: np.ndarray
    energy_delta: np.ndarray
    sample_ids: np.ndarray
    sample_seed: np.ndarray
    group_ids: np.ndarray
    normalized_family: np.ndarray
    rejected_rows: tuple[dict[str, Any], ...] = ()

    def __post_init__(self) -> None:
        features = np.asarray(self.features, dtype=np.float32)
        task = np.asarray(self.task_delta, dtype=np.float32).reshape(-1)
        energy = np.asarray(self.energy_delta, dtype=np.float32).reshape(-1)
        sample_ids = np.asarray(self.sample_ids, dtype=np.int64).reshape(-1)
        seeds = np.asarray(self.sample_seed, dtype=np.int64).reshape(-1)
        groups = np.asarray(self.group_ids).astype(str).reshape(-1)
        families = np.asarray(self.normalized_family).astype(str).reshape(-1)
        size = features.shape[0] if features.ndim == 2 else -1
        if features.ndim != 2 or len(self.feature_names) != features.shape[1]:
            raise ValueError("physical bridge features must be [row,feature] with matching names")
        if any(values.shape[0] != size for values in (task, energy, sample_ids, seeds, groups, families)):
            raise ValueError("physical bridge training fields must share the row dimension")
        if not np.all(np.isfinite(features)) or not np.all(np.isfinite(task)) or not np.all(np.isfinite(energy)):
            raise ValueError("physical bridge features and targets must be finite")
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "task_delta", task)
        object.__setattr__(self, "energy_delta", energy)
        object.__setattr__(self, "sample_ids", sample_ids)
        object.__setattr__(self, "sample_seed", seeds)
        object.__setattr__(self, "group_ids", groups)
        object.__setattr__(self, "normalized_family", families)


def _as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes"}


def build_physical_training_batch(
    physical_rows: Sequence[Mapping[str, Any]],
    selector_sample_ids: np.ndarray,
    selector_sample_seed: np.ndarray,
    selector_batch: Any,
) -> PhysicalBenefitTrainingBatch:
    """Pair candidates with their group default and attach observable context."""

    ids = np.asarray(selector_sample_ids, dtype=np.int64).reshape(-1)
    seeds = np.asarray(selector_sample_seed, dtype=np.int64).reshape(-1)
    context = np.asarray(selector_batch.context, dtype=np.float32)
    if ids.shape != seeds.shape or ids.shape[0] != context.shape[0]:
        raise ValueError("selector sample ids, seeds, and context must align")
    if len(ids) != len(set(int(value) for value in ids)):
        raise ValueError("selector sample ids must be unique")
    context_names = tuple(str(name) for name in selector_batch.context_feature_names)
    if len(context_names) != context.shape[1]:
        raise ValueError("selector context feature names are required for physical alignment")
    sample_lookup = {
        int(sample_id): (index, int(seed))
        for index, (sample_id, seed) in enumerate(zip(ids, seeds))
    }

    grouped: dict[tuple[int, int, float], list[Mapping[str, Any]]] = {}
    for row in physical_rows:
        key = (
            int(row["seed"]),
            int(row["sample_id"]),
            round(float(row["decision_time"]), 8),
        )
        grouped.setdefault(key, []).append(row)

    feature_rows: list[np.ndarray] = []
    task_delta: list[float] = []
    energy_delta: list[float] = []
    output_ids: list[int] = []
    output_seeds: list[int] = []
    group_ids: list[str] = []
    families: list[str] = []
    rejected: list[dict[str, Any]] = []
    for (seed, sample_id, decision_time), rows in sorted(grouped.items()):
        defaults = [row for row in rows if str(row.get("action_family", "")).lower() == "default"]
        if len(defaults) != 1:
            raise ValueError(
                f"physical group {(seed, sample_id, decision_time)} must contain exactly one default"
            )
        if sample_id not in sample_lookup:
            raise ValueError(f"physical sample_id missing from selector cache: {sample_id}")
        context_index, expected_seed = sample_lookup[sample_id]
        if expected_seed != seed:
            raise ValueError(f"physical sample {sample_id} seed differs from selector cache")
        baseline = defaults[0]
        if not _as_bool(baseline.get("action_applied", False)):
            raise ValueError("physical default action must be applied")
        for row in rows:
            if row is baseline:
                continue
            family = normalize_physical_family(str(row.get("action_family", "")))
            if family is None:
                rejected.append(
                    {"candidate_id": str(row.get("candidate_id", "")), "reason": "unsupported_action_family"}
                )
                continue
            if not _as_bool(row.get("action_applied", False)):
                rejected.append(
                    {"candidate_id": str(row.get("candidate_id", "")), "reason": "action_not_applied"}
                )
                continue
            descriptor, descriptor_names = physical_action_descriptor(row)
            feature_rows.append(np.concatenate((context[context_index], descriptor)).astype(np.float32))
            task_delta.append(float(row["task_utility"]) - float(baseline["task_utility"]))
            energy_delta.append(float(row["energy_total"]) - float(baseline["energy_total"]))
            output_ids.append(sample_id)
            output_seeds.append(seed)
            group_ids.append(f"{seed}:{sample_id}:{decision_time:.8f}")
            families.append(family)

    feature_names = context_names + descriptor_names if feature_rows else context_names + COMMON_DESCRIPTOR_NAMES
    features = (
        np.stack(feature_rows).astype(np.float32)
        if feature_rows
        else np.empty((0, len(feature_names)), dtype=np.float32)
    )
    return PhysicalBenefitTrainingBatch(
        features=features,
        feature_names=feature_names,
        task_delta=np.asarray(task_delta, dtype=np.float32),
        energy_delta=np.asarray(energy_delta, dtype=np.float32),
        sample_ids=np.asarray(output_ids, dtype=np.int64),
        sample_seed=np.asarray(output_seeds, dtype=np.int64),
        group_ids=np.asarray(group_ids, dtype=str),
        normalized_family=np.asarray(families, dtype=str),
        rejected_rows=tuple(rejected),
    )


def audit_physical_bridge_protocol(
    feature_names: Sequence[str],
    split_seed_sets: Mapping[str, Sequence[int]],
    matched_test_accessed: bool,
    external_holdout_accessed: bool,
) -> dict[str, Any]:
    """Audit leakage tokens, seed isolation, and locked-split access."""

    forbidden_tokens = ("actual", "future", "oracle", "outcome", "seed")
    names = tuple(str(name) for name in feature_names)
    forbidden = [
        name for name in names if any(token in name.lower() for token in forbidden_tokens)
    ]
    normalized = {
        str(name): set(int(seed) for seed in values)
        for name, values in split_seed_sets.items()
    }
    overlap: set[int] = set()
    split_names = list(normalized)
    overlap_pairs = []
    for left_index, left in enumerate(split_names):
        for right in split_names[left_index + 1 :]:
            shared = normalized[left] & normalized[right]
            if shared:
                overlap.update(shared)
                overlap_pairs.append({"left": left, "right": right, "seeds": sorted(shared)})
    passed = (
        not forbidden
        and not overlap
        and not bool(matched_test_accessed)
        and not bool(external_holdout_accessed)
    )
    return {
        "passed": passed,
        "forbidden_features": forbidden,
        "split_overlap_count": len(overlap),
        "split_overlap_pairs": overlap_pairs,
        "matched_test_accessed": bool(matched_test_accessed),
        "external_holdout_accessed": bool(external_holdout_accessed),
    }
