"""Step 3.2 raw-to-dataset batch validation contract.

This module deliberately stays JSON-native and delegates window semantics to
the frozen Step 3.1F model-ready sample contract.
"""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .model_ready_sample_contract_v1 import SCHEMA_VERSION as MODEL_READY_SAMPLE_SCHEMA_VERSION
from .model_ready_sample_contract_v1 import audit_future_action_references, build_sample, validate_sample


SCHEMA_VERSION = "PI-JWM-Step-3.2-Raw-to-Dataset-Batch-v1"
NORMALIZATION_FIELDS = (
    ("entity", "speed_mps"),
    ("entity", "canonical_acceleration_mps2"),
    ("task", "task_size"),
)


@dataclass(frozen=True)
class RawSource:
    path: Path
    split: str


MODEL_READY_SAMPLE_CONTRACT_VERSION = MODEL_READY_SAMPLE_SCHEMA_VERSION
FEATURE_UNITS = {
    "entity.speed_mps": "m/s",
    "entity.canonical_acceleration_mps2": "m/s^2",
    "task.task_size": "AirFogSim data-unit",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    payload = gzip.decompress(path.read_bytes()).decode("utf-8") if path.suffix == ".gz" else path.read_text(encoding="utf-8")
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError(f"raw source must be an object: {path}")
    return value


def _check_raw(raw: Mapping[str, Any], source: RawSource) -> str:
    scope = raw.get("scope", {})
    if scope.get("locked_test") or scope.get("training") or scope.get("gpu"):
        raise ValueError(f"raw source outside development scope: {source.path}")
    decisions = list(raw.get("decisions", []))
    steps = list(raw.get("steps", []))
    if len(decisions) != len(steps) + 1:
        raise ValueError(f"decision/step alignment mismatch: {source.path}")
    frames = [int(row.get("frame_index", -1)) for row in decisions]
    step_frames = [int(row.get("frame_index", -1)) for row in steps]
    if frames != list(range(len(frames))) or step_frames != list(range(len(steps))):
        raise ValueError(f"contiguous frame indices required: {source.path}")
    slot_duration = raw.get("environment", {}).get("slot_duration_s")
    if slot_duration is None or float(slot_duration) <= 0:
        raise ValueError(f"slot_duration_s is required: {source.path}")
    slot_duration = float(slot_duration)
    times = [float(row["simulation_time_s"]) for row in decisions]
    tolerance = max(1e-9, slot_duration * 1e-6)
    if any(abs((times[i + 1] - times[i]) - slot_duration) > tolerance for i in range(len(times) - 1)):
        raise ValueError(f"time grid gap detected: {source.path}")
    for index, step in enumerate(steps):
        execution = step.get("execution", {})
        start = execution.get("start_time_s")
        end = execution.get("end_time_s")
        outcome_time = step.get("outcome", {}).get("simulation_time_s")
        if start is None or end is None or outcome_time is None:
            raise ValueError(f"step timing metadata missing: {source.path}")
        if abs(float(start) - times[index]) > tolerance or abs(float(end) - float(start) - slot_duration) > tolerance or abs(float(outcome_time) - float(end)) > tolerance:
            raise ValueError(f"step timing does not align to time grid: {source.path}")
    ids = {str(row.get("trajectory_id", "")) for row in decisions}
    if len(ids) != 1 or "" in ids:
        raise ValueError(f"one non-empty trajectory_id required: {source.path}")
    trajectory_id = next(iter(ids))
    if source.split not in {"dev_train", "dev_validation"}:
        raise ValueError(f"unsupported development split: {source.split}")
    return trajectory_id


def _feature_value(row: Mapping[str, Any], field: str) -> tuple[Any, bool]:
    if row.get("presence") is False:
        return None, False
    value = row.get(field)
    object_mask = row.get("feature_mask", {})
    if isinstance(object_mask, Mapping) and field in object_mask and not bool(object_mask[field]):
        if isinstance(value, Mapping):
            return value.get("value"), False
        return value, False
    if isinstance(value, Mapping):
        return value.get("value"), bool(value.get("presence")) and bool(value.get("feature_mask")) and value.get("value") is not None
    return value, value is not None


def _iter_features(sample: Mapping[str, Any]):
    for frame in sample.get("history", []):
        for row in frame.get("entities", []):
            for field in ("speed_mps", "canonical_acceleration_mps2"):
                yield "entity", field, row
        for row in frame.get("tasks", []):
            yield "task", "task_size", row


def fit_train_normalization_stats(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    sums = {f"{kind}.{field}": [0.0, 0.0, 0] for kind, field in NORMALIZATION_FIELDS}
    for sample in samples:
        if sample.get("metadata", {}).get("split") != "dev_train":
            continue
        for kind, field, row in _iter_features(sample):
            value, valid = _feature_value(row, field)
            if valid:
                key = f"{kind}.{field}"
                sums[key][0] += float(value)
                sums[key][1] += float(value) ** 2
                sums[key][2] += 1
    features = {}
    for key, (total, total_sq, count) in sums.items():
        mean = total / count if count else 0.0
        variance = max(total_sq / count - mean * mean, 0.0) if count else 0.0
        features[key] = {
            "mean": mean,
            "std": variance ** 0.5 if variance > 1e-12 else 1.0,
            "count": count,
            "zero_variance_handling": "scale=1.0" if variance <= 1e-12 else "population_std",
            "source_field": field if (field := key.split(".", 1)[1]) else key,
            "unit": FEATURE_UNITS[key],
            "mask_policy": "presence=true AND feature_mask=true AND value!=null; train split only",
        }
    return {
        "schema_version": "PI-JWM-Step-3.2-Train-Only-Normalization-v1",
        "source_split": "dev_train",
        "features": features,
        "units": dict(FEATURE_UNITS),
    }


def apply_normalization(samples: Sequence[Mapping[str, Any]], stats: Mapping[str, Any]) -> list[dict[str, Any]]:
    output = copy.deepcopy(list(samples))
    features = stats.get("features", {})
    for sample in output:
        for kind, field, row in _iter_features(sample):
            key = f"{kind}.{field}"
            value, valid = _feature_value(row, field)
            feature = features[key]
            row[field]["normalized_value"] = (
                (float(value) - float(feature["mean"])) / float(feature["std"])
                if valid else None
            )
            row[field]["normalization_valid"] = valid
    return output


def build_batch(sources: Sequence[RawSource], *, history_steps: int = 2, horizon_steps: int = 2) -> dict[str, Any]:
    if not sources:
        raise ValueError("at least one raw source is required")
    seen: dict[str, Path] = {}
    lineage: dict[tuple[Any, Any], tuple[str, str]] = {}
    samples: list[dict[str, Any]] = []
    provenance = []
    for source in sources:
        path = Path(source.path)
        raw = _load(path)
        trajectory_id = _check_raw(raw, source)
        if trajectory_id in seen:
            raise ValueError(f"duplicate trajectory_id: {trajectory_id}")
        seen[trajectory_id] = path
        environment = raw.get("environment", {})
        seed = environment.get("seed")
        config_hash = environment.get("config_hash")
        lineage_key = (seed, config_hash)
        if lineage_key in lineage and lineage[lineage_key][1] != trajectory_id:
            raise ValueError(f"same run lineage cannot cross trajectories/splits: {trajectory_id}")
        lineage[lineage_key] = (source.split, trajectory_id)
        decisions = list(raw.get("decisions", []))
        steps = list(raw.get("steps", []))
        provenance.append({
            "trajectory_id": trajectory_id,
            "seed": seed,
            "source_path": str(path),
            "source_sha256": _sha256(path),
            "split": source.split,
            "schema_version": raw.get("schema_version"),
            "config_hash": config_hash,
            "decision_frame_range": [int(decisions[0]["frame_index"]), int(decisions[-1]["frame_index"])],
            "step_frame_range": [int(steps[0]["frame_index"]), int(steps[-1]["frame_index"])],
            "decision_time_range_s": [float(decisions[0]["simulation_time_s"]), float(decisions[-1]["simulation_time_s"])],
            "step_time_range_s": [float(steps[0]["execution"]["start_time_s"]), float(steps[-1]["execution"]["end_time_s"])],
            "slot_duration_s": float(environment["slot_duration_s"]),
            "model_ready_sample_contract_version": MODEL_READY_SAMPLE_CONTRACT_VERSION,
            "lineage_key": {"seed": seed, "config_hash": config_hash},
        })
        for anchor in range(history_steps - 1, len(raw.get("steps", [])) - horizon_steps + 1):
            sample = build_sample(raw, anchor_step=anchor)
            checks = validate_sample(sample)
            if not all(checks.values()):
                raise ValueError(f"invalid model-ready sample at {trajectory_id} anchor {anchor}")
            sample["metadata"]["split"] = source.split
            sample["metadata"]["source_split"] = source.split
            sample["metadata"]["sample_id"] = f"{trajectory_id}::anchor-{anchor:04d}"
            samples.append(sample)
    split_ids = {split: {row["trajectory_id"] for row in provenance if row["split"] == split} for split in ("dev_train", "dev_validation")}
    if split_ids["dev_train"] & split_ids["dev_validation"]:
        raise ValueError("trajectory_id crosses train and validation")
    return {
        "schema_version": SCHEMA_VERSION,
        "contract": {"history_steps": history_steps, "horizon_steps": horizon_steps, "split_policy": "trajectory_level_before_window_construction", "input_index_policy": "history_causal_observable_object_union"},
        "samples": samples,
        "provenance": provenance,
        "normalization_units": dict(FEATURE_UNITS),
        "checks": {
            "trajectory_ids_unique": len({row["trajectory_id"] for row in provenance}) == len(provenance),
            "no_trajectory_cross_split": not bool(split_ids["dev_train"] & split_ids["dev_validation"]),
            "source_sha256_present": all(bool(row["source_sha256"]) for row in provenance),
            "seed_source_lineage_auditable": all("seed" in row and "lineage_key" in row for row in provenance),
            "causal_windows_only": all(sample["metadata"]["future_action_frame_indices"][0] > sample["metadata"]["anchor_decision_frame"] - 1 for sample in samples),
        },
        "scope": {"locked_test": False, "training": False, "gpu": False, "formal_dataset": False},
    }


def audit_batch_future_action_references(sources: Sequence[RawSource], *, bundle: Mapping[str, Any], history_steps: int = 2, horizon_steps: int = 2) -> dict[str, Any]:
    by_family: dict[str, dict[str, int]] = {}
    by_object_kind: dict[str, dict[str, int]] = {}
    total_candidate = constructed = unresolved_windows = unresolved_count = 0
    for source in sources:
        raw = _load(Path(source.path))
        audit = audit_future_action_references(raw, history_steps=history_steps, horizon_steps=horizon_steps)
        total_candidate += int(audit["constructible_window_count"])
        unresolved_windows += int(audit["affected_window_count"])
        unresolved_count += int(audit["unresolved_future_reference_count"])
        for family, values in audit["by_action_family"].items():
            target = by_family.setdefault(family, {"windows": 0, "references": 0})
            target["windows"] += int(values["windows"])
            target["references"] += int(values["references"])
        for kind, values in audit["by_object_kind"].items():
            target = by_object_kind.setdefault(kind, {"windows": 0, "references": 0})
            target["windows"] += int(values["windows"])
            target["references"] += int(values["references"])
    constructed = len(bundle.get("samples", []))
    return {
        "schema_version": "PI-JWM-Step-3.2-Future-Reference-Audit-v1",
        "total_candidate_windows": total_candidate,
        "successfully_constructed_windows": constructed,
        "unresolved_reference_windows": unresolved_windows,
        "unresolved_reference_count": unresolved_count,
        "affected_window_rate": (unresolved_windows / total_candidate) if total_candidate else None,
        "by_action_family": by_family,
        "by_object_kind": by_object_kind,
        "observation_only": True,
        "decision": "RESEARCHER_DECISION_REQUIRED_IF_NONZERO",
    }


def evaluate_validation_checks(bundle: Mapping[str, Any], stats: Mapping[str, Any], *, deterministic_rebuild: bool) -> dict[str, bool]:
    provenance = list(bundle.get("provenance", []))
    train_ids = {row.get("trajectory_id") for row in provenance if row.get("split") == "dev_train"}
    validation_ids = {row.get("trajectory_id") for row in provenance if row.get("split") == "dev_validation"}
    scope = bundle.get("scope", {})
    checks = {
        "trajectory_level_split": bool(provenance) and len({row.get("trajectory_id") for row in provenance}) == len(provenance),
        "no_trajectory_cross_split": not bool(train_ids & validation_ids),
        "causal_windows_only": all(
            sample["metadata"]["history_frame_indices"][-1] == sample["metadata"]["anchor_decision_frame"]
            and sample["metadata"]["future_action_frame_indices"][0] == sample["metadata"]["anchor_decision_frame"]
            for sample in bundle.get("samples", [])
        ),
        "train_only_normalization_fit": stats.get("source_split") == "dev_train" and all(
            feature.get("mask_policy") == "presence=true AND feature_mask=true AND value!=null; train split only"
            for feature in stats.get("features", {}).values()
        ),
        "source_sha256_traceable": all(len(str(row.get("source_sha256", ""))) == 64 for row in provenance),
        "seed_source_lineage_auditable": all("seed" in row and "lineage_key" in row for row in provenance),
        "deterministic_rebuild": bool(deterministic_rebuild),
        "scope_locked_test_false": scope.get("locked_test") is False,
        "scope_training_false": scope.get("training") is False,
        "scope_gpu_false": scope.get("gpu") is False,
        "scope_formal_dataset_false": scope.get("formal_dataset") is False,
    }
    return checks


def collate_samples(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("cannot collate an empty batch")
    splits = {str(sample.get("metadata", {}).get("split")) for sample in samples}
    if len(splits) != 1:
        raise ValueError("a batch cannot mix splits")
    return {"schema_version": SCHEMA_VERSION, "split": next(iter(splits)), "sample_ids": [sample["metadata"]["sample_id"] for sample in samples], "samples": [copy.deepcopy(dict(sample)) for sample in samples]}


def write_batch_bundle(bundle: Mapping[str, Any], stats: Mapping[str, Any], normalized: Sequence[Mapping[str, Any]], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    payloads = {"batch.json": bundle, "train_normalization_stats.json": stats, "normalized_samples.json": list(normalized)}
    for name, payload in payloads.items():
        (output / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {"schema_version": SCHEMA_VERSION, "files": {name: _sha256(output / name) for name in payloads}, "scope": bundle.get("scope", {})}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_batch_bundle(output: Path) -> dict[str, Any]:
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest.get("files", {}).items():
        if _sha256(output / name) != expected:
            raise ValueError(f"artifact hash mismatch: {name}")
    return {"bundle": json.loads((output / "batch.json").read_text(encoding="utf-8")), "stats": json.loads((output / "train_normalization_stats.json").read_text(encoding="utf-8")), "normalized": json.loads((output / "normalized_samples.json").read_text(encoding="utf-8"))}


__all__ = ["MODEL_READY_SAMPLE_CONTRACT_VERSION", "RawSource", "apply_normalization", "audit_batch_future_action_references", "build_batch", "collate_samples", "evaluate_validation_checks", "fit_train_normalization_stats", "load_batch_bundle", "write_batch_bundle"]
