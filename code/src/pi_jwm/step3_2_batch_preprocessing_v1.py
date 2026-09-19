"""Step 3.2 raw-to-dataset batch validation contract.

This module deliberately stays JSON-native and delegates window semantics to
the frozen Step 3.1F model-ready sample contract.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .model_ready_sample_contract_v1 import build_sample, validate_sample


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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
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
    ids = {str(row.get("trajectory_id", "")) for row in decisions}
    if len(ids) != 1 or "" in ids:
        raise ValueError(f"one non-empty trajectory_id required: {source.path}")
    trajectory_id = next(iter(ids))
    if source.split not in {"dev_train", "dev_validation"}:
        raise ValueError(f"unsupported development split: {source.split}")
    return trajectory_id


def _feature_value(row: Mapping[str, Any], field: str) -> tuple[Any, bool]:
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
            "mask_policy": "presence=true AND feature_mask=true AND value!=null; train split only",
        }
    return {
        "schema_version": "PI-JWM-Step-3.2-Train-Only-Normalization-v1",
        "source_split": "dev_train",
        "features": features,
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
    samples: list[dict[str, Any]] = []
    provenance = []
    for source in sources:
        path = Path(source.path)
        raw = _load(path)
        trajectory_id = _check_raw(raw, source)
        if trajectory_id in seen:
            raise ValueError(f"duplicate trajectory_id: {trajectory_id}")
        seen[trajectory_id] = path
        provenance.append({"path": str(path), "sha256": _sha256(path), "trajectory_id": trajectory_id, "split": source.split, "schema_version": raw.get("schema_version")})
        for anchor in range(history_steps - 1, len(raw.get("steps", [])) - horizon_steps + 1):
            sample = build_sample(raw, anchor_step=anchor)
            checks = validate_sample(sample)
            if not all(checks.values()):
                raise ValueError(f"invalid model-ready sample at {trajectory_id} anchor {anchor}")
            sample["metadata"]["split"] = source.split
            sample["metadata"]["source_split"] = source.split
            sample["metadata"]["sample_id"] = f"{trajectory_id}::anchor-{anchor:04d}"
            samples.append(sample)
    return {
        "schema_version": SCHEMA_VERSION,
        "contract": {"history_steps": history_steps, "horizon_steps": horizon_steps, "split_policy": "trajectory_level_before_window_construction", "input_index_policy": "history_causal_observable_object_union"},
        "samples": samples,
        "provenance": provenance,
        "scope": {"locked_test": False, "training": False, "gpu": False, "formal_dataset": False},
    }


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


__all__ = ["RawSource", "apply_normalization", "build_batch", "collate_samples", "fit_train_normalization_stats", "load_batch_bundle", "write_batch_bundle"]
