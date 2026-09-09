"""Build a new non-locked formal tensor with causal node-motion features."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pi_jwm.airfogsim_window_dataset_v2 import fit_training_stats
from pi_jwm.formal_motion_state_v1 import (
    MOTION_FEATURES,
    MOTION_SCHEMA_VERSION,
    derive_causal_node_motion,
)


DATASET_SCHEMA = "PI-JWM-AirFogSim-formal-causal-motion-tensor-v1"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _motion_stats(
    output_root: Path, split_by_seed: dict[int, str]
) -> dict[str, Any]:
    sums = np.zeros(len(MOTION_FEATURES), dtype=np.float64)
    square_sums = np.zeros_like(sums)
    counts = np.zeros(len(MOTION_FEATURES), dtype=np.int64)
    for seed, split in split_by_seed.items():
        if split != "train":
            continue
        path = output_root / f"seed_{seed:03d}" / "trajectory_tensors.npz"
        with np.load(path, allow_pickle=False) as loaded:
            values = loaded["node_motion_state"].astype(np.float64)
            mask = loaded["node_motion_mask"].astype(bool)
            sums += np.where(mask, values, 0.0).sum(axis=(0, 1))
            square_sums += np.where(mask, values * values, 0.0).sum(axis=(0, 1))
            counts += mask.sum(axis=(0, 1))
    denominator = np.maximum(counts, 1)
    mean = sums / denominator
    variance = np.maximum(square_sums / denominator - mean * mean, 0.0)
    scale = np.maximum(np.sqrt(variance), 1e-6)
    return {
        "count": int(counts.max(initial=0)),
        "feature_count": counts.tolist(),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
    }


def _write_manifest(root: Path) -> dict[str, Any]:
    files = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "manifest.json":
            continue
        relative = path.relative_to(root).as_posix()
        files[relative] = {"size_bytes": path.stat().st_size, "sha256": _sha256(path)}
    manifest = {
        "schema_version": "PI-JWM-formal-causal-motion-tensor-manifest-v1",
        "files": files,
    }
    _write_json(root / "manifest.json", manifest)
    return manifest


def build_causal_motion_tensor(
    *, source_root: str | Path, output_root: str | Path
) -> dict[str, Any]:
    source_root, output_root = Path(source_root), Path(output_root)
    if any("locked_test" in str(path).lower() for path in (source_root, output_root)):
        raise ValueError("locked_test path is forbidden")
    if output_root.exists():
        raise FileExistsError(output_root)
    source_validation = _read_json(source_root / "validation_report.json")
    if source_validation.get("formal_tensor_ready") is not True:
        raise ValueError("source formal tensor is not ready")
    source_contract = _read_json(source_root / "tensor_contract.json")
    source_summary = _read_json(source_root / "dataset_summary.json")
    source_manifest_path = source_root / "manifest.json"
    window_path = source_root / "window_index.csv"
    window_rows = _read_csv(window_path)
    if any(row.get("split") == "locked_test" for row in window_rows):
        raise ValueError("source window index contains locked_test")
    slot_seconds_values = {float(row["simulation_interval"]) for row in window_rows}
    if len(slot_seconds_values) != 1:
        raise ValueError("source window index must contain one simulation interval")
    slot_seconds_value = slot_seconds_values.pop()

    output_root.mkdir(parents=True, exist_ok=False)
    shutil.copy2(window_path, output_root / "window_index.csv")
    split_by_seed: dict[int, str] = {}
    seed_summaries = []
    for tensor_path in sorted(source_root.glob("seed_*/trajectory_tensors.npz")):
        seed = int(tensor_path.parent.name.removeprefix("seed_"))
        source_report = _read_json(tensor_path.parent / "tensor_report.json")
        split = str(source_report["split"])
        if split == "locked_test":
            raise ValueError("locked_test tensor is forbidden")
        split_by_seed[seed] = split
        with np.load(tensor_path, allow_pickle=False) as loaded:
            arrays = {name: loaded[name] for name in loaded.files}
        motion = derive_causal_node_motion(
            arrays["node_state"],
            arrays["node_present"],
            slot_seconds=float(arrays["slot_seconds"]),
        )
        arrays["node_state"] = motion["node_state"]
        arrays["node_motion_state"] = motion["node_motion_state"]
        arrays["node_motion_mask"] = motion["node_motion_mask"]
        seed_dir = output_root / tensor_path.parent.name
        seed_dir.mkdir(parents=True)
        np.savez_compressed(seed_dir / "trajectory_tensors.npz", **arrays)
        velocity_valid = arrays["node_motion_mask"][..., :3].all(axis=-1)
        acceleration_valid = arrays["node_motion_mask"][..., 3:].all(axis=-1)
        report = {
            **source_report,
            "schema_version": DATASET_SCHEMA,
            "source_tensor_sha256": _sha256(tensor_path),
            "motion_schema_version": MOTION_SCHEMA_VERSION,
            "array_shapes": {name: list(value.shape) for name, value in arrays.items()},
            "array_dtypes": {name: str(value.dtype) for name, value in arrays.items()},
            "motion": {
                "velocity_valid_count": int(velocity_valid.sum()),
                "acceleration_valid_count": int(acceleration_valid.sum()),
                "nonzero_speed_count": int(
                    (np.abs(arrays["node_state"][..., 3]) > 1e-8).sum()
                ),
                "nonzero_acceleration_count": int(
                    (np.abs(arrays["node_state"][..., 4]) > 1e-8).sum()
                ),
            },
        }
        _write_json(seed_dir / "tensor_report.json", report)
        seed_summaries.append(report["motion"])
    if not seed_summaries:
        raise ValueError("source contains no seed tensors")

    contract = dict(source_contract)
    contract.update(
        {
            "schema_version": DATASET_SCHEMA,
            "motion_features": list(MOTION_FEATURES),
            "node_motion_contract": {
                "schema_version": MOTION_SCHEMA_VERSION,
                "derivative": "causal_backward_difference_v1",
                "source": "node_position_at_or_before_current_time",
                "slot_duration_field": "slot_seconds",
                "future_values_used": False,
            },
            "node_speed_acceleration_semantics": "norms_of_causal_velocity_and_acceleration_vectors_v1",
            "slot_seconds_value": slot_seconds_value,
        }
    )
    _write_json(output_root / "tensor_contract.json", contract)
    stats = fit_training_stats(output_root, split="train")
    stats["features"]["node_motion_state"] = _motion_stats(output_root, split_by_seed)
    stats["motion_schema_version"] = MOTION_SCHEMA_VERSION
    _write_json(output_root / "normalization_stats.json", stats)

    summary = dict(source_summary)
    summary.update(
        {
            "schema_version": DATASET_SCHEMA,
            "source_tensor_root": str(source_root),
            "source_tensor_manifest_sha256": _sha256(source_manifest_path),
            "tensor_contract": contract,
            "motion": {
                "seed_count": len(seed_summaries),
                "velocity_valid_count": sum(row["velocity_valid_count"] for row in seed_summaries),
                "acceleration_valid_count": sum(
                    row["acceleration_valid_count"] for row in seed_summaries
                ),
                "nonzero_speed_count": sum(row["nonzero_speed_count"] for row in seed_summaries),
                "nonzero_acceleration_count": sum(
                    row["nonzero_acceleration_count"] for row in seed_summaries
                ),
            },
        }
    )
    _write_json(output_root / "dataset_summary.json", summary)
    checks = {
        "source_tensor_ready": True,
        "source_and_output_are_distinct": source_root.resolve() != output_root.resolve(),
        "window_index_unchanged": _sha256(window_path) == _sha256(output_root / "window_index.csv"),
        "all_splits_nonlocked": all(split != "locked_test" for split in split_by_seed.values()),
        "locked_test_not_materialized": not (output_root / "locked_test").exists(),
        "motion_is_causal_backward_difference": True,
        "derived_velocity_nonzero": summary["motion"]["nonzero_speed_count"] > 0,
        "derived_acceleration_nonzero": summary["motion"]["nonzero_acceleration_count"] > 0,
        "normalization_source_is_train": stats.get("source_split") == "train",
    }
    validation = {
        "schema_version": "PI-JWM-formal-causal-motion-tensor-validation-v1",
        "formal_tensor_ready": all(checks.values()),
        "formal_training_ready": False,
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "locked_test_accessed": False,
        "formal_performance_claim_ready": False,
    }
    _write_json(output_root / "validation_report.json", validation)
    provenance = {
        "schema_version": "PI-JWM-formal-causal-motion-source-v1",
        "source_root": str(source_root),
        "source_manifest_sha256": _sha256(source_manifest_path),
        "window_index_sha256": _sha256(window_path),
        "locked_test_accessed": False,
    }
    _write_json(output_root / "source_provenance.json", provenance)
    manifest = _write_manifest(output_root)
    return {
        "status": "ready_for_model_integration" if validation["formal_tensor_ready"] else "blocked",
        "output_root": str(output_root),
        "validation": validation,
        "motion": summary["motion"],
        "manifest_file_count": len(manifest["files"]),
        "manifest_sha256": _sha256(output_root / "manifest.json"),
        "locked_test_accessed": False,
        "formal_performance_claim_ready": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = build_causal_motion_tensor(
        source_root=args.source_root, output_root=args.output_root
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["status"] == "ready_for_model_integration" else 1


if __name__ == "__main__":
    raise SystemExit(main())
