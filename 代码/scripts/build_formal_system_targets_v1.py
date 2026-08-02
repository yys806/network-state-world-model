"""Build nonlocked real system-outcome sidecars for formal PI-JWM tensors."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pi_jwm.formal_system_targets_v1 import SCHEMA_VERSION, build_system_target_arrays


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_formal_system_target_dataset(
    *,
    source_dir: str | Path,
    tensor_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    source_dir = Path(source_dir)
    tensor_dir = Path(tensor_dir)
    output_dir = Path(output_dir)
    seed_dirs = sorted(path for path in tensor_dir.glob("seed_*") if path.is_dir())
    if not seed_dirs:
        raise ValueError("formal tensor directory contains no nonlocked seed directories")

    output_dir.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, Any]] = []
    manifest_paths: list[Path] = []
    for tensor_seed_dir in seed_dirs:
        tensor_report = _read_json(tensor_seed_dir / "tensor_report.json")
        split = str(tensor_report.get("split"))
        if split == "locked_test" or int(tensor_report.get("seed", -1)) >= 600:
            raise ValueError("locked-test tensors are forbidden in the system-target builder")
        if split not in {"train", "validation", "calibration"}:
            raise ValueError(f"unsupported nonlocked split: {split}")
        seed = int(tensor_report["seed"])
        trajectory_id = str(tensor_report["trajectory_id"])
        trajectory_dir = source_dir / "trajectories" / trajectory_id
        graph = _read_json(trajectory_dir / "dual_graph_v2_bundle.json")
        resources = _read_json(trajectory_dir / "resource_bundle.json")
        with np.load(tensor_seed_dir / "trajectory_tensors.npz", allow_pickle=False) as tensors:
            time_values = tensors["time"].astype(np.float64, copy=True)
        arrays = build_system_target_arrays(
            time_values=time_values,
            node_vocab=[str(value) for value in tensor_report["node_vocab"]],
            task_vocab=[str(value) for value in tensor_report["task_vocab"]],
            task_snapshots=list(graph.get("source_task_snapshots", [])),
            energy_rows=list(resources.get("uav_energy_ledger", [])),
            transfer_events=list(graph.get("source_transfer_events", [])),
        )
        if not np.array_equal(
            arrays["task_completion_event"], arrays["completed_task_delay_valid"]
        ):
            raise ValueError("every completion event must have one direct completed-task delay")
        if not all(np.all(np.isfinite(value)) for value in arrays.values() if value.dtype.kind == "f"):
            raise ValueError("system-target arrays must contain only finite numeric values")

        seed_output = output_dir / f"seed_{seed:03d}"
        seed_output.mkdir(parents=True, exist_ok=True)
        npz_path = seed_output / "system_targets.npz"
        np.savez_compressed(npz_path, **arrays)
        report = {
            "schema_version": SCHEMA_VERSION,
            "seed": seed,
            "split": split,
            "trajectory_id": trajectory_id,
            "time_count": len(time_values),
            "completion_event_count": int(arrays["task_completion_event"].sum()),
            "on_time_completion_event_count": int(
                arrays["task_on_time_completion_event"].sum()
            ),
            "completed_delay_count": int(arrays["completed_task_delay_valid"].sum()),
            "uav_energy_row_count": int(arrays["uav_energy_valid"].sum()),
            "uav_energy_total": float(arrays["uav_energy_delta"].sum()),
            "source_service_total": float(arrays["source_service_delta"].sum()),
            "source_on_time_service_total": float(
                arrays["source_on_time_service_delta"].sum()
            ),
            "source_task_count": int(arrays["source_task_count"].sum()),
            "delivered_data_total": float(arrays["delivered_data_total"].sum()),
            "array_shapes": {name: list(value.shape) for name, value in arrays.items()},
            "array_dtypes": {name: str(value.dtype) for name, value in arrays.items()},
        }
        report_path = seed_output / "system_target_report.json"
        _write_json(report_path, report)
        reports.append(report)
        manifest_paths.extend((npz_path, report_path))

    checks = {
        "all_splits_nonlocked": all(row["split"] != "locked_test" for row in reports),
        "all_seeds_below_locked_range": all(int(row["seed"]) < 600 for row in reports),
        "all_completion_delays_direct": all(
            row["completion_event_count"] == row["completed_delay_count"] for row in reports
        ),
        "energy_targets_nonempty": sum(row["uav_energy_row_count"] for row in reports) > 0,
        "delivered_data_targets_nonempty": sum(row["delivered_data_total"] for row in reports) > 0,
        "source_task_counts_match_tensor_vocab": all(
            row["source_task_count"] == row["array_shapes"]["task_completion_event"][1]
            for row in reports
        ),
        "on_time_service_matches_on_time_events": all(
            row["source_on_time_service_total"] == row["on_time_completion_event_count"]
            for row in reports
        ),
    }
    summary = {
        "schema_version": "PI-JWM-formal-system-target-dataset-v1",
        "system_targets_ready": all(checks.values()),
        "nonlocked_seed_count": len(reports),
        "split_counts": {
            split: sum(row["split"] == split for row in reports)
            for split in ("train", "validation", "calibration")
        },
        "checks": checks,
        "locked_test_accessed": False,
        "totals": {
            "completion_events": sum(row["completion_event_count"] for row in reports),
            "on_time_completion_events": sum(
                row["on_time_completion_event_count"] for row in reports
            ),
            "source_tasks": sum(row["source_task_count"] for row in reports),
            "uav_energy_rows": sum(row["uav_energy_row_count"] for row in reports),
            "uav_energy": sum(row["uav_energy_total"] for row in reports),
            "delivered_data": sum(row["delivered_data_total"] for row in reports),
        },
    }
    summary_path = output_dir / "dataset_summary.json"
    _write_json(summary_path, summary)
    manifest_paths.append(summary_path)
    _write_json(
        output_dir / "manifest.json",
        {
            "schema_version": "PI-JWM-formal-system-target-manifest-v1",
            "files": {
                path.relative_to(output_dir).as_posix(): {
                    "sha256": _sha256(path),
                    "bytes": path.stat().st_size,
                }
                for path in manifest_paths
            },
        },
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=CODE_ROOT / "artifacts" / "datasets" / "airfogsim_formal_v1",
    )
    parser.add_argument(
        "--tensor-dir",
        type=Path,
        default=CODE_ROOT / "artifacts" / "datasets" / "airfogsim_formal_tensor_v1",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CODE_ROOT / "artifacts" / "datasets" / "airfogsim_formal_system_targets_v1",
    )
    args = parser.parse_args()
    result = build_formal_system_target_dataset(
        source_dir=args.source_dir,
        tensor_dir=args.tensor_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if result["system_targets_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
