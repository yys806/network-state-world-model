from __future__ import annotations

"""Tensorize the unlocked portion of the formal PI-JWM AirFogSim dataset."""

import argparse
import csv
import hashlib
import json
import shutil
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pi_jwm.airfogsim_tensor_v2 import (
    TensorContract,
    contract_from_dict,
    infer_tensor_contract,
    validate_seed_tensors,
)
from pi_jwm.airfogsim_dataset_v2 import build_multiseed_window_index
from pi_jwm.airfogsim_window_dataset_v2 import fit_training_stats
from pi_jwm.formal_airfogsim_graph_v1 import (
    FORMAL_ACTION_FEATURES,
    FORMAL_DAG_STATE_FEATURES,
    SCHEMA_VERSION as FORMAL_TENSOR_SCHEMA_VERSION,
    tensorize_formal_graph,
)


SCHEMA_VERSION = "PI-JWM-AirFogSim-formal-tensor-dataset-v1"


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


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_formal_tensor_dataset(
    *,
    source_dir: str | Path,
    output_dir: str | Path,
    graph_loader: Callable[[Mapping[str, str]], dict[str, Any]] | None = None,
    include_locked_test: bool = True,
    history_steps: int | None = None,
    horizon_steps: int | None = None,
    reuse_existing: bool = False,
) -> dict[str, Any]:
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    source_summary = _read_json(source_dir / "dataset_summary.json")
    if source_summary.get("schema_version") != "PI-JWM-AirFogSim-formal-dataset-v1":
        raise ValueError("source dataset must use the formal AirFogSim v1 schema")
    history_steps = int(
        source_summary.get("history_steps", 0)
        if history_steps is None
        else history_steps
    )
    horizon_steps = int(
        source_summary.get("horizon_steps", 0)
        if horizon_steps is None
        else horizon_steps
    )
    if history_steps <= 0 or horizon_steps <= 0:
        raise ValueError("source dataset has invalid history or horizon steps")

    trajectory_rows = _read_csv(source_dir / "trajectory_index.csv")
    unlocked_rows = [row for row in trajectory_rows if row["split"] != "locked_test"]
    locked_rows = [row for row in trajectory_rows if row["split"] == "locked_test"]
    if not unlocked_rows or (include_locked_test and not locked_rows):
        raise ValueError("formal source must contain unlocked and locked-test trajectories")
    if not include_locked_test:
        locked_rows = []
    if graph_loader is None:
        graph_loader = lambda row: _read_json(
            source_dir
            / "trajectories"
            / row["trajectory_id"]
            / "dual_graph_v2_bundle.json"
        )
    if reuse_existing:
        existing_contract_path = output_dir / "tensor_contract.json"
        if not existing_contract_path.exists():
            raise FileNotFoundError(
                "reuse_existing requires an existing tensor_contract.json"
            )
        existing_payload = _read_json(existing_contract_path)
        contract = contract_from_dict(existing_payload)
        if contract.history_steps != history_steps or contract.horizon_steps != horizon_steps:
            raise ValueError(
                "existing tensor contract does not match requested history/horizon"
            )
    else:
        per_trajectory_contracts = [
            infer_tensor_contract(
                [graph_loader(row)],
                history_steps=history_steps,
                horizon_steps=horizon_steps,
            )
            for row in unlocked_rows
        ]
        contract = TensorContract(
            max_nodes=max(row.max_nodes for row in per_trajectory_contracts),
            max_physical_edges=max(
                row.max_physical_edges for row in per_trajectory_contracts
            ),
            max_flows=max(row.max_flows for row in per_trajectory_contracts),
            max_tasks=max(row.max_tasks for row in per_trajectory_contracts),
            max_dag_edges=max(row.max_dag_edges for row in per_trajectory_contracts),
            history_steps=history_steps,
            horizon_steps=horizon_steps,
            n_rb=max(row.n_rb for row in per_trajectory_contracts),
        )
    contract_payload = contract.to_dict()
    contract_payload.update(
        {
            "schema_version": FORMAL_TENSOR_SCHEMA_VERSION,
            "action_features": list(FORMAL_ACTION_FEATURES),
            "dag_state_features": list(FORMAL_DAG_STATE_FEATURES),
            "action_source_endpoint_field": "task_action_source_node_index",
            "flow_task_mapping_field": "flow_task_index",
            "slot_duration_field": "slot_seconds",
            "normalization_source_split": "train",
            "locked_test_capacity_status": "not_inspected_before_model_freeze",
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "tensor_contract.json", contract_payload)
    reports: list[dict[str, Any]] = []
    time_counts: dict[int, int] = {}
    times_by_seed: dict[int, list[float]] = {}
    split_by_seed: dict[int, str] = {}
    for row in unlocked_rows:
        seed = int(row["seed"])
        seed_dir = output_dir / f"seed_{seed:03d}"
        if reuse_existing:
            tensor_path = seed_dir / "trajectory_tensors.npz"
            report_path = seed_dir / "tensor_report.json"
            if not tensor_path.exists() or not report_path.exists():
                raise FileNotFoundError(
                    f"reuse_existing requires complete seed artifacts for {seed}"
                )
            with np.load(tensor_path, allow_pickle=False) as loaded:
                arrays = {name: loaded[name] for name in loaded.files}
            report = _read_json(report_path)
            validation = validate_seed_tensors(arrays, contract)
            if not validation.get("tensor_valid", False):
                raise ValueError(f"existing seed tensor {seed} failed validation")
            report = {
                **report,
                "seed": seed,
                "trajectory_id": row["trajectory_id"],
                "split": row["split"],
                "validation": validation,
                "array_shapes": {name: list(value.shape) for name, value in arrays.items()},
                "array_dtypes": {name: str(value.dtype) for name, value in arrays.items()},
            }
        else:
            arrays, report = tensorize_formal_graph(graph_loader(row), contract)
            validation = validate_seed_tensors(arrays, contract)
            seed_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(seed_dir / "trajectory_tensors.npz", **arrays)
            report = {
                **report,
                "seed": seed,
                "trajectory_id": row["trajectory_id"],
                "split": row["split"],
                "validation": validation,
                "array_shapes": {name: list(value.shape) for name, value in arrays.items()},
                "array_dtypes": {name: str(value.dtype) for name, value in arrays.items()},
            }
            _write_json(seed_dir / "tensor_report.json", report)
        reports.append(report)
        time_counts[seed] = int(report["time_count"])
        times_by_seed[seed] = [round(float(value), 6) for value in arrays["time"].tolist()]
        split_by_seed[seed] = str(row["split"])

    # Rebuild windows from the tensorized time grids. This is deliberate: a
    # source dataset may have been collected with a shorter historical
    # horizon, but its verified trajectories can support a new, explicit
    # horizon without copying stale window rows.
    unlocked_windows = build_multiseed_window_index(
        times_by_seed,
        split_by_seed,
        history_steps=history_steps,
        horizon_steps=horizon_steps,
    )
    _write_csv(output_dir / "window_index.csv", unlocked_windows)

    stats = fit_training_stats(output_dir, split="train")
    _write_json(output_dir / "normalization_stats.json", stats)
    window_rows = _read_csv(output_dir / "window_index.csv")
    window_bounds_valid = all(
        int(row["seed"]) in time_counts
        and 0 <= int(row["input_start_index"]) < int(row["input_end_index"])
        and int(row["input_end_index"]) == int(row["label_start_index"])
        and int(row["label_start_index"])
        < int(row["label_end_index"])
        <= time_counts[int(row["seed"])]
        for row in window_rows
    )
    checks = {
        "source_schema_valid": True,
        "all_unlocked_tensors_valid": all(
            row["validation"]["tensor_valid"] for row in reports
        ),
        "window_bounds_valid": window_bounds_valid,
        "training_stats_are_train_only": stats.get("source_split") == "train",
        "formal_action_width_is_8": all(
            row["array_shapes"]["task_action"][-1] == len(FORMAL_ACTION_FEATURES)
            for row in reports
        ),
        "dag_state_is_present": all("task_dag_state" in row["array_shapes"] for row in reports),
        "locked_test_not_tensorized": all(
            not (output_dir / f"seed_{int(row['seed']):03d}").exists()
            for row in locked_rows
        ),
    }
    formal_tensor_ready = all(checks.values())
    validation = {
        "schema_version": "PI-JWM-AirFogSim-formal-tensor-validation-v1",
        "formal_tensor_ready": formal_tensor_ready,
        "formal_training_ready": False,
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
    }
    _write_json(output_dir / "validation_report.json", validation)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "source_dataset": str(source_dir.resolve()),
        "formal_tensor_ready": formal_tensor_ready,
        "formal_training_ready": False,
        "formal_training_blockers": [
            "model_and_training_protocol_not_frozen",
        "locked_test_remains_untensorized_until_explicit_unlock",
        ],
        "unlocked_trajectory_count": len(unlocked_rows),
        "locked_test_trajectory_count": len(locked_rows),
        "window_count": len(window_rows),
        "history_steps": history_steps,
        "horizon_steps": horizon_steps,
        "normalization_source_split": "train",
        "tensor_contract": contract_payload,
    }
    _write_json(output_dir / "dataset_summary.json", summary)
    manifest_paths = [
        output_dir / "tensor_contract.json",
        output_dir / "window_index.csv",
        output_dir / "normalization_stats.json",
        output_dir / "validation_report.json",
        output_dir / "dataset_summary.json",
    ]
    for row in unlocked_rows:
        seed_dir = output_dir / f"seed_{int(row['seed']):03d}"
        manifest_paths.extend(
            [seed_dir / "trajectory_tensors.npz", seed_dir / "tensor_report.json"]
        )
    _write_json(
        output_dir / "manifest.json",
        {
            "schema_version": "PI-JWM-AirFogSim-formal-tensor-manifest-v1",
            "files": {
                path.relative_to(output_dir).as_posix(): {
                    "sha256": _sha256(path),
                    "size_bytes": path.stat().st_size,
                }
                for path in manifest_paths
            },
        },
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Tensorize formal AirFogSim dataset v1.")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=CODE_ROOT / "artifacts" / "datasets" / "airfogsim_formal_v1",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CODE_ROOT / "artifacts" / "datasets" / "airfogsim_formal_tensor_v1",
    )
    parser.add_argument(
        "--unlocked-only",
        action="store_true",
        help="tensorize only train/validation/calibration rows without reading locked-test trajectories",
    )
    parser.add_argument(
        "--history",
        type=int,
        help="override the source history length when rebuilding windows",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        help="override the source horizon when rebuilding windows",
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="reuse complete seed tensor artifacts already present in output-dir",
    )
    args = parser.parse_args()
    result = build_formal_tensor_dataset(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        include_locked_test=not args.unlocked_only,
        history_steps=args.history,
        horizon_steps=args.horizon,
        reuse_existing=args.reuse_existing,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["formal_tensor_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
