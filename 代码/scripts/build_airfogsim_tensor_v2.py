from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pi_jwm.airfogsim_tensor_v2 import infer_tensor_contract, tensorize_seed_graph, validate_seed_tensors
from pi_jwm.airfogsim_window_dataset_v2 import fit_training_stats


DEFAULT_SOURCE = CODE_ROOT / "artifacts" / "datasets" / "airfogsim_multiseed_v2_dev"
DEFAULT_OUTPUT = CODE_ROOT / "artifacts" / "datasets" / "airfogsim_tensor_v2_dev"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_window_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def build_tensor_dataset(
    *,
    source_dir: str | Path,
    output_dir: str | Path,
    graph_loader: Callable[[int], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    source_summary = _read_json(source_dir / "dataset_summary.json")
    if source_summary.get("schema_version") != "PI-JWM-AirFogSim-multiseed-dataset-v2":
        raise ValueError("source dataset must use the AirFogSim multiseed v2 schema")
    seeds = [int(seed) for seed in source_summary.get("seeds", [])]
    if not seeds:
        raise ValueError("source dataset contains no seeds")
    history_steps = int(source_summary.get("history_steps", 0))
    horizon_steps = int(source_summary.get("horizon_steps", 0))
    if history_steps <= 0 or horizon_steps <= 0:
        raise ValueError("source dataset has invalid history or horizon steps")

    if graph_loader is None:
        graph_loader = lambda seed: _read_json(source_dir / f"seed_{seed:03d}" / "dual_graph_v2_bundle.json")
    graphs = {seed: graph_loader(seed) for seed in seeds}
    contract = infer_tensor_contract(
        list(graphs.values()),
        history_steps=history_steps,
        horizon_steps=horizon_steps,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "tensor_contract.json", contract.to_dict())
    shutil.copyfile(source_dir / "window_index.csv", output_dir / "window_index.csv")

    seed_reports: list[dict[str, Any]] = []
    time_counts: dict[int, int] = {}
    for seed in seeds:
        arrays, report = tensorize_seed_graph(graphs[seed], contract)
        validation = validate_seed_tensors(arrays, contract)
        seed_dir = output_dir / f"seed_{seed:03d}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(seed_dir / "trajectory_tensors.npz", **arrays)
        report = {
            **report,
            "seed": seed,
            "split": str(source_summary.get("split_by_seed", {}).get(str(seed), "")),
            "validation": validation,
            "array_shapes": {key: list(value.shape) for key, value in arrays.items()},
            "array_dtypes": {key: str(value.dtype) for key, value in arrays.items()},
        }
        _write_json(seed_dir / "tensor_report.json", report)
        seed_reports.append(report)
        time_counts[seed] = int(report["time_count"])

    stats = fit_training_stats(output_dir, split="dev_train")
    _write_json(output_dir / "normalization_stats.json", stats)
    window_rows = _load_window_rows(output_dir / "window_index.csv")
    window_bounds_valid = all(
        int(row["seed"]) in time_counts
        and 0 <= int(row["input_start_index"]) < int(row["input_end_index"])
        and int(row["input_end_index"]) == int(row["label_start_index"])
        and int(row["label_start_index"]) < int(row["label_end_index"]) <= time_counts[int(row["seed"])]
        for row in window_rows
    )
    checks = {
        "source_schema_valid": True,
        "all_seed_tensors_valid": all(report["validation"]["tensor_valid"] for report in seed_reports),
        "window_bounds_valid": window_bounds_valid,
        "training_stats_are_train_only": stats.get("source_split") == "dev_train",
        "nonempty_graph_components_have_observations": all(
            (report["counts"]["nodes"] == 0 or report["validation"]["present_counts"]["nodes"] > 0)
            and (report["counts"]["physical_edges"] == 0 or report["validation"]["present_counts"]["physical_edges"] > 0)
            and (report["counts"]["flows"] == 0 or report["validation"]["present_counts"]["flows"] > 0)
            and (report["counts"]["tasks"] == 0 or report["validation"]["present_counts"]["tasks"] > 0)
            for report in seed_reports
        ),
        "frozen_capacity_covers_all_seeds": all(
            report["counts"]["nodes"] <= contract.max_nodes
            and report["counts"]["physical_edges"] <= contract.max_physical_edges
            and report["counts"]["flows"] <= contract.max_flows
            and report["counts"]["tasks"] <= contract.max_tasks
            and report["counts"]["dag_edges"] <= contract.max_dag_edges
            for report in seed_reports
        ),
    }
    development_tensor_ready = all(checks.values())
    validation_report = {
        "schema_version": "PI-JWM-AirFogSim-tensor-validation-v2",
        "development_tensor_ready": development_tensor_ready,
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "seed_count": len(seeds),
        "window_count": len(window_rows),
    }
    _write_json(output_dir / "validation_report.json", validation_report)
    tensor_summary = {
        "schema_version": "PI-JWM-AirFogSim-tensor-dataset-v2",
        "source_dataset": str(source_dir.resolve()),
        "development_tensor_ready": development_tensor_ready,
        "formal_training_ready": False,
        "formal_training_blockers": [
            "only_development_seeds",
            "no_locked_test_split",
            "single_pass_multiseed_generation",
        ],
        "seeds": seeds,
        "split_by_seed": source_summary.get("split_by_seed", {}),
        "history_steps": history_steps,
        "horizon_steps": horizon_steps,
        "window_count": len(window_rows),
        "tensor_contract": contract.to_dict(),
        "seed_counts": {str(report["seed"]): report["counts"] for report in seed_reports},
    }
    _write_json(output_dir / "dataset_summary.json", tensor_summary)

    manifest_paths = [
        output_dir / "tensor_contract.json",
        output_dir / "window_index.csv",
        output_dir / "normalization_stats.json",
        output_dir / "validation_report.json",
        output_dir / "dataset_summary.json",
    ]
    for seed in seeds:
        manifest_paths.extend(
            [
                output_dir / f"seed_{seed:03d}" / "trajectory_tensors.npz",
                output_dir / f"seed_{seed:03d}" / "tensor_report.json",
            ]
        )
    manifest = {
        "schema_version": "PI-JWM-AirFogSim-tensor-manifest-v2",
        "files": {
            path.relative_to(output_dir).as_posix(): {
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in manifest_paths
        },
    }
    _write_json(output_dir / "manifest.json", manifest)
    return tensor_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tensorize the AirFogSim dual-graph v2 development dataset.")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_tensor_dataset(source_dir=args.source_dir, output_dir=args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
