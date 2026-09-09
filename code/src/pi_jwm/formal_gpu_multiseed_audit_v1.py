"""Audit reproducibility and non-locked boundaries for formal GPU seeds."""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any, Iterable, Mapping


NONLOCKED_SPLITS = ("train", "validation", "calibration")
LEARNED_METHOD = "coupled_dual_gnn_residual"
CONTRACT_KEYS = (
    "data_seed",
    "device",
    "splits",
    "train_limit",
    "evaluation_limit",
    "hidden_dim",
    "epochs",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "zero_init_residual_state_heads",
    "residual_state_scale",
    "learned_methods",
    "tensor_root",
    "dataset_manifest_sha256",
)
METRIC_KEYS = (
    "link_f1",
    "node_x_mae",
    "throughput_mae",
    "rb_occupancy_mae",
    "task_delay_mae",
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _manifest_mismatches(run_dir: Path) -> list[str]:
    manifest = _read_json(run_dir / "manifest.json")
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise ValueError(f"invalid manifest files object: {run_dir}")
    mismatches: list[str] = []
    for relative, expected in files.items():
        path = run_dir / str(relative)
        if not path.is_file():
            mismatches.append(f"{relative}:missing")
            continue
        data = path.read_bytes()
        if len(data) != int(expected["bytes"]):
            mismatches.append(f"{relative}:bytes")
        if hashlib.sha256(data).hexdigest() != str(expected["sha256"]):
            mismatches.append(f"{relative}:sha256")
    return mismatches


def _metric_summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("metric requires at least one value")
    return {
        "mean": float(statistics.mean(values)),
        "sample_std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "max": float(max(values)),
        "count": len(values),
    }


def _comparison_rows(run_dir: Path) -> dict[str, dict[str, str]]:
    with (run_dir / "comparison.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_method = {row.get("method", ""): row for row in rows}
    for method in ("last_persistence", LEARNED_METHOD):
        if method not in by_method:
            raise ValueError(f"missing comparison method {method}: {run_dir}")
    return by_method


def _validate_run(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    config = _read_json(run_dir / "config.json")
    summary = _read_json(run_dir / "run_summary.json")
    if config.get("locked_test_accessed") or summary.get("locked_test_accessed"):
        raise ValueError(f"locked_test access is forbidden: {run_dir}")
    if tuple(config.get("splits", ())) != NONLOCKED_SPLITS:
        raise ValueError(f"unsupported or locked_test split in {run_dir}")
    if summary.get("training_run_complete") is not True or summary.get("gpu_execution") is not True:
        raise ValueError(f"run is not a completed GPU execution: {run_dir}")
    counts = summary.get("sample_counts")
    if counts != {"train": 256, "validation": 128, "calibration": 128}:
        raise ValueError(f"unexpected sample counts: {run_dir}")
    for path in run_dir.rglob("*"):
        if "locked_test" in path.name.lower():
            raise ValueError(f"locked_test artifact is present: {path}")
    return config, summary, _manifest_mismatches(run_dir)


def audit_gpu_multiseed_runs(
    run_dirs: Iterable[str | Path],
    *,
    learned_method: str = LEARNED_METHOD,
) -> dict[str, Any]:
    """Audit completed, same-contract, non-locked GPU runs without evaluating new data."""

    paths = [Path(value) for value in run_dirs]
    if not paths:
        raise ValueError("at least one GPU run directory is required")
    records: list[dict[str, Any]] = []
    reference_contract: dict[str, Any] | None = None
    seen_seeds: set[int] = set()
    for run_dir in paths:
        config, summary, mismatches = _validate_run(run_dir)
        seed = int(config["seed"])
        if seed in seen_seeds:
            raise ValueError(f"duplicate seed: {seed}")
        seen_seeds.add(seed)
        contract = {key: config.get(key) for key in CONTRACT_KEYS}
        if reference_contract is None:
            reference_contract = contract
        elif contract != reference_contract:
            raise ValueError(f"contract drift detected for seed {seed}")
        comparison = _comparison_rows(run_dir)
        learned = comparison[learned_method]
        persistence = comparison["last_persistence"]
        metric_values: dict[str, float] = {}
        persistence_values: dict[str, float] = {}
        deltas: dict[str, float] = {}
        for metric in METRIC_KEYS:
            learned_value = float(learned[f"validation_{metric}"])
            persistence_value = float(persistence[f"validation_{metric}"])
            metric_values[metric] = learned_value
            persistence_values[metric] = persistence_value
            deltas[metric] = learned_value - persistence_value
        calibration_link_delta = float(learned["calibration_link_f1"]) - float(persistence["calibration_link_f1"])
        threshold_report = _read_json(
            run_dir / "metrics" / f"{learned_method}__threshold_selection.json"
        )
        records.append(
            {
                "seed": seed,
                "run_dir": str(run_dir.resolve()),
                "manifest_mismatch_count": len(mismatches),
                "validation": metric_values,
                "validation_persistence": persistence_values,
                "validation_deltas": deltas,
                "calibration_link_f1_delta": calibration_link_delta,
                "threshold_selection_split": threshold_report.get("selection_split"),
                "gpu_execution": bool(summary["gpu_execution"]),
                "locked_test_accessed": bool(summary["locked_test_accessed"]),
            }
        )
    if len(records) < 3:
        raise ValueError("at least three independent GPU seeds are required")
    metrics: dict[str, Any] = {}
    for metric in METRIC_KEYS:
        metrics[f"validation_{metric}_delta"] = _metric_summary(
            [row["validation_deltas"][metric] for row in records]
        )
    metrics["calibration_link_f1_delta"] = _metric_summary(
        [row["calibration_link_f1_delta"] for row in records]
    )
    mismatch_count = sum(int(row["manifest_mismatch_count"]) for row in records)
    return {
        "schema_version": "PI-JWM-formal-gpu-multiseed-audit-v1",
        "audit_passed": mismatch_count == 0,
        "formal_performance_claim_ready": False,
        "seed_count": len(records),
        "seeds": [row["seed"] for row in records],
        "learned_method": learned_method,
        "contract": reference_contract,
        "records": records,
        "metrics": metrics,
        "total_manifest_mismatches": mismatch_count,
        "execution_policy": {
            "gpu_execution_required": True,
            "locked_test_accessed": False,
            "per_rb_method_consumed": False,
            "result_boundary": "Nonlocked aggregate-baseline GPU training evidence only.",
        },
    }


__all__ = ["audit_gpu_multiseed_runs"]
