"""Evaluate real PI-JWM system outcomes on the exact nonlocked training-run samples."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader


CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pi_jwm.formal_airfogsim_window_v1 import FormalWindowConfig
from pi_jwm.formal_dual_graph_world_model_v1 import FormalDualGraphWorldModel, FormalWorldModelConfig
from pi_jwm.formal_system_outcome_metrics_v1 import compute_system_outcome_metrics
from pi_jwm.formal_system_prediction_v1 import system_predictions_from_batch
from pi_jwm.formal_system_window_v1 import FormalSystemWindowDataset
from pi_jwm.formal_world_model_baselines_v1 import build_rule_prediction
from run_formal_dual_graph_cpu_smoke_v1 import (
    _load_or_fit_stats,
    _manifest,
    _subset_for_ids,
    _write_json,
)
from run_formal_dual_graph_gpu_train_v1 import RULE_METHODS, move_nested_to_device


EVALUATION_SPLITS = ("validation", "calibration")


def _macro_metrics(seed_reports: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    names = sorted(
        {name for report in seed_reports for name in report["metrics"]}
    )
    result: dict[str, dict[str, Any]] = {}
    for name in names:
        records = [report["metrics"][name] for report in seed_reports]
        computed = [record for record in records if record["status"] == "computed"]
        if computed:
            values = [float(record["value"]) for record in computed]
            result[name] = {
                "value": float(np.mean(values)),
                "status": "computed",
                "numerator": float(np.sum(values)),
                "denominator": len(values),
                "count": len(values),
                "unit": computed[0]["unit"],
                "source_fields": computed[0]["source_fields"],
                "reason": None,
                "aggregation": "macro_mean_over_seed_trajectories",
            }
        else:
            result[name] = {
                **records[0],
                "aggregation": "macro_mean_over_seed_trajectories",
                "reason": records[0].get("reason") or "not computable for any seed",
            }
    return result


def _load_model(training_root: Path, method: str, device: torch.device) -> FormalDualGraphWorldModel:
    checkpoint_path = training_root / "checkpoints" / f"{method}__best.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = FormalDualGraphWorldModel(FormalWorldModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model


def _evaluate_method_split(
    *,
    method: str,
    model: FormalDualGraphWorldModel | None,
    loader: DataLoader,
    stats: Mapping[str, Any],
    device: torch.device,
    step_seconds: float,
) -> dict[str, Any]:
    grouped: dict[int, dict[str, Any]] = defaultdict(lambda: defaultdict(list))
    with torch.no_grad():
        for cpu_batch in loader:
            batch = move_nested_to_device(cpu_batch, device)
            prediction = (
                build_rule_prediction(method, batch, stats)
                if method in RULE_METHODS
                else model(batch)
            )
            converted = system_predictions_from_batch(
                prediction, batch, stats, horizon_index=0
            )
            seeds = batch["seed"].detach().cpu().tolist()
            for batch_index, seed_value in enumerate(seeds):
                seed = int(seed_value)
                for name, value in converted.items():
                    if name == "source_population_valid":
                        current = value[batch_index].detach().cpu().numpy()
                        previous = grouped[seed].get(name)
                        if previous is not None and not np.array_equal(previous, current):
                            raise ValueError(f"source population changed within seed {seed}")
                        grouped[seed][name] = current
                    elif value is None:
                        grouped[seed][name] = None
                    else:
                        grouped[seed][name].append(value[batch_index].detach().cpu().numpy())

    seed_reports = []
    for seed, values in sorted(grouped.items()):
        predicted_energy = values.get("predicted_uav_energy")
        report = compute_system_outcome_metrics(
            true_completion_event=np.stack(values["true_completion_event"]),
            predicted_completion_event=np.stack(values["predicted_completion_event"]),
            true_completed_delay=np.stack(values["true_completed_delay"]),
            predicted_task_delay=np.stack(values["predicted_task_delay"]),
            completed_delay_valid=np.stack(values["completed_delay_valid"]),
            true_delivered_data=np.stack(values["true_delivered_data"]),
            predicted_delivered_data=np.stack(values["predicted_delivered_data"]),
            step_seconds=step_seconds,
            true_uav_energy=np.stack(values["true_uav_energy"]),
            predicted_uav_energy=(
                None if predicted_energy is None else np.stack(predicted_energy)
            ),
            uav_energy_valid=np.stack(values["uav_energy_valid"]),
            true_source_service=np.stack(values["true_source_service"]),
            predicted_source_service=np.stack(values["predicted_source_service"]),
            source_population_valid=values["source_population_valid"],
        )
        report["seed"] = seed
        report["one_step_sample_count"] = len(values["true_delivered_data"])
        seed_reports.append(report)
    return {
        "schema_version": "PI-JWM-formal-system-evaluation-v1",
        "method": method,
        "protocol": "teacher_forced_k1_unique_window_targets",
        "seed_count": len(seed_reports),
        "sample_count": sum(row["one_step_sample_count"] for row in seed_reports),
        "macro_metrics": _macro_metrics(seed_reports),
        "per_seed": seed_reports,
        "locked_test_accessed": False,
    }


def _metric_value(report: Mapping[str, Any], name: str) -> float | None:
    record = report["macro_metrics"].get(name, {})
    return record.get("value") if record.get("status") == "computed" else None


def evaluate_training_run(
    *,
    tensor_root: str | Path,
    system_root: str | Path,
    training_root: str | Path,
    output_dir: str | Path,
    device: str = "cpu",
    batch_size: int = 32,
    step_seconds: float = 0.1,
) -> dict[str, Any]:
    if not np.isfinite(step_seconds) or step_seconds <= 0:
        raise ValueError("step_seconds must be finite and positive")
    tensor_root = Path(tensor_root)
    system_root = Path(system_root)
    training_root = Path(training_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device_value = torch.device(device)
    if device_value.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    summary = json.loads((training_root / "run_summary.json").read_text(encoding="utf-8"))
    if summary.get("locked_test_accessed"):
        raise ValueError("training run reports locked-test access")
    methods = [str(value) for value in summary["completed_methods"]]
    sample_ids = json.loads((training_root / "sample_ids.json").read_text(encoding="utf-8"))
    contract = json.loads((tensor_root / "tensor_contract.json").read_text(encoding="utf-8"))
    config = FormalWindowConfig(
        history_steps=int(contract["history_steps"]),
        horizon_steps=int(contract["horizon_steps"]),
    )
    stats = _load_or_fit_stats(tensor_root)
    loaders = {}
    for split in EVALUATION_SPLITS:
        dataset = FormalSystemWindowDataset(
            tensor_root,
            system_root=system_root,
            split=split,
            config=config,
            stats=stats,
            normalize=True,
        )
        subset = _subset_for_ids(dataset, list(sample_ids[split]))
        loaders[split] = DataLoader(subset, batch_size=batch_size, shuffle=False)

    reports: dict[tuple[str, str], dict[str, Any]] = {}
    for method in methods:
        model = None if method in RULE_METHODS else _load_model(training_root, method, device_value)
        for split in EVALUATION_SPLITS:
            report = _evaluate_method_split(
                method=method,
                model=model,
                loader=loaders[split],
                stats=stats,
                device=device_value,
                step_seconds=step_seconds,
            )
            reports[(method, split)] = report
            _write_json(output_dir / f"{method}__{split}.json", report)

    metric_names = {
        "completion_f1": "event.task_completion.f1",
        "completed_delay_mae": "system.latency.completed_task_delay.mae",
        "latency_p95_error": "system.latency.p95.absolute_error",
        "latency_p99_error": "system.latency.p99.absolute_error",
        "application_throughput_mae": "system.application_throughput.mae",
        "uav_energy_mae": "system.uav_energy.mae",
        "completion_fairness_error": "system.completion_fairness_jain.absolute_error",
    }
    comparison = []
    for method in methods:
        row: dict[str, Any] = {"method": method}
        for split in EVALUATION_SPLITS:
            for label, metric_name in metric_names.items():
                row[f"{split}_{label}"] = _metric_value(reports[(method, split)], metric_name)
        comparison.append(row)
    with (output_dir / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)

    result = {
        "schema_version": "PI-JWM-formal-system-evaluation-summary-v1",
        "system_evaluation_complete": True,
        "methods": methods,
        "splits": list(EVALUATION_SPLITS),
        "step_seconds": float(step_seconds),
        "locked_test_accessed": False,
        "result_boundary": "Nonlocked teacher-forced k=1 system outcomes; locked test remains sealed.",
    }
    _write_json(output_dir / "summary.json", result)
    manifest = _manifest(output_dir)
    manifest["schema_version"] = "PI-JWM-formal-system-evaluation-manifest-v1"
    _write_json(output_dir / "manifest.json", manifest)
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tensor-root", type=Path, required=True)
    parser.add_argument("--system-root", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--step-seconds", type=float, default=0.1)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = evaluate_training_run(
        tensor_root=args.tensor_root,
        system_root=args.system_root,
        training_root=args.training_root,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
        step_seconds=args.step_seconds,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
