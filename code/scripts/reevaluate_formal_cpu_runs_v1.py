"""Re-evaluate existing non-locked formal CPU runs after a metric-only repair."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pi_jwm.formal_airfogsim_window_v1 import FormalAirFogSimWindowDataset, FormalWindowConfig
from pi_jwm.formal_dual_graph_world_model_v1 import FormalDualGraphWorldModel, FormalWorldModelConfig
from pi_jwm.formal_world_model_baselines_v1 import build_rule_prediction, method_registry
from run_formal_dual_graph_cpu_smoke_v1 import (
    _evaluate,
    _load_or_fit_stats,
    _metric_value,
    _subset_for_ids,
    _choose_event_thresholds,
)


def _load_model(run_dir: Path, method: str) -> torch.nn.Module:
    checkpoint_path = run_dir / "checkpoints" / f"{method}__best.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = FormalDualGraphWorldModel(FormalWorldModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model


def reevaluate_run(*, run_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    output_dir = Path(output_dir)
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    if config.get("locked_test_accessed"):
        raise ValueError("locked_test run cannot be re-evaluated")
    if config.get("device") != "cpu":
        raise ValueError("only existing CPU runs may be re-evaluated")
    tensor_root = Path(config["tensor_root"])
    contract = json.loads((tensor_root / "tensor_contract.json").read_text(encoding="utf-8"))
    stats = _load_or_fit_stats(tensor_root)
    window_config = FormalWindowConfig(
        history_steps=int(contract["history_steps"]),
        horizon_steps=int(contract["horizon_steps"]),
    )
    sample_ids = json.loads((run_dir / "sample_ids.json").read_text(encoding="utf-8"))
    datasets = {
        split: FormalAirFogSimWindowDataset(
            tensor_root, split=split, config=window_config, stats=stats, normalize=True
        )
        for split in ("validation", "calibration")
    }
    batch_size = int(config.get("batch_size", 2))
    loaders = {
        split: DataLoader(
            _subset_for_ids(datasets[split], list(sample_ids[split])),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
        for split in datasets
    }
    method = "coupled_dual_gnn_residual"
    model = _load_model(run_dir, method)
    thresholds, threshold_report, _ = _choose_event_thresholds(method, model, loaders["calibration"], stats)
    reports = {
        "validation": _evaluate(method, model, loaders["validation"], stats, thresholds, True)[0],
        "calibration": _evaluate(method, model, loaders["calibration"], stats, thresholds, True)[0],
    }
    registry = method_registry()
    rows: list[dict[str, Any]] = []
    for baseline in ("last_persistence",):
        base_thresholds, _, _ = _choose_event_thresholds(baseline, None, loaders["calibration"], stats)
        base_reports = {
            split: _evaluate(baseline, None, loaders[split], stats, base_thresholds, False)[0]
            for split in reports
        }
        baseline_row = _comparison_row(baseline, base_thresholds, base_reports, None)
        rows.append(baseline_row)
    learned_row = _comparison_row(method, thresholds, reports, config)
    baseline_row = rows[0]
    for metric in ("link_f1", "node_x_mae", "throughput_mae", "rb_occupancy_mae", "task_delay_mae"):
        learned_row[f"validation_persistence_{metric}"] = baseline_row[f"validation_{metric}"]
    rows.append(learned_row)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics").mkdir(exist_ok=True)
    reevaluation_config = {
        "schema_version": "PI-JWM-formal-cpu-reevaluation-config-v1",
        "seed": config["seed"],
        "device": "cpu",
        "tensor_root": str(tensor_root.resolve()),
        "source_run": str(run_dir.resolve()),
        "threshold_selection_split": "calibration",
        "locked_test_accessed": False,
    }
    (output_dir / "config.json").write_text(
        json.dumps(reevaluation_config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for split, report in reports.items():
        (output_dir / "metrics" / f"{method}__{split}.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    for split, report in base_reports.items():
        (output_dir / "metrics" / f"last_persistence__{split}.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    (output_dir / "metrics" / f"{method}__threshold_selection.json").write_text(
        json.dumps(threshold_report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output_dir / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "schema_version": "PI-JWM-formal-cpu-reevaluation-v1",
        "source_run": str(run_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "device": "cpu",
        "metric_code_revision": "aggregate-rb-physical-unit-repair",
        "threshold_selection_split": "calibration",
        "locked_test_accessed": False,
        "gpu_started": False,
        "formal_performance_claim_ready": False,
        "sample_ids_reused": True,
        "methods": ["last_persistence", method],
        "registry_distribution_available": registry[method]["distribution_output"],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def _comparison_row(method: str, thresholds: dict[str, float], reports: dict[str, dict[str, Any]], config: dict[str, Any] | None) -> dict[str, Any]:
    row: dict[str, Any] = {"method": method, "threshold": thresholds["link_activity"], "thresholds": json.dumps(thresholds, sort_keys=True)}
    for split in ("validation", "calibration"):
        report = reports[split]
        row[f"{split}_link_f1"] = _metric_value(report, "event.link_activity.f1")
        row[f"{split}_node_x_mae"] = _metric_value(report, "state.node.x.mae")
        row[f"{split}_throughput_mae"] = _metric_value(report, "system.communication_throughput.mae")
        row[f"{split}_rb_occupancy_mae"] = _metric_value(report, "resource.rb_occupancy.mae")
        row[f"{split}_task_delay_mae"] = _metric_value(report, "state.task.delay.mae")
    row.update({"validation_persistence_link_f1": None, "validation_persistence_node_x_mae": None, "validation_persistence_throughput_mae": None, "validation_persistence_rb_occupancy_mae": None, "validation_persistence_task_delay_mae": None})
    row.update({"best_epoch": None if config is None else config.get("epochs"), "best_validation_loss": None, "parameter_count": 0, "train_seconds": 0.0, "peak_device_memory_bytes": 0})
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    for run_dir in args.run_dir:
        run_path = Path(run_dir)
        reevaluate_run(run_dir=run_path, output_dir=args.output_root / run_path.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
