"""Run controlled non-locked input-perturbation diagnostics on formal GPU checkpoints."""

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
from pi_jwm.formal_gpu_robustness_v1 import (
    CONTINUOUS_HISTORY_KEYS,
    execution_policy,
    perturb_history,
)
from pi_jwm.formal_world_model_metrics_v1 import FormalMetricAccumulator
from run_formal_dual_graph_cpu_smoke_v1 import _load_or_fit_stats, _subset_for_ids
from run_formal_dual_graph_gpu_train_v1 import move_nested_to_device


METRIC_NAMES = {
    "link_active_f1": "event.link_activity.f1",
    "node_x_mae": "state.node.x.mae",
    "throughput_mae": "system.communication_throughput.mae",
    "rb_occupancy_mae": "resource.rb_occupancy.mae",
    "task_delay_mae": "state.task.delay.mae",
}


def _metric(report: dict[str, Any], name: str) -> float | None:
    item = report.get("horizons", {}).get("overall", {}).get("metrics", {}).get(name, {})
    return float(item["value"]) if item.get("status") == "computed" else None


def _load_thresholds(run_dir: Path) -> dict[str, float]:
    report = json.loads(
        (run_dir / "metrics" / "coupled_dual_gnn_residual__threshold_selection.json").read_text(
            encoding="utf-8"
        )
    )
    return {
        str(name): float(value["selected"]["threshold"])
        for name, value in report["events"].items()
    }


def _load_model(run_dir: Path, device: torch.device) -> FormalDualGraphWorldModel:
    checkpoint_path = run_dir / "checkpoints" / "coupled_dual_gnn_residual__best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = FormalDualGraphWorldModel(FormalWorldModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model


def _validate_run(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    if config.get("locked_test_accessed") or summary.get("locked_test_accessed"):
        raise ValueError(f"locked_test access is forbidden: {run_dir}")
    if summary.get("gpu_execution") is not True or summary.get("training_run_complete") is not True:
        raise ValueError(f"run is not a completed GPU run: {run_dir}")
    if "locked_test" in str(run_dir).lower():
        raise ValueError(f"locked_test path is forbidden: {run_dir}")
    return config, summary


def evaluate_run_robustness(
    run_dir: str | Path,
    *,
    tensor_root: str | Path,
    noise_scales: tuple[float, ...] = (0.0, 0.05, 0.10, 0.20),
    device: str = "cuda",
    batch_size: int = 2,
    perturb_keys: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    config, summary = _validate_run(run_dir)
    if not torch.cuda.is_available() and str(device).startswith("cuda"):
        raise RuntimeError("CUDA is not available")
    device_obj = torch.device(device)
    policy = execution_policy(device_obj)
    tensor_root = Path(tensor_root)
    contract = json.loads((tensor_root / "tensor_contract.json").read_text(encoding="utf-8"))
    stats = _load_or_fit_stats(tensor_root)
    dataset = FormalAirFogSimWindowDataset(
        tensor_root,
        split="validation",
        config=FormalWindowConfig(
            history_steps=int(contract["history_steps"]),
            horizon_steps=int(contract["horizon_steps"]),
        ),
        stats=stats,
        normalize=True,
    )
    sample_ids = json.loads((run_dir / "sample_ids.json").read_text(encoding="utf-8"))["validation"]
    subset = _subset_for_ids(dataset, sample_ids)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)
    model = _load_model(run_dir, device_obj)
    thresholds = _load_thresholds(run_dir)
    reports: list[dict[str, Any]] = []
    for noise_scale in noise_scales:
        accumulator = FormalMetricAccumulator(stats, thresholds=thresholds, distribution_available=True)
        with torch.no_grad():
            for batch_index, cpu_batch in enumerate(loader):
                batch = move_nested_to_device(cpu_batch, device_obj)
                batch["history"] = perturb_history(
                    batch["history"],
                    noise_scale=float(noise_scale),
                    seed=int(config["seed"]) + batch_index,
                    perturb_keys=perturb_keys,
                )
                prediction = model(batch)
                accumulator.update(prediction, batch["target"], batch["static"])
        metric_report = accumulator.finalize()
        reports.append(
            {
                "noise_scale": float(noise_scale),
                "metrics": {
                    name: _metric(metric_report, source) for name, source in METRIC_NAMES.items()
                },
            }
        )
    clean = reports[0]["metrics"]
    for row in reports:
        row["metric_deltas_from_clean"] = {
            name: (value - clean[name]) if value is not None and clean[name] is not None else None
            for name, value in row["metrics"].items()
        }
    return {
        "schema_version": "PI-JWM-formal-gpu-robustness-v1",
        "audit_passed": True,
        "formal_performance_claim_ready": False,
        "seed": int(config["seed"]),
        "run_dir": str(run_dir.resolve()),
        "sample_count": len(sample_ids),
        "noise_protocol": {
            "scales_in_normalized_feature_std": list(noise_scales),
            "perturbed_history_keys": list(
                perturb_keys or [key for key, _ in CONTINUOUS_HISTORY_KEYS]
            ),
            "unchanged_keys": ["*_present", "task_action", "future_action", "static", "target"],
            "threshold_source": "clean calibration thresholds from the same run",
        },
        "reports": reports,
        "execution_policy": policy,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--tensor-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--perturb-key", action="append", dest="perturb_keys")
    args = parser.parse_args()
    reports = [
        evaluate_run_robustness(
            run_dir,
            tensor_root=args.tensor_root,
            device=args.device,
            perturb_keys=tuple(args.perturb_keys) if args.perturb_keys else None,
        )
        for run_dir in args.run_dir
    ]
    output = {
        "schema_version": "PI-JWM-formal-gpu-robustness-multiseed-v1",
        "audit_passed": all(report["audit_passed"] for report in reports),
        "formal_performance_claim_ready": False,
        "seed_count": len(reports),
        "reports": reports,
        "execution_policy": {
            "locked_test_accessed": False,
            "evaluation_devices": sorted({report["execution_policy"]["evaluation_device"] for report in reports}),
            "gpu_execution": all(report["execution_policy"]["gpu_execution"] for report in reports),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "gpu_robustness_multiseed.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "robustness_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["seed", "noise_scale", *METRIC_NAMES.keys(), *[f"delta_{name}" for name in METRIC_NAMES]]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for report in reports:
            for row in report["reports"]:
                writer.writerow({
                    "seed": report["seed"],
                    "noise_scale": row["noise_scale"],
                    **row["metrics"],
                    **{f"delta_{name}": value for name, value in row["metric_deltas_from_clean"].items()},
                })
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if output["audit_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
