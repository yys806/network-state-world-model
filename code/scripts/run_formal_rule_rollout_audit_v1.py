"""Replay formal rule-enabled checkpoints on CPU and audit every rollout rule step."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
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
from pi_jwm.formal_rule_rollout_audit_v1 import audit_rule_rollout
from pi_jwm.formal_world_model_metrics_v1 import FormalMetricAccumulator
from run_formal_dual_graph_cpu_smoke_v1 import _load_or_fit_stats, _subset_for_ids
from run_formal_dual_graph_gpu_train_v1 import move_nested_to_device


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _metric(report: dict[str, Any], horizon: str, name: str) -> float | None:
    item = report.get("horizons", {}).get(horizon, {}).get("metrics", {}).get(name, {})
    return float(item["value"]) if item.get("status") == "computed" else None


def _load_model(run_dir: Path) -> FormalDualGraphWorldModel:
    checkpoint = torch.load(
        run_dir / "checkpoints" / "coupled_dual_gnn_residual__best.pt",
        map_location="cpu",
        weights_only=True,
    )
    config = checkpoint.get("model_config")
    if not isinstance(config, dict) or not config.get("deterministic_rule_layer"):
        raise ValueError(f"checkpoint is not rule-layer enabled: {run_dir}")
    model = FormalDualGraphWorldModel(FormalWorldModelConfig(**config))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model


def _thresholds(run_dir: Path) -> dict[str, float]:
    report = _read_json(run_dir / "metrics" / "coupled_dual_gnn_residual__threshold_selection.json")
    return {str(name): float(value["selected"]["threshold"]) for name, value in report["events"].items()}


def audit_run(run_dir: str | Path, *, tensor_root: str | Path, batch_size: int = 2) -> dict[str, Any]:
    run_dir = Path(run_dir)
    tensor_root = Path(tensor_root)
    if "locked_test" in str(run_dir).lower() or "locked_test" in str(tensor_root).lower():
        raise ValueError("locked_test path is forbidden")
    config = _read_json(run_dir / "config.json")
    training = _read_json(run_dir / "run_summary.json")
    if config.get("locked_test_accessed") or training.get("locked_test_accessed"):
        raise ValueError(f"run declares locked-test access: {run_dir}")
    if not training.get("training_run_complete") or not training.get("gpu_execution"):
        raise ValueError(f"not a completed GPU checkpoint: {run_dir}")
    contract = _read_json(tensor_root / "tensor_contract.json")
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
    sample_ids = _read_json(run_dir / "sample_ids.json")["validation"]
    loader = DataLoader(_subset_for_ids(dataset, sample_ids), batch_size=batch_size, shuffle=False, num_workers=0)
    model = _load_model(run_dir)
    metric_accumulator = FormalMetricAccumulator(stats, thresholds=_thresholds(run_dir), distribution_available=True)
    rows: list[dict[str, Any]] = []
    violation_counts: Counter[str] = Counter()
    expected_steps = observed_steps = active_flows = active_tasks = 0
    with torch.no_grad():
        for batch_index, cpu_batch in enumerate(loader):
            batch = move_nested_to_device(cpu_batch, torch.device("cpu"))
            prediction, audit = audit_rule_rollout(model, batch, n_rb=int(model.config.n_rb))
            metric_accumulator.update(prediction, batch["target"], batch["static"])
            expected_steps += int(audit["expected_step_count"])
            observed_steps += int(audit["observed_step_count"])
            active_flows += int(audit["active_flow_count"])
            active_tasks += int(audit["active_task_count"])
            violation_counts.update(audit["violation_counts"])
            rows.append({
                "batch_index": batch_index,
                "expected_step_count": audit["expected_step_count"],
                "observed_step_count": audit["observed_step_count"],
                "audit_passed": audit["audit_passed"],
                "active_flow_count": audit["active_flow_count"],
                "active_task_count": audit["active_task_count"],
                "violations": ";".join(sorted(audit["violation_counts"])),
            })
    metrics = metric_accumulator.finalize()
    summary = {
        "schema_version": "PI-JWM-formal-rule-rollout-audit-v1",
        "audit_passed": observed_steps == expected_steps and not violation_counts,
        "formal_performance_claim_ready": False,
        "run_dir": str(run_dir.resolve()),
        "seed": int(config["seed"]),
        "sample_count": len(sample_ids),
        "rule_step_summary": {
            "expected_step_count": expected_steps,
            "observed_step_count": observed_steps,
            "active_flow_count": active_flows,
            "active_task_count": active_tasks,
            "violation_counts": dict(sorted(violation_counts.items())),
        },
        "state_error": {
            horizon: {
                "node_x_mae": _metric(metrics, horizon, "state.node.x.mae"),
                "task_delay_mae": _metric(metrics, horizon, "state.task.delay.mae"),
            }
            for horizon in ("k=1", "k=2", "k=3")
        },
        "execution_policy": {
            "evaluation_device": "cpu",
            "gpu_execution": False,
            "locked_test_accessed": False,
            "result_boundary": "Nonlocked CPU replay audit of rule-enabled aggregate checkpoints only.",
        },
        "batch_reports": rows,
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--tensor-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    reports = [audit_run(run_dir, tensor_root=args.tensor_root) for run_dir in args.run_dir]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "schema_version": "PI-JWM-formal-rule-rollout-multiseed-audit-v1",
        "audit_passed": all(report["audit_passed"] for report in reports),
        "formal_performance_claim_ready": False,
        "seed_count": len(reports),
        "reports": reports,
        "execution_policy": {"evaluation_device": "cpu", "gpu_execution": False, "locked_test_accessed": False},
    }
    (args.output_dir / "rule_rollout_multiseed_audit.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "rule_rollout_batches.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["seed", "batch_index", "expected_step_count", "observed_step_count", "audit_passed", "active_flow_count", "active_task_count", "violations"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for report in reports:
            for row in report["batch_reports"]:
                writer.writerow({"seed": report["seed"], **row})
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if output["audit_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
