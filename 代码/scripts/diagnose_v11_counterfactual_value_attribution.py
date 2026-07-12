"""Counterfactual action-value attribution for PI-JWM v11 candidate policy work.

This CPU-only diagnostic asks which future-action value components would most
improve frozen PI-JWM rollout if they were repaired. It is not a deployable
method and must not be selected by test performance.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from pi_jwm.v6_data import V6WorldModelDataset, collate_v6_world_model_batch
from pi_jwm.v6_dual_graph import V6DualGraphBatch
from pi_jwm.v8_training import collect_v8_predictions
from pi_jwm.v11_rollout_value_calibrator import freeze_module

from diagnose_v11_bridge_operating_point import OperatingPoint, load_context, make_bridge_dataset
from evaluate_v11_adaptive_bridge import AdaptivePolicyBridgeDataset


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts/reports/v11_counterfactual_value_attribution_20260621"
ACTION_NAMES = ["offload_count", "rb_task_count", "rb_total", "cpu_task_count", "cpu_total", "return_count"]


@dataclass(frozen=True)
class CounterfactualMode:
    name: str
    description: str


def _rmse(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(values**2)))


def make_counterfactual_actions(baseline: np.ndarray, truth: np.ndarray, mode: str) -> np.ndarray:
    """Replace selected positive baseline action values with true values."""

    baseline = np.asarray(baseline, dtype=np.float32)
    truth = np.asarray(truth, dtype=np.float32)
    if baseline.shape != truth.shape or baseline.ndim != 4:
        raise ValueError("baseline and truth must share shape [sample, horizon, edge, action_dim]")
    result = baseline.copy()
    if mode == "baseline":
        return result
    if mode == "all_true":
        return truth.copy()
    if mode == "all_values":
        mask = baseline > 1e-9
        result[mask] = truth[mask]
        return result
    if mode.startswith("dim:"):
        dim = int(mode.split(":", 1)[1])
        _check_dim(dim, baseline.shape[-1])
        mask = baseline[..., dim] > 1e-9
        result[..., dim][mask] = truth[..., dim][mask]
        return result
    if mode.startswith("step:"):
        step = int(mode.split(":", 1)[1])
        _check_step(step, baseline.shape[1])
        mask = baseline[:, step] > 1e-9
        result[:, step][mask] = truth[:, step][mask]
        return result
    if mode.startswith("step_dim:"):
        _, step_text, dim_text = mode.split(":")
        step = int(step_text)
        dim = int(dim_text)
        _check_step(step, baseline.shape[1])
        _check_dim(dim, baseline.shape[-1])
        mask = baseline[:, step, :, dim] > 1e-9
        result[:, step, :, dim][mask] = truth[:, step, :, dim][mask]
        return result
    raise ValueError(f"unknown counterfactual mode: {mode}")


def _check_step(step: int, horizon: int) -> None:
    if step < 0 or step >= horizon:
        raise ValueError(f"step out of range: {step}")


def _check_dim(dim: int, action_dim: int) -> None:
    if dim < 0 or dim >= action_dim:
        raise ValueError(f"action dim out of range: {dim}")


class RawFutureActionDataset(Dataset):
    def __init__(self, base_dataset: V6WorldModelDataset, raw_actions: np.ndarray, stats: dict) -> None:
        self.base_dataset = base_dataset
        self.raw_actions = np.asarray(raw_actions, dtype=np.float32)
        self.stats = stats
        if len(base_dataset) != self.raw_actions.shape[0]:
            raise ValueError("base_dataset length must match raw action sample count")

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, item: int):
        world_batch, target = self.base_dataset[item]
        normalized_future = self.normalize_future_actions(torch.from_numpy(self.raw_actions[item]))
        bridged = V6DualGraphBatch(
            node_history=world_batch.node_history,
            physical_edge_history=world_batch.physical_edge_history,
            info_edge_history=world_batch.info_edge_history,
            action_history=world_batch.action_history,
            future_actions=normalized_future,
            task_history=world_batch.task_history,
            link_rate_baseline=world_batch.link_rate_baseline,
        )
        return bridged, target

    def normalize_future_actions(self, raw_future: torch.Tensor) -> torch.Tensor:
        mean, std = self.stats["edge_a_future"]
        mean_t = torch.as_tensor(mean[0], dtype=torch.float32)
        std_t = torch.as_tensor(std[0], dtype=torch.float32)
        return ((raw_future - mean_t) / std_t).to(torch.float32)


def raw_future_from_normalized(normalized_future: torch.Tensor, stats: dict) -> np.ndarray:
    mean, std = stats["edge_a_future"]
    mean_t = torch.as_tensor(mean[0], dtype=torch.float32)
    std_t = torch.as_tensor(std[0], dtype=torch.float32)
    return (normalized_future * std_t + mean_t).to(torch.float32).numpy()


def collect_raw_actions(dataset, stats: dict) -> tuple[np.ndarray, np.ndarray]:
    predicted_rows = []
    truth_rows = []
    for idx in range(len(dataset)):
        bridged_batch, _ = dataset[idx]
        base_batch, _ = dataset.base_dataset[idx]
        predicted_rows.append(raw_future_from_normalized(bridged_batch.future_actions, stats))
        truth_rows.append(raw_future_from_normalized(base_batch.future_actions, stats))
    return np.stack(predicted_rows, axis=0), np.stack(truth_rows, axis=0)


def active_rate_row(mode: str, split: str, predictions: dict[str, np.ndarray], baseline_rmse: float) -> dict:
    truth = predictions["link_rate_true"].squeeze(-1)
    pred = predictions["link_rate_pred"].squeeze(-1)
    active = predictions["link_activity_true"].squeeze(-1) > 0.5
    active_error = pred[active] - truth[active]
    link_error = predictions["link_rate_pred"] - predictions["link_rate_true"]
    active_rmse = _rmse(active_error)
    return {
        "mode": mode,
        "split": split,
        "active_count": int(active.sum()),
        "active_rate_rmse": active_rmse,
        "active_rate_mae": float(np.mean(np.abs(active_error))) if active_error.size else float("nan"),
        "link_rmse": _rmse(link_error),
        "improvement_vs_baseline": float(baseline_rmse - active_rmse) if np.isfinite(baseline_rmse) else float("nan"),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def default_modes(horizon: int, action_dim: int) -> list[CounterfactualMode]:
    modes = [
        CounterfactualMode("baseline", "current deployable baseline actions"),
        CounterfactualMode("all_values", "replace all positive predicted action values with true values, preserving predicted activity"),
        CounterfactualMode("all_true", "replace the full future-action tensor with true future actions"),
    ]
    for dim in range(action_dim):
        name = ACTION_NAMES[dim] if dim < len(ACTION_NAMES) else f"dim_{dim}"
        modes.append(CounterfactualMode(f"dim:{dim}", f"replace positive predicted values for action dim {dim} ({name})"))
    for step in range(horizon):
        modes.append(CounterfactualMode(f"step:{step}", f"replace positive predicted values for horizon step {step}"))
    for step in range(horizon):
        for dim in range(action_dim):
            name = ACTION_NAMES[dim] if dim < len(ACTION_NAMES) else f"dim_{dim}"
            modes.append(CounterfactualMode(f"step_dim:{step}:{dim}", f"replace positive predicted values for step {step}, dim {dim} ({name})"))
    return modes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-experiment-dir", type=Path, default=PROJECT_ROOT / "artifacts/experiments/pi_jwm_v9_expanded_v2_gpu_20260619/v2_hurdle_baseline")
    parser.add_argument("--world-checkpoint", type=Path, default=PROJECT_ROOT / "artifacts/experiments/pi_jwm_v9_expanded_v2_gpu_20260619/v2_hurdle_baseline/checkpoints/v8_dual_best.pt")
    parser.add_argument("--policy-checkpoint", type=Path, default=PROJECT_ROOT / "artifacts/experiments/pi_jwm_v10_policy_bridge_gpu_20260620/v10_action_policy_bc/checkpoints/v7_action_policy_cross_attention_best.pt")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--split", choices=("val", "test", "both"), default="both")
    parser.add_argument("--policy-threshold", type=float, default=0.4)
    parser.add_argument("--value-scale", type=float, default=1.0)
    parser.add_argument("--baseline-mode", choices=("old", "adaptive"), default="adaptive")
    parser.add_argument("--new-policy-threshold", type=float, default=0.37)
    parser.add_argument("--new-value-scale", type=float, default=1.06)
    parser.add_argument("--gate-feature", choices=("step_rb_total", "step_cpu_total", "step_rb_cpu_total", "step_active_count"), default="step_rb_cpu_total")
    parser.add_argument("--gate-threshold", type=float, default=450.0)
    parser.add_argument("--value-codebook-size", type=int, default=9)
    return parser.parse_args()


def evaluate_counterfactuals(args: argparse.Namespace) -> dict:
    device = torch.device("cpu")
    context = load_context(args, device)
    summary, arrays, stats, world_model, policy_model, action_scale, value_vocab, splits = context
    world_model = freeze_module(world_model)
    policy_model = freeze_module(policy_model)
    config = summary["config"]
    split_names = ["val", "test"] if args.split == "both" else [args.split]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    mode_rows: list[dict] = []
    for split_name in split_names:
        indices = splits[split_name]
        base_dataset = V6WorldModelDataset(arrays, indices, stats)
        old_dataset = make_bridge_dataset(
            arrays,
            indices,
            stats,
            policy_model,
            action_scale,
            value_vocab,
            device,
            OperatingPoint("baseline", args.policy_threshold, args.value_scale, value_codebook_size=args.value_codebook_size),
            splits["train"],
        )
        if args.baseline_mode == "old":
            bridge_dataset = old_dataset
        else:
            new_dataset = make_bridge_dataset(
                arrays,
                indices,
                stats,
                policy_model,
                action_scale,
                value_vocab,
                device,
                OperatingPoint(
                    "new",
                    args.new_policy_threshold,
                    args.new_value_scale,
                    value_codebook_size=args.value_codebook_size,
                ),
                splits["train"],
            )
            bridge_dataset = AdaptivePolicyBridgeDataset(
                base_dataset,
                old_dataset,
                new_dataset,
                stats,
                args.gate_feature,
                args.gate_threshold,
            )
        baseline_actions, true_actions = collect_raw_actions(bridge_dataset, stats)
        horizon, _, action_dim = baseline_actions.shape[1:]
        modes = default_modes(horizon, action_dim)
        baseline_rmse = float("nan")
        split_rows = []
        for mode in modes:
            raw_actions = make_counterfactual_actions(baseline_actions, true_actions, mode.name)
            loader = DataLoader(
                RawFutureActionDataset(base_dataset, raw_actions, stats),
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=collate_v6_world_model_batch,
            )
            predictions = collect_v8_predictions(
                world_model,
                loader,
                device,
                stats,
                rate_output_mode=config.get("rate_output_mode", "main"),
                inactive_rate_value=float(config.get("inactive_rate_value", 0.0)),
            )
            row = active_rate_row(mode.name, split_name, predictions, baseline_rmse)
            row["description"] = mode.description
            if mode.name == "baseline":
                baseline_rmse = float(row["active_rate_rmse"])
                row["improvement_vs_baseline"] = 0.0
            split_rows.append(row)
            all_rows.append(row)
        split_rows.sort(key=lambda r: (-float(r["improvement_vs_baseline"]), str(r["mode"])))
        write_csv(args.output_dir / f"{split_name}_counterfactual_ranked.csv", split_rows)
        mode_rows.extend({"split": split_name, **mode.__dict__} for mode in modes)

    val_rows = [row for row in all_rows if row["split"] == "val"]
    val_ranked = sorted(val_rows, key=lambda r: (-float(r["improvement_vs_baseline"]), str(r["mode"])))
    write_csv(args.output_dir / "counterfactual_all_rows.csv", all_rows)
    write_csv(args.output_dir / "counterfactual_val_ranked.csv", val_ranked)
    write_csv(args.output_dir / "counterfactual_modes.csv", mode_rows)
    report_lines = [
        "# PI-JWM v11 Counterfactual Value Attribution",
        "",
        "CPU-only diagnostic. Counterfactual modes with true values are not deployable; they identify which action-value modules are worth training.",
        "",
        "| rank | mode | val_active_rmse | improvement_vs_baseline | description |",
        "|---:|---|---:|---:|---|",
    ]
    for rank, row in enumerate(val_ranked[:20], start=1):
        report_lines.append(
            f"| {rank} | {row['mode']} | {row['active_rate_rmse']:.6f} | {row['improvement_vs_baseline']:.6f} | {row['description']} |"
        )
    actionable = [
        row for row in val_ranked
        if row["mode"] not in {"baseline", "all_true", "all_values"} and float(row["improvement_vs_baseline"]) > 0.0
    ]
    if val_ranked:
        best = val_ranked[0]
        best_actionable = actionable[0] if actionable else None
        report_lines.extend(
            [
                "",
                "## Recommendation Signal",
                "",
                f"Best validation upper-bound mode: {best['mode']} with active-rate RMSE {best['active_rate_rmse']:.6f}.",
            ]
        )
        if best_actionable is not None:
            report_lines.append(
                f"Best actionable module: {best_actionable['mode']} with active-rate RMSE "
                f"{best_actionable['active_rate_rmse']:.6f} and improvement "
                f"{best_actionable['improvement_vs_baseline']:.6f}."
            )
            report_lines.append("Train/repair this action-value module first; do not expand broad policy training before this signal is reviewed.")
        else:
            report_lines.append("No actionable non-oracle value module improved validation; do not expand training.")
    (args.output_dir / "counterfactual_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    summary_out = {
        "framework": "PI-JWM",
        "candidate": "v11",
        "mode": "counterfactual_value_attribution",
        "baseline_mode": args.baseline_mode,
        "output_dir": str(args.output_dir),
        "rows": all_rows,
        "val_top": val_ranked[:20],
        "best_actionable": actionable[0] if actionable else None,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary_out, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary_out


def main() -> None:
    args = parse_args()
    result = evaluate_counterfactuals(args)
    print(json.dumps({"output_dir": result["output_dir"], "val_top": result["val_top"][:10]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
