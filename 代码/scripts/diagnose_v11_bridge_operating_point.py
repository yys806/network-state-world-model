"""Diagnose a PI-JWM v11 candidate bridge operating point against a baseline."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from pi_jwm.v6_data import V6WorldModelDataset, collate_v6_world_model_batch, load_world_model_arrays, make_normalization_stats
from pi_jwm.v8_training import collect_v8_predictions

from evaluate_v10_policy_bridge import (
    PolicyBridgeDataset,
    make_action_decoder_config,
    make_action_value_decoder_config,
    load_policy,
)
from evaluate_v11_adaptive_bridge import AdaptivePolicyBridgeDataset
from evaluate_v9_action_ablation import load_model_for_experiment, resolve_project_path
from run_v7_action_policy import V7ActionPolicyDataset
from run_world_model_v8_full_training import add_event_memory_features, resolve_seed_splits


@dataclass(frozen=True)
class OperatingPoint:
    name: str
    threshold: float
    value_scale: float
    value_decoder: str = "train_codebook_quantile"
    value_codebook_size: int = 9


def _rmse(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(values**2)))


def _bin_label(left: float, right: float) -> str:
    left_text = f"{left:g}"
    right_text = "inf" if np.isinf(right) else f"{right:g}"
    return f"{left_text}-{right_text}"


def summarize_rate_bins(
    truth: np.ndarray,
    active: np.ndarray,
    old_pred: np.ndarray,
    new_pred: np.ndarray,
    bins: list[float],
) -> list[dict[str, float | int | str]]:
    truth = np.asarray(truth, dtype=np.float64)
    active = np.asarray(active, dtype=bool)
    old_pred = np.asarray(old_pred, dtype=np.float64)
    new_pred = np.asarray(new_pred, dtype=np.float64)
    rows = []
    for left, right in zip(bins[:-1], bins[1:]):
        mask = active & (truth >= left) & (truth < right)
        old_error = old_pred[mask] - truth[mask]
        new_error = new_pred[mask] - truth[mask]
        old_rmse = _rmse(old_pred[mask] - truth[mask])
        new_rmse = _rmse(new_pred[mask] - truth[mask])
        old_sse = float(np.sum(old_error**2))
        new_sse = float(np.sum(new_error**2))
        rows.append(
            {
                "bin": _bin_label(float(left), float(right)),
                "count": int(mask.sum()),
                "old_rmse": old_rmse,
                "new_rmse": new_rmse,
                "delta_rmse": float(new_rmse - old_rmse),
                "old_sse": old_sse,
                "new_sse": new_sse,
                "delta_sse": float(new_sse - old_sse),
                "true_mean": float(np.mean(truth[mask])) if mask.any() else float("nan"),
                "old_pred_mean": float(np.mean(old_pred[mask])) if mask.any() else float("nan"),
                "new_pred_mean": float(np.mean(new_pred[mask])) if mask.any() else float("nan"),
            }
        )
    return rows


def top_active_improvements(
    truth: np.ndarray,
    active: np.ndarray,
    old_pred: np.ndarray,
    new_pred: np.ndarray,
    top_k: int = 25,
) -> list[dict[str, float | int]]:
    truth = np.asarray(truth, dtype=np.float64)
    active = np.asarray(active, dtype=bool)
    old_pred = np.asarray(old_pred, dtype=np.float64)
    new_pred = np.asarray(new_pred, dtype=np.float64)
    old_error = (old_pred - truth) ** 2
    new_error = (new_pred - truth) ** 2
    reduction = old_error - new_error
    coordinates = np.argwhere(active)
    rows = []
    for sample, step, edge in coordinates:
        rows.append(
            {
                "sample": int(sample),
                "step": int(step),
                "edge": int(edge),
                "true_rate": float(truth[sample, step, edge]),
                "old_pred": float(old_pred[sample, step, edge]),
                "new_pred": float(new_pred[sample, step, edge]),
                "squared_error_reduction": float(reduction[sample, step, edge]),
            }
        )
    rows.sort(key=lambda row: (-float(row["squared_error_reduction"]), row["sample"], row["step"], row["edge"]))
    return rows[:top_k]


def top_active_regressions(
    truth: np.ndarray,
    active: np.ndarray,
    old_pred: np.ndarray,
    new_pred: np.ndarray,
    top_k: int = 25,
) -> list[dict[str, float | int]]:
    truth = np.asarray(truth, dtype=np.float64)
    active = np.asarray(active, dtype=bool)
    old_pred = np.asarray(old_pred, dtype=np.float64)
    new_pred = np.asarray(new_pred, dtype=np.float64)
    old_error = (old_pred - truth) ** 2
    new_error = (new_pred - truth) ** 2
    increase = new_error - old_error
    coordinates = np.argwhere(active)
    rows = []
    for sample, step, edge in coordinates:
        rows.append(
            {
                "sample": int(sample),
                "step": int(step),
                "edge": int(edge),
                "true_rate": float(truth[sample, step, edge]),
                "old_pred": float(old_pred[sample, step, edge]),
                "new_pred": float(new_pred[sample, step, edge]),
                "squared_error_increase": float(increase[sample, step, edge]),
            }
        )
    rows.sort(key=lambda row: (-float(row["squared_error_increase"]), row["sample"], row["step"], row["edge"]))
    return rows[:top_k]


def summarize_step_rmse(truth: np.ndarray, active: np.ndarray, old_pred: np.ndarray, new_pred: np.ndarray) -> list[dict]:
    rows = []
    for step in range(truth.shape[1]):
        mask = active[:, step]
        old_rmse = _rmse(old_pred[:, step][mask] - truth[:, step][mask])
        new_rmse = _rmse(new_pred[:, step][mask] - truth[:, step][mask])
        rows.append(
            {
                "step": step,
                "active_count": int(mask.sum()),
                "old_rmse": old_rmse,
                "new_rmse": new_rmse,
                "delta_rmse": float(new_rmse - old_rmse),
            }
        )
    return rows


def summarize_three_way_overall(
    truth: np.ndarray,
    active: np.ndarray,
    old_pred: np.ndarray,
    global_pred: np.ndarray,
    adaptive_pred: np.ndarray,
    split: str,
) -> dict[str, float | int | str]:
    truth = np.asarray(truth, dtype=np.float64)
    active = np.asarray(active, dtype=bool)
    old_rmse = _rmse((np.asarray(old_pred, dtype=np.float64) - truth)[active])
    global_rmse = _rmse((np.asarray(global_pred, dtype=np.float64) - truth)[active])
    adaptive_rmse = _rmse((np.asarray(adaptive_pred, dtype=np.float64) - truth)[active])
    return {
        "split": split,
        "old_rmse": old_rmse,
        "global_rmse": global_rmse,
        "adaptive_rmse": adaptive_rmse,
        "global_vs_old_delta": float(global_rmse - old_rmse),
        "adaptive_vs_old_delta": float(adaptive_rmse - old_rmse),
        "adaptive_vs_global_delta": float(adaptive_rmse - global_rmse),
        "active_count": int(active.sum()),
    }


def summarize_three_way_bins(
    truth: np.ndarray,
    active: np.ndarray,
    old_pred: np.ndarray,
    global_pred: np.ndarray,
    adaptive_pred: np.ndarray,
    bins: list[float],
) -> list[dict[str, float | int | str]]:
    rows = []
    for left, right in zip(bins[:-1], bins[1:]):
        mask = active & (truth >= left) & (truth < right)
        old_rmse = _rmse(old_pred[mask] - truth[mask])
        global_rmse = _rmse(global_pred[mask] - truth[mask])
        adaptive_rmse = _rmse(adaptive_pred[mask] - truth[mask])
        rows.append(
            {
                "bin": _bin_label(float(left), float(right)),
                "count": int(mask.sum()),
                "old_rmse": old_rmse,
                "global_rmse": global_rmse,
                "adaptive_rmse": adaptive_rmse,
                "global_vs_old_delta": float(global_rmse - old_rmse),
                "adaptive_vs_old_delta": float(adaptive_rmse - old_rmse),
                "adaptive_vs_global_delta": float(adaptive_rmse - global_rmse),
            }
        )
    return rows


def summarize_three_way_steps(
    truth: np.ndarray,
    active: np.ndarray,
    old_pred: np.ndarray,
    global_pred: np.ndarray,
    adaptive_pred: np.ndarray,
) -> list[dict[str, float | int]]:
    rows = []
    for step in range(truth.shape[1]):
        mask = active[:, step]
        old_rmse = _rmse(old_pred[:, step][mask] - truth[:, step][mask])
        global_rmse = _rmse(global_pred[:, step][mask] - truth[:, step][mask])
        adaptive_rmse = _rmse(adaptive_pred[:, step][mask] - truth[:, step][mask])
        rows.append(
            {
                "step": step,
                "active_count": int(mask.sum()),
                "old_rmse": old_rmse,
                "global_rmse": global_rmse,
                "adaptive_rmse": adaptive_rmse,
                "global_vs_old_delta": float(global_rmse - old_rmse),
                "adaptive_vs_old_delta": float(adaptive_rmse - old_rmse),
                "adaptive_vs_global_delta": float(adaptive_rmse - global_rmse),
            }
        )
    return rows


def summarize_action_totals(old_actions: np.ndarray, new_actions: np.ndarray) -> list[dict]:
    names = ["offload_count", "rb_task_count", "rb_total", "cpu_task_count", "cpu_total", "return_count"]
    rows = []
    for dim, name in enumerate(names[: old_actions.shape[-1]]):
        old_total = old_actions[..., dim].sum(axis=(1, 2))
        new_total = new_actions[..., dim].sum(axis=(1, 2))
        rows.append(
            {
                "action_dim": dim,
                "action_name": name,
                "old_mean_total": float(np.mean(old_total)),
                "new_mean_total": float(np.mean(new_total)),
                "delta_mean_total": float(np.mean(new_total - old_total)),
                "old_nonzero": int(np.count_nonzero(old_actions[..., dim] > 1e-9)),
                "new_nonzero": int(np.count_nonzero(new_actions[..., dim] > 1e-9)),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-experiment-dir", type=Path, default=PROJECT_ROOT / "artifacts/experiments/pi_jwm_v9_expanded_v2_gpu_20260619/v2_hurdle_baseline")
    parser.add_argument("--world-checkpoint", type=Path, default=PROJECT_ROOT / "artifacts/experiments/pi_jwm_v9_expanded_v2_gpu_20260619/v2_hurdle_baseline/checkpoints/v8_dual_best.pt")
    parser.add_argument("--policy-checkpoint", type=Path, default=PROJECT_ROOT / "artifacts/experiments/pi_jwm_v10_policy_bridge_gpu_20260620/v10_action_policy_bc/checkpoints/v7_action_policy_cross_attention_best.pt")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts/reports/v11_cpu_multi_scheme_20260621/operating_point_diagnosis")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--split", choices=("val", "test", "both"), default="both")
    parser.add_argument("--old-threshold", type=float, default=0.4)
    parser.add_argument("--old-scale", type=float, default=1.0)
    parser.add_argument("--new-threshold", type=float, default=0.37)
    parser.add_argument("--new-scale", type=float, default=1.06)
    parser.add_argument("--adaptive-gate-feature", default="step_rb_cpu_total")
    parser.add_argument("--adaptive-gate-threshold", type=float, default=450.0)
    parser.add_argument("--top-k", type=int, default=25)
    return parser.parse_args()


def load_context(args: argparse.Namespace, device: torch.device):
    summary = json.loads((args.world_experiment_dir / "v8_full_training_summary.json").read_text(encoding="utf-8"))
    dataset_dir = resolve_project_path(summary["dataset_dir"])
    arrays = load_world_model_arrays(dataset_dir)
    if summary["config"].get("use_event_memory_features", False):
        arrays = add_event_memory_features(arrays)
    split = summary["split_seed_spec"]
    train_idx, val_idx, test_idx, _ = resolve_seed_splits(
        arrays["sample_seed"],
        train_seeds=split["train_seeds"],
        val_seeds=split["val_seeds"],
        test_seeds=split["test_seeds"],
    )
    stats = make_normalization_stats(arrays, train_idx)
    world_model = load_model_for_experiment(summary, arrays, args.world_checkpoint, device)
    policy_model, action_scale, learned_threshold, value_vocab = load_policy(args.policy_checkpoint, device)
    return summary, arrays, stats, world_model, policy_model, action_scale, value_vocab, {"val": val_idx, "test": test_idx, "train": train_idx}


def make_bridge_dataset(
    arrays: dict,
    indices: np.ndarray,
    stats: dict,
    policy_model,
    action_scale: np.ndarray,
    value_vocab: dict | None,
    device: torch.device,
    point: OperatingPoint,
    train_idx: np.ndarray,
) -> PolicyBridgeDataset:
    base = V6WorldModelDataset(arrays, indices, stats)
    policy = V7ActionPolicyDataset(arrays, indices, stats, action_scale)
    decoder_config = make_action_decoder_config("threshold", {}, 0.5)
    value_config = make_action_value_decoder_config(
        point.value_decoder,
        arrays,
        train_idx,
        value_quantile=0.75,
        value_codebook_size=point.value_codebook_size,
        value_scale=point.value_scale,
    )
    return PolicyBridgeDataset(
        base,
        policy,
        policy_model,
        action_scale,
        stats,
        device,
        point.threshold,
        "true_first_pred_rest",
        decoder_config,
        "policy",
        value_config,
        value_vocab,
    )


def collect_raw_actions(dataset: PolicyBridgeDataset) -> np.ndarray:
    rows = []
    for idx in range(len(dataset)):
        world_batch, _ = dataset.base_dataset[idx]
        policy_batch, _ = dataset.policy_dataset[idx]
        true_future = dataset.raw_future_from_normalized(world_batch.future_actions)
        predicted = dataset.generate_raw_actions(policy_batch, world_batch, true_future)
        predicted[0] = true_future[0]
        rows.append(predicted.numpy())
    return np.stack(rows, axis=0)


def run_split(args: argparse.Namespace, split_name: str, context: tuple, device: torch.device) -> dict:
    summary, arrays, stats, world_model, policy_model, action_scale, value_vocab, splits = context
    old_point = OperatingPoint("old", args.old_threshold, args.old_scale)
    new_point = OperatingPoint("new", args.new_threshold, args.new_scale)
    indices = splits[split_name]
    old_dataset = make_bridge_dataset(arrays, indices, stats, policy_model, action_scale, value_vocab, device, old_point, splits["train"])
    new_dataset = make_bridge_dataset(arrays, indices, stats, policy_model, action_scale, value_vocab, device, new_point, splits["train"])
    base_dataset = V6WorldModelDataset(arrays, indices, stats)
    adaptive_dataset = AdaptivePolicyBridgeDataset(
        base_dataset,
        old_dataset,
        new_dataset,
        stats,
        args.adaptive_gate_feature,
        args.adaptive_gate_threshold,
    )
    old_loader = DataLoader(old_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_v6_world_model_batch)
    new_loader = DataLoader(new_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_v6_world_model_batch)
    adaptive_loader = DataLoader(adaptive_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_v6_world_model_batch)
    config = summary["config"]
    old_pred = collect_v8_predictions(
        world_model,
        old_loader,
        device,
        stats,
        rate_output_mode=config.get("rate_output_mode", "main"),
        inactive_rate_value=float(config.get("inactive_rate_value", 0.0)),
    )
    new_pred = collect_v8_predictions(
        world_model,
        new_loader,
        device,
        stats,
        rate_output_mode=config.get("rate_output_mode", "main"),
        inactive_rate_value=float(config.get("inactive_rate_value", 0.0)),
    )
    adaptive_pred = collect_v8_predictions(
        world_model,
        adaptive_loader,
        device,
        stats,
        rate_output_mode=config.get("rate_output_mode", "main"),
        inactive_rate_value=float(config.get("inactive_rate_value", 0.0)),
    )
    truth = old_pred["link_rate_true"].squeeze(-1)
    active = old_pred["link_activity_true"].squeeze(-1) > 0.5
    old_rate = old_pred["link_rate_pred"].squeeze(-1)
    new_rate = new_pred["link_rate_pred"].squeeze(-1)
    adaptive_rate = adaptive_pred["link_rate_pred"].squeeze(-1)
    old_actions = collect_raw_actions(old_dataset)
    new_actions = collect_raw_actions(new_dataset)
    return {
        "split": split_name,
        "truth": truth,
        "active": active,
        "old_rate": old_rate,
        "new_rate": new_rate,
        "adaptive_rate": adaptive_rate,
        "three_way_overall": summarize_three_way_overall(truth, active, old_rate, new_rate, adaptive_rate, split_name),
        "three_way_steps": summarize_three_way_steps(truth, active, old_rate, new_rate, adaptive_rate),
        "three_way_bins": summarize_three_way_bins(
            truth,
            active,
            old_rate,
            new_rate,
            adaptive_rate,
            [0.0, 50.0, 100.0, 150.0, 250.0, float("inf")],
        ),
        "adaptive_gate": {
            "true_count": int(adaptive_dataset.gate_true_count),
            "total_count": int(adaptive_dataset.gate_total_count),
            "fraction": float(adaptive_dataset.gate_true_count / max(adaptive_dataset.gate_total_count, 1)),
        },
        "step_rows": summarize_step_rmse(truth, active, old_rate, new_rate),
        "bin_rows": summarize_rate_bins(truth, active, old_rate, new_rate, [0.0, 50.0, 100.0, 150.0, 250.0, float("inf")]),
        "top_rows": top_active_improvements(truth, active, old_rate, new_rate, args.top_k),
        "regression_rows": top_active_regressions(truth, active, old_rate, new_rate, args.top_k),
        "action_rows": summarize_action_totals(old_actions, new_actions),
    }


def main() -> None:
    args = parse_args()
    device = torch.device("cpu")
    context = load_context(args, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    splits = ["val", "test"] if args.split == "both" else [args.split]
    summary_rows = []
    three_way_rows = []
    for split_name in splits:
        result = run_split(args, split_name, context, device)
        prefix = args.output_dir / split_name
        write_csv(prefix.with_name(f"{split_name}_step_rmse.csv"), result["step_rows"])
        write_csv(prefix.with_name(f"{split_name}_rate_bins.csv"), result["bin_rows"])
        write_csv(prefix.with_name(f"{split_name}_top_improvements.csv"), result["top_rows"])
        write_csv(prefix.with_name(f"{split_name}_top_regressions.csv"), result["regression_rows"])
        write_csv(prefix.with_name(f"{split_name}_action_totals.csv"), result["action_rows"])
        write_csv(prefix.with_name(f"{split_name}_three_way_steps.csv"), result["three_way_steps"])
        write_csv(prefix.with_name(f"{split_name}_three_way_bins.csv"), result["three_way_bins"])
        old_active_rmse = _rmse((result["old_rate"] - result["truth"])[result["active"]])
        new_active_rmse = _rmse((result["new_rate"] - result["truth"])[result["active"]])
        adaptive_active_rmse = _rmse((result["adaptive_rate"] - result["truth"])[result["active"]])
        summary_rows.append(
            {
                "split": split_name,
                "old_active_rmse": old_active_rmse,
                "new_active_rmse": new_active_rmse,
                "delta_rmse": float(new_active_rmse - old_active_rmse),
                "active_count": int(result["active"].sum()),
            }
        )
        three_way_row = dict(result["three_way_overall"])
        three_way_row["adaptive_gate_fraction"] = result["adaptive_gate"]["fraction"]
        three_way_rows.append(three_way_row)
    write_csv(args.output_dir / "summary.csv", summary_rows)
    write_csv(args.output_dir / "three_way_summary.csv", three_way_rows)
    report = [
        "# PI-JWM v11 Candidate Operating Point Diagnosis",
        "",
        f"Old point: threshold={args.old_threshold:g}, scale={args.old_scale:g}.",
        f"New point: threshold={args.new_threshold:g}, scale={args.new_scale:g}.",
        "",
        "| split | old_active_rmse | new_active_rmse | delta_rmse | active_count |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        report.append(
            f"| {row['split']} | {row['old_active_rmse']:.6f} | {row['new_active_rmse']:.6f} | {row['delta_rmse']:.6f} | {row['active_count']} |"
        )
    report.extend(
        [
        "",
        "See per-split CSV files for step, rate-bin, top-improvement, and action-total diagnostics.",
        "Adaptive three-way diagnostics are in three_way_summary.csv and per-split three_way CSV files.",
    ]
    )
    (args.output_dir / "diagnosis_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary_rows, "output_dir": str(args.output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
