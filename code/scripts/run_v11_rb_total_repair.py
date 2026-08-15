"""CPU-only rb_total repair probes for PI-JWM v11 candidate policy work."""

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

from pi_jwm.v6_data import V6WorldModelDataset, collate_v6_world_model_batch
from pi_jwm.v6_metrics import activity_metrics
from pi_jwm.v8_training import collect_v8_predictions
from pi_jwm.v11_rollout_value_calibrator import freeze_module

from diagnose_v11_bridge_operating_point import OperatingPoint, load_context, make_bridge_dataset
from diagnose_v11_counterfactual_value_attribution import RawFutureActionDataset, collect_raw_actions
from evaluate_v11_adaptive_bridge import AdaptivePolicyBridgeDataset


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts/experiments/pi_jwm_v11_rb_total_repair_cpu_20260621"
RB_DIM = 2
CURRENT_CPU_BEST_VAL_ACTIVE_RMSE = 232.26805853434024


@dataclass(frozen=True)
class RbRepairRule:
    name: str
    mode: str
    scale: float = 1.0
    threshold: float = 0.0
    step_scales: np.ndarray | None = None
    bin_edges: np.ndarray | None = None
    bin_values: np.ndarray | None = None


def _rmse(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(values**2)))


def fit_global_scale_rule(baseline: np.ndarray, truth: np.ndarray, name: str = "global_scale") -> RbRepairRule:
    pred = np.asarray(baseline[..., RB_DIM], dtype=np.float64)
    target = np.asarray(truth[..., RB_DIM], dtype=np.float64)
    mask = pred > 1e-9
    scale = _least_squares_scale(pred[mask], target[mask]) if mask.any() else 1.0
    return RbRepairRule(name=name, mode="global_scale", scale=float(scale))


def fit_step_scale_rule(baseline: np.ndarray, truth: np.ndarray, name: str = "step_scale") -> RbRepairRule:
    pred = np.asarray(baseline[..., RB_DIM], dtype=np.float64)
    target = np.asarray(truth[..., RB_DIM], dtype=np.float64)
    step_scales = np.ones(pred.shape[1], dtype=np.float32)
    for step in range(pred.shape[1]):
        mask = pred[:, step] > 1e-9
        if mask.any():
            step_scales[step] = float(_least_squares_scale(pred[:, step][mask], target[:, step][mask]))
    return RbRepairRule(name=name, mode="step_scale", step_scales=step_scales)


def fit_step_bin_median_rule(
    baseline: np.ndarray,
    truth: np.ndarray,
    bin_edges: np.ndarray,
    name: str = "step_bin_median",
) -> RbRepairRule:
    pred = np.asarray(baseline[..., RB_DIM], dtype=np.float64)
    target = np.asarray(truth[..., RB_DIM], dtype=np.float64)
    bin_edges = np.asarray(bin_edges, dtype=np.float32)
    bin_count = int(len(bin_edges) + 1)
    values = np.full((pred.shape[1], bin_count), np.nan, dtype=np.float32)
    fallback = fit_step_scale_rule(baseline, truth).step_scales
    for step in range(pred.shape[1]):
        for bin_idx in range(bin_count):
            lower = -np.inf if bin_idx == 0 else float(bin_edges[bin_idx - 1])
            upper = np.inf if bin_idx == len(bin_edges) else float(bin_edges[bin_idx])
            mask = (pred[:, step] > 1e-9) & (pred[:, step] >= lower) & (pred[:, step] < upper)
            if mask.any():
                values[step, bin_idx] = float(np.median(target[:, step][mask]))
        missing = ~np.isfinite(values[step])
        if np.any(missing):
            centers = np.concatenate([[0.0], bin_edges.astype(np.float32), [float(bin_edges[-1] * 1.5 if len(bin_edges) else 1.0)]])
            values[step, missing] = centers[:bin_count][missing] * float(fallback[step])
    return RbRepairRule(name=name, mode="step_bin_value", bin_edges=bin_edges, bin_values=values)


def _least_squares_scale(pred: np.ndarray, target: np.ndarray) -> float:
    denom = float(np.sum(pred * pred))
    if denom <= 1e-12:
        return 1.0
    return float(np.clip(np.sum(pred * target) / denom, 0.0, 5.0))


def apply_rb_total_repair(actions: np.ndarray, rule: RbRepairRule, preserve_step0: bool = True) -> np.ndarray:
    repaired = np.asarray(actions, dtype=np.float32).copy()
    rb = repaired[..., RB_DIM]
    active = rb > 1e-9
    if rule.mode == "identity":
        return repaired
    if rule.mode == "global_scale":
        rb[active] = rb[active] * float(rule.scale)
    elif rule.mode == "step_scale":
        if rule.step_scales is None:
            raise ValueError("step_scale rule requires step_scales")
        scales = np.asarray(rule.step_scales, dtype=np.float32)
        if scales.shape != (repaired.shape[1],):
            raise ValueError("step_scales must match horizon")
        for step in range(repaired.shape[1]):
            step_active = active[:, step]
            rb[:, step][step_active] = rb[:, step][step_active] * float(scales[step])
    elif rule.mode == "step_bin_value":
        if rule.bin_edges is None or rule.bin_values is None:
            raise ValueError("step_bin_value rule requires bin_edges and bin_values")
        edges = np.asarray(rule.bin_edges, dtype=np.float32)
        values = np.asarray(rule.bin_values, dtype=np.float32)
        if values.shape[0] != repaired.shape[1] or values.shape[1] != len(edges) + 1:
            raise ValueError("bin_values must have shape [horizon, bin_count]")
        for step in range(repaired.shape[1]):
            idx = np.searchsorted(edges, rb[:, step], side="right")
            step_active = active[:, step]
            rb[:, step][step_active] = values[step][idx[step_active]]
    elif rule.mode == "zero_below_threshold":
        remove = active & (rb < float(rule.threshold))
        rb[remove] = 0.0
    else:
        raise ValueError(f"unknown rb repair mode: {rule.mode}")
    if preserve_step0:
        repaired[:, 0, :, RB_DIM] = actions[:, 0, :, RB_DIM]
    repaired[..., RB_DIM] = np.where(actions[..., RB_DIM] > 1e-9, np.clip(repaired[..., RB_DIM], 0.0, None), 0.0)
    return repaired


def rule_to_json(rule: RbRepairRule) -> dict:
    return {
        "name": rule.name,
        "mode": rule.mode,
        "scale": float(rule.scale),
        "threshold": float(rule.threshold),
        "step_scales": None if rule.step_scales is None else np.asarray(rule.step_scales, dtype=float).tolist(),
        "bin_edges": None if rule.bin_edges is None else np.asarray(rule.bin_edges, dtype=float).tolist(),
        "bin_values": None if rule.bin_values is None else np.asarray(rule.bin_values, dtype=float).tolist(),
    }


def active_rate_row(candidate: str, split: str, predictions: dict[str, np.ndarray], baseline_rmse: float) -> dict:
    truth = predictions["link_rate_true"].squeeze(-1)
    pred = predictions["link_rate_pred"].squeeze(-1)
    activity_true = predictions["link_activity_true"].squeeze(-1)
    active = activity_true > 0.5
    active_error = pred[active] - truth[active]
    active_rmse = _rmse(active_error)
    activity = activity_metrics(predictions["link_activity_prob"].squeeze(-1), activity_true, threshold=0.5)
    return {
        "candidate": candidate,
        "split": split,
        "active_count": int(active.sum()),
        "active_rate_rmse": active_rmse,
        "active_rate_mae": float(np.mean(np.abs(active_error))) if active_error.size else float("nan"),
        "activity_precision": float(activity["precision"]),
        "activity_recall": float(activity["recall"]),
        "activity_f1": float(activity["f1"]),
        "activity_accuracy": float(activity["accuracy"]),
        "activity_threshold": float(activity["threshold"]),
        "activity_tp": int(activity["tp"]),
        "activity_fp": int(activity["fp"]),
        "activity_fn": int(activity["fn"]),
        "activity_tn": int(activity["tn"]),
        "link_rmse": _rmse(predictions["link_rate_pred"] - predictions["link_rate_true"]),
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


def make_adaptive_dataset(args, arrays, indices, stats, policy_model, action_scale, value_vocab, device, train_idx):
    base_dataset = V6WorldModelDataset(arrays, indices, stats)
    old_dataset = make_bridge_dataset(
        arrays, indices, stats, policy_model, action_scale, value_vocab, device,
        OperatingPoint("old", args.policy_threshold, args.value_scale, value_codebook_size=args.value_codebook_size),
        train_idx,
    )
    new_dataset = make_bridge_dataset(
        arrays, indices, stats, policy_model, action_scale, value_vocab, device,
        OperatingPoint("new", args.new_policy_threshold, args.new_value_scale, value_codebook_size=args.value_codebook_size),
        train_idx,
    )
    return base_dataset, AdaptivePolicyBridgeDataset(base_dataset, old_dataset, new_dataset, stats, args.gate_feature, args.gate_threshold)


def evaluate_raw_actions(raw_actions: np.ndarray, base_dataset, stats: dict, world_model, world_config: dict, device: torch.device, batch_size: int) -> dict:
    loader = DataLoader(
        RawFutureActionDataset(base_dataset, raw_actions, stats),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_v6_world_model_batch,
    )
    return collect_v8_predictions(
        world_model, loader, device, stats,
        rate_output_mode=world_config.get("rate_output_mode", "main"),
        inactive_rate_value=float(world_config.get("inactive_rate_value", 0.0)),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-experiment-dir", type=Path, default=PROJECT_ROOT / "artifacts/experiments/pi_jwm_v9_expanded_v2_gpu_20260619/v2_hurdle_baseline")
    parser.add_argument("--world-checkpoint", type=Path, default=PROJECT_ROOT / "artifacts/experiments/pi_jwm_v9_expanded_v2_gpu_20260619/v2_hurdle_baseline/checkpoints/v8_dual_best.pt")
    parser.add_argument("--policy-checkpoint", type=Path, default=PROJECT_ROOT / "artifacts/experiments/pi_jwm_v10_policy_bridge_gpu_20260620/v10_action_policy_bc/checkpoints/v7_action_policy_cross_attention_best.pt")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--policy-threshold", type=float, default=0.4)
    parser.add_argument("--value-scale", type=float, default=1.0)
    parser.add_argument("--new-policy-threshold", type=float, default=0.37)
    parser.add_argument("--new-value-scale", type=float, default=1.06)
    parser.add_argument("--gate-feature", choices=("step_rb_total", "step_cpu_total", "step_rb_cpu_total", "step_active_count"), default="step_rb_cpu_total")
    parser.add_argument("--gate-threshold", type=float, default=450.0)
    parser.add_argument("--value-codebook-size", type=int, default=9)
    return parser.parse_args()


def run_experiment(args: argparse.Namespace) -> dict:
    device = torch.device("cpu")
    context = load_context(args, device)
    summary, arrays, stats, world_model, policy_model, action_scale, value_vocab, splits = context
    world_model = freeze_module(world_model)
    policy_model = freeze_module(policy_model)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_base, train_dataset = make_adaptive_dataset(args, arrays, splits["train"], stats, policy_model, action_scale, value_vocab, device, splits["train"])
    train_actions, train_truth = collect_raw_actions(train_dataset, stats)
    rules = [
        RbRepairRule("identity", "identity"),
        fit_global_scale_rule(train_actions, train_truth, name="global_scale_train_ls"),
        fit_step_scale_rule(train_actions, train_truth, name="step_scale_train_ls"),
        fit_step_bin_median_rule(train_actions, train_truth, np.array([1.0, 5.0, 10.0, 16.0, 25.0, 50.0, 100.0], dtype=np.float32), name="step_bin_median_train"),
    ]
    for threshold in [1.0, 2.0, 5.0, 10.0, 16.0, 25.0, 50.0, 100.0]:
        rules.append(RbRepairRule(f"zero_rb_below_{threshold:g}", "zero_below_threshold", threshold=float(threshold)))

    all_rows = []
    raw_cache = {}
    for split_name in ("val", "test"):
        base_dataset, adaptive_dataset = make_adaptive_dataset(args, arrays, splits[split_name], stats, policy_model, action_scale, value_vocab, device, splits["train"])
        baseline_actions, _ = collect_raw_actions(adaptive_dataset, stats)
        raw_cache[split_name] = baseline_actions
        baseline_predictions = evaluate_raw_actions(baseline_actions, base_dataset, stats, world_model, summary["config"], device, args.batch_size)
        baseline_row = active_rate_row("identity", split_name, baseline_predictions, float("nan"))
        baseline_rmse = float(baseline_row["active_rate_rmse"])
        baseline_row["improvement_vs_baseline"] = 0.0
        all_rows.append(baseline_row)
        for rule in rules[1:]:
            repaired = apply_rb_total_repair(baseline_actions, rule, preserve_step0=True)
            predictions = evaluate_raw_actions(repaired, base_dataset, stats, world_model, summary["config"], device, args.batch_size)
            all_rows.append(active_rate_row(rule.name, split_name, predictions, baseline_rmse))

    val_rows = [row for row in all_rows if row["split"] == "val"]
    val_ranked = sorted(val_rows, key=lambda row: (float(row["active_rate_rmse"]), str(row["candidate"])))
    write_csv(args.output_dir / "rb_total_repair_results.csv", all_rows)
    write_csv(args.output_dir / "rb_total_repair_val_ranked.csv", val_ranked)
    (args.output_dir / "rb_total_repair_rules.json").write_text(json.dumps([rule_to_json(rule) for rule in rules], ensure_ascii=False, indent=2), encoding="utf-8")
    best = val_ranked[0]
    report = [
        "# PI-JWM v11 rb_total Repair CPU Probe",
        "",
        "CPU-only deployable probe. Rules are fitted on train split only and selected by validation active-rate RMSE.",
        "",
        "| rank | candidate | val_active_rmse | improvement_vs_baseline | link_rmse |",
        "|---:|---|---:|---:|---:|",
    ]
    for rank, row in enumerate(val_ranked, start=1):
        report.append(f"| {rank} | {row['candidate']} | {row['active_rate_rmse']:.6f} | {row['improvement_vs_baseline']:.6f} | {row['link_rmse']:.6f} |")
    report.extend([
        "",
        "## Decision",
        "",
        f"Best validation candidate: {best['candidate']} with val active-rate RMSE {best['active_rate_rmse']:.6f}.",
    ])
    if float(best["active_rate_rmse"]) < CURRENT_CPU_BEST_VAL_ACTIVE_RMSE - 1e-3:
        report.append("This passes the CPU validation gate for further confirmation.")
    else:
        report.append("This does not pass the CPU validation gate; do not run GPU from this probe.")
    (args.output_dir / "rb_total_repair_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    result = {
        "framework": "PI-JWM",
        "candidate": "v11",
        "mode": "rb_total_repair_cpu_probe",
        "output_dir": str(args.output_dir),
        "rules": [rule_to_json(rule) for rule in rules],
        "rows": all_rows,
        "val_ranked": val_ranked,
        "best_val": best,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    args = parse_args()
    result = run_experiment(args)
    print(json.dumps({"output_dir": result["output_dir"], "best_val": result["best_val"], "val_ranked": result["val_ranked"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
