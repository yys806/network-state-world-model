"""CPU group-level rb_total calibration diagnostic for PI-JWM v11 candidate."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from pi_jwm.v11_rollout_value_calibrator import freeze_module

from diagnose_v11_counterfactual_value_attribution import collect_raw_actions
from run_v11_rb_total_repair import active_rate_row, evaluate_raw_actions, make_adaptive_dataset, write_csv
from run_v11_rb_total_value_head import load_context_limited


RB_DIM = 2
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts/experiments/pi_jwm_v11_rb_total_group_calibration_20260622"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-experiment-dir", type=Path, default=PROJECT_ROOT / "artifacts/experiments/pi_jwm_v9_expanded_v2_gpu_20260619/v2_hurdle_baseline")
    parser.add_argument("--world-checkpoint", type=Path, default=PROJECT_ROOT / "artifacts/experiments/pi_jwm_v9_expanded_v2_gpu_20260619/v2_hurdle_baseline/checkpoints/v8_dual_best.pt")
    parser.add_argument("--policy-checkpoint", type=Path, default=PROJECT_ROOT / "artifacts/experiments/pi_jwm_v10_policy_bridge_gpu_20260620/v10_action_policy_bc/checkpoints/v7_action_policy_cross_attention_best.pt")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--policy-threshold", type=float, default=0.4)
    parser.add_argument("--value-scale", type=float, default=1.0)
    parser.add_argument("--new-policy-threshold", type=float, default=0.37)
    parser.add_argument("--new-value-scale", type=float, default=1.06)
    parser.add_argument("--gate-feature", choices=("step_rb_total", "step_cpu_total", "step_rb_cpu_total", "step_active_count"), default="step_rb_cpu_total")
    parser.add_argument("--gate-threshold", type=float, default=450.0)
    parser.add_argument("--value-codebook-size", type=int, default=9)
    parser.add_argument("--max-train-samples", type=int, default=512)
    parser.add_argument("--max-val-samples", type=int, default=256)
    parser.add_argument("--max-test-samples", type=int, default=256)
    parser.add_argument("--limit-after-stats", action="store_true")
    parser.add_argument("--streaming-stats", action="store_true")
    parser.add_argument("--stats-chunk-size", type=int, default=512)
    parser.add_argument("--scales", type=float, nargs="+", default=[0.94, 0.96, 0.98, 1.0, 1.02, 1.04, 1.06, 1.08])
    parser.add_argument("--target-totals", type=float, nargs="+", default=[100.0, 125.0, 150.0, 175.0, 200.0, 225.0, 250.0, 300.0, 350.0])
    parser.add_argument("--blend-alpha", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--modes", choices=("scale", "cap_total", "blend_to_total"), nargs="+", default=["scale", "cap_total", "blend_to_total"])
    return parser.parse_args()


def calibrate_actions(actions: np.ndarray, mode: str, scale: float, target_total: float, alpha: float) -> np.ndarray:
    repaired = np.asarray(actions, dtype=np.float32).copy()
    rb = repaired[..., RB_DIM]
    base_rb = np.asarray(actions, dtype=np.float32)[..., RB_DIM]
    active = base_rb > 1e-9
    if mode == "scale":
        rb[active] = base_rb[active] * float(scale)
    elif mode in {"cap_total", "blend_to_total"}:
        totals = np.sum(np.where(active, base_rb, 0.0), axis=2, keepdims=True)
        safe_scale = np.ones_like(totals, dtype=np.float32)
        if mode == "cap_total":
            safe_scale = np.minimum(1.0, float(target_total) / np.maximum(totals, 1e-6))
        else:
            desired_scale = float(target_total) / np.maximum(totals, 1e-6)
            safe_scale = (1.0 - float(alpha)) + float(alpha) * desired_scale
        rb[:, 1:] = np.where(active[:, 1:], base_rb[:, 1:] * safe_scale[:, 1:], rb[:, 1:])
    else:
        raise ValueError(f"unknown mode: {mode}")
    repaired[:, 0, :, RB_DIM] = actions[:, 0, :, RB_DIM]
    repaired[..., RB_DIM] = np.where(actions[..., RB_DIM] > 1e-9, np.clip(repaired[..., RB_DIM], 0.0, None), 0.0)
    return repaired


def main() -> None:
    args = parse_args()
    device = torch.device("cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    summary, arrays, stats, world_model, policy_model, action_scale, value_vocab, splits = load_context_limited(args, device)
    splits = dict(splits)
    if args.limit_after_stats:
        from run_v11_rb_total_value_head import limit_indices

        splits["train"] = limit_indices(splits["train"], args.max_train_samples)
        splits["val"] = limit_indices(splits["val"], args.max_val_samples)
        splits["test"] = limit_indices(splits["test"], args.max_test_samples)
    world_model = freeze_module(world_model)
    policy_model = freeze_module(policy_model)

    payloads = {}
    baseline_rmse = {}
    rows = []
    for split_name in ("val", "test"):
        base_dataset, adaptive_dataset = make_adaptive_dataset(args, arrays, splits[split_name], stats, policy_model, action_scale, value_vocab, device, splits["train"])
        baseline_actions, _truth_actions = collect_raw_actions(adaptive_dataset, stats)
        baseline_predictions = evaluate_raw_actions(baseline_actions, base_dataset, stats, world_model, summary["config"], device, args.batch_size)
        base_row = active_rate_row("identity", split_name, baseline_predictions, float("nan"))
        base_row.update({"mode": "identity", "scale": 1.0, "target_total": "", "alpha": ""})
        rows.append(base_row)
        baseline_rmse[split_name] = float(base_row["active_rate_rmse"])
        payloads[split_name] = {"base_dataset": base_dataset, "baseline_actions": baseline_actions}

    candidates = []
    for mode in args.modes:
        if mode == "scale":
            for scale in args.scales:
                candidates.append((f"scale_{scale:g}", mode, float(scale), 0.0, 1.0))
        elif mode == "cap_total":
            for total in args.target_totals:
                candidates.append((f"cap_total_{total:g}", mode, 1.0, float(total), 1.0))
        else:
            for total in args.target_totals:
                for alpha in args.blend_alpha:
                    candidates.append((f"blend_total_{total:g}_alpha{alpha:g}", mode, 1.0, float(total), float(alpha)))

    for name, mode, scale, total, alpha in candidates:
        for split_name in ("val", "test"):
            payload = payloads[split_name]
            actions = calibrate_actions(payload["baseline_actions"], mode, scale=scale, target_total=total, alpha=alpha)
            predictions = evaluate_raw_actions(actions, payload["base_dataset"], stats, world_model, summary["config"], device, args.batch_size)
            row = active_rate_row(name, split_name, predictions, baseline_rmse[split_name])
            row.update({"mode": mode, "scale": scale, "target_total": total if total else "", "alpha": alpha})
            rows.append(row)
            print(f"{split_name} {name} active_rate_rmse={row['active_rate_rmse']:.6f} improvement={row['improvement_vs_baseline']:.6f}")

    write_csv(args.output_dir / "group_calibration_results.csv", rows)
    val_ranked = sorted([row for row in rows if row["split"] == "val"], key=lambda row: (float(row["active_rate_rmse"]), str(row["candidate"])))
    write_csv(args.output_dir / "group_calibration_val_ranked.csv", val_ranked)
    test_ranked = sorted([row for row in rows if row["split"] == "test"], key=lambda row: (float(row["active_rate_rmse"]), str(row["candidate"])))
    write_csv(args.output_dir / "group_calibration_test_ranked.csv", test_ranked)
    result = {
        "framework": "PI-JWM",
        "candidate": "v11",
        "mode": "rb_total_group_calibration",
        "output_dir": str(args.output_dir),
        "command": " ".join(sys.argv),
        "best_val": val_ranked[0],
        "best_test": test_ranked[0],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
