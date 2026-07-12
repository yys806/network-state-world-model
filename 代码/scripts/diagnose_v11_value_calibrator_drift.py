"""Diagnose action-value drift caused by a PI-JWM v11 rollout value calibrator."""

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

from pi_jwm.v6_data import V6WorldModelDataset, collate_v6_world_model_batch, load_world_model_arrays, make_normalization_stats
from pi_jwm.v6_dual_graph import V6DualGraphBatch
from pi_jwm.v11_rollout_value_calibrator import RolloutAlignedValueCalibrator

from evaluate_v10_policy_bridge import load_policy, make_positive_value_quantile_codebook
from evaluate_v9_action_ablation import load_model_for_experiment, resolve_project_path
from run_v11_rollout_value_calibrator import (
    build_calibrated_raw_action,
    choose_device,
    compute_step_score,
    inverse_normalize_action_tensor,
    mix_adaptive_new_action_by_step_gate,
)
from run_world_model_v8_full_training import add_event_memory_features, resolve_seed_splits


ACTION_NAMES = ["offload_count", "rb_task_count", "rb_total", "cpu_task_count", "cpu_total", "return_count"]


def summarize_delta(name: str, baseline: np.ndarray, calibrated: np.ndarray) -> dict[str, float | int | str]:
    baseline = np.asarray(baseline, dtype=np.float64)
    calibrated = np.asarray(calibrated, dtype=np.float64)
    if baseline.shape != calibrated.shape:
        raise ValueError("baseline and calibrated must share shape")
    if baseline.size == 0:
        return {
            "name": name,
            "count": 0,
            "baseline_mean": float("nan"),
            "calibrated_mean": float("nan"),
            "delta_mean": float("nan"),
            "abs_delta_mean": float("nan"),
            "baseline_sum": 0.0,
            "calibrated_sum": 0.0,
            "delta_sum": 0.0,
        }
    delta = calibrated - baseline
    return {
        "name": name,
        "count": int(baseline.size),
        "baseline_mean": float(np.mean(baseline)),
        "calibrated_mean": float(np.mean(calibrated)),
        "delta_mean": float(np.mean(delta)),
        "abs_delta_mean": float(np.mean(np.abs(delta))),
        "baseline_sum": float(np.sum(baseline)),
        "calibrated_sum": float(np.sum(calibrated)),
        "delta_sum": float(np.sum(delta)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--world-experiment-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts/experiments/pi_jwm_v9_expanded_v2_gpu_20260619/v2_hurdle_baseline",
    )
    parser.add_argument(
        "--world-checkpoint",
        type=Path,
        default=PROJECT_ROOT / "artifacts/experiments/pi_jwm_v9_expanded_v2_gpu_20260619/v2_hurdle_baseline/checkpoints/v8_dual_best.pt",
    )
    parser.add_argument(
        "--policy-checkpoint",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts/experiments/pi_jwm_v10_policy_bridge_gpu_20260620/v10_action_policy_bc/checkpoints/v7_action_policy_cross_attention_best.pt",
    )
    parser.add_argument("--calibrator-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--policy-threshold", type=float, default=0.4)
    parser.add_argument("--value-scale", type=float, default=1.0)
    parser.add_argument("--new-policy-threshold", type=float, default=0.37)
    parser.add_argument("--new-value-scale", type=float, default=1.06)
    parser.add_argument("--gate-feature", choices=("step_rb_total", "step_cpu_total", "step_rb_cpu_total", "step_active_count"), default="step_rb_cpu_total")
    parser.add_argument("--gate-threshold", type=float, default=450.0)
    parser.add_argument("--true-first", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def make_batch(world_batch, device: torch.device) -> V6DualGraphBatch:
    return V6DualGraphBatch(
        node_history=world_batch.node_history.unsqueeze(0).to(device),
        physical_edge_history=world_batch.physical_edge_history.unsqueeze(0).to(device),
        info_edge_history=world_batch.info_edge_history.unsqueeze(0).to(device),
        action_history=world_batch.action_history.unsqueeze(0).to(device),
        future_actions=world_batch.future_actions.unsqueeze(0).to(device),
        task_history=world_batch.task_history.unsqueeze(0).to(device),
        link_rate_baseline=None if world_batch.link_rate_baseline is None else world_batch.link_rate_baseline.unsqueeze(0).to(device),
    )


def load_calibrator(checkpoint_path: Path, device: torch.device) -> tuple[RolloutAlignedValueCalibrator, torch.Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    calibrator = RolloutAlignedValueCalibrator(**config).to(device)
    calibrator.load_state_dict(checkpoint["model_state"])
    calibrator.eval()
    codebook = torch.as_tensor(checkpoint["codebook"], dtype=torch.float32, device=device)
    return calibrator, codebook


def collect_split_actions(
    arrays: dict[str, np.ndarray],
    indices: np.ndarray,
    stats: dict,
    policy_model,
    calibrator: RolloutAlignedValueCalibrator,
    action_scale: torch.Tensor,
    codebook: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, np.ndarray]:
    dataset = V6WorldModelDataset(arrays, indices, stats)
    baseline_rows = []
    calibrated_rows = []
    true_rows = []
    gate_rows = []
    with torch.no_grad():
        for idx in range(len(dataset)):
            world_batch, _ = dataset[idx]
            batch = make_batch(world_batch, device)
            outputs = policy_model(batch)
            activity_prob = torch.sigmoid(outputs["action_logit"])
            policy_value = outputs["action_value"] * action_scale
            old_active = activity_prob >= float(args.policy_threshold)
            new_active = activity_prob >= float(args.new_policy_threshold)
            true_raw = inverse_normalize_action_tensor(batch.future_actions, stats["edge_a_future"])

            baseline_value = calibrator(policy_value, activity_prob, codebook, torch.ones_like(old_active), hard=True)
            calibrated_value = calibrator(policy_value, activity_prob, codebook, torch.ones_like(old_active), hard=True)
            identity_value = project_identity_to_codebook(policy_value, codebook)

            baseline_old = build_calibrated_raw_action(identity_value, old_active, args.value_scale, true_raw, args.true_first)
            baseline_new = build_calibrated_raw_action(identity_value, new_active, args.new_value_scale, true_raw, args.true_first)
            baseline_action = mix_adaptive_new_action_by_step_gate(
                baseline_old,
                baseline_new,
                args.gate_feature,
                args.gate_threshold,
                true_raw=true_raw,
                true_first=args.true_first,
            )

            calibrated_old = build_calibrated_raw_action(calibrated_value, old_active, args.value_scale, true_raw, args.true_first)
            calibrated_new = build_calibrated_raw_action(calibrated_value, new_active, args.new_value_scale, true_raw, args.true_first)
            calibrated_action = mix_adaptive_new_action_by_step_gate(
                calibrated_old,
                calibrated_new,
                args.gate_feature,
                args.gate_threshold,
                true_raw=true_raw,
                true_first=args.true_first,
            )

            gate = compute_step_score(baseline_new, args.gate_feature) >= float(args.gate_threshold)
            baseline_rows.append(baseline_action.squeeze(0).cpu().numpy())
            calibrated_rows.append(calibrated_action.squeeze(0).cpu().numpy())
            true_rows.append(true_raw.squeeze(0).cpu().numpy())
            gate_rows.append(gate.squeeze(0).cpu().numpy())
    return {
        "baseline": np.stack(baseline_rows, axis=0),
        "calibrated": np.stack(calibrated_rows, axis=0),
        "truth": np.stack(true_rows, axis=0),
        "gate": np.stack(gate_rows, axis=0).astype(bool),
    }


def project_identity_to_codebook(policy_value: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    distances = torch.abs(policy_value.unsqueeze(-1) - codebook.reshape(1, codebook.shape[0], 1, codebook.shape[1], codebook.shape[2]))
    nearest = torch.argmin(distances, dim=-1)
    expanded = codebook.reshape(1, codebook.shape[0], 1, codebook.shape[1], codebook.shape[2]).expand(*policy_value.shape, codebook.shape[2])
    return torch.gather(expanded, dim=-1, index=nearest.unsqueeze(-1)).squeeze(-1)


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys()) if rows else ["name"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize_split(split: str, data: dict[str, np.ndarray]) -> dict[str, list[dict]]:
    baseline = data["baseline"]
    calibrated = data["calibrated"]
    truth = data["truth"]
    gate = data["gate"]
    rows_overall = [summarize_delta(f"{split}/all", baseline, calibrated)]
    rows_step = []
    for step in range(baseline.shape[1]):
        rows_step.append(summarize_delta(f"{split}/step_{step}", baseline[:, step], calibrated[:, step]))
    rows_dim = []
    for dim in range(baseline.shape[-1]):
        name = ACTION_NAMES[dim] if dim < len(ACTION_NAMES) else f"dim_{dim}"
        rows_dim.append(summarize_delta(f"{split}/{name}", baseline[..., dim], calibrated[..., dim]))
    rows_gate = [
        summarize_delta(f"{split}/gate_true", baseline[gate], calibrated[gate]),
        summarize_delta(f"{split}/gate_false", baseline[~gate], calibrated[~gate]),
    ]
    active_truth = np.any(truth > 1e-9, axis=-1)
    rows_active = [
        summarize_delta(f"{split}/truth_active", baseline[active_truth], calibrated[active_truth]),
        summarize_delta(f"{split}/truth_inactive", baseline[~active_truth], calibrated[~active_truth]),
    ]
    return {
        "overall": rows_overall,
        "by_step": rows_step,
        "by_dim": rows_dim,
        "by_gate": rows_gate,
        "by_truth_activity": rows_active,
    }


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    world_summary = json.loads((args.world_experiment_dir / "v8_full_training_summary.json").read_text(encoding="utf-8"))
    dataset_dir = resolve_project_path(world_summary["dataset_dir"])
    arrays = load_world_model_arrays(dataset_dir)
    if world_summary["config"].get("use_event_memory_features", False):
        arrays = add_event_memory_features(arrays)
    split = world_summary["split_seed_spec"]
    train_idx, val_idx, test_idx, _ = resolve_seed_splits(
        arrays["sample_seed"],
        train_seeds=split["train_seeds"],
        val_seeds=split["val_seeds"],
        test_seeds=split["test_seeds"],
    )
    stats = make_normalization_stats(arrays, train_idx)
    _ = load_model_for_experiment(world_summary, arrays, args.world_checkpoint, device)
    policy_model, action_scale_np, _, value_vocab = load_policy(args.policy_checkpoint, device)
    if value_vocab is not None:
        raise ValueError("drift diagnosis currently expects continuous v10 policy checkpoint")
    policy_model.eval()
    action_scale = torch.as_tensor(action_scale_np.reshape(1, 1, 1, -1), dtype=torch.float32, device=device)
    calibrator, codebook = load_calibrator(args.calibrator_checkpoint, device)

    outputs = {}
    for split_name, indices in (("val", val_idx), ("test", test_idx)):
        data = collect_split_actions(
            arrays,
            indices,
            stats,
            policy_model,
            calibrator,
            action_scale,
            codebook,
            args,
            device,
        )
        summaries = summarize_split(split_name, data)
        outputs[split_name] = summaries
        for table_name, rows in summaries.items():
            write_csv(rows, args.output_dir / f"{split_name}_{table_name}.csv")

    summary_path = args.output_dir / "drift_summary.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8")
    report_lines = ["# PI-JWM v11 Value Calibrator Drift Diagnosis", ""]
    for split_name in ("val", "test"):
        report_lines.append(f"## {split_name}")
        for row in outputs[split_name]["overall"] + outputs[split_name]["by_gate"] + outputs[split_name]["by_truth_activity"]:
            report_lines.append(
                f"- {row['name']}: count={row['count']} delta_mean={row['delta_mean']:.6f} "
                f"abs_delta_mean={row['abs_delta_mean']:.6f} delta_sum={row['delta_sum']:.6f}"
            )
        report_lines.append("")
    (args.output_dir / "drift_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
