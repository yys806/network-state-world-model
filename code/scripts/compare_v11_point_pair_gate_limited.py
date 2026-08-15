"""Limited CPU actual-rollout sweep for PI-JWM v11 two-point bridge gates."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from pi_jwm.v6_data import V6WorldModelDataset, load_world_model_arrays, make_normalization_stats

from compare_v11_learned_point_selector import limit_indices
from diagnose_v11_bridge_operating_point import collect_raw_actions
from evaluate_v10_policy_bridge import load_policy
from evaluate_v11_adaptive_bridge import PointConfig, choose_device, make_point_dataset, mix_actions_by_step_gate
from evaluate_v9_action_ablation import load_model_for_experiment, resolve_project_path
from run_v11_rb_total_repair import active_rate_row, evaluate_raw_actions, write_csv
from run_world_model_v8_full_training import add_event_memory_features, resolve_seed_splits
from sweep_v11_adaptive_bridge_gate import GateRule, compute_rule_gate, parse_rule


DEFAULT_RULES = (
    "step_active_count:4",
    "step_active_count:6",
    "step_active_count:8",
    "step_active_count:10",
    "step_active_count:12",
    "step_rb_total:100",
    "step_rb_total:150",
    "step_rb_total:200",
    "step_rb_total:250",
    "step_rb_total:300",
)


def mix_actions_for_rule(old_actions: np.ndarray, new_actions: np.ndarray, rule: GateRule) -> tuple[np.ndarray, float]:
    old_actions_t = torch.as_tensor(old_actions, dtype=torch.float32)
    new_actions_t = torch.as_tensor(new_actions, dtype=torch.float32)
    mixed = []
    true_count = 0
    total_count = 0
    for old_sample, new_sample in zip(old_actions_t, new_actions_t):
        gate = compute_rule_gate(new_sample, rule)
        true_count += int(gate.sum().item())
        total_count += int(gate.numel())
        mixed.append(mix_actions_by_step_gate(old_sample, new_sample, gate).numpy())
    return np.stack(mixed, axis=0).astype(np.float32), float(true_count / max(total_count, 1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-experiment-dir", type=Path, default=PROJECT_ROOT / "artifacts/experiments/pi_jwm_v9_expanded_v2_gpu_20260619/v2_hurdle_baseline")
    parser.add_argument("--world-checkpoint", type=Path, default=PROJECT_ROOT / "artifacts/experiments/pi_jwm_v9_expanded_v2_gpu_20260619/v2_hurdle_baseline/checkpoints/v8_dual_best.pt")
    parser.add_argument("--policy-checkpoint", type=Path, default=PROJECT_ROOT / "artifacts/experiments/pi_jwm_v10_policy_bridge_gpu_20260620/v10_action_policy_bc/checkpoints/v7_action_policy_cross_attention_best.pt")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts/experiments/pi_jwm_v11_point_pair_gate_limited_20260629")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--old-threshold", type=float, required=True)
    parser.add_argument("--old-scale", type=float, required=True)
    parser.add_argument("--new-threshold", type=float, required=True)
    parser.add_argument("--new-scale", type=float, required=True)
    parser.add_argument("--value-codebook-size", type=int, default=9)
    parser.add_argument("--max-val-samples", type=int, default=384)
    parser.add_argument("--max-test-samples", type=int, default=384)
    parser.add_argument("--rule", action="append", default=None)
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
    splits = {
        "train": train_idx,
        "val": limit_indices(val_idx, args.max_val_samples),
        "test": limit_indices(test_idx, args.max_test_samples),
    }
    world_model = load_model_for_experiment(summary, arrays, args.world_checkpoint, device)
    policy_model, action_scale, _, value_vocab = load_policy(args.policy_checkpoint, device)
    return summary, arrays, stats, world_model, policy_model, action_scale, value_vocab, splits


def collect_point_actions(args, arrays, indices, stats, policy_model, action_scale, value_vocab, device, train_idx, threshold, scale):
    base = V6WorldModelDataset(arrays, indices, stats)
    point = PointConfig(float(threshold), float(scale), value_codebook_size=int(args.value_codebook_size))
    dataset = make_point_dataset(arrays, indices, stats, policy_model, action_scale, value_vocab, device, train_idx, point)
    actions = collect_raw_actions(dataset)
    return base, actions


def run(args: argparse.Namespace) -> dict:
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    device = choose_device(args.device)
    summary, arrays, stats, world_model, policy_model, action_scale, value_vocab, splits = load_context(args, device)
    rules = [parse_rule(text) for text in (args.rule or DEFAULT_RULES)]
    rows = []
    for split_name in ("val", "test"):
        base, old_actions = collect_point_actions(
            args, arrays, splits[split_name], stats, policy_model, action_scale, value_vocab, device, splits["train"], args.old_threshold, args.old_scale
        )
        _base, new_actions = collect_point_actions(
            args, arrays, splits[split_name], stats, policy_model, action_scale, value_vocab, device, splits["train"], args.new_threshold, args.new_scale
        )
        for tag, actions in (("old_point", old_actions), ("new_point", new_actions)):
            predictions = evaluate_raw_actions(actions, base, stats, world_model, summary["config"], device, args.batch_size)
            row = active_rate_row(tag, split_name, predictions, float("nan"))
            row["gate_fraction"] = float("nan")
            rows.append(row)
        for rule in rules:
            mixed, fraction = mix_actions_for_rule(old_actions, new_actions, rule)
            predictions = evaluate_raw_actions(mixed, base, stats, world_model, summary["config"], device, args.batch_size)
            row = active_rate_row(f"rule_{rule.slug}", split_name, predictions, float("nan"))
            row["gate_fraction"] = fraction
            rows.append(row)
            print(f"{split_name} rule={rule.slug} active_rmse={float(row['active_rate_rmse']):.6f}")
    write_csv(args.output_dir / "point_pair_gate_results.csv", rows)
    val_rows = [row for row in rows if row["split"] == "val"]
    test_rows = [row for row in rows if row["split"] == "test"]
    result = {
        "framework": "PI-JWM",
        "candidate": "v11",
        "mode": "point_pair_gate_limited",
        "old_point": {"threshold": float(args.old_threshold), "scale": float(args.old_scale)},
        "new_point": {"threshold": float(args.new_threshold), "scale": float(args.new_scale)},
        "rows": rows,
        "best_val": min(val_rows, key=lambda row: float(row["active_rate_rmse"])),
        "best_test": min(test_rows, key=lambda row: float(row["active_rate_rmse"])),
        "runtime_seconds": float(time.time() - started),
        "output_dir": str(args.output_dir),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    result = run(parse_args())
    print(
        f"best_val={result['best_val']['candidate']} active_rmse={float(result['best_val']['active_rate_rmse']):.6f} "
        f"best_test={result['best_test']['candidate']} test_active_rmse={float(result['best_test']['active_rate_rmse']):.6f}"
    )
    print(f"wrote {result['output_dir']}")


if __name__ == "__main__":
    main()
