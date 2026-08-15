"""CPU sweep for explicit RB-total shrink on PI-JWM v11 adaptive bridge."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from pi_jwm.v6_data import V6WorldModelDataset, collate_v6_world_model_batch, load_world_model_arrays, make_normalization_stats
from pi_jwm.v6_dual_graph import V6DualGraphBatch
from pi_jwm.v8_training import evaluate_v8_model

from evaluate_v10_policy_bridge import load_policy
from evaluate_v11_adaptive_bridge import PointConfig, choose_device, make_point_dataset, mix_actions_by_step_gate
from evaluate_v9_action_ablation import load_model_for_experiment, resolve_project_path
from run_v11_rollout_value_calibrator import compute_step_score
from run_world_model_v8_full_training import add_event_memory_features, resolve_seed_splits


def apply_rb_total_scale(
    actions: torch.Tensor,
    rb_scale: float,
    step_mask: torch.Tensor | None = None,
    true_raw: torch.Tensor | None = None,
    true_first: bool = True,
) -> torch.Tensor:
    if rb_scale < 0.0:
        raise ValueError("rb_scale must be non-negative")
    if actions.ndim != 4:
        raise ValueError("actions must have shape [batch, horizon, edge, action_dim]")
    scaled = actions.clone()
    if step_mask is None:
        scaled[..., 2] = scaled[..., 2] * float(rb_scale)
    else:
        if step_mask.shape != actions.shape[:2]:
            raise ValueError("step_mask must have shape [batch, horizon]")
        mask = step_mask.to(dtype=torch.bool).reshape(actions.shape[0], actions.shape[1], 1)
        scaled[..., 2] = torch.where(mask, scaled[..., 2] * float(rb_scale), scaled[..., 2])
    if true_first:
        if true_raw is None:
            raise ValueError("true_raw is required when true_first=True")
        if true_raw.shape != scaled.shape:
            raise ValueError("true_raw must match action shape")
        scaled[:, 0] = true_raw[:, 0]
    return scaled


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
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts/experiments/pi_jwm_v11_rb_shrink_sweep_cpu_20260621")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--old-threshold", type=float, default=0.4)
    parser.add_argument("--old-scale", type=float, default=1.0)
    parser.add_argument("--new-threshold", type=float, default=0.37)
    parser.add_argument("--new-scale", type=float, default=1.06)
    parser.add_argument("--gate-feature", choices=("step_rb_total", "step_cpu_total", "step_rb_cpu_total", "step_active_count"), default="step_rb_cpu_total")
    parser.add_argument("--gate-threshold", type=float, default=450.0)
    parser.add_argument("--value-codebook-size", type=int, default=9)
    parser.add_argument("--rb-scale", action="append", type=float, dest="rb_scales", default=None)
    parser.add_argument("--shrink-region", choices=("all", "gate_true", "gate_false"), default="all")
    parser.add_argument("--true-first", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


class CachedAdaptiveProposal:
    def __init__(self, world_batch: V6DualGraphBatch, target: dict, action: torch.Tensor, true_raw: torch.Tensor, gate: torch.Tensor) -> None:
        self.world_batch = world_batch
        self.target = target
        self.action = action
        self.true_raw = true_raw
        self.gate = gate


class RbShrinkDataset(Dataset):
    def __init__(self, proposals: list[CachedAdaptiveProposal], normalizer, rb_scale: float, shrink_region: str, true_first: bool) -> None:
        self.proposals = proposals
        self.normalizer = normalizer
        self.rb_scale = float(rb_scale)
        self.shrink_region = str(shrink_region)
        self.true_first = bool(true_first)
        self.gate_true_count = 0
        self.gate_total_count = 0

    def __len__(self) -> int:
        return len(self.proposals)

    def __getitem__(self, item: int):
        proposal = self.proposals[item]
        raw = apply_rb_total_scale(
            proposal.action.unsqueeze(0),
            self.rb_scale,
            step_mask=self.step_mask(proposal).unsqueeze(0),
            true_raw=proposal.true_raw.unsqueeze(0),
            true_first=self.true_first,
        ).squeeze(0)
        normalized = self.normalizer.normalize_future_actions(raw)
        self.gate_true_count += int(proposal.gate.sum().item())
        self.gate_total_count += int(proposal.gate.numel())
        batch = proposal.world_batch
        bridged = V6DualGraphBatch(
            node_history=batch.node_history,
            physical_edge_history=batch.physical_edge_history,
            info_edge_history=batch.info_edge_history,
            action_history=batch.action_history,
            future_actions=normalized,
            task_history=batch.task_history,
            link_rate_baseline=batch.link_rate_baseline,
        )
        return bridged, proposal.target

    def step_mask(self, proposal: CachedAdaptiveProposal) -> torch.Tensor | None:
        if self.shrink_region == "all":
            return None
        if self.shrink_region == "gate_true":
            return proposal.gate
        if self.shrink_region == "gate_false":
            return ~proposal.gate
        raise ValueError(f"unknown shrink_region: {self.shrink_region}")


def make_context(args: argparse.Namespace):
    device = choose_device(args.device)
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
    policy_model, action_scale, _, value_vocab = load_policy(args.policy_checkpoint, device)
    old_point = PointConfig(args.old_threshold, args.old_scale, value_codebook_size=args.value_codebook_size)
    new_point = PointConfig(args.new_threshold, args.new_scale, value_codebook_size=args.value_codebook_size)
    split_context = {}
    for split_name, indices in (("val", val_idx), ("test", test_idx)):
        base = V6WorldModelDataset(arrays, indices, stats)
        old_dataset = make_point_dataset(arrays, indices, stats, policy_model, action_scale, value_vocab, device, train_idx, old_point)
        new_dataset = make_point_dataset(arrays, indices, stats, policy_model, action_scale, value_vocab, device, train_idx, new_point)
        split_context[split_name] = {
            "old_dataset": old_dataset,
            "proposals": cache_adaptive_proposals(base, old_dataset, new_dataset, args),
        }
    return device, summary, world_model, stats, old_point, new_point, split_context


def cache_adaptive_proposals(base: V6WorldModelDataset, old_dataset, new_dataset, args: argparse.Namespace) -> list[CachedAdaptiveProposal]:
    proposals = []
    for idx in range(len(base)):
        world_batch, target = base[idx]
        old_batch, _ = old_dataset.policy_dataset[idx]
        new_batch, _ = new_dataset.policy_dataset[idx]
        true_raw = old_dataset.raw_future_from_normalized(world_batch.future_actions)
        old_action = old_dataset.generate_raw_actions(old_batch, world_batch, true_raw)
        new_action = new_dataset.generate_raw_actions(new_batch, world_batch, true_raw)
        if args.true_first:
            old_action[0] = true_raw[0]
            new_action[0] = true_raw[0]
        gate = compute_step_score(new_action.unsqueeze(0), args.gate_feature).squeeze(0) >= float(args.gate_threshold)
        adaptive = mix_actions_by_step_gate(old_action, new_action, gate)
        proposals.append(CachedAdaptiveProposal(world_batch, target, adaptive, true_raw, gate))
    return proposals


def evaluate_scale(rb_scale: float, device, summary: dict, world_model, stats: dict, split_context: dict, batch_size: int, shrink_region: str, true_first: bool) -> dict:
    result = {}
    config = summary["config"]
    for split_name, context in split_context.items():
        dataset = RbShrinkDataset(context["proposals"], context["old_dataset"], rb_scale, shrink_region, true_first)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_v6_world_model_batch)
        metrics = evaluate_v8_model(
            world_model,
            loader,
            device,
            stats,
            rate_output_mode=config.get("rate_output_mode", "main"),
            inactive_rate_value=float(config.get("inactive_rate_value", 0.0)),
        )
        metrics["adaptive_gate"] = {
            "true_count": int(dataset.gate_true_count),
            "total_count": int(dataset.gate_total_count),
            "fraction": float(dataset.gate_true_count / max(dataset.gate_total_count, 1)),
        }
        result[split_name] = metrics
    return result


def row_from_result(rb_scale: float, output_json: Path, payload: dict) -> dict:
    return {
        "rb_scale": float(rb_scale),
        "val_active_rate_rmse": float(payload["val"]["active_rate"]["active_rmse"]),
        "test_active_rate_rmse": float(payload["test"]["active_rate"]["active_rmse"]),
        "val_activity_f1": float(payload["val"]["activity"]["f1"]),
        "test_activity_f1": float(payload["test"]["activity"]["f1"]),
        "val_link_rmse": float(payload["val"]["link_rate"]["rmse"]),
        "test_link_rmse": float(payload["test"]["link_rate"]["rmse"]),
        "output_json": str(output_json),
    }


def write_csv(rows: list[dict], path: Path) -> None:
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    rb_scales = args.rb_scales or [1.0, 0.995, 0.99, 0.985, 0.98, 0.975, 0.97, 0.965, 0.96, 0.95]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_dir = args.output_dir / "scale_json"
    json_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    device, summary, world_model, stats, old_point, new_point, split_context = make_context(args)
    rows = []
    for rb_scale in rb_scales:
        metrics = evaluate_scale(rb_scale, device, summary, world_model, stats, split_context, args.batch_size, args.shrink_region, args.true_first)
        payload = {
            "framework": "PI-JWM",
            "candidate": "v11",
            "mode": "adaptive_rb_total_shrink_sweep",
            "old_point": old_point.__dict__,
            "new_point": new_point.__dict__,
            "gate_feature": args.gate_feature,
            "gate_threshold": float(args.gate_threshold),
            "rb_scale": float(rb_scale),
            "shrink_region": args.shrink_region,
            **metrics,
        }
        output_json = json_dir / f"rb_scale_{float(rb_scale):.6f}.json"
        output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        rows.append(row_from_result(rb_scale, output_json, payload))
        print(
            f"rb_scale={rb_scale:g} "
            f"val_active_rate_rmse={payload['val']['active_rate']['active_rmse']:.6f} "
            f"test_active_rate_rmse={payload['test']['active_rate']['active_rmse']:.6f} "
            f"test_link_rmse={payload['test']['link_rate']['rmse']:.6f}"
        )
    rows.sort(key=lambda row: (float(row["val_active_rate_rmse"]), float(row["test_active_rate_rmse"])))
    write_csv(rows, args.output_dir / "sweep_results.csv")
    best = rows[0]
    print(
        "best "
        f"rb_scale={best['rb_scale']:g} "
        f"val_active_rate_rmse={best['val_active_rate_rmse']:.6f} "
        f"test_active_rate_rmse={best['test_active_rate_rmse']:.6f} "
        f"test_f1={best['test_activity_f1']:.6f} "
        f"test_link_rmse={best['test_link_rmse']:.6f}"
    )


if __name__ == "__main__":
    main()
