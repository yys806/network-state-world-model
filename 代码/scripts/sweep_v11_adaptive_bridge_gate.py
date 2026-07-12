"""Sweep deployable step gates for a PI-JWM v11 candidate adaptive bridge.

The script reuses the frozen world model and frozen action policy. It caches the
old/new decoded action proposals once per split, then evaluates a small set of
interpretable step-gate rules without retraining.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
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
from run_world_model_v8_full_training import add_event_memory_features, resolve_seed_splits


@dataclass(frozen=True)
class GateRule:
    primary_feature: str
    primary_threshold: float
    min_active_count: float | None = None
    max_active_count: float | None = None

    @property
    def slug(self) -> str:
        parts = [self.primary_feature, _float_token(self.primary_threshold)]
        if self.min_active_count is not None:
            parts.extend(["minac", _float_token(self.min_active_count)])
        if self.max_active_count is not None:
            parts.extend(["maxac", _float_token(self.max_active_count)])
        return "_".join(parts)

    def to_json(self) -> dict:
        return {
            "primary_feature": self.primary_feature,
            "primary_threshold": float(self.primary_threshold),
            "min_active_count": None if self.min_active_count is None else float(self.min_active_count),
            "max_active_count": None if self.max_active_count is None else float(self.max_active_count),
        }


def _float_token(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


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
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts/experiments/pi_jwm_v11_adaptive_gate_rule_sweep_cpu_20260621",
    )
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--old-threshold", type=float, default=0.4)
    parser.add_argument("--old-scale", type=float, default=1.0)
    parser.add_argument("--new-threshold", type=float, default=0.37)
    parser.add_argument("--new-scale", type=float, default=1.06)
    parser.add_argument("--value-codebook-size", type=int, default=9)
    parser.add_argument("--rule", action="append", default=None, help="Rule spec: feature:threshold[:min_active[:max_active]].")
    parser.add_argument("--limit", type=int, default=0, help="Run only first N rules after enumeration; 0 means all.")
    return parser.parse_args()


def parse_rule(text: str) -> GateRule:
    parts = str(text).split(":")
    if len(parts) not in (2, 3, 4):
        raise ValueError("rule must be feature:threshold[:min_active[:max_active]]")
    feature = parts[0]
    if feature not in {"step_rb_total", "step_cpu_total", "step_rb_cpu_total", "step_active_count"}:
        raise ValueError(f"unknown gate feature: {feature}")
    min_active = None if len(parts) < 3 or parts[2] == "" else float(parts[2])
    max_active = None if len(parts) < 4 or parts[3] == "" else float(parts[3])
    return GateRule(feature, float(parts[1]), min_active, max_active)


def default_rules() -> list[GateRule]:
    rules: list[GateRule] = []
    for threshold in (375, 400, 425, 450, 475, 500):
        rules.append(GateRule("step_rb_cpu_total", threshold))
    for threshold in (325, 350, 375, 400):
        rules.append(GateRule("step_rb_total", threshold))
    for threshold in (85, 100, 115, 130):
        rules.append(GateRule("step_cpu_total", threshold))
    for threshold in (9, 11, 13, 15):
        rules.append(GateRule("step_active_count", threshold))
    for threshold in (400, 425, 450, 475):
        for min_active in (9, 11, 13):
            rules.append(GateRule("step_rb_cpu_total", threshold, min_active_count=min_active))
    return rules


def compute_score(actions: torch.Tensor, feature: str) -> torch.Tensor:
    if actions.ndim != 3:
        raise ValueError("actions must have shape [horizon, edge, action_dim]")
    if feature == "step_rb_total":
        return actions[..., 2].sum(dim=1)
    if feature == "step_cpu_total":
        return actions[..., 4].sum(dim=1)
    if feature == "step_rb_cpu_total":
        return actions[..., 2].sum(dim=1) + actions[..., 4].sum(dim=1)
    if feature == "step_active_count":
        return (actions > 1e-9).any(dim=-1).sum(dim=1).to(actions.dtype)
    raise ValueError(f"unknown feature: {feature}")


def compute_rule_gate(actions: torch.Tensor, rule: GateRule) -> torch.Tensor:
    gate = compute_score(actions, rule.primary_feature) >= float(rule.primary_threshold)
    active_count = compute_score(actions, "step_active_count")
    if rule.min_active_count is not None:
        gate = gate & (active_count >= float(rule.min_active_count))
    if rule.max_active_count is not None:
        gate = gate & (active_count <= float(rule.max_active_count))
    return gate


@dataclass
class CachedProposal:
    world_batch: V6DualGraphBatch
    target: dict
    old_actions: torch.Tensor
    new_actions: torch.Tensor


class CachedRuleDataset(Dataset):
    def __init__(self, proposals: list[CachedProposal], normalizer, rule: GateRule) -> None:
        self.proposals = proposals
        self.normalizer = normalizer
        self.rule = rule
        self.gate_true_count = 0
        self.gate_total_count = 0

    def __len__(self) -> int:
        return len(self.proposals)

    def __getitem__(self, item: int):
        proposal = self.proposals[item]
        gate = compute_rule_gate(proposal.new_actions, self.rule)
        self.gate_true_count += int(gate.sum().item())
        self.gate_total_count += int(gate.numel())
        mixed = mix_actions_by_step_gate(proposal.old_actions, proposal.new_actions, gate)
        normalized_future = self.normalizer.normalize_future_actions(mixed)
        world_batch = proposal.world_batch
        bridged = V6DualGraphBatch(
            node_history=world_batch.node_history,
            physical_edge_history=world_batch.physical_edge_history,
            info_edge_history=world_batch.info_edge_history,
            action_history=world_batch.action_history,
            future_actions=normalized_future,
            task_history=world_batch.task_history,
            link_rate_baseline=world_batch.link_rate_baseline,
        )
        return bridged, proposal.target


def cache_proposals(base: V6WorldModelDataset, old_dataset, new_dataset) -> list[CachedProposal]:
    proposals: list[CachedProposal] = []
    for idx in range(len(base)):
        world_batch, target = base[idx]
        old_batch, _ = old_dataset.policy_dataset[idx]
        new_batch, _ = new_dataset.policy_dataset[idx]
        true_future = old_dataset.raw_future_from_normalized(world_batch.future_actions)
        old_actions = old_dataset.generate_raw_actions(old_batch, world_batch, true_future)
        new_actions = new_dataset.generate_raw_actions(new_batch, world_batch, true_future)
        old_actions[0] = true_future[0]
        new_actions[0] = true_future[0]
        proposals.append(CachedProposal(world_batch, target, old_actions, new_actions))
    return proposals


def prepare_context(args: argparse.Namespace):
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
            "base": base,
            "old_dataset": old_dataset,
            "proposals": cache_proposals(base, old_dataset, new_dataset),
        }
    return device, summary, world_model, stats, old_point, new_point, split_context


def evaluate_rule(rule: GateRule, device, summary: dict, world_model, stats: dict, split_context: dict, batch_size: int) -> dict:
    result = {}
    config = summary["config"]
    for split_name, context in split_context.items():
        dataset = CachedRuleDataset(context["proposals"], context["old_dataset"], rule)
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


def row_from_result(rule: GateRule, output_json: Path, result: dict) -> dict:
    return {
        **rule.to_json(),
        "val_active_rate_rmse": float(result["val"]["active_rate"]["active_rmse"]),
        "test_active_rate_rmse": float(result["test"]["active_rate"]["active_rmse"]),
        "val_activity_f1": float(result["val"]["activity"]["f1"]),
        "test_activity_f1": float(result["test"]["activity"]["f1"]),
        "val_link_rmse": float(result["val"]["link_rate"]["rmse"]),
        "test_link_rmse": float(result["test"]["link_rate"]["rmse"]),
        "val_gate_fraction": float(result["val"]["adaptive_gate"]["fraction"]),
        "test_gate_fraction": float(result["test"]["adaptive_gate"]["fraction"]),
        "output_json": str(output_json),
    }


def write_csv(rows: list[dict], path: Path) -> None:
    fieldnames = [
        "primary_feature",
        "primary_threshold",
        "min_active_count",
        "max_active_count",
        "val_active_rate_rmse",
        "test_active_rate_rmse",
        "val_activity_f1",
        "test_activity_f1",
        "val_link_rmse",
        "test_link_rmse",
        "val_gate_fraction",
        "test_gate_fraction",
        "output_json",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def sort_rows(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=lambda row: (_nan_to_inf(row["val_active_rate_rmse"]), _nan_to_inf(row["test_active_rate_rmse"])))


def _nan_to_inf(value) -> float:
    value = float(value)
    return float("inf") if math.isnan(value) else value


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_dir = args.output_dir / "rule_json"
    json_dir.mkdir(parents=True, exist_ok=True)
    rules = [parse_rule(rule) for rule in args.rule] if args.rule else default_rules()
    if args.limit and args.limit > 0:
        rules = rules[: args.limit]
    command_line = " ".join(sys.argv)
    (args.output_dir / "command.txt").write_text(command_line + "\n", encoding="utf-8")

    device, summary, world_model, stats, old_point, new_point, split_context = prepare_context(args)
    rows = []
    for rule in rules:
        metrics = evaluate_rule(rule, device, summary, world_model, stats, split_context, args.batch_size)
        payload = {
            "framework": "PI-JWM",
            "candidate": "v11",
            "mode": "adaptive_bridge_gate_rule_sweep",
            "old_point": old_point.__dict__,
            "new_point": new_point.__dict__,
            "gate_rule": rule.to_json(),
            **metrics,
        }
        output_json = json_dir / f"{rule.slug}.json"
        output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        rows.append(row_from_result(rule, output_json, payload))
        print(
            f"rule={rule.slug} "
            f"val_active_rate_rmse={payload['val']['active_rate']['active_rmse']:.6f} "
            f"test_active_rate_rmse={payload['test']['active_rate']['active_rmse']:.6f}"
        )

    ranked = sort_rows(rows)
    write_csv(ranked, args.output_dir / "sweep_results.csv")
    best = ranked[0]
    print(
        "best "
        f"feature={best['primary_feature']} threshold={best['primary_threshold']:g} "
        f"min_active={best['min_active_count']} max_active={best['max_active_count']} "
        f"val_active_rate_rmse={best['val_active_rate_rmse']:.6f} "
        f"test_active_rate_rmse={best['test_active_rate_rmse']:.6f} "
        f"test_f1={best['test_activity_f1']:.6f} "
        f"test_link_rmse={best['test_link_rmse']:.6f}"
    )


if __name__ == "__main__":
    main()
