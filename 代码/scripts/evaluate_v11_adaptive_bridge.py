"""Evaluate an adaptive PI-JWM v11 candidate bridge operating point."""

from __future__ import annotations

import argparse
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

from pi_jwm.v6_data import V6WorldModelDataset, collate_v6_world_model_batch, load_world_model_arrays, make_normalization_stats
from pi_jwm.v6_dual_graph import V6DualGraphBatch
from pi_jwm.v8_training import evaluate_v8_model

from evaluate_v10_policy_bridge import (
    ActionDecoderConfig,
    PolicyBridgeDataset,
    evaluate_policy_bridge,
    load_policy,
    make_action_decoder_config,
    make_action_value_decoder_config,
)
from evaluate_v9_action_ablation import load_model_for_experiment, resolve_project_path
from run_v7_action_policy import V7ActionPolicyDataset
from run_world_model_v8_full_training import add_event_memory_features, resolve_seed_splits


@dataclass(frozen=True)
class PointConfig:
    threshold: float
    value_scale: float
    value_decoder: str = "train_codebook_quantile"
    value_codebook_size: int = 9


def compute_step_gate(
    actions: torch.Tensor,
    gate_feature: str,
    gate_threshold: float,
) -> torch.Tensor:
    if actions.ndim != 3:
        raise ValueError("actions must have shape [horizon, edge, action_dim]")
    if gate_feature == "step_rb_total":
        score = actions[..., 2].sum(dim=1)
    elif gate_feature == "step_cpu_total":
        score = actions[..., 4].sum(dim=1)
    elif gate_feature == "step_rb_cpu_total":
        score = actions[..., 2].sum(dim=1) + actions[..., 4].sum(dim=1)
    elif gate_feature == "step_active_count":
        score = (actions > 1e-9).any(dim=-1).sum(dim=1).to(actions.dtype)
    else:
        raise ValueError("unknown gate_feature")
    return score >= float(gate_threshold)


def mix_actions_by_step_gate(old_actions: torch.Tensor, new_actions: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    if old_actions.shape != new_actions.shape or old_actions.ndim != 3:
        raise ValueError("old_actions and new_actions must share shape [horizon, edge, action_dim]")
    if gate.shape != (old_actions.shape[0],):
        raise ValueError("gate must have shape [horizon]")
    return torch.where(gate.to(dtype=torch.bool).reshape(-1, 1, 1), new_actions, old_actions)


class AdaptivePolicyBridgeDataset(Dataset):
    def __init__(
        self,
        base_dataset: V6WorldModelDataset,
        old_dataset: PolicyBridgeDataset,
        new_dataset: PolicyBridgeDataset,
        stats: dict,
        gate_feature: str,
        gate_threshold: float,
    ) -> None:
        if len(base_dataset) != len(old_dataset) or len(base_dataset) != len(new_dataset):
            raise ValueError("datasets must share length")
        self.base_dataset = base_dataset
        self.old_dataset = old_dataset
        self.new_dataset = new_dataset
        self.stats = stats
        self.gate_feature = gate_feature
        self.gate_threshold = float(gate_threshold)
        self.gate_true_count = 0
        self.gate_total_count = 0

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, item: int):
        world_batch, target = self.base_dataset[item]
        old_batch, _ = self.old_dataset.policy_dataset[item]
        new_batch, _ = self.new_dataset.policy_dataset[item]
        true_future = self.old_dataset.raw_future_from_normalized(world_batch.future_actions)
        old_actions = self.old_dataset.generate_raw_actions(old_batch, world_batch, true_future, item)
        new_actions = self.new_dataset.generate_raw_actions(new_batch, world_batch, true_future, item)
        old_actions[0] = true_future[0]
        new_actions[0] = true_future[0]
        gate = compute_step_gate(new_actions, self.gate_feature, self.gate_threshold)
        self.gate_true_count += int(gate.sum().item())
        self.gate_total_count += int(gate.numel())
        mixed = mix_actions_by_step_gate(old_actions, new_actions, gate)
        normalized_future = self.old_dataset.normalize_future_actions(mixed)
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


def choose_device(value: str) -> torch.device:
    if value == "cpu":
        return torch.device("cpu")
    if value == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-experiment-dir", type=Path, default=PROJECT_ROOT / "artifacts/experiments/pi_jwm_v9_expanded_v2_gpu_20260619/v2_hurdle_baseline")
    parser.add_argument("--world-checkpoint", type=Path, default=PROJECT_ROOT / "artifacts/experiments/pi_jwm_v9_expanded_v2_gpu_20260619/v2_hurdle_baseline/checkpoints/v8_dual_best.pt")
    parser.add_argument("--policy-checkpoint", type=Path, default=PROJECT_ROOT / "artifacts/experiments/pi_jwm_v10_policy_bridge_gpu_20260620/v10_action_policy_bc/checkpoints/v7_action_policy_cross_attention_best.pt")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--old-threshold", type=float, default=0.4)
    parser.add_argument("--old-scale", type=float, default=1.0)
    parser.add_argument("--new-threshold", type=float, default=0.37)
    parser.add_argument("--new-scale", type=float, default=1.06)
    parser.add_argument("--value-codebook-size", type=int, default=9)
    parser.add_argument("--gate-feature", choices=("step_rb_total", "step_cpu_total", "step_rb_cpu_total", "step_active_count"), default="step_rb_total")
    parser.add_argument("--gate-threshold", type=float, required=True)
    return parser.parse_args()


def make_point_dataset(
    arrays: dict[str, np.ndarray],
    indices: np.ndarray,
    stats: dict,
    policy_model,
    action_scale: np.ndarray,
    value_vocab: dict | None,
    device: torch.device,
    train_idx: np.ndarray,
    point: PointConfig,
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


def evaluate_adaptive_bridge(args: argparse.Namespace) -> dict:
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
    metrics = {}
    for split_name, indices in (("val", val_idx), ("test", test_idx)):
        base = V6WorldModelDataset(arrays, indices, stats)
        old_dataset = make_point_dataset(arrays, indices, stats, policy_model, action_scale, value_vocab, device, train_idx, old_point)
        new_dataset = make_point_dataset(arrays, indices, stats, policy_model, action_scale, value_vocab, device, train_idx, new_point)
        adaptive = AdaptivePolicyBridgeDataset(base, old_dataset, new_dataset, stats, args.gate_feature, args.gate_threshold)
        loader = DataLoader(adaptive, batch_size=args.batch_size, shuffle=False, collate_fn=collate_v6_world_model_batch)
        config = summary["config"]
        metrics[split_name] = evaluate_v8_model(
            world_model,
            loader,
            device,
            stats,
            rate_output_mode=config.get("rate_output_mode", "main"),
            inactive_rate_value=float(config.get("inactive_rate_value", 0.0)),
        )
        metrics[split_name]["adaptive_gate"] = {
            "true_count": int(adaptive.gate_true_count),
            "total_count": int(adaptive.gate_total_count),
            "fraction": float(adaptive.gate_true_count / max(adaptive.gate_total_count, 1)),
        }
    result = {
        "framework": "PI-JWM",
        "candidate": "v11",
        "mode": "adaptive_bridge_posthoc",
        "old_point": old_point.__dict__,
        "new_point": new_point.__dict__,
        "gate_feature": args.gate_feature,
        "gate_threshold": float(args.gate_threshold),
        **metrics,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    args = parse_args()
    result = evaluate_adaptive_bridge(args)
    print(
        f"gate={result['gate_feature']} threshold={result['gate_threshold']:g} "
        f"val_active_rate_rmse={result['val']['active_rate']['active_rmse']:.6f} "
        f"test_active_rate_rmse={result['test']['active_rate']['active_rmse']:.6f} "
        f"test_f1={result['test']['activity']['f1']:.6f} "
        f"test_link_rmse={result['test']['link_rate']['rmse']:.6f}"
    )
    print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()
