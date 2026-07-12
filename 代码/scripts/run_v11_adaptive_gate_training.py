"""Train a small adaptive step gate for PI-JWM v11 candidate strategy calibration."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from pi_jwm.v6_data import V6WorldModelDataset, collate_v6_world_model_batch, load_world_model_arrays, make_normalization_stats
from pi_jwm.v6_dual_graph import V6DualGraphBatch
from pi_jwm.v8_training import compute_v8_loss, evaluate_v8_model, move_v8_batch_to_device, move_v8_target_to_device
from pi_jwm.v11_adaptive_gate import (
    StepAdaptiveGate,
    StepThresholdGate,
    extract_step_gate_features,
    hard_gate_from_probability,
    mix_actions_with_gate_probability,
)

from evaluate_v10_policy_bridge import PolicyBridgeDataset, load_policy, make_action_decoder_config, make_action_value_decoder_config
from evaluate_v9_action_ablation import load_model_for_experiment, resolve_project_path
from run_v7_action_policy import V7ActionPolicyDataset
from run_world_model_v8_full_training import add_event_memory_features, resolve_seed_splits


class GateTrainingDataset(Dataset):
    def __init__(self, base: V6WorldModelDataset, old_policy: PolicyBridgeDataset, new_policy: PolicyBridgeDataset):
        self.base = base
        self.old_policy = old_policy
        self.new_policy = new_policy

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, item: int):
        world_batch, target = self.base[item]
        old_policy_batch, _ = self.old_policy.policy_dataset[item]
        new_policy_batch, _ = self.new_policy.policy_dataset[item]
        true_future = self.old_policy.raw_future_from_normalized(world_batch.future_actions)
        old_actions = self.old_policy.generate_raw_actions(old_policy_batch, world_batch, true_future)
        new_actions = self.new_policy.generate_raw_actions(new_policy_batch, world_batch, true_future)
        old_actions[0] = true_future[0]
        new_actions[0] = true_future[0]
        return world_batch, target, old_actions, new_actions


def collate_gate_training_batch(items):
    batches, targets, old_actions, new_actions = zip(*items)
    batch, target = collate_v6_world_model_batch(list(zip(batches, targets)))
    return batch, target, torch.stack(old_actions), torch.stack(new_actions)


def normalize_future_actions(raw_future: torch.Tensor, stats: dict) -> torch.Tensor:
    mean, std = stats["edge_a_future"]
    mean_t = torch.as_tensor(mean[0], dtype=raw_future.dtype, device=raw_future.device)
    std_t = torch.as_tensor(std[0], dtype=raw_future.dtype, device=raw_future.device)
    return (raw_future - mean_t) / std_t


def replace_future_actions(batch: V6DualGraphBatch, future_actions: torch.Tensor) -> V6DualGraphBatch:
    return V6DualGraphBatch(
        node_history=batch.node_history,
        physical_edge_history=batch.physical_edge_history,
        info_edge_history=batch.info_edge_history,
        action_history=batch.action_history,
        future_actions=future_actions,
        task_history=batch.task_history,
        link_rate_baseline=batch.link_rate_baseline,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-experiment-dir", type=Path, default=PROJECT_ROOT / "artifacts/experiments/pi_jwm_v9_expanded_v2_gpu_20260619/v2_hurdle_baseline")
    parser.add_argument("--world-checkpoint", type=Path, default=PROJECT_ROOT / "artifacts/experiments/pi_jwm_v9_expanded_v2_gpu_20260619/v2_hurdle_baseline/checkpoints/v8_dual_best.pt")
    parser.add_argument("--policy-checkpoint", type=Path, default=PROJECT_ROOT / "artifacts/experiments/pi_jwm_v10_policy_bridge_gpu_20260620/v10_action_policy_bc/checkpoints/v7_action_policy_cross_attention_best.pt")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts/experiments/pi_jwm_v11_adaptive_gate_gpu_early_20260621")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--gate-model", choices=("mlp", "threshold"), default="mlp")
    parser.add_argument("--initial-threshold", type=float, default=450.0)
    parser.add_argument("--initial-temperature", type=float, default=25.0)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--bridge-weight", type=float, default=1.0)
    parser.add_argument("--bc-weight", type=float, default=0.2)
    parser.add_argument("--entropy-weight", type=float, default=0.001)
    parser.add_argument("--manual-gate-threshold", type=float, default=450.0)
    parser.add_argument("--selection-val-target", type=float, default=234.12055082202622)
    return parser.parse_args()


def choose_device(value: str) -> torch.device:
    if value == "cpu":
        return torch.device("cpu")
    if value == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_policy_dataset(arrays, indices, stats, policy_model, action_scale, value_vocab, device, train_idx, threshold, scale):
    base = V6WorldModelDataset(arrays, indices, stats)
    policy = V7ActionPolicyDataset(arrays, indices, stats, action_scale)
    decoder_config = make_action_decoder_config("threshold", {}, 0.5)
    value_config = make_action_value_decoder_config("train_codebook_quantile", arrays, train_idx, 0.75, 9, scale)
    return PolicyBridgeDataset(
        base,
        policy,
        policy_model,
        action_scale,
        stats,
        device,
        threshold,
        "true_first_pred_rest",
        decoder_config,
        "policy",
        value_config,
        value_vocab,
    )


def load_context(args, device):
    summary = json.loads((args.world_experiment_dir / "v8_full_training_summary.json").read_text(encoding="utf-8"))
    dataset_dir = resolve_project_path(summary["dataset_dir"])
    arrays = load_world_model_arrays(dataset_dir)
    if summary["config"].get("use_event_memory_features", False):
        arrays = add_event_memory_features(arrays)
    split = summary["split_seed_spec"]
    full_train_idx, val_idx, test_idx, _ = resolve_seed_splits(
        arrays["sample_seed"], train_seeds=split["train_seeds"], val_seeds=split["val_seeds"], test_seeds=split["test_seeds"]
    )
    stats_idx, train_idx = split_stats_and_training_indices(full_train_idx, args.max_train_samples)
    reference_idx, _ = split_reference_and_training_indices(full_train_idx, args.max_train_samples)
    stats = make_normalization_stats(arrays, stats_idx)
    world = load_model_for_experiment(summary, arrays, args.world_checkpoint, device)
    policy, action_scale, _, value_vocab = load_policy(args.policy_checkpoint, device)
    world.eval(); policy.eval()
    for module in (world, policy):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    return summary, arrays, stats, world, policy, action_scale, value_vocab, reference_idx, train_idx, val_idx, test_idx


def split_stats_and_training_indices(full_train_idx: np.ndarray, max_train_samples: int) -> tuple[np.ndarray, np.ndarray]:
    full_train_idx = np.asarray(full_train_idx, dtype=np.int64)
    if max_train_samples and max_train_samples > 0:
        return full_train_idx, full_train_idx[: min(int(max_train_samples), len(full_train_idx))]
    return full_train_idx, full_train_idx


def split_reference_and_training_indices(full_train_idx: np.ndarray, max_train_samples: int) -> tuple[np.ndarray, np.ndarray]:
    full_train_idx = np.asarray(full_train_idx, dtype=np.int64)
    if max_train_samples and max_train_samples > 0:
        return full_train_idx, full_train_idx[: min(int(max_train_samples), len(full_train_idx))]
    return full_train_idx, full_train_idx


def make_loader(arrays, indices, stats, policy, action_scale, value_vocab, device, train_idx, batch_size, shuffle):
    base = V6WorldModelDataset(arrays, indices, stats)
    old_ds = make_policy_dataset(arrays, indices, stats, policy, action_scale, value_vocab, device, train_idx, 0.4, 1.0)
    new_ds = make_policy_dataset(arrays, indices, stats, policy, action_scale, value_vocab, device, train_idx, 0.37, 1.06)
    return DataLoader(
        GateTrainingDataset(base, old_ds, new_ds),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_gate_training_batch,
    )


def compute_manual_gate(new_actions: torch.Tensor, threshold: float) -> torch.Tensor:
    score = new_actions[..., 2].sum(dim=2) + new_actions[..., 4].sum(dim=2)
    return (score >= float(threshold)).to(new_actions.dtype)


def compute_weighted_gate_loss(
    bridge_loss: torch.Tensor,
    bc_loss: torch.Tensor,
    entropy: torch.Tensor,
    bridge_weight: float,
    bc_weight: float,
    entropy_weight: float,
) -> torch.Tensor:
    return bridge_weight * bridge_loss + bc_weight * bc_loss - entropy_weight * entropy


def train_epoch(model, world, loader, optimizer, stats, world_config, args, device):
    model.train()
    rows = []
    for batch, target, old_actions, new_actions in loader:
        batch = move_v8_batch_to_device(batch, device)
        target = move_v8_target_to_device(target, device)
        old_actions = old_actions.to(device)
        new_actions = new_actions.to(device)
        gate = compute_gate_probability(model, old_actions, new_actions)
        mixed = mix_actions_with_gate_probability(old_actions, new_actions, gate)
        normalized = normalize_future_actions(mixed, stats)
        bridged = replace_future_actions(batch, normalized)
        outputs = world(bridged)
        bridge_loss, parts = compute_v8_loss(
            outputs,
            target,
            rate_loss_mode="active_only",
            rate_output_mode=world_config.get("rate_output_mode", "main"),
            inactive_rate_value=float(world_config.get("inactive_rate_value", 0.0)),
            node_loss_weight=0.0,
            activity_loss_weight=0.0,
            rate_loss_weight=1.0,
            task_loss_weight=0.0,
            activity_loss_mode=world_config.get("activity_loss_mode", "bce"),
            activity_pos_weight=float(world_config.get("activity_pos_weight", 80.0)),
        )
        manual_gate = compute_manual_gate(new_actions, args.manual_gate_threshold)
        bc_loss = nn.functional.binary_cross_entropy(gate, manual_gate)
        entropy = -(gate.clamp(1e-6, 1 - 1e-6) * torch.log(gate.clamp(1e-6, 1 - 1e-6)) + (1 - gate).clamp(1e-6, 1 - 1e-6) * torch.log((1 - gate).clamp(1e-6, 1 - 1e-6))).mean()
        loss = compute_weighted_gate_loss(
            bridge_loss,
            bc_loss,
            entropy,
            args.bridge_weight,
            args.bc_weight,
            args.entropy_weight,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        rows.append({"loss": float(loss.detach().cpu()), "bridge": float(bridge_loss.detach().cpu()), "bc": float(bc_loss.detach().cpu()), "gate_mean": float(gate.detach().mean().cpu())})
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


class LearnedGateEvalDataset(Dataset):
    def __init__(self, train_dataset: GateTrainingDataset, gate_model: StepAdaptiveGate, stats: dict, device: torch.device, hard: bool):
        self.train_dataset = train_dataset
        self.gate_model = gate_model
        self.stats = stats
        self.device = device
        self.hard = hard

    def __len__(self):
        return len(self.train_dataset)

    def __getitem__(self, item):
        batch, target, old_actions, new_actions = self.train_dataset[item]
        with torch.no_grad():
            old_b = old_actions.unsqueeze(0).to(self.device)
            new_b = new_actions.unsqueeze(0).to(self.device)
            gate = compute_gate_probability(self.gate_model, old_b, new_b)
            if self.hard:
                if isinstance(self.gate_model, StepThresholdGate):
                    gate = self.gate_model.hard(new_b).to(new_b.dtype)
                else:
                    gate = hard_gate_from_probability(gate).to(new_b.dtype)
            mixed = mix_actions_with_gate_probability(old_b, new_b, gate).squeeze(0).cpu()
        normalized = normalize_future_actions(mixed, self.stats)
        bridged = V6DualGraphBatch(batch.node_history, batch.physical_edge_history, batch.info_edge_history, batch.action_history, normalized, batch.task_history, batch.link_rate_baseline)
        return bridged, target


def evaluate_gate(model, arrays, indices, stats, policy, action_scale, value_vocab, device, train_idx, world, world_config, batch_size, hard):
    base = V6WorldModelDataset(arrays, indices, stats)
    old_ds = make_policy_dataset(arrays, indices, stats, policy, action_scale, value_vocab, device, train_idx, 0.4, 1.0)
    new_ds = make_policy_dataset(arrays, indices, stats, policy, action_scale, value_vocab, device, train_idx, 0.37, 1.06)
    raw_ds = GateTrainingDataset(base, old_ds, new_ds)
    eval_ds = LearnedGateEvalDataset(raw_ds, model, stats, device, hard=hard)
    loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_v6_world_model_batch)
    return evaluate_v8_model(
        world,
        loader,
        device,
        stats,
        rate_output_mode=world_config.get("rate_output_mode", "main"),
        inactive_rate_value=float(world_config.get("inactive_rate_value", 0.0)),
    )


def compute_gate_probability(model: nn.Module, old_actions: torch.Tensor, new_actions: torch.Tensor) -> torch.Tensor:
    if isinstance(model, StepThresholdGate):
        return model(new_actions)
    return model(extract_step_gate_features(old_actions, new_actions))


def main() -> None:
    args = parse_args()
    torch.manual_seed(20260621)
    np.random.seed(20260621)
    device = choose_device(args.device)
    summary, arrays, stats, world, policy, action_scale, value_vocab, reference_idx, train_idx, val_idx, test_idx = load_context(args, device)
    loader = make_loader(arrays, train_idx, stats, policy, action_scale, value_vocab, device, reference_idx, args.batch_size, shuffle=True)
    if args.gate_model == "threshold":
        model = StepThresholdGate(args.initial_threshold, args.initial_temperature).to(device)
    else:
        model = StepAdaptiveGate(hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    history = []
    val_soft = evaluate_gate(model, arrays, val_idx, stats, policy, action_scale, value_vocab, device, reference_idx, world, summary["config"], args.batch_size, hard=False)
    val_hard = evaluate_gate(model, arrays, val_idx, stats, policy, action_scale, value_vocab, device, reference_idx, world, summary["config"], args.batch_size, hard=True)
    best = {
        "epoch": 0,
        "train": None,
        "val_soft_active_rate_rmse": val_soft["active_rate"]["active_rmse"],
        "val_hard_active_rate_rmse": val_hard["active_rate"]["active_rmse"],
        "val_hard_f1": val_hard["activity"]["f1"],
        "val_hard_link_rmse": val_hard["link_rate"]["rmse"],
    }
    print(json.dumps(best), flush=True)
    history.append(best)
    best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    for epoch in range(1, args.epochs + 1):
        train = train_epoch(model, world, loader, optimizer, stats, summary["config"], args, device)
        val_soft = evaluate_gate(model, arrays, val_idx, stats, policy, action_scale, value_vocab, device, reference_idx, world, summary["config"], args.batch_size, hard=False)
        val_hard = evaluate_gate(model, arrays, val_idx, stats, policy, action_scale, value_vocab, device, reference_idx, world, summary["config"], args.batch_size, hard=True)
        row = {
            "epoch": epoch,
            "train": train,
            "val_soft_active_rate_rmse": val_soft["active_rate"]["active_rmse"],
            "val_hard_active_rate_rmse": val_hard["active_rate"]["active_rmse"],
            "val_hard_f1": val_hard["activity"]["f1"],
            "val_hard_link_rmse": val_hard["link_rate"]["rmse"],
        }
        print(json.dumps(row), flush=True)
        history.append(row)
        score = row["val_hard_active_rate_rmse"]
        if best is None or score < best["val_hard_active_rate_rmse"]:
            best = row
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    test_hard = evaluate_gate(model, arrays, test_idx, stats, policy, action_scale, value_vocab, device, reference_idx, world, summary["config"], args.batch_size, hard=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "best": best, "config": vars(args)}, args.output_dir / "adaptive_gate_best.pt")
    result = {"framework": "PI-JWM", "candidate": "v11", "history": history, "best": best, "test_hard": test_hard, "output_dir": str(args.output_dir)}
    (args.output_dir / "adaptive_gate_training_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("SUMMARY", json.dumps({"best": best, "test_active_rate_rmse": test_hard["active_rate"]["active_rmse"], "test_f1": test_hard["activity"]["f1"], "test_link_rmse": test_hard["link_rate"]["rmse"]}))


if __name__ == "__main__":
    main()
