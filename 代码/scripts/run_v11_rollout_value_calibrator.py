"""Train a conservative rollout-aligned value calibrator for PI-JWM v11 candidates."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from pi_jwm.v6_data import (
    V6WorldModelDataset,
    collate_v6_world_model_batch,
    load_world_model_arrays,
    make_normalization_stats,
)
from pi_jwm.v6_dual_graph import V6DualGraphBatch
from pi_jwm.v8_training import compute_v8_loss, evaluate_v8_model, move_v8_batch_to_device, move_v8_target_to_device
from pi_jwm.v11_rollout_value_calibrator import (
    RolloutAlignedValueCalibrator,
    compute_action_aggregate_loss,
    freeze_module,
)

from evaluate_v10_policy_bridge import load_policy, make_positive_value_quantile_codebook
from evaluate_v9_action_ablation import load_model_for_experiment, resolve_project_path
from run_world_model_v8_full_training import add_event_memory_features, resolve_seed_splits


DEFAULT_WORLD_EXPERIMENT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "experiments"
    / "pi_jwm_v9_expanded_v2_gpu_20260619"
    / "v2_hurdle_baseline"
)
DEFAULT_POLICY_CHECKPOINT = (
    PROJECT_ROOT
    / "artifacts"
    / "experiments"
    / "pi_jwm_v10_policy_bridge_gpu_20260620"
    / "v10_action_policy_bc"
    / "checkpoints"
    / "v7_action_policy_cross_attention_best.pt"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "experiments"
    / "pi_jwm_v11_rollout_value_calibrator_cpu_smoke_20260621"
)


def _tensor_stats(values: Tensor, stats: tuple[np.ndarray, np.ndarray]) -> tuple[Tensor, Tensor]:
    mean, std = stats
    mean_t = torch.as_tensor(mean, dtype=values.dtype, device=values.device)
    std_t = torch.as_tensor(std, dtype=values.dtype, device=values.device)
    return mean_t, std_t


def normalize_action_tensor(values: Tensor, stats: tuple[np.ndarray, np.ndarray]) -> Tensor:
    """Normalize raw future actions without leaving the autograd graph."""

    mean, std = _tensor_stats(values, stats)
    return (values - mean) / std


def inverse_normalize_action_tensor(values: Tensor, stats: tuple[np.ndarray, np.ndarray]) -> Tensor:
    """Restore normalized future actions without leaving the autograd graph."""

    mean, std = _tensor_stats(values, stats)
    return values * std + mean


def replace_future_actions(batch: V6DualGraphBatch, future_actions: Tensor) -> V6DualGraphBatch:
    """Return the same world-model batch with a differentiable action future."""

    return V6DualGraphBatch(
        node_history=batch.node_history,
        physical_edge_history=batch.physical_edge_history,
        info_edge_history=batch.info_edge_history,
        action_history=batch.action_history,
        future_actions=future_actions,
        task_history=batch.task_history,
        link_rate_baseline=batch.link_rate_baseline,
    )


def compute_positive_value_bc_loss(predicted: Tensor, target: Tensor) -> Tensor:
    """Apply behavior-cloning loss only where logged action values are positive."""

    positive = target > 1e-9
    if not torch.any(positive):
        return predicted.sum() * 0.0
    return nn.functional.huber_loss(predicted[positive], target[positive])


def compute_weighted_objective(
    bc_loss: Tensor,
    bridge_loss: Tensor,
    aggregate_loss: Tensor,
    bc_loss_weight: float,
    bridge_loss_weight: float,
    aggregate_loss_weight: float,
) -> Tensor:
    return (
        bc_loss_weight * bc_loss
        + bridge_loss_weight * bridge_loss
        + aggregate_loss_weight * aggregate_loss
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-experiment-dir", type=Path, default=DEFAULT_WORLD_EXPERIMENT_DIR)
    parser.add_argument("--world-checkpoint", type=Path, default=None)
    parser.add_argument("--policy-checkpoint", type=Path, default=DEFAULT_POLICY_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--max-train-samples", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--codebook-size", type=int, default=9)
    parser.add_argument("--max-relative-delta", type=float, default=0.25)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--policy-threshold", type=float, default=0.4)
    parser.add_argument("--value-scale", type=float, default=1.0)
    parser.add_argument("--new-policy-threshold", type=float, default=None)
    parser.add_argument("--new-value-scale", type=float, default=None)
    parser.add_argument("--gate-feature", choices=("none", "step_rb_total", "step_cpu_total", "step_rb_cpu_total", "step_active_count"), default="none")
    parser.add_argument("--gate-threshold", type=float, default=450.0)
    parser.add_argument("--true-first", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--expected-val-active-rmse", type=float, default=None)
    parser.add_argument("--baseline-tolerance", type=float, default=1e-3)
    parser.add_argument("--bc-loss-weight", type=float, default=1.0)
    parser.add_argument("--bridge-loss-weight", type=float, default=0.05)
    parser.add_argument("--aggregate-loss-weight", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260621)
    return parser.parse_args()


def choose_device(value: str) -> torch.device:
    if value == "cpu":
        return torch.device("cpu")
    if value == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda")
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raise ValueError(f"unknown device: {value}")


def summarize_baseline_reproduction(
    actual_val_active_rmse: float,
    expected_val_active_rmse: float | None,
    tolerance: float,
) -> dict[str, float | bool | None]:
    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative")
    if expected_val_active_rmse is None:
        return {
            "actual_val_active_rmse": float(actual_val_active_rmse),
            "expected_val_active_rmse": None,
            "delta": None,
            "tolerance": float(tolerance),
            "passed": None,
        }
    delta = float(actual_val_active_rmse) - float(expected_val_active_rmse)
    return {
        "actual_val_active_rmse": float(actual_val_active_rmse),
        "expected_val_active_rmse": float(expected_val_active_rmse),
        "delta": delta,
        "tolerance": float(tolerance),
        "passed": bool(abs(delta) <= float(tolerance)),
    }


def choose_positive_aware_subset(arrays: dict[str, np.ndarray], indices: np.ndarray, limit: int) -> np.ndarray:
    if limit <= 0:
        raise ValueError("max_train_samples must be positive")
    indices = np.asarray(indices, dtype=np.int64)
    has_positive = (arrays["edge_a_future"][indices] > 1e-9).any(axis=(1, 2, 3))
    positive_indices = indices[has_positive]
    negative_indices = indices[~has_positive]
    ordered = np.concatenate([positive_indices, negative_indices])
    return ordered[: min(limit, len(ordered))]


def make_bridge_loss(
    world_outputs: dict[str, Tensor],
    target: dict[str, Tensor],
    world_config: dict,
) -> tuple[Tensor, dict[str, float]]:
    return compute_v8_loss(
        world_outputs,
        target,
        rate_loss_mode="active_only",
        active_rate_auxiliary_weight=float(world_config.get("active_rate_auxiliary_weight", 0.0)),
        rate_output_mode=world_config.get("rate_output_mode", "main"),
        inactive_rate_value=float(world_config.get("inactive_rate_value", 0.0)),
        node_loss_weight=0.1,
        activity_loss_weight=0.1,
        rate_loss_weight=1.0,
        task_loss_weight=0.1,
        activity_loss_mode=world_config.get("activity_loss_mode", "bce"),
        activity_pos_weight=float(world_config.get("activity_pos_weight", 80.0)),
        activity_focal_gamma=float(world_config.get("activity_focal_gamma", 2.0)),
        inactive_loss_sample_ratio=1.0,
        hurdle_train_gate_mode="predicted",
        hurdle_train_gate_power=float(world_config.get("hurdle_train_gate_power", 1.0)),
    )


def build_calibrated_raw_action(
    calibrated_value: Tensor,
    fixed_activity: Tensor,
    value_scale: float,
    true_raw: Tensor | None = None,
    true_first: bool = True,
) -> Tensor:
    """Apply a fixed activity mask, value scale, and optional true-first bridge mode."""

    if calibrated_value.shape != fixed_activity.shape:
        raise ValueError("calibrated_value and fixed_activity must share shape")
    predicted_raw = torch.where(
        fixed_activity.to(dtype=torch.bool),
        calibrated_value * float(value_scale),
        torch.zeros_like(calibrated_value),
    )
    if true_first:
        if true_raw is None:
            raise ValueError("true_raw is required when true_first=True")
        if true_raw.shape != predicted_raw.shape:
            raise ValueError("true_raw must match predicted action shape")
        predicted_raw = predicted_raw.clone()
        predicted_raw[:, 0] = true_raw[:, 0]
    return predicted_raw


def compute_step_score(actions: Tensor, feature: str) -> Tensor:
    if actions.ndim != 4:
        raise ValueError("actions must have shape [batch, horizon, edge, action_dim]")
    if feature == "step_rb_total":
        return actions[..., 2].sum(dim=2)
    if feature == "step_cpu_total":
        return actions[..., 4].sum(dim=2)
    if feature == "step_rb_cpu_total":
        return actions[..., 2].sum(dim=2) + actions[..., 4].sum(dim=2)
    if feature == "step_active_count":
        return (actions > 1e-9).any(dim=-1).sum(dim=2).to(actions.dtype)
    raise ValueError(f"unknown gate feature: {feature}")


def mix_adaptive_new_action_by_step_gate(
    old_action: Tensor,
    new_action: Tensor,
    gate_feature: str,
    gate_threshold: float,
    true_raw: Tensor | None = None,
    true_first: bool = True,
) -> Tensor:
    if old_action.shape != new_action.shape or old_action.ndim != 4:
        raise ValueError("old_action and new_action must share shape [batch, horizon, edge, action_dim]")
    if gate_feature == "none":
        mixed = old_action
    else:
        gate = compute_step_score(new_action, gate_feature) >= float(gate_threshold)
        mixed = torch.where(gate.reshape(gate.shape[0], gate.shape[1], 1, 1), new_action, old_action)
    if true_first:
        if true_raw is None:
            raise ValueError("true_raw is required when true_first=True")
        if true_raw.shape != mixed.shape:
            raise ValueError("true_raw must match action shape")
        mixed = mixed.clone()
        mixed[:, 0] = true_raw[:, 0]
    return mixed


def forward_objective(
    calibrator: RolloutAlignedValueCalibrator,
    policy_model: nn.Module,
    world_model: nn.Module,
    batch: V6DualGraphBatch,
    target: dict[str, Tensor],
    action_scale: Tensor,
    action_stats: tuple[np.ndarray, np.ndarray],
    codebook: Tensor,
    policy_threshold: float,
    new_policy_threshold: float | None,
    bc_loss_weight: float,
    bridge_loss_weight: float,
    aggregate_loss_weight: float,
    world_config: dict,
    hard: bool,
    value_scale: float = 1.0,
    new_value_scale: float | None = None,
    gate_feature: str = "none",
    gate_threshold: float = 450.0,
    true_first: bool = True,
) -> tuple[Tensor, dict[str, float]]:
    with torch.no_grad():
        policy_outputs = policy_model(batch)
        activity_prob = torch.sigmoid(policy_outputs["action_logit"])
        base_value = policy_outputs["action_value"] * action_scale
        fixed_activity = activity_prob >= policy_threshold
        new_fixed_activity = activity_prob >= (policy_threshold if new_policy_threshold is None else new_policy_threshold)

    candidate_value = calibrator(
        base_value,
        activity_prob,
        codebook,
        torch.ones_like(fixed_activity),
        hard=hard,
    )
    true_raw = inverse_normalize_action_tensor(batch.future_actions, action_stats)
    old_raw = build_calibrated_raw_action(
        candidate_value,
        fixed_activity,
        value_scale=value_scale,
        true_raw=true_raw,
        true_first=true_first,
    )
    if gate_feature == "none":
        predicted_raw = old_raw
    else:
        new_raw = build_calibrated_raw_action(
            candidate_value,
            new_fixed_activity,
            value_scale=value_scale if new_value_scale is None else new_value_scale,
            true_raw=true_raw,
            true_first=true_first,
        )
        predicted_raw = mix_adaptive_new_action_by_step_gate(
            old_raw,
            new_raw,
            gate_feature=gate_feature,
            gate_threshold=gate_threshold,
            true_raw=true_raw,
            true_first=true_first,
        )
    normalized_predicted = normalize_action_tensor(predicted_raw, action_stats)
    world_outputs = world_model(replace_future_actions(batch, normalized_predicted))
    bridge_loss, bridge_parts = make_bridge_loss(world_outputs, target, world_config)
    bc_loss = compute_positive_value_bc_loss(candidate_value, true_raw)
    aggregate_loss = compute_action_aggregate_loss(predicted_raw, true_raw)
    total = compute_weighted_objective(
        bc_loss,
        bridge_loss,
        aggregate_loss,
        bc_loss_weight,
        bridge_loss_weight,
        aggregate_loss_weight,
    )
    metrics = {
        "total": float(total.detach().cpu()),
        "bc": float(bc_loss.detach().cpu()),
        "bridge": float(bridge_loss.detach().cpu()),
        "aggregate": float(aggregate_loss.detach().cpu()),
        "policy_active_fraction": float(fixed_activity.float().mean().detach().cpu()),
        "candidate_mean": float(candidate_value.mean().detach().cpu()),
        "predicted_active_mean": float(predicted_raw[fixed_activity].mean().detach().cpu()) if fixed_activity.any() else 0.0,
        "bridge_active_rate_loss": float(bridge_parts["active_rate_loss"]),
    }
    return total, metrics


class CalibratedBridgeDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_dataset: V6WorldModelDataset,
        policy_model: nn.Module,
        calibrator: RolloutAlignedValueCalibrator,
        action_scale: Tensor,
        action_stats: tuple[np.ndarray, np.ndarray],
        codebook: Tensor,
        policy_threshold: float,
        value_scale: float,
        new_policy_threshold: float | None,
        new_value_scale: float | None,
        gate_feature: str,
        gate_threshold: float,
        true_first: bool,
        device: torch.device,
        hard: bool = True,
    ) -> None:
        self.base_dataset = base_dataset
        self.policy_model = policy_model
        self.calibrator = calibrator
        self.action_scale = action_scale
        self.action_stats = action_stats
        self.codebook = codebook
        self.policy_threshold = float(policy_threshold)
        self.value_scale = float(value_scale)
        self.new_policy_threshold = None if new_policy_threshold is None else float(new_policy_threshold)
        self.new_value_scale = None if new_value_scale is None else float(new_value_scale)
        self.gate_feature = str(gate_feature)
        self.gate_threshold = float(gate_threshold)
        self.true_first = bool(true_first)
        self.device = device
        self.hard = bool(hard)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, item: int):
        world_batch, target = self.base_dataset[item]
        batch = V6DualGraphBatch(
            node_history=world_batch.node_history.unsqueeze(0).to(self.device),
            physical_edge_history=world_batch.physical_edge_history.unsqueeze(0).to(self.device),
            info_edge_history=world_batch.info_edge_history.unsqueeze(0).to(self.device),
            action_history=world_batch.action_history.unsqueeze(0).to(self.device),
            future_actions=world_batch.future_actions.unsqueeze(0).to(self.device),
            task_history=world_batch.task_history.unsqueeze(0).to(self.device),
            link_rate_baseline=None if world_batch.link_rate_baseline is None else world_batch.link_rate_baseline.unsqueeze(0).to(self.device),
        )
        with torch.no_grad():
            policy_outputs = self.policy_model(batch)
            activity_prob = torch.sigmoid(policy_outputs["action_logit"])
            base_value = policy_outputs["action_value"] * self.action_scale
            fixed_activity = activity_prob >= self.policy_threshold
            new_fixed_activity = activity_prob >= (
                self.policy_threshold if self.new_policy_threshold is None else self.new_policy_threshold
            )
            candidate_value = self.calibrator(
                base_value,
                activity_prob,
                self.codebook,
                torch.ones_like(fixed_activity),
                hard=self.hard,
            )
            true_raw = inverse_normalize_action_tensor(batch.future_actions, self.action_stats)
            old_raw = build_calibrated_raw_action(
                candidate_value,
                fixed_activity,
                value_scale=self.value_scale,
                true_raw=true_raw,
                true_first=self.true_first,
            )
            if self.gate_feature == "none":
                predicted_raw = old_raw
            else:
                new_raw = build_calibrated_raw_action(
                    candidate_value,
                    new_fixed_activity,
                    value_scale=self.value_scale if self.new_value_scale is None else self.new_value_scale,
                    true_raw=true_raw,
                    true_first=self.true_first,
                )
                predicted_raw = mix_adaptive_new_action_by_step_gate(
                    old_raw,
                    new_raw,
                    gate_feature=self.gate_feature,
                    gate_threshold=self.gate_threshold,
                    true_raw=true_raw,
                    true_first=self.true_first,
                )
            normalized_predicted = normalize_action_tensor(predicted_raw, self.action_stats).squeeze(0).cpu()
        bridged = replace_future_actions(world_batch, normalized_predicted)
        return bridged, target


def evaluate_calibrated_bridge(
    arrays: dict[str, np.ndarray],
    indices: np.ndarray,
    stats: dict,
    world_model: nn.Module,
    policy_model: nn.Module,
    calibrator: RolloutAlignedValueCalibrator,
    action_scale: Tensor,
    codebook: Tensor,
    policy_threshold: float,
    value_scale: float,
    new_policy_threshold: float | None,
    new_value_scale: float | None,
    gate_feature: str,
    gate_threshold: float,
    true_first: bool,
    world_config: dict,
    device: torch.device,
    batch_size: int,
) -> dict:
    dataset = CalibratedBridgeDataset(
        V6WorldModelDataset(arrays, indices, stats),
        policy_model,
        calibrator,
        action_scale,
        stats["edge_a_future"],
        codebook,
        policy_threshold,
        value_scale,
        new_policy_threshold,
        new_value_scale,
        gate_feature,
        gate_threshold,
        true_first,
        device,
        hard=True,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_v6_world_model_batch)
    return evaluate_v8_model(
        world_model,
        loader,
        device,
        stats,
        rate_output_mode=world_config.get("rate_output_mode", "main"),
        inactive_rate_value=float(world_config.get("inactive_rate_value", 0.0)),
    )


def average_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def run_experiment(args: argparse.Namespace) -> dict:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)

    world_experiment_dir = args.world_experiment_dir.resolve()
    world_summary = json.loads(
        (world_experiment_dir / "v8_full_training_summary.json").read_text(encoding="utf-8")
    )
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
    train_used = choose_positive_aware_subset(arrays, train_idx, args.max_train_samples)
    loader = DataLoader(
        V6WorldModelDataset(arrays, train_used, stats),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_v6_world_model_batch,
    )

    world_checkpoint = args.world_checkpoint or world_experiment_dir / "checkpoints" / "v8_dual_best.pt"
    world_model = freeze_module(load_model_for_experiment(world_summary, arrays, world_checkpoint, device))
    policy_model, action_scale_np, _, value_vocab = load_policy(args.policy_checkpoint.resolve(), device)
    if value_vocab is not None or not hasattr(policy_model, "action_value_head"):
        raise ValueError("rollout calibrator CPU smoke requires a continuous-value policy checkpoint")
    policy_model = freeze_module(policy_model)

    raw_train_actions = arrays["edge_a_future"][train_idx]
    codebook_np = make_positive_value_quantile_codebook(raw_train_actions, args.codebook_size)
    codebook = torch.as_tensor(codebook_np, dtype=torch.float32, device=device)
    horizon, _, action_dim = raw_train_actions.shape[1:]
    calibrator = RolloutAlignedValueCalibrator(
        horizon=horizon,
        action_dim=action_dim,
        codebook_size=args.codebook_size,
        hidden_dim=args.hidden_dim,
        max_relative_delta=args.max_relative_delta,
        temperature=args.temperature,
    ).to(device)
    optimizer = torch.optim.Adam(calibrator.parameters(), lr=args.learning_rate)
    action_scale = torch.as_tensor(
        action_scale_np.reshape(1, 1, 1, -1), dtype=torch.float32, device=device
    )
    new_policy_threshold = args.policy_threshold if args.new_policy_threshold is None else args.new_policy_threshold
    new_value_scale = args.value_scale if args.new_value_scale is None else args.new_value_scale

    calibrator.eval()
    initial_bridge_eval = {
        "val": evaluate_calibrated_bridge(
            arrays,
            val_idx,
            stats,
            world_model,
            policy_model,
            calibrator,
            action_scale,
            codebook,
            args.policy_threshold,
            args.value_scale,
            new_policy_threshold,
            new_value_scale,
            args.gate_feature,
            args.gate_threshold,
            args.true_first,
            world_summary["config"],
            device,
            args.batch_size,
        ),
        "test": evaluate_calibrated_bridge(
            arrays,
            test_idx,
            stats,
            world_model,
            policy_model,
            calibrator,
            action_scale,
            codebook,
            args.policy_threshold,
            args.value_scale,
            new_policy_threshold,
            new_value_scale,
            args.gate_feature,
            args.gate_threshold,
            args.true_first,
            world_summary["config"],
            device,
            args.batch_size,
        ),
    }
    baseline_reproduction = summarize_baseline_reproduction(
        initial_bridge_eval["val"]["active_rate"]["active_rmse"],
        args.expected_val_active_rmse,
        args.baseline_tolerance,
    )

    batches = list(loader)
    first_batch = move_v8_batch_to_device(batches[0][0], device)
    first_target = move_v8_target_to_device(batches[0][1], device)
    calibrator.eval()
    with torch.no_grad():
        _, initial_soft = forward_objective(
            calibrator, policy_model, world_model, first_batch, first_target, action_scale,
            stats["edge_a_future"], codebook, args.policy_threshold, new_policy_threshold, args.bc_loss_weight, args.bridge_loss_weight,
            args.aggregate_loss_weight, world_summary["config"], hard=False, value_scale=args.value_scale,
            new_value_scale=new_value_scale, gate_feature=args.gate_feature, gate_threshold=args.gate_threshold, true_first=args.true_first,
        )
        _, initial_hard = forward_objective(
            calibrator, policy_model, world_model, first_batch, first_target, action_scale,
            stats["edge_a_future"], codebook, args.policy_threshold, new_policy_threshold, args.bc_loss_weight, args.bridge_loss_weight,
            args.aggregate_loss_weight, world_summary["config"], hard=True, value_scale=args.value_scale,
            new_value_scale=new_value_scale, gate_feature=args.gate_feature, gate_threshold=args.gate_threshold, true_first=args.true_first,
        )

    history = []
    saw_nonzero_gradient = False
    for epoch in range(1, args.epochs + 1):
        calibrator.train()
        rows = []
        for raw_batch, raw_target in batches:
            batch = move_v8_batch_to_device(raw_batch, device)
            target = move_v8_target_to_device(raw_target, device)
            optimizer.zero_grad(set_to_none=True)
            total, metrics = forward_objective(
                calibrator, policy_model, world_model, batch, target, action_scale,
                stats["edge_a_future"], codebook, args.policy_threshold, new_policy_threshold, args.bc_loss_weight, args.bridge_loss_weight,
                args.aggregate_loss_weight, world_summary["config"], hard=False, value_scale=args.value_scale,
                new_value_scale=new_value_scale, gate_feature=args.gate_feature, gate_threshold=args.gate_threshold, true_first=args.true_first,
            )
            total.backward()
            grad_norm = torch.sqrt(
                sum((parameter.grad.detach().pow(2).sum() for parameter in calibrator.parameters() if parameter.grad is not None),
                    start=torch.zeros((), device=device))
            )
            saw_nonzero_gradient = saw_nonzero_gradient or float(grad_norm) > 0.0
            optimizer.step()
            metrics["gradient_norm"] = float(grad_norm.detach().cpu())
            rows.append(metrics)
        epoch_metrics = average_metrics(rows)
        epoch_metrics["epoch"] = epoch
        history.append(epoch_metrics)
        print(json.dumps(epoch_metrics, ensure_ascii=False))

    calibrator.eval()
    with torch.no_grad():
        _, final_soft = forward_objective(
            calibrator, policy_model, world_model, first_batch, first_target, action_scale,
            stats["edge_a_future"], codebook, args.policy_threshold, new_policy_threshold, args.bc_loss_weight, args.bridge_loss_weight,
            args.aggregate_loss_weight, world_summary["config"], hard=False, value_scale=args.value_scale,
            new_value_scale=new_value_scale, gate_feature=args.gate_feature, gate_threshold=args.gate_threshold, true_first=args.true_first,
        )
        _, final_hard = forward_objective(
            calibrator, policy_model, world_model, first_batch, first_target, action_scale,
            stats["edge_a_future"], codebook, args.policy_threshold, new_policy_threshold, args.bc_loss_weight, args.bridge_loss_weight,
            args.aggregate_loss_weight, world_summary["config"], hard=True, value_scale=args.value_scale,
            new_value_scale=new_value_scale, gate_feature=args.gate_feature, gate_threshold=args.gate_threshold, true_first=args.true_first,
        )
    final_bridge_eval = {
        "val": evaluate_calibrated_bridge(
            arrays,
            val_idx,
            stats,
            world_model,
            policy_model,
            calibrator,
            action_scale,
            codebook,
            args.policy_threshold,
            args.value_scale,
            new_policy_threshold,
            new_value_scale,
            args.gate_feature,
            args.gate_threshold,
            args.true_first,
            world_summary["config"],
            device,
            args.batch_size,
        ),
        "test": evaluate_calibrated_bridge(
            arrays,
            test_idx,
            stats,
            world_model,
            policy_model,
            calibrator,
            action_scale,
            codebook,
            args.policy_threshold,
            args.value_scale,
            new_policy_threshold,
            new_value_scale,
            args.gate_feature,
            args.gate_threshold,
            args.true_first,
            world_summary["config"],
            device,
            args.batch_size,
        ),
    }

    frozen_gradients_clear = all(parameter.grad is None for parameter in policy_model.parameters()) and all(
        parameter.grad is None for parameter in world_model.parameters()
    )
    summary = {
        "framework": "PI-JWM",
        "candidate": "v11",
        "status": "cpu_pipeline_smoke_only",
        "interpretation_guard": (
            "Uses the recovered v10 continuous-value action policy checkpoint by default. "
            "Interpret training quality only after baseline_reproduction is checked."
        ),
        "device": str(device),
        "dataset_dir": str(dataset_dir),
        "world_experiment_dir": str(world_experiment_dir),
        "world_checkpoint": str(Path(world_checkpoint).resolve()),
        "policy_checkpoint": str(args.policy_checkpoint.resolve()),
        "train_samples": int(len(train_used)),
        "config": vars(args) | {
            "world_experiment_dir": str(args.world_experiment_dir),
            "world_checkpoint": None if args.world_checkpoint is None else str(args.world_checkpoint),
            "policy_checkpoint": str(args.policy_checkpoint),
            "output_dir": str(args.output_dir),
        },
        "initial_first_batch_soft": initial_soft,
        "final_first_batch_soft": final_soft,
        "initial_first_batch_hard": initial_hard,
        "final_first_batch_hard": final_hard,
        "initial_bridge_eval": initial_bridge_eval,
        "final_bridge_eval": final_bridge_eval,
        "baseline_reproduction": baseline_reproduction,
        "soft_total_delta": final_soft["total"] - initial_soft["total"],
        "hard_total_delta": final_hard["total"] - initial_hard["total"],
        "saw_nonzero_calibrator_gradient": saw_nonzero_gradient,
        "frozen_policy_and_world_gradients_clear": frozen_gradients_clear,
        "history": history,
        "codebook": codebook_np.tolist(),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": calibrator.state_dict(),
            "config": {
                "horizon": horizon,
                "action_dim": action_dim,
                "codebook_size": args.codebook_size,
                "hidden_dim": args.hidden_dim,
                "max_relative_delta": args.max_relative_delta,
                "temperature": args.temperature,
            },
            "codebook": codebook_np,
            "policy_threshold": args.policy_threshold,
        },
        checkpoint_dir / "v11_rollout_value_calibrator_cpu_smoke.pt",
    )
    summary_path = args.output_dir / "v11_rollout_value_calibrator_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"summary={summary_path.resolve()}")
    return summary


if __name__ == "__main__":
    run_experiment(parse_args())
