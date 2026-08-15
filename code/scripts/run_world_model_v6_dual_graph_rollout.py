"""Local CPU sanity run for the PI-JWM v6 dual-graph rollout skeleton."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pi_jwm.paths import ARTIFACTS_DIR
from pi_jwm.v6_data import (
    V6WorldModelDataset,
    collate_v6_world_model_batch,
    inverse_transform_link_rate,
    inverse_normalize,
    load_world_model_arrays,
    make_normalization_stats,
    split_by_seed,
)
from pi_jwm.v6_dual_graph import V6DualGraphBatch, V6DualGraphConfig, V6DualGraphRollout
from pi_jwm.v6_metrics import active_rate_metrics, activity_metrics, regression_metrics


DEFAULT_DATASET_DIR = (
    ARTIFACTS_DIR
    / "experiments"
    / "airfogsim_v0"
    / "datasets"
    / "world_model_dataset_seed0_9_v0"
)
OUTPUT_DIR = ARTIFACTS_DIR / "experiments" / "pi_jwm_v6_dual_graph"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PI-JWM v6 dual-graph sanity checks.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--synthetic-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-train-samples", type=int, default=128)
    parser.add_argument("--max-val-samples", type=int, default=64)
    parser.add_argument("--max-test-samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--graph-mode",
        choices=("dual", "physical_only", "information_only", "all"),
        default="dual",
    )
    parser.add_argument(
        "--fusion-mode",
        choices=("concat", "gated", "cross_attention", "hybrid_attention"),
        default="concat",
        help="Dual-graph fusion operator. concat reproduces the original v6 setting.",
    )
    parser.add_argument("--fusion-num-heads", type=int, default=4)
    parser.add_argument(
        "--rate-loss-mode",
        choices=("weighted_all", "active_only", "active_mixed"),
        default="weighted_all",
        help="Rate-head training loss. weighted_all reproduces v6; active modes focus the loss on active links.",
    )
    parser.add_argument(
        "--rate-target-transform",
        choices=("raw", "log1p_raw", "residual_last_rate"),
        default="raw",
        help="Target space for link-rate regression. Metrics are converted back to raw rate space.",
    )
    parser.add_argument(
        "--rate-inverse-clip-quantile",
        type=float,
        default=0.0,
        help=(
            "Optional upper quantile in transformed train targets used to clip rate predictions before "
            "inverse-transforming them for raw-space metrics. 0 disables clipping."
        ),
    )
    parser.add_argument(
        "--rate-head-mode",
        choices=("direct", "activity_gated", "residual_activity_gated"),
        default="direct",
        help=(
            "direct reproduces the existing rate head; activity_gated multiplies the raw rate value by "
            "activity probability; residual_activity_gated adds an activity-gated residual to the base rate."
        ),
    )
    parser.add_argument(
        "--rate-gate-temperature",
        type=float,
        default=1.0,
        help="Temperature for activity-derived rate gates. Larger values make the gate softer.",
    )
    parser.add_argument(
        "--rate-gate-floor",
        type=float,
        default=0.0,
        help="Lower bound for activity-derived rate gates. Use with residual_activity_gated to avoid hard suppression.",
    )
    parser.add_argument(
        "--inactive-rate-weight",
        type=float,
        default=0.05,
        help="Inactive-link rate penalty used by --rate-loss-mode active_mixed.",
    )
    parser.add_argument(
        "--rate-teacher-mode",
        choices=("none", "random_forest_active", "ridge_active"),
        default="none",
        help="Optional active-rate teacher used as an auxiliary training target.",
    )
    parser.add_argument(
        "--rate-teacher-weight",
        type=float,
        default=0.0,
        help="Weight for optional active-rate teacher loss.",
    )
    parser.add_argument("--rate-teacher-n-estimators", type=int, default=400)
    parser.add_argument("--rate-teacher-min-samples-leaf", type=int, default=2)
    parser.add_argument("--rate-teacher-max-features", type=float, default=0.8)
    parser.add_argument("--rate-teacher-ridge-lambda", type=float, default=10.0)
    parser.add_argument(
        "--active-rate-auxiliary",
        action="store_true",
        help="Add an auxiliary neural rate head trained only on true active links.",
    )
    parser.add_argument(
        "--active-rate-auxiliary-weight",
        type=float,
        default=0.0,
        help="Loss weight for --active-rate-auxiliary.",
    )
    parser.add_argument(
        "--rate-output-mode",
        choices=("main", "aux_soft_zero", "aux_hard_zero", "aux_oracle_zero"),
        default="main",
        help=(
            "Final link-rate path used for training/evaluation. main uses the original link_rate head; "
            "aux_soft_zero uses activity probability times the auxiliary active-rate head; "
            "aux_hard_zero uses thresholded predicted activity; "
            "aux_oracle_zero uses true activity as a diagnostic upper-bound selector."
        ),
    )
    parser.add_argument(
        "--inactive-rate-value",
        type=float,
        default=None,
        help=(
            "Normalized inactive-rate value used by active-selected rate outputs. "
            "Defaults to the normalized value corresponding to raw rate 0."
        ),
    )
    parser.add_argument(
        "--best-metric",
        choices=("total", "val_active_rate_rmse", "val_link_rate_rmse", "val_task_rmse", "val_activity_f1"),
        default="total",
        help="Validation criterion for checkpoint selection.",
    )
    parser.add_argument("--seed", type=int, default=20260529)
    return parser.parse_args()


def build_synthetic_batch(config: V6DualGraphConfig) -> V6DualGraphBatch:
    batch_size = 4
    history = 5
    num_nodes = 6
    num_edges = 10

    generator = torch.Generator().manual_seed(20260529)
    return V6DualGraphBatch(
        node_history=torch.randn(batch_size, history, num_nodes, config.node_dim, generator=generator),
        physical_edge_history=torch.randn(
            batch_size,
            history,
            num_edges,
            config.physical_edge_dim,
            generator=generator,
        ),
        info_edge_history=torch.randn(batch_size, history, num_edges, config.info_edge_dim, generator=generator),
        action_history=torch.randn(batch_size, history, num_edges, config.action_dim, generator=generator),
        future_actions=torch.randn(batch_size, config.horizon, num_edges, config.action_dim, generator=generator),
        task_history=torch.randn(batch_size, history, config.task_dim, generator=generator),
        link_rate_baseline=torch.zeros(batch_size, config.horizon, num_edges, 1),
    )


def run_synthetic_smoke(
    seed: int,
    fusion_mode: str = "concat",
    fusion_num_heads: int = 4,
    rate_head_mode: str = "direct",
    rate_gate_temperature: float = 1.0,
    rate_gate_floor: float = 0.0,
) -> dict:
    torch.manual_seed(seed)
    config = V6DualGraphConfig(
        node_dim=6,
        physical_edge_dim=8,
        info_edge_dim=5,
        action_dim=4,
        task_dim=3,
        hidden_dim=32,
        horizon=4,
        fusion_mode=fusion_mode,
        fusion_num_heads=fusion_num_heads,
        rate_head_mode=rate_head_mode,
        rate_gate_temperature=rate_gate_temperature,
        rate_gate_floor=rate_gate_floor,
    )
    model = V6DualGraphRollout(config)
    batch = build_synthetic_batch(config)

    with torch.no_grad():
        outputs = model(batch)

    return {
        "status": "smoke",
        "framework": "PI-JWM",
        "note": "Synthetic-interface smoke run only; not a trained experiment result.",
        "config": config.__dict__,
        "output_shapes": {name: list(value.shape) for name, value in outputs.items()},
    }


def make_config_from_arrays(
    arrays: dict,
    hidden_dim: int,
    graph_mode: str,
    fusion_mode: str = "concat",
    fusion_num_heads: int = 4,
    rate_head_mode: str = "direct",
    rate_gate_temperature: float = 1.0,
    rate_gate_floor: float = 0.0,
    active_rate_auxiliary: bool = False,
) -> V6DualGraphConfig:
    return V6DualGraphConfig(
        node_dim=int(arrays["x_node"].shape[-1]),
        physical_edge_dim=8,
        info_edge_dim=int(arrays["x_link"].shape[-1]),
        action_dim=int(arrays["edge_a_hist"].shape[-1]),
        task_dim=int(arrays["x_task"].shape[-1]),
        hidden_dim=hidden_dim,
        horizon=int(arrays["edge_a_future"].shape[1]),
        graph_mode=graph_mode,
        fusion_mode=fusion_mode,
        fusion_num_heads=fusion_num_heads,
        rate_head_mode=rate_head_mode,
        rate_gate_temperature=rate_gate_temperature,
        rate_gate_floor=rate_gate_floor,
        active_rate_auxiliary=active_rate_auxiliary,
    )


def move_batch_to_device(batch: V6DualGraphBatch, device: torch.device) -> V6DualGraphBatch:
    return V6DualGraphBatch(
        node_history=batch.node_history.to(device),
        physical_edge_history=batch.physical_edge_history.to(device),
        info_edge_history=batch.info_edge_history.to(device),
        action_history=batch.action_history.to(device),
        future_actions=batch.future_actions.to(device),
        task_history=batch.task_history.to(device),
        link_rate_baseline=(
            batch.link_rate_baseline.to(device) if batch.link_rate_baseline is not None else None
        ),
    )


def move_target_to_device(target: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: value.to(device) for name, value in target.items()}


def compute_loss(
    outputs: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    rate_loss_mode: str = "weighted_all",
    inactive_rate_weight: float = 0.05,
    rate_teacher_weight: float = 0.0,
    active_rate_auxiliary_weight: float = 0.0,
    rate_output_mode: str = "main",
    inactive_rate_value: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    if rate_teacher_weight < 0.0:
        raise ValueError("rate_teacher_weight must be non-negative")
    if active_rate_auxiliary_weight < 0.0:
        raise ValueError("active_rate_auxiliary_weight must be non-negative")
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([80.0], device=outputs["link_activity_logit"].device))
    node_loss = mse(outputs["node"], target["node"])
    activity_loss = bce(outputs["link_activity_logit"], target["link_activity"])
    train_rate_output_mode = "main" if rate_output_mode == "aux_hard_zero" else rate_output_mode
    link_rate_output = select_link_rate_output(
        outputs,
        target=target,
        rate_output_mode=train_rate_output_mode,
        inactive_rate_value=inactive_rate_value,
    )
    rate_error = (link_rate_output - target["link_rate"]) ** 2
    active_mask = target["link_activity"] > 0.5
    inactive_mask = ~active_mask
    if active_mask.any():
        active_rate_loss = rate_error[active_mask].mean()
    else:
        active_rate_loss = rate_error.new_tensor(0.0)
    if inactive_mask.any():
        inactive_rate_loss = rate_error[inactive_mask].mean()
    else:
        inactive_rate_loss = rate_error.new_tensor(0.0)

    if rate_loss_mode == "weighted_all":
        active_weight = 1.0 + 50.0 * target["link_activity"]
        rate_loss = (rate_error * active_weight).mean()
    elif rate_loss_mode == "active_only":
        rate_loss = active_rate_loss
    elif rate_loss_mode == "active_mixed":
        rate_loss = active_rate_loss + inactive_rate_weight * inactive_rate_loss
    else:
        raise ValueError("rate_loss_mode must be one of: weighted_all, active_only, active_mixed")
    teacher_loss = rate_error.new_tensor(0.0)
    if rate_teacher_weight > 0.0 and "link_rate_teacher" in target and "link_rate_teacher_mask" in target:
        teacher_mask = target["link_rate_teacher_mask"] > 0.5
        if teacher_mask.any():
            teacher_error = (link_rate_output - target["link_rate_teacher"]) ** 2
            teacher_loss = teacher_error[teacher_mask].mean()
    active_rate_auxiliary_loss = rate_error.new_tensor(0.0)
    if active_rate_auxiliary_weight > 0.0 and "link_active_rate_aux" in outputs:
        auxiliary_error = (outputs["link_active_rate_aux"] - target["link_rate"]) ** 2
        if active_mask.any():
            active_rate_auxiliary_loss = auxiliary_error[active_mask].mean()
    task_loss = mse(outputs["task"], target["task"])
    total = (
        0.5 * node_loss
        + activity_loss
        + 0.3
        * (
            rate_loss
            + rate_teacher_weight * teacher_loss
            + active_rate_auxiliary_weight * active_rate_auxiliary_loss
        )
        + 0.8 * task_loss
    )
    parts = {
        "total": float(total.detach().cpu()),
        "node": float(node_loss.detach().cpu()),
        "activity": float(activity_loss.detach().cpu()),
        "rate": float(rate_loss.detach().cpu()),
        "rate_teacher": float(teacher_loss.detach().cpu()),
        "active_rate_auxiliary": float(active_rate_auxiliary_loss.detach().cpu()),
        "active_rate_loss": float(active_rate_loss.detach().cpu()),
        "inactive_rate_loss": float(inactive_rate_loss.detach().cpu()),
        "task": float(task_loss.detach().cpu()),
    }
    return total, parts


def select_link_rate_output(
    outputs: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor] | None = None,
    rate_output_mode: str = "main",
    inactive_rate_value: float = 0.0,
    activity_threshold: float | None = None,
) -> torch.Tensor:
    """Select the final normalized link-rate tensor from main or active-rate auxiliary heads."""
    if rate_output_mode == "main":
        return outputs["link_rate"]
    if "link_active_rate_aux" not in outputs:
        raise ValueError(f"{rate_output_mode} requires --active-rate-auxiliary")
    inactive = torch.full_like(outputs["link_active_rate_aux"], float(inactive_rate_value))
    if rate_output_mode == "aux_soft_zero":
        activity_prob = torch.sigmoid(outputs["link_activity_logit"])
        return activity_prob * outputs["link_active_rate_aux"] + (1.0 - activity_prob) * inactive
    if rate_output_mode == "aux_oracle_zero":
        if target is None:
            raise ValueError("aux_oracle_zero requires target link_activity")
        return torch.where(target["link_activity"] > 0.5, outputs["link_active_rate_aux"], inactive)
    if rate_output_mode == "aux_hard_zero":
        if activity_threshold is None:
            raise ValueError("aux_hard_zero requires an activity threshold")
        activity_mask = torch.sigmoid(outputs["link_activity_logit"]) >= activity_threshold
        return torch.where(activity_mask, outputs["link_active_rate_aux"], inactive)
    raise ValueError(
        "rate_output_mode must be one of: main, aux_soft_zero, aux_oracle_zero, aux_hard_zero"
    )


def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    rate_loss_mode: str = "weighted_all",
    inactive_rate_weight: float = 0.05,
    rate_teacher_weight: float = 0.0,
    active_rate_auxiliary_weight: float = 0.0,
    rate_output_mode: str = "main",
    inactive_rate_value: float = 0.0,
) -> dict[str, float]:
    model.eval()
    totals = []
    with torch.no_grad():
        for batch, target in loader:
            batch = move_batch_to_device(batch, device)
            target = move_target_to_device(target, device)
            outputs = model(batch)
            _, parts = compute_loss(
                outputs,
                target,
                rate_loss_mode=rate_loss_mode,
                inactive_rate_weight=inactive_rate_weight,
                rate_teacher_weight=rate_teacher_weight,
                active_rate_auxiliary_weight=active_rate_auxiliary_weight,
                rate_output_mode=rate_output_mode,
                inactive_rate_value=inactive_rate_value,
            )
            totals.append(parts)
    return _mean_metrics(totals)


def run_real_data_sanity(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    arrays = load_world_model_arrays(args.dataset_dir)
    train_idx, val_idx, test_idx = split_by_seed(arrays["sample_seed"])
    teacher_summary = None
    if args.rate_teacher_mode == "random_forest_active":
        from pi_jwm.v7_active_rate_teacher import add_random_forest_active_rate_teacher

        teacher_result = add_random_forest_active_rate_teacher(
            arrays,
            train_idx,
            seed=args.seed,
            n_estimators=args.rate_teacher_n_estimators,
            min_samples_leaf=args.rate_teacher_min_samples_leaf,
            max_features=args.rate_teacher_max_features,
        )
        arrays = teacher_result.arrays
        teacher_summary = teacher_result.summary
    elif args.rate_teacher_mode == "ridge_active":
        from pi_jwm.v7_active_rate_teacher import add_ridge_active_rate_teacher

        teacher_result = add_ridge_active_rate_teacher(
            arrays,
            train_idx,
            ridge_lambda=args.rate_teacher_ridge_lambda,
        )
        arrays = teacher_result.arrays
        teacher_summary = teacher_result.summary
    stats = make_normalization_stats(arrays, train_idx, rate_target_transform=args.rate_target_transform)
    stats["rate_inverse_clip_max"] = compute_rate_inverse_clip_max(
        arrays,
        train_idx,
        args.rate_target_transform,
        args.rate_inverse_clip_quantile,
    )
    if args.inactive_rate_value is None:
        args.inactive_rate_value = compute_normalized_inactive_rate_value(stats)

    train_ds = V6WorldModelDataset(arrays, train_idx, stats, rate_target_transform=args.rate_target_transform)
    val_ds = V6WorldModelDataset(arrays, val_idx, stats, rate_target_transform=args.rate_target_transform)
    test_ds = V6WorldModelDataset(arrays, test_idx, stats, rate_target_transform=args.rate_target_transform)
    train_subset = Subset(train_ds, range(min(args.max_train_samples, len(train_ds))))
    val_subset = Subset(val_ds, range(min(args.max_val_samples, len(val_ds))))
    test_subset = Subset(test_ds, range(min(args.max_test_samples, len(test_ds))))

    modes = ["dual", "physical_only", "information_only"] if args.graph_mode == "all" else [args.graph_mode]
    runs = {}
    for mode in modes:
        runs[mode] = train_one_mode(args, arrays, stats, train_subset, val_subset, test_subset, mode, device)

    return {
        "status": "sanity",
        "framework": "PI-JWM",
        "note": "Sanity run on real historical dataset; use full samples and enough epochs before interpreting results.",
        "dataset_dir": str(args.dataset_dir),
        "split_sizes": {
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
            "train_used": int(len(train_subset)),
            "val_used": int(len(val_subset)),
            "test_used": int(len(test_subset)),
        },
        "device": str(device),
        "rate_teacher": teacher_summary,
        "runs": runs,
    }


def compute_rate_inverse_clip_max(
    arrays: dict,
    train_idx: np.ndarray,
    rate_target_transform: str,
    rate_inverse_clip_quantile: float,
) -> float | None:
    if rate_inverse_clip_quantile <= 0.0:
        return None
    if not 0.0 < rate_inverse_clip_quantile <= 1.0:
        raise ValueError("rate_inverse_clip_quantile must be 0 or in (0, 1].")
    from pi_jwm.v6_data import transform_link_rate

    raw_train_rate = arrays["y_link_rate"][train_idx]
    positive_rate = raw_train_rate[raw_train_rate > 0.0]
    if positive_rate.size == 0:
        return None
    train_rate = transform_link_rate(positive_rate, rate_target_transform)
    return float(np.quantile(train_rate, rate_inverse_clip_quantile))


def compute_normalized_inactive_rate_value(stats: dict) -> float:
    transform = stats.get("rate_target_transform", "raw")
    if transform == "residual_last_rate":
        raise ValueError("active-selected rate outputs are not supported with residual_last_rate targets")
    from pi_jwm.v6_data import transform_link_rate

    transformed_zero = transform_link_rate(np.array([0.0], dtype=np.float32), transform)
    mean_arr, std_arr = stats["y_link_rate"]
    mean = float(np.asarray(mean_arr).reshape(-1)[0])
    std = float(np.asarray(std_arr).reshape(-1)[0])
    return float((transformed_zero[0] - mean) / std)


def train_one_mode(
    args: argparse.Namespace,
    arrays: dict,
    stats: dict,
    train_subset: Subset,
    val_subset: Subset,
    test_subset: Subset,
    graph_mode: str,
    device: torch.device,
) -> dict:
    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collate_v6_world_model_batch,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_v6_world_model_batch,
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_v6_world_model_batch,
    )

    config = make_config_from_arrays(
        arrays,
        args.hidden_dim,
        graph_mode,
        fusion_mode=args.fusion_mode,
        fusion_num_heads=args.fusion_num_heads,
        rate_head_mode=args.rate_head_mode,
        rate_gate_temperature=args.rate_gate_temperature,
        rate_gate_floor=args.rate_gate_floor,
        active_rate_auxiliary=args.active_rate_auxiliary,
    )
    model = V6DualGraphRollout(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    history = []
    best_state = None
    best_epoch = 0
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_parts = []
        for batch, target in train_loader:
            batch = move_batch_to_device(batch, device)
            target = move_target_to_device(target, device)
            outputs = model(batch)
            loss, parts = compute_loss(
                outputs,
                target,
                rate_loss_mode=args.rate_loss_mode,
                inactive_rate_weight=args.inactive_rate_weight,
                rate_teacher_weight=args.rate_teacher_weight,
                active_rate_auxiliary_weight=args.active_rate_auxiliary_weight,
                rate_output_mode=args.rate_output_mode,
                inactive_rate_value=args.inactive_rate_value,
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_parts.append(parts)
        train_metrics = _mean_metrics(epoch_parts)
        val_metrics = evaluate_loss(
            model,
            val_loader,
            device,
            rate_loss_mode=args.rate_loss_mode,
            inactive_rate_weight=args.inactive_rate_weight,
            rate_teacher_weight=args.rate_teacher_weight,
            active_rate_auxiliary_weight=args.active_rate_auxiliary_weight,
            rate_output_mode=args.rate_output_mode,
            inactive_rate_value=args.inactive_rate_value,
        )
        selection_score, selection_metrics = compute_selection_score(
            model,
            val_loader,
            device,
            stats,
            args.best_metric,
            val_metrics,
            rate_output_mode=args.rate_output_mode,
            inactive_rate_value=args.inactive_rate_value,
        )
        history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
                "selection": selection_metrics,
            }
        )
        if selection_score < best_val:
            best_val = selection_score
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
        print(
            f"[v6-{device.type}:{graph_mode}] epoch={epoch} "
            f"train_total={train_metrics['total']:.6f} val_total={val_metrics['total']:.6f} "
            f"best_metric={args.best_metric} score={selection_score:.6f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    checkpoint_dir = args.output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"v6_{graph_mode}_{args.fusion_mode}_best.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config.__dict__,
            "best_epoch": best_epoch,
            "best_metric": args.best_metric,
            "best_selection_score": best_val,
        },
        checkpoint_path,
    )

    threshold_search_mode = "main" if args.rate_output_mode == "aux_hard_zero" else args.rate_output_mode
    val_predictions = collect_predictions(
        model,
        val_loader,
        device,
        stats,
        rate_output_mode=threshold_search_mode,
        inactive_rate_value=args.inactive_rate_value,
    )
    threshold = choose_activity_threshold(
        val_predictions["link_activity_prob"],
        val_predictions["link_activity_true"],
    )
    if args.rate_output_mode == "aux_hard_zero":
        val_predictions = collect_predictions(
            model,
            val_loader,
            device,
            stats,
            rate_output_mode=args.rate_output_mode,
            inactive_rate_value=args.inactive_rate_value,
            activity_threshold=threshold,
        )
    val_eval = evaluate_predictions(val_predictions, threshold)
    test_eval = evaluate_predictions(
        collect_predictions(
            model,
            test_loader,
            device,
            stats,
            rate_output_mode=args.rate_output_mode,
            inactive_rate_value=args.inactive_rate_value,
            activity_threshold=threshold if args.rate_output_mode == "aux_hard_zero" else None,
        ),
        threshold,
    )
    return {
        "config": config.__dict__,
        "history": history,
        "best_epoch": int(best_epoch),
        "best_metric": args.best_metric,
        "best_selection_score": float(best_val),
        "activity_threshold": float(threshold),
        "checkpoint_path": str(checkpoint_path),
        "rate_head_mode": args.rate_head_mode,
        "rate_target_transform": args.rate_target_transform,
        "rate_inverse_clip_quantile": float(args.rate_inverse_clip_quantile),
        "rate_inverse_clip_max": (
            None if stats.get("rate_inverse_clip_max") is None else float(stats["rate_inverse_clip_max"])
        ),
        "rate_gate_temperature": float(args.rate_gate_temperature),
        "rate_gate_floor": float(args.rate_gate_floor),
        "rate_loss_mode": args.rate_loss_mode,
        "inactive_rate_weight": float(args.inactive_rate_weight),
        "rate_teacher_mode": args.rate_teacher_mode,
        "rate_teacher_weight": float(args.rate_teacher_weight),
        "active_rate_auxiliary": bool(args.active_rate_auxiliary),
        "active_rate_auxiliary_weight": float(args.active_rate_auxiliary_weight),
        "rate_output_mode": args.rate_output_mode,
        "inactive_rate_value": float(args.inactive_rate_value),
        "val_eval": val_eval,
        "test_eval": test_eval,
    }


def compute_selection_score(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    stats: dict,
    best_metric: str,
    val_loss_metrics: dict[str, float],
    rate_output_mode: str = "main",
    inactive_rate_value: float = 0.0,
) -> tuple[float, dict[str, float]]:
    if best_metric == "total":
        return float(val_loss_metrics["total"]), {"score": float(val_loss_metrics["total"])}

    threshold_search_mode = "main" if rate_output_mode == "aux_hard_zero" else rate_output_mode
    predictions = collect_predictions(
        model,
        val_loader,
        device,
        stats,
        rate_output_mode=threshold_search_mode,
        inactive_rate_value=inactive_rate_value,
    )
    threshold = choose_activity_threshold(predictions["link_activity_prob"], predictions["link_activity_true"])
    if rate_output_mode == "aux_hard_zero":
        predictions = collect_predictions(
            model,
            val_loader,
            device,
            stats,
            rate_output_mode=rate_output_mode,
            inactive_rate_value=inactive_rate_value,
            activity_threshold=threshold,
        )
    val_eval = evaluate_predictions(predictions, threshold)
    metrics = {
        "val_active_rate_rmse": float(val_eval["active_rate"]["active_rmse"]),
        "val_link_rate_rmse": float(val_eval["link_rate"]["rmse"]),
        "val_task_rmse": float(val_eval["task"]["rmse"]),
        "val_activity_f1": float(val_eval["activity"]["f1"]),
        "threshold": float(threshold),
    }
    if best_metric == "val_activity_f1":
        score = -metrics[best_metric]
    else:
        score = metrics[best_metric]
    metrics["score"] = float(score)
    return float(score), metrics


def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    stats: dict,
    rate_output_mode: str = "main",
    inactive_rate_value: float = 0.0,
    activity_threshold: float | None = None,
) -> dict[str, np.ndarray]:
    model.eval()
    rows = {
        "node_pred": [],
        "node_true": [],
        "link_activity_prob": [],
        "link_activity_true": [],
        "link_rate_pred": [],
        "link_active_rate_aux_pred": [],
        "link_rate_true": [],
        "task_pred": [],
        "task_true": [],
    }
    with torch.no_grad():
        for batch, target in loader:
            batch = move_batch_to_device(batch, device)
            outputs = model(batch)
            target_device = move_target_to_device(target, device)
            selected_link_rate = select_link_rate_output(
                outputs,
                target=target_device,
                rate_output_mode=rate_output_mode,
                inactive_rate_value=inactive_rate_value,
                activity_threshold=activity_threshold,
            )
            rows["node_pred"].append(inverse_normalize(outputs["node"].cpu().numpy(), stats["y_node"]))
            rows["node_true"].append(inverse_normalize(target["node"].numpy(), stats["y_node"]))
            rows["link_activity_prob"].append(torch.sigmoid(outputs["link_activity_logit"]).cpu().numpy())
            rows["link_activity_true"].append(target["link_activity"].numpy())
            rows["link_rate_pred"].append(
                denormalize_link_rate_prediction(
                    outputs["link_rate"].cpu().numpy(),
                    stats,
                    baseline=(
                        batch.link_rate_baseline.cpu().numpy()
                        if batch.link_rate_baseline is not None
                        else None
                    ),
                )
            )
            if rate_output_mode != "main":
                rows["link_rate_pred"][-1] = denormalize_link_rate_prediction(
                    selected_link_rate.cpu().numpy(),
                    stats,
                    baseline=(
                        batch.link_rate_baseline.cpu().numpy()
                        if batch.link_rate_baseline is not None
                        else None
                    ),
                )
            if "link_active_rate_aux" in outputs:
                rows["link_active_rate_aux_pred"].append(
                    denormalize_link_rate_prediction(
                        outputs["link_active_rate_aux"].cpu().numpy(),
                        stats,
                        baseline=(
                            batch.link_rate_baseline.cpu().numpy()
                            if batch.link_rate_baseline is not None
                            else None
                        ),
                    )
                )
            rows["link_rate_true"].append(
                denormalize_link_rate_prediction(
                    target["link_rate"].numpy(),
                    stats,
                    baseline=(
                        batch.link_rate_baseline.cpu().numpy()
                        if batch.link_rate_baseline is not None
                        else None
                    ),
                    clip_prediction=False,
                )
            )
            rows["task_pred"].append(inverse_normalize(outputs["task"].cpu().numpy(), stats["y_task"]))
            rows["task_true"].append(inverse_normalize(target["task"].numpy(), stats["y_task"]))
    return {name: np.concatenate(values, axis=0) for name, values in rows.items() if values}


def denormalize_link_rate_prediction(
    values: np.ndarray,
    stats: dict,
    baseline: np.ndarray | None = None,
    clip_prediction: bool = True,
) -> np.ndarray:
    rate_space = inverse_normalize(values, stats["y_link_rate"])
    transform = stats.get("rate_target_transform", "raw")
    if transform == "residual_last_rate":
        if baseline is None:
            raise ValueError("baseline is required for residual_last_rate predictions")
        return (rate_space + baseline).astype(np.float32)
    return inverse_transform_link_rate(
        rate_space,
        transform,
        clip_max=stats.get("rate_inverse_clip_max") if clip_prediction else None,
    )


def choose_activity_threshold(prob: np.ndarray, true: np.ndarray) -> float:
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(0.01, 0.99, 99):
        score = activity_metrics(prob, true, threshold=float(threshold))["f1"]
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)
    return best_threshold


def evaluate_predictions(predictions: dict[str, np.ndarray], activity_threshold: float) -> dict[str, dict[str, float]]:
    metrics = {
        "node": regression_metrics(predictions["node_pred"], predictions["node_true"]),
        "task": regression_metrics(predictions["task_pred"], predictions["task_true"]),
        "link_rate": regression_metrics(predictions["link_rate_pred"], predictions["link_rate_true"]),
        "active_rate": active_rate_metrics(
            predictions["link_rate_pred"],
            predictions["link_rate_true"],
            predictions["link_activity_true"],
        ),
        "activity": activity_metrics(
            predictions["link_activity_prob"],
            predictions["link_activity_true"],
            threshold=activity_threshold,
        ),
    }
    if "link_active_rate_aux_pred" in predictions:
        metrics["active_rate_auxiliary"] = active_rate_metrics(
            predictions["link_active_rate_aux_pred"],
            predictions["link_rate_true"],
            predictions["link_activity_true"],
        )
    return metrics


def _mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    return {key: float(sum(row[key] for row in rows) / len(rows)) for key in rows[0]}


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    synthetic = run_synthetic_smoke(
        args.seed,
        args.fusion_mode,
        args.fusion_num_heads,
        args.rate_head_mode,
        args.rate_gate_temperature,
        args.rate_gate_floor,
    )
    real_data = None if args.synthetic_only else run_real_data_sanity(args)
    summary = {
        "synthetic": synthetic,
        "real_data_sanity": real_data,
        "next_step": "Run full training and ablations on GPU after CPU sanity checks are accepted.",
    }
    summary_path = args.output_dir / "v6_dual_graph_smoke_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report_path = args.output_dir / "v6_dual_graph_sanity_report.md"
    report_path.write_text(render_report(summary), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"summary_path={summary_path}")
    print(f"report_path={report_path}")


def render_report(summary: dict) -> str:
    real_data = summary.get("real_data_sanity")
    lines = [
        "# PI-JWM v6 双图联合 Rollout Sanity",
        "",
        "本报告记录 v6 双图模型的接口与训练检查。短 epoch 结果只说明数据流、模型前向和反向传播可以跑通；全量多 epoch 结果才可用于初步消融判断。",
        "",
        "## 合成数据接口检查",
        "",
        f"- 状态：{summary['synthetic']['status']}",
        f"- 输出形状：`{summary['synthetic']['output_shapes']}`",
        "",
    ]
    if real_data is None:
        lines.extend(["## 真实数据检查", "", "- 未运行。"])
        return "\n".join(lines) + "\n"

    lines.extend(
        [
            "## 真实数据训练检查",
            "",
            f"- 数据目录：`{real_data['dataset_dir']}`",
            f"- 设备：`{real_data['device']}`",
            f"- 切分规模：`{real_data['split_sizes']}`",
            "",
            "| 模式 | best epoch | best metric | best score | threshold | val activity F1 | test activity F1 | val active-rate RMSE | test active-rate RMSE | test node RMSE | test task RMSE |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for mode, run in real_data["runs"].items():
        val_eval = run["val_eval"]
        test_eval = run["test_eval"]
        best_metric = run.get("best_metric", "total")
        best_score = run.get("best_selection_score", run.get("best_val_total", float("nan")))
        lines.append(
            f"| {mode} | {run['best_epoch']} | {best_metric} | {best_score:.6f} | {run['activity_threshold']:.2f} | "
            f"{val_eval['activity']['f1']:.6f} | {test_eval['activity']['f1']:.6f} | "
            f"{val_eval['active_rate']['active_rmse']:.6f} | {test_eval['active_rate']['active_rmse']:.6f} | "
            f"{test_eval['node']['rmse']:.6f} | {test_eval['task']['rmse']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## GPU 临界点",
            "",
            "接口、真实数据适配、三模式消融 sanity 已完成。下一步需要补 test seed 评估、曲线图和更稳定的多 seed 统计。",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
