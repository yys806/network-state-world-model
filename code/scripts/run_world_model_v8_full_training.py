"""Local training entry point for PI-JWM v8 full-world-model experiments."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pi_jwm.paths import ARTIFACTS_DIR
from pi_jwm.v6_data import (
    V6WorldModelDataset,
    collate_v6_world_model_batch,
    load_world_model_arrays,
    make_normalization_stats,
    split_by_seed,
)
from pi_jwm.v8_training import (
    build_v8_model_from_arrays,
    evaluate_v8_model,
    fit_lds_rate_reweighting,
    train_v8_one_epoch,
)


DEFAULT_DATASET_DIR = (
    ARTIFACTS_DIR
    / "experiments"
    / "airfogsim_v0"
    / "datasets"
    / "world_model_dataset_seed0_9_v0"
)
DEFAULT_OUTPUT_DIR = ARTIFACTS_DIR / "experiments" / "pi_jwm_v8_full_training_smoke"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PI-JWM v8 local training smoke.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-train-samples", type=int, default=64)
    parser.add_argument("--max-val-samples", type=int, default=32)
    parser.add_argument("--max-test-samples", type=int, default=32)
    parser.add_argument("--train-seeds", default=None, help="Comma/space separated train seed ids; default uses seeds 0-7.")
    parser.add_argument("--val-seeds", default=None, help="Comma/space separated validation seed ids; default uses seed 8.")
    parser.add_argument("--test-seeds", default=None, help="Comma/space separated test seed ids; default uses seed 9.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260614)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="cpu")
    parser.add_argument(
        "--graph-mode",
        choices=("dual", "physical_only", "information_only"),
        default="dual",
    )
    parser.add_argument(
        "--fusion-mode",
        choices=("gated", "cross_attention"),
        default="gated",
    )
    parser.add_argument("--fusion-num-heads", type=int, default=4)
    parser.add_argument(
        "--history-encoder",
        choices=("mean", "temporal_conv", "stgcn_light", "stgcn_full"),
        default="mean",
    )
    parser.add_argument(
        "--latent-transition-mode",
        choices=("message_passing", "recurrent"),
        default="message_passing",
    )
    parser.add_argument(
        "--adaptive-edge-context",
        choices=("none", "sparse_attention"),
        default="none",
    )
    parser.add_argument("--adaptive-edge-topk", type=int, default=8)
    parser.add_argument(
        "--rate-loss-mode",
        choices=("weighted_all", "active_only", "active_mixed"),
        default="active_mixed",
    )
    parser.add_argument("--inactive-rate-weight", type=float, default=0.05)
    parser.add_argument(
        "--future-action-mode",
        choices=("full", "first_step_only", "none"),
        default="full",
        help="full is planner-conditioned; first_step_only hides closed-loop actions not known at rollout start.",
    )
    parser.add_argument("--active-rate-auxiliary", action="store_true")
    parser.add_argument("--active-rate-auxiliary-weight", type=float, default=0.0)
    parser.add_argument(
        "--active-rate-head-mode",
        choices=("mlp", "moe"),
        default="mlp",
    )
    parser.add_argument("--num-rate-experts", type=int, default=4)
    parser.add_argument(
        "--rate-output-mode",
        choices=("main", "hurdle_gate", "dual_soft_blend", "active_mass_alloc", "aux_soft_zero", "aux_oracle_zero"),
        default="main",
    )
    parser.add_argument(
        "--model-rate-output-mode",
        choices=("direct", "hurdle_soft", "hurdle_dual", "hurdle_mass"),
        default="direct",
    )
    parser.add_argument(
        "--hurdle-train-gate-mode",
        choices=("none", "predicted", "detach", "teacher_forcing"),
        default="predicted",
    )
    parser.add_argument("--hurdle-train-gate-power", type=float, default=1.0)
    parser.add_argument("--use-event-memory-features", action="store_true")
    parser.add_argument(
        "--event-memory-routing",
        choices=("shared", "activity_only"),
        default="shared",
        help="shared appends event-memory features to all link branches; activity_only routes them only to the activity head.",
    )
    parser.add_argument("--inactive-rate-value", type=float, default=0.0)
    parser.add_argument(
        "--best-metric",
        choices=(
            "val_active_rate_rmse",
            "val_link_rate_rmse",
            "val_activity_f1",
            "val_node_rmse",
            "val_task_rmse",
            "val_composite",
            "val_precision_constrained_active_rate",
            "val_precision_constrained_composite",
            "val_link_f1_constrained_active_rate",
            "val_link_f1_constrained_composite",
        ),
        default="val_active_rate_rmse",
    )
    parser.add_argument("--best-min-precision", type=float, default=0.0)
    parser.add_argument("--best-min-recall", type=float, default=0.0)
    parser.add_argument("--best-precision-penalty-weight", type=float, default=10000.0)
    parser.add_argument("--best-recall-penalty-weight", type=float, default=1000.0)
    parser.add_argument("--best-min-f1", type=float, default=0.0)
    parser.add_argument("--best-max-link-rmse", type=float, default=0.0)
    parser.add_argument("--best-f1-penalty-weight", type=float, default=1000.0)
    parser.add_argument("--best-link-penalty-weight", type=float, default=10.0)
    parser.add_argument(
        "--metric-checkpoints",
        default="",
        help="Comma/space separated validation metric names to save extra best checkpoints for.",
    )
    parser.add_argument("--node-loss-weight", type=float, default=0.5)
    parser.add_argument("--activity-loss-weight", type=float, default=1.0)
    parser.add_argument("--rate-loss-weight", type=float, default=0.3)
    parser.add_argument("--task-loss-weight", type=float, default=0.8)
    parser.add_argument("--activity-loss-mode", choices=("bce", "focal"), default="bce")
    parser.add_argument("--activity-pos-weight", type=float, default=80.0)
    parser.add_argument("--activity-focal-gamma", type=float, default=2.0)
    parser.add_argument("--inactive-loss-sample-ratio", type=float, default=1.0)
    parser.add_argument("--false-positive-penalty-weight", type=float, default=0.0)
    parser.add_argument("--dynamic-hard-negative-weight", type=float, default=0.0)
    parser.add_argument("--dynamic-hard-negative-ratio", type=float, default=0.1)
    parser.add_argument("--eval-hurdle-gate-temperature", type=float, default=1.0)
    parser.add_argument("--eval-hurdle-gate-power", type=float, default=1.0)
    parser.add_argument("--positive-rate-specialist-weight", type=float, default=0.0)
    parser.add_argument(
        "--positive-rate-target-mode",
        choices=("raw", "log1p", "normalized", "log1p_normalized"),
        default="normalized",
    )
    parser.add_argument("--positive-rate-loss-mode", choices=("mse", "huber", "tweedie"), default="mse")
    parser.add_argument("--positive-rate-tweedie-power", type=float, default=1.5)
    parser.add_argument("--high-rate-weight", type=float, default=1.0)
    parser.add_argument("--high-rate-threshold", type=float, default=0.0)
    parser.add_argument("--active-rate-reweight-mode", choices=("none", "lds", "bmc"), default="none")
    parser.add_argument("--lds-bin-width", type=float, default=50.0)
    parser.add_argument("--lds-kernel-size", type=int, default=5)
    parser.add_argument("--lds-sigma", type=float, default=2.0)
    parser.add_argument("--lds-weight-min", type=float, default=0.5)
    parser.add_argument("--lds-weight-max", type=float, default=3.0)
    parser.add_argument("--lds-tail-quantile", type=float, default=0.995)
    parser.add_argument("--bmc-noise-sigma", type=float, default=1.0)
    parser.add_argument("--bmc-minimum-count", type=int, default=3)
    parser.add_argument("--active-mass-loss-weight", type=float, default=0.0)
    parser.add_argument("--active-mass-target-mode", choices=("normalized", "raw"), default="normalized")
    parser.add_argument(
        "--candidate-pruning-mode",
        choices=("none", "train_active_plus_hard_negatives"),
        default="none",
    )
    parser.add_argument("--candidate-hard-negative-count", type=int, default=0)
    parser.add_argument(
        "--candidate-pruning-scope",
        choices=("all_losses", "rate_only"),
        default="all_losses",
    )
    return parser.parse_args()


def parse_seed_list(value: str | None) -> list[int] | None:
    if value is None:
        return None
    parts = [part for part in value.replace(",", " ").split() if part]
    if not parts:
        return None
    return [int(part) for part in parts]


def parse_metric_list(value: str | None) -> list[str]:
    if value is None:
        return []
    return [part for part in value.replace(",", " ").split() if part]


def build_active_rate_lds_config(
    arrays: dict[str, np.ndarray],
    train_idx: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, object] | None:
    if getattr(args, "active_rate_reweight_mode", "none") != "lds":
        return None
    train_rates = arrays["y_link_rate"][train_idx]
    train_active = arrays["y_link_active"][train_idx] > 0.5
    return fit_lds_rate_reweighting(
        train_rates[train_active],
        bin_width=getattr(args, "lds_bin_width", 50.0),
        kernel_size=getattr(args, "lds_kernel_size", 5),
        sigma=getattr(args, "lds_sigma", 2.0),
        weight_min=getattr(args, "lds_weight_min", 0.5),
        weight_max=getattr(args, "lds_weight_max", 3.0),
        tail_quantile=getattr(args, "lds_tail_quantile", 0.995),
    )


def resolve_seed_splits(
    sample_seed: np.ndarray,
    train_seeds: list[int] | None = None,
    val_seeds: list[int] | None = None,
    test_seeds: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, list[int]]]:
    if train_seeds is None and val_seeds is None and test_seeds is None:
        unique_seeds = sorted(int(seed) for seed in np.unique(sample_seed))
        legacy_seeds = set(range(10))
        if not unique_seeds or not set(unique_seeds).issubset(legacy_seeds):
            raise ValueError(
                "Explicit seed splits are required unless all dataset seeds use the legacy 0-9 protocol; "
                f"found {len(unique_seeds)} seeds ranging from {unique_seeds[0] if unique_seeds else 'none'} "
                f"to {unique_seeds[-1] if unique_seeds else 'none'}."
            )
        train_idx, val_idx, test_idx = split_by_seed(sample_seed)
        if any(len(idx) == 0 for idx in (train_idx, val_idx, test_idx)):
            raise ValueError(
                "Explicit seed splits are required because the legacy 0-9 protocol produced an empty split."
            )
        spec = {"train_seeds": list(range(0, 8)), "val_seeds": [8], "test_seeds": [9]}
        return train_idx, val_idx, test_idx, spec
    train_seeds = list(range(0, 8)) if train_seeds is None else train_seeds
    val_seeds = [8] if val_seeds is None else val_seeds
    test_seeds = [9] if test_seeds is None else test_seeds
    seed_sets = {
        "train": set(int(seed) for seed in train_seeds),
        "val": set(int(seed) for seed in val_seeds),
        "test": set(int(seed) for seed in test_seeds),
    }
    if (
        seed_sets["train"] & seed_sets["val"]
        or seed_sets["train"] & seed_sets["test"]
        or seed_sets["val"] & seed_sets["test"]
    ):
        raise ValueError("train, val, and test seed sets must be disjoint")
    sample_seed = np.asarray(sample_seed)
    train_idx = np.where(np.isin(sample_seed, train_seeds))[0]
    val_idx = np.where(np.isin(sample_seed, val_seeds))[0]
    test_idx = np.where(np.isin(sample_seed, test_seeds))[0]
    for name, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        if len(idx) == 0:
            raise ValueError(f"Custom seed split produced an empty {name} split.")
    spec = {"train_seeds": [int(seed) for seed in train_seeds], "val_seeds": [int(seed) for seed in val_seeds], "test_seeds": [int(seed) for seed in test_seeds]}
    return train_idx, val_idx, test_idx, spec

def choose_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_candidate_loss_mask(
    arrays: dict[str, np.ndarray],
    train_idx: np.ndarray,
    mode: str = "none",
    hard_negative_count: int = 0,
) -> tuple[np.ndarray | None, dict[str, object]]:
    total_edges = int(arrays["y_link_active"].shape[2])
    if mode == "none":
        return None, {
            "mode": "none",
            "edge_count": total_edges,
            "total_edges": total_edges,
            "active_edge_count": total_edges,
            "hard_negative_count": 0,
            "edge_indices": list(range(total_edges)),
        }
    if mode != "train_active_plus_hard_negatives":
        raise ValueError("candidate_pruning_mode must be one of: none, train_active_plus_hard_negatives")
    if hard_negative_count < 0:
        raise ValueError("candidate_hard_negative_count must be non-negative")

    active_edges = arrays["y_link_active"][train_idx].sum(axis=(0, 1)) > 0.5
    hard_negative_edges = choose_hard_negative_edges(arrays, train_idx, active_edges, hard_negative_count)
    mask = active_edges.copy()
    mask[hard_negative_edges] = True
    if not mask.any():
        mask[:] = True
    edge_indices = np.where(mask)[0].astype(int).tolist()
    return mask.astype(bool), {
        "mode": mode,
        "edge_count": int(mask.sum()),
        "total_edges": total_edges,
        "active_edge_count": int(active_edges.sum()),
        "hard_negative_count": int(len(hard_negative_edges)),
        "edge_indices": edge_indices,
    }


def choose_hard_negative_edges(
    arrays: dict[str, np.ndarray],
    train_idx: np.ndarray,
    active_edges: np.ndarray,
    hard_negative_count: int,
) -> np.ndarray:
    if hard_negative_count <= 0:
        return np.array([], dtype=np.int64)
    inactive_indices = np.where(~active_edges)[0]
    if len(inactive_indices) == 0:
        return np.array([], dtype=np.int64)
    link_names = [str(name) for name in arrays.get("link_features", [])]
    feature_mean = arrays["x_link"][train_idx].mean(axis=(0, 1))

    def feature(name: str, default: float = 0.0) -> np.ndarray:
        if name not in link_names:
            return np.full(feature_mean.shape[0], default, dtype=np.float64)
        return feature_mean[:, link_names.index(name)].astype(np.float64)

    score = (
        descending_unit_score(-feature("distance"))
        + descending_unit_score(feature("csi_mean"))
        + descending_unit_score(feature("active_task_count"))
        + descending_unit_score(feature("allocated_rb_count"))
        + 0.5 * descending_unit_score(feature("rate_sum"))
    )
    order = sorted(inactive_indices.tolist(), key=lambda edge: (-float(score[edge]), int(edge)))
    return np.asarray(order[: min(hard_negative_count, len(order))], dtype=np.int64)


def descending_unit_score(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros_like(values, dtype=np.float64)
    safe = np.where(finite, values, np.nanmin(values[finite]))
    span = float(np.max(safe) - np.min(safe))
    if span < 1e-12:
        return np.zeros_like(safe, dtype=np.float64)
    return (safe - np.min(safe)) / span


def add_event_memory_features(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    arrays = dict(arrays)
    x_link = np.asarray(arrays["x_link"], dtype=np.float32)
    link_names = [str(name) for name in arrays.get("link_features", [])]
    if not link_names:
        link_names = [f"link_feature_{idx}" for idx in range(x_link.shape[-1])]
        if x_link.shape[-1] > 1:
            link_names[1] = "rate_sum"
    rate_idx = link_names.index("rate_sum") if "rate_sum" in link_names else 1
    active_task_idx = link_names.index("active_task_count") if "active_task_count" in link_names else None
    rate_history = x_link[..., rate_idx]
    if active_task_idx is not None:
        active_history = (x_link[..., active_task_idx] > 0.0) | (rate_history > 0.0)
    else:
        active_history = rate_history > 0.0
    active_frequency = active_history.mean(axis=1).astype(np.float32)
    reversed_active = active_history[:, ::-1, :]
    has_active = active_history.any(axis=1)
    last_active_gap = np.argmax(reversed_active, axis=1).astype(np.float32)
    last_active_gap = np.where(has_active, last_active_gap, active_history.shape[1]).astype(np.float32)
    last_nonzero_rate = np.zeros_like(active_frequency, dtype=np.float32)
    for step in range(rate_history.shape[1]):
        last_nonzero_rate = np.where(rate_history[:, step, :] > 0.0, rate_history[:, step, :], last_nonzero_rate)
    extra = np.stack([active_frequency, last_active_gap, last_nonzero_rate], axis=-1)
    extra_sequence = np.repeat(extra[:, None, :, :], x_link.shape[1], axis=1)
    arrays["x_link"] = np.concatenate([x_link, extra_sequence.astype(np.float32)], axis=-1)
    arrays["link_features"] = np.asarray(link_names + ["active_frequency", "last_active_gap", "last_nonzero_rate"])
    return arrays


def validate_training_arrays(arrays: dict[str, np.ndarray], future_action_mode: str = "full") -> dict:
    if future_action_mode not in {"full", "first_step_only", "none"}:
        raise ValueError("future_action_mode must be one of: full, first_step_only, none")
    required = (
        "x_node",
        "x_link",
        "x_task",
        "edge_a_hist",
        "edge_a_future",
        "y_node",
        "y_link_rate",
        "y_link_active",
        "y_task",
        "sample_seed",
    )
    missing = [key for key in required if key not in arrays]
    if missing:
        raise ValueError(f"Training dataset is missing arrays: {', '.join(missing)}")
    num_samples = len(arrays["sample_seed"])
    for key in required:
        value = np.asarray(arrays[key])
        if value.shape[0] != num_samples:
            raise ValueError(f"{key} sample dimension does not match sample_seed.")
        if not np.isfinite(value).all():
            raise ValueError(f"{key} contains non-finite values.")

    expected_ndim = {
        "x_node": 4,
        "x_link": 4,
        "x_task": 3,
        "edge_a_hist": 4,
        "edge_a_future": 4,
        "y_node": 4,
        "y_link_rate": 3,
        "y_link_active": 3,
        "y_task": 3,
    }
    for key, ndim in expected_ndim.items():
        if np.asarray(arrays[key]).ndim != ndim:
            raise ValueError(f"{key} must have {ndim} dimensions; found {np.asarray(arrays[key]).ndim}.")

    history = arrays["x_node"].shape[1]
    if any(arrays[key].shape[1] != history for key in ("x_link", "x_task", "edge_a_hist")):
        raise ValueError("State and action history dimensions are inconsistent.")
    horizon = arrays["y_node"].shape[1]
    if any(arrays[key].shape[1] != horizon for key in ("y_link_rate", "y_link_active", "y_task")):
        raise ValueError("Target horizon dimensions are inconsistent.")
    link_shape = arrays["y_link_rate"].shape[1:3]
    if arrays["y_link_active"].shape[1:3] != link_shape:
        raise ValueError("y_link_rate and y_link_active horizon and edge dimensions are inconsistent.")
    if arrays["x_link"].shape[2] != link_shape[1] or arrays["edge_a_hist"].shape[2] != link_shape[1]:
        raise ValueError("Link history, action history, and target edge dimensions are inconsistent.")
    if arrays["edge_a_future"].shape[1:3] != link_shape:
        raise ValueError("edge_a_future horizon and edge dimensions must match link targets.")
    if arrays["x_node"].shape[2] != arrays["y_node"].shape[2]:
        raise ValueError("Node history and target node dimensions are inconsistent.")

    future_nonzero = int(np.count_nonzero(arrays["edge_a_future"]))
    first_step_nonzero = int(np.count_nonzero(arrays["edge_a_future"][:, 0]))
    if future_action_mode == "full" and future_nonzero == 0:
        raise ValueError("edge_a_future is all zero; refusing conditioned training.")
    if future_action_mode == "first_step_only" and first_step_nonzero == 0:
        raise ValueError("edge_a_future first action step is all zero; refusing conditioned training.")
    return {
        "num_samples": int(num_samples),
        "future_action_mode": future_action_mode,
        "future_action_nonzero": future_nonzero,
        "first_step_action_nonzero": first_step_nonzero,
    }


def run_training(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = choose_device(args.device)
    arrays = load_world_model_arrays(args.dataset_dir)
    future_action_mode = getattr(args, "future_action_mode", "full")
    data_preflight = validate_training_arrays(arrays, future_action_mode=future_action_mode)
    if getattr(args, "use_event_memory_features", False):
        arrays = add_event_memory_features(arrays)
    event_memory_routing = getattr(args, "event_memory_routing", "shared")
    activity_memory_dim = 3 if getattr(args, "use_event_memory_features", False) and event_memory_routing == "activity_only" else 0
    train_idx, val_idx, test_idx, split_seed_spec = resolve_seed_splits(
        arrays["sample_seed"],
        train_seeds=parse_seed_list(getattr(args, "train_seeds", None)),
        val_seeds=parse_seed_list(getattr(args, "val_seeds", None)),
        test_seeds=parse_seed_list(getattr(args, "test_seeds", None)),
    )
    stats = make_normalization_stats(arrays, train_idx)
    active_rate_lds_config = build_active_rate_lds_config(arrays, train_idx, args)
    candidate_loss_mask, candidate_loss_mask_info = build_candidate_loss_mask(
        arrays,
        train_idx,
        mode=getattr(args, "candidate_pruning_mode", "none"),
        hard_negative_count=getattr(args, "candidate_hard_negative_count", 0),
    )
    candidate_pruning_scope = getattr(args, "candidate_pruning_scope", "all_losses")
    if candidate_pruning_scope == "rate_only":
        activity_candidate_loss_mask = None
        rate_candidate_loss_mask = candidate_loss_mask
    elif candidate_pruning_scope == "all_losses":
        activity_candidate_loss_mask = candidate_loss_mask
        rate_candidate_loss_mask = None
    else:
        raise ValueError("candidate_pruning_scope must be one of: all_losses, rate_only")

    train_ds = V6WorldModelDataset(arrays, train_idx, stats, future_action_mode=future_action_mode)
    val_ds = V6WorldModelDataset(arrays, val_idx, stats, future_action_mode=future_action_mode)
    test_ds = V6WorldModelDataset(arrays, test_idx, stats, future_action_mode=future_action_mode)
    train_subset = Subset(train_ds, range(min(args.max_train_samples, len(train_ds))))
    val_subset = Subset(val_ds, range(min(args.max_val_samples, len(val_ds))))
    test_subset = Subset(test_ds, range(min(args.max_test_samples, len(test_ds))))
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

    model = build_v8_model_from_arrays(
        arrays,
        hidden_dim=args.hidden_dim,
        graph_mode=args.graph_mode,
        fusion_mode=args.fusion_mode,
        fusion_num_heads=args.fusion_num_heads,
        active_rate_auxiliary=args.active_rate_auxiliary,
        active_rate_head_mode=args.active_rate_head_mode,
        num_rate_experts=args.num_rate_experts,
        rate_output_mode=getattr(args, "model_rate_output_mode", "direct"),
        history_encoder=args.history_encoder,
        latent_transition_mode=args.latent_transition_mode,
        adaptive_edge_context=getattr(args, "adaptive_edge_context", "none"),
        adaptive_edge_topk=getattr(args, "adaptive_edge_topk", 8),
        activity_memory_dim=activity_memory_dim,
        activity_memory_routing="activity_only" if activity_memory_dim > 0 else "none",
        return_message_diagnostics=False,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    history = []
    checkpoint_dir = args.output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_metric = float("inf")
    best_epoch = 0
    best_val_eval = None
    best_state = None
    metric_checkpoint_names = parse_metric_list(getattr(args, "metric_checkpoints", ""))
    metric_checkpoint_states: dict[str, dict] = {}
    metric_checkpoint_values = {name: float("inf") for name in metric_checkpoint_names}
    metric_checkpoint_epochs = {name: 0 for name in metric_checkpoint_names}
    metric_checkpoint_val_eval: dict[str, dict] = {}
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_v8_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            rate_loss_mode=args.rate_loss_mode,
            inactive_rate_weight=args.inactive_rate_weight,
            active_rate_auxiliary_weight=args.active_rate_auxiliary_weight,
            rate_output_mode=args.rate_output_mode,
            inactive_rate_value=args.inactive_rate_value,
            node_loss_weight=args.node_loss_weight,
            activity_loss_weight=args.activity_loss_weight,
            rate_loss_weight=args.rate_loss_weight,
            task_loss_weight=args.task_loss_weight,
            activity_loss_mode=getattr(args, "activity_loss_mode", "bce"),
            activity_pos_weight=getattr(args, "activity_pos_weight", 80.0),
            activity_focal_gamma=getattr(args, "activity_focal_gamma", 2.0),
            inactive_loss_sample_ratio=getattr(args, "inactive_loss_sample_ratio", 1.0),
            false_positive_penalty_weight=getattr(args, "false_positive_penalty_weight", 0.0),
            dynamic_hard_negative_weight=getattr(args, "dynamic_hard_negative_weight", 0.0),
            dynamic_hard_negative_ratio=getattr(args, "dynamic_hard_negative_ratio", 0.1),
            hurdle_train_gate_mode=getattr(args, "hurdle_train_gate_mode", "predicted"),
            hurdle_train_gate_power=getattr(args, "hurdle_train_gate_power", 1.0),
            positive_rate_specialist_weight=getattr(args, "positive_rate_specialist_weight", 0.0),
            positive_rate_target_mode=getattr(args, "positive_rate_target_mode", "raw"),
            positive_rate_loss_mode=getattr(args, "positive_rate_loss_mode", "mse"),
            positive_rate_tweedie_power=getattr(args, "positive_rate_tweedie_power", 1.5),
            positive_rate_raw_stats=stats.get("y_link_rate"),
            high_rate_weight=getattr(args, "high_rate_weight", 1.0),
            high_rate_threshold=getattr(args, "high_rate_threshold", 0.0),
            active_rate_reweight_mode=getattr(args, "active_rate_reweight_mode", "none"),
            active_rate_lds_config=active_rate_lds_config,
            active_rate_bmc_noise_sigma=getattr(args, "bmc_noise_sigma", 1.0),
            active_rate_bmc_minimum_count=getattr(args, "bmc_minimum_count", 3),
            active_mass_loss_weight=getattr(args, "active_mass_loss_weight", 0.0),
            active_mass_target_mode=getattr(args, "active_mass_target_mode", "normalized"),
            active_mass_raw_stats=stats.get("y_link_rate"),
            candidate_loss_mask=activity_candidate_loss_mask,
            candidate_rate_loss_mask=rate_candidate_loss_mask,
        )
        val_metrics = evaluate_v8_model(
            model,
            val_loader,
            device,
            stats,
            rate_output_mode=args.rate_output_mode,
            inactive_rate_value=args.inactive_rate_value,
            hurdle_gate_temperature=getattr(args, "eval_hurdle_gate_temperature", 1.0),
            hurdle_gate_power=getattr(args, "eval_hurdle_gate_power", 1.0),
        )
        best_metric_values = compute_best_metric_values(
            val_metrics,
            min_precision=getattr(args, "best_min_precision", 0.0),
            min_recall=getattr(args, "best_min_recall", 0.0),
            precision_penalty_weight=getattr(args, "best_precision_penalty_weight", 10000.0),
            recall_penalty_weight=getattr(args, "best_recall_penalty_weight", 1000.0),
            min_f1=getattr(args, "best_min_f1", 0.0),
            max_link_rmse=getattr(args, "best_max_link_rmse", 0.0),
            f1_penalty_weight=getattr(args, "best_f1_penalty_weight", 1000.0),
            link_penalty_weight=getattr(args, "best_link_penalty_weight", 10.0),
        )
        best_metric_name = getattr(args, "best_metric", "val_active_rate_rmse")
        selected_metric = best_metric_values[best_metric_name]
        history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
                "best_metric_values": best_metric_values,
            }
        )
        if np.isfinite(selected_metric) and selected_metric < best_metric:
            best_metric = selected_metric
            best_epoch = epoch
            best_val_eval = val_metrics
            best_state = copy.deepcopy(model.state_dict())
        for metric_name in metric_checkpoint_names:
            if metric_name not in best_metric_values:
                raise ValueError(f"Unknown metric checkpoint name: {metric_name}")
            metric_value = best_metric_values[metric_name]
            if np.isfinite(metric_value) and metric_value < metric_checkpoint_values[metric_name]:
                metric_checkpoint_values[metric_name] = float(metric_value)
                metric_checkpoint_epochs[metric_name] = int(epoch)
                metric_checkpoint_val_eval[metric_name] = val_metrics
                metric_checkpoint_states[metric_name] = copy.deepcopy(model.state_dict())
        print(
            f"[v8-{device.type}:{args.graph_mode}] epoch={epoch} "
            f"train_total={train_metrics['total']:.6f} "
            f"val_activity_f1={val_metrics['activity']['f1']:.6f} "
            f"val_active_rate_rmse={val_metrics['active_rate']['active_rmse']:.6f}"
        )

    if best_state is None:
        best_metric = float("nan")
        best_epoch = args.epochs
        best_val_eval = val_metrics
        best_state = copy.deepcopy(model.state_dict())

    checkpoint_path = checkpoint_dir / f"v8_{args.graph_mode}_last.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": model.config.__dict__,
            "epochs": args.epochs,
        },
        checkpoint_path,
    )
    best_checkpoint_path = checkpoint_dir / f"v8_{args.graph_mode}_best.pt"
    if best_state is not None:
        torch.save(
            {
                "model_state": best_state,
                "config": model.config.__dict__,
                "epochs": args.epochs,
                "best_epoch": best_epoch,
                "best_metric": best_metric,
                "best_metric_name": getattr(args, "best_metric", "val_active_rate_rmse"),
            },
            best_checkpoint_path,
        )
    saved_metric_checkpoints = {}
    for metric_name, state in metric_checkpoint_states.items():
        metric_checkpoint_path = checkpoint_dir / f"v8_{args.graph_mode}_best_{metric_name}.pt"
        torch.save(
            {
                "model_state": state,
                "config": model.config.__dict__,
                "epochs": args.epochs,
                "best_epoch": metric_checkpoint_epochs[metric_name],
                "best_metric": metric_checkpoint_values[metric_name],
                "best_metric_name": metric_name,
            },
            metric_checkpoint_path,
        )
        saved_metric_checkpoints[metric_name] = {
            "best_epoch": int(metric_checkpoint_epochs[metric_name]),
            "best_metric_value": float(metric_checkpoint_values[metric_name]),
            "best_val_eval": metric_checkpoint_val_eval.get(metric_name),
            "checkpoint_path": str(metric_checkpoint_path),
        }

    val_eval = evaluate_v8_model(
        model,
        val_loader,
        device,
        stats,
        rate_output_mode=args.rate_output_mode,
        inactive_rate_value=args.inactive_rate_value,
        hurdle_gate_temperature=getattr(args, "eval_hurdle_gate_temperature", 1.0),
        hurdle_gate_power=getattr(args, "eval_hurdle_gate_power", 1.0),
    )
    val_eval_pruned = None
    if candidate_loss_mask is not None:
        val_eval_pruned = evaluate_v8_model(
            model,
            val_loader,
            device,
            stats,
            rate_output_mode=args.rate_output_mode,
            inactive_rate_value=args.inactive_rate_value,
            hurdle_gate_temperature=getattr(args, "eval_hurdle_gate_temperature", 1.0),
            hurdle_gate_power=getattr(args, "eval_hurdle_gate_power", 1.0),
            candidate_eval_mask=candidate_loss_mask,
        )
    threshold = val_eval["activity"]["threshold"]
    test_eval = evaluate_v8_model(
        model,
        test_loader,
        device,
        stats,
        activity_threshold=threshold,
        rate_output_mode=args.rate_output_mode,
        inactive_rate_value=args.inactive_rate_value,
        hurdle_gate_temperature=getattr(args, "eval_hurdle_gate_temperature", 1.0),
        hurdle_gate_power=getattr(args, "eval_hurdle_gate_power", 1.0),
    )
    test_eval_pruned = None
    if candidate_loss_mask is not None:
        pruned_threshold = val_eval_pruned["activity"]["threshold"] if val_eval_pruned is not None else threshold
        test_eval_pruned = evaluate_v8_model(
            model,
            test_loader,
            device,
            stats,
            activity_threshold=pruned_threshold,
            rate_output_mode=args.rate_output_mode,
            inactive_rate_value=args.inactive_rate_value,
            hurdle_gate_temperature=getattr(args, "eval_hurdle_gate_temperature", 1.0),
            hurdle_gate_power=getattr(args, "eval_hurdle_gate_power", 1.0),
            candidate_eval_mask=candidate_loss_mask,
        )
    if best_state is not None:
        model.load_state_dict(best_state)
    best_threshold = best_val_eval["activity"]["threshold"] if best_val_eval is not None else threshold
    best_test_eval = evaluate_v8_model(
        model,
        test_loader,
        device,
        stats,
        activity_threshold=best_threshold,
        rate_output_mode=args.rate_output_mode,
        inactive_rate_value=args.inactive_rate_value,
        hurdle_gate_temperature=getattr(args, "eval_hurdle_gate_temperature", 1.0),
        hurdle_gate_power=getattr(args, "eval_hurdle_gate_power", 1.0),
    )
    best_test_eval_pruned = None
    if candidate_loss_mask is not None:
        best_val_eval_pruned = evaluate_v8_model(
            model,
            val_loader,
            device,
            stats,
            rate_output_mode=args.rate_output_mode,
            inactive_rate_value=args.inactive_rate_value,
            hurdle_gate_temperature=getattr(args, "eval_hurdle_gate_temperature", 1.0),
            hurdle_gate_power=getattr(args, "eval_hurdle_gate_power", 1.0),
            candidate_eval_mask=candidate_loss_mask,
        )
        best_test_eval_pruned = evaluate_v8_model(
            model,
            test_loader,
            device,
            stats,
            activity_threshold=best_val_eval_pruned["activity"]["threshold"],
            rate_output_mode=args.rate_output_mode,
            inactive_rate_value=args.inactive_rate_value,
            hurdle_gate_temperature=getattr(args, "eval_hurdle_gate_temperature", 1.0),
            hurdle_gate_power=getattr(args, "eval_hurdle_gate_power", 1.0),
            candidate_eval_mask=candidate_loss_mask,
        )

    return {
        "status": "training_complete",
        "framework": "PI-JWM",
        "version": "v8",
        "note": "Small local training run; use same-split full GPU training before interpreting performance.",
        "dataset_dir": str(args.dataset_dir),
        "device": str(device),
        "split_seed_spec": split_seed_spec,
        "split_sizes": {
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
            "train_used": int(len(train_subset)),
            "val_used": int(len(val_subset)),
            "test_used": int(len(test_subset)),
        },
        "config": {
            "hidden_dim": args.hidden_dim,
            "graph_mode": args.graph_mode,
            "fusion_mode": args.fusion_mode,
            "fusion_num_heads": args.fusion_num_heads,
            "history_encoder": args.history_encoder,
            "latent_transition_mode": args.latent_transition_mode,
            "adaptive_edge_context": getattr(args, "adaptive_edge_context", "none"),
            "adaptive_edge_topk": getattr(args, "adaptive_edge_topk", 8),
            "epochs": args.epochs,
            "seed": int(getattr(args, "seed", 0)),
            "rate_loss_mode": args.rate_loss_mode,
            "future_action_mode": future_action_mode,
            "inactive_rate_weight": args.inactive_rate_weight,
            "active_rate_auxiliary": args.active_rate_auxiliary,
            "active_rate_auxiliary_weight": args.active_rate_auxiliary_weight,
            "active_rate_head_mode": args.active_rate_head_mode,
            "num_rate_experts": args.num_rate_experts,
            "rate_output_mode": args.rate_output_mode,
            "model_rate_output_mode": getattr(args, "model_rate_output_mode", "direct"),
            "hurdle_train_gate_mode": getattr(args, "hurdle_train_gate_mode", "predicted"),
            "hurdle_train_gate_power": getattr(args, "hurdle_train_gate_power", 1.0),
            "use_event_memory_features": bool(getattr(args, "use_event_memory_features", False)),
            "event_memory_routing": event_memory_routing,
            "activity_memory_dim": int(activity_memory_dim),
            "info_edge_dim": int(arrays["x_link"].shape[-1]),
            "inactive_rate_value": args.inactive_rate_value,
            "best_metric": getattr(args, "best_metric", "val_active_rate_rmse"),
            "best_min_precision": getattr(args, "best_min_precision", 0.0),
            "best_min_recall": getattr(args, "best_min_recall", 0.0),
            "best_precision_penalty_weight": getattr(args, "best_precision_penalty_weight", 10000.0),
            "best_recall_penalty_weight": getattr(args, "best_recall_penalty_weight", 1000.0),
            "best_min_f1": getattr(args, "best_min_f1", 0.0),
            "best_max_link_rmse": getattr(args, "best_max_link_rmse", 0.0),
            "best_f1_penalty_weight": getattr(args, "best_f1_penalty_weight", 1000.0),
            "best_link_penalty_weight": getattr(args, "best_link_penalty_weight", 10.0),
            "metric_checkpoints": metric_checkpoint_names,
            "node_loss_weight": args.node_loss_weight,
            "activity_loss_weight": args.activity_loss_weight,
            "rate_loss_weight": args.rate_loss_weight,
            "task_loss_weight": args.task_loss_weight,
            "activity_loss_mode": getattr(args, "activity_loss_mode", "bce"),
            "activity_pos_weight": getattr(args, "activity_pos_weight", 80.0),
            "activity_focal_gamma": getattr(args, "activity_focal_gamma", 2.0),
            "inactive_loss_sample_ratio": getattr(args, "inactive_loss_sample_ratio", 1.0),
            "false_positive_penalty_weight": getattr(args, "false_positive_penalty_weight", 0.0),
            "dynamic_hard_negative_weight": getattr(args, "dynamic_hard_negative_weight", 0.0),
            "dynamic_hard_negative_ratio": getattr(args, "dynamic_hard_negative_ratio", 0.1),
            "eval_hurdle_gate_temperature": getattr(args, "eval_hurdle_gate_temperature", 1.0),
            "eval_hurdle_gate_power": getattr(args, "eval_hurdle_gate_power", 1.0),
            "positive_rate_specialist_weight": getattr(args, "positive_rate_specialist_weight", 0.0),
            "positive_rate_target_mode": getattr(args, "positive_rate_target_mode", "raw"),
            "positive_rate_loss_mode": getattr(args, "positive_rate_loss_mode", "mse"),
            "positive_rate_tweedie_power": getattr(args, "positive_rate_tweedie_power", 1.5),
            "positive_rate_raw_stats": bool(stats.get("y_link_rate") is not None),
            "high_rate_weight": getattr(args, "high_rate_weight", 1.0),
            "high_rate_threshold": getattr(args, "high_rate_threshold", 0.0),
            "active_rate_reweight_mode": getattr(args, "active_rate_reweight_mode", "none"),
            "active_rate_lds_config": active_rate_lds_config,
            "active_rate_bmc_noise_sigma": getattr(args, "bmc_noise_sigma", 1.0),
            "active_rate_bmc_minimum_count": getattr(args, "bmc_minimum_count", 3),
            "active_mass_loss_weight": getattr(args, "active_mass_loss_weight", 0.0),
            "active_mass_target_mode": getattr(args, "active_mass_target_mode", "normalized"),
            "candidate_pruning_mode": getattr(args, "candidate_pruning_mode", "none"),
            "candidate_hard_negative_count": getattr(args, "candidate_hard_negative_count", 0),
            "candidate_pruning_scope": candidate_pruning_scope,
        },
        "candidate_loss_mask": candidate_loss_mask_info,
        "data_preflight": data_preflight,
        "history": history,
        "val_eval": val_eval,
        "test_eval": test_eval,
        "val_eval_pruned": val_eval_pruned,
        "test_eval_pruned": test_eval_pruned,
        "best_epoch": int(best_epoch),
        "best_metric_name": getattr(args, "best_metric", "val_active_rate_rmse"),
        "best_metric_value": float(best_metric),
        "best_val_eval": best_val_eval,
        "best_test_eval": best_test_eval,
        "best_test_eval_pruned": best_test_eval_pruned,
        "metric_checkpoints": saved_metric_checkpoints,
        "checkpoint_path": str(checkpoint_path),
        "best_checkpoint_path": str(best_checkpoint_path),
    }


def render_report(summary: dict) -> str:
    val = summary["val_eval"]
    test = summary["test_eval"]
    return "\n".join(
        [
            "# PI-JWM v8 Full World Model Training Smoke",
            "",
            f"- framework: {summary['framework']}",
            f"- version: {summary['version']}",
            f"- status: {summary['status']}",
            f"- note: {summary['note']}",
            f"- dataset: `{summary['dataset_dir']}`",
            f"- device: `{summary['device']}`",
            f"- split_sizes: `{summary['split_sizes']}`",
            f"- config: `{summary['config']}`",
            f"- val activity F1: {val['activity']['f1']:.6f}",
            f"- val active-rate RMSE: {val['active_rate']['active_rmse']:.6f}",
            f"- val link-rate RMSE: {val['link_rate']['rmse']:.6f}",
            f"- val node RMSE: {val['node']['rmse']:.6f}",
            f"- val task RMSE: {val['task']['rmse']:.6f}",
            f"- test activity F1: {test['activity']['f1']:.6f}",
            f"- test active-rate RMSE: {test['active_rate']['active_rmse']:.6f}",
            f"- test link-rate RMSE: {test['link_rate']['rmse']:.6f}",
            f"- test node RMSE: {test['node']['rmse']:.6f}",
            f"- test task RMSE: {test['task']['rmse']:.6f}",
            f"- best epoch: {summary['best_epoch']}",
            f"- best metric: {summary['best_metric_name']}={summary['best_metric_value']:.6f}",
            f"- best-val active-rate RMSE: {summary['best_val_eval']['active_rate']['active_rmse']:.6f}",
            f"- best-test active-rate RMSE: {summary['best_test_eval']['active_rate']['active_rmse']:.6f}",
            f"- checkpoint: `{summary['checkpoint_path']}`",
            f"- best checkpoint: `{summary['best_checkpoint_path']}`",
        ]
    ) + "\n"


def compute_best_metric_values(
    val_metrics: dict,
    min_precision: float = 0.0,
    min_recall: float = 0.0,
    precision_penalty_weight: float = 10000.0,
    recall_penalty_weight: float = 1000.0,
    min_f1: float = 0.0,
    max_link_rmse: float = 0.0,
    f1_penalty_weight: float = 1000.0,
    link_penalty_weight: float = 10.0,
) -> dict[str, float]:
    active_rate_rmse = float(val_metrics["active_rate"]["active_rmse"])
    link_rate_rmse = float(val_metrics["link_rate"]["rmse"])
    node_rmse = float(val_metrics["node"]["rmse"])
    task_rmse = float(val_metrics["task"]["rmse"])
    activity_f1 = float(val_metrics["activity"]["f1"])
    activity_precision = float(val_metrics["activity"].get("precision", 0.0))
    activity_recall = float(val_metrics["activity"].get("recall", 0.0))
    precision_shortfall = max(0.0, float(min_precision) - activity_precision)
    recall_shortfall = max(0.0, float(min_recall) - activity_recall)
    operating_penalty = (
        float(precision_penalty_weight) * precision_shortfall
        + float(recall_penalty_weight) * recall_shortfall
    )
    f1_shortfall = max(0.0, float(min_f1) - activity_f1)
    link_excess = max(0.0, link_rate_rmse - float(max_link_rmse)) if max_link_rmse > 0.0 else 0.0
    link_f1_penalty = float(f1_penalty_weight) * f1_shortfall + float(link_penalty_weight) * link_excess
    composite = active_rate_rmse + link_rate_rmse + 0.1 * node_rmse + task_rmse - 10.0 * activity_f1
    return {
        "val_active_rate_rmse": active_rate_rmse,
        "val_link_rate_rmse": link_rate_rmse,
        "val_activity_f1": -activity_f1,
        "val_node_rmse": node_rmse,
        "val_task_rmse": task_rmse,
        "val_composite": composite,
        "val_precision_constrained_active_rate": active_rate_rmse + operating_penalty,
        "val_precision_constrained_composite": composite + operating_penalty,
        "val_link_f1_constrained_active_rate": active_rate_rmse + link_f1_penalty,
        "val_link_f1_constrained_composite": composite + link_f1_penalty,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = run_training(args)
    summary_path = args.output_dir / "v8_full_training_summary.json"
    report_path = args.output_dir / "v8_full_training_report.md"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(render_report(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"summary_path={summary_path}")
    print(f"report_path={report_path}")


if __name__ == "__main__":
    main()




