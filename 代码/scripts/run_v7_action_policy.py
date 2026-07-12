"""Train a PI-JWM v7 behavior-cloning policy for state-to-action prediction."""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pi_jwm.paths import ARTIFACTS_DIR
from pi_jwm.v6_data import (
    V6WorldModelDataset,
    load_world_model_arrays,
    make_normalization_stats,
    split_by_seed,
)
from pi_jwm.v6_dual_graph import V6DualGraphBatch
from pi_jwm.v7_action_policy import V7ActionPolicy, V7ActionPolicyConfig


DEFAULT_DATASET_DIR = (
    ARTIFACTS_DIR
    / "experiments"
    / "airfogsim_v0"
    / "datasets"
    / "world_model_dataset_seed0_9_v0"
)
OUTPUT_DIR = ARTIFACTS_DIR / "experiments" / "pi_jwm_v7_action_policy"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PI-JWM v7 state-to-action behavior-cloning policy.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max-train-samples", type=int, default=512)
    parser.add_argument("--max-val-samples", type=int, default=128)
    parser.add_argument("--max-test-samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument(
        "--fusion-mode",
        choices=("concat", "gated", "cross_attention", "hybrid_attention"),
        default="cross_attention",
    )
    parser.add_argument("--fusion-num-heads", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--train-seeds", default=None)
    parser.add_argument("--val-seeds", default=None)
    parser.add_argument("--test-seeds", default=None)
    parser.add_argument("--max-pos-weight", type=float, default=500.0)
    parser.add_argument("--value-loss-weight", type=float, default=1.0)
    parser.add_argument("--inactive-value-weight", type=float, default=0.02)
    parser.add_argument("--active-value-weight", type=float, default=0.0)
    parser.add_argument("--max-active-value-weight", type=float, default=20.0)
    parser.add_argument("--activity-value-weight", type=float, default=0.0)
    parser.add_argument("--max-activity-value-weight", type=float, default=20.0)
    parser.add_argument("--use-edge-activity-head", action="store_true")
    parser.add_argument("--edge-activity-loss-weight", type=float, default=0.0)
    parser.add_argument("--max-edge-pos-weight", type=float, default=500.0)
    parser.add_argument("--seed", type=int, default=20260609)
    return parser.parse_args()


class V7ActionPolicyDataset(Dataset):
    def __init__(
        self,
        arrays: dict[str, np.ndarray],
        indices: np.ndarray,
        stats: dict,
        action_scale: np.ndarray,
    ):
        self.arrays = arrays
        self.indices = np.asarray(indices, dtype=np.int64)
        self.base = V6WorldModelDataset(arrays, self.indices, stats)
        self.action_scale = torch.as_tensor(action_scale.reshape(1, 1, -1), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        batch, _ = self.base[item]
        source_idx = int(self.indices[item])
        raw_action = torch.from_numpy(self.arrays["edge_a_future"][source_idx].astype(np.float32))
        active = (raw_action > 1e-9).to(torch.float32)
        edge_active = (active.any(dim=-1)).to(torch.float32)
        value = raw_action / self.action_scale
        target = {
            "action_active": active,
            "edge_active": edge_active,
            "action_value": value,
            "action_raw": raw_action,
        }
        return batch, target


def collate_action_policy_batch(items):
    batches, targets = zip(*items)
    batch = V6DualGraphBatch(
        node_history=torch.stack([item.node_history for item in batches]),
        physical_edge_history=torch.stack([item.physical_edge_history for item in batches]),
        info_edge_history=torch.stack([item.info_edge_history for item in batches]),
        action_history=torch.stack([item.action_history for item in batches]),
        future_actions=torch.stack([item.future_actions for item in batches]),
        task_history=torch.stack([item.task_history for item in batches]),
    )
    target = {
        "action_active": torch.stack([item["action_active"] for item in targets]),
        "edge_active": torch.stack([item["edge_active"] for item in targets]),
        "action_value": torch.stack([item["action_value"] for item in targets]),
        "action_raw": torch.stack([item["action_raw"] for item in targets]),
    }
    return batch, target


def make_action_scale(arrays: dict[str, np.ndarray], train_idx: np.ndarray) -> np.ndarray:
    train_actions = arrays["edge_a_future"][train_idx]
    scale = np.nanmax(train_actions, axis=(0, 1, 2)).astype(np.float32)
    scale = np.where(scale < 1e-6, 1.0, scale)
    return scale


def make_pos_weight(arrays: dict[str, np.ndarray], train_idx: np.ndarray, max_pos_weight: float) -> torch.Tensor:
    active = arrays["edge_a_future"][train_idx] > 1e-9
    pos = active.sum(axis=(0, 1, 2)).astype(np.float64)
    total = np.prod(active.shape[:3])
    neg = total - pos
    weight = neg / np.maximum(pos, 1.0)
    weight = np.clip(weight, 1.0, max_pos_weight)
    return torch.as_tensor(weight.astype(np.float32))


def make_edge_pos_weight(arrays: dict[str, np.ndarray], train_idx: np.ndarray, max_pos_weight: float) -> torch.Tensor:
    active = np.any(arrays["edge_a_future"][train_idx] > 1e-9, axis=-1)
    pos = float(active.sum())
    total = float(np.prod(active.shape))
    neg = total - pos
    weight = neg / max(pos, 1.0)
    weight = float(np.clip(weight, 1.0, max_pos_weight))
    return torch.as_tensor([weight], dtype=torch.float32)


def make_config(arrays: dict[str, np.ndarray], args: argparse.Namespace) -> V7ActionPolicyConfig:
    return V7ActionPolicyConfig(
        node_dim=int(arrays["x_node"].shape[-1]),
        physical_edge_dim=8,
        info_edge_dim=int(arrays["x_link"].shape[-1]),
        action_dim=int(arrays["edge_a_hist"].shape[-1]),
        task_dim=int(arrays["x_task"].shape[-1]),
        hidden_dim=args.hidden_dim,
        horizon=int(arrays["edge_a_future"].shape[1]),
        fusion_mode=args.fusion_mode,
        fusion_num_heads=args.fusion_num_heads,
        use_edge_activity_head=bool(getattr(args, "use_edge_activity_head", False)),
    )


def move_batch_to_device(batch, device: torch.device):
    return type(batch)(
        node_history=batch.node_history.to(device),
        physical_edge_history=batch.physical_edge_history.to(device),
        info_edge_history=batch.info_edge_history.to(device),
        action_history=batch.action_history.to(device),
        future_actions=batch.future_actions.to(device),
        task_history=batch.task_history.to(device),
    )


def move_target_to_device(target: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: value.to(device) for name, value in target.items()}


def compute_policy_loss(
    outputs: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    pos_weight: torch.Tensor,
    value_loss_weight: float,
    inactive_value_weight: float,
    active_value_weight: float = 0.0,
    max_active_value_weight: float = 20.0,
    activity_value_weight: float = 0.0,
    max_activity_value_weight: float = 20.0,
    edge_activity_loss_weight: float = 0.0,
    edge_pos_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    active = target["action_active"]
    raw_activity_loss = nn.functional.binary_cross_entropy_with_logits(
        outputs["action_logit"],
        active,
        pos_weight=pos_weight.to(outputs["action_logit"].device),
        reduction="none",
    )
    if float(activity_value_weight) > 0.0:
        activity_raw = torch.log1p(torch.clamp(target["action_raw"], min=0.0))
        activity_weight = 1.0 + active * float(activity_value_weight) * activity_raw
        activity_weight = torch.clamp(activity_weight, min=1.0, max=max(float(max_activity_value_weight), 1.0))
        activity_loss = (raw_activity_loss * activity_weight).mean()
    else:
        activity_loss = raw_activity_loss.mean()

    if float(edge_activity_loss_weight) > 0.0:
        if "edge_logit" not in outputs:
            raise ValueError("edge_activity_loss_weight requires outputs['edge_logit']")
        edge_pos = torch.ones(1, device=outputs["edge_logit"].device) if edge_pos_weight is None else edge_pos_weight.to(outputs["edge_logit"].device)
        edge_activity_loss = nn.functional.binary_cross_entropy_with_logits(
            outputs["edge_logit"],
            target["edge_active"],
            pos_weight=edge_pos,
        )
    else:
        edge_activity_loss = raw_activity_loss.new_tensor(0.0)

    value_error = (outputs["action_value"] - target["action_value"]) ** 2
    active_mask = active > 0.5
    inactive_mask = ~active_mask
    if active_mask.any():
        if float(active_value_weight) > 0.0:
            active_raw = torch.log1p(torch.clamp(target["action_raw"], min=0.0))
            value_weight = 1.0 + active_raw * float(active_value_weight)
            value_weight = torch.clamp(value_weight, min=1.0, max=max(float(max_active_value_weight), 1.0))
            active_value_loss = (value_error[active_mask] * value_weight[active_mask]).mean()
        else:
            active_value_loss = value_error[active_mask].mean()
    else:
        active_value_loss = value_error.new_tensor(0.0)
    inactive_value_loss = (outputs["action_value"][inactive_mask] ** 2).mean() if inactive_mask.any() else value_error.new_tensor(0.0)
    total = (
        activity_loss
        + float(edge_activity_loss_weight) * edge_activity_loss
        + value_loss_weight * active_value_loss
        + inactive_value_weight * inactive_value_loss
    )
    return total, {
        "total": float(total.detach().cpu()),
        "activity": float(activity_loss.detach().cpu()),
        "edge_activity": float(edge_activity_loss.detach().cpu()),
        "active_value": float(active_value_loss.detach().cpu()),
        "inactive_value": float(inactive_value_loss.detach().cpu()),
    }


def train(args: argparse.Namespace) -> dict:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    arrays = load_world_model_arrays(args.dataset_dir)
    train_idx, val_idx, test_idx, split_seed_spec = resolve_policy_seed_splits(
        arrays["sample_seed"],
        train_seeds=parse_seed_list(args.train_seeds),
        val_seeds=parse_seed_list(args.val_seeds),
        test_seeds=parse_seed_list(args.test_seeds),
    )
    stats = make_normalization_stats(arrays, train_idx)
    action_scale = make_action_scale(arrays, train_idx)
    pos_weight = make_pos_weight(arrays, train_idx, args.max_pos_weight)
    edge_pos_weight = make_edge_pos_weight(arrays, train_idx, args.max_edge_pos_weight)

    train_ds = V7ActionPolicyDataset(arrays, train_idx, stats, action_scale)
    val_ds = V7ActionPolicyDataset(arrays, val_idx, stats, action_scale)
    test_ds = V7ActionPolicyDataset(arrays, test_idx, stats, action_scale)
    train_subset = Subset(train_ds, range(min(args.max_train_samples, len(train_ds))))
    val_subset = Subset(val_ds, range(min(args.max_val_samples, len(val_ds))))
    test_subset = Subset(test_ds, range(min(args.max_test_samples, len(test_ds))))

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collate_action_policy_batch,
    )
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_action_policy_batch)
    test_loader = DataLoader(test_subset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_action_policy_batch)

    config = make_config(arrays, args)
    model = V7ActionPolicy(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_state = None
    best_epoch = 0
    best_score = float("inf")
    history = []
    action_scale_t = torch.as_tensor(action_scale.reshape(1, 1, 1, -1), dtype=torch.float32, device=device)
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_rows = []
        for batch, target in train_loader:
            batch = move_batch_to_device(batch, device)
            target = move_target_to_device(target, device)
            outputs = model(batch)
            loss, parts = compute_policy_loss(
                outputs,
                target,
                pos_weight,
                value_loss_weight=args.value_loss_weight,
                inactive_value_weight=args.inactive_value_weight,
                active_value_weight=args.active_value_weight,
                max_active_value_weight=args.max_active_value_weight,
                activity_value_weight=args.activity_value_weight,
                max_activity_value_weight=args.max_activity_value_weight,
                edge_activity_loss_weight=args.edge_activity_loss_weight,
                edge_pos_weight=edge_pos_weight,
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_rows.append(parts)
        train_loss = mean_rows(epoch_rows)
        val_loss = evaluate_loss(model, val_loader, device, pos_weight, edge_pos_weight, args, action_scale_t)
        history.append({"epoch": epoch, "train": train_loss, "val": val_loss})
        score = select_policy_score(val_loss, edge_activity_loss_weight=args.edge_activity_loss_weight)
        if score < best_score:
            best_score = score
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
        print(
            f"[v7-policy:{device.type}] epoch={epoch} train_total={train_loss['total']:.6f} "
            f"val_f1={val_loss['activity_f1']:.6f} val_active_value_rmse={val_loss['active_value_rmse']:.6f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    val_predictions = collect_predictions(model, val_loader, device, action_scale_t)
    threshold = choose_threshold(val_predictions["prob"], val_predictions["active"])
    val_eval = evaluate_predictions(val_predictions, threshold)
    test_eval = evaluate_predictions(collect_predictions(model, test_loader, device, action_scale_t), threshold)

    checkpoint_dir = args.output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"v7_action_policy_{args.fusion_mode}_best.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config.__dict__,
            "best_epoch": best_epoch,
            "best_val_active_value_rmse": val_eval["active_value_rmse"],
            "best_score": best_score,
            "action_scale": action_scale.tolist(),
            "pos_weight": pos_weight.tolist(),
            "edge_pos_weight": edge_pos_weight.tolist(),
        },
        checkpoint_path,
    )

    return {
        "framework": "PI-JWM",
        "module": "v7_action_policy",
        "note": "Behavior-cloning policy for logged edge actions; offline supervised policy, not online RL.",
        "dataset_dir": str(args.dataset_dir),
        "split_sizes": {
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
            "train_used": int(len(train_subset)),
            "val_used": int(len(val_subset)),
            "test_used": int(len(test_subset)),
        },
        "split_seed_spec": split_seed_spec,
        "config": config.__dict__,
        "action_features": [str(x) for x in arrays["edge_action_features"].tolist()],
        "action_scale": action_scale.tolist(),
        "pos_weight": pos_weight.tolist(),
        "edge_pos_weight": edge_pos_weight.tolist(),
        "history": history,
        "best_epoch": int(best_epoch),
        "best_val_active_value_rmse": float(val_eval["active_value_rmse"]),
        "best_score": float(best_score),
        "activity_threshold": float(threshold),
        "checkpoint_path": str(checkpoint_path),
        "val_eval": val_eval,
        "test_eval": test_eval,
    }


def evaluate_loss(
    model,
    loader,
    device,
    pos_weight,
    edge_pos_weight,
    args,
    action_scale_t,
) -> dict[str, float]:
    model.eval()
    loss_rows = []
    with torch.no_grad():
        for batch, target in loader:
            batch = move_batch_to_device(batch, device)
            target = move_target_to_device(target, device)
            outputs = model(batch)
            _, parts = compute_policy_loss(
                outputs,
                target,
                pos_weight,
                value_loss_weight=args.value_loss_weight,
                inactive_value_weight=args.inactive_value_weight,
                active_value_weight=args.active_value_weight,
                max_active_value_weight=args.max_active_value_weight,
                activity_value_weight=args.activity_value_weight,
                max_activity_value_weight=args.max_activity_value_weight,
                edge_activity_loss_weight=args.edge_activity_loss_weight,
                edge_pos_weight=edge_pos_weight,
            )
            loss_rows.append(parts)
    predictions = collect_predictions(model, loader, device, action_scale_t)
    threshold = choose_threshold(predictions["prob"], predictions["active"])
    metrics = evaluate_predictions(predictions, threshold)
    return {**mean_rows(loss_rows), **metrics}


def collect_predictions(model, loader, device, action_scale_t) -> dict[str, np.ndarray]:
    model.eval()
    rows = {"prob": [], "active": [], "value_pred": [], "value_true": []}
    with torch.no_grad():
        for batch, target in loader:
            batch = move_batch_to_device(batch, device)
            target = move_target_to_device(target, device)
            outputs = model(batch)
            rows["prob"].append(torch.sigmoid(outputs["action_logit"]).cpu().numpy())
            if "edge_logit" in outputs:
                rows.setdefault("edge_prob", []).append(torch.sigmoid(outputs["edge_logit"]).cpu().numpy())
            rows["active"].append(target["action_active"].cpu().numpy())
            rows["value_pred"].append((outputs["action_value"] * action_scale_t).cpu().numpy())
            rows["value_true"].append(target["action_raw"].cpu().numpy())
    return {name: np.concatenate(values, axis=0) for name, values in rows.items()}


def choose_threshold(prob: np.ndarray, active: np.ndarray) -> float:
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(0.01, 0.99, 99):
        score = binary_metrics(prob >= threshold, active > 0.5)["f1"]
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)
    return best_threshold


def evaluate_predictions(predictions: dict[str, np.ndarray], threshold: float) -> dict[str, float]:
    active = predictions["active"] > 0.5
    pred_active = predictions["prob"] >= threshold
    metrics = binary_metrics(pred_active, active)
    any_metrics = binary_metrics(pred_active.any(axis=-1), active.any(axis=-1))
    if "edge_prob" in predictions:
        edge_pred = predictions["edge_prob"] >= threshold
    else:
        edge_pred = pred_active.any(axis=-1)
    edge_true = active.any(axis=-1)
    edge_metrics = binary_metrics(edge_pred, edge_true)
    active_count = int(active.sum())
    if active_count:
        active_value_rmse = float(np.sqrt(np.mean((predictions["value_pred"][active] - predictions["value_true"][active]) ** 2)))
        zero_value_rmse = float(np.sqrt(np.mean(predictions["value_true"][active] ** 2)))
    else:
        active_value_rmse = float("nan")
        zero_value_rmse = float("nan")
    return {
        "activity_threshold": float(threshold),
        "activity_precision": metrics["precision"],
        "activity_recall": metrics["recall"],
        "activity_f1": metrics["f1"],
        "activity_tp": float(metrics["tp"]),
        "activity_fp": float(metrics["fp"]),
        "activity_fn": float(metrics["fn"]),
        "activity_tn": float(metrics["tn"]),
        "edge_step_activity_f1": any_metrics["f1"],
        "edge_activity_precision": edge_metrics["precision"],
        "edge_activity_recall": edge_metrics["recall"],
        "edge_activity_f1": edge_metrics["f1"],
        "edge_activity_tp": float(edge_metrics["tp"]),
        "edge_activity_fp": float(edge_metrics["fp"]),
        "edge_activity_fn": float(edge_metrics["fn"]),
        "edge_activity_tn": float(edge_metrics["tn"]),
        "active_count": float(active_count),
        "active_value_rmse": active_value_rmse,
        "zero_policy_active_value_rmse": zero_value_rmse,
    }


def select_policy_score(metrics: dict[str, float], edge_activity_loss_weight: float = 0.0) -> float:
    if float(edge_activity_loss_weight) > 0.0:
        return -float(metrics["edge_activity_f1"])
    return float(metrics["active_value_rmse"])


def binary_metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, float | int]:
    pred = np.asarray(pred, dtype=bool)
    true = np.asarray(true, dtype=bool)
    tp = int((pred & true).sum())
    fp = int((pred & ~true).sum())
    fn = int((~pred & true).sum())
    tn = int((~pred & ~true).sum())
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2.0 * precision * recall, precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def mean_rows(rows: list[dict[str, float]]) -> dict[str, float]:
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


def parse_seed_list(value: str | None) -> list[int] | None:
    if value is None:
        return None
    parts = [part for part in value.replace(",", " ").split() if part]
    return [int(part) for part in parts]


def resolve_policy_seed_splits(
    sample_seed: np.ndarray,
    train_seeds: list[int] | None = None,
    val_seeds: list[int] | None = None,
    test_seeds: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, list[int]]]:
    if train_seeds is None and val_seeds is None and test_seeds is None:
        train_idx, val_idx, test_idx = split_by_seed(sample_seed)
        return train_idx, val_idx, test_idx, {"train_seeds": list(range(0, 8)), "val_seeds": [8], "test_seeds": [9]}
    train_seeds = list(range(0, 8)) if train_seeds is None else train_seeds
    val_seeds = [8] if val_seeds is None else val_seeds
    test_seeds = [9] if test_seeds is None else test_seeds
    sets = [set(train_seeds), set(val_seeds), set(test_seeds)]
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise ValueError("train, val, and test seed sets must be disjoint")
    sample_seed = np.asarray(sample_seed)
    train_idx = np.where(np.isin(sample_seed, train_seeds))[0]
    val_idx = np.where(np.isin(sample_seed, val_seeds))[0]
    test_idx = np.where(np.isin(sample_seed, test_seeds))[0]
    for name, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        if len(idx) == 0:
            raise ValueError(f"Custom seed split produced an empty {name} split.")
    return train_idx, val_idx, test_idx, {
        "train_seeds": [int(seed) for seed in train_seeds],
        "val_seeds": [int(seed) for seed in val_seeds],
        "test_seeds": [int(seed) for seed in test_seeds],
    }


def render_report(summary: dict) -> str:
    test_eval = summary["test_eval"]
    val_eval = summary["val_eval"]
    return "\n".join(
        [
            "# PI-JWM v7 State-to-Action Policy",
            "",
            "This report records a supervised behavior-cloning policy for logged edge-level actions.",
            "",
            "## Setup",
            "",
            f"- Fusion mode: `{summary['config']['fusion_mode']}`",
            f"- Hidden dim: `{summary['config']['hidden_dim']}`",
            f"- Horizon: `{summary['config']['horizon']}`",
            f"- Split sizes: `{summary['split_sizes']}`",
            f"- Best epoch: `{summary['best_epoch']}`",
            "",
            "## Test Metrics",
            "",
            f"- action activity F1: `{test_eval['activity_f1']:.6f}`",
            f"- action activity precision/recall: `{test_eval['activity_precision']:.6f}` / `{test_eval['activity_recall']:.6f}`",
            f"- edge-step activity F1: `{test_eval['edge_step_activity_f1']:.6f}`",
            f"- active action value RMSE: `{test_eval['active_value_rmse']:.6f}`",
            f"- zero-policy active value RMSE: `{test_eval['zero_policy_active_value_rmse']:.6f}`",
            "",
            "## Validation Metrics",
            "",
            f"- action activity F1: `{val_eval['activity_f1']:.6f}`",
            f"- active action value RMSE: `{val_eval['active_value_rmse']:.6f}`",
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = train(args)
    summary_path = args.output_dir / "v7_action_policy_summary.json"
    report_path = args.output_dir / "v7_action_policy_report.md"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(render_report(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"summary_path={summary_path}")
    print(f"report_path={report_path}")


if __name__ == "__main__":
    main()
