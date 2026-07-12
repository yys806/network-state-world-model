"""Train a PI-JWM v11 action policy with discrete positive-value bins."""

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
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from pi_jwm.paths import ARTIFACTS_DIR
from pi_jwm.v6_data import V6WorldModelDataset, load_world_model_arrays, make_normalization_stats
from pi_jwm.v6_dual_graph import V6DualGraphBatch
from pi_jwm.v7_action_policy import V7ActionPolicy, V7ActionPolicyConfig

from run_v7_action_policy import (
    binary_metrics,
    choose_device,
    choose_threshold,
    make_edge_pos_weight,
    make_pos_weight,
    mean_rows,
    move_batch_to_device,
    parse_seed_list,
    resolve_policy_seed_splits,
    select_policy_score,
)


DEFAULT_DATASET_DIR = ARTIFACTS_DIR / "experiments" / "airfogsim_v0" / "datasets" / "world_model_dataset_active_heavy_v2_60seed_20260619"
OUTPUT_DIR = ARTIFACTS_DIR / "experiments" / "pi_jwm_v11_discrete_value_policy_cpu"
IGNORE_INDEX = -100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PI-JWM v11 discrete value-bin action policy.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-train-samples", type=int, default=1024)
    parser.add_argument("--max-val-samples", type=int, default=256)
    parser.add_argument("--max-test-samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--fusion-mode", choices=("concat", "gated", "cross_attention", "hybrid_attention"), default="cross_attention")
    parser.add_argument("--fusion-num-heads", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--train-seeds", default=None)
    parser.add_argument("--val-seeds", default=None)
    parser.add_argument("--test-seeds", default=None)
    parser.add_argument("--max-pos-weight", type=float, default=500.0)
    parser.add_argument("--activity-value-weight", type=float, default=0.0)
    parser.add_argument("--max-activity-value-weight", type=float, default=20.0)
    parser.add_argument("--use-edge-activity-head", action="store_true")
    parser.add_argument("--edge-activity-loss-weight", type=float, default=0.0)
    parser.add_argument("--max-edge-pos-weight", type=float, default=500.0)
    parser.add_argument(
        "--active-edge-sample-weight",
        type=float,
        default=1.0,
        help="Multiplier for edge-activity BCE terms on sample/step rows that contain at least one true active edge.",
    )
    parser.add_argument(
        "--hard-negative-edge-weight",
        type=float,
        default=1.0,
        help="Multiplier for edge-activity BCE terms on inactive edges whose predicted probability exceeds --hard-negative-edge-threshold.",
    )
    parser.add_argument(
        "--hard-negative-edge-threshold",
        type=float,
        default=0.5,
        help="Probability threshold used to identify hard negative inactive edges for weighted edge-activity BCE.",
    )
    parser.add_argument(
        "--edge-tversky-loss-weight",
        type=float,
        default=0.0,
        help="Auxiliary soft Tversky loss weight for edge activity; alpha controls false-positive penalty.",
    )
    parser.add_argument(
        "--edge-tversky-alpha",
        type=float,
        default=0.7,
        help="False-positive penalty coefficient for edge Tversky loss.",
    )
    parser.add_argument(
        "--edge-tversky-beta",
        type=float,
        default=0.3,
        help="False-negative penalty coefficient for edge Tversky loss.",
    )
    parser.add_argument("--use-step-total-head", action="store_true")
    parser.add_argument("--step-total-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--value-mode",
        choices=("discrete_bins", "coupled_tokens", "hierarchical_tokens"),
        default="discrete_bins",
        help="Positive value target form. coupled_tokens predicts group tuples; hierarchical_tokens predicts count and total per group.",
    )
    parser.add_argument("--bin-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--save-epoch-checkpoints",
        action="store_true",
        help="Save one checkpoint per epoch for bridge-proxy checkpoint selection.",
    )
    parser.add_argument(
        "--value-bin-loss",
        choices=("ce", "inverse_sqrt_ce", "effective_ce", "focal", "inverse_sqrt_focal"),
        default="ce",
    )
    parser.add_argument("--focal-gamma", type=float, default=1.5)
    parser.add_argument("--effective-beta", type=float, default=0.999)
    parser.add_argument("--seed", type=int, default=20260620)
    return parser.parse_args()


def save_epoch_checkpoint(
    checkpoint_dir: Path,
    fusion_mode: str,
    epoch: int,
    model_state: dict[str, torch.Tensor],
    checkpoint_payload: dict,
) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"v11_discrete_value_policy_{fusion_mode}_epoch_{int(epoch):03d}.pt"
    torch.save(
        {
            **checkpoint_payload,
            "model_state": model_state,
            "epoch": int(epoch),
            "best_epoch": int(epoch),
            "best_metric_name": "epoch",
            "best_metric_value": float("nan"),
        },
        path,
    )
    return path


def default_coupled_value_groups(action_dim: int) -> list[list[int]]:
    if action_dim >= 6:
        return [[0], [1, 2], [3, 4], [5]]
    return [[dim] for dim in range(action_dim)]


def build_value_vocab(true_value: np.ndarray, eps: float = 1e-9) -> dict[str, np.ndarray | int]:
    true_value = np.asarray(true_value, dtype=np.float32)
    if true_value.ndim != 4:
        raise ValueError("true_value must have shape [sample, horizon, edge, action_dim]")
    action_dim = true_value.shape[-1]
    vocab_lists: list[np.ndarray] = []
    max_bins = 1
    for dim in range(action_dim):
        positives = true_value[..., dim][true_value[..., dim] > eps]
        values = np.unique(np.round(positives.astype(np.float32), 6)).astype(np.float32)
        if values.size == 0:
            values = np.array([0.0], dtype=np.float32)
        vocab_lists.append(values)
        max_bins = max(max_bins, int(values.size))
    values = np.zeros((action_dim, max_bins), dtype=np.float32)
    sizes = np.zeros(action_dim, dtype=np.int64)
    for dim, dim_values in enumerate(vocab_lists):
        sizes[dim] = int(dim_values.size)
        values[dim, : dim_values.size] = dim_values
    return {"values": values, "sizes": sizes, "max_bins": int(max_bins)}


def build_coupled_value_vocab(
    true_value: np.ndarray,
    groups: list[list[int]] | None = None,
    eps: float = 1e-9,
) -> dict[str, np.ndarray | int | list[list[int]] | str]:
    true_value = np.asarray(true_value, dtype=np.float32)
    if true_value.ndim != 4:
        raise ValueError("true_value must have shape [sample, horizon, edge, action_dim]")
    action_dim = int(true_value.shape[-1])
    groups = default_coupled_value_groups(action_dim) if groups is None else [[int(dim) for dim in group] for group in groups]
    for group in groups:
        if not group:
            raise ValueError("coupled value groups must not be empty")
        if min(group) < 0 or max(group) >= action_dim:
            raise ValueError("coupled value group dimension out of range")

    group_values: list[np.ndarray] = []
    max_tokens = 1
    flat = np.round(true_value.reshape(-1, action_dim).astype(np.float32), 6)
    for group in groups:
        active = np.any(flat[:, group] > eps, axis=-1)
        if np.any(active):
            full_values = np.zeros((int(active.sum()), action_dim), dtype=np.float32)
            full_values[:, group] = flat[active][:, group]
            positive_values = np.unique(full_values, axis=0).astype(np.float32)
        else:
            positive_values = np.zeros((0, action_dim), dtype=np.float32)
        zero = np.zeros((1, action_dim), dtype=np.float32)
        values = np.concatenate([zero, positive_values], axis=0)
        group_values.append(values)
        max_tokens = max(max_tokens, int(values.shape[0]))

    values = np.zeros((len(groups), max_tokens, action_dim), dtype=np.float32)
    sizes = np.zeros(len(groups), dtype=np.int64)
    for group_idx, group_vocab in enumerate(group_values):
        sizes[group_idx] = int(group_vocab.shape[0])
        values[group_idx, : group_vocab.shape[0]] = group_vocab
    return {
        "mode": "coupled_tokens",
        "groups": groups,
        "values": values,
        "sizes": sizes,
        "max_tokens": int(max_tokens),
        "action_dim": int(action_dim),
    }


def _group_count_total_dims(group: list[int]) -> tuple[int, int]:
    count_dim = int(group[0])
    total_dim = int(group[1] if len(group) > 1 else group[0])
    return count_dim, total_dim


def build_hierarchical_value_vocab(
    true_value: np.ndarray,
    groups: list[list[int]] | None = None,
    eps: float = 1e-9,
) -> dict[str, np.ndarray | int | list[list[int]] | str]:
    true_value = np.asarray(true_value, dtype=np.float32)
    if true_value.ndim != 4:
        raise ValueError("true_value must have shape [sample, horizon, edge, action_dim]")
    action_dim = int(true_value.shape[-1])
    groups = default_coupled_value_groups(action_dim) if groups is None else [[int(dim) for dim in group] for group in groups]
    for group in groups:
        if not group:
            raise ValueError("hierarchical value groups must not be empty")
        if min(group) < 0 or max(group) >= action_dim:
            raise ValueError("hierarchical value group dimension out of range")

    flat = np.round(true_value.reshape(-1, action_dim).astype(np.float32), 6)
    count_lists: list[np.ndarray] = []
    total_lists: list[np.ndarray] = []
    max_count_tokens = 1
    max_total_tokens = 1
    for group in groups:
        count_dim, total_dim = _group_count_total_dims(group)
        active = np.any(flat[:, group] > eps, axis=-1)
        count_positive = np.unique(flat[active, count_dim]).astype(np.float32) if np.any(active) else np.zeros(0, dtype=np.float32)
        total_positive = np.unique(flat[active, total_dim]).astype(np.float32) if np.any(active) else np.zeros(0, dtype=np.float32)
        count_values = np.concatenate([np.zeros(1, dtype=np.float32), count_positive[count_positive > eps]])
        total_values = np.concatenate([np.zeros(1, dtype=np.float32), total_positive[total_positive > eps]])
        count_lists.append(count_values)
        total_lists.append(total_values)
        max_count_tokens = max(max_count_tokens, int(count_values.shape[0]))
        max_total_tokens = max(max_total_tokens, int(total_values.shape[0]))

    count_values_arr = np.zeros((len(groups), max_count_tokens), dtype=np.float32)
    total_values_arr = np.zeros((len(groups), max_total_tokens), dtype=np.float32)
    count_sizes = np.zeros(len(groups), dtype=np.int64)
    total_sizes = np.zeros(len(groups), dtype=np.int64)
    for group_idx, values in enumerate(count_lists):
        count_sizes[group_idx] = int(values.shape[0])
        count_values_arr[group_idx, : values.shape[0]] = values
    for group_idx, values in enumerate(total_lists):
        total_sizes[group_idx] = int(values.shape[0])
        total_values_arr[group_idx, : values.shape[0]] = values
    return {
        "mode": "hierarchical_tokens",
        "groups": groups,
        "count_values": count_values_arr,
        "total_values": total_values_arr,
        "count_sizes": count_sizes,
        "total_sizes": total_sizes,
        "max_count_tokens": int(max_count_tokens),
        "max_total_tokens": int(max_total_tokens),
        "action_dim": int(action_dim),
    }


def serialize_value_vocab(vocab: dict) -> dict:
    if vocab.get("mode") == "hierarchical_tokens":
        return {
            "mode": "hierarchical_tokens",
            "groups": [[int(dim) for dim in group] for group in vocab["groups"]],
            "count_values": np.asarray(vocab["count_values"]).tolist(),
            "total_values": np.asarray(vocab["total_values"]).tolist(),
            "count_sizes": np.asarray(vocab["count_sizes"]).tolist(),
            "total_sizes": np.asarray(vocab["total_sizes"]).tolist(),
            "max_count_tokens": int(vocab["max_count_tokens"]),
            "max_total_tokens": int(vocab["max_total_tokens"]),
            "action_dim": int(vocab["action_dim"]),
        }
    payload = {
        "values": np.asarray(vocab["values"]).tolist(),
        "sizes": np.asarray(vocab["sizes"]).tolist(),
    }
    if vocab.get("mode") == "coupled_tokens":
        payload.update(
            {
                "mode": "coupled_tokens",
                "groups": [[int(dim) for dim in group] for group in vocab["groups"]],
                "max_tokens": int(vocab["max_tokens"]),
                "action_dim": int(vocab["action_dim"]),
            }
        )
    else:
        payload["max_bins"] = int(vocab["max_bins"])
    return payload


def encode_value_bins(raw_action: torch.Tensor, vocab: dict[str, np.ndarray | int], eps: float = 1e-9) -> torch.Tensor:
    if raw_action.ndim < 2:
        raise ValueError("raw_action must end with action_dim")
    values = np.asarray(vocab["values"], dtype=np.float32)
    sizes = np.asarray(vocab["sizes"], dtype=np.int64)
    if raw_action.shape[-1] != values.shape[0]:
        raise ValueError("raw_action action_dim does not match vocab")
    target = torch.full(raw_action.shape, IGNORE_INDEX, dtype=torch.long, device=raw_action.device)
    raw_cpu = raw_action.detach().cpu().numpy()
    for dim in range(raw_action.shape[-1]):
        dim_vocab = values[dim, : sizes[dim]]
        dim_raw = raw_cpu[..., dim]
        active = dim_raw > eps
        if not np.any(active):
            continue
        distances = np.abs(dim_raw[..., None] - dim_vocab.reshape((1,) * dim_raw.ndim + (-1,)))
        nearest = np.argmin(distances, axis=-1).astype(np.int64)
        dim_target = target[..., dim]
        dim_target[torch.as_tensor(active, dtype=torch.bool, device=raw_action.device)] = torch.as_tensor(
            nearest[active],
            dtype=torch.long,
            device=raw_action.device,
        )
    return target


def encode_coupled_value_tokens(
    raw_action: torch.Tensor,
    vocab: dict[str, np.ndarray | int | list[list[int]] | str],
    eps: float = 1e-9,
) -> torch.Tensor:
    if raw_action.ndim < 2:
        raise ValueError("raw_action must end with action_dim")
    values = np.asarray(vocab["values"], dtype=np.float32)
    sizes = np.asarray(vocab["sizes"], dtype=np.int64)
    groups = [[int(dim) for dim in group] for group in vocab["groups"]]
    action_dim = int(vocab.get("action_dim", values.shape[-1]))
    if raw_action.shape[-1] != action_dim:
        raise ValueError("raw_action action_dim does not match coupled vocab")
    target_shape = (*raw_action.shape[:-1], len(groups))
    target = torch.full(target_shape, IGNORE_INDEX, dtype=torch.long, device=raw_action.device)
    raw_cpu = raw_action.detach().cpu().numpy()
    for group_idx, group in enumerate(groups):
        group_vocab = values[group_idx, : sizes[group_idx], :][:, group]
        group_raw = raw_cpu[..., group]
        active = np.any(group_raw > eps, axis=-1)
        if not np.any(active):
            continue
        distances = np.abs(group_raw[..., None, :] - group_vocab.reshape((1,) * active.ndim + group_vocab.shape)).sum(axis=-1)
        nearest = np.argmin(distances, axis=-1).astype(np.int64)
        group_target = target[..., group_idx]
        group_target[torch.as_tensor(active, dtype=torch.bool, device=raw_action.device)] = torch.as_tensor(
            nearest[active],
            dtype=torch.long,
            device=raw_action.device,
        )
    return target


def encode_hierarchical_value_tokens(
    raw_action: torch.Tensor,
    vocab: dict[str, np.ndarray | int | list[list[int]] | str],
    eps: float = 1e-9,
) -> dict[str, torch.Tensor]:
    if raw_action.ndim < 2:
        raise ValueError("raw_action must end with action_dim")
    count_values = np.asarray(vocab["count_values"], dtype=np.float32)
    total_values = np.asarray(vocab["total_values"], dtype=np.float32)
    count_sizes = np.asarray(vocab["count_sizes"], dtype=np.int64)
    total_sizes = np.asarray(vocab["total_sizes"], dtype=np.int64)
    groups = [[int(dim) for dim in group] for group in vocab["groups"]]
    action_dim = int(vocab.get("action_dim", raw_action.shape[-1]))
    if raw_action.shape[-1] != action_dim:
        raise ValueError("raw_action action_dim does not match hierarchical vocab")
    target_shape = (*raw_action.shape[:-1], len(groups))
    count_target = torch.full(target_shape, IGNORE_INDEX, dtype=torch.long, device=raw_action.device)
    total_target = torch.full(target_shape, IGNORE_INDEX, dtype=torch.long, device=raw_action.device)
    raw_cpu = raw_action.detach().cpu().numpy()
    for group_idx, group in enumerate(groups):
        count_dim, total_dim = _group_count_total_dims(group)
        group_raw = raw_cpu[..., group]
        active = np.any(group_raw > eps, axis=-1)
        if not np.any(active):
            continue
        count_vocab = count_values[group_idx, : count_sizes[group_idx]]
        total_vocab = total_values[group_idx, : total_sizes[group_idx]]
        count_raw = raw_cpu[..., count_dim]
        total_raw = raw_cpu[..., total_dim]
        count_nearest = np.argmin(np.abs(count_raw[..., None] - count_vocab.reshape((1,) * count_raw.ndim + (-1,))), axis=-1)
        total_nearest = np.argmin(np.abs(total_raw[..., None] - total_vocab.reshape((1,) * total_raw.ndim + (-1,))), axis=-1)
        active_t = torch.as_tensor(active, dtype=torch.bool, device=raw_action.device)
        count_group_target = count_target[..., group_idx]
        total_group_target = total_target[..., group_idx]
        count_group_target[active_t] = torch.as_tensor(count_nearest[active], dtype=torch.long, device=raw_action.device)
        total_group_target[active_t] = torch.as_tensor(total_nearest[active], dtype=torch.long, device=raw_action.device)
    return {"count": count_target, "total": total_target}


def decode_value_bins(logits: torch.Tensor, vocab: dict[str, np.ndarray | int]) -> torch.Tensor:
    if logits.ndim != 5:
        raise ValueError("logits must have shape [batch, horizon, edge, action_dim, max_bins]")
    values = torch.as_tensor(np.asarray(vocab["values"], dtype=np.float32), dtype=logits.dtype, device=logits.device)
    sizes = torch.as_tensor(np.asarray(vocab["sizes"], dtype=np.int64), dtype=torch.long, device=logits.device)
    if logits.shape[-2] != values.shape[0]:
        raise ValueError("logits action_dim does not match vocab")
    if logits.shape[-1] != values.shape[1]:
        raise ValueError("logits max_bins does not match vocab")
    bin_ids = torch.arange(logits.shape[-1], device=logits.device).reshape(1, 1, 1, 1, -1)
    valid = bin_ids < sizes.reshape(1, 1, 1, -1, 1)
    masked_logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
    selected = torch.argmax(masked_logits, dim=-1)
    expanded_values = values.reshape(1, 1, 1, values.shape[0], values.shape[1]).expand(*selected.shape, values.shape[1])
    return torch.gather(expanded_values, dim=-1, index=selected.unsqueeze(-1)).squeeze(-1)


def decode_coupled_value_tokens(
    logits: torch.Tensor,
    vocab: dict[str, np.ndarray | int | list[list[int]] | str],
) -> torch.Tensor:
    if logits.ndim != 5:
        raise ValueError("logits must have shape [batch, horizon, edge, group, max_tokens]")
    values = torch.as_tensor(np.asarray(vocab["values"], dtype=np.float32), dtype=logits.dtype, device=logits.device)
    sizes = torch.as_tensor(np.asarray(vocab["sizes"], dtype=np.int64), dtype=torch.long, device=logits.device)
    groups = [[int(dim) for dim in group] for group in vocab["groups"]]
    action_dim = int(vocab.get("action_dim", values.shape[-1]))
    if logits.shape[-2] != values.shape[0]:
        raise ValueError("logits group count does not match coupled vocab")
    if logits.shape[-1] != values.shape[1]:
        raise ValueError("logits max_tokens does not match coupled vocab")
    token_ids = torch.arange(logits.shape[-1], device=logits.device).reshape(1, 1, 1, 1, -1)
    valid = token_ids < sizes.reshape(1, 1, 1, -1, 1)
    masked_logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
    selected = torch.argmax(masked_logits, dim=-1)
    decoded = torch.zeros((*selected.shape[:-1], action_dim), dtype=logits.dtype, device=logits.device)
    for group_idx, group in enumerate(groups):
        group_values = values[group_idx]
        selected_values = group_values[selected[..., group_idx]]
        decoded[..., group] = selected_values[..., group]
    return decoded


def _select_valid_tokens(logits: torch.Tensor, sizes: torch.Tensor) -> torch.Tensor:
    token_ids = torch.arange(logits.shape[-1], device=logits.device).reshape(1, 1, 1, 1, -1)
    valid = token_ids < sizes.reshape(1, 1, 1, -1, 1)
    masked_logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
    return torch.argmax(masked_logits, dim=-1)


def decode_hierarchical_value_tokens(
    count_logits: torch.Tensor,
    total_logits: torch.Tensor,
    vocab: dict[str, np.ndarray | int | list[list[int]] | str],
) -> torch.Tensor:
    if count_logits.ndim != 5 or total_logits.ndim != 5:
        raise ValueError("hierarchical logits must have shape [batch, horizon, edge, group, tokens]")
    count_values = torch.as_tensor(np.asarray(vocab["count_values"], dtype=np.float32), dtype=count_logits.dtype, device=count_logits.device)
    total_values = torch.as_tensor(np.asarray(vocab["total_values"], dtype=np.float32), dtype=total_logits.dtype, device=total_logits.device)
    count_sizes = torch.as_tensor(np.asarray(vocab["count_sizes"], dtype=np.int64), dtype=torch.long, device=count_logits.device)
    total_sizes = torch.as_tensor(np.asarray(vocab["total_sizes"], dtype=np.int64), dtype=torch.long, device=total_logits.device)
    groups = [[int(dim) for dim in group] for group in vocab["groups"]]
    action_dim = int(vocab.get("action_dim", max(max(group) for group in groups) + 1))
    if count_logits.shape[-2] != len(groups) or total_logits.shape[-2] != len(groups):
        raise ValueError("hierarchical logits group count does not match vocab")
    if count_logits.shape[-1] != count_values.shape[1]:
        raise ValueError("count logits token count does not match vocab")
    if total_logits.shape[-1] != total_values.shape[1]:
        raise ValueError("total logits token count does not match vocab")
    count_selected = _select_valid_tokens(count_logits, count_sizes)
    total_selected = _select_valid_tokens(total_logits, total_sizes)
    decoded = torch.zeros((*count_selected.shape[:-1], action_dim), dtype=count_logits.dtype, device=count_logits.device)
    for group_idx, group in enumerate(groups):
        count_dim, total_dim = _group_count_total_dims(group)
        selected_count_values = count_values[group_idx][count_selected[..., group_idx]]
        selected_total_values = total_values[group_idx][total_selected[..., group_idx]]
        decoded[..., count_dim] = selected_count_values
        decoded[..., total_dim] = selected_total_values
    return decoded


def make_value_bin_class_weights(
    true_value: np.ndarray,
    vocab: dict[str, np.ndarray | int],
    mode: str,
    beta: float = 0.999,
) -> np.ndarray | None:
    if mode == "inverse_sqrt":
        mode = "inverse_sqrt_ce"
    if mode in {"ce", "focal"}:
        return None
    values = np.asarray(vocab["values"], dtype=np.float32)
    sizes = np.asarray(vocab["sizes"], dtype=np.int64)
    weights = np.ones_like(values, dtype=np.float32)
    for dim in range(values.shape[0]):
        dim_values = values[dim, : sizes[dim]]
        positives = np.round(true_value[..., dim][true_value[..., dim] > 1e-9].astype(np.float32), 6)
        counts = np.array([max(int(np.sum(positives == value)), 1) for value in dim_values], dtype=np.float32)
        if mode in {"inverse_sqrt_ce", "inverse_sqrt_focal"}:
            dim_weights = 1.0 / np.sqrt(counts)
        elif mode == "effective_ce":
            if not 0.0 < beta < 1.0:
                raise ValueError("effective beta must be between 0 and 1")
            effective_num = 1.0 - np.power(float(beta), counts)
            dim_weights = (1.0 - float(beta)) / np.maximum(effective_num, 1e-12)
        else:
            raise ValueError(f"Unknown value bin weight mode: {mode}")
        dim_weights = dim_weights / np.mean(dim_weights)
        weights[dim, : sizes[dim]] = dim_weights.astype(np.float32)
    return weights


def make_hierarchical_total_class_weights(
    true_value: np.ndarray,
    vocab: dict[str, np.ndarray | int | list[list[int]] | str],
    mode: str,
    beta: float = 0.999,
) -> np.ndarray | None:
    if mode == "inverse_sqrt":
        mode = "inverse_sqrt_ce"
    if mode in {"ce", "focal"}:
        return None
    true_value = np.asarray(true_value, dtype=np.float32)
    total_values = np.asarray(vocab["total_values"], dtype=np.float32)
    total_sizes = np.asarray(vocab["total_sizes"], dtype=np.int64)
    groups = [[int(dim) for dim in group] for group in vocab["groups"]]
    flat = np.round(true_value.reshape(-1, true_value.shape[-1]).astype(np.float32), 6)
    weights = np.ones_like(total_values, dtype=np.float32)
    for group_idx, group in enumerate(groups):
        _, total_dim = _group_count_total_dims(group)
        active = np.any(flat[:, group] > 1e-9, axis=-1)
        positives = flat[active, total_dim]
        values = total_values[group_idx, : total_sizes[group_idx]]
        counts = np.array([max(int(np.sum(positives == value)), 1) for value in values], dtype=np.float32)
        if mode in {"inverse_sqrt_ce", "inverse_sqrt_focal"}:
            group_weights = 1.0 / np.sqrt(counts)
        elif mode == "effective_ce":
            if not 0.0 < beta < 1.0:
                raise ValueError("effective beta must be between 0 and 1")
            effective_num = 1.0 - np.power(float(beta), counts)
            group_weights = (1.0 - float(beta)) / np.maximum(effective_num, 1e-12)
        else:
            raise ValueError(f"Unknown hierarchical total weight mode: {mode}")
        group_weights = group_weights / np.mean(group_weights)
        weights[group_idx, : total_sizes[group_idx]] = group_weights.astype(np.float32)
    return weights


def compute_discrete_policy_loss(
    outputs: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    pos_weight: torch.Tensor,
    vocab: dict[str, np.ndarray | int],
    bin_loss_weight: float,
    value_bin_loss: str = "ce",
    class_weights: np.ndarray | None = None,
    focal_gamma: float = 1.5,
    activity_value_weight: float = 0.0,
    max_activity_value_weight: float = 20.0,
    edge_activity_loss_weight: float = 0.0,
    edge_pos_weight: torch.Tensor | None = None,
    active_edge_sample_weight: float = 1.0,
    hard_negative_edge_weight: float = 1.0,
    hard_negative_edge_threshold: float = 0.5,
    edge_tversky_loss_weight: float = 0.0,
    edge_tversky_alpha: float = 0.7,
    edge_tversky_beta: float = 0.3,
    step_total_loss_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    active = target["action_active"]
    raw_activity_loss = nn.functional.binary_cross_entropy_with_logits(
        outputs["action_logit"],
        active,
        pos_weight=pos_weight.to(outputs["action_logit"].device),
        reduction="none",
    )
    activity_weight = make_activity_value_weight(
        active,
        target["action_raw"].to(outputs["action_logit"].device),
        activity_value_weight=activity_value_weight,
        max_activity_value_weight=max_activity_value_weight,
    )
    activity_loss = (raw_activity_loss * activity_weight).mean()
    if float(edge_activity_loss_weight) > 0.0:
        if "edge_logit" not in outputs:
            raise ValueError("edge_activity_loss_weight requires outputs['edge_logit']")
        edge_pos = torch.ones(1, device=outputs["edge_logit"].device) if edge_pos_weight is None else edge_pos_weight.to(outputs["edge_logit"].device)
        edge_target = target["edge_active"].to(outputs["edge_logit"].device)
        raw_edge_activity_loss = nn.functional.binary_cross_entropy_with_logits(
            outputs["edge_logit"],
            edge_target,
            pos_weight=edge_pos,
            reduction="none",
        )
        edge_weight = make_edge_activity_weight(
            outputs["edge_logit"],
            edge_target,
            active_edge_sample_weight=active_edge_sample_weight,
            hard_negative_edge_weight=hard_negative_edge_weight,
            hard_negative_edge_threshold=hard_negative_edge_threshold,
        )
        edge_activity_loss = (raw_edge_activity_loss * edge_weight).mean()
    else:
        edge_activity_loss = raw_activity_loss.new_tensor(0.0)
        edge_target = target["edge_active"].to(outputs["action_logit"].device)
    if float(edge_tversky_loss_weight) > 0.0:
        if "edge_logit" not in outputs:
            raise ValueError("edge_tversky_loss_weight requires outputs['edge_logit']")
        edge_tversky = edge_tversky_loss(
            outputs["edge_logit"],
            edge_target,
            alpha=edge_tversky_alpha,
            beta=edge_tversky_beta,
        )
    else:
        edge_tversky = raw_activity_loss.new_tensor(0.0)
    if float(step_total_loss_weight) > 0.0:
        if "step_total_log" not in outputs:
            raise ValueError("step_total_loss_weight requires outputs['step_total_log']")
        step_total_loss = nn.functional.mse_loss(
            outputs["step_total_log"],
            target["step_total_log"].to(outputs["step_total_log"].device),
        )
    else:
        step_total_loss = raw_activity_loss.new_tensor(0.0)
    if "action_value_count_logit" in outputs:
        token_target = {
            "count": target["action_value_count_token"].to(outputs["action_value_count_logit"].device),
            "total": target["action_value_total_token"].to(outputs["action_value_total_logit"].device),
        }
        bin_loss = masked_hierarchical_token_cross_entropy(
            outputs["action_value_count_logit"],
            outputs["action_value_total_logit"],
            token_target,
            vocab,
            total_class_weights=class_weights,
            value_bin_loss=value_bin_loss,
            focal_gamma=focal_gamma,
        )
    elif "action_value_token_logit" in outputs:
        token_target = target["action_value_token"].to(outputs["action_value_token_logit"].device)
        bin_loss = masked_token_cross_entropy(outputs["action_value_token_logit"], token_target, vocab)
    else:
        bin_target = target["action_value_bin"].to(outputs["action_value_bin_logit"].device)
        bin_loss = masked_bin_cross_entropy(
            outputs["action_value_bin_logit"],
            bin_target,
            vocab,
            value_bin_loss=value_bin_loss,
            class_weights=class_weights,
            focal_gamma=focal_gamma,
        )
    total = (
        activity_loss
        + float(edge_activity_loss_weight) * edge_activity_loss
        + float(edge_tversky_loss_weight) * edge_tversky
        + float(step_total_loss_weight) * step_total_loss
        + float(bin_loss_weight) * bin_loss
    )
    return total, {
        "total": float(total.detach().cpu()),
        "activity": float(activity_loss.detach().cpu()),
        "edge_activity": float(edge_activity_loss.detach().cpu()),
        "edge_tversky": float(edge_tversky.detach().cpu()),
        "step_total": float(step_total_loss.detach().cpu()),
        "value_bin": float(bin_loss.detach().cpu()),
    }


def edge_tversky_loss(
    edge_logit: torch.Tensor,
    edge_active: torch.Tensor,
    alpha: float = 0.7,
    beta: float = 0.3,
    eps: float = 1e-6,
) -> torch.Tensor:
    target = edge_active.to(device=edge_logit.device, dtype=torch.float32)
    prob = torch.sigmoid(edge_logit)
    fp_weight = max(float(alpha), 0.0)
    fn_weight = max(float(beta), 0.0)
    dims = tuple(range(1, prob.ndim))
    true_positive = (prob * target).sum(dim=dims)
    false_positive = (prob * (1.0 - target)).sum(dim=dims)
    false_negative = ((1.0 - prob) * target).sum(dim=dims)
    index = (true_positive + eps) / (
        true_positive + fp_weight * false_positive + fn_weight * false_negative + eps
    )
    return (1.0 - index).mean()


def make_edge_activity_weight(
    edge_logit: torch.Tensor,
    edge_active: torch.Tensor,
    active_edge_sample_weight: float = 1.0,
    hard_negative_edge_weight: float = 1.0,
    hard_negative_edge_threshold: float = 0.5,
) -> torch.Tensor:
    edge_active = edge_active.to(device=edge_logit.device, dtype=torch.float32)
    weight = torch.ones_like(edge_active)
    active_multiplier = max(float(active_edge_sample_weight), 1.0)
    if active_multiplier > 1.0:
        active_rows = edge_active.any(dim=-1, keepdim=True)
        weight = torch.where(active_rows, weight * active_multiplier, weight)
    hard_multiplier = max(float(hard_negative_edge_weight), 1.0)
    if hard_multiplier > 1.0:
        threshold = min(max(float(hard_negative_edge_threshold), 0.0), 1.0)
        hard_negative = (edge_active <= 0.5) & (torch.sigmoid(edge_logit) >= threshold)
        weight = torch.where(hard_negative, weight * hard_multiplier, weight)
    return weight


def make_activity_value_weight(
    active: torch.Tensor,
    raw_action: torch.Tensor,
    activity_value_weight: float,
    max_activity_value_weight: float,
) -> torch.Tensor:
    active = active.to(dtype=torch.float32)
    raw_action = raw_action.to(device=active.device, dtype=torch.float32)
    if float(activity_value_weight) <= 0.0:
        return torch.ones_like(active)
    positive_value = torch.log1p(torch.clamp(raw_action, min=0.0))
    weight = 1.0 + active * float(activity_value_weight) * positive_value
    return torch.clamp(weight, min=1.0, max=max(float(max_activity_value_weight), 1.0))


def masked_bin_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    vocab: dict[str, np.ndarray | int],
    value_bin_loss: str = "ce",
    class_weights: np.ndarray | None = None,
    focal_gamma: float = 1.5,
) -> torch.Tensor:
    sizes = torch.as_tensor(np.asarray(vocab["sizes"], dtype=np.int64), dtype=torch.long, device=logits.device)
    bin_ids = torch.arange(logits.shape[-1], device=logits.device).reshape(1, 1, 1, 1, -1)
    valid = bin_ids < sizes.reshape(1, 1, 1, -1, 1)
    masked_logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
    flat_logits = masked_logits.reshape(-1, masked_logits.shape[-1])
    flat_target = target.reshape(-1)
    active = flat_target != IGNORE_INDEX
    if not torch.any(active):
        return logits.new_tensor(0.0)
    active_logits = flat_logits[active]
    active_target = flat_target[active]
    action_dim = logits.shape[-2]
    active_dim = torch.arange(action_dim, device=logits.device).reshape(1, 1, 1, action_dim).expand(target.shape).reshape(-1)[active]
    ce = nn.functional.cross_entropy(active_logits, active_target, reduction="none")
    if class_weights is not None:
        weights_t = torch.as_tensor(class_weights, dtype=logits.dtype, device=logits.device)
        ce = ce * weights_t[active_dim, active_target]
    if value_bin_loss in {"focal", "inverse_sqrt_focal"}:
        pt = torch.exp(-ce.detach()).clamp(min=1e-6, max=1.0)
        ce = ((1.0 - pt) ** float(focal_gamma)) * ce
    return ce.mean()


def masked_token_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    vocab: dict[str, np.ndarray | int | list[list[int]] | str],
) -> torch.Tensor:
    sizes = torch.as_tensor(np.asarray(vocab["sizes"], dtype=np.int64), dtype=torch.long, device=logits.device)
    token_ids = torch.arange(logits.shape[-1], device=logits.device).reshape(1, 1, 1, 1, -1)
    valid = token_ids < sizes.reshape(1, 1, 1, -1, 1)
    masked_logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
    flat_logits = masked_logits.reshape(-1, masked_logits.shape[-1])
    flat_target = target.reshape(-1)
    active = flat_target != IGNORE_INDEX
    if not torch.any(active):
        return logits.new_tensor(0.0)
    return nn.functional.cross_entropy(flat_logits[active], flat_target[active])


def _masked_group_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    sizes: np.ndarray,
    class_weights: np.ndarray | None = None,
    value_bin_loss: str = "ce",
    focal_gamma: float = 1.5,
) -> torch.Tensor:
    sizes_t = torch.as_tensor(np.asarray(sizes, dtype=np.int64), dtype=torch.long, device=logits.device)
    token_ids = torch.arange(logits.shape[-1], device=logits.device).reshape(1, 1, 1, 1, -1)
    valid = token_ids < sizes_t.reshape(1, 1, 1, -1, 1)
    masked_logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
    flat_logits = masked_logits.reshape(-1, masked_logits.shape[-1])
    flat_target = target.reshape(-1)
    active = flat_target != IGNORE_INDEX
    if not torch.any(active):
        return logits.new_tensor(0.0)
    active_logits = flat_logits[active]
    active_target = flat_target[active]
    ce = nn.functional.cross_entropy(active_logits, active_target, reduction="none")
    if class_weights is not None:
        weights_t = torch.as_tensor(class_weights, dtype=logits.dtype, device=logits.device)
        group_count = logits.shape[-2]
        active_group = torch.arange(group_count, device=logits.device).reshape(1, 1, 1, group_count).expand(target.shape).reshape(-1)[active]
        ce = ce * weights_t[active_group, active_target]
    if value_bin_loss in {"focal", "inverse_sqrt_focal"}:
        pt = torch.exp(-ce.detach()).clamp(min=1e-6, max=1.0)
        ce = ((1.0 - pt) ** float(focal_gamma)) * ce
    return ce.mean()


def masked_hierarchical_token_cross_entropy(
    count_logits: torch.Tensor,
    total_logits: torch.Tensor,
    target: dict[str, torch.Tensor],
    vocab: dict[str, np.ndarray | int | list[list[int]] | str],
    total_class_weights: np.ndarray | None = None,
    value_bin_loss: str = "ce",
    focal_gamma: float = 1.5,
) -> torch.Tensor:
    count_loss = _masked_group_cross_entropy(count_logits, target["count"], np.asarray(vocab["count_sizes"], dtype=np.int64))
    total_loss = _masked_group_cross_entropy(
        total_logits,
        target["total"],
        np.asarray(vocab["total_sizes"], dtype=np.int64),
        class_weights=total_class_weights,
        value_bin_loss=value_bin_loss,
        focal_gamma=focal_gamma,
    )
    return count_loss + total_loss


class V11DiscreteValuePolicyDataset(Dataset):
    def __init__(self, arrays: dict[str, np.ndarray], indices: np.ndarray, stats: dict, vocab: dict[str, np.ndarray | int]):
        self.arrays = arrays
        self.indices = np.asarray(indices, dtype=np.int64)
        self.base = V6WorldModelDataset(arrays, self.indices, stats)
        self.vocab = vocab

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        batch, _ = self.base[item]
        source_idx = int(self.indices[item])
        raw_action = torch.from_numpy(self.arrays["edge_a_future"][source_idx].astype(np.float32))
        active = (raw_action > 1e-9).to(torch.float32)
        edge_active = (active.any(dim=-1)).to(torch.float32)
        rb_total = raw_action[..., 2].sum(dim=-1) if raw_action.shape[-1] > 2 else torch.zeros_like(edge_active.sum(dim=-1))
        cpu_total = raw_action[..., 4].sum(dim=-1) if raw_action.shape[-1] > 4 else torch.zeros_like(edge_active.sum(dim=-1))
        step_total = torch.stack([edge_active.sum(dim=-1), rb_total, cpu_total], dim=-1)
        step_total_log = torch.log1p(torch.clamp(step_total, min=0.0))
        if self.vocab.get("mode") == "coupled_tokens":
            action_value_bin = torch.empty(raw_action.shape, dtype=torch.long)
            action_value_token = encode_coupled_value_tokens(raw_action, self.vocab)
            action_value_count_token = torch.empty((*raw_action.shape[:-1], 0), dtype=torch.long)
            action_value_total_token = torch.empty((*raw_action.shape[:-1], 0), dtype=torch.long)
        elif self.vocab.get("mode") == "hierarchical_tokens":
            action_value_bin = torch.empty(raw_action.shape, dtype=torch.long)
            action_value_token = torch.empty((*raw_action.shape[:-1], 0), dtype=torch.long)
            hierarchical_tokens = encode_hierarchical_value_tokens(raw_action, self.vocab)
            action_value_count_token = hierarchical_tokens["count"]
            action_value_total_token = hierarchical_tokens["total"]
        else:
            action_value_bin = encode_value_bins(raw_action, self.vocab)
            action_value_token = torch.empty((*raw_action.shape[:-1], 0), dtype=torch.long)
            action_value_count_token = torch.empty((*raw_action.shape[:-1], 0), dtype=torch.long)
            action_value_total_token = torch.empty((*raw_action.shape[:-1], 0), dtype=torch.long)
        target = {
            "action_active": active,
            "edge_active": edge_active,
            "action_value_bin": action_value_bin,
            "action_value_token": action_value_token,
            "action_value_count_token": action_value_count_token,
            "action_value_total_token": action_value_total_token,
            "action_raw": raw_action,
            "step_total_log": step_total_log,
        }
        return batch, target


def collate_discrete_policy_batch(items):
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
        "action_value_bin": torch.stack([item["action_value_bin"] for item in targets]),
        "action_value_token": torch.stack([item["action_value_token"] for item in targets]),
        "action_value_count_token": torch.stack([item["action_value_count_token"] for item in targets]),
        "action_value_total_token": torch.stack([item["action_value_total_token"] for item in targets]),
        "action_raw": torch.stack([item["action_raw"] for item in targets]),
        "step_total_log": torch.stack([item["step_total_log"] for item in targets]),
    }
    return batch, target


def move_target_to_device(target: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: value.to(device) for name, value in target.items()}


def make_config(
    arrays: dict[str, np.ndarray],
    args: argparse.Namespace,
    max_bins: int,
    value_token_group_count: int = 0,
    max_count_tokens: int | None = None,
    max_total_tokens: int | None = None,
) -> V7ActionPolicyConfig:
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
        value_mode=args.value_mode,
        max_value_bins=int(max_bins),
        value_token_group_count=int(value_token_group_count),
        max_value_tokens=int(max_bins),
        max_value_count_tokens=int(max_bins if max_count_tokens is None else max_count_tokens),
        max_value_total_tokens=int(max_bins if max_total_tokens is None else max_total_tokens),
        use_edge_activity_head=bool(getattr(args, "use_edge_activity_head", False)),
        use_step_total_head=bool(getattr(args, "use_step_total_head", False)),
    )


def take_positive_aware_subset_indices(arrays: dict[str, np.ndarray], indices: np.ndarray, max_samples: int) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    if max_samples <= 0 or max_samples >= len(indices):
        return indices
    has_positive = (arrays["edge_a_future"][indices] > 1e-9).any(axis=(1, 2, 3))
    positives = indices[has_positive]
    negatives = indices[~has_positive]
    if len(positives) >= max_samples:
        return positives[:max_samples]
    need = max_samples - len(positives)
    return np.concatenate([positives, negatives[:need]])


def collect_predictions(model, loader, device, vocab: dict[str, np.ndarray | int]) -> dict[str, np.ndarray]:
    model.eval()
    rows = {"prob": [], "active": [], "value_pred": [], "value_true": [], "bin_pred": [], "bin_true": []}
    with torch.no_grad():
        for batch, target in loader:
            batch = move_batch_to_device(batch, device)
            target = move_target_to_device(target, device)
            outputs = model(batch)
            if "edge_logit" in outputs:
                rows.setdefault("edge_prob", []).append(torch.sigmoid(outputs["edge_logit"]).cpu().numpy())
            if "action_value_count_logit" in outputs:
                value_pred = decode_hierarchical_value_tokens(
                    outputs["action_value_count_logit"],
                    outputs["action_value_total_logit"],
                    vocab,
                )
                count_pred = torch.argmax(outputs["action_value_count_logit"], dim=-1)
                total_pred = torch.argmax(outputs["action_value_total_logit"], dim=-1)
                rows.setdefault("count_token_pred", []).append(count_pred.cpu().numpy())
                rows.setdefault("count_token_true", []).append(target["action_value_count_token"].cpu().numpy())
                rows.setdefault("total_token_pred", []).append(total_pred.cpu().numpy())
                rows.setdefault("total_token_true", []).append(target["action_value_total_token"].cpu().numpy())
                bin_pred = torch.empty_like(target["action_value_bin"])
            elif "action_value_token_logit" in outputs:
                value_pred = decode_coupled_value_tokens(outputs["action_value_token_logit"], vocab)
                token_pred = torch.argmax(outputs["action_value_token_logit"], dim=-1)
                rows.setdefault("token_pred", []).append(token_pred.cpu().numpy())
                rows.setdefault("token_true", []).append(target["action_value_token"].cpu().numpy())
                bin_pred = torch.empty_like(target["action_value_bin"])
            else:
                value_pred = decode_value_bins(outputs["action_value_bin_logit"], vocab)
                bin_pred = torch.argmax(outputs["action_value_bin_logit"], dim=-1)
            rows["prob"].append(torch.sigmoid(outputs["action_logit"]).cpu().numpy())
            rows["active"].append(target["action_active"].cpu().numpy())
            rows["value_pred"].append(value_pred.cpu().numpy())
            rows["value_true"].append(target["action_raw"].cpu().numpy())
            rows["bin_pred"].append(bin_pred.cpu().numpy())
            rows["bin_true"].append(target["action_value_bin"].cpu().numpy())
    return {name: np.concatenate(values, axis=0) for name, values in rows.items()}


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
        if "count_token_pred" in predictions:
            count_true = predictions["count_token_true"]
            total_true = predictions["total_token_true"]
            count_active = count_true != IGNORE_INDEX
            total_active = total_true != IGNORE_INDEX
            count_acc = (
                float(np.mean(predictions["count_token_pred"][count_active] == count_true[count_active]))
                if count_active.any()
                else float("nan")
            )
            total_acc = (
                float(np.mean(predictions["total_token_pred"][total_active] == total_true[total_active]))
                if total_active.any()
                else float("nan")
            )
            bin_acc = float(np.nanmean([count_acc, total_acc]))
        elif "token_pred" in predictions:
            token_true = predictions["token_true"]
            token_active = token_true != IGNORE_INDEX
            bin_acc = float(np.mean(predictions["token_pred"][token_active] == token_true[token_active])) if token_active.any() else float("nan")
        else:
            bin_acc = float(np.mean(predictions["bin_pred"][active] == predictions["bin_true"][active]))
    else:
        active_value_rmse = float("nan")
        bin_acc = float("nan")
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
        "active_value_bin_accuracy": bin_acc,
    }


def evaluate_loss(model, loader, device, pos_weight, edge_pos_weight, args, vocab) -> dict[str, float]:
    model.eval()
    rows = []
    with torch.no_grad():
        for batch, target in loader:
            batch = move_batch_to_device(batch, device)
            target = move_target_to_device(target, device)
            outputs = model(batch)
            _, parts = compute_discrete_policy_loss(
                outputs,
                target,
                pos_weight,
                vocab,
                args.bin_loss_weight,
                value_bin_loss=args.value_bin_loss,
                class_weights=getattr(args, "value_bin_class_weights", None),
                focal_gamma=args.focal_gamma,
                activity_value_weight=args.activity_value_weight,
                max_activity_value_weight=args.max_activity_value_weight,
                edge_activity_loss_weight=args.edge_activity_loss_weight,
                edge_pos_weight=edge_pos_weight,
                active_edge_sample_weight=args.active_edge_sample_weight,
                hard_negative_edge_weight=args.hard_negative_edge_weight,
                hard_negative_edge_threshold=args.hard_negative_edge_threshold,
                edge_tversky_loss_weight=args.edge_tversky_loss_weight,
                edge_tversky_alpha=args.edge_tversky_alpha,
                edge_tversky_beta=args.edge_tversky_beta,
                step_total_loss_weight=args.step_total_loss_weight,
            )
            rows.append(parts)
    predictions = collect_predictions(model, loader, device, vocab)
    threshold = choose_threshold(predictions["prob"], predictions["active"])
    return {**mean_rows(rows), **evaluate_predictions(predictions, threshold)}


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
    if args.value_mode == "coupled_tokens":
        vocab = build_coupled_value_vocab(arrays["edge_a_future"][train_idx])
        value_bin_class_weights = None
    elif args.value_mode == "hierarchical_tokens":
        vocab = build_hierarchical_value_vocab(arrays["edge_a_future"][train_idx])
        value_bin_class_weights = make_hierarchical_total_class_weights(
            arrays["edge_a_future"][train_idx],
            vocab,
            mode=args.value_bin_loss,
            beta=args.effective_beta,
        )
    else:
        vocab = build_value_vocab(arrays["edge_a_future"][train_idx])
        value_bin_class_weights = make_value_bin_class_weights(
            arrays["edge_a_future"][train_idx],
            vocab,
            mode=args.value_bin_loss,
            beta=args.effective_beta,
        )
    args.value_bin_class_weights = value_bin_class_weights
    pos_weight = make_pos_weight(arrays, train_idx, args.max_pos_weight)
    edge_pos_weight = make_edge_pos_weight(arrays, train_idx, args.max_edge_pos_weight)

    train_ds = V11DiscreteValuePolicyDataset(arrays, train_idx, stats, vocab)
    val_ds = V11DiscreteValuePolicyDataset(arrays, val_idx, stats, vocab)
    test_ds = V11DiscreteValuePolicyDataset(arrays, test_idx, stats, vocab)
    train_used_idx = take_positive_aware_subset_indices(arrays, train_idx, args.max_train_samples)
    val_used_idx = take_positive_aware_subset_indices(arrays, val_idx, args.max_val_samples)
    test_used_idx = take_positive_aware_subset_indices(arrays, test_idx, args.max_test_samples)
    train_ds = V11DiscreteValuePolicyDataset(arrays, train_used_idx, stats, vocab)
    val_ds = V11DiscreteValuePolicyDataset(arrays, val_used_idx, stats, vocab)
    test_ds = V11DiscreteValuePolicyDataset(arrays, test_used_idx, stats, vocab)
    train_subset = Subset(train_ds, range(len(train_ds)))
    val_subset = Subset(val_ds, range(len(val_ds)))
    test_subset = Subset(test_ds, range(len(test_ds)))

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collate_discrete_policy_batch,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_discrete_policy_batch,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_discrete_policy_batch,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )

    if args.value_mode == "coupled_tokens":
        max_units = int(vocab["max_tokens"])
        group_count = int(len(vocab["groups"]))
        config = make_config(arrays, args, max_units, group_count)
    elif args.value_mode == "hierarchical_tokens":
        group_count = int(len(vocab["groups"]))
        config = make_config(
            arrays,
            args,
            max_bins=int(max(int(vocab["max_count_tokens"]), int(vocab["max_total_tokens"]))),
            value_token_group_count=group_count,
            max_count_tokens=int(vocab["max_count_tokens"]),
            max_total_tokens=int(vocab["max_total_tokens"]),
        )
    else:
        max_units = int(vocab["max_bins"])
        config = make_config(arrays, args, max_units, 0)
    model = V7ActionPolicy(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_state = None
    best_epoch = 0
    best_score = float("inf")
    metric_states: dict[str, dict[str, torch.Tensor]] = {}
    metric_epochs = {"val_bin_accuracy": 0, "val_activity_f1": 0, "val_edge_activity_f1": 0}
    metric_values = {"val_bin_accuracy": float("-inf"), "val_activity_f1": float("-inf"), "val_edge_activity_f1": float("-inf")}
    checkpoint_dir = args.output_dir / "checkpoints"
    checkpoint_payload = {
        "config": config.__dict__,
        "value_vocab": serialize_value_vocab(vocab),
        "value_bin_loss": args.value_bin_loss,
        "value_bin_class_weights": None if value_bin_class_weights is None else value_bin_class_weights.tolist(),
        "focal_gamma": float(args.focal_gamma),
        "activity_value_weight": float(args.activity_value_weight),
        "max_activity_value_weight": float(args.max_activity_value_weight),
        "edge_activity_loss_weight": float(args.edge_activity_loss_weight),
        "active_edge_sample_weight": float(args.active_edge_sample_weight),
        "hard_negative_edge_weight": float(args.hard_negative_edge_weight),
        "hard_negative_edge_threshold": float(args.hard_negative_edge_threshold),
        "edge_tversky_loss_weight": float(args.edge_tversky_loss_weight),
        "edge_tversky_alpha": float(args.edge_tversky_alpha),
        "edge_tversky_beta": float(args.edge_tversky_beta),
        "step_total_loss_weight": float(args.step_total_loss_weight),
        "pos_weight": pos_weight.tolist(),
        "edge_pos_weight": edge_pos_weight.tolist(),
    }
    epoch_checkpoints = {}
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        rows = []
        for batch, target in train_loader:
            batch = move_batch_to_device(batch, device)
            target = move_target_to_device(target, device)
            outputs = model(batch)
            loss, parts = compute_discrete_policy_loss(
                outputs,
                target,
                pos_weight,
                vocab,
                args.bin_loss_weight,
                value_bin_loss=args.value_bin_loss,
                class_weights=value_bin_class_weights,
                focal_gamma=args.focal_gamma,
                activity_value_weight=args.activity_value_weight,
                max_activity_value_weight=args.max_activity_value_weight,
                edge_activity_loss_weight=args.edge_activity_loss_weight,
                edge_pos_weight=edge_pos_weight,
                active_edge_sample_weight=args.active_edge_sample_weight,
                hard_negative_edge_weight=args.hard_negative_edge_weight,
                hard_negative_edge_threshold=args.hard_negative_edge_threshold,
                edge_tversky_loss_weight=args.edge_tversky_loss_weight,
                edge_tversky_alpha=args.edge_tversky_alpha,
                edge_tversky_beta=args.edge_tversky_beta,
                step_total_loss_weight=args.step_total_loss_weight,
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            rows.append(parts)
        train_loss = mean_rows(rows)
        val_loss = evaluate_loss(model, val_loader, device, pos_weight, edge_pos_weight, args, vocab)
        history.append({"epoch": epoch, "train": train_loss, "val": val_loss})
        score = select_policy_score(val_loss, edge_activity_loss_weight=args.edge_activity_loss_weight)
        if np.isfinite(score) and score < best_score:
            best_score = float(score)
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
        for metric_name, metric_value in (
            ("val_bin_accuracy", val_loss["active_value_bin_accuracy"]),
            ("val_activity_f1", val_loss["activity_f1"]),
            ("val_edge_activity_f1", val_loss["edge_activity_f1"]),
        ):
            if np.isfinite(metric_value) and metric_value > metric_values[metric_name]:
                metric_values[metric_name] = float(metric_value)
                metric_epochs[metric_name] = int(epoch)
                metric_states[metric_name] = deepcopy(model.state_dict())
        if args.save_epoch_checkpoints:
            epoch_threshold = float(val_loss["activity_threshold"])
            epoch_path = save_epoch_checkpoint(
                checkpoint_dir=checkpoint_dir,
                fusion_mode=args.fusion_mode,
                epoch=epoch,
                model_state=deepcopy(model.state_dict()),
                checkpoint_payload={**checkpoint_payload, "activity_threshold": epoch_threshold},
            )
            epoch_checkpoints[str(epoch)] = str(epoch_path)
        print(
            f"[v11-discrete-policy:{device.type}] epoch={epoch} train_total={train_loss['total']:.6f} "
            f"val_f1={val_loss['activity_f1']:.6f} val_bin_acc={val_loss['active_value_bin_accuracy']:.6f} "
            f"val_edge_f1={val_loss['edge_activity_f1']:.6f} val_active_value_rmse={val_loss['active_value_rmse']:.6f}"
        )

    last_state = deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)

    val_predictions = collect_predictions(model, val_loader, device, vocab)
    threshold = choose_threshold(val_predictions["prob"], val_predictions["active"])
    val_eval = evaluate_predictions(val_predictions, threshold)
    test_eval = evaluate_predictions(collect_predictions(model, test_loader, device, vocab), threshold)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"v11_discrete_value_policy_{args.fusion_mode}_best.pt"
    torch.save(
        {
            **checkpoint_payload,
            "model_state": model.state_dict(),
            "best_epoch": best_epoch,
            "best_val_active_value_rmse": val_eval["active_value_rmse"],
            "best_score": best_score,
            "activity_threshold": float(threshold),
        },
        checkpoint_path,
    )
    extra_checkpoints = {}
    checkpoint_payload = {**checkpoint_payload, "activity_threshold": float(threshold)}
    for name, state in {**metric_states, "last": last_state}.items():
        extra_path = checkpoint_dir / f"v11_discrete_value_policy_{args.fusion_mode}_{name}.pt"
        torch.save(
            {
                **checkpoint_payload,
                "model_state": state,
                "best_epoch": int(metric_epochs.get(name, args.epochs)),
                "best_metric_name": name,
                "best_metric_value": float(metric_values.get(name, float("nan"))),
            },
            extra_path,
        )
        extra_checkpoints[name] = str(extra_path)

    return {
        "framework": "PI-JWM",
        "module": "v11_discrete_value_policy",
        "note": "Behavior-cloning policy with hurdle activity head and positive action value-bin classification.",
        "dataset_dir": str(args.dataset_dir),
        "split_sizes": {
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
            "train_used": int(len(train_used_idx)),
            "val_used": int(len(val_used_idx)),
            "test_used": int(len(test_used_idx)),
        },
        "split_seed_spec": split_seed_spec,
        "config": config.__dict__,
        "action_features": [str(x) for x in arrays["edge_action_features"].tolist()],
        "value_vocab": serialize_value_vocab(vocab),
        "value_bin_loss": args.value_bin_loss,
        "value_bin_class_weights": None if value_bin_class_weights is None else value_bin_class_weights.tolist(),
        "focal_gamma": float(args.focal_gamma),
        "activity_value_weight": float(args.activity_value_weight),
        "max_activity_value_weight": float(args.max_activity_value_weight),
        "edge_activity_loss_weight": float(args.edge_activity_loss_weight),
        "active_edge_sample_weight": float(args.active_edge_sample_weight),
        "hard_negative_edge_weight": float(args.hard_negative_edge_weight),
        "hard_negative_edge_threshold": float(args.hard_negative_edge_threshold),
        "edge_tversky_loss_weight": float(args.edge_tversky_loss_weight),
        "edge_tversky_alpha": float(args.edge_tversky_alpha),
        "edge_tversky_beta": float(args.edge_tversky_beta),
        "step_total_loss_weight": float(args.step_total_loss_weight),
        "pos_weight": pos_weight.tolist(),
        "edge_pos_weight": edge_pos_weight.tolist(),
        "history": history,
        "best_epoch": int(best_epoch),
        "best_val_active_value_rmse": float(val_eval["active_value_rmse"]),
        "best_score": float(best_score),
        "activity_threshold": float(threshold),
        "checkpoint_path": str(checkpoint_path),
        "extra_checkpoints": extra_checkpoints,
        "epoch_checkpoints": epoch_checkpoints,
        "val_eval": val_eval,
        "test_eval": test_eval,
    }


def render_report(summary: dict) -> str:
    value_vocab = summary["value_vocab"]
    if value_vocab.get("mode") == "hierarchical_tokens":
        vocab_sizes = {
            "count": value_vocab["count_sizes"],
            "total": value_vocab["total_sizes"],
        }
    else:
        vocab_sizes = value_vocab["sizes"]
    return "\n".join(
        [
            "# PI-JWM v11 Discrete Value Policy",
            "",
            "Supervised policy with activity hurdle gate and positive value-bin classification.",
            "",
            f"- Best epoch: `{summary['best_epoch']}`",
            f"- Checkpoint: `{summary['checkpoint_path']}`",
            f"- Value vocab sizes: `{vocab_sizes}`",
            "",
            "## Test Metrics",
            "",
            f"- activity F1: `{summary['test_eval']['activity_f1']:.6f}`",
            f"- active value RMSE: `{summary['test_eval']['active_value_rmse']:.6f}`",
            f"- active value-bin accuracy: `{summary['test_eval']['active_value_bin_accuracy']:.6f}`",
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = train(args)
    summary_path = args.output_dir / "v11_discrete_value_policy_summary.json"
    report_path = args.output_dir / "v11_discrete_value_policy_report.md"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(render_report(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"summary_path={summary_path}")
    print(f"report_path={report_path}")


if __name__ == "__main__":
    main()
