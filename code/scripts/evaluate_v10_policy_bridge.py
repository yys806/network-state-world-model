"""Evaluate a frozen PI-JWM world model with predicted future actions."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.ensemble import HistGradientBoostingRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from pi_jwm.v6_data import V6WorldModelDataset, collate_v6_world_model_batch, load_world_model_arrays, make_normalization_stats
from pi_jwm.v6_dual_graph import V6DualGraphBatch
from pi_jwm.evaluation.result_protocol import build_result_protocol, classify_bridge_result
from pi_jwm.v7_action_policy import V7ActionPolicy
from pi_jwm.v8_training import build_v8_model_from_arrays, evaluate_v8_model

from evaluate_v9_action_ablation import load_model_for_experiment, resolve_project_path
from run_v7_action_policy import V7ActionPolicyDataset, collate_action_policy_batch, collect_predictions
from run_v11_discrete_value_policy import decode_hierarchical_value_tokens
from run_world_model_v8_full_training import add_event_memory_features, resolve_seed_splits


DEFAULT_WORLD_EXPERIMENT_DIR = (
    PROJECT_ROOT / "artifacts" / "experiments" / "pi_jwm_v10_action_aligned_20260619" / "source_summaries"
)


def make_bridge_result_protocol(
    action_generator: str,
    action_decoder: str,
    mode: str,
    fit_splits: tuple[str, ...] = ("train", "val"),
) -> dict[str, object]:
    return build_result_protocol(
        result_kind=classify_bridge_result(action_generator, action_decoder, mode),
        fit_splits=fit_splits,
        selection_split="val",
        evaluation_split="test",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bridge a behavior-cloned action policy into a frozen PI-JWM world model.")
    parser.add_argument("--world-experiment-dir", type=Path, required=True)
    parser.add_argument("--world-checkpoint", type=Path, default=None)
    parser.add_argument("--policy-checkpoint", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--policy-threshold", type=float, default=None)
    parser.add_argument(
        "--action-decoder",
        choices=(
            "threshold",
            "val_mean_topk",
            "val_quantile_topk",
            "probability_mass_topk",
            "edge_val_mean_topk",
            "edge_val_quantile_topk",
            "edge_probability_mass_topk",
            "edge_threshold",
            "edge_threshold_topk",
            "oracle_topk",
        ),
        default="threshold",
        help=(
            "How to convert policy probabilities into sparse future actions. "
            "threshold keeps the original global threshold behavior; top-k modes constrain active action counts."
        ),
    )
    parser.add_argument(
        "--budget-quantile",
        type=float,
        default=0.5,
        help="Validation count quantile for val_quantile_topk. 0.5 is median.",
    )
    parser.add_argument(
        "--mode",
        choices=("predicted_all", "true_first_pred_rest"),
        default="true_first_pred_rest",
    )
    parser.add_argument(
        "--action-generator",
        choices=(
            "policy",
            "repeat_first_future",
            "repeat_last_history",
            "true_future",
            "true_activity_policy_value",
            "policy_activity_true_value",
        ),
        default="policy",
        help=(
            "Future-action source before mode masking. policy uses the learned V7 policy; "
            "repeat_first_future copies the known first future action across the rollout; "
            "repeat_last_history repeats the latest historical action; diagnostic true_* modes isolate "
            "action-location and action-value errors."
        ),
    )
    parser.add_argument(
        "--value-decoder",
        choices=(
            "policy",
            "train_mean",
            "train_median",
            "train_q75",
            "train_edge_median",
            "train_median_dim_scaled",
            "train_median_step_scaled",
            "train_codebook_quantile",
        ),
        default="policy",
        help=(
            "How to fill action magnitudes after activity decoding. policy uses the V7 value head; "
            "train_* modes use positive-action prototypes fitted on the training split."
        ),
    )
    parser.add_argument(
        "--value-quantile",
        type=float,
        default=0.75,
        help="Positive-action quantile used by train_q75-style value prototypes.",
    )
    parser.add_argument(
        "--value-codebook-size",
        type=int,
        default=5,
        help="Number of positive-action quantile bins for train_codebook_quantile.",
    )
    parser.add_argument(
        "--value-scale",
        type=float,
        default=1.0,
        help="Multiplicative calibration applied after value decoding.",
    )
    parser.add_argument("--rb-dim-scale", type=float, default=1.0, help="Additional scale for RB count/total action dimensions.")
    parser.add_argument("--cpu-dim-scale", type=float, default=1.0, help="Additional scale for CPU count/total action dimensions.")
    parser.add_argument(
        "--step-total-calibrator",
        choices=("none", "val_count_quantile", "policy_step_total"),
        default="none",
        help="Optional validation-fitted per-step RB/CPU total controller applied after action decoding.",
    )
    parser.add_argument(
        "--step-total-quantile",
        type=float,
        default=0.5,
        help="Validation quantile used by --step-total-calibrator val_count_quantile.",
    )
    parser.add_argument("--max-val-samples", type=int, default=0, help="Limit validation samples for memory-bounded CPU diagnostics; <=0 means all.")
    parser.add_argument("--max-test-samples", type=int, default=0, help="Limit test samples for memory-bounded CPU diagnostics; <=0 means all.")
    parser.add_argument(
        "--active-aware-sample-limit",
        action="store_true",
        help="When limiting val/test samples, prefer samples with active future actions or active link targets.",
    )
    parser.add_argument(
        "--pretrim-arrays",
        action="store_true",
        help="Load only train plus limited val/test samples into memory. Requires max-val-samples and max-test-samples.",
    )
    parser.add_argument("--edge-reranker", choices=("none", "logistic_val"), default="none")
    parser.add_argument("--edge-reranker-max-train-rows", type=int, default=200000)
    parser.add_argument("--edge-count-controller", choices=("none", "hgb_val"), default="none")
    parser.add_argument("--edge-count-offset", type=int, default=0)
    return parser.parse_args()


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def limit_eval_indices(indices: np.ndarray, max_samples: int) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    if int(max_samples) <= 0:
        return indices
    return indices[: min(int(max_samples), indices.shape[0])].astype(np.int64)


def limit_eval_indices_active_aware(
    indices: np.ndarray,
    max_samples: int,
    edge_a_future: np.ndarray,
    y_link_active: np.ndarray,
) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    if int(max_samples) <= 0 or int(max_samples) >= len(indices):
        return indices
    action_active = np.any(np.asarray(edge_a_future[indices]) > 1e-9, axis=(1, 2, 3))
    link_active = np.any(np.asarray(y_link_active[indices]) > 0.5, axis=(1, 2))
    preferred = indices[action_active | link_active]
    fallback = indices[~(action_active | link_active)]
    selected = np.concatenate([preferred, fallback])[: int(max_samples)]
    return selected.astype(np.int64)


def load_bridge_arrays_and_splits(
    dataset_dir: Path,
    split: dict,
    max_val_samples: int = 0,
    max_test_samples: int = 0,
    pretrim_arrays: bool = False,
    active_aware_sample_limit: bool = False,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    if not pretrim_arrays:
        arrays = load_world_model_arrays(dataset_dir)
        train_idx, val_idx, test_idx, _ = resolve_seed_splits(
            arrays["sample_seed"],
            train_seeds=split["train_seeds"],
            val_seeds=split["val_seeds"],
            test_seeds=split["test_seeds"],
        )
        if active_aware_sample_limit:
            val_idx = limit_eval_indices_active_aware(val_idx, max_val_samples, arrays["edge_a_future"], arrays["y_link_active"])
            test_idx = limit_eval_indices_active_aware(test_idx, max_test_samples, arrays["edge_a_future"], arrays["y_link_active"])
        else:
            val_idx = limit_eval_indices(val_idx, max_val_samples)
            test_idx = limit_eval_indices(test_idx, max_test_samples)
        return arrays, train_idx, val_idx, test_idx
    if int(max_val_samples) <= 0 or int(max_test_samples) <= 0:
        raise ValueError("--pretrim-arrays requires positive --max-val-samples and --max-test-samples")
    dataset_path = Path(dataset_dir) / "world_model_dataset_v0_samples.npz"
    with np.load(dataset_path, allow_pickle=True) as data:
        sample_seed = np.asarray(data["sample_seed"])
        train_idx, val_idx, test_idx, _ = resolve_seed_splits(
            sample_seed,
            train_seeds=split["train_seeds"],
            val_seeds=split["val_seeds"],
            test_seeds=split["test_seeds"],
        )
        if active_aware_sample_limit:
            val_idx = limit_eval_indices_active_aware(val_idx, max_val_samples, data["edge_a_future"], data["y_link_active"])
            test_idx = limit_eval_indices_active_aware(test_idx, max_test_samples, data["edge_a_future"], data["y_link_active"])
        else:
            val_idx = limit_eval_indices(val_idx, max_val_samples)
            test_idx = limit_eval_indices(test_idx, max_test_samples)
        selected_idx = np.concatenate([train_idx, val_idx, test_idx]).astype(np.int64)
        train_local = np.arange(0, len(train_idx), dtype=np.int64)
        val_local = np.arange(len(train_idx), len(train_idx) + len(val_idx), dtype=np.int64)
        test_local = np.arange(len(train_idx) + len(val_idx), len(selected_idx), dtype=np.int64)
        arrays: dict[str, np.ndarray] = {}
        sample_count = int(sample_seed.shape[0])
        for key in data.files:
            value = data[key]
            if getattr(value, "shape", ()) and value.shape[0] == sample_count:
                arrays[key] = value[selected_idx]
            else:
                arrays[key] = value[()]
        return arrays, train_local, val_local, test_local


def load_policy(checkpoint_path: Path, device: torch.device) -> tuple[V7ActionPolicy, np.ndarray, float, dict | None]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    model = V7ActionPolicy(type("Config", (), config)()).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    threshold = float(checkpoint.get("activity_threshold", 0.5))
    value_vocab = checkpoint.get("value_vocab")
    if value_vocab is None:
        action_scale = np.asarray(checkpoint["action_scale"], dtype=np.float32)
    else:
        action_dim = int(config["action_dim"] if isinstance(config, dict) else getattr(config, "action_dim"))
        action_scale = np.ones(action_dim, dtype=np.float32)
    return model, action_scale, threshold, value_vocab


@dataclass(frozen=True)
class ActionDecoderConfig:
    name: str = "threshold"
    count_budget: np.ndarray | None = None
    probability_budget_scales: np.ndarray | None = None
    budget_quantile: float = 0.5

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "count_budget": None if self.count_budget is None else self.count_budget.astype(int).tolist(),
            "probability_budget_scales": (
                None if self.probability_budget_scales is None else self.probability_budget_scales.astype(float).tolist()
            ),
            "budget_quantile": float(self.budget_quantile),
        }


@dataclass(frozen=True)
class ActionValueDecoderConfig:
    name: str = "policy"
    prototype: np.ndarray | None = None
    value_quantile: float = 0.75
    value_codebook_size: int = 5
    value_scale: float = 1.0

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "prototype_shape": None if self.prototype is None else list(self.prototype.shape),
            "prototype_min": None if self.prototype is None else float(np.min(self.prototype)),
            "prototype_max": None if self.prototype is None else float(np.max(self.prototype)),
            "value_quantile": float(self.value_quantile),
            "value_codebook_size": int(self.value_codebook_size),
            "value_scale": float(self.value_scale),
        }


@dataclass(frozen=True)
class StepTotalCalibrator:
    name: str = "none"
    target_totals: np.ndarray | None = None
    quantile: float = 0.5
    rb_total_dim: int = 2
    cpu_total_dim: int = 4

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "target_totals_shape": None if self.target_totals is None else list(self.target_totals.shape),
            "target_totals_min": None if self.target_totals is None else float(np.min(self.target_totals)),
            "target_totals_max": None if self.target_totals is None else float(np.max(self.target_totals)),
            "quantile": float(self.quantile),
            "rb_total_dim": int(self.rb_total_dim),
            "cpu_total_dim": int(self.cpu_total_dim),
        }


def make_action_value_decoder_config(
    decoder: str,
    arrays: dict[str, np.ndarray],
    train_idx: np.ndarray,
    value_quantile: float,
    value_codebook_size: int,
    value_scale: float,
    val_predictions: dict[str, np.ndarray] | None = None,
    policy_threshold: float = 0.5,
) -> ActionValueDecoderConfig:
    if value_scale < 0.0:
        raise ValueError("value_scale must be non-negative")
    if decoder == "policy":
        return ActionValueDecoderConfig(
            name=decoder,
            value_quantile=value_quantile,
            value_codebook_size=value_codebook_size,
            value_scale=value_scale,
        )
    if not 0.0 <= float(value_quantile) <= 1.0:
        raise ValueError("value_quantile must be between 0 and 1")
    if value_codebook_size <= 0:
        raise ValueError("value_codebook_size must be positive")
    prototype = fit_action_value_prototype(
        arrays["edge_a_future"][train_idx],
        decoder=decoder,
        value_quantile=value_quantile,
        value_codebook_size=value_codebook_size,
        val_predictions=val_predictions,
        policy_threshold=policy_threshold,
    )
    return ActionValueDecoderConfig(
        name=decoder,
        prototype=prototype,
        value_quantile=value_quantile,
        value_codebook_size=value_codebook_size,
        value_scale=value_scale,
    )


def fit_action_value_prototype(
    true_value: np.ndarray,
    decoder: str,
    value_quantile: float,
    value_codebook_size: int = 5,
    val_predictions: dict[str, np.ndarray] | None = None,
    policy_threshold: float = 0.5,
) -> np.ndarray:
    true_value = np.asarray(true_value, dtype=np.float32)
    if true_value.ndim != 4:
        raise ValueError("true_value must have shape [sample, horizon, edge, action_dim]")
    if decoder == "train_mean":
        reducer = lambda values: float(np.mean(values))
        return reduce_positive_values_by_step_action(true_value, reducer)
    if decoder == "train_median":
        reducer = lambda values: float(np.median(values))
        return reduce_positive_values_by_step_action(true_value, reducer)
    if decoder == "train_q75":
        reducer = lambda values: float(np.quantile(values, float(value_quantile)))
        return reduce_positive_values_by_step_action(true_value, reducer)
    if decoder == "train_edge_median":
        fallback = reduce_positive_values_by_step_action(true_value, lambda values: float(np.median(values)))
        return reduce_positive_values_by_step_edge_action(true_value, fallback, lambda values: float(np.median(values)))
    if decoder == "train_median_dim_scaled":
        base = reduce_positive_values_by_step_action(true_value, lambda values: float(np.median(values)))
        return calibrate_value_prototype_scale(base, val_predictions, policy_threshold, mode="dim")
    if decoder == "train_median_step_scaled":
        base = reduce_positive_values_by_step_action(true_value, lambda values: float(np.median(values)))
        return calibrate_value_prototype_scale(base, val_predictions, policy_threshold, mode="step")
    if decoder == "train_codebook_quantile":
        return make_positive_value_quantile_codebook(true_value, value_codebook_size)
    raise ValueError(f"Unknown action value decoder: {decoder}")


def calibrate_value_prototype_scale(
    prototype: np.ndarray,
    val_predictions: dict[str, np.ndarray] | None,
    policy_threshold: float,
    mode: str,
) -> np.ndarray:
    if val_predictions is None:
        raise ValueError(f"{mode} value scaling requires validation predictions")
    true_value = np.asarray(val_predictions["value_true"], dtype=np.float32)
    prob = np.asarray(val_predictions["prob"], dtype=np.float32)
    active = prob >= float(policy_threshold)
    if prototype.shape != (true_value.shape[1], true_value.shape[-1]):
        raise ValueError("prototype must have shape [horizon, action_dim]")
    candidates = np.unique(np.concatenate([np.array([1.0, 1.12, 2.0], dtype=np.float32), np.linspace(0.5, 2.5, 41)]))
    scaled = prototype.astype(np.float32).copy()
    if mode == "dim":
        for dim in range(true_value.shape[-1]):
            pred_base = prototype[None, :, None, dim]
            mask = active[:, :, :, dim]
            target = true_value[:, :, :, dim]
            scaled[:, dim] *= choose_value_scale(pred_base, mask, target, candidates)
        return scaled
    if mode == "step":
        for step in range(true_value.shape[1]):
            for dim in range(true_value.shape[-1]):
                pred_base = prototype[step, dim]
                mask = active[:, step, :, dim]
                target = true_value[:, step, :, dim]
                scaled[step, dim] *= choose_value_scale(pred_base, mask, target, candidates)
        return scaled
    raise ValueError(f"Unknown value scale calibration mode: {mode}")


def choose_value_scale(pred_base: np.ndarray | float, mask: np.ndarray, target: np.ndarray, candidates: np.ndarray) -> float:
    mask = np.asarray(mask, dtype=bool)
    target = np.asarray(target, dtype=np.float32)
    best_scale = 1.0
    best_mse = float("inf")
    if not mask.any():
        return best_scale
    for scale in candidates:
        pred = np.where(mask, np.asarray(pred_base, dtype=np.float32) * float(scale), 0.0)
        mse = float(np.mean((pred - target) ** 2))
        if mse < best_mse:
            best_mse = mse
            best_scale = float(scale)
    return best_scale


def reduce_positive_values_by_step_action(true_value: np.ndarray, reducer) -> np.ndarray:
    _, horizon, _, action_dim = true_value.shape
    prototype = np.zeros((horizon, action_dim), dtype=np.float32)
    for step in range(horizon):
        for dim in range(action_dim):
            values = true_value[:, step, :, dim]
            positives = values[values > 1e-9]
            prototype[step, dim] = reducer(positives) if positives.size else 0.0
    return prototype


def reduce_positive_values_by_step_edge_action(true_value: np.ndarray, fallback: np.ndarray, reducer) -> np.ndarray:
    _, horizon, num_edges, action_dim = true_value.shape
    prototype = np.zeros((horizon, num_edges, action_dim), dtype=np.float32)
    for step in range(horizon):
        for edge in range(num_edges):
            for dim in range(action_dim):
                values = true_value[:, step, edge, dim]
                positives = values[values > 1e-9]
                prototype[step, edge, dim] = reducer(positives) if positives.size else fallback[step, dim]
    return prototype


def make_positive_value_quantile_codebook(true_value: np.ndarray, codebook_size: int) -> np.ndarray:
    if codebook_size <= 0:
        raise ValueError("codebook_size must be positive")
    _, horizon, _, action_dim = true_value.shape
    codebook = np.zeros((horizon, action_dim, codebook_size), dtype=np.float32)
    quantiles = (np.arange(codebook_size, dtype=np.float32) + 1.0) / float(codebook_size + 1)
    for step in range(horizon):
        for dim in range(action_dim):
            values = true_value[:, step, :, dim]
            positives = values[values > 1e-9]
            if positives.size:
                codebook[step, dim] = np.quantile(positives, quantiles).astype(np.float32)
    return codebook


def project_policy_value_to_codebook(policy_value: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    if policy_value.ndim != 3:
        raise ValueError("policy_value must have shape [horizon, edge, action_dim]")
    if codebook.ndim != 3:
        raise ValueError("codebook must have shape [horizon, action_dim, codebook_size]")
    if codebook.shape[:2] != (policy_value.shape[0], policy_value.shape[2]):
        raise ValueError("codebook first dimensions must match policy_value [horizon, action_dim]")
    distances = torch.abs(policy_value.unsqueeze(-1) - codebook[:, None, :, :])
    nearest = torch.argmin(distances, dim=-1)
    expanded_codebook = codebook[:, None, :, :].expand(policy_value.shape[0], policy_value.shape[1], policy_value.shape[2], -1)
    return torch.gather(expanded_codebook, dim=-1, index=nearest.unsqueeze(-1)).squeeze(-1)


def decode_discrete_policy_value(logits: torch.Tensor, value_vocab: dict) -> torch.Tensor:
    if logits.ndim != 5:
        raise ValueError("logits must have shape [batch, horizon, edge, action_dim, max_bins]")
    values = torch.as_tensor(np.asarray(value_vocab["values"], dtype=np.float32), dtype=logits.dtype, device=logits.device)
    sizes = torch.as_tensor(np.asarray(value_vocab["sizes"], dtype=np.int64), dtype=torch.long, device=logits.device)
    if logits.shape[-2] != values.shape[0]:
        raise ValueError("logits action_dim does not match value_vocab")
    if logits.shape[-1] != values.shape[1]:
        raise ValueError("logits max_bins does not match value_vocab")
    bin_ids = torch.arange(logits.shape[-1], device=logits.device).reshape(1, 1, 1, 1, -1)
    valid = bin_ids < sizes.reshape(1, 1, 1, -1, 1)
    masked_logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
    selected = torch.argmax(masked_logits, dim=-1)
    expanded_values = values.reshape(1, 1, 1, values.shape[0], values.shape[1]).expand(*selected.shape, values.shape[1])
    return torch.gather(expanded_values, dim=-1, index=selected.unsqueeze(-1)).squeeze(-1)


def decode_coupled_policy_value(logits: torch.Tensor, value_vocab: dict) -> torch.Tensor:
    if logits.ndim != 5:
        raise ValueError("logits must have shape [batch, horizon, edge, group, max_tokens]")
    values = torch.as_tensor(np.asarray(value_vocab["values"], dtype=np.float32), dtype=logits.dtype, device=logits.device)
    sizes = torch.as_tensor(np.asarray(value_vocab["sizes"], dtype=np.int64), dtype=torch.long, device=logits.device)
    groups = [[int(dim) for dim in group] for group in value_vocab["groups"]]
    action_dim = int(value_vocab.get("action_dim", values.shape[-1]))
    if logits.shape[-2] != values.shape[0]:
        raise ValueError("logits group count does not match value_vocab")
    if logits.shape[-1] != values.shape[1]:
        raise ValueError("logits max_tokens does not match value_vocab")
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


def decode_hierarchical_policy_value(
    count_logits: torch.Tensor,
    total_logits: torch.Tensor,
    value_vocab: dict,
) -> torch.Tensor:
    return decode_hierarchical_value_tokens(count_logits, total_logits, value_vocab)


def collect_bridge_policy_predictions(
    model: V7ActionPolicy,
    loader: DataLoader,
    device: torch.device,
    action_scale_t: torch.Tensor,
    value_vocab: dict | None,
) -> dict[str, np.ndarray]:
    model.eval()
    rows = {"prob": [], "active": [], "value_pred": [], "value_true": []}
    with torch.no_grad():
        for batch, target in loader:
            batch = V6DualGraphBatch(
                node_history=batch.node_history.to(device),
                physical_edge_history=batch.physical_edge_history.to(device),
                info_edge_history=batch.info_edge_history.to(device),
                action_history=batch.action_history.to(device),
                future_actions=batch.future_actions.to(device),
                task_history=batch.task_history.to(device),
            )
            outputs = model(batch)
            rows["prob"].append(torch.sigmoid(outputs["action_logit"]).cpu().numpy())
            if "edge_logit" in outputs:
                rows.setdefault("edge_prob", []).append(torch.sigmoid(outputs["edge_logit"]).cpu().numpy())
            if "step_total_log" in outputs:
                rows.setdefault("step_total_pred", []).append(torch.expm1(torch.clamp(outputs["step_total_log"], min=0.0)).cpu().numpy())
            rows["active"].append(target["action_active"].cpu().numpy())
            if value_vocab is None:
                value_pred = outputs["action_value"] * action_scale_t
            elif "action_value_count_logit" in outputs:
                value_pred = decode_hierarchical_policy_value(
                    outputs["action_value_count_logit"],
                    outputs["action_value_total_logit"],
                    value_vocab,
                )
            elif "action_value_token_logit" in outputs:
                value_pred = decode_coupled_policy_value(outputs["action_value_token_logit"], value_vocab)
            else:
                value_pred = decode_discrete_policy_value(outputs["action_value_bin_logit"], value_vocab)
            rows["value_pred"].append(value_pred.cpu().numpy())
            rows["value_true"].append(target["action_raw"].cpu().numpy())
    return {name: np.concatenate(values, axis=0) for name, values in rows.items()}


def make_action_decoder_config(
    decoder: str,
    val_predictions: dict[str, np.ndarray],
    budget_quantile: float,
) -> ActionDecoderConfig:
    if decoder == "threshold":
        return ActionDecoderConfig(name=decoder, budget_quantile=budget_quantile)
    if decoder == "edge_threshold":
        return ActionDecoderConfig(name=decoder, budget_quantile=budget_quantile)
    if not 0.0 <= float(budget_quantile) <= 1.0:
        raise ValueError("budget_quantile must be between 0 and 1")
    true_value = np.asarray(val_predictions["value_true"], dtype=np.float32)
    if decoder == "val_mean_topk":
        count_budget = summarize_action_count_budget(true_value, reducer="mean", quantile=budget_quantile)
        return ActionDecoderConfig(name=decoder, count_budget=count_budget, budget_quantile=budget_quantile)
    if decoder == "val_quantile_topk":
        count_budget = summarize_action_count_budget(true_value, reducer="quantile", quantile=budget_quantile)
        return ActionDecoderConfig(name=decoder, count_budget=count_budget, budget_quantile=budget_quantile)
    if decoder == "edge_val_mean_topk":
        count_budget = summarize_edge_count_budget(true_value, reducer="mean", quantile=budget_quantile)
        return ActionDecoderConfig(name=decoder, count_budget=count_budget, budget_quantile=budget_quantile)
    if decoder == "edge_val_quantile_topk":
        count_budget = summarize_edge_count_budget(true_value, reducer="quantile", quantile=budget_quantile)
        return ActionDecoderConfig(name=decoder, count_budget=count_budget, budget_quantile=budget_quantile)
    if decoder == "edge_threshold_topk":
        count_budget = summarize_edge_count_budget(true_value, reducer="quantile", quantile=budget_quantile)
        return ActionDecoderConfig(name=decoder, count_budget=count_budget, budget_quantile=budget_quantile)
    if decoder == "probability_mass_topk":
        scales = calibrate_probability_budget_scales(
            np.asarray(val_predictions["prob"], dtype=np.float32),
            true_value,
        )
        return ActionDecoderConfig(name=decoder, probability_budget_scales=scales, budget_quantile=budget_quantile)
    if decoder == "edge_probability_mass_topk":
        scales = calibrate_edge_probability_budget_scales(
            np.asarray(val_predictions["prob"], dtype=np.float32),
            true_value,
            edge_prob=val_predictions.get("edge_prob"),
        )
        return ActionDecoderConfig(name=decoder, probability_budget_scales=scales, budget_quantile=budget_quantile)
    if decoder == "oracle_topk":
        return ActionDecoderConfig(name=decoder, budget_quantile=budget_quantile)
    raise ValueError(f"Unknown action decoder: {decoder}")


def summarize_action_count_budget(true_value: np.ndarray, reducer: str, quantile: float) -> np.ndarray:
    active_counts = (np.asarray(true_value) > 1e-9).sum(axis=2)
    if reducer == "mean":
        budget = np.rint(active_counts.mean(axis=0))
    elif reducer == "quantile":
        budget = np.rint(np.quantile(active_counts, float(quantile), axis=0))
    else:
        raise ValueError(f"Unknown action count reducer: {reducer}")
    return np.clip(budget, 0, true_value.shape[2]).astype(np.int64)


def summarize_edge_count_budget(true_value: np.ndarray, reducer: str, quantile: float) -> np.ndarray:
    active_counts = np.any(np.asarray(true_value) > 1e-9, axis=-1).sum(axis=2)
    if reducer == "mean":
        budget = np.rint(active_counts.mean(axis=0))
    elif reducer == "quantile":
        budget = np.rint(np.quantile(active_counts, float(quantile), axis=0))
    else:
        raise ValueError(f"Unknown edge count reducer: {reducer}")
    return np.clip(budget, 0, true_value.shape[2]).astype(np.int64)


def calibrate_probability_budget_scales(prob: np.ndarray, true_value: np.ndarray) -> np.ndarray:
    prob = np.asarray(prob, dtype=np.float32)
    true_active = np.asarray(true_value) > 1e-9
    true_counts = true_active.sum(axis=2)
    prob_mass = prob.sum(axis=2)
    _, horizon, num_edges, action_dim = prob.shape
    scales = np.zeros((horizon, action_dim), dtype=np.float32)
    for step in range(horizon):
        for dim in range(action_dim):
            mass = prob_mass[:, step, dim]
            counts = true_counts[:, step, dim]
            base = safe_div(float(counts.sum()), float(mass.sum()))
            upper = max(base * 4.0, 1e-4)
            candidates = np.unique(
                np.concatenate(
                    [
                        np.array([0.0, base], dtype=np.float32),
                        np.linspace(max(base * 0.1, 1e-5), upper, 50, dtype=np.float32),
                    ]
                )
            )
            best_scale = float(base)
            best_f1 = -1.0
            for scale in candidates:
                pred_counts = np.zeros((prob.shape[0], horizon, action_dim), dtype=np.int64)
                pred_counts[:, step, dim] = np.clip(np.rint(mass * float(scale)), 0, num_edges).astype(np.int64)
                pred_active = decode_action_activity_topk_np(prob[:, :, :, dim : dim + 1], pred_counts[:, :, dim : dim + 1])
                score = binary_metrics_np(pred_active[:, step, :, 0], true_active[:, step, :, dim])["f1"]
                if score > best_f1:
                    best_f1 = score
                    best_scale = float(scale)
            scales[step, dim] = best_scale
    return scales


def edge_probability_score(prob: np.ndarray) -> np.ndarray:
    prob = np.asarray(prob, dtype=np.float32)
    if prob.ndim != 4:
        raise ValueError("prob must have shape [sample, horizon, edge, action_dim]")
    return prob.max(axis=-1)


def resolve_edge_probability_score(prob: np.ndarray, edge_prob: np.ndarray | None = None) -> np.ndarray:
    prob = np.asarray(prob, dtype=np.float32)
    if edge_prob is None:
        return edge_probability_score(prob)
    edge_prob = np.asarray(edge_prob, dtype=np.float32)
    if edge_prob.shape != prob.shape[:3]:
        raise ValueError("edge_prob must have shape [sample, horizon, edge]")
    return edge_prob


def calibrate_edge_probability_budget_scales(
    prob: np.ndarray,
    true_value: np.ndarray,
    edge_prob: np.ndarray | None = None,
) -> np.ndarray:
    prob = np.asarray(prob, dtype=np.float32)
    true_edge_active = np.any(np.asarray(true_value) > 1e-9, axis=-1)
    true_counts = true_edge_active.sum(axis=2)
    edge_score = resolve_edge_probability_score(prob, edge_prob)
    edge_mass = edge_score.sum(axis=2)
    _, horizon, num_edges = edge_score.shape
    scales = np.zeros(horizon, dtype=np.float32)
    for step in range(horizon):
        mass = edge_mass[:, step]
        counts = true_counts[:, step]
        base = safe_div(float(counts.sum()), float(mass.sum()))
        upper = max(base * 4.0, 1e-4)
        candidates = np.unique(
            np.concatenate(
                [
                    np.array([0.0, base], dtype=np.float32),
                    np.linspace(max(base * 0.1, 1e-5), upper, 50, dtype=np.float32),
                ]
            )
        )
        best_scale = float(base)
        best_f1 = -1.0
        for scale in candidates:
            pred_counts = np.zeros((prob.shape[0], horizon), dtype=np.int64)
            pred_counts[:, step] = np.clip(np.rint(mass * float(scale)), 0, num_edges).astype(np.int64)
            pred_active = decode_edge_activity_topk_np(prob, pred_counts, edge_prob=edge_score)
            score = binary_metrics_np(pred_active[:, step], true_edge_active[:, step])["f1"]
            if score > best_f1:
                best_f1 = score
                best_scale = float(scale)
        scales[step] = best_scale
    return scales


def decode_action_activity_topk_np(prob: np.ndarray, counts: np.ndarray) -> np.ndarray:
    prob = np.asarray(prob, dtype=np.float32)
    counts = np.asarray(counts, dtype=np.int64)
    if prob.ndim != 4:
        raise ValueError("prob must have shape [sample, horizon, edge, action_dim]")
    if counts.shape != (prob.shape[0], prob.shape[1], prob.shape[3]):
        raise ValueError("counts must have shape [sample, horizon, action_dim]")
    decoded = np.zeros_like(prob, dtype=bool)
    num_edges = prob.shape[2]
    for sample_idx in range(prob.shape[0]):
        for step in range(prob.shape[1]):
            for dim in range(prob.shape[3]):
                k = int(np.clip(counts[sample_idx, step, dim], 0, num_edges))
                if k <= 0:
                    continue
                top_idx = np.argpartition(prob[sample_idx, step, :, dim], -k)[-k:]
                decoded[sample_idx, step, top_idx, dim] = True
    return decoded


def decode_edge_activity_topk_np(prob: np.ndarray, counts: np.ndarray, edge_prob: np.ndarray | None = None) -> np.ndarray:
    prob = np.asarray(prob, dtype=np.float32)
    counts = np.asarray(counts, dtype=np.int64)
    if prob.ndim != 4:
        raise ValueError("prob must have shape [sample, horizon, edge, action_dim]")
    if counts.shape != (prob.shape[0], prob.shape[1]):
        raise ValueError("counts must have shape [sample, horizon]")
    edge_score = resolve_edge_probability_score(prob, edge_prob)
    decoded = np.zeros(edge_score.shape, dtype=bool)
    num_edges = prob.shape[2]
    for sample_idx in range(prob.shape[0]):
        for step in range(prob.shape[1]):
            k = int(np.clip(counts[sample_idx, step], 0, num_edges))
            if k <= 0:
                continue
            top_idx = np.argpartition(edge_score[sample_idx, step], -k)[-k:]
            decoded[sample_idx, step, top_idx] = True
    return decoded


def decode_edge_threshold_np(prob: np.ndarray, threshold: float, edge_prob: np.ndarray | None = None) -> np.ndarray:
    return resolve_edge_probability_score(prob, edge_prob) >= float(threshold)


def decode_edge_threshold_topk_np(
    prob: np.ndarray,
    counts: np.ndarray,
    threshold: float,
    edge_prob: np.ndarray | None = None,
) -> np.ndarray:
    prob = np.asarray(prob, dtype=np.float32)
    counts = np.asarray(counts, dtype=np.int64)
    active = decode_edge_activity_topk_np(prob, counts, edge_prob=edge_prob)
    return active & decode_edge_threshold_np(prob, threshold, edge_prob=edge_prob)


def decode_action_activity_topk(prob: torch.Tensor, counts: torch.Tensor | np.ndarray) -> torch.Tensor:
    if prob.ndim != 3:
        raise ValueError("prob must have shape [horizon, edge, action_dim]")
    counts_t = torch.as_tensor(counts, dtype=torch.long, device=prob.device)
    if counts_t.shape != (prob.shape[0], prob.shape[2]):
        raise ValueError("counts must have shape [horizon, action_dim]")
    active = torch.zeros_like(prob, dtype=torch.bool)
    num_edges = int(prob.shape[1])
    for step in range(int(prob.shape[0])):
        for dim in range(int(prob.shape[2])):
            k = int(torch.clamp(counts_t[step, dim], min=0, max=num_edges).item())
            if k <= 0:
                continue
            top_idx = torch.topk(prob[step, :, dim], k=k).indices
            active[step, top_idx, dim] = True
    return active


def decode_edge_activity_topk(
    prob: torch.Tensor,
    counts: torch.Tensor | np.ndarray,
    edge_prob: torch.Tensor | None = None,
) -> torch.Tensor:
    if prob.ndim != 3:
        raise ValueError("prob must have shape [horizon, edge, action_dim]")
    counts_t = torch.as_tensor(counts, dtype=torch.long, device=prob.device)
    if counts_t.shape != (prob.shape[0],):
        raise ValueError("counts must have shape [horizon]")
    edge_score = torch.max(prob, dim=-1).values if edge_prob is None else edge_prob
    if edge_score.shape != prob.shape[:2]:
        raise ValueError("edge_prob must have shape [horizon, edge]")
    active_edge = torch.zeros_like(edge_score, dtype=torch.bool)
    num_edges = int(prob.shape[1])
    for step in range(int(prob.shape[0])):
        k = int(torch.clamp(counts_t[step], min=0, max=num_edges).item())
        if k <= 0:
            continue
        top_idx = torch.topk(edge_score[step], k=k).indices
        active_edge[step, top_idx] = True
    return active_edge[:, :, None].expand_as(prob)


def decode_edge_threshold(prob: torch.Tensor, threshold: float, edge_prob: torch.Tensor | None = None) -> torch.Tensor:
    if prob.ndim != 3:
        raise ValueError("prob must have shape [horizon, edge, action_dim]")
    edge_score = torch.max(prob, dim=-1).values if edge_prob is None else edge_prob
    if edge_score.shape != prob.shape[:2]:
        raise ValueError("edge_prob must have shape [horizon, edge]")
    active_edge = edge_score >= float(threshold)
    return active_edge[:, :, None].expand_as(prob)


def decode_edge_threshold_topk(
    prob: torch.Tensor,
    counts: torch.Tensor | np.ndarray,
    threshold: float,
    edge_prob: torch.Tensor | None = None,
) -> torch.Tensor:
    if prob.ndim != 3:
        raise ValueError("prob must have shape [horizon, edge, action_dim]")
    counts_t = torch.as_tensor(counts, dtype=torch.long, device=prob.device)
    if counts_t.shape != (prob.shape[0],):
        raise ValueError("counts must have shape [horizon]")
    edge_score = torch.max(prob, dim=-1).values if edge_prob is None else edge_prob
    if edge_score.shape != prob.shape[:2]:
        raise ValueError("edge_prob must have shape [horizon, edge]")
    active_edge = torch.zeros_like(edge_score, dtype=torch.bool)
    num_edges = int(prob.shape[1])
    for step in range(int(prob.shape[0])):
        candidates = torch.nonzero(edge_score[step] >= float(threshold), as_tuple=False).flatten()
        if candidates.numel() == 0:
            continue
        k = min(int(torch.clamp(counts_t[step], min=0, max=num_edges).item()), int(candidates.numel()))
        if k <= 0:
            continue
        local = torch.topk(edge_score[step, candidates], k=k).indices
        active_edge[step, candidates[local]] = True
    return active_edge[:, :, None].expand_as(prob)


def predict_counts_from_probability_mass(prob: torch.Tensor, scales: np.ndarray) -> torch.Tensor:
    scales_t = torch.as_tensor(scales, dtype=prob.dtype, device=prob.device)
    if scales_t.shape != (prob.shape[0], prob.shape[2]):
        raise ValueError("probability budget scales must have shape [horizon, action_dim]")
    counts = torch.round(prob.sum(dim=1) * scales_t).to(torch.long)
    return torch.clamp(counts, min=0, max=int(prob.shape[1]))


def predict_edge_counts_from_probability_mass(
    prob: torch.Tensor,
    scales: np.ndarray,
    edge_prob: torch.Tensor | None = None,
) -> torch.Tensor:
    scales_t = torch.as_tensor(scales, dtype=prob.dtype, device=prob.device)
    if scales_t.shape != (prob.shape[0],):
        raise ValueError("edge probability budget scales must have shape [horizon]")
    edge_score = torch.max(prob, dim=-1).values if edge_prob is None else edge_prob
    if edge_score.shape != prob.shape[:2]:
        raise ValueError("edge_prob must have shape [horizon, edge]")
    counts = torch.round(edge_score.sum(dim=1) * scales_t).to(torch.long)
    return torch.clamp(counts, min=0, max=int(prob.shape[1]))


def scale_action_value_groups(value: torch.Tensor, rb_dim_scale: float = 1.0, cpu_dim_scale: float = 1.0) -> torch.Tensor:
    scaled = value.clone()
    scaled[..., 1] = scaled[..., 1] * float(rb_dim_scale)
    scaled[..., 2] = scaled[..., 2] * float(rb_dim_scale)
    scaled[..., 3] = scaled[..., 3] * float(cpu_dim_scale)
    scaled[..., 4] = scaled[..., 4] * float(cpu_dim_scale)
    return scaled


def fit_step_total_calibrator(
    pred_actions: np.ndarray,
    true_actions: np.ndarray,
    quantile: float = 0.5,
    rb_total_dim: int = 2,
    cpu_total_dim: int = 4,
) -> StepTotalCalibrator:
    pred_actions = np.asarray(pred_actions, dtype=np.float32)
    true_actions = np.asarray(true_actions, dtype=np.float32)
    if pred_actions.shape != true_actions.shape:
        raise ValueError("pred_actions and true_actions must have the same shape")
    if pred_actions.ndim != 4:
        raise ValueError("actions must have shape [sample, horizon, edge, action_dim]")
    if not 0.0 <= float(quantile) <= 1.0:
        raise ValueError("quantile must be between 0 and 1")
    _, horizon, num_edges, action_dim = pred_actions.shape
    if max(rb_total_dim, cpu_total_dim) >= action_dim:
        raise ValueError("total dimensions are outside action_dim")
    pred_counts = np.any(pred_actions > 1e-9, axis=-1).sum(axis=2).astype(np.int64)
    true_rb_totals = true_actions[..., int(rb_total_dim)].sum(axis=2)
    true_cpu_totals = true_actions[..., int(cpu_total_dim)].sum(axis=2)
    targets = np.zeros((horizon, num_edges + 1, 2), dtype=np.float32)
    for step in range(horizon):
        fallback_rb = float(np.quantile(true_rb_totals[:, step], float(quantile))) if true_rb_totals.shape[0] else 0.0
        fallback_cpu = float(np.quantile(true_cpu_totals[:, step], float(quantile))) if true_cpu_totals.shape[0] else 0.0
        for count in range(num_edges + 1):
            if count == 0:
                continue
            mask = pred_counts[:, step] == count
            if np.any(mask):
                targets[step, count, 0] = float(np.quantile(true_rb_totals[mask, step], float(quantile)))
                targets[step, count, 1] = float(np.quantile(true_cpu_totals[mask, step], float(quantile)))
            else:
                targets[step, count, 0] = fallback_rb
                targets[step, count, 1] = fallback_cpu
    return StepTotalCalibrator(
        name="val_count_quantile",
        target_totals=targets,
        quantile=float(quantile),
        rb_total_dim=int(rb_total_dim),
        cpu_total_dim=int(cpu_total_dim),
    )


def make_step_total_calibrator(
    name: str,
    pred_actions: np.ndarray,
    true_actions: np.ndarray,
    quantile: float = 0.5,
) -> StepTotalCalibrator:
    if name == "none":
        return StepTotalCalibrator(name="none", quantile=float(quantile))
    if name == "policy_step_total":
        return StepTotalCalibrator(name="policy_step_total", quantile=float(quantile))
    if name == "val_count_quantile":
        return fit_step_total_calibrator(pred_actions, true_actions, quantile=quantile)
    raise ValueError(f"Unknown step total calibrator: {name}")


def apply_step_total_calibration_np(
    actions: np.ndarray,
    calibrator: StepTotalCalibrator,
    step_total_pred: np.ndarray | None = None,
) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32).copy()
    if calibrator.name == "none":
        return actions
    if calibrator.name == "policy_step_total":
        if step_total_pred is None:
            raise ValueError("policy_step_total requires step_total_pred")
        targets = np.asarray(step_total_pred, dtype=np.float32)
        if targets.shape != (actions.shape[0], actions.shape[1], 3):
            raise ValueError("step_total_pred must have shape [sample, horizon, 3]")
        for sample_idx in range(actions.shape[0]):
            for step in range(actions.shape[1]):
                for dim, target in (
                    (int(calibrator.rb_total_dim), float(targets[sample_idx, step, 1])),
                    (int(calibrator.cpu_total_dim), float(targets[sample_idx, step, 2])),
                ):
                    current = float(actions[sample_idx, step, :, dim].sum())
                    if current > 1e-9:
                        actions[sample_idx, step, :, dim] *= float(max(target, 0.0) / current)
        return actions
    if calibrator.name != "val_count_quantile" or calibrator.target_totals is None:
        raise ValueError(f"Unsupported step total calibrator: {calibrator.name}")
    targets = np.asarray(calibrator.target_totals, dtype=np.float32)
    active_counts = np.any(actions > 1e-9, axis=-1).sum(axis=2).astype(np.int64)
    _, horizon, num_edges, _ = actions.shape
    if targets.shape[:2] != (horizon, num_edges + 1):
        raise ValueError("calibrator target shape does not match actions")
    for sample_idx in range(actions.shape[0]):
        for step in range(horizon):
            count = int(np.clip(active_counts[sample_idx, step], 0, num_edges))
            if count <= 0:
                actions[sample_idx, step, :, int(calibrator.rb_total_dim)] = 0.0
                actions[sample_idx, step, :, int(calibrator.cpu_total_dim)] = 0.0
                continue
            rb_target = float(targets[step, count, 0])
            cpu_target = float(targets[step, count, 1])
            for dim, target in ((int(calibrator.rb_total_dim), rb_target), (int(calibrator.cpu_total_dim), cpu_target)):
                current = float(actions[sample_idx, step, :, dim].sum())
                if current > 1e-9:
                    actions[sample_idx, step, :, dim] *= float(target / current)
                else:
                    active_edges = np.any(actions[sample_idx, step] > 1e-9, axis=-1)
                    if np.any(active_edges):
                        actions[sample_idx, step, active_edges, dim] = float(target / np.sum(active_edges))
    return actions


def apply_step_total_calibration(
    actions: torch.Tensor,
    calibrator: StepTotalCalibrator,
    step_total_pred: torch.Tensor | None = None,
) -> torch.Tensor:
    if calibrator.name == "none":
        return actions
    step_total_np = None if step_total_pred is None else step_total_pred.detach().cpu().numpy()
    calibrated = apply_step_total_calibration_np(actions.detach().cpu().numpy(), calibrator, step_total_pred=step_total_np)
    return torch.as_tensor(calibrated, dtype=actions.dtype, device=actions.device)


def decode_action_value_predictions_np(policy_value: np.ndarray, value_config: ActionValueDecoderConfig) -> np.ndarray:
    policy_value = np.asarray(policy_value, dtype=np.float32)
    if value_config.name == "policy":
        return policy_value * float(value_config.value_scale)
    if value_config.prototype is None:
        raise ValueError(f"{value_config.name} requires a value prototype")
    prototype = np.asarray(value_config.prototype, dtype=np.float32)
    if value_config.name == "train_codebook_quantile":
        value_t = torch.as_tensor(policy_value, dtype=torch.float32)
        codebook_t = torch.as_tensor(prototype, dtype=torch.float32)
        decoded = torch.stack([project_policy_value_to_codebook(row, codebook_t) for row in value_t], dim=0).numpy()
        return decoded.astype(np.float32) * float(value_config.value_scale)
    if prototype.ndim == 2:
        return np.broadcast_to(prototype[None, :, None, :], policy_value.shape).astype(np.float32) * float(value_config.value_scale)
    if prototype.ndim == 3:
        if prototype.shape == policy_value.shape[1:]:
            return np.broadcast_to(prototype[None, :, :, :], policy_value.shape).astype(np.float32) * float(value_config.value_scale)
        if prototype.shape == (policy_value.shape[1], policy_value.shape[3], prototype.shape[2]):
            raise ValueError("codebook prototypes require train_codebook_quantile")
    raise ValueError("Value prototype must have shape [horizon, action_dim] or [horizon, edge, action_dim]")


def decode_policy_predictions_to_actions_np(
    predictions: dict[str, np.ndarray],
    decoder_config: ActionDecoderConfig,
    value_config: ActionValueDecoderConfig,
    threshold: float,
    rb_dim_scale: float = 1.0,
    cpu_dim_scale: float = 1.0,
    override_edge_prob: np.ndarray | None = None,
    edge_count_controller_counts: np.ndarray | None = None,
) -> np.ndarray:
    prob = np.asarray(predictions["prob"], dtype=np.float32)
    value = decode_action_value_predictions_np(predictions["value_pred"], value_config)
    value_t = scale_action_value_groups(
        torch.as_tensor(value, dtype=torch.float32),
        rb_dim_scale=rb_dim_scale,
        cpu_dim_scale=cpu_dim_scale,
    )
    active_rows = []
    true_value = np.asarray(predictions["value_true"], dtype=np.float32)
    edge_prob = override_edge_prob if override_edge_prob is not None else predictions.get("edge_prob")
    for sample_idx in range(prob.shape[0]):
        edge_prob_sample = None if edge_prob is None else torch.as_tensor(edge_prob[sample_idx], dtype=torch.float32)
        if edge_count_controller_counts is not None:
            active = decode_edge_activity_topk(
                torch.as_tensor(prob[sample_idx], dtype=torch.float32),
                edge_count_controller_counts[sample_idx],
                edge_prob=edge_prob_sample,
            )
        else:
            active = PolicyBridgeDataset.decode_activity_static(
                torch.as_tensor(prob[sample_idx], dtype=torch.float32),
                torch.as_tensor(true_value[sample_idx], dtype=torch.float32),
                decoder_config,
                float(threshold),
                edge_prob=edge_prob_sample,
            )
        active_rows.append(active.cpu().numpy())
    active_np = np.stack(active_rows, axis=0).astype(bool)
    return np.where(active_np, value_t.numpy(), 0.0).astype(np.float32)


def binary_metrics_np(pred: np.ndarray, true: np.ndarray) -> dict[str, float | int]:
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


def build_edge_reranker_features(predictions: dict[str, np.ndarray]) -> np.ndarray:
    prob = np.asarray(predictions["prob"], dtype=np.float32)
    if prob.ndim != 4:
        raise ValueError("prob must have shape [sample, horizon, edge, action_dim]")
    samples, horizon, edges, _ = prob.shape
    edge_score = np.asarray(predictions.get("edge_prob", prob.max(axis=-1)), dtype=np.float32)
    if edge_score.shape != (samples, horizon, edges):
        raise ValueError("edge_prob must have shape [sample, horizon, edge]")
    value = np.asarray(predictions.get("value_pred", np.zeros_like(prob)), dtype=np.float32)
    if value.shape != prob.shape:
        raise ValueError("value_pred must match prob shape")
    step = np.broadcast_to(
        (np.arange(horizon, dtype=np.float32) / max(horizon - 1, 1)).reshape(1, horizon, 1),
        (samples, horizon, edges),
    )
    edge = np.broadcast_to(
        (np.arange(edges, dtype=np.float32) / max(edges - 1, 1)).reshape(1, 1, edges),
        (samples, horizon, edges),
    )
    features = [
        edge_score,
        prob.max(axis=-1),
        prob.mean(axis=-1),
        prob.std(axis=-1),
        value.max(axis=-1),
        value.mean(axis=-1),
        value.std(axis=-1),
        step,
        edge,
    ]
    if value.shape[-1] > 2:
        features.append(value[..., 2])
    if value.shape[-1] > 4:
        features.append(value[..., 4])
    return np.stack(features, axis=-1).reshape(-1, len(features)).astype(np.float32)


def fit_logistic_edge_reranker(predictions: dict[str, np.ndarray], max_train_rows: int, seed: int = 20260629):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    features = build_edge_reranker_features(predictions)
    target = np.any(np.asarray(predictions["value_true"], dtype=np.float32) > 1e-9, axis=-1).reshape(-1)
    if not np.any(target) or np.all(target):
        raise ValueError("edge reranker requires both positive and negative validation edges")
    max_train_rows = int(max_train_rows)
    if max_train_rows > 0 and features.shape[0] > max_train_rows:
        rng = np.random.default_rng(seed)
        pos_idx = np.flatnonzero(target)
        neg_idx = np.flatnonzero(~target)
        neg_take = max(0, max_train_rows - len(pos_idx))
        if neg_take < len(neg_idx):
            neg_idx = rng.choice(neg_idx, size=neg_take, replace=False)
        selected = np.concatenate([pos_idx, neg_idx])
        rng.shuffle(selected)
        features = features[selected]
        target = target[selected]
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=500, class_weight="balanced", solver="lbfgs"),
    )
    return model.fit(features, target)


def predict_edge_reranker_scores(model, predictions: dict[str, np.ndarray]) -> np.ndarray:
    prob = np.asarray(predictions["prob"], dtype=np.float32)
    features = build_edge_reranker_features(predictions)
    scores = model.predict_proba(features)[:, 1].astype(np.float32)
    return scores.reshape(prob.shape[:3])


def build_edge_count_features(predictions: dict[str, np.ndarray]) -> np.ndarray:
    prob = np.asarray(predictions["prob"], dtype=np.float32)
    if prob.ndim != 4:
        raise ValueError("prob must have shape [sample, horizon, edge, action_dim]")
    samples, horizon, edges, _ = prob.shape
    edge_score = np.asarray(predictions.get("edge_prob", prob.max(axis=-1)), dtype=np.float32)
    if edge_score.shape != (samples, horizon, edges):
        raise ValueError("edge_prob must have shape [sample, horizon, edge]")
    sorted_score = np.sort(edge_score, axis=2)
    step = np.broadcast_to(
        (np.arange(horizon, dtype=np.float32) / max(horizon - 1, 1)).reshape(1, horizon),
        (samples, horizon),
    )
    features = [
        edge_score.sum(axis=2),
        edge_score.mean(axis=2),
        edge_score.std(axis=2),
        sorted_score[:, :, -1],
        sorted_score[:, :, -min(3, edges) :].sum(axis=2),
        sorted_score[:, :, -min(8, edges) :].sum(axis=2),
        (edge_score >= 0.50).sum(axis=2).astype(np.float32),
        (edge_score >= 0.80).sum(axis=2).astype(np.float32),
        (edge_score >= 0.90).sum(axis=2).astype(np.float32),
        (edge_score >= 0.95).sum(axis=2).astype(np.float32),
        (edge_score >= 0.99).sum(axis=2).astype(np.float32),
        step,
    ]
    if "value_pred" in predictions:
        value = np.asarray(predictions["value_pred"], dtype=np.float32)
        if value.shape != prob.shape:
            raise ValueError("value_pred must match prob shape")
        if value.shape[-1] > 2:
            features.append(np.clip(value[..., 2], 0.0, None).sum(axis=2))
        if value.shape[-1] > 4:
            features.append(np.clip(value[..., 4], 0.0, None).sum(axis=2))
        features.extend([value.max(axis=(2, 3)), value.mean(axis=(2, 3))])
    return np.stack(features, axis=-1).reshape(samples * horizon, len(features)).astype(np.float32)


def fit_edge_count_regressor(predictions: dict[str, np.ndarray], seed: int = 20260629) -> dict:
    features = build_edge_count_features(predictions)
    target = np.any(np.asarray(predictions["value_true"], dtype=np.float32) > 1e-9, axis=-1).sum(axis=2).astype(np.float32).reshape(-1)
    prob = np.asarray(predictions["prob"], dtype=np.float32)
    unique = np.unique(target)
    if unique.size <= 1:
        return {"model": None, "constant": float(unique[0]) if unique.size else 0.0, "num_edges": int(prob.shape[2])}
    model = HistGradientBoostingRegressor(
        max_iter=100,
        max_leaf_nodes=15,
        learning_rate=0.05,
        l2_regularization=0.01,
        random_state=int(seed),
    )
    model.fit(features, target)
    return {"model": model, "constant": None, "num_edges": int(prob.shape[2])}


def predict_edge_count_regressor(model_info: dict, predictions: dict[str, np.ndarray]) -> np.ndarray:
    prob = np.asarray(predictions["prob"], dtype=np.float32)
    if model_info.get("model") is None:
        raw = np.full((prob.shape[0] * prob.shape[1],), float(model_info.get("constant", 0.0)), dtype=np.float32)
    else:
        raw = np.asarray(model_info["model"].predict(build_edge_count_features(predictions)), dtype=np.float32)
    counts = np.rint(raw).reshape(prob.shape[0], prob.shape[1]).astype(np.int64)
    return np.clip(counts, 0, prob.shape[2])


def apply_edge_count_offset(counts: np.ndarray, offset: int, num_edges: int) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.int64)
    shifted = counts + int(offset)
    return np.clip(shifted, 0, int(num_edges)).astype(np.int64)


def predictions_with_edge_prob(predictions: dict[str, np.ndarray], edge_prob: np.ndarray | None) -> dict[str, np.ndarray]:
    if edge_prob is None:
        return predictions
    updated = dict(predictions)
    updated["edge_prob"] = np.asarray(edge_prob, dtype=np.float32)
    return updated


def choose_threshold_by_f1(scores: np.ndarray, true: np.ndarray, candidates: np.ndarray | None = None) -> float:
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    true = np.asarray(true, dtype=bool).reshape(-1)
    if candidates is None:
        candidates = np.unique(np.quantile(scores, np.linspace(0.0, 1.0, 101))).astype(np.float32)
    best_threshold = float(candidates[0]) if len(candidates) else 0.5
    best_f1 = -1.0
    for threshold in np.asarray(candidates, dtype=np.float32).reshape(-1):
        f1 = binary_metrics_np(scores >= float(threshold), true)["f1"]
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)
    return best_threshold


def repeat_first_future_actions(true_future: torch.Tensor) -> torch.Tensor:
    if true_future.ndim != 3:
        raise ValueError("true_future must have shape [horizon, edge, action_dim]")
    return true_future[0:1].expand_as(true_future).clone()


def repeat_last_history_actions(history_actions: torch.Tensor, horizon: int) -> torch.Tensor:
    if history_actions.ndim != 3:
        raise ValueError("history_actions must have shape [history, edge, action_dim]")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    return history_actions[-1:].expand(horizon, -1, -1).clone()


class PolicyBridgeDataset(Dataset):
    def __init__(
        self,
        base_dataset: V6WorldModelDataset,
        policy_dataset: V7ActionPolicyDataset,
        policy_model: V7ActionPolicy,
        action_scale: np.ndarray,
        stats: dict,
        device: torch.device,
        threshold: float,
        mode: str,
        decoder_config: ActionDecoderConfig,
        action_generator: str,
        value_decoder_config: ActionValueDecoderConfig,
        value_vocab: dict | None = None,
        rb_dim_scale: float = 1.0,
        cpu_dim_scale: float = 1.0,
        step_total_calibrator: StepTotalCalibrator | None = None,
        override_edge_prob: np.ndarray | None = None,
        edge_count_controller_counts: np.ndarray | None = None,
    ):
        self.base_dataset = base_dataset
        self.policy_dataset = policy_dataset
        self.policy_model = policy_model
        self.action_scale = torch.as_tensor(action_scale.reshape(1, 1, 1, -1), dtype=torch.float32, device=device)
        self.stats = stats
        self.device = device
        self.threshold = float(threshold)
        self.mode = mode
        self.decoder_config = decoder_config
        self.action_generator = action_generator
        self.value_decoder_config = value_decoder_config
        self.value_vocab = value_vocab
        self.rb_dim_scale = float(rb_dim_scale)
        self.cpu_dim_scale = float(cpu_dim_scale)
        self.step_total_calibrator = step_total_calibrator or StepTotalCalibrator(name="none")
        self.override_edge_prob = None if override_edge_prob is None else np.asarray(override_edge_prob, dtype=np.float32)
        self.edge_count_controller_counts = (
            None if edge_count_controller_counts is None else np.asarray(edge_count_controller_counts, dtype=np.int64)
        )
        if self.override_edge_prob is not None and self.override_edge_prob.shape[:1] != (len(base_dataset),):
            raise ValueError("override_edge_prob first dimension must match dataset length")
        if self.edge_count_controller_counts is not None and self.edge_count_controller_counts.shape[:1] != (len(base_dataset),):
            raise ValueError("edge_count_controller_counts first dimension must match dataset length")

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, item: int):
        world_batch, target = self.base_dataset[item]
        policy_batch, _ = self.policy_dataset[item]
        true_future = self.raw_future_from_normalized(world_batch.future_actions)
        predicted_raw = self.generate_raw_actions(policy_batch, world_batch, true_future, item)
        if self.mode == "true_first_pred_rest":
            predicted_raw[0] = true_future[0]
        normalized_future = self.normalize_future_actions(predicted_raw)
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

    def generate_raw_actions(
        self,
        policy_batch: V6DualGraphBatch,
        world_batch: V6DualGraphBatch,
        true_future: torch.Tensor,
        item: int,
    ) -> torch.Tensor:
        if self.action_generator == "policy":
            return self.predict_raw_actions(policy_batch, true_future, item=item)
        if self.action_generator == "repeat_first_future":
            return repeat_first_future_actions(true_future)
        if self.action_generator == "repeat_last_history":
            raw_history = self.raw_history_from_normalized(world_batch.action_history)
            return repeat_last_history_actions(raw_history, horizon=true_future.shape[0])
        if self.action_generator == "true_future":
            return true_future.clone()
        if self.action_generator == "true_activity_policy_value":
            return self.predict_raw_actions(policy_batch, true_future, item=item, activity_override=true_future > 1e-9)
        if self.action_generator == "policy_activity_true_value":
            policy_active = self.predict_action_activity(policy_batch, true_future)
            return torch.where(policy_active.cpu(), true_future, torch.zeros_like(true_future))
        raise ValueError(f"Unknown action generator: {self.action_generator}")

    def predict_action_outputs(self, batch: V6DualGraphBatch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        batch = V6DualGraphBatch(
            node_history=batch.node_history.unsqueeze(0).to(self.device),
            physical_edge_history=batch.physical_edge_history.unsqueeze(0).to(self.device),
            info_edge_history=batch.info_edge_history.unsqueeze(0).to(self.device),
            action_history=batch.action_history.unsqueeze(0).to(self.device),
            future_actions=batch.future_actions.unsqueeze(0).to(self.device),
            task_history=batch.task_history.unsqueeze(0).to(self.device),
            link_rate_baseline=None,
        )
        with torch.no_grad():
            outputs = self.policy_model(batch)
            prob = torch.sigmoid(outputs["action_logit"]).squeeze(0)
            edge_prob = torch.sigmoid(outputs["edge_logit"]).squeeze(0) if "edge_logit" in outputs else None
            step_total_pred = torch.expm1(torch.clamp(outputs["step_total_log"], min=0.0)).squeeze(0) if "step_total_log" in outputs else None
            if "action_value_count_logit" in outputs:
                if self.value_vocab is None:
                    raise ValueError("Hierarchical-token policy checkpoint requires value_vocab")
                value = decode_hierarchical_policy_value(
                    outputs["action_value_count_logit"],
                    outputs["action_value_total_logit"],
                    self.value_vocab,
                ).squeeze(0)
            elif "action_value_token_logit" in outputs:
                if self.value_vocab is None:
                    raise ValueError("Coupled-token policy checkpoint requires value_vocab")
                value = decode_coupled_policy_value(outputs["action_value_token_logit"], self.value_vocab).squeeze(0)
            elif "action_value_bin_logit" in outputs:
                if self.value_vocab is None:
                    raise ValueError("Discrete policy checkpoint requires value_vocab")
                value = decode_discrete_policy_value(outputs["action_value_bin_logit"], self.value_vocab).squeeze(0)
            else:
                value = (outputs["action_value"] * self.action_scale).squeeze(0)
        return prob, value, edge_prob, step_total_pred

    def predict_action_activity(self, batch: V6DualGraphBatch, true_future: torch.Tensor) -> torch.Tensor:
        prob, _, edge_prob, _ = self.predict_action_outputs(batch)
        return self.decode_activity(prob, true_future.to(self.device), edge_prob=edge_prob)

    def predict_raw_actions(
        self,
        batch: V6DualGraphBatch,
        true_future: torch.Tensor,
        item: int = 0,
        activity_override: torch.Tensor | None = None,
    ) -> torch.Tensor:
        prob, policy_value, edge_prob, step_total_pred = self.predict_action_outputs(batch)
        if self.override_edge_prob is not None:
            edge_prob = torch.as_tensor(self.override_edge_prob[int(item)], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            active = (
                activity_override.to(self.device)
                if activity_override is not None
                else self.decode_activity(prob, true_future.to(self.device), edge_prob=edge_prob, item=item)
            )
            value = scale_action_value_groups(
                self.decode_action_value(policy_value),
                rb_dim_scale=self.rb_dim_scale,
                cpu_dim_scale=self.cpu_dim_scale,
            )
            raw = torch.where(active, value, torch.zeros_like(value))
            raw = apply_step_total_calibration(
                raw.unsqueeze(0),
                self.step_total_calibrator,
                step_total_pred=None if step_total_pred is None else step_total_pred.unsqueeze(0),
            ).squeeze(0)
        return raw.cpu()

    def decode_action_value(self, policy_value: torch.Tensor) -> torch.Tensor:
        if self.value_decoder_config.name == "policy":
            return policy_value * float(self.value_decoder_config.value_scale)
        if self.value_decoder_config.prototype is None:
            raise ValueError(f"{self.value_decoder_config.name} requires a value prototype")
        prototype = torch.as_tensor(self.value_decoder_config.prototype, dtype=policy_value.dtype, device=policy_value.device)
        if self.value_decoder_config.name == "train_codebook_quantile":
            return project_policy_value_to_codebook(policy_value, prototype) * float(self.value_decoder_config.value_scale)
        if prototype.ndim == 2:
            return prototype[:, None, :].expand_as(policy_value) * float(self.value_decoder_config.value_scale)
        if prototype.ndim == 3:
            if prototype.shape != policy_value.shape:
                raise ValueError(
                    f"Value prototype shape {tuple(prototype.shape)} does not match policy value shape {tuple(policy_value.shape)}"
                )
            return prototype * float(self.value_decoder_config.value_scale)
        raise ValueError("Value prototype must have shape [horizon, action_dim] or [horizon, edge, action_dim]")

    def decode_activity(
        self,
        prob: torch.Tensor,
        true_future: torch.Tensor,
        edge_prob: torch.Tensor | None = None,
        item: int = 0,
    ) -> torch.Tensor:
        if self.edge_count_controller_counts is not None:
            counts = self.edge_count_controller_counts[int(item)]
            return decode_edge_activity_topk(prob, counts, edge_prob=edge_prob)
        return self.decode_activity_static(prob, true_future, self.decoder_config, self.threshold, edge_prob=edge_prob)

    @staticmethod
    def decode_activity_static(
        prob: torch.Tensor,
        true_future: torch.Tensor,
        decoder_config: ActionDecoderConfig,
        threshold: float,
        edge_prob: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if decoder_config.name == "threshold":
            return prob >= float(threshold)
        if decoder_config.name == "edge_threshold":
            return decode_edge_threshold(prob, float(threshold), edge_prob=edge_prob)
        if decoder_config.name in {"val_mean_topk", "val_quantile_topk"}:
            if decoder_config.count_budget is None:
                raise ValueError(f"{decoder_config.name} requires count_budget")
            return decode_action_activity_topk(prob, decoder_config.count_budget)
        if decoder_config.name in {"edge_val_mean_topk", "edge_val_quantile_topk"}:
            if decoder_config.count_budget is None:
                raise ValueError(f"{decoder_config.name} requires count_budget")
            return decode_edge_activity_topk(prob, decoder_config.count_budget, edge_prob=edge_prob)
        if decoder_config.name == "edge_threshold_topk":
            if decoder_config.count_budget is None:
                raise ValueError("edge_threshold_topk requires count_budget")
            return decode_edge_threshold_topk(prob, decoder_config.count_budget, float(threshold), edge_prob=edge_prob)
        if decoder_config.name == "probability_mass_topk":
            if decoder_config.probability_budget_scales is None:
                raise ValueError("probability_mass_topk requires probability_budget_scales")
            counts = predict_counts_from_probability_mass(prob, decoder_config.probability_budget_scales)
            return decode_action_activity_topk(prob, counts)
        if decoder_config.name == "edge_probability_mass_topk":
            if decoder_config.probability_budget_scales is None:
                raise ValueError("edge_probability_mass_topk requires probability_budget_scales")
            counts = predict_edge_counts_from_probability_mass(prob, decoder_config.probability_budget_scales, edge_prob=edge_prob)
            return decode_edge_activity_topk(prob, counts, edge_prob=edge_prob)
        if decoder_config.name == "oracle_topk":
            counts = (true_future > 1e-9).sum(dim=1).to(torch.long)
            return decode_action_activity_topk(prob, counts)
        raise ValueError(f"Unknown action decoder: {decoder_config.name}")

    def raw_future_from_normalized(self, normalized_future: torch.Tensor) -> torch.Tensor:
        mean, std = self.stats["edge_a_future"]
        mean_t = torch.as_tensor(mean[0], dtype=torch.float32)
        std_t = torch.as_tensor(std[0], dtype=torch.float32)
        return normalized_future * std_t + mean_t

    def raw_history_from_normalized(self, normalized_history: torch.Tensor) -> torch.Tensor:
        mean, std = self.stats["edge_a_hist"]
        mean_t = torch.as_tensor(mean[0], dtype=torch.float32)
        std_t = torch.as_tensor(std[0], dtype=torch.float32)
        return normalized_history * std_t + mean_t

    def normalize_future_actions(self, raw_future: torch.Tensor) -> torch.Tensor:
        mean, std = self.stats["edge_a_future"]
        mean_t = torch.as_tensor(mean[0], dtype=torch.float32)
        std_t = torch.as_tensor(std[0], dtype=torch.float32)
        return ((raw_future - mean_t) / std_t).to(torch.float32)


def evaluate_policy_bridge(
    world_experiment_dir: Path,
    world_checkpoint: Path | None,
    policy_checkpoint: Path,
    output_json: Path,
    device: torch.device,
    batch_size: int,
    policy_threshold: float | None,
    mode: str,
    action_decoder: str,
    budget_quantile: float,
    action_generator: str,
    value_decoder: str,
    value_quantile: float,
    value_codebook_size: int,
    value_scale: float,
    rb_dim_scale: float = 1.0,
    cpu_dim_scale: float = 1.0,
    step_total_calibrator_name: str = "none",
    step_total_quantile: float = 0.5,
    max_val_samples: int = 0,
    max_test_samples: int = 0,
    pretrim_arrays: bool = False,
    active_aware_sample_limit: bool = False,
    edge_reranker: str = "none",
    edge_reranker_max_train_rows: int = 200000,
    edge_count_controller: str = "none",
    edge_count_offset: int = 0,
) -> dict:
    summary = json.loads((world_experiment_dir / "v8_full_training_summary.json").read_text(encoding="utf-8"))
    dataset_dir = resolve_project_path(summary["dataset_dir"])
    split = summary["split_seed_spec"]
    arrays, train_idx, val_idx, test_idx = load_bridge_arrays_and_splits(
        dataset_dir,
        split,
        max_val_samples=max_val_samples,
        max_test_samples=max_test_samples,
        pretrim_arrays=pretrim_arrays,
        active_aware_sample_limit=active_aware_sample_limit,
    )
    if summary["config"].get("use_event_memory_features", False):
        arrays = add_event_memory_features(arrays)
    stats = make_normalization_stats(arrays, train_idx)
    world_checkpoint = world_checkpoint or world_experiment_dir / "checkpoints" / "v8_dual_best.pt"
    world_model = load_model_for_experiment(summary, arrays, world_checkpoint, device)
    policy_model, action_scale, learned_threshold, value_vocab = load_policy(policy_checkpoint, device)
    threshold = learned_threshold if policy_threshold is None else float(policy_threshold)

    val_base = V6WorldModelDataset(arrays, val_idx, stats)
    test_base = V6WorldModelDataset(arrays, test_idx, stats)
    val_policy = V7ActionPolicyDataset(arrays, val_idx, stats, action_scale)
    test_policy = V7ActionPolicyDataset(arrays, test_idx, stats, action_scale)
    action_scale_t = torch.as_tensor(action_scale.reshape(1, 1, 1, -1), dtype=torch.float32, device=device)
    val_policy_loader = DataLoader(
        val_policy,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_action_policy_batch,
    )
    val_policy_predictions = collect_bridge_policy_predictions(policy_model, val_policy_loader, device, action_scale_t, value_vocab)
    edge_reranker_info = {"name": edge_reranker}
    val_edge_prob_for_decode = val_policy_predictions.get("edge_prob")
    edge_reranker_model = None
    if edge_reranker == "logistic_val":
        edge_reranker_model = fit_logistic_edge_reranker(
            val_policy_predictions,
            max_train_rows=edge_reranker_max_train_rows,
        )
        val_edge_prob_for_decode = predict_edge_reranker_scores(edge_reranker_model, val_policy_predictions)
        true_edge = np.any(np.asarray(val_policy_predictions["value_true"], dtype=np.float32) > 1e-9, axis=-1)
        if policy_threshold is None:
            threshold = choose_threshold_by_f1(val_edge_prob_for_decode, true_edge)
        edge_reranker_info.update(
            {
                "max_train_rows": int(edge_reranker_max_train_rows),
                "selected_threshold": float(threshold),
            }
        )
    elif edge_reranker != "none":
        raise ValueError(f"Unknown edge reranker: {edge_reranker}")
    edge_count_info = {"name": edge_count_controller}
    edge_count_model = None
    val_edge_counts = None
    if edge_count_controller == "hgb_val":
        val_for_count = predictions_with_edge_prob(val_policy_predictions, val_edge_prob_for_decode)
        edge_count_model = fit_edge_count_regressor(val_for_count, seed=20260629)
        val_edge_counts = predict_edge_count_regressor(edge_count_model, val_for_count)
        val_edge_counts = apply_edge_count_offset(val_edge_counts, edge_count_offset, val_policy_predictions["prob"].shape[2])
        true_counts = np.any(np.asarray(val_policy_predictions["value_true"], dtype=np.float32) > 1e-9, axis=-1).sum(axis=2)
        edge_count_info.update(
            {
                "count_offset": int(edge_count_offset),
                "train_rows": int(val_edge_counts.size),
                "pred_count_min": int(np.min(val_edge_counts)) if val_edge_counts.size else 0,
                "pred_count_max": int(np.max(val_edge_counts)) if val_edge_counts.size else 0,
                "pred_count_mean": float(np.mean(val_edge_counts)) if val_edge_counts.size else 0.0,
                "true_count_mean": float(np.mean(true_counts)) if true_counts.size else 0.0,
            }
        )
    elif edge_count_controller != "none":
        raise ValueError(f"Unknown edge count controller: {edge_count_controller}")
    decoder_config = make_action_decoder_config(action_decoder, val_policy_predictions, budget_quantile)
    value_decoder_config = make_action_value_decoder_config(
        value_decoder,
        arrays,
        train_idx,
        value_quantile,
        value_codebook_size,
        value_scale,
        val_predictions=val_policy_predictions,
        policy_threshold=threshold,
    )
    val_pred_actions_for_calibration = decode_policy_predictions_to_actions_np(
        val_policy_predictions,
        decoder_config,
        value_decoder_config,
        threshold,
        rb_dim_scale=rb_dim_scale,
        cpu_dim_scale=cpu_dim_scale,
        override_edge_prob=val_edge_prob_for_decode,
        edge_count_controller_counts=val_edge_counts,
    )
    step_total_calibrator = make_step_total_calibrator(
        step_total_calibrator_name,
        val_pred_actions_for_calibration,
        val_policy_predictions["value_true"],
        quantile=step_total_quantile,
    )
    test_edge_prob_for_decode = None
    test_edge_counts = None
    if edge_reranker_model is not None or edge_count_model is not None:
        test_policy_loader_for_decode = DataLoader(
            test_policy,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_action_policy_batch,
        )
        test_policy_predictions = collect_bridge_policy_predictions(
            policy_model,
            test_policy_loader_for_decode,
            device,
            action_scale_t,
            value_vocab,
        )
        test_edge_prob_for_decode = test_policy_predictions.get("edge_prob")
        if edge_reranker_model is not None:
            test_edge_prob_for_decode = predict_edge_reranker_scores(edge_reranker_model, test_policy_predictions)
        if edge_count_model is not None:
            test_edge_counts = predict_edge_count_regressor(
                edge_count_model,
                predictions_with_edge_prob(test_policy_predictions, test_edge_prob_for_decode),
            )
            test_edge_counts = apply_edge_count_offset(test_edge_counts, edge_count_offset, test_policy_predictions["prob"].shape[2])
            edge_count_info.update(
                {
                    "test_pred_count_min": int(np.min(test_edge_counts)) if test_edge_counts.size else 0,
                    "test_pred_count_max": int(np.max(test_edge_counts)) if test_edge_counts.size else 0,
                    "test_pred_count_mean": float(np.mean(test_edge_counts)) if test_edge_counts.size else 0.0,
                }
            )
    val_loader = DataLoader(
        PolicyBridgeDataset(
            val_base,
            val_policy,
            policy_model,
            action_scale,
            stats,
            device,
            threshold,
            mode,
            decoder_config,
            action_generator,
            value_decoder_config,
            value_vocab,
            rb_dim_scale=rb_dim_scale,
            cpu_dim_scale=cpu_dim_scale,
            step_total_calibrator=step_total_calibrator,
            override_edge_prob=val_edge_prob_for_decode,
            edge_count_controller_counts=val_edge_counts,
        ),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_v6_world_model_batch,
    )
    test_loader = DataLoader(
        PolicyBridgeDataset(
            test_base,
            test_policy,
            policy_model,
            action_scale,
            stats,
            device,
            threshold,
            mode,
            decoder_config,
            action_generator,
            value_decoder_config,
            value_vocab,
            rb_dim_scale=rb_dim_scale,
            cpu_dim_scale=cpu_dim_scale,
            step_total_calibrator=step_total_calibrator,
            override_edge_prob=test_edge_prob_for_decode,
            edge_count_controller_counts=test_edge_counts,
        ),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_v6_world_model_batch,
    )
    config = summary["config"]
    val_metrics = evaluate_v8_model(
        world_model,
        val_loader,
        device,
        stats,
        rate_output_mode=config.get("rate_output_mode", "main"),
        inactive_rate_value=float(config.get("inactive_rate_value", 0.0)),
        hurdle_gate_temperature=float(config.get("eval_hurdle_gate_temperature", 1.0)),
        hurdle_gate_power=float(config.get("eval_hurdle_gate_power", 1.0)),
    )
    activity_threshold = val_metrics["activity"]["threshold"]
    test_metrics = evaluate_v8_model(
        world_model,
        test_loader,
        device,
        stats,
        activity_threshold=activity_threshold,
        rate_output_mode=config.get("rate_output_mode", "main"),
        inactive_rate_value=float(config.get("inactive_rate_value", 0.0)),
        hurdle_gate_temperature=float(config.get("eval_hurdle_gate_temperature", 1.0)),
        hurdle_gate_power=float(config.get("eval_hurdle_gate_power", 1.0)),
    )
    result = {
        "result_protocol": make_bridge_result_protocol(action_generator, decoder_config.name, mode),
        "mode": mode,
        "policy_checkpoint": str(policy_checkpoint),
        "world_experiment_dir": str(world_experiment_dir),
        "world_checkpoint": str(world_checkpoint),
        "policy_threshold": float(threshold),
        "action_decoder": decoder_config.to_json(),
        "action_generator": action_generator,
        "value_decoder": value_decoder_config.to_json(),
        "rb_dim_scale": float(rb_dim_scale),
        "cpu_dim_scale": float(cpu_dim_scale),
        "step_total_calibrator": step_total_calibrator.to_json(),
        "edge_reranker": edge_reranker_info,
        "edge_count_controller": edge_count_info,
        "eval_sample_limits": {
            "max_val_samples": int(max_val_samples),
            "max_test_samples": int(max_test_samples),
            "val_used": int(len(val_idx)),
            "test_used": int(len(test_idx)),
            "pretrim_arrays": bool(pretrim_arrays),
            "active_aware_sample_limit": bool(active_aware_sample_limit),
        },
        "val_threshold": float(activity_threshold),
        "val": val_metrics,
        "test": test_metrics,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    args = parse_args()
    result = evaluate_policy_bridge(
        args.world_experiment_dir,
        args.world_checkpoint,
        args.policy_checkpoint,
        args.output_json,
        choose_device(args.device),
        args.batch_size,
        args.policy_threshold,
        args.mode,
        args.action_decoder,
        args.budget_quantile,
        args.action_generator,
        args.value_decoder,
        args.value_quantile,
        args.value_codebook_size,
        args.value_scale,
        args.rb_dim_scale,
        args.cpu_dim_scale,
        args.step_total_calibrator,
        args.step_total_quantile,
        args.max_val_samples,
        args.max_test_samples,
        args.pretrim_arrays,
        args.active_aware_sample_limit,
        args.edge_reranker,
        args.edge_reranker_max_train_rows,
        args.edge_count_controller,
        args.edge_count_offset,
    )
    test = result["test"]
    print(
        f"mode={result['mode']} generator={result['action_generator']} decoder={result['action_decoder']['name']} "
        f"value={result['value_decoder']['name']} "
        f"active_rate_rmse={test['active_rate']['active_rmse']:.6f} "
        f"f1={test['activity']['f1']:.6f} link_rmse={test['link_rate']['rmse']:.6f}"
    )
    print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()
