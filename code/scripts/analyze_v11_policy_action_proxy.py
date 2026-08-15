"""Lightweight PI-JWM v11 policy-action proxy diagnostics.

This script evaluates generated scheduler actions directly, without running the
frozen world model. It is a memory-bounded CPU gate for bridge experiments.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from pi_jwm.v6_data import load_world_model_arrays, make_normalization_stats

from evaluate_v10_policy_bridge import (
    ActionDecoderConfig,
    ActionValueDecoderConfig,
    decode_edge_activity_topk_np,
    decode_edge_threshold_np,
    decode_edge_threshold_topk_np,
    apply_step_total_calibration_np,
    collect_bridge_policy_predictions,
    decode_action_activity_topk_np,
    fit_action_value_prototype,
    limit_eval_indices,
    load_policy,
    make_action_decoder_config,
    make_step_total_calibrator,
)
from run_v7_action_policy import V7ActionPolicyDataset, collate_action_policy_batch, parse_seed_list, resolve_policy_seed_splits


DEFAULT_DATASET_DIR = PROJECT_ROOT / "artifacts/experiments/airfogsim_v0/datasets/world_model_dataset_active_heavy_v2_60seed_20260619"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts/experiments/pi_jwm_v11_policy_action_proxy_20260629"


RB_TOTAL_DIM = 2
CPU_TOTAL_DIM = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument(
        "--policy-checkpoint",
        type=Path,
        action="append",
        required=True,
        help="Policy checkpoint. Repeat to average support/value predictions as a small checkpoint ensemble.",
    )
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_DIR / "summary.json")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-seeds", default=None)
    parser.add_argument("--val-seeds", default=None)
    parser.add_argument("--test-seeds", default=None)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument(
        "--max-budget-samples",
        type=int,
        default=0,
        help="Validation samples used to fit count/probability budgets; <=0 uses the full validation split.",
    )
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
    )
    parser.add_argument("--budget-quantile", type=float, default=0.5)
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
    )
    parser.add_argument("--value-quantile", type=float, default=0.75)
    parser.add_argument("--value-codebook-size", type=int, default=5)
    parser.add_argument("--value-scale", type=float, default=1.0)
    parser.add_argument("--rb-dim-scale", type=float, default=1.0)
    parser.add_argument("--cpu-dim-scale", type=float, default=1.0)
    parser.add_argument("--step-total-calibrator", choices=("none", "val_count_quantile", "policy_step_total"), default="none")
    parser.add_argument("--step-total-quantile", type=float, default=0.5)
    parser.add_argument("--edge-reranker", choices=("none", "logistic_val"), default="none")
    parser.add_argument("--edge-reranker-max-train-rows", type=int, default=200000)
    parser.add_argument("--edge-count-controller", choices=("none", "hgb_val"), default="none")
    return parser.parse_args()


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def binary_metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    pred = np.asarray(pred, dtype=bool)
    true = np.asarray(true, dtype=bool)
    tp = int(np.sum(pred & true))
    fp = int(np.sum(pred & ~true))
    fn = int(np.sum(~pred & true))
    tn = int(np.sum(~pred & ~true))
    precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
    f1 = float(2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": float(tp), "fp": float(fp), "fn": float(fn), "tn": float(tn)}


def step_total_rmse(pred: np.ndarray, true: np.ndarray, dim: int) -> float:
    pred_total = np.asarray(pred, dtype=np.float32)[..., int(dim)].sum(axis=2)
    true_total = np.asarray(true, dtype=np.float32)[..., int(dim)].sum(axis=2)
    return float(np.sqrt(np.mean((pred_total - true_total) ** 2)))


def safe_ratio(num: float, den: float) -> float:
    return float(num / den) if abs(float(den)) > 1e-9 else float("nan")


def compute_action_proxy_metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    pred = np.asarray(pred, dtype=np.float32)
    true = np.asarray(true, dtype=np.float32)
    if pred.shape != true.shape:
        raise ValueError("pred and true actions must have the same shape")
    pred_active = np.any(pred > 1e-9, axis=-1)
    true_active = np.any(true > 1e-9, axis=-1)
    activity = binary_metrics(pred_active, true_active)
    rb_pred_sum = float(np.sum(pred[..., RB_TOTAL_DIM]))
    rb_true_sum = float(np.sum(true[..., RB_TOTAL_DIM]))
    cpu_pred_sum = float(np.sum(pred[..., CPU_TOTAL_DIM]))
    cpu_true_sum = float(np.sum(true[..., CPU_TOTAL_DIM]))
    active_mask = true_active
    rb_active_rmse = (
        float(np.sqrt(np.mean((pred[..., RB_TOTAL_DIM][active_mask] - true[..., RB_TOTAL_DIM][active_mask]) ** 2)))
        if np.any(active_mask)
        else float("nan")
    )
    result = {
        "activity_precision": activity["precision"],
        "activity_recall": activity["recall"],
        "activity_f1": activity["f1"],
        "activity_tp": activity["tp"],
        "activity_fp": activity["fp"],
        "activity_fn": activity["fn"],
        "activity_tn": activity["tn"],
        "rb_total_sum_ratio": safe_ratio(rb_pred_sum, rb_true_sum),
        "cpu_total_sum_ratio": safe_ratio(cpu_pred_sum, cpu_true_sum),
        "rb_total_step_rmse": step_total_rmse(pred, true, RB_TOTAL_DIM),
        "cpu_total_step_rmse": step_total_rmse(pred, true, CPU_TOTAL_DIM),
        "rb_total_active_rmse": rb_active_rmse,
        "pred_active_count": float(np.sum(pred_active)),
        "true_active_count": float(np.sum(true_active)),
        "rb_pred_sum": rb_pred_sum,
        "rb_true_sum": rb_true_sum,
        "cpu_pred_sum": cpu_pred_sum,
        "cpu_true_sum": cpu_true_sum,
    }
    return result


def average_policy_predictions(prediction_rows: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not prediction_rows:
        raise ValueError("prediction_rows must not be empty")
    result: dict[str, np.ndarray] = {}
    for key in ("prob", "edge_prob", "value_pred", "step_total_pred"):
        present = [np.asarray(row[key], dtype=np.float32) for row in prediction_rows if key in row]
        if present:
            reference_shape = present[0].shape
            if any(row.shape != reference_shape for row in present):
                raise ValueError(f"all {key} arrays must have the same shape")
            result[key] = np.mean(np.stack(present, axis=0), axis=0).astype(np.float32)
    for key in ("active", "value_true"):
        if key in prediction_rows[0]:
            result[key] = np.asarray(prediction_rows[0][key])
            for row in prediction_rows[1:]:
                if key in row and not np.array_equal(result[key], row[key]):
                    raise ValueError(f"all {key} arrays must match")
    return result


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
    if value.shape[-1] > RB_TOTAL_DIM:
        features.append(value[..., RB_TOTAL_DIM])
    if value.shape[-1] > CPU_TOTAL_DIM:
        features.append(value[..., CPU_TOTAL_DIM])
    return np.stack(features, axis=-1).reshape(-1, len(features)).astype(np.float32)


def choose_threshold_by_f1(
    scores: np.ndarray,
    true: np.ndarray,
    candidates: np.ndarray | None = None,
) -> float:
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    true = np.asarray(true, dtype=bool).reshape(-1)
    if candidates is None:
        candidates = np.unique(np.quantile(scores, np.linspace(0.0, 1.0, 101))).astype(np.float32)
    best_threshold = float(candidates[0]) if len(candidates) else 0.5
    best_f1 = -1.0
    for threshold in np.asarray(candidates, dtype=np.float32).reshape(-1):
        f1 = binary_metrics(scores >= float(threshold), true)["f1"]
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)
    return best_threshold


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
    top1 = sorted_score[:, :, -1]
    top3 = sorted_score[:, :, -min(3, edges) :].sum(axis=2)
    top8 = sorted_score[:, :, -min(8, edges) :].sum(axis=2)
    step = np.broadcast_to(
        (np.arange(horizon, dtype=np.float32) / max(horizon - 1, 1)).reshape(1, horizon),
        (samples, horizon),
    )
    features = [
        edge_score.sum(axis=2),
        edge_score.mean(axis=2),
        edge_score.std(axis=2),
        top1,
        top3,
        top8,
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
        features.extend(
            [
                np.clip(value[..., RB_TOTAL_DIM], 0.0, None).sum(axis=2),
                np.clip(value[..., CPU_TOTAL_DIM], 0.0, None).sum(axis=2),
                value.max(axis=(2, 3)),
                value.mean(axis=(2, 3)),
            ]
        )
    return np.stack(features, axis=-1).reshape(samples * horizon, len(features)).astype(np.float32)


def _edge_count_target(predictions: dict[str, np.ndarray]) -> np.ndarray:
    true_value = np.asarray(predictions["value_true"], dtype=np.float32)
    if true_value.ndim != 4:
        raise ValueError("value_true must have shape [sample, horizon, edge, action_dim]")
    return np.any(true_value > 1e-9, axis=-1).sum(axis=2).astype(np.float32).reshape(-1)


def fit_edge_count_regressor(predictions: dict[str, np.ndarray], seed: int = 20260629) -> dict:
    features = build_edge_count_features(predictions)
    target = _edge_count_target(predictions)
    num_edges = int(np.asarray(predictions["prob"]).shape[2])
    horizon = int(np.asarray(predictions["prob"]).shape[1])
    unique = np.unique(target)
    if unique.size <= 1:
        return {"model": None, "constant": float(unique[0]) if unique.size else 0.0, "num_edges": num_edges, "horizon": horizon}
    model = HistGradientBoostingRegressor(
        max_iter=100,
        max_leaf_nodes=15,
        learning_rate=0.05,
        l2_regularization=0.01,
        random_state=int(seed),
    )
    model.fit(features, target)
    return {"model": model, "constant": None, "num_edges": num_edges, "horizon": horizon}


def predict_edge_count_regressor(model_info: dict, predictions: dict[str, np.ndarray]) -> np.ndarray:
    prob = np.asarray(predictions["prob"], dtype=np.float32)
    if model_info.get("model") is None:
        raw = np.full((prob.shape[0] * prob.shape[1],), float(model_info.get("constant", 0.0)), dtype=np.float32)
    else:
        raw = np.asarray(model_info["model"].predict(build_edge_count_features(predictions)), dtype=np.float32)
    counts = np.rint(raw).reshape(prob.shape[0], prob.shape[1]).astype(np.int64)
    return np.clip(counts, 0, prob.shape[2])


def predictions_with_edge_prob(predictions: dict[str, np.ndarray], edge_prob: np.ndarray | None) -> dict[str, np.ndarray]:
    if edge_prob is None:
        return predictions
    updated = dict(predictions)
    updated["edge_prob"] = np.asarray(edge_prob, dtype=np.float32)
    return updated


def decode_edge_count_controlled_activity(
    prob: np.ndarray,
    edge_counts: np.ndarray,
    edge_prob: np.ndarray | None = None,
) -> np.ndarray:
    edge_active = decode_edge_activity_topk_np(prob, edge_counts, edge_prob=edge_prob)
    return edge_active[:, :, :, None].repeat(prob.shape[3], axis=-1)


def make_value_decoder_config(
    name: str,
    arrays: dict[str, np.ndarray],
    train_idx: np.ndarray,
    value_quantile: float,
    value_codebook_size: int,
    value_scale: float,
) -> ActionValueDecoderConfig:
    if name == "policy":
        return ActionValueDecoderConfig(name=name, value_quantile=value_quantile, value_codebook_size=value_codebook_size, value_scale=value_scale)
    prototype = fit_action_value_prototype(
        arrays["edge_a_future"][train_idx],
        decoder=name,
        value_quantile=value_quantile,
        value_codebook_size=value_codebook_size,
    )
    return ActionValueDecoderConfig(
        name=name,
        prototype=prototype,
        value_quantile=value_quantile,
        value_codebook_size=value_codebook_size,
        value_scale=value_scale,
    )


def decode_values(policy_value: np.ndarray, value_config: ActionValueDecoderConfig) -> np.ndarray:
    policy_value = np.asarray(policy_value, dtype=np.float32)
    if value_config.name == "policy":
        return policy_value * float(value_config.value_scale)
    if value_config.prototype is None:
        raise ValueError(f"{value_config.name} requires prototype")
    prototype = np.asarray(value_config.prototype, dtype=np.float32)
    if prototype.ndim == 2:
        return np.broadcast_to(prototype[None, :, None, :], policy_value.shape).astype(np.float32) * float(value_config.value_scale)
    if prototype.ndim == 3:
        if value_config.name == "train_codebook_quantile":
            return project_policy_value_to_codebook_np(policy_value, prototype) * float(value_config.value_scale)
        return np.broadcast_to(prototype[None, :, :, :], policy_value.shape).astype(np.float32) * float(value_config.value_scale)
    if prototype.ndim == 4:
        return prototype.astype(np.float32) * float(value_config.value_scale)
    raise ValueError("unsupported prototype shape")


def apply_group_value_scales(values: np.ndarray, rb_dim_scale: float, cpu_dim_scale: float) -> np.ndarray:
    scaled = np.asarray(values, dtype=np.float32).copy()
    scaled[..., 1] *= float(rb_dim_scale)
    scaled[..., 2] *= float(rb_dim_scale)
    scaled[..., 3] *= float(cpu_dim_scale)
    scaled[..., 4] *= float(cpu_dim_scale)
    return scaled


def project_policy_value_to_codebook_np(policy_value: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    policy_value = np.asarray(policy_value, dtype=np.float32)
    codebook = np.asarray(codebook, dtype=np.float32)
    if policy_value.ndim != 4:
        raise ValueError("policy_value must have shape [sample, horizon, edge, action_dim]")
    if codebook.ndim != 3:
        raise ValueError("codebook must have shape [horizon, action_dim, codebook_size]")
    if codebook.shape[:2] != (policy_value.shape[1], policy_value.shape[3]):
        raise ValueError("codebook first dimensions must match policy_value [horizon, action_dim]")
    distances = np.abs(policy_value[..., None] - codebook[None, :, None, :, :])
    nearest = np.argmin(distances, axis=-1)
    expanded = np.broadcast_to(codebook[None, :, None, :, :], (*policy_value.shape, codebook.shape[-1]))
    return np.take_along_axis(expanded, nearest[..., None], axis=-1).squeeze(-1).astype(np.float32)


def decode_activity_np(
    prob: np.ndarray,
    true_value: np.ndarray,
    decoder_config: ActionDecoderConfig,
    threshold: float,
    edge_prob: np.ndarray | None = None,
) -> np.ndarray:
    if decoder_config.name == "threshold":
        return np.asarray(prob, dtype=np.float32) >= float(threshold)
    if decoder_config.name == "edge_threshold":
        edge_active = decode_edge_threshold_np(prob, threshold, edge_prob=edge_prob)
        return edge_active[:, :, :, None].repeat(prob.shape[3], axis=-1)
    if decoder_config.name in {"val_mean_topk", "val_quantile_topk"}:
        counts = np.asarray(decoder_config.count_budget, dtype=np.int64)
        expanded = np.broadcast_to(counts[None, :, :], (prob.shape[0], prob.shape[1], prob.shape[3]))
        return decode_action_activity_topk_np(prob, expanded)
    if decoder_config.name in {"edge_val_mean_topk", "edge_val_quantile_topk"}:
        counts = np.asarray(decoder_config.count_budget, dtype=np.int64)
        expanded = np.broadcast_to(counts[None, :], (prob.shape[0], prob.shape[1]))
        return decode_edge_activity_topk_np(prob, expanded, edge_prob=edge_prob)[:, :, :, None].repeat(prob.shape[3], axis=-1)
    if decoder_config.name == "edge_threshold_topk":
        counts = np.asarray(decoder_config.count_budget, dtype=np.int64)
        expanded = np.broadcast_to(counts[None, :], (prob.shape[0], prob.shape[1]))
        edge_active = decode_edge_threshold_topk_np(prob, expanded, threshold, edge_prob=edge_prob)
        return edge_active[:, :, :, None].repeat(prob.shape[3], axis=-1)
    if decoder_config.name == "oracle_topk":
        counts = (np.asarray(true_value) > 1e-9).sum(axis=2)
        return decode_action_activity_topk_np(prob, counts)
    if decoder_config.name == "probability_mass_topk":
        if decoder_config.probability_budget_scales is None:
            raise ValueError("probability_mass_topk requires scales")
        counts = np.rint(prob.sum(axis=2) * np.asarray(decoder_config.probability_budget_scales)[None, :, :]).astype(np.int64)
        return decode_action_activity_topk_np(prob, np.clip(counts, 0, prob.shape[2]))
    if decoder_config.name == "edge_probability_mass_topk":
        if decoder_config.probability_budget_scales is None:
            raise ValueError("edge_probability_mass_topk requires scales")
        edge_score = prob.max(axis=-1)
        if edge_prob is not None:
            edge_score = np.asarray(edge_prob, dtype=np.float32)
        counts = np.rint(edge_score.sum(axis=2) * np.asarray(decoder_config.probability_budget_scales)[None, :]).astype(np.int64)
        edge_active = decode_edge_activity_topk_np(prob, np.clip(counts, 0, prob.shape[2]), edge_prob=edge_prob)
        return edge_active[:, :, :, None].repeat(prob.shape[3], axis=-1)
    raise ValueError(f"unknown decoder: {decoder_config.name}")


def run(args: argparse.Namespace) -> dict:
    device = choose_device(args.device)
    arrays = load_world_model_arrays(args.dataset_dir)
    train_idx, val_idx, test_idx, split_spec = resolve_policy_seed_splits(
        arrays["sample_seed"],
        train_seeds=parse_seed_list(args.train_seeds),
        val_seeds=parse_seed_list(args.val_seeds),
        test_seeds=parse_seed_list(args.test_seeds),
    )
    eval_idx = val_idx if args.split == "val" else test_idx
    eval_idx = limit_eval_indices(eval_idx, args.max_samples)
    stats = make_normalization_stats(arrays, train_idx)
    checkpoint_paths = [Path(path) for path in args.policy_checkpoint]
    loaded = [load_policy(path, device) for path in checkpoint_paths]
    model, action_scale, learned_threshold, value_vocab = loaded[0]
    learned_thresholds = [float(item[2]) for item in loaded]
    threshold = float(np.mean(learned_thresholds)) if args.policy_threshold is None else float(args.policy_threshold)

    budget_val_idx = limit_eval_indices(val_idx, args.max_budget_samples)
    val_policy = V7ActionPolicyDataset(arrays, budget_val_idx, stats, action_scale)
    val_loader = DataLoader(val_policy, batch_size=args.batch_size, shuffle=False, collate_fn=collate_action_policy_batch)
    val_prediction_rows = []
    for model_i, action_scale_i, _, value_vocab_i in loaded:
        action_scale_t = torch.as_tensor(action_scale_i.reshape(1, 1, 1, -1), dtype=torch.float32, device=device)
        val_prediction_rows.append(collect_bridge_policy_predictions(model_i, val_loader, device, action_scale_t, value_vocab_i))
    val_predictions = average_policy_predictions(val_prediction_rows)
    edge_reranker_info = {"name": args.edge_reranker}
    val_edge_prob_for_decode = val_predictions.get("edge_prob")
    edge_reranker_model = None
    if args.edge_reranker == "logistic_val":
        edge_reranker_model = fit_logistic_edge_reranker(
            val_predictions,
            max_train_rows=args.edge_reranker_max_train_rows,
        )
        val_edge_prob_for_decode = predict_edge_reranker_scores(edge_reranker_model, val_predictions)
        true_edge = np.any(np.asarray(val_predictions["value_true"], dtype=np.float32) > 1e-9, axis=-1)
        if args.policy_threshold is None:
            threshold = choose_threshold_by_f1(val_edge_prob_for_decode, true_edge)
        edge_reranker_info.update(
            {
                "max_train_rows": int(args.edge_reranker_max_train_rows),
                "selected_threshold": float(threshold),
            }
        )
    edge_count_info = {"name": args.edge_count_controller}
    edge_count_model = None
    if args.edge_count_controller == "hgb_val":
        val_for_count = predictions_with_edge_prob(val_predictions, val_edge_prob_for_decode)
        edge_count_model = fit_edge_count_regressor(val_for_count, seed=20260629)
        val_count_pred = predict_edge_count_regressor(edge_count_model, val_for_count)
        val_count_true = np.any(np.asarray(val_predictions["value_true"], dtype=np.float32) > 1e-9, axis=-1).sum(axis=2)
        edge_count_info.update(
            {
                "train_rows": int(val_count_pred.size),
                "pred_count_min": int(np.min(val_count_pred)) if val_count_pred.size else 0,
                "pred_count_max": int(np.max(val_count_pred)) if val_count_pred.size else 0,
                "pred_count_mean": float(np.mean(val_count_pred)) if val_count_pred.size else 0.0,
                "true_count_mean": float(np.mean(val_count_true)) if val_count_true.size else 0.0,
            }
        )
    decoder_config = make_action_decoder_config(args.action_decoder, val_predictions, args.budget_quantile)
    value_config = make_value_decoder_config(args.value_decoder, arrays, train_idx, args.value_quantile, args.value_codebook_size, args.value_scale)
    val_pred_value = decode_values(val_predictions["value_pred"], value_config)
    val_pred_value = apply_group_value_scales(val_pred_value, args.rb_dim_scale, args.cpu_dim_scale)
    if edge_count_model is not None:
        val_counts = predict_edge_count_regressor(edge_count_model, predictions_with_edge_prob(val_predictions, val_edge_prob_for_decode))
        val_pred_active = decode_edge_count_controlled_activity(val_predictions["prob"], val_counts, edge_prob=val_edge_prob_for_decode)
    else:
        val_pred_active = decode_activity_np(
            val_predictions["prob"],
            val_predictions["value_true"],
            decoder_config,
            threshold,
            edge_prob=val_edge_prob_for_decode,
        )
    val_pred_actions = np.where(val_pred_active, val_pred_value, 0.0).astype(np.float32)
    step_total_calibrator = make_step_total_calibrator(
        args.step_total_calibrator,
        val_pred_actions,
        val_predictions["value_true"],
        quantile=args.step_total_quantile,
    )

    eval_policy = V7ActionPolicyDataset(arrays, eval_idx, stats, action_scale)
    eval_loader = DataLoader(eval_policy, batch_size=args.batch_size, shuffle=False, collate_fn=collate_action_policy_batch)
    prediction_rows = []
    for model_i, action_scale_i, _, value_vocab_i in loaded:
        action_scale_t = torch.as_tensor(action_scale_i.reshape(1, 1, 1, -1), dtype=torch.float32, device=device)
        prediction_rows.append(collect_bridge_policy_predictions(model_i, eval_loader, device, action_scale_t, value_vocab_i))
    predictions = average_policy_predictions(prediction_rows)
    edge_prob_for_decode = predictions.get("edge_prob")
    if edge_reranker_model is not None:
        edge_prob_for_decode = predict_edge_reranker_scores(edge_reranker_model, predictions)
    pred_value = decode_values(predictions["value_pred"], value_config)
    pred_value = apply_group_value_scales(pred_value, args.rb_dim_scale, args.cpu_dim_scale)
    if edge_count_model is not None:
        pred_counts = predict_edge_count_regressor(edge_count_model, predictions_with_edge_prob(predictions, edge_prob_for_decode))
        pred_active = decode_edge_count_controlled_activity(predictions["prob"], pred_counts, edge_prob=edge_prob_for_decode)
        edge_count_info.update(
            {
                "eval_pred_count_min": int(np.min(pred_counts)) if pred_counts.size else 0,
                "eval_pred_count_max": int(np.max(pred_counts)) if pred_counts.size else 0,
                "eval_pred_count_mean": float(np.mean(pred_counts)) if pred_counts.size else 0.0,
            }
        )
    else:
        pred_active = decode_activity_np(
            predictions["prob"],
            predictions["value_true"],
            decoder_config,
            threshold,
            edge_prob=edge_prob_for_decode,
        )
    pred_actions = np.where(pred_active, pred_value, 0.0).astype(np.float32)
    pred_actions = apply_step_total_calibration_np(
        pred_actions,
        step_total_calibrator,
        step_total_pred=predictions.get("step_total_pred"),
    )
    metrics = compute_action_proxy_metrics(pred_actions, predictions["value_true"])
    result = {
        "framework": "PI-JWM",
        "candidate": "v11",
        "mode": "policy_action_proxy",
        "policy_checkpoint": str(checkpoint_paths[0]),
        "policy_checkpoints": [str(path) for path in checkpoint_paths],
        "ensemble_size": int(len(checkpoint_paths)),
        "dataset_dir": str(args.dataset_dir),
        "split": args.split,
        "split_seed_spec": split_spec,
        "eval_used": int(len(eval_idx)),
        "budget_val_used": int(len(budget_val_idx)),
        "policy_threshold": float(threshold),
        "action_decoder": decoder_config.to_json(),
        "value_decoder": value_config.to_json(),
        "rb_dim_scale": float(args.rb_dim_scale),
        "cpu_dim_scale": float(args.cpu_dim_scale),
        "step_total_calibrator": step_total_calibrator.to_json(),
        "edge_reranker": edge_reranker_info,
        "edge_count_controller": edge_count_info,
        "metrics": metrics,
        "command": " ".join(sys.argv),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    result = run(parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
