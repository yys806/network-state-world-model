"""Strong CPU diagnostic for PI-JWM v11 candidate rb_total value reconstruction.

This script focuses on the value bottleneck exposed by prior diagnostics:
predicted rank can be useful when true rb_total values are injected, but
nearest-support deployable values still damage rollout.  It tests value targets
that are more scheduler-aware than direct absolute-value regression:
residual, log-ratio, per-task total, and per-task residual, plus optional
count-total constraints.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import average_precision_score, mean_absolute_error, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from pi_jwm.v11_rollout_value_calibrator import freeze_module

from diagnose_v11_counterfactual_value_attribution import collect_raw_actions
from diagnose_v11_rb_total_latent_identifiability import collect_rollout_edge_context, rows_from_context
from diagnose_v11_rb_total_oracle_value_scope import rankdata, safe_corr, select_topk_indices, write_json
from diagnose_v11_repair_benefit_shortlist import build_step_load_rows, make_edge_targets, make_feature_matrix
from run_v11_rb_total_repair import active_rate_row, evaluate_raw_actions, make_adaptive_dataset, write_csv
from run_v11_rb_total_value_head import (
    RB_DIM,
    _make_inference_examples,
    _standardize,
    collect_edge_gradient_improvement,
    limit_indices,
    load_context_limited,
    make_critical_examples,
)


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts/experiments/pi_jwm_v11_rb_total_value_reconstruction_20260622"
CPU_DIM = 4
RB_COUNT_DIM = 1
EPS = 1e-6


class ConstantSupportClassifier:
    def __init__(self, probability: float) -> None:
        self.probability = float(np.clip(probability, 0.0, 1.0))

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features)
        positive = np.full((features.shape[0],), self.probability, dtype=np.float32)
        return np.stack([1.0 - positive, positive], axis=1)


def make_support_sample_weight(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.float32).reshape(-1)
    if labels.size == 0:
        return np.zeros((0,), dtype=np.float32)
    positive = labels > 0.5
    pos_count = int(np.sum(positive))
    neg_count = int(labels.size - pos_count)
    if pos_count == 0 or neg_count == 0:
        return np.ones_like(labels, dtype=np.float32)
    weight = np.empty_like(labels, dtype=np.float32)
    weight[positive] = labels.size / (2.0 * pos_count)
    weight[~positive] = labels.size / (2.0 * neg_count)
    mean_weight = float(np.mean(weight))
    if mean_weight > EPS:
        weight = weight / mean_weight
    return weight.astype(np.float32)


def fit_support_classifier(kind: str, features: np.ndarray, labels: np.ndarray, seed: int, rf_trees: int):
    features = np.asarray(features, dtype=np.float32)
    labels = (np.asarray(labels, dtype=np.float32).reshape(-1) > 0.5).astype(np.int64)
    if features.shape[0] != labels.shape[0]:
        raise ValueError("features and labels must have the same row count")
    if labels.size == 0 or np.unique(labels).size < 2:
        return ConstantSupportClassifier(float(np.mean(labels)) if labels.size else 0.0)
    sample_weight = make_support_sample_weight(labels)
    if kind == "rf":
        model = RandomForestClassifier(
            n_estimators=int(rf_trees),
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight=None,
            random_state=int(seed),
            n_jobs=-1,
        )
    elif kind == "hgb":
        model = HistGradientBoostingClassifier(
            max_iter=180,
            learning_rate=0.04,
            l2_regularization=0.02,
            min_samples_leaf=8,
            random_state=int(seed),
        )
    else:
        raise ValueError(f"unknown support classifier kind: {kind}")
    model.fit(features, labels, sample_weight=sample_weight)
    return model


def predict_support_probability(model, features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    if features.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    probability = np.asarray(model.predict_proba(features), dtype=np.float32)
    if probability.ndim != 2:
        raise ValueError("support classifier predict_proba must return a 2D array")
    if probability.shape[1] == 1:
        return np.zeros((features.shape[0],), dtype=np.float32)
    return np.clip(probability[:, 1], 0.0, 1.0).astype(np.float32)


def support_classification_metrics(labels: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    labels = (np.asarray(labels, dtype=np.float32).reshape(-1) > 0.5).astype(np.int64)
    probability = np.asarray(probability, dtype=np.float32).reshape(-1)
    if labels.shape[0] != probability.shape[0]:
        raise ValueError("labels and probability must have the same row count")
    if labels.size == 0 or np.unique(labels).size < 2:
        return {"roc_auc": float("nan"), "average_precision": float("nan"), "positive_rate": float(np.mean(labels)) if labels.size else 0.0}
    return {
        "roc_auc": float(roc_auc_score(labels, probability)),
        "average_precision": float(average_precision_score(labels, probability)),
        "positive_rate": float(np.mean(labels)),
    }


def select_support_gated_topk_indices(
    coordinates: np.ndarray,
    scores: np.ndarray,
    support_probability: np.ndarray,
    top_k: int,
    scope: str,
    min_support_probability: float,
    score_mode: str = "rank_only",
) -> np.ndarray:
    coordinates = np.asarray(coordinates, dtype=np.int64).reshape(-1, 3)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    support_probability = np.asarray(support_probability, dtype=np.float32).reshape(-1)
    if not (coordinates.shape[0] == scores.shape[0] == support_probability.shape[0]):
        raise ValueError("coordinates, scores, and support probabilities must have the same row count")
    if top_k <= 0 or coordinates.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64)
    eligible = np.where(support_probability >= float(min_support_probability))[0]
    if eligible.size == 0:
        return np.zeros((0,), dtype=np.int64)
    if score_mode == "rank_only":
        gated_scores = scores[eligible]
    elif score_mode == "rank_times_support":
        gated_scores = scores[eligible] * support_probability[eligible]
    else:
        raise ValueError(f"unknown support score mode: {score_mode}")
    local_selected = select_topk_indices(coordinates[eligible], gated_scores, int(top_k), scope)
    return eligible[local_selected].astype(np.int64)


def make_step_support_labels(
    coordinates: np.ndarray,
    true_value: np.ndarray,
    min_effective_value: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coordinates = np.asarray(coordinates, dtype=np.int64).reshape(-1, 3)
    true_value = np.asarray(true_value, dtype=np.float32).reshape(-1)
    if coordinates.shape[0] != true_value.shape[0]:
        raise ValueError("coordinates and true_value must have the same row count")
    if coordinates.shape[0] == 0:
        return (
            np.zeros((0, 2), dtype=np.int64),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )
    keys, inverse = np.unique(coordinates[:, :2], axis=0, return_inverse=True)
    labels = np.zeros((keys.shape[0],), dtype=np.float32)
    positive_rows = true_value >= float(min_effective_value)
    if np.any(positive_rows):
        labels[np.unique(inverse[positive_rows])] = 1.0
    return keys.astype(np.int64), labels.astype(np.float32), labels[inverse].astype(np.float32)


def make_step_scheduler_features(actions: np.ndarray, step_keys: np.ndarray) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    step_keys = np.asarray(step_keys, dtype=np.int64).reshape(-1, 2)
    if step_keys.shape[0] == 0:
        return np.zeros((0, 11), dtype=np.float32)
    horizon = max(1, actions.shape[1] - 1)
    rows = []
    for sample, step in step_keys:
        step_action = actions[int(sample), int(step)]
        rb_total = np.clip(step_action[:, RB_DIM], 0.0, None)
        cpu_total = np.clip(step_action[:, CPU_DIM], 0.0, None)
        rb_count = np.clip(step_action[:, RB_COUNT_DIM], 0.0, None)
        cpu_count = np.clip(step_action[:, 3], 0.0, None)
        active = (step_action > EPS).any(axis=-1).astype(np.float32)
        rows.append(
            [
                float(step) / float(horizon),
                float(np.log1p(np.sum(rb_total))),
                float(np.log1p(np.sum(cpu_total))),
                float(np.log1p(np.sum(rb_total) + np.sum(cpu_total))),
                float(np.sum(rb_total > EPS)),
                float(np.sum(cpu_total > EPS)),
                float(np.sum(active)),
                float(np.log1p(np.sum(rb_count))),
                float(np.log1p(np.sum(cpu_count))),
                float(np.mean(rb_total[rb_total > EPS])) if np.any(rb_total > EPS) else 0.0,
                float(np.max(rb_total)) if rb_total.size else 0.0,
            ]
        )
    return np.asarray(rows, dtype=np.float32)


def map_step_probability_to_rows(step_keys: np.ndarray, step_probability: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    step_keys = np.asarray(step_keys, dtype=np.int64).reshape(-1, 2)
    step_probability = np.asarray(step_probability, dtype=np.float32).reshape(-1)
    coordinates = np.asarray(coordinates, dtype=np.int64).reshape(-1, 3)
    if step_keys.shape[0] != step_probability.shape[0]:
        raise ValueError("step_keys and step_probability must have the same row count")
    lookup = {(int(sample), int(step)): float(prob) for (sample, step), prob in zip(step_keys, step_probability)}
    return np.asarray([lookup.get((int(sample), int(step)), 0.0) for sample, step, _edge in coordinates], dtype=np.float32)

class NeuralValueDecoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def fit_neural_value_decoder(
    features: np.ndarray,
    targets: np.ndarray,
    sample_weight: np.ndarray | None = None,
    hidden_dim: int = 64,
    epochs: int = 80,
    lr: float = 1e-3,
    batch_size: int = 256,
    seed: int = 37,
) -> NeuralValueDecoder:
    features = np.asarray(features, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32).reshape(-1)
    if features.shape[0] != targets.shape[0]:
        raise ValueError("features and targets must have the same row count")
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    model = NeuralValueDecoder(features.shape[1], hidden_dim=int(hidden_dim))
    if features.shape[0] == 0:
        model.eval()
        return model
    if sample_weight is None:
        weight = np.ones_like(targets, dtype=np.float32)
    else:
        weight = np.asarray(sample_weight, dtype=np.float32).reshape(-1)
        if weight.shape[0] != targets.shape[0]:
            raise ValueError("sample_weight must match targets")
        weight = np.clip(weight, 0.0, None).astype(np.float32)
        mean_weight = float(np.mean(weight)) if weight.size else 1.0
        if mean_weight > EPS:
            weight = weight / mean_weight
        else:
            weight = np.ones_like(targets, dtype=np.float32)
    dataset = TensorDataset(
        torch.as_tensor(features, dtype=torch.float32),
        torch.as_tensor(targets, dtype=torch.float32),
        torch.as_tensor(weight, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1e-4)
    model.train()
    for _epoch in range(int(epochs)):
        for batch_x, batch_y, batch_w in loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch_x)
            loss = torch.mean(batch_w * torch.nn.functional.smooth_l1_loss(prediction, batch_y, reduction="none"))
            loss.backward()
            optimizer.step()
    model.eval()
    return model


def predict_neural_value_decoder(model: NeuralValueDecoder, features: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    if features.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    rows = []
    with torch.no_grad():
        for start in range(0, features.shape[0], int(batch_size)):
            batch = torch.as_tensor(features[start:start + int(batch_size)], dtype=torch.float32)
            rows.append(model(batch).cpu().numpy())
    return np.concatenate(rows, axis=0).astype(np.float32)



def make_value_fit_mask(true_value: np.ndarray, min_effective_value: float, fit_mode: str = "positive_only") -> np.ndarray:
    true_value = np.asarray(true_value, dtype=np.float32).reshape(-1)
    if fit_mode == "positive_only":
        return true_value >= float(min_effective_value)
    if fit_mode == "all":
        return np.ones_like(true_value, dtype=bool)
    raise ValueError(f"unknown value fit mode: {fit_mode}")
def make_value_sample_weight(true_value: np.ndarray, rb_count: np.ndarray, power: float = 0.5) -> np.ndarray:
    true_value = np.asarray(true_value, dtype=np.float32).reshape(-1)
    rb_count = np.asarray(rb_count, dtype=np.float32).reshape(-1)
    if true_value.shape != rb_count.shape:
        raise ValueError("true_value and rb_count must have the same shape")
    scale = np.log1p(np.clip(true_value, 0.0, None)) + 0.25 * np.log1p(np.clip(rb_count, 0.0, None))
    positive = scale > 0.0
    if not np.any(positive):
        return np.ones_like(true_value, dtype=np.float32)
    median = float(np.median(scale[positive]))
    if median <= EPS:
        median = 1.0
    weight = np.power(np.clip(scale / median, 0.1, 10.0), float(power)).astype(np.float32)
    return np.clip(weight, 0.25, 4.0).astype(np.float32)



def prepare_value_training_data(
    features: np.ndarray,
    true_value: np.ndarray,
    baseline_value: np.ndarray,
    rb_count: np.ndarray,
    target_mode: str,
    min_effective_value: float,
    fit_mode: str,
    weight_power: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    features = np.asarray(features, dtype=np.float32)
    mask = make_value_fit_mask(true_value, min_effective_value=min_effective_value, fit_mode=fit_mode)
    target = make_value_target(target_mode, np.asarray(true_value)[mask], np.asarray(baseline_value)[mask], np.asarray(rb_count)[mask])
    weight = make_value_sample_weight(np.asarray(true_value)[mask], np.asarray(rb_count)[mask], power=weight_power)
    return features[mask].astype(np.float32), target.astype(np.float32), weight.astype(np.float32), mask.astype(bool)
def make_value_target(mode: str, true_value: np.ndarray, baseline_value: np.ndarray, rb_count: np.ndarray) -> np.ndarray:
    true_value = np.asarray(true_value, dtype=np.float32).reshape(-1)
    baseline_value = np.asarray(baseline_value, dtype=np.float32).reshape(-1)
    rb_count = np.asarray(rb_count, dtype=np.float32).reshape(-1)
    if not (true_value.shape == baseline_value.shape == rb_count.shape):
        raise ValueError("true, baseline, and rb_count must have the same shape")
    if mode == "abs":
        target = true_value
    elif mode == "residual":
        target = true_value - baseline_value
    elif mode == "log_ratio":
        target = np.log((np.clip(true_value, 0.0, None) + EPS) / (np.clip(baseline_value, 0.0, None) + EPS))
    elif mode == "per_task":
        denom = np.maximum(rb_count, 1.0)
        target = true_value / denom
    elif mode == "per_task_residual":
        denom = np.maximum(rb_count, 1.0)
        target = (true_value - baseline_value) / denom
    else:
        raise ValueError(f"unknown value target mode: {mode}")
    return target.astype(np.float32)


def invert_value_target(mode: str, prediction: np.ndarray, baseline_value: np.ndarray, rb_count: np.ndarray) -> np.ndarray:
    prediction = np.asarray(prediction, dtype=np.float32).reshape(-1)
    baseline_value = np.asarray(baseline_value, dtype=np.float32).reshape(-1)
    rb_count = np.asarray(rb_count, dtype=np.float32).reshape(-1)
    if not (prediction.shape == baseline_value.shape == rb_count.shape):
        raise ValueError("prediction, baseline, and rb_count must have the same shape")
    if mode == "abs":
        value = prediction
    elif mode == "residual":
        value = baseline_value + prediction
    elif mode == "log_ratio":
        value = (np.clip(baseline_value, 0.0, None) + EPS) * np.exp(np.clip(prediction, -20.0, 20.0)) - EPS
    elif mode == "per_task":
        value = prediction * np.maximum(rb_count, 1.0)
    elif mode == "per_task_residual":
        value = baseline_value + prediction * np.maximum(rb_count, 1.0)
    else:
        raise ValueError(f"unknown value target mode: {mode}")
    return np.clip(value, 0.0, None).astype(np.float32)


def apply_count_total_constraint(
    values: np.ndarray,
    baseline_values: np.ndarray,
    rb_count: np.ndarray,
    max_rb_per_task: float = 64.0,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    baseline_values = np.asarray(baseline_values, dtype=np.float32).reshape(-1)
    rb_count = np.asarray(rb_count, dtype=np.float32).reshape(-1)
    if not (values.shape == baseline_values.shape == rb_count.shape):
        raise ValueError("values, baseline, and rb_count must have the same shape")
    constrained = np.clip(values, 0.0, None).astype(np.float32)
    inactive = baseline_values <= EPS
    zero_count = rb_count <= EPS
    constrained[inactive] = 0.0
    constrained[~inactive & zero_count] = baseline_values[~inactive & zero_count]
    positive_count = ~inactive & ~zero_count
    max_total = np.maximum(rb_count[positive_count], 1.0) * float(max_rb_per_task)
    constrained[positive_count] = np.minimum(constrained[positive_count], max_total).astype(np.float32)
    return constrained.astype(np.float32)


def audit_coordinate_values(coordinates: np.ndarray, values: np.ndarray, truth_actions: np.ndarray) -> dict:
    coordinates = np.asarray(coordinates, dtype=np.int64).reshape(-1, 3)
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    truth_actions = np.asarray(truth_actions, dtype=np.float32)
    if coordinates.shape[0] != values.shape[0]:
        raise ValueError("coordinates and values must have the same row count")
    if coordinates.shape[0] == 0:
        return {"row_count": 0, "mismatch_count": 0, "max_abs_error": 0.0, "mean_abs_error": 0.0}
    truth_values = truth_actions[coordinates[:, 0], coordinates[:, 1], coordinates[:, 2], RB_DIM]
    abs_error = np.abs(truth_values - values)
    return {
        "row_count": int(values.shape[0]),
        "mismatch_count": int(np.sum(abs_error > 1e-5)),
        "max_abs_error": float(np.max(abs_error)),
        "mean_abs_error": float(np.mean(abs_error)),
    }


def coordinate_action_values(actions: np.ndarray, coordinates: np.ndarray, dim: int) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    coordinates = np.asarray(coordinates, dtype=np.int64).reshape(-1, 3)
    if coordinates.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    return actions[coordinates[:, 0], coordinates[:, 1], coordinates[:, 2], int(dim)].astype(np.float32)


def apply_selected_value_repair(
    actions: np.ndarray,
    coordinates: np.ndarray,
    values: np.ndarray,
    selected: np.ndarray,
    alpha: float = 1.0,
    constrain: bool = False,
    rb_count: np.ndarray | None = None,
    max_rb_per_task: float = 64.0,
) -> np.ndarray:
    repaired = np.asarray(actions, dtype=np.float32).copy()
    coordinates = np.asarray(coordinates, dtype=np.int64).reshape(-1, 3)
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if coordinates.shape[0] != values.shape[0]:
        raise ValueError("coordinates and values must have the same row count")
    baseline = coordinate_action_values(actions, coordinates, RB_DIM)
    if constrain:
        if rb_count is None:
            rb_count = coordinate_action_values(actions, coordinates, RB_COUNT_DIM)
        values = apply_count_total_constraint(values, baseline, rb_count, max_rb_per_task=max_rb_per_task)
    alpha = float(alpha)
    for idx in np.asarray(selected, dtype=np.int64).reshape(-1):
        sample, step, edge = coordinates[int(idx)]
        if step == 0 or actions[sample, step, edge, RB_DIM] <= EPS:
            continue
        baseline_value = float(actions[sample, step, edge, RB_DIM])
        target_value = float(values[int(idx)])
        repaired[sample, step, edge, RB_DIM] = max((1.0 - alpha) * baseline_value + alpha * target_value, 0.0)
    repaired[:, 0, :, RB_DIM] = actions[:, 0, :, RB_DIM]
    repaired[..., RB_DIM] = np.where(actions[..., RB_DIM] > EPS, np.clip(repaired[..., RB_DIM], 0.0, None), 0.0)
    return repaired.astype(np.float32)


def fit_value_model(kind: str, features: np.ndarray, targets: np.ndarray, seed: int, rf_trees: int, args: argparse.Namespace | None = None, sample_weight: np.ndarray | None = None):
    if kind == "rf":
        model = RandomForestRegressor(
            n_estimators=int(rf_trees),
            min_samples_leaf=3,
            max_features="sqrt",
            random_state=int(seed),
            n_jobs=-1,
        )
    elif kind == "hgb":
        model = HistGradientBoostingRegressor(
            max_iter=220,
            learning_rate=0.04,
            l2_regularization=0.02,
            min_samples_leaf=12,
            random_state=int(seed),
        )
    elif kind == "nn":
        hidden_dim = getattr(args, "nn_hidden_dim", 64) if args is not None else 64
        epochs = getattr(args, "nn_epochs", 80) if args is not None else 80
        lr = getattr(args, "nn_lr", 1e-3) if args is not None else 1e-3
        batch_size = getattr(args, "nn_batch_size", 256) if args is not None else 256
        return fit_neural_value_decoder(
            features,
            targets,
            sample_weight=sample_weight,
            hidden_dim=int(hidden_dim),
            epochs=int(epochs),
            lr=float(lr),
            batch_size=int(batch_size),
            seed=int(seed),
        )
    else:
        raise ValueError(f"unknown model kind: {kind}")
    if sample_weight is None:
        model.fit(features, targets)
    else:
        model.fit(features, targets, sample_weight=np.asarray(sample_weight, dtype=np.float32).reshape(-1))
    return model


def predict_value_model(model, features: np.ndarray) -> np.ndarray:
    if isinstance(model, NeuralValueDecoder):
        return predict_neural_value_decoder(model, features)
    return np.asarray(model.predict(features), dtype=np.float32)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-experiment-dir", type=Path, default=PROJECT_ROOT / "artifacts/experiments/pi_jwm_v9_expanded_v2_gpu_20260619/v2_hurdle_baseline")
    parser.add_argument("--world-checkpoint", type=Path, default=PROJECT_ROOT / "artifacts/experiments/pi_jwm_v9_expanded_v2_gpu_20260619/v2_hurdle_baseline/checkpoints/v8_dual_best.pt")
    parser.add_argument("--policy-checkpoint", type=Path, default=PROJECT_ROOT / "artifacts/experiments/pi_jwm_v10_policy_bridge_gpu_20260620/v10_action_policy_bc/checkpoints/v7_action_policy_cross_attention_best.pt")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--policy-threshold", type=float, default=0.4)
    parser.add_argument("--value-scale", type=float, default=1.0)
    parser.add_argument("--new-policy-threshold", type=float, default=0.37)
    parser.add_argument("--new-value-scale", type=float, default=1.06)
    parser.add_argument("--gate-feature", choices=("step_rb_total", "step_cpu_total", "step_rb_cpu_total", "step_active_count"), default="step_rb_cpu_total")
    parser.add_argument("--gate-threshold", type=float, default=450.0)
    parser.add_argument("--value-codebook-size", type=int, default=9)
    parser.add_argument("--min-effective-rb-total", type=float, default=1.0)
    parser.add_argument("--steps", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--max-train-samples", type=int, default=512)
    parser.add_argument("--max-val-samples", type=int, default=256)
    parser.add_argument("--max-test-samples", type=int, default=256)
    parser.add_argument("--limit-after-stats", action="store_true")
    parser.add_argument("--streaming-stats", action="store_true")
    parser.add_argument("--stats-chunk-size", type=int, default=512)
    parser.add_argument("--feature-sets", choices=("action", "load", "action_load", "action_latent_load"), nargs="+", default=["action_load", "action_latent_load"])
    parser.add_argument("--model-kinds", choices=("rf", "hgb", "nn"), nargs="+", default=["rf"])
    parser.add_argument("--rank-model-kind", choices=("rf", "hgb", "nn"), default="rf")
    parser.add_argument("--target-modes", choices=("abs", "residual", "log_ratio", "per_task", "per_task_residual"), nargs="+", default=["residual", "log_ratio", "per_task", "per_task_residual"])
    parser.add_argument("--rf-trees", type=int, default=100)
    parser.add_argument("--nn-hidden-dim", type=int, default=64)
    parser.add_argument("--nn-epochs", type=int, default=80)
    parser.add_argument("--nn-lr", type=float, default=0.001)
    parser.add_argument("--nn-batch-size", type=int, default=256)
    parser.add_argument("--positive-value-weight-power", type=float, default=0.5)
    parser.add_argument("--value-fit-mode", choices=("positive_only", "all"), default="positive_only")
    parser.add_argument("--support-classifier-kinds", choices=("rf", "hgb"), nargs="+", default=[])
    parser.add_argument("--support-thresholds", type=float, nargs="+", default=[0.5])
    parser.add_argument("--support-score-modes", choices=("rank_only", "rank_times_support"), nargs="+", default=["rank_times_support"])
    parser.add_argument("--step-support-modes", choices=("oracle", "learned_rf", "learned_hgb"), nargs="+", default=[])
    parser.add_argument("--step-support-thresholds", type=float, nargs="+", default=[0.5])
    parser.add_argument("--top-k", type=int, nargs="+", default=[4, 16, 64])
    parser.add_argument("--scopes", choices=("per_sample", "per_sample_step", "global"), nargs="+", default=["per_sample"])
    parser.add_argument("--blend-alpha", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    parser.add_argument("--constraint", choices=("none", "count_total"), nargs="+", default=["none", "count_total"])
    parser.add_argument("--max-rb-per-task", type=float, default=64.0)
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--include-oracle-rank", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict:
    started = time.time()
    device = torch.device("cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    context = load_context_limited(args, device)
    summary, arrays, stats, world_model, policy_model, action_scale, value_vocab, splits = context
    splits = dict(splits)
    if args.limit_after_stats:
        splits["train"] = limit_indices(splits["train"], args.max_train_samples)
        splits["val"] = limit_indices(splits["val"], args.max_val_samples)
        splits["test"] = limit_indices(splits["test"], args.max_test_samples)
    world_model = freeze_module(world_model)
    policy_model = freeze_module(policy_model)
    steps = tuple(int(step) for step in args.steps)

    train_base, train_dataset = make_adaptive_dataset(args, arrays, splits["train"], stats, policy_model, action_scale, value_vocab, device, splits["train"])
    train_actions, train_truth = collect_raw_actions(train_dataset, stats)
    train_examples = make_critical_examples(train_actions, train_truth, steps=steps)
    train_edge_improvement = collect_edge_gradient_improvement(
        world_model, train_base, train_actions, train_truth, stats, summary["config"], device, args.batch_size
    )
    train_score, train_true_value = make_edge_targets(train_examples, train_truth, train_edge_improvement)
    train_rb_count = coordinate_action_values(train_actions, train_examples.coordinates, RB_COUNT_DIM)
    train_latent_context = collect_rollout_edge_context(world_model, train_base, train_actions, stats, device, args.batch_size)
    train_latent = rows_from_context(train_latent_context, train_examples.coordinates)
    train_load = build_step_load_rows(train_actions, train_examples.coordinates)
    train_audit = audit_coordinate_values(train_examples.coordinates, train_true_value, train_truth)
    train_step_keys, train_step_labels, _train_row_step_labels = make_step_support_labels(
        train_examples.coordinates,
        train_true_value,
        min_effective_value=args.min_effective_rb_total,
    )
    train_step_features_raw = make_step_scheduler_features(train_actions, train_step_keys)

    split_payload = {}
    split_step_features_raw = {}
    for split_name in ("val", "test"):
        base_dataset, adaptive_dataset = make_adaptive_dataset(args, arrays, splits[split_name], stats, policy_model, action_scale, value_vocab, device, splits["train"])
        baseline_actions, truth_actions = collect_raw_actions(adaptive_dataset, stats)
        examples = _make_inference_examples(baseline_actions, truth_actions.shape, steps)
        edge_improvement = collect_edge_gradient_improvement(
            world_model, base_dataset, baseline_actions, truth_actions, stats, summary["config"], device, args.batch_size
        )
        oracle_score, true_value = make_edge_targets(examples, truth_actions, edge_improvement)
        rb_count = coordinate_action_values(baseline_actions, examples.coordinates, RB_COUNT_DIM)
        latent_context = collect_rollout_edge_context(world_model, base_dataset, baseline_actions, stats, device, args.batch_size)
        latent_rows = rows_from_context(latent_context, examples.coordinates)
        load_rows = build_step_load_rows(baseline_actions, examples.coordinates)
        step_keys, step_labels, row_step_labels = make_step_support_labels(
            examples.coordinates,
            true_value,
            min_effective_value=args.min_effective_rb_total,
        )
        split_step_features_raw[split_name] = make_step_scheduler_features(baseline_actions, step_keys)
        split_payload[split_name] = {
            "base_dataset": base_dataset,
            "baseline_actions": baseline_actions,
            "truth_actions": truth_actions,
            "examples": examples,
            "oracle_score": oracle_score,
            "true_value": true_value,
            "rb_count": rb_count,
            "latent": latent_rows,
            "load": load_rows,
            "audit": audit_coordinate_values(examples.coordinates, true_value, truth_actions),
            "step_keys": step_keys,
            "step_labels": step_labels,
            "row_step_labels": row_step_labels,
        }

    rows = []
    diagnostics = {
        "train_examples": int(train_examples.coordinates.shape[0]),
        "train_positive_value_count": int(np.sum(train_true_value >= float(args.min_effective_rb_total))),
        "train_positive_score_count": int(np.sum(train_score > 0.0)),
        "train_value_audit": train_audit,
        "train_step_support_count": int(np.sum(train_step_labels > 0.5)),
        "train_step_count": int(train_step_labels.shape[0]),
        "feature_sets": list(args.feature_sets),
        "model_kinds": list(args.model_kinds),
        "rank_model_kind": str(args.rank_model_kind),
        "target_modes": list(args.target_modes),
        "value_fit_mode": str(args.value_fit_mode),
        "support_classifier_kinds": list(args.support_classifier_kinds),
        "support_thresholds": [float(item) for item in args.support_thresholds],
        "support_score_modes": list(args.support_score_modes),
        "step_support_modes": list(args.step_support_modes),
        "step_support_thresholds": [float(item) for item in args.step_support_thresholds],
    }

    positive_train = train_true_value >= float(args.min_effective_rb_total)
    step_standardized = _standardize(train_step_features_raw, *(split_step_features_raw[name] for name in ("val", "test")))
    train_step_features = step_standardized[0]
    split_step_features = {"val": step_standardized[1], "test": step_standardized[2]}
    step_support_models = {}
    if "learned_rf" in args.step_support_modes:
        step_support_models["learned_rf"] = fit_support_classifier("rf", train_step_features, train_step_labels, args.seed + 41, args.rf_trees)
    if "learned_hgb" in args.step_support_modes:
        step_support_models["learned_hgb"] = fit_support_classifier("hgb", train_step_features, train_step_labels, args.seed + 43, args.rf_trees)
    fit_features_by_set = {}
    split_features_by_set = {}
    for feature_set in args.feature_sets:
        train_features_raw = make_feature_matrix(train_examples.features, train_latent, train_load, feature_set)
        split_features_raw = {
            split_name: make_feature_matrix(payload["examples"].features, payload["latent"], payload["load"], feature_set)
            for split_name, payload in split_payload.items()
        }
        standardized = _standardize(train_features_raw, *(split_features_raw[name] for name in ("val", "test")))
        fit_features_by_set[feature_set] = standardized[0]
        split_features_by_set[feature_set] = {"val": standardized[1], "test": standardized[2]}

    for feature_set in args.feature_sets:
        train_features = fit_features_by_set[feature_set]
        support_labels = train_true_value >= float(args.min_effective_rb_total)
        support_models = {
            support_kind: fit_support_classifier(support_kind, train_features, support_labels, args.seed + 23, args.rf_trees)
            for support_kind in args.support_classifier_kinds
        }
        for model_kind in args.model_kinds:
            # A lightweight benefit model is used only for deployable rank; value diagnostics are isolated below.
            rank_model = fit_value_model(args.rank_model_kind, train_features, np.log1p(np.clip(train_score, 0.0, None)), args.seed, args.rf_trees, args=args)
            for target_mode in args.target_modes:
                value_features, target, value_weight, fit_mask = prepare_value_training_data(
                    train_features,
                    train_true_value,
                    train_examples.baseline_values,
                    train_rb_count,
                    target_mode=target_mode,
                    min_effective_value=args.min_effective_rb_total,
                    fit_mode=args.value_fit_mode,
                    weight_power=args.positive_value_weight_power,
                )
                diagnostics[f"{feature_set}_{model_kind}_{target_mode}_fit_row_count"] = int(np.sum(fit_mask))
                value_model = fit_value_model(model_kind, value_features, target, args.seed + 11, args.rf_trees, args=args, sample_weight=value_weight)

                for split_name, payload in split_payload.items():
                    baseline_predictions = evaluate_raw_actions(
                        payload["baseline_actions"], payload["base_dataset"], stats, world_model, summary["config"], device, args.batch_size
                    )
                    baseline_row = active_rate_row("identity", split_name, baseline_predictions, float("nan"))
                    baseline_rmse = float(baseline_row["active_rate_rmse"])
                    baseline_row.update({"feature_set": feature_set, "model": model_kind, "method": "identity", "target_mode": target_mode, "scope": "none", "top_k": 0})
                    rows.append(baseline_row)

                    split_features = split_features_by_set[feature_set][split_name]
                    raw_prediction = predict_value_model(value_model, split_features)
                    pred_value = invert_value_target(target_mode, raw_prediction, payload["examples"].baseline_values, payload["rb_count"])
                    pred_score = predict_value_model(rank_model, split_features)
                    support_probability_by_kind = {
                        support_kind: predict_support_probability(support_model, split_features)
                        for support_kind, support_model in support_models.items()
                    }
                    step_support_probability_by_mode = {}
                    if "oracle" in args.step_support_modes:
                        step_support_probability_by_mode["oracle"] = payload["row_step_labels"].astype(np.float32)
                    for step_mode, step_model in step_support_models.items():
                        step_probability = predict_support_probability(step_model, split_step_features[split_name])
                        step_support_probability_by_mode[step_mode] = map_step_probability_to_rows(
                            payload["step_keys"],
                            step_probability,
                            payload["examples"].coordinates,
                        )
                        step_metrics = support_classification_metrics(payload["step_labels"], step_probability)
                        for metric_name, metric_value in step_metrics.items():
                            diagnostics[f"{split_name}_{step_mode}_step_support_{metric_name}"] = float(metric_value)
                    positive_mask = payload["true_value"] >= float(args.min_effective_rb_total)
                    diagnostics[f"{split_name}_{feature_set}_{model_kind}_{target_mode}_value_mae_nonzero_true"] = (
                        float(mean_absolute_error(payload["true_value"][positive_mask], pred_value[positive_mask]))
                        if np.any(positive_mask)
                        else float("nan")
                    )
                    diagnostics[f"{split_name}_{feature_set}_{model_kind}_{target_mode}_value_pearson"] = safe_corr(pred_value, payload["true_value"])
                    diagnostics[f"{split_name}_{feature_set}_{model_kind}_rank_spearman"] = safe_corr(rankdata(pred_score), rankdata(payload["oracle_score"]))
                    for support_kind, support_probability in support_probability_by_kind.items():
                        support_metrics = support_classification_metrics(positive_mask, support_probability)
                        for metric_name, metric_value in support_metrics.items():
                            diagnostics[f"{split_name}_{feature_set}_{support_kind}_support_{metric_name}"] = float(metric_value)

                    for rank_name, rank_score in (("pred_rank", pred_score), ("oracle_rank", payload["oracle_score"])):
                        if rank_name == "oracle_rank" and not args.include_oracle_rank:
                            continue
                        for constraint in args.constraint:
                            constrain = constraint == "count_total"
                            for scope in args.scopes:
                                for top_k in args.top_k:
                                    selected = select_topk_indices(payload["examples"].coordinates, rank_score, int(top_k), scope)
                                    for alpha in args.blend_alpha:
                                        actions = apply_selected_value_repair(
                                            payload["baseline_actions"],
                                            payload["examples"].coordinates,
                                            pred_value,
                                            selected,
                                            alpha=float(alpha),
                                            constrain=constrain,
                                            rb_count=payload["rb_count"],
                                            max_rb_per_task=float(args.max_rb_per_task),
                                        )
                                        predictions = evaluate_raw_actions(actions, payload["base_dataset"], stats, world_model, summary["config"], device, args.batch_size)
                                        prefix = "diagnostic_only__" if rank_name == "oracle_rank" else ""
                                        candidate = (
                                            f"{prefix}{feature_set}__{model_kind}__value_{target_mode}__{rank_name}__"
                                            f"{constraint}__{scope}__top{int(top_k)}__alpha{float(alpha):g}"
                                        )
                                        row = active_rate_row(candidate, split_name, predictions, baseline_rmse)
                                        row.update(
                                            {
                                                "feature_set": feature_set,
                                                "model": model_kind,
                                                "method": "value_reconstruction" if rank_name == "pred_rank" else "diagnostic_only_oracle_rank_value_reconstruction",
                                                "target_mode": target_mode,
                                                "rank_mode": rank_name,
                                                "constraint": constraint,
                                                "scope": scope,
                                                "top_k": int(top_k),
                                                "alpha": float(alpha),
                                                "selected_count": int(selected.size),
                                                "selected_oracle_mass": float(np.sum(payload["oracle_score"][selected])) if selected.size else 0.0,
                                            }
                                        )
                                        rows.append(row)

                    for support_kind, support_probability in support_probability_by_kind.items():
                        for constraint in args.constraint:
                            constrain = constraint == "count_total"
                            for scope in args.scopes:
                                for top_k in args.top_k:
                                    for support_threshold in args.support_thresholds:
                                        for support_score_mode in args.support_score_modes:
                                            selected = select_support_gated_topk_indices(
                                                payload["examples"].coordinates,
                                                pred_score,
                                                support_probability,
                                                int(top_k),
                                                scope,
                                                min_support_probability=float(support_threshold),
                                                score_mode=support_score_mode,
                                            )
                                            for alpha in args.blend_alpha:
                                                actions = apply_selected_value_repair(
                                                    payload["baseline_actions"],
                                                    payload["examples"].coordinates,
                                                    pred_value,
                                                    selected,
                                                    alpha=float(alpha),
                                                    constrain=constrain,
                                                    rb_count=payload["rb_count"],
                                                    max_rb_per_task=float(args.max_rb_per_task),
                                                )
                                                predictions = evaluate_raw_actions(actions, payload["base_dataset"], stats, world_model, summary["config"], device, args.batch_size)
                                                candidate = (
                                                    f"{feature_set}__{model_kind}__value_{target_mode}__pred_rank__support_{support_kind}"
                                                    f"_p{float(support_threshold):g}_{support_score_mode}__{constraint}__{scope}__top{int(top_k)}__alpha{float(alpha):g}"
                                                )
                                                row = active_rate_row(candidate, split_name, predictions, baseline_rmse)
                                                row.update(
                                                    {
                                                        "feature_set": feature_set,
                                                        "model": model_kind,
                                                        "method": "support_gated_value_reconstruction",
                                                        "target_mode": target_mode,
                                                        "rank_mode": "pred_rank",
                                                        "support_classifier": support_kind,
                                                        "support_threshold": float(support_threshold),
                                                        "support_score_mode": support_score_mode,
                                                        "constraint": constraint,
                                                        "scope": scope,
                                                        "top_k": int(top_k),
                                                        "alpha": float(alpha),
                                                        "selected_count": int(selected.size),
                                                        "selected_oracle_mass": float(np.sum(payload["oracle_score"][selected])) if selected.size else 0.0,
                                                        "selected_support_mean": float(np.mean(support_probability[selected])) if selected.size else float("nan"),
                                                        "selected_true_positive_value_count": int(np.sum(positive_mask[selected])) if selected.size else 0,
                                                    }
                                                )
                                                rows.append(row)

                    for step_mode, step_support_probability in step_support_probability_by_mode.items():
                        for constraint in args.constraint:
                            constrain = constraint == "count_total"
                            for scope in args.scopes:
                                for top_k in args.top_k:
                                    for step_threshold in args.step_support_thresholds:
                                        selected = select_support_gated_topk_indices(
                                            payload["examples"].coordinates,
                                            pred_score,
                                            step_support_probability,
                                            int(top_k),
                                            scope,
                                            min_support_probability=float(step_threshold),
                                            score_mode="rank_only",
                                        )
                                        for alpha in args.blend_alpha:
                                            actions = apply_selected_value_repair(
                                                payload["baseline_actions"],
                                                payload["examples"].coordinates,
                                                pred_value,
                                                selected,
                                                alpha=float(alpha),
                                                constrain=constrain,
                                                rb_count=payload["rb_count"],
                                                max_rb_per_task=float(args.max_rb_per_task),
                                            )
                                            predictions = evaluate_raw_actions(actions, payload["base_dataset"], stats, world_model, summary["config"], device, args.batch_size)
                                            prefix = "diagnostic_only__" if step_mode == "oracle" else ""
                                            method = "diagnostic_only_step_support_value_reconstruction" if step_mode == "oracle" else "step_support_value_reconstruction"
                                            candidate = (
                                                f"{prefix}{feature_set}__{model_kind}__value_{target_mode}__pred_rank__step_support_{step_mode}"
                                                f"_p{float(step_threshold):g}__{constraint}__{scope}__top{int(top_k)}__alpha{float(alpha):g}"
                                            )
                                            row = active_rate_row(candidate, split_name, predictions, baseline_rmse)
                                            row.update(
                                                {
                                                    "feature_set": feature_set,
                                                    "model": model_kind,
                                                    "method": method,
                                                    "target_mode": target_mode,
                                                    "rank_mode": "pred_rank",
                                                    "step_support_mode": step_mode,
                                                    "step_support_threshold": float(step_threshold),
                                                    "constraint": constraint,
                                                    "scope": scope,
                                                    "top_k": int(top_k),
                                                    "alpha": float(alpha),
                                                    "selected_count": int(selected.size),
                                                    "selected_oracle_mass": float(np.sum(payload["oracle_score"][selected])) if selected.size else 0.0,
                                                    "selected_step_support_mean": float(np.mean(step_support_probability[selected])) if selected.size else float("nan"),
                                                    "selected_true_positive_value_count": int(np.sum(positive_mask[selected])) if selected.size else 0,
                                                }
                                            )
                                            rows.append(row)

    write_csv(args.output_dir / "value_reconstruction_results.csv", rows)
    val_ranked = sorted([row for row in rows if row["split"] == "val"], key=lambda row: (float(row["active_rate_rmse"]), str(row["candidate"])))
    deployable_val_ranked = sorted(
        [row for row in rows if row["split"] == "val" and row.get("method") in {"value_reconstruction", "support_gated_value_reconstruction", "step_support_value_reconstruction"}],
        key=lambda row: (float(row["active_rate_rmse"]), str(row["candidate"])),
    )
    diagnostic_val_ranked = sorted(
        [row for row in rows if row["split"] == "val" and str(row.get("method", "")).startswith("diagnostic_only")],
        key=lambda row: (float(row["active_rate_rmse"]), str(row["candidate"])),
    )
    write_csv(args.output_dir / "value_reconstruction_val_ranked.csv", val_ranked)
    write_csv(args.output_dir / "value_reconstruction_deployable_val_ranked.csv", deployable_val_ranked)
    write_csv(args.output_dir / "value_reconstruction_diagnostic_val_ranked.csv", diagnostic_val_ranked)
    test_by_candidate = {str(row["candidate"]): row for row in rows if row["split"] == "test"}
    best_val = deployable_val_ranked[0] if deployable_val_ranked else (val_ranked[0] if val_ranked else None)
    best_diagnostic_val = diagnostic_val_ranked[0] if diagnostic_val_ranked else None
    matched_test = test_by_candidate.get(str(best_val["candidate"])) if best_val else None
    matched_diagnostic_test = test_by_candidate.get(str(best_diagnostic_val["candidate"])) if best_diagnostic_val else None
    diagnostics["split_audits"] = {name: payload["audit"] for name, payload in split_payload.items()}
    diagnostics["runtime_seconds"] = float(time.time() - started)
    result = {
        "framework": "PI-JWM",
        "candidate": "v11",
        "mode": "rb_total_value_reconstruction",
        "output_dir": str(args.output_dir),
        "command": " ".join(sys.argv),
        "diagnostics": diagnostics,
        "best_val": best_val,
        "matched_test_for_best_val": matched_test,
        "best_diagnostic_val": best_diagnostic_val,
        "matched_test_for_best_diagnostic_val": matched_diagnostic_test,
    }
    write_json(args.output_dir / "summary.json", result)
    return result


def main() -> None:
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
