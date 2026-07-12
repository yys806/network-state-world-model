"""CPU-first rb_total critical value-head repair for PI-JWM v11 candidate.

This is a deployable probe: it never uses true future values at inference.
It trains a small critical-position classifier plus support-constrained rb_total
codebook head on the train split, selects candidates by validation rollout, and
reports test only after validation ranking.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from pi_jwm.v11_rollout_value_calibrator import freeze_module

from pi_jwm.v6_data import (
    build_link_rate_target,
    build_physical_edge_history,
    inverse_transform_link_rate,
    load_world_model_arrays,
    make_normalization_stats,
)
from pi_jwm.v6_dual_graph import V6DualGraphBatch

from diagnose_v11_bridge_operating_point import load_context
from evaluate_v10_policy_bridge import load_policy
from evaluate_v9_action_ablation import load_model_for_experiment, resolve_project_path
from diagnose_v11_counterfactual_value_attribution import collect_raw_actions
from run_world_model_v8_full_training import add_event_memory_features, resolve_seed_splits
from run_v11_rb_total_repair import (
    CURRENT_CPU_BEST_VAL_ACTIVE_RMSE,
    active_rate_row,
    evaluate_raw_actions,
    make_adaptive_dataset,
    write_csv,
)


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts/experiments/pi_jwm_v11_rb_total_value_head_cpu_20260621"
RB_DIM = 2
EPS = 1e-9
MIN_EFFECTIVE_RB_TOTAL = 1.0


@dataclass(frozen=True)
class StepCodebook:
    step: int
    values: np.ndarray
    edges: np.ndarray


@dataclass(frozen=True)
class RbTotalExamples:
    features: np.ndarray
    labels: np.ndarray
    coordinates: np.ndarray
    baseline_values: np.ndarray
    true_values: np.ndarray


@dataclass(frozen=True)
class RetrievalPrototypeModel:
    features: np.ndarray
    values: np.ndarray


class TinyHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 64, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TinyRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def build_step_codebooks(
    truth: np.ndarray,
    steps: tuple[int, ...] = (1, 2),
    bin_count: int = 9,
    min_effective_value: float = MIN_EFFECTIVE_RB_TOTAL,
) -> dict[int, StepCodebook]:
    truth = np.asarray(truth, dtype=np.float32)
    codebooks: dict[int, StepCodebook] = {}
    for step in steps:
        values = truth[:, step, :, RB_DIM].reshape(-1)
        values = values[np.isfinite(values) & (values >= float(min_effective_value))]
        if values.size == 0:
            codebooks[step] = StepCodebook(step=step, values=np.array([0.0], dtype=np.float32), edges=np.array([], dtype=np.float32))
            continue
        quantiles = np.linspace(0.0, 1.0, max(1, bin_count), dtype=np.float64)
        centers = np.unique(np.quantile(values, quantiles).astype(np.float32))
        if centers.size == 0:
            centers = np.array([float(np.median(values))], dtype=np.float32)
        centers = np.sort(centers.astype(np.float32))
        if centers.size == 1:
            edges = np.array([], dtype=np.float32)
        else:
            edges = ((centers[:-1] + centers[1:]) * 0.5).astype(np.float32)
        codebooks[step] = StepCodebook(step=step, values=centers, edges=edges)
    return codebooks


def _encode_values(values: np.ndarray, codebook: StepCodebook) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return np.searchsorted(codebook.edges, values, side="right").astype(np.int64)


def _feature_rows(baseline: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    rows = []
    horizon = max(1, baseline.shape[1] - 1)
    for sample, step, edge in coordinates:
        action = baseline[sample, step, edge]
        step_actions = baseline[sample, step]
        rb_values = step_actions[:, RB_DIM]
        cpu_values = step_actions[:, 4]
        active_rb = rb_values > EPS
        active_cpu = cpu_values > EPS
        rows.append(
            [
                float(step) / float(horizon),
                float(action[0]),
                float(action[1]),
                float(np.log1p(max(action[RB_DIM], 0.0))),
                float(action[3]),
                float(np.log1p(max(action[4], 0.0))),
                float(action[5]),
                float(np.log1p(np.sum(np.clip(rb_values, 0.0, None)))),
                float(np.log1p(np.sum(np.clip(cpu_values, 0.0, None)))),
                float(np.sum(active_rb)),
                float(np.sum(active_cpu)),
                float(np.log1p(np.sum(np.clip(rb_values, 0.0, None)) + np.sum(np.clip(cpu_values, 0.0, None)))),
            ]
        )
    return np.asarray(rows, dtype=np.float32)


def append_state_features(features: np.ndarray, state_features: np.ndarray | None, coordinates: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    if state_features is None:
        return features
    state_features = np.asarray(state_features, dtype=np.float32)
    coordinates = np.asarray(coordinates, dtype=np.int64).reshape(-1, 3)
    if coordinates.shape[0] != features.shape[0]:
        raise ValueError("coordinates must match feature rows")
    if features.shape[0] == 0:
        return np.concatenate([features, np.zeros((0, state_features.shape[-1]), dtype=np.float32)], axis=1)
    rows = state_features[coordinates[:, 0], coordinates[:, 2]]
    return np.concatenate([features, rows.astype(np.float32)], axis=1)


def extract_state_features(base_dataset) -> np.ndarray:
    rows = []
    for idx in range(len(base_dataset)):
        batch, _ = base_dataset[idx]
        physical = batch.physical_edge_history.numpy()
        info = batch.info_edge_history.numpy()
        action = batch.action_history.numpy()
        parts = [
            physical.mean(axis=0),
            physical[-1],
            info.mean(axis=0),
            info[-1],
            action.mean(axis=0),
            action[-1],
        ]
        rows.append(np.concatenate(parts, axis=-1).astype(np.float32))
    return np.stack(rows, axis=0) if rows else np.zeros((0, 0, 0), dtype=np.float32)


def make_rb_total_examples(
    baseline: np.ndarray,
    truth: np.ndarray,
    codebooks: dict[int, StepCodebook],
    steps: tuple[int, ...] = (1, 2),
    min_effective_value: float = MIN_EFFECTIVE_RB_TOTAL,
) -> RbTotalExamples:
    baseline = np.asarray(baseline, dtype=np.float32)
    truth = np.asarray(truth, dtype=np.float32)
    coordinates = []
    labels = []
    baseline_values = []
    true_values = []
    for step in steps:
        if step not in codebooks:
            continue
        pred_rb = baseline[:, step, :, RB_DIM]
        true_rb = truth[:, step, :, RB_DIM]
        sample_idx, edge_idx = np.where((pred_rb > EPS) & (true_rb >= float(min_effective_value)))
        for sample, edge in zip(sample_idx, edge_idx):
            coordinates.append((int(sample), int(step), int(edge)))
            value = float(true_rb[sample, edge])
            labels.append(int(_encode_values(np.array([value], dtype=np.float32), codebooks[step])[0]))
            baseline_values.append(float(pred_rb[sample, edge]))
            true_values.append(value)
    coords = np.asarray(coordinates, dtype=np.int64).reshape(-1, 3)
    features = _feature_rows(baseline, coords) if len(coords) else np.zeros((0, 12), dtype=np.float32)
    return RbTotalExamples(
        features=features,
        labels=np.asarray(labels, dtype=np.int64),
        coordinates=coords,
        baseline_values=np.asarray(baseline_values, dtype=np.float32),
        true_values=np.asarray(true_values, dtype=np.float32),
    )


def make_critical_examples(baseline: np.ndarray, truth: np.ndarray, steps: tuple[int, ...] = (1, 2)) -> RbTotalExamples:
    baseline = np.asarray(baseline, dtype=np.float32)
    truth = np.asarray(truth, dtype=np.float32)
    coordinates = []
    labels = []
    baseline_values = []
    true_values = []
    for step in steps:
        pred_rb = baseline[:, step, :, RB_DIM]
        true_rb = truth[:, step, :, RB_DIM]
        sample_idx, edge_idx = np.where(pred_rb > EPS)
        for sample, edge in zip(sample_idx, edge_idx):
            coordinates.append((int(sample), int(step), int(edge)))
            value = float(true_rb[sample, edge])
            labels.append(1 if value >= MIN_EFFECTIVE_RB_TOTAL else 0)
            baseline_values.append(float(pred_rb[sample, edge]))
            true_values.append(value)
    coords = np.asarray(coordinates, dtype=np.int64).reshape(-1, 3)
    features = _feature_rows(baseline, coords) if len(coords) else np.zeros((0, 12), dtype=np.float32)
    return RbTotalExamples(
        features=features,
        labels=np.asarray(labels, dtype=np.int64),
        coordinates=coords,
        baseline_values=np.asarray(baseline_values, dtype=np.float32),
        true_values=np.asarray(true_values, dtype=np.float32),
    )


def make_teacher_critical_labels(
    coordinates: np.ndarray,
    truth: np.ndarray,
    step_improvement: np.ndarray,
    min_effective_value: float = MIN_EFFECTIVE_RB_TOTAL,
    min_improvement: float = 0.0,
) -> np.ndarray:
    coordinates = np.asarray(coordinates, dtype=np.int64).reshape(-1, 3)
    truth = np.asarray(truth, dtype=np.float32)
    step_improvement = np.asarray(step_improvement, dtype=np.float32)
    labels = np.zeros((coordinates.shape[0],), dtype=np.int64)
    for row_idx, (sample, step, edge) in enumerate(coordinates):
        if truth[sample, step, edge, RB_DIM] >= float(min_effective_value) and step_improvement[sample, step] > float(min_improvement):
            labels[row_idx] = 1
    return labels


def first_order_edge_improvement(
    gradient: np.ndarray,
    baseline: np.ndarray,
    truth: np.ndarray,
    rb_dim: int = RB_DIM,
) -> np.ndarray:
    gradient = np.asarray(gradient, dtype=np.float32)
    baseline = np.asarray(baseline, dtype=np.float32)
    truth = np.asarray(truth, dtype=np.float32)
    return (-gradient[..., rb_dim] * (truth[..., rb_dim] - baseline[..., rb_dim])).astype(np.float32)


def make_edge_teacher_labels(
    coordinates: np.ndarray,
    truth: np.ndarray,
    edge_improvement: np.ndarray,
    min_effective_value: float = MIN_EFFECTIVE_RB_TOTAL,
    min_improvement: float = 0.0,
) -> np.ndarray:
    coordinates = np.asarray(coordinates, dtype=np.int64).reshape(-1, 3)
    truth = np.asarray(truth, dtype=np.float32)
    edge_improvement = np.asarray(edge_improvement, dtype=np.float32)
    labels = np.zeros((coordinates.shape[0],), dtype=np.int64)
    for row_idx, (sample, step, edge) in enumerate(coordinates):
        if truth[sample, step, edge, RB_DIM] >= float(min_effective_value) and edge_improvement[sample, step, edge] > float(min_improvement):
            labels[row_idx] = 1
    return labels


def decode_probabilities(
    probabilities: np.ndarray,
    codebook: StepCodebook,
    decoder: str = "argmax",
) -> tuple[np.ndarray, np.ndarray]:
    probs = np.asarray(probabilities, dtype=np.float32)
    values = np.asarray(codebook.values, dtype=np.float32)
    if probs.ndim != 2 or probs.shape[1] != values.shape[0]:
        raise ValueError("probabilities must have shape [N, codebook_size]")
    confidence = np.max(probs, axis=1).astype(np.float32)
    if decoder == "argmax":
        decoded = values[np.argmax(probs, axis=1)]
    elif decoder == "expected":
        decoded = probs @ values
    elif decoder == "conservative":
        cumulative = np.cumsum(probs, axis=1)
        idx = np.argmax(cumulative >= 0.4, axis=1)
        decoded = values[idx]
    else:
        raise ValueError(f"unknown decoder: {decoder}")
    return decoded.astype(np.float32), confidence


def apply_value_head_repair(
    actions: np.ndarray,
    coordinates: np.ndarray,
    values: np.ndarray,
    confidence: np.ndarray,
    min_confidence: float = 0.0,
    zero_low_confidence: bool = False,
) -> np.ndarray:
    repaired = np.asarray(actions, dtype=np.float32).copy()
    coords = np.asarray(coordinates, dtype=np.int64).reshape(-1, 3)
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    confidence = np.asarray(confidence, dtype=np.float32).reshape(-1)
    if not (coords.shape[0] == values.shape[0] == confidence.shape[0]):
        raise ValueError("coordinates, values, and confidence must have the same row count")
    for (sample, step, edge), value, conf in zip(coords, values, confidence):
        if step == 0:
            continue
        if actions[sample, step, edge, RB_DIM] <= EPS:
            continue
        if conf >= min_confidence:
            repaired[sample, step, edge, RB_DIM] = max(float(value), 0.0)
        elif zero_low_confidence:
            repaired[sample, step, edge, RB_DIM] = 0.0
    repaired[:, 0, :, RB_DIM] = actions[:, 0, :, RB_DIM]
    repaired[..., RB_DIM] = np.where(actions[..., RB_DIM] > EPS, np.clip(repaired[..., RB_DIM], 0.0, None), 0.0)
    return repaired


def apply_topk_score_repair(
    actions: np.ndarray,
    coordinates: np.ndarray,
    values: np.ndarray,
    scores: np.ndarray,
    top_k: int,
) -> np.ndarray:
    repaired = np.asarray(actions, dtype=np.float32).copy()
    coords = np.asarray(coordinates, dtype=np.int64).reshape(-1, 3)
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if not (coords.shape[0] == values.shape[0] == scores.shape[0]):
        raise ValueError("coordinates, values, and scores must have the same row count")
    if top_k <= 0 or coords.shape[0] == 0:
        return repaired
    selected = np.argsort(-scores)[: min(int(top_k), coords.shape[0])]
    for idx in selected:
        sample, step, edge = coords[idx]
        if step == 0 or actions[sample, step, edge, RB_DIM] <= EPS:
            continue
        repaired[sample, step, edge, RB_DIM] = max(float(values[idx]), 0.0)
    repaired[:, 0, :, RB_DIM] = actions[:, 0, :, RB_DIM]
    repaired[..., RB_DIM] = np.where(actions[..., RB_DIM] > EPS, np.clip(repaired[..., RB_DIM], 0.0, None), 0.0)
    return repaired


def _standardize(train: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, ...]:
    mean = np.mean(train, axis=0, keepdims=True).astype(np.float32)
    std = np.std(train, axis=0, keepdims=True).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std)
    arrays = [((train - mean) / std).astype(np.float32)]
    arrays.extend(((arr - mean) / std).astype(np.float32) for arr in others)
    return tuple(arrays)


def _class_weights(labels: np.ndarray, class_count: int) -> torch.Tensor:
    counts = np.bincount(labels.astype(np.int64), minlength=class_count).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    weights = np.sqrt(np.sum(counts) / counts)
    weights = weights / np.mean(weights)
    return torch.as_tensor(weights, dtype=torch.float32)


def make_balanced_indices(labels: np.ndarray, positive_multiplier: int = 1, seed: int = 0) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    base = np.arange(labels.shape[0], dtype=np.int64)
    if positive_multiplier <= 1 or labels.shape[0] == 0:
        return base
    positive = np.where(labels == 1)[0]
    if positive.size == 0:
        return base
    rng = np.random.default_rng(seed)
    extra = rng.choice(positive, size=int((positive_multiplier - 1) * positive.size), replace=True)
    return np.concatenate([base, extra.astype(np.int64)])


def binary_focal_cross_entropy(logits: torch.Tensor, labels: torch.Tensor, gamma: float = 2.0, weight: torch.Tensor | None = None) -> torch.Tensor:
    log_prob = torch.log_softmax(logits, dim=-1)
    prob = torch.exp(log_prob)
    gather_index = labels.view(-1, 1)
    target_log_prob = log_prob.gather(1, gather_index).squeeze(1)
    target_prob = prob.gather(1, gather_index).squeeze(1)
    loss = -((1.0 - target_prob).clamp_min(0.0) ** float(gamma)) * target_log_prob
    if weight is not None:
        loss = loss * weight.to(logits.device)[labels]
    return loss.mean()


def fit_retrieval_prototypes(
    features: np.ndarray,
    labels: np.ndarray,
    values: np.ndarray,
    max_prototypes: int = 512,
) -> RetrievalPrototypeModel:
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    values = np.asarray(values, dtype=np.float32)
    positive_idx = np.where(labels == 1)[0]
    if positive_idx.size == 0:
        return RetrievalPrototypeModel(features=np.zeros((0, features.shape[1]), dtype=np.float32), values=np.zeros((0,), dtype=np.float32))
    if positive_idx.size > max_prototypes:
        order = np.linspace(0, positive_idx.size - 1, max_prototypes).round().astype(np.int64)
        positive_idx = positive_idx[order]
    return RetrievalPrototypeModel(features=features[positive_idx].astype(np.float32), values=values[positive_idx].astype(np.float32))


def predict_retrieval_values(features: np.ndarray, model: RetrievalPrototypeModel) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = np.asarray(features, dtype=np.float32)
    if model.features.shape[0] == 0:
        return (
            np.zeros((features.shape[0],), dtype=np.float32),
            np.zeros((features.shape[0],), dtype=np.float32),
            np.full((features.shape[0],), np.inf, dtype=np.float32),
        )
    diff = features[:, None, :] - model.features[None, :, :]
    distance = np.sqrt(np.mean(diff * diff, axis=2))
    nearest = np.argmin(distance, axis=1)
    nearest_distance = distance[np.arange(features.shape[0]), nearest].astype(np.float32)
    confidence = (1.0 / (1.0 + nearest_distance)).astype(np.float32)
    return model.values[nearest].astype(np.float32), confidence, nearest_distance


def train_classifier(
    features: np.ndarray,
    labels: np.ndarray,
    output_dim: int,
    hidden_dim: int,
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int,
    loss_mode: str = "ce",
    focal_gamma: float = 2.0,
    positive_multiplier: int = 1,
) -> TinyHead:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = TinyHead(features.shape[1], output_dim, hidden_dim=hidden_dim)
    if features.shape[0] == 0:
        return model
    row_idx = make_balanced_indices(labels, positive_multiplier=positive_multiplier, seed=seed) if output_dim == 2 else np.arange(labels.shape[0], dtype=np.int64)
    x = torch.as_tensor(features[row_idx], dtype=torch.float32)
    y = torch.as_tensor(labels[row_idx], dtype=torch.long)
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)
    class_weight = _class_weights(labels, output_dim)
    criterion = nn.CrossEntropyLoss(weight=class_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    model.train()
    for _ in range(epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            if loss_mode == "focal" and output_dim == 2:
                loss = binary_focal_cross_entropy(logits, batch_y, gamma=focal_gamma, weight=class_weight)
            else:
                loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
    model.eval()
    return model


def train_regressor(
    features: np.ndarray,
    targets: np.ndarray,
    hidden_dim: int,
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int,
) -> TinyRegressor:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = TinyRegressor(features.shape[1], hidden_dim=hidden_dim)
    if features.shape[0] == 0:
        return model
    x = torch.as_tensor(features, dtype=torch.float32)
    y = torch.as_tensor(targets, dtype=torch.float32)
    weights = torch.ones_like(y)
    positive = y > 0
    if bool(torch.any(positive)):
        weights[positive] = max(1.0, float((~positive).sum().item()) / max(1.0, float(positive.sum().item()))) ** 0.5
    loader = DataLoader(TensorDataset(x, y, weights), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    model.train()
    for _ in range(epochs):
        for batch_x, batch_y, batch_w in loader:
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch_x)
            loss = torch.mean(batch_w * torch.nn.functional.smooth_l1_loss(pred, batch_y, reduction="none"))
            loss.backward()
            optimizer.step()
    model.eval()
    return model


def train_pairwise_ranker(
    features: np.ndarray,
    targets: np.ndarray,
    hidden_dim: int,
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int,
    pairs_per_epoch: int = 4096,
    min_target_gap: float = 1e-6,
) -> TinyRegressor:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = TinyRegressor(features.shape[1], hidden_dim=hidden_dim)
    features = np.asarray(features, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32).reshape(-1)
    if features.shape[0] < 2:
        return model
    positive = np.where(targets > float(min_target_gap))[0]
    negative = np.where(targets <= float(min_target_gap))[0]
    if positive.size == 0 or negative.size == 0:
        return train_regressor(features, targets, hidden_dim, epochs, lr, batch_size, seed)
    rng = np.random.default_rng(seed)
    x = torch.as_tensor(features, dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    model.train()
    for _ in range(epochs):
        pos_idx = rng.choice(positive, size=int(pairs_per_epoch), replace=True)
        neg_idx = rng.choice(negative, size=int(pairs_per_epoch), replace=True)
        target_gap = targets[pos_idx] - targets[neg_idx]
        keep = target_gap > float(min_target_gap)
        if not np.any(keep):
            continue
        pos_idx = pos_idx[keep]
        neg_idx = neg_idx[keep]
        order = rng.permutation(pos_idx.shape[0])
        for start in range(0, order.shape[0], batch_size):
            batch_order = order[start:start + batch_size]
            batch_pos = torch.as_tensor(pos_idx[batch_order], dtype=torch.long)
            batch_neg = torch.as_tensor(neg_idx[batch_order], dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            pos_score = model(x[batch_pos])
            neg_score = model(x[batch_neg])
            loss = torch.nn.functional.softplus(-(pos_score - neg_score)).mean()
            loss.backward()
            optimizer.step()
    model.eval()
    return model


def predict_scores(model: nn.Module, features: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    if features.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    rows = []
    with torch.no_grad():
        for start in range(0, features.shape[0], batch_size):
            batch = torch.as_tensor(features[start:start + batch_size], dtype=torch.float32)
            rows.append(model(batch).cpu().numpy())
    return np.concatenate(rows, axis=0).astype(np.float32)


def predict_probabilities(model: nn.Module, features: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    if features.shape[0] == 0:
        output_dim = model.net[-1].out_features if isinstance(model, TinyHead) else 0
        return np.zeros((0, output_dim), dtype=np.float32)
    rows = []
    with torch.no_grad():
        for start in range(0, features.shape[0], batch_size):
            batch = torch.as_tensor(features[start:start + batch_size], dtype=torch.float32)
            rows.append(torch.softmax(model(batch), dim=-1).cpu().numpy())
    return np.concatenate(rows, axis=0).astype(np.float32)


def _candidate_actions(
    baseline_actions: np.ndarray,
    examples: RbTotalExamples,
    critical_probs: np.ndarray,
    value_probs_by_step: dict[int, np.ndarray],
    codebooks: dict[int, StepCodebook],
    decoder: str,
    critical_threshold: float,
    value_conf_threshold: float,
    mode: str,
) -> np.ndarray:
    if examples.coordinates.shape[0] == 0:
        return baseline_actions.copy()
    replacement_values = np.zeros((examples.coordinates.shape[0],), dtype=np.float32)
    value_conf = np.zeros_like(replacement_values)
    for step, codebook in codebooks.items():
        step_mask = examples.coordinates[:, 1] == step
        if not np.any(step_mask):
            continue
        decoded, conf = decode_probabilities(value_probs_by_step[step], codebook, decoder=decoder)
        replacement_values[step_mask] = decoded
        value_conf[step_mask] = conf
    critical_pos_prob = critical_probs[:, 1] if critical_probs.shape[1] > 1 else np.ones(examples.coordinates.shape[0], dtype=np.float32)
    confidence = np.minimum(critical_pos_prob, value_conf)
    if mode == "repair_only":
        eligible = (critical_pos_prob >= critical_threshold) & (value_conf >= value_conf_threshold)
        return apply_value_head_repair(baseline_actions, examples.coordinates, replacement_values, eligible.astype(np.float32), min_confidence=0.5)
    if mode == "zero_lowcrit_repair_high":
        repaired = apply_value_head_repair(
            baseline_actions,
            examples.coordinates,
            replacement_values,
            ((critical_pos_prob >= critical_threshold) & (value_conf >= value_conf_threshold)).astype(np.float32),
            min_confidence=0.5,
        )
        low = critical_pos_prob < critical_threshold
        zero_values = np.zeros_like(replacement_values)
        repaired = apply_value_head_repair(
            repaired,
            examples.coordinates,
            zero_values,
            low.astype(np.float32),
            min_confidence=0.5,
            zero_low_confidence=False,
        )
        return repaired
    if mode == "zero_lowcrit_only":
        low = critical_pos_prob < critical_threshold
        return apply_value_head_repair(
            baseline_actions,
            examples.coordinates,
            np.zeros_like(replacement_values),
            low.astype(np.float32),
            min_confidence=0.5,
            zero_low_confidence=False,
        )
    raise ValueError(f"unknown candidate mode: {mode}")


def make_step_dim_true_rb_actions(baseline: np.ndarray, truth: np.ndarray, step: int) -> np.ndarray:
    repaired = np.asarray(baseline, dtype=np.float32).copy()
    mask = baseline[:, step, :, RB_DIM] > EPS
    repaired[:, step, :, RB_DIM][mask] = truth[:, step, :, RB_DIM][mask]
    repaired[:, 0, :, RB_DIM] = baseline[:, 0, :, RB_DIM]
    repaired[..., RB_DIM] = np.where(baseline[..., RB_DIM] > EPS, np.clip(repaired[..., RB_DIM], 0.0, None), 0.0)
    return repaired


def sample_step_improvement_from_predictions(
    baseline_predictions: dict[str, np.ndarray],
    repaired_predictions: dict[str, np.ndarray],
) -> np.ndarray:
    truth = baseline_predictions["link_rate_true"].squeeze(-1)
    active = baseline_predictions["link_activity_true"].squeeze(-1) > 0.5
    base_pred = baseline_predictions["link_rate_pred"].squeeze(-1)
    repaired_pred = repaired_predictions["link_rate_pred"].squeeze(-1)
    base_error = (base_pred - truth) ** 2
    repaired_error = (repaired_pred - truth) ** 2
    improvement = np.sum((base_error - repaired_error) * active, axis=2)
    return improvement.astype(np.float32)


def _select_link_rate_tensor(outputs: dict[str, torch.Tensor], rate_output_mode: str) -> torch.Tensor:
    if rate_output_mode == "main":
        return outputs["link_rate"]
    if rate_output_mode == "positive" and "link_positive_rate" in outputs:
        return outputs["link_positive_rate"]
    if rate_output_mode == "hurdle_dual" and "link_hurdle_rate" in outputs:
        return outputs["link_hurdle_rate"]
    if rate_output_mode == "hurdle_mass" and "link_active_mass_rate" in outputs:
        return outputs["link_active_mass_rate"]
    return outputs["link_rate"]


def denormalize_link_rate_tensor(
    normalized_rate: torch.Tensor,
    stats: dict,
    baseline: torch.Tensor | None,
) -> torch.Tensor:
    mean_np, std_np = stats["y_link_rate"]
    mean = torch.as_tensor(mean_np, dtype=normalized_rate.dtype, device=normalized_rate.device)
    std = torch.as_tensor(std_np, dtype=normalized_rate.dtype, device=normalized_rate.device)
    raw = normalized_rate * std + mean
    transform = stats.get("rate_target_transform", "raw")
    if transform == "log1p_raw":
        raw = torch.expm1(raw).clamp_min(0.0)
    elif transform == "residual_last_rate":
        if baseline is None:
            raise ValueError("residual_last_rate requires baseline")
        raw = raw + baseline
    return raw


class RawFutureActionSingleDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, raw_actions: np.ndarray, stats: dict) -> None:
        self.base_dataset = base_dataset
        self.raw_actions = np.asarray(raw_actions, dtype=np.float32)
        self.stats = stats
        if len(base_dataset) != self.raw_actions.shape[0]:
            raise ValueError("base_dataset length must match raw action rows")

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, item: int):
        base_batch, target = self.base_dataset[item]
        mean, std = self.stats["edge_a_future"]
        normalized = (self.raw_actions[item] - mean[0]) / std[0]
        batch = V6DualGraphBatch(
            node_history=base_batch.node_history,
            physical_edge_history=base_batch.physical_edge_history,
            info_edge_history=base_batch.info_edge_history,
            action_history=base_batch.action_history,
            future_actions=torch.as_tensor(normalized, dtype=torch.float32),
            task_history=base_batch.task_history,
            link_rate_baseline=base_batch.link_rate_baseline,
        )
        return batch, target


def collate_raw_future_action_batch(items):
    batches, targets = zip(*items)
    return V6DualGraphBatch(
        node_history=torch.stack([item.node_history for item in batches]),
        physical_edge_history=torch.stack([item.physical_edge_history for item in batches]),
        info_edge_history=torch.stack([item.info_edge_history for item in batches]),
        action_history=torch.stack([item.action_history for item in batches]),
        future_actions=torch.stack([item.future_actions for item in batches]),
        task_history=torch.stack([item.task_history for item in batches]),
        link_rate_baseline=torch.stack([item.link_rate_baseline for item in batches]) if batches[0].link_rate_baseline is not None else None,
    ), {
        key: torch.stack([target[key] for target in targets])
        for key in targets[0]
    }


def collect_edge_gradient_improvement(
    world_model,
    base_dataset,
    baseline_actions: np.ndarray,
    truth_actions: np.ndarray,
    stats: dict,
    world_config: dict,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    world_model.eval()
    dataset = RawFutureActionSingleDataset(base_dataset, baseline_actions, stats)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_raw_future_action_batch)
    rows = []
    rate_output_mode = world_config.get("rate_output_mode", "main")
    for batch, target in loader:
        batch = V6DualGraphBatch(
            node_history=batch.node_history.to(device),
            physical_edge_history=batch.physical_edge_history.to(device),
            info_edge_history=batch.info_edge_history.to(device),
            action_history=batch.action_history.to(device),
            future_actions=batch.future_actions.to(device).detach().clone().requires_grad_(True),
            task_history=batch.task_history.to(device),
            link_rate_baseline=(batch.link_rate_baseline.to(device) if batch.link_rate_baseline is not None else None),
        )
        target_rate = target["link_rate"].to(device)
        target_activity = target["link_activity"].to(device) > 0.5
        world_model.zero_grad(set_to_none=True)
        outputs = world_model(batch)
        selected_rate = _select_link_rate_tensor(outputs, rate_output_mode)
        pred_raw = denormalize_link_rate_tensor(selected_rate, stats, batch.link_rate_baseline)
        true_raw = denormalize_link_rate_tensor(target_rate, stats, batch.link_rate_baseline)
        loss = torch.sum(((pred_raw - true_raw) ** 2) * target_activity.to(pred_raw.dtype))
        loss.backward()
        grad_norm = batch.future_actions.grad.detach().cpu().numpy()
        mean, std = stats["edge_a_future"]
        grad_raw = grad_norm / std[0]
        start = len(rows) * batch_size
        stop = start + grad_raw.shape[0]
        improvement = first_order_edge_improvement(grad_raw, baseline_actions[start:stop], truth_actions[start:stop], rb_dim=RB_DIM)
        rows.append(improvement)
    return np.concatenate(rows, axis=0).astype(np.float32)


def _make_inference_examples(baseline: np.ndarray, truth_shape: tuple[int, ...], steps: tuple[int, ...]) -> RbTotalExamples:
    dummy_truth = np.zeros(truth_shape, dtype=np.float32)
    return make_critical_examples(baseline, dummy_truth, steps=steps)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def limit_indices(indices: np.ndarray, limit: int | None) -> np.ndarray:
    indices = np.asarray(indices)
    if limit is None or limit <= 0 or limit >= len(indices):
        return indices
    return indices[: int(limit)]


def select_context_indices(
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    max_train_samples: int = 0,
    max_val_samples: int = 0,
    max_test_samples: int = 0,
    limit_after_stats: bool = False,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    train_idx = np.asarray(train_idx, dtype=np.int64)
    val_idx = np.asarray(val_idx, dtype=np.int64)
    test_idx = np.asarray(test_idx, dtype=np.int64)
    if limit_after_stats:
        stats_train_idx = train_idx
        split_train_idx = limit_indices(train_idx, max_train_samples)
    else:
        split_train_idx = limit_indices(train_idx, max_train_samples)
        stats_train_idx = split_train_idx
    return stats_train_idx, {
        "train": split_train_idx,
        "val": limit_indices(val_idx, max_val_samples),
        "test": limit_indices(test_idx, max_test_samples),
    }


def streaming_fit_stats(chunks) -> tuple[np.ndarray, np.ndarray]:
    total_count = 0
    total_sum = None
    total_sumsq = None
    for chunk in chunks:
        values = np.asarray(chunk, dtype=np.float64)
        if values.size == 0:
            continue
        axes = (0,)
        chunk_count = values.shape[0]
        chunk_sum = values.sum(axis=axes, keepdims=True)
        chunk_sumsq = np.square(values).sum(axis=axes, keepdims=True)
        total_count += int(chunk_count)
        total_sum = chunk_sum if total_sum is None else total_sum + chunk_sum
        total_sumsq = chunk_sumsq if total_sumsq is None else total_sumsq + chunk_sumsq
    if total_count <= 0 or total_sum is None or total_sumsq is None:
        raise ValueError("streaming_fit_stats requires at least one non-empty chunk")
    mean = total_sum / float(total_count)
    variance = np.maximum(total_sumsq / float(total_count) - np.square(mean), 0.0)
    std = np.sqrt(variance)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def _index_chunks(indices: np.ndarray, chunk_size: int):
    indices = np.asarray(indices, dtype=np.int64)
    for start in range(0, len(indices), max(1, int(chunk_size))):
        yield indices[start:start + max(1, int(chunk_size))]


def make_normalization_stats_streaming(
    arrays: dict[str, np.ndarray],
    train_idx: np.ndarray,
    rate_target_transform: str = "raw",
    chunk_size: int = 512,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    train_idx = np.asarray(train_idx, dtype=np.int64)
    return {
        "rate_target_transform": rate_target_transform,
        "x_node": streaming_fit_stats(arrays["x_node"][chunk] for chunk in _index_chunks(train_idx, chunk_size)),
        "x_link": streaming_fit_stats(arrays["x_link"][chunk] for chunk in _index_chunks(train_idx, chunk_size)),
        "x_physical_edge": streaming_fit_stats(
            build_physical_edge_history(
                arrays["x_node"][chunk],
                arrays["edge_src_idx"],
                arrays["edge_dst_idx"],
                arrays["valid_edge_node"],
            ).numpy()
            for chunk in _index_chunks(train_idx, chunk_size)
        ),
        "x_task": streaming_fit_stats(arrays["x_task"][chunk] for chunk in _index_chunks(train_idx, chunk_size)),
        "edge_a_hist": streaming_fit_stats(arrays["edge_a_hist"][chunk] for chunk in _index_chunks(train_idx, chunk_size)),
        "edge_a_future": streaming_fit_stats(arrays["edge_a_future"][chunk] for chunk in _index_chunks(train_idx, chunk_size)),
        "y_node": streaming_fit_stats(arrays["y_node"][chunk] for chunk in _index_chunks(train_idx, chunk_size)),
        "y_task": streaming_fit_stats(arrays["y_task"][chunk] for chunk in _index_chunks(train_idx, chunk_size)),
        "y_link_rate": streaming_fit_stats(
            build_link_rate_target(arrays, chunk, rate_target_transform)
            for chunk in _index_chunks(train_idx, chunk_size)
        ),
    }


def load_context_limited(args: argparse.Namespace, device: torch.device):
    if args.max_train_samples <= 0 and args.max_val_samples <= 0 and args.max_test_samples <= 0:
        return load_context(args, device)
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
    stats_train_idx, split_indices = select_context_indices(
        train_idx,
        val_idx,
        test_idx,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        max_test_samples=args.max_test_samples,
        limit_after_stats=args.limit_after_stats,
    )
    stats = (
        make_normalization_stats_streaming(arrays, stats_train_idx, chunk_size=args.stats_chunk_size)
        if args.streaming_stats
        else make_normalization_stats(arrays, stats_train_idx)
    )
    world_model = load_model_for_experiment(summary, arrays, args.world_checkpoint, device)
    policy_model, action_scale, learned_threshold, value_vocab = load_policy(args.policy_checkpoint, device)
    return summary, arrays, stats, world_model, policy_model, action_scale, value_vocab, split_indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-experiment-dir", type=Path, default=PROJECT_ROOT / "artifacts/experiments/pi_jwm_v9_expanded_v2_gpu_20260619/v2_hurdle_baseline")
    parser.add_argument("--world-checkpoint", type=Path, default=PROJECT_ROOT / "artifacts/experiments/pi_jwm_v9_expanded_v2_gpu_20260619/v2_hurdle_baseline/checkpoints/v8_dual_best.pt")
    parser.add_argument("--policy-checkpoint", type=Path, default=PROJECT_ROOT / "artifacts/experiments/pi_jwm_v10_policy_bridge_gpu_20260620/v10_action_policy_bc/checkpoints/v7_action_policy_cross_attention_best.pt")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--policy-threshold", type=float, default=0.4)
    parser.add_argument("--value-scale", type=float, default=1.0)
    parser.add_argument("--new-policy-threshold", type=float, default=0.37)
    parser.add_argument("--new-value-scale", type=float, default=1.06)
    parser.add_argument("--gate-feature", choices=("step_rb_total", "step_cpu_total", "step_rb_cpu_total", "step_active_count"), default="step_rb_cpu_total")
    parser.add_argument("--gate-threshold", type=float, default=450.0)
    parser.add_argument("--value-codebook-size", type=int, default=9)
    parser.add_argument("--rb-bin-count", type=int, default=9)
    parser.add_argument("--min-effective-rb-total", type=float, default=MIN_EFFECTIVE_RB_TOTAL)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--head-batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--steps", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--quick-sweep", action="store_true", help="Evaluate a small validation candidate set before wider CPU sweeps.")
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--include-state-features", action="store_true")
    parser.add_argument("--critical-label-mode", choices=("truth_positive", "teacher_step", "teacher_edge_grad"), default="truth_positive")
    parser.add_argument("--teacher-min-improvement", type=float, default=0.0)
    parser.add_argument("--critical-loss-mode", choices=("ce", "focal"), default="ce")
    parser.add_argument("--critical-focal-gamma", type=float, default=2.0)
    parser.add_argument("--critical-positive-multiplier", type=int, default=1)
    parser.add_argument("--score-loss-mode", choices=("regression", "pairwise"), default="regression")
    parser.add_argument("--score-pairs-per-epoch", type=int, default=4096)
    parser.add_argument("--score-min-target-gap", type=float, default=1e-6)
    parser.add_argument("--retrieval-max-prototypes", type=int, default=512)
    parser.add_argument("--limit-after-stats", action="store_true", help="Use full train stats, then limit split indices for comparable smoke evaluation.")
    parser.add_argument("--streaming-stats", action="store_true", help="Compute normalization stats in chunks to avoid large temporary arrays.")
    parser.add_argument("--stats-chunk-size", type=int, default=512)
    return parser.parse_args()


def run_experiment(args: argparse.Namespace) -> dict:
    started = time.time()
    device = torch.device("cpu")
    if args.limit_after_stats and not args.streaming_stats:
        context = load_context(args, device)
    else:
        context = load_context_limited(args, device)
    summary, arrays, stats, world_model, policy_model, action_scale, value_vocab, splits = context
    splits = dict(splits)
    if args.limit_after_stats:
        splits["train"] = limit_indices(splits["train"], args.max_train_samples)
        splits["val"] = limit_indices(splits["val"], args.max_val_samples)
        splits["test"] = limit_indices(splits["test"], args.max_test_samples)
    world_model = freeze_module(world_model)
    policy_model = freeze_module(policy_model)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    steps = tuple(int(step) for step in args.steps)

    train_base, train_dataset = make_adaptive_dataset(args, arrays, splits["train"], stats, policy_model, action_scale, value_vocab, device, splits["train"])
    train_actions, train_truth = collect_raw_actions(train_dataset, stats)
    global MIN_EFFECTIVE_RB_TOTAL
    MIN_EFFECTIVE_RB_TOTAL = float(args.min_effective_rb_total)
    codebooks = build_step_codebooks(train_truth, steps=steps, bin_count=args.rb_bin_count, min_effective_value=args.min_effective_rb_total)
    critical_train = make_critical_examples(train_actions, train_truth, steps=steps)
    value_train = make_rb_total_examples(train_actions, train_truth, codebooks, steps=steps, min_effective_value=args.min_effective_rb_total)
    teacher_step_improvement = None
    teacher_edge_improvement = None
    if args.critical_label_mode == "teacher_step":
        baseline_train_predictions = evaluate_raw_actions(
            train_actions, train_base, stats, world_model, summary["config"], device, args.batch_size
        )
        teacher_step_improvement = np.zeros((train_actions.shape[0], train_actions.shape[1]), dtype=np.float32)
        for step in steps:
            repaired_train_actions = make_step_dim_true_rb_actions(train_actions, train_truth, step)
            repaired_predictions = evaluate_raw_actions(
                repaired_train_actions, train_base, stats, world_model, summary["config"], device, args.batch_size
            )
            improvement = sample_step_improvement_from_predictions(baseline_train_predictions, repaired_predictions)
            teacher_step_improvement[:, step] = improvement[:, step]
        critical_train = RbTotalExamples(
            features=critical_train.features,
            labels=make_teacher_critical_labels(
                critical_train.coordinates,
                train_truth,
                teacher_step_improvement,
                min_effective_value=args.min_effective_rb_total,
                min_improvement=args.teacher_min_improvement,
            ),
            coordinates=critical_train.coordinates,
            baseline_values=critical_train.baseline_values,
            true_values=critical_train.true_values,
        )
    elif args.critical_label_mode == "teacher_edge_grad":
        teacher_edge_improvement = collect_edge_gradient_improvement(
            world_model,
            train_base,
            train_actions,
            train_truth,
            stats,
            summary["config"],
            device,
            args.batch_size,
        )
        critical_train = RbTotalExamples(
            features=critical_train.features,
            labels=make_edge_teacher_labels(
                critical_train.coordinates,
                train_truth,
                teacher_edge_improvement,
                min_effective_value=args.min_effective_rb_total,
                min_improvement=args.teacher_min_improvement,
            ),
            coordinates=critical_train.coordinates,
            baseline_values=critical_train.baseline_values,
            true_values=critical_train.true_values,
        )
    train_state_features = extract_state_features(train_base) if args.include_state_features else None
    critical_train = RbTotalExamples(
        features=append_state_features(critical_train.features, train_state_features, critical_train.coordinates),
        labels=critical_train.labels,
        coordinates=critical_train.coordinates,
        baseline_values=critical_train.baseline_values,
        true_values=critical_train.true_values,
    )
    value_train = RbTotalExamples(
        features=append_state_features(value_train.features, train_state_features, value_train.coordinates),
        labels=value_train.labels,
        coordinates=value_train.coordinates,
        baseline_values=value_train.baseline_values,
        true_values=value_train.true_values,
    )

    split_payload = {}
    for split_name in ("val", "test"):
        base_dataset, adaptive_dataset = make_adaptive_dataset(args, arrays, splits[split_name], stats, policy_model, action_scale, value_vocab, device, splits["train"])
        baseline_actions, truth_actions = collect_raw_actions(adaptive_dataset, stats)
        split_payload[split_name] = {
            "base_dataset": base_dataset,
            "baseline_actions": baseline_actions,
            "truth_actions": truth_actions,
            "critical_examples": _make_inference_examples(baseline_actions, truth_actions.shape, steps),
            "state_features": extract_state_features(base_dataset) if args.include_state_features else None,
        }

    feature_sets = [critical_train.features]
    for payload in split_payload.values():
        examples = payload["critical_examples"]
        payload["critical_examples"] = RbTotalExamples(
            features=append_state_features(examples.features, payload["state_features"], examples.coordinates),
            labels=examples.labels,
            coordinates=examples.coordinates,
            baseline_values=examples.baseline_values,
            true_values=examples.true_values,
        )
        feature_sets.append(payload["critical_examples"].features)
    standardized = _standardize(*feature_sets)
    critical_train_features = standardized[0]
    offset = 1
    for split_name in ("val", "test"):
        split_payload[split_name]["critical_features"] = standardized[offset]
        offset += 1

    value_features_by_step = {}
    value_labels_by_step = {}
    value_models = {}
    for step, codebook in codebooks.items():
        mask = value_train.coordinates[:, 1] == step
        features = value_train.features[mask]
        labels = value_train.labels[mask]
        per_split_features = []
        for split_name in ("val", "test"):
            split_examples = split_payload[split_name]["critical_examples"]
            split_mask = split_examples.coordinates[:, 1] == step
            per_split_features.append(split_examples.features[split_mask])
        standardized_step = _standardize(features, *per_split_features) if features.shape[0] else (features, *per_split_features)
        value_features_by_step[step] = {"train": standardized_step[0]}
        split_idx = 1
        for split_name in ("val", "test"):
            value_features_by_step[step][split_name] = standardized_step[split_idx]
            split_idx += 1
        value_labels_by_step[step] = labels
        value_models[step] = train_classifier(
            value_features_by_step[step]["train"],
            labels,
            output_dim=len(codebook.values),
            hidden_dim=args.hidden_dim,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.head_batch_size,
            seed=args.seed + step,
        )

    critical_model = train_classifier(
        critical_train_features,
        critical_train.labels,
        output_dim=2,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.head_batch_size,
        seed=args.seed,
        loss_mode=args.critical_loss_mode,
        focal_gamma=args.critical_focal_gamma,
        positive_multiplier=args.critical_positive_multiplier,
    )
    score_model = None
    if teacher_edge_improvement is not None:
        score_targets = np.zeros((critical_train.coordinates.shape[0],), dtype=np.float32)
        for row_idx, (sample, step, edge) in enumerate(critical_train.coordinates):
            score_targets[row_idx] = np.log1p(max(float(teacher_edge_improvement[sample, step, edge]), 0.0))
        if args.score_loss_mode == "pairwise":
            score_model = train_pairwise_ranker(
                critical_train_features,
                score_targets,
                hidden_dim=args.hidden_dim,
                epochs=args.epochs,
                lr=args.lr,
                batch_size=args.head_batch_size,
                seed=args.seed + 1009,
                pairs_per_epoch=args.score_pairs_per_epoch,
                min_target_gap=args.score_min_target_gap,
            )
        else:
            score_model = train_regressor(
                critical_train_features,
                score_targets,
                hidden_dim=args.hidden_dim,
                epochs=args.epochs,
                lr=args.lr,
                batch_size=args.head_batch_size,
                seed=args.seed + 1009,
            )
    retrieval_model = fit_retrieval_prototypes(
        critical_train_features,
        critical_train.labels,
        critical_train.true_values,
        max_prototypes=args.retrieval_max_prototypes,
    )

    all_rows = []
    candidate_actions_cache = {}
    baseline_rmse_by_split = {}
    for split_name in ("val", "test"):
        payload = split_payload[split_name]
        baseline_predictions = evaluate_raw_actions(
            payload["baseline_actions"], payload["base_dataset"], stats, world_model, summary["config"], device, args.batch_size
        )
        baseline_row = active_rate_row("identity", split_name, baseline_predictions, float("nan"))
        baseline_row["improvement_vs_baseline"] = 0.0
        all_rows.append(baseline_row)
        baseline_rmse_by_split[split_name] = float(baseline_row["active_rate_rmse"])

    for split_name in ("val", "test"):
        payload = split_payload[split_name]
        examples = payload["critical_examples"]
        critical_probs = predict_probabilities(critical_model, payload["critical_features"])
        value_probs_by_step = {}
        for step in codebooks:
            step_mask = examples.coordinates[:, 1] == step
            probs = predict_probabilities(value_models[step], value_features_by_step[step][split_name])
            if probs.shape[0] != int(np.sum(step_mask)):
                raise RuntimeError("step probability row count mismatch")
            value_probs_by_step[step] = probs
        candidate_actions_cache[split_name] = {}
        if score_model is not None:
            predicted_scores = predict_scores(score_model, payload["critical_features"])
            score_repair_values = np.zeros((examples.coordinates.shape[0],), dtype=np.float32)
            for step, codebook in codebooks.items():
                step_mask = examples.coordinates[:, 1] == step
                if not np.any(step_mask):
                    continue
                decoded, conf = decode_probabilities(value_probs_by_step[step], codebook, decoder="argmax")
                score_repair_values[step_mask] = decoded
            for top_k in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512):
                name = f"score_topk__{top_k}"
                candidate_actions_cache[split_name][name] = apply_topk_score_repair(
                    payload["baseline_actions"],
                    examples.coordinates,
                    score_repair_values,
                    predicted_scores,
                    top_k=top_k,
                )
        retrieval_values, retrieval_confidence, retrieval_distance = predict_retrieval_values(payload["critical_features"], retrieval_model)
        for distance_threshold in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0):
            confidence = (retrieval_distance <= distance_threshold).astype(np.float32)
            name = f"retrieval_repair__dist{distance_threshold:g}"
            candidate_actions_cache[split_name][name] = apply_value_head_repair(
                payload["baseline_actions"],
                examples.coordinates,
                retrieval_values,
                confidence,
                min_confidence=0.5,
            )
        if args.quick_sweep:
            sweep_modes = ("repair_only", "zero_lowcrit_repair_high")
            sweep_decoders = ("argmax", "conservative")
            sweep_critical_thresholds = (0.05, 0.1, 0.2, 0.35, 0.5, 0.65)
            sweep_value_conf_thresholds = (0.0, 0.5)
        else:
            sweep_modes = ("repair_only", "zero_lowcrit_only", "zero_lowcrit_repair_high")
            sweep_decoders = ("argmax", "expected", "conservative")
            sweep_critical_thresholds = (0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8)
            sweep_value_conf_thresholds = (0.0, 0.35, 0.5, 0.65)
        for mode in sweep_modes:
            for decoder in sweep_decoders:
                for critical_threshold in sweep_critical_thresholds:
                    for value_conf_threshold in sweep_value_conf_thresholds:
                        if mode == "zero_lowcrit_only" and (decoder != "argmax" or value_conf_threshold != 0.0):
                            continue
                        name = f"{mode}__{decoder}__crit{critical_threshold:g}__vconf{value_conf_threshold:g}"
                        actions = _candidate_actions(
                            payload["baseline_actions"],
                            examples,
                            critical_probs,
                            value_probs_by_step,
                            codebooks,
                            decoder=decoder,
                            critical_threshold=critical_threshold,
                            value_conf_threshold=value_conf_threshold,
                            mode=mode,
                        )
                        candidate_actions_cache[split_name][name] = actions

    val_rows = []
    for name, actions in candidate_actions_cache["val"].items():
        payload = split_payload["val"]
        predictions = evaluate_raw_actions(actions, payload["base_dataset"], stats, world_model, summary["config"], device, args.batch_size)
        row = active_rate_row(name, "val", predictions, baseline_rmse_by_split["val"])
        val_rows.append(row)
        all_rows.append(row)
    val_ranked = sorted([row for row in all_rows if row["split"] == "val"], key=lambda row: (float(row["active_rate_rmse"]), str(row["candidate"])))
    selected = val_ranked[0]["candidate"]
    if selected != "identity":
        payload = split_payload["test"]
        predictions = evaluate_raw_actions(candidate_actions_cache["test"][selected], payload["base_dataset"], stats, world_model, summary["config"], device, args.batch_size)
        all_rows.append(active_rate_row(selected, "test", predictions, baseline_rmse_by_split["test"]))

    test_rows = [row for row in all_rows if row["split"] == "test"]
    write_csv(args.output_dir / "rb_total_value_head_results.csv", all_rows)
    write_csv(args.output_dir / "rb_total_value_head_val_ranked.csv", val_ranked)
    write_csv(args.output_dir / "rb_total_value_head_test_selected.csv", test_rows)

    codebook_json = {
        str(step): {"values": cb.values.astype(float).tolist(), "edges": cb.edges.astype(float).tolist()}
        for step, cb in codebooks.items()
    }
    diagnostics = {
        "train_critical_examples": int(critical_train.features.shape[0]),
        "train_critical_positive_rate": float(np.mean(critical_train.labels)) if critical_train.labels.size else float("nan"),
        "train_value_examples": int(value_train.features.shape[0]),
        "codebooks": codebook_json,
        "runtime_seconds": float(time.time() - started),
        "critical_label_mode": args.critical_label_mode,
        "teacher_min_improvement": float(args.teacher_min_improvement),
        "critical_loss_mode": args.critical_loss_mode,
        "critical_focal_gamma": float(args.critical_focal_gamma),
        "critical_positive_multiplier": int(args.critical_positive_multiplier),
        "score_loss_mode": args.score_loss_mode,
        "score_pairs_per_epoch": int(args.score_pairs_per_epoch),
        "score_min_target_gap": float(args.score_min_target_gap),
        "retrieval_prototype_count": int(retrieval_model.features.shape[0]),
    }
    if teacher_step_improvement is not None:
        positive = teacher_step_improvement[teacher_step_improvement > 0.0]
        diagnostics["teacher_positive_step_count"] = int(positive.size)
        diagnostics["teacher_positive_step_improvement_mean"] = float(np.mean(positive)) if positive.size else float("nan")
        diagnostics["teacher_positive_step_improvement_p90"] = float(np.quantile(positive, 0.9)) if positive.size else float("nan")
    if teacher_edge_improvement is not None:
        positive = teacher_edge_improvement[teacher_edge_improvement > 0.0]
        diagnostics["teacher_positive_edge_count"] = int(positive.size)
        diagnostics["teacher_positive_edge_improvement_mean"] = float(np.mean(positive)) if positive.size else float("nan")
        diagnostics["teacher_positive_edge_improvement_p90"] = float(np.quantile(positive, 0.9)) if positive.size else float("nan")
    write_json(args.output_dir / "diagnostics.json", diagnostics)

    best = val_ranked[0]
    report = [
        "# PI-JWM v11 rb_total Value Head CPU Probe",
        "",
        "CPU-only deployable probe. Model is selected by validation active-rate RMSE; test is reported only for the selected validation candidate.",
        "",
        "## Diagnostics",
        "",
        f"- train critical examples: {diagnostics['train_critical_examples']}",
        f"- train critical positive rate: {diagnostics['train_critical_positive_rate']:.6f}",
        f"- train value examples: {diagnostics['train_value_examples']}",
        f"- runtime seconds: {diagnostics['runtime_seconds']:.1f}",
        "",
        "## Validation Ranking",
        "",
        "| rank | candidate | val_active_rmse | improvement_vs_baseline | link_rmse |",
        "|---:|---|---:|---:|---:|",
    ]
    for rank, row in enumerate(val_ranked[:20], start=1):
        report.append(f"| {rank} | {row['candidate']} | {row['active_rate_rmse']:.6f} | {row['improvement_vs_baseline']:.6f} | {row['link_rmse']:.6f} |")
    report.extend(["", "## Decision", ""])
    report.append(f"Best validation candidate: {best['candidate']} with val active-rate RMSE {best['active_rate_rmse']:.6f}.")
    if float(best["active_rate_rmse"]) < CURRENT_CPU_BEST_VAL_ACTIVE_RMSE - 1e-3 and float(best["link_rmse"]) <= 90.0:
        report.append("This passes the CPU validation gate for confirmation.")
    else:
        report.append("This does not pass the CPU validation gate for GPU.")
    (args.output_dir / "rb_total_value_head_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    result = {
        "framework": "PI-JWM",
        "candidate": "v11",
        "mode": "rb_total_value_head_cpu_probe",
        "output_dir": str(args.output_dir),
        "diagnostics": diagnostics,
        "rows": all_rows,
        "val_ranked": val_ranked,
        "selected_by_val": best,
        "test_rows": test_rows,
    }
    write_json(args.output_dir / "summary.json", result)
    return result


def main() -> None:
    args = parse_args()
    result = run_experiment(args)
    print(
        json.dumps(
            {
                "output_dir": result["output_dir"],
                "diagnostics": result["diagnostics"],
                "selected_by_val": result["selected_by_val"],
                "test_rows": result["test_rows"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
