'''CPU-first graph support generator for PI-JWM v11 candidate.

This probe tests a stronger base-policy replacement than template selection:
the new selector scores all edges at each sample-step, including edges that the
BC policy did not activate.  Selected edges receive a conservative RB value and
then a hard step-total projection keeps the action within a controlled budget.
'''

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRanker
except ImportError:  # pragma: no cover - exercised in environments without xgboost
    XGBRanker = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))

from pi_jwm.v11_rollout_value_calibrator import freeze_module

from diagnose_v11_counterfactual_value_attribution import collect_raw_actions
from diagnose_v11_rb_total_latent_identifiability import (
    build_models,
    collect_rollout_edge_context,
    invert_value_target,
    make_value_target,
    rows_from_context,
)
from diagnose_v11_rb_total_oracle_value_scope import safe_corr, select_topk_indices, write_json
from diagnose_v11_scheduler_ranked_allocation import predict_conservative_value_target, resolve_torch_device
from run_v11_rb_total_repair import active_rate_row, evaluate_raw_actions, make_adaptive_dataset, write_csv
from run_v11_rb_total_value_head import (
    RB_DIM,
    collect_edge_gradient_improvement,
    load_context_limited,
)


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / 'artifacts/experiments/pi_jwm_v11_graph_support_generator_20260627'
EPS = 1e-9


def limit_indices_window(
    indices: np.ndarray,
    limit: int | None,
    start: int = 0,
    shuffle_seed: int | None = None,
) -> np.ndarray:
    indices = np.asarray(indices)
    if shuffle_seed is not None and int(shuffle_seed) >= 0:
        rng = np.random.default_rng(int(shuffle_seed))
        indices = indices[rng.permutation(indices.shape[0])]
    if limit is None or int(limit) <= 0:
        return indices
    start = max(int(start), 0)
    if start >= indices.shape[0]:
        return indices[:0]
    return indices[start : start + int(limit)]


def parse_gate_threshold(source: str, prefix: str) -> float:
    if not str(source).startswith(prefix):
        raise ValueError(f'{source} does not start with {prefix}')
    text = str(source)[len(prefix):].replace('p', '.')
    return float(text)


def parse_selected_gate_source(source: str) -> tuple[str, float]:
    parts = str(source).split('_t')
    if len(parts) != 2 or not parts[0].startswith('selected_gate_'):
        raise ValueError(f'invalid selected gate source: {source}')
    quantile = parts[0].removeprefix('selected_gate_')
    threshold = float(parts[1].replace('p', '.'))
    return f'train_positive_{quantile}', threshold


def parse_step_budget_gated_source(source: str) -> tuple[str, float]:
    prefix = 'step_budget_gated_pred_'
    if not str(source).startswith(prefix) or '_t' not in str(source):
        raise ValueError(f'invalid gated step budget source: {source}')
    mode_text, threshold_text = str(source).removeprefix(prefix).rsplit('_t', 1)
    return mode_text, float(threshold_text.replace('p', '.'))


def parse_step_budget_filter_source(source: str) -> tuple[str, str]:
    prefix = 'step_budget_filter_'
    if not str(source).startswith(prefix):
        raise ValueError(f'invalid filtered step budget source: {source}')
    text = str(source).removeprefix(prefix)
    if text.startswith('pred_'):
        return 'pred', text.removeprefix('pred_')
    if text.startswith('true_'):
        return 'true', text.removeprefix('true_')
    raise ValueError(f'invalid filtered step budget source: {source}')


def parse_step_budget_step_gate_source(source: str) -> tuple[str, str, float]:
    prefix = 'step_budget_stepgate_'
    if not str(source).startswith(prefix) or '_t' not in str(source):
        raise ValueError(f'invalid step-gated step budget source: {source}')
    text, threshold_text = str(source).removeprefix(prefix).rsplit('_t', 1)
    if text.startswith('pred_'):
        return 'pred', text.removeprefix('pred_'), float(threshold_text.replace('p', '.'))
    if text.startswith('true_'):
        return 'true', text.removeprefix('true_'), float(threshold_text.replace('p', '.'))
    raise ValueError(f'invalid step-gated step budget source: {source}')


def parse_step_budget_reconstruct_gate_source(source: str) -> tuple[str, str, float]:
    prefix = 'step_budget_reconstruct_gate_'
    if not str(source).startswith(prefix) or '_t' not in str(source):
        raise ValueError(f'invalid reconstruct-gated step budget source: {source}')
    text, threshold_text = str(source).removeprefix(prefix).rsplit('_t', 1)
    if text.startswith('pred_'):
        return 'pred', text.removeprefix('pred_'), float(threshold_text.replace('p', '.'))
    if text.startswith('true_'):
        return 'true', text.removeprefix('true_'), float(threshold_text.replace('p', '.'))
    raise ValueError(f'invalid reconstruct-gated step budget source: {source}')


def parse_step_budget_soft_source(source: str) -> tuple[str, str, float]:
    prefix = 'step_budget_soft_'
    if not str(source).startswith(prefix) or '_b' not in str(source):
        raise ValueError(f'invalid soft step budget source: {source}')
    text, beta_text = str(source).removeprefix(prefix).rsplit('_b', 1)
    if text.startswith('pred_'):
        return 'pred', text.removeprefix('pred_'), float(beta_text.replace('p', '.'))
    if text.startswith('true_'):
        return 'true', text.removeprefix('true_'), float(beta_text.replace('p', '.'))
    raise ValueError(f'invalid soft step budget source: {source}')


def parse_mass_preserve_source(source: str) -> tuple[str, str, float]:
    prefix = 'mass_preserve_'
    if not str(source).startswith(prefix) or '_b' not in str(source):
        raise ValueError(f'invalid mass-preserving source: {source}')
    text, beta_text = str(source).removeprefix(prefix).rsplit('_b', 1)
    if text.startswith('pred_'):
        return 'pred', text.removeprefix('pred_'), float(beta_text.replace('p', '.'))
    if text.startswith('true_'):
        return 'true', text.removeprefix('true_'), float(beta_text.replace('p', '.'))
    raise ValueError(f'invalid mass-preserving source: {source}')


def parse_active_value_new_fixed_mode(group_mode: str) -> tuple[int, int] | None:
    prefix = 'baseline_active_value_top'
    middle = '_plus_new'
    text = str(group_mode)
    if not text.startswith(prefix) or middle not in text:
        return None
    active_text, new_text = text.removeprefix(prefix).split(middle, 1)
    try:
        active_k = int(active_text)
        new_k = int(new_text)
    except ValueError:
        return None
    if active_k < 0 or new_k < 0:
        raise ValueError(f'active/new counts must be non-negative in group mode: {group_mode}')
    return active_k, new_k


def normalized_group_rank(coordinates: np.ndarray, score: np.ndarray) -> np.ndarray:
    coords = np.asarray(coordinates, dtype=np.int64).reshape(-1, 3)
    score = np.asarray(score, dtype=np.float32).reshape(-1)
    if coords.shape[0] != score.shape[0]:
        raise ValueError('coordinates and score must have the same row count')
    ranks = np.zeros((coords.shape[0],), dtype=np.float32)
    groups: dict[tuple[int, int], list[int]] = {}
    for row_idx, (sample, step, _edge) in enumerate(coords):
        groups.setdefault((int(sample), int(step)), []).append(int(row_idx))
    for group_rows in groups.values():
        rows = np.asarray(group_rows, dtype=np.int64)
        order = rows[np.argsort(-score[rows], kind='mergesort')]
        denom = max(float(rows.size - 1), 1.0)
        for rank_idx, row in enumerate(order):
            ranks[int(row)] = 1.0 - float(rank_idx) / denom
    return ranks.astype(np.float32)


def make_candidate_value_features(
    edge_features: np.ndarray,
    coordinates: np.ndarray,
    support_score: np.ndarray,
    selection_score: np.ndarray,
    baseline_value: np.ndarray,
) -> np.ndarray:
    edge_features = np.asarray(edge_features, dtype=np.float32)
    if edge_features.ndim != 2:
        raise ValueError('edge_features must be 2D')
    support_score = np.asarray(support_score, dtype=np.float32).reshape(-1)
    selection_score = np.asarray(selection_score, dtype=np.float32).reshape(-1)
    baseline_value = np.asarray(baseline_value, dtype=np.float32).reshape(-1)
    if not (edge_features.shape[0] == support_score.shape[0] == selection_score.shape[0] == baseline_value.shape[0]):
        raise ValueError('candidate value feature inputs must have the same row count')
    rank = normalized_group_rank(coordinates, selection_score)
    extra = np.stack(
        [
            support_score,
            selection_score,
            np.clip(baseline_value, 0.0, None),
            np.log1p(np.clip(baseline_value, 0.0, None)),
            rank,
        ],
        axis=1,
    ).astype(np.float32)
    return np.concatenate([edge_features, extra], axis=1).astype(np.float32)


def make_allocation_filter_features(
    edge_features: np.ndarray,
    coordinates: np.ndarray,
    support_score: np.ndarray,
    selection_score: np.ndarray,
    baseline_value: np.ndarray,
    pred_value: np.ndarray,
    total_by_row: np.ndarray,
    count_by_row: np.ndarray,
) -> np.ndarray:
    edge_features = np.asarray(edge_features, dtype=np.float32)
    coords = np.asarray(coordinates, dtype=np.int64).reshape(-1, 3)
    support_score = np.asarray(support_score, dtype=np.float32).reshape(-1)
    selection_score = np.asarray(selection_score, dtype=np.float32).reshape(-1)
    baseline_value = np.asarray(baseline_value, dtype=np.float32).reshape(-1)
    pred_value = np.asarray(pred_value, dtype=np.float32).reshape(-1)
    total_by_row = np.asarray(total_by_row, dtype=np.float32).reshape(-1)
    count_by_row = np.asarray(count_by_row, dtype=np.float32).reshape(-1)
    if edge_features.ndim != 2:
        raise ValueError('edge_features must be 2D')
    if not (
        edge_features.shape[0]
        == coords.shape[0]
        == support_score.shape[0]
        == selection_score.shape[0]
        == baseline_value.shape[0]
        == pred_value.shape[0]
        == total_by_row.shape[0]
        == count_by_row.shape[0]
    ):
        raise ValueError('allocation filter feature inputs must have the same row count')
    rank = normalized_group_rank(coords, selection_score)
    baseline = np.clip(baseline_value, 0.0, None)
    pred = np.clip(pred_value, 0.0, None)
    total = np.clip(total_by_row, 0.0, None)
    count = np.clip(count_by_row, 0.0, None)
    extra = np.stack(
        [
            support_score,
            selection_score,
            baseline,
            np.log1p(baseline),
            pred,
            np.log1p(pred),
            total,
            np.log1p(total),
            count,
            rank,
            baseline / np.clip(total, 1.0, None),
            pred / np.clip(total, 1.0, None),
        ],
        axis=1,
    ).astype(np.float32)
    return np.concatenate([edge_features, extra], axis=1).astype(np.float32)


def make_step_budget_features(
    edge_features: np.ndarray,
    coordinates: np.ndarray,
    support_score: np.ndarray,
    selection_score: np.ndarray,
    baseline_value: np.ndarray,
    pred_value: np.ndarray,
) -> StepBudgetFeatures:
    edge_features = np.asarray(edge_features, dtype=np.float32)
    coords = np.asarray(coordinates, dtype=np.int64).reshape(-1, 3)
    support_score = np.asarray(support_score, dtype=np.float32).reshape(-1)
    selection_score = np.asarray(selection_score, dtype=np.float32).reshape(-1)
    baseline_value = np.asarray(baseline_value, dtype=np.float32).reshape(-1)
    pred_value = np.asarray(pred_value, dtype=np.float32).reshape(-1)
    if not (edge_features.shape[0] == coords.shape[0] == support_score.shape[0] == selection_score.shape[0] == baseline_value.shape[0] == pred_value.shape[0]):
        raise ValueError('step budget feature inputs must have the same row count')
    if coords.shape[0] == 0:
        return StepBudgetFeatures(
            keys=np.zeros((0, 2), dtype=np.int64),
            features=np.zeros((0, edge_features.shape[1] * 2 + 12), dtype=np.float32),
            row_to_step=np.zeros((0,), dtype=np.int64),
        )
    keys, inverse = np.unique(coords[:, :2], axis=0, return_inverse=True)
    rows = []
    for step_idx in range(keys.shape[0]):
        mask = inverse == step_idx
        ef = edge_features[mask]
        support = support_score[mask]
        selection = selection_score[mask]
        baseline = np.clip(baseline_value[mask], 0.0, None)
        pred = np.clip(pred_value[mask], 0.0, None)
        active_baseline = baseline > EPS
        top_support = np.sort(support)[-min(8, support.shape[0]):] if support.size else np.zeros((0,), dtype=np.float32)
        top_selection = np.sort(selection)[-min(8, selection.shape[0]):] if selection.size else np.zeros((0,), dtype=np.float32)
        rows.append(
            np.concatenate(
                [
                    np.array(
                        [
                            float(keys[step_idx, 1]),
                            float(np.sum(baseline)),
                            float(np.sum(active_baseline)),
                            float(np.sum(pred)),
                            float(np.mean(support)) if support.size else 0.0,
                            float(np.max(support)) if support.size else 0.0,
                            float(np.mean(selection)) if selection.size else 0.0,
                            float(np.max(selection)) if selection.size else 0.0,
                            float(np.sum(top_support)),
                            float(np.sum(top_selection)),
                            float(np.log1p(np.sum(baseline))),
                            float(np.log1p(np.sum(pred))),
                        ],
                        dtype=np.float32,
                    ),
                    ef.mean(axis=0),
                    ef.max(axis=0),
                ]
            ).astype(np.float32)
        )
    return StepBudgetFeatures(
        keys=keys.astype(np.int64),
        features=np.asarray(rows, dtype=np.float32),
        row_to_step=inverse.astype(np.int64),
    )


def make_step_budget_targets(
    step_keys: np.ndarray,
    coordinates: np.ndarray,
    true_value: np.ndarray,
    min_effective_value: float,
    baseline_value: np.ndarray | None = None,
) -> StepBudgetTargets:
    keys = np.asarray(step_keys, dtype=np.int64).reshape(-1, 2)
    coords = np.asarray(coordinates, dtype=np.int64).reshape(-1, 3)
    true_value = np.asarray(true_value, dtype=np.float32).reshape(-1)
    if baseline_value is None:
        baseline_value = np.zeros((coords.shape[0],), dtype=np.float32)
    baseline_value = np.asarray(baseline_value, dtype=np.float32).reshape(-1)
    if coords.shape[0] != true_value.shape[0]:
        raise ValueError('coordinates and true_value must have the same row count')
    if baseline_value.shape[0] != coords.shape[0]:
        raise ValueError('baseline_value must have the same row count as coordinates')
    totals = np.zeros((keys.shape[0],), dtype=np.float32)
    counts = np.zeros((keys.shape[0],), dtype=np.float32)
    new_counts = np.zeros((keys.shape[0],), dtype=np.float32)
    key_to_idx = {(int(sample), int(step)): idx for idx, (sample, step) in enumerate(keys)}
    for row_idx, (sample, step, _edge) in enumerate(coords):
        idx = key_to_idx.get((int(sample), int(step)))
        if idx is None:
            continue
        value = max(float(true_value[row_idx]), 0.0)
        if value >= float(min_effective_value):
            totals[idx] += value
            counts[idx] += 1.0
            if float(baseline_value[row_idx]) <= EPS:
                new_counts[idx] += 1.0
    return StepBudgetTargets(total=totals.astype(np.float32), count=counts.astype(np.float32), new_count=new_counts.astype(np.float32))


@dataclass(frozen=True)
class AllEdgeExamples:
    coordinates: np.ndarray
    baseline_values: np.ndarray
    true_values: np.ndarray


@dataclass(frozen=True)
class CandidateValueBinModel:
    centers: np.ndarray
    classifier: object


@dataclass(frozen=True)
class StepBudgetFeatures:
    keys: np.ndarray
    features: np.ndarray
    row_to_step: np.ndarray


@dataclass(frozen=True)
class StepBudgetTargets:
    total: np.ndarray
    count: np.ndarray
    new_count: np.ndarray


def make_all_edge_coordinates(actions: np.ndarray, steps: tuple[int, ...] = (1, 2)) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    coords = []
    for sample in range(actions.shape[0]):
        for step in steps:
            if 0 <= int(step) < actions.shape[1]:
                for edge in range(actions.shape[2]):
                    coords.append((int(sample), int(step), int(edge)))
    return np.asarray(coords, dtype=np.int64).reshape(-1, 3)


def make_all_edge_examples(baseline: np.ndarray, truth: np.ndarray, steps: tuple[int, ...]) -> AllEdgeExamples:
    baseline = np.asarray(baseline, dtype=np.float32)
    truth = np.asarray(truth, dtype=np.float32)
    coords = make_all_edge_coordinates(baseline, steps=steps)
    if coords.shape[0] == 0:
        return AllEdgeExamples(
            coordinates=coords,
            baseline_values=np.zeros((0,), dtype=np.float32),
            true_values=np.zeros((0,), dtype=np.float32),
        )
    return AllEdgeExamples(
        coordinates=coords,
        baseline_values=baseline[coords[:, 0], coords[:, 1], coords[:, 2], RB_DIM].astype(np.float32),
        true_values=truth[coords[:, 0], coords[:, 1], coords[:, 2], RB_DIM].astype(np.float32),
    )


def make_all_edge_targets(
    examples: AllEdgeExamples,
    truth_actions: np.ndarray,
    edge_improvement: np.ndarray,
    min_effective_value: float,
    min_improvement: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coords = np.asarray(examples.coordinates, dtype=np.int64).reshape(-1, 3)
    truth_actions = np.asarray(truth_actions, dtype=np.float32)
    edge_improvement = np.asarray(edge_improvement, dtype=np.float32)
    scores = np.zeros((coords.shape[0],), dtype=np.float32)
    values = np.zeros((coords.shape[0],), dtype=np.float32)
    labels = np.zeros((coords.shape[0],), dtype=np.int64)
    for row_idx, (sample, step, edge) in enumerate(coords):
        improvement = max(float(edge_improvement[sample, step, edge]), 0.0)
        value = max(float(truth_actions[sample, step, edge, RB_DIM]), 0.0)
        scores[row_idx] = np.log1p(improvement)
        values[row_idx] = value
        if value >= float(min_effective_value) and improvement > float(min_improvement):
            labels[row_idx] = 1
    return labels, scores, values


def downsample_support_training_rows(
    labels: np.ndarray,
    scores: np.ndarray,
    max_rows: int,
    negative_ratio: float,
    seed: int,
) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    positive = np.flatnonzero(labels > 0)
    negative = np.flatnonzero(labels <= 0)
    if int(max_rows) <= 0 or labels.shape[0] <= int(max_rows):
        return np.arange(labels.shape[0], dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    max_negative = max(int(len(positive) * float(negative_ratio)), int(max_rows) - len(positive))
    max_negative = max(0, min(max_negative, int(max_rows) - min(len(positive), int(max_rows))))
    if positive.shape[0] > int(max_rows):
        weights = scores[positive] + 1e-6
        weights = weights / float(np.sum(weights))
        chosen_positive = rng.choice(positive, size=int(max_rows), replace=False, p=weights)
        chosen_negative = np.zeros((0,), dtype=np.int64)
    else:
        chosen_positive = positive
        if negative.shape[0] > max_negative:
            chosen_negative = rng.choice(negative, size=max_negative, replace=False)
        else:
            chosen_negative = negative
    rows = np.concatenate([chosen_positive, chosen_negative]).astype(np.int64)
    rng.shuffle(rows)
    return rows


def downsample_hard_negative_support_training_rows(
    labels: np.ndarray,
    scores: np.ndarray,
    warm_scores: np.ndarray,
    max_rows: int,
    hard_fraction: float,
    seed: int,
) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    warm_scores = np.asarray(warm_scores, dtype=np.float32).reshape(-1)
    if int(max_rows) <= 0 or labels.shape[0] <= int(max_rows):
        return np.arange(labels.shape[0], dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    positive = np.flatnonzero(labels > 0)
    negative = np.flatnonzero(labels <= 0)
    if positive.shape[0] >= int(max_rows):
        weights = scores[positive] + 1e-6
        weights = weights / float(np.sum(weights))
        rows = rng.choice(positive, size=int(max_rows), replace=False, p=weights)
        rng.shuffle(rows)
        return rows.astype(np.int64)

    negative_slots = int(max_rows) - int(positive.shape[0])
    hard_count = int(round(float(negative_slots) * float(hard_fraction)))
    hard_count = max(0, min(hard_count, negative.shape[0], negative_slots))
    if hard_count > 0:
        hard_order = np.argsort(-warm_scores[negative], kind='mergesort')[:hard_count]
        chosen_hard = negative[hard_order]
    else:
        chosen_hard = np.zeros((0,), dtype=np.int64)

    remaining = np.setdiff1d(negative, chosen_hard, assume_unique=False)
    random_count = max(0, min(negative_slots - int(chosen_hard.shape[0]), remaining.shape[0]))
    if random_count > 0:
        chosen_random = rng.choice(remaining, size=random_count, replace=False)
    else:
        chosen_random = np.zeros((0,), dtype=np.int64)
    rows = np.concatenate([positive, chosen_hard, chosen_random]).astype(np.int64)
    rng.shuffle(rows)
    return rows


def make_group_rank_targets(
    coordinates: np.ndarray,
    oracle_scores: np.ndarray,
    true_values: np.ndarray,
    mode: str = 'gain_norm',
) -> np.ndarray:
    coords = np.asarray(coordinates, dtype=np.int64).reshape(-1, 3)
    oracle_scores = np.asarray(oracle_scores, dtype=np.float32).reshape(-1)
    true_values = np.asarray(true_values, dtype=np.float32).reshape(-1)
    targets = np.zeros((coords.shape[0],), dtype=np.float32)
    groups: dict[tuple[int, int], list[int]] = {}
    for row_idx, (sample, step, _edge) in enumerate(coords):
        groups.setdefault((int(sample), int(step)), []).append(int(row_idx))

    for group_rows in groups.values():
        rows = np.asarray(group_rows, dtype=np.int64)
        gain = np.clip(oracle_scores[rows], 0.0, None)
        if str(mode) == 'gain':
            targets[rows] = gain
        elif str(mode) == 'gain_norm':
            denom = float(np.max(gain)) if gain.size else 0.0
            if denom > EPS:
                targets[rows] = gain / denom
        elif str(mode) == 'value_gain_norm':
            mixed = gain * np.log1p(np.clip(true_values[rows], 0.0, None))
            denom = float(np.max(mixed)) if mixed.size else 0.0
            if denom > EPS:
                targets[rows] = mixed / denom
        elif str(mode) == 'true_value_norm':
            value = np.clip(true_values[rows], 0.0, None)
            denom = float(np.max(value)) if value.size else 0.0
            if denom > EPS:
                targets[rows] = value / denom
        else:
            raise ValueError(f'unknown rank target mode: {mode}')
    return targets.astype(np.float32)


def make_pairwise_preference_examples(
    features: np.ndarray,
    coordinates: np.ndarray,
    rank_targets: np.ndarray,
    max_pairs: int,
    negatives_per_positive: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    features = np.asarray(features, dtype=np.float32)
    coords = np.asarray(coordinates, dtype=np.int64).reshape(-1, 3)
    rank_targets = np.asarray(rank_targets, dtype=np.float32).reshape(-1)
    rng = np.random.default_rng(int(seed))
    groups: dict[tuple[int, int], list[int]] = {}
    for row_idx, (sample, step, _edge) in enumerate(coords):
        groups.setdefault((int(sample), int(step)), []).append(int(row_idx))

    diffs: list[np.ndarray] = []
    labels: list[int] = []
    for group_rows in groups.values():
        rows = np.asarray(group_rows, dtype=np.int64)
        positive = rows[rank_targets[rows] > 0.0]
        negative = rows[rank_targets[rows] <= 0.0]
        if positive.size == 0 or negative.size == 0:
            continue
        neg_order = negative[np.argsort(-rank_targets[negative], kind='mergesort')]
        for pos in positive:
            if len(labels) >= int(max_pairs) * 2:
                break
            if neg_order.size > int(negatives_per_positive):
                chosen_negative = rng.choice(neg_order, size=int(negatives_per_positive), replace=False)
            else:
                chosen_negative = neg_order
            for neg in chosen_negative:
                diff = features[int(pos)] - features[int(neg)]
                diffs.append(diff)
                labels.append(1)
                diffs.append(-diff)
                labels.append(0)
        if len(labels) >= int(max_pairs) * 2:
            break
    if not diffs:
        return np.zeros((0, features.shape[1]), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.asarray(diffs, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def order_rows_by_group(coordinates: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coords = np.asarray(coordinates, dtype=np.int64).reshape(-1, 3)
    rows = np.asarray(rows, dtype=np.int64).reshape(-1)
    if rows.size == 0:
        return rows.astype(np.int64), np.zeros((0,), dtype=np.int64)

    group_map: dict[tuple[int, int], list[int]] = {}
    for row in rows:
        sample, step, _edge = coords[int(row)]
        group_map.setdefault((int(sample), int(step)), []).append(int(row))

    ordered_rows: list[int] = []
    group_sizes: list[int] = []
    for key in sorted(group_map.keys()):
        group_rows = sorted(group_map[key], key=lambda item: int(coords[int(item), 2]))
        ordered_rows.extend(group_rows)
        group_sizes.append(len(group_rows))
    return np.asarray(ordered_rows, dtype=np.int64), np.asarray(group_sizes, dtype=np.int64)


def fit_xgb_ranker(
    features: np.ndarray,
    coordinates: np.ndarray,
    targets: np.ndarray,
    rows: np.ndarray,
    seed: int,
    trees: int,
):
    if XGBRanker is None:
        raise RuntimeError('xgboost is required for support model kind xgb_rank')
    ordered_rows, group_sizes = order_rows_by_group(coordinates, rows)
    if ordered_rows.size == 0 or group_sizes.size == 0:
        return ConstantSupportClassifier(0.0)
    model = XGBRanker(
        objective='rank:pairwise',
        n_estimators=max(4, int(trees)),
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=5.0,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        random_state=int(seed),
        n_jobs=-1,
        tree_method='hist',
    )
    model.fit(
        np.asarray(features, dtype=np.float32)[ordered_rows],
        np.asarray(targets, dtype=np.float32).reshape(-1)[ordered_rows],
        group=group_sizes.astype(np.uint32),
        verbose=False,
    )
    return model


def select_support_indices(
    coordinates: np.ndarray,
    score: np.ndarray,
    top_k: int,
    group_mode: str,
    baseline_value: np.ndarray | None = None,
    threshold: float = 0.0,
) -> np.ndarray:
    coords = np.asarray(coordinates, dtype=np.int64).reshape(-1, 3)
    score = np.asarray(score, dtype=np.float32).reshape(-1)
    if str(group_mode) == 'fixed':
        return select_topk_indices(coords, score, int(top_k), 'per_sample_step')

    baseline_value_arr = None if baseline_value is None else np.asarray(baseline_value, dtype=np.float32).reshape(-1)
    groups: dict[tuple[int, int], list[int]] = {}
    for row_idx, (sample, step, _edge) in enumerate(coords):
        groups.setdefault((int(sample), int(step)), []).append(int(row_idx))

    selected: list[int] = []
    for group_rows in groups.values():
        rows = np.asarray(group_rows, dtype=np.int64)
        if rows.size == 0:
            continue
        if str(group_mode) == 'baseline_active_count':
            if baseline_value_arr is None:
                raise ValueError('baseline_active_count requires baseline_value')
            k = int(np.sum(baseline_value_arr[rows] > EPS))
            k = max(1, min(int(top_k), k))
            chosen = rows[np.argsort(-score[rows], kind='mergesort')[:k]]
        elif str(group_mode) == 'baseline_active_plus_topk':
            if baseline_value_arr is None:
                raise ValueError('baseline_active_plus_topk requires baseline_value')
            active_rows = rows[baseline_value_arr[rows] > EPS]
            top_rows = rows[np.argsort(-score[rows], kind='mergesort')[: int(top_k)]]
            chosen = np.union1d(active_rows, top_rows).astype(np.int64)
        elif str(group_mode) == 'baseline_active_plus_new_topk':
            if baseline_value_arr is None:
                raise ValueError('baseline_active_plus_new_topk requires baseline_value')
            active_rows = rows[baseline_value_arr[rows] > EPS]
            inactive_rows = rows[baseline_value_arr[rows] <= EPS]
            top_new_rows = inactive_rows[np.argsort(-score[inactive_rows], kind='mergesort')[: int(top_k)]]
            chosen = np.union1d(active_rows, top_new_rows).astype(np.int64)
        elif str(group_mode) == 'baseline_active_plus_new_topk_threshold':
            if baseline_value_arr is None:
                raise ValueError('baseline_active_plus_new_topk_threshold requires baseline_value')
            active_rows = rows[baseline_value_arr[rows] > EPS]
            inactive_rows = rows[baseline_value_arr[rows] <= EPS]
            top_new_rows = inactive_rows[np.argsort(-score[inactive_rows], kind='mergesort')[: int(top_k)]]
            top_new_rows = top_new_rows[score[top_new_rows] >= float(threshold)]
            chosen = np.union1d(active_rows, top_new_rows).astype(np.int64)
        elif str(group_mode) == 'baseline_active_topk_plus_new_topk':
            if baseline_value_arr is None:
                raise ValueError('baseline_active_topk_plus_new_topk requires baseline_value')
            active_rows = rows[baseline_value_arr[rows] > EPS]
            inactive_rows = rows[baseline_value_arr[rows] <= EPS]
            top_active_rows = active_rows[np.argsort(-score[active_rows], kind='mergesort')[: int(top_k)]]
            top_new_rows = inactive_rows[np.argsort(-score[inactive_rows], kind='mergesort')[: int(top_k)]]
            chosen = np.union1d(top_active_rows, top_new_rows).astype(np.int64)
        elif str(group_mode) == 'baseline_active_value_topk_plus_new_topk':
            if baseline_value_arr is None:
                raise ValueError('baseline_active_value_topk_plus_new_topk requires baseline_value')
            active_rows = rows[baseline_value_arr[rows] > EPS]
            inactive_rows = rows[baseline_value_arr[rows] <= EPS]
            top_active_rows = active_rows[np.argsort(-baseline_value_arr[active_rows], kind='mergesort')[: int(top_k)]]
            top_new_rows = inactive_rows[np.argsort(-score[inactive_rows], kind='mergesort')[: int(top_k)]]
            chosen = np.union1d(top_active_rows, top_new_rows).astype(np.int64)
        elif parse_active_value_new_fixed_mode(str(group_mode)) is not None:
            if baseline_value_arr is None:
                raise ValueError(f'{group_mode} requires baseline_value')
            active_k, new_k = parse_active_value_new_fixed_mode(str(group_mode))
            active_rows = rows[baseline_value_arr[rows] > EPS]
            inactive_rows = rows[baseline_value_arr[rows] <= EPS]
            top_active_rows = active_rows[np.argsort(-baseline_value_arr[active_rows], kind='mergesort')[: int(active_k)]]
            top_new_rows = inactive_rows[np.argsort(-score[inactive_rows], kind='mergesort')[: int(new_k)]]
            chosen = np.union1d(top_active_rows, top_new_rows).astype(np.int64)
        elif str(group_mode) == 'support_threshold':
            top_rows = rows[np.argsort(-score[rows], kind='mergesort')[: int(top_k)]]
            chosen = top_rows[score[top_rows] >= float(threshold)]
        else:
            raise ValueError(f'unknown support group mode: {group_mode}')
        selected.extend(int(row) for row in chosen)
    return np.asarray(selected, dtype=np.int64)


def oracle_mass_at_selected_budget(
    coordinates: np.ndarray,
    oracle_score: np.ndarray,
    selected: np.ndarray,
) -> float:
    coords = np.asarray(coordinates, dtype=np.int64).reshape(-1, 3)
    oracle_score = np.asarray(oracle_score, dtype=np.float32).reshape(-1)
    selected = np.asarray(selected, dtype=np.int64).reshape(-1)
    if coords.shape[0] != oracle_score.shape[0]:
        raise ValueError('coordinates and oracle_score must have the same row count')
    if selected.size == 0:
        return 0.0

    selected_budget_by_group: dict[tuple[int, int], int] = {}
    for row_idx in selected:
        sample, step, _edge = coords[int(row_idx)]
        key = (int(sample), int(step))
        selected_budget_by_group[key] = selected_budget_by_group.get(key, 0) + 1

    group_rows_by_key: dict[tuple[int, int], list[int]] = {}
    for row_idx, (sample, step, _edge) in enumerate(coords):
        group_rows_by_key.setdefault((int(sample), int(step)), []).append(int(row_idx))

    total = 0.0
    for key, budget in selected_budget_by_group.items():
        rows = np.asarray(group_rows_by_key.get(key, []), dtype=np.int64)
        if rows.size == 0 or int(budget) <= 0:
            continue
        top = rows[np.argsort(-oracle_score[rows], kind='mergesort')[: int(budget)]]
        total += float(np.sum(np.clip(oracle_score[top], 0.0, None)))
    return float(total)


def support_selection_diagnostics(
    coordinates: np.ndarray,
    oracle_score: np.ndarray,
    labels: np.ndarray,
    selected: np.ndarray,
) -> dict[str, float | int]:
    oracle_score = np.asarray(oracle_score, dtype=np.float32).reshape(-1)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    selected = np.asarray(selected, dtype=np.int64).reshape(-1)
    if oracle_score.shape[0] != labels.shape[0]:
        raise ValueError('oracle_score and labels must have the same row count')

    selected_oracle_mass = float(np.sum(np.clip(oracle_score[selected], 0.0, None))) if selected.size else 0.0
    oracle_budget_mass = oracle_mass_at_selected_budget(coordinates, oracle_score, selected)
    total_positive = int(np.sum(labels > 0))
    selected_positive = int(np.sum(labels[selected] > 0)) if selected.size else 0
    precision = float(selected_positive / selected.size) if selected.size else 0.0
    recall = float(selected_positive / total_positive) if total_positive else 0.0
    ratio = float(selected_oracle_mass / oracle_budget_mass) if oracle_budget_mass > EPS else 0.0
    return {
        'selected_count': int(selected.size),
        'selected_oracle_mass': selected_oracle_mass,
        'oracle_budget_mass': float(oracle_budget_mass),
        'oracle_mass_ratio_at_budget': ratio,
        'selected_positive_count': selected_positive,
        'total_positive_count': total_positive,
        'selected_positive_precision': precision,
        'selected_positive_recall': recall,
    }


def repair_values_for_source(
    pred_value: np.ndarray,
    true_value: np.ndarray,
    source: str,
    value_lookup: dict[str, float] | None = None,
    support_score: np.ndarray | None = None,
) -> np.ndarray:
    pred_value = np.asarray(pred_value, dtype=np.float32).reshape(-1)
    true_value = np.asarray(true_value, dtype=np.float32).reshape(-1)
    if pred_value.shape != true_value.shape:
        raise ValueError('pred_value and true_value must share shape')
    if source in {'pred_value', 'pred_value_reconstruct'}:
        return pred_value.astype(np.float32)
    if source == 'true_value':
        return true_value.astype(np.float32)
    if str(source).startswith('train_positive_'):
        lookup = value_lookup or {}
        if source not in lookup:
            raise ValueError(f'missing value lookup for repair source: {source}')
        return np.full(pred_value.shape, float(lookup[source]), dtype=np.float32)
    if str(source).startswith('support_score_q'):
        quantile_source = 'train_positive_' + str(source).removeprefix('support_score_')
        lookup = value_lookup or {}
        if quantile_source not in lookup:
            raise ValueError(f'missing value lookup for repair source: {quantile_source}')
        if support_score is None:
            raise ValueError(f'{source} requires support_score')
        support_score = np.asarray(support_score, dtype=np.float32).reshape(-1)
        if support_score.shape != pred_value.shape:
            raise ValueError('support_score and pred_value must share shape')
        return (np.clip(support_score, 0.0, 1.0) * float(lookup[quantile_source])).astype(np.float32)
    if str(source).startswith('pred_value_gate_'):
        if support_score is None:
            raise ValueError(f'{source} requires support_score')
        support_score = np.asarray(support_score, dtype=np.float32).reshape(-1)
        if support_score.shape != pred_value.shape:
            raise ValueError('support_score and pred_value must share shape')
        threshold = parse_gate_threshold(str(source), 'pred_value_gate_')
        return np.where(support_score >= threshold, pred_value, 0.0).astype(np.float32)
    raise ValueError(f'unknown repair value source: {source}')


def candidate_family_for_repair_value_source(source: str) -> str:
    if source in {'pred_value', 'pred_value_reconstruct'}:
        return 'graph_support_generator'
    if str(source).startswith('train_positive_'):
        return 'graph_support_generator'
    if str(source).startswith('support_score_q'):
        return 'graph_support_generator'
    if str(source).startswith('pred_value_gate_'):
        return 'graph_support_generator'
    if source == 'candidate_value':
        return 'graph_support_generator'
    if source == 'positive_candidate_value':
        return 'graph_support_generator'
    if str(source).startswith('candidate_bin_'):
        return 'graph_support_generator'
    if str(source).startswith('positive_candidate_bin_'):
        return 'graph_support_generator'
    if str(source).startswith('all_candidate_bin_'):
        return 'graph_support_generator'
    if str(source).startswith('selected_gate_'):
        return 'graph_support_generator'
    if source == 'structured_stepwise_pred_reconstruct':
        return 'graph_support_generator'
    if source == 'structured_branch_value_pred_reconstruct':
        return 'graph_support_generator'
    if str(source).startswith('step_budget_pred_'):
        return 'graph_support_generator'
    if str(source).startswith('step_budget_gated_pred_'):
        return 'graph_support_generator'
    if str(source).startswith('step_budget_stepgate_pred_'):
        return 'graph_support_generator'
    if str(source).startswith('step_budget_reconstruct_gate_pred_'):
        return 'graph_support_generator'
    if str(source).startswith('step_budget_soft_pred_'):
        return 'graph_support_generator'
    if str(source).startswith('mass_preserve_pred_'):
        return 'graph_support_generator'
    if str(source).startswith('step_budget_filter_pred_'):
        return 'graph_support_generator'
    if str(source).startswith('step_budget_ranker_pred_'):
        return 'graph_support_generator'
    if str(source).startswith('step_budget_newq_pred_'):
        return 'graph_support_generator'
    if str(source).startswith('step_budget_oracle_alloc_pred_'):
        return 'diagnostic_only'
    if str(source).startswith('step_budget_true_'):
        return 'diagnostic_only'
    if str(source).startswith('step_budget_stepgate_true_'):
        return 'diagnostic_only'
    if str(source).startswith('step_budget_reconstruct_gate_true_'):
        return 'diagnostic_only'
    if str(source).startswith('step_budget_soft_true_'):
        return 'diagnostic_only'
    if str(source).startswith('mass_preserve_true_'):
        return 'diagnostic_only'
    if str(source).startswith('step_budget_filter_true_'):
        return 'diagnostic_only'
    if str(source).startswith('step_budget_ranker_true_'):
        return 'diagnostic_only'
    if str(source).startswith('step_budget_newq_true_'):
        return 'diagnostic_only'
    if source == 'true_value':
        return 'diagnostic_only'
    raise ValueError(f'unknown repair value source: {source}')


def candidate_family_for_score_and_value_source(score_name: str, score_mode: str, repair_value_source: str) -> str:
    if str(score_name).startswith('diagnostic_only'):
        return 'diagnostic_only'
    if str(score_mode) == 'oracle_support':
        return 'diagnostic_only'
    return candidate_family_for_repair_value_source(str(repair_value_source))


def apply_support_generator_repair(
    actions: np.ndarray,
    coordinates: np.ndarray,
    values: np.ndarray,
    selected: np.ndarray,
    alpha: float,
    step_total_cap_scale: float,
    edge_value_cap_scale: float,
    new_edge_value_cap: float,
) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    coords = np.asarray(coordinates, dtype=np.int64).reshape(-1, 3)
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    selected = np.asarray(selected, dtype=np.int64).reshape(-1)
    repaired = actions.copy()
    alpha = float(alpha)
    for row_idx in selected:
        sample, step, edge = coords[int(row_idx)]
        if int(step) == 0:
            continue
        baseline = float(actions[sample, step, edge, RB_DIM])
        target = max(float(values[int(row_idx)]), 0.0)
        if baseline > EPS and float(edge_value_cap_scale) > 0.0:
            target = min(target, baseline * float(edge_value_cap_scale))
        elif baseline <= EPS and float(new_edge_value_cap) > 0.0:
            target = min(target, float(new_edge_value_cap))
        repaired[sample, step, edge, RB_DIM] = max((1.0 - alpha) * baseline + alpha * target, 0.0)

    if float(step_total_cap_scale) > 0.0:
        original_totals = np.sum(np.clip(actions[..., RB_DIM], 0.0, None), axis=2)
        repaired_totals = np.sum(np.clip(repaired[..., RB_DIM], 0.0, None), axis=2)
        caps = np.clip(original_totals * float(step_total_cap_scale), 0.0, None)
        for sample in range(repaired.shape[0]):
            for step in range(1, repaired.shape[1]):
                total = float(repaired_totals[sample, step])
                cap = float(caps[sample, step])
                if total > max(cap, EPS):
                    repaired[sample, step, :, RB_DIM] *= cap / total
    repaired[:, 0, :, RB_DIM] = actions[:, 0, :, RB_DIM]
    repaired[..., RB_DIM] = np.clip(repaired[..., RB_DIM], 0.0, None)
    return repaired.astype(np.float32)


def apply_support_generator_reconstruction(
    actions: np.ndarray,
    coordinates: np.ndarray,
    values: np.ndarray,
    selected: np.ndarray,
    alpha: float,
    step_total_cap_scale: float,
    edge_value_cap_scale: float,
    new_edge_value_cap: float,
) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    coords = np.asarray(coordinates, dtype=np.int64).reshape(-1, 3)
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    selected = np.asarray(selected, dtype=np.int64).reshape(-1)
    repaired = actions.copy()
    repaired[:, 1:, :, RB_DIM] = 0.0
    alpha = float(alpha)
    for row_idx in selected:
        sample, step, edge = coords[int(row_idx)]
        if int(step) == 0:
            continue
        baseline = float(actions[sample, step, edge, RB_DIM])
        target = max(float(values[int(row_idx)]), 0.0)
        if baseline > EPS and float(edge_value_cap_scale) > 0.0:
            target = min(target, baseline * float(edge_value_cap_scale))
        elif baseline <= EPS and float(new_edge_value_cap) > 0.0:
            target = min(target, float(new_edge_value_cap))
        repaired[sample, step, edge, RB_DIM] = max((1.0 - alpha) * baseline + alpha * target, 0.0)
    if float(step_total_cap_scale) > 0.0:
        original_totals = np.sum(np.clip(actions[..., RB_DIM], 0.0, None), axis=2)
        repaired_totals = np.sum(np.clip(repaired[..., RB_DIM], 0.0, None), axis=2)
        caps = np.clip(original_totals * float(step_total_cap_scale), 0.0, None)
        for sample in range(repaired.shape[0]):
            for step in range(1, repaired.shape[1]):
                total = float(repaired_totals[sample, step])
                cap = float(caps[sample, step])
                if total > max(cap, EPS):
                    repaired[sample, step, :, RB_DIM] *= cap / total
    repaired[:, 0, :, RB_DIM] = actions[:, 0, :, RB_DIM]
    repaired[..., RB_DIM] = np.clip(repaired[..., RB_DIM], 0.0, None)
    return repaired.astype(np.float32)


@dataclass(frozen=True)
class StructuredActionCandidate:
    name: str
    actions: np.ndarray
    step_scores: np.ndarray


def make_structured_step_score_matrix(
    actions: np.ndarray,
    coordinates: np.ndarray,
    selected: np.ndarray,
    selection_score: np.ndarray,
    pred_value: np.ndarray,
    baseline_value: np.ndarray,
    new_edge_penalty: float = 0.0,
    selected_count_penalty: float = 0.0,
) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    coords = np.asarray(coordinates, dtype=np.int64).reshape(-1, 3)
    selected = np.asarray(selected, dtype=np.int64).reshape(-1)
    selection_score = np.asarray(selection_score, dtype=np.float32).reshape(-1)
    pred_value = np.asarray(pred_value, dtype=np.float32).reshape(-1)
    baseline_value = np.asarray(baseline_value, dtype=np.float32).reshape(-1)
    if not (coords.shape[0] == selection_score.shape[0] == pred_value.shape[0] == baseline_value.shape[0]):
        raise ValueError('structured score inputs must have the same row count')
    scores = np.zeros((actions.shape[0], actions.shape[1]), dtype=np.float32)
    if selected.size == 0:
        return scores
    selected_counts = np.zeros_like(scores, dtype=np.float32)
    for row_idx in selected:
        sample, step, _edge = coords[int(row_idx)]
        if int(step) == 0:
            continue
        value_gain = max(float(pred_value[int(row_idx)]) - float(baseline_value[int(row_idx)]), 0.0)
        if float(baseline_value[int(row_idx)]) <= EPS:
            value_gain = max(value_gain - float(new_edge_penalty), 0.0)
        scores[int(sample), int(step)] += max(float(selection_score[int(row_idx)]), 0.0) * np.log1p(value_gain)
        selected_counts[int(sample), int(step)] += 1.0
    if float(selected_count_penalty) > 0.0:
        scores = scores - float(selected_count_penalty) * selected_counts
    return scores.astype(np.float32)


def compose_structured_stepwise_actions(
    baseline_actions: np.ndarray,
    candidates: list[StructuredActionCandidate],
) -> tuple[np.ndarray, dict[str, int]]:
    baseline_actions = np.asarray(baseline_actions, dtype=np.float32)
    composed = baseline_actions.copy()
    counts = {candidate.name: 0 for candidate in candidates}
    if not candidates:
        return composed.astype(np.float32), counts
    for candidate in candidates:
        candidate_actions = np.asarray(candidate.actions, dtype=np.float32)
        candidate_scores = np.asarray(candidate.step_scores, dtype=np.float32)
        if candidate_actions.shape != baseline_actions.shape:
            raise ValueError('structured candidate actions must match baseline action shape')
        if candidate_scores.shape != baseline_actions.shape[:2]:
            raise ValueError('structured candidate step_scores must match sample/step shape')
    stacked_scores = np.stack([np.asarray(candidate.step_scores, dtype=np.float32) for candidate in candidates], axis=0)
    best_indices = np.argmax(stacked_scores, axis=0)
    for sample in range(baseline_actions.shape[0]):
        for step in range(1, baseline_actions.shape[1]):
            candidate_idx = int(best_indices[sample, step])
            best_candidate = candidates[candidate_idx]
            if float(stacked_scores[candidate_idx, sample, step]) <= 0.0:
                continue
            composed[sample, step] = np.asarray(best_candidate.actions, dtype=np.float32)[sample, step]
            counts[best_candidate.name] = counts.get(best_candidate.name, 0) + 1
    composed[:, 0] = baseline_actions[:, 0]
    composed[..., RB_DIM] = np.clip(composed[..., RB_DIM], 0.0, None)
    return composed.astype(np.float32), counts


def _key_to_index(keys: np.ndarray) -> dict[tuple[int, int], int]:
    keys = np.asarray(keys, dtype=np.int64).reshape(-1, 2)
    return {(int(sample), int(step)): int(idx) for idx, (sample, step) in enumerate(keys)}


def make_action_step_keys(actions: np.ndarray) -> np.ndarray:
    actions = np.asarray(actions)
    if actions.ndim < 2:
        raise ValueError('actions must have at least sample and step dimensions')
    return np.asarray(
        [[sample, step] for sample in range(actions.shape[0]) for step in range(actions.shape[1])],
        dtype=np.int64,
    )


def make_branch_step_features(
    keys: np.ndarray,
    coordinates: np.ndarray,
    selected: np.ndarray,
    selection_score: np.ndarray,
    pred_value: np.ndarray,
    baseline_value: np.ndarray,
) -> np.ndarray:
    keys = np.asarray(keys, dtype=np.int64).reshape(-1, 2)
    coords = np.asarray(coordinates, dtype=np.int64).reshape(-1, 3)
    selected = np.asarray(selected, dtype=np.int64).reshape(-1)
    selection_score = np.asarray(selection_score, dtype=np.float32).reshape(-1)
    pred_value = np.asarray(pred_value, dtype=np.float32).reshape(-1)
    baseline_value = np.asarray(baseline_value, dtype=np.float32).reshape(-1)
    if not (coords.shape[0] == selection_score.shape[0] == pred_value.shape[0] == baseline_value.shape[0]):
        raise ValueError('branch feature inputs must have the same row count')
    features = np.zeros((keys.shape[0], 10), dtype=np.float32)
    key_index = _key_to_index(keys)
    selected_by_key: dict[tuple[int, int], list[int]] = {}
    for row_idx in selected:
        sample, step, _edge = coords[int(row_idx)]
        selected_by_key.setdefault((int(sample), int(step)), []).append(int(row_idx))
    for key, group_rows in selected_by_key.items():
        step_idx = key_index.get(key)
        if step_idx is None:
            continue
        rows = np.asarray(group_rows, dtype=np.int64)
        values = np.clip(pred_value[rows], 0.0, None)
        base = np.clip(baseline_value[rows], 0.0, None)
        scores = np.clip(selection_score[rows], 0.0, None)
        gains = np.clip(values - base, 0.0, None)
        new_mask = base <= EPS
        active_mask = base > EPS
        features[step_idx] = np.array(
            [
                float(rows.size),
                float(np.sum(new_mask)),
                float(np.sum(active_mask)),
                float(np.sum(scores)),
                float(np.max(scores)) if scores.size else 0.0,
                float(np.mean(scores)) if scores.size else 0.0,
                float(np.sum(values)),
                float(np.sum(gains)),
                float(np.max(values)) if values.size else 0.0,
                float(np.sum(base)),
            ],
            dtype=np.float32,
        )
    return features.astype(np.float32)


def make_branch_step_targets(
    keys: np.ndarray,
    coordinates: np.ndarray,
    selected: np.ndarray,
    true_value: np.ndarray,
    labels: np.ndarray,
    baseline_value: np.ndarray,
) -> np.ndarray:
    keys = np.asarray(keys, dtype=np.int64).reshape(-1, 2)
    coords = np.asarray(coordinates, dtype=np.int64).reshape(-1, 3)
    selected = np.asarray(selected, dtype=np.int64).reshape(-1)
    true_value = np.asarray(true_value, dtype=np.float32).reshape(-1)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    baseline_value = np.asarray(baseline_value, dtype=np.float32).reshape(-1)
    if not (coords.shape[0] == true_value.shape[0] == labels.shape[0] == baseline_value.shape[0]):
        raise ValueError('branch target inputs must have the same row count')
    targets = np.zeros((keys.shape[0],), dtype=np.float32)
    key_index = _key_to_index(keys)
    for row_idx in selected:
        sample, step, _edge = coords[int(row_idx)]
        step_idx = key_index.get((int(sample), int(step)))
        if step_idx is None:
            continue
        value_gain = max(float(true_value[int(row_idx)]) - float(baseline_value[int(row_idx)]), 0.0)
        if int(labels[int(row_idx)]) > 0 or float(true_value[int(row_idx)]) > EPS:
            targets[step_idx] += value_gain
    return targets.astype(np.float32)


def fit_branch_value_regressor(features: np.ndarray, targets: np.ndarray, seed: int):
    features = np.asarray(features, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32).reshape(-1)
    if features.shape[0] != targets.shape[0]:
        raise ValueError('branch value features and targets must share row count')
    if features.shape[0] == 0:
        return ConstantValueRegressor(0.0)
    if features.shape[0] < 8 or float(np.max(targets) - np.min(targets)) <= EPS:
        return ConstantValueRegressor(float(np.mean(targets)) if targets.size else 0.0)
    weights = np.ones((targets.shape[0],), dtype=np.float32)
    weights[targets > EPS] = 4.0
    model = HistGradientBoostingRegressor(
        max_iter=160,
        learning_rate=0.05,
        l2_regularization=0.03,
        min_samples_leaf=4,
        random_state=int(seed),
    )
    model.fit(features, targets, sample_weight=weights)
    return model


def structured_branch_specs(
    coordinates: np.ndarray,
    score: np.ndarray,
    baseline_value: np.ndarray,
    primary_selected: np.ndarray,
) -> list[tuple[str, np.ndarray]]:
    return [
        ('primary', np.asarray(primary_selected, dtype=np.int64).reshape(-1)),
        (
            'top16_value_new',
            select_support_indices(
                coordinates,
                score,
                16,
                'baseline_active_value_topk_plus_new_topk',
                baseline_value=baseline_value,
            ),
        ),
        (
            'top24_value_new',
            select_support_indices(
                coordinates,
                score,
                24,
                'baseline_active_value_topk_plus_new_topk',
                baseline_value=baseline_value,
            ),
        ),
        (
            'active16_new8',
            select_support_indices(
                coordinates,
                score,
                0,
                'baseline_active_value_top16_plus_new8',
                baseline_value=baseline_value,
            ),
        ),
    ]


def allocate_step_budget_values(
    coordinates: np.ndarray,
    selected: np.ndarray,
    score: np.ndarray,
    total_by_row: np.ndarray,
    count_by_row: np.ndarray,
    allocation_mode: str,
    baseline_value: np.ndarray | None = None,
    new_count_by_row: np.ndarray | None = None,
    rate_score: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    coords = np.asarray(coordinates, dtype=np.int64).reshape(-1, 3)
    selected = np.asarray(selected, dtype=np.int64).reshape(-1)
    score = np.asarray(score, dtype=np.float32).reshape(-1)
    total_by_row = np.asarray(total_by_row, dtype=np.float32).reshape(-1)
    count_by_row = np.asarray(count_by_row, dtype=np.float32).reshape(-1)
    if baseline_value is None:
        baseline_value = np.zeros((coords.shape[0],), dtype=np.float32)
    baseline_value = np.asarray(baseline_value, dtype=np.float32).reshape(-1)
    if new_count_by_row is None:
        new_count_by_row = np.zeros((coords.shape[0],), dtype=np.float32)
    new_count_by_row = np.asarray(new_count_by_row, dtype=np.float32).reshape(-1)
    if rate_score is None:
        rate_score = np.ones((coords.shape[0],), dtype=np.float32)
    rate_score = np.asarray(rate_score, dtype=np.float32).reshape(-1)
    if not (coords.shape[0] == score.shape[0] == total_by_row.shape[0] == count_by_row.shape[0]):
        raise ValueError('allocation inputs must have the same row count')
    if baseline_value.shape[0] != coords.shape[0]:
        raise ValueError('baseline_value must have the same row count as coordinates')
    if new_count_by_row.shape[0] != coords.shape[0]:
        raise ValueError('new_count_by_row must have the same row count as coordinates')
    if rate_score.shape[0] != coords.shape[0]:
        raise ValueError('rate_score must have the same row count as coordinates')
    values = np.zeros((coords.shape[0],), dtype=np.float32)
    if selected.size == 0:
        return selected.astype(np.int64), values
    selected_by_group: dict[tuple[int, int], list[int]] = {}
    for row_idx in selected:
        sample, step, _edge = coords[int(row_idx)]
        selected_by_group.setdefault((int(sample), int(step)), []).append(int(row_idx))
    allocated = []
    for group_rows in selected_by_group.values():
        rows = np.asarray(group_rows, dtype=np.int64)
        if allocation_mode == 'baseline_score':
            active_priority = (baseline_value[rows] > EPS).astype(np.float32)
            order = np.lexsort((-score[rows], -active_priority))
            row_order = rows[order]
        elif allocation_mode == 'baseline_value_score':
            order = np.lexsort((-score[rows], -baseline_value[rows]))
            row_order = rows[order]
        elif allocation_mode == 'rate_score':
            effective_score = np.clip(score[rows], 0.0, None) * np.log1p(np.clip(rate_score[rows], 0.0, None))
            row_order = rows[np.argsort(-effective_score, kind='mergesort')]
        else:
            row_order = rows[np.argsort(-score[rows], kind='mergesort')]
        count = int(np.clip(np.rint(float(np.nanmean(count_by_row[rows]))), 0, rows.size))
        total = max(float(np.nanmean(total_by_row[rows])), 0.0)
        if count <= 0 or total <= EPS:
            continue
        if allocation_mode == 'new_quota_score':
            new_count = int(np.clip(np.rint(float(np.nanmean(new_count_by_row[rows]))), 0, count))
            inactive_rows = rows[baseline_value[rows] <= EPS]
            active_rows = rows[baseline_value[rows] > EPS]
            chosen_new = inactive_rows[np.argsort(-score[inactive_rows], kind='mergesort')[:new_count]]
            remaining_count = max(0, count - int(chosen_new.shape[0]))
            remaining_pool = np.setdiff1d(rows, chosen_new, assume_unique=False)
            active_first = remaining_pool[baseline_value[remaining_pool] > EPS]
            inactive_rest = remaining_pool[baseline_value[remaining_pool] <= EPS]
            active_order = active_first[np.argsort(-score[active_first], kind='mergesort')]
            inactive_order = inactive_rest[np.argsort(-score[inactive_rest], kind='mergesort')]
            chosen_rest = np.concatenate([active_order, inactive_order])[:remaining_count]
            chosen = np.concatenate([chosen_new, chosen_rest]).astype(np.int64)
        else:
            chosen = row_order[:count]
        allocated.extend(chosen.tolist())
        if allocation_mode == 'uniform':
            weights = np.ones((chosen.shape[0],), dtype=np.float32)
        elif allocation_mode in {'score', 'baseline_score', 'baseline_value_score', 'new_quota_score'}:
            raw = np.clip(score[chosen], 0.0, None).astype(np.float32)
            weights = raw if float(np.sum(raw)) > EPS else np.ones((chosen.shape[0],), dtype=np.float32)
        elif allocation_mode == 'rate_score':
            raw = (
                np.clip(score[chosen], 0.0, None)
                * np.log1p(np.clip(rate_score[chosen], 0.0, None))
            ).astype(np.float32)
            weights = raw if float(np.sum(raw)) > EPS else np.ones((chosen.shape[0],), dtype=np.float32)
        else:
            raise ValueError(f'unknown step budget allocation mode: {allocation_mode}')
        weights = weights / max(float(np.sum(weights)), EPS)
        values[chosen] = (weights * total).astype(np.float32)
    return np.asarray(allocated, dtype=np.int64), values.astype(np.float32)


def apply_step_budget_reconstruction(
    actions: np.ndarray,
    coordinates: np.ndarray,
    values: np.ndarray,
    selected: np.ndarray,
) -> np.ndarray:
    repaired = np.asarray(actions, dtype=np.float32).copy()
    coords = np.asarray(coordinates, dtype=np.int64).reshape(-1, 3)
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    selected = np.asarray(selected, dtype=np.int64).reshape(-1)
    if coords.shape[0] != values.shape[0]:
        raise ValueError('coordinates and values must have the same row count')
    affected_steps = set()
    for row_idx in selected:
        sample, step, _edge = coords[int(row_idx)]
        if int(step) != 0:
            affected_steps.add((int(sample), int(step)))
    for sample, step in affected_steps:
        repaired[sample, step, :, RB_DIM] = 0.0
    for row_idx in selected:
        sample, step, edge = coords[int(row_idx)]
        if int(step) == 0:
            continue
        repaired[int(sample), int(step), int(edge), RB_DIM] = max(float(values[int(row_idx)]), 0.0)
    repaired[:, 0, :, RB_DIM] = actions[:, 0, :, RB_DIM]
    repaired[..., RB_DIM] = np.clip(repaired[..., RB_DIM], 0.0, None)
    return repaired.astype(np.float32)


def apply_gated_step_budget_reconstruction(
    fallback_actions: np.ndarray,
    coordinates: np.ndarray,
    values: np.ndarray,
    selected: np.ndarray,
    confidence: np.ndarray,
    threshold: float,
) -> np.ndarray:
    repaired = np.asarray(fallback_actions, dtype=np.float32).copy()
    coords = np.asarray(coordinates, dtype=np.int64).reshape(-1, 3)
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    selected = np.asarray(selected, dtype=np.int64).reshape(-1)
    confidence = np.asarray(confidence, dtype=np.float32).reshape(-1)
    if not (coords.shape[0] == values.shape[0] == confidence.shape[0]):
        raise ValueError('coordinates, values, and confidence must have the same row count')
    gated_selected = [int(row) for row in selected if float(confidence[int(row)]) >= float(threshold)]
    affected_steps = set()
    for row_idx in gated_selected:
        sample, step, _edge = coords[row_idx]
        if int(step) != 0:
            affected_steps.add((int(sample), int(step)))
    for sample, step in affected_steps:
        repaired[sample, step, :, RB_DIM] = 0.0
    for row_idx in gated_selected:
        sample, step, edge = coords[row_idx]
        if int(step) == 0:
            continue
        repaired[int(sample), int(step), int(edge), RB_DIM] = max(float(values[row_idx]), 0.0)
    repaired[..., RB_DIM] = np.clip(repaired[..., RB_DIM], 0.0, None)
    return repaired.astype(np.float32)


def apply_soft_step_budget_repair(
    fallback_actions: np.ndarray,
    coordinates: np.ndarray,
    values: np.ndarray,
    selected: np.ndarray,
    beta: float,
) -> np.ndarray:
    repaired = np.asarray(fallback_actions, dtype=np.float32).copy()
    coords = np.asarray(coordinates, dtype=np.int64).reshape(-1, 3)
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    selected = np.asarray(selected, dtype=np.int64).reshape(-1)
    if coords.shape[0] != values.shape[0]:
        raise ValueError('coordinates and values must have the same row count')
    beta = float(np.clip(float(beta), 0.0, 1.0))
    for row_idx in selected:
        sample, step, edge = coords[int(row_idx)]
        if int(step) == 0:
            continue
        current = float(repaired[int(sample), int(step), int(edge), RB_DIM])
        target = max(float(values[int(row_idx)]), 0.0)
        repaired[int(sample), int(step), int(edge), RB_DIM] = max((1.0 - beta) * current + beta * target, 0.0)
    repaired[..., RB_DIM] = np.clip(repaired[..., RB_DIM], 0.0, None)
    return repaired.astype(np.float32)


def apply_mass_preserving_reallocation(
    fallback_actions: np.ndarray,
    coordinates: np.ndarray,
    score: np.ndarray,
    selected: np.ndarray,
    beta: float,
) -> np.ndarray:
    repaired = np.asarray(fallback_actions, dtype=np.float32).copy()
    coords = np.asarray(coordinates, dtype=np.int64).reshape(-1, 3)
    score = np.asarray(score, dtype=np.float32).reshape(-1)
    selected = np.asarray(selected, dtype=np.int64).reshape(-1)
    if coords.shape[0] != score.shape[0]:
        raise ValueError('coordinates and score must have the same row count')
    beta = float(np.clip(float(beta), 0.0, 1.0))
    if selected.size == 0 or beta <= 0.0:
        return repaired.astype(np.float32)
    selected_by_group: dict[tuple[int, int], list[int]] = {}
    for row_idx in selected:
        sample, step, _edge = coords[int(row_idx)]
        if int(step) == 0:
            continue
        selected_by_group.setdefault((int(sample), int(step)), []).append(int(row_idx))
    for (sample, step), group_rows in selected_by_group.items():
        rows = np.asarray(group_rows, dtype=np.int64)
        current = repaired[int(sample), int(step), :, RB_DIM].copy()
        total = float(np.sum(np.clip(current, 0.0, None)))
        if total <= EPS:
            continue
        raw = np.clip(score[rows], 0.0, None).astype(np.float32)
        weights = raw if float(np.sum(raw)) > EPS else np.ones((rows.shape[0],), dtype=np.float32)
        weights = weights / max(float(np.sum(weights)), EPS)
        target = current.copy()
        target[:] = 0.0
        for row_idx, weight in zip(rows, weights):
            _sample, _step, edge = coords[int(row_idx)]
            target[int(edge)] = float(weight) * total
        repaired[int(sample), int(step), :, RB_DIM] = (1.0 - beta) * current + beta * target
    repaired[..., RB_DIM] = np.clip(repaired[..., RB_DIM], 0.0, None)
    return repaired.astype(np.float32)


def action_budget_diagnostics(
    actions: np.ndarray,
    baseline_actions: np.ndarray,
    truth_actions: np.ndarray,
) -> dict:
    action_rb = np.clip(np.asarray(actions, dtype=np.float32)[..., RB_DIM], 0.0, None)
    baseline_rb = np.clip(np.asarray(baseline_actions, dtype=np.float32)[..., RB_DIM], 0.0, None)
    truth_rb = np.clip(np.asarray(truth_actions, dtype=np.float32)[..., RB_DIM], 0.0, None)
    if not (action_rb.shape == baseline_rb.shape == truth_rb.shape):
        raise ValueError('actions, baseline_actions, and truth_actions must share the same shape')
    step_slice = slice(1, None)
    action_step_total = np.sum(action_rb[:, step_slice, :], axis=-1)
    truth_step_total = np.sum(truth_rb[:, step_slice, :], axis=-1)
    step_error = action_step_total - truth_step_total
    truth_total = float(np.sum(truth_rb[:, step_slice, :]))
    step_rmse = float(np.sqrt(np.mean(np.square(step_error)))) if step_error.size else float('nan')
    return {
        'action_rb_total': float(np.sum(action_rb[:, step_slice, :])),
        'baseline_rb_total': float(np.sum(baseline_rb[:, step_slice, :])),
        'truth_rb_total': truth_total,
        'action_rb_total_ratio_to_truth': float(np.sum(action_rb[:, step_slice, :]) / max(truth_total, EPS)),
        'action_rb_nonzero_count': int(np.sum(action_rb[:, step_slice, :] > EPS)),
        'baseline_rb_nonzero_count': int(np.sum(baseline_rb[:, step_slice, :] > EPS)),
        'truth_rb_nonzero_count': int(np.sum(truth_rb[:, step_slice, :] > EPS)),
        'action_step_total_mae_vs_truth': float(np.mean(np.abs(step_error))) if step_error.size else float('nan'),
        'action_step_total_rmse_vs_truth': step_rmse,
    }


def apply_step_budget_step_gate_reconstruction(
    fallback_actions: np.ndarray,
    coordinates: np.ndarray,
    values: np.ndarray,
    selected: np.ndarray,
    step_gate_by_row: np.ndarray,
    threshold: float,
) -> np.ndarray:
    repaired = np.asarray(fallback_actions, dtype=np.float32).copy()
    coords = np.asarray(coordinates, dtype=np.int64).reshape(-1, 3)
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    selected = np.asarray(selected, dtype=np.int64).reshape(-1)
    step_gate_by_row = np.asarray(step_gate_by_row, dtype=np.float32).reshape(-1)
    if not (coords.shape[0] == values.shape[0] == step_gate_by_row.shape[0]):
        raise ValueError('coordinates, values, and step_gate_by_row must have the same row count')
    gated_selected = [int(row) for row in selected if float(step_gate_by_row[int(row)]) >= float(threshold)]
    affected_steps = set()
    for row_idx in gated_selected:
        sample, step, _edge = coords[row_idx]
        if int(step) != 0:
            affected_steps.add((int(sample), int(step)))
    for sample, step in affected_steps:
        repaired[sample, step, :, RB_DIM] = 0.0
    for row_idx in gated_selected:
        sample, step, edge = coords[row_idx]
        if int(step) == 0:
            continue
        repaired[int(sample), int(step), int(edge), RB_DIM] = max(float(values[row_idx]), 0.0)
    repaired[..., RB_DIM] = np.clip(repaired[..., RB_DIM], 0.0, None)
    return repaired.astype(np.float32)


def fit_support_classifier(kind: str, features: np.ndarray, labels: np.ndarray, sample_weight: np.ndarray, seed: int, trees: int):
    labels = np.asarray(labels, dtype=np.int64)
    if np.unique(labels).shape[0] < 2:
        return ConstantSupportClassifier(float(np.mean(labels > 0)))
    if kind == 'rf':
        model = RandomForestClassifier(
            n_estimators=int(trees),
            min_samples_leaf=5,
            max_features='sqrt',
            class_weight='balanced_subsample',
            random_state=int(seed),
            n_jobs=-1,
        )
    elif kind == 'hgb':
        model = HistGradientBoostingClassifier(
            max_iter=160,
            learning_rate=0.05,
            l2_regularization=0.02,
            min_samples_leaf=20,
            random_state=int(seed),
        )
    else:
        raise ValueError(f'unknown support model kind: {kind}')
    model.fit(np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int64), sample_weight=np.asarray(sample_weight, dtype=np.float32))
    return model


class ConstantSupportClassifier:
    def __init__(self, positive_probability: float):
        self.positive_probability = float(np.clip(positive_probability, 0.0, 1.0))

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        n_rows = int(np.asarray(features).shape[0])
        pos = np.full((n_rows,), self.positive_probability, dtype=np.float32)
        return np.stack([1.0 - pos, pos], axis=1)


class ConstantValueRegressor:
    def __init__(self, value: float):
        self.value = float(value)

    def predict(self, features: np.ndarray) -> np.ndarray:
        n_rows = int(np.asarray(features).shape[0])
        return np.full((n_rows,), self.value, dtype=np.float32)


class ConstantValueBinClassifier:
    def __init__(self, probabilities: np.ndarray):
        probabilities = np.asarray(probabilities, dtype=np.float32).reshape(-1)
        if probabilities.size == 0:
            probabilities = np.ones((1,), dtype=np.float32)
        total = float(np.sum(np.clip(probabilities, 0.0, None)))
        if total <= EPS:
            probabilities = np.zeros_like(probabilities, dtype=np.float32)
            probabilities[0] = 1.0
        else:
            probabilities = np.clip(probabilities, 0.0, None) / total
        self.probabilities = probabilities.astype(np.float32)
        self.classes_ = np.arange(self.probabilities.shape[0], dtype=np.int64)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        n_rows = int(np.asarray(features).shape[0])
        return np.tile(self.probabilities[None, :], (n_rows, 1)).astype(np.float32)


class PairwiseLinearRanker:
    def __init__(self, model):
        self.model = model

    def predict(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=np.float32)
        if hasattr(self.model, 'decision_function'):
            return np.asarray(self.model.decision_function(features), dtype=np.float32).reshape(-1)
        return np.asarray(self.model.predict_proba(features)[:, 1], dtype=np.float32).reshape(-1)


def predict_support_score(model, features: np.ndarray) -> np.ndarray:
    if hasattr(model, 'predict_proba'):
        proba = np.asarray(model.predict_proba(features), dtype=np.float32)
        if proba.ndim == 2 and proba.shape[1] > 1:
            return proba[:, 1].astype(np.float32)
        return proba.reshape(-1).astype(np.float32)
    score = np.asarray(model.predict(features), dtype=np.float32).reshape(-1)
    finite = score[np.isfinite(score)]
    if finite.size and (float(np.min(finite)) < 0.0 or float(np.max(finite)) > 1.0):
        lo = float(np.min(finite))
        hi = float(np.max(finite))
        if hi > lo:
            score = (score - lo) / (hi - lo)
    return score.astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--world-experiment-dir', type=Path, default=PROJECT_ROOT / 'artifacts/experiments/pi_jwm_v9_expanded_v2_gpu_20260619/v2_hurdle_baseline')
    parser.add_argument('--world-checkpoint', type=Path, default=PROJECT_ROOT / 'artifacts/experiments/pi_jwm_v9_expanded_v2_gpu_20260619/v2_hurdle_baseline/checkpoints/v8_dual_best.pt')
    parser.add_argument('--policy-checkpoint', type=Path, default=PROJECT_ROOT / 'artifacts/experiments/pi_jwm_v10_policy_bridge_gpu_20260620/v10_action_policy_bc/checkpoints/v7_action_policy_cross_attention_best.pt')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='cpu')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--policy-threshold', type=float, default=0.4)
    parser.add_argument('--value-scale', type=float, default=1.0)
    parser.add_argument('--new-policy-threshold', type=float, default=0.37)
    parser.add_argument('--new-value-scale', type=float, default=1.06)
    parser.add_argument('--gate-feature', choices=('step_rb_total', 'step_cpu_total', 'step_rb_cpu_total', 'step_active_count'), default='step_rb_cpu_total')
    parser.add_argument('--gate-threshold', type=float, default=450.0)
    parser.add_argument('--value-codebook-size', type=int, default=9)
    parser.add_argument('--min-effective-rb-total', type=float, default=1.0)
    parser.add_argument('--min-improvement', type=float, default=0.0)
    parser.add_argument('--steps', type=int, nargs='+', default=[1, 2])
    parser.add_argument('--max-train-samples', type=int, default=512)
    parser.add_argument('--max-val-samples', type=int, default=256)
    parser.add_argument('--max-test-samples', type=int, default=256)
    parser.add_argument('--train-sample-start', type=int, default=0)
    parser.add_argument('--val-sample-start', type=int, default=0)
    parser.add_argument('--test-sample-start', type=int, default=0)
    parser.add_argument('--val-shuffle-seed', type=int, default=-1)
    parser.add_argument('--test-shuffle-seed', type=int, default=-1)
    parser.add_argument('--limit-after-stats', action='store_true')
    parser.add_argument('--streaming-stats', action='store_true')
    parser.add_argument('--stats-chunk-size', type=int, default=512)
    parser.add_argument('--support-model-kinds', choices=('rf', 'hgb', 'rank_hgb', 'pairwise_linear', 'xgb_rank'), nargs='+', default=['rf'])
    parser.add_argument('--support-training-strategies', choices=('random', 'hard_negative'), nargs='+', default=['random'])
    parser.add_argument('--rf-trees', type=int, default=80)
    parser.add_argument('--support-max-train-rows', type=int, default=100000)
    parser.add_argument('--support-negative-ratio', type=float, default=20.0)
    parser.add_argument('--hard-negative-fraction', type=float, default=0.7)
    parser.add_argument('--rank-target-mode', choices=('gain', 'gain_norm', 'value_gain_norm', 'true_value_norm'), default='gain_norm')
    parser.add_argument('--pairwise-max-pairs', type=int, default=50000)
    parser.add_argument('--pairwise-negatives-per-positive', type=int, default=16)
    parser.add_argument('--top-k', type=int, nargs='+', default=[16, 32])
    parser.add_argument('--selection-group-modes', nargs='+', default=['fixed'])
    parser.add_argument('--support-thresholds', type=float, nargs='+', default=[0.05])
    parser.add_argument('--blend-alpha', type=float, nargs='+', default=[0.95, 1.0])
    parser.add_argument('--step-total-cap-scale', type=float, nargs='+', default=[1.1, 1.15])
    parser.add_argument('--edge-value-cap-scale', type=float, nargs='+', default=[1.15, 1.25])
    parser.add_argument('--new-edge-value-cap', type=float, nargs='+', default=[5.0, 10.0])
    parser.add_argument('--value-target-mode', choices=('abs', 'log', 'residual', 'ratio'), default='abs')
    parser.add_argument('--value-training-mode', choices=('positive_only', 'all_weighted'), default='positive_only')
    parser.add_argument('--value-positive-weight', type=float, default=8.0)
    parser.add_argument('--candidate-value-positive-weight', type=float, default=16.0)
    parser.add_argument('--candidate-value-bin-positive-weight', type=float, default=16.0)
    parser.add_argument('--candidate-value-bin-count', type=int, default=4)
    parser.add_argument(
        '--repair-value-sources',
        choices=(
            'pred_value',
            'pred_value_reconstruct',
            'train_positive_q50',
            'train_positive_q75',
            'train_positive_q90',
            'support_score_q50',
            'support_score_q75',
            'support_score_q90',
            'pred_value_gate_0p2',
            'pred_value_gate_0p5',
            'pred_value_gate_0p8',
            'candidate_value',
            'positive_candidate_value',
            'candidate_bin_expected',
            'candidate_bin_argmax',
            'candidate_bin_conservative',
            'positive_candidate_bin_expected',
            'positive_candidate_bin_argmax',
            'positive_candidate_bin_conservative',
            'all_candidate_bin_expected',
            'all_candidate_bin_argmax',
            'all_candidate_bin_conservative',
            'selected_gate_q50_t0p01',
            'selected_gate_q50_t0p05',
            'selected_gate_q75_t0p01',
            'selected_gate_q75_t0p05',
            'selected_gate_q75_t0p1',
            'selected_gate_q75_t0p2',
            'structured_stepwise_pred_reconstruct',
            'structured_branch_value_pred_reconstruct',
            'step_budget_pred_score',
            'step_budget_pred_uniform',
            'step_budget_pred_baseline_score',
            'step_budget_pred_baseline_value_score',
            'step_budget_pred_rate_score',
            'step_budget_gated_pred_score_t0p2',
            'step_budget_gated_pred_score_t0p5',
            'step_budget_gated_pred_score_t0p8',
            'step_budget_stepgate_pred_score_t0p2',
            'step_budget_stepgate_pred_score_t0p5',
            'step_budget_stepgate_pred_score_t0p8',
            'step_budget_reconstruct_gate_pred_rate_score_t0p2',
            'step_budget_reconstruct_gate_pred_rate_score_t0p5',
            'step_budget_reconstruct_gate_pred_rate_score_t0p8',
            'step_budget_soft_pred_rate_score_b0p25',
            'step_budget_soft_pred_rate_score_b0p5',
            'step_budget_soft_pred_rate_score_b0p75',
            'mass_preserve_pred_rate_score_b0p1',
            'mass_preserve_pred_rate_score_b0p25',
            'mass_preserve_pred_rate_score_b0p5',
            'mass_preserve_pred_rate_score_b1p0',
            'step_budget_filter_pred_score',
            'step_budget_filter_pred_uniform',
            'step_budget_filter_pred_baseline_score',
            'step_budget_ranker_pred_score',
            'step_budget_newq_pred_score',
            'step_budget_oracle_alloc_pred_score',
            'step_budget_oracle_alloc_pred_uniform',
            'step_budget_oracle_alloc_pred_baseline_score',
            'step_budget_true_score',
            'step_budget_true_uniform',
            'step_budget_true_baseline_score',
            'step_budget_true_baseline_value_score',
            'step_budget_true_rate_score',
            'step_budget_stepgate_true_score_t0p2',
            'step_budget_stepgate_true_score_t0p5',
            'step_budget_stepgate_true_score_t0p8',
            'step_budget_reconstruct_gate_true_rate_score_t0p2',
            'step_budget_reconstruct_gate_true_rate_score_t0p5',
            'step_budget_reconstruct_gate_true_rate_score_t0p8',
            'step_budget_soft_true_rate_score_b0p25',
            'step_budget_soft_true_rate_score_b0p5',
            'step_budget_soft_true_rate_score_b0p75',
            'mass_preserve_true_rate_score_b0p1',
            'mass_preserve_true_rate_score_b0p25',
            'mass_preserve_true_rate_score_b0p5',
            'mass_preserve_true_rate_score_b1p0',
            'step_budget_filter_true_score',
            'step_budget_filter_true_uniform',
            'step_budget_filter_true_baseline_score',
            'step_budget_ranker_true_score',
            'step_budget_newq_true_score',
            'true_value',
        ),
        nargs='+',
        default=['pred_value'],
    )
    parser.add_argument('--selection-score-mode', choices=('support', 'support_minus_value_delta', 'support_value', 'support_rate_value', 'support_gain', 'active_keep_support', 'active_keep_support_value', 'active_keep_support_gain', 'active_keep_support_minus_value_delta', 'oracle_support'), default=None)
    parser.add_argument('--selection-score-modes', choices=('support', 'support_minus_value_delta', 'support_value', 'support_rate_value', 'support_gain', 'active_keep_support', 'active_keep_support_value', 'active_keep_support_gain', 'active_keep_support_minus_value_delta', 'oracle_support'), nargs='+', default=None)
    parser.add_argument('--risk-weight', type=float, default=0.02)
    parser.add_argument('--seed', type=int, default=20260627)
    return parser.parse_args()


def _value_training_rows(labels: np.ndarray, true_values: np.ndarray, min_effective_value: float) -> np.ndarray:
    rows = np.flatnonzero((np.asarray(labels).reshape(-1) > 0) | (np.asarray(true_values).reshape(-1) >= float(min_effective_value)))
    return rows.astype(np.int64)


def make_value_training_rows_and_weights(
    labels: np.ndarray,
    true_values: np.ndarray,
    min_effective_value: float,
    mode: str,
    positive_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    true_values = np.asarray(true_values, dtype=np.float32).reshape(-1)
    if labels.shape != true_values.shape:
        raise ValueError('labels and true_values must share shape')
    positive_mask = (labels > 0) | (true_values >= float(min_effective_value))
    if mode == 'positive_only':
        rows = np.flatnonzero(positive_mask).astype(np.int64)
        return rows, np.ones((rows.shape[0],), dtype=np.float32)
    if mode == 'all_weighted':
        rows = np.arange(true_values.shape[0], dtype=np.int64)
        weights = np.ones((rows.shape[0],), dtype=np.float32)
        weights[positive_mask] = float(positive_weight)
        return rows, weights
    raise ValueError(f'unknown value training mode: {mode}')


def fit_candidate_value_regressor(
    features: np.ndarray,
    true_values: np.ndarray,
    labels: np.ndarray,
    selected_rows: np.ndarray,
    min_effective_value: float,
    positive_weight: float,
    seed: int,
):
    features = np.asarray(features, dtype=np.float32)
    true_values = np.asarray(true_values, dtype=np.float32).reshape(-1)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    selected_rows = np.asarray(selected_rows, dtype=np.int64).reshape(-1)
    if not (features.shape[0] == true_values.shape[0] == labels.shape[0]):
        raise ValueError('features, true_values, and labels must have the same row count')
    if selected_rows.size == 0:
        return ConstantValueRegressor(0.0)
    selected_values = np.clip(true_values[selected_rows], 0.0, None)
    if selected_rows.size < 8 or float(np.max(selected_values)) <= EPS:
        return ConstantValueRegressor(float(np.mean(selected_values)) if selected_values.size else 0.0)
    selected_labels = labels[selected_rows]
    positive_mask = (selected_labels > 0) | (selected_values >= float(min_effective_value))
    weights = np.ones((selected_rows.shape[0],), dtype=np.float32)
    weights[positive_mask] = float(positive_weight)
    model = HistGradientBoostingRegressor(
        max_iter=180,
        learning_rate=0.04,
        l2_regularization=0.03,
        min_samples_leaf=8,
        random_state=int(seed),
    )
    model.fit(features[selected_rows], selected_values, sample_weight=weights)
    return model


def build_candidate_value_codebook(
    values: np.ndarray,
    min_effective_value: float,
    positive_bin_count: int,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    positive = values[np.isfinite(values) & (values >= float(min_effective_value))]
    if positive.size == 0:
        return np.array([0.0], dtype=np.float32)
    quantiles = np.linspace(0.0, 1.0, max(1, int(positive_bin_count)), dtype=np.float64)
    centers = np.unique(np.quantile(positive, quantiles).astype(np.float32))
    centers = centers[np.isfinite(centers) & (centers >= float(min_effective_value))]
    if centers.size == 0:
        centers = np.array([float(np.median(positive))], dtype=np.float32)
    centers = np.sort(centers.astype(np.float32))
    return np.concatenate([np.array([0.0], dtype=np.float32), centers]).astype(np.float32)


def encode_candidate_value_bins(values: np.ndarray, centers: np.ndarray, min_effective_value: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    centers = np.asarray(centers, dtype=np.float32).reshape(-1)
    if centers.size == 0:
        raise ValueError('centers must not be empty')
    labels = np.zeros((values.shape[0],), dtype=np.int64)
    if centers.size == 1:
        return labels
    positive_centers = centers[1:]
    positive_rows = values >= float(min_effective_value)
    if np.any(positive_rows):
        distances = np.abs(values[positive_rows, None] - positive_centers[None, :])
        labels[positive_rows] = 1 + np.argmin(distances, axis=1).astype(np.int64)
    return labels.astype(np.int64)


def fit_candidate_value_bin_classifier(
    features: np.ndarray,
    true_values: np.ndarray,
    labels: np.ndarray,
    selected_rows: np.ndarray,
    min_effective_value: float,
    positive_weight: float,
    positive_bin_count: int,
    seed: int,
) -> CandidateValueBinModel:
    features = np.asarray(features, dtype=np.float32)
    true_values = np.asarray(true_values, dtype=np.float32).reshape(-1)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    selected_rows = np.asarray(selected_rows, dtype=np.int64).reshape(-1)
    if not (features.shape[0] == true_values.shape[0] == labels.shape[0]):
        raise ValueError('features, true_values, and labels must have the same row count')
    if selected_rows.size == 0:
        centers = np.array([0.0], dtype=np.float32)
        return CandidateValueBinModel(centers=centers, classifier=ConstantValueBinClassifier(np.array([1.0], dtype=np.float32)))
    selected_values = np.clip(true_values[selected_rows], 0.0, None)
    centers = build_candidate_value_codebook(
        selected_values,
        min_effective_value=float(min_effective_value),
        positive_bin_count=int(positive_bin_count),
    )
    class_labels = encode_candidate_value_bins(selected_values, centers, min_effective_value=float(min_effective_value))
    positive_mask = (labels[selected_rows] > 0) | (selected_values >= float(min_effective_value))
    weights = np.ones((selected_rows.shape[0],), dtype=np.float32)
    weights[positive_mask] = float(positive_weight)
    bincount = np.bincount(class_labels, weights=weights, minlength=centers.shape[0]).astype(np.float32)
    if selected_rows.size < 8 or np.unique(class_labels).shape[0] < 2:
        return CandidateValueBinModel(centers=centers, classifier=ConstantValueBinClassifier(bincount))
    model = HistGradientBoostingClassifier(
        max_iter=180,
        learning_rate=0.04,
        l2_regularization=0.03,
        min_samples_leaf=8,
        random_state=int(seed),
    )
    model.fit(features[selected_rows], class_labels, sample_weight=weights)
    return CandidateValueBinModel(centers=centers, classifier=model)


def make_selected_gate_weights(labels: np.ndarray) -> np.ndarray:
    labels = (np.asarray(labels, dtype=np.int64).reshape(-1) > 0).astype(np.int64)
    if labels.size == 0:
        return np.zeros((0,), dtype=np.float32)
    positive = labels > 0
    pos_count = int(np.sum(positive))
    neg_count = int(labels.size - pos_count)
    if pos_count == 0 or neg_count == 0:
        return np.ones((labels.shape[0],), dtype=np.float32)
    weights = np.ones((labels.shape[0],), dtype=np.float32)
    weights[positive] = labels.size / (2.0 * pos_count)
    weights[~positive] = labels.size / (2.0 * neg_count)
    return (weights / max(float(np.mean(weights)), EPS)).astype(np.float32)


def fit_selected_candidate_gate(
    features: np.ndarray,
    true_values: np.ndarray,
    labels: np.ndarray,
    selected_rows: np.ndarray,
    min_effective_value: float,
    seed: int,
):
    features = np.asarray(features, dtype=np.float32)
    true_values = np.asarray(true_values, dtype=np.float32).reshape(-1)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    selected_rows = np.asarray(selected_rows, dtype=np.int64).reshape(-1)
    if not (features.shape[0] == true_values.shape[0] == labels.shape[0]):
        raise ValueError('features, true_values, and labels must have the same row count')
    if selected_rows.size == 0:
        return ConstantSupportClassifier(0.0)
    gate_labels = ((labels[selected_rows] > 0) | (true_values[selected_rows] >= float(min_effective_value))).astype(np.int64)
    if np.unique(gate_labels).shape[0] < 2:
        return ConstantSupportClassifier(float(np.mean(gate_labels)) if gate_labels.size else 0.0)
    model = HistGradientBoostingClassifier(
        max_iter=180,
        learning_rate=0.04,
        l2_regularization=0.03,
        min_samples_leaf=8,
        random_state=int(seed),
    )
    model.fit(features[selected_rows], gate_labels, sample_weight=make_selected_gate_weights(gate_labels))
    return model


def fit_allocation_filter(
    features: np.ndarray,
    true_values: np.ndarray,
    labels: np.ndarray,
    selected_rows: np.ndarray,
    min_effective_value: float,
    seed: int,
):
    features = np.asarray(features, dtype=np.float32)
    true_values = np.asarray(true_values, dtype=np.float32).reshape(-1)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    selected_rows = np.asarray(selected_rows, dtype=np.int64).reshape(-1)
    if not (features.shape[0] == true_values.shape[0] == labels.shape[0]):
        raise ValueError('features, true_values, and labels must have the same row count')
    if selected_rows.size == 0:
        return ConstantSupportClassifier(0.0)
    target = ((labels[selected_rows] > 0) | (true_values[selected_rows] >= float(min_effective_value))).astype(np.int64)
    if np.unique(target).shape[0] < 2:
        return ConstantSupportClassifier(float(np.mean(target)) if target.size else 0.0)
    sample_weight = make_selected_gate_weights(target)
    positive_value = np.log1p(np.clip(true_values[selected_rows], 0.0, None)).astype(np.float32)
    sample_weight = (sample_weight * (1.0 + 0.5 * positive_value)).astype(np.float32)
    sample_weight = sample_weight / max(float(np.mean(sample_weight)), EPS)
    model = HistGradientBoostingClassifier(
        max_iter=220,
        learning_rate=0.035,
        l2_regularization=0.04,
        min_samples_leaf=8,
        random_state=int(seed),
    )
    model.fit(features[selected_rows], target, sample_weight=sample_weight)
    return model


def fit_allocation_ranker(
    features: np.ndarray,
    coordinates: np.ndarray,
    true_values: np.ndarray,
    labels: np.ndarray,
    selected_rows: np.ndarray,
    min_effective_value: float,
    seed: int,
):
    features = np.asarray(features, dtype=np.float32)
    coords = np.asarray(coordinates, dtype=np.int64).reshape(-1, 3)
    true_values = np.asarray(true_values, dtype=np.float32).reshape(-1)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    selected_rows = np.asarray(selected_rows, dtype=np.int64).reshape(-1)
    if not (features.shape[0] == coords.shape[0] == true_values.shape[0] == labels.shape[0]):
        raise ValueError('allocation ranker inputs must have the same row count')
    if selected_rows.size == 0:
        return ConstantValueRegressor(0.0)
    target = make_group_rank_targets(coords, true_values, true_values, mode='true_value_norm')
    selected_target = np.asarray(target[selected_rows], dtype=np.float32).reshape(-1)
    if selected_rows.size < 8 or float(np.max(selected_target) - np.min(selected_target)) <= EPS:
        return ConstantValueRegressor(float(np.mean(selected_target)) if selected_target.size else 0.0)
    positive = ((labels[selected_rows] > 0) | (true_values[selected_rows] >= float(min_effective_value))).astype(np.float32)
    value_scale = np.log1p(np.clip(true_values[selected_rows], 0.0, None)).astype(np.float32)
    sample_weight = (1.0 + 8.0 * positive + 0.25 * value_scale).astype(np.float32)
    sample_weight = sample_weight / max(float(np.mean(sample_weight)), EPS)
    model = HistGradientBoostingRegressor(
        max_iter=260,
        learning_rate=0.035,
        l2_regularization=0.04,
        min_samples_leaf=8,
        random_state=int(seed),
    )
    model.fit(features[selected_rows], selected_target, sample_weight=sample_weight)
    return model


def is_active_keep_score_mode(mode: str) -> bool:
    return str(mode).startswith('active_keep_')


def active_keep_base_score_mode(mode: str) -> str:
    text = str(mode)
    if not is_active_keep_score_mode(text):
        return text
    base = text.removeprefix('active_keep_')
    if base not in {'support', 'support_value', 'support_gain', 'support_minus_value_delta'}:
        raise ValueError(f'unknown active keep base score mode: {mode}')
    return base


def fit_active_keep_classifier(
    features: np.ndarray,
    true_values: np.ndarray,
    labels: np.ndarray,
    baseline_values: np.ndarray,
    min_effective_value: float,
    seed: int,
):
    features = np.asarray(features, dtype=np.float32)
    true_values = np.asarray(true_values, dtype=np.float32).reshape(-1)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    baseline_values = np.asarray(baseline_values, dtype=np.float32).reshape(-1)
    if not (features.shape[0] == true_values.shape[0] == labels.shape[0] == baseline_values.shape[0]):
        raise ValueError('active keep inputs must have the same row count')
    active_rows = np.flatnonzero(baseline_values > EPS).astype(np.int64)
    if active_rows.size == 0:
        return ConstantSupportClassifier(0.0)
    target = ((labels[active_rows] > 0) | (true_values[active_rows] >= float(min_effective_value))).astype(np.int64)
    if np.unique(target).shape[0] < 2:
        return ConstantSupportClassifier(float(np.mean(target)) if target.size else 0.0)
    sample_weight = make_selected_gate_weights(target)
    baseline_scale = np.log1p(np.clip(baseline_values[active_rows], 0.0, None)).astype(np.float32)
    sample_weight = (sample_weight * (1.0 + 0.25 * baseline_scale)).astype(np.float32)
    sample_weight = sample_weight / max(float(np.mean(sample_weight)), EPS)
    model = HistGradientBoostingClassifier(
        max_iter=220,
        learning_rate=0.035,
        l2_regularization=0.04,
        min_samples_leaf=8,
        random_state=int(seed),
    )
    model.fit(features[active_rows], target, sample_weight=sample_weight)
    return model


def fit_step_budget_regressors(features: np.ndarray, targets: StepBudgetTargets, seed: int):
    features = np.asarray(features, dtype=np.float32)
    total = np.asarray(targets.total, dtype=np.float32).reshape(-1)
    count = np.asarray(targets.count, dtype=np.float32).reshape(-1)
    if not (features.shape[0] == total.shape[0] == count.shape[0]):
        raise ValueError('step budget features and targets must share row count')
    total_model = HistGradientBoostingRegressor(
        max_iter=180,
        learning_rate=0.04,
        l2_regularization=0.03,
        min_samples_leaf=8,
        random_state=int(seed),
    )
    count_model = HistGradientBoostingRegressor(
        max_iter=180,
        learning_rate=0.04,
        l2_regularization=0.03,
        min_samples_leaf=8,
        random_state=int(seed) + 1,
    )
    if features.shape[0] == 0:
        return ConstantValueRegressor(0.0), ConstantValueRegressor(0.0)
    total_model.fit(features, total)
    count_model.fit(features, count)
    return total_model, count_model


def fit_step_budget_new_count_regressor(features: np.ndarray, targets: StepBudgetTargets, seed: int):
    features = np.asarray(features, dtype=np.float32)
    new_count = np.asarray(targets.new_count, dtype=np.float32).reshape(-1)
    if features.shape[0] != new_count.shape[0]:
        raise ValueError('step budget features and new-count targets must share row count')
    if features.shape[0] == 0:
        return ConstantValueRegressor(0.0)
    model = HistGradientBoostingRegressor(
        max_iter=180,
        learning_rate=0.04,
        l2_regularization=0.03,
        min_samples_leaf=8,
        random_state=int(seed),
    )
    model.fit(features, new_count)
    return model


def fit_step_gate_classifier(features: np.ndarray, targets: StepBudgetTargets, seed: int):
    features = np.asarray(features, dtype=np.float32)
    labels = (np.asarray(targets.total, dtype=np.float32).reshape(-1) > EPS).astype(np.int64)
    if features.shape[0] != labels.shape[0]:
        raise ValueError('step gate features and targets must share row count')
    if features.shape[0] == 0:
        return ConstantSupportClassifier(0.0)
    if np.unique(labels).shape[0] < 2:
        return ConstantSupportClassifier(float(np.mean(labels)))
    weights = make_selected_gate_weights(labels)
    model = HistGradientBoostingClassifier(
        max_iter=180,
        learning_rate=0.04,
        l2_regularization=0.03,
        min_samples_leaf=8,
        random_state=int(seed),
    )
    model.fit(features, labels, sample_weight=weights)
    return model


def map_step_predictions_to_rows(row_to_step: np.ndarray, step_values: np.ndarray) -> np.ndarray:
    row_to_step = np.asarray(row_to_step, dtype=np.int64).reshape(-1)
    step_values = np.asarray(step_values, dtype=np.float32).reshape(-1)
    if row_to_step.size == 0:
        return np.zeros((0,), dtype=np.float32)
    return step_values[np.clip(row_to_step, 0, max(step_values.shape[0] - 1, 0))].astype(np.float32)


def selected_step_confidence(
    coordinates: np.ndarray,
    selected: np.ndarray,
    support_score: np.ndarray,
) -> np.ndarray:
    coords = np.asarray(coordinates, dtype=np.int64).reshape(-1, 3)
    selected = np.asarray(selected, dtype=np.int64).reshape(-1)
    support_score = np.asarray(support_score, dtype=np.float32).reshape(-1)
    if coords.shape[0] != support_score.shape[0]:
        raise ValueError('coordinates and support_score must share row count')
    confidence = np.zeros((coords.shape[0],), dtype=np.float32)
    if selected.size == 0:
        return confidence
    groups: dict[tuple[int, int], list[int]] = {}
    for row_idx in selected:
        sample, step, _edge = coords[int(row_idx)]
        groups.setdefault((int(sample), int(step)), []).append(int(row_idx))
    for group_rows in groups.values():
        rows = np.asarray(group_rows, dtype=np.int64)
        value = float(np.mean(np.clip(support_score[rows], 0.0, 1.0))) if rows.size else 0.0
        confidence[rows] = value
    return confidence.astype(np.float32)


def predict_candidate_value_bin_probabilities(model: CandidateValueBinModel, features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    centers = np.asarray(model.centers, dtype=np.float32).reshape(-1)
    raw = np.asarray(model.classifier.predict_proba(features), dtype=np.float32)
    if raw.ndim != 2:
        raise ValueError('candidate value bin classifier must return a 2D probability array')
    probabilities = np.zeros((features.shape[0], centers.shape[0]), dtype=np.float32)
    classes = np.asarray(getattr(model.classifier, 'classes_', np.arange(raw.shape[1])), dtype=np.int64).reshape(-1)
    for col, cls in enumerate(classes):
        if 0 <= int(cls) < centers.shape[0]:
            probabilities[:, int(cls)] = raw[:, col]
    totals = probabilities.sum(axis=1, keepdims=True)
    zero_rows = totals[:, 0] <= EPS
    if np.any(zero_rows):
        probabilities[zero_rows, 0] = 1.0
        totals = probabilities.sum(axis=1, keepdims=True)
    return (probabilities / np.clip(totals, EPS, None)).astype(np.float32)


def decode_candidate_value_bins(model: CandidateValueBinModel, features: np.ndarray, mode: str) -> np.ndarray:
    probabilities = predict_candidate_value_bin_probabilities(model, features)
    centers = np.asarray(model.centers, dtype=np.float32).reshape(-1)
    if mode == 'expected':
        return np.sum(probabilities * centers[None, :], axis=1).astype(np.float32)
    if mode == 'argmax':
        return centers[np.argmax(probabilities, axis=1)].astype(np.float32)
    if mode == 'conservative':
        if centers.shape[0] <= 1:
            return np.zeros((probabilities.shape[0],), dtype=np.float32)
        positive_probability = np.sum(probabilities[:, 1:], axis=1)
        positive_mass = np.sum(probabilities[:, 1:] * centers[None, 1:], axis=1)
        conditional_value = positive_mass / np.clip(positive_probability, EPS, None)
        return np.where(positive_probability >= 0.5, conditional_value, 0.0).astype(np.float32)
    raise ValueError(f'unknown candidate value bin decode mode: {mode}')


def _fit_value_model(
    args,
    features: np.ndarray,
    true_values: np.ndarray,
    baseline_values: np.ndarray,
    labels: np.ndarray | None = None,
):
    factories = build_models(seed=int(args.seed), rf_trees=int(args.rf_trees))
    value_model = factories['rf']()
    if labels is None:
        labels = np.zeros_like(np.asarray(true_values).reshape(-1), dtype=np.int64)
    rows, weights = make_value_training_rows_and_weights(
        labels,
        true_values,
        min_effective_value=float(args.min_effective_rb_total),
        mode=str(args.value_training_mode),
        positive_weight=float(args.value_positive_weight),
    )
    if rows.size == 0:
        rows = np.arange(features.shape[0], dtype=np.int64)
        weights = np.ones((rows.shape[0],), dtype=np.float32)
    target = make_value_target(str(args.value_target_mode), true_values[rows], baseline_values[rows])
    value_model.fit(features[rows], target, sample_weight=weights)
    return value_model


def _make_split_payload(args, split_name, arrays, splits, stats, policy_model, action_scale, value_vocab, world_model, summary, device, support_models, value_model, steps):
    base_dataset, adaptive_dataset = make_adaptive_dataset(
        args, arrays, splits[split_name], stats, policy_model, action_scale, value_vocab, device, splits['train']
    )
    baseline_actions, truth_actions = collect_raw_actions(adaptive_dataset, stats)
    examples = make_all_edge_examples(baseline_actions, truth_actions, steps=steps)
    edge_improvement = collect_edge_gradient_improvement(
        world_model, base_dataset, baseline_actions, truth_actions, stats, summary['config'], device, args.batch_size
    )
    labels, oracle_score, true_value = make_all_edge_targets(
        examples,
        truth_actions,
        edge_improvement,
        min_effective_value=float(args.min_effective_rb_total),
        min_improvement=float(args.min_improvement),
    )
    context = collect_rollout_edge_context(world_model, base_dataset, baseline_actions, stats, device, args.batch_size)
    features = rows_from_context(context, examples.coordinates)
    raw_pred_value = predict_conservative_value_target(value_model, features, mode='mean', beta=0.0)
    pred_value = invert_value_target(str(args.value_target_mode), raw_pred_value, examples.baseline_values)
    support_scores = {name: predict_support_score(model, features) for name, model in support_models.items()}
    return {
        'base_dataset': base_dataset,
        'baseline_actions': baseline_actions,
        'truth_actions': truth_actions,
        'examples': examples,
        'labels': labels,
        'oracle_score': oracle_score,
        'true_value': true_value,
        'features': features,
        'pred_value': pred_value,
        'support_scores': support_scores,
    }


def _selection_score(
    mode: str,
    support_score: np.ndarray,
    baseline_value: np.ndarray,
    pred_value: np.ndarray,
    oracle_score: np.ndarray,
    risk_weight: float,
    edge_features: np.ndarray | None = None,
) -> np.ndarray:
    support_score = np.asarray(support_score, dtype=np.float32).reshape(-1)
    baseline_value = np.asarray(baseline_value, dtype=np.float32).reshape(-1)
    pred_value = np.asarray(pred_value, dtype=np.float32).reshape(-1)
    oracle_score = np.asarray(oracle_score, dtype=np.float32).reshape(-1)
    if mode == 'support':
        return support_score
    if mode == 'support_value':
        value_scale = np.log1p(np.clip(pred_value, 0.0, None))
        return (support_score * value_scale).astype(np.float32)
    if mode == 'support_gain':
        gain_scale = np.log1p(np.clip(pred_value - baseline_value, 0.0, None))
        return (support_score * gain_scale).astype(np.float32)
    if mode == 'support_rate_value':
        if edge_features is None:
            raise ValueError('support_rate_value requires edge_features')
        edge_features = np.asarray(edge_features, dtype=np.float32)
        if edge_features.ndim != 2 or edge_features.shape[0] != support_score.shape[0]:
            raise ValueError('edge_features must be 2D and share row count with support_score')
        rate_context = np.mean(np.clip(edge_features[:, -3:], 0.0, None), axis=1)
        rate_scale = np.log1p(rate_context)
        value_scale = np.log1p(np.clip(pred_value, 0.0, None))
        return (support_score * rate_scale * value_scale).astype(np.float32)
    if mode == 'support_minus_value_delta':
        delta = np.clip(pred_value - baseline_value, 0.0, None) / np.maximum(np.clip(baseline_value, 0.0, None), 1.0)
        return (support_score - float(risk_weight) * delta).astype(np.float32)
    if mode == 'oracle_support':
        return oracle_score
    raise ValueError(f'unknown selection score mode: {mode}')


def run(args: argparse.Namespace) -> dict:
    started = time.time()
    device = resolve_torch_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary, arrays, stats, world_model, policy_model, action_scale, value_vocab, splits = load_context_limited(args, device)
    splits = dict(splits)
    if args.limit_after_stats:
        splits['train'] = limit_indices_window(
            splits['train'],
            args.max_train_samples,
            start=int(args.train_sample_start),
        )
        splits['val'] = limit_indices_window(
            splits['val'],
            args.max_val_samples,
            start=int(args.val_sample_start),
            shuffle_seed=int(args.val_shuffle_seed),
        )
        splits['test'] = limit_indices_window(
            splits['test'],
            args.max_test_samples,
            start=int(args.test_sample_start),
            shuffle_seed=int(args.test_shuffle_seed),
        )
    world_model = freeze_module(world_model)
    policy_model = freeze_module(policy_model)
    steps = tuple(int(step) for step in args.steps)

    train_base, train_dataset = make_adaptive_dataset(args, arrays, splits['train'], stats, policy_model, action_scale, value_vocab, device, splits['train'])
    train_actions, train_truth = collect_raw_actions(train_dataset, stats)
    train_examples = make_all_edge_examples(train_actions, train_truth, steps=steps)
    train_edge_improvement = collect_edge_gradient_improvement(
        world_model, train_base, train_actions, train_truth, stats, summary['config'], device, args.batch_size
    )
    train_labels, train_score, train_value = make_all_edge_targets(
        train_examples,
        train_truth,
        train_edge_improvement,
        min_effective_value=float(args.min_effective_rb_total),
        min_improvement=float(args.min_improvement),
    )
    train_context = collect_rollout_edge_context(world_model, train_base, train_actions, stats, device, args.batch_size)
    train_features = rows_from_context(train_context, train_examples.coordinates)
    positive_train_values = np.asarray(train_value, dtype=np.float32).reshape(-1)
    positive_train_values = positive_train_values[positive_train_values >= float(args.min_effective_rb_total)]
    if positive_train_values.size:
        repair_value_lookup = {
            'train_positive_q50': float(np.quantile(positive_train_values, 0.50)),
            'train_positive_q75': float(np.quantile(positive_train_values, 0.75)),
            'train_positive_q90': float(np.quantile(positive_train_values, 0.90)),
        }
    else:
        repair_value_lookup = {
            'train_positive_q50': 0.0,
            'train_positive_q75': 0.0,
            'train_positive_q90': 0.0,
        }
    train_rank_target = make_group_rank_targets(
        train_examples.coordinates,
        train_score,
        train_value,
        mode=str(args.rank_target_mode),
    )
    train_rows = downsample_support_training_rows(
        train_labels,
        train_score,
        max_rows=int(args.support_max_train_rows),
        negative_ratio=float(args.support_negative_ratio),
        seed=int(args.seed),
    )
    support_models = {}
    support_train_rows_used = {}
    support_score_train_corr = {}
    for kind in args.support_model_kinds:
        if kind == 'rank_hgb':
            for strategy in args.support_training_strategies:
                if strategy == 'random':
                    rank_rows = train_rows
                else:
                    warm_weight = (1.0 + 10.0 * train_score[train_rows]).astype(np.float32)
                    warm_model = fit_support_classifier(
                        'hgb',
                        train_features[train_rows],
                        train_labels[train_rows],
                        warm_weight,
                        int(args.seed),
                        int(args.rf_trees),
                    )
                    warm_scores = predict_support_score(warm_model, train_features)
                    rank_rows = downsample_hard_negative_support_training_rows(
                        train_labels,
                        train_score,
                        warm_scores,
                        max_rows=int(args.support_max_train_rows),
                        hard_fraction=float(args.hard_negative_fraction),
                        seed=int(args.seed) + 31,
                    )
                model = HistGradientBoostingRegressor(
                    max_iter=220,
                    learning_rate=0.04,
                    l2_regularization=0.03,
                    min_samples_leaf=20,
                    random_state=int(args.seed) + 41,
                )
                rank_weight = (1.0 + 20.0 * train_rank_target[rank_rows] + 5.0 * (train_labels[rank_rows] > 0)).astype(np.float32)
                model.fit(train_features[rank_rows], train_rank_target[rank_rows], sample_weight=rank_weight)
                model_name = f'{kind}_{strategy}'
                support_models[model_name] = model
                support_train_rows_used[model_name] = int(rank_rows.shape[0])
                support_score_train_corr[model_name] = safe_corr(predict_support_score(model, train_features), train_rank_target)
            continue

        if kind == 'pairwise_linear':
            pair_features, pair_labels = make_pairwise_preference_examples(
                train_features,
                train_examples.coordinates,
                train_rank_target,
                max_pairs=int(args.pairwise_max_pairs),
                negatives_per_positive=int(args.pairwise_negatives_per_positive),
                seed=int(args.seed) + 53,
            )
            if pair_labels.size == 0 or np.unique(pair_labels).shape[0] < 2:
                model = ConstantSupportClassifier(0.0)
            else:
                model = PairwiseLinearRanker(
                    make_pipeline(
                        StandardScaler(),
                        LogisticRegression(
                            max_iter=1000,
                            class_weight='balanced',
                            random_state=int(args.seed) + 59,
                        ),
                    ).fit(pair_features, pair_labels)
                )
            model_name = f'{kind}_pairwise'
            support_models[model_name] = model
            support_train_rows_used[model_name] = int(pair_labels.shape[0])
            support_score_train_corr[model_name] = safe_corr(predict_support_score(model, train_features), train_rank_target)
            continue

        if kind == 'xgb_rank':
            for strategy in args.support_training_strategies:
                if strategy == 'random':
                    rank_rows = train_rows
                else:
                    warm_weight = (1.0 + 10.0 * train_score[train_rows]).astype(np.float32)
                    warm_model = fit_support_classifier(
                        'hgb',
                        train_features[train_rows],
                        train_labels[train_rows],
                        warm_weight,
                        int(args.seed),
                        int(args.rf_trees),
                    )
                    warm_scores = predict_support_score(warm_model, train_features)
                    rank_rows = downsample_hard_negative_support_training_rows(
                        train_labels,
                        train_score,
                        warm_scores,
                        max_rows=int(args.support_max_train_rows),
                        hard_fraction=float(args.hard_negative_fraction),
                        seed=int(args.seed) + 67,
                    )
                model_name = f'{kind}_{strategy}'
                support_models[model_name] = fit_xgb_ranker(
                    train_features,
                    train_examples.coordinates,
                    train_rank_target,
                    rank_rows,
                    seed=int(args.seed) + 71,
                    trees=int(args.rf_trees),
                )
                support_train_rows_used[model_name] = int(rank_rows.shape[0])
                support_score_train_corr[model_name] = safe_corr(
                    predict_support_score(support_models[model_name], train_features),
                    train_rank_target,
                )
            continue

        if 'random' in args.support_training_strategies:
            sample_weight = (1.0 + 10.0 * train_score[train_rows]).astype(np.float32)
            model_name = f'{kind}_random'
            support_models[model_name] = fit_support_classifier(
                kind,
                train_features[train_rows],
                train_labels[train_rows],
                sample_weight,
                int(args.seed),
                int(args.rf_trees),
            )
            support_train_rows_used[model_name] = int(train_rows.shape[0])
            support_score_train_corr[model_name] = safe_corr(predict_support_score(support_models[model_name], train_features), train_labels)
        if 'hard_negative' in args.support_training_strategies:
            warm_weight = (1.0 + 10.0 * train_score[train_rows]).astype(np.float32)
            warm_model = fit_support_classifier(
                kind,
                train_features[train_rows],
                train_labels[train_rows],
                warm_weight,
                int(args.seed),
                int(args.rf_trees),
            )
            warm_scores = predict_support_score(warm_model, train_features)
            hard_rows = downsample_hard_negative_support_training_rows(
                train_labels,
                train_score,
                warm_scores,
                max_rows=int(args.support_max_train_rows),
                hard_fraction=float(args.hard_negative_fraction),
                seed=int(args.seed) + 17,
            )
            hard_weight = (1.0 + 10.0 * train_score[hard_rows]).astype(np.float32)
            model_name = f'{kind}_hardneg'
            support_models[model_name] = fit_support_classifier(
                kind,
                train_features[hard_rows],
                train_labels[hard_rows],
                hard_weight,
                int(args.seed) + 23,
                int(args.rf_trees),
            )
            support_train_rows_used[model_name] = int(hard_rows.shape[0])
            support_score_train_corr[model_name] = safe_corr(predict_support_score(support_models[model_name], train_features), train_labels)
    value_model = _fit_value_model(args, train_features, train_value, train_examples.baseline_values, train_labels)
    train_raw_pred_value = predict_conservative_value_target(value_model, train_features, mode='mean', beta=0.0)
    train_pred_value = invert_value_target(str(args.value_target_mode), train_raw_pred_value, train_examples.baseline_values)
    train_support_scores = {name: predict_support_score(model, train_features) for name, model in support_models.items()}
    selection_score_modes = list(args.selection_score_modes or ([args.selection_score_mode] if args.selection_score_mode else ['support']))
    active_keep_models = {}
    active_keep_train_corr = {}
    active_keep_target = ((train_labels > 0) | (train_value >= float(args.min_effective_rb_total))).astype(np.int64)
    active_train_rows = np.flatnonzero(train_examples.baseline_values > EPS).astype(np.int64)
    for model_name, train_support_score in train_support_scores.items():
        for score_mode in selection_score_modes:
            if not is_active_keep_score_mode(str(score_mode)):
                continue
            base_score_mode = active_keep_base_score_mode(str(score_mode))
            train_selection_score = _selection_score(
                base_score_mode,
                train_support_score,
                train_examples.baseline_values,
                train_pred_value,
                train_score,
                float(args.risk_weight),
                edge_features=train_features,
            )
            train_active_keep_features = make_candidate_value_features(
                train_features,
                train_examples.coordinates,
                train_support_score,
                train_selection_score,
                train_examples.baseline_values,
            )
            key = (model_name, str(score_mode))
            active_keep_models[key] = fit_active_keep_classifier(
                train_active_keep_features,
                train_value,
                train_labels,
                train_examples.baseline_values,
                min_effective_value=float(args.min_effective_rb_total),
                seed=int(args.seed) + 1709,
            )
            if active_train_rows.size:
                active_keep_train_corr[f'{model_name}__{score_mode}'] = safe_corr(
                    predict_support_score(active_keep_models[key], train_active_keep_features)[active_train_rows],
                    active_keep_target[active_train_rows],
                )
            else:
                active_keep_train_corr[f'{model_name}__{score_mode}'] = float('nan')
    candidate_value_model_cache = {}
    step_budget_model_cache = {}

    split_payload = {
        name: _make_split_payload(args, name, arrays, splits, stats, policy_model, action_scale, value_vocab, world_model, summary, device, support_models, value_model, steps)
        for name in ('val', 'test')
    }

    rows = []
    baseline_rmse_by_split = {}
    for split_name, payload in split_payload.items():
        baseline_predictions = evaluate_raw_actions(
            payload['baseline_actions'], payload['base_dataset'], stats, world_model, summary['config'], device, args.batch_size
        )
        baseline_row = active_rate_row('identity_bc_reference', split_name, baseline_predictions, float('nan'))
        baseline_row.update({'family': 'identity', 'support_model': 'none', 'top_k': 0})
        rows.append(baseline_row)
        baseline_rmse_by_split[split_name] = float(baseline_row['active_rate_rmse'])

        candidate_scores: list[tuple[str, str, str, np.ndarray]] = []
        for model_name, score in payload['support_scores'].items():
            for score_mode in selection_score_modes:
                base_score_mode = active_keep_base_score_mode(str(score_mode))
                selection_score = _selection_score(
                    base_score_mode,
                    score,
                    payload['examples'].baseline_values,
                    payload['pred_value'],
                    payload['oracle_score'],
                    float(args.risk_weight),
                    edge_features=payload['features'],
                )
                if is_active_keep_score_mode(str(score_mode)):
                    active_keep_features = make_candidate_value_features(
                        payload['features'],
                        payload['examples'].coordinates,
                        score,
                        selection_score,
                        payload['examples'].baseline_values,
                    )
                    active_keep_score = predict_support_score(active_keep_models[(model_name, str(score_mode))], active_keep_features)
                    combined_score = np.asarray(selection_score, dtype=np.float32).copy()
                    active_mask = np.asarray(payload['examples'].baseline_values, dtype=np.float32).reshape(-1) > EPS
                    combined_score[active_mask] = active_keep_score[active_mask]
                    selection_score = combined_score.astype(np.float32)
                candidate_scores.append((f'{model_name}__{score_mode}', model_name, str(score_mode), selection_score))
        candidate_scores.append(('diagnostic_only__oracle_support', 'diagnostic_only', 'oracle_support', payload['oracle_score']))

        for score_name, model_name, score_mode, score in candidate_scores:
            for top_k in args.top_k:
                for group_mode in args.selection_group_modes:
                    thresholds = args.support_thresholds if str(group_mode) in {'support_threshold', 'baseline_active_plus_new_topk_threshold'} else [float('nan')]
                    for threshold in thresholds:
                        threshold_tag = '' if np.isnan(float(threshold)) else f'__thr{float(threshold):g}'
                        selected = select_support_indices(
                            payload['examples'].coordinates,
                            score,
                            int(top_k),
                            str(group_mode),
                            baseline_value=payload['examples'].baseline_values,
                            threshold=float(threshold),
                        )
                        selection_diag = support_selection_diagnostics(
                            payload['examples'].coordinates,
                            payload['oracle_score'],
                            payload['labels'],
                            selected,
                        )
                        for repair_value_source in args.repair_value_sources:
                            use_step_reconstruction = False
                            candidate_family = candidate_family_for_score_and_value_source(
                                score_name,
                                score_mode,
                                str(repair_value_source),
                            )
                            value_support_score = payload['support_scores'].get(model_name, score)
                            if str(repair_value_source) in {'candidate_value', 'positive_candidate_value'}:
                                model_key = (
                                    str(repair_value_source),
                                    model_name,
                                    score_mode,
                                    str(group_mode),
                                    None if np.isnan(float(threshold)) else float(threshold),
                                    int(top_k),
                                )
                                if model_key not in candidate_value_model_cache:
                                    train_support_score = train_support_scores.get(model_name, train_score)
                                    train_selection_score = _selection_score(
                                        str(score_mode),
                                        train_support_score,
                                        train_examples.baseline_values,
                                        train_pred_value,
                                        train_score,
                                        float(args.risk_weight),
                                        edge_features=train_features,
                                    )
                                    train_candidate_features = make_candidate_value_features(
                                        train_features,
                                        train_examples.coordinates,
                                        train_support_score,
                                        train_selection_score,
                                        train_examples.baseline_values,
                                    )
                                    if str(repair_value_source) == 'positive_candidate_value':
                                        train_selected = np.flatnonzero(
                                            (train_labels > 0) | (train_value >= float(args.min_effective_rb_total))
                                        ).astype(np.int64)
                                    else:
                                        train_selected = select_support_indices(
                                            train_examples.coordinates,
                                            train_selection_score,
                                            int(top_k),
                                            str(group_mode),
                                            baseline_value=train_examples.baseline_values,
                                            threshold=float(threshold),
                                        )
                                    candidate_value_model_cache[model_key] = fit_candidate_value_regressor(
                                        train_candidate_features,
                                        train_value,
                                        train_labels,
                                        train_selected,
                                        min_effective_value=float(args.min_effective_rb_total),
                                        positive_weight=float(args.candidate_value_positive_weight),
                                        seed=int(args.seed) + 503,
                                    )
                                candidate_features = make_candidate_value_features(
                                    payload['features'],
                                    payload['examples'].coordinates,
                                    value_support_score,
                                    score,
                                    payload['examples'].baseline_values,
                                )
                                repair_values = np.clip(
                                    np.asarray(candidate_value_model_cache[model_key].predict(candidate_features), dtype=np.float32).reshape(-1),
                                    0.0,
                                    None,
                                )
                            elif (
                                str(repair_value_source).startswith('candidate_bin_')
                                or str(repair_value_source).startswith('positive_candidate_bin_')
                                or str(repair_value_source).startswith('all_candidate_bin_')
                            ):
                                source_text = str(repair_value_source)
                                decode_mode = (
                                    source_text
                                    .removeprefix('positive_candidate_bin_')
                                    .removeprefix('all_candidate_bin_')
                                    .removeprefix('candidate_bin_')
                                )
                                model_key = (
                                    source_text,
                                    model_name,
                                    score_mode,
                                    str(group_mode),
                                    None if np.isnan(float(threshold)) else float(threshold),
                                    int(top_k),
                                    int(args.candidate_value_bin_count),
                                )
                                if model_key not in candidate_value_model_cache:
                                    train_support_score = train_support_scores.get(model_name, train_score)
                                    train_selection_score = _selection_score(
                                        str(score_mode),
                                        train_support_score,
                                        train_examples.baseline_values,
                                        train_pred_value,
                                        train_score,
                                        float(args.risk_weight),
                                        edge_features=train_features,
                                    )
                                    train_candidate_features = make_candidate_value_features(
                                        train_features,
                                        train_examples.coordinates,
                                        train_support_score,
                                        train_selection_score,
                                        train_examples.baseline_values,
                                    )
                                    if source_text.startswith('positive_candidate_bin_'):
                                        train_selected = np.flatnonzero(
                                            (train_labels > 0) | (train_value >= float(args.min_effective_rb_total))
                                        ).astype(np.int64)
                                    elif source_text.startswith('all_candidate_bin_'):
                                        train_selected = np.arange(train_value.shape[0], dtype=np.int64)
                                    else:
                                        train_selected = select_support_indices(
                                            train_examples.coordinates,
                                            train_selection_score,
                                            int(top_k),
                                            str(group_mode),
                                            baseline_value=train_examples.baseline_values,
                                            threshold=float(threshold),
                                        )
                                    candidate_value_model_cache[model_key] = fit_candidate_value_bin_classifier(
                                        train_candidate_features,
                                        train_value,
                                        train_labels,
                                        train_selected,
                                        min_effective_value=float(args.min_effective_rb_total),
                                        positive_weight=float(args.candidate_value_bin_positive_weight),
                                        positive_bin_count=int(args.candidate_value_bin_count),
                                        seed=int(args.seed) + 701,
                                    )
                                candidate_features = make_candidate_value_features(
                                    payload['features'],
                                    payload['examples'].coordinates,
                                    value_support_score,
                                    score,
                                    payload['examples'].baseline_values,
                                )
                                repair_values = np.clip(
                                    decode_candidate_value_bins(candidate_value_model_cache[model_key], candidate_features, decode_mode),
                                    0.0,
                                    None,
                                )
                            elif str(repair_value_source).startswith('selected_gate_'):
                                value_lookup_key, gate_threshold = parse_selected_gate_source(str(repair_value_source))
                                if value_lookup_key not in repair_value_lookup:
                                    raise ValueError(f'missing value lookup for repair source: {value_lookup_key}')
                                model_key = (
                                    str(repair_value_source),
                                    model_name,
                                    score_mode,
                                    str(group_mode),
                                    None if np.isnan(float(threshold)) else float(threshold),
                                    int(top_k),
                                )
                                if model_key not in candidate_value_model_cache:
                                    train_support_score = train_support_scores.get(model_name, train_score)
                                    train_selection_score = _selection_score(
                                        str(score_mode),
                                        train_support_score,
                                        train_examples.baseline_values,
                                        train_pred_value,
                                        train_score,
                                        float(args.risk_weight),
                                        edge_features=train_features,
                                    )
                                    train_candidate_features = make_candidate_value_features(
                                        train_features,
                                        train_examples.coordinates,
                                        train_support_score,
                                        train_selection_score,
                                        train_examples.baseline_values,
                                    )
                                    train_selected = select_support_indices(
                                        train_examples.coordinates,
                                        train_selection_score,
                                        int(top_k),
                                        str(group_mode),
                                        baseline_value=train_examples.baseline_values,
                                        threshold=float(threshold),
                                    )
                                    candidate_value_model_cache[model_key] = fit_selected_candidate_gate(
                                        train_candidate_features,
                                        train_value,
                                        train_labels,
                                        train_selected,
                                        min_effective_value=float(args.min_effective_rb_total),
                                        seed=int(args.seed) + 907,
                                    )
                                candidate_features = make_candidate_value_features(
                                    payload['features'],
                                    payload['examples'].coordinates,
                                    value_support_score,
                                    score,
                                    payload['examples'].baseline_values,
                                )
                                gate_probability = predict_support_score(candidate_value_model_cache[model_key], candidate_features)
                                repair_values = np.where(
                                    gate_probability >= float(gate_threshold),
                                    float(repair_value_lookup[value_lookup_key]),
                                    0.0,
                                ).astype(np.float32)
                            elif str(repair_value_source) in {'structured_stepwise_pred_reconstruct', 'structured_branch_value_pred_reconstruct'}:
                                repair_values = payload['pred_value']
                                branch_value_model = None
                                if str(repair_value_source) == 'structured_branch_value_pred_reconstruct':
                                    branch_model_key = (
                                        'structured_branch_value',
                                        model_name,
                                        score_mode,
                                        str(group_mode),
                                        int(top_k),
                                        float(threshold) if not np.isnan(float(threshold)) else None,
                                    )
                                    if branch_model_key not in candidate_value_model_cache:
                                        train_support_score = train_support_scores.get(model_name, train_score)
                                        train_selection_score = _selection_score(
                                            str(score_mode),
                                            train_support_score,
                                            train_examples.baseline_values,
                                            train_pred_value,
                                            train_score,
                                            float(args.risk_weight),
                                            edge_features=train_features,
                                        )
                                        train_primary_selected = select_support_indices(
                                            train_examples.coordinates,
                                            train_selection_score,
                                            int(top_k),
                                            str(group_mode),
                                            baseline_value=train_examples.baseline_values,
                                            threshold=float(threshold),
                                        )
                                        train_step_keys = make_action_step_keys(train_actions)
                                        train_branch_features = []
                                        train_branch_targets = []
                                        for _branch_name, train_branch_selected in structured_branch_specs(
                                            train_examples.coordinates,
                                            train_selection_score,
                                            train_examples.baseline_values,
                                            train_primary_selected,
                                        ):
                                            train_step_features = make_branch_step_features(
                                                train_step_keys,
                                                train_examples.coordinates,
                                                train_branch_selected,
                                                train_selection_score,
                                                train_pred_value,
                                                train_examples.baseline_values,
                                            )
                                            train_step_targets = make_branch_step_targets(
                                                train_step_keys,
                                                train_examples.coordinates,
                                                train_branch_selected,
                                                train_value,
                                                train_labels,
                                                train_examples.baseline_values,
                                            )
                                            train_branch_features.append(train_step_features)
                                            train_branch_targets.append(train_step_targets)
                                        candidate_value_model_cache[branch_model_key] = fit_branch_value_regressor(
                                            np.concatenate(train_branch_features, axis=0),
                                            np.concatenate(train_branch_targets, axis=0),
                                            seed=int(args.seed) + 1901,
                                        )
                                    branch_value_model = candidate_value_model_cache[branch_model_key]
                            elif str(repair_value_source).startswith('mass_preserve_'):
                                use_step_reconstruction = False
                                mass_source_kind, mass_allocation_mode, mass_beta = parse_mass_preserve_source(str(repair_value_source))
                                if mass_source_kind == 'true':
                                    mass_score = payload['true_value']
                                elif mass_allocation_mode == 'rate_score':
                                    rate_score = np.mean(np.clip(payload['features'][:, -3:], 0.0, None), axis=1)
                                    mass_score = (
                                        np.clip(score, 0.0, None)
                                        * np.log1p(np.clip(rate_score, 0.0, None))
                                    ).astype(np.float32)
                                elif mass_allocation_mode == 'score':
                                    mass_score = score
                                else:
                                    raise ValueError(f'unknown mass-preserving allocation mode in source: {repair_value_source}')
                                repair_values = payload['pred_value']
                            elif str(repair_value_source).startswith('step_budget_'):
                                use_step_reconstruction = True
                                source_text = str(repair_value_source)
                                gated_step_threshold = None
                                step_gate_threshold = None
                                reconstruct_gate_threshold = None
                                soft_beta = None
                                step_gate_by_row = None
                                filter_step_budget = False
                                rank_step_budget = False
                                new_quota_step_budget = False
                                oracle_allocation_score = False
                                budget_source_kind = 'pred'
                                if source_text.startswith('step_budget_gated_pred_'):
                                    allocation_mode, gated_step_threshold = parse_step_budget_gated_source(source_text)
                                elif source_text.startswith('step_budget_stepgate_'):
                                    budget_source_kind, allocation_mode, step_gate_threshold = parse_step_budget_step_gate_source(source_text)
                                elif source_text.startswith('step_budget_reconstruct_gate_'):
                                    budget_source_kind, allocation_mode, reconstruct_gate_threshold = parse_step_budget_reconstruct_gate_source(source_text)
                                elif source_text.startswith('step_budget_soft_'):
                                    budget_source_kind, allocation_mode, soft_beta = parse_step_budget_soft_source(source_text)
                                elif source_text.startswith('step_budget_filter_'):
                                    budget_source_kind, allocation_mode = parse_step_budget_filter_source(source_text)
                                    filter_step_budget = True
                                elif source_text.startswith('step_budget_ranker_'):
                                    budget_source_kind, allocation_mode = parse_step_budget_filter_source(source_text.replace('step_budget_ranker_', 'step_budget_filter_', 1))
                                    rank_step_budget = True
                                elif source_text.startswith('step_budget_newq_'):
                                    budget_source_kind, allocation_mode = parse_step_budget_filter_source(source_text.replace('step_budget_newq_', 'step_budget_filter_', 1))
                                    allocation_mode = 'new_quota_score'
                                    new_quota_step_budget = True
                                elif source_text.startswith('step_budget_oracle_alloc_pred_'):
                                    allocation_mode = source_text.removeprefix('step_budget_oracle_alloc_pred_')
                                    oracle_allocation_score = True
                                elif source_text.startswith('step_budget_pred_'):
                                    allocation_mode = source_text.removeprefix('step_budget_pred_')
                                elif source_text.startswith('step_budget_true_'):
                                    allocation_mode = source_text.removeprefix('step_budget_true_')
                                    budget_source_kind = 'true'
                                else:
                                    raise ValueError(f'unknown step budget source: {source_text}')
                                if allocation_mode not in {'score', 'uniform', 'baseline_score', 'baseline_value_score', 'new_quota_score', 'rate_score'}:
                                    raise ValueError(f'unknown step budget allocation mode in source: {source_text}')
                                train_support_score = train_support_scores.get(model_name, train_score)
                                train_selection_score = _selection_score(
                                    str(score_mode),
                                    train_support_score,
                                    train_examples.baseline_values,
                                    train_pred_value,
                                    train_score,
                                    float(args.risk_weight),
                                    edge_features=train_features,
                                )
                                train_step_features = make_step_budget_features(
                                    train_features,
                                    train_examples.coordinates,
                                    train_support_score,
                                    train_selection_score,
                                    train_examples.baseline_values,
                                    train_pred_value,
                                )
                                train_step_targets = make_step_budget_targets(
                                    train_step_features.keys,
                                    train_examples.coordinates,
                                    train_value,
                                    min_effective_value=float(args.min_effective_rb_total),
                                    baseline_value=train_examples.baseline_values,
                                )
                                step_features = make_step_budget_features(
                                    payload['features'],
                                    payload['examples'].coordinates,
                                    value_support_score,
                                    score,
                                    payload['examples'].baseline_values,
                                    payload['pred_value'],
                                )
                                step_targets = make_step_budget_targets(
                                    step_features.keys,
                                    payload['examples'].coordinates,
                                    payload['true_value'],
                                    min_effective_value=float(args.min_effective_rb_total),
                                    baseline_value=payload['examples'].baseline_values,
                                )
                                if budget_source_kind == 'true':
                                    total_by_row = map_step_predictions_to_rows(step_features.row_to_step, step_targets.total)
                                    count_by_row = map_step_predictions_to_rows(step_features.row_to_step, step_targets.count)
                                    new_count_by_row = map_step_predictions_to_rows(step_features.row_to_step, step_targets.new_count)
                                else:
                                    model_key = ('step_budget_pred', model_name, score_mode)
                                    if model_key not in step_budget_model_cache:
                                        step_budget_model_cache[model_key] = fit_step_budget_regressors(
                                            train_step_features.features,
                                            train_step_targets,
                                            seed=int(args.seed) + 1103,
                                        )
                                    total_model, count_model = step_budget_model_cache[model_key]
                                    step_total = np.clip(np.asarray(total_model.predict(step_features.features), dtype=np.float32).reshape(-1), 0.0, None)
                                    step_count = np.clip(np.asarray(count_model.predict(step_features.features), dtype=np.float32).reshape(-1), 0.0, None)
                                    total_by_row = map_step_predictions_to_rows(step_features.row_to_step, step_total)
                                    count_by_row = map_step_predictions_to_rows(step_features.row_to_step, step_count)
                                    if new_quota_step_budget:
                                        new_count_model_key = ('step_budget_new_count_pred', model_name, score_mode)
                                        if new_count_model_key not in step_budget_model_cache:
                                            step_budget_model_cache[new_count_model_key] = fit_step_budget_new_count_regressor(
                                                train_step_features.features,
                                                train_step_targets,
                                                seed=int(args.seed) + 1601,
                                            )
                                        new_count_model = step_budget_model_cache[new_count_model_key]
                                        step_new_count = np.clip(np.asarray(new_count_model.predict(step_features.features), dtype=np.float32).reshape(-1), 0.0, None)
                                        new_count_by_row = map_step_predictions_to_rows(step_features.row_to_step, step_new_count)
                                    else:
                                        new_count_by_row = np.zeros_like(count_by_row, dtype=np.float32)
                                if step_gate_threshold is not None:
                                    if budget_source_kind == 'true':
                                        step_gate = (step_targets.total > EPS).astype(np.float32)
                                    else:
                                        gate_model_key = ('step_gate_pred', model_name, score_mode)
                                        if gate_model_key not in step_budget_model_cache:
                                            step_budget_model_cache[gate_model_key] = fit_step_gate_classifier(
                                                train_step_features.features,
                                                train_step_targets,
                                                seed=int(args.seed) + 1423,
                                            )
                                        step_gate_model = step_budget_model_cache[gate_model_key]
                                        step_gate = predict_support_score(step_gate_model, step_features.features)
                                    step_gate_by_row = map_step_predictions_to_rows(step_features.row_to_step, step_gate)
                                allocation_score = score
                                if oracle_allocation_score:
                                    allocation_score = payload['true_value']
                                if filter_step_budget:
                                    train_step_total = map_step_predictions_to_rows(train_step_features.row_to_step, train_step_targets.total)
                                    train_step_count = map_step_predictions_to_rows(train_step_features.row_to_step, train_step_targets.count)
                                    if budget_source_kind == 'pred':
                                        train_model_key = ('step_budget_pred', model_name, score_mode)
                                        if train_model_key not in step_budget_model_cache:
                                            step_budget_model_cache[train_model_key] = fit_step_budget_regressors(
                                                train_step_features.features,
                                                train_step_targets,
                                                seed=int(args.seed) + 1103,
                                            )
                                        train_total_model, train_count_model = step_budget_model_cache[train_model_key]
                                        train_step_total_pred = np.clip(
                                            np.asarray(train_total_model.predict(train_step_features.features), dtype=np.float32).reshape(-1),
                                            0.0,
                                            None,
                                        )
                                        train_step_count_pred = np.clip(
                                            np.asarray(train_count_model.predict(train_step_features.features), dtype=np.float32).reshape(-1),
                                            0.0,
                                            None,
                                        )
                                        train_step_total = map_step_predictions_to_rows(train_step_features.row_to_step, train_step_total_pred)
                                        train_step_count = map_step_predictions_to_rows(train_step_features.row_to_step, train_step_count_pred)
                                    train_filter_features = make_allocation_filter_features(
                                        train_features,
                                        train_examples.coordinates,
                                        train_support_score,
                                        train_selection_score,
                                        train_examples.baseline_values,
                                        train_pred_value,
                                        train_step_total,
                                        train_step_count,
                                    )
                                    filter_model_key = ('step_budget_filter', budget_source_kind, model_name, score_mode, str(group_mode), int(top_k), float(threshold) if not np.isnan(float(threshold)) else None)
                                    if filter_model_key not in candidate_value_model_cache:
                                        train_selected_for_filter = select_support_indices(
                                            train_examples.coordinates,
                                            train_selection_score,
                                            int(top_k),
                                            str(group_mode),
                                            baseline_value=train_examples.baseline_values,
                                            threshold=float(threshold),
                                        )
                                        candidate_value_model_cache[filter_model_key] = fit_allocation_filter(
                                            train_filter_features,
                                            train_value,
                                            train_labels,
                                            train_selected_for_filter,
                                            min_effective_value=float(args.min_effective_rb_total),
                                            seed=int(args.seed) + 1307,
                                        )
                                    filter_features = make_allocation_filter_features(
                                        payload['features'],
                                        payload['examples'].coordinates,
                                        value_support_score,
                                        score,
                                        payload['examples'].baseline_values,
                                        payload['pred_value'],
                                        total_by_row,
                                        count_by_row,
                                    )
                                    allocation_score = predict_support_score(candidate_value_model_cache[filter_model_key], filter_features)
                                if rank_step_budget:
                                    train_step_total = map_step_predictions_to_rows(train_step_features.row_to_step, train_step_targets.total)
                                    train_step_count = map_step_predictions_to_rows(train_step_features.row_to_step, train_step_targets.count)
                                    if budget_source_kind == 'pred':
                                        train_model_key = ('step_budget_pred', model_name, score_mode)
                                        if train_model_key not in step_budget_model_cache:
                                            step_budget_model_cache[train_model_key] = fit_step_budget_regressors(
                                                train_step_features.features,
                                                train_step_targets,
                                                seed=int(args.seed) + 1103,
                                            )
                                        train_total_model, train_count_model = step_budget_model_cache[train_model_key]
                                        train_step_total_pred = np.clip(
                                            np.asarray(train_total_model.predict(train_step_features.features), dtype=np.float32).reshape(-1),
                                            0.0,
                                            None,
                                        )
                                        train_step_count_pred = np.clip(
                                            np.asarray(train_count_model.predict(train_step_features.features), dtype=np.float32).reshape(-1),
                                            0.0,
                                            None,
                                        )
                                        train_step_total = map_step_predictions_to_rows(train_step_features.row_to_step, train_step_total_pred)
                                        train_step_count = map_step_predictions_to_rows(train_step_features.row_to_step, train_step_count_pred)
                                    train_rank_features = make_allocation_filter_features(
                                        train_features,
                                        train_examples.coordinates,
                                        train_support_score,
                                        train_selection_score,
                                        train_examples.baseline_values,
                                        train_pred_value,
                                        train_step_total,
                                        train_step_count,
                                    )
                                    ranker_model_key = ('step_budget_ranker', budget_source_kind, model_name, score_mode, str(group_mode), int(top_k), float(threshold) if not np.isnan(float(threshold)) else None)
                                    if ranker_model_key not in candidate_value_model_cache:
                                        train_selected_for_ranker = select_support_indices(
                                            train_examples.coordinates,
                                            train_selection_score,
                                            int(top_k),
                                            str(group_mode),
                                            baseline_value=train_examples.baseline_values,
                                            threshold=float(threshold),
                                        )
                                        candidate_value_model_cache[ranker_model_key] = fit_allocation_ranker(
                                            train_rank_features,
                                            train_examples.coordinates,
                                            train_value,
                                            train_labels,
                                            train_selected_for_ranker,
                                            min_effective_value=float(args.min_effective_rb_total),
                                            seed=int(args.seed) + 1777,
                                        )
                                    rank_features = make_allocation_filter_features(
                                        payload['features'],
                                        payload['examples'].coordinates,
                                        value_support_score,
                                        score,
                                        payload['examples'].baseline_values,
                                        payload['pred_value'],
                                        total_by_row,
                                        count_by_row,
                                    )
                                    allocation_score = np.clip(
                                        np.asarray(candidate_value_model_cache[ranker_model_key].predict(rank_features), dtype=np.float32).reshape(-1),
                                        0.0,
                                        None,
                                    )
                                selected, repair_values = allocate_step_budget_values(
                                    payload['examples'].coordinates,
                                    selected,
                                    allocation_score,
                                    total_by_row,
                                    count_by_row,
                                    allocation_mode=allocation_mode,
                                    baseline_value=payload['examples'].baseline_values,
                                    new_count_by_row=new_count_by_row,
                                    rate_score=np.mean(np.clip(payload['features'][:, -3:], 0.0, None), axis=1),
                                )
                                if gated_step_threshold is not None:
                                    step_confidence = selected_step_confidence(
                                        payload['examples'].coordinates,
                                        selected,
                                        value_support_score,
                                    )
                            else:
                                repair_values = repair_values_for_source(
                                    payload['pred_value'],
                                    payload['true_value'],
                                    str(repair_value_source),
                                    repair_value_lookup,
                                    value_support_score,
                                )
                            for alpha in args.blend_alpha:
                                for cap_scale in args.step_total_cap_scale:
                                    for edge_cap in args.edge_value_cap_scale:
                                        for new_edge_cap in args.new_edge_value_cap:
                                            value_tag = '' if str(repair_value_source) == 'pred_value' else f'__vsrc{repair_value_source}'
                                            candidate = f'{score_name}__g{group_mode}{threshold_tag}__top{int(top_k)}{value_tag}__alpha{float(alpha):g}__cap{float(cap_scale):g}__ecap{float(edge_cap):g}__newcap{float(new_edge_cap):g}'
                                            if candidate_family == 'diagnostic_only' and not candidate.startswith('diagnostic_only'):
                                                candidate = f'diagnostic_only__{candidate}'
                                            if use_step_reconstruction and str(repair_value_source).startswith('step_budget_stepgate_'):
                                                fallback_actions = apply_support_generator_repair(
                                                    payload['baseline_actions'],
                                                    payload['examples'].coordinates,
                                                    payload['pred_value'],
                                                    selected,
                                                    alpha=float(alpha),
                                                    step_total_cap_scale=float(cap_scale),
                                                    edge_value_cap_scale=float(edge_cap),
                                                    new_edge_value_cap=float(new_edge_cap),
                                                )
                                                actions = apply_step_budget_step_gate_reconstruction(
                                                    fallback_actions,
                                                    payload['examples'].coordinates,
                                                    repair_values,
                                                    selected,
                                                    step_gate_by_row,
                                                    threshold=float(step_gate_threshold),
                                                )
                                            elif use_step_reconstruction and str(repair_value_source).startswith('step_budget_gated_pred_'):
                                                fallback_actions = apply_support_generator_repair(
                                                    payload['baseline_actions'],
                                                    payload['examples'].coordinates,
                                                    payload['pred_value'],
                                                    selected,
                                                    alpha=float(alpha),
                                                    step_total_cap_scale=float(cap_scale),
                                                    edge_value_cap_scale=float(edge_cap),
                                                    new_edge_value_cap=float(new_edge_cap),
                                                )
                                                actions = apply_gated_step_budget_reconstruction(
                                                    fallback_actions,
                                                    payload['examples'].coordinates,
                                                    repair_values,
                                                    selected,
                                                    step_confidence,
                                                    threshold=float(gated_step_threshold),
                                                )
                                            elif use_step_reconstruction and str(repair_value_source).startswith('step_budget_reconstruct_gate_'):
                                                fallback_actions = apply_support_generator_reconstruction(
                                                    payload['baseline_actions'],
                                                    payload['examples'].coordinates,
                                                    payload['pred_value'],
                                                    selected,
                                                    alpha=float(alpha),
                                                    step_total_cap_scale=float(cap_scale),
                                                    edge_value_cap_scale=float(edge_cap),
                                                    new_edge_value_cap=float(new_edge_cap),
                                                )
                                                step_confidence = selected_step_confidence(
                                                    payload['examples'].coordinates,
                                                    selected,
                                                    value_support_score,
                                                )
                                                actions = apply_gated_step_budget_reconstruction(
                                                    fallback_actions,
                                                    payload['examples'].coordinates,
                                                    repair_values,
                                                    selected,
                                                    step_confidence,
                                                    threshold=float(reconstruct_gate_threshold),
                                                )
                                            elif use_step_reconstruction and str(repair_value_source).startswith('step_budget_soft_'):
                                                fallback_actions = apply_support_generator_reconstruction(
                                                    payload['baseline_actions'],
                                                    payload['examples'].coordinates,
                                                    payload['pred_value'],
                                                    selected,
                                                    alpha=float(alpha),
                                                    step_total_cap_scale=float(cap_scale),
                                                    edge_value_cap_scale=float(edge_cap),
                                                    new_edge_value_cap=float(new_edge_cap),
                                                )
                                                actions = apply_soft_step_budget_repair(
                                                    fallback_actions,
                                                    payload['examples'].coordinates,
                                                    repair_values,
                                                    selected,
                                                    beta=float(soft_beta),
                                                )
                                            elif str(repair_value_source).startswith('mass_preserve_'):
                                                fallback_actions = apply_support_generator_reconstruction(
                                                    payload['baseline_actions'],
                                                    payload['examples'].coordinates,
                                                    payload['pred_value'],
                                                    selected,
                                                    alpha=float(alpha),
                                                    step_total_cap_scale=float(cap_scale),
                                                    edge_value_cap_scale=float(edge_cap),
                                                    new_edge_value_cap=float(new_edge_cap),
                                                )
                                                actions = apply_mass_preserving_reallocation(
                                                    fallback_actions,
                                                    payload['examples'].coordinates,
                                                    mass_score,
                                                    selected,
                                                    beta=float(mass_beta),
                                                )
                                            elif use_step_reconstruction:
                                                actions = apply_step_budget_reconstruction(
                                                    payload['baseline_actions'],
                                                    payload['examples'].coordinates,
                                                    repair_values,
                                                    selected,
                                                )
                                            elif str(repair_value_source) == 'pred_value_reconstruct':
                                                actions = apply_support_generator_reconstruction(
                                                    payload['baseline_actions'],
                                                    payload['examples'].coordinates,
                                                    repair_values,
                                                    selected,
                                                    alpha=float(alpha),
                                                    step_total_cap_scale=float(cap_scale),
                                                    edge_value_cap_scale=float(edge_cap),
                                                    new_edge_value_cap=float(new_edge_cap),
                                                )
                                            elif str(repair_value_source) in {'structured_stepwise_pred_reconstruct', 'structured_branch_value_pred_reconstruct'}:
                                                structured_specs = structured_branch_specs(
                                                    payload['examples'].coordinates,
                                                    score,
                                                    payload['examples'].baseline_values,
                                                    selected,
                                                )
                                                structured_candidates = []
                                                payload_step_keys = make_action_step_keys(payload['baseline_actions'])
                                                for structured_name, structured_selected in structured_specs:
                                                    structured_actions = apply_support_generator_reconstruction(
                                                        payload['baseline_actions'],
                                                        payload['examples'].coordinates,
                                                        repair_values,
                                                        structured_selected,
                                                        alpha=float(alpha),
                                                        step_total_cap_scale=float(cap_scale),
                                                        edge_value_cap_scale=float(edge_cap),
                                                        new_edge_value_cap=float(new_edge_cap),
                                                    )
                                                    if str(repair_value_source) == 'structured_branch_value_pred_reconstruct':
                                                        branch_features = make_branch_step_features(
                                                            payload_step_keys,
                                                            payload['examples'].coordinates,
                                                            structured_selected,
                                                            score,
                                                            payload['pred_value'],
                                                            payload['examples'].baseline_values,
                                                        )
                                                        branch_predictions = np.clip(
                                                            np.asarray(branch_value_model.predict(branch_features), dtype=np.float32).reshape(-1),
                                                            0.0,
                                                            None,
                                                        )
                                                        structured_scores = branch_predictions.reshape(payload['baseline_actions'].shape[:2]).astype(np.float32)
                                                    else:
                                                        structured_scores = make_structured_step_score_matrix(
                                                            structured_actions,
                                                            payload['examples'].coordinates,
                                                            structured_selected,
                                                            score,
                                                            payload['pred_value'],
                                                            payload['examples'].baseline_values,
                                                            new_edge_penalty=float(new_edge_cap),
                                                            selected_count_penalty=0.08,
                                                        )
                                                    structured_candidates.append(
                                                        StructuredActionCandidate(
                                                            name=structured_name,
                                                            actions=structured_actions,
                                                            step_scores=structured_scores,
                                                        )
                                                    )
                                                actions, structured_counts = compose_structured_stepwise_actions(
                                                    payload['baseline_actions'],
                                                    structured_candidates,
                                                )
                                            else:
                                                actions = apply_support_generator_repair(
                                                    payload['baseline_actions'],
                                                    payload['examples'].coordinates,
                                                    repair_values,
                                                    selected,
                                                    alpha=float(alpha),
                                                    step_total_cap_scale=float(cap_scale),
                                                    edge_value_cap_scale=float(edge_cap),
                                                    new_edge_value_cap=float(new_edge_cap),
                                                )
                                            predictions = evaluate_raw_actions(actions, payload['base_dataset'], stats, world_model, summary['config'], device, args.batch_size)
                                            row = active_rate_row(candidate, split_name, predictions, baseline_rmse_by_split[split_name])
                                            new_support_count = int(np.sum((actions[..., RB_DIM] > EPS) & (payload['baseline_actions'][..., RB_DIM] <= EPS)))
                                            row.update(
                                                {
                                                    'family': candidate_family,
                                                    'support_model': model_name,
                                                    'selection_score_mode': score_mode,
                                                    'selection_group_mode': str(group_mode),
                                                    'support_threshold': None if np.isnan(float(threshold)) else float(threshold),
                                                    'repair_value_source': str(repair_value_source),
                                                    'top_k': int(top_k),
                                                    'alpha': float(alpha),
                                                    'step_total_cap_scale': float(cap_scale),
                                                    'edge_value_cap_scale': float(edge_cap),
                                                    'new_edge_value_cap': float(new_edge_cap),
                                                    'selected_count': int(selected.size),
                                                    'new_support_count': new_support_count,
                                                    **action_budget_diagnostics(
                                                        actions,
                                                        payload['baseline_actions'],
                                                        payload['truth_actions'],
                                                    ),
                                                    **selection_diag,
                                                }
                                            )
                                            if str(repair_value_source) == 'structured_stepwise_pred_reconstruct':
                                                row['structured_choice_counts_json'] = json.dumps(structured_counts, sort_keys=True)
                                            rows.append(row)

    write_csv(args.output_dir / 'graph_support_generator_results.csv', rows)
    val_ranked = sorted([row for row in rows if row['split'] == 'val'], key=lambda row: (float(row['active_rate_rmse']), str(row['candidate'])))
    deployable_val_ranked = [
        row for row in val_ranked
        if str(row.get('candidate')) != 'identity_bc_reference' and not str(row.get('candidate')).startswith('diagnostic_only')
    ]
    diagnostic_val_ranked = [row for row in val_ranked if str(row.get('candidate')).startswith('diagnostic_only')]
    write_csv(args.output_dir / 'graph_support_generator_val_ranked.csv', val_ranked)
    write_csv(args.output_dir / 'graph_support_generator_deployable_val_ranked.csv', deployable_val_ranked)
    test_by_candidate = {str(row['candidate']): row for row in rows if row['split'] == 'test'}
    best_val = deployable_val_ranked[0] if deployable_val_ranked else None
    best_diagnostic_val = diagnostic_val_ranked[0] if diagnostic_val_ranked else None
    diagnostics = {
        'train_all_edge_examples': int(train_examples.coordinates.shape[0]),
        'train_rows_used': int(train_rows.shape[0]),
        'support_train_rows_used_by_model': support_train_rows_used,
        'support_score_train_corr_by_model': support_score_train_corr,
        'active_keep_train_corr_by_model': active_keep_train_corr,
        'train_positive_label_count': int(np.sum(train_labels > 0)),
        'train_positive_value_count': int(np.sum(train_value >= float(args.min_effective_rb_total))),
        'repair_value_lookup': repair_value_lookup,
        'support_model_kinds': list(args.support_model_kinds),
        'support_training_strategies': list(args.support_training_strategies),
        'rank_target_mode': str(args.rank_target_mode),
        'value_training_mode': str(args.value_training_mode),
        'value_positive_weight': float(args.value_positive_weight),
        'candidate_value_positive_weight': float(args.candidate_value_positive_weight),
        'candidate_value_bin_positive_weight': float(args.candidate_value_bin_positive_weight),
        'candidate_value_bin_count': int(args.candidate_value_bin_count),
        'pairwise_max_pairs': int(args.pairwise_max_pairs),
        'pairwise_negatives_per_positive': int(args.pairwise_negatives_per_positive),
        'selection_score_modes': selection_score_modes,
        'selection_group_modes': list(args.selection_group_modes),
        'train_sample_start': int(args.train_sample_start),
        'val_sample_start': int(args.val_sample_start),
        'test_sample_start': int(args.test_sample_start),
        'val_shuffle_seed': int(args.val_shuffle_seed),
        'test_shuffle_seed': int(args.test_shuffle_seed),
        'val_support_label_count': int(np.sum(split_payload['val']['labels'] > 0)),
        'test_support_label_count': int(np.sum(split_payload['test']['labels'] > 0)),
        'val_value_mae_nonzero_true': float(mean_absolute_error(split_payload['val']['true_value'][split_payload['val']['true_value'] >= float(args.min_effective_rb_total)], split_payload['val']['pred_value'][split_payload['val']['true_value'] >= float(args.min_effective_rb_total)])) if np.any(split_payload['val']['true_value'] >= float(args.min_effective_rb_total)) else float('nan'),
        'test_value_mae_nonzero_true': float(mean_absolute_error(split_payload['test']['true_value'][split_payload['test']['true_value'] >= float(args.min_effective_rb_total)], split_payload['test']['pred_value'][split_payload['test']['true_value'] >= float(args.min_effective_rb_total)])) if np.any(split_payload['test']['true_value'] >= float(args.min_effective_rb_total)) else float('nan'),
        'val_first_support_score_corr': safe_corr(next(iter(split_payload['val']['support_scores'].values())), split_payload['val']['labels']) if split_payload['val']['support_scores'] else float('nan'),
        'test_first_support_score_corr': safe_corr(next(iter(split_payload['test']['support_scores'].values())), split_payload['test']['labels']) if split_payload['test']['support_scores'] else float('nan'),
        'runtime_seconds': float(time.time() - started),
    }
    result = {
        'framework': 'PI-JWM',
        'candidate': 'v11',
        'mode': 'graph_support_generator_cpu',
        'output_dir': str(args.output_dir),
        'command': ' '.join(sys.argv),
        'device': str(device),
        'diagnostics': diagnostics,
        'best_val': best_val,
        'matched_test_for_best_val': test_by_candidate.get(str(best_val['candidate'])) if best_val else None,
        'best_diagnostic_val': best_diagnostic_val,
        'matched_test_for_best_diagnostic_val': test_by_candidate.get(str(best_diagnostic_val['candidate'])) if best_diagnostic_val else None,
    }
    write_json(args.output_dir / 'summary.json', result)
    return result


def main() -> None:
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
