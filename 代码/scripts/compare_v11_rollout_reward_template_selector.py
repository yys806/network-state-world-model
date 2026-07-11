'''Actual-rollout reward template selector for PI-JWM v11 candidate.

This CPU-first probe builds a small deployable action-template library, scores
each template with the frozen PI-JWM world model, and trains a shallow
sample-level selector from actual rollout errors.  The point is to test whether
per-sample template choice has real rollout headroom before any GPU expansion.
'''

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))

from pi_jwm.v11_rollout_value_calibrator import freeze_module
from pi_jwm.evaluation.candidate_selection import (
    choice_rmse_from_sample_sse,
    choose_best_single_by_sample_sse,
    mix_actions_by_sample,
    sample_active_sse,
    sample_rmse_from_sse,
)
from pi_jwm.evaluation.result_protocol import build_result_protocol

from compare_v11_base_policy_candidates import (
    _apply_candidate_actions,
    _fit_models,
    advantage_weighted_scores,
    expectile_baseline,
)
from diagnose_v11_counterfactual_value_attribution import collect_raw_actions
from diagnose_v11_rb_total_latent_identifiability import (
    collect_rollout_edge_context,
    invert_value_target,
    make_targets,
    rows_from_context,
)
from diagnose_v11_rb_total_oracle_value_scope import select_topk_indices, write_json
from diagnose_v11_scheduler_ranked_allocation import (
    link_aware_selection_score,
    predict_conservative_value_target,
    resolve_torch_device,
)
from run_v11_rb_total_repair import active_rate_row, evaluate_raw_actions, make_adaptive_dataset, write_csv
from run_v11_rb_total_value_head import (
    RB_DIM,
    _make_inference_examples,
    collect_edge_gradient_improvement,
    limit_indices,
    load_context_limited,
    make_critical_examples,
)


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / 'artifacts/experiments/pi_jwm_v11_rollout_reward_template_selector_20260629'
EPS = 1e-9


@dataclass(frozen=True)
class SplitPayload:
    base_dataset: object
    baseline_actions: np.ndarray
    coordinates: np.ndarray
    baseline_values: np.ndarray
    pred_score: np.ndarray
    pred_value: np.ndarray


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    family: str
    selection_rule: str
    top_k: int
    alpha: float
    step_total_cap_scale: float
    edge_value_cap_scale: float


def make_selector_result_protocol(candidate_name: str, pairwise_fit_split: str) -> dict[str, object]:
    uses_pairwise_fit = str(candidate_name).startswith('rollout_reward_pairwise_')
    fit_splits = ('train', 'val') if uses_pairwise_fit and str(pairwise_fit_split) == 'train_val' else ('train',)
    result_kind = 'test_best_diagnostic' if fit_splits == ('train', 'val') else 'deployable'
    return build_result_protocol(
        result_kind=result_kind,
        fit_splits=fit_splits,
        selection_split='val',
        evaluation_split='test',
    )


def make_sample_summary_features(actions: np.ndarray) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 4:
        raise ValueError('actions must have shape [sample, step, edge, dim]')
    rb = np.clip(actions[..., RB_DIM], 0.0, None)
    active = rb > EPS
    step_total = rb.sum(axis=2)
    step_count = active.sum(axis=2).astype(np.float32)
    features = [
        step_total,
        step_count,
        step_total.mean(axis=1, keepdims=True),
        step_total.max(axis=1, keepdims=True),
        step_count.mean(axis=1, keepdims=True),
        step_count.max(axis=1, keepdims=True),
    ]
    return np.concatenate(features, axis=1).astype(np.float32)


def make_forecast_selector_features(
    candidate_actions: list[np.ndarray],
    predictions_by_spec: list[dict[str, np.ndarray]],
) -> np.ndarray:
    if not candidate_actions or not predictions_by_spec:
        raise ValueError('candidate actions and predictions must not be empty')
    if len(candidate_actions) != len(predictions_by_spec):
        raise ValueError('candidate actions and predictions must have the same length')
    features = []
    base_rate = None
    base_prob = None
    def col(values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.float32).reshape(-1, 1)

    for actions, predictions in zip(candidate_actions, predictions_by_spec):
        rb = np.clip(np.asarray(actions, dtype=np.float32)[..., RB_DIM], 0.0, None)
        prob = np.asarray(predictions['link_activity_prob'], dtype=np.float32).squeeze(-1)
        rate = np.asarray(predictions['link_rate_pred'], dtype=np.float32).squeeze(-1)
        if base_rate is None:
            base_rate = rate
            base_prob = prob
        if rb.shape != rate.shape or prob.shape != rate.shape:
            raise ValueError('candidate action and prediction tensors must share [sample, step, edge] shape')
        predicted_active_mass = np.sum(prob * np.clip(rate, 0.0, None), axis=(1, 2), keepdims=False)
        parts = [
            col(np.sum(rb, axis=(1, 2))),
            col(np.max(rb, axis=(1, 2))),
            col(np.sum(rb > EPS, axis=(1, 2))),
            col(np.sum(prob, axis=(1, 2))),
            col(np.max(prob, axis=(1, 2))),
            col(np.mean(prob, axis=(1, 2))),
            col(np.sum(np.clip(rate, 0.0, None), axis=(1, 2))),
            col(np.max(np.clip(rate, 0.0, None), axis=(1, 2))),
            col(predicted_active_mass),
            col(np.mean(rate - base_rate, axis=(1, 2))),
            col(np.mean(prob - base_prob, axis=(1, 2))),
        ]
        features.append(np.concatenate(parts, axis=1))
    return np.concatenate(features, axis=1).astype(np.float32)


def make_candidate_error_features(
    candidate_actions: list[np.ndarray],
    predictions_by_spec: list[dict[str, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    if not candidate_actions or not predictions_by_spec:
        raise ValueError('candidate actions and predictions must not be empty')
    if len(candidate_actions) != len(predictions_by_spec):
        raise ValueError('candidate actions and predictions must have the same length')
    rows = []
    candidate_idx = []
    base_rate = np.asarray(predictions_by_spec[0]['link_rate_pred'], dtype=np.float32).squeeze(-1)
    base_prob = np.asarray(predictions_by_spec[0]['link_activity_prob'], dtype=np.float32).squeeze(-1)
    for idx, (actions, predictions) in enumerate(zip(candidate_actions, predictions_by_spec)):
        rb = np.clip(np.asarray(actions, dtype=np.float32)[..., RB_DIM], 0.0, None)
        prob = np.asarray(predictions['link_activity_prob'], dtype=np.float32).squeeze(-1)
        rate = np.asarray(predictions['link_rate_pred'], dtype=np.float32).squeeze(-1)
        if rb.shape != rate.shape or prob.shape != rate.shape:
            raise ValueError('candidate action and prediction tensors must share [sample, step, edge] shape')
        predicted_active_mass = np.sum(prob * np.clip(rate, 0.0, None), axis=(1, 2))
        features = np.stack(
            [
                np.sum(rb, axis=(1, 2)),
                np.max(rb, axis=(1, 2)),
                np.sum(rb > EPS, axis=(1, 2)).astype(np.float32),
                np.sum(prob, axis=(1, 2)),
                np.max(prob, axis=(1, 2)),
                np.mean(prob, axis=(1, 2)),
                np.sum(np.clip(rate, 0.0, None), axis=(1, 2)),
                np.max(np.clip(rate, 0.0, None), axis=(1, 2)),
                predicted_active_mass,
                np.mean(rate - base_rate, axis=(1, 2)),
                np.mean(prob - base_prob, axis=(1, 2)),
                np.full((rate.shape[0],), float(idx) / max(len(candidate_actions) - 1, 1), dtype=np.float32),
            ],
            axis=1,
        )
        rows.append(features.astype(np.float32))
        candidate_idx.append(np.full((rate.shape[0],), int(idx), dtype=np.int64))
    return np.concatenate(rows, axis=0).astype(np.float32), np.concatenate(candidate_idx, axis=0)


def _candidate_gate_summary(actions: np.ndarray, predictions: dict[str, np.ndarray], feature_mode: str = 'global') -> np.ndarray:
    rb = np.clip(np.asarray(actions, dtype=np.float32)[..., RB_DIM], 0.0, None)
    prob = np.asarray(predictions['link_activity_prob'], dtype=np.float32).squeeze(-1)
    rate = np.asarray(predictions['link_rate_pred'], dtype=np.float32).squeeze(-1)
    if rb.shape != rate.shape or prob.shape != rate.shape:
        raise ValueError('candidate action and prediction tensors must share [sample, step, edge] shape')
    predicted_active_mass = np.sum(prob * np.clip(rate, 0.0, None), axis=(1, 2))
    active = (rb > EPS).astype(np.float32)
    clipped_rate = np.clip(rate, 0.0, None)
    step_blocks = [
        np.sum(rb, axis=2),
        np.max(rb, axis=2),
        np.sum(active, axis=2),
        np.sum(prob, axis=2),
        np.max(prob, axis=2),
        np.mean(prob, axis=2),
        np.sum(clipped_rate, axis=2),
        np.max(clipped_rate, axis=2),
        np.sum(prob * clipped_rate, axis=2),
    ]
    global_block = np.stack(
        [
            np.sum(rb, axis=(1, 2)),
            np.max(rb, axis=(1, 2)),
            np.sum(active, axis=(1, 2)),
            np.sum(prob, axis=(1, 2)),
            np.max(prob, axis=(1, 2)),
            np.mean(prob, axis=(1, 2)),
            np.sum(clipped_rate, axis=(1, 2)),
            np.max(clipped_rate, axis=(1, 2)),
            predicted_active_mass,
        ],
        axis=1,
    )
    feature_mode = str(feature_mode)
    if feature_mode == 'global':
        return global_block.astype(np.float32)
    if feature_mode != 'rich':
        raise ValueError(f'unknown pairwise feature mode: {feature_mode}')
    return np.concatenate([global_block] + step_blocks, axis=1).astype(np.float32)


def make_pairwise_gate_features(
    candidate_actions: list[np.ndarray],
    predictions_by_spec: list[dict[str, np.ndarray]],
    challenger_idx: int,
    default_idx: int,
    feature_mode: str = 'global',
) -> np.ndarray:
    challenger_idx = int(challenger_idx)
    default_idx = int(default_idx)
    left = _candidate_gate_summary(candidate_actions[challenger_idx], predictions_by_spec[challenger_idx], feature_mode)
    right = _candidate_gate_summary(candidate_actions[default_idx], predictions_by_spec[default_idx], feature_mode)
    return np.concatenate([left, right, left - right], axis=1).astype(np.float32)


def make_candidate_specs(args: argparse.Namespace) -> list[CandidateSpec]:
    specs = [
        CandidateSpec('identity', 'identity', 'none', 0, 0.0, 0.0, 0.0),
    ]
    for family, rule in [
        ('ranked_rf', 'minus_delta'),
        ('awr_selector', 'awr_temp1'),
        ('iql_expectile_selector', 'expectile0.7'),
    ]:
        for top_k in args.top_k:
            for alpha in args.blend_alpha:
                for cap in args.step_total_cap_scale:
                    for ecap in args.edge_value_cap_scale:
                        name = f'{family}__{rule}__top{int(top_k)}__alpha{float(alpha):g}__cap{float(cap):g}__ecap{float(ecap):g}'
                        specs.append(CandidateSpec(name, family, rule, int(top_k), float(alpha), float(cap), float(ecap)))
    return specs


def make_split_payload(
    args: argparse.Namespace,
    split_name: str,
    arrays: dict[str, np.ndarray],
    split_indices: np.ndarray,
    train_indices: np.ndarray,
    stats: dict,
    policy_model,
    action_scale: np.ndarray,
    value_vocab,
    world_model,
    world_config: dict,
    device,
    score_model,
    value_model,
    steps: tuple[int, ...],
) -> SplitPayload:
    base_dataset, adaptive_dataset = make_adaptive_dataset(
        args, arrays, split_indices, stats, policy_model, action_scale, value_vocab, device, train_indices
    )
    baseline_actions, truth_actions = collect_raw_actions(adaptive_dataset, stats)
    examples = _make_inference_examples(baseline_actions, truth_actions.shape, steps)
    context = collect_rollout_edge_context(world_model, base_dataset, baseline_actions, stats, device, args.batch_size)
    features = rows_from_context(context, examples.coordinates)
    pred_score = np.asarray(score_model.predict(features), dtype=np.float32).reshape(-1)
    raw_pred_value = predict_conservative_value_target(value_model, features, mode='mean', beta=0.0)
    pred_value = invert_value_target('abs', raw_pred_value, examples.baseline_values)
    return SplitPayload(
        base_dataset=base_dataset,
        baseline_actions=baseline_actions,
        coordinates=examples.coordinates,
        baseline_values=examples.baseline_values,
        pred_score=pred_score,
        pred_value=pred_value,
    )


def candidate_score_for_spec(spec: CandidateSpec, payload: SplitPayload, train_score: np.ndarray) -> np.ndarray:
    base_score = link_aware_selection_score(
        payload.pred_score,
        payload.baseline_values,
        payload.pred_value,
        mode='minus_delta',
        risk_weight=0.05,
    )
    if spec.family == 'ranked_rf':
        return base_score
    if spec.family == 'awr_selector':
        advantage = payload.pred_score - expectile_baseline(train_score, expectile=0.5)
        return advantage_weighted_scores(base_score, advantage, temperature=1.0, max_weight=20.0)
    if spec.family == 'iql_expectile_selector':
        baseline = expectile_baseline(payload.pred_score, expectile=0.7)
        advantage = payload.pred_score - baseline
        return advantage_weighted_scores(base_score, advantage, temperature=1.0, max_weight=20.0)
    raise ValueError(f'no score for family: {spec.family}')


def build_candidate_actions(specs: list[CandidateSpec], payload: SplitPayload, train_score: np.ndarray) -> list[np.ndarray]:
    actions_by_spec = []
    for spec in specs:
        if spec.family == 'identity':
            actions_by_spec.append(payload.baseline_actions.copy())
            continue
        score = candidate_score_for_spec(spec, payload, train_score)
        selected = select_topk_indices(payload.coordinates, score, int(spec.top_k), 'per_sample_step')
        actions = _apply_candidate_actions(
            spec.family,
            payload.baseline_actions,
            payload.coordinates,
            payload.pred_value,
            selected,
            alpha=float(spec.alpha),
            cap_scale=float(spec.step_total_cap_scale),
            edge_cap_scale=float(spec.edge_value_cap_scale),
        )
        actions_by_spec.append(actions)
    return actions_by_spec


def evaluate_candidate_library(
    specs: list[CandidateSpec],
    actions_by_spec: list[np.ndarray],
    payload: SplitPayload,
    stats: dict,
    world_model,
    world_config: dict,
    device,
    batch_size: int,
    split_name: str,
) -> tuple[list[dict], list[dict[str, np.ndarray]], np.ndarray, np.ndarray]:
    rows = []
    predictions_by_spec = []
    sample_sse = []
    sample_count = None
    baseline_rmse = float('nan')
    for spec, actions in zip(specs, actions_by_spec):
        predictions = evaluate_raw_actions(actions, payload.base_dataset, stats, world_model, world_config, device, batch_size)
        row = active_rate_row(spec.name, split_name, predictions, baseline_rmse)
        if spec.family == 'identity':
            baseline_rmse = float(row['active_rate_rmse'])
            row['improvement_vs_baseline'] = float('nan')
        else:
            row['improvement_vs_baseline'] = float(baseline_rmse - float(row['active_rate_rmse']))
        row.update({
            'family': spec.family,
            'selection_rule': spec.selection_rule,
            'top_k': int(spec.top_k),
            'alpha': float(spec.alpha),
            'step_total_cap_scale': float(spec.step_total_cap_scale),
            'edge_value_cap_scale': float(spec.edge_value_cap_scale),
        })
        sse, count = sample_active_sse(predictions)
        if sample_count is None:
            sample_count = count
        rows.append(row)
        predictions_by_spec.append(predictions)
        sample_sse.append(sse)
    return rows, predictions_by_spec, np.stack(sample_sse, axis=1), np.asarray(sample_count, dtype=np.int64)


def train_template_selector(features: np.ndarray, labels: np.ndarray, active_count: np.ndarray, seed: int):
    keep = np.asarray(active_count).reshape(-1) > 0
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if not np.any(keep) or np.unique(labels[keep]).shape[0] < 2:
        return None
    model = RandomForestClassifier(
        n_estimators=80,
        max_depth=6,
        min_samples_leaf=3,
        class_weight='balanced_subsample',
        random_state=int(seed),
        n_jobs=-1,
    )
    model.fit(np.asarray(features, dtype=np.float32)[keep], labels[keep])
    return model


def train_pairwise_gate(
    features: np.ndarray,
    labels: np.ndarray,
    active_count: np.ndarray,
    seed: int,
    model_name: str = 'rf',
):
    keep = np.asarray(active_count).reshape(-1) > 0
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if not np.any(keep) or np.unique(labels[keep]).shape[0] < 2:
        return None
    model_name = str(model_name)
    if model_name == 'rf':
        model = RandomForestClassifier(
            n_estimators=120,
            max_depth=5,
            min_samples_leaf=6,
            class_weight='balanced_subsample',
            random_state=int(seed),
            n_jobs=-1,
        )
    elif model_name == 'extratrees':
        model = ExtraTreesClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=4,
            class_weight='balanced',
            random_state=int(seed),
            n_jobs=-1,
        )
    elif model_name == 'logreg':
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=0.5,
                class_weight='balanced',
                max_iter=1000,
                random_state=int(seed),
            ),
        )
    elif model_name == 'gb':
        model = GradientBoostingClassifier(
            n_estimators=120,
            learning_rate=0.04,
            max_depth=2,
            min_samples_leaf=5,
            random_state=int(seed),
        )
    elif model_name == 'hgb':
        model = HistGradientBoostingClassifier(
            learning_rate=0.04,
            max_iter=120,
            max_leaf_nodes=15,
            l2_regularization=0.1,
            random_state=int(seed),
        )
    else:
        raise ValueError(f'unknown pairwise gate model: {model_name}')
    model.fit(np.asarray(features, dtype=np.float32)[keep], labels[keep])
    return model


def train_error_regressor(features: np.ndarray, target_mse: np.ndarray, active_count: np.ndarray, seed: int):
    features = np.asarray(features, dtype=np.float32)
    target = np.asarray(target_mse, dtype=np.float32).reshape(-1)
    active_count = np.asarray(active_count, dtype=np.int64).reshape(-1)
    if features.shape[0] != target.shape[0] or target.shape[0] != active_count.shape[0]:
        raise ValueError('features, target_mse, and active_count must share row count')
    keep = (active_count > 0) & np.isfinite(target)
    if int(np.sum(keep)) < 8:
        return None
    model = RandomForestRegressor(
        n_estimators=120,
        max_depth=8,
        min_samples_leaf=4,
        random_state=int(seed),
        n_jobs=-1,
    )
    model.fit(features[keep], target[keep])
    return model


def train_improvement_regressor(features: np.ndarray, target_gain: np.ndarray, active_count: np.ndarray, seed: int):
    features = np.asarray(features, dtype=np.float32)
    target = np.asarray(target_gain, dtype=np.float32).reshape(-1)
    active_count = np.asarray(active_count, dtype=np.int64).reshape(-1)
    if features.shape[0] != target.shape[0] or target.shape[0] != active_count.shape[0]:
        raise ValueError('features, target_gain, and active_count must share row count')
    keep = (active_count > 0) & np.isfinite(target)
    if int(np.sum(keep)) < 8:
        return None
    model = RandomForestRegressor(
        n_estimators=160,
        max_depth=6,
        min_samples_leaf=6,
        random_state=int(seed),
        n_jobs=-1,
    )
    model.fit(features[keep], target[keep])
    return model


def train_pairwise_gain_regressor(
    features: np.ndarray,
    target_gain: np.ndarray,
    active_count: np.ndarray,
    seed: int,
    model_name: str = 'gbr',
):
    features = np.asarray(features, dtype=np.float32)
    target = np.asarray(target_gain, dtype=np.float32).reshape(-1)
    active_count = np.asarray(active_count, dtype=np.int64).reshape(-1)
    if features.shape[0] != target.shape[0] or target.shape[0] != active_count.shape[0]:
        raise ValueError('features, target_gain, and active_count must share row count')
    keep = (active_count > 0) & np.isfinite(target)
    if int(np.sum(keep)) < 8:
        return None
    model_name = str(model_name)
    if model_name == 'gbr':
        model = GradientBoostingRegressor(
            n_estimators=160,
            learning_rate=0.03,
            max_depth=2,
            min_samples_leaf=5,
            random_state=int(seed),
        )
    elif model_name == 'rf':
        model = RandomForestRegressor(
            n_estimators=180,
            max_depth=6,
            min_samples_leaf=5,
            random_state=int(seed),
            n_jobs=-1,
        )
    else:
        raise ValueError(f'unknown pairwise gain model: {model_name}')
    model.fit(features[keep], target[keep])
    return model


def stack_pairwise_training_blocks(
    blocks: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not blocks:
        raise ValueError('blocks must not be empty')
    features = []
    labels = []
    gains = []
    counts = []
    for feature_block, label_block, gain_block, count_block in blocks:
        feature_block = np.asarray(feature_block, dtype=np.float32)
        label_block = np.asarray(label_block, dtype=np.int64).reshape(-1)
        gain_block = np.asarray(gain_block, dtype=np.float32).reshape(-1)
        count_block = np.asarray(count_block, dtype=np.int64).reshape(-1)
        if feature_block.ndim != 2:
            raise ValueError('feature blocks must have shape [sample, feature]')
        if not (feature_block.shape[0] == label_block.shape[0] == gain_block.shape[0] == count_block.shape[0]):
            raise ValueError('feature, label, gain, and count blocks must share row count')
        features.append(feature_block)
        labels.append(label_block)
        gains.append(gain_block)
        counts.append(count_block)
    return (
        np.concatenate(features, axis=0).astype(np.float32),
        np.concatenate(labels, axis=0).astype(np.int64),
        np.concatenate(gains, axis=0).astype(np.float32),
        np.concatenate(counts, axis=0).astype(np.int64),
    )


def predict_or_default(model, features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    if model is None:
        return np.zeros((features.shape[0],), dtype=np.int64)
    return np.asarray(model.predict(features), dtype=np.int64)


def predict_error_regressor_choice(model, features: np.ndarray, sample_count: int, candidate_count: int) -> np.ndarray:
    if model is None:
        return np.zeros((int(sample_count),), dtype=np.int64)
    pred = np.asarray(model.predict(np.asarray(features, dtype=np.float32)), dtype=np.float32)
    pred = pred.reshape(int(candidate_count), int(sample_count)).T
    return np.argmin(pred, axis=1).astype(np.int64)


def select_pairwise_threshold_by_rmse(
    score: np.ndarray,
    sample_sse: np.ndarray,
    active_count: np.ndarray,
    default_idx: int,
    challenger_idx: int,
    thresholds: list[float] | tuple[float, ...] | np.ndarray,
) -> tuple[float, float, np.ndarray]:
    score = np.asarray(score, dtype=np.float32).reshape(-1)
    sample_sse = np.asarray(sample_sse, dtype=np.float64)
    active_count = np.asarray(active_count, dtype=np.int64).reshape(-1)
    if sample_sse.ndim != 2 or sample_sse.shape[0] != score.shape[0] or active_count.shape[0] != score.shape[0]:
        raise ValueError('score, sample_sse, and active_count must share sample count')
    thresholds = [float(value) for value in thresholds]
    if not thresholds:
        raise ValueError('thresholds must not be empty')
    best_threshold = thresholds[0]
    best_choice = np.full((score.shape[0],), int(default_idx), dtype=np.int64)
    best_rmse = float('inf')
    for threshold in thresholds:
        choice = np.where(score >= float(threshold), int(challenger_idx), int(default_idx)).astype(np.int64)
        rmse = choice_rmse_from_sample_sse(sample_sse, active_count, choice)
        if rmse < best_rmse - 1e-12:
            best_threshold = float(threshold)
            best_rmse = float(rmse)
            best_choice = choice
    return float(best_threshold), float(best_rmse), best_choice


def make_stratified_fit_calibration_masks(
    labels: np.ndarray,
    active_count: np.ndarray,
    calibration_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    active_count = np.asarray(active_count, dtype=np.int64).reshape(-1)
    if labels.shape[0] != active_count.shape[0]:
        raise ValueError('labels and active_count must share row count')
    eligible = np.where(active_count > 0)[0]
    fit_mask = np.zeros((labels.shape[0],), dtype=bool)
    calibration_mask = np.zeros((labels.shape[0],), dtype=bool)
    if eligible.shape[0] == 0:
        return fit_mask, calibration_mask
    rng = np.random.default_rng(int(seed))
    calibration_fraction = float(np.clip(calibration_fraction, 0.05, 0.8))
    for label in np.unique(labels[eligible]):
        idx = eligible[labels[eligible] == int(label)]
        idx = idx.copy()
        rng.shuffle(idx)
        if idx.shape[0] < 2:
            fit_mask[idx] = True
            continue
        calibration_count = int(round(idx.shape[0] * calibration_fraction))
        calibration_count = max(1, min(idx.shape[0] - 1, calibration_count))
        calibration_mask[idx[:calibration_count]] = True
        fit_mask[idx[calibration_count:]] = True
    if np.unique(labels[fit_mask]).shape[0] < 2:
        fit_mask[eligible] = True
        calibration_mask[:] = False
    return fit_mask, calibration_mask


def conservative_improvement_choice(
    predicted_gain: np.ndarray,
    default_idx: int,
    min_predicted_gain: float,
    allowed_candidates: np.ndarray | None = None,
) -> np.ndarray:
    predicted_gain = np.asarray(predicted_gain, dtype=np.float32)
    if predicted_gain.ndim != 2:
        raise ValueError('predicted_gain must have shape [sample, candidate]')
    default_idx = int(default_idx)
    if default_idx < 0 or default_idx >= predicted_gain.shape[1]:
        raise ValueError('default_idx is out of range')
    score = predicted_gain.copy()
    if allowed_candidates is not None:
        allowed = np.asarray(allowed_candidates, dtype=bool).reshape(-1)
        if allowed.shape[0] != score.shape[1]:
            raise ValueError('allowed_candidates length must match candidate count')
        score[:, ~allowed] = -np.inf
    score[:, default_idx] = 0.0
    best = np.argmax(score, axis=1).astype(np.int64)
    best_gain = score[np.arange(score.shape[0]), best]
    return np.where(best_gain >= float(min_predicted_gain), best, default_idx).astype(np.int64)


def predict_improvement_regressor_choice(
    model,
    features: np.ndarray,
    sample_count: int,
    candidate_count: int,
    default_idx: int,
    min_predicted_gain: float,
    allowed_candidates: np.ndarray | None = None,
) -> np.ndarray:
    if model is None:
        return np.full((int(sample_count),), int(default_idx), dtype=np.int64)
    pred = np.asarray(model.predict(np.asarray(features, dtype=np.float32)), dtype=np.float32)
    pred = pred.reshape(int(candidate_count), int(sample_count)).T
    return conservative_improvement_choice(pred, default_idx, min_predicted_gain, allowed_candidates)


def _model_classes(model) -> np.ndarray:
    if hasattr(model, 'classes_'):
        return np.asarray(model.classes_, dtype=np.int64)
    if hasattr(model, 'steps') and model.steps:
        return np.asarray(model.steps[-1][1].classes_, dtype=np.int64)
    return np.asarray([], dtype=np.int64)


def predict_pairwise_gate_choice(
    model,
    features: np.ndarray,
    default_idx: int,
    challenger_idx: int,
    min_probability: float,
    invert: bool = False,
) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    default_idx = int(default_idx)
    challenger_idx = int(challenger_idx)
    if model is None:
        return np.full((features.shape[0],), default_idx, dtype=np.int64)
    positive_prob = predict_pairwise_gate_score(model, features)
    if bool(invert):
        return np.where(positive_prob <= float(min_probability), challenger_idx, default_idx).astype(np.int64)
    return np.where(positive_prob >= float(min_probability), challenger_idx, default_idx).astype(np.int64)


def predict_pairwise_quota_choice(
    model,
    features: np.ndarray,
    default_idx: int,
    challenger_idx: int,
    default_fraction: float,
) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    default_idx = int(default_idx)
    challenger_idx = int(challenger_idx)
    if model is None:
        return np.full((features.shape[0],), default_idx, dtype=np.int64)
    positive_prob = predict_pairwise_gate_score(model, features)
    sample_count = int(positive_prob.shape[0])
    default_count = int(round(sample_count * float(np.clip(default_fraction, 0.0, 1.0))))
    if default_count <= 0:
        return np.full((sample_count,), challenger_idx, dtype=np.int64)
    if default_count >= sample_count:
        return np.full((sample_count,), default_idx, dtype=np.int64)
    choice = np.full((sample_count,), challenger_idx, dtype=np.int64)
    default_score = 1.0 - positive_prob
    default_indices = np.argpartition(default_score, -default_count)[-default_count:]
    choice[default_indices] = default_idx
    return choice.astype(np.int64)


def predict_pairwise_gate_score(model, features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    if model is None:
        return np.zeros((features.shape[0],), dtype=np.float32)
    prob = np.asarray(model.predict_proba(features), dtype=np.float32)
    classes = _model_classes(model)
    positive_columns = np.where(classes == 1)[0]
    if positive_columns.shape[0] == 0:
        return np.zeros((features.shape[0],), dtype=np.float32)
    return prob[:, int(positive_columns[0])].astype(np.float32)


def predict_pairwise_gain_choice(
    model,
    features: np.ndarray,
    default_idx: int,
    challenger_idx: int,
    min_gain: float,
) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    default_idx = int(default_idx)
    challenger_idx = int(challenger_idx)
    if model is None:
        return np.full((features.shape[0],), default_idx, dtype=np.int64)
    gain = np.asarray(model.predict(features), dtype=np.float32).reshape(-1)
    return np.where(gain >= float(min_gain), challenger_idx, default_idx).astype(np.int64)


def evaluate_selector_choice(
    name: str,
    split_name: str,
    choice: np.ndarray,
    specs: list[CandidateSpec],
    actions_by_spec: list[np.ndarray],
    payload: SplitPayload,
    stats: dict,
    world_model,
    world_config: dict,
    device,
    batch_size: int,
    baseline_rmse: float,
) -> dict:
    mixed_actions = mix_actions_by_sample(actions_by_spec, choice)
    predictions = evaluate_raw_actions(mixed_actions, payload.base_dataset, stats, world_model, world_config, device, batch_size)
    row = active_rate_row(name, split_name, predictions, baseline_rmse)
    bincount = np.bincount(np.asarray(choice, dtype=np.int64), minlength=len(specs))
    row.update({
        'family': 'rollout_reward_template_selector',
        'selection_rule': name,
        'choice_histogram': json.dumps({specs[i].name: int(bincount[i]) for i in range(len(specs))}, ensure_ascii=False),
    })
    return row


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
    parser.add_argument('--steps', type=int, nargs='+', default=[1, 2])
    parser.add_argument('--max-train-samples', type=int, default=256)
    parser.add_argument('--max-val-samples', type=int, default=128)
    parser.add_argument('--max-test-samples', type=int, default=128)
    parser.add_argument('--limit-after-stats', action='store_true')
    parser.add_argument('--streaming-stats', action='store_true')
    parser.add_argument('--stats-chunk-size', type=int, default=512)
    parser.add_argument('--rf-trees', type=int, default=50)
    parser.add_argument('--top-k', type=int, nargs='+', default=[32])
    parser.add_argument('--blend-alpha', type=float, nargs='+', default=[1.0])
    parser.add_argument('--step-total-cap-scale', type=float, nargs='+', default=[1.1, 1.15])
    parser.add_argument('--edge-value-cap-scale', type=float, nargs='+', default=[1.15, 1.25])
    parser.add_argument('--selector-min-predicted-gain', type=float, nargs='+', default=[0.0, 1000.0, 2500.0, 5000.0])
    parser.add_argument('--pairwise-feature-mode', choices=('global', 'rich'), default='global')
    parser.add_argument('--pairwise-gate-model', choices=('rf', 'extratrees', 'logreg', 'gb', 'hgb'), nargs='+', default=['rf'])
    parser.add_argument('--pairwise-fit-split', choices=('train', 'train_val'), default='train')
    parser.add_argument('--pairwise-gate-probability', type=float, nargs='+', default=[0.5, 0.6, 0.7, 0.8, 0.9])
    parser.add_argument('--pairwise-default-fraction', type=float, nargs='+', default=[])
    parser.add_argument('--pairwise-gain-model', choices=('gbr', 'rf'), nargs='+', default=[])
    parser.add_argument('--pairwise-gain-threshold', type=float, nargs='+', default=[0.0, 250.0, 500.0, 1000.0])
    parser.add_argument('--pairwise-calibrated', action='store_true')
    parser.add_argument('--pairwise-calibration-fraction', type=float, default=0.35)
    parser.add_argument('--selector-min-oracle-wins', type=int, default=1)
    parser.add_argument('--seed', type=int, default=20260629)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict:
    started = time.time()
    device = resolve_torch_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary, arrays, stats, world_model, policy_model, action_scale, value_vocab, splits = load_context_limited(args, device)
    splits = dict(splits)
    if args.limit_after_stats:
        splits['train'] = limit_indices(splits['train'], args.max_train_samples)
        splits['val'] = limit_indices(splits['val'], args.max_val_samples)
        splits['test'] = limit_indices(splits['test'], args.max_test_samples)
    world_model = freeze_module(world_model)
    policy_model = freeze_module(policy_model)
    steps = tuple(int(step) for step in args.steps)

    train_base, train_dataset = make_adaptive_dataset(args, arrays, splits['train'], stats, policy_model, action_scale, value_vocab, device, splits['train'])
    train_actions, train_truth = collect_raw_actions(train_dataset, stats)
    train_examples = make_critical_examples(train_actions, train_truth, steps=steps)
    train_edge_improvement = collect_edge_gradient_improvement(world_model, train_base, train_actions, train_truth, stats, summary['config'], device, args.batch_size)
    train_score, train_value = make_targets(train_examples, train_truth, train_edge_improvement)
    train_context = collect_rollout_edge_context(world_model, train_base, train_actions, stats, device, args.batch_size)
    train_features = rows_from_context(train_context, train_examples.coordinates)
    score_model, value_model = _fit_models(args, train_features, train_score, train_value, train_examples.baseline_values)

    split_payload = {
        'train': make_split_payload(args, 'train', arrays, splits['train'], splits['train'], stats, policy_model, action_scale, value_vocab, world_model, summary['config'], device, score_model, value_model, steps),
        'val': make_split_payload(args, 'val', arrays, splits['val'], splits['train'], stats, policy_model, action_scale, value_vocab, world_model, summary['config'], device, score_model, value_model, steps),
        'test': make_split_payload(args, 'test', arrays, splits['test'], splits['train'], stats, policy_model, action_scale, value_vocab, world_model, summary['config'], device, score_model, value_model, steps),
    }

    specs = make_candidate_specs(args)
    rows = []
    actions_by_split = {}
    predictions_by_split = {}
    sse_by_split = {}
    count_by_split = {}
    for split_name, payload in split_payload.items():
        actions = build_candidate_actions(specs, payload, train_score)
        split_rows, predictions, sample_sse, sample_count = evaluate_candidate_library(
            specs, actions, payload, stats, world_model, summary['config'], device, args.batch_size, split_name
        )
        rows.extend(split_rows)
        actions_by_split[split_name] = actions
        predictions_by_split[split_name] = predictions
        sse_by_split[split_name] = sample_sse
        count_by_split[split_name] = sample_count

    train_labels = np.argmin(sse_by_split['train'] / np.maximum(count_by_split['train'][:, None], 1), axis=1).astype(np.int64)
    action_selector = train_template_selector(
        make_sample_summary_features(split_payload['train'].baseline_actions),
        train_labels,
        count_by_split['train'],
        int(args.seed),
    )
    forecast_selector = train_template_selector(
        make_forecast_selector_features(actions_by_split['train'], predictions_by_split['train']),
        train_labels,
        count_by_split['train'],
        int(args.seed) + 17,
    )
    train_error_features, train_error_candidate_idx = make_candidate_error_features(actions_by_split['train'], predictions_by_split['train'])
    train_sample_mse = sse_by_split['train'] / np.maximum(count_by_split['train'][:, None], 1)
    train_error_target = train_sample_mse.T.reshape(-1)
    train_error_active_count = np.repeat(count_by_split['train'][None, :], len(specs), axis=0).reshape(-1)
    error_regressor = train_error_regressor(
        train_error_features,
        train_error_target,
        train_error_active_count,
        int(args.seed) + 31,
    )
    default_candidate_idx, train_single_rmse = choose_best_single_by_sample_sse(sse_by_split['train'], count_by_split['train'])
    train_oracle_win_count = np.bincount(train_labels, minlength=len(specs))
    allowed_candidates = train_oracle_win_count >= int(args.selector_min_oracle_wins)
    allowed_candidates[int(default_candidate_idx)] = True
    train_default_mse = train_sample_mse[:, int(default_candidate_idx)]
    train_improvement_target = (train_default_mse[:, None] - train_sample_mse).T.reshape(-1)
    improvement_regressor = train_improvement_regressor(
        train_error_features,
        train_improvement_target,
        train_error_active_count,
        int(args.seed) + 47,
    )
    identity_idx = 0
    pairwise_gates = {}
    pairwise_gain_regressors = {}
    calibrated_pairwise_gates = {}
    if int(default_candidate_idx) != identity_idx:
        train_pair_features = make_pairwise_gate_features(
            actions_by_split['train'],
            predictions_by_split['train'],
            identity_idx,
            int(default_candidate_idx),
            args.pairwise_feature_mode,
        )
        train_pair_labels = (
            train_sample_mse[:, identity_idx] < train_sample_mse[:, int(default_candidate_idx)]
        ).astype(np.int64)
        train_pair_gain = train_sample_mse[:, int(default_candidate_idx)] - train_sample_mse[:, identity_idx]
        pair_fit_features = train_pair_features
        pair_fit_labels = train_pair_labels
        pair_fit_gain = train_pair_gain
        pair_fit_count = count_by_split['train']
        if str(args.pairwise_fit_split) == 'train_val':
            val_sample_mse = sse_by_split['val'] / np.maximum(count_by_split['val'][:, None], 1)
            val_pair_features = make_pairwise_gate_features(
                actions_by_split['val'],
                predictions_by_split['val'],
                identity_idx,
                int(default_candidate_idx),
                args.pairwise_feature_mode,
            )
            val_pair_labels = (
                val_sample_mse[:, identity_idx] < val_sample_mse[:, int(default_candidate_idx)]
            ).astype(np.int64)
            val_pair_gain = val_sample_mse[:, int(default_candidate_idx)] - val_sample_mse[:, identity_idx]
            pair_fit_features, pair_fit_labels, pair_fit_gain, pair_fit_count = stack_pairwise_training_blocks([
                (train_pair_features, train_pair_labels, train_pair_gain, count_by_split['train']),
                (val_pair_features, val_pair_labels, val_pair_gain, count_by_split['val']),
            ])
        for model_offset, model_name in enumerate(args.pairwise_gate_model):
            pairwise_gates[str(model_name)] = train_pairwise_gate(
                pair_fit_features,
                pair_fit_labels,
                pair_fit_count,
                int(args.seed) + 59 + model_offset,
                str(model_name),
            )
        for model_offset, model_name in enumerate(args.pairwise_gain_model):
            pairwise_gain_regressors[str(model_name)] = train_pairwise_gain_regressor(
                pair_fit_features,
                pair_fit_gain,
                pair_fit_count,
                int(args.seed) + 83 + model_offset,
                str(model_name),
            )
        if bool(args.pairwise_calibrated):
            fit_mask, calibration_mask = make_stratified_fit_calibration_masks(
                pair_fit_labels,
                pair_fit_count,
                float(args.pairwise_calibration_fraction),
                int(args.seed) + 101,
            )
            threshold_grid = sorted(set(float(value) for value in args.pairwise_gate_probability))
            for model_offset, model_name in enumerate(args.pairwise_gate_model):
                model = train_pairwise_gate(
                    pair_fit_features[fit_mask],
                    pair_fit_labels[fit_mask],
                    pair_fit_count[fit_mask],
                    int(args.seed) + 131 + model_offset,
                    str(model_name),
                )
                if model is None or not np.any(calibration_mask):
                    calibrated_pairwise_gates[str(model_name)] = {
                        'model': None,
                        'threshold': float('nan'),
                        'calibration_rmse': float('nan'),
                        'fit_count': int(np.sum(fit_mask)),
                        'calibration_count': int(np.sum(calibration_mask)),
                    }
                    continue
                calibration_score = predict_pairwise_gate_score(model, pair_fit_features[calibration_mask])
                calibration_sse = np.concatenate([sse_by_split['train'], sse_by_split['val']], axis=0) if str(args.pairwise_fit_split) == 'train_val' else sse_by_split['train']
                threshold, calibration_rmse, _calibration_choice = select_pairwise_threshold_by_rmse(
                    calibration_score,
                    calibration_sse[calibration_mask],
                    pair_fit_count[calibration_mask],
                    int(default_candidate_idx),
                    identity_idx,
                    threshold_grid,
                )
                calibrated_pairwise_gates[str(model_name)] = {
                    'model': model,
                    'threshold': float(threshold),
                    'calibration_rmse': float(calibration_rmse),
                    'fit_count': int(np.sum(fit_mask)),
                    'calibration_count': int(np.sum(calibration_mask)),
                }

    baseline_rmse = {
        split: float(next(row['active_rate_rmse'] for row in rows if row['split'] == split and row['candidate'] == 'identity'))
        for split in ('train', 'val', 'test')
    }

    selector_rows = []
    for split_name in ('val', 'test'):
        sample_features = make_sample_summary_features(split_payload[split_name].baseline_actions)
        oracle_choice = np.argmin(sse_by_split[split_name] / np.maximum(count_by_split[split_name][:, None], 1), axis=1).astype(np.int64)
        selector_rows.append(evaluate_selector_choice(
            f'rollout_reward_oracle_sample_{split_name}',
            split_name,
            oracle_choice,
            specs,
            actions_by_split[split_name],
            split_payload[split_name],
            stats,
            world_model,
            summary['config'],
            device,
            args.batch_size,
            baseline_rmse[split_name],
        ))
        if int(default_candidate_idx) != identity_idx:
            pair_mse = sse_by_split[split_name][:, [identity_idx, int(default_candidate_idx)]] / np.maximum(
                count_by_split[split_name][:, None], 1
            )
            pair_choice = np.where(
                np.argmin(pair_mse, axis=1) == 0,
                identity_idx,
                int(default_candidate_idx),
            ).astype(np.int64)
            selector_rows.append(evaluate_selector_choice(
                f'rollout_reward_oracle_pair_identity_vs_default_{split_name}',
                split_name,
                pair_choice,
                specs,
                actions_by_split[split_name],
                split_payload[split_name],
                stats,
                world_model,
                summary['config'],
                device,
                args.batch_size,
                baseline_rmse[split_name],
            ))
        predicted_choice = predict_or_default(action_selector, sample_features)
        selector_rows.append(evaluate_selector_choice(
            'rollout_reward_rf_action_selector',
            split_name,
            predicted_choice,
            specs,
            actions_by_split[split_name],
            split_payload[split_name],
            stats,
            world_model,
            summary['config'],
            device,
            args.batch_size,
            baseline_rmse[split_name],
        ))
        forecast_features = make_forecast_selector_features(actions_by_split[split_name], predictions_by_split[split_name])
        forecast_choice = predict_or_default(forecast_selector, forecast_features)
        selector_rows.append(evaluate_selector_choice(
            'rollout_reward_rf_forecast_selector',
            split_name,
            forecast_choice,
            specs,
            actions_by_split[split_name],
            split_payload[split_name],
            stats,
            world_model,
            summary['config'],
            device,
            args.batch_size,
            baseline_rmse[split_name],
        ))
        error_features, _error_candidate_idx = make_candidate_error_features(actions_by_split[split_name], predictions_by_split[split_name])
        error_choice = predict_error_regressor_choice(error_regressor, error_features, split_payload[split_name].baseline_actions.shape[0], len(specs))
        selector_rows.append(evaluate_selector_choice(
            'rollout_reward_rf_error_regressor',
            split_name,
            error_choice,
            specs,
            actions_by_split[split_name],
            split_payload[split_name],
            stats,
            world_model,
            summary['config'],
            device,
            args.batch_size,
            baseline_rmse[split_name],
        ))
        for min_gain in args.selector_min_predicted_gain:
            improvement_choice = predict_improvement_regressor_choice(
                improvement_regressor,
                error_features,
                split_payload[split_name].baseline_actions.shape[0],
                len(specs),
                int(default_candidate_idx),
                float(min_gain),
                allowed_candidates,
            )
            safe_gain = str(float(min_gain)).replace('.', 'p').replace('-', 'm')
            selector_rows.append(evaluate_selector_choice(
                f'rollout_reward_rf_improvement_gate_gain{safe_gain}',
                split_name,
                improvement_choice,
                specs,
                actions_by_split[split_name],
                split_payload[split_name],
                stats,
                world_model,
                summary['config'],
                device,
                args.batch_size,
                baseline_rmse[split_name],
            ))
        if int(default_candidate_idx) != identity_idx:
            pair_features = make_pairwise_gate_features(
                actions_by_split[split_name],
                predictions_by_split[split_name],
                identity_idx,
                int(default_candidate_idx),
                args.pairwise_feature_mode,
            )
            for model_name, pairwise_gate in pairwise_gates.items():
                for probability in args.pairwise_gate_probability:
                    pair_choice = predict_pairwise_gate_choice(
                        pairwise_gate,
                        pair_features,
                        int(default_candidate_idx),
                        identity_idx,
                        float(probability),
                    )
                    safe_probability = str(float(probability)).replace('.', 'p').replace('-', 'm')
                    selector_rows.append(evaluate_selector_choice(
                        f'rollout_reward_pairwise_{model_name}_identity_gate_p{safe_probability}',
                        split_name,
                        pair_choice,
                        specs,
                        actions_by_split[split_name],
                        split_payload[split_name],
                        stats,
                        world_model,
                        summary['config'],
                        device,
                        args.batch_size,
                        baseline_rmse[split_name],
                    ))
                for default_fraction in args.pairwise_default_fraction:
                    quota_choice = predict_pairwise_quota_choice(
                        pairwise_gate,
                        pair_features,
                        int(default_candidate_idx),
                        identity_idx,
                        float(default_fraction),
                    )
                    safe_fraction = str(float(default_fraction)).replace('.', 'p').replace('-', 'm')
                    selector_rows.append(evaluate_selector_choice(
                        f'rollout_reward_pairwise_{model_name}_identity_quota_q{safe_fraction}',
                        split_name,
                        quota_choice,
                        specs,
                        actions_by_split[split_name],
                        split_payload[split_name],
                        stats,
                        world_model,
                        summary['config'],
                        device,
                        args.batch_size,
                        baseline_rmse[split_name],
                    ))
            for model_name, calibrated in calibrated_pairwise_gates.items():
                threshold = float(calibrated.get('threshold', float('nan')))
                calibrated_choice = predict_pairwise_gate_choice(
                    calibrated.get('model'),
                    pair_features,
                    int(default_candidate_idx),
                    identity_idx,
                    threshold,
                )
                safe_threshold = str(threshold).replace('.', 'p').replace('-', 'm')
                selector_rows.append(evaluate_selector_choice(
                    f'rollout_reward_pairwise_calibrated_{model_name}_identity_gate_p{safe_threshold}',
                    split_name,
                    calibrated_choice,
                    specs,
                    actions_by_split[split_name],
                    split_payload[split_name],
                    stats,
                    world_model,
                    summary['config'],
                    device,
                    args.batch_size,
                    baseline_rmse[split_name],
                ))
            for model_name, gain_regressor in pairwise_gain_regressors.items():
                for min_gain in args.pairwise_gain_threshold:
                    pair_gain_choice = predict_pairwise_gain_choice(
                        gain_regressor,
                        pair_features,
                        int(default_candidate_idx),
                        identity_idx,
                        float(min_gain),
                    )
                    safe_gain = str(float(min_gain)).replace('.', 'p').replace('-', 'm')
                    selector_rows.append(evaluate_selector_choice(
                        f'rollout_reward_pairwise_gain_{model_name}_identity_gate_gain{safe_gain}',
                        split_name,
                        pair_gain_choice,
                        specs,
                        actions_by_split[split_name],
                        split_payload[split_name],
                        stats,
                        world_model,
                        summary['config'],
                        device,
                        args.batch_size,
                        baseline_rmse[split_name],
                    ))
                    inverted_pair_choice = predict_pairwise_gate_choice(
                        pairwise_gate,
                        pair_features,
                        int(default_candidate_idx),
                        identity_idx,
                        float(probability),
                        invert=True,
                    )
                    selector_rows.append(evaluate_selector_choice(
                        f'rollout_reward_pairwise_{model_name}_identity_gate_inv_p{safe_probability}',
                        split_name,
                        inverted_pair_choice,
                        specs,
                        actions_by_split[split_name],
                        split_payload[split_name],
                        stats,
                        world_model,
                        summary['config'],
                        device,
                        args.batch_size,
                        baseline_rmse[split_name],
                    ))
    rows.extend(selector_rows)

    write_csv(args.output_dir / 'rollout_reward_template_selector_results.csv', rows)
    val_ranked = sorted([row for row in rows if row['split'] == 'val'], key=lambda row: (float(row['active_rate_rmse']), str(row['candidate'])))
    write_csv(args.output_dir / 'rollout_reward_template_selector_val_ranked.csv', val_ranked)
    test_by_candidate = {str(row['candidate']): row for row in rows if row['split'] == 'test'}
    deployable_val_ranked = [row for row in val_ranked if not str(row['candidate']).startswith('rollout_reward_oracle')]
    best_val = deployable_val_ranked[0] if deployable_val_ranked else val_ranked[0]
    diagnostics = {
        'candidate_count': len(specs),
        'candidate_names': [spec.name for spec in specs],
        'train_examples': int(train_examples.coordinates.shape[0]),
        'train_positive_score_count': int(np.sum(train_score > 0.0)),
        'train_positive_value_count': int(np.sum(train_value >= float(args.min_effective_rb_total))),
        'train_selector_class_count': int(np.unique(train_labels[count_by_split['train'] > 0]).shape[0]) if np.any(count_by_split['train'] > 0) else 0,
        'train_selector_label_histogram': {specs[i].name: int(np.sum(train_labels == i)) for i in range(len(specs))},
        'action_selector_available': action_selector is not None,
        'forecast_selector_available': forecast_selector is not None,
        'error_regressor_available': error_regressor is not None,
        'improvement_regressor_available': improvement_regressor is not None,
        'pairwise_gate_models': [str(value) for value in args.pairwise_gate_model],
        'pairwise_feature_mode': str(args.pairwise_feature_mode),
        'pairwise_fit_split': str(args.pairwise_fit_split),
        'pairwise_identity_gate_available': {name: model is not None for name, model in pairwise_gates.items()},
        'pairwise_gain_models': [str(value) for value in args.pairwise_gain_model],
        'pairwise_gain_regressor_available': {name: model is not None for name, model in pairwise_gain_regressors.items()},
        'pairwise_calibrated': bool(args.pairwise_calibrated),
        'pairwise_calibration_fraction': float(args.pairwise_calibration_fraction),
        'pairwise_calibrated_gate_available': {name: values.get('model') is not None for name, values in calibrated_pairwise_gates.items()},
        'pairwise_calibrated_threshold': {name: float(values.get('threshold', float('nan'))) for name, values in calibrated_pairwise_gates.items()},
        'pairwise_calibrated_calibration_rmse': {name: float(values.get('calibration_rmse', float('nan'))) for name, values in calibrated_pairwise_gates.items()},
        'pairwise_calibrated_fit_count': {name: int(values.get('fit_count', 0)) for name, values in calibrated_pairwise_gates.items()},
        'pairwise_calibrated_calibration_count': {name: int(values.get('calibration_count', 0)) for name, values in calibrated_pairwise_gates.items()},
        'default_candidate_idx': int(default_candidate_idx),
        'default_candidate_name': specs[int(default_candidate_idx)].name,
        'train_single_rmse': {specs[i].name: float(train_single_rmse[i]) for i in range(len(specs))},
        'selector_min_predicted_gain': [float(value) for value in args.selector_min_predicted_gain],
        'pairwise_gate_probability': [float(value) for value in args.pairwise_gate_probability],
        'pairwise_default_fraction': [float(value) for value in args.pairwise_default_fraction],
        'pairwise_gain_threshold': [float(value) for value in args.pairwise_gain_threshold],
        'selector_min_oracle_wins': int(args.selector_min_oracle_wins),
        'allowed_candidate_names': [specs[i].name for i in range(len(specs)) if bool(allowed_candidates[i])],
        'train_oracle_win_count': {specs[i].name: int(train_oracle_win_count[i]) for i in range(len(specs))},
        'train_error_feature_rows': int(train_error_features.shape[0]),
        'train_error_candidate_idx_checksum': int(np.sum(train_error_candidate_idx)),
        'runtime_seconds': float(time.time() - started),
    }
    result = {
        'framework': 'PI-JWM',
        'candidate': 'v11',
        'mode': 'rollout_reward_template_selector_cpu',
        'result_protocol': make_selector_result_protocol(
            str(best_val['candidate']),
            str(args.pairwise_fit_split),
        ),
        'output_dir': str(args.output_dir),
        'command': ' '.join(sys.argv),
        'device': str(device),
        'diagnostics': diagnostics,
        'best_val': best_val,
        'matched_test_for_best_val': test_by_candidate.get(str(best_val['candidate'])),
    }
    write_json(args.output_dir / 'summary.json', result)
    return result


def main() -> None:
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
