'''CPU-first ranked scheduler allocation diagnostic for PI-JWM v11 candidate.

This script keeps the useful signal from the latent edge-ranking diagnostics,
but constrains generated RB totals at each sample-step.  It tests whether a
ranking-aware scheduler repair can improve active-rate without the link-RMSE
blow-up caused by unconstrained value replacement.
'''

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))

from pi_jwm.v11_rollout_value_calibrator import freeze_module

from diagnose_v11_counterfactual_value_attribution import collect_raw_actions
from evaluate_v10_policy_bridge import collect_bridge_policy_predictions, load_policy
from diagnose_v11_rb_total_latent_identifiability import (
    build_models,
    collect_rollout_edge_context,
    invert_value_target,
    make_targets,
    make_value_target,
    rows_from_context,
)
from diagnose_v11_rb_total_oracle_value_scope import rankdata, safe_corr, select_topk_indices, write_json
from run_v11_rb_total_repair import active_rate_row, evaluate_raw_actions, make_adaptive_dataset, write_csv
from run_v11_rb_total_value_head import (
    RB_DIM,
    _make_inference_examples,
    collect_edge_gradient_improvement,
    limit_indices,
    load_context_limited,
    make_critical_examples,
)
from run_v7_action_policy import V7ActionPolicyDataset, collate_action_policy_batch


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / 'artifacts/experiments/pi_jwm_v11_scheduler_ranked_allocation_20260622'
EPS = 1e-9


def resolve_torch_device(device_arg: str) -> torch.device:
    if device_arg == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device_arg == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but torch.cuda.is_available() is false')
    if device_arg not in ('cpu', 'cuda'):
        raise ValueError(f'unknown device: {device_arg}')
    return torch.device(device_arg)


def link_aware_selection_score(
    active_score: np.ndarray,
    baseline_value: np.ndarray,
    predicted_value: np.ndarray,
    mode: str,
    risk_weight: float,
) -> np.ndarray:
    active_score = np.asarray(active_score, dtype=np.float32).reshape(-1)
    baseline_value = np.asarray(baseline_value, dtype=np.float32).reshape(-1)
    predicted_value = np.asarray(predicted_value, dtype=np.float32).reshape(-1)
    if not (active_score.shape[0] == baseline_value.shape[0] == predicted_value.shape[0]):
        raise ValueError('scores and values must have the same row count')
    positive_delta = np.clip(predicted_value - baseline_value, 0.0, None)
    if mode == 'raw':
        penalty = 0.0
    elif mode == 'minus_delta':
        penalty = positive_delta
    elif mode == 'minus_ratio':
        penalty = positive_delta / np.maximum(np.clip(baseline_value, 0.0, None), 1.0)
    else:
        raise ValueError(f'unknown selection score mode: {mode}')
    return (active_score - float(risk_weight) * penalty).astype(np.float32)


def blend_policy_prior_selection_score(
    ranked_score: np.ndarray,
    policy_prior: np.ndarray,
    policy_weight: float,
) -> np.ndarray:
    ranked_score = np.asarray(ranked_score, dtype=np.float32).reshape(-1)
    policy_prior = np.asarray(policy_prior, dtype=np.float32).reshape(-1)
    if ranked_score.shape[0] != policy_prior.shape[0]:
        raise ValueError('ranked_score and policy_prior must have the same row count')
    if not np.all(np.isfinite(ranked_score)) or not np.all(np.isfinite(policy_prior)):
        raise ValueError('ranked_score and policy_prior must be finite')
    weight = float(policy_weight)
    if not 0.0 <= weight <= 1.0:
        raise ValueError('policy_weight must be between 0 and 1')
    return ((1.0 - weight) * ranked_score + weight * policy_prior).astype(np.float32)


def collect_policy_prior_for_examples(
    arrays: dict[str, np.ndarray],
    indices: np.ndarray,
    stats: dict,
    policy_checkpoint: Path,
    device: torch.device,
    batch_size: int,
    coordinates: np.ndarray,
) -> np.ndarray:
    prior_model, action_scale, _, value_vocab = load_policy(policy_checkpoint, device)
    prior_model = freeze_module(prior_model)
    prior_dataset = V7ActionPolicyDataset(arrays, indices, stats, action_scale)
    prior_loader = DataLoader(
        prior_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_action_policy_batch,
    )
    action_scale_t = torch.as_tensor(action_scale.reshape(1, 1, 1, -1), dtype=torch.float32, device=device)
    predictions = collect_bridge_policy_predictions(prior_model, prior_loader, device, action_scale_t, value_vocab)
    probability = np.asarray(predictions['prob'], dtype=np.float32)
    coordinates = np.asarray(coordinates, dtype=np.int64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError('coordinates must have shape [row, 3]')
    return probability[coordinates[:, 0], coordinates[:, 1], coordinates[:, 2], RB_DIM].astype(np.float32)


def predict_conservative_value_target(model, features: np.ndarray, mode: str, beta: float = 0.0) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    mean_prediction = np.asarray(model.predict(features), dtype=np.float32).reshape(-1)
    mode = str(mode)
    estimators = getattr(model, 'estimators_', None)
    if mode == 'mean' or estimators is None:
        return mean_prediction.astype(np.float32)

    tree_predictions = np.stack(
        [np.asarray(estimator.predict(features), dtype=np.float32).reshape(-1) for estimator in estimators],
        axis=0,
    )
    if mode == 'lcb':
        return (mean_prediction - float(beta) * np.std(tree_predictions, axis=0)).astype(np.float32)
    if mode.startswith('q'):
        quantile = float(mode[1:]) / 100.0
        if not (0.0 <= quantile <= 1.0):
            raise ValueError(f'quantile mode out of range: {mode}')
        return np.quantile(tree_predictions, quantile, axis=0).astype(np.float32)
    raise ValueError(f'unknown value prediction mode: {mode}')


def make_selector_target(
    mode: str,
    gradient_score: np.ndarray,
    true_value: np.ndarray,
    min_effective_value: float,
) -> np.ndarray:
    gradient_score = np.asarray(gradient_score, dtype=np.float32).reshape(-1)
    true_value = np.asarray(true_value, dtype=np.float32).reshape(-1)
    if gradient_score.shape[0] != true_value.shape[0]:
        raise ValueError('gradient_score and true_value must have the same row count')
    if mode == 'gradient':
        return gradient_score.astype(np.float32)
    if mode == 'value_binary':
        return (true_value >= float(min_effective_value)).astype(np.float32)
    if mode == 'value_log':
        return np.log1p(np.clip(true_value, 0.0, None)).astype(np.float32)
    raise ValueError(f'unknown selector target mode: {mode}')


def apply_selected_blend_repair_with_step_cap(
    actions: np.ndarray,
    coordinates: np.ndarray,
    values: np.ndarray,
    selected: np.ndarray,
    alpha: float,
    step_total_cap_scale: float,
    edge_value_cap_scale: float = 0.0,
) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    coords = np.asarray(coordinates, dtype=np.int64).reshape(-1, 3)
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    selected = np.asarray(selected, dtype=np.int64).reshape(-1)
    if coords.shape[0] != values.shape[0]:
        raise ValueError('coordinates and values must have the same row count')
    repaired = actions.copy()
    alpha = float(alpha)
    for row_idx in selected:
        sample, step, edge = coords[int(row_idx)]
        if int(step) == 0 or actions[sample, step, edge, RB_DIM] <= EPS:
            continue
        baseline_value = float(actions[sample, step, edge, RB_DIM])
        target_value = max(float(values[int(row_idx)]), 0.0)
        if float(edge_value_cap_scale) > 0.0:
            target_value = min(target_value, baseline_value * float(edge_value_cap_scale))
        repaired[sample, step, edge, RB_DIM] = max((1.0 - alpha) * baseline_value + alpha * target_value, 0.0)

    repaired[:, 0, :, RB_DIM] = actions[:, 0, :, RB_DIM]
    repaired[..., RB_DIM] = np.where(actions[..., RB_DIM] > EPS, np.clip(repaired[..., RB_DIM], 0.0, None), 0.0)

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
    repaired[..., RB_DIM] = np.where(actions[..., RB_DIM] > EPS, np.clip(repaired[..., RB_DIM], 0.0, None), 0.0)
    return repaired.astype(np.float32)


def apply_ranked_step_redistribution(
    actions: np.ndarray,
    coordinates: np.ndarray,
    values: np.ndarray,
    selected: np.ndarray,
    alpha: float,
    step_total_scale: float,
    edge_value_cap_scale: float = 0.0,
) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    coords = np.asarray(coordinates, dtype=np.int64).reshape(-1, 3)
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    selected = np.asarray(selected, dtype=np.int64).reshape(-1)
    if coords.shape[0] != values.shape[0]:
        raise ValueError('coordinates and values must have the same row count')
    repaired = actions.copy()
    groups: dict[tuple[int, int], list[int]] = {}
    for row_idx in selected:
        sample, step, edge = coords[int(row_idx)]
        if int(step) == 0 or actions[sample, step, edge, RB_DIM] <= EPS:
            continue
        groups.setdefault((int(sample), int(step)), []).append(int(row_idx))

    for (sample, step), row_indices in groups.items():
        edges = coords[row_indices, 2].astype(np.int64)
        baseline = np.clip(actions[sample, step, edges, RB_DIM].astype(np.float32), 0.0, None)
        if baseline.size == 0 or float(np.sum(baseline)) <= EPS:
            continue
        scores = np.clip(values[row_indices].astype(np.float32), 0.0, None)
        if float(np.sum(scores)) <= EPS:
            scores = baseline.copy()
        weights = scores / max(float(np.sum(scores)), EPS)
        target_total = float(np.sum(baseline)) * max(float(step_total_scale), 0.0)
        target = weights * target_total
        if float(edge_value_cap_scale) > 0.0:
            target = np.minimum(target, baseline * float(edge_value_cap_scale))
        alpha = float(alpha)
        repaired_values = (1.0 - alpha) * baseline + alpha * target
        repaired[sample, step, edges, RB_DIM] = np.clip(repaired_values, 0.0, None)

    repaired[:, 0, :, RB_DIM] = actions[:, 0, :, RB_DIM]
    repaired[..., RB_DIM] = np.where(actions[..., RB_DIM] > EPS, np.clip(repaired[..., RB_DIM], 0.0, None), 0.0)
    return repaired.astype(np.float32)


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
    parser.add_argument('--max-train-samples', type=int, default=512)
    parser.add_argument('--max-val-samples', type=int, default=256)
    parser.add_argument('--max-test-samples', type=int, default=256)
    parser.add_argument('--limit-after-stats', action='store_true')
    parser.add_argument('--streaming-stats', action='store_true')
    parser.add_argument('--stats-chunk-size', type=int, default=512)
    parser.add_argument('--top-k', type=int, nargs='+', default=[16, 32, 64])
    parser.add_argument('--scopes', choices=('per_sample', 'per_sample_step'), nargs='+', default=['per_sample', 'per_sample_step'])
    parser.add_argument('--rf-trees', type=int, default=100)
    parser.add_argument('--feature-sets', choices=('latent', 'action_latent'), nargs='+', default=['latent'])
    parser.add_argument('--model-kinds', choices=('rf', 'hgb', 'xgb'), nargs='+', default=['rf'])
    parser.add_argument('--selector-target-modes', choices=('gradient', 'value_binary', 'value_log'), nargs='+', default=['gradient'])
    parser.add_argument('--value-target-modes', choices=('abs', 'log', 'residual', 'ratio'), nargs='+', default=['abs', 'residual'])
    parser.add_argument('--value-prediction-modes', nargs='+', default=['mean'])
    parser.add_argument('--value-lcb-beta', type=float, nargs='+', default=[0.0])
    parser.add_argument('--blend-alpha', type=float, nargs='+', default=[0.5, 0.75, 1.0])
    parser.add_argument('--allocation-modes', choices=('blend', 'redistribute'), nargs='+', default=['blend'])
    parser.add_argument('--step-total-cap-scale', type=float, nargs='+', default=[1.0, 1.1, 1.25, 1.5])
    parser.add_argument('--edge-value-cap-scale', type=float, nargs='+', default=[0.0])
    parser.add_argument('--selection-score-modes', choices=('raw', 'minus_delta', 'minus_ratio'), nargs='+', default=['raw'])
    parser.add_argument('--risk-weight', type=float, nargs='+', default=[0.0])
    parser.add_argument('--policy-prior-checkpoint', type=Path, default=None)
    parser.add_argument('--policy-prior-weight', type=float, nargs='+', default=[0.0])
    parser.add_argument('--seed', type=int, default=20260622)
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
    train_latent = rows_from_context(train_context, train_examples.coordinates)
    train_action = train_examples.features.astype(np.float32)
    train_feature_sets = {
        'latent': train_latent,
        'action_latent': np.concatenate([train_action, train_latent], axis=1).astype(np.float32),
    }

    split_payload = {}
    for split_name in ('val', 'test'):
        base_dataset, adaptive_dataset = make_adaptive_dataset(args, arrays, splits[split_name], stats, policy_model, action_scale, value_vocab, device, splits['train'])
        baseline_actions, truth_actions = collect_raw_actions(adaptive_dataset, stats)
        examples = _make_inference_examples(baseline_actions, truth_actions.shape, steps)
        edge_improvement = collect_edge_gradient_improvement(world_model, base_dataset, baseline_actions, truth_actions, stats, summary['config'], device, args.batch_size)
        oracle_score, true_value = make_targets(examples, truth_actions, edge_improvement)
        latent_context = collect_rollout_edge_context(world_model, base_dataset, baseline_actions, stats, device, args.batch_size)
        latent_rows = rows_from_context(latent_context, examples.coordinates)
        action_rows = examples.features.astype(np.float32)
        policy_prior = None
        if args.policy_prior_checkpoint is not None:
            policy_prior = collect_policy_prior_for_examples(
                arrays,
                splits[split_name],
                stats,
                args.policy_prior_checkpoint,
                device,
                args.batch_size,
                examples.coordinates,
            )
        split_payload[split_name] = {
            'base_dataset': base_dataset,
            'baseline_actions': baseline_actions,
            'examples': examples,
            'oracle_score': oracle_score,
            'true_value': true_value,
            'policy_prior': policy_prior,
            'features': {
                'latent': latent_rows,
                'action_latent': np.concatenate([action_rows, latent_rows], axis=1).astype(np.float32),
            },
        }

    rows = []
    diagnostics = {
        'train_examples': int(train_examples.coordinates.shape[0]),
        'train_positive_score_count': int(np.sum(train_score > 0.0)),
        'train_positive_value_count': int(np.sum(train_value >= float(args.min_effective_rb_total))),
        'feature_sets': list(args.feature_sets),
        'model_kinds': list(args.model_kinds),
        'selector_target_modes': list(args.selector_target_modes),
        'value_target_modes': list(args.value_target_modes),
        'value_prediction_modes': list(args.value_prediction_modes),
        'value_lcb_beta': [float(item) for item in args.value_lcb_beta],
        'allocation_modes': list(args.allocation_modes),
        'step_total_cap_scale': [float(item) for item in args.step_total_cap_scale],
        'edge_value_cap_scale': [float(item) for item in args.edge_value_cap_scale],
        'selection_score_modes': list(args.selection_score_modes),
        'risk_weight': [float(item) for item in args.risk_weight],
        'policy_prior_checkpoint': None if args.policy_prior_checkpoint is None else str(args.policy_prior_checkpoint),
        'policy_prior_weight': [float(item) for item in args.policy_prior_weight],
    }
    factories = build_models(seed=int(args.seed), rf_trees=args.rf_trees)
    positive_value_mask = train_value >= float(args.min_effective_rb_total)

    for feature_set in args.feature_sets:
        train_features = train_feature_sets[feature_set]
        fitted = {}
        for model_name in args.model_kinds:
            score_models = {}
            for selector_target_mode in args.selector_target_modes:
                score_model = factories[model_name]()
                score_target = make_selector_target(
                    selector_target_mode,
                    train_score,
                    train_value,
                    min_effective_value=float(args.min_effective_rb_total),
                )
                score_model.fit(train_features, score_target)
                score_models[selector_target_mode] = score_model
            value_models = {}
            for value_mode in args.value_target_modes:
                value_model = factories[model_name]()
                if np.any(positive_value_mask):
                    target = make_value_target(value_mode, train_value[positive_value_mask], train_examples.baseline_values[positive_value_mask])
                    value_model.fit(train_features[positive_value_mask], target)
                else:
                    target = make_value_target(value_mode, train_value, train_examples.baseline_values)
                    value_model.fit(train_features, target)
                value_models[value_mode] = value_model
            fitted[model_name] = {'score_models': score_models, 'value_models': value_models}

        for split_name, payload in split_payload.items():
            baseline_predictions = evaluate_raw_actions(payload['baseline_actions'], payload['base_dataset'], stats, world_model, summary['config'], device, args.batch_size)
            baseline_row = active_rate_row('identity', split_name, baseline_predictions, float('nan'))
            baseline_rmse = float(baseline_row['active_rate_rmse'])
            baseline_row.update({'feature_set': feature_set, 'model': 'identity', 'value_mode': 'none', 'scope': 'none', 'top_k': 0, 'alpha': 0.0, 'step_total_cap_scale': 0.0})
            rows.append(baseline_row)

            split_features = payload['features'][feature_set]
            if payload['policy_prior'] is not None:
                support_target = (payload['true_value'] >= float(args.min_effective_rb_total)).astype(np.float32)
                diagnostics[f'{split_name}_policy_prior_support_pearson'] = safe_corr(payload['policy_prior'], support_target)
                diagnostics[f'{split_name}_policy_prior_support_spearman'] = safe_corr(rankdata(payload['policy_prior']), rankdata(support_target))
            for model_name, models in fitted.items():
                pred_scores = {}
                for selector_target_mode, score_model in models['score_models'].items():
                    pred_score = np.asarray(score_model.predict(split_features), dtype=np.float32)
                    pred_scores[selector_target_mode] = pred_score
                    diagnostics[f'{split_name}_{feature_set}_{model_name}_{selector_target_mode}_score_pearson'] = safe_corr(pred_score, payload['oracle_score'])
                    diagnostics[f'{split_name}_{feature_set}_{model_name}_{selector_target_mode}_score_spearman'] = safe_corr(rankdata(pred_score), rankdata(payload['oracle_score']))
                    support_target = (payload['true_value'] >= float(args.min_effective_rb_total)).astype(np.float32)
                    diagnostics[f'{split_name}_{feature_set}_{model_name}_{selector_target_mode}_support_pearson'] = safe_corr(pred_score, support_target)
                    diagnostics[f'{split_name}_{feature_set}_{model_name}_{selector_target_mode}_support_spearman'] = safe_corr(rankdata(pred_score), rankdata(support_target))
                for value_mode, value_model in models['value_models'].items():
                    raw_pred_value = predict_conservative_value_target(value_model, split_features, mode='mean', beta=0.0)
                    mean_pred_value = invert_value_target(value_mode, raw_pred_value, payload['examples'].baseline_values)
                    positive_mask = payload['true_value'] >= float(args.min_effective_rb_total)
                    diagnostics[f'{split_name}_{feature_set}_{model_name}_{value_mode}_value_mae_nonzero_true'] = (
                        float(mean_absolute_error(payload['true_value'][positive_mask], mean_pred_value[positive_mask]))
                        if np.any(positive_mask)
                        else float('nan')
                    )
                    for value_prediction_mode in args.value_prediction_modes:
                        beta_values = args.value_lcb_beta if value_prediction_mode == 'lcb' else [0.0]
                        for lcb_beta in beta_values:
                            raw_pred_value = predict_conservative_value_target(
                                value_model,
                                split_features,
                                mode=value_prediction_mode,
                                beta=float(lcb_beta),
                            )
                            pred_value = invert_value_target(value_mode, raw_pred_value, payload['examples'].baseline_values)
                            prediction_suffix = f'{value_prediction_mode}b{float(lcb_beta):g}' if value_prediction_mode == 'lcb' else value_prediction_mode
                            for selector_target_mode, pred_score in pred_scores.items():
                                for score_mode in args.selection_score_modes:
                                    for risk_weight in args.risk_weight:
                                        adjusted_score = link_aware_selection_score(
                                            pred_score,
                                            payload['examples'].baseline_values,
                                            pred_value,
                                            mode=score_mode,
                                            risk_weight=float(risk_weight),
                                        )
                                        for prior_weight in args.policy_prior_weight:
                                            if float(prior_weight) != 0.0 and payload['policy_prior'] is None:
                                                raise ValueError('policy_prior_weight requires --policy-prior-checkpoint when nonzero')
                                            selection_score = (
                                                adjusted_score
                                                if float(prior_weight) == 0.0
                                                else blend_policy_prior_selection_score(
                                                    adjusted_score,
                                                    payload['policy_prior'],
                                                    policy_weight=float(prior_weight),
                                                )
                                            )
                                            for scope in args.scopes:
                                                for top_k in args.top_k:
                                                    selected = select_topk_indices(payload['examples'].coordinates, selection_score, int(top_k), scope)
                                                for alpha in args.blend_alpha:
                                                    for cap_scale in args.step_total_cap_scale:
                                                        for edge_cap_scale in args.edge_value_cap_scale:
                                                            for allocation_mode in args.allocation_modes:
                                                                if allocation_mode == 'blend':
                                                                    actions = apply_selected_blend_repair_with_step_cap(
                                                                        payload['baseline_actions'],
                                                                        payload['examples'].coordinates,
                                                                        pred_value,
                                                                        selected,
                                                                        alpha=float(alpha),
                                                                        step_total_cap_scale=float(cap_scale),
                                                                        edge_value_cap_scale=float(edge_cap_scale),
                                                                    )
                                                                elif allocation_mode == 'redistribute':
                                                                    actions = apply_ranked_step_redistribution(
                                                                        payload['baseline_actions'],
                                                                        payload['examples'].coordinates,
                                                                        pred_value,
                                                                        selected,
                                                                        alpha=float(alpha),
                                                                        step_total_scale=float(cap_scale),
                                                                        edge_value_cap_scale=float(edge_cap_scale),
                                                                    )
                                                                else:
                                                                    raise ValueError(f'unknown allocation mode: {allocation_mode}')
                                                                predictions = evaluate_raw_actions(actions, payload['base_dataset'], stats, world_model, summary['config'], device, args.batch_size)
                                                                candidate = f'{feature_set}__{model_name}__score_{selector_target_mode}__value_{value_mode}_{prediction_suffix}__alloc_{allocation_mode}__{score_mode}_rw{float(risk_weight):g}__priorw{float(prior_weight):g}__{scope}__top{int(top_k)}__alpha{float(alpha):g}__cap{float(cap_scale):g}__ecap{float(edge_cap_scale):g}'
                                                                row = active_rate_row(candidate, split_name, predictions, baseline_rmse)
                                                                row.update(
                                                                    {
                                                                        'feature_set': feature_set,
                                                                        'model': model_name,
                                                                        'selector_target_mode': selector_target_mode,
                                                                        'value_mode': value_mode,
                                                                        'value_prediction_mode': value_prediction_mode,
                                                                        'value_lcb_beta': float(lcb_beta),
                                                                        'allocation_mode': allocation_mode,
                                                                        'selection_score_mode': score_mode,
                                                                        'risk_weight': float(risk_weight),
                                                                        'policy_prior_weight': float(prior_weight),
                                                                        'scope': scope,
                                                                        'top_k': int(top_k),
                                                                        'alpha': float(alpha),
                                                                        'step_total_cap_scale': float(cap_scale),
                                                                        'edge_value_cap_scale': float(edge_cap_scale),
                                                                        'selected_count': int(selected.size),
                                                                        'selected_oracle_mass': float(np.sum(payload['oracle_score'][selected])) if selected.size else 0.0,
                                                                    }
                                                                )
                                                                rows.append(row)

    write_csv(args.output_dir / 'scheduler_ranked_allocation_results.csv', rows)
    val_ranked = sorted([row for row in rows if row['split'] == 'val'], key=lambda row: (float(row['active_rate_rmse']), str(row['candidate'])))
    deployable_val_ranked = [row for row in val_ranked if row.get('method') != 'identity' and str(row.get('candidate')) != 'identity']
    write_csv(args.output_dir / 'scheduler_ranked_allocation_val_ranked.csv', val_ranked)
    write_csv(args.output_dir / 'scheduler_ranked_allocation_deployable_val_ranked.csv', deployable_val_ranked)
    test_by_candidate = {str(row['candidate']): row for row in rows if row['split'] == 'test'}
    best_val = deployable_val_ranked[0] if deployable_val_ranked else None
    diagnostics['runtime_seconds'] = float(time.time() - started)
    result = {
        'framework': 'PI-JWM',
        'candidate': 'v11',
        'mode': 'scheduler_ranked_allocation_cpu',
        'output_dir': str(args.output_dir),
        'command': ' '.join(sys.argv),
        'device': str(device),
        'diagnostics': diagnostics,
        'best_val': best_val,
        'matched_test_for_best_val': test_by_candidate.get(str(best_val['candidate'])) if best_val else None,
    }
    write_json(args.output_dir / 'summary.json', result)
    return result


def main() -> None:
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
