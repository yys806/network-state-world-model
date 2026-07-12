'''CPU-first template-level policy improvement for PI-JWM v11 candidate.

This is a controlled base-policy replacement probe.  Instead of multiplying the
same edge scores, it learns a small reward-aware selector over complete ranked
allocation templates.  The selector chooses a template per sample-step, then the
chosen template generates the constrained RB repair for that sample-step.
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))

from pi_jwm.v11_rollout_value_calibrator import freeze_module

from diagnose_v11_counterfactual_value_attribution import collect_raw_actions
from diagnose_v11_rb_total_latent_identifiability import (
    build_models,
    collect_rollout_edge_context,
    invert_value_target,
    make_targets,
    make_value_target,
    rows_from_context,
)
from diagnose_v11_rb_total_oracle_value_scope import safe_corr, select_topk_indices, write_json
from diagnose_v11_scheduler_ranked_allocation import (
    link_aware_selection_score,
    make_selector_target,
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


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / 'artifacts/experiments/pi_jwm_v11_template_policy_selector_20260627'
EPS = 1e-9


@dataclass(frozen=True)
class SchedulerTemplate:
    top_k: int
    alpha: float
    step_total_cap_scale: float
    edge_value_cap_scale: float

    @property
    def name(self) -> str:
        return (
            f'top{int(self.top_k)}'
            f'__alpha{float(self.alpha):g}'
            f'__cap{float(self.step_total_cap_scale):g}'
            f'__ecap{float(self.edge_value_cap_scale):g}'
        )


def template_advantage_weights(
    rewards: np.ndarray,
    baseline: float,
    temperature: float = 1.0,
    max_weight: float = 20.0,
) -> np.ndarray:
    rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)
    temp = max(float(temperature), EPS)
    weights = np.exp(np.clip((rewards - float(baseline)) / temp, -20.0, np.log(max(float(max_weight), 1.0))))
    return weights.astype(np.float32)


def expectile_value(values: np.ndarray, expectile: float = 0.8, iterations: int = 100) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    tau = float(expectile)
    if not 0.0 < tau < 1.0:
        raise ValueError('expectile must be in (0, 1)')
    estimate = float(np.mean(values))
    for _ in range(int(iterations)):
        diff = values - estimate
        weights = np.where(diff >= 0.0, tau, 1.0 - tau)
        denom = float(np.sum(weights))
        if denom <= EPS:
            break
        updated = float(np.sum(weights * values) / denom)
        if abs(updated - estimate) < 1e-9:
            estimate = updated
            break
        estimate = updated
    return estimate


def _ordered_sample_step_groups(coordinates: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    coords = np.asarray(coordinates, dtype=np.int64).reshape(-1, 3)
    group_map: dict[tuple[int, int], list[int]] = {}
    order: list[tuple[int, int]] = []
    for row_idx, (sample, step, _edge) in enumerate(coords):
        key = (int(sample), int(step))
        if key not in group_map:
            group_map[key] = []
            order.append(key)
        group_map[key].append(int(row_idx))
    keys = np.asarray(order, dtype=np.int64).reshape(-1, 2)
    groups = [np.asarray(group_map[key], dtype=np.int64) for key in order]
    return keys, groups


def make_sample_step_feature_rows(
    actions: np.ndarray,
    coordinates: np.ndarray,
    selection_score: np.ndarray,
    predicted_value: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    actions = np.asarray(actions, dtype=np.float32)
    coords = np.asarray(coordinates, dtype=np.int64).reshape(-1, 3)
    selection_score = np.asarray(selection_score, dtype=np.float32).reshape(-1)
    predicted_value = np.asarray(predicted_value, dtype=np.float32).reshape(-1)
    if not (coords.shape[0] == selection_score.shape[0] == predicted_value.shape[0]):
        raise ValueError('coordinates, selection_score, and predicted_value must have the same row count')
    keys, groups = _ordered_sample_step_groups(coords)
    rows = []
    horizon = max(1, actions.shape[1] - 1)
    for key, indices in zip(keys, groups):
        sample, step = int(key[0]), int(key[1])
        rb = np.clip(actions[sample, step, :, RB_DIM], 0.0, None)
        group_scores = selection_score[indices]
        group_values = predicted_value[indices]
        baseline_values = actions[coords[indices, 0], coords[indices, 1], coords[indices, 2], RB_DIM]
        rows.append(
            [
                float(np.sum(rb)),
                float(np.count_nonzero(rb > EPS)),
                float(np.mean(group_scores)) if group_scores.size else 0.0,
                float(np.max(group_scores)) if group_scores.size else 0.0,
                float(np.mean(group_values)) if group_values.size else 0.0,
                float(np.max(group_values)) if group_values.size else 0.0,
                float(np.mean(baseline_values)) if baseline_values.size else 0.0,
                float(np.sum(np.clip(group_values - baseline_values, 0.0, None))) if group_values.size else 0.0,
                float(step) / float(horizon),
                float(indices.size),
            ]
        )
    return keys, np.asarray(rows, dtype=np.float32)


def template_reward_matrix(
    coordinates: np.ndarray,
    oracle_score: np.ndarray,
    baseline_value: np.ndarray,
    predicted_value: np.ndarray,
    selection_score: np.ndarray,
    templates: list[SchedulerTemplate],
    value_penalty_weight: float = 0.02,
) -> tuple[np.ndarray, np.ndarray]:
    coords = np.asarray(coordinates, dtype=np.int64).reshape(-1, 3)
    oracle_score = np.asarray(oracle_score, dtype=np.float32).reshape(-1)
    baseline_value = np.asarray(baseline_value, dtype=np.float32).reshape(-1)
    predicted_value = np.asarray(predicted_value, dtype=np.float32).reshape(-1)
    selection_score = np.asarray(selection_score, dtype=np.float32).reshape(-1)
    if not (coords.shape[0] == oracle_score.shape[0] == baseline_value.shape[0] == predicted_value.shape[0] == selection_score.shape[0]):
        raise ValueError('all edge-level arrays must have the same row count')
    keys, groups = _ordered_sample_step_groups(coords)
    rewards = np.zeros((len(groups), len(templates)), dtype=np.float32)
    positive_delta = np.clip(predicted_value - baseline_value, 0.0, None)
    for group_idx, indices in enumerate(groups):
        if indices.size == 0:
            continue
        order = indices[np.argsort(selection_score[indices])[::-1]]
        for template_idx, template in enumerate(templates):
            selected = order[: max(0, min(int(template.top_k), order.size))]
            if selected.size == 0:
                continue
            raw_gain = float(np.sum(oracle_score[selected]))
            penalty = float(value_penalty_weight) * float(np.sum(positive_delta[selected]))
            rewards[group_idx, template_idx] = raw_gain - penalty
    return keys, rewards


def sample_step_rewards_from_predictions(
    predictions: dict[str, np.ndarray],
    group_keys: np.ndarray,
    link_penalty_weight: float = 0.0,
) -> np.ndarray:
    truth = np.asarray(predictions['link_rate_true'], dtype=np.float32).squeeze(-1)
    pred = np.asarray(predictions['link_rate_pred'], dtype=np.float32).squeeze(-1)
    activity_true = np.asarray(predictions['link_activity_true'], dtype=np.float32).squeeze(-1)
    group_keys = np.asarray(group_keys, dtype=np.int64).reshape(-1, 2)
    rewards = np.zeros((group_keys.shape[0],), dtype=np.float32)
    for row_idx, (sample, step) in enumerate(group_keys):
        sample = int(sample)
        step = int(step)
        active = activity_true[sample, step] > 0.5
        step_error = pred[sample, step] - truth[sample, step]
        if np.any(active):
            active_rmse = float(np.sqrt(np.mean(np.square(step_error[active]))))
        else:
            active_rmse = float(np.sqrt(np.mean(np.square(step_error))))
        link_rmse = float(np.sqrt(np.mean(np.square(step_error))))
        rewards[row_idx] = -active_rmse - float(link_penalty_weight) * link_rmse
    return rewards.astype(np.float32)


def apply_template_assignments(
    actions: np.ndarray,
    coordinates: np.ndarray,
    predicted_value: np.ndarray,
    selection_score: np.ndarray,
    group_keys: np.ndarray,
    template_assignments: np.ndarray,
    templates: list[SchedulerTemplate],
) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    coords = np.asarray(coordinates, dtype=np.int64).reshape(-1, 3)
    predicted_value = np.asarray(predicted_value, dtype=np.float32).reshape(-1)
    selection_score = np.asarray(selection_score, dtype=np.float32).reshape(-1)
    group_keys = np.asarray(group_keys, dtype=np.int64).reshape(-1, 2)
    template_assignments = np.asarray(template_assignments, dtype=np.int64).reshape(-1)
    if not (coords.shape[0] == predicted_value.shape[0] == selection_score.shape[0]):
        raise ValueError('coordinates, predicted_value, and selection_score must have the same row count')
    if group_keys.shape[0] != template_assignments.shape[0]:
        raise ValueError('group_keys and template_assignments must have the same row count')
    if len(templates) == 0:
        raise ValueError('templates must not be empty')

    repaired = actions.copy()
    _keys, groups = _ordered_sample_step_groups(coords)
    group_by_key = {(int(key[0]), int(key[1])): group for key, group in zip(_keys, groups)}
    for key, template_idx in zip(group_keys, template_assignments):
        sample, step = int(key[0]), int(key[1])
        if step == 0:
            continue
        group = group_by_key.get((sample, step))
        if group is None or group.size == 0:
            continue
        template = templates[int(np.clip(template_idx, 0, len(templates) - 1))]
        ordered = group[np.argsort(selection_score[group])[::-1]]
        selected = ordered[: max(0, min(int(template.top_k), ordered.size))]
        for row_idx in selected:
            _sample, _step, edge = coords[int(row_idx)]
            if actions[sample, step, edge, RB_DIM] <= EPS:
                continue
            baseline = float(actions[sample, step, edge, RB_DIM])
            target = max(float(predicted_value[int(row_idx)]), 0.0)
            if float(template.edge_value_cap_scale) > 0.0:
                target = min(target, baseline * float(template.edge_value_cap_scale))
            repaired[sample, step, edge, RB_DIM] = max((1.0 - float(template.alpha)) * baseline + float(template.alpha) * target, 0.0)

        if float(template.step_total_cap_scale) > 0.0:
            original_total = float(np.sum(np.clip(actions[sample, step, :, RB_DIM], 0.0, None)))
            repaired_total = float(np.sum(np.clip(repaired[sample, step, :, RB_DIM], 0.0, None)))
            cap = max(original_total * float(template.step_total_cap_scale), 0.0)
            if repaired_total > max(cap, EPS):
                repaired[sample, step, :, RB_DIM] *= cap / repaired_total

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
    parser.add_argument('--rf-trees', type=int, default=100)
    parser.add_argument('--selector-rf-trees', type=int, default=80)
    parser.add_argument('--top-k', type=int, nargs='+', default=[16, 32])
    parser.add_argument('--blend-alpha', type=float, nargs='+', default=[0.95, 1.0])
    parser.add_argument('--step-total-cap-scale', type=float, nargs='+', default=[1.1, 1.15])
    parser.add_argument('--edge-value-cap-scale', type=float, nargs='+', default=[1.15, 1.25])
    parser.add_argument('--risk-weight', type=float, default=0.05)
    parser.add_argument('--value-penalty-weight', type=float, default=0.02)
    parser.add_argument('--template-label-source', choices=('first_order', 'rollout'), default='first_order')
    parser.add_argument('--rollout-link-penalty-weight', type=float, default=0.0)
    parser.add_argument('--awr-temperature', type=float, default=1.0)
    parser.add_argument('--iql-temperature', type=float, default=1.0)
    parser.add_argument('--iql-expectile', type=float, default=0.8)
    parser.add_argument('--max-weight', type=float, default=20.0)
    parser.add_argument('--seed', type=int, default=20260627)
    return parser.parse_args()


def build_templates(args: argparse.Namespace) -> list[SchedulerTemplate]:
    templates = []
    for top_k in args.top_k:
        for alpha in args.blend_alpha:
            for cap in args.step_total_cap_scale:
                for ecap in args.edge_value_cap_scale:
                    templates.append(
                        SchedulerTemplate(
                            top_k=int(top_k),
                            alpha=float(alpha),
                            step_total_cap_scale=float(cap),
                            edge_value_cap_scale=float(ecap),
                        )
                    )
    return templates


def _fit_edge_models(args, train_features, train_score, train_value, baseline_values):
    factories = build_models(seed=int(args.seed), rf_trees=int(args.rf_trees))
    score_model = factories['rf']()
    score_model.fit(train_features, make_selector_target('gradient', train_score, train_value, float(args.min_effective_rb_total)))
    value_model = factories['rf']()
    positive_value_mask = train_value >= float(args.min_effective_rb_total)
    if np.any(positive_value_mask):
        target = make_value_target('abs', train_value[positive_value_mask], baseline_values[positive_value_mask])
        value_model.fit(train_features[positive_value_mask], target)
    else:
        target = make_value_target('abs', train_value, baseline_values)
        value_model.fit(train_features, target)
    return score_model, value_model


def _make_payload(args, split_name, arrays, splits, stats, policy_model, action_scale, value_vocab, world_model, summary, device, score_model, value_model, steps):
    base_dataset, adaptive_dataset = make_adaptive_dataset(
        args, arrays, splits[split_name], stats, policy_model, action_scale, value_vocab, device, splits['train']
    )
    baseline_actions, truth_actions = collect_raw_actions(adaptive_dataset, stats)
    examples = _make_inference_examples(baseline_actions, truth_actions.shape, steps)
    edge_improvement = collect_edge_gradient_improvement(
        world_model, base_dataset, baseline_actions, truth_actions, stats, summary['config'], device, args.batch_size
    )
    oracle_score, true_value = make_targets(examples, truth_actions, edge_improvement)
    latent_context = collect_rollout_edge_context(world_model, base_dataset, baseline_actions, stats, device, args.batch_size)
    features = rows_from_context(latent_context, examples.coordinates)
    pred_score = np.asarray(score_model.predict(features), dtype=np.float32).reshape(-1)
    raw_pred_value = predict_conservative_value_target(value_model, features, mode='mean', beta=0.0)
    pred_value = invert_value_target('abs', raw_pred_value, examples.baseline_values)
    selection_score = link_aware_selection_score(
        pred_score,
        examples.baseline_values,
        pred_value,
        mode='minus_delta',
        risk_weight=float(args.risk_weight),
    )
    group_keys, group_features = make_sample_step_feature_rows(baseline_actions, examples.coordinates, selection_score, pred_value)
    return {
        'base_dataset': base_dataset,
        'baseline_actions': baseline_actions,
        'examples': examples,
        'oracle_score': oracle_score,
        'true_value': true_value,
        'features': features,
        'pred_score': pred_score,
        'pred_value': pred_value,
        'selection_score': selection_score,
        'group_keys': group_keys,
        'group_features': group_features,
    }


def _fit_template_selector(features: np.ndarray, labels: np.ndarray, weights: np.ndarray, args) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=int(args.selector_rf_trees),
        min_samples_leaf=3,
        max_features='sqrt',
        random_state=int(args.seed),
        n_jobs=-1,
        class_weight=None,
    )
    model.fit(np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int64), sample_weight=np.asarray(weights, dtype=np.float32))
    return model


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
    templates = build_templates(args)

    train_base, train_dataset = make_adaptive_dataset(args, arrays, splits['train'], stats, policy_model, action_scale, value_vocab, device, splits['train'])
    train_actions, train_truth = collect_raw_actions(train_dataset, stats)
    train_examples = make_critical_examples(train_actions, train_truth, steps=steps)
    train_edge_improvement = collect_edge_gradient_improvement(
        world_model, train_base, train_actions, train_truth, stats, summary['config'], device, args.batch_size
    )
    train_score, train_value = make_targets(train_examples, train_truth, train_edge_improvement)
    train_context = collect_rollout_edge_context(world_model, train_base, train_actions, stats, device, args.batch_size)
    train_edge_features = rows_from_context(train_context, train_examples.coordinates)
    score_model, value_model = _fit_edge_models(args, train_edge_features, train_score, train_value, train_examples.baseline_values)
    train_pred_score = np.asarray(score_model.predict(train_edge_features), dtype=np.float32).reshape(-1)
    train_raw_pred_value = predict_conservative_value_target(value_model, train_edge_features, mode='mean', beta=0.0)
    train_pred_value = invert_value_target('abs', train_raw_pred_value, train_examples.baseline_values)
    train_selection_score = link_aware_selection_score(
        train_pred_score,
        train_examples.baseline_values,
        train_pred_value,
        mode='minus_delta',
        risk_weight=float(args.risk_weight),
    )
    train_group_keys, train_group_features = make_sample_step_feature_rows(
        train_actions, train_examples.coordinates, train_selection_score, train_pred_value
    )
    _, train_rewards = template_reward_matrix(
        train_examples.coordinates,
        train_score,
        train_examples.baseline_values,
        train_pred_value,
        train_selection_score,
        templates,
        value_penalty_weight=float(args.value_penalty_weight),
    )
    if args.template_label_source == 'rollout':
        rollout_rewards = np.zeros_like(train_rewards, dtype=np.float32)
        for template_idx, _template in enumerate(templates):
            train_assignments = np.full((train_group_keys.shape[0],), template_idx, dtype=np.int64)
            train_template_actions = apply_template_assignments(
                train_actions,
                train_examples.coordinates,
                train_pred_value,
                train_selection_score,
                train_group_keys,
                train_assignments,
                templates,
            )
            train_predictions = evaluate_raw_actions(
                train_template_actions,
                train_base,
                stats,
                world_model,
                summary['config'],
                device,
                args.batch_size,
            )
            rollout_rewards[:, template_idx] = sample_step_rewards_from_predictions(
                train_predictions,
                train_group_keys,
                link_penalty_weight=float(args.rollout_link_penalty_weight),
            )
        train_rewards = rollout_rewards
    train_labels = np.argmax(train_rewards, axis=1).astype(np.int64)
    train_best_reward = np.max(train_rewards, axis=1).astype(np.float32)
    global_template_idx = int(np.argmax(np.mean(train_rewards, axis=0))) if train_rewards.size else 0

    awac_weights = template_advantage_weights(
        train_best_reward,
        baseline=float(np.mean(train_best_reward)) if train_best_reward.size else 0.0,
        temperature=float(args.awr_temperature),
        max_weight=float(args.max_weight),
    )
    iql_weights = template_advantage_weights(
        train_best_reward,
        baseline=expectile_value(train_best_reward, expectile=float(args.iql_expectile)),
        temperature=float(args.iql_temperature),
        max_weight=float(args.max_weight),
    )
    awac_selector = _fit_template_selector(train_group_features, train_labels, awac_weights, args)
    iql_selector = _fit_template_selector(train_group_features, train_labels, iql_weights, args)

    split_payload = {
        name: _make_payload(args, name, arrays, splits, stats, policy_model, action_scale, value_vocab, world_model, summary, device, score_model, value_model, steps)
        for name in ('val', 'test')
    }

    rows = []
    baseline_rmse_by_split = {}
    for split_name, payload in split_payload.items():
        baseline_predictions = evaluate_raw_actions(
            payload['baseline_actions'], payload['base_dataset'], stats, world_model, summary['config'], device, args.batch_size
        )
        row = active_rate_row('identity_bc_reference', split_name, baseline_predictions, float('nan'))
        row.update({'family': 'identity', 'selector': 'none'})
        rows.append(row)
        baseline_rmse_by_split[split_name] = float(row['active_rate_rmse'])

        fixed_candidates: list[tuple[str, np.ndarray]] = []
        fixed_candidates.append((
            f'global_train_best_template__{templates[global_template_idx].name}',
            np.full((payload['group_keys'].shape[0],), global_template_idx, dtype=np.int64),
        ))
        fixed_candidates.append((
            'awac_template_selector',
            np.asarray(awac_selector.predict(payload['group_features']), dtype=np.int64).reshape(-1),
        ))
        fixed_candidates.append((
            'iql_template_selector',
            np.asarray(iql_selector.predict(payload['group_features']), dtype=np.int64).reshape(-1),
        ))

        _, oracle_rewards = template_reward_matrix(
            payload['examples'].coordinates,
            payload['oracle_score'],
            payload['examples'].baseline_values,
            payload['pred_value'],
            payload['selection_score'],
            templates,
            value_penalty_weight=float(args.value_penalty_weight),
        )
        fixed_candidates.append((
            'diagnostic_only__oracle_template_selector',
            np.argmax(oracle_rewards, axis=1).astype(np.int64),
        ))

        for template_idx, template in enumerate(templates):
            fixed_candidates.append((
                f'fixed_template__{template.name}',
                np.full((payload['group_keys'].shape[0],), template_idx, dtype=np.int64),
            ))

        for candidate, assignments in fixed_candidates:
            actions = apply_template_assignments(
                payload['baseline_actions'],
                payload['examples'].coordinates,
                payload['pred_value'],
                payload['selection_score'],
                payload['group_keys'],
                assignments,
                templates,
            )
            predictions = evaluate_raw_actions(actions, payload['base_dataset'], stats, world_model, summary['config'], device, args.batch_size)
            row = active_rate_row(candidate, split_name, predictions, baseline_rmse_by_split[split_name])
            row.update(
                {
                    'family': 'template_policy',
                    'selector': candidate.split('__')[0],
                    'unique_templates': int(np.unique(assignments).size) if assignments.size else 0,
                    'template_assignment_histogram': json.dumps(
                        {templates[int(idx)].name: int(count) for idx, count in zip(*np.unique(assignments, return_counts=True))},
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    if assignments.size
                    else '{}',
                }
            )
            rows.append(row)

    write_csv(args.output_dir / 'template_policy_selector_results.csv', rows)
    val_ranked = sorted([row for row in rows if row['split'] == 'val'], key=lambda row: (float(row['active_rate_rmse']), str(row['candidate'])))
    deployable_val_ranked = [
        row for row in val_ranked
        if str(row.get('candidate')) != 'identity_bc_reference' and not str(row.get('candidate')).startswith('diagnostic_only')
    ]
    diagnostic_val_ranked = [row for row in val_ranked if str(row.get('candidate')).startswith('diagnostic_only')]
    write_csv(args.output_dir / 'template_policy_selector_val_ranked.csv', val_ranked)
    write_csv(args.output_dir / 'template_policy_selector_deployable_val_ranked.csv', deployable_val_ranked)
    test_by_candidate = {str(row['candidate']): row for row in rows if row['split'] == 'test'}
    best_val = deployable_val_ranked[0] if deployable_val_ranked else None
    best_diagnostic_val = diagnostic_val_ranked[0] if diagnostic_val_ranked else None
    diagnostics = {
        'templates': [template.name for template in templates],
        'template_label_source': str(args.template_label_source),
        'global_template': templates[global_template_idx].name if templates else '',
        'train_group_count': int(train_group_features.shape[0]),
        'train_examples': int(train_examples.coordinates.shape[0]),
        'train_positive_score_count': int(np.sum(train_score > 0.0)),
        'train_positive_value_count': int(np.sum(train_value >= float(args.min_effective_rb_total))),
        'train_best_reward_mean': float(np.mean(train_best_reward)) if train_best_reward.size else 0.0,
        'train_best_reward_expectile': float(expectile_value(train_best_reward, expectile=float(args.iql_expectile))),
        'val_score_pearson': safe_corr(split_payload['val']['pred_score'], split_payload['val']['oracle_score']),
        'test_score_pearson': safe_corr(split_payload['test']['pred_score'], split_payload['test']['oracle_score']),
        'val_value_mae_nonzero_true': float(mean_absolute_error(split_payload['val']['true_value'][split_payload['val']['true_value'] >= float(args.min_effective_rb_total)], split_payload['val']['pred_value'][split_payload['val']['true_value'] >= float(args.min_effective_rb_total)])) if np.any(split_payload['val']['true_value'] >= float(args.min_effective_rb_total)) else float('nan'),
        'test_value_mae_nonzero_true': float(mean_absolute_error(split_payload['test']['true_value'][split_payload['test']['true_value'] >= float(args.min_effective_rb_total)], split_payload['test']['pred_value'][split_payload['test']['true_value'] >= float(args.min_effective_rb_total)])) if np.any(split_payload['test']['true_value'] >= float(args.min_effective_rb_total)) else float('nan'),
        'runtime_seconds': float(time.time() - started),
    }
    result = {
        'framework': 'PI-JWM',
        'candidate': 'v11',
        'mode': 'template_policy_selector_cpu',
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
