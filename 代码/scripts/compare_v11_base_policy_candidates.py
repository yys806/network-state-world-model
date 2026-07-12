'''CPU-first base-policy tournament for PI-JWM v11 candidate schedulers.

This script compares small deployable base-policy alternatives before any GPU
run.  It intentionally reuses the ranked-allocation evaluation path so every
candidate is judged by the same PI-JWM rollout metrics.
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
from diagnose_v11_rb_total_oracle_value_scope import rankdata, safe_corr, select_topk_indices, write_json
from diagnose_v11_scheduler_ranked_allocation import (
    apply_ranked_step_redistribution,
    apply_selected_blend_repair_with_step_cap,
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


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / 'artifacts/experiments/pi_jwm_v11_base_policy_tournament_20260626'
EPS = 1e-9


def candidate_families() -> list[str]:
    return ['identity', 'ranked_rf', 'awr_selector', 'iql_expectile_selector']


def expectile_baseline(values: np.ndarray, expectile: float = 0.8, iterations: int = 100) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    tau = float(expectile)
    if not 0.0 < tau < 1.0:
        raise ValueError('expectile must be in (0, 1)')
    baseline = float(np.mean(values))
    for _ in range(int(iterations)):
        diff = values - baseline
        weights = np.where(diff >= 0.0, tau, 1.0 - tau)
        denom = float(np.sum(weights))
        if denom <= EPS:
            break
        new_baseline = float(np.sum(weights * values) / denom)
        if abs(new_baseline - baseline) < 1e-9:
            baseline = new_baseline
            break
        baseline = new_baseline
    return baseline


def advantage_weighted_scores(
    behavior_score: np.ndarray,
    advantage: np.ndarray,
    temperature: float = 1.0,
    max_weight: float = 20.0,
) -> np.ndarray:
    behavior_score = np.asarray(behavior_score, dtype=np.float32).reshape(-1)
    advantage = np.asarray(advantage, dtype=np.float32).reshape(-1)
    if behavior_score.shape[0] != advantage.shape[0]:
        raise ValueError('behavior_score and advantage must have the same row count')
    temp = max(float(temperature), EPS)
    weights = np.exp(np.clip(advantage / temp, -20.0, np.log(max(float(max_weight), 1.0))))
    return (behavior_score * weights.astype(np.float32)).astype(np.float32)


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
    parser.add_argument('--top-k', type=int, nargs='+', default=[16, 32])
    parser.add_argument('--blend-alpha', type=float, nargs='+', default=[0.95, 1.0])
    parser.add_argument('--step-total-cap-scale', type=float, nargs='+', default=[1.1, 1.15])
    parser.add_argument('--edge-value-cap-scale', type=float, nargs='+', default=[1.15, 1.25])
    parser.add_argument('--risk-weight', type=float, nargs='+', default=[0.05])
    parser.add_argument('--awr-temperature', type=float, nargs='+', default=[1.0, 2.0])
    parser.add_argument('--awr-max-weight', type=float, default=20.0)
    parser.add_argument('--expectile', type=float, nargs='+', default=[0.7, 0.8])
    parser.add_argument('--seed', type=int, default=20260626)
    return parser.parse_args()


def _fit_models(args: argparse.Namespace, train_features: np.ndarray, train_score: np.ndarray, train_value: np.ndarray, baseline_values: np.ndarray):
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


def _apply_candidate_actions(
    family: str,
    baseline_actions: np.ndarray,
    coordinates: np.ndarray,
    pred_value: np.ndarray,
    selected: np.ndarray,
    alpha: float,
    cap_scale: float,
    edge_cap_scale: float,
) -> np.ndarray:
    if family == 'ranked_rf' or family.startswith('awr_selector') or family.startswith('iql_expectile_selector'):
        return apply_selected_blend_repair_with_step_cap(
            baseline_actions,
            coordinates,
            pred_value,
            selected,
            alpha=float(alpha),
            step_total_cap_scale=float(cap_scale),
            edge_value_cap_scale=float(edge_cap_scale),
        )
    raise ValueError(f'unknown candidate family for action application: {family}')


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
    train_advantage = train_score - expectile_baseline(train_score, expectile=0.8)

    split_payload = {}
    for split_name in ('val', 'test'):
        base_dataset, adaptive_dataset = make_adaptive_dataset(args, arrays, splits[split_name], stats, policy_model, action_scale, value_vocab, device, splits['train'])
        baseline_actions, truth_actions = collect_raw_actions(adaptive_dataset, stats)
        examples = _make_inference_examples(baseline_actions, truth_actions.shape, steps)
        edge_improvement = collect_edge_gradient_improvement(world_model, base_dataset, baseline_actions, truth_actions, stats, summary['config'], device, args.batch_size)
        oracle_score, true_value = make_targets(examples, truth_actions, edge_improvement)
        latent_context = collect_rollout_edge_context(world_model, base_dataset, baseline_actions, stats, device, args.batch_size)
        features = rows_from_context(latent_context, examples.coordinates)
        pred_score = np.asarray(score_model.predict(features), dtype=np.float32).reshape(-1)
        raw_pred_value = predict_conservative_value_target(value_model, features, mode='mean', beta=0.0)
        pred_value = invert_value_target('abs', raw_pred_value, examples.baseline_values)
        split_payload[split_name] = {
            'base_dataset': base_dataset,
            'baseline_actions': baseline_actions,
            'examples': examples,
            'oracle_score': oracle_score,
            'true_value': true_value,
            'features': features,
            'pred_score': pred_score,
            'pred_value': pred_value,
        }

    rows = []
    baseline_rmse_by_split = {}
    for split_name, payload in split_payload.items():
        predictions = evaluate_raw_actions(payload['baseline_actions'], payload['base_dataset'], stats, world_model, summary['config'], device, args.batch_size)
        row = active_rate_row('identity', split_name, predictions, float('nan'))
        row.update({'family': 'identity', 'selection_rule': 'none', 'top_k': 0, 'alpha': 0.0, 'step_total_cap_scale': 0.0, 'edge_value_cap_scale': 0.0})
        rows.append(row)
        baseline_rmse_by_split[split_name] = float(row['active_rate_rmse'])

    for split_name, payload in split_payload.items():
        baseline_rmse = baseline_rmse_by_split[split_name]
        base_score = link_aware_selection_score(
            payload['pred_score'],
            payload['examples'].baseline_values,
            payload['pred_value'],
            mode='minus_delta',
            risk_weight=float(args.risk_weight[0]),
        )
        score_variants: list[tuple[str, str, np.ndarray]] = [('ranked_rf', 'minus_delta', base_score)]
        for temperature in args.awr_temperature:
            advantage = payload['pred_score'] - expectile_baseline(train_score, expectile=0.5)
            awr_score = advantage_weighted_scores(base_score, advantage, temperature=float(temperature), max_weight=float(args.awr_max_weight))
            score_variants.append(('awr_selector', f'awr_temp{float(temperature):g}', awr_score))
        for expectile in args.expectile:
            baseline = expectile_baseline(payload['pred_score'], expectile=float(expectile))
            iql_advantage = payload['pred_score'] - baseline
            iql_score = advantage_weighted_scores(base_score, iql_advantage, temperature=1.0, max_weight=float(args.awr_max_weight))
            score_variants.append(('iql_expectile_selector', f'expectile{float(expectile):g}', iql_score))

        for family, selection_rule, score in score_variants:
            for top_k in args.top_k:
                selected = select_topk_indices(payload['examples'].coordinates, score, int(top_k), 'per_sample_step')
                for alpha in args.blend_alpha:
                    for cap_scale in args.step_total_cap_scale:
                        for edge_cap_scale in args.edge_value_cap_scale:
                            actions = _apply_candidate_actions(
                                family,
                                payload['baseline_actions'],
                                payload['examples'].coordinates,
                                payload['pred_value'],
                                selected,
                                alpha=float(alpha),
                                cap_scale=float(cap_scale),
                                edge_cap_scale=float(edge_cap_scale),
                            )
                            predictions = evaluate_raw_actions(actions, payload['base_dataset'], stats, world_model, summary['config'], device, args.batch_size)
                            candidate = f'{family}__{selection_rule}__top{int(top_k)}__alpha{float(alpha):g}__cap{float(cap_scale):g}__ecap{float(edge_cap_scale):g}'
                            row = active_rate_row(candidate, split_name, predictions, baseline_rmse)
                            row.update({
                                'family': family,
                                'selection_rule': selection_rule,
                                'top_k': int(top_k),
                                'alpha': float(alpha),
                                'step_total_cap_scale': float(cap_scale),
                                'edge_value_cap_scale': float(edge_cap_scale),
                                'selected_count': int(selected.size),
                                'selected_oracle_mass': float(np.sum(payload['oracle_score'][selected])) if selected.size else 0.0,
                            })
                            rows.append(row)

    val_ranked = sorted([row for row in rows if row['split'] == 'val' and row['candidate'] != 'identity'], key=lambda row: (float(row['active_rate_rmse']), str(row['candidate'])))
    test_by_candidate = {str(row['candidate']): row for row in rows if row['split'] == 'test'}
    best_val = val_ranked[0] if val_ranked else None
    write_csv(args.output_dir / 'base_policy_tournament_results.csv', rows)
    write_csv(args.output_dir / 'base_policy_tournament_val_ranked.csv', val_ranked)
    diagnostics = {
        'candidate_families': candidate_families(),
        'train_examples': int(train_examples.coordinates.shape[0]),
        'train_positive_score_count': int(np.sum(train_score > 0.0)),
        'train_positive_value_count': int(np.sum(train_value >= float(args.min_effective_rb_total))),
        'train_score_expectile_0.8': float(expectile_baseline(train_score, expectile=0.8)),
        'train_advantage_mean': float(np.mean(train_advantage)),
        'runtime_seconds': float(time.time() - started),
        'val_score_pearson': safe_corr(split_payload['val']['pred_score'], split_payload['val']['oracle_score']),
        'test_score_pearson': safe_corr(split_payload['test']['pred_score'], split_payload['test']['oracle_score']),
        'val_value_mae_nonzero_true': float(mean_absolute_error(split_payload['val']['true_value'][split_payload['val']['true_value'] >= float(args.min_effective_rb_total)], split_payload['val']['pred_value'][split_payload['val']['true_value'] >= float(args.min_effective_rb_total)])) if np.any(split_payload['val']['true_value'] >= float(args.min_effective_rb_total)) else float('nan'),
        'test_value_mae_nonzero_true': float(mean_absolute_error(split_payload['test']['true_value'][split_payload['test']['true_value'] >= float(args.min_effective_rb_total)], split_payload['test']['pred_value'][split_payload['test']['true_value'] >= float(args.min_effective_rb_total)])) if np.any(split_payload['test']['true_value'] >= float(args.min_effective_rb_total)) else float('nan'),
    }
    result = {
        'framework': 'PI-JWM',
        'candidate': 'v11',
        'mode': 'base_policy_tournament_cpu',
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
