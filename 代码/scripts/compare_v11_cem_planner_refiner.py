'''CEM-style support-score refiner for PI-JWM v11 candidate.

This planner is a lightweight upper-bound/proposal diagnostic.  It refines
per-edge support scores by sampling top-k masks within each sample-step group,
then evaluates the refined support through the standard PI-JWM action repair
and rollout path.
'''

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))

from pi_jwm.v11_rollout_value_calibrator import freeze_module

from compare_v11_gnn_greedy_scheduler import _fit_value_model, _make_split_payload
from compare_v11_graph_support_generator import EPS, apply_support_generator_repair, make_all_edge_examples, make_all_edge_targets, select_support_indices
from diagnose_v11_counterfactual_value_attribution import collect_raw_actions
from diagnose_v11_rb_total_latent_identifiability import collect_rollout_edge_context, rows_from_context
from diagnose_v11_rb_total_oracle_value_scope import safe_corr, write_json
from diagnose_v11_scheduler_ranked_allocation import resolve_torch_device
from run_v11_rb_total_repair import active_rate_row, evaluate_raw_actions, make_adaptive_dataset, write_csv
from run_v11_rb_total_value_head import RB_DIM, collect_edge_gradient_improvement, limit_indices, load_context_limited


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / 'artifacts/experiments/pi_jwm_v11_cem_planner_refiner_20260627'


def cem_refine_scores(
    coordinates: np.ndarray,
    base_score: np.ndarray,
    top_k: int,
    iterations: int,
    samples_per_group: int,
    elite_frac: float,
    noise_std: float,
    momentum: float,
    seed: int,
) -> np.ndarray:
    coords = np.asarray(coordinates, dtype=np.int64).reshape(-1, 3)
    base = np.asarray(base_score, dtype=np.float32).reshape(-1)
    refined = np.zeros_like(base)
    rng = np.random.default_rng(int(seed))
    groups: dict[tuple[int, int], list[int]] = {}
    for row_idx, (sample, step, _edge) in enumerate(coords):
        groups.setdefault((int(sample), int(step)), []).append(int(row_idx))
    for group_rows in groups.values():
        rows = np.asarray(group_rows, dtype=np.int64)
        score = base[rows].astype(np.float32)
        if float(np.std(score)) > 1e-6:
            mean = (score - float(np.mean(score))) / float(np.std(score))
        else:
            mean = score.copy()
        k = max(1, min(int(top_k), rows.size))
        elite_count = max(1, int(round(int(samples_per_group) * float(elite_frac))))
        for _ in range(int(iterations)):
            sampled = mean[None, :] + rng.normal(0.0, float(noise_std), size=(int(samples_per_group), rows.size)).astype(np.float32)
            top_idx = np.argpartition(-sampled, kth=k - 1, axis=1)[:, :k]
            rewards = np.take_along_axis(score[None, :], top_idx, axis=1).sum(axis=1)
            elite = np.argsort(-rewards, kind='mergesort')[:elite_count]
            freq = np.zeros((rows.size,), dtype=np.float32)
            for sample_idx in elite:
                freq[top_idx[sample_idx]] += 1.0
            freq /= float(elite_count)
            mean = (1.0 - float(momentum)) * mean + float(momentum) * freq
        refined[rows] = mean
    finite = refined[np.isfinite(refined)]
    if finite.size:
        lo = float(np.min(finite))
        hi = float(np.max(finite))
        if hi > lo:
            refined = (refined - lo) / (hi - lo)
    return refined.astype(np.float32)


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
    parser.add_argument('--limit-after-stats', action='store_true')
    parser.add_argument('--streaming-stats', action='store_true')
    parser.add_argument('--stats-chunk-size', type=int, default=512)
    parser.add_argument('--value-target-mode', choices=('abs', 'log', 'residual', 'ratio'), default='abs')
    parser.add_argument('--cem-iterations', type=int, default=3)
    parser.add_argument('--cem-samples-per-group', type=int, default=32)
    parser.add_argument('--cem-elite-frac', type=float, default=0.25)
    parser.add_argument('--cem-noise-std', type=float, default=0.7)
    parser.add_argument('--cem-momentum', type=float, default=0.7)
    parser.add_argument('--top-k', type=int, nargs='+', default=[4, 8, 16])
    parser.add_argument('--selection-score-modes', choices=('cem', 'cem_value', 'cem_gain', 'oracle_support'), nargs='+', default=['cem'])
    parser.add_argument('--selection-group-modes', choices=('fixed', 'baseline_active_count', 'support_threshold'), nargs='+', default=['baseline_active_count'])
    parser.add_argument('--support-thresholds', type=float, nargs='+', default=[0.05])
    parser.add_argument('--blend-alpha', type=float, nargs='+', default=[1.0])
    parser.add_argument('--step-total-cap-scale', type=float, nargs='+', default=[1.1])
    parser.add_argument('--edge-value-cap-scale', type=float, nargs='+', default=[1.15])
    parser.add_argument('--new-edge-value-cap', type=float, nargs='+', default=[2.0, 5.0])
    parser.add_argument('--seed', type=int, default=20260627)
    return parser.parse_args()


def _base_score(payload) -> np.ndarray:
    gain = np.clip(payload.pred_value - payload.baseline_values, 0.0, None)
    return np.log1p(gain).astype(np.float32)


def _selection_score(mode: str, cem_score: np.ndarray, payload) -> np.ndarray:
    if mode == 'cem':
        return cem_score
    if mode == 'cem_value':
        return (cem_score * np.log1p(np.clip(payload.pred_value, 0.0, None))).astype(np.float32)
    if mode == 'cem_gain':
        return (cem_score * np.log1p(np.clip(payload.pred_value - payload.baseline_values, 0.0, None))).astype(np.float32)
    if mode == 'oracle_support':
        return np.asarray(payload.oracle_score, dtype=np.float32)
    raise ValueError(f'unknown score mode: {mode}')


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
    train_examples = make_all_edge_examples(train_actions, train_truth, steps=steps)
    train_edge_improvement = collect_edge_gradient_improvement(world_model, train_base, train_actions, train_truth, stats, summary['config'], device, args.batch_size)
    train_labels, _train_score, train_value = make_all_edge_targets(train_examples, train_truth, train_edge_improvement, float(args.min_effective_rb_total), float(args.min_improvement))
    train_context = collect_rollout_edge_context(world_model, train_base, train_actions, stats, device, args.batch_size)
    train_features = rows_from_context(train_context, train_examples.coordinates)
    value_model = _fit_value_model(args, train_features, train_value, train_examples.baseline_values)
    split_payload = {name: _make_split_payload(args, name, arrays, splits, stats, policy_model, action_scale, value_vocab, world_model, summary, device, value_model, steps) for name in ('val', 'test')}

    cem_scores_by_split = {}
    for split_name, payload in split_payload.items():
        base = _base_score(payload)
        # Use the largest requested top-k for CEM refinement; final evaluation still sweeps all top-k values.
        refine_top_k = max(int(k) for k in args.top_k)
        cem_scores_by_split[split_name] = cem_refine_scores(payload.coordinates, base, refine_top_k, int(args.cem_iterations), int(args.cem_samples_per_group), float(args.cem_elite_frac), float(args.cem_noise_std), float(args.cem_momentum), int(args.seed))

    rows = []
    baseline_rmse_by_split = {}
    for split_name, payload in split_payload.items():
        baseline_predictions = evaluate_raw_actions(payload.baseline_actions, payload.base_dataset, stats, world_model, summary['config'], device, args.batch_size)
        baseline_row = active_rate_row('identity_bc_reference', split_name, baseline_predictions, float('nan'))
        baseline_row.update({'family': 'identity', 'support_model': 'none', 'top_k': 0})
        rows.append(baseline_row)
        baseline_rmse_by_split[split_name] = float(baseline_row['active_rate_rmse'])
        for score_mode in args.selection_score_modes:
            score = _selection_score(str(score_mode), cem_scores_by_split[split_name], payload)
            for top_k in args.top_k:
                for group_mode in args.selection_group_modes:
                    thresholds = args.support_thresholds if str(group_mode) == 'support_threshold' else [float('nan')]
                    for threshold in thresholds:
                        selected = select_support_indices(payload.coordinates, score, int(top_k), str(group_mode), baseline_value=payload.baseline_values, threshold=float(threshold))
                        for alpha in args.blend_alpha:
                            for cap_scale in args.step_total_cap_scale:
                                for edge_cap in args.edge_value_cap_scale:
                                    for new_edge_cap in args.new_edge_value_cap:
                                        actions = apply_support_generator_repair(payload.baseline_actions, payload.coordinates, payload.pred_value, selected, float(alpha), float(cap_scale), float(edge_cap), float(new_edge_cap))
                                        predictions = evaluate_raw_actions(actions, payload.base_dataset, stats, world_model, summary['config'], device, args.batch_size)
                                        threshold_tag = '' if np.isnan(float(threshold)) else f'__thr{float(threshold):g}'
                                        candidate = f'cem_refiner__{score_mode}__g{group_mode}{threshold_tag}__top{int(top_k)}__alpha{float(alpha):g}__cap{float(cap_scale):g}__ecap{float(edge_cap):g}__newcap{float(new_edge_cap):g}'
                                        row = active_rate_row(candidate, split_name, predictions, baseline_rmse_by_split[split_name])
                                        new_support_count = int(np.sum((actions[..., RB_DIM] > EPS) & (payload.baseline_actions[..., RB_DIM] <= EPS)))
                                        row.update({'family': 'cem_planner_refiner', 'support_model': 'cem_score_refiner', 'selection_score_mode': str(score_mode), 'selection_group_mode': str(group_mode), 'support_threshold': None if np.isnan(float(threshold)) else float(threshold), 'top_k': int(top_k), 'alpha': float(alpha), 'step_total_cap_scale': float(cap_scale), 'edge_value_cap_scale': float(edge_cap), 'new_edge_value_cap': float(new_edge_cap), 'selected_count': int(selected.size), 'new_support_count': new_support_count, 'selected_oracle_mass': float(np.sum(payload.oracle_score[selected])) if selected.size else 0.0})
                                        rows.append(row)

    write_csv(args.output_dir / 'cem_planner_refiner_results.csv', rows)
    val_ranked = sorted([row for row in rows if row['split'] == 'val'], key=lambda row: (float(row['active_rate_rmse']), str(row['candidate'])))
    deployable_val_ranked = [row for row in val_ranked if str(row.get('candidate')) != 'identity_bc_reference']
    write_csv(args.output_dir / 'cem_planner_refiner_val_ranked.csv', val_ranked)
    write_csv(args.output_dir / 'cem_planner_refiner_deployable_val_ranked.csv', deployable_val_ranked)
    test_by_candidate = {str(row['candidate']): row for row in rows if row['split'] == 'test'}
    best_val = deployable_val_ranked[0] if deployable_val_ranked else None
    diagnostics = {'train_all_edge_examples': int(train_examples.coordinates.shape[0]), 'train_positive_label_count': int(np.sum(train_labels > 0)), 'val_support_label_count': int(np.sum(split_payload['val'].labels > 0)), 'test_support_label_count': int(np.sum(split_payload['test'].labels > 0)), 'val_cem_score_corr_label': safe_corr(cem_scores_by_split['val'], split_payload['val'].labels), 'test_cem_score_corr_label': safe_corr(cem_scores_by_split['test'], split_payload['test'].labels), 'runtime_seconds': float(time.time() - started)}
    result = {'framework': 'PI-JWM', 'candidate': 'v11', 'mode': 'cem_planner_refiner', 'output_dir': str(args.output_dir), 'command': ' '.join(sys.argv), 'device': str(device), 'diagnostics': diagnostics, 'best_val': best_val, 'matched_test_for_best_val': test_by_candidate.get(str(best_val['candidate'])) if best_val else None}
    write_json(args.output_dir / 'summary.json', result)
    return result


def main() -> None:
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
