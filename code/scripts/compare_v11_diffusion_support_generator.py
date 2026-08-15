'''Discrete denoising support-mask generator for PI-JWM v11 candidate.

This is a GPU-first method family, but the script supports tiny CPU smoke tests
to verify data flow, hard projection, and metric logging before spending GPU.
The denoiser learns to recover useful support masks from noisy masks, then the
same constrained action repair/evaluation path is reused.
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
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))

from pi_jwm.v11_rollout_value_calibrator import freeze_module

from diagnose_v11_counterfactual_value_attribution import collect_raw_actions
from compare_v11_graph_support_generator import (
    EPS,
    apply_support_generator_repair,
    make_all_edge_examples,
    make_all_edge_targets,
    make_group_rank_targets,
    select_support_indices,
)
from diagnose_v11_rb_total_latent_identifiability import (
    build_models,
    collect_rollout_edge_context,
    invert_value_target,
    make_value_target,
    rows_from_context,
)
from diagnose_v11_rb_total_oracle_value_scope import safe_corr, write_json
from diagnose_v11_scheduler_ranked_allocation import predict_conservative_value_target, resolve_torch_device
from run_v11_rb_total_repair import active_rate_row, evaluate_raw_actions, make_adaptive_dataset, write_csv
from run_v11_rb_total_value_head import (
    RB_DIM,
    collect_edge_gradient_improvement,
    limit_indices,
    load_context_limited,
)


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / 'artifacts/experiments/pi_jwm_v11_diffusion_support_generator_20260627'


@dataclass(frozen=True)
class SplitPayload:
    base_dataset: object
    baseline_actions: np.ndarray
    truth_actions: np.ndarray
    coordinates: np.ndarray
    baseline_values: np.ndarray
    labels: np.ndarray
    oracle_score: np.ndarray
    true_value: np.ndarray
    features: np.ndarray
    pred_value: np.ndarray


class DenoisingSupportMLP(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(feature_dim) + 2, int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(self, features: torch.Tensor, noisy_mask: torch.Tensor, noise_level: torch.Tensor) -> torch.Tensor:
        x = torch.cat([features, noisy_mask[:, None], noise_level[:, None]], dim=1)
        return self.net(x).squeeze(1)


def make_diffusion_support_targets(
    coordinates: np.ndarray,
    labels: np.ndarray,
    rank_target: np.ndarray,
    target_top_k: int,
) -> np.ndarray:
    coords = np.asarray(coordinates, dtype=np.int64).reshape(-1, 3)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    rank_target = np.asarray(rank_target, dtype=np.float32).reshape(-1)
    target = (labels > 0).astype(np.float32)
    groups: dict[tuple[int, int], list[int]] = {}
    for row_idx, (sample, step, _edge) in enumerate(coords):
        groups.setdefault((int(sample), int(step)), []).append(int(row_idx))
    for group_rows in groups.values():
        rows = np.asarray(group_rows, dtype=np.int64)
        positive_rank = rows[rank_target[rows] > 0.0]
        if positive_rank.size == 0:
            continue
        order = positive_rank[np.argsort(-rank_target[positive_rank], kind='mergesort')]
        target[order[: int(target_top_k)]] = 1.0
    return target.astype(np.float32)


def make_noisy_mask(target: torch.Tensor, noise_level: torch.Tensor, false_positive_scale: float) -> torch.Tensor:
    keep = torch.rand_like(target) > noise_level
    false_positive = torch.rand_like(target) < (noise_level * float(false_positive_scale))
    noisy = torch.where(target > 0.5, keep.float(), false_positive.float())
    return noisy


def train_denoiser(
    features: np.ndarray,
    target: np.ndarray,
    rank_target: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[DenoisingSupportMLP, dict]:
    rng = np.random.default_rng(int(args.seed))
    features_arr = np.asarray(features, dtype=np.float32)
    target_arr = np.asarray(target, dtype=np.float32).reshape(-1)
    rank_arr = np.asarray(rank_target, dtype=np.float32).reshape(-1)
    if int(args.diffusion_max_train_rows) > 0 and features_arr.shape[0] > int(args.diffusion_max_train_rows):
        positive = np.flatnonzero(target_arr > 0.5)
        negative = np.flatnonzero(target_arr <= 0.5)
        keep_positive = positive
        remaining = max(0, int(args.diffusion_max_train_rows) - keep_positive.shape[0])
        if negative.shape[0] > remaining:
            keep_negative = rng.choice(negative, size=remaining, replace=False)
        else:
            keep_negative = negative
        rows = np.concatenate([keep_positive, keep_negative]).astype(np.int64)
        rng.shuffle(rows)
        features_arr = features_arr[rows]
        target_arr = target_arr[rows]
        rank_arr = rank_arr[rows]

    mean = features_arr.mean(axis=0, keepdims=True)
    std = features_arr.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    features_norm = (features_arr - mean) / std

    model = DenoisingSupportMLP(features_norm.shape[1], int(args.diffusion_hidden_dim), float(args.diffusion_dropout)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.diffusion_lr), weight_decay=float(args.diffusion_weight_decay))
    x = torch.as_tensor(features_norm, dtype=torch.float32, device=device)
    y = torch.as_tensor(target_arr, dtype=torch.float32, device=device)
    rank = torch.as_tensor(rank_arr, dtype=torch.float32, device=device)
    pos_count = float(np.sum(target_arr > 0.5))
    neg_count = float(target_arr.shape[0] - pos_count)
    pos_weight_value = min(float(args.diffusion_pos_weight), max(1.0, neg_count / max(pos_count, 1.0)))
    sample_weight = 1.0 + rank * float(args.diffusion_rank_weight)
    sample_weight = sample_weight + y * (pos_weight_value - 1.0)
    n_rows = x.shape[0]
    history = []
    for epoch in range(int(args.diffusion_epochs)):
        perm = torch.randperm(n_rows, device=device)
        losses = []
        for start in range(0, n_rows, int(args.diffusion_batch_rows)):
            idx = perm[start : start + int(args.diffusion_batch_rows)]
            noise = torch.rand((idx.numel(),), dtype=torch.float32, device=device) * float(args.diffusion_max_noise)
            noisy_mask = make_noisy_mask(y[idx], noise, float(args.diffusion_false_positive_scale))
            logits = model(x[idx], noisy_mask, noise)
            loss = F.binary_cross_entropy_with_logits(logits, y[idx], weight=sample_weight[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append({'epoch': int(epoch + 1), 'loss': float(np.mean(losses)) if losses else float('nan')})
    model.feature_mean = torch.as_tensor(mean.astype(np.float32), device=device)
    model.feature_std = torch.as_tensor(std.astype(np.float32), device=device)
    diagnostics = {
        'train_rows_used': int(n_rows),
        'target_positive_count': int(np.sum(target_arr > 0.5)),
        'pos_weight_used': float(pos_weight_value),
        'history': history,
    }
    return model, diagnostics


@torch.no_grad()
def sample_denoiser_scores(model: DenoisingSupportMLP, features: np.ndarray, args: argparse.Namespace, device: torch.device) -> np.ndarray:
    model.eval()
    features_arr = np.asarray(features, dtype=np.float32)
    x = torch.as_tensor(features_arr, dtype=torch.float32, device=device)
    x = (x - model.feature_mean) / model.feature_std
    current = torch.full((x.shape[0],), float(args.diffusion_initial_mask_prob), dtype=torch.float32, device=device)
    for step in range(int(args.diffusion_sampling_steps), 0, -1):
        noise_level = torch.full((x.shape[0],), float(step) / float(max(int(args.diffusion_sampling_steps), 1)), dtype=torch.float32, device=device)
        logits = model(x, current, noise_level)
        probs = torch.sigmoid(logits)
        current = probs
    return current.detach().cpu().numpy().astype(np.float32)


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
    parser.add_argument('--rank-target-mode', choices=('gain', 'gain_norm', 'value_gain_norm'), default='gain_norm')
    parser.add_argument('--target-top-k', type=int, default=4)
    parser.add_argument('--diffusion-epochs', type=int, default=3)
    parser.add_argument('--diffusion-hidden-dim', type=int, default=128)
    parser.add_argument('--diffusion-dropout', type=float, default=0.05)
    parser.add_argument('--diffusion-lr', type=float, default=3e-4)
    parser.add_argument('--diffusion-weight-decay', type=float, default=1e-4)
    parser.add_argument('--diffusion-batch-rows', type=int, default=4096)
    parser.add_argument('--diffusion-max-train-rows', type=int, default=120000)
    parser.add_argument('--diffusion-pos-weight', type=float, default=80.0)
    parser.add_argument('--diffusion-rank-weight', type=float, default=10.0)
    parser.add_argument('--diffusion-max-noise', type=float, default=0.95)
    parser.add_argument('--diffusion-false-positive-scale', type=float, default=0.25)
    parser.add_argument('--diffusion-sampling-steps', type=int, default=4)
    parser.add_argument('--diffusion-initial-mask-prob', type=float, default=0.5)
    parser.add_argument('--top-k', type=int, nargs='+', default=[4, 8, 16])
    parser.add_argument('--selection-score-modes', choices=('diffusion', 'diffusion_value', 'diffusion_gain', 'oracle_support'), nargs='+', default=['diffusion'])
    parser.add_argument('--selection-group-modes', choices=('fixed', 'baseline_active_count', 'support_threshold'), nargs='+', default=['baseline_active_count'])
    parser.add_argument('--support-thresholds', type=float, nargs='+', default=[0.05])
    parser.add_argument('--blend-alpha', type=float, nargs='+', default=[1.0])
    parser.add_argument('--step-total-cap-scale', type=float, nargs='+', default=[1.1])
    parser.add_argument('--edge-value-cap-scale', type=float, nargs='+', default=[1.15])
    parser.add_argument('--new-edge-value-cap', type=float, nargs='+', default=[2.0, 5.0])
    parser.add_argument('--seed', type=int, default=20260627)
    return parser.parse_args()


def _fit_value_model(args, features: np.ndarray, true_values: np.ndarray, baseline_values: np.ndarray):
    factories = build_models(seed=int(args.seed), rf_trees=50)
    value_model = factories['rf']()
    rows = np.flatnonzero(np.asarray(true_values).reshape(-1) >= float(args.min_effective_rb_total))
    if rows.size == 0:
        rows = np.arange(features.shape[0], dtype=np.int64)
    target = make_value_target(str(args.value_target_mode), true_values[rows], baseline_values[rows])
    value_model.fit(features[rows], target)
    return value_model


def _make_split_payload(args, split_name, arrays, splits, stats, policy_model, action_scale, value_vocab, world_model, summary, device, value_model, steps) -> SplitPayload:
    base_dataset, adaptive_dataset = make_adaptive_dataset(args, arrays, splits[split_name], stats, policy_model, action_scale, value_vocab, device, splits['train'])
    baseline_actions, truth_actions = collect_raw_actions(adaptive_dataset, stats)
    examples = make_all_edge_examples(baseline_actions, truth_actions, steps=steps)
    edge_improvement = collect_edge_gradient_improvement(world_model, base_dataset, baseline_actions, truth_actions, stats, summary['config'], device, args.batch_size)
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
    return SplitPayload(
        base_dataset=base_dataset,
        baseline_actions=baseline_actions,
        truth_actions=truth_actions,
        coordinates=examples.coordinates,
        baseline_values=examples.baseline_values,
        labels=labels,
        oracle_score=oracle_score,
        true_value=true_value,
        features=features,
        pred_value=pred_value,
    )


def _selection_score(mode: str, diffusion_score: np.ndarray, baseline_value: np.ndarray, pred_value: np.ndarray, oracle_score: np.ndarray) -> np.ndarray:
    diffusion_score = np.asarray(diffusion_score, dtype=np.float32).reshape(-1)
    baseline_value = np.asarray(baseline_value, dtype=np.float32).reshape(-1)
    pred_value = np.asarray(pred_value, dtype=np.float32).reshape(-1)
    oracle_score = np.asarray(oracle_score, dtype=np.float32).reshape(-1)
    if mode == 'diffusion':
        return diffusion_score
    if mode == 'diffusion_value':
        return (diffusion_score * np.log1p(np.clip(pred_value, 0.0, None))).astype(np.float32)
    if mode == 'diffusion_gain':
        gain = np.log1p(np.clip(pred_value - baseline_value, 0.0, None))
        return (diffusion_score * gain).astype(np.float32)
    if mode == 'oracle_support':
        return oracle_score
    raise ValueError(f'unknown selection score mode: {mode}')


def run(args: argparse.Namespace) -> dict:
    started = time.time()
    device = resolve_torch_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

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
    train_labels, train_score, train_value = make_all_edge_targets(
        train_examples,
        train_truth,
        train_edge_improvement,
        min_effective_value=float(args.min_effective_rb_total),
        min_improvement=float(args.min_improvement),
    )
    train_context = collect_rollout_edge_context(world_model, train_base, train_actions, stats, device, args.batch_size)
    train_features = rows_from_context(train_context, train_examples.coordinates)
    train_rank_target = make_group_rank_targets(train_examples.coordinates, train_score, train_value, mode=str(args.rank_target_mode))
    train_support_target = make_diffusion_support_targets(train_examples.coordinates, train_labels, train_rank_target, int(args.target_top_k))
    value_model = _fit_value_model(args, train_features, train_value, train_examples.baseline_values)
    denoiser, denoiser_diagnostics = train_denoiser(train_features, train_support_target, train_rank_target, args, device)

    split_payload = {
        name: _make_split_payload(args, name, arrays, splits, stats, policy_model, action_scale, value_vocab, world_model, summary, device, value_model, steps)
        for name in ('val', 'test')
    }
    diffusion_scores = {
        name: sample_denoiser_scores(denoiser, payload.features, args, device)
        for name, payload in split_payload.items()
    }

    rows = []
    baseline_rmse_by_split = {}
    for split_name, payload in split_payload.items():
        baseline_predictions = evaluate_raw_actions(payload.baseline_actions, payload.base_dataset, stats, world_model, summary['config'], device, args.batch_size)
        baseline_row = active_rate_row('identity_bc_reference', split_name, baseline_predictions, float('nan'))
        baseline_row.update({'family': 'identity', 'support_model': 'none', 'top_k': 0})
        rows.append(baseline_row)
        baseline_rmse_by_split[split_name] = float(baseline_row['active_rate_rmse'])

        for score_mode in args.selection_score_modes:
            score = _selection_score(str(score_mode), diffusion_scores[split_name], payload.baseline_values, payload.pred_value, payload.oracle_score)
            for top_k in args.top_k:
                for group_mode in args.selection_group_modes:
                    thresholds = args.support_thresholds if str(group_mode) == 'support_threshold' else [float('nan')]
                    for threshold in thresholds:
                        selected = select_support_indices(payload.coordinates, score, int(top_k), str(group_mode), baseline_value=payload.baseline_values, threshold=float(threshold))
                        for alpha in args.blend_alpha:
                            for cap_scale in args.step_total_cap_scale:
                                for edge_cap in args.edge_value_cap_scale:
                                    for new_edge_cap in args.new_edge_value_cap:
                                        actions = apply_support_generator_repair(
                                            payload.baseline_actions,
                                            payload.coordinates,
                                            payload.pred_value,
                                            selected,
                                            alpha=float(alpha),
                                            step_total_cap_scale=float(cap_scale),
                                            edge_value_cap_scale=float(edge_cap),
                                            new_edge_value_cap=float(new_edge_cap),
                                        )
                                        predictions = evaluate_raw_actions(actions, payload.base_dataset, stats, world_model, summary['config'], device, args.batch_size)
                                        threshold_tag = '' if np.isnan(float(threshold)) else f'__thr{float(threshold):g}'
                                        candidate = f'diffusion_denoiser__{score_mode}__g{group_mode}{threshold_tag}__top{int(top_k)}__alpha{float(alpha):g}__cap{float(cap_scale):g}__ecap{float(edge_cap):g}__newcap{float(new_edge_cap):g}'
                                        row = active_rate_row(candidate, split_name, predictions, baseline_rmse_by_split[split_name])
                                        new_support_count = int(np.sum((actions[..., RB_DIM] > EPS) & (payload.baseline_actions[..., RB_DIM] <= EPS)))
                                        row.update(
                                            {
                                                'family': 'diffusion_support_generator',
                                                'support_model': 'denoising_mlp',
                                                'selection_score_mode': str(score_mode),
                                                'selection_group_mode': str(group_mode),
                                                'support_threshold': None if np.isnan(float(threshold)) else float(threshold),
                                                'top_k': int(top_k),
                                                'alpha': float(alpha),
                                                'step_total_cap_scale': float(cap_scale),
                                                'edge_value_cap_scale': float(edge_cap),
                                                'new_edge_value_cap': float(new_edge_cap),
                                                'selected_count': int(selected.size),
                                                'new_support_count': new_support_count,
                                                'selected_oracle_mass': float(np.sum(payload.oracle_score[selected])) if selected.size else 0.0,
                                            }
                                        )
                                        rows.append(row)

    write_csv(args.output_dir / 'diffusion_support_generator_results.csv', rows)
    val_ranked = sorted([row for row in rows if row['split'] == 'val'], key=lambda row: (float(row['active_rate_rmse']), str(row['candidate'])))
    deployable_val_ranked = [row for row in val_ranked if str(row.get('candidate')) != 'identity_bc_reference']
    write_csv(args.output_dir / 'diffusion_support_generator_val_ranked.csv', val_ranked)
    write_csv(args.output_dir / 'diffusion_support_generator_deployable_val_ranked.csv', deployable_val_ranked)
    test_by_candidate = {str(row['candidate']): row for row in rows if row['split'] == 'test'}
    best_val = deployable_val_ranked[0] if deployable_val_ranked else None
    diagnostics = {
        'train_all_edge_examples': int(train_examples.coordinates.shape[0]),
        'train_positive_label_count': int(np.sum(train_labels > 0)),
        'train_positive_value_count': int(np.sum(train_value >= float(args.min_effective_rb_total))),
        'train_support_target_positive_count': int(np.sum(train_support_target > 0.5)),
        'rank_target_mode': str(args.rank_target_mode),
        'target_top_k': int(args.target_top_k),
        'denoiser': denoiser_diagnostics,
        'val_support_label_count': int(np.sum(split_payload['val'].labels > 0)),
        'test_support_label_count': int(np.sum(split_payload['test'].labels > 0)),
        'val_diffusion_score_corr_label': safe_corr(diffusion_scores['val'], split_payload['val'].labels),
        'test_diffusion_score_corr_label': safe_corr(diffusion_scores['test'], split_payload['test'].labels),
        'runtime_seconds': float(time.time() - started),
    }
    result = {
        'framework': 'PI-JWM',
        'candidate': 'v11',
        'mode': 'diffusion_support_generator',
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
