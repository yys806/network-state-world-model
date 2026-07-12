'''GNN-guided greedy support scheduler for PI-JWM v11 candidate.

The scorer predicts per-edge utility for each sample-step, optionally smoothing
edge states through a line-graph adjacency.  Decoding still uses the existing
hard support selection, RB repair, and PI-JWM rollout evaluation path.
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
from compare_v11_diffusion_support_generator import make_diffusion_support_targets
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


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / 'artifacts/experiments/pi_jwm_v11_gnn_greedy_scheduler_20260627'


@dataclass(frozen=True)
class GroupedRows:
    features: np.ndarray
    target: np.ndarray
    rank_target: np.ndarray
    coordinates: np.ndarray
    group_count: int
    edge_count: int


@dataclass(frozen=True)
class SplitPayload:
    base_dataset: object
    baseline_actions: np.ndarray
    coordinates: np.ndarray
    baseline_values: np.ndarray
    labels: np.ndarray
    oracle_score: np.ndarray
    true_value: np.ndarray
    features: np.ndarray
    pred_value: np.ndarray


class EdgeGraphScorer(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, layers: int, dropout: float):
        super().__init__()
        self.input = nn.Linear(int(feature_dim), int(hidden_dim))
        self.layers = nn.ModuleList([nn.Linear(int(hidden_dim) * 2, int(hidden_dim)) for _ in range(int(layers))])
        self.dropout = nn.Dropout(float(dropout))
        self.out = nn.Linear(int(hidden_dim), 1)

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.input(features))
        for layer in self.layers:
            neighbor = torch.einsum('ij,bjh->bih', adjacency, h)
            h = F.silu(layer(torch.cat([h, neighbor], dim=-1)))
            h = self.dropout(h)
        return self.out(h).squeeze(-1)


def build_edge_adjacency(edge_count: int, world_model=None, device: torch.device | None = None, dtype=torch.float32) -> torch.Tensor:
    device = device or torch.device('cpu')
    edge_count = int(edge_count)
    adjacency = torch.eye(edge_count, device=device, dtype=dtype)
    src = getattr(world_model, 'edge_src_idx', None)
    dst = getattr(world_model, 'edge_dst_idx', None)
    if src is not None and dst is not None:
        src_t = torch.as_tensor(src, device=device).reshape(-1)[:edge_count]
        dst_t = torch.as_tensor(dst, device=device).reshape(-1)[:edge_count]
        if src_t.numel() == edge_count and dst_t.numel() == edge_count:
            shares = (src_t[:, None] == src_t[None, :]) | (src_t[:, None] == dst_t[None, :]) | (dst_t[:, None] == src_t[None, :]) | (dst_t[:, None] == dst_t[None, :])
            adjacency = shares.to(dtype=dtype)
            adjacency.fill_diagonal_(1.0)
    degree = adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0)
    return adjacency / degree


def group_rows_by_sample_step(
    features: np.ndarray,
    coordinates: np.ndarray,
    target: np.ndarray,
    rank_target: np.ndarray,
) -> GroupedRows:
    features = np.asarray(features, dtype=np.float32)
    coords = np.asarray(coordinates, dtype=np.int64).reshape(-1, 3)
    target = np.asarray(target, dtype=np.float32).reshape(-1)
    rank_target = np.asarray(rank_target, dtype=np.float32).reshape(-1)
    groups: dict[tuple[int, int], list[int]] = {}
    for row_idx, (sample, step, _edge) in enumerate(coords):
        groups.setdefault((int(sample), int(step)), []).append(int(row_idx))
    ordered = sorted(groups)
    if not ordered:
        return GroupedRows(
            features=np.zeros((0, 0, features.shape[1]), dtype=np.float32),
            target=np.zeros((0, 0), dtype=np.float32),
            rank_target=np.zeros((0, 0), dtype=np.float32),
            coordinates=np.zeros((0, 3), dtype=np.int64),
            group_count=0,
            edge_count=0,
        )
    edge_count = len(groups[ordered[0]])
    group_features = []
    group_target = []
    group_rank = []
    ordered_rows = []
    for key in ordered:
        rows = np.asarray(groups[key], dtype=np.int64)
        if rows.shape[0] != edge_count:
            raise ValueError('all sample-step groups must have the same edge count')
        rows = rows[np.argsort(coords[rows, 2], kind='mergesort')]
        ordered_rows.append(rows)
        group_features.append(features[rows])
        group_target.append(target[rows])
        group_rank.append(rank_target[rows])
    flat_rows = np.concatenate(ordered_rows)
    return GroupedRows(
        features=np.asarray(group_features, dtype=np.float32),
        target=np.asarray(group_target, dtype=np.float32),
        rank_target=np.asarray(group_rank, dtype=np.float32),
        coordinates=coords[flat_rows],
        group_count=len(ordered),
        edge_count=edge_count,
    )


def train_gnn_scorer(grouped: GroupedRows, adjacency: torch.Tensor, args: argparse.Namespace, device: torch.device):
    features = grouped.features
    mean = features.reshape(-1, features.shape[-1]).mean(axis=0, keepdims=True)
    std = features.reshape(-1, features.shape[-1]).std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    features_norm = (features - mean) / std
    x = torch.as_tensor(features_norm, dtype=torch.float32, device=device)
    y = torch.as_tensor(grouped.target, dtype=torch.float32, device=device)
    rank = torch.as_tensor(grouped.rank_target, dtype=torch.float32, device=device)
    model = EdgeGraphScorer(features.shape[-1], int(args.gnn_hidden_dim), int(args.gnn_layers), float(args.gnn_dropout)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.gnn_lr), weight_decay=float(args.gnn_weight_decay))
    pos = float(np.sum(grouped.target > 0.5))
    neg = float(grouped.target.size - pos)
    pos_weight = min(float(args.gnn_pos_weight), max(1.0, neg / max(pos, 1.0)))
    sample_weight = 1.0 + rank * float(args.gnn_rank_weight) + y * (pos_weight - 1.0)
    n_groups = x.shape[0]
    history = []
    for epoch in range(int(args.gnn_epochs)):
        perm = torch.randperm(n_groups, device=device)
        losses = []
        for start in range(0, n_groups, int(args.gnn_batch_groups)):
            idx = perm[start:start + int(args.gnn_batch_groups)]
            logits = model(x[idx], adjacency)
            loss = F.binary_cross_entropy_with_logits(logits, y[idx], weight=sample_weight[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        history.append({'epoch': int(epoch + 1), 'loss': float(np.mean(losses)) if losses else float('nan')})
    model.feature_mean = torch.as_tensor(mean.astype(np.float32), device=device)
    model.feature_std = torch.as_tensor(std.astype(np.float32), device=device)
    return model, {'group_count': int(n_groups), 'edge_count': int(grouped.edge_count), 'target_positive_count': int(pos), 'pos_weight_used': float(pos_weight), 'history': history}


@torch.no_grad()
def predict_gnn_scores(model: EdgeGraphScorer, features: np.ndarray, coordinates: np.ndarray, adjacency: torch.Tensor, device: torch.device) -> np.ndarray:
    dummy = np.zeros((np.asarray(features).shape[0],), dtype=np.float32)
    grouped = group_rows_by_sample_step(features, coordinates, dummy, dummy)
    x = torch.as_tensor(grouped.features, dtype=torch.float32, device=device)
    x = (x - model.feature_mean) / model.feature_std
    model.eval()
    scores = []
    for start in range(0, x.shape[0], 64):
        logits = model(x[start:start + 64], adjacency)
        scores.append(torch.sigmoid(logits).detach().cpu().numpy())
    grouped_scores = np.concatenate(scores, axis=0).reshape(-1)
    return grouped_scores.astype(np.float32)


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
    parser.add_argument('--gnn-epochs', type=int, default=3)
    parser.add_argument('--gnn-hidden-dim', type=int, default=128)
    parser.add_argument('--gnn-layers', type=int, default=2)
    parser.add_argument('--gnn-dropout', type=float, default=0.05)
    parser.add_argument('--gnn-lr', type=float, default=3e-4)
    parser.add_argument('--gnn-weight-decay', type=float, default=1e-4)
    parser.add_argument('--gnn-batch-groups', type=int, default=64)
    parser.add_argument('--gnn-pos-weight', type=float, default=80.0)
    parser.add_argument('--gnn-rank-weight', type=float, default=10.0)
    parser.add_argument('--top-k', type=int, nargs='+', default=[4, 8, 16])
    parser.add_argument('--selection-score-modes', choices=('gnn', 'gnn_value', 'gnn_gain', 'oracle_support'), nargs='+', default=['gnn'])
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
    labels, oracle_score, true_value = make_all_edge_targets(examples, truth_actions, edge_improvement, float(args.min_effective_rb_total), float(args.min_improvement))
    context = collect_rollout_edge_context(world_model, base_dataset, baseline_actions, stats, device, args.batch_size)
    features = rows_from_context(context, examples.coordinates)
    raw_pred_value = predict_conservative_value_target(value_model, features, mode='mean', beta=0.0)
    pred_value = invert_value_target(str(args.value_target_mode), raw_pred_value, examples.baseline_values)
    return SplitPayload(base_dataset, baseline_actions, examples.coordinates, examples.baseline_values, labels, oracle_score, true_value, features, pred_value)


def _selection_score(mode: str, gnn_score: np.ndarray, baseline_value: np.ndarray, pred_value: np.ndarray, oracle_score: np.ndarray) -> np.ndarray:
    if mode == 'gnn':
        return np.asarray(gnn_score, dtype=np.float32)
    if mode == 'gnn_value':
        return (np.asarray(gnn_score, dtype=np.float32) * np.log1p(np.clip(pred_value, 0.0, None))).astype(np.float32)
    if mode == 'gnn_gain':
        return (np.asarray(gnn_score, dtype=np.float32) * np.log1p(np.clip(pred_value - baseline_value, 0.0, None))).astype(np.float32)
    if mode == 'oracle_support':
        return np.asarray(oracle_score, dtype=np.float32)
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
    train_labels, train_score, train_value = make_all_edge_targets(train_examples, train_truth, train_edge_improvement, float(args.min_effective_rb_total), float(args.min_improvement))
    train_context = collect_rollout_edge_context(world_model, train_base, train_actions, stats, device, args.batch_size)
    train_features = rows_from_context(train_context, train_examples.coordinates)
    train_rank_target = make_group_rank_targets(train_examples.coordinates, train_score, train_value, mode=str(args.rank_target_mode))
    train_support_target = make_diffusion_support_targets(train_examples.coordinates, train_labels, train_rank_target, int(args.target_top_k))
    grouped = group_rows_by_sample_step(train_features, train_examples.coordinates, train_support_target, train_rank_target)
    adjacency = build_edge_adjacency(grouped.edge_count, world_model, device=device, dtype=torch.float32)
    gnn_model, gnn_diagnostics = train_gnn_scorer(grouped, adjacency, args, device)
    value_model = _fit_value_model(args, train_features, train_value, train_examples.baseline_values)

    split_payload = {name: _make_split_payload(args, name, arrays, splits, stats, policy_model, action_scale, value_vocab, world_model, summary, device, value_model, steps) for name in ('val', 'test')}
    gnn_scores = {name: predict_gnn_scores(gnn_model, payload.features, payload.coordinates, adjacency, device) for name, payload in split_payload.items()}

    rows = []
    baseline_rmse_by_split = {}
    for split_name, payload in split_payload.items():
        baseline_predictions = evaluate_raw_actions(payload.baseline_actions, payload.base_dataset, stats, world_model, summary['config'], device, args.batch_size)
        baseline_row = active_rate_row('identity_bc_reference', split_name, baseline_predictions, float('nan'))
        baseline_row.update({'family': 'identity', 'support_model': 'none', 'top_k': 0})
        rows.append(baseline_row)
        baseline_rmse_by_split[split_name] = float(baseline_row['active_rate_rmse'])
        for score_mode in args.selection_score_modes:
            score = _selection_score(str(score_mode), gnn_scores[split_name], payload.baseline_values, payload.pred_value, payload.oracle_score)
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
                                        candidate = f'gnn_greedy__{score_mode}__g{group_mode}{threshold_tag}__top{int(top_k)}__alpha{float(alpha):g}__cap{float(cap_scale):g}__ecap{float(edge_cap):g}__newcap{float(new_edge_cap):g}'
                                        row = active_rate_row(candidate, split_name, predictions, baseline_rmse_by_split[split_name])
                                        new_support_count = int(np.sum((actions[..., RB_DIM] > EPS) & (payload.baseline_actions[..., RB_DIM] <= EPS)))
                                        row.update({'family': 'gnn_guided_greedy_scheduler', 'support_model': 'edge_graph_scorer', 'selection_score_mode': str(score_mode), 'selection_group_mode': str(group_mode), 'support_threshold': None if np.isnan(float(threshold)) else float(threshold), 'top_k': int(top_k), 'alpha': float(alpha), 'step_total_cap_scale': float(cap_scale), 'edge_value_cap_scale': float(edge_cap), 'new_edge_value_cap': float(new_edge_cap), 'selected_count': int(selected.size), 'new_support_count': new_support_count, 'selected_oracle_mass': float(np.sum(payload.oracle_score[selected])) if selected.size else 0.0})
                                        rows.append(row)

    write_csv(args.output_dir / 'gnn_greedy_scheduler_results.csv', rows)
    val_ranked = sorted([row for row in rows if row['split'] == 'val'], key=lambda row: (float(row['active_rate_rmse']), str(row['candidate'])))
    deployable_val_ranked = [row for row in val_ranked if str(row.get('candidate')) != 'identity_bc_reference']
    write_csv(args.output_dir / 'gnn_greedy_scheduler_val_ranked.csv', val_ranked)
    write_csv(args.output_dir / 'gnn_greedy_scheduler_deployable_val_ranked.csv', deployable_val_ranked)
    test_by_candidate = {str(row['candidate']): row for row in rows if row['split'] == 'test'}
    best_val = deployable_val_ranked[0] if deployable_val_ranked else None
    diagnostics = {'train_all_edge_examples': int(train_examples.coordinates.shape[0]), 'train_positive_label_count': int(np.sum(train_labels > 0)), 'train_support_target_positive_count': int(np.sum(train_support_target > 0.5)), 'gnn': gnn_diagnostics, 'val_support_label_count': int(np.sum(split_payload['val'].labels > 0)), 'test_support_label_count': int(np.sum(split_payload['test'].labels > 0)), 'val_gnn_score_corr_label': safe_corr(gnn_scores['val'], split_payload['val'].labels), 'test_gnn_score_corr_label': safe_corr(gnn_scores['test'], split_payload['test'].labels), 'runtime_seconds': float(time.time() - started)}
    result = {'framework': 'PI-JWM', 'candidate': 'v11', 'mode': 'gnn_guided_greedy_scheduler', 'output_dir': str(args.output_dir), 'command': ' '.join(sys.argv), 'device': str(device), 'diagnostics': diagnostics, 'best_val': best_val, 'matched_test_for_best_val': test_by_candidate.get(str(best_val['candidate'])) if best_val else None}
    write_json(args.output_dir / 'summary.json', result)
    return result


def main() -> None:
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
