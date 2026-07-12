"""Diagnose PI-JWM v11 candidate rb_total ranking/value/scope bottlenecks.

This script is diagnostic only.  It intentionally includes non-deployable
oracle variants to isolate whether the current gap is caused by edge ranking,
replacement values, or global top-K selection scope.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from pi_jwm.v11_rollout_value_calibrator import freeze_module

from diagnose_v11_counterfactual_value_attribution import collect_raw_actions
from run_v11_rb_total_repair import active_rate_row, evaluate_raw_actions, make_adaptive_dataset, write_csv
from run_v11_rb_total_value_head import (
    RB_DIM,
    StepCodebook,
    _make_inference_examples,
    _standardize,
    apply_topk_score_repair,
    append_state_features,
    build_step_codebooks,
    collect_edge_gradient_improvement,
    decode_probabilities,
    extract_state_features,
    load_context_limited,
    make_critical_examples,
    make_edge_teacher_labels,
    make_rb_total_examples,
    predict_probabilities,
    predict_scores,
    train_classifier,
    train_pairwise_ranker,
    train_regressor,
)


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts/experiments/pi_jwm_v11_rb_total_oracle_value_scope_diagnostic_20260622"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def select_topk_indices(
    coordinates: np.ndarray,
    scores: np.ndarray,
    top_k: int,
    scope: str,
) -> np.ndarray:
    coords = np.asarray(coordinates, dtype=np.int64).reshape(-1, 3)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if top_k <= 0 or coords.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64)
    if scope == "global":
        return np.argsort(-scores)[: min(int(top_k), coords.shape[0])].astype(np.int64)
    groups: dict[tuple[int, ...], list[int]] = {}
    for idx, (sample, step, _edge) in enumerate(coords):
        if scope == "per_sample":
            key = (int(sample),)
        elif scope == "per_sample_step":
            key = (int(sample), int(step))
        else:
            raise ValueError(f"unknown scope: {scope}")
        groups.setdefault(key, []).append(idx)
    selected = []
    for indices in groups.values():
        group_idx = np.asarray(indices, dtype=np.int64)
        order = group_idx[np.argsort(-scores[group_idx])[: min(int(top_k), group_idx.shape[0])]]
        selected.extend(int(x) for x in order)
    return np.asarray(selected, dtype=np.int64)


def apply_selected_repair(
    actions: np.ndarray,
    coordinates: np.ndarray,
    values: np.ndarray,
    selected: np.ndarray,
) -> np.ndarray:
    repaired = np.asarray(actions, dtype=np.float32).copy()
    coords = np.asarray(coordinates, dtype=np.int64).reshape(-1, 3)
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    for idx in np.asarray(selected, dtype=np.int64).reshape(-1):
        sample, step, edge = coords[int(idx)]
        if step == 0 or actions[sample, step, edge, RB_DIM] <= 1e-9:
            continue
        repaired[sample, step, edge, RB_DIM] = max(float(values[int(idx)]), 0.0)
    repaired[:, 0, :, RB_DIM] = actions[:, 0, :, RB_DIM]
    repaired[..., RB_DIM] = np.where(actions[..., RB_DIM] > 1e-9, np.clip(repaired[..., RB_DIM], 0.0, None), 0.0)
    return repaired


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 2:
        return float("nan")
    a = a[mask]
    b = b[mask]
    if float(np.std(a)) < 1e-12 or float(np.std(b)) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    ranks[order] = np.arange(values.shape[0], dtype=np.float64)
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    del unique
    sums = np.bincount(inverse, weights=ranks)
    avg = sums / counts
    return avg[inverse]


def make_predicted_values(
    examples,
    codebooks: dict[int, StepCodebook],
    value_models: dict[int, torch.nn.Module],
    value_features_by_step: dict[int, dict[str, np.ndarray]],
    split_name: str,
) -> np.ndarray:
    values = np.zeros((examples.coordinates.shape[0],), dtype=np.float32)
    for step, codebook in codebooks.items():
        step_mask = examples.coordinates[:, 1] == step
        if not np.any(step_mask):
            continue
        probs = predict_probabilities(value_models[step], value_features_by_step[step][split_name])
        decoded, _conf = decode_probabilities(probs, codebook, decoder="argmax")
        values[step_mask] = decoded
    return values


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
    parser.add_argument("--rb-bin-count", type=int, default=9)
    parser.add_argument("--min-effective-rb-total", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--head-batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--steps", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--max-train-samples", type=int, default=512)
    parser.add_argument("--max-val-samples", type=int, default=256)
    parser.add_argument("--max-test-samples", type=int, default=256)
    parser.add_argument("--limit-after-stats", action="store_true")
    parser.add_argument("--streaming-stats", action="store_true")
    parser.add_argument("--stats-chunk-size", type=int, default=512)
    parser.add_argument("--include-state-features", action="store_true")
    parser.add_argument("--score-loss-mode", choices=("regression", "pairwise"), default="regression")
    parser.add_argument("--score-pairs-per-epoch", type=int, default=4096)
    parser.add_argument("--score-min-target-gap", type=float, default=1e-6)
    parser.add_argument("--top-k", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128])
    parser.add_argument("--scopes", choices=("global", "per_sample", "per_sample_step"), nargs="+", default=["global", "per_sample", "per_sample_step"])
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict:
    started = time.time()
    device = torch.device("cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    context = load_context_limited(args, device)
    summary, arrays, stats, world_model, policy_model, action_scale, value_vocab, splits = context
    splits = dict(splits)
    if args.limit_after_stats:
        from run_v11_rb_total_value_head import limit_indices

        splits["train"] = limit_indices(splits["train"], args.max_train_samples)
        splits["val"] = limit_indices(splits["val"], args.max_val_samples)
        splits["test"] = limit_indices(splits["test"], args.max_test_samples)
    world_model = freeze_module(world_model)
    policy_model = freeze_module(policy_model)
    steps = tuple(int(step) for step in args.steps)

    train_base, train_dataset = make_adaptive_dataset(args, arrays, splits["train"], stats, policy_model, action_scale, value_vocab, device, splits["train"])
    train_actions, train_truth = collect_raw_actions(train_dataset, stats)
    codebooks = build_step_codebooks(train_truth, steps=steps, bin_count=args.rb_bin_count, min_effective_value=args.min_effective_rb_total)
    critical_train = make_critical_examples(train_actions, train_truth, steps=steps)
    value_train = make_rb_total_examples(train_actions, train_truth, codebooks, steps=steps, min_effective_value=args.min_effective_rb_total)
    train_edge_improvement = collect_edge_gradient_improvement(
        world_model, train_base, train_actions, train_truth, stats, summary["config"], device, args.batch_size
    )
    critical_train = type(critical_train)(
        features=critical_train.features,
        labels=make_edge_teacher_labels(
            critical_train.coordinates,
            train_truth,
            train_edge_improvement,
            min_effective_value=args.min_effective_rb_total,
            min_improvement=0.0,
        ),
        coordinates=critical_train.coordinates,
        baseline_values=critical_train.baseline_values,
        true_values=critical_train.true_values,
    )

    train_state_features = extract_state_features(train_base) if args.include_state_features else None
    critical_train = type(critical_train)(
        features=append_state_features(critical_train.features, train_state_features, critical_train.coordinates),
        labels=critical_train.labels,
        coordinates=critical_train.coordinates,
        baseline_values=critical_train.baseline_values,
        true_values=critical_train.true_values,
    )
    value_train = type(value_train)(
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
        examples = _make_inference_examples(baseline_actions, truth_actions.shape, steps)
        state_features = extract_state_features(base_dataset) if args.include_state_features else None
        examples = type(examples)(
            features=append_state_features(examples.features, state_features, examples.coordinates),
            labels=examples.labels,
            coordinates=examples.coordinates,
            baseline_values=examples.baseline_values,
            true_values=examples.true_values,
        )
        oracle_edge_improvement = collect_edge_gradient_improvement(
            world_model, base_dataset, baseline_actions, truth_actions, stats, summary["config"], device, args.batch_size
        )
        oracle_scores = np.zeros((examples.coordinates.shape[0],), dtype=np.float32)
        true_values = np.zeros((examples.coordinates.shape[0],), dtype=np.float32)
        for row_idx, (sample, step, edge) in enumerate(examples.coordinates):
            oracle_scores[row_idx] = np.log1p(max(float(oracle_edge_improvement[sample, step, edge]), 0.0))
            true_values[row_idx] = truth_actions[sample, step, edge, RB_DIM]
        split_payload[split_name] = {
            "base_dataset": base_dataset,
            "baseline_actions": baseline_actions,
            "truth_actions": truth_actions,
            "examples": examples,
            "oracle_scores": oracle_scores,
            "true_values": true_values,
        }

    feature_sets = [critical_train.features]
    feature_sets.extend(split_payload[name]["examples"].features for name in ("val", "test"))
    standardized = _standardize(*feature_sets)
    critical_train_features = standardized[0]
    split_payload["val"]["critical_features"] = standardized[1]
    split_payload["test"]["critical_features"] = standardized[2]

    value_features_by_step = {}
    value_models = {}
    for step, codebook in codebooks.items():
        mask = value_train.coordinates[:, 1] == step
        features = value_train.features[mask]
        labels = value_train.labels[mask]
        per_split_features = []
        for split_name in ("val", "test"):
            examples = split_payload[split_name]["examples"]
            per_split_features.append(examples.features[examples.coordinates[:, 1] == step])
        standardized_step = _standardize(features, *per_split_features) if features.shape[0] else (features, *per_split_features)
        value_features_by_step[step] = {"train": standardized_step[0], "val": standardized_step[1], "test": standardized_step[2]}
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

    score_targets = np.zeros((critical_train.coordinates.shape[0],), dtype=np.float32)
    for row_idx, (sample, step, edge) in enumerate(critical_train.coordinates):
        score_targets[row_idx] = np.log1p(max(float(train_edge_improvement[sample, step, edge]), 0.0))
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

    rows = []
    diagnostics = {
        "train_critical_examples": int(critical_train.coordinates.shape[0]),
        "train_critical_positive_rate": float(np.mean(critical_train.labels)) if critical_train.labels.size else float("nan"),
        "train_value_examples": int(value_train.coordinates.shape[0]),
        "score_loss_mode": args.score_loss_mode,
    }
    for split_name in ("val", "test"):
        payload = split_payload[split_name]
        baseline_predictions = evaluate_raw_actions(
            payload["baseline_actions"], payload["base_dataset"], stats, world_model, summary["config"], device, args.batch_size
        )
        baseline_row = active_rate_row("identity", split_name, baseline_predictions, float("nan"))
        baseline_rmse = float(baseline_row["active_rate_rmse"])
        baseline_row.update({"rank_source": "identity", "value_source": "identity", "scope": "none", "top_k": 0, "selected_count": 0})
        rows.append(baseline_row)

        examples = payload["examples"]
        predicted_scores = predict_scores(score_model, payload["critical_features"])
        predicted_values = make_predicted_values(examples, codebooks, value_models, value_features_by_step, split_name)
        oracle_scores = payload["oracle_scores"]
        true_values = payload["true_values"]
        diagnostics[f"{split_name}_pred_oracle_pearson"] = safe_corr(predicted_scores, oracle_scores)
        diagnostics[f"{split_name}_pred_oracle_spearman"] = safe_corr(rankdata(predicted_scores), rankdata(oracle_scores))
        diagnostics[f"{split_name}_value_pred_true_pearson"] = safe_corr(predicted_values, true_values)
        diagnostics[f"{split_name}_value_mae_nonzero_true"] = float(np.mean(np.abs(predicted_values[true_values >= args.min_effective_rb_total] - true_values[true_values >= args.min_effective_rb_total]))) if np.any(true_values >= args.min_effective_rb_total) else float("nan")

        variants = [
            ("oracle_rank", oracle_scores, "true_value", true_values),
            ("oracle_rank", oracle_scores, "pred_value", predicted_values),
            ("pred_rank", predicted_scores, "true_value", true_values),
            ("pred_rank", predicted_scores, "pred_value", predicted_values),
        ]
        for rank_name, scores, value_name, values in variants:
            for scope in args.scopes:
                for top_k in args.top_k:
                    selected = select_topk_indices(examples.coordinates, scores, int(top_k), scope)
                    actions = apply_selected_repair(payload["baseline_actions"], examples.coordinates, values, selected)
                    predictions = evaluate_raw_actions(actions, payload["base_dataset"], stats, world_model, summary["config"], device, args.batch_size)
                    candidate = f"{rank_name}__{value_name}__{scope}__top{top_k}"
                    row = active_rate_row(candidate, split_name, predictions, baseline_rmse)
                    selected_oracle_mass = float(np.sum(oracle_scores[selected])) if selected.size else 0.0
                    oracle_top = select_topk_indices(examples.coordinates, oracle_scores, int(top_k), scope)
                    oracle_top_mass = float(np.sum(oracle_scores[oracle_top])) if oracle_top.size else 0.0
                    row.update(
                        {
                            "rank_source": rank_name,
                            "value_source": value_name,
                            "scope": scope,
                            "top_k": int(top_k),
                            "selected_count": int(selected.size),
                            "selected_oracle_mass": selected_oracle_mass,
                            "oracle_mass_ratio_vs_same_scope": float(selected_oracle_mass / oracle_top_mass) if oracle_top_mass > 0 else float("nan"),
                        }
                    )
                    rows.append(row)

    write_csv(args.output_dir / "oracle_value_scope_results.csv", rows)
    val_rows = [row for row in rows if row["split"] == "val"]
    val_ranked = sorted(val_rows, key=lambda row: (float(row["active_rate_rmse"]), str(row["candidate"])))
    write_csv(args.output_dir / "oracle_value_scope_val_ranked.csv", val_ranked)
    diagnostics["runtime_seconds"] = float(time.time() - started)
    diagnostics["best_val_candidate"] = val_ranked[0]["candidate"] if val_ranked else None
    diagnostics["best_val_active_rate_rmse"] = float(val_ranked[0]["active_rate_rmse"]) if val_ranked else float("nan")
    result = {
        "framework": "PI-JWM",
        "candidate": "v11",
        "mode": "rb_total_oracle_value_scope_diagnostic",
        "output_dir": str(args.output_dir),
        "command": " ".join(sys.argv),
        "diagnostics": diagnostics,
        "best_val": val_ranked[0] if val_ranked else None,
    }
    write_json(args.output_dir / "summary.json", result)
    return result


def main() -> None:
    result = run(parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
