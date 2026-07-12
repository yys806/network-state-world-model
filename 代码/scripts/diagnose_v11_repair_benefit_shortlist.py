"""CPU shortlist diagnostic for PI-JWM v11 candidate rb_total repair.

This script tests three deployable repair ideas in one comparable run:

1. benefit/uplift-style edge ranking over first-order repair benefit labels;
2. conformal/selective lower-bound gating that can abstain to identity;
3. support-constrained rb_total value repair using local train-set values.

It is CPU-first and diagnostic.  It does not finalize v11 and it does not use
true future values at inference except in explicitly recorded diagnostics.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from pi_jwm.v11_rollout_value_calibrator import freeze_module

from diagnose_v11_counterfactual_value_attribution import collect_raw_actions
from diagnose_v11_rb_total_latent_identifiability import (
    apply_selected_blend_repair,
    collect_rollout_edge_context,
    rows_from_context,
)
from diagnose_v11_rb_total_oracle_value_scope import rankdata, safe_corr, select_topk_indices, write_json
from run_v11_rb_total_repair import active_rate_row, evaluate_raw_actions, make_adaptive_dataset, write_csv
from run_v11_rb_total_value_head import (
    RB_DIM,
    _make_inference_examples,
    _standardize,
    collect_edge_gradient_improvement,
    limit_indices,
    load_context_limited,
    make_critical_examples,
)


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts/experiments/pi_jwm_v11_repair_benefit_shortlist_20260622"
EPS = 1e-9


def conformal_lower_bound(
    predictions: np.ndarray,
    calibration_predictions: np.ndarray,
    calibration_targets: np.ndarray,
    residual_quantile: float = 0.8,
) -> np.ndarray:
    """Return a conservative lower confidence score from calibration residuals."""
    predictions = np.asarray(predictions, dtype=np.float32).reshape(-1)
    calibration_predictions = np.asarray(calibration_predictions, dtype=np.float32).reshape(-1)
    calibration_targets = np.asarray(calibration_targets, dtype=np.float32).reshape(-1)
    if calibration_predictions.shape[0] != calibration_targets.shape[0]:
        raise ValueError("calibration predictions and targets must have the same row count")
    if calibration_predictions.shape[0] == 0:
        return predictions.astype(np.float32)
    q = float(np.clip(residual_quantile, 0.0, 1.0))
    residual = np.abs(calibration_targets - calibration_predictions)
    penalty = float(np.quantile(residual, q))
    return (predictions - penalty).astype(np.float32)


def select_positive_topk_indices(
    coordinates: np.ndarray,
    scores: np.ndarray,
    top_k: int,
    scope: str,
    min_score: float = 0.0,
) -> np.ndarray:
    """Select top-K per scope after discarding scores below min_score."""
    coordinates = np.asarray(coordinates, dtype=np.int64).reshape(-1, 3)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if coordinates.shape[0] != scores.shape[0]:
        raise ValueError("coordinates and scores must have the same row count")
    if top_k <= 0 or coordinates.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64)
    eligible = np.where(scores > float(min_score))[0]
    if eligible.size == 0:
        return np.zeros((0,), dtype=np.int64)
    local_selected = select_topk_indices(coordinates[eligible], scores[eligible], int(top_k), scope)
    return eligible[local_selected].astype(np.int64)


def predict_support_values(
    train_features: np.ndarray,
    train_values: np.ndarray,
    query_features: np.ndarray,
    k: int = 8,
    fallback_values: np.ndarray | None = None,
    chunk_size: int = 2048,
) -> np.ndarray:
    """Predict supported values by nearest-neighbor median in feature space."""
    train_features = np.asarray(train_features, dtype=np.float32)
    train_values = np.asarray(train_values, dtype=np.float32).reshape(-1)
    query_features = np.asarray(query_features, dtype=np.float32)
    if train_features.ndim != 2 or query_features.ndim != 2:
        raise ValueError("features must be 2D")
    if train_features.shape[0] != train_values.shape[0]:
        raise ValueError("train features and values must have the same row count")
    if fallback_values is None:
        fallback = np.zeros((query_features.shape[0],), dtype=np.float32)
    else:
        fallback = np.asarray(fallback_values, dtype=np.float32).reshape(-1)
        if fallback.shape[0] != query_features.shape[0]:
            raise ValueError("fallback values must match query rows")
    if train_features.shape[0] == 0 or query_features.shape[0] == 0:
        return fallback.astype(np.float32)
    k = max(1, min(int(k), train_features.shape[0]))
    predictions = np.empty((query_features.shape[0],), dtype=np.float32)
    train_norm = np.sum(train_features * train_features, axis=1, keepdims=True).T
    for start in range(0, query_features.shape[0], int(chunk_size)):
        stop = min(start + int(chunk_size), query_features.shape[0])
        query = query_features[start:stop]
        distances = np.sum(query * query, axis=1, keepdims=True) + train_norm - 2.0 * (query @ train_features.T)
        nearest = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
        predictions[start:stop] = np.median(train_values[nearest], axis=1).astype(np.float32)
    return np.clip(predictions, 0.0, None).astype(np.float32)


def make_edge_targets(examples, truth_actions: np.ndarray, edge_improvement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scores = np.zeros((examples.coordinates.shape[0],), dtype=np.float32)
    values = np.zeros((examples.coordinates.shape[0],), dtype=np.float32)
    for row_idx, (sample, step, edge) in enumerate(examples.coordinates):
        scores[row_idx] = np.log1p(max(float(edge_improvement[sample, step, edge]), 0.0))
        values[row_idx] = truth_actions[sample, step, edge, RB_DIM]
    return scores, values


def build_step_load_rows(actions: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    coordinates = np.asarray(coordinates, dtype=np.int64).reshape(-1, 3)
    if coordinates.shape[0] == 0:
        return np.zeros((0, 6), dtype=np.float32)
    rb = np.clip(actions[..., RB_DIM], 0.0, None)
    cpu = np.clip(actions[..., 4], 0.0, None)
    active = (actions > EPS).any(axis=-1).astype(np.float32)
    step_rb = rb.sum(axis=2)
    step_cpu = cpu.sum(axis=2)
    step_active = active.sum(axis=2)
    rows = []
    for sample, step, edge in coordinates:
        base = float(rb[sample, step, edge])
        rows.append(
            [
                float(np.log1p(step_rb[sample, step])),
                float(np.log1p(step_cpu[sample, step])),
                float(np.log1p(step_rb[sample, step] + step_cpu[sample, step])),
                float(step_active[sample, step]),
                float(np.log1p(base)),
                float(base / max(float(step_rb[sample, step]), 1.0)),
            ]
        )
    return np.asarray(rows, dtype=np.float32)


def make_feature_matrix(action_features: np.ndarray, latent_rows: np.ndarray, load_rows: np.ndarray, feature_set: str) -> np.ndarray:
    parts = []
    if feature_set in {"action", "action_latent", "action_load", "action_latent_load"}:
        parts.append(np.asarray(action_features, dtype=np.float32))
    if feature_set in {"latent", "action_latent", "latent_load", "action_latent_load"}:
        parts.append(np.asarray(latent_rows, dtype=np.float32))
    if feature_set in {"load", "action_load", "latent_load", "action_latent_load"}:
        parts.append(np.asarray(load_rows, dtype=np.float32))
    if not parts:
        raise ValueError(f"unknown feature set: {feature_set}")
    return np.concatenate(parts, axis=1).astype(np.float32)


def fit_benefit_model(kind: str, features: np.ndarray, targets: np.ndarray, seed: int, rf_trees: int):
    if kind == "rf":
        model = RandomForestRegressor(
            n_estimators=int(rf_trees),
            min_samples_leaf=5,
            max_features="sqrt",
            random_state=int(seed),
            n_jobs=-1,
        )
    elif kind == "hgb":
        model = HistGradientBoostingRegressor(
            max_iter=160,
            learning_rate=0.05,
            l2_regularization=0.01,
            min_samples_leaf=20,
            random_state=int(seed),
        )
    else:
        raise ValueError(f"unknown model kind: {kind}")
    model.fit(features, targets)
    return model


def diagnostic_candidate_name(rank_mode: str, value_mode: str, scope: str, top_k: int, support_k: int) -> str:
    return f"diagnostic_only__{rank_mode}__{value_mode}__support{int(support_k)}__{scope}__top{int(top_k)}"

def split_fit_calibration(coordinates: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    coordinates = np.asarray(coordinates, dtype=np.int64).reshape(-1, 3)
    if coordinates.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(coordinates.shape[0])
    cut = max(1, int(round(order.shape[0] * 0.7)))
    if cut >= order.shape[0]:
        cut = max(1, order.shape[0] - 1)
    return np.sort(order[:cut]).astype(np.int64), np.sort(order[cut:]).astype(np.int64)


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
    parser.add_argument("--min-effective-rb-total", type=float, default=1.0)
    parser.add_argument("--steps", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--max-train-samples", type=int, default=512)
    parser.add_argument("--max-val-samples", type=int, default=256)
    parser.add_argument("--max-test-samples", type=int, default=256)
    parser.add_argument("--limit-after-stats", action="store_true")
    parser.add_argument("--streaming-stats", action="store_true")
    parser.add_argument("--stats-chunk-size", type=int, default=512)
    parser.add_argument("--top-k", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument("--scopes", choices=("per_sample", "per_sample_step", "global"), nargs="+", default=["per_sample", "per_sample_step"])
    parser.add_argument("--feature-sets", choices=("action", "latent", "load", "action_latent", "action_load", "latent_load", "action_latent_load"), nargs="+", default=["action_latent_load"])
    parser.add_argument("--model-kinds", choices=("rf", "hgb"), nargs="+", default=["rf"])
    parser.add_argument("--rf-trees", type=int, default=100)
    parser.add_argument("--support-k", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--residual-quantile", type=float, nargs="+", default=[0.5, 0.8])
    parser.add_argument("--blend-alpha", type=float, nargs="+", default=[1.0])
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--include-diagnostic-oracles", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict:
    started = time.time()
    device = torch.device("cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    context = load_context_limited(args, device)
    summary, arrays, stats, world_model, policy_model, action_scale, value_vocab, splits = context
    splits = dict(splits)
    if args.limit_after_stats:
        splits["train"] = limit_indices(splits["train"], args.max_train_samples)
        splits["val"] = limit_indices(splits["val"], args.max_val_samples)
        splits["test"] = limit_indices(splits["test"], args.max_test_samples)
    world_model = freeze_module(world_model)
    policy_model = freeze_module(policy_model)
    steps = tuple(int(step) for step in args.steps)

    train_base, train_dataset = make_adaptive_dataset(args, arrays, splits["train"], stats, policy_model, action_scale, value_vocab, device, splits["train"])
    train_actions, train_truth = collect_raw_actions(train_dataset, stats)
    train_examples = make_critical_examples(train_actions, train_truth, steps=steps)
    train_edge_improvement = collect_edge_gradient_improvement(
        world_model, train_base, train_actions, train_truth, stats, summary["config"], device, args.batch_size
    )
    train_score, train_value = make_edge_targets(train_examples, train_truth, train_edge_improvement)
    train_latent_context = collect_rollout_edge_context(world_model, train_base, train_actions, stats, device, args.batch_size)
    train_latent = rows_from_context(train_latent_context, train_examples.coordinates)
    train_load = build_step_load_rows(train_actions, train_examples.coordinates)

    split_payload = {}
    for split_name in ("val", "test"):
        base_dataset, adaptive_dataset = make_adaptive_dataset(args, arrays, splits[split_name], stats, policy_model, action_scale, value_vocab, device, splits["train"])
        baseline_actions, truth_actions = collect_raw_actions(adaptive_dataset, stats)
        examples = _make_inference_examples(baseline_actions, truth_actions.shape, steps)
        edge_improvement = collect_edge_gradient_improvement(
            world_model, base_dataset, baseline_actions, truth_actions, stats, summary["config"], device, args.batch_size
        )
        oracle_score, true_value = make_edge_targets(examples, truth_actions, edge_improvement)
        latent_context = collect_rollout_edge_context(world_model, base_dataset, baseline_actions, stats, device, args.batch_size)
        latent_rows = rows_from_context(latent_context, examples.coordinates)
        load_rows = build_step_load_rows(baseline_actions, examples.coordinates)
        split_payload[split_name] = {
            "base_dataset": base_dataset,
            "baseline_actions": baseline_actions,
            "examples": examples,
            "oracle_score": oracle_score,
            "true_value": true_value,
            "latent": latent_rows,
            "load": load_rows,
        }

    fit_idx, calib_idx = split_fit_calibration(train_examples.coordinates, args.seed)
    rows = []
    diagnostics = {
        "train_examples": int(train_examples.coordinates.shape[0]),
        "fit_examples": int(fit_idx.shape[0]),
        "calibration_examples": int(calib_idx.shape[0]),
        "train_positive_score_count": int(np.sum(train_score > 0.0)),
        "train_positive_value_count": int(np.sum(train_value >= float(args.min_effective_rb_total))),
        "feature_sets": list(args.feature_sets),
        "model_kinds": list(args.model_kinds),
    }

    for feature_set in args.feature_sets:
        train_features_raw = make_feature_matrix(train_examples.features, train_latent, train_load, feature_set)
        split_features_raw = {}
        for split_name, payload in split_payload.items():
            split_features_raw[split_name] = make_feature_matrix(
                payload["examples"].features,
                payload["latent"],
                payload["load"],
                feature_set,
            )
        standardized = _standardize(train_features_raw, *(split_features_raw[name] for name in ("val", "test")))
        train_features = standardized[0]
        split_features = {"val": standardized[1], "test": standardized[2]}

        for model_kind in args.model_kinds:
            model = fit_benefit_model(model_kind, train_features[fit_idx], train_score[fit_idx], args.seed, args.rf_trees)
            calib_pred = np.asarray(model.predict(train_features[calib_idx]), dtype=np.float32)
            calib_target = train_score[calib_idx]
            positive_fit = fit_idx[train_value[fit_idx] >= float(args.min_effective_rb_total)]
            support_features = train_features[positive_fit]
            support_values = train_value[positive_fit]

            for split_name, payload in split_payload.items():
                baseline_predictions = evaluate_raw_actions(
                    payload["baseline_actions"], payload["base_dataset"], stats, world_model, summary["config"], device, args.batch_size
                )
                baseline_row = active_rate_row("identity", split_name, baseline_predictions, float("nan"))
                baseline_rmse = float(baseline_row["active_rate_rmse"])
                baseline_row.update({"feature_set": feature_set, "model": model_kind, "method": "identity", "scope": "none", "top_k": 0})
                rows.append(baseline_row)

                pred_score = np.asarray(model.predict(split_features[split_name]), dtype=np.float32)
                diagnostics[f"{split_name}_{feature_set}_{model_kind}_score_pearson"] = safe_corr(pred_score, payload["oracle_score"])
                diagnostics[f"{split_name}_{feature_set}_{model_kind}_score_spearman"] = safe_corr(rankdata(pred_score), rankdata(payload["oracle_score"]))
                positive_mask = payload["true_value"] >= float(args.min_effective_rb_total)

                for support_k in args.support_k:
                    support_value = predict_support_values(
                        support_features,
                        support_values,
                        split_features[split_name],
                        k=int(support_k),
                        fallback_values=payload["examples"].baseline_values,
                    )
                    diagnostics[f"{split_name}_{feature_set}_{model_kind}_support{support_k}_value_mae_nonzero_true"] = (
                        float(mean_absolute_error(payload["true_value"][positive_mask], support_value[positive_mask]))
                        if np.any(positive_mask)
                        else float("nan")
                    )
                    if args.include_diagnostic_oracles:
                        for scope in args.scopes:
                            for top_k in args.top_k:
                                oracle_selected = select_topk_indices(payload["examples"].coordinates, payload["oracle_score"], int(top_k), scope)
                                support_actions = apply_selected_blend_repair(
                                    payload["baseline_actions"],
                                    payload["examples"].coordinates,
                                    support_value,
                                    oracle_selected,
                                    alpha=1.0,
                                )
                                support_predictions = evaluate_raw_actions(
                                    support_actions, payload["base_dataset"], stats, world_model, summary["config"], device, args.batch_size
                                )
                                support_candidate = diagnostic_candidate_name("oracle_rank", "support_value", scope, int(top_k), int(support_k))
                                support_row = active_rate_row(support_candidate, split_name, support_predictions, baseline_rmse)
                                support_row.update(
                                    {
                                        "feature_set": feature_set,
                                        "model": model_kind,
                                        "method": "diagnostic_only_oracle_rank_support_value",
                                        "support_k": int(support_k),
                                        "score_mode": "oracle_rank",
                                        "scope": scope,
                                        "top_k": int(top_k),
                                        "alpha": 1.0,
                                        "selected_count": int(oracle_selected.size),
                                        "selected_oracle_mass": float(np.sum(payload["oracle_score"][oracle_selected])) if oracle_selected.size else 0.0,
                                    }
                                )
                                rows.append(support_row)

                                pred_selected = select_topk_indices(payload["examples"].coordinates, pred_score, int(top_k), scope)
                                true_actions = apply_selected_blend_repair(
                                    payload["baseline_actions"],
                                    payload["examples"].coordinates,
                                    payload["true_value"],
                                    pred_selected,
                                    alpha=1.0,
                                )
                                true_predictions = evaluate_raw_actions(
                                    true_actions, payload["base_dataset"], stats, world_model, summary["config"], device, args.batch_size
                                )
                                true_candidate = diagnostic_candidate_name("pred_rank", "true_value", scope, int(top_k), int(support_k))
                                true_row = active_rate_row(true_candidate, split_name, true_predictions, baseline_rmse)
                                true_row.update(
                                    {
                                        "feature_set": feature_set,
                                        "model": model_kind,
                                        "method": "diagnostic_only_pred_rank_true_value",
                                        "support_k": int(support_k),
                                        "score_mode": "pred_rank_true_value",
                                        "scope": scope,
                                        "top_k": int(top_k),
                                        "alpha": 1.0,
                                        "selected_count": int(pred_selected.size),
                                        "selected_oracle_mass": float(np.sum(payload["oracle_score"][pred_selected])) if pred_selected.size else 0.0,
                                    }
                                )
                                rows.append(true_row)
                    for score_mode in ("raw", "lcb"):
                        score_variants = []
                        if score_mode == "raw":
                            score_variants.append(("raw_q0", pred_score))
                        else:
                            for q in args.residual_quantile:
                                score_variants.append((f"lcb_q{float(q):g}", conformal_lower_bound(pred_score, calib_pred, calib_target, q)))
                        for score_name, selection_score in score_variants:
                            for scope in args.scopes:
                                for top_k in args.top_k:
                                    if score_mode == "raw":
                                        selected = select_topk_indices(payload["examples"].coordinates, selection_score, int(top_k), scope)
                                    else:
                                        selected = select_positive_topk_indices(payload["examples"].coordinates, selection_score, int(top_k), scope, min_score=0.0)
                                    for alpha in args.blend_alpha:
                                        actions = apply_selected_blend_repair(
                                            payload["baseline_actions"],
                                            payload["examples"].coordinates,
                                            support_value,
                                            selected,
                                            alpha=float(alpha),
                                        )
                                        predictions = evaluate_raw_actions(actions, payload["base_dataset"], stats, world_model, summary["config"], device, args.batch_size)
                                        candidate = (
                                            f"{feature_set}__{model_kind}__support{support_k}__{score_name}__"
                                            f"{scope}__top{int(top_k)}__alpha{float(alpha):g}"
                                        )
                                        row = active_rate_row(candidate, split_name, predictions, baseline_rmse)
                                        row.update(
                                            {
                                                "feature_set": feature_set,
                                                "model": model_kind,
                                                "method": "support_benefit" if score_mode == "raw" else "support_selective_lcb",
                                                "support_k": int(support_k),
                                                "score_mode": score_name,
                                                "scope": scope,
                                                "top_k": int(top_k),
                                                "alpha": float(alpha),
                                                "selected_count": int(selected.size),
                                                "selected_oracle_mass": float(np.sum(payload["oracle_score"][selected])) if selected.size else 0.0,
                                            }
                                        )
                                        rows.append(row)

    write_csv(args.output_dir / "repair_benefit_shortlist_results.csv", rows)
    val_ranked = sorted([row for row in rows if row["split"] == "val"], key=lambda row: (float(row["active_rate_rmse"]), str(row["candidate"])))
    write_csv(args.output_dir / "repair_benefit_shortlist_val_ranked.csv", val_ranked)
    test_by_candidate = {str(row["candidate"]): row for row in rows if row["split"] == "test"}
    best_val = val_ranked[0] if val_ranked else None
    matched_test = test_by_candidate.get(str(best_val["candidate"])) if best_val else None
    diagnostics["runtime_seconds"] = float(time.time() - started)
    result = {
        "framework": "PI-JWM",
        "candidate": "v11",
        "mode": "repair_benefit_shortlist",
        "output_dir": str(args.output_dir),
        "command": " ".join(sys.argv),
        "diagnostics": diagnostics,
        "best_val": best_val,
        "matched_test_for_best_val": matched_test,
    }
    write_json(args.output_dir / "summary.json", result)
    return result


def main() -> None:
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
