"""CPU tree baselines for PI-JWM v11 candidate rb_total identifiability.

This is a diagnostic, not a production strategy.  It tests whether stronger
nonlinear tabular models can recover oracle edge-gradient ranking or rb_total
values from the current deployable features.
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
from diagnose_v11_rb_total_oracle_value_scope import (
    apply_selected_repair,
    rankdata,
    safe_corr,
    select_topk_indices,
    write_json,
)
from run_v11_rb_total_repair import active_rate_row, evaluate_raw_actions, make_adaptive_dataset, write_csv
from run_v11_rb_total_value_head import (
    RB_DIM,
    _make_inference_examples,
    append_state_features,
    collect_edge_gradient_improvement,
    extract_state_features,
    load_context_limited,
    make_critical_examples,
)


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts/experiments/pi_jwm_v11_rb_total_tree_identifiability_20260622"


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
    parser.add_argument("--include-state-features", action="store_true")
    parser.add_argument("--top-k", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128])
    parser.add_argument("--scopes", choices=("global", "per_sample", "per_sample_step"), nargs="+", default=["global", "per_sample", "per_sample_step"])
    parser.add_argument("--rf-trees", type=int, default=200)
    return parser.parse_args()


def make_targets(examples, truth_actions: np.ndarray, edge_improvement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scores = np.zeros((examples.coordinates.shape[0],), dtype=np.float32)
    values = np.zeros((examples.coordinates.shape[0],), dtype=np.float32)
    for row_idx, (sample, step, edge) in enumerate(examples.coordinates):
        scores[row_idx] = np.log1p(max(float(edge_improvement[sample, step, edge]), 0.0))
        values[row_idx] = truth_actions[sample, step, edge, RB_DIM]
    return scores, values


def make_models(seed: int, rf_trees: int):
    return {
        "rf": lambda: RandomForestRegressor(
            n_estimators=int(rf_trees),
            min_samples_leaf=5,
            max_features="sqrt",
            random_state=int(seed),
            n_jobs=-1,
        ),
        "hgb": lambda: HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.05,
            l2_regularization=0.01,
            min_samples_leaf=20,
            random_state=int(seed),
        ),
    }


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
    train_examples = make_critical_examples(train_actions, train_truth, steps=steps)
    train_state_features = extract_state_features(train_base) if args.include_state_features else None
    train_features = append_state_features(train_examples.features, train_state_features, train_examples.coordinates)
    train_edge_improvement = collect_edge_gradient_improvement(
        world_model, train_base, train_actions, train_truth, stats, summary["config"], device, args.batch_size
    )
    train_score_target, train_value_target_all = make_targets(train_examples, train_truth, train_edge_improvement)
    positive_value_mask = train_value_target_all >= float(args.min_effective_rb_total)

    split_payload = {}
    for split_name in ("val", "test"):
        base_dataset, adaptive_dataset = make_adaptive_dataset(args, arrays, splits[split_name], stats, policy_model, action_scale, value_vocab, device, splits["train"])
        baseline_actions, truth_actions = collect_raw_actions(adaptive_dataset, stats)
        examples = _make_inference_examples(baseline_actions, truth_actions.shape, steps)
        state_features = extract_state_features(base_dataset) if args.include_state_features else None
        features = append_state_features(examples.features, state_features, examples.coordinates)
        edge_improvement = collect_edge_gradient_improvement(
            world_model, base_dataset, baseline_actions, truth_actions, stats, summary["config"], device, args.batch_size
        )
        oracle_score, true_value = make_targets(examples, truth_actions, edge_improvement)
        split_payload[split_name] = {
            "base_dataset": base_dataset,
            "baseline_actions": baseline_actions,
            "examples": examples,
            "features": features,
            "oracle_score": oracle_score,
            "true_value": true_value,
        }

    model_factories = make_models(seed=17, rf_trees=args.rf_trees)
    fitted = {}
    diagnostics = {
        "train_examples": int(train_features.shape[0]),
        "train_positive_score_count": int(np.sum(train_score_target > 0.0)),
        "train_positive_value_count": int(np.sum(positive_value_mask)),
        "include_state_features": bool(args.include_state_features),
    }
    for model_name, factory in model_factories.items():
        score_model = factory()
        score_model.fit(train_features, train_score_target)
        value_all_model = factory()
        value_all_model.fit(train_features, train_value_target_all)
        value_pos_model = factory()
        if np.any(positive_value_mask):
            value_pos_model.fit(train_features[positive_value_mask], train_value_target_all[positive_value_mask])
        else:
            value_pos_model = value_all_model
        fitted[model_name] = {
            "score": score_model,
            "value_all": value_all_model,
            "value_pos": value_pos_model,
        }

    rows = []
    for split_name, payload in split_payload.items():
        baseline_predictions = evaluate_raw_actions(
            payload["baseline_actions"], payload["base_dataset"], stats, world_model, summary["config"], device, args.batch_size
        )
        baseline_row = active_rate_row("identity", split_name, baseline_predictions, float("nan"))
        baseline_rmse = float(baseline_row["active_rate_rmse"])
        baseline_row.update({"model": "identity", "value_model": "identity", "scope": "none", "top_k": 0})
        rows.append(baseline_row)
        for model_name, models in fitted.items():
            pred_score = np.asarray(models["score"].predict(payload["features"]), dtype=np.float32)
            diagnostics[f"{split_name}_{model_name}_score_pearson"] = safe_corr(pred_score, payload["oracle_score"])
            diagnostics[f"{split_name}_{model_name}_score_spearman"] = safe_corr(rankdata(pred_score), rankdata(payload["oracle_score"]))
            for value_name in ("value_all", "value_pos"):
                pred_value = np.asarray(models[value_name].predict(payload["features"]), dtype=np.float32)
                pred_value = np.clip(pred_value, 0.0, None)
                positive_mask = payload["true_value"] >= float(args.min_effective_rb_total)
                diagnostics[f"{split_name}_{model_name}_{value_name}_pearson"] = safe_corr(pred_value, payload["true_value"])
                diagnostics[f"{split_name}_{model_name}_{value_name}_mae_nonzero_true"] = (
                    float(mean_absolute_error(payload["true_value"][positive_mask], pred_value[positive_mask]))
                    if np.any(positive_mask)
                    else float("nan")
                )
                for scope in args.scopes:
                    for top_k in args.top_k:
                        selected = select_topk_indices(payload["examples"].coordinates, pred_score, int(top_k), scope)
                        actions = apply_selected_repair(payload["baseline_actions"], payload["examples"].coordinates, pred_value, selected)
                        predictions = evaluate_raw_actions(actions, payload["base_dataset"], stats, world_model, summary["config"], device, args.batch_size)
                        candidate = f"{model_name}__{value_name}__{scope}__top{top_k}"
                        row = active_rate_row(candidate, split_name, predictions, baseline_rmse)
                        row.update(
                            {
                                "model": model_name,
                                "value_model": value_name,
                                "scope": scope,
                                "top_k": int(top_k),
                                "selected_count": int(selected.size),
                                "selected_oracle_mass": float(np.sum(payload["oracle_score"][selected])) if selected.size else 0.0,
                            }
                        )
                        rows.append(row)

    write_csv(args.output_dir / "tree_identifiability_results.csv", rows)
    val_ranked = sorted([row for row in rows if row["split"] == "val"], key=lambda row: (float(row["active_rate_rmse"]), str(row["candidate"])))
    write_csv(args.output_dir / "tree_identifiability_val_ranked.csv", val_ranked)
    diagnostics["runtime_seconds"] = float(time.time() - started)
    diagnostics["best_val_candidate"] = val_ranked[0]["candidate"] if val_ranked else None
    diagnostics["best_val_active_rate_rmse"] = float(val_ranked[0]["active_rate_rmse"]) if val_ranked else float("nan")
    result = {
        "framework": "PI-JWM",
        "candidate": "v11",
        "mode": "rb_total_tree_identifiability",
        "output_dir": str(args.output_dir),
        "command": " ".join(sys.argv),
        "diagnostics": diagnostics,
        "best_val": val_ranked[0] if val_ranked else None,
    }
    write_json(args.output_dir / "summary.json", result)
    return result


def main() -> None:
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
