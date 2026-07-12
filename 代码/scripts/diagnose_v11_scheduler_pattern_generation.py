"""CPU-first scheduler-pattern generation diagnostic for PI-JWM v11 candidate.

This diagnostic tests whether structure-level action generation is a better
direction than post-hoc rb_total repair.  It retrieves whole step-level action
patterns from the train split and evaluates replacing each predicted future
step with the retrieved pattern.  Oracle variants are explicitly marked as
diagnostic-only.
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
from sklearn.multioutput import MultiOutputRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from pi_jwm.v11_rollout_value_calibrator import freeze_module

from diagnose_v11_counterfactual_value_attribution import collect_raw_actions
from diagnose_v11_rb_total_value_reconstruction import make_step_scheduler_features
from run_v11_rb_total_repair import active_rate_row, evaluate_raw_actions, make_adaptive_dataset, write_csv
from run_v11_rb_total_value_head import limit_indices, load_context_limited


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts/experiments/pi_jwm_v11_scheduler_pattern_generation_cpu_20260622"
RB_DIM = 2
CPU_DIM = 4
EPS = 1e-9


def make_step_keys(actions: np.ndarray, steps: tuple[int, ...]) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    keys = []
    for sample in range(actions.shape[0]):
        for step in steps:
            if 0 <= int(step) < actions.shape[1]:
                keys.append((int(sample), int(step)))
    return np.asarray(keys, dtype=np.int64).reshape(-1, 2)


def step_action_rows(actions: np.ndarray, step_keys: np.ndarray) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    step_keys = np.asarray(step_keys, dtype=np.int64).reshape(-1, 2)
    if step_keys.shape[0] == 0:
        return np.zeros((0, actions.shape[2], actions.shape[3]), dtype=np.float32)
    return actions[step_keys[:, 0], step_keys[:, 1]].astype(np.float32)


def retrieve_nearest_step_patterns(
    query_features: np.ndarray,
    prototype_features: np.ndarray,
    prototype_actions: np.ndarray,
    chunk_size: int = 2048,
) -> tuple[np.ndarray, np.ndarray]:
    query_features = np.asarray(query_features, dtype=np.float32)
    prototype_features = np.asarray(prototype_features, dtype=np.float32)
    prototype_actions = np.asarray(prototype_actions, dtype=np.float32)
    if query_features.ndim != 2 or prototype_features.ndim != 2:
        raise ValueError("features must be 2D")
    if prototype_features.shape[0] != prototype_actions.shape[0]:
        raise ValueError("prototype features and actions must have the same row count")
    if prototype_features.shape[0] == 0:
        empty = np.zeros((query_features.shape[0], *prototype_actions.shape[1:]), dtype=np.float32)
        return empty, np.full((query_features.shape[0],), -1, dtype=np.int64)
    nearest_all = np.empty((query_features.shape[0],), dtype=np.int64)
    proto_norm = np.sum(prototype_features * prototype_features, axis=1, keepdims=True).T
    for start in range(0, query_features.shape[0], int(chunk_size)):
        stop = min(start + int(chunk_size), query_features.shape[0])
        query = query_features[start:stop]
        distance = np.sum(query * query, axis=1, keepdims=True) + proto_norm - 2.0 * (query @ prototype_features.T)
        nearest_all[start:stop] = np.argmin(distance, axis=1).astype(np.int64)
    return prototype_actions[nearest_all].astype(np.float32), nearest_all


def apply_step_pattern_replacement(
    actions: np.ndarray,
    step_keys: np.ndarray,
    replacements: np.ndarray,
    mode: str = "all",
) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    step_keys = np.asarray(step_keys, dtype=np.int64).reshape(-1, 2)
    replacements = np.asarray(replacements, dtype=np.float32)
    if step_keys.shape[0] != replacements.shape[0]:
        raise ValueError("step keys and replacements must have the same row count")
    repaired = actions.copy()
    if mode == "all":
        dims = list(range(actions.shape[-1]))
    elif mode == "rb_cpu":
        dims = [1, RB_DIM, 3, CPU_DIM]
    elif mode == "rb_only":
        dims = [1, RB_DIM]
    else:
        raise ValueError(f"unknown replacement mode: {mode}")
    for row_idx, (sample, step) in enumerate(step_keys):
        if int(step) == 0:
            continue
        repaired[int(sample), int(step), :, dims] = replacements[row_idx, :, dims]
    repaired[:, 0] = actions[:, 0]
    return np.clip(repaired, 0.0, None).astype(np.float32)


def step_group_totals(step_actions: np.ndarray) -> np.ndarray:
    step_actions = np.asarray(step_actions, dtype=np.float32)
    if step_actions.ndim != 3:
        raise ValueError("step_actions must have shape [row, edge, action_dim]")
    rb_total = np.sum(np.clip(step_actions[:, :, RB_DIM], 0.0, None), axis=1)
    cpu_total = np.sum(np.clip(step_actions[:, :, CPU_DIM], 0.0, None), axis=1)
    return np.stack([rb_total, cpu_total], axis=1).astype(np.float32)


def fit_group_total_model(kind: str, features: np.ndarray, targets: np.ndarray, seed: int, rf_trees: int):
    features = np.asarray(features, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32).reshape(-1, 2)
    if features.shape[0] != targets.shape[0]:
        raise ValueError("features and targets must have the same row count")
    if kind == "rf":
        model = RandomForestRegressor(
            n_estimators=int(rf_trees),
            min_samples_leaf=3,
            max_features="sqrt",
            random_state=int(seed),
            n_jobs=-1,
        )
    elif kind == "hgb":
        model = MultiOutputRegressor(
            HistGradientBoostingRegressor(
                max_iter=180,
                learning_rate=0.04,
                l2_regularization=0.02,
                min_samples_leaf=10,
                random_state=int(seed),
            )
        )
    else:
        raise ValueError(f"unknown group total model kind: {kind}")
    model.fit(features, targets)
    return model


def predict_group_totals(model, features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    if features.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    prediction = np.asarray(model.predict(features), dtype=np.float32).reshape(-1, 2)
    return np.clip(prediction, 0.0, None).astype(np.float32)


def apply_step_group_total_scaling(
    actions: np.ndarray,
    step_keys: np.ndarray,
    target_totals: np.ndarray,
    mode: str = "rb_cpu",
) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    step_keys = np.asarray(step_keys, dtype=np.int64).reshape(-1, 2)
    target_totals = np.asarray(target_totals, dtype=np.float32).reshape(-1, 2)
    if step_keys.shape[0] != target_totals.shape[0]:
        raise ValueError("step keys and target totals must have the same row count")
    repaired = actions.copy()
    dims = []
    if mode in {"rb_cpu", "rb_only"}:
        dims.append((RB_DIM, 0))
    if mode == "rb_cpu":
        dims.append((CPU_DIM, 1))
    if mode not in {"rb_cpu", "rb_only"}:
        raise ValueError(f"unknown group-total scaling mode: {mode}")
    for row_idx, (sample, step) in enumerate(step_keys):
        if int(step) == 0:
            continue
        for action_dim, total_idx in dims:
            current = np.clip(repaired[int(sample), int(step), :, action_dim], 0.0, None)
            current_sum = float(np.sum(current))
            if current_sum <= EPS:
                continue
            target_sum = max(float(target_totals[row_idx, total_idx]), 0.0)
            repaired[int(sample), int(step), :, action_dim] = current * (target_sum / current_sum)
    repaired[:, 0] = actions[:, 0]
    return np.clip(repaired, 0.0, None).astype(np.float32)


def standardize(train_features: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, ...]:
    train_features = np.asarray(train_features, dtype=np.float32)
    mean = train_features.mean(axis=0, keepdims=True) if train_features.shape[0] else np.zeros((1, train_features.shape[1]), dtype=np.float32)
    std = train_features.std(axis=0, keepdims=True) if train_features.shape[0] else np.ones_like(mean)
    std = np.where(std < 1e-6, 1.0, std)
    result = [(train_features - mean) / std]
    for item in others:
        result.append((np.asarray(item, dtype=np.float32) - mean) / std)
    return tuple(arr.astype(np.float32) for arr in result)


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
    parser.add_argument("--steps", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--max-train-samples", type=int, default=512)
    parser.add_argument("--max-val-samples", type=int, default=256)
    parser.add_argument("--max-test-samples", type=int, default=256)
    parser.add_argument("--limit-after-stats", action="store_true")
    parser.add_argument("--streaming-stats", action="store_true")
    parser.add_argument("--stats-chunk-size", type=int, default=512)
    parser.add_argument("--prototype-sources", choices=("train_truth", "train_policy"), nargs="+", default=["train_truth", "train_policy"])
    parser.add_argument(
        "--replacement-modes",
        choices=("all", "rb_cpu", "rb_only", "group_total_rb_cpu", "group_total_rb_only"),
        nargs="+",
        default=["group_total_rb_cpu", "rb_cpu"],
    )
    parser.add_argument("--group-total-models", choices=("rf", "hgb"), nargs="+", default=[])
    parser.add_argument("--include-oracle-group-total", action="store_true")
    parser.add_argument("--rf-trees", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260622)
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
    train_policy_actions, train_truth = collect_raw_actions(train_dataset, stats)
    train_step_keys = make_step_keys(train_policy_actions, steps)
    train_features_raw = make_step_scheduler_features(train_policy_actions, train_step_keys)
    train_policy_step_actions = step_action_rows(train_policy_actions, train_step_keys)
    train_truth_step_actions = step_action_rows(train_truth, train_step_keys)
    train_truth_group_totals = step_group_totals(train_truth_step_actions)

    split_payload = {}
    split_features_raw = {}
    for split_name in ("val", "test"):
        base_dataset, adaptive_dataset = make_adaptive_dataset(args, arrays, splits[split_name], stats, policy_model, action_scale, value_vocab, device, splits["train"])
        policy_actions, truth_actions = collect_raw_actions(adaptive_dataset, stats)
        step_keys = make_step_keys(policy_actions, steps)
        features_raw = make_step_scheduler_features(policy_actions, step_keys)
        split_features_raw[split_name] = features_raw
        split_payload[split_name] = {
            "base_dataset": base_dataset,
            "policy_actions": policy_actions,
            "truth_actions": truth_actions,
            "step_keys": step_keys,
        }

    standardized = standardize(train_features_raw, *(split_features_raw[name] for name in ("val", "test")))
    train_features = standardized[0]
    split_features = {"val": standardized[1], "test": standardized[2]}

    prototype_actions_by_source = {
        "train_policy": train_policy_step_actions,
        "train_truth": train_truth_step_actions,
    }
    group_total_models = {
        model_kind: fit_group_total_model(model_kind, train_features, train_truth_group_totals, args.seed + 101, args.rf_trees)
        for model_kind in args.group_total_models
    }

    rows = []
    for split_name, payload in split_payload.items():
        baseline_predictions = evaluate_raw_actions(
            payload["policy_actions"], payload["base_dataset"], stats, world_model, summary["config"], device, args.batch_size
        )
        baseline_row = active_rate_row("identity", split_name, baseline_predictions, float("nan"))
        baseline_row.update({"method": "identity"})
        baseline_rmse = float(baseline_row["active_rate_rmse"])
        rows.append(baseline_row)

        for source in args.prototype_sources:
            retrieved, nearest = retrieve_nearest_step_patterns(
                split_features[split_name],
                train_features,
                prototype_actions_by_source[source],
            )
            for mode in args.replacement_modes:
                if mode == "group_total_rb_cpu":
                    repaired = apply_step_group_total_scaling(
                        payload["policy_actions"], payload["step_keys"], step_group_totals(retrieved), mode="rb_cpu"
                    )
                elif mode == "group_total_rb_only":
                    repaired = apply_step_group_total_scaling(
                        payload["policy_actions"], payload["step_keys"], step_group_totals(retrieved), mode="rb_only"
                    )
                else:
                    repaired = apply_step_pattern_replacement(payload["policy_actions"], payload["step_keys"], retrieved, mode=mode)
                predictions = evaluate_raw_actions(repaired, payload["base_dataset"], stats, world_model, summary["config"], device, args.batch_size)
                prefix = "diagnostic_only__" if source == "train_truth" else ""
                candidate = f"{prefix}nearest_step_pattern__{source}__{mode}"
                row = active_rate_row(candidate, split_name, predictions, baseline_rmse)
                row.update(
                    {
                        "method": "diagnostic_only_pattern" if source == "train_truth" else "deployable_pattern",
                        "prototype_source": source,
                        "replacement_mode": mode,
                        "unique_prototypes": int(np.unique(nearest[nearest >= 0]).size),
                    }
                )
                rows.append(row)

        group_total_candidates: list[tuple[str, str, np.ndarray]] = []
        for model_kind, model in group_total_models.items():
            group_total_candidates.append((f"learned_group_total_{model_kind}", "deployable_group_total", predict_group_totals(model, split_features[split_name])))
        if args.include_oracle_group_total:
            truth_step_actions = step_action_rows(payload["truth_actions"], payload["step_keys"])
            group_total_candidates.append(("diagnostic_only__oracle_group_total", "diagnostic_only_group_total", step_group_totals(truth_step_actions)))
        for candidate, method, target_totals in group_total_candidates:
            for mode in ("rb_cpu", "rb_only"):
                repaired = apply_step_group_total_scaling(payload["policy_actions"], payload["step_keys"], target_totals, mode=mode)
                predictions = evaluate_raw_actions(repaired, payload["base_dataset"], stats, world_model, summary["config"], device, args.batch_size)
                row = active_rate_row(f"{candidate}__{mode}", split_name, predictions, baseline_rmse)
                row.update(
                    {
                        "method": method,
                        "replacement_mode": mode,
                        "target_total_mean_rb": float(np.mean(target_totals[:, 0])) if target_totals.size else 0.0,
                        "target_total_mean_cpu": float(np.mean(target_totals[:, 1])) if target_totals.size else 0.0,
                    }
                )
                rows.append(row)

    write_csv(args.output_dir / "scheduler_pattern_results.csv", rows)
    val_ranked = sorted([row for row in rows if row["split"] == "val"], key=lambda row: (float(row["active_rate_rmse"]), str(row["candidate"])))
    deployable_val_ranked = sorted(
        [row for row in rows if row["split"] == "val" and row.get("method") in {"deployable_pattern", "deployable_group_total"}],
        key=lambda row: (float(row["active_rate_rmse"]), str(row["candidate"])),
    )
    diagnostic_val_ranked = sorted(
        [row for row in rows if row["split"] == "val" and str(row.get("method", "")).startswith("diagnostic_only")],
        key=lambda row: (float(row["active_rate_rmse"]), str(row["candidate"])),
    )
    write_csv(args.output_dir / "scheduler_pattern_val_ranked.csv", val_ranked)
    write_csv(args.output_dir / "scheduler_pattern_deployable_val_ranked.csv", deployable_val_ranked)
    write_csv(args.output_dir / "scheduler_pattern_diagnostic_val_ranked.csv", diagnostic_val_ranked)
    test_by_candidate = {str(row["candidate"]): row for row in rows if row["split"] == "test"}
    best_val = deployable_val_ranked[0] if deployable_val_ranked else None
    best_diagnostic_val = diagnostic_val_ranked[0] if diagnostic_val_ranked else None
    result = {
        "framework": "PI-JWM",
        "candidate": "v11",
        "mode": "scheduler_pattern_generation_cpu",
        "output_dir": str(args.output_dir),
        "command": " ".join(sys.argv),
        "diagnostics": {
            "train_step_count": int(train_step_keys.shape[0]),
            "runtime_seconds": float(time.time() - started),
            "steps": list(steps),
        },
        "best_val": best_val,
        "matched_test_for_best_val": test_by_candidate.get(str(best_val["candidate"])) if best_val else None,
        "best_diagnostic_val": best_diagnostic_val,
        "matched_test_for_best_diagnostic_val": test_by_candidate.get(str(best_diagnostic_val["candidate"])) if best_diagnostic_val else None,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
