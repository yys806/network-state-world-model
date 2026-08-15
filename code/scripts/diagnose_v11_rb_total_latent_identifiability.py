"""PI-JWM v11 candidate rb_total latent/context feature diagnostic.

This diagnostic extracts deployable edge latent context from the frozen
PI-JWM world model and tests whether those features make rb_total repair
ranking/value targets identifiable on CPU.
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
from torch.utils.data import DataLoader

try:
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover
    XGBRegressor = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from pi_jwm.v11_rollout_value_calibrator import freeze_module
from pi_jwm.v6_dual_graph import V6DualGraphBatch

from diagnose_v11_counterfactual_value_attribution import collect_raw_actions
from diagnose_v11_rb_total_oracle_value_scope import (
    rankdata,
    safe_corr,
    select_topk_indices,
    write_json,
)
from run_v11_rb_total_repair import active_rate_row, evaluate_raw_actions, make_adaptive_dataset, write_csv
from run_v11_rb_total_value_head import (
    RB_DIM,
    _make_inference_examples,
    RawFutureActionSingleDataset,
    collate_raw_future_action_batch,
    collect_edge_gradient_improvement,
    load_context_limited,
    make_critical_examples,
)


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts/experiments/pi_jwm_v11_rb_total_latent_identifiability_20260622"


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
    parser.add_argument("--rf-trees", type=int, default=100)
    parser.add_argument("--feature-sets", choices=("latent", "action_latent"), nargs="+", default=["latent", "action_latent"])
    parser.add_argument("--blend-alpha", type=float, nargs="+", default=[1.0])
    parser.add_argument("--model-kinds", choices=("rf", "hgb", "xgb"), nargs="+", default=["rf", "hgb"])
    parser.add_argument("--value-target-modes", choices=("abs", "log", "residual", "ratio"), nargs="+", default=["abs"])
    return parser.parse_args()


def collect_rollout_edge_context(world_model, base_dataset, raw_actions: np.ndarray, stats: dict, device: torch.device, batch_size: int) -> np.ndarray:
    """Return deployable per-future-step per-edge context features.

    Features are computed from history plus the provided raw future actions.
    No true future labels are used.
    """
    if not hasattr(world_model, "initial_message_passing"):
        raise TypeError("latent diagnostic currently expects V8FullWorldModelRollout")
    world_model.eval()
    dataset = RawFutureActionSingleDataset(base_dataset, raw_actions, stats)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_raw_future_action_batch)
    chunks = []
    with torch.no_grad():
        for batch, _target in loader:
            batch = V6DualGraphBatch(
                node_history=batch.node_history.to(device),
                physical_edge_history=batch.physical_edge_history.to(device),
                info_edge_history=batch.info_edge_history.to(device),
                action_history=batch.action_history.to(device),
                future_actions=batch.future_actions.to(device),
                task_history=batch.task_history.to(device),
                link_rate_baseline=(batch.link_rate_baseline.to(device) if batch.link_rate_baseline is not None else None),
            )
            world_model._validate_batch(batch)
            if world_model.config.history_encoder == "stgcn_full":
                node_state, physical_edge_state, info_edge_state, action_state, task_state = world_model._encode_stgcn_full_history(batch)
                activity_memory_state = world_model._encode_activity_memory_history(batch)
            elif world_model.config.history_encoder == "stgcn_light":
                node_state, physical_edge_state, info_edge_state, action_state, task_state = world_model._encode_stgcn_light_history(batch)
                activity_memory_state = world_model._encode_activity_memory_history(batch)
            else:
                node_state = world_model._encode_history(batch.node_history, world_model.node_encoder, world_model.node_temporal_encoder)
                physical_edge_state = world_model._encode_history(
                    batch.physical_edge_history,
                    world_model.physical_edge_encoder,
                    world_model.physical_edge_temporal_encoder,
                )
                info_edge_state = world_model._encode_history(
                    world_model._base_info_edge_history(batch.info_edge_history),
                    world_model.info_edge_encoder,
                    world_model.info_edge_temporal_encoder,
                )
                activity_memory_state = world_model._encode_activity_memory_history(batch)
                action_state = world_model._encode_history(batch.action_history, world_model.action_encoder, world_model.action_temporal_encoder)
                task_state = world_model._encode_history(batch.task_history, world_model.task_encoder, world_model.task_temporal_encoder)

            node_state, edge_state, _diagnostics = world_model.initial_message_passing(
                node_state=node_state,
                physical_edge_state=physical_edge_state,
                info_edge_state=info_edge_state,
                action_state=action_state,
                edge_src_idx=world_model.edge_src_idx,
                edge_dst_idx=world_model.edge_dst_idx,
            )

            step_features = []
            for step in range(world_model.config.horizon):
                future_action = world_model.action_encoder(batch.future_actions[:, step])
                candidate_node_state, candidate_edge_state, _step_diagnostics = world_model.rollout_message_passing(
                    node_state=node_state,
                    physical_edge_state=edge_state,
                    info_edge_state=edge_state,
                    action_state=future_action,
                    edge_src_idx=world_model.edge_src_idx,
                    edge_dst_idx=world_model.edge_dst_idx,
                )
                node_state, edge_state = world_model._rollout_latent_transition(
                    node_state=node_state,
                    edge_state=edge_state,
                    candidate_node_state=candidate_node_state,
                    candidate_edge_state=candidate_edge_state,
                )
                if world_model.adaptive_edge_context is not None:
                    edge_state, _attention = world_model.adaptive_edge_context(edge_state)

                global_edge_state = edge_state.mean(dim=1)
                global_node_state = node_state.mean(dim=1)
                task_state = world_model.task_rollout(
                    torch.cat([task_state, global_edge_state, global_node_state], dim=-1),
                    task_state,
                )

                activity_input = world_model._activity_head_input(edge_state, activity_memory_state)
                activity_logit = world_model.link_activity_head(activity_input)
                direct_rate = world_model.link_rate_head(edge_state)
                if world_model.link_positive_rate_head is not None:
                    positive_rate = world_model.link_positive_rate_head(edge_state)
                else:
                    positive_rate = torch.zeros_like(direct_rate)
                if world_model.link_active_rate_aux_head is not None:
                    aux_rate, _weights = world_model._predict_active_rate_aux(edge_state)
                else:
                    aux_rate = torch.zeros_like(direct_rate)

                expanded_global_edge = global_edge_state.unsqueeze(1).expand_as(edge_state)
                expanded_global_node = global_node_state.unsqueeze(1).expand_as(edge_state)
                expanded_task = task_state.unsqueeze(1).expand_as(edge_state)
                step_feature = torch.cat(
                    [
                        edge_state,
                        expanded_global_edge,
                        expanded_global_node,
                        expanded_task,
                        activity_logit,
                        direct_rate,
                        positive_rate,
                        aux_rate,
                    ],
                    dim=-1,
                )
                step_features.append(step_feature.detach().cpu().numpy().astype(np.float32))
            chunks.append(np.stack(step_features, axis=1))
    return np.concatenate(chunks, axis=0).astype(np.float32)


def rows_from_context(context: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    coords = np.asarray(coordinates, dtype=np.int64).reshape(-1, 3)
    if coords.shape[0] == 0:
        return np.zeros((0, context.shape[-1]), dtype=np.float32)
    return context[coords[:, 0], coords[:, 1], coords[:, 2]].astype(np.float32)


def apply_selected_blend_repair(
    actions: np.ndarray,
    coordinates: np.ndarray,
    values: np.ndarray,
    selected: np.ndarray,
    alpha: float,
) -> np.ndarray:
    repaired = np.asarray(actions, dtype=np.float32).copy()
    coords = np.asarray(coordinates, dtype=np.int64).reshape(-1, 3)
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    alpha = float(alpha)
    for idx in np.asarray(selected, dtype=np.int64).reshape(-1):
        sample, step, edge = coords[int(idx)]
        if step == 0 or actions[sample, step, edge, RB_DIM] <= 1e-9:
            continue
        baseline_value = float(actions[sample, step, edge, RB_DIM])
        target_value = max(float(values[int(idx)]), 0.0)
        repaired[sample, step, edge, RB_DIM] = max((1.0 - alpha) * baseline_value + alpha * target_value, 0.0)
    repaired[:, 0, :, RB_DIM] = actions[:, 0, :, RB_DIM]
    repaired[..., RB_DIM] = np.where(actions[..., RB_DIM] > 1e-9, np.clip(repaired[..., RB_DIM], 0.0, None), 0.0)
    return repaired


def make_targets(examples, truth_actions: np.ndarray, edge_improvement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scores = np.zeros((examples.coordinates.shape[0],), dtype=np.float32)
    values = np.zeros((examples.coordinates.shape[0],), dtype=np.float32)
    for row_idx, (sample, step, edge) in enumerate(examples.coordinates):
        scores[row_idx] = np.log1p(max(float(edge_improvement[sample, step, edge]), 0.0))
        values[row_idx] = truth_actions[sample, step, edge, RB_DIM]
    return scores, values


def build_models(seed: int, rf_trees: int):
    factories = {
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
    if XGBRegressor is not None:
        factories["xgb"] = lambda: XGBRegressor(
            n_estimators=max(4, int(rf_trees)),
            learning_rate=0.05,
            max_depth=4,
            min_child_weight=2.0,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            random_state=int(seed),
            n_jobs=-1,
            tree_method="hist",
            objective="reg:squarederror",
        )
    return factories


def make_value_target(mode: str, true_value: np.ndarray, baseline_value: np.ndarray) -> np.ndarray:
    true_value = np.asarray(true_value, dtype=np.float32)
    baseline_value = np.asarray(baseline_value, dtype=np.float32)
    if mode == "abs":
        return true_value
    if mode == "log":
        return np.log1p(np.clip(true_value, 0.0, None)).astype(np.float32)
    if mode == "residual":
        return (true_value - baseline_value).astype(np.float32)
    if mode == "ratio":
        ratio = np.clip(true_value, 0.0, None) / np.maximum(np.clip(baseline_value, 0.0, None), 1e-6)
        return np.log(np.clip(ratio, 1e-6, 1e6)).astype(np.float32)
    raise ValueError(f"unknown value target mode: {mode}")


def invert_value_target(mode: str, prediction: np.ndarray, baseline_value: np.ndarray) -> np.ndarray:
    prediction = np.asarray(prediction, dtype=np.float32)
    baseline_value = np.asarray(baseline_value, dtype=np.float32)
    if mode == "abs":
        value = prediction
    elif mode == "log":
        value = np.expm1(prediction)
    elif mode == "residual":
        value = baseline_value + prediction
    elif mode == "ratio":
        value = np.clip(baseline_value, 0.0, None) * np.exp(np.clip(prediction, -20.0, 20.0))
    else:
        raise ValueError(f"unknown value target mode: {mode}")
    return np.clip(value, 0.0, None).astype(np.float32)


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
    train_edge_improvement = collect_edge_gradient_improvement(
        world_model, train_base, train_actions, train_truth, stats, summary["config"], device, args.batch_size
    )
    train_score, train_value = make_targets(train_examples, train_truth, train_edge_improvement)
    train_context = collect_rollout_edge_context(world_model, train_base, train_actions, stats, device, args.batch_size)
    train_latent = rows_from_context(train_context, train_examples.coordinates)
    train_action = train_examples.features.astype(np.float32)
    train_feature_sets = {
        "latent": train_latent,
        "action_latent": np.concatenate([train_action, train_latent], axis=1).astype(np.float32),
    }

    split_payload = {}
    for split_name in ("val", "test"):
        base_dataset, adaptive_dataset = make_adaptive_dataset(args, arrays, splits[split_name], stats, policy_model, action_scale, value_vocab, device, splits["train"])
        baseline_actions, truth_actions = collect_raw_actions(adaptive_dataset, stats)
        examples = _make_inference_examples(baseline_actions, truth_actions.shape, steps)
        edge_improvement = collect_edge_gradient_improvement(
            world_model, base_dataset, baseline_actions, truth_actions, stats, summary["config"], device, args.batch_size
        )
        oracle_score, true_value = make_targets(examples, truth_actions, edge_improvement)
        latent_context = collect_rollout_edge_context(world_model, base_dataset, baseline_actions, stats, device, args.batch_size)
        latent_rows = rows_from_context(latent_context, examples.coordinates)
        action_rows = examples.features.astype(np.float32)
        split_payload[split_name] = {
            "base_dataset": base_dataset,
            "baseline_actions": baseline_actions,
            "examples": examples,
            "oracle_score": oracle_score,
            "true_value": true_value,
            "features": {
                "latent": latent_rows,
                "action_latent": np.concatenate([action_rows, latent_rows], axis=1).astype(np.float32),
            },
        }

    rows = []
    diagnostics = {
        "train_examples": int(train_examples.coordinates.shape[0]),
        "train_positive_score_count": int(np.sum(train_score > 0.0)),
        "train_positive_value_count": int(np.sum(train_value >= float(args.min_effective_rb_total))),
        "latent_feature_dim": int(train_latent.shape[1]),
        "feature_sets": list(args.feature_sets),
        "model_kinds": list(args.model_kinds),
        "value_target_modes": list(args.value_target_modes),
    }
    all_model_factories = build_models(seed=23, rf_trees=args.rf_trees)
    model_factories = {name: all_model_factories[name] for name in args.model_kinds}
    positive_value_mask = train_value >= float(args.min_effective_rb_total)

    for feature_set in args.feature_sets:
        train_features = train_feature_sets[feature_set]
        fitted = {}
        for model_name, factory in model_factories.items():
            score_model = factory()
            score_model.fit(train_features, train_score)
            value_models = {}
            for value_mode in args.value_target_modes:
                value_pos_model = factory()
                if np.any(positive_value_mask):
                    target = make_value_target(
                        value_mode,
                        train_value[positive_value_mask],
                        train_examples.baseline_values[positive_value_mask],
                    )
                    value_pos_model.fit(train_features[positive_value_mask], target)
                else:
                    target = make_value_target(value_mode, train_value, train_examples.baseline_values)
                    value_pos_model.fit(train_features, target)
                value_models[value_mode] = value_pos_model
            fitted[model_name] = {"score": score_model, "value_models": value_models}

        for split_name, payload in split_payload.items():
            baseline_predictions = evaluate_raw_actions(
                payload["baseline_actions"], payload["base_dataset"], stats, world_model, summary["config"], device, args.batch_size
            )
            baseline_row = active_rate_row("identity", split_name, baseline_predictions, float("nan"))
            baseline_rmse = float(baseline_row["active_rate_rmse"])
            baseline_row.update({"feature_set": feature_set, "model": "identity", "scope": "none", "top_k": 0})
            rows.append(baseline_row)

            split_features = payload["features"][feature_set]
            for model_name, models in fitted.items():
                pred_score = np.asarray(models["score"].predict(split_features), dtype=np.float32)
                diagnostics[f"{split_name}_{feature_set}_{model_name}_score_pearson"] = safe_corr(pred_score, payload["oracle_score"])
                diagnostics[f"{split_name}_{feature_set}_{model_name}_score_spearman"] = safe_corr(rankdata(pred_score), rankdata(payload["oracle_score"]))
                for value_mode, value_model in models["value_models"].items():
                    raw_pred_value = np.asarray(value_model.predict(split_features), dtype=np.float32)
                    pred_value = invert_value_target(value_mode, raw_pred_value, payload["examples"].baseline_values)
                    positive_mask = payload["true_value"] >= float(args.min_effective_rb_total)
                    diagnostics[f"{split_name}_{feature_set}_{model_name}_{value_mode}_value_pearson"] = safe_corr(pred_value, payload["true_value"])
                    diagnostics[f"{split_name}_{feature_set}_{model_name}_{value_mode}_value_mae_nonzero_true"] = (
                        float(mean_absolute_error(payload["true_value"][positive_mask], pred_value[positive_mask]))
                        if np.any(positive_mask)
                        else float("nan")
                    )
                    for scope in args.scopes:
                        for top_k in args.top_k:
                            selected = select_topk_indices(payload["examples"].coordinates, pred_score, int(top_k), scope)
                            for alpha in args.blend_alpha:
                                actions = apply_selected_blend_repair(
                                    payload["baseline_actions"],
                                    payload["examples"].coordinates,
                                    pred_value,
                                    selected,
                                    alpha=float(alpha),
                                )
                                predictions = evaluate_raw_actions(actions, payload["base_dataset"], stats, world_model, summary["config"], device, args.batch_size)
                                candidate = f"{feature_set}__{model_name}__value_{value_mode}__{scope}__top{top_k}__alpha{float(alpha):g}"
                                row = active_rate_row(candidate, split_name, predictions, baseline_rmse)
                                row.update(
                                    {
                                        "feature_set": feature_set,
                                        "model": model_name,
                                        "value_mode": value_mode,
                                        "scope": scope,
                                        "top_k": int(top_k),
                                        "alpha": float(alpha),
                                        "selected_count": int(selected.size),
                                        "selected_oracle_mass": float(np.sum(payload["oracle_score"][selected])) if selected.size else 0.0,
                                    }
                                )
                                rows.append(row)

    write_csv(args.output_dir / "latent_identifiability_results.csv", rows)
    val_ranked = sorted([row for row in rows if row["split"] == "val"], key=lambda row: (float(row["active_rate_rmse"]), str(row["candidate"])))
    write_csv(args.output_dir / "latent_identifiability_val_ranked.csv", val_ranked)
    diagnostics["runtime_seconds"] = float(time.time() - started)
    diagnostics["best_val_candidate"] = val_ranked[0]["candidate"] if val_ranked else None
    diagnostics["best_val_active_rate_rmse"] = float(val_ranked[0]["active_rate_rmse"]) if val_ranked else float("nan")
    result = {
        "framework": "PI-JWM",
        "candidate": "v11",
        "mode": "rb_total_latent_identifiability",
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
