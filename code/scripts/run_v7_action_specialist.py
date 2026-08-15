"""Train sparse state-to-action specialists for PI-JWM v7.

This script turns logged scheduler actions into an offline supervised policy
diagnostic. It follows the same two-stage idea as the active-rate specialist:
classify rare active action entries first, then regress the action value only
on positive entries.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pi_jwm.paths import ARTIFACTS_DIR
from pi_jwm.v6_data import build_physical_edge_history, load_world_model_arrays, split_by_seed


DEFAULT_DATASET_DIR = (
    ARTIFACTS_DIR
    / "experiments"
    / "airfogsim_v0"
    / "datasets"
    / "world_model_dataset_seed0_9_v0"
)
OUTPUT_DIR = ARTIFACTS_DIR / "experiments" / "pi_jwm_v7_action_specialist"


@dataclass
class SpecialistFamily:
    name: str
    classifier_factory: object
    regressor_factory: object


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PI-JWM v7 sparse state-to-action specialists.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--max-neg-per-pos", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260609)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    arrays = load_world_model_arrays(args.dataset_dir)
    train_idx, val_idx, test_idx = split_by_seed(arrays["sample_seed"])
    features, feature_names, target = build_action_policy_table(arrays)
    action_names = [str(name) for name in arrays["edge_action_features"].tolist()]
    horizon = int(arrays["edge_a_future"].shape[1])
    num_edges = int(arrays["edge_a_future"].shape[2])

    split_rows = {
        "train": sample_indices_to_flat_indices(train_idx, horizon, num_edges),
        "val": sample_indices_to_flat_indices(val_idx, horizon, num_edges),
        "test": sample_indices_to_flat_indices(test_idx, horizon, num_edges),
    }

    family_rows = []
    per_action_rows = []
    family_outputs = {}
    for family in make_families(args.seed):
        result = fit_family(
            family,
            features,
            target,
            split_rows,
            action_names,
            max_neg_per_pos=args.max_neg_per_pos,
            seed=args.seed,
        )
        family_rows.append(result["summary_row"])
        per_action_rows.extend(result["per_action_rows"])
        family_outputs[family.name] = result

    zero_row = evaluate_zero_policy(target, split_rows)
    family_rows.append(zero_row)
    family_rows = sorted(family_rows, key=lambda row: (-row["test_action_f1"], row["test_active_value_rmse"]))
    best = family_rows[0]

    metrics_csv = args.output_dir / "v7_action_specialist_metrics.csv"
    write_csv(metrics_csv, family_rows)
    per_action_csv = args.output_dir / "v7_action_specialist_per_action_metrics.csv"
    write_csv(per_action_csv, per_action_rows)
    fig_path = args.output_dir / "v7_action_specialist_comparison.png"
    plot_family_metrics(fig_path, family_rows)

    summary = {
        "framework": "PI-JWM",
        "module": "v7_action_specialist",
        "note": "Offline behavior-cloning specialist for logged state-to-action learning. It is not online RL.",
        "dataset_dir": str(args.dataset_dir),
        "feature_dim": int(features.shape[1]),
        "feature_names": feature_names,
        "action_names": action_names,
        "split_sizes": {name: int(len(rows)) for name, rows in split_rows.items()},
        "max_neg_per_pos": int(args.max_neg_per_pos),
        "rows": family_rows,
        "per_action_rows": per_action_rows,
        "best": best,
        "outputs": {
            "metrics_csv": str(metrics_csv),
            "per_action_csv": str(per_action_csv),
            "figure": str(fig_path),
        },
    }
    summary_path = args.output_dir / "v7_action_specialist_summary.json"
    report_path = args.output_dir / "v7_action_specialist_report.md"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(render_report(summary), encoding="utf-8")

    meeting_dir = WORKSPACE_ROOT / "\u6587\u6863" / "\u5f00\u4f1a" / "6.9"
    fig_meeting_dir = meeting_dir / "figs"
    fig_meeting_dir.mkdir(parents=True, exist_ok=True)
    write_csv(meeting_dir / "pi_jwm_v7_action_specialist_metrics.csv", family_rows)
    write_csv(meeting_dir / "pi_jwm_v7_action_specialist_per_action_metrics.csv", per_action_rows)
    plot_family_metrics(fig_meeting_dir / "pi_jwm_v7_action_specialist_comparison.png", family_rows)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"summary_path={summary_path}")
    print(f"report_path={report_path}")


def build_action_policy_table(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str], np.ndarray]:
    num_samples, horizon, num_edges, _ = arrays["edge_a_future"].shape
    physical = build_physical_edge_history(
        arrays["x_node"],
        arrays["edge_src_idx"],
        arrays["edge_dst_idx"],
        arrays["valid_edge_node"],
    ).numpy()

    last_link = arrays["x_link"][:, -1]
    mean_link = arrays["x_link"].mean(axis=1)
    last_physical = physical[:, -1]
    mean_physical = physical.mean(axis=1)
    last_action = arrays["edge_a_hist"][:, -1]
    sum_action_hist = arrays["edge_a_hist"].sum(axis=1)
    task_last = arrays["x_task"][:, -1]
    task_mean = arrays["x_task"].mean(axis=1)
    node_last_global = arrays["x_node"][:, -1].mean(axis=1)
    node_mean_global = arrays["x_node"].mean(axis=(1, 2))

    edge_id = np.repeat((np.arange(num_edges, dtype=np.float32) / max(num_edges - 1, 1))[None, :, None], num_samples, axis=0)
    src_idx = np.repeat(arrays["edge_src_idx"][None, :, None].astype(np.float32), num_samples, axis=0)
    dst_idx = np.repeat(arrays["edge_dst_idx"][None, :, None].astype(np.float32), num_samples, axis=0)
    valid_edge = np.repeat(arrays["valid_edge_node"][None, :, None].astype(np.float32), num_samples, axis=0)

    edge_parts = []
    names = []
    add_edge_features(edge_parts, names, last_link, [f"last_link_{name}" for name in arrays["link_features"]])
    add_edge_features(edge_parts, names, mean_link, [f"mean_link_{name}" for name in arrays["link_features"]])
    add_edge_features(edge_parts, names, last_physical, [f"last_phys_{i}" for i in range(last_physical.shape[-1])])
    add_edge_features(edge_parts, names, mean_physical, [f"mean_phys_{i}" for i in range(mean_physical.shape[-1])])
    add_edge_features(edge_parts, names, last_action, [f"last_action_{name}" for name in arrays["edge_action_features"]])
    add_edge_features(edge_parts, names, sum_action_hist, [f"sum_action_{name}" for name in arrays["edge_action_features"]])
    add_edge_features(edge_parts, names, edge_id, ["edge_id"])
    add_edge_features(edge_parts, names, src_idx, ["src_idx"])
    add_edge_features(edge_parts, names, dst_idx, ["dst_idx"])
    add_edge_features(edge_parts, names, valid_edge, ["valid_edge_node"])

    sample_features = np.concatenate([task_last, task_mean, node_last_global, node_mean_global], axis=-1)
    sample_feature_names = [f"task_last_{name}" for name in arrays["task_features"]]
    sample_feature_names += [f"task_mean_{name}" for name in arrays["task_features"]]
    sample_feature_names += [f"node_last_mean_{name}" for name in arrays["node_features"]]
    sample_feature_names += [f"node_hist_mean_{name}" for name in arrays["node_features"]]

    per_step_features = []
    full_names = None
    for step in range(horizon):
        step_parts = [part for part in edge_parts]
        repeated_sample = np.repeat(sample_features[:, None, :], num_edges, axis=1)
        step_parts.append(repeated_sample)
        step_col = np.full((num_samples, num_edges, 1), float(step), dtype=np.float32)
        step_parts.append(step_col)
        if step == 0:
            full_names = names + sample_feature_names + ["horizon_step"]
        per_step_features.append(np.concatenate(step_parts, axis=-1))

    features = np.stack(per_step_features, axis=1).reshape(num_samples * horizon * num_edges, -1).astype(np.float32)
    target = arrays["edge_a_future"].reshape(num_samples * horizon * num_edges, -1).astype(np.float32)
    return features, list(full_names or []), target


def add_edge_features(parts: list[np.ndarray], names: list[str], values: np.ndarray, value_names: list[str]) -> None:
    parts.append(values.astype(np.float32))
    names.extend([str(name) for name in value_names])


def sample_indices_to_flat_indices(sample_idx: np.ndarray, horizon: int, num_edges: int) -> np.ndarray:
    sample_idx = np.asarray(sample_idx, dtype=np.int64)
    offsets = sample_idx[:, None, None] * horizon * num_edges
    h = np.arange(horizon, dtype=np.int64)[None, :, None] * num_edges
    e = np.arange(num_edges, dtype=np.int64)[None, None, :]
    return (offsets + h + e).reshape(-1)


def make_families(seed: int) -> list[SpecialistFamily]:
    return [
        SpecialistFamily(
            "random_forest",
            lambda: RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=1,
                max_features="sqrt",
                class_weight="balanced_subsample",
                random_state=seed,
                n_jobs=-1,
            ),
            lambda: RandomForestRegressor(
                n_estimators=300,
                min_samples_leaf=1,
                max_features=0.8,
                random_state=seed,
                n_jobs=-1,
            ),
        ),
        SpecialistFamily(
            "extra_trees",
            lambda: ExtraTreesClassifier(
                n_estimators=300,
                min_samples_leaf=1,
                max_features="sqrt",
                class_weight="balanced",
                random_state=seed,
                n_jobs=-1,
            ),
            lambda: ExtraTreesRegressor(
                n_estimators=300,
                min_samples_leaf=1,
                max_features=0.8,
                random_state=seed,
                n_jobs=-1,
            ),
        ),
        SpecialistFamily(
            "hist_gbr",
            lambda: HistGradientBoostingClassifier(
                learning_rate=0.04,
                max_iter=250,
                max_leaf_nodes=31,
                l2_regularization=0.01,
                random_state=seed,
            ),
            lambda: HistGradientBoostingRegressor(
                loss="squared_error",
                learning_rate=0.04,
                max_iter=250,
                max_leaf_nodes=31,
                l2_regularization=0.01,
                random_state=seed,
            ),
        ),
        SpecialistFamily(
            "logistic_ridge",
            lambda: make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear"),
            ),
            lambda: make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        ),
    ]


def fit_family(
    family: SpecialistFamily,
    features: np.ndarray,
    target: np.ndarray,
    split_rows: dict[str, np.ndarray],
    action_names: list[str],
    max_neg_per_pos: int,
    seed: int,
) -> dict:
    train_rows = split_rows["train"]
    x_train = features[train_rows]
    y_train = target[train_rows]
    models = []
    val_prob = np.zeros((len(split_rows["val"]), target.shape[1]), dtype=np.float32)
    val_value = np.zeros_like(val_prob)
    test_prob = np.zeros((len(split_rows["test"]), target.shape[1]), dtype=np.float32)
    test_value = np.zeros_like(test_prob)
    train_active_counts = []
    for action_dim in range(target.shape[1]):
        active_train = y_train[:, action_dim] > 1e-9
        train_active_counts.append(int(active_train.sum()))
        classifier = family.classifier_factory()
        regressor = family.regressor_factory()
        if active_train.any():
            selected = make_balanced_training_indices(
                active_train,
                max_neg_per_pos=max_neg_per_pos,
                seed=seed + action_dim,
            )
            classifier.fit(x_train[selected], active_train[selected])
            regressor.fit(x_train[active_train], y_train[active_train, action_dim])
            val_prob[:, action_dim] = predict_positive_probability(classifier, features[split_rows["val"]])
            test_prob[:, action_dim] = predict_positive_probability(classifier, features[split_rows["test"]])
            val_value[:, action_dim] = np.clip(regressor.predict(features[split_rows["val"]]), 0.0, None)
            test_value[:, action_dim] = np.clip(regressor.predict(features[split_rows["test"]]), 0.0, None)
        models.append((classifier, regressor))

    val_true = target[split_rows["val"]]
    test_true = target[split_rows["test"]]
    thresholds = choose_thresholds(val_prob, val_true > 1e-9)
    val_eval = evaluate_predictions(val_prob, val_value, val_true, thresholds)
    test_eval = evaluate_predictions(test_prob, test_value, test_true, thresholds)
    budget_scales = calibrate_probability_budget_scales(val_prob, val_value, val_true, num_edges=188)
    val_budget_counts = predict_budget_counts(val_prob, budget_scales, num_edges=188)
    test_budget_counts = predict_budget_counts(test_prob, budget_scales, num_edges=188)
    val_budget_eval = evaluate_decoded_predictions(
        decode_budgeted_topk(val_prob, val_budget_counts, num_edges=188),
        val_value,
        val_true,
    )
    test_budget_eval = evaluate_decoded_predictions(
        decode_budgeted_topk(test_prob, test_budget_counts, num_edges=188),
        test_value,
        test_true,
    )
    test_oracle_counts = true_budget_counts(test_true, num_edges=188)
    test_oracle_budget_eval = evaluate_decoded_predictions(
        decode_budgeted_topk(test_prob, test_oracle_counts, num_edges=188),
        test_value,
        test_true,
    )
    per_action_rows = []
    for action_dim, action_name in enumerate(action_names):
        dim_eval = evaluate_predictions(
            test_prob[:, action_dim : action_dim + 1],
            test_value[:, action_dim : action_dim + 1],
            test_true[:, action_dim : action_dim + 1],
            np.array([thresholds[action_dim]], dtype=np.float32),
        )
        per_action_rows.append(
            {
                "family": family.name,
                "action": action_name,
                "train_positive_count": train_active_counts[action_dim],
                "test_positive_count": int((test_true[:, action_dim] > 1e-9).sum()),
                "threshold": float(thresholds[action_dim]),
                "test_f1": dim_eval["action_f1"],
                "test_precision": dim_eval["action_precision"],
                "test_recall": dim_eval["action_recall"],
                "test_active_value_rmse": dim_eval["active_value_rmse"],
            }
        )
    summary_row = {
        "family": family.name,
        "val_action_f1": val_eval["action_f1"],
        "val_any_edge_f1": val_eval["any_edge_f1"],
        "val_active_value_rmse": val_eval["active_value_rmse"],
        "val_budget_action_f1": val_budget_eval["action_f1"],
        "val_budget_active_value_rmse": val_budget_eval["active_value_rmse"],
        "test_action_f1": test_eval["action_f1"],
        "test_action_precision": test_eval["action_precision"],
        "test_action_recall": test_eval["action_recall"],
        "test_any_edge_f1": test_eval["any_edge_f1"],
        "test_active_value_rmse": test_eval["active_value_rmse"],
        "test_budget_action_f1": test_budget_eval["action_f1"],
        "test_budget_precision": test_budget_eval["action_precision"],
        "test_budget_recall": test_budget_eval["action_recall"],
        "test_budget_any_edge_f1": test_budget_eval["any_edge_f1"],
        "test_budget_active_value_rmse": test_budget_eval["active_value_rmse"],
        "test_oracle_budget_action_f1": test_oracle_budget_eval["action_f1"],
        "test_oracle_budget_active_value_rmse": test_oracle_budget_eval["active_value_rmse"],
        "test_oracle_active_value_rmse": test_eval["oracle_active_value_rmse"],
        "test_zero_active_value_rmse": test_eval["zero_active_value_rmse"],
        "test_active_count": test_eval["active_count"],
        "thresholds": ";".join(f"{x:.3f}" for x in thresholds),
        "budget_scales": ";".join(f"{x:.5f}" for x in budget_scales),
    }
    return {
        "family": family.name,
        "summary_row": summary_row,
        "per_action_rows": per_action_rows,
        "thresholds": thresholds.tolist(),
        "val_eval": val_eval,
        "test_eval": test_eval,
    }


def make_balanced_training_indices(active: np.ndarray, max_neg_per_pos: int, seed: int) -> np.ndarray:
    active = np.asarray(active, dtype=bool)
    pos_idx = np.where(active)[0]
    neg_idx = np.where(~active)[0]
    if len(pos_idx) == 0:
        return np.arange(min(len(active), max_neg_per_pos), dtype=np.int64)
    rng = np.random.default_rng(seed)
    max_neg = min(len(neg_idx), int(max_neg_per_pos) * len(pos_idx))
    sampled_neg = rng.choice(neg_idx, size=max_neg, replace=False) if max_neg else np.array([], dtype=np.int64)
    selected = np.concatenate([pos_idx, sampled_neg.astype(np.int64)])
    rng.shuffle(selected)
    return selected.astype(np.int64)


def predict_positive_probability(classifier, x: np.ndarray) -> np.ndarray:
    if hasattr(classifier, "predict_proba"):
        proba = classifier.predict_proba(x)
        if isinstance(proba, list):
            proba = proba[0]
        if proba.shape[1] == 1:
            return np.zeros(x.shape[0], dtype=np.float32)
        return proba[:, 1].astype(np.float32)
    return classifier.predict(x).astype(np.float32)


def choose_thresholds(prob: np.ndarray, active: np.ndarray) -> np.ndarray:
    thresholds = []
    for dim in range(prob.shape[1]):
        best_threshold = 0.5
        best_f1 = -1.0
        for threshold in np.linspace(0.01, 0.99, 99):
            f1 = binary_metrics(prob[:, dim] >= threshold, active[:, dim])["f1"]
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = float(threshold)
        thresholds.append(best_threshold)
    return np.asarray(thresholds, dtype=np.float32)


def calibrate_probability_budget_scales(
    prob: np.ndarray,
    value_pred: np.ndarray,
    true_value: np.ndarray,
    num_edges: int,
) -> np.ndarray:
    true_active = true_value > 1e-9
    grouped_prob = prob.reshape(-1, num_edges, prob.shape[-1])
    true_counts = true_budget_counts(true_value, num_edges)
    scales = []
    for dim in range(prob.shape[-1]):
        sum_prob = grouped_prob[:, :, dim].sum(axis=1)
        base = safe_div(float(true_counts[:, dim].sum()), float(sum_prob.sum()))
        candidates = np.unique(
            np.concatenate(
                [
                    np.array([0.0, base], dtype=np.float32),
                    np.linspace(max(base * 0.1, 1e-5), max(base * 4.0, 1e-4), 50, dtype=np.float32),
                ]
            )
        )
        best_scale = float(base)
        best_f1 = -1.0
        for scale in candidates:
            counts = np.zeros_like(true_counts)
            counts[:, dim] = np.rint(sum_prob * float(scale)).astype(np.int64)
            decoded = decode_budgeted_topk(prob[:, dim : dim + 1], counts[:, dim : dim + 1], num_edges)
            score = binary_metrics(decoded[:, 0], true_active[:, dim])["f1"]
            if score > best_f1:
                best_f1 = score
                best_scale = float(scale)
        scales.append(best_scale)
    return np.asarray(scales, dtype=np.float32)


def predict_budget_counts(prob: np.ndarray, scales: np.ndarray, num_edges: int) -> np.ndarray:
    grouped_prob = prob.reshape(-1, num_edges, prob.shape[-1])
    counts = np.rint(grouped_prob.sum(axis=1) * scales.reshape(1, -1)).astype(np.int64)
    return np.clip(counts, 0, num_edges)


def true_budget_counts(true_value: np.ndarray, num_edges: int) -> np.ndarray:
    return (true_value.reshape(-1, num_edges, true_value.shape[-1]) > 1e-9).sum(axis=1).astype(np.int64)


def decode_budgeted_topk(prob: np.ndarray, counts: np.ndarray, num_edges: int) -> np.ndarray:
    grouped_prob = prob.reshape(-1, num_edges, prob.shape[-1])
    counts = np.asarray(counts, dtype=np.int64).reshape(grouped_prob.shape[0], grouped_prob.shape[-1])
    decoded = np.zeros_like(grouped_prob, dtype=bool)
    for group_idx in range(grouped_prob.shape[0]):
        for action_dim in range(grouped_prob.shape[-1]):
            k = int(np.clip(counts[group_idx, action_dim], 0, num_edges))
            if k <= 0:
                continue
            top_idx = np.argpartition(grouped_prob[group_idx, :, action_dim], -k)[-k:]
            decoded[group_idx, top_idx, action_dim] = True
    return decoded.reshape(prob.shape)


def evaluate_predictions(
    prob: np.ndarray,
    value_pred: np.ndarray,
    true_value: np.ndarray,
    thresholds: np.ndarray,
) -> dict[str, float]:
    true_active = true_value > 1e-9
    pred_active = prob >= thresholds.reshape(1, -1)
    end_to_end_value = np.where(pred_active, value_pred, 0.0)
    activity = binary_metrics(pred_active, true_active)
    any_edge = binary_metrics(pred_active.any(axis=-1), true_active.any(axis=-1))
    active_count = int(true_active.sum())
    if active_count:
        active_value_rmse = rmse(true_value[true_active], end_to_end_value[true_active])
        oracle_active_value_rmse = rmse(true_value[true_active], value_pred[true_active])
        zero_active_value_rmse = rmse(true_value[true_active], np.zeros(active_count, dtype=np.float32))
        active_value_mae = float(mean_absolute_error(true_value[true_active], end_to_end_value[true_active]))
    else:
        active_value_rmse = float("nan")
        oracle_active_value_rmse = float("nan")
        zero_active_value_rmse = float("nan")
        active_value_mae = float("nan")
    return {
        "action_precision": activity["precision"],
        "action_recall": activity["recall"],
        "action_f1": activity["f1"],
        "action_tp": float(activity["tp"]),
        "action_fp": float(activity["fp"]),
        "action_fn": float(activity["fn"]),
        "action_tn": float(activity["tn"]),
        "any_edge_f1": any_edge["f1"],
        "any_edge_precision": any_edge["precision"],
        "any_edge_recall": any_edge["recall"],
        "active_count": float(active_count),
        "active_value_rmse": active_value_rmse,
        "active_value_mae": active_value_mae,
        "oracle_active_value_rmse": oracle_active_value_rmse,
        "zero_active_value_rmse": zero_active_value_rmse,
    }


def evaluate_decoded_predictions(
    pred_active: np.ndarray,
    value_pred: np.ndarray,
    true_value: np.ndarray,
) -> dict[str, float]:
    true_active = true_value > 1e-9
    activity = binary_metrics(pred_active, true_active)
    any_edge = binary_metrics(pred_active.any(axis=-1), true_active.any(axis=-1))
    end_to_end_value = np.where(pred_active, value_pred, 0.0)
    active_count = int(true_active.sum())
    if active_count:
        active_value_rmse = rmse(true_value[true_active], end_to_end_value[true_active])
        active_value_mae = float(mean_absolute_error(true_value[true_active], end_to_end_value[true_active]))
    else:
        active_value_rmse = float("nan")
        active_value_mae = float("nan")
    return {
        "action_precision": activity["precision"],
        "action_recall": activity["recall"],
        "action_f1": activity["f1"],
        "action_tp": float(activity["tp"]),
        "action_fp": float(activity["fp"]),
        "action_fn": float(activity["fn"]),
        "action_tn": float(activity["tn"]),
        "any_edge_f1": any_edge["f1"],
        "any_edge_precision": any_edge["precision"],
        "any_edge_recall": any_edge["recall"],
        "active_count": float(active_count),
        "active_value_rmse": active_value_rmse,
        "active_value_mae": active_value_mae,
    }


def evaluate_zero_policy(target: np.ndarray, split_rows: dict[str, np.ndarray]) -> dict:
    test_true = target[split_rows["test"]]
    zeros = np.zeros_like(test_true)
    eval_row = evaluate_predictions(zeros, zeros, test_true, np.full(test_true.shape[1], 0.5, dtype=np.float32))
    return {
        "family": "zero_policy",
        "val_action_f1": 0.0,
        "val_any_edge_f1": 0.0,
        "val_active_value_rmse": float("nan"),
        "val_budget_action_f1": 0.0,
        "val_budget_active_value_rmse": float("nan"),
        "test_action_f1": eval_row["action_f1"],
        "test_action_precision": eval_row["action_precision"],
        "test_action_recall": eval_row["action_recall"],
        "test_any_edge_f1": eval_row["any_edge_f1"],
        "test_active_value_rmse": eval_row["active_value_rmse"],
        "test_budget_action_f1": 0.0,
        "test_budget_precision": 0.0,
        "test_budget_recall": 0.0,
        "test_budget_any_edge_f1": 0.0,
        "test_budget_active_value_rmse": eval_row["active_value_rmse"],
        "test_oracle_budget_action_f1": 0.0,
        "test_oracle_budget_active_value_rmse": eval_row["active_value_rmse"],
        "test_oracle_active_value_rmse": eval_row["oracle_active_value_rmse"],
        "test_zero_active_value_rmse": eval_row["zero_active_value_rmse"],
        "test_active_count": eval_row["active_count"],
        "thresholds": "0.500;0.500;0.500;0.500;0.500;0.500",
        "budget_scales": "0.00000;0.00000;0.00000;0.00000;0.00000;0.00000",
    }


def binary_metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, float | int]:
    pred = np.asarray(pred, dtype=bool)
    true = np.asarray(true, dtype=bool)
    tp = int((pred & true).sum())
    fp = int((pred & ~true).sum())
    fn = int((~pred & true).sum())
    tn = int((~pred & ~true).sum())
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2.0 * precision * recall, precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def rmse(true: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(true, pred)))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_family_metrics(path: Path, rows: list[dict]) -> None:
    plot_rows = [row for row in rows if row["family"] != "zero_policy"]
    labels = [row["family"] for row in plot_rows]
    f1 = [row["test_action_f1"] for row in plot_rows]
    budget_f1 = [row["test_budget_action_f1"] for row in plot_rows]
    rmse_values = [row["test_active_value_rmse"] for row in plot_rows]
    budget_rmse_values = [row["test_budget_active_value_rmse"] for row in plot_rows]

    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(len(labels))
    width = 0.36
    axes[0].bar(x - width / 2, f1, width=width, color="#2f855a", label="threshold")
    axes[0].bar(x + width / 2, budget_f1, width=width, color="#805ad5", label="budget top-k")
    axes[0].set_title("Action activity F1")
    axes[0].set_ylim(0, max(1.0, max(f1 + budget_f1) * 1.15 if f1 else 1.0))
    axes[0].set_xticks(x, labels, rotation=20)
    axes[0].legend(frameon=False, fontsize=9)

    axes[1].bar(x - width / 2, rmse_values, width=width, color="#2b6cb0", label="threshold")
    axes[1].bar(x + width / 2, budget_rmse_values, width=width, color="#b7791f", label="budget top-k")
    axes[1].set_title("End-to-end active action RMSE")
    axes[1].set_xticks(x, labels, rotation=20)
    axes[1].legend(frameon=False, fontsize=9)

    fig.suptitle("PI-JWM v7 sparse state-to-action specialist")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def render_report(summary: dict) -> str:
    best = summary["best"]
    return "\n".join(
        [
            "# PI-JWM v7 Sparse State-to-Action Specialist",
            "",
            "This report records an offline behavior-cloning specialist for logged scheduler actions.",
            "",
            "## Best Result",
            "",
            f"- best family: `{best['family']}`",
            f"- test action F1: `{best['test_action_f1']:.6f}`",
            f"- test budgeted action F1: `{best['test_budget_action_f1']:.6f}`",
            f"- test any-edge F1: `{best['test_any_edge_f1']:.6f}`",
            f"- test end-to-end active action RMSE: `{best['test_active_value_rmse']:.6f}`",
            f"- test budgeted active action RMSE: `{best['test_budget_active_value_rmse']:.6f}`",
            f"- zero-policy active action RMSE: `{best['test_zero_active_value_rmse']:.6f}`",
            "",
            "## Interpretation",
            "",
            "- Logged scheduler actions are extremely sparse, so the specialist uses negative downsampling and per-action thresholds.",
            "- This module answers where the current action `a` comes from: in logs it comes from the simulator scheduler; in PI-JWM it can be approximated by this supervised `s -> a` policy.",
            "- The current result is an offline diagnostic and a policy-prior candidate for future closed-loop rollout.",
            "",
        ]
    )


if __name__ == "__main__":
    main()
