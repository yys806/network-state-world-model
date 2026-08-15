import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

from train_baseline_v0 import fit_best_ridge, inverse_standardize, predict_ridge, standardize
from run_edge_level_link_prediction_v0 import (
    build_edge_samples,
    evaluate_model,
    load_inputs,
    split_by_seed,
    train_ridge_residual,
)


ROOT = Path(__file__).resolve().parents[1]
EDGE_ACTION_DIR = ROOT / "datasets" / "edge_action_v0"
OUTPUT_DIR = ROOT / "reports" / "edge_action_link_model_v0"
FIGURE_DIR = ROOT / "figures"
ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]


def fit_standardizer(x, eps=1e-6):
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std = np.where(std < eps, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def transform(x, mean, std):
    return ((x - mean) / std).astype(np.float32)


def active_label(y):
    return np.any(y > 1e-6, axis=1).astype(np.int32)


def load_edge_actions():
    with np.load(EDGE_ACTION_DIR / "edge_action_v0_samples.npz", allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def build_edge_action_features(edge_actions):
    hist = edge_actions["edge_a_hist"].astype(np.float32)
    future = edge_actions["edge_a_future"].astype(np.float32)
    num_samples, history, num_edges, feat = hist.shape
    horizon = future.shape[1]
    hist_flat = hist.transpose(0, 2, 1, 3).reshape(num_samples * num_edges, history * feat)
    future_flat = future.transpose(0, 2, 1, 3).reshape(num_samples * num_edges, horizon * feat)
    hist_sum = hist.sum(axis=1).reshape(num_samples * num_edges, feat)
    future_sum = future.sum(axis=1).reshape(num_samples * num_edges, feat)
    return np.concatenate([hist_flat, future_flat, hist_sum, future_sum], axis=1).astype(np.float32)


def train_activity_classifier(x, y_active, train_idx, val_idx, test_idx):
    mean, std = fit_standardizer(x[train_idx])
    x_train = transform(x[train_idx], mean, std)
    x_val = transform(x[val_idx], mean, std)
    x_test = transform(x[test_idx], mean, std)
    clf = SGDClassifier(
        loss="log_loss",
        penalty="elasticnet",
        alpha=1e-4,
        l1_ratio=0.05,
        class_weight="balanced",
        max_iter=2500,
        tol=1e-4,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(x_train, y_active[train_idx])
    return {
        "val_prob": clf.predict_proba(x_val)[:, 1].astype(np.float32),
        "test_prob": clf.predict_proba(x_test)[:, 1].astype(np.float32),
    }


def train_active_rate_regressor(x, y, train_idx, val_idx, test_idx, y_active):
    active_train = train_idx[y_active[train_idx] == 1]
    x_parts, _, _ = standardize(x[active_train], x[active_train], x[val_idx], x[test_idx])
    x_train, x_val, x_test = [part.astype(np.float32) for part in x_parts]
    y_parts, y_mean, y_std = standardize(y[active_train], y[active_train], y[val_idx], y[test_idx])
    y_train, y_val, _ = [part.astype(np.float32) for part in y_parts]
    alpha, weights, val_mse = fit_best_ridge(x_train, y_train, x_val, y_val, ALPHAS)
    val_pred = np.clip(inverse_standardize(predict_ridge(x_val, weights), y_mean, y_std), 0.0, None)
    test_pred = np.clip(inverse_standardize(predict_ridge(x_test, weights), y_mean, y_std), 0.0, None)
    return {
        "alpha": float(alpha),
        "val_mse_scaled": float(val_mse),
        "active_train_edges": int(len(active_train)),
        "val_rate_pred": val_pred.astype(np.float32),
        "test_rate_pred": test_pred.astype(np.float32),
    }


def choose_threshold(y_val, prob, rate_pred):
    y_active = active_label(y_val)
    candidates = np.unique(
        np.concatenate(
            [
                np.linspace(0.01, 0.99, 99, dtype=np.float32),
                np.quantile(prob, np.linspace(0.50, 0.995, 40)).astype(np.float32),
            ]
        )
    )
    rows = []
    best_f1 = None
    best_rmse = None
    for threshold in candidates:
        pred_active = prob >= threshold
        pred = np.where(pred_active[:, None], rate_pred, 0.0)
        err = pred - y_val
        row = {
            "threshold": float(threshold),
            "rmse": float(np.sqrt(np.mean(err**2))),
            "mae": float(np.mean(np.abs(err))),
            "precision": float(precision_score(y_active, pred_active, zero_division=0)),
            "recall": float(recall_score(y_active, pred_active, zero_division=0)),
            "f1": float(f1_score(y_active, pred_active, zero_division=0)),
            "predicted_active": int(pred_active.sum()),
        }
        rows.append(row)
        if best_f1 is None or (row["f1"], -row["rmse"]) > (best_f1["f1"], -best_f1["rmse"]):
            best_f1 = row
        if best_rmse is None or (row["rmse"], -row["f1"]) < (best_rmse["rmse"], -best_rmse["f1"]):
            best_rmse = row
    return best_f1, best_rmse, pd.DataFrame(rows)


def classification_row(name, split, y_true, prob, pred_active):
    row = {
        "model": name,
        "split": split,
        "active_ratio": float(y_true.mean()),
        "predicted_active_ratio": float(pred_active.mean()),
        "precision": float(precision_score(y_true, pred_active, zero_division=0)),
        "recall": float(recall_score(y_true, pred_active, zero_division=0)),
        "f1": float(f1_score(y_true, pred_active, zero_division=0)),
        "average_precision": float(average_precision_score(y_true, prob)),
    }
    try:
        row["roc_auc"] = float(roc_auc_score(y_true, prob))
    except ValueError:
        row["roc_auc"] = float("nan")
    return row


def plot_activity_metrics(cls_df):
    test = cls_df[cls_df["split"] == "test_seed_4"].set_index("model")
    path = FIGURE_DIR / "edge_action_activity_metrics.png"
    metrics = ["precision", "recall", "f1", "average_precision", "roc_auc"]
    ax = test.loc[["global_action", "edge_action"], metrics].plot(kind="bar", figsize=(8.2, 4.4))
    ax.set_ylabel("score")
    ax.set_title("Future link activity detection with edge-level actions")
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()
    return path


def plot_rmse(metrics_df):
    test = metrics_df[(metrics_df["split"] == "test_seed_4") & (metrics_df["link_type"] == "all")]
    order = [
        "zero_rate",
        "edge_state_action_ridge",
        "edge_action_ridge",
        "edge_action_two_stage_f1",
        "edge_action_oracle_activity",
    ]
    test = test.set_index("model").loc[order].reset_index()
    path = FIGURE_DIR / "edge_action_link_rmse_compare.png"
    plt.figure(figsize=(8.6, 4.4))
    bars = plt.bar(test["model"], test["rmse"], color=["#B8C0CC", "#1F77B4", "#004BFF", "#D62728", "#2CA02C"])
    plt.ylabel("RMSE of edge-level rate_sum")
    plt.title("Edge-level actions for link prediction")
    plt.xticks(rotation=14, ha="right")
    for bar in bars:
        value = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()
    return path


def write_report(summary, metrics_df, cls_df):
    lines = [
        "# Edge-action link model report v0",
        "",
        "## Goal",
        "",
        "This experiment maps strict scheduler actions onto specific candidate edges, then checks whether edge-level action features improve future link activity detection and edge-level rate prediction.",
        "",
        "## Edge-action features",
        "",
        "For each edge and time step, the constructed features are:",
        "",
        "- `offload_count`",
        "- `rb_task_count`",
        "- `rb_total`",
        "- `cpu_task_count`",
        "- `cpu_total`",
        "- `return_count`",
        "",
        "The aligned tensors are:",
        "",
        f"- `edge_a_hist`: `{tuple(summary['edge_action_shapes']['edge_a_hist'])}`",
        f"- `edge_a_future`: `{tuple(summary['edge_action_shapes']['edge_a_future'])}`",
        "",
        "## Activity detection",
        "",
        cls_df.to_markdown(index=False),
        "",
        "## Rate prediction metrics",
        "",
        metrics_df.to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        "- Edge-level action features make the input more faithful to the actual scheduler decision path, because actions are no longer only global aggregates.",
        "- If activity precision/recall remains low, the missing factor is likely more detailed task-to-edge alignment or the fact that many scheduler actions are not included in the current candidate communication-edge vocabulary.",
        "- This step is still useful even without a large RMSE gain, because it establishes a traceable path from strict AirFogSim actions to edge-level world-model inputs.",
        "",
        "## Outputs",
        "",
    ]
    for key, value in summary["outputs"].items():
        lines.append(f"- {key}: `{value}`")
    path = OUTPUT_DIR / "edge_action_link_model_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    arrays, actions, node_vocab, edge_vocab = load_inputs()
    edge_actions = load_edge_actions()
    x_state, x_global_action, y, persistence, sample_seed, edge_index, link_type, meta = build_edge_samples(
        arrays, actions, node_vocab, edge_vocab
    )
    edge_action_features = build_edge_action_features(edge_actions)
    x_edge_action = np.concatenate([x_state, edge_action_features], axis=1).astype(np.float32)
    y_active = active_label(y)
    train_idx, val_idx, test_idx = split_by_seed(sample_seed)

    global_classifier = train_activity_classifier(x_global_action, y_active, train_idx, val_idx, test_idx)
    edge_classifier = train_activity_classifier(x_edge_action, y_active, train_idx, val_idx, test_idx)
    global_reg = train_active_rate_regressor(x_global_action, y, train_idx, val_idx, test_idx, y_active)
    edge_reg = train_active_rate_regressor(x_edge_action, y, train_idx, val_idx, test_idx, y_active)
    global_best_f1, _, global_thresholds = choose_threshold(y[val_idx], global_classifier["val_prob"], global_reg["val_rate_pred"])
    edge_best_f1, edge_best_rmse, edge_thresholds = choose_threshold(y[val_idx], edge_classifier["val_prob"], edge_reg["val_rate_pred"])

    one_stage_global = train_ridge_residual(x_global_action, y, persistence, train_idx, val_idx, test_idx)
    one_stage_edge = train_ridge_residual(x_edge_action, y, persistence, train_idx, val_idx, test_idx)

    zero_val = np.zeros_like(y[val_idx])
    zero_test = np.zeros_like(y[test_idx])
    edge_val_active_f1 = edge_classifier["val_prob"] >= edge_best_f1["threshold"]
    edge_test_active_f1 = edge_classifier["test_prob"] >= edge_best_f1["threshold"]
    edge_val_active_rmse = edge_classifier["val_prob"] >= edge_best_rmse["threshold"]
    edge_test_active_rmse = edge_classifier["test_prob"] >= edge_best_rmse["threshold"]
    edge_two_stage_f1_val = np.where(edge_val_active_f1[:, None], edge_reg["val_rate_pred"], 0.0).astype(np.float32)
    edge_two_stage_f1_test = np.where(edge_test_active_f1[:, None], edge_reg["test_rate_pred"], 0.0).astype(np.float32)
    edge_two_stage_rmse_val = np.where(edge_val_active_rmse[:, None], edge_reg["val_rate_pred"], 0.0).astype(np.float32)
    edge_two_stage_rmse_test = np.where(edge_test_active_rmse[:, None], edge_reg["test_rate_pred"], 0.0).astype(np.float32)
    edge_oracle_val = np.where(y_active[val_idx][:, None] == 1, edge_reg["val_rate_pred"], 0.0).astype(np.float32)
    edge_oracle_test = np.where(y_active[test_idx][:, None] == 1, edge_reg["test_rate_pred"], 0.0).astype(np.float32)

    val_lt = link_type[val_idx]
    test_lt = link_type[test_idx]
    rows = []
    models = [
        ("zero_rate", zero_val, zero_test),
        ("persistence", persistence[val_idx], persistence[test_idx]),
        ("edge_state_action_ridge", one_stage_global["val_pred"], one_stage_global["test_pred"]),
        ("edge_action_ridge", one_stage_edge["val_pred"], one_stage_edge["test_pred"]),
        ("edge_action_two_stage_f1", edge_two_stage_f1_val, edge_two_stage_f1_test),
        ("edge_action_two_stage_rmse", edge_two_stage_rmse_val, edge_two_stage_rmse_test),
        ("edge_action_oracle_activity", edge_oracle_val, edge_oracle_test),
    ]
    for name, val_pred, test_pred in models:
        rows.extend(evaluate_model("val_seed_3", name, y[val_idx], val_pred, val_lt))
        rows.extend(evaluate_model("test_seed_4", name, y[test_idx], test_pred, test_lt))
    metrics_df = pd.DataFrame(rows)
    metrics_path = OUTPUT_DIR / "edge_action_link_model_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    global_val_active = global_classifier["val_prob"] >= global_best_f1["threshold"]
    global_test_active = global_classifier["test_prob"] >= global_best_f1["threshold"]
    cls_df = pd.DataFrame(
        [
            classification_row("global_action", "val_seed_3", y_active[val_idx], global_classifier["val_prob"], global_val_active),
            classification_row("global_action", "test_seed_4", y_active[test_idx], global_classifier["test_prob"], global_test_active),
            classification_row("edge_action", "val_seed_3", y_active[val_idx], edge_classifier["val_prob"], edge_val_active_f1),
            classification_row("edge_action", "test_seed_4", y_active[test_idx], edge_classifier["test_prob"], edge_test_active_f1),
        ]
    )
    cls_path = OUTPUT_DIR / "edge_action_activity_metrics.csv"
    cls_df.to_csv(cls_path, index=False, encoding="utf-8-sig")
    global_thresholds.to_csv(OUTPUT_DIR / "global_action_threshold_curve.csv", index=False, encoding="utf-8-sig")
    edge_thresholds.to_csv(OUTPUT_DIR / "edge_action_threshold_curve.csv", index=False, encoding="utf-8-sig")

    activity_plot = plot_activity_metrics(cls_df)
    rmse_plot = plot_rmse(metrics_df)
    summary = {
        "dataset": meta,
        "edge_action_shapes": {
            "edge_a_hist": list(edge_actions["edge_a_hist"].shape),
            "edge_a_future": list(edge_actions["edge_a_future"].shape),
        },
        "feature_dims": {
            "x_global_action": int(x_global_action.shape[1]),
            "edge_action_features": int(edge_action_features.shape[1]),
            "x_edge_action": int(x_edge_action.shape[1]),
        },
        "activity": {
            "active_items": int(y_active.sum()),
            "total_items": int(len(y_active)),
            "active_ratio": float(y_active.mean()),
        },
        "thresholds": {
            "global_best_f1": global_best_f1,
            "edge_best_f1": edge_best_f1,
            "edge_best_rmse": edge_best_rmse,
        },
        "outputs": {
            "metrics_csv": str(metrics_path),
            "activity_metrics_csv": str(cls_path),
            "activity_plot": str(activity_plot),
            "rmse_plot": str(rmse_plot),
        },
    }
    summary_path = OUTPUT_DIR / "edge_action_link_model_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = write_report(summary, metrics_df, cls_df)
    summary["outputs"]["report_md"] = str(report_path)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
