import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

from train_baseline_v0 import fit_best_ridge, inverse_standardize, predict_ridge, standardize
from run_edge_level_link_prediction_v0 import (
    LINK_TYPES,
    build_edge_samples,
    evaluate_model,
    load_inputs,
    split_by_seed,
    train_ridge_residual,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "reports" / "two_stage_link_model_v0"
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
        max_iter=2000,
        tol=1e-4,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(x_train, y_active[train_idx])
    val_prob = clf.predict_proba(x_val)[:, 1].astype(np.float32)
    test_prob = clf.predict_proba(x_test)[:, 1].astype(np.float32)
    return {
        "classifier": clf,
        "mean": mean,
        "std": std,
        "val_prob": val_prob,
        "test_prob": test_prob,
    }


def train_active_rate_regressor(x, y, train_idx, val_idx, test_idx, y_active):
    active_train = train_idx[y_active[train_idx] == 1]
    if len(active_train) < 10:
        raise ValueError("Too few active training edges for rate regression.")
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


def choose_threshold(y_val, val_prob, val_rate_pred):
    y_active = active_label(y_val)
    candidates = np.unique(
        np.concatenate(
            [
                np.linspace(0.01, 0.99, 99, dtype=np.float32),
                np.quantile(val_prob, np.linspace(0.50, 0.995, 40)).astype(np.float32),
            ]
        )
    )
    rows = []
    best = None
    for threshold in candidates:
        pred_active = val_prob >= threshold
        pred = np.where(pred_active[:, None], val_rate_pred, 0.0)
        err = pred - y_val
        rmse = float(np.sqrt(np.mean(err**2)))
        mae = float(np.mean(np.abs(err)))
        f1 = float(f1_score(y_active, pred_active, zero_division=0))
        precision = float(precision_score(y_active, pred_active, zero_division=0))
        recall = float(recall_score(y_active, pred_active, zero_division=0))
        row = {
            "threshold": float(threshold),
            "rmse": rmse,
            "mae": mae,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "predicted_active": int(pred_active.sum()),
        }
        rows.append(row)
        if best is None or (rmse, -f1) < (best["rmse"], -best["f1"]):
            best = row
    return best, pd.DataFrame(rows)


def predict_topk_by_sample(prob, rate_pred, sample_id, k):
    pred_active = np.zeros(len(prob), dtype=bool)
    if k > 0:
        for sid in np.unique(sample_id):
            idx = np.where(sample_id == sid)[0]
            if len(idx) == 0:
                continue
            take = idx[np.argsort(prob[idx])[-min(k, len(idx)) :]]
            pred_active[take] = True
    return np.where(pred_active[:, None], rate_pred, 0.0).astype(np.float32), pred_active


def choose_topk(y_val, val_prob, val_rate_pred, sample_id_val, max_k=5):
    y_active = active_label(y_val)
    rows = []
    best = None
    for k in range(max_k + 1):
        pred, pred_active = predict_topk_by_sample(val_prob, val_rate_pred, sample_id_val, k)
        err = pred - y_val
        rmse = float(np.sqrt(np.mean(err**2)))
        mae = float(np.mean(np.abs(err)))
        f1 = float(f1_score(y_active, pred_active, zero_division=0))
        precision = float(precision_score(y_active, pred_active, zero_division=0))
        recall = float(recall_score(y_active, pred_active, zero_division=0))
        row = {
            "topk": int(k),
            "rmse": rmse,
            "mae": mae,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "predicted_active": int(pred_active.sum()),
        }
        rows.append(row)
        if best is None or (rmse, -f1) < (best["rmse"], -best["f1"]):
            best = row
    return best, pd.DataFrame(rows)


def classification_metrics(split, y_true, prob, pred, method):
    out = {
        "split": split,
        "method": method,
        "active_ratio": float(y_true.mean()),
        "predicted_active_ratio": float(pred.mean()),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "average_precision": float(average_precision_score(y_true, prob)),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, prob))
    except ValueError:
        out["roc_auc"] = float("nan")
    return out


def plot_classifier_pr(threshold_curve):
    path = FIGURE_DIR / "two_stage_link_threshold_curve.png"
    plt.figure(figsize=(7.6, 4.2))
    plt.plot(threshold_curve["threshold"], threshold_curve["rmse"], label="validation RMSE", color="#1F77B4")
    ax2 = plt.gca().twinx()
    ax2.plot(threshold_curve["threshold"], threshold_curve["f1"], label="validation F1", color="#D62728")
    plt.gca().set_xlabel("activity threshold")
    plt.gca().set_ylabel("RMSE")
    ax2.set_ylabel("F1")
    plt.title("Two-stage link model threshold selection")
    lines, labels = plt.gca().get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    plt.legend(lines + lines2, labels + labels2, loc="upper right")
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()
    return path


def plot_topk_curve(topk_curve):
    path = FIGURE_DIR / "two_stage_link_topk_curve.png"
    plt.figure(figsize=(7.6, 4.2))
    plt.plot(topk_curve["topk"], topk_curve["rmse"], marker="o", label="validation RMSE", color="#1F77B4")
    ax2 = plt.gca().twinx()
    ax2.plot(topk_curve["topk"], topk_curve["f1"], marker="o", label="validation F1", color="#D62728")
    plt.gca().set_xlabel("selected active edges per original sample")
    plt.gca().set_ylabel("RMSE")
    ax2.set_ylabel("F1")
    plt.title("Top-k activity selection")
    lines, labels = plt.gca().get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    plt.legend(lines + lines2, labels + labels2, loc="upper right")
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()
    return path


def plot_rmse_compare(metrics_df):
    test = metrics_df[(metrics_df["split"] == "test_seed_4") & (metrics_df["link_type"] == "all")]
    order = [
        "zero_rate",
        "persistence",
        "edge_state_action_ridge",
        "two_stage_threshold",
        "two_stage_topk",
        "oracle_activity_regression",
    ]
    test = test.set_index("model").loc[order].reset_index()
    path = FIGURE_DIR / "two_stage_link_rmse_compare.png"
    colors = ["#B8C0CC", "#8A94A6", "#1F77B4", "#D62728", "#FF7F0E", "#2CA02C"]
    plt.figure(figsize=(8.2, 4.3))
    bars = plt.bar(test["model"], test["rmse"], color=colors)
    plt.ylabel("RMSE of edge-level rate_sum")
    plt.title("Two-stage edge-level link model on held-out seed 4")
    plt.xticks(rotation=14, ha="right")
    for bar in bars:
        value = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()
    return path


def write_report(summary, metrics_df, cls_df, threshold_df):
    test_all = metrics_df[(metrics_df["split"] == "test_seed_4") & (metrics_df["link_type"] == "all")].set_index("model")
    lines = [
        "# Two-stage link model report v0",
        "",
        "## Goal",
        "",
        "This experiment treats link prediction as a sparse two-stage problem. The first stage predicts whether an edge will be active in the future horizon. The second stage predicts future `rate_sum` only for edges predicted as active.",
        "",
        "## Why this matters",
        "",
        f"Only `{summary['activity']['active_edge_items']}` of `{summary['activity']['total_edge_items']}` edge-level items are active, so the active ratio is `{summary['activity']['active_ratio']:.4f}`. A pure regression model is easily biased by inactive edges.",
        "",
        "## Models",
        "",
        "- `zero_rate`: always predicts zero link rate.",
        "- `persistence`: repeats the last observed rate of the same edge.",
        "- `edge_state_action_ridge`: one-stage residual regression with strict actions.",
        "- `two_stage_threshold`: activity classifier plus active-edge rate regressor, using a validation-selected probability threshold.",
        "- `two_stage_topk`: selects a fixed number of likely active edges per original sample, then applies the same rate regressor.",
        "- `oracle_activity_regression`: uses true future activity as an upper-bound diagnostic, then applies the same rate regressor.",
        "",
        "## Activity classifier",
        "",
        cls_df.to_markdown(index=False),
        "",
        f"Selected threshold from validation: `{summary['threshold']['threshold']:.4f}`.",
        f"Selected top-k from validation: `{summary['topk']['topk']}` edge(s) per original sample.",
        "",
        "## Prediction metrics",
        "",
        metrics_df.to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        f"- On test seed 4, persistence all-edge RMSE is `{test_all.loc['persistence', 'rmse']:.3f}`.",
        f"- One-stage edge-state-action Ridge all-edge RMSE is `{test_all.loc['edge_state_action_ridge', 'rmse']:.3f}`.",
        f"- Threshold two-stage model all-edge RMSE is `{test_all.loc['two_stage_threshold', 'rmse']:.3f}`.",
        f"- Top-k two-stage model all-edge RMSE is `{test_all.loc['two_stage_topk', 'rmse']:.3f}`.",
        f"- Oracle activity regression all-edge RMSE is `{test_all.loc['oracle_activity_regression', 'rmse']:.3f}`.",
        "- If the two-stage model is much worse than oracle activity, the bottleneck is activity detection. If both are weak, the rate regressor also needs better graph features.",
        "- This experiment converts the link-side problem from one vague regression target into two concrete subproblems: future activity detection and active-edge rate regression.",
        "",
        "## Outputs",
        "",
    ]
    for key, value in summary["outputs"].items():
        lines.append(f"- {key}: `{value}`")
    path = OUTPUT_DIR / "two_stage_link_model_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    arrays, actions, node_vocab, edge_vocab = load_inputs()
    x_state, x_state_action, y, persistence, sample_seed, edge_index, link_type, meta = build_edge_samples(
        arrays, actions, node_vocab, edge_vocab
    )
    sample_id = np.repeat(np.arange(meta["num_original_samples"]), meta["num_edges"])
    y_active = active_label(y)
    train_idx, val_idx, test_idx = split_by_seed(sample_seed)

    classifier = train_activity_classifier(x_state_action, y_active, train_idx, val_idx, test_idx)
    rate_regressor = train_active_rate_regressor(x_state_action, y, train_idx, val_idx, test_idx, y_active)
    best_threshold, threshold_df = choose_threshold(y[val_idx], classifier["val_prob"], rate_regressor["val_rate_pred"])
    threshold = best_threshold["threshold"]
    best_topk, topk_df = choose_topk(
        y[val_idx],
        classifier["val_prob"],
        rate_regressor["val_rate_pred"],
        sample_id[val_idx],
        max_k=5,
    )

    edge_action = train_ridge_residual(x_state_action, y, persistence, train_idx, val_idx, test_idx)
    zero_val = np.zeros_like(y[val_idx])
    zero_test = np.zeros_like(y[test_idx])
    threshold_val_active = classifier["val_prob"] >= threshold
    threshold_test_active = classifier["test_prob"] >= threshold
    two_stage_threshold_val = np.where(
        (classifier["val_prob"] >= threshold)[:, None],
        rate_regressor["val_rate_pred"],
        0.0,
    ).astype(np.float32)
    two_stage_threshold_test = np.where(
        (classifier["test_prob"] >= threshold)[:, None],
        rate_regressor["test_rate_pred"],
        0.0,
    ).astype(np.float32)
    two_stage_topk_val, topk_val_active = predict_topk_by_sample(
        classifier["val_prob"],
        rate_regressor["val_rate_pred"],
        sample_id[val_idx],
        best_topk["topk"],
    )
    two_stage_topk_test, topk_test_active = predict_topk_by_sample(
        classifier["test_prob"],
        rate_regressor["test_rate_pred"],
        sample_id[test_idx],
        best_topk["topk"],
    )
    oracle_val = np.where(y_active[val_idx][:, None] == 1, rate_regressor["val_rate_pred"], 0.0).astype(np.float32)
    oracle_test = np.where(y_active[test_idx][:, None] == 1, rate_regressor["test_rate_pred"], 0.0).astype(np.float32)

    rows = []
    val_lt = link_type[val_idx]
    test_lt = link_type[test_idx]
    models = [
        ("zero_rate", zero_val, zero_test),
        ("persistence", persistence[val_idx], persistence[test_idx]),
        ("edge_state_action_ridge", edge_action["val_pred"], edge_action["test_pred"]),
        ("two_stage_threshold", two_stage_threshold_val, two_stage_threshold_test),
        ("two_stage_topk", two_stage_topk_val, two_stage_topk_test),
        ("oracle_activity_regression", oracle_val, oracle_test),
    ]
    for name, val_pred, test_pred in models:
        rows.extend(evaluate_model("val_seed_3", name, y[val_idx], val_pred, val_lt))
        rows.extend(evaluate_model("test_seed_4", name, y[test_idx], test_pred, test_lt))
    metrics_df = pd.DataFrame(rows)
    metrics_path = OUTPUT_DIR / "two_stage_link_model_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    cls_rows = [
        classification_metrics("val_seed_3", y_active[val_idx], classifier["val_prob"], threshold_val_active, "threshold"),
        classification_metrics("test_seed_4", y_active[test_idx], classifier["test_prob"], threshold_test_active, "threshold"),
        classification_metrics("val_seed_3", y_active[val_idx], classifier["val_prob"], topk_val_active, "topk"),
        classification_metrics("test_seed_4", y_active[test_idx], classifier["test_prob"], topk_test_active, "topk"),
    ]
    cls_df = pd.DataFrame(cls_rows)
    cls_path = OUTPUT_DIR / "two_stage_link_activity_metrics.csv"
    cls_df.to_csv(cls_path, index=False, encoding="utf-8-sig")
    threshold_path = OUTPUT_DIR / "two_stage_link_threshold_curve.csv"
    threshold_df.to_csv(threshold_path, index=False, encoding="utf-8-sig")
    topk_path = OUTPUT_DIR / "two_stage_link_topk_curve.csv"
    topk_df.to_csv(topk_path, index=False, encoding="utf-8-sig")

    threshold_plot = plot_classifier_pr(threshold_df)
    topk_plot = plot_topk_curve(topk_df)
    rmse_plot = plot_rmse_compare(metrics_df)

    summary = {
        "dataset": meta,
        "activity": {
            "total_edge_items": int(len(y_active)),
            "active_edge_items": int(y_active.sum()),
            "active_ratio": float(y_active.mean()),
            "train_active": int(y_active[train_idx].sum()),
            "val_active": int(y_active[val_idx].sum()),
            "test_active": int(y_active[test_idx].sum()),
        },
        "threshold": best_threshold,
        "topk": best_topk,
        "classifier": {
            "type": "SGDClassifier(log_loss, class_weight=balanced)",
        },
        "active_rate_regressor": {
            "selected_alpha": rate_regressor["alpha"],
            "active_train_edges": rate_regressor["active_train_edges"],
            "val_mse_scaled": rate_regressor["val_mse_scaled"],
        },
        "one_stage_edge_action": {
            "selected_alpha": edge_action["alpha"],
            "val_mse_scaled": edge_action["val_mse_scaled"],
        },
        "outputs": {
            "metrics_csv": str(metrics_path),
            "activity_metrics_csv": str(cls_path),
            "threshold_curve_csv": str(threshold_path),
            "topk_curve_csv": str(topk_path),
            "threshold_curve_plot": str(threshold_plot),
            "topk_curve_plot": str(topk_plot),
            "rmse_compare_plot": str(rmse_plot),
        },
    }
    summary_path = OUTPUT_DIR / "two_stage_link_model_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = write_report(summary, metrics_df, cls_df, threshold_df)
    summary["outputs"]["report_md"] = str(report_path)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
