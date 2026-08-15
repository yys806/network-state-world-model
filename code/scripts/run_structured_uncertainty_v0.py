import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_structured_dual_branch_baseline_v0 import (
    build_branch_features,
    build_targets_and_persistence,
    fit_standardizer,
    load_arrays,
    predict_model,
    split_by_seed,
    train_model,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "reports" / "structured_uncertainty_v0"
FIGURE_DIR = ROOT / "figures"
INTERVALS = [0.80, 0.90]


def interval_metrics(y_true, pred, residual_abs, link_dim):
    rows = []
    for level in INTERVALS:
        q = level
        width = np.quantile(residual_abs, q, axis=0, keepdims=True)
        lower = np.maximum(pred - width, 0.0)
        upper = pred + width
        covered = (y_true >= lower) & (y_true <= upper)
        for name, sl in {
            "all": slice(None),
            "link_rate_by_type": slice(0, link_dim),
            "task_state": slice(link_dim, None),
        }.items():
            rows.append(
                {
                    "interval": f"{int(level * 100)}%",
                    "target_group": name,
                    "coverage": float(np.mean(covered[:, sl])),
                    "mean_width": float(np.mean((upper - lower)[:, sl])),
                }
            )
    return rows


def plot_intervals(y_true, pred, residual_abs, meta, output_path):
    task_features = meta["task_features"]
    feat = "num_finished"
    feat_idx = task_features.index(feat)
    step = 2
    target_idx = meta["link_target_dim"] + step * len(task_features) + feat_idx
    width90 = np.quantile(residual_abs, 0.90, axis=0)[target_idx]
    x = np.arange(len(y_true))
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.plot(x, y_true[:, target_idx], label="true", color="#111827", lw=1.8)
    ax.plot(x, pred[:, target_idx], label="prediction", color="#2563eb", lw=1.5)
    ax.fill_between(
        x,
        np.maximum(pred[:, target_idx] - width90, 0.0),
        pred[:, target_idx] + width90,
        color="#60a5fa",
        alpha=0.25,
        label="90% interval",
    )
    ax.set_title("Structured state-action uncertainty: num_finished, t+3")
    ax.set_xlabel("test sample index")
    ax.set_ylabel(feat)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_report(metrics_df, summary):
    pivot = metrics_df.pivot(index="interval", columns="target_group", values="coverage")
    lines = [
        "# Structured uncertainty report v0",
        "",
        "## Purpose",
        "",
        "The previous uncertainty experiment was built on the Ridge residual baseline. This report applies the same validation-residual quantile idea to the structured state-action model, so uncertainty evaluation is aligned with the current model direction.",
        "",
        "## Method",
        "",
        "- Train structured state-action model on seeds 0, 1, 2.",
        "- Use seed 3 validation residuals to estimate absolute residual quantiles.",
        "- Apply the resulting 80% and 90% intervals on held-out seed 4.",
        "",
        "## Metrics",
        "",
        metrics_df.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Key takeaways",
        "",
        f"- 90% all-target coverage: `{float(pivot.loc['90%', 'all']):.3f}`.",
        f"- 90% task-state coverage: `{float(pivot.loc['90%', 'task_state']):.3f}`.",
        f"- 80% all-target coverage: `{float(pivot.loc['80%', 'all']):.3f}`.",
        "- This remains a lightweight residual-quantile interval, not a full Bayesian uncertainty model.",
        "",
        "## Outputs",
        "",
    ]
    for key, value in summary["outputs"].items():
        lines.append(f"- {key}: `{value}`")
    path = OUTPUT_DIR / "structured_uncertainty_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    arrays, actions, node_vocab, edge_vocab = load_arrays()
    features = build_branch_features(arrays, actions, node_vocab, edge_vocab)
    y, persistence = build_targets_and_persistence(arrays, edge_vocab)
    train_idx, val_idx, test_idx = split_by_seed(arrays["sample_seed"])
    y_res = y - persistence
    y_mean, y_std = fit_standardizer(y_res[train_idx])
    y_res_scaled = ((y_res - y_mean) / y_std).astype(np.float32)

    model, stats, _, train_info = train_model(features, y_res_scaled, train_idx, val_idx, use_action=True, seed=73)

    val_res_pred = predict_model(model, features, val_idx, stats, use_action=True)
    test_res_pred = predict_model(model, features, test_idx, stats, use_action=True)
    val_pred = np.maximum(persistence[val_idx] + val_res_pred * y_std + y_mean, 0.0)
    test_pred = np.maximum(persistence[test_idx] + test_res_pred * y_std + y_mean, 0.0)
    val_abs_residual = np.abs(y[val_idx] - val_pred)

    meta = {
        "horizon": int(arrays["y_link"].shape[1]),
        "link_target_dim": int(3 * arrays["y_link"].shape[1]),
        "task_features": arrays["task_features"].tolist(),
    }
    rows = interval_metrics(y[test_idx], test_pred, val_abs_residual, meta["link_target_dim"])
    metrics_df = pd.DataFrame(rows)
    metrics_path = OUTPUT_DIR / "structured_uncertainty_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    figure_path = FIGURE_DIR / "structured_uncertainty_prediction_intervals.png"
    plot_intervals(y[test_idx], test_pred, val_abs_residual, meta, figure_path)

    summary = {
        "training": train_info,
        "method": "validation absolute residual quantiles",
        "outputs": {
            "metrics_csv": str(metrics_path),
            "figure": str(figure_path),
        },
    }
    report_path = write_report(metrics_df, summary)
    summary["outputs"]["report_md"] = str(report_path)
    summary_path = OUTPUT_DIR / "structured_uncertainty_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
