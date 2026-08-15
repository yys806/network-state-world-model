import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from train_baseline_v0 import (
    DATASET_DIR,
    LINK_TYPES,
    build_xy,
    chronological_split,
    clip_outputs,
    fit_best_ridge,
    load_dataset,
    predict_ridge,
    standardize,
)


OUTPUT_DIR = DATASET_DIR / "uncertainty_v0"
ALPHAS = [0.10, 0.20]


def fit_residual_ridge(arrays, edge_vocab):
    x, y, persistence, meta = build_xy(arrays, edge_vocab)
    train_idx, val_idx, test_idx = chronological_split(len(x))
    (x_train, x_val, x_test), x_mean, x_std = standardize(x[train_idx], x[train_idx], x[val_idx], x[test_idx])

    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]
    persistence_train = persistence[train_idx]
    persistence_val = persistence[val_idx]
    persistence_test = persistence[test_idx]
    y_train_res = y_train - persistence_train
    y_val_res = y_val - persistence_val

    (y_train_s, y_val_s, _), y_mean, y_std = standardize(
        y_train_res,
        y_train_res,
        y_val_res,
        y_test - persistence_test,
    )
    alpha, weights, val_mse = fit_best_ridge(
        x_train,
        y_train_s,
        x_val,
        y_val_s,
        alphas=[0.1, 1.0, 10.0, 100.0, 1000.0],
    )

    val_pred = clip_outputs(persistence_val + predict_ridge(x_val, weights) * y_std + y_mean)
    test_pred = clip_outputs(persistence_test + predict_ridge(x_test, weights) * y_std + y_mean)
    return {
        "y_val": y_val,
        "val_pred": val_pred,
        "y_test": y_test,
        "test_pred": test_pred,
        "meta": meta,
        "alpha": alpha,
        "val_mse_scaled": val_mse,
    }


def interval_from_residuals(y_val, val_pred, test_pred, alpha):
    residual = y_val - val_pred
    low_q = np.quantile(residual, alpha / 2, axis=0, keepdims=True)
    high_q = np.quantile(residual, 1 - alpha / 2, axis=0, keepdims=True)
    lower = np.maximum(0.0, test_pred + low_q)
    upper = np.maximum(lower, test_pred + high_q)
    return lower, upper


def summarize_interval(y_true, pred, lower, upper, link_dim):
    coverage = (y_true >= lower) & (y_true <= upper)
    width = upper - lower
    out = {}
    slices = {
        "all": slice(None),
        "link_rate_by_type": slice(0, link_dim),
        "task_state": slice(link_dim, None),
    }
    err = pred - y_true
    for name, sl in slices.items():
        out[f"{name}_mae"] = float(np.mean(np.abs(err[:, sl])))
        out[f"{name}_coverage"] = float(np.mean(coverage[:, sl]))
        out[f"{name}_mean_width"] = float(np.mean(width[:, sl]))
    return out


def plot_interval(output_dir, y_true, pred, lower, upper, meta):
    output_dir.mkdir(parents=True, exist_ok=True)
    task_features = meta["task_features"]
    link_dim = meta["link_target_dim"]

    targets = [
        ("U2I rate, t+1", 0),
        ("V2I rate, t+1", 1),
        ("num_to_offload, t+1", link_dim + task_features.index("num_to_offload")),
        ("num_finished, t+3", link_dim + 2 * len(task_features) + task_features.index("num_finished")),
    ]
    x_axis = np.arange(len(y_true))
    fig, axes = plt.subplots(len(targets), 1, figsize=(12, 9), sharex=True)
    for ax, (title, idx) in zip(axes, targets):
        ax.fill_between(x_axis, lower[:, idx], upper[:, idx], color="#93c5fd", alpha=0.35, label="80% interval")
        ax.plot(x_axis, y_true[:, idx], color="#111827", lw=1.7, label="true")
        ax.plot(x_axis, pred[:, idx], color="#2563eb", lw=1.5, label="ridge mean")
        ax.set_title(title)
        ax.grid(alpha=0.25)
    axes[0].legend(loc="upper left")
    axes[-1].set_xlabel("test sample index")
    fig.suptitle("Uncertainty interval from validation residual quantiles")
    fig.tight_layout()
    path = output_dir / "uncertainty_prediction_intervals.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    arrays, edge_vocab = load_dataset(DATASET_DIR)
    fit = fit_residual_ridge(arrays, edge_vocab)
    rows = []
    interval_for_plot = None
    for alpha in ALPHAS:
        lower, upper = interval_from_residuals(fit["y_val"], fit["val_pred"], fit["test_pred"], alpha)
        rows.append(
            {
                "interval": f"{int((1 - alpha) * 100)}%",
                "alpha": alpha,
                **summarize_interval(
                    fit["y_test"],
                    fit["test_pred"],
                    lower,
                    upper,
                    fit["meta"]["link_target_dim"],
                ),
            }
        )
        if alpha == 0.20:
            interval_for_plot = (lower, upper)

    metrics_df = pd.DataFrame(rows)
    metrics_path = OUTPUT_DIR / "uncertainty_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    plot_path = plot_interval(
        OUTPUT_DIR,
        fit["y_test"],
        fit["test_pred"],
        interval_for_plot[0],
        interval_for_plot[1],
        fit["meta"],
    )
    summary = {
        "dataset_dir": str(DATASET_DIR),
        "output_dir": str(OUTPUT_DIR),
        "method": "validation residual quantile interval around ridge residual prediction",
        "ridge_alpha": fit["alpha"],
        "metrics_csv": str(metrics_path),
        "plot_png": str(plot_path),
        "metrics": rows,
    }
    (OUTPUT_DIR / "uncertainty_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
