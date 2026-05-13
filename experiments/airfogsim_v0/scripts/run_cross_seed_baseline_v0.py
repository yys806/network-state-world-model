import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from train_baseline_v0 import (
    LINK_TYPES,
    build_xy,
    clip_outputs,
    fit_best_ridge,
    inverse_standardize,
    metrics,
    predict_ridge,
    standardize,
)


EXAMPLE_DIR = Path(__file__).resolve().parent
DATASET_DIR = EXAMPLE_DIR / "outputs" / "dataset_multiseed_v0"
OUTPUT_DIR = DATASET_DIR / "cross_seed_baseline_v0"


def load_multiseed_dataset(dataset_dir):
    with np.load(dataset_dir / "dataset_multiseed_v0_samples.npz", allow_pickle=True) as data:
        arrays = {key: data[key] for key in data.files}
    edge_vocab = pd.read_csv(dataset_dir / "edge_vocab.csv")
    return arrays, edge_vocab


def split_by_seed(sample_seed, train_seeds=(0, 1, 2), val_seed=3, test_seed=4):
    sample_seed = np.asarray(sample_seed)
    train_idx = np.where(np.isin(sample_seed, train_seeds))[0]
    val_idx = np.where(sample_seed == val_seed)[0]
    test_idx = np.where(sample_seed == test_seed)[0]
    if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
        raise ValueError("Cross-seed split produced an empty partition.")
    return train_idx, val_idx, test_idx


def evaluate_model(name, y_true, y_pred, link_dim, split_name):
    return {
        "split": split_name,
        "model": name,
        **metrics(y_true, y_pred, link_dim),
    }


def plot_metric_bars(output_dir, metrics_df):
    subset = metrics_df[metrics_df["split"] == "test_seed_4"].copy()
    names = subset["model"].tolist()
    values = subset["all_rmse"].tolist()
    colors = ["#6b7280", "#2563eb"][: len(values)]

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.bar(names, values, color=colors)
    ax.set_ylabel("RMSE")
    ax.set_title("Cross-seed test error on held-out seed 4")
    ax.grid(axis="y", alpha=0.25)
    for idx, value in enumerate(values):
        ax.text(idx, value, f"{value:.3f}", ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    path = output_dir / "cross_seed_rmse_bar.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_seed4_link_predictions(output_dir, y_true, predictions, meta):
    horizon = meta["horizon"]
    fig, axes = plt.subplots(len(LINK_TYPES), horizon, figsize=(14, 7), sharex=True)
    if axes.ndim == 1:
        axes = axes.reshape(len(LINK_TYPES), horizon)
    x_axis = np.arange(len(y_true))
    for type_idx, link_type in enumerate(LINK_TYPES):
        for step in range(horizon):
            target_idx = step * len(LINK_TYPES) + type_idx
            ax = axes[type_idx, step]
            ax.plot(x_axis, y_true[:, target_idx], label="true", color="#111827", lw=1.8)
            for name, pred in predictions.items():
                ax.plot(x_axis, pred[:, target_idx], label=name, lw=1.25, alpha=0.9)
            ax.set_title(f"{link_type}, t+{step + 1}")
            ax.grid(alpha=0.25)
    axes[0, 0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Held-out seed 4: future mean link rate by type")
    fig.tight_layout()
    path = output_dir / "cross_seed_link_rate_predictions_seed4.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_seed4_task_predictions(output_dir, y_true, predictions, meta):
    task_features = meta["task_features"]
    horizon = meta["horizon"]
    selected = ["num_tasks", "num_to_offload", "num_finished"]
    feature_indices = [task_features.index(name) for name in selected]
    fig, axes = plt.subplots(len(selected), horizon, figsize=(14, 7), sharex=True)
    if axes.ndim == 1:
        axes = axes.reshape(len(selected), horizon)
    x_axis = np.arange(len(y_true))
    for row_idx, feat_idx in enumerate(feature_indices):
        for step in range(horizon):
            target_idx = meta["link_target_dim"] + step * len(task_features) + feat_idx
            ax = axes[row_idx, step]
            ax.plot(x_axis, y_true[:, target_idx], label="true", color="#111827", lw=1.8)
            for name, pred in predictions.items():
                ax.plot(x_axis, pred[:, target_idx], label=name, lw=1.25, alpha=0.9)
            ax.set_title(f"{selected[row_idx]}, t+{step + 1}")
            ax.grid(alpha=0.25)
    axes[0, 0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Held-out seed 4: future task statistics")
    fig.tight_layout()
    path = output_dir / "cross_seed_task_predictions_seed4.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_report(output_dir, summary, metrics_df):
    test = metrics_df[metrics_df["split"] == "test_seed_4"].set_index("model")
    persistence_rmse = float(test.loc["persistence", "all_rmse"])
    ridge_rmse = float(test.loc["ridge_residual", "all_rmse"])
    delta = ridge_rmse - persistence_rmse
    if delta < 0:
        interpretation = (
            "Ridge residual improves over persistence on the held-out seed. "
            "This suggests the compact baseline captures part of the cross-seed transition pattern."
        )
    else:
        interpretation = (
            "Ridge residual is worse than persistence on the held-out seed. "
            "This means the current compact baseline has limited cross-seed generalization, "
            "and a structured graph/world-model design is still necessary."
        )

    lines = [
        "# Cross-seed baseline report v0",
        "",
        "## Goal",
        "",
        "This experiment checks whether the current training sample format can support evaluation across different stochastic AirFogSim trajectories.",
        "",
        "## Split",
        "",
        "- Train: seed 0, seed 1, seed 2",
        "- Validation: seed 3",
        "- Test: seed 4",
        "- Input: history window of node/link/task tensors",
        "- Target: future link-rate statistics and task-state statistics",
        "",
        "## Metrics",
        "",
        metrics_df.to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        interpretation,
        "",
        "This result should be presented as a generalization check, not as the final world-model result.",
        "",
        "## Outputs",
        "",
    ]
    for key, value in summary["outputs"].items():
        lines.append(f"- {key}: `{value}`")
    path = output_dir / "cross_seed_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    arrays, edge_vocab = load_multiseed_dataset(DATASET_DIR)
    x, y, persistence, meta = build_xy(arrays, edge_vocab)
    train_idx, val_idx, test_idx = split_by_seed(arrays["sample_seed"])

    (x_train, x_val, x_test), x_mean, x_std = standardize(
        x[train_idx], x[train_idx], x[val_idx], x[test_idx]
    )
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]
    persistence_train = persistence[train_idx]
    persistence_val = persistence[val_idx]
    persistence_test = persistence[test_idx]

    y_train_res = y_train - persistence_train
    y_val_res = y_val - persistence_val
    (y_train_s, y_val_s), y_mean, y_std = standardize(y_train_res, y_train_res, y_val_res)

    ridge_alpha, ridge_weights, ridge_val_mse = fit_best_ridge(
        x_train,
        y_train_s,
        x_val,
        y_val_s,
        alphas=[0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0],
    )
    ridge_val = clip_outputs(
        persistence_val + inverse_standardize(predict_ridge(x_val, ridge_weights), y_mean, y_std)
    )
    ridge_test = clip_outputs(
        persistence_test + inverse_standardize(predict_ridge(x_test, ridge_weights), y_mean, y_std)
    )

    rows = []
    rows.append(evaluate_model("persistence", y_val, persistence_val, meta["link_target_dim"], "val_seed_3"))
    rows.append(evaluate_model("ridge_residual", y_val, ridge_val, meta["link_target_dim"], "val_seed_3"))
    rows.append(evaluate_model("persistence", y_test, persistence_test, meta["link_target_dim"], "test_seed_4"))
    rows.append(evaluate_model("ridge_residual", y_test, ridge_test, meta["link_target_dim"], "test_seed_4"))
    metrics_df = pd.DataFrame(rows)
    metrics_path = OUTPUT_DIR / "cross_seed_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    predictions = {
        "persistence": persistence_test,
        "ridge": ridge_test,
    }
    bar_plot = plot_metric_bars(OUTPUT_DIR, metrics_df)
    link_plot = plot_seed4_link_predictions(OUTPUT_DIR, y_test, predictions, meta)
    task_plot = plot_seed4_task_predictions(OUTPUT_DIR, y_test, predictions, meta)

    summary = {
        "dataset_dir": str(DATASET_DIR),
        "output_dir": str(OUTPUT_DIR),
        "split": {
            "train_seeds": [0, 1, 2],
            "val_seed": 3,
            "test_seed": 4,
            "train_samples": int(len(train_idx)),
            "val_samples": int(len(val_idx)),
            "test_samples": int(len(test_idx)),
        },
        "meta": meta,
        "ridge": {"selected_alpha": float(ridge_alpha), "val_mse_scaled": float(ridge_val_mse)},
        "metrics": rows,
        "outputs": {
            "metrics_csv": str(metrics_path),
            "bar_plot": str(bar_plot),
            "link_plot": str(link_plot),
            "task_plot": str(task_plot),
        },
    }
    summary_path = OUTPUT_DIR / "cross_seed_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = write_report(OUTPUT_DIR, summary, metrics_df)
    summary["outputs"]["report_md"] = str(report_path)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
