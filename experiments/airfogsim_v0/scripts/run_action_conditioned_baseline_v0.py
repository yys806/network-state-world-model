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
from run_cross_seed_baseline_v0 import load_multiseed_dataset, split_by_seed


EXAMPLE_DIR = Path(__file__).resolve().parent
DATASET_DIR = EXAMPLE_DIR / "outputs" / "dataset_multiseed_v0"
ACTION_DIR = EXAMPLE_DIR / "outputs" / "strict_action_logs_v0"
OUTPUT_DIR = DATASET_DIR / "action_conditioned_baseline_v0"


def load_actions(action_dir):
    with np.load(action_dir / "strict_action_v0_samples.npz", allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def build_action_features(actions):
    a_hist = actions["a_hist"].astype(np.float32)
    a_future = actions["a_future"].astype(np.float32)
    # For action-conditioned rollout, future actions are treated as planned controls.
    return np.concatenate(
        [
            a_hist.reshape(len(a_hist), -1),
            a_future.reshape(len(a_future), -1),
        ],
        axis=1,
    ).astype(np.float32)


def train_ridge_residual(x, y, persistence, train_idx, val_idx, test_idx, alphas):
    (x_train, x_val, x_test), _, _ = standardize(x[train_idx], x[train_idx], x[val_idx], x[test_idx])
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]
    persistence_train = persistence[train_idx]
    persistence_val = persistence[val_idx]
    persistence_test = persistence[test_idx]

    y_train_res = y_train - persistence_train
    y_val_res = y_val - persistence_val
    (y_train_s, y_val_s), y_mean, y_std = standardize(y_train_res, y_train_res, y_val_res)
    alpha, weights, val_mse = fit_best_ridge(x_train, y_train_s, x_val, y_val_s, alphas=alphas)
    val_pred = clip_outputs(
        persistence_val + inverse_standardize(predict_ridge(x_val, weights), y_mean, y_std)
    )
    test_pred = clip_outputs(
        persistence_test + inverse_standardize(predict_ridge(x_test, weights), y_mean, y_std)
    )
    return {
        "alpha": float(alpha),
        "val_mse_scaled": float(val_mse),
        "val_pred": val_pred,
        "test_pred": test_pred,
    }


def evaluate_rows(model_name, y_val, y_test, val_pred, test_pred, link_dim):
    return [
        {"split": "val_seed_3", "model": model_name, **metrics(y_val, val_pred, link_dim)},
        {"split": "test_seed_4", "model": model_name, **metrics(y_test, test_pred, link_dim)},
    ]


def plot_rmse(output_dir, metrics_df):
    subset = metrics_df[metrics_df["split"] == "test_seed_4"].copy()
    order = ["persistence", "state_only_ridge", "state_action_ridge"]
    subset["model"] = pd.Categorical(subset["model"], categories=order, ordered=True)
    subset = subset.sort_values("model")

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    colors = ["#6b7280", "#2563eb", "#16a34a"]
    ax.bar(subset["model"].astype(str), subset["all_rmse"], color=colors[: len(subset)])
    ax.set_ylabel("RMSE")
    ax.set_title("Held-out seed 4: state-only vs action-conditioned baseline")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", labelrotation=15)
    for idx, value in enumerate(subset["all_rmse"]):
        ax.text(idx, value, f"{value:.3f}", ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    path = output_dir / "action_conditioned_rmse_bar.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_link_predictions(output_dir, y_true, predictions, meta):
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
                ax.plot(x_axis, pred[:, target_idx], label=name, lw=1.15, alpha=0.9)
            ax.set_title(f"{link_type}, t+{step + 1}")
            ax.grid(alpha=0.25)
    axes[0, 0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Held-out seed 4: link-rate prediction with strict actions")
    fig.tight_layout()
    path = output_dir / "action_conditioned_link_predictions_seed4.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_task_predictions(output_dir, y_true, predictions, meta):
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
                ax.plot(x_axis, pred[:, target_idx], label=name, lw=1.15, alpha=0.9)
            ax.set_title(f"{selected[row_idx]}, t+{step + 1}")
            ax.grid(alpha=0.25)
    axes[0, 0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Held-out seed 4: task-state prediction with strict actions")
    fig.tight_layout()
    path = output_dir / "action_conditioned_task_predictions_seed4.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_report(output_dir, summary, metrics_df):
    test = metrics_df[metrics_df["split"] == "test_seed_4"].set_index("model")
    state_rmse = float(test.loc["state_only_ridge", "all_rmse"])
    action_rmse = float(test.loc["state_action_ridge", "all_rmse"])
    delta = action_rmse - state_rmse
    if delta < 0:
        interpretation = (
            "Adding strict actions improves the Ridge residual baseline on the held-out seed. "
            "This suggests action variables contain useful transition information."
        )
    else:
        interpretation = (
            "Adding strict actions does not improve the current Ridge residual baseline on the held-out seed. "
            "This does not invalidate action-conditioned modeling; it indicates that a linear compact baseline "
            "cannot fully use the action signal yet."
        )

    lines = [
        "# Action-conditioned baseline report v0",
        "",
        "## Goal",
        "",
        "This experiment tests whether strict scheduler actions help cross-seed prediction.",
        "",
        "## Split",
        "",
        "- Train: seed 0, seed 1, seed 2",
        "- Validation: seed 3",
        "- Test: seed 4",
        "",
        "## Compared inputs",
        "",
        "- `persistence`: repeat the last observed future target.",
        "- `state_only_ridge`: historical node/link/task states only.",
        "- `state_action_ridge`: historical node/link/task states plus strict historical/future action tensors.",
        "",
        "## Metrics",
        "",
        metrics_df.to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        interpretation,
        "",
        "## Outputs",
        "",
    ]
    for key, value in summary["outputs"].items():
        lines.append(f"- {key}: `{value}`")
    path = output_dir / "action_conditioned_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    arrays, edge_vocab = load_multiseed_dataset(DATASET_DIR)
    actions = load_actions(ACTION_DIR)
    if not np.array_equal(arrays["sample_seed"], actions["sample_seed"]):
        raise ValueError("State samples and action samples are not aligned by seed order.")

    x_state, y, persistence, meta = build_xy(arrays, edge_vocab)
    x_action = build_action_features(actions)
    x_state_action = np.concatenate([x_state, x_action], axis=1).astype(np.float32)

    train_idx, val_idx, test_idx = split_by_seed(arrays["sample_seed"])
    y_val, y_test = y[val_idx], y[test_idx]
    persistence_val, persistence_test = persistence[val_idx], persistence[test_idx]

    alphas = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
    state_result = train_ridge_residual(x_state, y, persistence, train_idx, val_idx, test_idx, alphas)
    action_result = train_ridge_residual(x_state_action, y, persistence, train_idx, val_idx, test_idx, alphas)

    rows = []
    rows.extend(evaluate_rows("persistence", y_val, y_test, persistence_val, persistence_test, meta["link_target_dim"]))
    rows.extend(
        evaluate_rows(
            "state_only_ridge",
            y_val,
            y_test,
            state_result["val_pred"],
            state_result["test_pred"],
            meta["link_target_dim"],
        )
    )
    rows.extend(
        evaluate_rows(
            "state_action_ridge",
            y_val,
            y_test,
            action_result["val_pred"],
            action_result["test_pred"],
            meta["link_target_dim"],
        )
    )
    metrics_df = pd.DataFrame(rows)
    metrics_path = OUTPUT_DIR / "action_conditioned_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    predictions = {
        "persistence": persistence_test,
        "state": state_result["test_pred"],
        "state+action": action_result["test_pred"],
    }
    bar_plot = plot_rmse(OUTPUT_DIR, metrics_df)
    link_plot = plot_link_predictions(OUTPUT_DIR, y_test, predictions, meta)
    task_plot = plot_task_predictions(OUTPUT_DIR, y_test, predictions, meta)

    summary = {
        "dataset_dir": str(DATASET_DIR),
        "action_dir": str(ACTION_DIR),
        "output_dir": str(OUTPUT_DIR),
        "split": {
            "train_seeds": [0, 1, 2],
            "val_seed": 3,
            "test_seed": 4,
            "train_samples": int(len(train_idx)),
            "val_samples": int(len(val_idx)),
            "test_samples": int(len(test_idx)),
        },
        "feature_dims": {
            "state_features": int(x_state.shape[1]),
            "action_features": int(x_action.shape[1]),
            "state_action_features": int(x_state_action.shape[1]),
            "targets": int(y.shape[1]),
        },
        "state_only_ridge": {
            "selected_alpha": state_result["alpha"],
            "val_mse_scaled": state_result["val_mse_scaled"],
        },
        "state_action_ridge": {
            "selected_alpha": action_result["alpha"],
            "val_mse_scaled": action_result["val_mse_scaled"],
        },
        "metrics": rows,
        "outputs": {
            "metrics_csv": str(metrics_path),
            "bar_plot": str(bar_plot),
            "link_plot": str(link_plot),
            "task_plot": str(task_plot),
        },
    }
    summary_path = OUTPUT_DIR / "action_conditioned_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = write_report(OUTPUT_DIR, summary, metrics_df)
    summary["outputs"]["report_md"] = str(report_path)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
