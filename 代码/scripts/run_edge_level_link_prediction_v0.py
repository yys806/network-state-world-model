import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from train_baseline_v0 import (
    fit_best_ridge,
    inverse_standardize,
    predict_ridge,
    standardize,
)


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "datasets" / "dataset_multiseed_v0"
ACTION_DIR = ROOT / "datasets" / "strict_action_v0"
OUTPUT_DIR = ROOT / "reports" / "edge_level_link_prediction_v0"
FIGURE_DIR = ROOT / "figures"

NODE_TYPES = ["cloud", "rsu", "uav", "vehicle"]
LINK_TYPES = ["U2I", "V2I", "V2U"]
ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]


def load_inputs():
    with np.load(DATASET_DIR / "dataset_multiseed_v0_samples.npz", allow_pickle=True) as data:
        arrays = {key: data[key] for key in data.files}
    with np.load(ACTION_DIR / "strict_action_v0_samples.npz", allow_pickle=True) as data:
        actions = {key: data[key] for key in data.files}
    if not np.array_equal(arrays["sample_seed"], actions["sample_seed"]):
        raise ValueError("State samples and strict action samples are not aligned.")
    node_vocab = pd.read_csv(DATASET_DIR / "node_vocab.csv")
    edge_vocab = pd.read_csv(DATASET_DIR / "edge_vocab.csv")
    return arrays, actions, node_vocab, edge_vocab


def split_by_seed(sample_seed, train_seeds=(0, 1, 2), val_seed=3, test_seed=4):
    sample_seed = np.asarray(sample_seed)
    return (
        np.where(np.isin(sample_seed, train_seeds))[0],
        np.where(sample_seed == val_seed)[0],
        np.where(sample_seed == test_seed)[0],
    )


def grouped_stats(tensor, labels, groups):
    labels = np.asarray(labels)
    parts = []
    for group in groups:
        mask = labels == group
        if mask.sum() == 0:
            b, h, _, f = tensor.shape
            parts.append(np.zeros((b, h, f * 3), dtype=np.float32))
            continue
        sub = tensor[:, :, mask, :]
        parts.append(np.concatenate([sub.mean(axis=2), sub.std(axis=2), sub.max(axis=2)], axis=-1))
    return np.concatenate(parts, axis=-1).reshape(len(tensor), -1).astype(np.float32)


def build_edge_samples(arrays, actions, node_vocab, edge_vocab):
    x_node = arrays["x_node"].astype(np.float32)
    x_link = arrays["x_link"].astype(np.float32)
    x_task = arrays["x_task"].astype(np.float32)
    y_link = arrays["y_link"].astype(np.float32)
    a_hist = actions["a_hist"].astype(np.float32)
    a_future = actions["a_future"].astype(np.float32)

    num_samples, history, num_edges, link_features = x_link.shape
    horizon = y_link.shape[1]
    rate_feature_idx = list(arrays["link_features"]).index("rate_sum")

    global_node = grouped_stats(x_node, node_vocab["node_type"].to_numpy(), NODE_TYPES)
    global_link = grouped_stats(x_link, edge_vocab["link_type"].to_numpy(), LINK_TYPES)
    global_task = x_task.reshape(num_samples, -1)
    global_state = np.concatenate([global_node, global_link, global_task], axis=1).astype(np.float32)
    global_action = np.concatenate([a_hist.reshape(num_samples, -1), a_future.reshape(num_samples, -1)], axis=1)

    edge_type = pd.get_dummies(edge_vocab["link_type"]).reindex(columns=LINK_TYPES, fill_value=0).to_numpy(dtype=np.float32)
    edge_identity = np.eye(num_edges, dtype=np.float32)
    edge_static = np.concatenate([edge_type, edge_identity], axis=1)

    repeated_global_state = np.repeat(global_state, num_edges, axis=0)
    repeated_global_action = np.repeat(global_action, num_edges, axis=0)
    repeated_edge_static = np.tile(edge_static, (num_samples, 1))
    edge_history = x_link.transpose(0, 2, 1, 3).reshape(num_samples * num_edges, history * link_features)

    x_state = np.concatenate([edge_history, repeated_edge_static, repeated_global_state], axis=1).astype(np.float32)
    x_state_action = np.concatenate([x_state, repeated_global_action], axis=1).astype(np.float32)
    y = y_link[..., rate_feature_idx].transpose(0, 2, 1).reshape(num_samples * num_edges, horizon).astype(np.float32)
    persistence = np.repeat(x_link[:, -1, :, rate_feature_idx], horizon, axis=1).reshape(num_samples, num_edges, horizon)
    persistence = persistence.reshape(num_samples * num_edges, horizon).astype(np.float32)

    sample_seed = np.repeat(arrays["sample_seed"], num_edges)
    edge_index = np.tile(np.arange(num_edges), num_samples)
    link_type = edge_vocab["link_type"].to_numpy()[edge_index]

    meta = {
        "num_original_samples": int(num_samples),
        "num_edge_samples": int(num_samples * num_edges),
        "history": int(history),
        "horizon": int(horizon),
        "num_edges": int(num_edges),
        "link_features": arrays["link_features"].tolist(),
        "rate_feature_idx": int(rate_feature_idx),
        "state_features": int(x_state.shape[1]),
        "state_action_features": int(x_state_action.shape[1]),
        "target_dim": int(y.shape[1]),
    }
    return x_state, x_state_action, y, persistence, sample_seed, edge_index, link_type, meta


def train_ridge_residual(x, y, persistence, train_idx, val_idx, test_idx):
    x_parts, x_mean, x_std = standardize(x[train_idx], x[train_idx], x[val_idx], x[test_idx])
    x_train, x_val, x_test = [part.astype(np.float32) for part in x_parts]
    y_res = y - persistence
    y_parts, y_mean, y_std = standardize(y_res[train_idx], y_res[train_idx], y_res[val_idx], y_res[test_idx])
    y_train, y_val, _ = [part.astype(np.float32) for part in y_parts]
    alpha, weights, val_mse = fit_best_ridge(x_train, y_train, x_val, y_val, ALPHAS)
    val_pred = persistence[val_idx] + inverse_standardize(predict_ridge(x_val, weights), y_mean, y_std)
    test_pred = persistence[test_idx] + inverse_standardize(predict_ridge(x_test, weights), y_mean, y_std)
    val_pred = np.clip(val_pred, 0.0, None)
    test_pred = np.clip(test_pred, 0.0, None)
    return {
        "alpha": float(alpha),
        "val_mse_scaled": float(val_mse),
        "val_pred": val_pred.astype(np.float32),
        "test_pred": test_pred.astype(np.float32),
        "x_mean": x_mean,
        "x_std": x_std,
    }


def metrics(y_true, y_pred):
    err = y_pred - y_true
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "bias": float(np.mean(err)),
    }


def evaluate_model(split, model, y_true, y_pred, link_type):
    rows = [{"split": split, "model": model, "link_type": "all", **metrics(y_true, y_pred)}]
    for lt in LINK_TYPES:
        mask = link_type == lt
        if mask.sum() == 0:
            continue
        rows.append({"split": split, "model": model, "link_type": lt, **metrics(y_true[mask], y_pred[mask])})
    active_mask = np.any(y_true > 1e-6, axis=1)
    inactive_mask = ~active_mask
    if active_mask.sum() > 0:
        rows.append({"split": split, "model": model, "link_type": "active_edges", **metrics(y_true[active_mask], y_pred[active_mask])})
    if inactive_mask.sum() > 0:
        rows.append(
            {"split": split, "model": model, "link_type": "inactive_edges", **metrics(y_true[inactive_mask], y_pred[inactive_mask])}
        )
    return rows


def plot_overall_rmse(metrics_df):
    test = metrics_df[(metrics_df["split"] == "test_seed_4") & (metrics_df["link_type"] == "all")]
    order = ["persistence", "edge_state_ridge", "edge_state_action_ridge"]
    test = test.set_index("model").loc[order].reset_index()
    path = FIGURE_DIR / "edge_level_link_rmse_bar.png"
    plt.figure(figsize=(7.2, 4.2))
    bars = plt.bar(test["model"], test["rmse"], color=["#8A94A6", "#1F77B4", "#D62728"])
    plt.ylabel("RMSE of edge-level rate_sum")
    plt.title("Edge-level link prediction on held-out seed 4")
    plt.xticks(rotation=12, ha="right")
    for bar in bars:
        value = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()
    return path


def plot_by_link_type(metrics_df):
    test = metrics_df[(metrics_df["split"] == "test_seed_4") & (metrics_df["link_type"].isin(LINK_TYPES))]
    pivot = test.pivot(index="link_type", columns="model", values="rmse").reindex(LINK_TYPES)
    path = FIGURE_DIR / "edge_level_link_rmse_by_type.png"
    ax = pivot[["persistence", "edge_state_ridge", "edge_state_action_ridge"]].plot(
        kind="bar",
        figsize=(7.5, 4.3),
        color=["#8A94A6", "#1F77B4", "#D62728"],
    )
    ax.set_ylabel("RMSE of edge-level rate_sum")
    ax.set_title("Edge-level link prediction by link type")
    ax.tick_params(axis="x", rotation=0)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()
    return path


def plot_prediction_scatter(y_true, predictions):
    path = FIGURE_DIR / "edge_level_link_prediction_scatter.png"
    flat_true = y_true.reshape(-1)
    rng = np.random.default_rng(42)
    if len(flat_true) > 4000:
        idx = rng.choice(len(flat_true), size=4000, replace=False)
    else:
        idx = np.arange(len(flat_true))
    max_value = float(max(np.percentile(flat_true, 99.9), *(np.percentile(pred.reshape(-1), 99.9) for pred in predictions.values()), 1.0))
    plt.figure(figsize=(8, 4))
    for i, (name, pred) in enumerate(predictions.items(), start=1):
        plt.subplot(1, 2, i)
        flat_pred = pred.reshape(-1)
        plt.scatter(flat_true[idx], flat_pred[idx], s=4, alpha=0.25, color="#1F77B4" if i == 1 else "#D62728")
        plt.plot([0, max_value], [0, max_value], color="#222222", linewidth=1)
        plt.xlim(0, max_value)
        plt.ylim(0, max_value)
        plt.xlabel("true rate_sum")
        plt.ylabel("predicted rate_sum")
        plt.title(name)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()
    return path


def write_report(summary, metrics_df):
    test_all = metrics_df[(metrics_df["split"] == "test_seed_4") & (metrics_df["link_type"] == "all")].set_index("model")
    p_rmse = test_all.loc["persistence", "rmse"]
    s_rmse = test_all.loc["edge_state_ridge", "rmse"]
    a_rmse = test_all.loc["edge_state_action_ridge", "rmse"]
    action_delta = s_rmse - a_rmse
    persistence_gain = p_rmse - a_rmse
    active = metrics_df[(metrics_df["split"] == "test_seed_4") & (metrics_df["link_type"] == "active_edges")].set_index("model")
    inactive = metrics_df[(metrics_df["split"] == "test_seed_4") & (metrics_df["link_type"] == "inactive_edges")].set_index("model")
    lines = [
        "# Edge-level link prediction report v0",
        "",
        "## Goal",
        "",
        "This experiment checks whether the current link-side bottleneck is caused by aggregating 188 candidate links into a few type-level statistics. Instead of predicting link-type averages, this version turns every edge into a training item and predicts the future `rate_sum` of each edge.",
        "",
        "## Dataset and split",
        "",
        f"- Original samples: `{summary['meta']['num_original_samples']}`",
        f"- Candidate edges: `{summary['meta']['num_edges']}`",
        f"- Edge-level samples: `{summary['meta']['num_edge_samples']}`",
        f"- History window H: `{summary['meta']['history']}`",
        f"- Prediction horizon K: `{summary['meta']['horizon']}`",
        "- Train seeds: `0, 1, 2`; validation seed: `3`; test seed: `4`.",
        "",
        "## Compared models",
        "",
        "- `persistence`: repeats the last observed rate of the same edge.",
        "- `edge_state_ridge`: predicts a residual from edge history, edge identity/type, and global node/link/task context.",
        "- `edge_state_action_ridge`: adds strict scheduler actions to the edge-state features.",
        "",
        "## Metrics",
        "",
        metrics_df.to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        f"- On held-out seed 4, `edge_state_action_ridge` changes RMSE by `{action_delta:.3f}` compared with `edge_state_ridge`.",
        f"- Compared with persistence, `edge_state_action_ridge` lowers all-edge RMSE by `{persistence_gain:.3f}`, but its MAE is higher because it gives positive estimates to many inactive or near-zero links.",
        f"- On active edges, persistence RMSE is `{active.loc['persistence', 'rmse']:.3f}` and edge-state-action RMSE is `{active.loc['edge_state_action_ridge', 'rmse']:.3f}`.",
        f"- On inactive edges, persistence RMSE is `{inactive.loc['persistence', 'rmse']:.3f}` and edge-state-action RMSE is `{inactive.loc['edge_state_action_ridge', 'rmse']:.3f}`.",
        "- This result is useful because it separates two link-side problems: rate regression on active links and activity/zero-link filtering on inactive links.",
        "- Short-horizon persistence remains strong because the simulator step is short and many edge rates change smoothly over only three future steps.",
        "- The next reasonable step is not a larger MLP on compressed features, but an edge-level graph model that keeps sender, receiver, distance, RB allocation, and interference context for each edge.",
        "",
        "## Outputs",
        "",
    ]
    for key, value in summary["outputs"].items():
        lines.append(f"- {key}: `{value}`")
    path = OUTPUT_DIR / "edge_level_link_prediction_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    arrays, actions, node_vocab, edge_vocab = load_inputs()
    x_state, x_state_action, y, persistence, sample_seed, edge_index, link_type, meta = build_edge_samples(
        arrays, actions, node_vocab, edge_vocab
    )
    train_idx, val_idx, test_idx = split_by_seed(sample_seed)
    val_link_type = link_type[val_idx]
    test_link_type = link_type[test_idx]

    state_result = train_ridge_residual(x_state, y, persistence, train_idx, val_idx, test_idx)
    action_result = train_ridge_residual(x_state_action, y, persistence, train_idx, val_idx, test_idx)

    rows = []
    rows.extend(evaluate_model("val_seed_3", "persistence", y[val_idx], persistence[val_idx], val_link_type))
    rows.extend(evaluate_model("test_seed_4", "persistence", y[test_idx], persistence[test_idx], test_link_type))
    rows.extend(evaluate_model("val_seed_3", "edge_state_ridge", y[val_idx], state_result["val_pred"], val_link_type))
    rows.extend(evaluate_model("test_seed_4", "edge_state_ridge", y[test_idx], state_result["test_pred"], test_link_type))
    rows.extend(
        evaluate_model("val_seed_3", "edge_state_action_ridge", y[val_idx], action_result["val_pred"], val_link_type)
    )
    rows.extend(
        evaluate_model("test_seed_4", "edge_state_action_ridge", y[test_idx], action_result["test_pred"], test_link_type)
    )
    metrics_df = pd.DataFrame(rows)
    metrics_path = OUTPUT_DIR / "edge_level_link_prediction_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    rmse_plot = plot_overall_rmse(metrics_df)
    type_plot = plot_by_link_type(metrics_df)
    scatter_plot = plot_prediction_scatter(
        y[test_idx],
        {
            "edge_state_ridge": state_result["test_pred"],
            "edge_state_action_ridge": action_result["test_pred"],
        },
    )

    summary = {
        "dataset_dir": str(DATASET_DIR),
        "action_dir": str(ACTION_DIR),
        "output_dir": str(OUTPUT_DIR),
        "meta": meta,
        "split": {
            "train_seeds": [0, 1, 2],
            "val_seed": 3,
            "test_seed": 4,
            "train_edge_samples": int(len(train_idx)),
            "val_edge_samples": int(len(val_idx)),
            "test_edge_samples": int(len(test_idx)),
        },
        "state_ridge": {
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
            "overall_rmse_plot": str(rmse_plot),
            "rmse_by_type_plot": str(type_plot),
            "prediction_scatter": str(scatter_plot),
        },
    }
    summary_path = OUTPUT_DIR / "edge_level_link_prediction_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = write_report(summary, metrics_df)
    summary["outputs"]["report_md"] = str(report_path)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
