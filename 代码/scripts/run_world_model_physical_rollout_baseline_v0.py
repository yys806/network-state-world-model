import json

import numpy as np
import pandas as pd

from run_world_model_v0 import FIGURE_DIR, ROOT, load_dataset, regression_metrics, split_by_seed
from run_world_model_v4_dual_graph_rollout import display_path


OUTPUT_DIR = ROOT / "reports" / "world_model_physical_rollout_baseline_v0"
EDGE_VOCAB_PATH = ROOT / "datasets" / "world_model_dataset_v0" / "edge_vocab.csv"


def persistence_nodes(x_node, horizon):
    last = x_node[:, -1:, :, :]
    return np.repeat(last, horizon, axis=1).astype(np.float32)


def linear_extrapolate_nodes(x_node, horizon):
    x_node = np.asarray(x_node, dtype=np.float32)
    last = x_node[:, -1, :, :].copy()
    if x_node.shape[1] >= 2:
        delta = x_node[:, -1, :, :] - x_node[:, -2, :, :]
    else:
        delta = np.zeros_like(last)
    pred = []
    for step in range(1, horizon + 1):
        current = last + delta * float(step)
        current[..., 4:] = last[..., 4:]
        pred.append(current)
    return np.stack(pred, axis=1).astype(np.float32)


def edge_distance_from_nodes(nodes, edge_src_idx, edge_dst_idx):
    src = np.asarray(edge_src_idx, dtype=np.int64).clip(min=0)
    dst = np.asarray(edge_dst_idx, dtype=np.int64).clip(min=0)
    src_xyz = nodes[:, :, src, :3]
    dst_xyz = nodes[:, :, dst, :3]
    return np.linalg.norm(dst_xyz - src_xyz, axis=-1).astype(np.float32)


def metric_row(split, model, y_node, pred_node, edge_src_idx, edge_dst_idx, edge_vocab):
    position_true = y_node[..., :3]
    position_pred = pred_node[..., :3]
    speed_true = y_node[..., 3:4]
    speed_pred = pred_node[..., 3:4]
    true_dist = edge_distance_from_nodes(y_node, edge_src_idx, edge_dst_idx)
    pred_dist = edge_distance_from_nodes(pred_node, edge_src_idx, edge_dst_idx)
    pos = regression_metrics(position_true, position_pred)
    speed = regression_metrics(speed_true, speed_pred)
    dist = regression_metrics(true_dist, pred_dist)
    row = {
        "split": split,
        "model": model,
        "position_mae": pos["mae"],
        "position_rmse": pos["rmse"],
        "speed_mae": speed["mae"],
        "speed_rmse": speed["rmse"],
        "edge_distance_mae": dist["mae"],
        "edge_distance_rmse": dist["rmse"],
    }
    for link_type, part in edge_vocab.groupby("link_type"):
        edge_idx = part["edge_index"].to_numpy(dtype=int)
        type_metric = regression_metrics(true_dist[:, :, edge_idx], pred_dist[:, :, edge_idx])
        row[f"{link_type}_distance_rmse"] = type_metric["rmse"]
    return row


def plot_physical_metrics(metrics_df):
    import matplotlib.pyplot as plt

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / "world_model_physical_rollout_baseline_v0.png"
    test = metrics_df[metrics_df["split"] == "test_seed_4"].set_index("model")
    cols = ["position_rmse", "speed_rmse", "edge_distance_rmse"]
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.8))
    colors = ["#2563eb", "#16a34a"]
    for ax, col in zip(axes, cols):
        test[col].plot(kind="bar", ax=ax, color=colors)
        ax.set_title(col.replace("_", " "))
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Physical graph rollout baseline")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def write_report(summary, metrics_df):
    test = metrics_df[metrics_df["split"] == "test_seed_4"].copy()
    best = test.sort_values("edge_distance_rmse").iloc[0]
    lines = [
        "# Physical graph rollout baseline v0",
        "",
        "## Goal",
        "",
        "This report adds the first explicit physical-graph future-rollout metrics. It uses CPU-only persistence and linear extrapolation baselines over future node states, then derives candidate-edge distances from predicted endpoint positions.",
        "",
        "## Metrics",
        "",
        metrics_df.to_markdown(index=False),
        "",
        "## Main Result",
        "",
        f"- Best test edge-distance RMSE baseline: `{best['model']}` with `{best['edge_distance_rmse']:.6f}`.",
        "- These baselines provide a non-neural reference for the next dual-graph model stage, where the physical branch should predict future mobility and coverage rather than only reuse historical physical-edge features.",
        "",
        "## Outputs",
        "",
    ]
    for key, value in summary["outputs"].items():
        lines.append(f"- {key}: `{value}`")
    path = OUTPUT_DIR / "world_model_physical_rollout_baseline_v0_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    arrays = load_dataset()
    edge_vocab = pd.read_csv(EDGE_VOCAB_PATH)
    _, val_idx, test_idx = split_by_seed(arrays["sample_seed"])
    horizon = arrays["y_node"].shape[1]
    models = {
        "physical_persistence": persistence_nodes,
        "physical_linear_extrapolation": linear_extrapolate_nodes,
    }
    rows = []
    for split, idx in [("val_seed_3", val_idx), ("test_seed_4", test_idx)]:
        x_node = arrays["x_node"][idx]
        y_node = arrays["y_node"][idx]
        for name, fn in models.items():
            pred = fn(x_node, horizon)
            rows.append(
                metric_row(
                    split,
                    name,
                    y_node,
                    pred,
                    arrays["edge_src_idx"],
                    arrays["edge_dst_idx"],
                    edge_vocab,
                )
            )
    metrics_df = pd.DataFrame(rows)
    metrics_path = OUTPUT_DIR / "world_model_physical_rollout_baseline_v0_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    plot_path = plot_physical_metrics(metrics_df)
    summary = {
        "output_dir": display_path(OUTPUT_DIR),
        "horizon": int(horizon),
        "models": list(models.keys()),
        "outputs": {
            "metrics_csv": display_path(metrics_path),
            "summary_plot": display_path(plot_path),
        },
    }
    report_path = write_report(summary, metrics_df)
    summary["outputs"]["report_md"] = display_path(report_path)
    summary_path = OUTPUT_DIR / "world_model_physical_rollout_baseline_v0_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
