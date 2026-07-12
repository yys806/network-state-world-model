import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXAMPLE_DIR = Path(__file__).resolve().parent
DEFAULT_RUN_DIR = EXAMPLE_DIR / "outputs" / "demo_run_20260507_190930"
DEFAULT_DATASET_DIR = EXAMPLE_DIR / "outputs" / "dataset_v0_from_demo_run_20260507_190930"


def parse_args():
    parser = argparse.ArgumentParser(description="Create presentation-friendly visuals for dataset_v0.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR, help="AirFogSim raw output directory.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="dataset_v0 output directory.",
    )
    parser.add_argument("--sample-id", type=int, default=60, help="Sample used in the history/label diagram.")
    return parser.parse_args()


def setup_style():
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 220,
            "font.size": 11,
            "axes.titlesize": 15,
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
        }
    )


def read_data(run_dir, dataset_dir):
    node = pd.read_csv(run_dir / "node_states.csv")
    link = pd.read_csv(run_dir / "link_states.csv")
    task = pd.read_csv(run_dir / "task_states.csv")
    sample_index = pd.read_csv(dataset_dir / "sample_index.csv")
    node_vocab = pd.read_csv(dataset_dir / "node_vocab.csv")
    edge_vocab = pd.read_csv(dataset_dir / "edge_vocab.csv")
    task_stats = pd.read_csv(dataset_dir / "task_time_stats.csv")
    with np.load(dataset_dir / "dataset_v0_samples.npz", allow_pickle=True) as npz:
        arrays = {key: npz[key] for key in npz.files}
    return node, link, task, sample_index, node_vocab, edge_vocab, task_stats, arrays


def ensure_visual_dir(dataset_dir):
    visual_dir = dataset_dir / "visuals"
    visual_dir.mkdir(parents=True, exist_ok=True)
    return visual_dir


def save_current(path):
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return str(path)


def plot_raw_to_dataset_pipeline(visual_dir, summary):
    fig, ax = plt.subplots(figsize=(13.2, 4.6))
    ax.axis("off")

    boxes = [
        ("AirFogSim raw logs", "node_states.csv\nlink_states.csv\ntask_states.csv"),
        ("Field alignment", "node vocabulary\nedge vocabulary\ntime index"),
        ("Graph-time tensors", "node tensor: T x N x F\nlink tensor: T x E x F\ntask tensor: T x F"),
        ("Supervised samples", "history window H=8\nfuture horizon K=3\n190 samples"),
        ("Next modeling step", "baseline prediction\njoint world model\ntransfer evaluation"),
    ]
    xs = np.linspace(0.08, 0.92, len(boxes))
    colors = ["#eef6ff", "#f5f1ff", "#effaf2", "#fff7e8", "#f0f5f5"]
    for idx, ((title, body), x, color) in enumerate(zip(boxes, xs, colors)):
        ax.text(
            x,
            0.58,
            f"{title}\n\n{body}",
            ha="center",
            va="center",
            linespacing=1.35,
            bbox=dict(boxstyle="round,pad=0.55,rounding_size=0.08", fc=color, ec="#475569", lw=1.2),
            transform=ax.transAxes,
        )
        if idx < len(boxes) - 1:
            ax.annotate(
                "",
                xy=(xs[idx + 1] - 0.085, 0.58),
                xytext=(x + 0.085, 0.58),
                xycoords=ax.transAxes,
                arrowprops=dict(arrowstyle="-|>", lw=1.6, color="#334155"),
            )

    caption = (
        f"Current run: {summary['num_nodes']} nodes, {summary['num_edges']} edges, "
        f"{summary['num_times']} time steps, {summary['num_samples']} samples"
    )
    ax.text(0.5, 0.12, caption, ha="center", va="center", color="#334155", transform=ax.transAxes)
    return save_current(visual_dir / "01_raw_to_dataset_pipeline.png")


def select_snapshot_time(link):
    active = link.groupby("time")["rate_sum"].apply(lambda s: (s > 0).sum())
    if active.max() > 0:
        return float(active.idxmax())
    return float(link["time"].median())


def plot_scene_snapshot(visual_dir, node, link):
    t = select_snapshot_time(link)
    node_t = node[np.isclose(node["time"], t)].copy()
    link_t = link[(np.isclose(link["time"], t)) & (link["rate_sum"] > 0)].copy()
    pos = {row.node_id: (row.x, row.y) for row in node_t.itertuples(index=False)}

    fig, ax = plt.subplots(figsize=(8.8, 7.4))
    type_style = {
        "vehicle": ("#64748b", "o", 32, "vehicle"),
        "uav": ("#f97316", "^", 135, "UAV"),
        "rsu": ("#2563eb", "s", 110, "RSU"),
        "cloud": ("#111827", "P", 145, "cloud"),
    }
    edge_style = {
        "V2U": ("#f97316", 1.5, 0.50),
        "V2I": ("#2563eb", 1.0, 0.28),
        "U2I": ("#16a34a", 1.8, 0.55),
    }

    for link_row in link_t.itertuples(index=False):
        if link_row.tx_id not in pos or link_row.rx_id not in pos:
            continue
        color, width, alpha = edge_style.get(link_row.link_type, ("#94a3b8", 1.0, 0.25))
        x1, y1 = pos[link_row.tx_id]
        x2, y2 = pos[link_row.rx_id]
        ax.plot([x1, x2], [y1, y2], color=color, lw=width, alpha=alpha, zorder=1)

    for node_type, (color, marker, size, label) in type_style.items():
        part = node_t[node_t["node_type"] == node_type]
        if len(part) == 0:
            continue
        ax.scatter(part["x"], part["y"], c=color, marker=marker, s=size, label=label, edgecolor="white", lw=0.8, zorder=3)

    for row in node_t[node_t["node_type"].isin(["uav", "rsu", "cloud"])].itertuples(index=False):
        ax.text(row.x + 8, row.y + 8, row.node_id, fontsize=9, color="#0f172a")

    ax.set_title(f"Scenario snapshot at t={t:.1f}s: physical entities and active communication links")
    ax.set_xlabel("x position")
    ax.set_ylabel("y position")
    ax.legend(loc="upper right", ncol=2)
    ax.set_aspect("equal", adjustable="box")
    return save_current(visual_dir / "02_scene_snapshot_physical_comm.png")


def plot_dual_graph_snapshot(visual_dir, node, link):
    t = select_snapshot_time(link)
    node_t = node[np.isclose(node["time"], t)].copy()
    link_t = link[np.isclose(link["time"], t)].copy()
    pos = {row.node_id: (row.x, row.y) for row in node_t.itertuples(index=False)}

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.2), sharex=True, sharey=True)
    for ax, title in zip(axes, ["Physical graph: distance-neighborhood structure", "Communication graph: active/rate-bearing links"]):
        ax.set_title(title)
        ax.set_xlabel("x position")
        ax.set_ylabel("y position")
        ax.set_aspect("equal", adjustable="box")

    moving = node_t[node_t["node_type"].isin(["vehicle", "uav"])]
    coords = moving[["x", "y"]].to_numpy()
    ids = moving["node_id"].to_list()
    max_edges = 110
    edges = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            dist = float(np.linalg.norm(coords[i] - coords[j]))
            if dist <= 280:
                edges.append((dist, ids[i], ids[j]))
    for _, src, dst in sorted(edges)[:max_edges]:
        x1, y1 = pos[src]
        x2, y2 = pos[dst]
        axes[0].plot([x1, x2], [y1, y2], color="#94a3b8", lw=0.8, alpha=0.28)

    active = link_t[link_t["rate_sum"] > 0]
    color_by_type = {"V2U": "#f97316", "V2I": "#2563eb", "U2I": "#16a34a"}
    for row in active.itertuples(index=False):
        if row.tx_id not in pos or row.rx_id not in pos:
            continue
        x1, y1 = pos[row.tx_id]
        x2, y2 = pos[row.rx_id]
        axes[1].plot([x1, x2], [y1, y2], color=color_by_type.get(row.link_type, "#64748b"), lw=1.5, alpha=0.55)

    type_style = {
        "vehicle": ("#64748b", "o", 24, "vehicle"),
        "uav": ("#f97316", "^", 105, "UAV"),
        "rsu": ("#2563eb", "s", 92, "RSU"),
        "cloud": ("#111827", "P", 120, "cloud"),
    }
    for ax in axes:
        for node_type, (color, marker, size, label) in type_style.items():
            part = node_t[node_t["node_type"] == node_type]
            ax.scatter(part["x"], part["y"], c=color, marker=marker, s=size, label=label, edgecolor="white", lw=0.7, zorder=3)
    axes[1].legend(loc="upper right", ncol=2)
    return save_current(visual_dir / "03_dual_graph_snapshot.png")


def plot_history_label_window(visual_dir, sample_index, sample_id):
    sample_id = int(np.clip(sample_id, 0, len(sample_index) - 1))
    row = sample_index.iloc[sample_id]
    fig, ax = plt.subplots(figsize=(11.8, 2.8))
    ax.set_title(f"Sample construction: history window H=8 predicts future horizon K=3 (sample {sample_id})")
    ax.set_yticks([])
    ax.set_xlabel("simulation time")

    hist_times = np.linspace(row["input_start_time"], row["input_end_time"], 8)
    label_times = np.linspace(row["label_start_time"], row["label_end_time"], 3)
    ax.scatter(hist_times, np.ones_like(hist_times), s=220, color="#2563eb", label="input history")
    ax.scatter(label_times, np.ones_like(label_times), s=260, color="#f97316", marker="s", label="future label")
    ax.axvline(row["input_end_time"], color="#334155", linestyle="--", lw=1.4)
    ax.text(row["input_end_time"], 1.18, "current t", ha="center", color="#334155")

    for time in hist_times:
        ax.text(time, 0.82, f"{time:.1f}", ha="center", fontsize=9)
    for time in label_times:
        ax.text(time, 0.82, f"{time:.1f}", ha="center", fontsize=9)

    ax.set_ylim(0.65, 1.35)
    ax.legend(loc="upper left", ncol=2)
    return save_current(visual_dir / "04_history_to_future_window.png")


def plot_tensor_shapes(visual_dir, summary):
    fig, ax = plt.subplots(figsize=(11.8, 5.2))
    ax.axis("off")
    rows = [
        ("Node state", "x_node", summary["x_node_shape"], "position / speed / acceleration / CPU / storage"),
        ("Link state", "x_link", summary["x_link_shape"], "distance / rate / CSI / active tasks / RB usage"),
        ("Task state", "x_task", summary["x_task_shape"], "task volume / CPU demand / lifecycle counts"),
        ("Future labels", "y_*", [summary["num_samples"], summary["horizon"], "..."], "same fields shifted to future K steps"),
    ]
    ax.text(0.02, 0.92, "dataset_v0 tensor organization", fontsize=18, fontweight="bold", transform=ax.transAxes)
    ax.text(0.02, 0.84, "Each sample keeps aligned physical, communication, and task observations.", color="#475569", transform=ax.transAxes)
    y = 0.68
    for name, symbol, shape, desc in rows:
        ax.text(
            0.05,
            y,
            name,
            ha="left",
            va="center",
            fontsize=13,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.35", fc="#f8fafc", ec="#cbd5e1"),
            transform=ax.transAxes,
        )
        ax.text(0.28, y, symbol, ha="left", va="center", fontsize=14, color="#1d4ed8", transform=ax.transAxes)
        ax.text(0.43, y, str(shape), ha="left", va="center", fontsize=13, color="#0f172a", transform=ax.transAxes)
        ax.text(0.67, y, desc, ha="left", va="center", fontsize=11, color="#475569", transform=ax.transAxes)
        y -= 0.15
    return save_current(visual_dir / "05_dataset_tensor_shapes.png")


def plot_prediction_targets_dashboard(visual_dir, link, task_stats):
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.0))
    rate_by_type = link.groupby(["time", "link_type"])["rate_sum"].mean().unstack(fill_value=0)
    rate_by_type.plot(ax=axes[0, 0], lw=2)
    axes[0, 0].set_title("Future target 1: link rate trend")
    axes[0, 0].set_xlabel("time")
    axes[0, 0].set_ylabel("mean rate_sum")

    rb_by_type = link.groupby(["time", "link_type"])["allocated_rb_count"].mean().unstack(fill_value=0)
    rb_by_type.plot(ax=axes[0, 1], lw=2)
    axes[0, 1].set_title("Resource observation: allocated RB count")
    axes[0, 1].set_xlabel("time")
    axes[0, 1].set_ylabel("mean allocated_rb_count")

    lifecycle_cols = ["num_to_offload", "num_computing", "num_returning", "num_finished"]
    task_stats.plot(x="time", y=lifecycle_cols, ax=axes[1, 0], lw=2)
    axes[1, 0].set_title("Future target 2: task lifecycle counts")
    axes[1, 0].set_xlabel("time")
    axes[1, 0].set_ylabel("task count")

    task_stats.plot(x="time", y=["total_task_size", "total_task_cpu"], ax=axes[1, 1], lw=2)
    axes[1, 1].set_title("Future target 3: aggregate task demand")
    axes[1, 1].set_xlabel("time")
    axes[1, 1].set_ylabel("aggregate demand")
    return save_current(visual_dir / "06_prediction_targets_dashboard.png")


def plot_task_flow_sankey_like(visual_dir, task):
    transitions = []
    for _, part in task.sort_values(["task_id", "time"]).groupby("task_id"):
        states = part["lifecycle_state"].dropna().to_list()
        for src, dst in zip(states[:-1], states[1:]):
            if src != dst:
                transitions.append((src, dst))
    counts = pd.Series(transitions).value_counts()
    states = ["to_offload", "computing", "returning", "finished"]
    y_pos = dict(zip(states, np.linspace(0.78, 0.22, len(states))))
    fig, ax = plt.subplots(figsize=(9.6, 5.6))
    ax.axis("off")
    ax.set_title("Task lifecycle transitions observed in the exported run")

    for state in states:
        ax.text(
            0.12,
            y_pos[state],
            state,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.45", fc="#f8fafc", ec="#94a3b8"),
            transform=ax.transAxes,
        )
        ax.text(
            0.86,
            y_pos[state],
            state,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.45", fc="#fff7ed", ec="#fdba74"),
            transform=ax.transAxes,
        )

    # Put a full label beside each curve so counts cannot be visually matched to
    # the wrong transition when multiple curves are close to each other.
    transition_styles = {
        ("to_offload", "computing"): dict(rad=0.22, color="#2563eb", label_y=0.695, label_x=0.51),
        ("to_offload", "finished"): dict(rad=-0.10, color="#7c3aed", label_y=0.545, label_x=0.52),
        ("computing", "finished"): dict(rad=0.16, color="#0891b2", label_y=0.405, label_x=0.50),
    }
    max_count = max(counts.max(), 1) if len(counts) else 1
    for (src, dst), count in counts.items():
        y1 = y_pos.get(src, 0.5)
        y2 = y_pos.get(dst, 0.5)
        width = 0.8 + 4.5 * count / max_count
        style = transition_styles.get((src, dst), dict(rad=0.12, color="#2563eb", label_y=(y1 + y2) / 2, label_x=0.5))
        ax.annotate(
            "",
            xy=(0.78, y2),
            xytext=(0.22, y1),
            xycoords=ax.transAxes,
            arrowprops=dict(
                arrowstyle="-|>",
                lw=width,
                color=style["color"],
                alpha=0.36,
                connectionstyle=f"arc3,rad={style['rad']}",
            ),
        )
        ax.text(
            style["label_x"],
            style["label_y"],
            f"{src} -> {dst}: {int(count)}",
            ha="center",
            va="center",
            fontsize=10,
            color="#0f172a",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=style["color"], alpha=0.92),
            transform=ax.transAxes,
        )

    ax.text(0.12, 0.92, "previous state", ha="center", color="#475569", transform=ax.transAxes)
    ax.text(0.86, 0.92, "next state", ha="center", color="#475569", transform=ax.transAxes)
    return save_current(visual_dir / "07_task_lifecycle_transitions.png")


def write_visual_report(visual_dir, generated):
    lines = [
        "# dataset_v0 可视化说明",
        "",
        "这些图片用于本周汇报，重点说明 AirFogSim 原始输出已经被组织成可建模的数据样本。",
        "",
    ]
    descriptions = {
        "01_raw_to_dataset_pipeline.png": "从原始 CSV 到联合建模样本的整体处理链路。",
        "02_scene_snapshot_physical_comm.png": "某一时刻的实体位置与活跃通信链路，可说明空地协同场景。",
        "03_dual_graph_snapshot.png": "同一场景下物理图和通信图的区别，突出联合建模不是只看一种关系。",
        "04_history_to_future_window.png": "历史窗口 H=8 到未来标签 K=3 的样本构造方式。",
        "05_dataset_tensor_shapes.png": "dataset_v0 的张量形状，可直接用于解释输入输出。",
        "06_prediction_targets_dashboard.png": "后续 baseline 可以预测的链路、任务、资源相关目标。",
        "07_task_lifecycle_transitions.png": "任务状态转移关系，用于解释任务流程和卸载链路。",
    }
    for path in generated:
        name = Path(path).name
        lines.append(f"- `{name}`：{descriptions.get(name, '补充可视化。')}")
    (visual_dir / "visual_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    setup_style()
    run_dir = args.run_dir.resolve()
    dataset_dir = args.dataset_dir.resolve()
    visual_dir = ensure_visual_dir(dataset_dir)

    node, link, task, sample_index, node_vocab, edge_vocab, task_stats, arrays = read_data(run_dir, dataset_dir)
    summary = json.loads((dataset_dir / "dataset_summary.json").read_text(encoding="utf-8"))
    summary["num_times"] = len(arrays["times"])

    generated = [
        plot_raw_to_dataset_pipeline(visual_dir, summary),
        plot_scene_snapshot(visual_dir, node, link),
        plot_dual_graph_snapshot(visual_dir, node, link),
        plot_history_label_window(visual_dir, sample_index, args.sample_id),
        plot_tensor_shapes(visual_dir, summary),
        plot_prediction_targets_dashboard(visual_dir, link, task_stats),
        plot_task_flow_sankey_like(visual_dir, task),
    ]
    write_visual_report(visual_dir, generated)
    print(json.dumps({"visual_dir": str(visual_dir), "generated": generated}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
