import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


EXAMPLE_DIR = Path(__file__).resolve().parent
DEFAULT_RAW_ROOT = EXAMPLE_DIR / "outputs" / "multiseed_raw_v0"
DEFAULT_OUTPUT_DIR = EXAMPLE_DIR / "outputs" / "dataset_multiseed_v0"

NODE_FEATURES = ["x", "y", "z", "speed", "acceleration", "cpu", "storage"]
LINK_FEATURES = ["distance", "rate_sum", "csi_mean", "active_task_count", "allocated_rb_count"]
TASK_LIFECYCLES = ["to_offload", "computing", "returning", "finished"]


def parse_args():
    parser = argparse.ArgumentParser(description="Build dataset_multiseed_v0 from multi-seed AirFogSim CSV logs.")
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--history", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def read_all_runs(raw_root):
    run_dirs = sorted(path for path in raw_root.glob("seed_*") if path.is_dir())
    if not run_dirs:
        raise FileNotFoundError(f"No seed_* run directories found under {raw_root}")
    nodes = []
    links = []
    tasks = []
    for run_dir in run_dirs:
        seed = int(run_dir.name.split("_")[-1])
        node = pd.read_csv(run_dir / "node_states.csv")
        link = pd.read_csv(run_dir / "link_states.csv")
        task = pd.read_csv(run_dir / "task_states.csv")
        for df in [node, link, task]:
            if "seed" not in df.columns:
                df["seed"] = seed
            df["time"] = df["time"].round(3)
        nodes.append(node)
        links.append(link)
        tasks.append(task)
    return pd.concat(nodes, ignore_index=True), pd.concat(links, ignore_index=True), pd.concat(tasks, ignore_index=True)


def build_vocab(node, link):
    node_vocab = (
        node[["node_id", "node_type"]]
        .drop_duplicates()
        .sort_values(["node_type", "node_id"])
        .reset_index(drop=True)
    )
    node_vocab["node_index"] = np.arange(len(node_vocab))

    edge_vocab = (
        link[["tx_id", "rx_id", "link_type"]]
        .drop_duplicates()
        .sort_values(["link_type", "tx_id", "rx_id"])
        .reset_index(drop=True)
    )
    edge_vocab["edge_index"] = np.arange(len(edge_vocab))
    return node_vocab, edge_vocab


def build_node_tensor(node, times, node_vocab):
    node_index = dict(zip(node_vocab["node_id"], node_vocab["node_index"]))
    tensor = np.zeros((len(times), len(node_vocab), len(NODE_FEATURES)), dtype=np.float32)
    mask = np.zeros((len(times), len(node_vocab)), dtype=bool)
    time_index = {round(float(time), 3): idx for idx, time in enumerate(times)}
    filled = node.copy()
    filled[NODE_FEATURES] = filled[NODE_FEATURES].fillna(0.0)
    for row in filled.itertuples(index=False):
        ti = time_index[round(float(row.time), 3)]
        ni = node_index[row.node_id]
        tensor[ti, ni, :] = [getattr(row, feature) for feature in NODE_FEATURES]
        mask[ti, ni] = True
    return tensor, mask


def build_link_tensor(link, times, edge_vocab):
    edge_key_to_index = {
        (row.tx_id, row.rx_id, row.link_type): row.edge_index
        for row in edge_vocab.itertuples(index=False)
    }
    tensor = np.zeros((len(times), len(edge_vocab), len(LINK_FEATURES)), dtype=np.float32)
    mask = np.zeros((len(times), len(edge_vocab)), dtype=bool)
    time_index = {round(float(time), 3): idx for idx, time in enumerate(times)}
    filled = link.copy()
    filled[LINK_FEATURES] = filled[LINK_FEATURES].fillna(0.0)
    for row in filled.itertuples(index=False):
        ti = time_index[round(float(row.time), 3)]
        ei = edge_key_to_index[(row.tx_id, row.rx_id, row.link_type)]
        tensor[ti, ei, :] = [getattr(row, feature) for feature in LINK_FEATURES]
        mask[ti, ei] = True
    return tensor, mask


def build_task_tensor(task, times):
    columns = [
        "num_tasks",
        "total_task_size",
        "total_task_cpu",
        "mean_deadline",
        "mean_priority",
        *[f"num_{state}" for state in TASK_LIFECYCLES],
    ]
    rows = []
    for time in times:
        part = task[task["time"] == time]
        row = {
            "time": time,
            "num_tasks": len(part),
            "total_task_size": float(part["task_size"].sum()) if len(part) else 0.0,
            "total_task_cpu": float(part["task_cpu"].sum()) if len(part) else 0.0,
            "mean_deadline": float(part["deadline"].mean()) if len(part) else 0.0,
            "mean_priority": float(part["priority"].mean()) if len(part) else 0.0,
        }
        counts = part["lifecycle_state"].value_counts()
        for state in TASK_LIFECYCLES:
            row[f"num_{state}"] = int(counts.get(state, 0))
        rows.append(row)
    task_stats = pd.DataFrame(rows)
    return task_stats[columns].to_numpy(dtype=np.float32), task_stats


def build_sample_index_for_seed(seed, times, history, horizon, sample_offset):
    rows = []
    for end_idx in range(history - 1, len(times) - horizon):
        rows.append(
            {
                "sample_id": sample_offset + len(rows),
                "seed": seed,
                "input_start_idx": end_idx - history + 1,
                "input_end_idx": end_idx,
                "label_start_idx": end_idx + 1,
                "label_end_idx": end_idx + horizon,
                "input_start_time": times[end_idx - history + 1],
                "input_end_time": times[end_idx],
                "label_start_time": times[end_idx + 1],
                "label_end_time": times[end_idx + horizon],
            }
        )
    return pd.DataFrame(rows)


def slice_samples(tensor, sample_index, start_col, end_col):
    samples = []
    for row in sample_index.itertuples(index=False):
        start = getattr(row, start_col)
        end = getattr(row, end_col)
        samples.append(tensor[start : end + 1])
    return np.stack(samples, axis=0)


def plot_reports(output_dir, sample_index, node, link, task):
    if plt is None:
        return []
    generated = []
    sample_counts = sample_index.groupby("seed")["sample_id"].count()
    sample_counts.plot(kind="bar", figsize=(7, 4), color="#2563eb")
    plt.xlabel("seed")
    plt.ylabel("sample count")
    plt.title("dataset_multiseed_v0 samples per seed")
    plt.tight_layout()
    path = output_dir / "samples_per_seed.png"
    plt.savefig(path, dpi=200)
    plt.close()
    generated.append(str(path))

    final_tasks = task.groupby("seed")["task_id"].nunique()
    final_tasks.plot(kind="bar", figsize=(7, 4), color="#16a34a")
    plt.xlabel("seed")
    plt.ylabel("unique task count")
    plt.title("Unique tasks per seed")
    plt.tight_layout()
    path = output_dir / "tasks_per_seed.png"
    plt.savefig(path, dpi=200)
    plt.close()
    generated.append(str(path))

    rate = link.groupby(["seed", "link_type"])["rate_sum"].mean().unstack(fill_value=0)
    rate.plot(kind="bar", figsize=(8, 4))
    plt.xlabel("seed")
    plt.ylabel("mean rate_sum")
    plt.title("Mean link rate by seed")
    plt.tight_layout()
    path = output_dir / "rate_by_seed.png"
    plt.savefig(path, dpi=200)
    plt.close()
    generated.append(str(path))
    return generated


def write_report(output_dir, summary):
    lines = [
        "# dataset_multiseed_v0 数据集说明",
        "",
        "## 构造目的",
        "",
        "这一版数据集把多个 AirFogSim 随机种子的仿真日志整理成统一的历史窗口-未来标签样本，用于后续验证模型是否能跨随机轨迹泛化。",
        "",
        "## 核心设置",
        "",
        f"- seed 数量：{summary['num_seeds']}",
        f"- seeds：{summary['seeds']}",
        f"- 历史窗口 H：{summary['history']}",
        f"- 预测步长 K：{summary['horizon']}",
        f"- 总样本数：{summary['num_samples']}",
        f"- 节点 vocab 大小：{summary['num_nodes']}",
        f"- 边 vocab 大小：{summary['num_edges']}",
        "",
        "## 张量形状",
        "",
        f"- `x_node`: `{summary['x_node_shape']}`",
        f"- `x_link`: `{summary['x_link_shape']}`",
        f"- `x_task`: `{summary['x_task_shape']}`",
        f"- `y_node`: `{summary['y_node_shape']}`",
        f"- `y_link`: `{summary['y_link_shape']}`",
        f"- `y_task`: `{summary['y_task_shape']}`",
        "",
        "## 每个 seed 的样本数",
        "",
    ]
    for seed, count in summary["samples_per_seed"].items():
        lines.append(f"- seed {seed}: {count}")
    lines.extend(
        [
            "",
            "## 当前结论",
            "",
            "这一步把前面的多 seed 随机性分析推进成了可训练数据集。后续可以按 seed 做训练/测试划分，例如用 seed 0-3 训练、seed 4 测试，从而验证模型是否真正学到跨轨迹规律，而不是只记住单条轨迹。",
        ]
    )
    (output_dir / "dataset_multiseed_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    node_all, link_all, task_all = read_all_runs(args.raw_root)
    node_vocab, edge_vocab = build_vocab(node_all, link_all)

    x_node_parts = []
    x_link_parts = []
    x_task_parts = []
    y_node_parts = []
    y_link_parts = []
    y_task_parts = []
    sample_indices = []
    seed_summaries = {}
    sample_offset = 0

    for seed in sorted(node_all["seed"].unique()):
        node = node_all[node_all["seed"] == seed].copy()
        link = link_all[link_all["seed"] == seed].copy()
        task = task_all[task_all["seed"] == seed].copy()
        times = np.array(
            sorted({round(float(time), 3) for time in set(node["time"]).union(set(link["time"])).union(set(task["time"]))}),
            dtype=np.float64,
        )
        node_tensor, _ = build_node_tensor(node, times, node_vocab)
        link_tensor, _ = build_link_tensor(link, times, edge_vocab)
        task_tensor, task_stats = build_task_tensor(task, times)
        sample_index = build_sample_index_for_seed(seed, times, args.history, args.horizon, sample_offset)
        sample_offset += len(sample_index)

        x_node_parts.append(slice_samples(node_tensor, sample_index, "input_start_idx", "input_end_idx"))
        x_link_parts.append(slice_samples(link_tensor, sample_index, "input_start_idx", "input_end_idx"))
        x_task_parts.append(slice_samples(task_tensor, sample_index, "input_start_idx", "input_end_idx"))
        y_node_parts.append(slice_samples(node_tensor, sample_index, "label_start_idx", "label_end_idx"))
        y_link_parts.append(slice_samples(link_tensor, sample_index, "label_start_idx", "label_end_idx"))
        y_task_parts.append(slice_samples(task_tensor, sample_index, "label_start_idx", "label_end_idx"))
        sample_indices.append(sample_index)
        seed_summaries[int(seed)] = {
            "num_times": int(len(times)),
            "num_samples": int(len(sample_index)),
            "node_rows": int(len(node)),
            "link_rows": int(len(link)),
            "task_rows": int(len(task)),
        }

    sample_index_all = pd.concat(sample_indices, ignore_index=True)
    x_node = np.concatenate(x_node_parts, axis=0)
    x_link = np.concatenate(x_link_parts, axis=0)
    x_task = np.concatenate(x_task_parts, axis=0)
    y_node = np.concatenate(y_node_parts, axis=0)
    y_link = np.concatenate(y_link_parts, axis=0)
    y_task = np.concatenate(y_task_parts, axis=0)

    node_vocab.to_csv(args.output_dir / "node_vocab.csv", index=False, encoding="utf-8-sig")
    edge_vocab.to_csv(args.output_dir / "edge_vocab.csv", index=False, encoding="utf-8-sig")
    sample_index_all.to_csv(args.output_dir / "sample_index.csv", index=False, encoding="utf-8-sig")

    np.savez_compressed(
        args.output_dir / "dataset_multiseed_v0_samples.npz",
        node_features=np.array(NODE_FEATURES),
        link_features=np.array(LINK_FEATURES),
        task_features=np.array(
            [
                "num_tasks",
                "total_task_size",
                "total_task_cpu",
                "mean_deadline",
                "mean_priority",
                *[f"num_{state}" for state in TASK_LIFECYCLES],
            ]
        ),
        x_node=x_node,
        x_link=x_link,
        x_task=x_task,
        y_node=y_node,
        y_link=y_link,
        y_task=y_task,
        sample_seed=sample_index_all["seed"].to_numpy(dtype=np.int32),
        sample_id=sample_index_all["sample_id"].to_numpy(dtype=np.int64),
    )

    summary = {
        "raw_root": str(args.raw_root),
        "output_dir": str(args.output_dir),
        "history": args.history,
        "horizon": args.horizon,
        "num_seeds": int(sample_index_all["seed"].nunique()),
        "seeds": [int(seed) for seed in sorted(sample_index_all["seed"].unique())],
        "num_samples": int(len(sample_index_all)),
        "samples_per_seed": {str(k): int(v) for k, v in sample_index_all.groupby("seed")["sample_id"].count().items()},
        "num_nodes": int(len(node_vocab)),
        "num_edges": int(len(edge_vocab)),
        "node_rows": int(len(node_all)),
        "link_rows": int(len(link_all)),
        "task_rows": int(len(task_all)),
        "x_node_shape": list(x_node.shape),
        "x_link_shape": list(x_link.shape),
        "x_task_shape": list(x_task.shape),
        "y_node_shape": list(y_node.shape),
        "y_link_shape": list(y_link.shape),
        "y_task_shape": list(y_task.shape),
        "seed_summaries": seed_summaries,
    }
    plot_paths = plot_reports(args.output_dir, sample_index_all, node_all, link_all, task_all)
    summary["plot_files"] = plot_paths
    (args.output_dir / "dataset_multiseed_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(args.output_dir, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
