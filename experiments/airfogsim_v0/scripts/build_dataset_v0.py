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
DEFAULT_RUN_DIR = EXAMPLE_DIR / "outputs" / "demo_run_20260507_190930"

NODE_FEATURES = ["x", "y", "z", "speed", "acceleration", "cpu", "storage"]
LINK_FEATURES = ["distance", "rate_sum", "csi_mean", "active_task_count", "allocated_rb_count"]
TASK_LIFECYCLES = ["to_offload", "computing", "returning", "finished"]


def parse_args():
    parser = argparse.ArgumentParser(description="Build dataset_v0 from AirFogSim exported CSV files.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR, help="AirFogSim output run directory.")
    parser.add_argument("--history", type=int, default=8, help="History window length H.")
    parser.add_argument("--horizon", type=int, default=3, help="Prediction horizon K.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory for dataset_v0.")
    return parser.parse_args()


def read_inputs(run_dir):
    node = pd.read_csv(run_dir / "node_states.csv")
    link = pd.read_csv(run_dir / "link_states.csv")
    task = pd.read_csv(run_dir / "task_states.csv")
    for df in [node, link, task]:
        df["time"] = df["time"].round(3)
    return node, link, task


def make_output_dir(run_dir, output_dir):
    if output_dir is None:
        output_dir = run_dir.parent / f"dataset_v0_from_{run_dir.name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


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
    tensor = task_stats[columns].to_numpy(dtype=np.float32)
    return tensor, task_stats


def build_sample_index(times, history, horizon):
    rows = []
    for end_idx in range(history - 1, len(times) - horizon):
        sample_id = len(rows)
        rows.append(
            {
                "sample_id": sample_id,
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


def write_field_mapping(output_dir):
    text = """# AirFogSim 字段映射表

## node_states.csv

| 原始字段 | 建模符号 | 含义 |
|---|---|---|
| `time` | $t$ | 离散时间步 |
| `node_id` | 节点编号 | UAV、vehicle、RSU、cloud 的唯一标识 |
| `node_type` | 节点类型 | 区分无人机、车辆、RSU、云节点 |
| `x, y, z` | $\\mathbf{p}_{i,t}$ | 节点空间位置 |
| `speed` | $\\|\\mathbf{v}_{i,t}\\|$ | 节点速度大小 |
| `acceleration` | 加速度状态 | 节点运动变化趋势 |
| `cpu` | $c_{i,t}$ 或 $C_{r,t}^{\\mathrm{edge}}$ | 本地或边缘可用算力 |
| `storage` | 缓存/存储资源 | 节点可用存储资源 |

## link_states.csv

| 原始字段 | 建模符号 | 含义 |
|---|---|---|
| `tx_id, rx_id` | $(i,j)$ | 链路发送端和接收端 |
| `link_type` | 链路类型 | V2U、V2I、U2I |
| `distance` | $d_{ij,t}$ | 节点间距离，可用于物理边 |
| `rate_sum` | $r_{ij,t}$ | 链路传输速率 |
| `csi_mean` | $\\gamma_{ij,t}$ | CSI/SINR 摘要的近似输入 |
| `active_task_count` | 任务占用强度 | 当前链路上激活任务数 |
| `allocated_rb_count` | $\\eta_{ij,t}$ | 资源块占用数 |

## task_states.csv

| 原始字段 | 建模符号 | 含义 |
|---|---|---|
| `task_id` | 任务编号 | 单个任务唯一标识 |
| `task_node_id` | 任务源节点 | 任务产生位置 |
| `current_node_id` | 当前所在节点 | 任务当前执行或传输位置 |
| `assigned_to` | 卸载目标 | 当前任务被分配到的节点 |
| `task_size` | $D_{m,t}$ | 任务数据量 |
| `task_cpu` | $C_{m,t}$ | 任务计算需求 |
| `deadline` | $\\tau_{m,t}$ | 任务时限 |
| `priority` | $\\rho_{m,t}$ | 任务优先级 |
| `lifecycle_state` | $\\mathbf{Y}^{\\mathrm{task}}$ | 任务生命周期状态 |
"""
    (output_dir / "field_mapping.md").write_text(text, encoding="utf-8")


def plot_reports(output_dir, node, link, task_stats):
    if plt is None:
        return []
    generated = []
    node_counts = node.groupby(["time", "node_type"])["node_id"].nunique().unstack(fill_value=0)
    node_counts.plot(figsize=(8, 4))
    plt.xlabel("time")
    plt.ylabel("node count")
    plt.tight_layout()
    path = output_dir / "node_counts_over_time.png"
    plt.savefig(path, dpi=200)
    plt.close()
    generated.append(str(path))

    rate_by_type = link.groupby(["time", "link_type"])["rate_sum"].mean().unstack(fill_value=0)
    rate_by_type.plot(figsize=(8, 4))
    plt.xlabel("time")
    plt.ylabel("mean rate_sum")
    plt.tight_layout()
    path = output_dir / "link_rate_by_type.png"
    plt.savefig(path, dpi=200)
    plt.close()
    generated.append(str(path))

    lifecycle_cols = [f"num_{state}" for state in TASK_LIFECYCLES]
    task_stats.plot(x="time", y=lifecycle_cols, figsize=(8, 4))
    plt.xlabel("time")
    plt.ylabel("task count")
    plt.tight_layout()
    path = output_dir / "task_lifecycle_counts.png"
    plt.savefig(path, dpi=200)
    plt.close()
    generated.append(str(path))
    return generated


def write_quality_report(output_dir, summary):
    lines = [
        "# dataset_v0 数据质量报告",
        "",
        f"- 源目录：`{summary['source_run_dir']}`",
        f"- 时间步数量：{summary['num_times']}",
        f"- 时间范围：{summary['time_min']} 到 {summary['time_max']}",
        f"- 历史窗口 H：{summary['history']}",
        f"- 预测步长 K：{summary['horizon']}",
        f"- 可构造样本数：{summary['num_samples']}",
        f"- 节点数：{summary['num_nodes']}",
        f"- 边数：{summary['num_edges']}",
        f"- 节点原始行数：{summary['node_rows']}",
        f"- 链路原始行数：{summary['link_rows']}",
        f"- 任务原始行数：{summary['task_rows']}",
        "",
        "## 节点类型",
        "",
    ]
    for key, value in summary["node_type_counts"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## 链路类型行数", ""])
    for key, value in summary["link_type_rows"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## 任务状态行数", ""])
    for key, value in summary["task_lifecycle_rows"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            "## 当前结论",
            "",
            "- 当前输出已经可以支撑第一版物理图和通信图构造。",
            "- 节点和链路在 200 个时间步上都有记录；任务从 0.3 秒开始出现，前两个时间步任务为空，构造样本时已用聚合零向量补齐。",
            "- `active_task_count` 与 `allocated_rb_count` 可以作为动作/资源占用的近似观测，但严格动作变量后续还需要从调度模块中进一步导出。",
        ]
    )
    (output_dir / "data_quality_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = make_output_dir(run_dir, args.output_dir)

    node, link, task = read_inputs(run_dir)
    times = np.array(
        sorted({round(float(time), 3) for time in set(node["time"]).union(set(link["time"])).union(set(task["time"]))}),
        dtype=np.float64,
    )
    node_vocab, edge_vocab = build_vocab(node, link)
    node_tensor, node_mask = build_node_tensor(node, times, node_vocab)
    link_tensor, link_mask = build_link_tensor(link, times, edge_vocab)
    task_tensor, task_stats = build_task_tensor(task, times)
    sample_index = build_sample_index(times, args.history, args.horizon)

    x_node = slice_samples(node_tensor, sample_index, "input_start_idx", "input_end_idx")
    x_link = slice_samples(link_tensor, sample_index, "input_start_idx", "input_end_idx")
    x_task = slice_samples(task_tensor, sample_index, "input_start_idx", "input_end_idx")
    y_node = slice_samples(node_tensor, sample_index, "label_start_idx", "label_end_idx")
    y_link = slice_samples(link_tensor, sample_index, "label_start_idx", "label_end_idx")
    y_task = slice_samples(task_tensor, sample_index, "label_start_idx", "label_end_idx")

    node_vocab.to_csv(output_dir / "node_vocab.csv", index=False, encoding="utf-8-sig")
    edge_vocab.to_csv(output_dir / "edge_vocab.csv", index=False, encoding="utf-8-sig")
    task_stats.to_csv(output_dir / "task_time_stats.csv", index=False, encoding="utf-8-sig")
    sample_index.to_csv(output_dir / "sample_index.csv", index=False, encoding="utf-8-sig")

    np.savez_compressed(
        output_dir / "dataset_v0_samples.npz",
        times=times,
        node_features=np.array(NODE_FEATURES),
        link_features=np.array(LINK_FEATURES),
        task_features=np.array(list(task_stats.columns[1:])),
        node_tensor=node_tensor,
        node_mask=node_mask,
        link_tensor=link_tensor,
        link_mask=link_mask,
        task_tensor=task_tensor,
        x_node=x_node,
        x_link=x_link,
        x_task=x_task,
        y_node=y_node,
        y_link=y_link,
        y_task=y_task,
    )

    summary = {
        "source_run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "history": args.history,
        "horizon": args.horizon,
        "num_times": int(len(times)),
        "time_min": float(times.min()),
        "time_max": float(times.max()),
        "num_samples": int(len(sample_index)),
        "num_nodes": int(len(node_vocab)),
        "num_edges": int(len(edge_vocab)),
        "node_rows": int(len(node)),
        "link_rows": int(len(link)),
        "task_rows": int(len(task)),
        "node_tensor_shape": list(node_tensor.shape),
        "link_tensor_shape": list(link_tensor.shape),
        "task_tensor_shape": list(task_tensor.shape),
        "x_node_shape": list(x_node.shape),
        "x_link_shape": list(x_link.shape),
        "x_task_shape": list(x_task.shape),
        "y_node_shape": list(y_node.shape),
        "y_link_shape": list(y_link.shape),
        "y_task_shape": list(y_task.shape),
        "node_type_counts": node.groupby("node_type")["node_id"].nunique().to_dict(),
        "link_type_rows": link["link_type"].value_counts().to_dict(),
        "task_lifecycle_rows": task["lifecycle_state"].value_counts().to_dict(),
    }
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_field_mapping(output_dir)
    write_quality_report(output_dir, summary)
    plot_paths = plot_reports(output_dir, node, link, task_stats)
    summary["plot_files"] = plot_paths
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
