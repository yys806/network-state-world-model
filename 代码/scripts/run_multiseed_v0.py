import csv
import json
import os
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from airfogsim import AirFogSimEnv, BaseAlgorithmModule
from airfogsim.scheduler import RewardScheduler, TaskScheduler


EXAMPLE_DIR = Path(__file__).resolve().parent
DATASET_DIR = EXAMPLE_DIR / "outputs" / "dataset_v0_from_demo_run_20260507_190930"
OUTPUT_DIR = DATASET_DIR / "multiseed_v0"
SEEDS = [0, 1, 2, 3, 4]


def load_config(path):
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def make_config(config, max_time=10):
    config["simulation"]["max_simulation_time"] = max_time
    config["traffic"]["max_n_vehicles"] = 50
    config["traffic"]["max_n_UAVs"] = 2
    config["traffic"]["RSU_positions"] = [
        [100, 100, 0],
        [700, 100, 0],
        [100, 700, 0],
        [700, 700, 0],
    ]
    config["task_profile"]["task_node_gen_poss"] = 0.8
    return config


def as_float(value, default=0.0):
    try:
        if hasattr(value, "get"):
            value = value.get()
        if hasattr(value, "item"):
            value = value.item()
        return float(value)
    except Exception:
        return default


def run_one_seed(seed, max_time=10):
    np.random.seed(seed)
    random.seed(seed)
    config = make_config(load_config(EXAMPLE_DIR / "config.yaml"), max_time=max_time)

    env = AirFogSimEnv(config, interactive_mode=None)
    algorithm_module = BaseAlgorithmModule()
    algorithm_module.initialize(env)
    RewardScheduler.setModel(env, "REWARD", "1/task_delay")

    rows = []
    try:
        while not env.isDone():
            algorithm_module.scheduleStep(env)
            env.step()
            done = TaskScheduler.getDoneTaskNum(env)
            failed = TaskScheduler.getOutOfDDLTasks(env)
            rows.append(
                {
                    "seed": seed,
                    "time": round(env.simulation_time, 3),
                    "num_vehicle": len(env.vehicles),
                    "num_uav": len(env.UAVs),
                    "num_rsu": len(env.RSUs),
                    "num_cloud": len(env.cloudServers),
                    "num_all_tasks": len(env.task_manager.getAllTasks()),
                    "num_done_tasks": done,
                    "num_failed_tasks": failed,
                    "success_ratio": done / max(1, done + failed),
                    "v2u_avg_rate": as_float(env.getChannelAvgRate("V2U")),
                    "v2i_avg_rate": as_float(env.getChannelAvgRate("V2I")),
                    "u2i_avg_rate": as_float(env.getChannelAvgRate("U2I")),
                }
            )
    finally:
        env.close()
    return rows


def summarize(df):
    final = df.sort_values(["seed", "time"]).groupby("seed").tail(1).copy()
    summary = {
        "num_seeds": int(final["seed"].nunique()),
        "seeds": [int(x) for x in sorted(final["seed"].unique())],
        "max_time": float(final["time"].max()),
    }
    metrics = [
        "num_vehicle",
        "num_all_tasks",
        "num_done_tasks",
        "num_failed_tasks",
        "success_ratio",
        "v2u_avg_rate",
        "v2i_avg_rate",
        "u2i_avg_rate",
    ]
    for metric in metrics:
        values = final[metric].to_numpy(dtype=float)
        summary[metric] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return final, summary


def plot_multiseed(df, final):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    for seed, sub in df.groupby("seed"):
        axes[0, 0].plot(sub["time"], sub["num_vehicle"], label=f"seed {seed}", lw=1.2)
        axes[0, 1].plot(sub["time"], sub["num_all_tasks"], label=f"seed {seed}", lw=1.2)
        axes[1, 0].plot(sub["time"], sub["v2i_avg_rate"], label=f"seed {seed}", lw=1.2)
        axes[1, 1].plot(sub["time"], sub["success_ratio"], label=f"seed {seed}", lw=1.2)

    titles = [
        "Vehicle count over time",
        "Generated task count over time",
        "V2I average rate over time",
        "Task success ratio over time",
    ]
    for ax, title in zip(axes.ravel(), titles):
        ax.set_title(title)
        ax.set_xlabel("simulation time (s)")
        ax.grid(alpha=0.25)
    axes[0, 0].set_ylabel("count")
    axes[0, 1].set_ylabel("count")
    axes[1, 0].set_ylabel("rate")
    axes[1, 1].set_ylabel("ratio")
    axes[0, 0].legend(fontsize=8, ncol=2)
    fig.suptitle("Multi-seed AirFogSim trajectories")
    fig.tight_layout()
    trajectory_path = OUTPUT_DIR / "multiseed_trajectories.png"
    fig.savefig(trajectory_path, dpi=220)
    plt.close(fig)

    metrics = ["num_vehicle", "num_all_tasks", "num_done_tasks", "v2i_avg_rate", "success_ratio"]
    labels = ["vehicles", "tasks", "done tasks", "V2I avg rate", "success ratio"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(14, 4))
    for ax, metric, label in zip(axes, metrics, labels):
        ax.bar(final["seed"].astype(str), final[metric], color="#2563eb")
        ax.set_title(label)
        ax.set_xlabel("seed")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Final-state variation across seeds")
    fig.tight_layout()
    final_path = OUTPUT_DIR / "multiseed_final_variation.png"
    fig.savefig(final_path, dpi=220)
    plt.close(fig)

    return trajectory_path, final_path


def write_rows(path, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(summary, trajectory_path, final_path):
    v = summary
    report = f"""# multiseed_v0 阶段小结

## 任务设置

- 目的：验证 AirFogSim 当前场景中确实存在内生随机性，并为后续多场景/多 seed 预训练做准备。
- 随机种子：`{v['seeds']}`
- 每个 seed 仿真时长：`{v['max_time']:.1f}s`
- 统计变量：车辆数量、任务数量、任务完成情况、V2U/V2I/U2I 平均链路速率。

## 当前结果

| 指标 | 均值 | 标准差 | 最小值 | 最大值 |
|---|---:|---:|---:|---:|
| 最终车辆数 | {v['num_vehicle']['mean']:.2f} | {v['num_vehicle']['std']:.2f} | {v['num_vehicle']['min']:.2f} | {v['num_vehicle']['max']:.2f} |
| 最终任务数 | {v['num_all_tasks']['mean']:.2f} | {v['num_all_tasks']['std']:.2f} | {v['num_all_tasks']['min']:.2f} | {v['num_all_tasks']['max']:.2f} |
| 完成任务数 | {v['num_done_tasks']['mean']:.2f} | {v['num_done_tasks']['std']:.2f} | {v['num_done_tasks']['min']:.2f} | {v['num_done_tasks']['max']:.2f} |
| 失败任务数 | {v['num_failed_tasks']['mean']:.2f} | {v['num_failed_tasks']['std']:.2f} | {v['num_failed_tasks']['min']:.2f} | {v['num_failed_tasks']['max']:.2f} |
| 成功率 | {v['success_ratio']['mean']:.3f} | {v['success_ratio']['std']:.3f} | {v['success_ratio']['min']:.3f} | {v['success_ratio']['max']:.3f} |
| V2U 平均速率 | {v['v2u_avg_rate']['mean']:.3f} | {v['v2u_avg_rate']['std']:.3f} | {v['v2u_avg_rate']['min']:.3f} | {v['v2u_avg_rate']['max']:.3f} |
| V2I 平均速率 | {v['v2i_avg_rate']['mean']:.3f} | {v['v2i_avg_rate']['std']:.3f} | {v['v2i_avg_rate']['min']:.3f} | {v['v2i_avg_rate']['max']:.3f} |
| U2I 平均速率 | {v['u2i_avg_rate']['mean']:.3f} | {v['u2i_avg_rate']['std']:.3f} | {v['u2i_avg_rate']['min']:.3f} | {v['u2i_avg_rate']['max']:.3f} |

## 怎么解释

同一个小场景在不同 seed 下会产生不同车辆到达、车辆路线、任务生成、任务属性和信道变化，因此最终任务数量、完成情况和链路速率会发生变化。这说明当前平台不是只有一条固定轨迹，而是可以通过 seed 和配置生成多条轨迹。

当前 `dataset_v0` 仍然只来自 seed 0 的单条轨迹，因此它适合做第一版 pipeline 和 baseline 验证，但还不足以支撑“多场景预训练”结论。下一步应该把多 seed 输出整理成 `dataset_multiseed_v0`，用于训练/测试跨随机轨迹的泛化能力。

## 输出文件

- `{trajectory_path.name}`
- `{final_path.name}`
- `multiseed_timeseries.csv`
- `multiseed_final_summary.csv`
- `multiseed_summary.json`
"""
    (OUTPUT_DIR / "multiseed_report.md").write_text(report, encoding="utf-8")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in SEEDS:
        print(f"Running seed {seed}...")
        rows.extend(run_one_seed(seed, max_time=10))

    write_rows(OUTPUT_DIR / "multiseed_timeseries.csv", rows)
    df = pd.DataFrame(rows)
    final, summary = summarize(df)
    final.to_csv(OUTPUT_DIR / "multiseed_final_summary.csv", index=False, encoding="utf-8-sig")
    trajectory_path, final_path = plot_multiseed(df, final)
    with open(OUTPUT_DIR / "multiseed_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    write_report(summary, trajectory_path, final_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Output dir: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
