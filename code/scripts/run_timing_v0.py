import json
import os
import random
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from airfogsim import AirFogSimEnv, BaseAlgorithmModule
from airfogsim.scheduler import RewardScheduler
from train_baseline_v0 import (
    build_xy,
    chronological_split,
    fit_best_ridge,
    load_dataset,
    predict_ridge,
    standardize,
)


EXAMPLE_DIR = Path(__file__).resolve().parent
DATASET_DIR = EXAMPLE_DIR / "outputs" / "dataset_v0_from_demo_run_20260507_190930"
OUTPUT_DIR = DATASET_DIR / "timing_v0"


def load_config(path):
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def make_timing_config(config, max_time=10):
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


def percentile(values, q):
    if len(values) == 0:
        return 0.0
    return float(np.percentile(np.asarray(values), q))


def benchmark_airfogsim(max_time=10, seed=0):
    np.random.seed(seed)
    random.seed(seed)
    config = make_timing_config(load_config(EXAMPLE_DIR / "config.yaml"), max_time=max_time)

    env = AirFogSimEnv(config, interactive_mode=None)
    algorithm_module = BaseAlgorithmModule()
    algorithm_module.initialize(env)
    RewardScheduler.setModel(env, "REWARD", "1/task_delay")

    schedule_ms = []
    step_ms = []
    total_ms = []

    try:
        while not env.isDone():
            t0 = time.perf_counter()
            algorithm_module.scheduleStep(env)
            t1 = time.perf_counter()
            env.step()
            t2 = time.perf_counter()
            schedule_ms.append((t1 - t0) * 1000)
            step_ms.append((t2 - t1) * 1000)
            total_ms.append((t2 - t0) * 1000)
    finally:
        env.close()

    return {
        "num_steps": len(total_ms),
        "schedule_ms_mean": float(np.mean(schedule_ms)),
        "schedule_ms_p50": percentile(schedule_ms, 50),
        "schedule_ms_p95": percentile(schedule_ms, 95),
        "step_ms_mean": float(np.mean(step_ms)),
        "step_ms_p50": percentile(step_ms, 50),
        "step_ms_p95": percentile(step_ms, 95),
        "schedule_plus_step_ms_mean": float(np.mean(total_ms)),
        "schedule_plus_step_ms_p50": percentile(total_ms, 50),
        "schedule_plus_step_ms_p95": percentile(total_ms, 95),
        "all_schedule_ms": schedule_ms,
        "all_step_ms": step_ms,
        "all_total_ms": total_ms,
    }


def benchmark_model_inference(repeats=500):
    arrays, edge_vocab = load_dataset(DATASET_DIR)
    x, y, persistence, meta = build_xy(arrays, edge_vocab)
    train_idx, val_idx, test_idx = chronological_split(len(x))

    y_residual = y - persistence
    (x_parts, x_mean, x_std) = standardize(x[train_idx], x[train_idx], x[val_idx], x[test_idx])
    x_train, x_val, x_test = x_parts

    y_mean = y_residual[train_idx].mean(axis=0, keepdims=True)
    y_std = y_residual[train_idx].std(axis=0, keepdims=True)
    y_std = np.where(y_std < 1e-6, 1.0, y_std)
    y_train_s = (y_residual[train_idx] - y_mean) / y_std
    y_val_s = (y_residual[val_idx] - y_mean) / y_std

    alpha, weights, _ = fit_best_ridge(x_train, y_train_s, x_val, y_val_s, [0.1, 1.0, 10.0, 100.0])

    batch_times = []
    per_sample_times = []
    persistence_batch_times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _ = predict_ridge(x_test, weights)
        t1 = time.perf_counter()
        batch_times.append((t1 - t0) * 1000)

        t0 = time.perf_counter()
        for row in x_test:
            _ = predict_ridge(row[None, :], weights)
        t1 = time.perf_counter()
        per_sample_times.append((t1 - t0) * 1000 / len(x_test))

        t0 = time.perf_counter()
        _ = persistence[test_idx].copy()
        t1 = time.perf_counter()
        persistence_batch_times.append((t1 - t0) * 1000)

    return {
        "num_test_samples": int(len(test_idx)),
        "num_features": int(meta["num_features"]),
        "num_targets": int(meta["num_targets"]),
        "horizon": int(meta["horizon"]),
        "ridge_alpha": float(alpha),
        "ridge_batch_ms_mean": float(np.mean(batch_times)),
        "ridge_batch_ms_p50": percentile(batch_times, 50),
        "ridge_batch_ms_p95": percentile(batch_times, 95),
        "ridge_per_sample_ms_mean": float(np.mean(per_sample_times)),
        "ridge_per_sample_ms_p50": percentile(per_sample_times, 50),
        "ridge_per_sample_ms_p95": percentile(per_sample_times, 95),
        "persistence_batch_ms_mean": float(np.mean(persistence_batch_times)),
        "persistence_batch_ms_p50": percentile(persistence_batch_times, 50),
        "persistence_batch_ms_p95": percentile(persistence_batch_times, 95),
    }


def plot_results(summary):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    air = summary["airfogsim"]
    model = summary["model_inference"]
    horizon = model["horizon"]

    labels = [
        "AirFogSim\none step",
        f"AirFogSim\n{horizon}-step rollout",
        f"Ridge\n{horizon}-step/sample",
        f"Persistence\nbatch/{model['num_test_samples']} samples",
    ]
    values = [
        air["schedule_plus_step_ms_mean"],
        air["schedule_plus_step_ms_mean"] * horizon,
        model["ridge_per_sample_ms_mean"],
        model["persistence_batch_ms_mean"],
    ]
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd"]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(labels, values, color=colors)
    ax.set_ylabel("Average wall-clock time (ms, log scale)")
    ax.set_yscale("log")
    ax.set_title("Timing comparison: simulator rollout vs baseline inference")
    ax.grid(axis="y", alpha=0.25, which="both")
    for idx, value in enumerate(values):
        ax.text(idx, value * 1.1, f"{value:.4f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    comparison_path = OUTPUT_DIR / "timing_comparison_logscale.png"
    fig.savefig(comparison_path, dpi=220)
    plt.close(fig)

    series = pd.DataFrame(
        {
            "schedule_ms": air["all_schedule_ms"],
            "env_step_ms": air["all_step_ms"],
            "schedule_plus_step_ms": air["all_total_ms"],
        }
    )
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(series.index, series["schedule_ms"], label="scheduler", lw=1.2)
    ax.plot(series.index, series["env_step_ms"], label="env.step", lw=1.2)
    ax.plot(series.index, series["schedule_plus_step_ms"], label="total", lw=1.4)
    ax.set_xlabel("simulation step")
    ax.set_ylabel("wall-clock time (ms)")
    ax.set_title("AirFogSim per-step timing")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    step_path = OUTPUT_DIR / "airfogsim_step_timing.png"
    fig.savefig(step_path, dpi=220)
    plt.close(fig)

    series.to_csv(OUTPUT_DIR / "airfogsim_step_timing.csv", index=False, encoding="utf-8-sig")
    return comparison_path, step_path


def write_report(summary, comparison_path, step_path):
    air = summary["airfogsim"]
    model = summary["model_inference"]
    horizon = model["horizon"]
    sim_k = air["schedule_plus_step_ms_mean"] * horizon
    ridge = model["ridge_per_sample_ms_mean"]
    ratio = sim_k / ridge if ridge > 0 else float("inf")

    report = f"""# timing_v0 阶段小结

## 任务设置

- 目的：补充“复杂度/计算量”问题的第一版实测证据。
- 仿真侧：运行 AirFogSim demo 场景，统计 `scheduleStep + env.step()` 的逐步耗时。
- 模型侧：复用 `dataset_v0`，训练 Ridge residual baseline，统计测试集推理耗时。
- 预测设置：历史窗口 `H=8`，预测步长 `K={horizon}`。

## 当前结果

| 项目 | 平均耗时 |
|---|---:|
| AirFogSim scheduler | {air['schedule_ms_mean']:.4f} ms/step |
| AirFogSim env.step | {air['step_ms_mean']:.4f} ms/step |
| AirFogSim scheduler + env.step | {air['schedule_plus_step_ms_mean']:.4f} ms/step |
| AirFogSim 估算 {horizon}-step rollout | {sim_k:.4f} ms |
| Ridge residual baseline | {ridge:.6f} ms/sample |
| Persistence baseline batch | {model['persistence_batch_ms_mean']:.6f} ms/{model['num_test_samples']} samples |

按当前小场景和 Ridge baseline 估算，AirFogSim 的 {horizon}-step 显式 rollout 平均耗时约为 Ridge 单样本推理的 `{ratio:.1f}` 倍。这个倍数只用于说明当前小场景下存在在线推理加速空间，不能直接外推到更大场景或最终模型。

## 怎么解释

AirFogSim 的每一步会显式执行交通、任务、通信、计算、存储和能量等模块，因此多步 rollout 需要重复调用完整仿真流程。学习式模型在训练完成后，可以把历史窗口编码为特征，并用一次前向推理输出未来 `K` 步预测，所以在线推理可能更快。

需要强调的是：这不是说当前 Ridge baseline 已经可以替代 AirFogSim。当前结果只说明“复杂度/耗时评估流程已经建立”，后续还需要在误差可接受的前提下，对更强的双图模型或动作条件 world model 做同样计时。

## 输出文件

- `{comparison_path.name}`
- `{step_path.name}`
- `airfogsim_step_timing.csv`
- `timing_summary.json`
"""
    (OUTPUT_DIR / "timing_report.md").write_text(report, encoding="utf-8")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    airfogsim_summary = benchmark_airfogsim(max_time=10, seed=0)
    model_summary = benchmark_model_inference(repeats=500)
    summary = {
        "airfogsim": {k: v for k, v in airfogsim_summary.items() if not k.startswith("all_")},
        "model_inference": model_summary,
    }
    plot_payload = {
        "airfogsim": airfogsim_summary,
        "model_inference": model_summary,
    }
    comparison_path, step_path = plot_results(plot_payload)
    with open(OUTPUT_DIR / "timing_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    write_report(summary, comparison_path, step_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Output dir: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
