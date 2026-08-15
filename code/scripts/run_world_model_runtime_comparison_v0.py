import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from run_world_model_v0 import FIGURE_DIR, ROOT, load_dataset, split_by_seed
from run_world_model_v4_dual_graph_rollout import (
    DualGraphWorldModelDataset,
    augment_arrays_with_physical_edges,
    display_path,
    make_model,
    make_stats,
)


OUTPUT_DIR = ROOT / "reports" / "world_model_runtime_comparison_v0"
AIRFOGSIM_TIMING_SUMMARY = ROOT / "reports" / "timing_summary.json"


def summarize_times(values, warmup=0):
    values = np.asarray(values, dtype=np.float64)
    if warmup > 0:
        values = values[warmup:]
    if values.size == 0:
        return {
            "num_repeats": 0,
            "mean_ms": float("nan"),
            "p50_ms": float("nan"),
            "p95_ms": float("nan"),
            "min_ms": float("nan"),
            "max_ms": float("nan"),
        }
    return {
        "num_repeats": int(values.size),
        "mean_ms": float(values.mean()),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "min_ms": float(values.min()),
        "max_ms": float(values.max()),
    }


def compute_speedup(sim_step_ms, horizon, model_sample_ms):
    sim_k_step_ms = float(sim_step_ms) * int(horizon)
    model_sample_ms = float(model_sample_ms)
    return {
        "sim_step_ms": float(sim_step_ms),
        "horizon": int(horizon),
        "sim_k_step_ms": sim_k_step_ms,
        "model_sample_ms": model_sample_ms,
        "speedup": sim_k_step_ms / model_sample_ms if model_sample_ms > 0 else float("inf"),
    }


def load_airfogsim_timing():
    if not AIRFOGSIM_TIMING_SUMMARY.exists():
        raise FileNotFoundError(f"Missing AirFogSim timing summary: {AIRFOGSIM_TIMING_SUMMARY}")
    data = json.loads(AIRFOGSIM_TIMING_SUMMARY.read_text(encoding="utf-8"))
    air = data["airfogsim"]
    return {
        "source": display_path(AIRFOGSIM_TIMING_SUMMARY),
        "schedule_plus_step_ms_mean": float(air["schedule_plus_step_ms_mean"]),
        "schedule_plus_step_ms_p50": float(air["schedule_plus_step_ms_p50"]),
        "schedule_plus_step_ms_p95": float(air["schedule_plus_step_ms_p95"]),
        "schedule_ms_mean": float(air["schedule_ms_mean"]),
        "step_ms_mean": float(air["step_ms_mean"]),
        "num_steps": int(air["num_steps"]),
    }


def make_timing_batch(arrays, idx, stats, batch_size):
    ds = DualGraphWorldModelDataset(arrays, idx[:batch_size], stats)
    batch = [item for item in ds]
    stacked = []
    for field_idx in range(9):
        stacked.append(torch.stack([item[field_idx] for item in batch], dim=0))
    return stacked[:6]


def benchmark_v4_forward(batch_size=64, repeats=260, warmup=30):
    arrays = augment_arrays_with_physical_edges(load_dataset())
    train_idx, _, test_idx = split_by_seed(arrays["sample_seed"])
    stats = make_stats(arrays, train_idx)
    model = make_model(arrays)
    model.eval()
    inputs = make_timing_batch(arrays, test_idx, stats, min(batch_size, len(test_idx)))
    horizon = int(arrays["y_link_rate"].shape[1])
    actual_batch = int(inputs[0].shape[0])

    times = []
    with torch.no_grad():
        for _ in range(repeats):
            t0 = time.perf_counter()
            _ = model(*inputs)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)
    summary = summarize_times(times, warmup=warmup)
    summary.update(
        {
            "model": "world_model_v4_dual_graph_untrained_forward",
            "device": "cpu",
            "batch_size": actual_batch,
            "horizon": horizon,
            "raw_repeats": int(repeats),
            "warmup": int(warmup),
            "per_sample_mean_ms": float(summary["mean_ms"] / actual_batch),
            "per_sample_p50_ms": float(summary["p50_ms"] / actual_batch),
            "per_sample_p95_ms": float(summary["p95_ms"] / actual_batch),
        }
    )
    return summary


def plot_runtime(rows):
    import matplotlib.pyplot as plt

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / "world_model_runtime_comparison_v0.png"
    df = pd.DataFrame(rows)
    labels = df["label"].tolist()
    values = df["ms"].astype(float).to_numpy()
    plt.figure(figsize=(9.5, 4.6))
    plt.bar(labels, values, color=["#2563eb", "#1d4ed8", "#dc2626", "#d97706"])
    plt.yscale("log")
    plt.ylabel("wall-clock time (ms, log scale)")
    plt.title("AirFogSim rollout vs world-model forward inference")
    plt.grid(axis="y", alpha=0.25, which="both")
    for idx, value in enumerate(values):
        plt.text(idx, value * 1.15, f"{value:.4f}", ha="center", va="bottom", fontsize=8)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()
    return path


def write_report(summary, metrics_df):
    speed = summary["speedup"]
    lines = [
        "# World model runtime comparison v0",
        "",
        "## Goal",
        "",
        "This report compares the existing AirFogSim per-step timing with a CPU forward pass of the v4 dual-graph world-model architecture. The v4 timing uses the model architecture with random weights, so it measures inference cost rather than prediction quality.",
        "",
        "## Main Result",
        "",
        f"- AirFogSim mean step time: `{summary['airfogsim']['schedule_plus_step_ms_mean']:.6f}` ms.",
        f"- AirFogSim estimated `{speed['horizon']}`-step rollout time: `{speed['sim_k_step_ms']:.6f}` ms.",
        f"- v4 dual-graph CPU forward time per sample: `{speed['model_sample_ms']:.6f}` ms.",
        f"- Estimated online speedup for one `{speed['horizon']}`-step sample: `{speed['speedup']:.2f}x`.",
        "",
        "## Metrics",
        "",
        metrics_df.to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        "- This result supports the decision-interface motivation: once trained, a world model can evaluate many candidate futures faster than repeatedly stepping the simulator.",
        "- The timing is CPU-only and architecture-level. A final claim still needs trained-model timing, larger scenarios, and candidate-action ranking/regret evaluation.",
        "",
        "## Outputs",
        "",
    ]
    for key, value in summary["outputs"].items():
        lines.append(f"- {key}: `{value}`")
    path = OUTPUT_DIR / "world_model_runtime_comparison_v0_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    air = load_airfogsim_timing()
    model = benchmark_v4_forward()
    speedup = compute_speedup(
        air["schedule_plus_step_ms_mean"],
        model["horizon"],
        model["per_sample_mean_ms"],
    )
    rows = [
        {"label": "AirFogSim one step", "ms": air["schedule_plus_step_ms_mean"], "kind": "simulator"},
        {"label": f"AirFogSim {model['horizon']}-step", "ms": speedup["sim_k_step_ms"], "kind": "simulator"},
        {"label": f"v4 batch {model['batch_size']}", "ms": model["mean_ms"], "kind": "world_model"},
        {"label": "v4 per sample", "ms": model["per_sample_mean_ms"], "kind": "world_model"},
    ]
    metrics_df = pd.DataFrame(
        [
            {"component": "airfogsim", **air},
            {"component": "world_model_v4_forward", **model},
            {"component": "speedup", **speedup},
        ]
    )
    plot_path = plot_runtime(rows)
    metrics_path = OUTPUT_DIR / "world_model_runtime_comparison_v0_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    summary = {
        "output_dir": display_path(OUTPUT_DIR),
        "airfogsim": air,
        "world_model_v4_forward": model,
        "speedup": speedup,
        "outputs": {
            "metrics_csv": display_path(metrics_path),
            "runtime_plot": display_path(plot_path),
        },
    }
    report_path = write_report(summary, metrics_df)
    summary["outputs"]["report_md"] = display_path(report_path)
    summary_path = OUTPUT_DIR / "world_model_runtime_comparison_v0_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
