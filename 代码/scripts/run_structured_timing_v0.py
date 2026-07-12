import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from run_structured_dual_branch_baseline_v0 import (
    build_branch_features,
    build_targets_and_persistence,
    fit_standardizer,
    load_arrays,
    predict_model,
    split_by_seed,
    train_model,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports"
OUTPUT_DIR = REPORT_DIR / "structured_timing_v0"
FIGURE_DIR = ROOT / "figures"


def benchmark(fn, warmup=30, repeat=300):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    arr = np.asarray(times, dtype=np.float64)
    return {
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "repeat": repeat,
    }


def plot_timing(rows, output_path):
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    ax.bar(df["item"], df["mean_ms"], color=["#6b7280", "#2563eb", "#16a34a", "#dc2626"])
    ax.set_yscale("log")
    ax.set_ylabel("Mean time (ms, log scale)")
    ax.set_title("Simulator rollout vs learned model inference")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", labelrotation=15)
    for idx, value in enumerate(df["mean_ms"]):
        ax.text(idx, value, f"{value:.4f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def write_report(summary, rows, output_path):
    df = pd.DataFrame(rows)
    sim = float(df[df["item"] == "AirFogSim 3-step rollout"]["mean_ms"].iloc[0])
    structured = float(df[df["item"] == "structured state-action"]["mean_ms"].iloc[0])
    ratio = sim / structured if structured > 0 else float("inf")
    lines = [
        "# Structured timing report v0",
        "",
        "## Purpose",
        "",
        "This report adds online inference timing for the structured state-action baseline, so the complexity discussion is not limited to Ridge.",
        "",
        "## Timing table",
        "",
        df.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Interpretation",
        "",
        f"- AirFogSim estimated 3-step rollout mean time is `{sim:.4f}` ms.",
        f"- Structured state-action model mean inference time is `{structured:.4f}` ms/sample on CPU.",
        f"- In this small scenario, AirFogSim 3-step rollout is about `{ratio:.1f}x` slower than structured model inference.",
        "- This only supports the online-inference speed argument. It does not mean the current structured model is accurate enough to replace the simulator.",
        "",
        "## Outputs",
        "",
    ]
    for key, value in summary["outputs"].items():
        lines.append(f"- {key}: `{value}`")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    arrays, actions, node_vocab, edge_vocab = load_arrays()
    features = build_branch_features(arrays, actions, node_vocab, edge_vocab)
    y, persistence = build_targets_and_persistence(arrays, edge_vocab)
    train_idx, val_idx, test_idx = split_by_seed(arrays["sample_seed"])
    y_res = y - persistence
    y_mean, y_std = fit_standardizer(y_res[train_idx])
    y_res_scaled = ((y_res - y_mean) / y_std).astype(np.float32)

    action_model, action_stats, _, action_info = train_model(
        features, y_res_scaled, train_idx, val_idx, use_action=True, seed=63
    )
    _ = predict_model(action_model, features, test_idx, action_stats, use_action=True)

    def predict_batch():
        predict_model(action_model, features, test_idx, action_stats, use_action=True)

    def predict_single():
        predict_model(action_model, features, test_idx[:1], action_stats, use_action=True)

    batch_stats = benchmark(predict_batch)
    single_stats = benchmark(predict_single)

    timing_summary_path = REPORT_DIR / "timing_summary.json"
    timing_summary = json.loads(timing_summary_path.read_text(encoding="utf-8"))
    airfogsim_3step = timing_summary["airfogsim"]["schedule_plus_step_ms_mean"] * 3
    ridge_sample = timing_summary["model_inference"]["ridge_per_sample_ms_mean"]
    persistence_batch = timing_summary["model_inference"]["persistence_batch_ms_mean"]

    rows = [
        {"item": "persistence batch", "mean_ms": persistence_batch, "p50_ms": np.nan, "p95_ms": np.nan, "note": "29-sample batch from timing_v0"},
        {"item": "Ridge per sample", "mean_ms": ridge_sample, "p50_ms": timing_summary["model_inference"]["ridge_per_sample_ms_p50"], "p95_ms": timing_summary["model_inference"]["ridge_per_sample_ms_p95"], "note": "from timing_v0"},
        {"item": "structured state-action", "mean_ms": single_stats["mean_ms"], "p50_ms": single_stats["p50_ms"], "p95_ms": single_stats["p95_ms"], "note": "CPU, single sample"},
        {"item": "AirFogSim 3-step rollout", "mean_ms": airfogsim_3step, "p50_ms": np.nan, "p95_ms": np.nan, "note": "schedule+env.step mean x 3"},
    ]
    rows_batch = rows + [
        {"item": "structured state-action batch", "mean_ms": batch_stats["mean_ms"], "p50_ms": batch_stats["p50_ms"], "p95_ms": batch_stats["p95_ms"], "note": f"CPU, {len(test_idx)}-sample batch"}
    ]
    metrics_path = OUTPUT_DIR / "structured_timing_metrics.csv"
    pd.DataFrame(rows_batch).to_csv(metrics_path, index=False, encoding="utf-8-sig")
    fig_path = FIGURE_DIR / "structured_timing_comparison_logscale.png"
    plot_timing(rows, fig_path)
    summary = {
        "training": {"structured_state_action": action_info},
        "test_samples": int(len(test_idx)),
        "single_sample": single_stats,
        "batch": batch_stats,
        "outputs": {
            "metrics_csv": str(metrics_path),
            "figure": str(fig_path),
        },
    }
    report_path = OUTPUT_DIR / "structured_timing_report.md"
    summary["outputs"]["report_md"] = str(report_path)
    write_report(summary, rows_batch, report_path)
    summary_path = OUTPUT_DIR / "structured_timing_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
