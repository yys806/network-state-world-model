import json

import numpy as np
import pandas as pd

from run_world_model_v0 import FIGURE_DIR, ROOT, choose_threshold, load_dataset, split_by_seed
from run_world_model_v3_graph_rollout import evaluate
from run_world_model_v4_dual_graph_ablation import make_physical_variant
from run_world_model_v4_dual_graph_rollout import (
    augment_arrays_with_physical_edges,
    display_path,
    make_stats,
    predict,
    train_model,
)


OUTPUT_DIR = ROOT / "reports" / "world_model_v4_seed_stability"
DEFAULT_VARIANTS = ["dual_full", "no_physical"]
DEFAULT_TORCH_SEEDS = [11, 42, 73]


def summarize_stability(metrics_df):
    test = metrics_df[metrics_df["split"] == "test_seed_4"].copy()
    wanted_cols = [
        "activity_f1",
        "activity_precision",
        "activity_recall",
        "rate_all_rmse",
        "rate_active_rmse",
        "task_rmse",
    ]
    metric_cols = [col for col in wanted_cols if col in test.columns]
    grouped = test.groupby("physical_variant", dropna=False)
    rows = []
    for variant, part in grouped:
        row = {"physical_variant": variant, "runs": int(len(part))}
        for col in metric_cols:
            row[f"{col}_mean"] = float(part[col].mean())
            row[f"{col}_std"] = float(part[col].std(ddof=1)) if len(part) > 1 else 0.0
            row[f"{col}_min"] = float(part[col].min())
            row[f"{col}_max"] = float(part[col].max())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("physical_variant").reset_index(drop=True)


def run_one(base_arrays, train_idx, val_idx, test_idx, variant, torch_seed):
    arrays = make_physical_variant(base_arrays, variant)
    stats = make_stats(arrays, train_idx)
    model, history, train_info = train_model(arrays, train_idx, val_idx, stats, torch_seed=torch_seed)
    val_pred = predict(model, arrays, val_idx, stats)
    test_pred = predict(model, arrays, test_idx, stats)
    best_threshold, threshold_df = choose_threshold(arrays["y_link_active"][val_idx], val_pred["active_prob"])
    threshold = best_threshold["threshold"]
    rows = []
    for split, idx, pred in [("val_seed_3", val_idx, val_pred), ("test_seed_4", test_idx, test_pred)]:
        rows.append(
            {
                "split": split,
                "model": f"world_model_v4_{variant}",
                "physical_variant": variant,
                "torch_seed": int(torch_seed),
                "threshold": float(threshold),
                **evaluate(arrays, idx, pred, threshold),
            }
        )
    return {
        "metrics": rows,
        "history": history.assign(physical_variant=variant, torch_seed=torch_seed),
        "threshold_curve": threshold_df.assign(physical_variant=variant, torch_seed=torch_seed),
        "train_info": train_info,
        "selected_threshold": best_threshold,
    }


def plot_stability(summary_df):
    import matplotlib.pyplot as plt

    path = FIGURE_DIR / "world_model_v4_seed_stability_summary.png"
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6))
    labels = summary_df["physical_variant"]
    axes[0].bar(labels, summary_df["activity_f1_mean"], yerr=summary_df["activity_f1_std"], color="#d62728")
    axes[0].set_title("activity F1 mean/std")
    axes[1].bar(labels, summary_df["rate_all_rmse_mean"], yerr=summary_df["rate_all_rmse_std"], color="#1f77b4")
    axes[1].set_title("link RMSE mean/std")
    axes[2].bar(labels, summary_df["task_rmse_mean"], yerr=summary_df["task_rmse_std"], color="#2ca02c")
    axes[2].set_title("task RMSE mean/std")
    for ax in axes:
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()
    return path


def write_report(summary, metrics_df, stability_df):
    test = metrics_df[metrics_df["split"] == "test_seed_4"].copy()
    lines = [
        "# World model v4 seed-stability report",
        "",
        "## Goal",
        "",
        "This experiment checks whether the v4 physical branch is stable across model-initialization seeds. It keeps the dataset seed split fixed and repeats training for selected physical variants.",
        "",
        "## Per-run Test Metrics",
        "",
        test.to_markdown(index=False),
        "",
        "## Stability Summary",
        "",
        stability_df.to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        "- Large standard deviation means the current training/threshold pipeline is not stable enough for a strong method claim.",
        "- Compare `dual_full` against `no_physical` by both mean value and variance, not by a single run.",
        "",
        "## Outputs",
        "",
    ]
    for key, value in summary["outputs"].items():
        lines.append(f"- {key}: `{value}`")
    path = OUTPUT_DIR / "world_model_v4_seed_stability_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    base_arrays = augment_arrays_with_physical_edges(load_dataset())
    train_idx, val_idx, test_idx = split_by_seed(base_arrays["sample_seed"])

    results = []
    for variant in DEFAULT_VARIANTS:
        for torch_seed in DEFAULT_TORCH_SEEDS:
            print(f"[v4-stability] variant={variant} torch_seed={torch_seed}", flush=True)
            results.append(run_one(base_arrays, train_idx, val_idx, test_idx, variant, torch_seed))

    metrics_df = pd.DataFrame([row for result in results for row in result["metrics"]])
    history_df = pd.concat([result["history"] for result in results], ignore_index=True)
    threshold_df = pd.concat([result["threshold_curve"] for result in results], ignore_index=True)
    stability_df = summarize_stability(metrics_df)

    metrics_path = OUTPUT_DIR / "world_model_v4_seed_stability_metrics.csv"
    history_path = OUTPUT_DIR / "world_model_v4_seed_stability_training_history.csv"
    threshold_path = OUTPUT_DIR / "world_model_v4_seed_stability_threshold_curve.csv"
    stability_path = OUTPUT_DIR / "world_model_v4_seed_stability_summary.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    history_df.to_csv(history_path, index=False, encoding="utf-8-sig")
    threshold_df.to_csv(threshold_path, index=False, encoding="utf-8-sig")
    stability_df.to_csv(stability_path, index=False, encoding="utf-8-sig")
    plot_path = plot_stability(stability_df)

    summary = {
        "output_dir": display_path(OUTPUT_DIR),
        "variants": DEFAULT_VARIANTS,
        "torch_seeds": DEFAULT_TORCH_SEEDS,
        "split": {
            "train_seeds": [0, 1, 2],
            "val_seed": 3,
            "test_seed": 4,
            "train_samples": int(len(train_idx)),
            "val_samples": int(len(val_idx)),
            "test_samples": int(len(test_idx)),
        },
        "outputs": {
            "metrics_csv": display_path(metrics_path),
            "training_history_csv": display_path(history_path),
            "threshold_curve_csv": display_path(threshold_path),
            "stability_summary_csv": display_path(stability_path),
            "summary_plot": display_path(plot_path),
        },
    }
    report_path = write_report(summary, metrics_df, stability_df)
    summary["outputs"]["report_md"] = display_path(report_path)
    summary_path = OUTPUT_DIR / "world_model_v4_seed_stability_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
