import json

import numpy as np
import pandas as pd

from run_world_model_v0 import FIGURE_DIR, ROOT, choose_threshold, load_dataset, split_by_seed
from run_world_model_v3_graph_rollout import evaluate
from run_world_model_v4_dual_graph_rollout import (
    PHYSICAL_EDGE_FEATURES,
    augment_arrays_with_physical_edges,
    display_path,
    make_stats,
    predict,
    train_model,
)


OUTPUT_DIR = ROOT / "reports" / "world_model_v4_dual_graph_ablation"
VARIANTS = {
    "dual_full": None,
    "no_physical": [],
    "distance_only": ["distance_3d"],
    "distance_height_speed": ["distance_3d", "abs_speed_delta", "abs_dz"],
}


def make_physical_variant(arrays, variant):
    if variant not in VARIANTS:
        raise ValueError(f"unknown physical variant: {variant}")
    result = dict(arrays)
    physical = np.array(arrays["x_phy_edge"], copy=True)
    keep_names = VARIANTS[variant]
    if keep_names is not None:
        mask = np.zeros(physical.shape[-1], dtype=np.float32)
        for name in keep_names:
            mask[PHYSICAL_EDGE_FEATURES.index(name)] = 1.0
        physical = physical * mask.reshape(1, 1, 1, -1)
    result["x_phy_edge"] = physical.astype(np.float32)
    result["physical_variant"] = np.asarray([variant], dtype=object)
    result["physical_variant_features"] = np.asarray(
        PHYSICAL_EDGE_FEATURES if keep_names is None else keep_names,
        dtype=object,
    )
    return result


def run_variant(base_arrays, train_idx, val_idx, test_idx, variant):
    arrays = make_physical_variant(base_arrays, variant)
    stats = make_stats(arrays, train_idx)
    model, history, train_info = train_model(arrays, train_idx, val_idx, stats)
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
                "threshold": float(threshold),
                **evaluate(arrays, idx, pred, threshold),
            }
        )
    return {
        "variant": variant,
        "metrics": rows,
        "history": history.assign(physical_variant=variant),
        "threshold_curve": threshold_df.assign(physical_variant=variant),
        "train_info": train_info,
        "selected_threshold": best_threshold,
    }


def plot_ablation(metrics_df):
    import matplotlib.pyplot as plt

    path = FIGURE_DIR / "world_model_v4_dual_graph_ablation_compare.png"
    test = metrics_df[metrics_df["split"] == "test_seed_4"].copy()
    test["label"] = test["physical_variant"]
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.6))
    axes[0].bar(test["label"], test["activity_f1"], color="#d62728")
    axes[0].set_title("activity F1")
    axes[1].bar(test["label"], test["rate_all_rmse"], color="#1f77b4")
    axes[1].set_title("link RMSE")
    axes[2].bar(test["label"], test["task_rmse"], color="#2ca02c")
    axes[2].set_title("task RMSE")
    for ax in axes:
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()
    return path


def write_report(summary, metrics_df):
    test = metrics_df[metrics_df["split"] == "test_seed_4"].copy()
    best_activity = test.sort_values("activity_f1", ascending=False).iloc[0]
    best_task = test.sort_values("task_rmse", ascending=True).iloc[0]
    lines = [
        "# World model v4 dual-graph ablation report",
        "",
        "## Goal",
        "",
        "This experiment checks which part of the physical-edge branch contributes to the v4 improvement. All variants reuse the same `world_model_dataset_v0`, seed split, candidate communication edges, edge actions, and training schedule.",
        "",
        "## Variants",
        "",
        "- `dual_full`: all physical-edge features.",
        "- `no_physical`: the physical-edge tensor is zeroed while the architecture is kept unchanged.",
        "- `distance_only`: keeps only `distance_3d`.",
        "- `distance_height_speed`: keeps `distance_3d`, `abs_speed_delta`, and `abs_dz`.",
        "",
        "## Test-seed-4 summary",
        "",
        test.to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        f"- Best activity F1: `{best_activity['physical_variant']}` with `{best_activity['activity_f1']:.6f}`.",
        f"- Best task RMSE: `{best_task['physical_variant']}` with `{best_task['task_rmse']:.6f}`.",
        "- Compare `dual_full` against `no_physical` to decide whether the dual-graph gain comes from endpoint geometry instead of only extra parameters.",
        "",
        "## Outputs",
        "",
    ]
    for key, value in summary["outputs"].items():
        lines.append(f"- {key}: `{value}`")
    path = OUTPUT_DIR / "world_model_v4_dual_graph_ablation_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    base_arrays = augment_arrays_with_physical_edges(load_dataset())
    train_idx, val_idx, test_idx = split_by_seed(base_arrays["sample_seed"])

    results = []
    for variant in VARIANTS:
        print(f"[v4-ablation] running {variant}", flush=True)
        results.append(run_variant(base_arrays, train_idx, val_idx, test_idx, variant))

    metrics_df = pd.DataFrame([row for result in results for row in result["metrics"]])
    history_df = pd.concat([result["history"] for result in results], ignore_index=True)
    threshold_df = pd.concat([result["threshold_curve"] for result in results], ignore_index=True)

    metrics_path = OUTPUT_DIR / "world_model_v4_dual_graph_ablation_metrics.csv"
    history_path = OUTPUT_DIR / "world_model_v4_dual_graph_ablation_training_history.csv"
    threshold_path = OUTPUT_DIR / "world_model_v4_dual_graph_ablation_threshold_curve.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    history_df.to_csv(history_path, index=False, encoding="utf-8-sig")
    threshold_df.to_csv(threshold_path, index=False, encoding="utf-8-sig")
    plot_path = plot_ablation(metrics_df)

    summary = {
        "output_dir": display_path(OUTPUT_DIR),
        "variants": {name: features for name, features in VARIANTS.items()},
        "physical_edge_features": PHYSICAL_EDGE_FEATURES,
        "split": {
            "train_seeds": [0, 1, 2],
            "val_seed": 3,
            "test_seed": 4,
            "train_samples": int(len(train_idx)),
            "val_samples": int(len(val_idx)),
            "test_samples": int(len(test_idx)),
        },
        "train_info": {result["variant"]: result["train_info"] for result in results},
        "selected_thresholds": {result["variant"]: result["selected_threshold"] for result in results},
        "outputs": {
            "metrics_csv": display_path(metrics_path),
            "training_history_csv": display_path(history_path),
            "threshold_curve_csv": display_path(threshold_path),
            "compare_plot": display_path(plot_path),
        },
    }
    report_path = write_report(summary, metrics_df)
    summary["outputs"]["report_md"] = display_path(report_path)
    summary_path = OUTPUT_DIR / "world_model_v4_dual_graph_ablation_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
