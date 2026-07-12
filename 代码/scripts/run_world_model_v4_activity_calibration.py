import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

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


OUTPUT_DIR = ROOT / "reports" / "world_model_v4_activity_calibration"
CACHE_DIR = OUTPUT_DIR / "prediction_cache"
DEFAULT_TORCH_SEEDS = [11, 42, 73]
DEFAULT_VARIANT = "dual_full"
TEMPERATURE_GRID = np.array([0.4, 0.5, 0.65, 0.8, 1.0, 1.25, 1.5, 2.0, 2.75, 3.5, 5.0], dtype=np.float32)


def safe_logit(prob, eps=1e-6):
    prob = np.asarray(prob, dtype=np.float64)
    prob = np.clip(prob, eps, 1.0 - eps)
    return np.log(prob / (1.0 - prob))


def apply_temperature(prob, temperature):
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    logits = safe_logit(prob)
    return (1.0 / (1.0 + np.exp(-logits / float(temperature)))).astype(np.float32)


def binary_cross_entropy(y_true, prob):
    y = np.asarray(y_true, dtype=np.float64).reshape(-1)
    p = np.clip(np.asarray(prob, dtype=np.float64).reshape(-1), 1e-6, 1.0 - 1e-6)
    return float(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)).mean())


def fit_temperature_grid(y_val, prob_val, temperatures=TEMPERATURE_GRID):
    rows = []
    best = None
    for temperature in temperatures:
        calibrated = apply_temperature(prob_val, float(temperature))
        row = {
            "temperature": float(temperature),
            "val_bce": binary_cross_entropy(y_val, calibrated),
        }
        rows.append(row)
        if best is None or row["val_bce"] < best["val_bce"]:
            best = row
    return best, pd.DataFrame(rows)


def threshold_metrics(y_true, prob, threshold):
    y_flat = np.asarray(y_true).reshape(-1).astype(int)
    p_flat = np.asarray(prob).reshape(-1)
    pred = p_flat >= float(threshold)
    out = {
        "threshold": float(threshold),
        "precision": float(precision_score(y_flat, pred, zero_division=0)),
        "recall": float(recall_score(y_flat, pred, zero_division=0)),
        "f1": float(f1_score(y_flat, pred, zero_division=0)),
        "predicted_active_ratio": float(pred.mean()),
        "predicted_active_count": int(pred.sum()),
        "true_active_ratio": float(y_flat.mean()),
        "true_active_count": int(y_flat.sum()),
    }
    if y_flat.min() != y_flat.max():
        out["average_precision"] = float(average_precision_score(y_flat, p_flat))
        out["roc_auc"] = float(roc_auc_score(y_flat, p_flat))
    else:
        out["average_precision"] = float("nan")
        out["roc_auc"] = float("nan")
    return out


def threshold_candidates(prob):
    p_flat = np.asarray(prob).reshape(-1)
    quantiles = np.quantile(p_flat, np.linspace(0.80, 0.9999, 80))
    return np.unique(np.concatenate([np.linspace(0.01, 0.99, 99), quantiles]))


def select_ratio_threshold(y_val, prob_val):
    y_flat = np.asarray(y_val).reshape(-1).astype(int)
    p_flat = np.asarray(prob_val).reshape(-1)
    target_count = int(y_flat.sum())
    if target_count <= 0:
        threshold = float(np.max(p_flat) + 1e-6)
    elif target_count >= p_flat.size:
        threshold = float(np.min(p_flat) - 1e-6)
    else:
        threshold = float(np.partition(p_flat, p_flat.size - target_count)[p_flat.size - target_count])
    metrics = threshold_metrics(y_flat, p_flat, threshold)
    metrics["strategy_note"] = "match_validation_active_ratio"
    return metrics


def select_precision_constrained_threshold(y_val, prob_val, min_precision=0.8):
    best_feasible = None
    best_any = None
    for threshold in threshold_candidates(prob_val):
        row = threshold_metrics(y_val, prob_val, threshold)
        if best_any is None or (row["precision"], row["f1"], row["recall"]) > (
            best_any["precision"],
            best_any["f1"],
            best_any["recall"],
        ):
            best_any = row
        if row["precision"] >= min_precision and row["predicted_active_count"] > 0:
            if best_feasible is None or (row["recall"], row["f1"]) > (
                best_feasible["recall"],
                best_feasible["f1"],
            ):
                best_feasible = row
    selected = dict(best_feasible if best_feasible is not None else best_any)
    selected["strategy_note"] = (
        f"precision_at_least_{min_precision:.2f}" if best_feasible is not None else "best_precision_fallback"
    )
    return selected


def select_f1_threshold(y_val, prob_val):
    best, _ = choose_threshold(y_val, prob_val)
    out = threshold_metrics(y_val, prob_val, best["threshold"])
    out["strategy_note"] = "best_validation_f1"
    return out


def prediction_cache_path(variant, torch_seed):
    return CACHE_DIR / f"v4_activity_{variant}_seed_{int(torch_seed):03d}_predictions.npz"


def load_or_train_predictions(base_arrays, train_idx, val_idx, test_idx, variant, torch_seed):
    cache_path = prediction_cache_path(variant, torch_seed)
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=True) as data:
            val_pred = {
                "active_prob": data["val_active_prob"],
                "rate_pred": data["val_rate_pred"],
                "task_pred": data["val_task_pred"],
            }
            test_pred = {
                "active_prob": data["test_active_prob"],
                "rate_pred": data["test_rate_pred"],
                "task_pred": data["test_task_pred"],
            }
            train_info = json.loads(str(data["train_info_json"]))
            history = pd.DataFrame(json.loads(str(data["history_json"])))
        return val_pred, test_pred, history, train_info, True

    arrays = make_physical_variant(base_arrays, variant)
    stats = make_stats(arrays, train_idx)
    model, history, train_info = train_model(arrays, train_idx, val_idx, stats, torch_seed=torch_seed)
    val_pred = predict(model, arrays, val_idx, stats)
    test_pred = predict(model, arrays, test_idx, stats)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        val_active_prob=val_pred["active_prob"],
        val_rate_pred=val_pred["rate_pred"],
        val_task_pred=val_pred["task_pred"],
        test_active_prob=test_pred["active_prob"],
        test_rate_pred=test_pred["rate_pred"],
        test_task_pred=test_pred["task_pred"],
        history_json=history.to_json(orient="records", force_ascii=False),
        train_info_json=json.dumps(train_info, ensure_ascii=False),
    )
    return val_pred, test_pred, history, train_info, False


def strategy_table_for_seed(arrays, val_idx, test_idx, val_pred, test_pred, torch_seed):
    y_val = arrays["y_link_active"][val_idx]
    y_test = arrays["y_link_active"][test_idx]
    best_temp, temp_curve = fit_temperature_grid(y_val, val_pred["active_prob"])
    temp_val_prob = apply_temperature(val_pred["active_prob"], best_temp["temperature"])
    temp_test_prob = apply_temperature(test_pred["active_prob"], best_temp["temperature"])

    strategy_specs = [
        ("fixed_0.50_raw", val_pred["active_prob"], test_pred["active_prob"], {"threshold": 0.5}),
        ("val_f1_raw", val_pred["active_prob"], test_pred["active_prob"], select_f1_threshold(y_val, val_pred["active_prob"])),
        (
            "val_ratio_raw",
            val_pred["active_prob"],
            test_pred["active_prob"],
            select_ratio_threshold(y_val, val_pred["active_prob"]),
        ),
        (
            "precision_0.80_raw",
            val_pred["active_prob"],
            test_pred["active_prob"],
            select_precision_constrained_threshold(y_val, val_pred["active_prob"], min_precision=0.8),
        ),
        (
            "val_f1_temp",
            temp_val_prob,
            temp_test_prob,
            select_f1_threshold(y_val, temp_val_prob),
        ),
        (
            "val_ratio_temp",
            temp_val_prob,
            temp_test_prob,
            select_ratio_threshold(y_val, temp_val_prob),
        ),
    ]

    rows = []
    for name, val_prob, test_prob, selected in strategy_specs:
        threshold = float(selected["threshold"])
        val_metrics = threshold_metrics(y_val, val_prob, threshold)
        test_metrics = threshold_metrics(y_test, test_prob, threshold)
        for split, metrics in [("val_seed_3", val_metrics), ("test_seed_4", test_metrics)]:
            row = {
                "torch_seed": int(torch_seed),
                "strategy": name,
                "split": split,
                "threshold": threshold,
                "temperature": float(best_temp["temperature"]) if name.endswith("_temp") else 1.0,
                "val_selected_precision": float(selected.get("precision", np.nan)),
                "val_selected_recall": float(selected.get("recall", np.nan)),
                "val_selected_f1": float(selected.get("f1", np.nan)),
                "strategy_note": selected.get("strategy_note", ""),
                **metrics,
            }
            rows.append(row)
    return pd.DataFrame(rows), temp_curve.assign(torch_seed=int(torch_seed))


def plot_activity_calibration(strategy_df):
    import matplotlib.pyplot as plt

    path = FIGURE_DIR / "world_model_v4_activity_calibration_summary.png"
    test = strategy_df[strategy_df["split"] == "test_seed_4"].copy()
    pivot_f1 = test.pivot(index="strategy", columns="torch_seed", values="f1").sort_index()
    pivot_precision = test.pivot(index="strategy", columns="torch_seed", values="precision").reindex(pivot_f1.index)
    pivot_ratio = test.pivot(index="strategy", columns="torch_seed", values="predicted_active_ratio").reindex(pivot_f1.index)

    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.4))
    pivot_f1.plot(kind="bar", ax=axes[0])
    axes[0].set_title("test activity F1")
    pivot_precision.plot(kind="bar", ax=axes[1])
    axes[1].set_title("test precision")
    pivot_ratio.plot(kind="bar", ax=axes[2])
    axes[2].set_title("predicted active ratio")
    for ax in axes:
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(title="torch seed", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()
    return path


def summarize_strategies(strategy_df):
    test = strategy_df[strategy_df["split"] == "test_seed_4"].copy()
    rows = []
    for strategy, part in test.groupby("strategy", dropna=False):
        row = {"strategy": strategy, "runs": int(len(part))}
        for col in ["precision", "recall", "f1", "predicted_active_ratio"]:
            row[f"{col}_mean"] = float(part[col].mean())
            row[f"{col}_std"] = float(part[col].std(ddof=1)) if len(part) > 1 else 0.0
            row[f"{col}_min"] = float(part[col].min())
            row[f"{col}_max"] = float(part[col].max())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("strategy").reset_index(drop=True)


def write_report(summary, strategy_df, strategy_summary):
    test = strategy_df[strategy_df["split"] == "test_seed_4"].copy()
    best_mean = strategy_summary.sort_values(["f1_mean", "precision_mean"], ascending=False).iloc[0]
    lines = [
        "# World model v4 activity-calibration report",
        "",
        "## Goal",
        "",
        "This experiment isolates the activity decision layer after v4 dual-graph training. It reuses the same model probabilities and compares threshold-transfer and probability-temperature strategies on validation seed 3 and test seed 4.",
        "",
        "## Test Metrics by Run",
        "",
        test[
            [
                "torch_seed",
                "strategy",
                "threshold",
                "temperature",
                "precision",
                "recall",
                "f1",
                "predicted_active_ratio",
                "true_active_ratio",
            ]
        ].to_markdown(index=False),
        "",
        "## Strategy Stability Summary",
        "",
        strategy_summary.to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        f"- The best mean-F1 strategy in this run is `{best_mean['strategy']}` with mean test F1 `{best_mean['f1_mean']:.6f}` over {int(best_mean['runs'])} torch seeds.",
        "- Raw validation-F1 thresholding can select a threshold that transfers poorly when the predicted active ratio changes across seeds.",
        "- Ratio matching is useful for controlling active-edge sparsity, while precision-constrained thresholding is useful when false active links are more harmful than missed links.",
        "- Temperature scaling changes probability confidence but preserves ranking; it mainly supports threshold interpretability and does not replace activity-threshold validation.",
        "",
        "## Outputs",
        "",
    ]
    for key, value in summary["outputs"].items():
        lines.append(f"- {key}: `{value}`")
    path = OUTPUT_DIR / "world_model_v4_activity_calibration_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    base_arrays = augment_arrays_with_physical_edges(load_dataset())
    arrays = make_physical_variant(base_arrays, DEFAULT_VARIANT)
    train_idx, val_idx, test_idx = split_by_seed(arrays["sample_seed"])

    strategy_parts = []
    temp_parts = []
    histories = []
    cache_status = []
    for torch_seed in DEFAULT_TORCH_SEEDS:
        print(f"[v4-activity-calibration] variant={DEFAULT_VARIANT} torch_seed={torch_seed}", flush=True)
        val_pred, test_pred, history, train_info, cache_hit = load_or_train_predictions(
            base_arrays, train_idx, val_idx, test_idx, DEFAULT_VARIANT, torch_seed
        )
        strategy_df, temp_curve = strategy_table_for_seed(arrays, val_idx, test_idx, val_pred, test_pred, torch_seed)
        strategy_parts.append(strategy_df)
        temp_parts.append(temp_curve)
        histories.append(history.assign(torch_seed=int(torch_seed)))
        cache_status.append(
            {
                "torch_seed": int(torch_seed),
                "cache_hit": bool(cache_hit),
                "cache_path": display_path(prediction_cache_path(DEFAULT_VARIANT, torch_seed)),
                "train_info": train_info,
            }
        )

    strategy_df = pd.concat(strategy_parts, ignore_index=True)
    temp_curve_df = pd.concat(temp_parts, ignore_index=True)
    history_df = pd.concat(histories, ignore_index=True)
    strategy_summary = summarize_strategies(strategy_df)

    strategy_path = OUTPUT_DIR / "world_model_v4_activity_calibration_metrics.csv"
    temp_path = OUTPUT_DIR / "world_model_v4_activity_temperature_curve.csv"
    history_path = OUTPUT_DIR / "world_model_v4_activity_training_history.csv"
    summary_csv_path = OUTPUT_DIR / "world_model_v4_activity_calibration_summary.csv"
    strategy_df.to_csv(strategy_path, index=False, encoding="utf-8-sig")
    temp_curve_df.to_csv(temp_path, index=False, encoding="utf-8-sig")
    history_df.to_csv(history_path, index=False, encoding="utf-8-sig")
    strategy_summary.to_csv(summary_csv_path, index=False, encoding="utf-8-sig")
    plot_path = plot_activity_calibration(strategy_df)

    summary = {
        "output_dir": display_path(OUTPUT_DIR),
        "variant": DEFAULT_VARIANT,
        "torch_seeds": DEFAULT_TORCH_SEEDS,
        "split": {
            "train_seeds": [0, 1, 2],
            "val_seed": 3,
            "test_seed": 4,
            "train_samples": int(len(train_idx)),
            "val_samples": int(len(val_idx)),
            "test_samples": int(len(test_idx)),
        },
        "cache_status": cache_status,
        "outputs": {
            "metrics_csv": display_path(strategy_path),
            "temperature_curve_csv": display_path(temp_path),
            "training_history_csv": display_path(history_path),
            "strategy_summary_csv": display_path(summary_csv_path),
            "summary_plot": display_path(plot_path),
        },
    }
    report_path = write_report(summary, strategy_df, strategy_summary)
    summary["outputs"]["report_md"] = display_path(report_path)
    summary_path = OUTPUT_DIR / "world_model_v4_activity_calibration_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
