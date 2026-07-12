import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, f1_score, precision_score, recall_score

from run_world_model_v0 import FIGURE_DIR, ROOT, choose_threshold, split_by_seed
from run_world_model_v3_graph_rollout import evaluate, load_dataset, make_stats, predict, train_model


OUTPUT_DIR = ROOT / "reports" / "world_model_v3_diagnostics"
NOISE_LEVELS = [0.0, 0.05, 0.10, 0.20, 0.30]
INTERVAL_LEVELS = [0.80, 0.90]
SEED = 20260518


def noisy_arrays(arrays, level, seed):
    rng = np.random.default_rng(seed)
    out = {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in arrays.items()}
    if level <= 0:
        return out

    node_scale = np.array([15.0, 15.0, 3.0, 1.0, 0.5, 0.0, 0.0], dtype=np.float32) * level
    out["x_node"] = out["x_node"].astype(np.float32) + rng.normal(
        0.0, node_scale, size=out["x_node"].shape
    ).astype(np.float32)

    link = out["x_link"].astype(np.float32)
    link[..., 0] = np.maximum(0.0, link[..., 0] + rng.normal(0.0, 20.0 * level, size=link[..., 0].shape))
    link[..., 1] = np.maximum(0.0, link[..., 1] * (1.0 + rng.normal(0.0, 0.50 * level, size=link[..., 1].shape)))
    link[..., 2] = np.maximum(0.0, link[..., 2] * (1.0 + rng.normal(0.0, 0.30 * level, size=link[..., 2].shape)))
    link[..., 3] = np.maximum(0.0, link[..., 3] + rng.normal(0.0, 1.0 * level, size=link[..., 3].shape))
    link[..., 4] = np.maximum(0.0, link[..., 4] + rng.normal(0.0, 1.0 * level, size=link[..., 4].shape))
    out["x_link"] = link.astype(np.float32)

    task = out["x_task"].astype(np.float32)
    scale = np.maximum(np.std(task, axis=(0, 1), keepdims=True), 1.0)
    out["x_task"] = np.maximum(0.0, task + rng.normal(0.0, level, size=task.shape).astype(np.float32) * scale)
    return out


def threshold_rows(y_val, p_val, y_test, p_test, f1_threshold):
    yv = y_val.reshape(-1).astype(int)
    pv = p_val.reshape(-1)
    yt = y_test.reshape(-1).astype(int)
    pt = p_test.reshape(-1)
    active_ratio = float(yv.mean())
    prevalence_threshold = float(np.quantile(pv, max(0.0, 1.0 - active_ratio)))
    rows = []
    for name, threshold in [
        ("val_f1_threshold", float(f1_threshold)),
        ("val_prevalence_threshold", prevalence_threshold),
        ("fixed_0_5_threshold", 0.5),
    ]:
        for split, y, p in [("val_seed_3", yv, pv), ("test_seed_4", yt, pt)]:
            pred = p >= threshold
            rows.append(
                {
                    "split": split,
                    "threshold_rule": name,
                    "threshold": threshold,
                    "precision": float(precision_score(y, pred, zero_division=0)),
                    "recall": float(recall_score(y, pred, zero_division=0)),
                    "f1": float(f1_score(y, pred, zero_division=0)),
                    "predicted_active_ratio": float(pred.mean()),
                    "true_active_ratio": float(y.mean()),
                }
            )
    return rows


def threshold_by_link_type(edge_vocab, y_true, prob, threshold):
    rows = []
    for link_type, edge_part in edge_vocab.groupby("link_type"):
        edge_idx = edge_part["edge_index"].to_numpy(dtype=int)
        y = y_true[:, :, edge_idx].reshape(-1).astype(int)
        p = prob[:, :, edge_idx].reshape(-1)
        pred = p >= threshold
        rows.append(
            {
                "link_type": link_type,
                "threshold": float(threshold),
                "precision": float(precision_score(y, pred, zero_division=0)),
                "recall": float(recall_score(y, pred, zero_division=0)),
                "f1": float(f1_score(y, pred, zero_division=0)),
                "true_active_ratio": float(y.mean()),
                "predicted_active_ratio": float(pred.mean()),
            }
        )
    return rows


def calibration_rows(y_true, prob, split, n_bins=10):
    y = y_true.reshape(-1).astype(float)
    p = prob.reshape(-1).astype(float)
    quantiles = np.unique(np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1)))
    rows = []
    for i in range(len(quantiles) - 1):
        lo, hi = quantiles[i], quantiles[i + 1]
        mask = (p >= lo) & (p <= hi if i == len(quantiles) - 2 else p < hi)
        if not mask.any():
            continue
        rows.append(
            {
                "split": split,
                "bin": i,
                "prob_min": float(lo),
                "prob_max": float(hi),
                "mean_prob": float(p[mask].mean()),
                "empirical_active_ratio": float(y[mask].mean()),
                "count": int(mask.sum()),
            }
        )
    return rows


def expected_calibration_error(cal_df):
    total = cal_df["count"].sum()
    if total == 0:
        return float("nan")
    gap = np.abs(cal_df["mean_prob"] - cal_df["empirical_active_ratio"])
    return float((gap * cal_df["count"]).sum() / total)


def interval_rows(arrays, val_idx, test_idx, val_pred, test_pred):
    rows = []

    val_link_abs = np.abs(arrays["y_link_rate"][val_idx] - val_pred["rate_pred"])
    y_link_test = arrays["y_link_rate"][test_idx]
    p_link_test = test_pred["rate_pred"]
    test_active = arrays["y_link_active"][test_idx] > 0.5
    val_active = arrays["y_link_active"][val_idx] > 0.5

    for level in INTERVAL_LEVELS:
        width = np.quantile(val_link_abs, level, axis=0, keepdims=True)
        lower = np.maximum(p_link_test - width, 0.0)
        upper = p_link_test + width
        covered = (y_link_test >= lower) & (y_link_test <= upper)
        rows.append(
            {
                "target": "link_rate",
                "subset": "all_edges_axis_quantile",
                "interval": f"{int(level * 100)}%",
                "coverage": float(np.mean(covered)),
                "mean_width": float(np.mean(upper - lower)),
            }
        )
        if test_active.any():
            rows.append(
                {
                    "target": "link_rate",
                    "subset": "active_edges_axis_quantile",
                    "interval": f"{int(level * 100)}%",
                    "coverage": float(np.mean(covered[test_active])),
                    "mean_width": float(np.mean((upper - lower)[test_active])),
                }
            )
        if val_active.any() and test_active.any():
            active_width = float(np.quantile(val_link_abs[val_active], level))
            active_lower = np.maximum(p_link_test - active_width, 0.0)
            active_upper = p_link_test + active_width
            active_covered = (y_link_test >= active_lower) & (y_link_test <= active_upper)
            rows.append(
                {
                    "target": "link_rate",
                    "subset": "active_edges_scalar_active_quantile",
                    "interval": f"{int(level * 100)}%",
                    "coverage": float(np.mean(active_covered[test_active])),
                    "mean_width": float(2.0 * active_width),
                }
            )

    val_task_abs = np.abs(arrays["y_task"][val_idx] - val_pred["task_pred"])
    y_task_test = arrays["y_task"][test_idx]
    p_task_test = test_pred["task_pred"]
    for level in INTERVAL_LEVELS:
        width = np.quantile(val_task_abs, level, axis=0, keepdims=True)
        lower = np.maximum(p_task_test - width, 0.0)
        upper = p_task_test + width
        covered = (y_task_test >= lower) & (y_task_test <= upper)
        rows.append(
            {
                "target": "task_state",
                "subset": "all",
                "interval": f"{int(level * 100)}%",
                "coverage": float(np.mean(covered)),
                "mean_width": float(np.mean(upper - lower)),
            }
        )
    return rows


def robustness_rows(model, arrays, test_idx, stats, threshold):
    rows = []
    for level in NOISE_LEVELS:
        noisy = noisy_arrays(arrays, level, SEED + int(level * 1000))
        pred = predict(model, noisy, test_idx, stats)
        metrics = evaluate(arrays, test_idx, pred, threshold)
        rows.append({"noise_level": level, **metrics})
    return rows


def plot_calibration(cal_df, path):
    plt.figure(figsize=(5.2, 4.6))
    for split, part in cal_df.groupby("split"):
        plt.plot(part["mean_prob"], part["empirical_active_ratio"], marker="o", label=split)
    max_x = max(0.01, float(cal_df["mean_prob"].max()))
    plt.plot([0, max_x], [0, max_x], "--", color="#6b7280", label="perfect calibration")
    plt.xlabel("mean predicted probability")
    plt.ylabel("empirical active ratio")
    plt.title("World model v3 link-activity calibration")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def plot_robustness(df, path):
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8))
    for ax, metric, title in [
        (axes[0], "activity_f1", "Activity F1"),
        (axes[1], "rate_all_rmse", "Rate RMSE"),
        (axes[2], "task_rmse", "Task RMSE"),
    ]:
        ax.plot(df["noise_level"], df[metric], marker="o", color="#2563eb")
        ax.set_xlabel("noise level")
        ax.set_title(title)
        ax.grid(alpha=0.25)
    fig.suptitle("World model v3 robustness under input perturbations")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_active_rate_interval(arrays, test_idx, test_pred, active_width, path):
    y = arrays["y_link_rate"][test_idx]
    p = test_pred["rate_pred"]
    active = arrays["y_link_active"][test_idx] > 0.5
    if not active.any():
        return None
    y_active = y[active]
    p_active = p[active]
    order = np.argsort(y_active)
    x = np.arange(len(order))
    plt.figure(figsize=(10.0, 4.2))
    plt.plot(x, y_active[order], label="true active-edge rate", color="#111827", lw=1.8)
    plt.plot(x, p_active[order], label="prediction", color="#2563eb", lw=1.4)
    plt.fill_between(
        x,
        np.maximum(p_active[order] - active_width, 0.0),
        p_active[order] + active_width,
        color="#60a5fa",
        alpha=0.25,
        label="90% active-edge residual interval",
    )
    plt.xlabel("active edge item sorted by true rate")
    plt.ylabel("rate_sum")
    plt.title("World model v3 active-edge rate interval")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()
    return path


def write_report(summary, threshold_df, type_df, calibration_summary_df, interval_df, robustness_df):
    lines = [
        "# World model v3 diagnostics report",
        "",
        "## Goal",
        "",
        "This report evaluates the first graph-structured latent rollout baseline from three angles: threshold transfer, active-edge rate intervals, and robustness under synthetic input perturbations.",
        "",
        "## Threshold transfer",
        "",
        threshold_df.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Link-type threshold metrics on test seed 4",
        "",
        type_df.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Calibration summary",
        "",
        calibration_summary_df.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Prediction interval metrics",
        "",
        interval_df.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Robustness metrics",
        "",
        robustness_df.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Interpretation",
        "",
        "- v3 should be read as a graph-rollout baseline, not a final model.",
        "- The main diagnostic question is whether the improved active-edge F1 remains stable under threshold transfer and input perturbation.",
        "- Active-edge rate remains the bottleneck, so active-only residual intervals are reported separately from all-edge intervals.",
        "",
        "## Outputs",
        "",
    ]
    for key, value in summary["outputs"].items():
        lines.append(f"- {key}: `{value}`")
    path = OUTPUT_DIR / "world_model_v3_diagnostics_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    arrays = load_dataset()
    train_idx, val_idx, test_idx = split_by_seed(arrays["sample_seed"])
    stats = make_stats(arrays, train_idx)
    model, history, train_info = train_model(arrays, train_idx, val_idx, stats)
    val_pred = predict(model, arrays, val_idx, stats)
    test_pred = predict(model, arrays, test_idx, stats)
    best_threshold, _ = choose_threshold(arrays["y_link_active"][val_idx], val_pred["active_prob"])
    threshold = float(best_threshold["threshold"])

    threshold_df = pd.DataFrame(
        threshold_rows(
            arrays["y_link_active"][val_idx],
            val_pred["active_prob"],
            arrays["y_link_active"][test_idx],
            test_pred["active_prob"],
            threshold,
        )
    )
    edge_vocab = pd.read_csv(ROOT / "datasets" / "world_model_dataset_v0" / "edge_vocab.csv")
    type_df = pd.DataFrame(
        threshold_by_link_type(edge_vocab, arrays["y_link_active"][test_idx], test_pred["active_prob"], threshold)
    )

    val_cal = pd.DataFrame(calibration_rows(arrays["y_link_active"][val_idx], val_pred["active_prob"], "val_seed_3"))
    test_cal = pd.DataFrame(calibration_rows(arrays["y_link_active"][test_idx], test_pred["active_prob"], "test_seed_4"))
    calibration_df = pd.concat([val_cal, test_cal], ignore_index=True)
    calibration_summary_df = pd.DataFrame(
        [
            {
                "split": "val_seed_3",
                "brier": float(
                    brier_score_loss(
                        arrays["y_link_active"][val_idx].reshape(-1),
                        val_pred["active_prob"].reshape(-1),
                    )
                ),
                "ece_quantile_bins": expected_calibration_error(val_cal),
            },
            {
                "split": "test_seed_4",
                "brier": float(
                    brier_score_loss(
                        arrays["y_link_active"][test_idx].reshape(-1),
                        test_pred["active_prob"].reshape(-1),
                    )
                ),
                "ece_quantile_bins": expected_calibration_error(test_cal),
            },
        ]
    )

    interval_df = pd.DataFrame(interval_rows(arrays, val_idx, test_idx, val_pred, test_pred))
    robustness_df = pd.DataFrame(robustness_rows(model, arrays, test_idx, stats, threshold))

    threshold_path = OUTPUT_DIR / "world_model_v3_threshold_transfer.csv"
    type_path = OUTPUT_DIR / "world_model_v3_threshold_by_link_type.csv"
    calibration_path = OUTPUT_DIR / "world_model_v3_activity_calibration.csv"
    calibration_summary_path = OUTPUT_DIR / "world_model_v3_activity_calibration_summary.csv"
    interval_path = OUTPUT_DIR / "world_model_v3_prediction_intervals.csv"
    robustness_path = OUTPUT_DIR / "world_model_v3_robustness_metrics.csv"
    threshold_df.to_csv(threshold_path, index=False, encoding="utf-8-sig")
    type_df.to_csv(type_path, index=False, encoding="utf-8-sig")
    calibration_df.to_csv(calibration_path, index=False, encoding="utf-8-sig")
    calibration_summary_df.to_csv(calibration_summary_path, index=False, encoding="utf-8-sig")
    interval_df.to_csv(interval_path, index=False, encoding="utf-8-sig")
    robustness_df.to_csv(robustness_path, index=False, encoding="utf-8-sig")

    calibration_plot = FIGURE_DIR / "world_model_v3_activity_calibration.png"
    robustness_plot = FIGURE_DIR / "world_model_v3_robustness_noise_vs_error.png"
    active_interval_plot = FIGURE_DIR / "world_model_v3_active_rate_interval.png"
    plot_calibration(calibration_df, calibration_plot)
    plot_robustness(robustness_df, robustness_plot)
    val_link_abs = np.abs(arrays["y_link_rate"][val_idx] - val_pred["rate_pred"])
    val_active = arrays["y_link_active"][val_idx] > 0.5
    active_width90 = float(np.quantile(val_link_abs[val_active], 0.90)) if val_active.any() else float("nan")
    active_plot = plot_active_rate_interval(arrays, test_idx, test_pred, active_width90, active_interval_plot)

    summary = {
        "train_info": train_info,
        "selected_threshold": best_threshold,
        "active_edge_rate_width90": active_width90,
        "outputs": {
            "threshold_transfer_csv": str(threshold_path),
            "threshold_by_link_type_csv": str(type_path),
            "activity_calibration_csv": str(calibration_path),
            "activity_calibration_summary_csv": str(calibration_summary_path),
            "prediction_intervals_csv": str(interval_path),
            "robustness_metrics_csv": str(robustness_path),
            "activity_calibration_plot": str(calibration_plot),
            "robustness_plot": str(robustness_plot),
            "active_rate_interval_plot": str(active_plot) if active_plot else None,
        },
    }
    report_path = write_report(summary, threshold_df, type_df, calibration_summary_df, interval_df, robustness_df)
    summary["outputs"]["report_md"] = str(report_path)
    summary_path = OUTPUT_DIR / "world_model_v3_diagnostics_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
