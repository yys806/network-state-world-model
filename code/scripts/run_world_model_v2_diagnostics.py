import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, f1_score, precision_score, recall_score

from run_world_model_v0 import FIGURE_DIR, ROOT, activity_metrics, split_by_seed
from run_world_model_v2_latent_rollout import (
    evaluate,
    load_dataset,
    make_stats,
    predict,
    regression_metrics,
    train_model,
)


OUTPUT_DIR = ROOT / "reports" / "world_model_v2_diagnostics"
NOISE_LEVELS = [0.0, 0.05, 0.10, 0.20, 0.30]
INTERVAL_LEVELS = [0.80, 0.90]
SEED = 20260517


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
    for target, y_key, pred_key in [
        ("link_rate", "y_link_rate", "rate_pred"),
        ("task_state", "y_task", "task_pred"),
    ]:
        val_abs = np.abs(arrays[y_key][val_idx] - val_pred[pred_key])
        y_test = arrays[y_key][test_idx]
        p_test = test_pred[pred_key]
        active_mask = arrays["y_link_active"][test_idx] > 0.5 if target == "link_rate" else None
        for level in INTERVAL_LEVELS:
            width = np.quantile(val_abs, level, axis=0, keepdims=True)
            lower = np.maximum(p_test - width, 0.0)
            upper = p_test + width
            covered = (y_test >= lower) & (y_test <= upper)
            rows.append(
                {
                    "target": target,
                    "subset": "all",
                    "interval": f"{int(level * 100)}%",
                    "coverage": float(np.mean(covered)),
                    "mean_width": float(np.mean(upper - lower)),
                }
            )
            if active_mask is not None and active_mask.any():
                rows.append(
                    {
                        "target": target,
                        "subset": "active_edges",
                        "interval": f"{int(level * 100)}%",
                        "coverage": float(np.mean(covered[active_mask])),
                        "mean_width": float(np.mean((upper - lower)[active_mask])),
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
    plt.title("World model v2 link-activity calibration")
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
    fig.suptitle("World model v2 robustness under input perturbations")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_intervals(arrays, test_idx, test_pred, val_abs, path):
    feature_names = [str(x) for x in arrays["task_features"].tolist()]
    feat_idx = feature_names.index("num_finished")
    step = 2
    width90 = np.quantile(val_abs, 0.90, axis=0)[step, feat_idx]
    y = arrays["y_task"][test_idx, step, feat_idx]
    p = test_pred["task_pred"][:, step, feat_idx]
    x = np.arange(len(y))
    plt.figure(figsize=(10, 4.3))
    plt.plot(x, y, label="true", color="#111827", lw=1.8)
    plt.plot(x, p, label="prediction", color="#2563eb", lw=1.5)
    plt.fill_between(x, np.maximum(p - width90, 0.0), p + width90, color="#60a5fa", alpha=0.25, label="90% interval")
    plt.xlabel("test sample index")
    plt.ylabel("num_finished")
    plt.title("World model v2 task-state prediction interval: num_finished, t+3")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def write_report(summary, threshold_df, calibration_summary_df, interval_df, robustness_df):
    lines = [
        "# World model v2 diagnostics report",
        "",
        "## Goal",
        "",
        "This report evaluates the v2 latent-rollout world model from three angles: link-activity threshold transfer, residual-quantile prediction intervals, and robustness under synthetic input perturbations.",
        "",
        "## Literature-backed rationale",
        "",
        "- Validation-residual quantile intervals are a lightweight conformal-style uncertainty estimate, following the split-conformal idea summarized by Angelopoulos and Bates (2021).",
        "- Activity probability calibration is checked because the previous v2 result had high AP/AUC but low F1, meaning ranking and thresholding behave differently under cross-seed transfer.",
        "- Perturbation testing follows the same clean-train/noisy-test setting used earlier in this project, and is used as a stress test rather than a claim of real-world channel noise modeling.",
        "",
        "## Threshold transfer",
        "",
        threshold_df.to_markdown(index=False, floatfmt=".4f"),
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
        "- v2 ranks active links well, but the validation-selected threshold does not transfer cleanly to the held-out seed.",
        "- Regression intervals can be computed from validation residuals, but the active-edge rate interval remains wide because active links are rare and high-valued.",
        "- Under synthetic input perturbations, v2 diagnostics provide a concrete robustness curve; this is still evaluation-only, not yet perturbation-trained robustness.",
        "",
        "## Outputs",
        "",
    ]
    for key, value in summary["outputs"].items():
        lines.append(f"- {key}: `{value}`")
    path = OUTPUT_DIR / "world_model_v2_diagnostics_report.md"
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

    selected_threshold, _ = activity_metrics, None
    from run_world_model_v0 import choose_threshold

    best_threshold, threshold_curve = choose_threshold(arrays["y_link_active"][val_idx], val_pred["active_prob"])
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
    val_cal = pd.DataFrame(calibration_rows(arrays["y_link_active"][val_idx], val_pred["active_prob"], "val_seed_3"))
    test_cal = pd.DataFrame(calibration_rows(arrays["y_link_active"][test_idx], test_pred["active_prob"], "test_seed_4"))
    calibration_df = pd.concat([val_cal, test_cal], ignore_index=True)
    calibration_summary_df = pd.DataFrame(
        [
            {
                "split": "val_seed_3",
                "brier": float(brier_score_loss(arrays["y_link_active"][val_idx].reshape(-1), val_pred["active_prob"].reshape(-1))),
                "ece_quantile_bins": expected_calibration_error(val_cal),
            },
            {
                "split": "test_seed_4",
                "brier": float(brier_score_loss(arrays["y_link_active"][test_idx].reshape(-1), test_pred["active_prob"].reshape(-1))),
                "ece_quantile_bins": expected_calibration_error(test_cal),
            },
        ]
    )
    interval_df = pd.DataFrame(interval_rows(arrays, val_idx, test_idx, val_pred, test_pred))
    robustness_df = pd.DataFrame(robustness_rows(model, arrays, test_idx, stats, threshold))

    threshold_path = OUTPUT_DIR / "world_model_v2_threshold_transfer.csv"
    calibration_path = OUTPUT_DIR / "world_model_v2_activity_calibration.csv"
    calibration_summary_path = OUTPUT_DIR / "world_model_v2_activity_calibration_summary.csv"
    interval_path = OUTPUT_DIR / "world_model_v2_prediction_intervals.csv"
    robustness_path = OUTPUT_DIR / "world_model_v2_robustness_metrics.csv"
    threshold_df.to_csv(threshold_path, index=False, encoding="utf-8-sig")
    calibration_df.to_csv(calibration_path, index=False, encoding="utf-8-sig")
    calibration_summary_df.to_csv(calibration_summary_path, index=False, encoding="utf-8-sig")
    interval_df.to_csv(interval_path, index=False, encoding="utf-8-sig")
    robustness_df.to_csv(robustness_path, index=False, encoding="utf-8-sig")

    calibration_plot = FIGURE_DIR / "world_model_v2_activity_calibration.png"
    robustness_plot = FIGURE_DIR / "world_model_v2_robustness_noise_vs_error.png"
    interval_plot = FIGURE_DIR / "world_model_v2_prediction_interval_num_finished.png"
    plot_calibration(calibration_df, calibration_plot)
    plot_robustness(robustness_df, robustness_plot)
    val_task_abs = np.abs(arrays["y_task"][val_idx] - val_pred["task_pred"])
    plot_intervals(arrays, test_idx, test_pred, val_task_abs, interval_plot)

    summary = {
        "train_info": train_info,
        "selected_threshold": best_threshold,
        "outputs": {
            "threshold_transfer_csv": str(threshold_path),
            "activity_calibration_csv": str(calibration_path),
            "activity_calibration_summary_csv": str(calibration_summary_path),
            "prediction_intervals_csv": str(interval_path),
            "robustness_metrics_csv": str(robustness_path),
            "activity_calibration_plot": str(calibration_plot),
            "robustness_plot": str(robustness_plot),
            "interval_plot": str(interval_plot),
        },
    }
    report_path = write_report(summary, threshold_df, calibration_summary_df, interval_df, robustness_df)
    summary["outputs"]["report_md"] = str(report_path)
    summary_path = OUTPUT_DIR / "world_model_v2_diagnostics_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
