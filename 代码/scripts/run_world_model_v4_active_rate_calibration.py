import json

import numpy as np
import pandas as pd

from run_world_model_v0 import FIGURE_DIR, ROOT, choose_threshold, load_dataset, split_by_seed
from run_world_model_v3_active_rate_calibration import (
    apply_activity_policy,
    build_rate_features,
    flatten_targets,
    interval_rows,
    metrics_row,
    predict_rate_ridge,
    train_rate_ridge,
)
from run_world_model_v4_dual_graph_rollout import (
    augment_arrays_with_physical_edges,
    display_path,
    make_stats,
    predict,
    train_model,
)


OUTPUT_DIR = ROOT / "reports" / "world_model_v4_active_rate_calibration"
PREDICTION_CACHE = OUTPUT_DIR / "world_model_v4_active_rate_predictions.npz"


def build_rate_features_with_physical(arrays, idx, pred=None):
    base = build_rate_features(arrays, idx, pred=None)
    x_phy_edge = arrays["x_phy_edge"][idx].astype(np.float32)
    num_samples, history, num_edges, feat_dim = x_phy_edge.shape
    horizon = arrays["edge_a_future"].shape[1]
    physical_hist = x_phy_edge.transpose(0, 2, 1, 3).reshape(num_samples, num_edges, history * feat_dim)
    physical_flat = np.broadcast_to(
        physical_hist[:, None, :, :],
        (num_samples, horizon, num_edges, history * feat_dim),
    ).reshape(num_samples * horizon * num_edges, history * feat_dim)
    parts = [base, physical_flat.astype(np.float32)]
    if pred is not None:
        prob = np.clip(pred["active_prob"].astype(np.float32).reshape(-1, 1), 1e-6, 1.0 - 1e-6)
        rate = np.clip(pred["rate_pred"].astype(np.float32).reshape(-1, 1), 0.0, None)
        parts.extend([prob, np.log(prob / (1.0 - prob)).astype(np.float32), rate, np.log1p(rate).astype(np.float32)])
    return np.concatenate(parts, axis=1).astype(np.float32)


def get_v4_predictions(arrays, train_idx, val_idx, test_idx, force_retrain=False):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if PREDICTION_CACHE.exists() and not force_retrain:
        data = np.load(PREDICTION_CACHE, allow_pickle=True)
        train_pred = {"active_prob": data["train_active_prob"], "rate_pred": data["train_rate_pred"]}
        val_pred = {"active_prob": data["val_active_prob"], "rate_pred": data["val_rate_pred"]}
        test_pred = {"active_prob": data["test_active_prob"], "rate_pred": data["test_rate_pred"]}
        return train_pred, val_pred, test_pred, {"source": "cache", "cache_file": display_path(PREDICTION_CACHE)}

    stats = make_stats(arrays, train_idx)
    model, history, train_info = train_model(arrays, train_idx, val_idx, stats)
    train_pred = predict(model, arrays, train_idx, stats)
    val_pred = predict(model, arrays, val_idx, stats)
    test_pred = predict(model, arrays, test_idx, stats)
    np.savez_compressed(
        PREDICTION_CACHE,
        train_active_prob=train_pred["active_prob"],
        train_rate_pred=train_pred["rate_pred"],
        val_active_prob=val_pred["active_prob"],
        val_rate_pred=val_pred["rate_pred"],
        test_active_prob=test_pred["active_prob"],
        test_rate_pred=test_pred["rate_pred"],
        train_info=json.dumps(train_info, ensure_ascii=False),
    )
    history.to_csv(OUTPUT_DIR / "world_model_v4_active_rate_training_history.csv", index=False, encoding="utf-8-sig")
    return train_pred, val_pred, test_pred, {"source": "trained", "train_info": train_info}


def plot_rmse_compare(metrics_df):
    import matplotlib.pyplot as plt

    path = FIGURE_DIR / "world_model_v4_active_rate_rmse_compare.png"
    test = metrics_df[
        (metrics_df["split"] == "test_seed_4")
        & (metrics_df["activity_policy"].isin(["oracle_activity", "v4_predicted_activity"]))
    ].copy()
    test["label"] = test["model"] + "\n" + test["activity_policy"]
    plt.figure(figsize=(9.5, 4.2))
    plt.bar(test["label"], test["active_rmse"], color="#1f77b4")
    plt.ylabel("RMSE on true active edges")
    plt.xticks(rotation=20, ha="right")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()
    return path


def write_report(summary, metrics_df, intervals_df, tuning_df):
    test = metrics_df[metrics_df["split"] == "test_seed_4"].copy()
    candidate = test[test["activity_policy"] == "v4_predicted_activity"].sort_values("active_rmse")
    best = candidate.iloc[0]
    oracle = test[(test["model"] == best["model"]) & (test["activity_policy"] == "oracle_activity")]
    oracle_rmse = float(oracle.iloc[0]["active_rmse"]) if len(oracle) else float("nan")
    lines = [
        "# World model v4 active-rate calibration report",
        "",
        "## Goal",
        "",
        "This experiment isolates the active-rate bottleneck after v4 dual-graph rollout. It compares the v4 rate head with active-only Ridge regressors under oracle activity masks and v4-predicted activity masks.",
        "",
        "## Metrics",
        "",
        test.to_markdown(index=False),
        "",
        "## Ridge Tuning",
        "",
        tuning_df.to_markdown(index=False),
        "",
        "## Residual-Quantile Intervals",
        "",
        intervals_df.to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        f"- Best v4-predicted-mask active-rate model: `{best['model']}` with active RMSE `{best['active_rmse']:.3f}`.",
        f"- Its oracle-mask active RMSE is `{oracle_rmse:.3f}`, which estimates the remaining rate-regression difficulty after activity masking is solved.",
        "- The gap between oracle and predicted masks shows how much activity errors still affect rate evaluation.",
        "",
        "## Outputs",
        "",
    ]
    for key, value in summary["outputs"].items():
        lines.append(f"- {key}: `{value}`")
    path = OUTPUT_DIR / "world_model_v4_active_rate_calibration_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    arrays = augment_arrays_with_physical_edges(load_dataset())
    train_idx, val_idx, test_idx = split_by_seed(arrays["sample_seed"])

    train_pred, val_pred, test_pred, pred_summary = get_v4_predictions(arrays, train_idx, val_idx, test_idx)
    best_threshold, threshold_df = choose_threshold(arrays["y_link_active"][val_idx], val_pred["active_prob"])
    threshold = best_threshold["threshold"]

    train_meta = flatten_targets(arrays, train_idx)
    val_meta = flatten_targets(arrays, val_idx)
    test_meta = flatten_targets(arrays, test_idx)
    train_meta["pred_active"] = train_pred["active_prob"].reshape(-1) >= threshold
    val_meta["pred_active"] = val_pred["active_prob"].reshape(-1) >= threshold
    test_meta["pred_active"] = test_pred["active_prob"].reshape(-1) >= threshold

    active_train = train_meta["y_active"]
    active_val = val_meta["y_active"]
    x_train = build_rate_features_with_physical(arrays, train_idx)
    x_val = build_rate_features_with_physical(arrays, val_idx)
    x_test = build_rate_features_with_physical(arrays, test_idx)
    x_train_aug = build_rate_features_with_physical(arrays, train_idx, train_pred)
    x_val_aug = build_rate_features_with_physical(arrays, val_idx, val_pred)
    x_test_aug = build_rate_features_with_physical(arrays, test_idx, test_pred)

    v4_train_rate = train_pred["rate_pred"].reshape(-1)
    v4_val_rate = val_pred["rate_pred"].reshape(-1)
    v4_test_rate = test_pred["rate_pred"].reshape(-1)
    models = [
        train_rate_ridge(
            "v4_physical_active_rate_ridge",
            x_train[active_train],
            train_meta["y_rate"][active_train],
            x_val[active_val],
            val_meta["y_rate"][active_val],
        ),
        train_rate_ridge(
            "v4_aug_active_rate_ridge",
            x_train_aug[active_train],
            train_meta["y_rate"][active_train],
            x_val_aug[active_val],
            val_meta["y_rate"][active_val],
        ),
        train_rate_ridge(
            "v4_aug_rate_log_ridge",
            x_train_aug[active_train],
            train_meta["y_rate"][active_train],
            x_val_aug[active_val],
            val_meta["y_rate"][active_val],
            target_mode="log",
        ),
        train_rate_ridge(
            "v4_residual_ridge",
            x_train_aug[active_train],
            train_meta["y_rate"][active_train],
            x_val_aug[active_val],
            val_meta["y_rate"][active_val],
            target_mode="residual",
            base_train=v4_train_rate[active_train],
            base_val=v4_val_rate[active_val],
        ),
    ]

    val_rates = {"v4_rate_head": v4_val_rate}
    test_rates = {"v4_rate_head": v4_test_rate}
    for model in models:
        if model["name"] == "v4_physical_active_rate_ridge":
            val_x, test_x = x_val, x_test
            val_base = test_base = None
        else:
            val_x, test_x = x_val_aug, x_test_aug
            val_base = v4_val_rate if model["target_mode"] == "residual" else None
            test_base = v4_test_rate if model["target_mode"] == "residual" else None
        val_rates[model["name"]] = predict_rate_ridge(model, val_x, base=val_base)
        test_rates[model["name"]] = predict_rate_ridge(model, test_x, base=test_base)

    rows = []
    for split, meta, rates in [("val_seed_3", val_meta, val_rates), ("test_seed_4", test_meta, test_rates)]:
        rows.append(
            metrics_row(
                split,
                "zero_rate",
                "none",
                None,
                meta["y_rate"],
                np.zeros_like(meta["y_rate"]),
                meta["y_active"],
                None,
            )
        )
        for name, rate in rates.items():
            for policy in ["oracle_activity", "v4_predicted_activity"]:
                gated = apply_activity_policy(
                    rate,
                    meta["y_active"],
                    meta["pred_active"],
                    "oracle_activity" if policy == "oracle_activity" else "v3_predicted_activity",
                )
                rows.append(
                    metrics_row(
                        split,
                        name,
                        policy,
                        threshold,
                        meta["y_rate"],
                        gated,
                        meta["y_active"],
                        meta["pred_active"],
                    )
                )

    metrics_df = pd.DataFrame(rows)
    interval_df = pd.DataFrame(interval_rows(val_meta, test_meta, val_rates, test_rates, threshold))
    interval_df["activity_policy"] = interval_df["activity_policy"].replace(
        {"v3_predicted_activity": "v4_predicted_activity"}
    )
    tuning_df = pd.DataFrame([row for model in models for row in model["tuning_rows"]])

    metrics_path = OUTPUT_DIR / "world_model_v4_active_rate_metrics.csv"
    intervals_path = OUTPUT_DIR / "world_model_v4_active_rate_intervals.csv"
    tuning_path = OUTPUT_DIR / "world_model_v4_active_rate_tuning.csv"
    threshold_path = OUTPUT_DIR / "world_model_v4_active_rate_threshold_curve.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    interval_df.to_csv(intervals_path, index=False, encoding="utf-8-sig")
    tuning_df.to_csv(tuning_path, index=False, encoding="utf-8-sig")
    threshold_df.to_csv(threshold_path, index=False, encoding="utf-8-sig")
    plot_path = plot_rmse_compare(metrics_df)

    summary = {
        "output_dir": display_path(OUTPUT_DIR),
        "v4_prediction_source": pred_summary,
        "selected_threshold": best_threshold,
        "active_items": {
            "train": int(active_train.sum()),
            "val": int(active_val.sum()),
            "test": int(test_meta["y_active"].sum()),
        },
        "models": {
            model["name"]: {
                "target_mode": model["target_mode"],
                "best_alpha": model["best_alpha"],
                "best_val_active_rmse": model["best_val_active_rmse"],
            }
            for model in models
        },
        "outputs": {
            "metrics_csv": display_path(metrics_path),
            "intervals_csv": display_path(intervals_path),
            "tuning_csv": display_path(tuning_path),
            "threshold_curve_csv": display_path(threshold_path),
            "rmse_compare_plot": display_path(plot_path),
        },
    }
    report_path = write_report(summary, metrics_df, interval_df, tuning_df)
    summary["outputs"]["report_md"] = display_path(report_path)
    summary_path = OUTPUT_DIR / "world_model_v4_active_rate_calibration_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
