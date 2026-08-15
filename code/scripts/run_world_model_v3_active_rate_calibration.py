import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_world_model_v0 import FIGURE_DIR, ROOT, choose_threshold, split_by_seed
from run_world_model_v3_graph_rollout import load_dataset, make_stats, predict, train_model


OUTPUT_DIR = ROOT / "reports" / "world_model_v3_active_rate_calibration"
EDGE_VOCAB_PATH = ROOT / "datasets" / "world_model_dataset_v0" / "edge_vocab.csv"
PREDICTION_CACHE = OUTPUT_DIR / "world_model_v3_active_rate_v3_predictions.npz"
ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0]
INTERVAL_LEVELS = [0.80, 0.90]
LINK_TYPES = ["U2I", "V2I", "V2U"]


def fit_standardizer(x, eps=1e-6):
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std = np.where(std < eps, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def transform(x, mean, std):
    return ((x - mean) / std).astype(np.float32)


def fit_ridge(x, y, alpha):
    x_aug = np.concatenate([x, np.ones((len(x), 1), dtype=x.dtype)], axis=1).astype(np.float64)
    y = y.reshape(-1, 1).astype(np.float64)
    eye = np.eye(x_aug.shape[1], dtype=np.float64)
    eye[-1, -1] = 0.0
    return np.linalg.solve(x_aug.T @ x_aug + alpha * eye, x_aug.T @ y)


def predict_ridge(x, weights):
    x_aug = np.concatenate([x, np.ones((len(x), 1), dtype=x.dtype)], axis=1).astype(np.float64)
    return (x_aug @ weights).reshape(-1).astype(np.float32)


def inverse_standardize_1d(y, mean, std):
    return (y * float(std.reshape(-1)[0]) + float(mean.reshape(-1)[0])).astype(np.float32)


def build_edge_metadata(num_samples, horizon, num_edges):
    sample_pos = np.broadcast_to(np.arange(num_samples)[:, None, None], (num_samples, horizon, num_edges))
    horizon_pos = np.broadcast_to(np.arange(horizon)[None, :, None], (num_samples, horizon, num_edges))
    edge_pos = np.broadcast_to(np.arange(num_edges)[None, None, :], (num_samples, horizon, num_edges))
    return sample_pos.reshape(-1), horizon_pos.reshape(-1), edge_pos.reshape(-1)


def flatten_targets(arrays, idx):
    y_rate = arrays["y_link_rate"][idx].astype(np.float32).reshape(-1)
    y_active = (arrays["y_link_active"][idx].reshape(-1) > 0.5)
    sample_pos, horizon_pos, edge_pos = build_edge_metadata(
        len(idx), arrays["y_link_rate"].shape[1], arrays["y_link_rate"].shape[2]
    )
    global_sample = np.asarray(idx)[sample_pos]
    return {
        "y_rate": y_rate,
        "y_active": y_active,
        "sample_pos": sample_pos,
        "global_sample": global_sample,
        "horizon": horizon_pos,
        "edge": edge_pos,
    }


def broadcast_edge_feature(edge_feature, horizon):
    num_samples, num_edges, feat_dim = edge_feature.shape
    return np.broadcast_to(edge_feature[:, None, :, :], (num_samples, horizon, num_edges, feat_dim)).reshape(
        num_samples * horizon * num_edges, feat_dim
    )


def broadcast_sample_feature(sample_feature, horizon, num_edges):
    num_samples, feat_dim = sample_feature.shape
    return np.broadcast_to(sample_feature[:, None, None, :], (num_samples, horizon, num_edges, feat_dim)).reshape(
        num_samples * horizon * num_edges, feat_dim
    )


def broadcast_edge_static(edge_feature, num_samples, horizon):
    num_edges, feat_dim = edge_feature.shape
    return np.broadcast_to(edge_feature[None, None, :, :], (num_samples, horizon, num_edges, feat_dim)).reshape(
        num_samples * horizon * num_edges, feat_dim
    )


def build_rate_features(arrays, idx, pred=None):
    x_node = arrays["x_node"][idx].astype(np.float32)
    x_link = arrays["x_link"][idx].astype(np.float32)
    x_task = arrays["x_task"][idx].astype(np.float32)
    edge_a_hist = arrays["edge_a_hist"][idx].astype(np.float32)
    edge_a_future = arrays["edge_a_future"][idx].astype(np.float32)
    num_samples, history, num_edges, link_feat = x_link.shape
    horizon = edge_a_future.shape[1]

    node_hist = x_node.transpose(0, 2, 1, 3).reshape(num_samples, x_node.shape[2], -1)
    src_idx = arrays["edge_src_idx"].astype(int).clip(min=0)
    dst_idx = arrays["edge_dst_idx"].astype(int).clip(min=0)
    src_hist = node_hist[:, src_idx, :]
    dst_hist = node_hist[:, dst_idx, :]
    edge_hist = x_link.transpose(0, 2, 1, 3).reshape(num_samples, num_edges, history * link_feat)
    edge_action_hist = edge_a_hist.transpose(0, 2, 1, 3).reshape(num_samples, num_edges, -1)
    future_step = edge_a_future.reshape(num_samples * horizon * num_edges, edge_a_future.shape[-1])
    future_prefix = np.cumsum(edge_a_future, axis=1).reshape(
        num_samples * horizon * num_edges, edge_a_future.shape[-1]
    )
    task_hist = x_task.reshape(num_samples, -1)

    edge_vocab = pd.read_csv(EDGE_VOCAB_PATH)
    type_index = {name: i for i, name in enumerate(LINK_TYPES)}
    link_type = edge_vocab.sort_values("edge_index")["link_type"].to_numpy()
    type_onehot = np.zeros((num_edges, len(LINK_TYPES)), dtype=np.float32)
    for edge_i, name in enumerate(link_type):
        if name in type_index:
            type_onehot[edge_i, type_index[name]] = 1.0
    edge_scalar = (np.arange(num_edges, dtype=np.float32) / max(1, num_edges - 1)).reshape(num_edges, 1)
    horizon_onehot = np.eye(horizon, dtype=np.float32)
    horizon_feat = np.broadcast_to(
        horizon_onehot[None, :, None, :], (num_samples, horizon, num_edges, horizon)
    ).reshape(num_samples * horizon * num_edges, horizon)
    horizon_scalar = (
        np.broadcast_to(
            (np.arange(horizon, dtype=np.float32) / max(1, horizon - 1))[None, :, None],
            (num_samples, horizon, num_edges),
        )
        .reshape(-1, 1)
        .astype(np.float32)
    )

    parts = [
        broadcast_edge_feature(edge_hist, horizon),
        broadcast_edge_feature(edge_action_hist, horizon),
        future_step,
        future_prefix,
        broadcast_edge_feature(src_hist, horizon),
        broadcast_edge_feature(dst_hist, horizon),
        broadcast_sample_feature(task_hist, horizon, num_edges),
        broadcast_edge_static(type_onehot, num_samples, horizon),
        broadcast_edge_static(edge_scalar, num_samples, horizon),
        horizon_feat,
        horizon_scalar,
    ]

    if pred is not None:
        prob = np.clip(pred["active_prob"].astype(np.float32).reshape(-1, 1), 1e-6, 1.0 - 1e-6)
        rate = pred["rate_pred"].astype(np.float32).reshape(-1, 1)
        parts.extend([prob, np.log(prob / (1.0 - prob)).astype(np.float32), rate, np.log1p(rate).astype(np.float32)])

    return np.concatenate(parts, axis=1).astype(np.float32)


def train_rate_ridge(
    name,
    x_train,
    y_train,
    x_val,
    y_val,
    target_mode="raw",
    base_train=None,
    base_val=None,
):
    x_mean, x_std = fit_standardizer(x_train)
    x_train_s = transform(x_train, x_mean, x_std)
    x_val_s = transform(x_val, x_mean, x_std)

    if target_mode == "log":
        target_train = np.log1p(y_train).astype(np.float32)
        target_val = np.log1p(y_val).astype(np.float32)
    elif target_mode == "residual":
        target_train = (y_train - base_train).astype(np.float32)
        target_val = (y_val - base_val).astype(np.float32)
    else:
        target_train = y_train.astype(np.float32)
        target_val = y_val.astype(np.float32)

    y_mean, y_std = fit_standardizer(target_train.reshape(-1, 1))
    y_train_s = transform(target_train.reshape(-1, 1), y_mean, y_std).reshape(-1)
    best = None
    rows = []
    for alpha in ALPHAS:
        weights = fit_ridge(x_train_s, y_train_s, alpha)
        pred_scaled = predict_ridge(x_val_s, weights)
        pred_target = inverse_standardize_1d(pred_scaled, y_mean, y_std)
        if target_mode == "log":
            pred_rate = np.expm1(pred_target)
        elif target_mode == "residual":
            pred_rate = base_val + pred_target
        else:
            pred_rate = pred_target
        pred_rate = np.clip(pred_rate, 0.0, None)
        err = pred_rate - y_val
        row = {
            "model": name,
            "alpha": float(alpha),
            "val_active_mae": float(np.mean(np.abs(err))),
            "val_active_rmse": float(np.sqrt(np.mean(err**2))),
            "val_active_bias": float(np.mean(err)),
        }
        rows.append(row)
        if best is None or row["val_active_rmse"] < best["val_active_rmse"]:
            best = {**row, "weights": weights}

    return {
        "name": name,
        "target_mode": target_mode,
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "weights": best["weights"],
        "best_alpha": best["alpha"],
        "best_val_active_rmse": best["val_active_rmse"],
        "tuning_rows": rows,
    }


def predict_rate_ridge(model, x, base=None):
    x_s = transform(x, model["x_mean"], model["x_std"])
    pred_scaled = predict_ridge(x_s, model["weights"])
    pred_target = inverse_standardize_1d(pred_scaled, model["y_mean"], model["y_std"])
    if model["target_mode"] == "log":
        pred = np.expm1(pred_target)
    elif model["target_mode"] == "residual":
        pred = base + pred_target
    else:
        pred = pred_target
    return np.clip(pred, 0.0, None).astype(np.float32)


def apply_activity_policy(rate, y_active, pred_active, policy):
    if policy == "none":
        return np.clip(rate, 0.0, None).astype(np.float32)
    if policy == "oracle_activity":
        return np.where(y_active, rate, 0.0).astype(np.float32)
    if policy == "v3_predicted_activity":
        return np.where(pred_active, rate, 0.0).astype(np.float32)
    raise ValueError(f"unknown activity policy: {policy}")


def metric_values(y_true, y_pred, mask=None):
    if mask is not None:
        y_true = y_true[mask]
        y_pred = y_pred[mask]
    if y_true.size == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "bias": float("nan")}
    err = y_pred - y_true
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "bias": float(np.mean(err)),
    }


def metrics_row(split, model, policy, threshold, y_true, rate_pred, y_active, pred_active):
    all_m = metric_values(y_true, rate_pred)
    active_m = metric_values(y_true, rate_pred, y_active)
    inactive_m = metric_values(y_true, rate_pred, ~y_active)
    return {
        "split": split,
        "model": model,
        "activity_policy": policy,
        "threshold": float(threshold) if threshold is not None else np.nan,
        "true_active_ratio": float(y_active.mean()),
        "predicted_active_ratio": float(pred_active.mean()) if pred_active is not None else np.nan,
        "all_mae": all_m["mae"],
        "all_rmse": all_m["rmse"],
        "all_bias": all_m["bias"],
        "active_mae": active_m["mae"],
        "active_rmse": active_m["rmse"],
        "active_bias": active_m["bias"],
        "inactive_mae": inactive_m["mae"],
        "inactive_rmse": inactive_m["rmse"],
        "inactive_bias": inactive_m["bias"],
    }


def load_existing_edge_action_reference():
    path = ROOT / "reports" / "edge_action_link_model_v0" / "edge_action_link_model_metrics.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    rows = []
    for split in ["val_seed_3", "test_seed_4"]:
        for model in ["zero_rate", "edge_action_ridge"]:
            part = df[(df["split"] == split) & (df["model"] == model)]
            all_row = part[part["link_type"] == "all"]
            active_row = part[part["link_type"] == "active_edges"]
            inactive_row = part[part["link_type"] == "inactive_edges"]
            if all_row.empty or active_row.empty:
                continue
            all_item = all_row.iloc[0]
            active_item = active_row.iloc[0]
            inactive_item = inactive_row.iloc[0] if len(inactive_row) else None
            rows.append(
                {
                    "split": split,
                    "model": f"{model}_existing",
                    "activity_policy": "archived_edge_action_report",
                    "threshold": np.nan,
                    "true_active_ratio": np.nan,
                    "predicted_active_ratio": np.nan,
                    "all_mae": float(all_item["mae"]),
                    "all_rmse": float(all_item["rmse"]),
                    "all_bias": float(all_item["bias"]),
                    "active_mae": float(active_item["mae"]),
                    "active_rmse": float(active_item["rmse"]),
                    "active_bias": float(active_item["bias"]),
                    "inactive_mae": float(inactive_item["mae"]) if inactive_item is not None else np.nan,
                    "inactive_rmse": float(inactive_item["rmse"]) if inactive_item is not None else np.nan,
                    "inactive_bias": float(inactive_item["bias"]) if inactive_item is not None else np.nan,
                }
            )
    return pd.DataFrame(rows)


def interval_rows(val_meta, test_meta, val_rates, test_rates, threshold):
    rows = []
    y_val = val_meta["y_rate"]
    y_test = test_meta["y_rate"]
    active_val = val_meta["y_active"]
    active_test = test_meta["y_active"]
    pred_active_val = val_meta["pred_active"]
    pred_active_test = test_meta["pred_active"]
    for model_name, val_rate in val_rates.items():
        test_rate = test_rates[model_name]
        for policy in ["oracle_activity", "v3_predicted_activity"]:
            val_policy = apply_activity_policy(val_rate, active_val, pred_active_val, policy)
            test_policy = apply_activity_policy(test_rate, active_test, pred_active_test, policy)
            residuals = np.abs(y_val[active_val] - val_policy[active_val])
            for level in INTERVAL_LEVELS:
                width = float(np.quantile(residuals, level)) if residuals.size else float("nan")
                lower = np.maximum(test_policy - width, 0.0)
                upper = test_policy + width
                covered = (y_test >= lower) & (y_test <= upper)
                rows.append(
                    {
                        "model": model_name,
                        "activity_policy": policy,
                        "threshold": float(threshold),
                        "interval": f"{int(level * 100)}%",
                        "val_active_residual_count": int(residuals.size),
                        "active_coverage": float(np.mean(covered[active_test])) if active_test.any() else np.nan,
                        "active_mean_width": float(2.0 * width),
                        "active_residual_width": width,
                    }
                )
    return rows


def per_link_type_rows(split, edge_vocab, meta, model_rates, threshold):
    rows = []
    link_type = edge_vocab.sort_values("edge_index")["link_type"].to_numpy()
    for model_name, rates in model_rates.items():
        for policy in ["oracle_activity", "v3_predicted_activity"]:
            pred = apply_activity_policy(rates, meta["y_active"], meta["pred_active"], policy)
            for current_type in LINK_TYPES:
                mask = (link_type[meta["edge"]] == current_type) & meta["y_active"]
                values = metric_values(meta["y_rate"], pred, mask)
                rows.append(
                    {
                        "split": split,
                        "model": model_name,
                        "activity_policy": policy,
                        "threshold": float(threshold),
                        "link_type": current_type,
                        "active_count": int(mask.sum()),
                        "active_mae": values["mae"],
                        "active_rmse": values["rmse"],
                        "active_bias": values["bias"],
                    }
                )
    return rows


def get_v3_predictions(arrays, train_idx, val_idx, test_idx, force_retrain=False):
    if PREDICTION_CACHE.exists() and not force_retrain:
        with np.load(PREDICTION_CACHE, allow_pickle=True) as data:
            train_pred = {
                "active_prob": data["train_active_prob"],
                "rate_pred": data["train_rate_pred"],
                "task_pred": data["train_task_pred"],
            }
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
            cached_summary = json.loads(str(data["summary"].item()))
            cached_summary = {**cached_summary, "source": "cache", "cache_file": str(PREDICTION_CACHE)}
        return train_pred, val_pred, test_pred, pd.DataFrame(), cached_summary

    stats = make_stats(arrays, train_idx)
    model, history, train_info = train_model(arrays, train_idx, val_idx, stats)
    train_pred = predict(model, arrays, train_idx, stats)
    val_pred = predict(model, arrays, val_idx, stats)
    test_pred = predict(model, arrays, test_idx, stats)
    summary = {"source": "trained", "train_info": train_info}
    np.savez_compressed(
        PREDICTION_CACHE,
        train_active_prob=train_pred["active_prob"],
        train_rate_pred=train_pred["rate_pred"],
        train_task_pred=train_pred["task_pred"],
        val_active_prob=val_pred["active_prob"],
        val_rate_pred=val_pred["rate_pred"],
        val_task_pred=val_pred["task_pred"],
        test_active_prob=test_pred["active_prob"],
        test_rate_pred=test_pred["rate_pred"],
        test_task_pred=test_pred["task_pred"],
        summary=json.dumps(summary, ensure_ascii=False),
    )
    return train_pred, val_pred, test_pred, history, summary


def plot_rmse_compare(metrics_df, path):
    test = metrics_df[metrics_df["split"] == "test_seed_4"].copy()
    order = [
        ("zero_rate", "none"),
        ("v3_rate_head", "none"),
        ("v3_rate_head", "v3_predicted_activity"),
        ("active_rate_ridge", "oracle_activity"),
        ("active_rate_ridge", "v3_predicted_activity"),
        ("active_rate_log_ridge", "v3_predicted_activity"),
        ("v3_aug_rate_ridge", "v3_predicted_activity"),
        ("v3_aug_rate_log_ridge", "v3_predicted_activity"),
        ("v3_residual_ridge", "v3_predicted_activity"),
    ]
    rows = []
    for model, policy in order:
        part = test[(test["model"] == model) & (test["activity_policy"] == policy)]
        if len(part):
            item = part.iloc[0].copy()
            item["label"] = f"{model}\n{policy}"
            rows.append(item)
    plot_df = pd.DataFrame(rows)
    if plot_df.empty:
        return None
    plt.figure(figsize=(11.0, 4.8))
    colors = ["#9ca3af", "#ef4444", "#f97316", "#16a34a", "#22c55e", "#0f766e", "#14b8a6", "#7c3aed"]
    bars = plt.bar(plot_df["label"], plot_df["active_rmse"], color=colors[: len(plot_df)])
    plt.ylabel("RMSE on true active edges")
    plt.title("Active-edge rate prediction on held-out seed 4")
    plt.xticks(rotation=16, ha="right", fontsize=8)
    plt.grid(axis="y", alpha=0.25)
    for bar in bars:
        value = bar.get_height()
        if np.isfinite(value):
            plt.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.1f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()
    return path


def plot_active_predictions(meta, model_name, pred, path):
    active = meta["y_active"]
    if not active.any():
        return None
    y = meta["y_rate"][active]
    p = pred[active]
    order = np.argsort(y)
    x = np.arange(len(order))
    plt.figure(figsize=(9.5, 4.2))
    plt.plot(x, y[order], label="true active-edge rate", color="#111827", lw=1.8)
    plt.plot(x, p[order], label=model_name, color="#2563eb", lw=1.4)
    plt.xlabel("test active edge item sorted by true rate")
    plt.ylabel("rate_sum")
    plt.title("Active-edge rate fit on held-out seed 4")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()
    return path


def plot_interval_calibration(interval_df, path):
    part = interval_df[interval_df["interval"] == "90%"].copy()
    if part.empty:
        return None
    part["label"] = part["model"] + "\n" + part["activity_policy"]
    fig, ax1 = plt.subplots(figsize=(10.5, 4.6))
    x = np.arange(len(part))
    bars = ax1.bar(x - 0.18, part["active_coverage"], width=0.36, color="#2563eb", label="coverage")
    ax1.axhline(0.90, color="#111827", linestyle="--", linewidth=1.0, label="nominal 90%")
    ax1.set_ylabel("active-edge coverage")
    ax1.set_ylim(0.0, 1.05)
    ax2 = ax1.twinx()
    ax2.bar(x + 0.18, part["active_mean_width"], width=0.36, color="#f97316", alpha=0.75, label="mean width")
    ax2.set_ylabel("mean interval width")
    ax1.set_xticks(x)
    ax1.set_xticklabels(part["label"], rotation=18, ha="right", fontsize=8)
    ax1.set_title("Residual-quantile intervals for active-edge rate")
    ax1.grid(axis="y", alpha=0.25)
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left", fontsize=8)
    for bar in bars:
        value = bar.get_height()
        if np.isfinite(value):
            ax1.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def write_test_predictions(edge_vocab, test_idx, test_meta, test_pred, test_rates, threshold, path):
    active = test_meta["y_active"]
    edge_type = edge_vocab.sort_values("edge_index")["link_type"].to_numpy()
    df = pd.DataFrame(
        {
            "global_sample": test_meta["global_sample"][active],
            "seed": 4,
            "horizon": test_meta["horizon"][active],
            "edge_index": test_meta["edge"][active],
            "link_type": edge_type[test_meta["edge"][active]],
            "true_rate": test_meta["y_rate"][active],
            "v3_active_prob": test_pred["active_prob"].reshape(-1)[active],
            "v3_pred_active": test_meta["pred_active"][active],
            "v3_rate_head": test_rates["v3_rate_head"][active],
            "active_rate_ridge": test_rates["active_rate_ridge"][active],
            "active_rate_log_ridge": test_rates["active_rate_log_ridge"][active],
            "v3_aug_rate_ridge": test_rates["v3_aug_rate_ridge"][active],
            "v3_aug_rate_log_ridge": test_rates["v3_aug_rate_log_ridge"][active],
            "v3_residual_ridge": test_rates["v3_residual_ridge"][active],
        }
    )
    df["threshold"] = float(threshold)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return df


def write_report(summary, metrics_df, intervals_df, per_type_df, tuning_df):
    test = metrics_df[metrics_df["split"] == "test_seed_4"].copy()
    candidate = test[
        (test["activity_policy"] == "v3_predicted_activity")
        & (
            test["model"].isin(
                [
                    "v3_rate_head",
                    "active_rate_ridge",
                    "active_rate_log_ridge",
                    "v3_aug_rate_ridge",
                    "v3_aug_rate_log_ridge",
                    "v3_residual_ridge",
                ]
            )
        )
    ]
    best_line = ""
    if len(candidate):
        best = candidate.sort_values("active_rmse").iloc[0]
        best_line = (
            f"The best v3-thresholded active-rate variant on held-out seed 4 is "
            f"`{best['model']}` with active-edge RMSE `{best['active_rmse']:.3f}`."
        )

    lines = [
        "# World model v3 active-rate calibration report",
        "",
        "## Goal",
        "",
        "This experiment isolates the link-rate bottleneck after v3 graph latent rollout improved active-edge detection. It keeps the existing `world_model_dataset_v0`, seed split, and v3 activity threshold, then evaluates active-only rate regressors and residual-quantile intervals on true active edges.",
        "",
        "## Setup",
        "",
        f"- Train seeds: `{summary['split']['train_seeds']}`",
        f"- Validation seed: `{summary['split']['val_seed']}`",
        f"- Test seed: `{summary['split']['test_seed']}`",
        f"- Active train/val/test items: `{summary['active_items']['train']}` / `{summary['active_items']['val']}` / `{summary['active_items']['test']}`",
        f"- Selected v3 threshold: `{summary['selected_threshold']['threshold']:.4f}`",
        "",
        "The pure active-rate regressors use edge history, edge-action history and future actions, source/destination node histories, task history, link type, and horizon index. The `v3_aug_*` and residual variants additionally use the v3 activity probability and v3 rate-head output. Training is restricted to active train edges.",
        "",
        "## Main Metrics",
        "",
        metrics_df.to_markdown(index=False),
        "",
        "## Active-Rate Intervals",
        "",
        intervals_df.to_markdown(index=False),
        "",
        "## Link-Type Breakdown",
        "",
        per_type_df.to_markdown(index=False),
        "",
        "## Ridge Tuning",
        "",
        tuning_df.to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        best_line,
        "",
        "- `oracle_activity` is an upper-bound rate-only view because it uses the true active mask.",
        "- `v3_predicted_activity` is the deployable two-stage view in this experiment, because it gates rates with the v3 activity threshold selected on validation seed 3.",
        "- The archived edge-action Ridge result remains an important specialized baseline for all-edge RMSE, but its archived active-edge row uses a different grouping definition. This report therefore keeps the main active-edge table on a strict sample-horizon-edge active definition.",
        "- These results should not be described as replacing AirFogSim. They only test whether the learned rollout can become better calibrated on active communication edges.",
        "",
        "## Outputs",
        "",
    ]
    for key, value in summary["outputs"].items():
        lines.append(f"- {key}: `{value}`")
    path = OUTPUT_DIR / "world_model_v3_active_rate_calibration_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-retrain-v3", action="store_true", help="ignore cached v3 predictions and retrain v3")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    arrays = load_dataset()
    edge_vocab = pd.read_csv(EDGE_VOCAB_PATH)
    train_idx, val_idx, test_idx = split_by_seed(arrays["sample_seed"])
    train_pred, val_pred, test_pred, history, v3_summary = get_v3_predictions(
        arrays, train_idx, val_idx, test_idx, force_retrain=args.force_retrain_v3
    )
    best_threshold, threshold_df = choose_threshold(arrays["y_link_active"][val_idx], val_pred["active_prob"])
    threshold = float(best_threshold["threshold"])

    train_meta = flatten_targets(arrays, train_idx)
    val_meta = flatten_targets(arrays, val_idx)
    test_meta = flatten_targets(arrays, test_idx)
    train_meta["pred_active"] = train_pred["active_prob"].reshape(-1) >= threshold
    val_meta["pred_active"] = val_pred["active_prob"].reshape(-1) >= threshold
    test_meta["pred_active"] = test_pred["active_prob"].reshape(-1) >= threshold

    print("[active-rate] building features", flush=True)
    x_train = build_rate_features(arrays, train_idx, pred=None)
    x_val = build_rate_features(arrays, val_idx, pred=None)
    x_test = build_rate_features(arrays, test_idx, pred=None)
    x_train_aug = build_rate_features(arrays, train_idx, train_pred)
    x_val_aug = build_rate_features(arrays, val_idx, val_pred)
    x_test_aug = build_rate_features(arrays, test_idx, test_pred)

    active_train = train_meta["y_active"]
    active_val = val_meta["y_active"]
    print(
        f"[active-rate] active items train={int(active_train.sum())} "
        f"val={int(active_val.sum())} test={int(test_meta['y_active'].sum())}",
        flush=True,
    )

    v3_train_rate = train_pred["rate_pred"].reshape(-1).astype(np.float32)
    v3_val_rate = val_pred["rate_pred"].reshape(-1).astype(np.float32)
    v3_test_rate = test_pred["rate_pred"].reshape(-1).astype(np.float32)

    models = []
    model_specs = []
    models.append(
        train_rate_ridge(
            "active_rate_ridge",
            x_train[active_train],
            train_meta["y_rate"][active_train],
            x_val[active_val],
            val_meta["y_rate"][active_val],
            target_mode="raw",
        )
    )
    model_specs.append((models[-1], x_train, x_val, x_test))
    models.append(
        train_rate_ridge(
            "active_rate_log_ridge",
            x_train[active_train],
            train_meta["y_rate"][active_train],
            x_val[active_val],
            val_meta["y_rate"][active_val],
            target_mode="log",
        )
    )
    model_specs.append((models[-1], x_train, x_val, x_test))
    models.append(
        train_rate_ridge(
            "v3_aug_rate_ridge",
            x_train_aug[active_train],
            train_meta["y_rate"][active_train],
            x_val_aug[active_val],
            val_meta["y_rate"][active_val],
            target_mode="raw",
        )
    )
    model_specs.append((models[-1], x_train_aug, x_val_aug, x_test_aug))
    models.append(
        train_rate_ridge(
            "v3_aug_rate_log_ridge",
            x_train_aug[active_train],
            train_meta["y_rate"][active_train],
            x_val_aug[active_val],
            val_meta["y_rate"][active_val],
            target_mode="log",
        )
    )
    model_specs.append((models[-1], x_train_aug, x_val_aug, x_test_aug))
    models.append(
        train_rate_ridge(
            "v3_residual_ridge",
            x_train_aug[active_train],
            train_meta["y_rate"][active_train],
            x_val_aug[active_val],
            val_meta["y_rate"][active_val],
            target_mode="residual",
            base_train=v3_train_rate[active_train],
            base_val=v3_val_rate[active_val],
        )
    )
    model_specs.append((models[-1], x_train_aug, x_val_aug, x_test_aug))

    val_rates = {"v3_rate_head": v3_val_rate}
    test_rates = {"v3_rate_head": v3_test_rate}
    train_rates = {"v3_rate_head": v3_train_rate}
    tuning_rows = []
    for model, current_x_train, current_x_val, current_x_test in model_specs:
        tuning_rows.extend(model["tuning_rows"])
        name = model["name"]
        train_rates[name] = predict_rate_ridge(
            model, current_x_train, base=v3_train_rate if model["target_mode"] == "residual" else None
        )
        val_rates[name] = predict_rate_ridge(
            model, current_x_val, base=v3_val_rate if model["target_mode"] == "residual" else None
        )
        test_rates[name] = predict_rate_ridge(
            model, current_x_test, base=v3_test_rate if model["target_mode"] == "residual" else None
        )
        print(
            f"[active-rate] {name} alpha={model['best_alpha']} "
            f"val_active_rmse={model['best_val_active_rmse']:.3f}",
            flush=True,
        )

    legacy_reference_df = load_existing_edge_action_reference()
    legacy_reference_path = OUTPUT_DIR / "world_model_v3_active_rate_legacy_edge_action_reference.csv"
    if len(legacy_reference_df):
        legacy_reference_df.to_csv(legacy_reference_path, index=False, encoding="utf-8-sig")
    rows = []
    for split, meta, pred_rates in [
        ("train_seed_0_1_2", train_meta, train_rates),
        ("val_seed_3", val_meta, val_rates),
        ("test_seed_4", test_meta, test_rates),
    ]:
        zero = np.zeros_like(meta["y_rate"], dtype=np.float32)
        rows.append(metrics_row(split, "zero_rate", "none", None, meta["y_rate"], zero, meta["y_active"], None))
        for model_name, rate in pred_rates.items():
            policies = ["none"] if model_name == "v3_rate_head" else []
            policies.extend(["oracle_activity", "v3_predicted_activity"])
            for policy in policies:
                gated = apply_activity_policy(rate, meta["y_active"], meta["pred_active"], policy)
                rows.append(
                    metrics_row(
                        split,
                        model_name,
                        policy,
                        threshold if policy != "none" else None,
                        meta["y_rate"],
                        gated,
                        meta["y_active"],
                        meta["pred_active"],
                    )
                )

    metrics_df = pd.DataFrame(rows)
    interval_df = pd.DataFrame(interval_rows(val_meta, test_meta, val_rates, test_rates, threshold))
    per_type_df = pd.DataFrame(per_link_type_rows("test_seed_4", edge_vocab, test_meta, test_rates, threshold))
    tuning_df = pd.DataFrame(tuning_rows)
    threshold_df.to_csv(OUTPUT_DIR / "world_model_v3_active_rate_threshold_curve.csv", index=False, encoding="utf-8-sig")
    if len(history):
        history.to_csv(OUTPUT_DIR / "world_model_v3_active_rate_v3_training_history.csv", index=False, encoding="utf-8-sig")

    metrics_path = OUTPUT_DIR / "world_model_v3_active_rate_metrics.csv"
    intervals_path = OUTPUT_DIR / "world_model_v3_active_rate_intervals.csv"
    per_type_path = OUTPUT_DIR / "world_model_v3_active_rate_by_link_type.csv"
    tuning_path = OUTPUT_DIR / "world_model_v3_active_rate_tuning.csv"
    predictions_path = OUTPUT_DIR / "world_model_v3_active_rate_test_predictions.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    interval_df.to_csv(intervals_path, index=False, encoding="utf-8-sig")
    per_type_df.to_csv(per_type_path, index=False, encoding="utf-8-sig")
    tuning_df.to_csv(tuning_path, index=False, encoding="utf-8-sig")
    test_active_df = write_test_predictions(
        edge_vocab, test_idx, test_meta, test_pred, test_rates, threshold, predictions_path
    )

    rmse_plot = plot_rmse_compare(metrics_df, FIGURE_DIR / "world_model_v3_active_rate_rmse_compare.png")
    candidate = metrics_df[
        (metrics_df["split"] == "test_seed_4")
        & (metrics_df["activity_policy"] == "v3_predicted_activity")
        & (
            metrics_df["model"].isin(
                [
                    "active_rate_ridge",
                    "active_rate_log_ridge",
                    "v3_aug_rate_ridge",
                    "v3_aug_rate_log_ridge",
                    "v3_residual_ridge",
                ]
            )
        )
    ]
    if len(candidate):
        best_model_name = candidate.sort_values("active_rmse").iloc[0]["model"]
    else:
        best_model_name = "active_rate_ridge"
    prediction_plot = plot_active_predictions(
        test_meta,
        best_model_name,
        apply_activity_policy(test_rates[best_model_name], test_meta["y_active"], test_meta["pred_active"], "v3_predicted_activity"),
        FIGURE_DIR / "world_model_v3_active_rate_prediction_trace.png",
    )
    interval_plot = plot_interval_calibration(
        interval_df, FIGURE_DIR / "world_model_v3_active_rate_interval_calibration.png"
    )

    summary = {
        "output_dir": str(OUTPUT_DIR),
        "prediction_cache": str(PREDICTION_CACHE),
        "v3_prediction_source": v3_summary,
        "split": {
            "train_seeds": [0, 1, 2],
            "val_seed": 3,
            "test_seed": 4,
            "train_samples": int(len(train_idx)),
            "val_samples": int(len(val_idx)),
            "test_samples": int(len(test_idx)),
        },
        "active_items": {
            "train": int(train_meta["y_active"].sum()),
            "val": int(val_meta["y_active"].sum()),
            "test": int(test_meta["y_active"].sum()),
        },
        "selected_threshold": best_threshold,
        "ridge_models": [
            {
                "name": model["name"],
                "target_mode": model["target_mode"],
                "best_alpha": model["best_alpha"],
                "best_val_active_rmse": model["best_val_active_rmse"],
            }
            for model in models
        ],
        "test_active_prediction_rows": int(len(test_active_df)),
        "outputs": {
            "metrics_csv": str(metrics_path),
            "intervals_csv": str(intervals_path),
            "by_link_type_csv": str(per_type_path),
            "tuning_csv": str(tuning_path),
            "test_predictions_csv": str(predictions_path),
            "legacy_edge_action_reference_csv": str(legacy_reference_path) if len(legacy_reference_df) else None,
            "threshold_curve_csv": str(OUTPUT_DIR / "world_model_v3_active_rate_threshold_curve.csv"),
            "rmse_compare_plot": str(rmse_plot),
            "prediction_trace_plot": str(prediction_plot),
            "interval_calibration_plot": str(interval_plot),
        },
    }
    summary_path = OUTPUT_DIR / "world_model_v3_active_rate_calibration_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = write_report(summary, metrics_df, interval_df, per_type_df, tuning_df)
    summary["outputs"]["report_md"] = str(report_path)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
