import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from train_baseline_v0 import (
    DATASET_DIR,
    LINK_TYPES,
    build_xy,
    chronological_split,
    clip_outputs,
    fit_best_ridge,
    load_dataset,
    metrics,
    predict_ridge,
    standardize,
)


OUTPUT_DIR = DATASET_DIR / "robustness_v0"
NOISE_LEVELS = [0.0, 0.05, 0.10, 0.20, 0.30]
SEED = 20260513


def noisy_copy(arrays, level, seed):
    rng = np.random.default_rng(seed)
    out = {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in arrays.items()}
    if level <= 0:
        return out

    # Node features: x, y, z, speed, acceleration, cpu, storage.
    node_scale = np.array([15.0, 15.0, 3.0, 1.0, 0.5, 0.0, 0.0], dtype=np.float32) * level
    for key in ["x_node"]:
        noise = rng.normal(0.0, node_scale, size=out[key].shape).astype(np.float32)
        out[key] = out[key] + noise

    # Link features: distance, rate_sum, csi_mean, active_task_count, allocated_rb_count.
    for key in ["x_link"]:
        arr = out[key]
        arr[..., 0] = np.maximum(0.0, arr[..., 0] + rng.normal(0.0, 20.0 * level, size=arr[..., 0].shape))
        arr[..., 1] = np.maximum(0.0, arr[..., 1] * (1.0 + rng.normal(0.0, 0.50 * level, size=arr[..., 1].shape)))
        arr[..., 2] = np.maximum(0.0, arr[..., 2] * (1.0 + rng.normal(0.0, 0.30 * level, size=arr[..., 2].shape)))
        arr[..., 3] = np.maximum(0.0, arr[..., 3] + rng.normal(0.0, 1.0 * level, size=arr[..., 3].shape))
        arr[..., 4] = np.maximum(0.0, arr[..., 4] + rng.normal(0.0, 1.0 * level, size=arr[..., 4].shape))
        out[key] = arr.astype(np.float32)

    # Task aggregate features are counts/demands; keep them non-negative.
    for key in ["x_task"]:
        arr = out[key]
        scale = np.maximum(np.std(arr, axis=(0, 1), keepdims=True), 1.0)
        arr = arr + rng.normal(0.0, level, size=arr.shape).astype(np.float32) * scale
        out[key] = np.maximum(0.0, arr).astype(np.float32)

    return out


def train_ridge_on_arrays(arrays, edge_vocab):
    x, y, persistence, meta = build_xy(arrays, edge_vocab)
    train_idx, val_idx, test_idx = chronological_split(len(x))
    (x_train, x_val, x_test), _, _ = standardize(x[train_idx], x[train_idx], x[val_idx], x[test_idx])
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]
    persistence_train = persistence[train_idx]
    persistence_val = persistence[val_idx]
    persistence_test = persistence[test_idx]
    y_train_res = y_train - persistence_train
    y_val_res = y_val - persistence_val
    (y_train_s, y_val_s, _), y_mean, y_std = standardize(y_train_res, y_train_res, y_val_res, y_test - persistence_test)
    alpha, weights, val_mse = fit_best_ridge(
        x_train,
        y_train_s,
        x_val,
        y_val_s,
        alphas=[0.1, 1.0, 10.0, 100.0, 1000.0],
    )
    model = {
        "alpha": alpha,
        "weights": weights,
        "x_train_ref": x[train_idx],
        "x_mean": x[train_idx].mean(axis=0, keepdims=True),
        "x_std": np.where(x[train_idx].std(axis=0, keepdims=True) < 1e-6, 1.0, x[train_idx].std(axis=0, keepdims=True)),
        "y_mean": y_mean,
        "y_std": y_std,
        "meta": meta,
        "test_idx": test_idx,
        "val_mse_scaled": val_mse,
    }
    return model


def predict_ridge_model(model, arrays, edge_vocab):
    x, y, persistence, meta = build_xy(arrays, edge_vocab)
    test_idx = model["test_idx"]
    x_test = (x[test_idx] - model["x_mean"]) / model["x_std"]
    pred_res = predict_ridge(x_test, model["weights"]) * model["y_std"] + model["y_mean"]
    pred = clip_outputs(persistence[test_idx] + pred_res)
    return y[test_idx], persistence[test_idx], pred, meta


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    arrays, edge_vocab = load_dataset(DATASET_DIR)
    clean_model = train_ridge_on_arrays(arrays, edge_vocab)

    rows = []
    predictions_for_plot = {}
    for level in NOISE_LEVELS:
        noisy_arrays = noisy_copy(arrays, level, SEED + int(level * 1000))
        y_true, persistence_pred, ridge_pred, meta = predict_ridge_model(clean_model, noisy_arrays, edge_vocab)
        p_metrics = metrics(y_true, persistence_pred, meta["link_target_dim"])
        r_metrics = metrics(y_true, ridge_pred, meta["link_target_dim"])
        rows.append({"noise_level": level, "model": "persistence", **p_metrics})
        rows.append({"noise_level": level, "model": "ridge_residual", **r_metrics})
        if level in [0.0, 0.2, 0.3]:
            predictions_for_plot[level] = (y_true, persistence_pred, ridge_pred, meta)

    df = pd.DataFrame(rows)
    metrics_path = OUTPUT_DIR / "robustness_metrics.csv"
    df.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, metric, title in [
        (axes[0], "all_rmse", "All targets RMSE"),
        (axes[1], "link_rate_by_type_rmse", "Link rate RMSE"),
        (axes[2], "task_state_rmse", "Task state RMSE"),
    ]:
        for model in df["model"].unique():
            part = df[df["model"] == model]
            ax.plot(part["noise_level"], part[metric], marker="o", label=model)
        ax.set_xlabel("noise level")
        ax.set_ylabel(metric)
        ax.set_title(title)
        ax.grid(alpha=0.25)
    axes[0].legend()
    fig.suptitle("Robustness baseline under synthetic perturbations")
    fig.tight_layout()
    curve_path = OUTPUT_DIR / "robustness_noise_vs_error.png"
    fig.savefig(curve_path, dpi=180)
    plt.close(fig)

    summary = {
        "dataset_dir": str(DATASET_DIR),
        "output_dir": str(OUTPUT_DIR),
        "noise_levels": NOISE_LEVELS,
        "noise_design": {
            "node": "position/speed/acceleration Gaussian noise, CPU/storage unchanged",
            "link": "distance additive noise; rate_sum/csi_mean multiplicative noise; task/RB counts additive noise",
            "task": "aggregate task features Gaussian noise scaled by feature std, clipped non-negative",
        },
        "clean_ridge_alpha": clean_model["alpha"],
        "metrics_csv": str(metrics_path),
        "curve_png": str(curve_path),
    }
    (OUTPUT_DIR / "robustness_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
