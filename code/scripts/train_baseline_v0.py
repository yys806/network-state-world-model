import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXAMPLE_DIR = Path(__file__).resolve().parent
DATASET_DIR = EXAMPLE_DIR / "outputs" / "dataset_v0_from_demo_run_20260507_190930"
OUTPUT_DIR = DATASET_DIR / "baseline_v0"
LINK_TYPES = ["U2I", "V2I", "V2U"]


def load_dataset(dataset_dir):
    with np.load(dataset_dir / "dataset_v0_samples.npz", allow_pickle=True) as data:
        arrays = {key: data[key] for key in data.files}
    edge_vocab = pd.read_csv(dataset_dir / "edge_vocab.csv")
    return arrays, edge_vocab


def make_link_type_masks(edge_vocab):
    return {link_type: (edge_vocab["link_type"].to_numpy() == link_type) for link_type in LINK_TYPES}


def mean_rate_by_type(link_tensor, masks):
    # link_tensor shape: samples x steps x edges x features, feature 1 is rate_sum.
    rate = link_tensor[..., 1]
    outputs = []
    for link_type in LINK_TYPES:
        mask = masks[link_type]
        if mask.sum() == 0:
            outputs.append(np.zeros(rate.shape[:2], dtype=np.float32))
        else:
            outputs.append(rate[..., mask].mean(axis=-1))
    return np.stack(outputs, axis=-1)


def build_xy(arrays, edge_vocab):
    masks = make_link_type_masks(edge_vocab)
    x_node = arrays["x_node"].astype(np.float32)
    x_link = arrays["x_link"].astype(np.float32)
    x_task = arrays["x_task"].astype(np.float32)
    y_link = arrays["y_link"].astype(np.float32)
    y_task = arrays["y_task"].astype(np.float32)

    x_flat = np.concatenate(
        [
            x_node.reshape(len(x_node), -1),
            x_link.reshape(len(x_link), -1),
            x_task.reshape(len(x_task), -1),
        ],
        axis=1,
    )
    x = build_compact_features(x_node, x_link, x_task, edge_vocab)
    y_link_type = mean_rate_by_type(y_link, masks).reshape(len(y_link), -1)
    y_task_flat = y_task.reshape(len(y_task), -1)
    y = np.concatenate([y_link_type, y_task_flat], axis=1)

    last_link_type = mean_rate_by_type(x_link[:, -1:, :, :], masks)
    last_link_pred = np.repeat(last_link_type, y_link.shape[1], axis=1).reshape(len(x_link), -1)
    last_task_pred = np.repeat(x_task[:, -1:, :], y_task.shape[1], axis=1).reshape(len(x_task), -1)
    persistence = np.concatenate([last_link_pred, last_task_pred], axis=1)

    meta = {
        "num_samples": len(x),
        "num_features": x.shape[1],
        "num_flat_features": x_flat.shape[1],
        "num_targets": y.shape[1],
        "link_target_dim": y_link_type.shape[1],
        "task_target_dim": y_task_flat.shape[1],
        "horizon": int(y_link.shape[1]),
        "link_types": LINK_TYPES,
        "task_features": arrays["task_features"].tolist(),
    }
    return x, y, persistence, meta


def build_compact_features(x_node, x_link, x_task, edge_vocab):
    parts = []
    for reducers in [(np.mean, "mean"), (np.std, "std"), (np.max, "max")]:
        fn, _ = reducers
        parts.append(fn(x_node, axis=2).reshape(len(x_node), -1))

    link_types = edge_vocab["link_type"].to_numpy()
    for link_type in LINK_TYPES:
        mask = link_types == link_type
        link_part = x_link[:, :, mask, :]
        if link_part.shape[2] == 0:
            parts.append(np.zeros((len(x_link), x_link.shape[1] * x_link.shape[3] * 3), dtype=np.float32))
            continue
        parts.append(link_part.mean(axis=2).reshape(len(x_link), -1))
        parts.append(link_part.std(axis=2).reshape(len(x_link), -1))
        parts.append(link_part.max(axis=2).reshape(len(x_link), -1))

    parts.append(x_task.reshape(len(x_task), -1))
    return np.concatenate(parts, axis=1).astype(np.float32)


def chronological_split(n):
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    train = np.arange(0, n_train)
    val = np.arange(n_train, n_train + n_val)
    test = np.arange(n_train + n_val, n)
    return train, val, test


def standardize(train_x, *parts, eps=1e-6):
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std = np.where(std < eps, 1.0, std)
    return [(part - mean) / std for part in parts], mean, std


def fit_ridge(x, y, alpha=10.0):
    x_aug = np.concatenate([x, np.ones((len(x), 1), dtype=x.dtype)], axis=1)
    eye = np.eye(x_aug.shape[1], dtype=np.float32)
    eye[-1, -1] = 0.0
    return np.linalg.solve(x_aug.T @ x_aug + alpha * eye, x_aug.T @ y)


def predict_ridge(x, weights):
    x_aug = np.concatenate([x, np.ones((len(x), 1), dtype=x.dtype)], axis=1)
    return x_aug @ weights


def fit_best_ridge(x_train, y_train, x_val, y_val, alphas):
    best = None
    best_score = float("inf")
    for alpha in alphas:
        weights = fit_ridge(x_train, y_train, alpha=alpha)
        pred = predict_ridge(x_val, weights)
        score = float(np.mean((pred - y_val) ** 2))
        if score < best_score:
            best_score = score
            best = (alpha, weights)
    return best[0], best[1], best_score


def train_mlp(x_train, y_train, x_val, y_val, hidden=64, epochs=600, lr=1e-3, seed=42):
    rng = np.random.default_rng(seed)
    w1 = rng.normal(0, 1 / np.sqrt(x_train.shape[1]), size=(x_train.shape[1], hidden)).astype(np.float32)
    b1 = np.zeros((1, hidden), dtype=np.float32)
    w2 = rng.normal(0, 1 / np.sqrt(hidden), size=(hidden, y_train.shape[1])).astype(np.float32)
    b2 = np.zeros((1, y_train.shape[1]), dtype=np.float32)
    best = None
    best_val = float("inf")
    history = []

    for epoch in range(1, epochs + 1):
        h = np.maximum(x_train @ w1 + b1, 0.0)
        pred = h @ w2 + b2
        err = pred - y_train
        loss = float(np.mean(err**2))

        grad_pred = (2.0 / len(x_train)) * err / y_train.shape[1]
        grad_w2 = h.T @ grad_pred
        grad_b2 = grad_pred.sum(axis=0, keepdims=True)
        grad_h = grad_pred @ w2.T
        grad_z1 = grad_h * (h > 0)
        grad_w1 = x_train.T @ grad_z1
        grad_b1 = grad_z1.sum(axis=0, keepdims=True)

        w1 -= lr * grad_w1
        b1 -= lr * grad_b1
        w2 -= lr * grad_w2
        b2 -= lr * grad_b2

        if epoch % 25 == 0 or epoch == 1:
            val_pred = np.maximum(x_val @ w1 + b1, 0.0) @ w2 + b2
            val_loss = float(np.mean((val_pred - y_val) ** 2))
            history.append({"epoch": epoch, "train_mse": loss, "val_mse": val_loss})
            if val_loss < best_val:
                best_val = val_loss
                best = tuple(arr.copy() for arr in (w1, b1, w2, b2))

    return best, history


def predict_mlp(x, params):
    w1, b1, w2, b2 = params
    return np.maximum(x @ w1 + b1, 0.0) @ w2 + b2


def metrics(y_true, y_pred, link_dim):
    out = {}
    slices = {
        "all": slice(None),
        "link_rate_by_type": slice(0, link_dim),
        "task_state": slice(link_dim, None),
    }
    for name, sl in slices.items():
        err = y_pred[:, sl] - y_true[:, sl]
        out[f"{name}_mae"] = float(np.mean(np.abs(err)))
        out[f"{name}_rmse"] = float(np.sqrt(np.mean(err**2)))
    return out


def inverse_standardize(y_scaled, mean, std):
    return y_scaled * std + mean


def clip_outputs(y_pred):
    return np.maximum(y_pred, 0.0)


def plot_link_predictions(output_dir, y_true, predictions, meta):
    output_dir.mkdir(parents=True, exist_ok=True)
    horizon = meta["horizon"]
    fig, axes = plt.subplots(len(LINK_TYPES), horizon, figsize=(14, 7), sharex=True)
    if axes.ndim == 1:
        axes = axes.reshape(len(LINK_TYPES), horizon)
    x_axis = np.arange(len(y_true))
    for type_idx, link_type in enumerate(LINK_TYPES):
        for step in range(horizon):
            target_idx = step * len(LINK_TYPES) + type_idx
            ax = axes[type_idx, step]
            ax.plot(x_axis, y_true[:, target_idx], label="true", color="#111827", lw=1.7)
            for name, pred in predictions.items():
                ax.plot(x_axis, pred[:, target_idx], label=name, lw=1.2, alpha=0.85)
            ax.set_title(f"{link_type}, t+{step + 1}")
            ax.grid(alpha=0.25)
    axes[0, 0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Baseline prediction: future mean link rate by type")
    fig.tight_layout()
    path = output_dir / "baseline_link_rate_predictions.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_task_predictions(output_dir, y_true, predictions, meta):
    output_dir.mkdir(parents=True, exist_ok=True)
    task_features = meta["task_features"]
    horizon = meta["horizon"]
    selected = ["num_to_offload", "num_computing", "num_finished"]
    feature_indices = [task_features.index(name) for name in selected]
    fig, axes = plt.subplots(len(selected), horizon, figsize=(14, 7), sharex=True)
    x_axis = np.arange(len(y_true))
    for row_idx, feat_idx in enumerate(feature_indices):
        for step in range(horizon):
            target_idx = meta["link_target_dim"] + step * len(task_features) + feat_idx
            ax = axes[row_idx, step]
            ax.plot(x_axis, y_true[:, target_idx], label="true", color="#111827", lw=1.7)
            for name, pred in predictions.items():
                ax.plot(x_axis, pred[:, target_idx], label=name, lw=1.2, alpha=0.85)
            ax.set_title(f"{selected[row_idx]}, t+{step + 1}")
            ax.grid(alpha=0.25)
    axes[0, 0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Baseline prediction: future task state statistics")
    fig.tight_layout()
    path = output_dir / "baseline_task_state_predictions.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    arrays, edge_vocab = load_dataset(DATASET_DIR)
    x, y, persistence, meta = build_xy(arrays, edge_vocab)
    train_idx, val_idx, test_idx = chronological_split(len(x))

    (x_train, x_val, x_test), x_mean, x_std = standardize(x[train_idx], x[train_idx], x[val_idx], x[test_idx])
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]
    persistence_train = persistence[train_idx]
    persistence_val = persistence[val_idx]
    persistence_test = persistence[test_idx]

    # Dense simulation steps make current-state persistence a strong baseline.
    # Learned models therefore predict residual change over persistence.
    y_train_res = y_train - persistence_train
    y_val_res = y_val - persistence_val
    y_test_res = y_test - persistence_test
    (y_train_s, y_val_s, y_test_s), y_mean, y_std = standardize(y_train_res, y_train_res, y_val_res, y_test_res)

    ridge_alpha, ridge_weights, ridge_val_mse = fit_best_ridge(
        x_train,
        y_train_s,
        x_val,
        y_val_s,
        alphas=[0.1, 1.0, 10.0, 100.0, 1000.0],
    )
    ridge_test = clip_outputs(persistence_test + inverse_standardize(predict_ridge(x_test, ridge_weights), y_mean, y_std))
    mlp_params, mlp_history = train_mlp(x_train, y_train_s, x_val, y_val_s, hidden=64, epochs=800, lr=5e-4)
    mlp_test = clip_outputs(persistence_test + inverse_standardize(predict_mlp(x_test, mlp_params), y_mean, y_std))

    preds = {
        "persistence": persistence_test,
        "ridge": ridge_test,
        "mlp": mlp_test,
    }
    rows = []
    for name, pred in preds.items():
        rows.append({"model": name, **metrics(y_test, pred, meta["link_target_dim"])})
    metrics_df = pd.DataFrame(rows)
    metrics_path = OUTPUT_DIR / "baseline_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    history_path = OUTPUT_DIR / "mlp_training_history.csv"
    pd.DataFrame(mlp_history).to_csv(history_path, index=False, encoding="utf-8-sig")
    link_plot = plot_link_predictions(OUTPUT_DIR, y_test, preds, meta)
    task_plot = plot_task_predictions(OUTPUT_DIR, y_test, preds, meta)

    summary = {
        "dataset_dir": str(DATASET_DIR),
        "output_dir": str(OUTPUT_DIR),
        "split": {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)},
        "meta": meta,
        "ridge": {"selected_alpha": ridge_alpha, "val_mse_scaled": ridge_val_mse},
        "metrics": rows,
        "outputs": {
            "metrics_csv": str(metrics_path),
            "mlp_training_history_csv": str(history_path),
            "link_plot": str(link_plot),
            "task_plot": str(task_plot),
        },
    }
    (OUTPUT_DIR / "baseline_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
