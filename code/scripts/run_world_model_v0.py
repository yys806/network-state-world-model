import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "datasets" / "world_model_dataset_v0"
OUTPUT_DIR = ROOT / "reports" / "world_model_v0"
FIGURE_DIR = ROOT / "figures"


def load_dataset():
    with np.load(DATASET_DIR / "world_model_dataset_v0_samples.npz", allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def split_by_seed(sample_seed, train_seeds=(0, 1, 2), val_seed=3, test_seed=4):
    sample_seed = np.asarray(sample_seed)
    return (
        np.where(np.isin(sample_seed, train_seeds))[0],
        np.where(sample_seed == val_seed)[0],
        np.where(sample_seed == test_seed)[0],
    )


def fit_stats(x):
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def normalize(x, stats):
    mean, std = stats
    return ((x - mean) / std).astype(np.float32)


def inverse_normalize(x, stats):
    mean, std = stats
    return (x * std + mean).astype(np.float32)


class WorldModelDataset(Dataset):
    def __init__(self, arrays, idx, stats):
        self.x_node = torch.from_numpy(normalize(arrays["x_node"][idx], stats["x_node"]))
        self.x_link = torch.from_numpy(normalize(arrays["x_link"][idx], stats["x_link"]))
        self.x_task = torch.from_numpy(normalize(arrays["x_task"][idx], stats["x_task"]))
        self.edge_a_hist = torch.from_numpy(normalize(arrays["edge_a_hist"][idx], stats["edge_a_hist"]))
        self.edge_a_future = torch.from_numpy(normalize(arrays["edge_a_future"][idx], stats["edge_a_future"]))
        self.y_task = torch.from_numpy(normalize(arrays["y_task"][idx], stats["y_task"]))
        self.y_rate = torch.from_numpy(normalize(arrays["y_link_rate"][idx], stats["y_link_rate"]))
        self.y_active = torch.from_numpy(arrays["y_link_active"][idx].astype(np.float32))

    def __len__(self):
        return len(self.y_task)

    def __getitem__(self, i):
        return (
            self.x_node[i],
            self.x_link[i],
            self.x_task[i],
            self.edge_a_hist[i],
            self.edge_a_future[i],
            self.y_active[i],
            self.y_rate[i],
            self.y_task[i],
        )


class EdgeActionWorldModel(nn.Module):
    def __init__(self, node_dim, link_dim, task_context_dim, task_out_dim, edge_action_dim, horizon, hidden=64):
        super().__init__()
        self.horizon = horizon
        self.task_out_dim = task_out_dim
        self.node_proj = nn.Sequential(nn.Linear(node_dim, hidden), nn.ReLU(), nn.LayerNorm(hidden))
        self.link_proj = nn.Sequential(nn.Linear(link_dim, hidden), nn.ReLU(), nn.LayerNorm(hidden))
        self.edge_action_proj = nn.Sequential(nn.Linear(edge_action_dim, hidden), nn.ReLU(), nn.LayerNorm(hidden))
        self.task_proj = nn.Sequential(nn.Linear(task_context_dim, hidden), nn.ReLU(), nn.LayerNorm(hidden))
        self.edge_fuse = nn.Sequential(
            nn.Linear(hidden * 4, hidden * 2),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
        )
        self.active_head = nn.Linear(hidden, horizon)
        self.rate_head = nn.Linear(hidden, horizon)
        self.task_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden * 2),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden * 2, horizon * task_out_dim),
        )

    def forward(self, x_node, x_link, x_task, edge_a_hist, edge_a_future):
        # x_node: B,H,N,F; x_link: B,H,E,F; edge actions: B,H/E,K,E,F
        node_context = self.node_proj(x_node.mean(dim=(1, 2)))
        task_context = self.task_proj(x_task.reshape(x_task.shape[0], -1))
        edge_hist = x_link.permute(0, 2, 1, 3).reshape(x_link.shape[0], x_link.shape[2], -1)
        edge_action = torch.cat(
            [
                edge_a_hist.permute(0, 2, 1, 3).reshape(edge_a_hist.shape[0], edge_a_hist.shape[2], -1),
                edge_a_future.permute(0, 2, 1, 3).reshape(edge_a_future.shape[0], edge_a_future.shape[2], -1),
            ],
            dim=-1,
        )
        edge_state_h = self.link_proj(edge_hist)
        edge_action_h = self.edge_action_proj(edge_action)
        node_h = node_context[:, None, :].expand_as(edge_state_h)
        task_h = task_context[:, None, :].expand_as(edge_state_h)
        edge_h = self.edge_fuse(torch.cat([edge_state_h, edge_action_h, node_h, task_h], dim=-1))
        active_logits = self.active_head(edge_h).transpose(1, 2)
        rate_delta = self.rate_head(edge_h).transpose(1, 2)
        task_pred = self.task_head(torch.cat([node_context, task_context], dim=-1)).reshape(
            x_node.shape[0], self.horizon, self.task_out_dim
        )
        return active_logits, rate_delta, task_pred


def make_stats(arrays, train_idx):
    edge_action_combined = np.concatenate(
        [
            arrays["edge_a_hist"][train_idx].reshape(len(train_idx), arrays["edge_a_hist"].shape[2], -1),
            arrays["edge_a_future"][train_idx].reshape(len(train_idx), arrays["edge_a_future"].shape[2], -1),
        ],
        axis=-1,
    )
    return {
        "x_node": fit_stats(arrays["x_node"][train_idx]),
        "x_link": fit_stats(arrays["x_link"][train_idx]),
        "x_task": fit_stats(arrays["x_task"][train_idx]),
        "edge_a_hist": fit_stats(arrays["edge_a_hist"][train_idx]),
        "edge_a_future": fit_stats(arrays["edge_a_future"][train_idx]),
        "edge_action_combined": fit_stats(edge_action_combined),
        "y_task": fit_stats(arrays["y_task"][train_idx]),
        "y_link_rate": fit_stats(arrays["y_link_rate"][train_idx]),
    }


def train_model(arrays, train_idx, val_idx, stats):
    torch.manual_seed(42)
    train_ds = WorldModelDataset(arrays, train_idx, stats)
    val_ds = WorldModelDataset(arrays, val_idx, stats)
    loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    horizon = arrays["y_link_rate"].shape[1]
    model = EdgeActionWorldModel(
        node_dim=arrays["x_node"].shape[3],
        link_dim=arrays["x_link"].shape[1] * arrays["x_link"].shape[3],
        task_context_dim=arrays["x_task"].shape[1] * arrays["x_task"].shape[2],
        task_out_dim=arrays["y_task"].shape[2],
        edge_action_dim=arrays["edge_a_hist"].shape[1] * arrays["edge_a_hist"].shape[3]
        + arrays["edge_a_future"].shape[1] * arrays["edge_a_future"].shape[3],
        horizon=horizon,
        hidden=64,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([120.0]))
    mse = nn.MSELoss()
    best_state = None
    best_val = float("inf")
    best_epoch = 0
    history = []
    patience = 60
    last_rate_idx = 1

    for epoch in range(1, 501):
        model.train()
        losses = []
        for batch in loader:
            x_node, x_link, x_task, edge_a_hist, edge_a_future, y_active, y_rate, y_task = batch
            active_logits, rate_delta, task_pred = model(x_node, x_link, x_task, edge_a_hist, edge_a_future)
            rate_pred = rate_delta
            active_loss = bce(active_logits, y_active)
            active_mask = (y_active > 0.5).float()
            if active_mask.sum() > 0:
                rate_loss = (((rate_pred - y_rate) ** 2) * active_mask).sum() / active_mask.sum()
            else:
                rate_loss = torch.tensor(0.0)
            task_loss = mse(task_pred, y_task)
            loss = active_loss + 0.02 * rate_loss + 0.8 * task_loss
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach()))

        if epoch == 1 or epoch % 10 == 0:
            val_loss = evaluate_val_loss(model, val_loader, bce, mse)
            history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": val_loss})
            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if epoch - best_epoch > patience:
                break
    model.load_state_dict(best_state)
    return model, pd.DataFrame(history), {"best_epoch": best_epoch, "best_val_loss": best_val}


def evaluate_val_loss(model, loader, bce, mse):
    model.eval()
    vals = []
    with torch.no_grad():
        for batch in loader:
            x_node, x_link, x_task, edge_a_hist, edge_a_future, y_active, y_rate, y_task = batch
            active_logits, rate_delta, task_pred = model(x_node, x_link, x_task, edge_a_hist, edge_a_future)
            rate_pred = rate_delta
            active_loss = bce(active_logits, y_active)
            active_mask = (y_active > 0.5).float()
            rate_loss = (((rate_pred - y_rate) ** 2) * active_mask).sum() / active_mask.sum().clamp(min=1)
            task_loss = mse(task_pred, y_task)
            vals.append(float(active_loss + 0.02 * rate_loss + 0.8 * task_loss))
    return float(np.mean(vals))


def predict(model, arrays, idx, stats):
    ds = WorldModelDataset(arrays, idx, stats)
    loader = DataLoader(ds, batch_size=64, shuffle=False)
    probs = []
    rates = []
    tasks = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x_node, x_link, x_task, edge_a_hist, edge_a_future, *_ = batch
            active_logits, rate_delta, task_pred = model(x_node, x_link, x_task, edge_a_hist, edge_a_future)
            probs.append(torch.sigmoid(active_logits).cpu().numpy())
            rates.append(rate_delta.cpu().numpy())
            tasks.append(task_pred.cpu().numpy())
    task_norm = np.concatenate(tasks, axis=0)
    return {
        "active_prob": np.concatenate(probs, axis=0).astype(np.float32),
        "rate_pred": np.clip(inverse_normalize(np.concatenate(rates, axis=0), stats["y_link_rate"]), 0.0, None),
        "task_pred": inverse_normalize(task_norm, stats["y_task"]),
    }


def activity_metrics(y_true, prob, threshold):
    y_flat = y_true.reshape(-1).astype(int)
    p_flat = prob.reshape(-1)
    pred = p_flat >= threshold
    return {
        "precision": float(precision_score(y_flat, pred, zero_division=0)),
        "recall": float(recall_score(y_flat, pred, zero_division=0)),
        "f1": float(f1_score(y_flat, pred, zero_division=0)),
        "average_precision": float(average_precision_score(y_flat, p_flat)),
        "roc_auc": float(roc_auc_score(y_flat, p_flat)),
        "predicted_active_ratio": float(pred.mean()),
    }


def choose_threshold(y_val, prob_val):
    y_flat = y_val.reshape(-1).astype(int)
    p_flat = prob_val.reshape(-1)
    best = None
    rows = []
    for threshold in np.unique(np.concatenate([np.linspace(0.01, 0.99, 99), np.quantile(p_flat, np.linspace(0.8, 0.999, 50))])):
        pred = p_flat >= threshold
        row = {
            "threshold": float(threshold),
            "precision": float(precision_score(y_flat, pred, zero_division=0)),
            "recall": float(recall_score(y_flat, pred, zero_division=0)),
            "f1": float(f1_score(y_flat, pred, zero_division=0)),
            "predicted_active_ratio": float(pred.mean()),
        }
        rows.append(row)
        if best is None or (row["f1"], row["recall"]) > (best["f1"], best["recall"]):
            best = row
    return best, pd.DataFrame(rows)


def regression_metrics(y_true, y_pred, active_mask=None):
    if active_mask is not None:
        y_true = y_true[active_mask]
        y_pred = y_pred[active_mask]
    err = y_pred - y_true
    return {
        "mae": float(np.mean(np.abs(err))) if err.size else float("nan"),
        "rmse": float(np.sqrt(np.mean(err**2))) if err.size else float("nan"),
    }


def plot_training(history):
    path = FIGURE_DIR / "world_model_v0_training_curve.png"
    plt.figure(figsize=(7.2, 4.0))
    plt.plot(history["epoch"], history["train_loss"], label="train")
    plt.plot(history["epoch"], history["val_loss"], label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("World model v0 training curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()
    return path


def plot_compare(metrics_df):
    path = FIGURE_DIR / "world_model_v0_metric_compare.png"
    test = metrics_df[metrics_df["split"] == "test_seed_4"].set_index("model")
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.7))
    test[["activity_f1"]].plot(kind="bar", ax=axes[0], legend=False, color="#D62728")
    axes[0].set_title("link activity F1")
    test[["rate_all_rmse"]].plot(kind="bar", ax=axes[1], legend=False, color="#1F77B4")
    axes[1].set_title("link rate RMSE")
    test[["task_rmse"]].plot(kind="bar", ax=axes[2], legend=False, color="#2CA02C")
    axes[2].set_title("task RMSE")
    for ax in axes:
        ax.tick_params(axis="x", rotation=18)
        ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()
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
    best_threshold, threshold_df = choose_threshold(arrays["y_link_active"][val_idx], val_pred["active_prob"])
    threshold = best_threshold["threshold"]

    rows = []
    for split, idx, pred in [("val_seed_3", val_idx, val_pred), ("test_seed_4", test_idx, test_pred)]:
        act = activity_metrics(arrays["y_link_active"][idx], pred["active_prob"], threshold)
        rate_all = regression_metrics(arrays["y_link_rate"][idx], pred["rate_pred"])
        active_mask = arrays["y_link_active"][idx] > 0.5
        rate_active = regression_metrics(arrays["y_link_rate"][idx], pred["rate_pred"], active_mask=active_mask)
        task = regression_metrics(arrays["y_task"][idx], pred["task_pred"])
        rows.append(
            {
                "split": split,
                "model": "world_model_v0",
                "threshold": float(threshold),
                "activity_precision": act["precision"],
                "activity_recall": act["recall"],
                "activity_f1": act["f1"],
                "activity_ap": act["average_precision"],
                "activity_auc": act["roc_auc"],
                "rate_all_mae": rate_all["mae"],
                "rate_all_rmse": rate_all["rmse"],
                "rate_active_mae": rate_active["mae"],
                "rate_active_rmse": rate_active["rmse"],
                "task_mae": task["mae"],
                "task_rmse": task["rmse"],
            }
        )
    metrics_df = pd.DataFrame(rows)
    baseline_rows = load_baseline_rows()
    if baseline_rows:
        metrics_df = pd.concat([pd.DataFrame(baseline_rows), metrics_df], ignore_index=True)
    metrics_path = OUTPUT_DIR / "world_model_v0_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    history_path = OUTPUT_DIR / "world_model_v0_training_history.csv"
    history.to_csv(history_path, index=False, encoding="utf-8-sig")
    threshold_path = OUTPUT_DIR / "world_model_v0_threshold_curve.csv"
    threshold_df.to_csv(threshold_path, index=False, encoding="utf-8-sig")
    train_plot = plot_training(history)
    compare_plot = plot_compare(metrics_df)
    summary = {
        "dataset_dir": str(DATASET_DIR),
        "output_dir": str(OUTPUT_DIR),
        "split": {
            "train_seeds": [0, 1, 2],
            "val_seed": 3,
            "test_seed": 4,
            "train_samples": int(len(train_idx)),
            "val_samples": int(len(val_idx)),
            "test_samples": int(len(test_idx)),
        },
        "model": "EdgeActionWorldModel",
        "train_info": train_info,
        "selected_threshold": best_threshold,
        "metrics": rows,
        "outputs": {
            "metrics_csv": str(metrics_path),
            "training_history_csv": str(history_path),
            "threshold_curve_csv": str(threshold_path),
            "training_curve": str(train_plot),
            "metric_compare_plot": str(compare_plot),
        },
    }
    summary_path = OUTPUT_DIR / "world_model_v0_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = write_report(summary, metrics_df)
    summary["outputs"]["report_md"] = str(report_path)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def write_report(summary, metrics_df):
    lines = [
        "# World model v0 report",
        "",
        "## Goal",
        "",
        "This is the first integrated action-conditioned network world model. It consumes node history, link history, task history, and edge-level actions, then predicts future link activity, future link rate, and future task state.",
        "",
        "## Model",
        "",
        "`EdgeActionWorldModel` contains four encoders: node context, link history, task context, and edge-level action context. The fused edge representation feeds two link heads, while the global representation feeds a task head.",
        "",
        "Outputs:",
        "",
        "- link activity logits: future active / inactive state for every edge and horizon step.",
        "- link rate prediction: future `rate_sum` for every edge and horizon step.",
        "- task-state prediction: future task aggregate state.",
        "",
        "## Metrics",
        "",
        metrics_df.to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        "- This version is a method skeleton, not yet the final latent world model.",
        "- It already unifies physical-side node state, communication-side link state, task state, and edge-level scheduler actions.",
        "- The next upgrade is to add an explicit latent state `z_t` and autoregressive rollout, then compare multi-step stability with the current direct multi-horizon decoder.",
        "",
        "## Outputs",
        "",
    ]
    for key, value in summary["outputs"].items():
        lines.append(f"- {key}: `{value}`")
    path = OUTPUT_DIR / "world_model_v0_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def load_baseline_rows():
    rows = []
    edge_action_metrics = ROOT / "reports" / "edge_action_link_model_v0" / "edge_action_link_model_metrics.csv"
    edge_action_activity = ROOT / "reports" / "edge_action_link_model_v0" / "edge_action_activity_metrics.csv"
    structured_metrics = ROOT / "reports" / "structured_dual_branch_baseline_v0" / "structured_dual_branch_metrics.csv"
    if edge_action_metrics.exists() and edge_action_activity.exists():
        link_df = pd.read_csv(edge_action_metrics)
        activity_df = pd.read_csv(edge_action_activity)
        activity_test = activity_df[(activity_df["split"] == "test_seed_4") & (activity_df["model"] == "edge_action")]
        act = activity_test.iloc[0] if len(activity_test) else None
        for model in ["edge_state_action_ridge", "edge_action_ridge"]:
            test = link_df[(link_df["split"] == "test_seed_4") & (link_df["model"] == model) & (link_df["link_type"] == "all")]
            if len(test):
                item = test.iloc[0]
                rows.append(
                    {
                        "split": "test_seed_4",
                        "model": model,
                        "threshold": np.nan,
                        "activity_precision": float(act["precision"]) if act is not None and model == "edge_action_ridge" else np.nan,
                        "activity_recall": float(act["recall"]) if act is not None and model == "edge_action_ridge" else np.nan,
                        "activity_f1": float(act["f1"]) if act is not None and model == "edge_action_ridge" else np.nan,
                        "activity_ap": float(act["average_precision"]) if act is not None and model == "edge_action_ridge" else np.nan,
                        "activity_auc": float(act["roc_auc"]) if act is not None and model == "edge_action_ridge" else np.nan,
                        "rate_all_mae": float(item["mae"]),
                        "rate_all_rmse": float(item["rmse"]),
                        "rate_active_mae": np.nan,
                        "rate_active_rmse": np.nan,
                        "task_mae": np.nan,
                        "task_rmse": np.nan,
                    }
                )
    if structured_metrics.exists():
        df = pd.read_csv(structured_metrics)
        test = df[(df["split"] == "test_seed_4") & (df["model"] == "structured_state_action")]
        if len(test):
            item = test.iloc[0]
            rows.append(
                {
                    "split": "test_seed_4",
                    "model": "structured_state_action",
                    "threshold": np.nan,
                    "activity_precision": np.nan,
                    "activity_recall": np.nan,
                    "activity_f1": np.nan,
                    "activity_ap": np.nan,
                    "activity_auc": np.nan,
                    "rate_all_mae": np.nan,
                    "rate_all_rmse": float(item.get("link_rate_by_type_rmse", np.nan)),
                    "rate_active_mae": np.nan,
                    "rate_active_rmse": np.nan,
                    "task_mae": float(item.get("task_state_mae", np.nan)),
                    "task_rmse": float(item.get("task_state_rmse", np.nan)),
                }
            )
    return rows


if __name__ == "__main__":
    main()
