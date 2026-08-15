import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from run_world_model_v0 import (
    DATASET_DIR,
    FIGURE_DIR,
    ROOT,
    EdgeActionWorldModel,
    activity_metrics,
    choose_threshold,
    fit_stats,
    inverse_normalize,
    load_baseline_rows,
    load_dataset,
    normalize,
    regression_metrics,
    split_by_seed,
)


OUTPUT_DIR = ROOT / "reports" / "world_model_v1_staged"


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


def make_stats(arrays, train_idx):
    return {
        "x_node": fit_stats(arrays["x_node"][train_idx]),
        "x_link": fit_stats(arrays["x_link"][train_idx]),
        "x_task": fit_stats(arrays["x_task"][train_idx]),
        "edge_a_hist": fit_stats(arrays["edge_a_hist"][train_idx]),
        "edge_a_future": fit_stats(arrays["edge_a_future"][train_idx]),
        "y_task": fit_stats(arrays["y_task"][train_idx]),
        "y_link_rate": fit_stats(arrays["y_link_rate"][train_idx]),
    }


def focal_loss_with_logits(logits, targets, alpha=0.75, gamma=2.0):
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    prob = torch.sigmoid(logits)
    p_t = prob * targets + (1 - prob) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    return (alpha_t * (1 - p_t).pow(gamma) * bce).mean()


def make_model(arrays):
    return EdgeActionWorldModel(
        node_dim=arrays["x_node"].shape[3],
        link_dim=arrays["x_link"].shape[1] * arrays["x_link"].shape[3],
        task_context_dim=arrays["x_task"].shape[1] * arrays["x_task"].shape[2],
        task_out_dim=arrays["y_task"].shape[2],
        edge_action_dim=arrays["edge_a_hist"].shape[1] * arrays["edge_a_hist"].shape[3]
        + arrays["edge_a_future"].shape[1] * arrays["edge_a_future"].shape[3],
        horizon=arrays["y_link_rate"].shape[1],
        hidden=64,
    )


def run_epoch(model, loader, opt, stage):
    model.train()
    losses = []
    mse = nn.MSELoss()
    for batch in loader:
        x_node, x_link, x_task, edge_a_hist, edge_a_future, y_active, y_rate, y_task = batch
        active_logits, rate_pred, task_pred = model(x_node, x_link, x_task, edge_a_hist, edge_a_future)
        if stage == "activity":
            loss = focal_loss_with_logits(active_logits, y_active, alpha=0.85, gamma=2.0)
        elif stage == "rate":
            weights = 1.0 + 80.0 * y_active
            loss = (((rate_pred - y_rate) ** 2) * weights).mean()
        elif stage == "task":
            loss = mse(task_pred, y_task)
        else:
            activity = focal_loss_with_logits(active_logits, y_active, alpha=0.85, gamma=2.0)
            weights = 1.0 + 60.0 * y_active
            rate = (((rate_pred - y_rate) ** 2) * weights).mean()
            task = mse(task_pred, y_task)
            loss = activity + 0.5 * rate + 0.8 * task
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        losses.append(float(loss.detach()))
    return float(np.mean(losses))


def val_stage_loss(model, loader, stage):
    model.eval()
    losses = []
    mse = nn.MSELoss()
    with torch.no_grad():
        for batch in loader:
            x_node, x_link, x_task, edge_a_hist, edge_a_future, y_active, y_rate, y_task = batch
            active_logits, rate_pred, task_pred = model(x_node, x_link, x_task, edge_a_hist, edge_a_future)
            if stage == "activity":
                loss = focal_loss_with_logits(active_logits, y_active, alpha=0.85, gamma=2.0)
            elif stage == "rate":
                weights = 1.0 + 80.0 * y_active
                loss = (((rate_pred - y_rate) ** 2) * weights).mean()
            elif stage == "task":
                loss = mse(task_pred, y_task)
            else:
                activity = focal_loss_with_logits(active_logits, y_active, alpha=0.85, gamma=2.0)
                weights = 1.0 + 60.0 * y_active
                rate = (((rate_pred - y_rate) ** 2) * weights).mean()
                task = mse(task_pred, y_task)
                loss = activity + 0.5 * rate + 0.8 * task
            losses.append(float(loss))
    return float(np.mean(losses))


def train_staged(arrays, train_idx, val_idx, stats):
    torch.manual_seed(42)
    model = make_model(arrays)
    train_ds = WorldModelDataset(arrays, train_idx, stats)
    val_ds = WorldModelDataset(arrays, val_idx, stats)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    stages = [
        ("activity", 120, 1e-3),
        ("rate", 160, 8e-4),
        ("task", 120, 8e-4),
        ("joint", 120, 5e-4),
    ]
    history = []
    best_state = None
    best_val = float("inf")
    global_epoch = 0
    for stage, max_epochs, lr in stages:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        patience = 35
        best_stage_epoch = 0
        best_stage_val = float("inf")
        for epoch in range(1, max_epochs + 1):
            global_epoch += 1
            train_loss = run_epoch(model, train_loader, opt, stage)
            if epoch == 1 or epoch % 10 == 0:
                val_loss = val_stage_loss(model, val_loader, stage)
                history.append(
                    {
                        "global_epoch": global_epoch,
                        "stage": stage,
                        "stage_epoch": epoch,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                    }
                )
                if val_loss < best_stage_val:
                    best_stage_val = val_loss
                    best_stage_epoch = epoch
                if stage == "joint" and val_loss < best_val:
                    best_val = val_loss
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                if epoch - best_stage_epoch > patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, pd.DataFrame(history), {"best_joint_val_loss": best_val}


def predict(model, arrays, idx, stats):
    ds = WorldModelDataset(arrays, idx, stats)
    loader = DataLoader(ds, batch_size=64, shuffle=False)
    probs, rates, tasks = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x_node, x_link, x_task, edge_a_hist, edge_a_future, *_ = batch
            active_logits, rate_pred, task_pred = model(x_node, x_link, x_task, edge_a_hist, edge_a_future)
            probs.append(torch.sigmoid(active_logits).cpu().numpy())
            rates.append(rate_pred.cpu().numpy())
            tasks.append(task_pred.cpu().numpy())
    return {
        "active_prob": np.concatenate(probs, axis=0).astype(np.float32),
        "rate_pred": np.clip(inverse_normalize(np.concatenate(rates, axis=0), stats["y_link_rate"]), 0.0, None),
        "task_pred": inverse_normalize(np.concatenate(tasks, axis=0), stats["y_task"]),
    }


def plot_history(history):
    path = FIGURE_DIR / "world_model_v1_staged_training_curve.png"
    plt.figure(figsize=(8.0, 4.2))
    for stage, part in history.groupby("stage"):
        plt.plot(part["global_epoch"], part["val_loss"], marker="o", label=stage)
    plt.xlabel("global epoch")
    plt.ylabel("validation loss")
    plt.title("World model v1 staged training")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()
    return path


def plot_compare(metrics_df):
    path = FIGURE_DIR / "world_model_v1_staged_metric_compare.png"
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


def evaluate(arrays, idx, pred, threshold):
    act = activity_metrics(arrays["y_link_active"][idx], pred["active_prob"], threshold)
    rate_all = regression_metrics(arrays["y_link_rate"][idx], pred["rate_pred"])
    active_mask = arrays["y_link_active"][idx] > 0.5
    rate_active = regression_metrics(arrays["y_link_rate"][idx], pred["rate_pred"], active_mask=active_mask)
    task = regression_metrics(arrays["y_task"][idx], pred["task_pred"])
    return {
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


def write_report(summary, metrics_df):
    lines = [
        "# World model v1 staged report",
        "",
        "## Goal",
        "",
        "This experiment keeps the integrated world-model input interface from v0, but changes the training procedure according to three literature-backed issues: extreme class imbalance, multi-task loss interference, and world-model latent-state development.",
        "",
        "## Literature-backed design",
        "",
        "- Link activity is extremely sparse, so the activity head uses focal loss, following Lin et al. (ICCV 2017).",
        "- Link activity, rate regression, and task regression have different scales, so v1 uses staged training instead of one naive joint loss; this follows the multi-task loss-balancing concern discussed by Kendall et al. (CVPR 2018).",
        "- The current model is still a direct multi-horizon decoder. It is a stepping stone toward Dreamer/RSSM-style latent rollout, following Hafner et al. (ICLR 2020).",
        "",
        "## Metrics",
        "",
        metrics_df.to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        "- v1 is intended to test whether training protocol fixes the v0 bottleneck before increasing model complexity.",
        "- If specialized baselines remain stronger, the next step should preserve the integrated dataset but move to explicit latent state and graph message passing.",
        "",
        "## Outputs",
        "",
    ]
    for key, value in summary["outputs"].items():
        lines.append(f"- {key}: `{value}`")
    path = OUTPUT_DIR / "world_model_v1_staged_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    arrays = load_dataset()
    train_idx, val_idx, test_idx = split_by_seed(arrays["sample_seed"])
    stats = make_stats(arrays, train_idx)
    model, history, train_info = train_staged(arrays, train_idx, val_idx, stats)
    val_pred = predict(model, arrays, val_idx, stats)
    test_pred = predict(model, arrays, test_idx, stats)
    best_threshold, threshold_df = choose_threshold(arrays["y_link_active"][val_idx], val_pred["active_prob"])
    threshold = best_threshold["threshold"]
    rows = []
    for split, idx, pred in [("val_seed_3", val_idx, val_pred), ("test_seed_4", test_idx, test_pred)]:
        row = {
            "split": split,
            "model": "world_model_v1_staged",
            "threshold": float(threshold),
            **evaluate(arrays, idx, pred, threshold),
        }
        rows.append(row)
    baseline_rows = load_baseline_rows()
    metrics_df = pd.concat([pd.DataFrame(baseline_rows), pd.DataFrame(rows)], ignore_index=True)
    metrics_path = OUTPUT_DIR / "world_model_v1_staged_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    history_path = OUTPUT_DIR / "world_model_v1_staged_training_history.csv"
    history.to_csv(history_path, index=False, encoding="utf-8-sig")
    threshold_path = OUTPUT_DIR / "world_model_v1_staged_threshold_curve.csv"
    threshold_df.to_csv(threshold_path, index=False, encoding="utf-8-sig")
    train_plot = plot_history(history)
    compare_plot = plot_compare(metrics_df)
    summary = {
        "dataset_dir": str(DATASET_DIR),
        "output_dir": str(OUTPUT_DIR),
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
    summary_path = OUTPUT_DIR / "world_model_v1_staged_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = write_report(summary, metrics_df)
    summary["outputs"]["report_md"] = str(report_path)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
