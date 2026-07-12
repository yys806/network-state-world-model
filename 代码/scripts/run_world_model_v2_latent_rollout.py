import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from run_world_model_v0 import (
    DATASET_DIR,
    FIGURE_DIR,
    ROOT,
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
from run_world_model_v1_staged import focal_loss_with_logits


OUTPUT_DIR = ROOT / "reports" / "world_model_v2_latent_rollout"


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


class LatentRolloutWorldModel(nn.Module):
    """Deterministic latent rollout model for the first world-model-shaped baseline."""

    def __init__(
        self,
        node_dim,
        link_hist_dim,
        task_hist_dim,
        task_out_dim,
        edge_action_hist_dim,
        edge_action_step_dim,
        horizon,
        edge_src_idx,
        edge_dst_idx,
        valid_edge_node,
        hidden=64,
        latent=64,
    ):
        super().__init__()
        self.horizon = horizon
        self.task_out_dim = task_out_dim
        self.register_buffer("edge_src_idx", torch.as_tensor(edge_src_idx, dtype=torch.long))
        self.register_buffer("edge_dst_idx", torch.as_tensor(edge_dst_idx, dtype=torch.long))
        self.register_buffer("valid_edge_node", torch.as_tensor(valid_edge_node, dtype=torch.float32))

        self.node_proj = nn.Sequential(nn.Linear(node_dim, hidden), nn.ReLU(), nn.LayerNorm(hidden))
        self.link_hist_proj = nn.Sequential(nn.Linear(link_hist_dim, hidden), nn.ReLU(), nn.LayerNorm(hidden))
        self.edge_action_hist_proj = nn.Sequential(
            nn.Linear(edge_action_hist_dim, hidden), nn.ReLU(), nn.LayerNorm(hidden)
        )
        self.edge_action_step_proj = nn.Sequential(
            nn.Linear(edge_action_step_dim, hidden), nn.ReLU(), nn.LayerNorm(hidden)
        )
        self.task_hist_proj = nn.Sequential(nn.Linear(task_hist_dim, hidden), nn.ReLU(), nn.LayerNorm(hidden))
        self.edge_init = nn.Sequential(
            nn.Linear(hidden * 5, hidden * 2),
            nn.ReLU(),
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, latent),
            nn.Tanh(),
        )
        self.edge_dynamics = nn.GRUCell(hidden, latent)
        self.task_dynamics = nn.GRUCell(hidden, latent)
        self.task_init = nn.Sequential(nn.Linear(hidden * 2, latent), nn.Tanh())
        self.active_head = nn.Linear(latent, 1)
        self.rate_head = nn.Sequential(nn.Linear(latent, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.task_head = nn.Sequential(nn.Linear(latent * 2, hidden), nn.ReLU(), nn.Linear(hidden, task_out_dim))

    def gather_endpoint_context(self, node_context, idx):
        safe_idx = idx.clamp(min=0)
        gathered = node_context[:, safe_idx, :]
        return gathered * (idx[None, :, None] >= 0).float()

    def encode_initial_state(self, x_node, x_link, x_task, edge_a_hist):
        node_context = self.node_proj(x_node.mean(dim=1))
        src_context = self.gather_endpoint_context(node_context, self.edge_src_idx)
        dst_context = self.gather_endpoint_context(node_context, self.edge_dst_idx)
        edge_hist = x_link.permute(0, 2, 1, 3).reshape(x_link.shape[0], x_link.shape[2], -1)
        edge_action_hist = edge_a_hist.permute(0, 2, 1, 3).reshape(edge_a_hist.shape[0], edge_a_hist.shape[2], -1)
        edge_state = self.link_hist_proj(edge_hist)
        edge_action = self.edge_action_hist_proj(edge_action_hist)
        task_context = self.task_hist_proj(x_task.reshape(x_task.shape[0], -1))
        task_edge = task_context[:, None, :].expand_as(edge_state)
        valid = self.valid_edge_node[None, :, None]
        edge_z = self.edge_init(
            torch.cat([edge_state, edge_action, src_context, dst_context, task_edge], dim=-1)
        )
        edge_z = edge_z * valid
        node_global = node_context.mean(dim=1)
        task_z = self.task_init(torch.cat([node_global, task_context], dim=-1))
        return edge_z, task_z

    def forward(self, x_node, x_link, x_task, edge_a_hist, edge_a_future):
        edge_z, task_z = self.encode_initial_state(x_node, x_link, x_task, edge_a_hist)
        active_logits, rates, tasks = [], [], []
        for step in range(self.horizon):
            action_step = self.edge_action_step_proj(edge_a_future[:, step, :, :])
            bsz, num_edges, _ = action_step.shape
            edge_z = self.edge_dynamics(action_step.reshape(bsz * num_edges, -1), edge_z.reshape(bsz * num_edges, -1))
            edge_z = edge_z.reshape(bsz, num_edges, -1) * self.valid_edge_node[None, :, None]
            action_global = action_step.mean(dim=1)
            task_z = self.task_dynamics(action_global, task_z)
            edge_global = edge_z.mean(dim=1)
            active_logits.append(self.active_head(edge_z).squeeze(-1))
            rates.append(self.rate_head(edge_z).squeeze(-1))
            tasks.append(self.task_head(torch.cat([task_z, edge_global], dim=-1)))
        return torch.stack(active_logits, dim=1), torch.stack(rates, dim=1), torch.stack(tasks, dim=1)


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


def make_model(arrays):
    return LatentRolloutWorldModel(
        node_dim=arrays["x_node"].shape[3],
        link_hist_dim=arrays["x_link"].shape[1] * arrays["x_link"].shape[3],
        task_hist_dim=arrays["x_task"].shape[1] * arrays["x_task"].shape[2],
        task_out_dim=arrays["y_task"].shape[2],
        edge_action_hist_dim=arrays["edge_a_hist"].shape[1] * arrays["edge_a_hist"].shape[3],
        edge_action_step_dim=arrays["edge_a_future"].shape[3],
        horizon=arrays["y_link_rate"].shape[1],
        edge_src_idx=arrays["edge_src_idx"],
        edge_dst_idx=arrays["edge_dst_idx"],
        valid_edge_node=arrays["valid_edge_node"],
        hidden=64,
        latent=64,
    )


def compute_loss(active_logits, rate_pred, task_pred, y_active, y_rate, y_task, stage):
    mse = nn.MSELoss()
    if stage == "activity":
        return focal_loss_with_logits(active_logits, y_active, alpha=0.88, gamma=2.0)
    if stage == "rate":
        weights = 1.0 + 90.0 * y_active
        return (((rate_pred - y_rate) ** 2) * weights).mean()
    if stage == "task":
        return mse(task_pred, y_task)
    activity = focal_loss_with_logits(active_logits, y_active, alpha=0.88, gamma=2.0)
    weights = 1.0 + 60.0 * y_active
    rate = (((rate_pred - y_rate) ** 2) * weights).mean()
    task = mse(task_pred, y_task)
    return activity + 0.35 * rate + 0.8 * task


def run_epoch(model, loader, opt, stage):
    model.train()
    losses = []
    for batch in loader:
        x_node, x_link, x_task, edge_a_hist, edge_a_future, y_active, y_rate, y_task = batch
        active_logits, rate_pred, task_pred = model(x_node, x_link, x_task, edge_a_hist, edge_a_future)
        loss = compute_loss(active_logits, rate_pred, task_pred, y_active, y_rate, y_task, stage)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        losses.append(float(loss.detach()))
    return float(np.mean(losses))


def val_loss(model, loader, stage):
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            x_node, x_link, x_task, edge_a_hist, edge_a_future, y_active, y_rate, y_task = batch
            active_logits, rate_pred, task_pred = model(x_node, x_link, x_task, edge_a_hist, edge_a_future)
            losses.append(float(compute_loss(active_logits, rate_pred, task_pred, y_active, y_rate, y_task, stage)))
    return float(np.mean(losses))


def train_model(arrays, train_idx, val_idx, stats):
    torch.manual_seed(42)
    model = make_model(arrays)
    train_ds = WorldModelDataset(arrays, train_idx, stats)
    val_ds = WorldModelDataset(arrays, val_idx, stats)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    stages = [
        ("activity", 80, 1e-3),
        ("rate", 120, 8e-4),
        ("task", 100, 8e-4),
        ("joint", 120, 5e-4),
    ]
    history = []
    best_state = None
    best_val = float("inf")
    global_epoch = 0
    for stage, max_epochs, lr in stages:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        patience = 30
        best_stage_epoch = 0
        best_stage_val = float("inf")
        for epoch in range(1, max_epochs + 1):
            global_epoch += 1
            train_loss = run_epoch(model, train_loader, opt, stage)
            if epoch == 1 or epoch % 10 == 0:
                current_val = val_loss(model, val_loader, stage)
                history.append(
                    {
                        "global_epoch": global_epoch,
                        "stage": stage,
                        "stage_epoch": epoch,
                        "train_loss": train_loss,
                        "val_loss": current_val,
                    }
                )
                if current_val < best_stage_val:
                    best_stage_val = current_val
                    best_stage_epoch = epoch
                if stage == "joint" and current_val < best_val:
                    best_val = current_val
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


def plot_history(history):
    path = FIGURE_DIR / "world_model_v2_latent_rollout_training_curve.png"
    plt.figure(figsize=(8.0, 4.2))
    for stage, part in history.groupby("stage"):
        plt.plot(part["global_epoch"], part["val_loss"], marker="o", label=stage)
    plt.xlabel("global epoch")
    plt.ylabel("validation loss")
    plt.title("World model v2 latent rollout training")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()
    return path


def plot_compare(metrics_df):
    path = FIGURE_DIR / "world_model_v2_latent_rollout_metric_compare.png"
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


def write_report(summary, metrics_df):
    lines = [
        "# World model v2 latent rollout report",
        "",
        "## Goal",
        "",
        "This experiment upgrades the integrated v1 interface from direct multi-horizon decoding to deterministic latent rollout. The purpose is to make the model form closer to a world model: encode history into an explicit latent state, update the latent state with future edge-level actions, and decode each future step.",
        "",
        "## Literature-backed design",
        "",
        "- Dreamer learns behavior through latent imagination, so v2 introduces an explicit latent state and action-conditioned rollout instead of one-shot decoding.",
        "- STGCN-style spatio-temporal forecasting motivates using endpoint node context and edge history together, although this version is still a lightweight message-free encoder.",
        "- Focal loss is retained for sparse link activity detection, and staged training is retained for multi-task stability.",
        "",
        "## Model interface",
        "",
        "`z_e,t = Enc(edge_history, src_node_history, dst_node_history, task_history, edge_action_history)`",
        "",
        "`z_e,t+k = GRU(z_e,t+k-1, a_e,t+k-1)`",
        "",
        "`Dec(z_e,t+k) -> link activity, link rate; Dec(mean_e z_e,t+k) -> task state`",
        "",
        "## Metrics",
        "",
        metrics_df.to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        "- v2 is the first implemented latent-rollout world-model baseline in this project.",
        "- If performance is weaker than specialized baselines, the result should be read as a structural step rather than a final accuracy claim.",
        "- The next technical upgrade should add real graph message passing and/or posterior-prior regularization, because the current latent state is deterministic and supervised only.",
        "",
        "## Outputs",
        "",
    ]
    for key, value in summary["outputs"].items():
        lines.append(f"- {key}: `{value}`")
    path = OUTPUT_DIR / "world_model_v2_latent_rollout_report.md"
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
    best_threshold, threshold_df = choose_threshold(arrays["y_link_active"][val_idx], val_pred["active_prob"])
    threshold = best_threshold["threshold"]
    rows = []
    for split, idx, pred in [("val_seed_3", val_idx, val_pred), ("test_seed_4", test_idx, test_pred)]:
        rows.append(
            {
                "split": split,
                "model": "world_model_v2_latent_rollout",
                "threshold": float(threshold),
                **evaluate(arrays, idx, pred, threshold),
            }
        )
    baseline_rows = load_baseline_rows()
    for report_name in ["world_model_v0", "world_model_v1_staged"]:
        p = ROOT / "reports" / report_name / f"{report_name}_metrics.csv"
        if p.exists():
            df = pd.read_csv(p)
            test = df[(df["split"] == "test_seed_4") & (df["model"] == report_name)]
            baseline_rows.extend(test.to_dict("records"))
    metrics_df = pd.concat([pd.DataFrame(baseline_rows), pd.DataFrame(rows)], ignore_index=True)
    metrics_path = OUTPUT_DIR / "world_model_v2_latent_rollout_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    history_path = OUTPUT_DIR / "world_model_v2_latent_rollout_training_history.csv"
    history.to_csv(history_path, index=False, encoding="utf-8-sig")
    threshold_path = OUTPUT_DIR / "world_model_v2_latent_rollout_threshold_curve.csv"
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
    summary_path = OUTPUT_DIR / "world_model_v2_latent_rollout_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = write_report(summary, metrics_df)
    summary["outputs"]["report_md"] = str(report_path)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
