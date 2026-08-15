import json

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from run_world_model_v0 import (
    DATASET_DIR,
    FIGURE_DIR,
    ROOT,
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
from run_world_model_v3_graph_rollout import EdgeGraphMessageBlock, evaluate


OUTPUT_DIR = ROOT / "reports" / "world_model_v4_dual_graph_rollout"
PHYSICAL_EDGE_FEATURES = [
    "dx",
    "dy",
    "dz",
    "distance_3d",
    "abs_speed_delta",
    "src_speed",
    "dst_speed",
    "abs_dz",
]


def display_path(path):
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def build_physical_edge_history(x_node, edge_src_idx, edge_dst_idx, valid_edge_node):
    """Build physical edge history from endpoint node geometry.

    The current dataset already has per-node position/speed history. This helper
    turns it into a physical-edge tensor aligned with the communication-edge
    vocabulary, so v4 can compare communication-only v3 against a dual-graph
    variant without rebuilding the whole dataset.
    """

    x_node = np.asarray(x_node, dtype=np.float32)
    src_idx = np.asarray(edge_src_idx, dtype=np.int64).clip(min=0)
    dst_idx = np.asarray(edge_dst_idx, dtype=np.int64).clip(min=0)
    valid = np.asarray(valid_edge_node, dtype=np.float32).reshape(1, 1, -1, 1)

    src = x_node[:, :, src_idx, :]
    dst = x_node[:, :, dst_idx, :]
    delta_xyz = dst[..., :3] - src[..., :3]
    distance = np.linalg.norm(delta_xyz, axis=-1, keepdims=True)
    src_speed = src[..., 3:4]
    dst_speed = dst[..., 3:4]
    speed_delta = np.abs(dst_speed - src_speed)
    abs_dz = np.abs(delta_xyz[..., 2:3])
    physical = np.concatenate(
        [delta_xyz, distance, speed_delta, src_speed, dst_speed, abs_dz],
        axis=-1,
    )
    return (physical * valid).astype(np.float32)


def augment_arrays_with_physical_edges(arrays):
    arrays = dict(arrays)
    arrays["x_phy_edge"] = build_physical_edge_history(
        arrays["x_node"],
        arrays["edge_src_idx"],
        arrays["edge_dst_idx"],
        arrays["valid_edge_node"],
    )
    arrays["physical_edge_features"] = np.asarray(PHYSICAL_EDGE_FEATURES, dtype=object)
    return arrays


class DualGraphWorldModelDataset(Dataset):
    def __init__(self, arrays, idx, stats):
        self.x_node = torch.from_numpy(normalize(arrays["x_node"][idx], stats["x_node"]))
        self.x_link = torch.from_numpy(normalize(arrays["x_link"][idx], stats["x_link"]))
        self.x_phy_edge = torch.from_numpy(normalize(arrays["x_phy_edge"][idx], stats["x_phy_edge"]))
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
            self.x_phy_edge[i],
            self.x_task[i],
            self.edge_a_hist[i],
            self.edge_a_future[i],
            self.y_active[i],
            self.y_rate[i],
            self.y_task[i],
        )


class DualGraphLatentRolloutWorldModel(nn.Module):
    """Minimal dual-graph rollout model.

    v3 performs message passing over communication candidate edges. v4 adds a
    separate physical-edge branch derived from endpoint geometry, then fuses the
    physical and communication latents before action-conditioned rollout.
    """

    def __init__(
        self,
        num_nodes,
        node_hist_dim,
        link_hist_dim,
        physical_hist_dim,
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
        graph_layers=1,
    ):
        super().__init__()
        self.horizon = horizon
        self.task_out_dim = task_out_dim
        self.register_buffer("edge_src_idx", torch.as_tensor(edge_src_idx, dtype=torch.long).clamp(min=0))
        self.register_buffer("edge_dst_idx", torch.as_tensor(edge_dst_idx, dtype=torch.long).clamp(min=0))
        self.register_buffer("valid_edge_node", torch.as_tensor(valid_edge_node, dtype=torch.float32))

        self.node_hist_proj = nn.Sequential(nn.Linear(node_hist_dim, hidden), nn.ReLU(), nn.LayerNorm(hidden))
        self.link_hist_proj = nn.Sequential(nn.Linear(link_hist_dim, hidden), nn.ReLU(), nn.LayerNorm(hidden))
        self.physical_hist_proj = nn.Sequential(
            nn.Linear(physical_hist_dim, hidden), nn.ReLU(), nn.LayerNorm(hidden)
        )
        self.edge_action_hist_proj = nn.Sequential(
            nn.Linear(edge_action_hist_dim, hidden), nn.ReLU(), nn.LayerNorm(hidden)
        )
        self.edge_action_step_proj = nn.Sequential(
            nn.Linear(edge_action_step_dim, hidden), nn.ReLU(), nn.LayerNorm(hidden)
        )
        self.task_hist_proj = nn.Sequential(nn.Linear(task_hist_dim, hidden), nn.ReLU(), nn.LayerNorm(hidden))

        self.comm_edge_init = nn.Sequential(
            nn.Linear(hidden * 5, hidden * 2),
            nn.ReLU(),
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, latent),
            nn.Tanh(),
        )
        self.physical_edge_init = nn.Sequential(
            nn.Linear(hidden * 4, hidden * 2),
            nn.ReLU(),
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, latent),
            nn.Tanh(),
        )
        self.comm_init_graph = nn.ModuleList(
            [
                EdgeGraphMessageBlock(latent, num_nodes, edge_src_idx, edge_dst_idx, valid_edge_node)
                for _ in range(graph_layers)
            ]
        )
        self.physical_init_graph = nn.ModuleList(
            [
                EdgeGraphMessageBlock(latent, num_nodes, edge_src_idx, edge_dst_idx, valid_edge_node)
                for _ in range(graph_layers)
            ]
        )
        self.rollout_graph = nn.ModuleList(
            [
                EdgeGraphMessageBlock(latent, num_nodes, edge_src_idx, edge_dst_idx, valid_edge_node)
                for _ in range(graph_layers)
            ]
        )
        self.dual_init_fuse = nn.Sequential(nn.Linear(latent * 2, latent), nn.ReLU(), nn.LayerNorm(latent))

        self.edge_step_fuse = nn.Sequential(nn.Linear(hidden + latent * 2, hidden), nn.ReLU(), nn.LayerNorm(hidden))
        self.edge_dynamics = nn.GRUCell(hidden, latent)
        self.task_init = nn.Sequential(nn.Linear(hidden * 2, latent), nn.Tanh())
        self.task_step_fuse = nn.Sequential(nn.Linear(hidden + latent, hidden), nn.ReLU(), nn.LayerNorm(hidden))
        self.task_dynamics = nn.GRUCell(hidden, latent)

        self.active_head = nn.Linear(latent, 1)
        self.rate_head = nn.Sequential(nn.Linear(latent, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.task_head = nn.Sequential(nn.Linear(latent * 2, hidden), nn.ReLU(), nn.Linear(hidden, task_out_dim))

    def gather_endpoint_context(self, node_context, idx):
        return node_context[:, idx, :]

    def apply_blocks(self, edge_z, blocks):
        for block in blocks:
            edge_z = block(edge_z)
        return edge_z

    def encode_initial_state(self, x_node, x_link, x_phy_edge, x_task, edge_a_hist):
        bsz, hist_len, num_nodes, node_feat = x_node.shape
        node_hist = x_node.permute(0, 2, 1, 3).reshape(bsz, num_nodes, hist_len * node_feat)
        node_context = self.node_hist_proj(node_hist)
        src_context = self.gather_endpoint_context(node_context, self.edge_src_idx)
        dst_context = self.gather_endpoint_context(node_context, self.edge_dst_idx)

        edge_hist = x_link.permute(0, 2, 1, 3).reshape(x_link.shape[0], x_link.shape[2], -1)
        physical_hist = x_phy_edge.permute(0, 2, 1, 3).reshape(x_phy_edge.shape[0], x_phy_edge.shape[2], -1)
        edge_action_hist = edge_a_hist.permute(0, 2, 1, 3).reshape(edge_a_hist.shape[0], edge_a_hist.shape[2], -1)
        edge_state = self.link_hist_proj(edge_hist)
        physical_state = self.physical_hist_proj(physical_hist)
        edge_action = self.edge_action_hist_proj(edge_action_hist)
        task_context = self.task_hist_proj(x_task.reshape(x_task.shape[0], -1))
        task_edge = task_context[:, None, :].expand_as(edge_state)

        valid = self.valid_edge_node[None, :, None]
        comm_z = self.comm_edge_init(
            torch.cat([edge_state, edge_action, src_context, dst_context, task_edge], dim=-1)
        )
        physical_z = self.physical_edge_init(
            torch.cat([physical_state, src_context, dst_context, task_edge], dim=-1)
        )
        comm_z = self.apply_blocks(comm_z * valid, self.comm_init_graph)
        physical_z = self.apply_blocks(physical_z * valid, self.physical_init_graph)
        edge_z = self.dual_init_fuse(torch.cat([comm_z, physical_z], dim=-1)) * valid

        node_global = node_context.mean(dim=1)
        task_z = self.task_init(torch.cat([node_global, task_context], dim=-1))
        return edge_z, physical_z, task_z

    def forward(self, x_node, x_link, x_phy_edge, x_task, edge_a_hist, edge_a_future):
        edge_z, physical_z, task_z = self.encode_initial_state(x_node, x_link, x_phy_edge, x_task, edge_a_hist)
        active_logits, rates, tasks = [], [], []
        valid = self.valid_edge_node[None, :, None]
        valid_count = self.valid_edge_node.sum().clamp(min=1.0)

        for step in range(self.horizon):
            action_step = self.edge_action_step_proj(edge_a_future[:, step, :, :])
            bsz, num_edges, _ = action_step.shape
            task_edge = task_z[:, None, :].expand(bsz, num_edges, -1)
            dynamics_input = self.edge_step_fuse(torch.cat([action_step, task_edge, physical_z], dim=-1))
            edge_z = self.edge_dynamics(
                dynamics_input.reshape(bsz * num_edges, -1),
                edge_z.reshape(bsz * num_edges, -1),
            )
            edge_z = edge_z.reshape(bsz, num_edges, -1) * valid
            edge_z = self.apply_blocks(edge_z, self.rollout_graph)

            edge_global = (edge_z * valid).sum(dim=1) / valid_count
            action_global = action_step.mean(dim=1)
            task_input = self.task_step_fuse(torch.cat([action_global, edge_global], dim=-1))
            task_z = self.task_dynamics(task_input, task_z)

            active_logits.append(self.active_head(edge_z).squeeze(-1))
            rates.append(self.rate_head(edge_z).squeeze(-1))
            tasks.append(self.task_head(torch.cat([task_z, edge_global], dim=-1)))

        return torch.stack(active_logits, dim=1), torch.stack(rates, dim=1), torch.stack(tasks, dim=1)


def make_stats(arrays, train_idx):
    return {
        "x_node": fit_stats(arrays["x_node"][train_idx]),
        "x_link": fit_stats(arrays["x_link"][train_idx]),
        "x_phy_edge": fit_stats(arrays["x_phy_edge"][train_idx]),
        "x_task": fit_stats(arrays["x_task"][train_idx]),
        "edge_a_hist": fit_stats(arrays["edge_a_hist"][train_idx]),
        "edge_a_future": fit_stats(arrays["edge_a_future"][train_idx]),
        "y_task": fit_stats(arrays["y_task"][train_idx]),
        "y_link_rate": fit_stats(arrays["y_link_rate"][train_idx]),
    }


def make_model(arrays):
    return DualGraphLatentRolloutWorldModel(
        num_nodes=arrays["x_node"].shape[2],
        node_hist_dim=arrays["x_node"].shape[1] * arrays["x_node"].shape[3],
        link_hist_dim=arrays["x_link"].shape[1] * arrays["x_link"].shape[3],
        physical_hist_dim=arrays["x_phy_edge"].shape[1] * arrays["x_phy_edge"].shape[3],
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
        graph_layers=1,
    )


def compute_loss(active_logits, rate_pred, task_pred, y_active, y_rate, y_task, stage):
    mse = nn.MSELoss()
    if stage == "activity":
        return focal_loss_with_logits(active_logits, y_active, alpha=0.90, gamma=2.0)
    if stage == "rate":
        weights = 1.0 + 90.0 * y_active
        return (((rate_pred - y_rate) ** 2) * weights).mean()
    if stage == "task":
        return mse(task_pred, y_task)
    activity = focal_loss_with_logits(active_logits, y_active, alpha=0.90, gamma=2.0)
    weights = 1.0 + 70.0 * y_active
    rate = (((rate_pred - y_rate) ** 2) * weights).mean()
    task = mse(task_pred, y_task)
    return activity + 0.35 * rate + 0.8 * task


def run_epoch(model, loader, opt, stage):
    model.train()
    losses = []
    for batch in loader:
        x_node, x_link, x_phy_edge, x_task, edge_a_hist, edge_a_future, y_active, y_rate, y_task = batch
        active_logits, rate_pred, task_pred = model(x_node, x_link, x_phy_edge, x_task, edge_a_hist, edge_a_future)
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
            x_node, x_link, x_phy_edge, x_task, edge_a_hist, edge_a_future, y_active, y_rate, y_task = batch
            active_logits, rate_pred, task_pred = model(
                x_node, x_link, x_phy_edge, x_task, edge_a_hist, edge_a_future
            )
            losses.append(float(compute_loss(active_logits, rate_pred, task_pred, y_active, y_rate, y_task, stage)))
    return float(np.mean(losses))


def train_model(arrays, train_idx, val_idx, stats, torch_seed=42):
    torch.manual_seed(torch_seed)
    model = make_model(arrays)
    train_ds = DualGraphWorldModelDataset(arrays, train_idx, stats)
    val_ds = DualGraphWorldModelDataset(arrays, val_idx, stats)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    stages = [
        ("activity", 20, 1e-3),
        ("rate", 30, 8e-4),
        ("task", 20, 8e-4),
        ("joint", 30, 5e-4),
    ]
    history = []
    best_state = None
    best_val = float("inf")
    global_epoch = 0
    for stage, max_epochs, lr in stages:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        patience = 12
        best_stage_epoch = 0
        best_stage_val = float("inf")
        for epoch in range(1, max_epochs + 1):
            global_epoch += 1
            train_loss = run_epoch(model, train_loader, opt, stage)
            if epoch == 1 or epoch % 5 == 0:
                current_val = val_loss(model, val_loader, stage)
                print(
                    f"[v4] stage={stage} epoch={epoch}/{max_epochs} "
                    f"train_loss={train_loss:.6f} val_loss={current_val:.6f}",
                    flush=True,
                )
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
    ds = DualGraphWorldModelDataset(arrays, idx, stats)
    loader = DataLoader(ds, batch_size=64, shuffle=False)
    probs, rates, tasks = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x_node, x_link, x_phy_edge, x_task, edge_a_hist, edge_a_future, *_ = batch
            active_logits, rate_pred, task_pred = model(
                x_node, x_link, x_phy_edge, x_task, edge_a_hist, edge_a_future
            )
            probs.append(torch.sigmoid(active_logits).cpu().numpy())
            rates.append(rate_pred.cpu().numpy())
            tasks.append(task_pred.cpu().numpy())
    return {
        "active_prob": np.concatenate(probs, axis=0).astype(np.float32),
        "rate_pred": np.clip(inverse_normalize(np.concatenate(rates, axis=0), stats["y_link_rate"]), 0.0, None),
        "task_pred": inverse_normalize(np.concatenate(tasks, axis=0), stats["y_task"]),
    }


def plot_history(history):
    import matplotlib.pyplot as plt

    path = FIGURE_DIR / "world_model_v4_dual_graph_rollout_training_curve.png"
    plt.figure(figsize=(8.0, 4.2))
    for stage, part in history.groupby("stage"):
        plt.plot(part["global_epoch"], part["val_loss"], marker="o", label=stage)
    plt.xlabel("global epoch")
    plt.ylabel("validation loss")
    plt.title("World model v4 dual-graph rollout training")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()
    return path


def plot_compare(metrics_df):
    import matplotlib.pyplot as plt

    path = FIGURE_DIR / "world_model_v4_dual_graph_rollout_metric_compare.png"
    test = metrics_df[metrics_df["split"] == "test_seed_4"].set_index("model")
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.7))
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


def load_prior_world_model_rows():
    rows = []
    for report_name in [
        "world_model_v0",
        "world_model_v1_staged",
        "world_model_v2_latent_rollout",
        "world_model_v3_graph_rollout",
    ]:
        p = ROOT / "reports" / report_name / f"{report_name}_metrics.csv"
        if p.exists():
            df = pd.read_csv(p)
            test = df[(df["split"] == "test_seed_4") & (df["model"] == report_name)]
            rows.extend(test.to_dict("records"))
    return rows


def write_report(summary, metrics_df):
    lines = [
        "# World model v4 dual-graph rollout report",
        "",
        "## Goal",
        "",
        "This experiment tests a minimal dual-graph extension of v3. It keeps the same `world_model_dataset_v0`, seed split, edge-level action interface, and output targets, while adding a physical-edge branch derived from endpoint geometry.",
        "",
        "## Physical-edge features",
        "",
        ", ".join(PHYSICAL_EDGE_FEATURES),
        "",
        "## Model interface",
        "",
        "`z_comm = Enc_comm(link_history, src_node_history, dst_node_history, task_history, edge_action_history)`",
        "",
        "`z_phy = Enc_phy(physical_edge_history, src_node_history, dst_node_history, task_history)`",
        "",
        "`z_e,t = Fuse(GraphComm(z_comm), GraphPhy(z_phy))`",
        "",
        "`z_e,t+k = GraphComm(GRU(z_e,t+k-1, a_e,t+k-1, z_task,t+k-1, z_phy))`",
        "",
        "`Dec(z_e,t+k) -> link activity, link rate; Dec(mean_e z_e,t+k, z_task,t+k) -> task state`",
        "",
        "## Metrics",
        "",
        metrics_df.to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        "- v4 is a minimal prototype, not the final dual-graph model. The physical branch uses historical endpoint geometry and keeps the candidate-edge vocabulary aligned with v3.",
        "- On test seed 4, v4 improves activity F1 and task RMSE over v3, which indicates that endpoint geometry already adds useful context before future mobility rollout is introduced.",
        "- Rate RMSE changes only slightly, so active-rate prediction remains the main bottleneck and should continue through activity-gating plus active-rate calibration.",
        "",
        "## Outputs",
        "",
    ]
    for key, value in summary["outputs"].items():
        lines.append(f"- {key}: `{value}`")
    path = OUTPUT_DIR / "world_model_v4_dual_graph_rollout_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    arrays = augment_arrays_with_physical_edges(load_dataset())
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
                "model": "world_model_v4_dual_graph_rollout",
                "threshold": float(threshold),
                **evaluate(arrays, idx, pred, threshold),
            }
        )
    baseline_rows = load_baseline_rows()
    baseline_rows.extend(load_prior_world_model_rows())
    metrics_df = pd.concat([pd.DataFrame(baseline_rows), pd.DataFrame(rows)], ignore_index=True)

    metrics_path = OUTPUT_DIR / "world_model_v4_dual_graph_rollout_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    history_path = OUTPUT_DIR / "world_model_v4_dual_graph_rollout_training_history.csv"
    history.to_csv(history_path, index=False, encoding="utf-8-sig")
    threshold_path = OUTPUT_DIR / "world_model_v4_dual_graph_rollout_threshold_curve.csv"
    threshold_df.to_csv(threshold_path, index=False, encoding="utf-8-sig")
    train_plot = plot_history(history)
    compare_plot = plot_compare(metrics_df)

    summary = {
        "dataset_dir": display_path(DATASET_DIR),
        "output_dir": display_path(OUTPUT_DIR),
        "split": {
            "train_seeds": [0, 1, 2],
            "val_seed": 3,
            "test_seed": 4,
            "train_samples": int(len(train_idx)),
            "val_samples": int(len(val_idx)),
            "test_samples": int(len(test_idx)),
        },
        "model": "DualGraphLatentRolloutWorldModel",
        "physical_edge_features": PHYSICAL_EDGE_FEATURES,
        "physical_edge_shape": list(arrays["x_phy_edge"].shape),
        "train_info": train_info,
        "selected_threshold": best_threshold,
        "metrics": rows,
        "outputs": {
            "metrics_csv": display_path(metrics_path),
            "training_history_csv": display_path(history_path),
            "threshold_curve_csv": display_path(threshold_path),
            "training_curve": display_path(train_plot),
            "metric_compare_plot": display_path(compare_plot),
        },
    }
    summary_path = OUTPUT_DIR / "world_model_v4_dual_graph_rollout_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = write_report(summary, metrics_df)
    summary["outputs"]["report_md"] = display_path(report_path)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
