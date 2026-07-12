import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "datasets" / "dataset_multiseed_v0"
ACTION_DIR = ROOT / "datasets" / "strict_action_v0"
OUTPUT_DIR = ROOT / "reports" / "structured_dual_branch_baseline_v0"
FIGURE_DIR = ROOT / "figures"

LINK_TYPES = ["U2I", "V2I", "V2U"]
NODE_TYPES = ["cloud", "rsu", "uav", "vehicle"]


def load_arrays():
    with np.load(DATASET_DIR / "dataset_multiseed_v0_samples.npz", allow_pickle=True) as data:
        arrays = {key: data[key] for key in data.files}
    with np.load(ACTION_DIR / "strict_action_v0_samples.npz", allow_pickle=True) as data:
        actions = {key: data[key] for key in data.files}
    edge_vocab = pd.read_csv(DATASET_DIR / "edge_vocab.csv")
    node_vocab = pd.read_csv(DATASET_DIR / "node_vocab.csv")
    if not np.array_equal(arrays["sample_seed"], actions["sample_seed"]):
        raise ValueError("State samples and strict action samples are not aligned.")
    return arrays, actions, node_vocab, edge_vocab


def split_by_seed(sample_seed, train_seeds=(0, 1, 2), val_seed=3, test_seed=4):
    sample_seed = np.asarray(sample_seed)
    return (
        np.where(np.isin(sample_seed, train_seeds))[0],
        np.where(sample_seed == val_seed)[0],
        np.where(sample_seed == test_seed)[0],
    )


def grouped_stats(tensor, labels, groups):
    # tensor: B x H x Items x F. Output keeps temporal order and summarizes each group.
    parts = []
    labels = np.asarray(labels)
    for group in groups:
        mask = labels == group
        if mask.sum() == 0:
            b, h, _, f = tensor.shape
            parts.append(np.zeros((b, h, f * 3), dtype=np.float32))
            continue
        sub = tensor[:, :, mask, :]
        parts.append(
            np.concatenate(
                [sub.mean(axis=2), sub.std(axis=2), sub.max(axis=2)],
                axis=-1,
            )
        )
    return np.concatenate(parts, axis=-1).reshape(len(tensor), -1).astype(np.float32)


def mean_rate_by_type(link_tensor, edge_vocab):
    rate = link_tensor[..., 1]
    labels = edge_vocab["link_type"].to_numpy()
    outputs = []
    for link_type in LINK_TYPES:
        mask = labels == link_type
        if mask.sum() == 0:
            outputs.append(np.zeros(rate.shape[:2], dtype=np.float32))
        else:
            outputs.append(rate[..., mask].mean(axis=-1))
    return np.stack(outputs, axis=-1).astype(np.float32)


def build_targets_and_persistence(arrays, edge_vocab):
    y_link_type = mean_rate_by_type(arrays["y_link"].astype(np.float32), edge_vocab).reshape(len(arrays["y_link"]), -1)
    y_task = arrays["y_task"].astype(np.float32).reshape(len(arrays["y_task"]), -1)
    y = np.concatenate([y_link_type, y_task], axis=1).astype(np.float32)

    last_link_type = mean_rate_by_type(arrays["x_link"][:, -1:, :, :].astype(np.float32), edge_vocab)
    last_link = np.repeat(last_link_type, arrays["y_link"].shape[1], axis=1).reshape(len(y), -1)
    last_task = np.repeat(arrays["x_task"][:, -1:, :].astype(np.float32), arrays["y_task"].shape[1], axis=1).reshape(len(y), -1)
    persistence = np.concatenate([last_link, last_task], axis=1).astype(np.float32)
    return y, persistence


def build_branch_features(arrays, actions, node_vocab, edge_vocab):
    x_node = arrays["x_node"].astype(np.float32)
    x_link = arrays["x_link"].astype(np.float32)
    x_task = arrays["x_task"].astype(np.float32)
    node_branch = grouped_stats(x_node, node_vocab["node_type"].to_numpy(), NODE_TYPES)
    link_branch = grouped_stats(x_link, edge_vocab["link_type"].to_numpy(), LINK_TYPES)
    task_branch = x_task.reshape(len(x_task), -1).astype(np.float32)
    action_branch = np.concatenate(
        [
            actions["a_hist"].astype(np.float32).reshape(len(x_task), -1),
            actions["a_future"].astype(np.float32).reshape(len(x_task), -1),
        ],
        axis=1,
    ).astype(np.float32)
    return {
        "node": node_branch,
        "link": link_branch,
        "task": task_branch,
        "action": action_branch,
    }


def fit_standardizer(x):
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def apply_standardizer(x, mean, std):
    return ((x - mean) / std).astype(np.float32)


class StructuredPredictor(nn.Module):
    def __init__(self, dims, out_dim, hidden=64, use_action=False):
        super().__init__()
        self.use_action = use_action
        self.node_enc = self._branch(dims["node"], hidden)
        self.link_enc = self._branch(dims["link"], hidden)
        self.task_enc = self._branch(dims["task"], hidden)
        if use_action:
            self.action_enc = self._branch(dims["action"], hidden)
            fusion_in = hidden * 4
        else:
            self.action_enc = None
            fusion_in = hidden * 3
        self.head = nn.Sequential(
            nn.Linear(fusion_in, hidden * 2),
            nn.ReLU(),
            nn.Dropout(0.08),
            nn.Linear(hidden * 2, out_dim),
        )

    @staticmethod
    def _branch(in_dim, hidden):
        return nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

    def forward(self, node, link, task, action=None):
        parts = [self.node_enc(node), self.link_enc(link), self.task_enc(task)]
        if self.use_action:
            parts.append(self.action_enc(action))
        return self.head(torch.cat(parts, dim=-1))


def to_tensor_dict(features, idx, stats):
    out = {}
    for name, x in features.items():
        mean, std = stats[name]
        out[name] = torch.from_numpy(apply_standardizer(x[idx], mean, std))
    return out


def train_model(features, y_res_scaled, train_idx, val_idx, use_action, seed=42):
    torch.manual_seed(seed)
    stats = {name: fit_standardizer(x[train_idx]) for name, x in features.items()}
    train = to_tensor_dict(features, train_idx, stats)
    val = to_tensor_dict(features, val_idx, stats)
    y_train = torch.from_numpy(y_res_scaled[train_idx].astype(np.float32))
    y_val = torch.from_numpy(y_res_scaled[val_idx].astype(np.float32))

    dataset_tensors = [train["node"], train["link"], train["task"]]
    if use_action:
        dataset_tensors.append(train["action"])
    dataset_tensors.append(y_train)
    loader = DataLoader(TensorDataset(*dataset_tensors), batch_size=64, shuffle=True)

    dims = {name: x.shape[1] for name, x in features.items()}
    model = StructuredPredictor(dims, y_train.shape[1], hidden=64, use_action=use_action)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    best_state = None
    best_val = float("inf")
    best_epoch = 0
    patience = 80
    history = []

    for epoch in range(1, 801):
        model.train()
        losses = []
        for batch in loader:
            if use_action:
                node, link, task, action, target = batch
                pred = model(node, link, task, action)
            else:
                node, link, task, target = batch
                pred = model(node, link, task)
            loss = loss_fn(pred, target)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach()))

        if epoch == 1 or epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                if use_action:
                    val_pred = model(val["node"], val["link"], val["task"], val["action"])
                else:
                    val_pred = model(val["node"], val["link"], val["task"])
                val_loss = float(loss_fn(val_pred, y_val))
            history.append({"epoch": epoch, "train_mse": float(np.mean(losses)), "val_mse": val_loss})
            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if epoch - best_epoch > patience:
                break

    model.load_state_dict(best_state)
    return model, stats, pd.DataFrame(history), {"best_epoch": best_epoch, "best_val_mse": best_val}


def predict_model(model, features, idx, stats, use_action):
    model.eval()
    batch = to_tensor_dict(features, idx, stats)
    with torch.no_grad():
        if use_action:
            pred = model(batch["node"], batch["link"], batch["task"], batch["action"])
        else:
            pred = model(batch["node"], batch["link"], batch["task"])
    return pred.numpy().astype(np.float32)


def metrics(y_true, y_pred, link_dim):
    out = {}
    for name, sl in {
        "all": slice(None),
        "link_rate_by_type": slice(0, link_dim),
        "task_state": slice(link_dim, None),
    }.items():
        err = y_pred[:, sl] - y_true[:, sl]
        out[f"{name}_mae"] = float(np.mean(np.abs(err)))
        out[f"{name}_rmse"] = float(np.sqrt(np.mean(err**2)))
    return out


def plot_rmse(metrics_df, output_path):
    subset = metrics_df[metrics_df["split"] == "test_seed_4"].copy()
    order = ["persistence", "structured_state", "structured_state_action"]
    subset["model"] = pd.Categorical(subset["model"], categories=order, ordered=True)
    subset = subset.sort_values("model")
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    colors = ["#6b7280", "#2563eb", "#16a34a"]
    values = subset["all_rmse"].to_numpy()
    ax.bar(subset["model"].astype(str), values, color=colors)
    ax.set_ylabel("RMSE")
    ax.set_title("Held-out seed 4: structured branch baseline")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", labelrotation=12)
    for i, v in enumerate(values):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_task_curves(y_true, predictions, meta, output_path):
    task_features = meta["task_features"]
    selected = ["num_tasks", "num_to_offload", "num_finished"]
    feature_indices = [task_features.index(name) for name in selected]
    horizon = meta["horizon"]
    fig, axes = plt.subplots(len(selected), horizon, figsize=(14, 7), sharex=True)
    x_axis = np.arange(len(y_true))
    for row, feat_idx in enumerate(feature_indices):
        for step in range(horizon):
            target_idx = meta["link_target_dim"] + step * len(task_features) + feat_idx
            ax = axes[row, step]
            ax.plot(x_axis, y_true[:, target_idx], label="true", color="#111827", lw=1.7)
            for name, pred in predictions.items():
                ax.plot(x_axis, pred[:, target_idx], label=name, lw=1.1, alpha=0.85)
            ax.set_title(f"{selected[row]}, t+{step + 1}")
            ax.grid(alpha=0.25)
    axes[0, 0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Held-out seed 4: structured baseline task prediction")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_report(metrics_df, summary):
    test = metrics_df[metrics_df["split"] == "test_seed_4"].set_index("model")
    lines = [
        "# Structured dual-branch baseline v0",
        "",
        "## Goal",
        "",
        "This experiment moves beyond compact Ridge features and tests a lightweight structured model with separate node, link, task, and optional action branches.",
        "",
        "## Model input organization",
        "",
        "- Node branch: grouped node statistics by node type: cloud, RSU, UAV, vehicle.",
        "- Link branch: grouped link statistics by link type: U2I, V2I, V2U.",
        "- Task branch: historical task-state tensor.",
        "- Action branch: strict scheduler actions, including offloading, RB, CPU, and UAV movement records.",
        "",
        "## Split",
        "",
        "- Train: seed 0, seed 1, seed 2",
        "- Validation: seed 3",
        "- Test: seed 4",
        "",
        "## Metrics",
        "",
        metrics_df.to_markdown(index=False),
        "",
        "## Interpretation",
        "",
    ]
    persistence = float(test.loc["persistence", "all_rmse"])
    state = float(test.loc["structured_state", "all_rmse"])
    action = float(test.loc["structured_state_action", "all_rmse"])
    lines.extend(
        [
            f"- Persistence RMSE on held-out seed 4: `{persistence:.3f}`.",
            f"- Structured state-only RMSE: `{state:.3f}`.",
            f"- Structured state-action RMSE: `{action:.3f}`.",
        ]
    )
    if action < state:
        lines.append("- Adding strict actions improves the structured model, which supports action-conditioned transition modeling.")
    else:
        lines.append("- Adding strict actions does not improve this first structured model yet, so the action interface still needs better architecture or training.")
    if min(state, action) < persistence:
        lines.append("- The structured model beats persistence in this setting, so it is a stronger candidate baseline.")
    else:
        lines.append("- Persistence remains a strong short-horizon baseline; the next model must explicitly address this.")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
        ]
    )
    for key, value in summary["outputs"].items():
        lines.append(f"- {key}: `{value}`")
    report_path = OUTPUT_DIR / "structured_dual_branch_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    arrays, actions, node_vocab, edge_vocab = load_arrays()
    features = build_branch_features(arrays, actions, node_vocab, edge_vocab)
    y, persistence = build_targets_and_persistence(arrays, edge_vocab)
    train_idx, val_idx, test_idx = split_by_seed(arrays["sample_seed"])

    y_res = y - persistence
    y_mean, y_std = fit_standardizer(y_res[train_idx])
    y_res_scaled = apply_standardizer(y_res, y_mean, y_std)

    meta = {
        "horizon": int(arrays["y_link"].shape[1]),
        "link_target_dim": int(len(LINK_TYPES) * arrays["y_link"].shape[1]),
        "task_features": arrays["task_features"].tolist(),
        "node_features": arrays["node_features"].tolist(),
        "link_features": arrays["link_features"].tolist(),
        "action_features": actions["action_features"].tolist(),
        "branch_dims": {k: int(v.shape[1]) for k, v in features.items()},
    }

    state_model, state_stats, state_hist, state_info = train_model(
        features, y_res_scaled, train_idx, val_idx, use_action=False, seed=42
    )
    action_model, action_stats, action_hist, action_info = train_model(
        features, y_res_scaled, train_idx, val_idx, use_action=True, seed=43
    )

    preds = {}
    for split_name, idx in [("val_seed_3", val_idx), ("test_seed_4", test_idx)]:
        y_true = y[idx]
        persistence_pred = persistence[idx]
        state_res = predict_model(state_model, features, idx, state_stats, use_action=False)
        action_res = predict_model(action_model, features, idx, action_stats, use_action=True)
        state_pred = np.maximum(persistence_pred + state_res * y_std + y_mean, 0.0)
        action_pred = np.maximum(persistence_pred + action_res * y_std + y_mean, 0.0)
        preds[split_name] = {
            "y_true": y_true,
            "persistence": persistence_pred,
            "structured_state": state_pred,
            "structured_state_action": action_pred,
        }

    rows = []
    for split_name, split_preds in preds.items():
        y_true = split_preds["y_true"]
        for model_name in ["persistence", "structured_state", "structured_state_action"]:
            rows.append({"split": split_name, "model": model_name, **metrics(y_true, split_preds[model_name], meta["link_target_dim"])})
    metrics_df = pd.DataFrame(rows)
    metrics_path = OUTPUT_DIR / "structured_dual_branch_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    state_hist.to_csv(OUTPUT_DIR / "structured_state_training_history.csv", index=False, encoding="utf-8-sig")
    action_hist.to_csv(OUTPUT_DIR / "structured_state_action_training_history.csv", index=False, encoding="utf-8-sig")

    rmse_plot = FIGURE_DIR / "structured_dual_branch_rmse_bar.png"
    task_plot = FIGURE_DIR / "structured_dual_branch_task_predictions_seed4.png"
    plot_rmse(metrics_df, rmse_plot)
    plot_task_curves(
        preds["test_seed_4"]["y_true"],
        {
            "persistence": preds["test_seed_4"]["persistence"],
            "state": preds["test_seed_4"]["structured_state"],
            "state+action": preds["test_seed_4"]["structured_state_action"],
        },
        meta,
        task_plot,
    )

    summary = {
        "dataset_dir": str(DATASET_DIR),
        "action_dir": str(ACTION_DIR),
        "output_dir": str(OUTPUT_DIR),
        "split": {
            "train_seeds": [0, 1, 2],
            "val_seed": 3,
            "test_seed": 4,
            "train_samples": int(len(train_idx)),
            "val_samples": int(len(val_idx)),
            "test_samples": int(len(test_idx)),
        },
        "meta": meta,
        "training": {
            "structured_state": state_info,
            "structured_state_action": action_info,
        },
        "metrics": rows,
        "outputs": {
            "metrics_csv": str(metrics_path),
            "rmse_plot": str(rmse_plot),
            "task_prediction_plot": str(task_plot),
            "state_history_csv": str(OUTPUT_DIR / "structured_state_training_history.csv"),
            "state_action_history_csv": str(OUTPUT_DIR / "structured_state_action_training_history.csv"),
        },
    }
    summary_path = OUTPUT_DIR / "structured_dual_branch_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = write_report(metrics_df, summary)
    summary["outputs"]["report_md"] = str(report_path)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
