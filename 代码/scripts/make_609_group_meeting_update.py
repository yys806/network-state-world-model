"""Generate PI-JWM 6.9 group meeting materials.

The script adds three advisor-facing pieces of evidence:

1. v6 dual/physical/information ablation from the existing full80 run.
2. A same-split flat MLP baseline that flattens graph structure.
3. An active-only rate diagnostic that estimates headroom for active-rate RMSE.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from copy import deepcopy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pi_jwm.paths import ARTIFACTS_DIR
from pi_jwm.v6_data import (
    build_physical_edge_history,
    inverse_normalize,
    load_world_model_arrays,
    make_normalization_stats,
    normalize,
    split_by_seed,
)
from pi_jwm.v6_flat_baseline import V6FlatBaseline, V6FlatBaselineConfig
from pi_jwm.v6_metrics import active_rate_metrics, activity_metrics, regression_metrics


DEFAULT_DATASET_DIR = (
    ARTIFACTS_DIR
    / "experiments"
    / "airfogsim_v0"
    / "datasets"
    / "world_model_dataset_seed0_9_v0"
)
DEFAULT_V6_SUMMARY = (
    ARTIFACTS_DIR
    / "experiments"
    / "pi_jwm_v6_eval_full80"
    / "v6_dual_graph_smoke_summary.json"
)
DEFAULT_ARTIFACT_DIR = ARTIFACTS_DIR / "experiments" / "pi_jwm_609_group_meeting"
DEFAULT_MEETING_DIR = WORKSPACE_ROOT / "文档" / "组会" / "6.9"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create PI-JWM 6.9 meeting assets.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--v6-summary", type=Path, default=DEFAULT_V6_SUMMARY)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--meeting-dir", type=Path, default=DEFAULT_MEETING_DIR)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260608)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    figs_dir = args.meeting_dir / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    arrays = load_world_model_arrays(args.dataset_dir)
    train_idx, val_idx, test_idx = split_by_seed(arrays["sample_seed"])
    stats = make_normalization_stats(arrays, train_idx)
    full_physical = build_physical_edge_history(
        arrays["x_node"],
        arrays["edge_src_idx"],
        arrays["edge_dst_idx"],
        arrays["valid_edge_node"],
    ).numpy()

    v6_runs = load_v6_runs(args.v6_summary)
    flat_result = run_flat_mlp_baseline(args, arrays, full_physical, stats, train_idx, val_idx, test_idx)
    active_rate_result = run_active_only_rate_diagnostic(arrays, full_physical, train_idx, val_idx, test_idx)

    summary = {
        "framework": "PI-JWM",
        "meeting_date": "2026-06-09",
        "dataset": {
            "path": str(args.dataset_dir),
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
            "history": int(arrays["x_node"].shape[1]),
            "horizon": int(arrays["y_node"].shape[1]),
            "num_nodes": int(arrays["x_node"].shape[2]),
            "num_edges": int(arrays["x_link"].shape[2]),
            "test_active_edges": int((arrays["y_link_active"][test_idx] > 0.5).sum()),
            "test_active_ratio": float((arrays["y_link_active"][test_idx] > 0.5).mean()),
        },
        "v6_full80": v6_runs,
        "flat_mlp": flat_result,
        "active_rate_diagnostic": active_rate_result,
    }

    write_json(args.artifact_dir / "pi_jwm_609_summary.json", summary)
    write_metrics_csv(args.artifact_dir / "pi_jwm_609_metrics.csv", summary)
    write_metrics_csv(args.meeting_dir / "pi_jwm_609_metrics.csv", summary)
    write_figures(figs_dir, summary)
    write_meeting_update(args.meeting_dir / "PI-JWM_6.9组会阶段更新.md", summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"meeting_dir={args.meeting_dir}")
    print(f"artifact_dir={args.artifact_dir}")


def load_v6_runs(path: Path) -> dict[str, dict[str, float]]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    runs = summary["real_data_sanity"]["runs"]
    compact = {}
    for mode, run in runs.items():
        test = run["test_eval"]
        compact[mode] = {
            "best_epoch": int(run["best_epoch"]),
            "threshold": float(run["activity_threshold"]),
            "activity_f1": float(test["activity"]["f1"]),
            "activity_tp": float(test["activity"]["tp"]),
            "activity_fp": float(test["activity"]["fp"]),
            "activity_fn": float(test["activity"]["fn"]),
            "activity_tn": float(test["activity"]["tn"]),
            "active_rate_rmse": float(test["active_rate"]["active_rmse"]),
            "active_rate_mae": float(test["active_rate"]["active_mae"]),
            "link_rate_rmse": float(test["link_rate"]["rmse"]),
            "link_rate_mae": float(test["link_rate"]["mae"]),
            "node_rmse": float(test["node"]["rmse"]),
            "task_rmse": float(test["task"]["rmse"]),
        }
    return compact


def run_flat_mlp_baseline(
    args: argparse.Namespace,
    arrays: dict[str, np.ndarray],
    full_physical: np.ndarray,
    stats: dict[str, tuple[np.ndarray, np.ndarray]],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> dict[str, float | int | dict[str, float]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data = make_flat_tensors(arrays, full_physical, stats, train_idx)
    val_data = make_flat_tensors(arrays, full_physical, stats, val_idx)
    test_data = make_flat_tensors(arrays, full_physical, stats, test_idx)

    train_loader = make_loader(train_data, args.batch_size, shuffle=True, seed=args.seed)
    val_loader = make_loader(val_data, args.batch_size, shuffle=False, seed=args.seed)
    test_loader = make_loader(test_data, args.batch_size, shuffle=False, seed=args.seed)

    config = V6FlatBaselineConfig(
        input_dim=int(train_data.tensors[0].shape[1]),
        node_dim=int(arrays["y_node"].shape[-1]),
        task_dim=int(arrays["y_task"].shape[-1]),
        num_nodes=int(arrays["y_node"].shape[2]),
        num_edges=int(arrays["y_link_rate"].shape[2]),
        horizon=int(arrays["y_link_rate"].shape[1]),
        hidden_dim=args.hidden_dim,
    )
    model = V6FlatBaseline(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_state = None
    best_epoch = 0
    best_val = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_parts = []
        for batch in train_loader:
            inputs, target = move_flat_batch(batch, device)
            outputs = model(inputs)
            loss, parts = compute_loss(outputs, target)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_parts.append(parts)
        val_parts = evaluate_loss(model, val_loader, device)
        train_avg = mean_metrics(train_parts)
        history.append({"epoch": epoch, "train": train_avg, "val": val_parts})
        if val_parts["total"] < best_val:
            best_val = val_parts["total"]
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
        print(
            f"[flat-{device.type}] epoch={epoch} "
            f"train_total={train_avg['total']:.6f} val_total={val_parts['total']:.6f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    val_predictions = collect_flat_predictions(model, val_loader, device, stats)
    threshold = choose_activity_threshold(val_predictions["link_activity_prob"], val_predictions["link_activity_true"])
    test_eval = evaluate_predictions(collect_flat_predictions(model, test_loader, device, stats), threshold)
    return {
        "model": "flat_mlp",
        "note": "Same-split local baseline; it flattens node/edge tensors and does not use graph message passing.",
        "device": str(device),
        "epochs": int(args.epochs),
        "best_epoch": int(best_epoch),
        "best_val_total": float(best_val),
        "threshold": float(threshold),
        "history": history,
        "test_eval": flatten_eval(test_eval),
    }


def make_flat_tensors(
    arrays: dict[str, np.ndarray],
    full_physical: np.ndarray,
    stats: dict[str, tuple[np.ndarray, np.ndarray]],
    indices: np.ndarray,
) -> TensorDataset:
    parts = [
        normalize(arrays["x_node"][indices], stats["x_node"]).reshape(len(indices), -1),
        normalize(full_physical[indices], stats["x_physical_edge"]).reshape(len(indices), -1),
        normalize(arrays["x_link"][indices], stats["x_link"]).reshape(len(indices), -1),
        normalize(arrays["edge_a_hist"][indices], stats["edge_a_hist"]).reshape(len(indices), -1),
        normalize(arrays["edge_a_future"][indices], stats["edge_a_future"]).reshape(len(indices), -1),
        normalize(arrays["x_task"][indices], stats["x_task"]).reshape(len(indices), -1),
    ]
    inputs = np.concatenate(parts, axis=1).astype(np.float32)
    y_node = normalize(arrays["y_node"][indices], stats["y_node"])
    y_activity = arrays["y_link_active"][indices, ..., None].astype(np.float32)
    y_rate = normalize(arrays["y_link_rate"][indices, ..., None], stats["y_link_rate"])
    y_task = normalize(arrays["y_task"][indices], stats["y_task"])
    return TensorDataset(
        torch.from_numpy(inputs),
        torch.from_numpy(y_node),
        torch.from_numpy(y_activity),
        torch.from_numpy(y_rate),
        torch.from_numpy(y_task),
    )


def make_loader(dataset: TensorDataset, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def move_flat_batch(batch: tuple[torch.Tensor, ...], device: torch.device) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    inputs, y_node, y_activity, y_rate, y_task = batch
    target = {
        "node": y_node.to(device),
        "link_activity": y_activity.to(device),
        "link_rate": y_rate.to(device),
        "task": y_task.to(device),
    }
    return inputs.to(device), target


def compute_loss(outputs: dict[str, torch.Tensor], target: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
    mse = nn.MSELoss()
    pos_weight = torch.tensor([80.0], device=outputs["link_activity_logit"].device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    node_loss = mse(outputs["node"], target["node"])
    activity_loss = bce(outputs["link_activity_logit"], target["link_activity"])
    active_weight = 1.0 + 50.0 * target["link_activity"]
    rate_loss = (((outputs["link_rate"] - target["link_rate"]) ** 2) * active_weight).mean()
    task_loss = mse(outputs["task"], target["task"])
    total = 0.5 * node_loss + activity_loss + 0.3 * rate_loss + 0.8 * task_loss
    return total, {
        "total": float(total.detach().cpu()),
        "node": float(node_loss.detach().cpu()),
        "activity": float(activity_loss.detach().cpu()),
        "rate": float(rate_loss.detach().cpu()),
        "task": float(task_loss.detach().cpu()),
    }


def evaluate_loss(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in loader:
            inputs, target = move_flat_batch(batch, device)
            _, parts = compute_loss(model(inputs), target)
            rows.append(parts)
    return mean_metrics(rows)


def collect_flat_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    stats: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, np.ndarray]:
    model.eval()
    rows = {
        "node_pred": [],
        "node_true": [],
        "link_activity_prob": [],
        "link_activity_true": [],
        "link_rate_pred": [],
        "link_rate_true": [],
        "task_pred": [],
        "task_true": [],
    }
    with torch.no_grad():
        for batch in loader:
            inputs, target = move_flat_batch(batch, device)
            outputs = model(inputs)
            rows["node_pred"].append(inverse_normalize(outputs["node"].cpu().numpy(), stats["y_node"]))
            rows["node_true"].append(inverse_normalize(target["node"].cpu().numpy(), stats["y_node"]))
            rows["link_activity_prob"].append(torch.sigmoid(outputs["link_activity_logit"]).cpu().numpy())
            rows["link_activity_true"].append(target["link_activity"].cpu().numpy())
            rows["link_rate_pred"].append(inverse_normalize(outputs["link_rate"].cpu().numpy(), stats["y_link_rate"]))
            rows["link_rate_true"].append(inverse_normalize(target["link_rate"].cpu().numpy(), stats["y_link_rate"]))
            rows["task_pred"].append(inverse_normalize(outputs["task"].cpu().numpy(), stats["y_task"]))
            rows["task_true"].append(inverse_normalize(target["task"].cpu().numpy(), stats["y_task"]))
    return {name: np.concatenate(values, axis=0) for name, values in rows.items()}


def choose_activity_threshold(prob: np.ndarray, true: np.ndarray) -> float:
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(0.01, 0.99, 99):
        f1 = activity_metrics(prob, true, threshold=float(threshold))["f1"]
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)
    return best_threshold


def evaluate_predictions(predictions: dict[str, np.ndarray], threshold: float) -> dict[str, dict[str, float]]:
    return {
        "node": regression_metrics(predictions["node_pred"], predictions["node_true"]),
        "task": regression_metrics(predictions["task_pred"], predictions["task_true"]),
        "link_rate": regression_metrics(predictions["link_rate_pred"], predictions["link_rate_true"]),
        "active_rate": active_rate_metrics(
            predictions["link_rate_pred"],
            predictions["link_rate_true"],
            predictions["link_activity_true"],
        ),
        "activity": activity_metrics(predictions["link_activity_prob"], predictions["link_activity_true"], threshold),
    }


def flatten_eval(eval_result: dict[str, dict[str, float]]) -> dict[str, float]:
    return {
        "activity_f1": float(eval_result["activity"]["f1"]),
        "activity_precision": float(eval_result["activity"]["precision"]),
        "activity_recall": float(eval_result["activity"]["recall"]),
        "activity_tp": float(eval_result["activity"]["tp"]),
        "activity_fp": float(eval_result["activity"]["fp"]),
        "activity_fn": float(eval_result["activity"]["fn"]),
        "activity_tn": float(eval_result["activity"]["tn"]),
        "active_rate_rmse": float(eval_result["active_rate"]["active_rmse"]),
        "active_rate_mae": float(eval_result["active_rate"]["active_mae"]),
        "link_rate_rmse": float(eval_result["link_rate"]["rmse"]),
        "link_rate_mae": float(eval_result["link_rate"]["mae"]),
        "node_rmse": float(eval_result["node"]["rmse"]),
        "task_rmse": float(eval_result["task"]["rmse"]),
    }


def run_active_only_rate_diagnostic(
    arrays: dict[str, np.ndarray],
    full_physical: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> dict[str, float]:
    x_train, y_train, active_train = build_active_rate_rows(arrays, full_physical, train_idx)
    x_val, y_val, active_val = build_active_rate_rows(arrays, full_physical, val_idx)
    x_test, y_test, active_test = build_active_rate_rows(arrays, full_physical, test_idx)

    x_train = x_train[active_train].astype(np.float64)
    y_train = y_train[active_train].astype(np.float64)
    x_val = x_val[active_val].astype(np.float64)
    y_val = y_val[active_val].astype(np.float64)
    x_test = x_test[active_test].astype(np.float64)
    y_test = y_test[active_test].astype(np.float64)

    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    x_train_n = add_bias((x_train - mean) / std)
    x_val_n = add_bias((x_val - mean) / std)
    x_test_n = add_bias((x_test - mean) / std)

    best = None
    for ridge_lambda in (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0):
        weights = fit_ridge(x_train_n, y_train, ridge_lambda)
        val_pred = x_val_n @ weights
        val_rmse = rmse(val_pred, y_val)
        if best is None or val_rmse < best["val_rmse"]:
            test_pred = x_test_n @ weights
            best = {
                "model": "active_only_ridge_oracle_activity",
                "note": "Diagnostic upper-bound style result using true-active edges for fitting/evaluation.",
                "lambda": float(ridge_lambda),
                "train_active_count": int(len(y_train)),
                "val_active_count": int(len(y_val)),
                "test_active_count": int(len(y_test)),
                "val_rmse": float(val_rmse),
                "test_rmse": float(rmse(test_pred, y_test)),
                "test_mae": float(np.mean(np.abs(test_pred - y_test))),
                "test_true_mean": float(np.mean(y_test)),
                "test_true_min": float(np.min(y_test)),
                "test_true_max": float(np.max(y_test)),
            }

    active_mean = float(np.mean(y_train))
    best["active_mean_baseline_rmse"] = float(rmse(np.full_like(y_test, active_mean), y_test))
    best["active_mean_baseline_mae"] = float(np.mean(np.abs(active_mean - y_test)))
    return best


def build_active_rate_rows(
    arrays: dict[str, np.ndarray],
    full_physical: np.ndarray,
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_edges = arrays["x_link"].shape[2]
    horizon = arrays["edge_a_future"].shape[1]
    base = np.concatenate(
        [
            full_physical[indices, -1],
            arrays["x_link"][indices, -1],
            arrays["edge_a_hist"][indices, -1],
            np.repeat(arrays["x_task"][indices, -1][:, None, :], num_edges, axis=1),
        ],
        axis=-1,
    )
    features = []
    rates = []
    active = []
    eye = np.eye(horizon, dtype=np.float32)
    for step in range(horizon):
        step_one_hot = np.broadcast_to(eye[step], (len(indices), num_edges, horizon))
        features.append(np.concatenate([base, arrays["edge_a_future"][indices, step], step_one_hot], axis=-1))
        rates.append(arrays["y_link_rate"][indices, step])
        active.append(arrays["y_link_active"][indices, step] > 0.5)
    return np.stack(features, axis=1), np.stack(rates, axis=1), np.stack(active, axis=1)


def add_bias(values: np.ndarray) -> np.ndarray:
    return np.concatenate([values, np.ones((values.shape[0], 1), dtype=values.dtype)], axis=1)


def fit_ridge(x: np.ndarray, y: np.ndarray, ridge_lambda: float) -> np.ndarray:
    regularizer = ridge_lambda * np.eye(x.shape[1], dtype=x.dtype)
    regularizer[-1, -1] = 0.0
    return np.linalg.solve(x.T @ x + regularizer, x.T @ y)


def rmse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    return {key: float(sum(row[key] for row in rows) / len(rows)) for key in rows[0]}


def write_json(path: Path, summary: dict) -> None:
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def write_metrics_csv(path: Path, summary: dict) -> None:
    rows = []
    for mode, metrics in summary["v6_full80"].items():
        row = {"family": "PI-JWM v6 full80", "model": mode}
        row.update(metrics)
        rows.append(row)
    flat = {"family": "ordinary model local baseline", "model": "flat_mlp"}
    flat.update(summary["flat_mlp"]["test_eval"])
    flat["best_epoch"] = summary["flat_mlp"]["best_epoch"]
    flat["threshold"] = summary["flat_mlp"]["threshold"]
    rows.append(flat)
    active = summary["active_rate_diagnostic"]
    rows.append(
        {
            "family": "active-rate diagnostic",
            "model": active["model"],
            "active_rate_rmse": active["test_rmse"],
            "active_rate_mae": active["test_mae"],
            "activity_f1": "",
            "link_rate_rmse": "",
            "node_rmse": "",
            "task_rmse": "",
            "best_epoch": "",
            "threshold": "",
        }
    )
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_figures(figs_dir: Path, summary: dict) -> None:
    plot_v6_ablation(figs_dir / "pi_jwm_v6_dual_graph_ablation_609.png", summary)
    plot_dual_gain(figs_dir / "pi_jwm_dual_fusion_gain_609.png", summary)
    plot_flat_comparison(figs_dir / "pi_jwm_vs_flat_mlp_609.png", summary)
    plot_active_rate_headroom(figs_dir / "pi_jwm_active_rate_headroom_609.png", summary)


def plot_v6_ablation(path: Path, summary: dict) -> None:
    modes = ["dual", "physical_only", "information_only"]
    metrics = [
        ("link_rate_rmse", "Link-rate RMSE"),
        ("active_rate_rmse", "Active-rate RMSE"),
        ("node_rmse", "Node RMSE"),
        ("task_rmse", "Task RMSE"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), dpi=180)
    colors_base = ["#8da0cb", "#a6a6a6", "#a6a6a6"]
    for ax, (key, title) in zip(axes.ravel(), metrics):
        values = [summary["v6_full80"][mode][key] for mode in modes]
        best_idx = int(np.argmin(values))
        colors = colors_base.copy()
        colors[best_idx] = "#2ca25f"
        ax.bar(["dual", "physical", "info"], values, color=colors)
        ax.set_title(title)
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        for i, value in enumerate(values):
            ax.text(i, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("PI-JWM v6 full80 ablation: lower RMSE is better", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_dual_gain(path: Path, summary: dict) -> None:
    keys = ["link_rate_rmse", "active_rate_rmse", "node_rmse", "task_rmse"]
    labels = ["Link rate", "Active rate", "Node", "Task"]
    dual = np.array([summary["v6_full80"]["dual"][key] for key in keys], dtype=float)
    best_single = np.array(
        [
            min(summary["v6_full80"]["physical_only"][key], summary["v6_full80"]["information_only"][key])
            for key in keys
        ],
        dtype=float,
    )
    gain = (best_single - dual) / best_single * 100.0
    colors = ["#2ca25f" if value >= 0 else "#de2d26" for value in gain]
    fig, ax = plt.subplots(figsize=(9.5, 4.5), dpi=180)
    ax.axhline(0, color="#444444", linewidth=0.9)
    ax.bar(labels, gain, color=colors)
    ax.set_ylabel("Dual improvement vs best single graph (%)")
    ax.set_title("Where dual fusion helps, and where single-graph branches still win")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    for i, value in enumerate(gain):
        ax.text(i, value, f"{value:+.1f}%", ha="center", va="bottom" if value >= 0 else "top", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_flat_comparison(path: Path, summary: dict) -> None:
    keys = ["link_rate_rmse", "active_rate_rmse", "node_rmse", "task_rmse"]
    labels = ["Link", "Active", "Node", "Task"]
    dual = np.array([summary["v6_full80"]["dual"][key] for key in keys], dtype=float)
    flat = np.array([summary["flat_mlp"]["test_eval"][key] for key in keys], dtype=float)
    ratio = dual / flat
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), dpi=180)
    x = np.arange(len(keys))
    width = 0.36
    axes[0].bar(x - width / 2, flat, width, label="Flat MLP", color="#bdbdbd")
    axes[0].bar(x + width / 2, dual, width, label="PI-JWM dual", color="#3182bd")
    axes[0].set_xticks(x, labels)
    axes[0].set_title("Same-split ordinary model comparison")
    axes[0].set_ylabel("RMSE")
    axes[0].grid(axis="y", linestyle="--", alpha=0.25)
    axes[0].legend()
    colors = ["#2ca25f" if value < 1.0 else "#de2d26" for value in ratio]
    axes[1].axhline(1.0, color="#444444", linewidth=0.9)
    axes[1].bar(labels, ratio, color=colors)
    axes[1].set_title("PI-JWM dual / flat MLP RMSE ratio")
    axes[1].set_ylabel("Ratio (<1 is better)")
    axes[1].grid(axis="y", linestyle="--", alpha=0.25)
    for i, value in enumerate(ratio):
        axes[1].text(i, value, f"{value:.2f}x", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_active_rate_headroom(path: Path, summary: dict) -> None:
    active = summary["active_rate_diagnostic"]
    labels = [
        "PI-JWM dual\ncurrent head",
        "Info-only\ncurrent head",
        "Active-only ridge\noracle activity",
        "Active mean\nbaseline",
    ]
    values = [
        summary["v6_full80"]["dual"]["active_rate_rmse"],
        summary["v6_full80"]["information_only"]["active_rate_rmse"],
        active["test_rmse"],
        active["active_mean_baseline_rmse"],
    ]
    colors = ["#3182bd", "#9ecae1", "#2ca25f", "#bdbdbd"]
    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=180)
    ax.bar(labels, values, color=colors)
    ax.set_ylabel("Active-rate RMSE")
    ax.set_title("Active-rate bottleneck and estimated improvement headroom")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    for i, value in enumerate(values):
        ax.text(i, value, f"{value:.1f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def write_meeting_update(path: Path, summary: dict) -> None:
    v6 = summary["v6_full80"]
    flat = summary["flat_mlp"]
    active = summary["active_rate_diagnostic"]
    dataset = summary["dataset"]
    dual = v6["dual"]
    text = f"""# PI-JWM 6.9 组会阶段更新

## 1. 这次补上的工作

本次更新围绕 PI-JWM 主线补了三块可以直接汇报的内容：

1. **双图融合证据**：整理 v6 full80 的 dual / physical-only / information-only 三模式消融，明确哪些指标由双图融合带来收益，哪些指标仍由单图分支占优。
2. **active-rate 提升空间**：新增 active-only rate 诊断，只在真实活跃链路上拟合速率数值，估计当前 rate head 的可改进空间。
3. **普通模型对比**：新增同数据切分的 flat MLP baseline。它使用同类历史状态和动作输入，把所有节点、链路、动作和任务特征展平成向量，作为缺少显式双图消息传递的普通对照。

数据切分保持 v6 full80 口径：train={dataset["train"]}，val={dataset["val"]}，test={dataset["test"]}；历史窗口 H={dataset["history"]}，预测步长 K={dataset["horizon"]}，节点数 {dataset["num_nodes"]}，候选链路数 {dataset["num_edges"]}。测试集中真实 active 链路格点数为 {dataset["test_active_edges"]}，占比约 {dataset["test_active_ratio"] * 100:.4f}%。

## 2. 当前关键结果

### 2.1 v6 双图消融

| 模型模式 | activity F1 | active-rate RMSE | link-rate RMSE | node RMSE | task RMSE |
|---|---:|---:|---:|---:|---:|
| dual | {v6["dual"]["activity_f1"]:.3f} | {v6["dual"]["active_rate_rmse"]:.3f} | {v6["dual"]["link_rate_rmse"]:.3f} | {v6["dual"]["node_rmse"]:.3f} | {v6["dual"]["task_rmse"]:.3f} |
| physical-only | {v6["physical_only"]["activity_f1"]:.3f} | {v6["physical_only"]["active_rate_rmse"]:.3f} | {v6["physical_only"]["link_rate_rmse"]:.3f} | {v6["physical_only"]["node_rmse"]:.3f} | {v6["physical_only"]["task_rmse"]:.3f} |
| information-only | {v6["information_only"]["activity_f1"]:.3f} | {v6["information_only"]["active_rate_rmse"]:.3f} | {v6["information_only"]["link_rate_rmse"]:.3f} | {v6["information_only"]["node_rmse"]:.3f} | {v6["information_only"]["task_rmse"]:.3f} |

可以讲的结论：

- dual 在 link-rate RMSE 和 active-rate RMSE 上最好，说明物理图与信息图联合后，对链路速率幅值预测有帮助。
- physical-only 的 node RMSE 最低，说明节点运动/位置相关状态更依赖物理几何关系。
- information-only 的 task RMSE 最低，说明任务状态演化更依赖通信历史、动作历史和队列信息。
- activity F1 当前为 1.0，但 active 样本极稀疏，所以后续不能只看 activity，还要继续看 active-rate 的数值回归。

对应图表：

- `figs/pi_jwm_v6_dual_graph_ablation_609.png`
- `figs/pi_jwm_dual_fusion_gain_609.png`

### 2.2 普通模型对比

flat MLP baseline 的测试结果，训练 epoch 数为 {flat["epochs"]}：

| 模型 | activity F1 | active-rate RMSE | link-rate RMSE | node RMSE | task RMSE |
|---|---:|---:|---:|---:|---:|
| flat MLP | {flat["test_eval"]["activity_f1"]:.3f} | {flat["test_eval"]["active_rate_rmse"]:.3f} | {flat["test_eval"]["link_rate_rmse"]:.3f} | {flat["test_eval"]["node_rmse"]:.3f} | {flat["test_eval"]["task_rmse"]:.3f} |
| PI-JWM dual | {dual["activity_f1"]:.3f} | {dual["active_rate_rmse"]:.3f} | {dual["link_rate_rmse"]:.3f} | {dual["node_rmse"]:.3f} | {dual["task_rmse"]:.3f} |

讲法建议：

- 这个普通模型用的是同一批 train/val/test 样本，也输入历史状态和未来动作。
- 它把所有实体展平成一个大向量，缺少物理边、信息边和边级消息传递的结构约束。
- 这版可以作为明天的普通模型对照起点；后续会继续补 Ridge、Transformer-style flat encoder，并统一训练预算。

对应图表：

- `figs/pi_jwm_vs_flat_mlp_609.png`

### 2.3 active-rate 提升空间

active-only rate 诊断结果：

| 方法 | 说明 | active-rate RMSE | active-rate MAE |
|---|---|---:|---:|
| PI-JWM dual 当前 rate head | v6 full80 当前输出头 | {dual["active_rate_rmse"]:.3f} | {dual["active_rate_mae"]:.3f} |
| active-only Ridge | 只在真实 active 边上拟合速率，用于估计提升空间 | {active["test_rmse"]:.3f} | {active["test_mae"]:.3f} |
| active mean baseline | 用训练集 active 速率均值预测 | {active["active_mean_baseline_rmse"]:.3f} | {active["active_mean_baseline_mae"]:.3f} |

讲法建议：

- 当前 active-rate RMSE 是主要瓶颈，因为整体 link-rate RMSE 会被大量 inactive 边稀释。
- active-only 诊断显示，只要把 activity gating 和 active-rate regression 更好地分开，active-rate 还有明显下降空间。
- 下一版会把这个诊断并入 PI-JWM：先判断 active，再在 active 边上用专门 rate head 预测速率幅值，并加入残差校准。

对应图表：

- `figs/pi_jwm_active_rate_headroom_609.png`

## 3. 明天可以这样讲

第一步，说明当前主线是 PI-JWM：输入历史状态和动作，预测未来节点、链路、任务状态。动作当前来自日志或候选动作，完整的策略模块还没有做。

第二步，讲 v6 双图结果：dual 对链路速率最有帮助，physical-only 更适合节点状态，information-only 更适合任务状态。这说明双图融合方向是有价值的，但融合方式还可以增强。

第三步，讲普通模型对比：flat MLP 已经接上同 split 对比，它缺少显式图结构，后续会补 Ridge 和 Transformer-style flat encoder，作为 PI-JWM 的正式对照组。

第四步，讲 active-rate：当前 activity 识别稳定，但 active-rate 数值预测仍是瓶颈；active-only 诊断给出了明确改进方向。

## 4. 下一步计划

### P0：增强双图融合

- 从简单 concat fusion 改成 gated fusion 或 cross-attention fusion，让模型自动学习物理图和信息图在不同目标上的权重。
- 增加 edge-level residual head：在共享 rollout 表征上，给 link-rate 和 active-rate 单独接残差预测分支。
- 扩展未来物理图输入：当前 future action 已有，后续把未来几步的物理边变化也纳入 rollout。

### P0：提升 active-rate

- 将 activity gating 和 rate regression 分开训练：先稳定 active mask，再对 active 边做专门速率回归。
- 加入 active-only loss / focal-style weighting，减少 inactive 边对 rate head 的稀释。
- 做 grouped residual calibration：按距离、链路类型、RB 分配强度或 horizon 分组校准残差。

### P0：普通模型正式对比

- 在同一 train/val/test split 下补齐 ordinary baselines：Ridge、flat MLP full-budget、Transformer-style flat encoder。
- 统一输出 activity F1、active-rate RMSE、link-rate RMSE、node RMSE、task RMSE。
- 汇报时只把同 split、同训练预算的结果作为正式对比。

### P1：稳健性和泛化

- 做 seed-heldout / cross-seed 评估，避免只对 seed 9 成立。
- 加输入扰动和置信区间，观察 activity 与 active-rate 的稳定性。
- 在 state rollout 改善后，再接候选动作评估和策略闭环。
"""
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
