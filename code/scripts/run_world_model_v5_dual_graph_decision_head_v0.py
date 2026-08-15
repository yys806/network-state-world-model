import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from run_world_model_v5_classical_ranker_v0 import StandardizedRidgeRegressor
from run_world_model_v5_utility_ranking_smoke import (
    DEFAULT_STATE_DATASET_NPZ,
    DEFAULT_STATE_SAMPLE_INDEX_CSV,
    ROOT,
    add_resource_aware_utility,
    build_group_pairwise_pairs,
    build_candidate_features,
    display_path,
    ensure_decision_group_id,
    filter_state_available_groups,
    grouped_ranking_metrics,
    load_state_arrays,
    apply_feature_standardizer,
    fit_feature_standardizer,
    predict_utility,
    split_by_test_seeds,
    split_decision_groups,
    train_utility_head,
)


ACTION_FAMILY_NAMES = ["rb_count", "offload_target", "mixed_offload_rb", "cpu_scale", "return_route", "other"]
DEFAULT_CANDIDATE_CSV = (
    ROOT
    / "reports"
    / "airfogsim_counterfactual_offload_scaled_v0"
    / "airfogsim_counterfactual_multifamily_v0_labels.csv"
)
OUTPUT_DIR = ROOT / "reports" / "world_model_v5_dual_graph_decision_head_v0"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a v5 dual-graph state-action decision head.")
    parser.add_argument("--candidate-csv", type=Path, default=DEFAULT_CANDIDATE_CSV)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--state-sample-index-csv", type=Path, default=DEFAULT_STATE_SAMPLE_INDEX_CSV)
    parser.add_argument("--state-dataset-npz", type=Path, default=DEFAULT_STATE_DATASET_NPZ)
    parser.add_argument("--utility-column", type=str, default="resource_aware_utility")
    parser.add_argument("--rb-penalty", type=float, default=0.001)
    parser.add_argument("--test-fraction", type=float, default=0.35)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--test-seeds", type=int, nargs="*", default=None)
    parser.add_argument("--require-state-available", action="store_true")
    parser.add_argument("--model-kind", choices=["ridge", "mlp_rank", "family_mlp_rank"], default="ridge")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--reg-weight", type=float, default=1.0)
    parser.add_argument("--rank-weight", type=float, default=1.0)
    parser.add_argument("--winner-weight", type=float, default=0.0)
    parser.add_argument("--winner-gap-weight-power", type=float, default=0.0)
    parser.add_argument("--include-interactions", action="store_true")
    parser.add_argument("--compact-interactions", action="store_true")
    parser.add_argument("--anchor-mode", choices=["none", "minus_total_rb", "plus_total_rb"], default="none")
    return parser.parse_args()


def feature_index(names, name):
    values = [str(item) for item in names]
    return values.index(name) if name in values else None


def safe_edge_feature(array, names, feature_name):
    idx = feature_index(names, feature_name)
    if idx is None:
        return np.zeros(array.shape[:-1], dtype=np.float32)
    return np.asarray(array[..., idx], dtype=np.float32)


def valid_edge_mask(arrays):
    if "valid_edge_node" not in arrays:
        return None
    return np.asarray(arrays["valid_edge_node"], dtype=np.float32) > 0.5


def masked_values(values, mask):
    values = np.asarray(values, dtype=np.float32)
    if mask is None:
        return values.reshape(-1)
    return values[..., mask].reshape(-1)


def safe_mean(values):
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return 0.0
    return float(np.nanmean(values))


def safe_sum(values):
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return 0.0
    return float(np.nansum(values))


def build_action_family_ids(df, family_names=None):
    family_names = list(family_names or ACTION_FAMILY_NAMES)
    if "other" not in family_names:
        family_names.append("other")
    mapping = {name: idx for idx, name in enumerate(family_names)}
    other_idx = mapping["other"]
    if "action_family" not in df.columns:
        return np.full(len(df), other_idx, dtype=np.int64), family_names
    values = df["action_family"].astype(str)
    ids = values.map(lambda name: mapping.get(name, other_idx)).to_numpy(dtype=np.int64)
    return ids, family_names


def build_anchor_score(df, mode="none"):
    if mode == "none":
        return np.zeros(len(df), dtype=np.float32)
    if mode == "minus_total_rb":
        if "total_rb" not in df.columns:
            raise KeyError("total_rb is required for anchor mode minus_total_rb")
        return -np.log1p(np.maximum(df["total_rb"].to_numpy(dtype=np.float32), 0.0)).astype(np.float32)
    if mode == "plus_total_rb":
        if "total_rb" not in df.columns:
            raise KeyError("total_rb is required for anchor mode plus_total_rb")
        return np.log1p(np.maximum(df["total_rb"].to_numpy(dtype=np.float32), 0.0)).astype(np.float32)
    raise ValueError(f"unsupported anchor mode: {mode}")


def resolve_family_anchor_mode(model_kind, anchor_mode):
    if model_kind != "family_mlp_rank":
        return str(anchor_mode)
    return str(anchor_mode)


class FamilySpecificUtilityHead(nn.Module):
    def __init__(self, in_dim, num_families, hidden=16):
        super().__init__()
        embed_dim = min(8, max(2, int(num_families)))
        self.family_embedding = nn.Embedding(num_families, embed_dim)
        self.shared = nn.Sequential(
            nn.Linear(in_dim + embed_dim, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.residual = nn.Linear(hidden, 1)
        self.family_bias = nn.Embedding(num_families, 1)

    def forward(self, x, family_ids, anchor_score=None):
        family_ids = family_ids.clamp(min=0, max=self.family_embedding.num_embeddings - 1)
        emb = self.family_embedding(family_ids)
        hidden = self.shared(torch.cat([x, emb], dim=-1))
        residual = self.residual(hidden).squeeze(-1) + self.family_bias(family_ids).squeeze(-1)
        if anchor_score is None:
            return residual
        return anchor_score + residual


def group_winner_cross_entropy(pred, utility, groups, gap_weight_power=0.0, return_group_weights=False):
    groups = np.asarray(groups).reshape(-1)
    if pred.ndim != 1 or utility.ndim != 1:
        raise ValueError("pred and utility must be 1-D tensors")
    if pred.numel() != utility.numel() or pred.numel() != groups.size:
        raise ValueError("pred, utility, and groups must have the same length")
    losses = []
    gap_weights = []
    for group in sorted(set(groups.tolist())):
        idx_np = np.flatnonzero(groups == group)
        if idx_np.size < 2:
            continue
        idx = torch.as_tensor(idx_np, dtype=torch.long, device=pred.device)
        group_utility = utility[idx]
        winner = torch.argmax(group_utility).view(1)
        losses.append(nn.functional.cross_entropy(pred[idx].view(1, -1), winner))
        if float(gap_weight_power) > 0.0:
            sorted_utility = torch.sort(group_utility, descending=True).values
            gap = torch.clamp(sorted_utility[0] - sorted_utility[1], min=0.0)
            gap_weights.append(torch.pow(gap + 1e-6, float(gap_weight_power)))
    if not losses:
        loss = torch.tensor(0.0, dtype=pred.dtype, device=pred.device)
        if return_group_weights:
            return loss, torch.empty(0, dtype=pred.dtype, device=pred.device)
        return loss
    stacked = torch.stack(losses)
    if float(gap_weight_power) > 0.0:
        weights = torch.stack(gap_weights).to(dtype=pred.dtype, device=pred.device)
        weights = weights / torch.clamp(weights.mean(), min=1e-6)
    else:
        weights = torch.ones_like(stacked)
    loss = (stacked * weights).mean()
    if return_group_weights:
        return loss, weights
    return loss


def build_physical_edge_tensor(x_node, edge_src_idx, edge_dst_idx, valid_mask=None):
    x_node = np.asarray(x_node, dtype=np.float32)
    src_idx = np.asarray(edge_src_idx, dtype=np.int64).clip(min=0)
    dst_idx = np.asarray(edge_dst_idx, dtype=np.int64).clip(min=0)
    src = x_node[:, src_idx, :]
    dst = x_node[:, dst_idx, :]
    delta_xyz = dst[..., :3] - src[..., :3]
    distance = np.linalg.norm(delta_xyz, axis=-1)
    src_speed = src[..., 3] if src.shape[-1] > 3 else np.zeros_like(distance)
    dst_speed = dst[..., 3] if dst.shape[-1] > 3 else np.zeros_like(distance)
    speed_delta = np.abs(dst_speed - src_speed)
    if valid_mask is not None:
        valid = np.asarray(valid_mask, dtype=np.float32).reshape(1, -1)
        distance = distance * valid
        speed_delta = speed_delta * valid
        src_speed = src_speed * valid
        dst_speed = dst_speed * valid
    return {
        "distance": distance.astype(np.float32),
        "speed_delta": speed_delta.astype(np.float32),
        "src_speed": src_speed.astype(np.float32),
        "dst_speed": dst_speed.astype(np.float32),
        "abs_dz": np.abs(delta_xyz[..., 2]).astype(np.float32),
    }


def build_dual_graph_state_summary(arrays, sample_idx):
    idx = int(sample_idx)
    x_node = np.asarray(arrays["x_node"][idx], dtype=np.float32)
    x_link = np.asarray(arrays["x_link"][idx], dtype=np.float32)
    edge_a_hist = np.asarray(arrays["edge_a_hist"][idx], dtype=np.float32)
    link_features = arrays["link_features"]
    edge_action_features = arrays["edge_action_features"]
    mask = valid_edge_mask(arrays)
    physical = build_physical_edge_tensor(
        x_node,
        arrays["edge_src_idx"],
        arrays["edge_dst_idx"],
        mask,
    )

    last_rate = safe_edge_feature(x_link, link_features, "rate_sum")[-1]
    last_rb = safe_edge_feature(x_link, link_features, "allocated_rb_count")[-1]
    last_active_tasks = safe_edge_feature(x_link, link_features, "active_task_count")[-1]
    hist_rb = safe_edge_feature(edge_a_hist, edge_action_features, "rb_total")
    hist_offload = safe_edge_feature(edge_a_hist, edge_action_features, "offload_count")

    rate_valid = masked_values(last_rate, mask)
    rb_valid = masked_values(last_rb, mask)
    active_task_valid = masked_values(last_active_tasks, mask)
    distance_last = masked_values(physical["distance"][-1], mask)
    speed_delta_last = masked_values(physical["speed_delta"][-1], mask)
    src_speed_last = masked_values(physical["src_speed"][-1], mask)
    dst_speed_last = masked_values(physical["dst_speed"][-1], mask)
    rb_hist_valid = masked_values(hist_rb, mask)
    offload_hist_valid = masked_values(hist_offload, mask)

    active = rate_valid > 1e-6
    active_distance = distance_last[active] if active.any() else np.asarray([], dtype=np.float32)
    active_rate = rate_valid[active] if active.any() else np.asarray([], dtype=np.float32)
    return {
        "state_available": 1.0,
        "dual_comm_rate_sum_last": safe_sum(np.maximum(rate_valid, 0.0)),
        "dual_comm_rate_mean_last": safe_mean(np.maximum(rate_valid, 0.0)),
        "dual_comm_active_ratio_last": float(np.mean(active)) if rate_valid.size else 0.0,
        "dual_comm_allocated_rb_sum_last": safe_sum(np.maximum(rb_valid, 0.0)),
        "dual_comm_active_task_sum_last": safe_sum(np.maximum(active_task_valid, 0.0)),
        "dual_action_rb_hist_sum": safe_sum(np.maximum(rb_hist_valid, 0.0)),
        "dual_action_offload_hist_sum": safe_sum(np.maximum(offload_hist_valid, 0.0)),
        "dual_phy_distance_mean_last": safe_mean(distance_last),
        "dual_phy_distance_active_mean_last": safe_mean(active_distance),
        "dual_phy_speed_delta_mean_last": safe_mean(speed_delta_last),
        "dual_phy_src_speed_mean_last": safe_mean(src_speed_last),
        "dual_phy_dst_speed_mean_last": safe_mean(dst_speed_last),
        "dual_comm_phy_rate_per_distance_last": safe_sum(np.maximum(active_rate, 0.0))
        / max(safe_sum(np.maximum(active_distance, 0.0)), 1e-6),
    }


def empty_dual_graph_summary():
    keys = [
        "state_available",
        "dual_comm_rate_sum_last",
        "dual_comm_rate_mean_last",
        "dual_comm_active_ratio_last",
        "dual_comm_allocated_rb_sum_last",
        "dual_comm_active_task_sum_last",
        "dual_action_rb_hist_sum",
        "dual_action_offload_hist_sum",
        "dual_phy_distance_mean_last",
        "dual_phy_distance_active_mean_last",
        "dual_phy_speed_delta_mean_last",
        "dual_phy_src_speed_mean_last",
        "dual_phy_dst_speed_mean_last",
        "dual_comm_phy_rate_per_distance_last",
    ]
    return {key: 0.0 for key in keys}


def enrich_candidates_with_dual_graph_features(df, sample_index, arrays):
    lookup = {}
    for row in sample_index.itertuples(index=False):
        lookup[(int(getattr(row, "seed")), round(float(getattr(row, "input_end_time")), 6))] = int(
            getattr(row, "sample_id")
        )
    rows = []
    zero = empty_dual_graph_summary()
    for row in df.itertuples(index=False):
        key = (int(getattr(row, "seed")), round(float(getattr(row, "decision_time")), 6))
        sample_id = lookup.get(key)
        if sample_id is None:
            summary = dict(zero)
            summary["dual_sample_id"] = -1.0
        else:
            summary = build_dual_graph_state_summary(arrays, sample_id)
            summary["dual_sample_id"] = float(sample_id)
        rows.append(summary)
    feature_df = pd.DataFrame(rows)
    out = df.reset_index(drop=True).copy()
    for col in feature_df.columns:
        out[col] = feature_df[col].to_numpy(dtype=np.float32)
    return out


def build_interaction_features(action_features, state_features, action_names, state_names):
    action = np.asarray(action_features, dtype=np.float32)
    state = np.asarray(state_features, dtype=np.float32)
    if action.shape[0] != state.shape[0]:
        raise ValueError("action and state features must have the same number of rows")
    values = []
    names = []
    for action_idx, action_name in enumerate(action_names):
        for state_idx, state_name in enumerate(state_names):
            values.append(action[:, action_idx] * state[:, state_idx])
            names.append(f"{action_name}_x_{state_name}")
    if not values:
        return np.zeros((action.shape[0], 0), dtype=np.float32), []
    return np.column_stack(values).astype(np.float32), names


def select_action_interaction_columns(df):
    cols = [
        "rb_scale",
        "total_rb",
        "num_rb_tasks",
        "num_offload_overrides",
        "context_num_to_offload_tasks",
    ]
    selected = [col for col in cols if col in df.columns]
    if "action_family" in df.columns:
        family = df["action_family"].astype(str)
        for name in ["rb_count", "offload_target", "mixed_offload_rb"]:
            col = f"family_{name}"
            df[col] = family.eq(name).astype(np.float32)
            selected.append(col)
    return selected


def add_compact_dual_graph_features(df):
    out = df.copy()
    rb = np.log1p(np.maximum(out.get("total_rb", 0.0), 0.0)).astype(np.float32)
    offload = np.log1p(np.maximum(out.get("num_offload_overrides", 0.0), 0.0)).astype(np.float32)
    num_rb_tasks = np.log1p(np.maximum(out.get("num_rb_tasks", 0.0), 0.0)).astype(np.float32)
    rate = np.log1p(np.maximum(out.get("dual_comm_rate_sum_last", 0.0), 0.0)).astype(np.float32)
    distance = np.log1p(np.maximum(out.get("dual_phy_distance_mean_last", 0.0), 0.0)).astype(np.float32)
    active_ratio = np.asarray(out.get("dual_comm_active_ratio_last", 0.0), dtype=np.float32)
    rate_per_distance = np.log1p(np.maximum(out.get("dual_comm_phy_rate_per_distance_last", 0.0), 0.0)).astype(np.float32)
    speed_delta = np.log1p(np.maximum(out.get("dual_phy_speed_delta_mean_last", 0.0), 0.0)).astype(np.float32)
    definitions = {
        "compact_rb_x_rate": rb * rate,
        "compact_rb_x_active_ratio": rb * active_ratio,
        "compact_rb_x_rate_per_distance": rb * rate_per_distance,
        "compact_offload_x_distance": offload * distance,
        "compact_offload_x_speed_delta": offload * speed_delta,
        "compact_num_rb_tasks_x_rate": num_rb_tasks * rate,
    }
    for col, values in definitions.items():
        out[col] = np.asarray(values, dtype=np.float32)
    return out, list(definitions)


def evaluate_dual_graph_decision_head(df, feature_cols, train_idx, test_idx, model_kind="ridge"):
    missing = sorted({"decision_group_id", "target_utility", *feature_cols}.difference(df.columns))
    if missing:
        raise KeyError(f"missing dual-graph decision columns: {missing}")
    x = df.loc[:, feature_cols].to_numpy(dtype=np.float32)
    y = df["target_utility"].to_numpy(dtype=np.float32)
    model = StandardizedRidgeRegressor(alpha=1.0)
    model.fit(x[train_idx], y[train_idx])
    pred = np.asarray(model.predict(x), dtype=np.float32)
    out_df = df.copy()
    out_df["v5_predicted_utility"] = pred
    rows = {}
    for split_name, split_idx in [("train", train_idx), ("test", test_idx), ("all", np.arange(len(out_df)))]:
        metrics = grouped_ranking_metrics(out_df.iloc[split_idx].copy())
        for key, value in metrics.items():
            rows[f"{split_name}_{key}"] = value
    return rows, pred


def evaluate_dual_graph_mlp_rank_head(
    df,
    feature_cols,
    train_idx,
    test_idx,
    epochs=200,
    hidden=16,
    lr=0.01,
    device="cpu",
    reg_weight=1.0,
    rank_weight=1.0,
):
    missing = sorted({"decision_group_id", "target_utility", *feature_cols}.difference(df.columns))
    if missing:
        raise KeyError(f"missing dual-graph decision columns: {missing}")
    x = df.loc[:, feature_cols].to_numpy(dtype=np.float32)
    y = df["target_utility"].to_numpy(dtype=np.float32)
    mean, std = fit_feature_standardizer(x[train_idx])
    x_std = apply_feature_standardizer(x, mean, std)
    model, _, history = train_utility_head(
        x_std[train_idx],
        y[train_idx],
        epochs=epochs,
        lr=lr,
        hidden=hidden,
        device=device,
        groups=df.iloc[train_idx]["decision_group_id"].to_numpy(),
        pair_scope="group",
        reg_weight=reg_weight,
        rank_weight=rank_weight,
    )
    pred = predict_utility(model, x_std, device)
    out_df = df.copy()
    out_df["v5_predicted_utility"] = pred
    rows = {}
    for split_name, split_idx in [("train", train_idx), ("test", test_idx), ("all", np.arange(len(out_df)))]:
        metrics = grouped_ranking_metrics(out_df.iloc[split_idx].copy())
        for key, value in metrics.items():
            rows[f"{split_name}_{key}"] = value
    return rows, pred, history


def train_family_specific_head(
    features,
    utility,
    family_ids,
    anchor_score,
    groups,
    epochs=200,
    hidden=16,
    lr=0.01,
    device="cpu",
    reg_weight=1.0,
    rank_weight=1.0,
    winner_weight=0.0,
    winner_gap_weight_power=0.0,
):
    x_np = np.asarray(features, dtype=np.float32)
    y_np = np.asarray(utility, dtype=np.float32).reshape(-1)
    family_np = np.asarray(family_ids, dtype=np.int64).reshape(-1)
    anchor_np = np.asarray(anchor_score, dtype=np.float32).reshape(-1)
    groups_np = np.asarray(groups).reshape(-1)
    if not (x_np.shape[0] == y_np.size == family_np.size == anchor_np.size == groups_np.size):
        raise ValueError("features, utility, family_ids, anchor_score, and groups must have the same row count")
    num_families = int(max(family_np.max(initial=0) + 1, len(ACTION_FAMILY_NAMES)))
    pairs = build_group_pairwise_pairs(y_np, groups_np)
    model = FamilySpecificUtilityHead(x_np.shape[1], num_families=num_families, hidden=hidden).to(device)
    x = torch.from_numpy(x_np).to(device)
    y = torch.from_numpy(y_np).to(device)
    f = torch.from_numpy(family_np).to(device)
    anchor = torch.from_numpy(anchor_np).to(device)
    pair_tensor = torch.tensor(pairs, dtype=torch.long, device=device) if pairs else None
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    mse = nn.MSELoss()
    history = []
    for epoch in range(1, epochs + 1):
        pred = model(x, f, anchor)
        reg_loss = mse(pred, y)
        if pair_tensor is not None and len(pairs) > 0:
            better = pair_tensor[:, 0]
            worse = pair_tensor[:, 1]
            rank_loss = torch.relu(0.05 - (pred[better] - pred[worse])).mean()
        else:
            rank_loss = torch.tensor(0.0, device=device)
        winner_loss = group_winner_cross_entropy(
            pred,
            y,
            groups_np,
            gap_weight_power=winner_gap_weight_power,
        )
        loss = float(reg_weight) * reg_loss + float(rank_weight) * rank_loss + float(winner_weight) * winner_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        if epoch == 1 or epoch == epochs or epoch % max(1, epochs // 10) == 0:
            history.append(
                {
                    "epoch": epoch,
                    "loss": float(loss.detach().cpu()),
                    "reg_loss": float(reg_loss.detach().cpu()),
                    "rank_loss": float(rank_loss.detach().cpu()),
                    "winner_loss": float(winner_loss.detach().cpu()),
                }
            )
    return model, pd.DataFrame(history)


def predict_family_specific_utility(model, features, family_ids, anchor_score, device):
    x = torch.from_numpy(np.asarray(features, dtype=np.float32)).to(device)
    f = torch.from_numpy(np.asarray(family_ids, dtype=np.int64)).to(device)
    anchor = torch.from_numpy(np.asarray(anchor_score, dtype=np.float32)).to(device)
    model.eval()
    with torch.no_grad():
        return model(x, f, anchor).detach().cpu().numpy().astype(np.float32)


def evaluate_dual_graph_family_mlp_rank_head(
    df,
    feature_cols,
    train_idx,
    test_idx,
    epochs=200,
    hidden=16,
    lr=0.01,
    device="cpu",
    reg_weight=1.0,
    rank_weight=1.0,
    anchor_mode="minus_total_rb",
    winner_weight=0.0,
    winner_gap_weight_power=0.0,
):
    missing = sorted({"decision_group_id", "target_utility", *feature_cols}.difference(df.columns))
    if missing:
        raise KeyError(f"missing dual-graph decision columns: {missing}")
    x = df.loc[:, feature_cols].to_numpy(dtype=np.float32)
    y = df["target_utility"].to_numpy(dtype=np.float32)
    mean, std = fit_feature_standardizer(x[train_idx])
    x_std = apply_feature_standardizer(x, mean, std)
    family_ids, family_names = build_action_family_ids(df)
    anchor = build_anchor_score(df, mode=anchor_mode)
    model, history = train_family_specific_head(
        x_std[train_idx],
        y[train_idx],
        family_ids[train_idx],
        anchor[train_idx],
        df.iloc[train_idx]["decision_group_id"].to_numpy(),
        epochs=epochs,
        hidden=hidden,
        lr=lr,
        device=device,
        reg_weight=reg_weight,
        rank_weight=rank_weight,
        winner_weight=winner_weight,
        winner_gap_weight_power=winner_gap_weight_power,
    )
    pred = predict_family_specific_utility(model, x_std, family_ids, anchor, device)
    out_df = df.copy()
    out_df["v5_predicted_utility"] = pred
    rows = {}
    for split_name, split_idx in [("train", train_idx), ("test", test_idx), ("all", np.arange(len(out_df)))]:
        metrics = grouped_ranking_metrics(out_df.iloc[split_idx].copy())
        for key, value in metrics.items():
            rows[f"{split_name}_{key}"] = value
    rows["num_action_families"] = int(len(family_names))
    return rows, pred, history, family_names


def write_report(summary, metrics_df, output_dir):
    lines = [
        "# World model v5 dual-graph decision head v0",
        "",
        "## Goal",
        "",
        "This experiment tests whether communication-edge history plus physical endpoint geometry improves resource-aware offload/RB candidate ranking.",
        "",
        "## Setup",
        "",
        f"- candidate_csv: `{summary['candidate_csv']}`",
        f"- utility_column: `{summary['utility_column']}`",
        f"- rb_penalty: `{summary['rb_penalty']}`",
        f"- split_strategy: `{summary['split_strategy']}`",
        f"- feature_dim: `{summary['feature_dim']}`",
        f"- state_available_ratio: `{summary['state_available_ratio']}`",
        "",
        "## Metrics",
        "",
        metrics_df.to_markdown(index=False),
        "",
        "## Outputs",
        "",
    ]
    for key, value in summary["outputs"].items():
        lines.append(f"- {key}: `{value}`")
    path = output_dir / "world_model_v5_dual_graph_decision_head_v0_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = ensure_decision_group_id(pd.read_csv(args.candidate_csv))
    df = add_resource_aware_utility(df, args.rb_penalty)
    sample_index = pd.read_csv(args.state_sample_index_csv)
    arrays = load_state_arrays(args.state_dataset_npz)
    df = enrich_candidates_with_dual_graph_features(df, sample_index, arrays)
    if args.require_state_available:
        df = filter_state_available_groups(df)
        if df.empty:
            raise ValueError("no decision groups remain after requiring state_available for every candidate")
    if args.utility_column not in df.columns:
        raise KeyError(f"utility column not found: {args.utility_column}")
    df["target_utility"] = df[args.utility_column].to_numpy(dtype=np.float32)
    action_features = build_candidate_features(df)
    action_cols = [f"action_feature_{idx:03d}" for idx in range(action_features.shape[1])]
    action_df = pd.DataFrame(action_features, columns=action_cols)
    dual_cols = sorted(col for col in df.columns if col.startswith("dual_") and col != "dual_sample_id")
    model_df = pd.concat([df.reset_index(drop=True), action_df], axis=1)
    feature_cols = action_cols + dual_cols
    interaction_cols = []
    if args.include_interactions:
        interaction_action_cols = select_action_interaction_columns(model_df)
        interaction_state_cols = [
            col
            for col in dual_cols
            if col
            in {
                "dual_comm_rate_sum_last",
                "dual_comm_active_ratio_last",
                "dual_comm_allocated_rb_sum_last",
                "dual_phy_distance_mean_last",
                "dual_phy_distance_active_mean_last",
                "dual_phy_speed_delta_mean_last",
                "dual_comm_phy_rate_per_distance_last",
            }
        ]
        interaction_values, interaction_cols = build_interaction_features(
            model_df[interaction_action_cols].to_numpy(dtype=np.float32),
            model_df[interaction_state_cols].to_numpy(dtype=np.float32),
            interaction_action_cols,
            interaction_state_cols,
        )
        interaction_df = pd.DataFrame(interaction_values, columns=[f"interaction_{idx:03d}" for idx in range(len(interaction_cols))])
        model_df = pd.concat([model_df, interaction_df], axis=1)
        feature_cols += list(interaction_df.columns)
    compact_cols = []
    if args.compact_interactions:
        model_df, compact_cols = add_compact_dual_graph_features(model_df)
        feature_cols += compact_cols
    if args.test_seeds:
        train_idx, test_idx = split_by_test_seeds(model_df, args.test_seeds)
        split_strategy = f"test_seeds_{','.join(str(seed) for seed in args.test_seeds)}"
    else:
        train_idx, test_idx = split_decision_groups(model_df, test_fraction=args.test_fraction, seed=args.split_seed)
        split_strategy = f"group_fraction_{args.test_fraction}_seed_{args.split_seed}"
    history = pd.DataFrame()
    if args.model_kind == "mlp_rank":
        metrics, pred, history = evaluate_dual_graph_mlp_rank_head(
            model_df,
            feature_cols,
            train_idx,
            test_idx,
            epochs=args.epochs,
            hidden=args.hidden,
            lr=args.lr,
            device=args.device,
            reg_weight=args.reg_weight,
            rank_weight=args.rank_weight,
        )
        action_family_names = []
    elif args.model_kind == "family_mlp_rank":
        anchor_mode = resolve_family_anchor_mode(args.model_kind, args.anchor_mode)
        metrics, pred, history, action_family_names = evaluate_dual_graph_family_mlp_rank_head(
            model_df,
            feature_cols,
            train_idx,
            test_idx,
            epochs=args.epochs,
            hidden=args.hidden,
            lr=args.lr,
            device=args.device,
            reg_weight=args.reg_weight,
            rank_weight=args.rank_weight,
            anchor_mode=anchor_mode,
            winner_weight=args.winner_weight,
            winner_gap_weight_power=args.winner_gap_weight_power,
        )
    else:
        metrics, pred = evaluate_dual_graph_decision_head(model_df, feature_cols, train_idx, test_idx, args.model_kind)
        action_family_names = []
    out_df = df.copy()
    out_df["v5_predicted_utility"] = pred
    out_df["split"] = "unused"
    out_df.loc[train_idx, "split"] = "train"
    out_df.loc[test_idx, "split"] = "test"
    metrics_df = pd.DataFrame([{**metrics, "model_kind": args.model_kind}])
    predictions_path = args.output_dir / "world_model_v5_dual_graph_decision_head_v0_predictions.csv"
    metrics_path = args.output_dir / "world_model_v5_dual_graph_decision_head_v0_metrics.csv"
    history_path = args.output_dir / "world_model_v5_dual_graph_decision_head_v0_history.csv"
    out_df.to_csv(predictions_path, index=False, encoding="utf-8-sig")
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    if not history.empty:
        history.to_csv(history_path, index=False, encoding="utf-8-sig")
    summary = {
        "candidate_csv": display_path(args.candidate_csv),
        "utility_column": args.utility_column,
        "rb_penalty": float(args.rb_penalty),
        "num_candidates": int(len(df)),
        "num_decision_groups": int(df["decision_group_id"].nunique()),
        "num_train_groups": int(out_df.loc[train_idx, "decision_group_id"].nunique()),
        "num_test_groups": int(out_df.loc[test_idx, "decision_group_id"].nunique()),
        "split_strategy": split_strategy,
        "feature_dim": int(len(feature_cols)),
        "epochs": int(args.epochs),
        "hidden": int(args.hidden),
        "lr": float(args.lr),
        "device": str(args.device),
        "reg_weight": float(args.reg_weight),
        "rank_weight": float(args.rank_weight),
        "winner_weight": float(args.winner_weight),
        "winner_gap_weight_power": float(args.winner_gap_weight_power),
        "anchor_mode": resolve_family_anchor_mode(args.model_kind, args.anchor_mode),
        "action_family_names": action_family_names,
        "dual_feature_columns": dual_cols,
        "include_interactions": bool(args.include_interactions),
        "compact_interactions": bool(args.compact_interactions),
        "interaction_feature_names": interaction_cols,
        "compact_feature_columns": compact_cols,
        "state_available_ratio": float(df["state_available"].mean()) if "state_available" in df.columns else 0.0,
        "metrics": metrics,
        "outputs": {
            "predictions_csv": display_path(predictions_path),
            "metrics_csv": display_path(metrics_path),
        },
    }
    if not history.empty:
        summary["outputs"]["history_csv"] = display_path(history_path)
    report_path = write_report(summary, metrics_df, args.output_dir)
    summary["outputs"]["report_md"] = display_path(report_path)
    summary_path = args.output_dir / "world_model_v5_dual_graph_decision_head_v0_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
