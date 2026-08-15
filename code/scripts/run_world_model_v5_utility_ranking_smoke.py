import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATE_CSV = (
    ROOT
    / "reports"
    / "airfogsim_counterfactual_action_smoke_v0"
    / "airfogsim_counterfactual_action_smoke_v0_candidates.csv"
)
DEFAULT_STATE_SAMPLE_INDEX_CSV = ROOT / "datasets" / "dataset_multiseed_v0" / "sample_index.csv"
DEFAULT_STATE_DATASET_NPZ = ROOT / "datasets" / "world_model_dataset_v0" / "world_model_dataset_v0_samples.npz"
OUTPUT_DIR = ROOT / "reports" / "world_model_v5_utility_ranking_smoke"


def parse_args():
    parser = argparse.ArgumentParser(description="GPU smoke test for a v5 utility/ranking head.")
    parser.add_argument("--candidate-csv", type=Path, default=DEFAULT_CANDIDATE_CSV)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--test-fraction", type=float, default=0.35)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--test-seeds", type=int, nargs="*", default=None)
    parser.add_argument("--include-outcome-features", action="store_true")
    parser.add_argument("--state-sample-index-csv", type=Path, default=None)
    parser.add_argument("--state-dataset-npz", type=Path, default=None)
    parser.add_argument("--require-state-available", action="store_true")
    parser.add_argument("--pair-scope", choices=["global", "group"], default="group")
    parser.add_argument("--utility-column", type=str, default="airfogsim_utility")
    parser.add_argument("--rb-penalty", type=float, default=0.0)
    parser.add_argument("--reg-weight", type=float, default=1.0)
    parser.add_argument("--rank-weight", type=float, default=1.0)
    return parser.parse_args()


def display_path(path):
    path = Path(path)
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def build_candidate_features(df, include_outcome_features=False):
    required = ["rb_scale", "total_rb", "num_rb_tasks", "seed", "decision_time", "horizon"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"missing candidate feature columns: {missing}")
    parts = [
        df["rb_scale"].to_numpy(dtype=np.float32),
        np.log1p(df["total_rb"].to_numpy(dtype=np.float32)),
        np.log1p(df["num_rb_tasks"].to_numpy(dtype=np.float32)),
        df["seed"].to_numpy(dtype=np.float32) / 10.0,
        df["decision_time"].to_numpy(dtype=np.float32) / 20.0,
        df["horizon"].to_numpy(dtype=np.float32) / 10.0,
    ]
    optional_numeric = [
        "cpu_scale",
        "total_cpu",
        "num_offload_overrides",
        "num_cpu_overrides",
        "num_return_route_overrides",
        "context_num_to_offload_tasks",
        "context_num_computing_tasks",
        "context_num_waiting_return_tasks",
        "offload_default_distance",
        "offload_alternative_distance",
        "offload_distance_delta",
        "offload_distance_ratio",
        "offload_default_is_vehicle",
        "offload_default_is_uav",
        "offload_default_is_rsu",
        "offload_default_is_cloud",
        "offload_alternative_is_vehicle",
        "offload_alternative_is_uav",
        "offload_alternative_is_rsu",
        "offload_alternative_is_cloud",
        "offload_target_type_changed",
    ]
    for col in optional_numeric:
        if col in df.columns:
            values = df[col].fillna(0.0).to_numpy(dtype=np.float32)
            if col in {
                "total_cpu",
                "context_num_to_offload_tasks",
                "context_num_computing_tasks",
                "context_num_waiting_return_tasks",
                "offload_default_distance",
                "offload_alternative_distance",
                "offload_distance_ratio",
            }:
                values = np.log1p(np.maximum(values, 0.0))
            elif col == "offload_distance_delta":
                values = np.sign(values) * np.log1p(np.abs(values))
            parts.append(values)
    if "action_family" in df.columns:
        family = df["action_family"].astype(str)
        for name in ["rb_count", "offload_target", "mixed_offload_rb", "cpu_scale", "return_route"]:
            parts.append(family.eq(name).to_numpy(dtype=np.float32))
    if include_outcome_features:
        outcome_required = ["throughput", "delta_done", "delta_failed"]
        missing_outcome = [col for col in outcome_required if col not in df.columns]
        if missing_outcome:
            raise KeyError(f"missing outcome feature columns: {missing_outcome}")
        parts.extend(
            [
                np.log1p(df["throughput"].to_numpy(dtype=np.float32)),
                df["delta_done"].to_numpy(dtype=np.float32),
                df["delta_failed"].to_numpy(dtype=np.float32),
            ]
        )
    state_cols = [col for col in df.columns if col.startswith("state_")]
    for col in sorted(state_cols):
        parts.append(df[col].to_numpy(dtype=np.float32))
    features = np.column_stack(parts)
    return features.astype(np.float32)


def add_resource_aware_utility(df, rb_penalty):
    out = df.copy()
    if "airfogsim_utility" not in out.columns or "total_rb" not in out.columns:
        raise KeyError("airfogsim_utility and total_rb are required for resource-aware utility")
    out["resource_aware_utility"] = (
        out["airfogsim_utility"].to_numpy(dtype=np.float64)
        - float(rb_penalty) * out["total_rb"].to_numpy(dtype=np.float64)
    )
    return out


def load_state_arrays(path):
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def feature_index(names, name):
    values = [str(item) for item in names]
    return values.index(name) if name in values else None


def safe_last_feature_sum(array, names, feature_name):
    idx = feature_index(names, feature_name)
    if idx is None:
        return 0.0
    return float(np.nansum(array[-1, ..., idx]))


def safe_last_feature_mean(array, names, feature_name):
    idx = feature_index(names, feature_name)
    if idx is None:
        return 0.0
    values = np.asarray(array[-1, ..., idx], dtype=np.float32)
    if values.size == 0:
        return 0.0
    return float(np.nanmean(values))


def summarize_world_model_state(arrays, sample_idx):
    idx = int(sample_idx)
    x_task = np.asarray(arrays["x_task"][idx], dtype=np.float32)
    x_link = np.asarray(arrays["x_link"][idx], dtype=np.float32)
    x_node = np.asarray(arrays["x_node"][idx], dtype=np.float32)
    edge_a_hist = np.asarray(arrays["edge_a_hist"][idx], dtype=np.float32)
    task_features = arrays["task_features"]
    link_features = arrays["link_features"]
    node_features = arrays["node_features"]
    edge_action_features = arrays["edge_action_features"]

    rate_idx = feature_index(link_features, "rate_sum")
    last_rate = x_link[-1, :, rate_idx] if rate_idx is not None else np.zeros((x_link.shape[1],), dtype=np.float32)
    dist_idx = feature_index(link_features, "distance")
    last_distance = x_link[-1, :, dist_idx] if dist_idx is not None else np.zeros((x_link.shape[1],), dtype=np.float32)
    speed_idx = feature_index(node_features, "speed")
    last_speed = x_node[-1, :, speed_idx] if speed_idx is not None else np.zeros((x_node.shape[1],), dtype=np.float32)
    rb_idx = feature_index(edge_action_features, "rb_total")
    hist_rb = edge_a_hist[..., rb_idx] if rb_idx is not None else np.zeros(edge_a_hist.shape[:2], dtype=np.float32)

    return {
        "state_available": 1.0,
        "state_num_tasks_last": safe_last_feature_sum(x_task, task_features, "num_tasks"),
        "state_total_task_size_last": safe_last_feature_sum(x_task, task_features, "total_task_size"),
        "state_total_task_cpu_last": safe_last_feature_sum(x_task, task_features, "total_task_cpu"),
        "state_num_to_offload_last": safe_last_feature_sum(x_task, task_features, "num_to_offload"),
        "state_num_computing_last": safe_last_feature_sum(x_task, task_features, "num_computing"),
        "state_num_returning_last": safe_last_feature_sum(x_task, task_features, "num_returning"),
        "state_num_finished_last": safe_last_feature_sum(x_task, task_features, "num_finished"),
        "state_rate_sum_last": float(np.nansum(np.maximum(last_rate, 0.0))),
        "state_rate_mean_last": float(np.nanmean(np.maximum(last_rate, 0.0))) if last_rate.size else 0.0,
        "state_active_link_ratio_last": float(np.mean(last_rate > 1e-6)) if last_rate.size else 0.0,
        "state_allocated_rb_last": safe_last_feature_sum(x_link, link_features, "allocated_rb_count"),
        "state_active_task_count_last": safe_last_feature_sum(x_link, link_features, "active_task_count"),
        "state_mean_link_distance_last": float(np.nanmean(np.maximum(last_distance, 0.0))) if last_distance.size else 0.0,
        "state_mean_node_speed_last": float(np.nanmean(last_speed)) if last_speed.size else 0.0,
        "state_max_node_speed_last": float(np.nanmax(last_speed)) if last_speed.size else 0.0,
        "state_rb_total_hist_sum": float(np.nansum(np.maximum(hist_rb, 0.0))),
        "state_rb_total_last": float(np.nansum(np.maximum(hist_rb[-1], 0.0))) if hist_rb.size else 0.0,
    }


def empty_state_summary():
    keys = [
        "state_available",
        "state_num_tasks_last",
        "state_total_task_size_last",
        "state_total_task_cpu_last",
        "state_num_to_offload_last",
        "state_num_computing_last",
        "state_num_returning_last",
        "state_num_finished_last",
        "state_rate_sum_last",
        "state_rate_mean_last",
        "state_active_link_ratio_last",
        "state_allocated_rb_last",
        "state_active_task_count_last",
        "state_mean_link_distance_last",
        "state_mean_node_speed_last",
        "state_max_node_speed_last",
        "state_rb_total_hist_sum",
        "state_rb_total_last",
    ]
    return {key: 0.0 for key in keys}


def enrich_candidates_with_state_features(df, sample_index, arrays, time_tolerance=1e-6):
    lookup = {}
    for row in sample_index.itertuples(index=False):
        seed = int(getattr(row, "seed"))
        input_end_time = round(float(getattr(row, "input_end_time")), 6)
        sample_id = int(getattr(row, "sample_id"))
        lookup[(seed, input_end_time)] = sample_id
    rows = []
    zero = empty_state_summary()
    for row in df.itertuples(index=False):
        seed = int(getattr(row, "seed"))
        decision_time = round(float(getattr(row, "decision_time")), 6)
        sample_id = lookup.get((seed, decision_time))
        if sample_id is None and time_tolerance > 0:
            candidates = [key for key in lookup if key[0] == seed and abs(key[1] - decision_time) <= time_tolerance]
            sample_id = lookup[candidates[0]] if candidates else None
        if sample_id is None:
            summary = dict(zero)
            summary["state_sample_id"] = -1.0
        else:
            summary = summarize_world_model_state(arrays, sample_id)
            summary["state_sample_id"] = float(sample_id)
        rows.append(summary)
    feature_df = pd.DataFrame(rows)
    out = df.reset_index(drop=True).copy()
    for col in feature_df.columns:
        out[col] = feature_df[col].to_numpy(dtype=np.float32)
    return out


def filter_state_available_groups(df):
    if "state_available" not in df.columns:
        raise KeyError("state_available column is required to filter state-covered groups")
    if "decision_group_id" not in df.columns:
        raise KeyError("decision_group_id column is required to filter state-covered groups")
    group_ok = df.groupby("decision_group_id")["state_available"].transform(lambda values: bool(np.all(values > 0.5)))
    return df[group_ok].reset_index(drop=True).copy()


def ensure_decision_group_id(df):
    df = df.copy()
    if "decision_group_id" in df.columns:
        df["decision_group_id"] = df["decision_group_id"].astype(str)
        return df
    if {"seed", "decision_time"}.issubset(df.columns):
        df["decision_group_id"] = df.apply(
            lambda row: f"seed{int(row['seed'])}_t{float(row['decision_time']):.3f}",
            axis=1,
        )
    else:
        df["decision_group_id"] = "single_group"
    return df


def split_decision_groups(df, test_fraction=0.35, seed=42):
    if "decision_group_id" not in df.columns:
        raise KeyError("decision_group_id column is required for grouped splitting")
    groups = np.array(sorted(df["decision_group_id"].astype(str).unique()))
    if groups.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    if groups.size == 1:
        idx = np.arange(len(df), dtype=int)
        return idx, idx
    rng = np.random.default_rng(seed)
    shuffled = groups.copy()
    rng.shuffle(shuffled)
    n_test = int(round(groups.size * float(test_fraction)))
    n_test = min(groups.size - 1, max(1, n_test))
    test_groups = set(shuffled[:n_test])
    group_values = df["decision_group_id"].astype(str)
    test_mask = group_values.isin(test_groups).to_numpy()
    train_idx = np.flatnonzero(~test_mask)
    test_idx = np.flatnonzero(test_mask)
    return train_idx.astype(int), test_idx.astype(int)


def split_by_test_seeds(df, test_seeds):
    if "seed" not in df.columns:
        raise KeyError("seed column is required for seed-heldout splitting")
    test_seed_set = {int(seed) for seed in test_seeds}
    if not test_seed_set:
        raise ValueError("test_seeds must contain at least one seed")
    seed_values = df["seed"].astype(int)
    test_mask = seed_values.isin(test_seed_set).to_numpy()
    if not test_mask.any():
        raise ValueError(f"no rows found for requested test seeds: {sorted(test_seed_set)}")
    if test_mask.all():
        raise ValueError("all rows are in test seeds; no training rows remain")
    return np.flatnonzero(~test_mask).astype(int), np.flatnonzero(test_mask).astype(int)


def build_pairwise_pairs(utility):
    utility = np.asarray(utility, dtype=np.float32).reshape(-1)
    pairs = []
    for i in range(len(utility)):
        for j in range(len(utility)):
            if utility[i] > utility[j]:
                pairs.append((i, j))
    return pairs


def build_group_pairwise_pairs(utility, groups):
    utility = np.asarray(utility, dtype=np.float32).reshape(-1)
    groups = np.asarray(groups).reshape(-1)
    if utility.size != groups.size:
        raise ValueError("utility and groups must have the same size")
    pairs = []
    for group in sorted(set(groups.tolist())):
        idx = np.flatnonzero(groups == group)
        group_pairs = build_pairwise_pairs(utility[idx])
        pairs.extend((int(idx[better]), int(idx[worse])) for better, worse in group_pairs)
    return pairs


class UtilityHead(nn.Module):
    def __init__(self, in_dim, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_utility_head(
    features,
    utility,
    epochs=200,
    lr=3e-3,
    hidden=32,
    device="cpu",
    groups=None,
    pair_scope="group",
    reg_weight=1.0,
    rank_weight=1.0,
):
    x_np = np.asarray(features, dtype=np.float32)
    y_np = np.asarray(utility, dtype=np.float32).reshape(-1)
    if x_np.shape[0] != y_np.shape[0]:
        raise ValueError("features and utility must have the same number of rows")
    if pair_scope == "group":
        if groups is None:
            raise ValueError("groups are required when pair_scope='group'")
        pairs = build_group_pairwise_pairs(y_np, np.asarray(groups))
    else:
        pairs = build_pairwise_pairs(y_np)
    model = UtilityHead(x_np.shape[1], hidden=hidden).to(device)
    x = torch.from_numpy(x_np).to(device)
    y = torch.from_numpy(y_np).to(device)
    pair_tensor = torch.tensor(pairs, dtype=torch.long, device=device) if pairs else None
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    history = []
    mse = nn.MSELoss()
    for epoch in range(1, epochs + 1):
        pred = model(x)
        reg_loss = mse(pred, y)
        if pair_tensor is not None and len(pairs) > 0:
            better = pair_tensor[:, 0]
            worse = pair_tensor[:, 1]
            rank_loss = torch.relu(0.05 - (pred[better] - pred[worse])).mean()
        else:
            rank_loss = torch.tensor(0.0, device=device)
        loss = float(reg_weight) * reg_loss + float(rank_weight) * rank_loss
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
                }
            )
    with torch.no_grad():
        pred = model(x).detach().cpu().numpy().astype(np.float32)
    return model, pred, pd.DataFrame(history)


def fit_feature_standardizer(train_features):
    train = np.asarray(train_features, dtype=np.float32)
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def apply_feature_standardizer(features, mean, std):
    return ((np.asarray(features, dtype=np.float32) - mean) / std).astype(np.float32)


def ranking_metrics(true_utility, predicted_utility):
    true = np.asarray(true_utility, dtype=np.float64)
    pred = np.asarray(predicted_utility, dtype=np.float64)
    true_order = np.argsort(true)
    pred_order = np.argsort(pred)
    if len(true) > 1:
        spearman = float(np.corrcoef(np.argsort(true_order), np.argsort(pred_order))[0, 1])
    else:
        spearman = float("nan")
    best_true = int(np.argmax(true))
    best_pred = int(np.argmax(pred))
    denom = float(np.max(true) - np.min(true))
    regret = float(np.max(true) - true[best_pred])
    return {
        "spearman": spearman,
        "best_true_idx": best_true,
        "best_pred_idx": best_pred,
        "top1_hit": float(best_true == best_pred),
        "top1_regret": regret,
        "normalized_top1_regret": regret / denom if denom > 1e-12 else 0.0,
        "utility_rmse": float(np.sqrt(np.mean((true - pred) ** 2))),
    }


def grouped_ranking_metrics(df, group_col="decision_group_id"):
    required = [group_col, "target_utility", "v5_predicted_utility"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"missing grouped ranking columns: {missing}")
    rows = []
    for group_id, part in df.groupby(group_col, dropna=False):
        if len(part) < 2:
            continue
        metrics = ranking_metrics(part["target_utility"], part["v5_predicted_utility"])
        metrics[group_col] = str(group_id)
        metrics["num_candidates"] = int(len(part))
        rows.append(metrics)
    group_df = pd.DataFrame(rows)
    if group_df.empty:
        return {
            "num_groups": 0,
            "top1_hit_mean": float("nan"),
            "normalized_top1_regret_mean": float("nan"),
            "spearman_mean": float("nan"),
            "utility_rmse_mean": float("nan"),
        }
    return {
        "num_groups": int(len(group_df)),
        "top1_hit_mean": float(group_df["top1_hit"].mean()),
        "normalized_top1_regret_mean": float(group_df["normalized_top1_regret"].mean()),
        "spearman_mean": float(group_df["spearman"].mean(skipna=True)),
        "utility_rmse_mean": float(group_df["utility_rmse"].mean()),
    }


def predict_utility(model, features, device):
    x = torch.from_numpy(np.asarray(features, dtype=np.float32)).to(device)
    model.eval()
    with torch.no_grad():
        return model(x).detach().cpu().numpy().astype(np.float32)


def write_report(summary, candidate_df, history_df, group_metrics_df, output_dir):
    lines = [
        "# World model v5 utility/ranking GPU smoke",
        "",
        "## Goal",
        "",
        "This smoke test verifies that a utility/ranking head can train on counterfactual AirFogSim candidate labels using the selected device.",
        "",
        "## Device",
        "",
        f"- device: `{summary['device']}`",
        f"- cuda_available: `{summary['cuda_available']}`",
        f"- gpu_name: `{summary['gpu_name']}`",
        f"- feature_mode: `{summary['feature_mode']}`",
        f"- pair_scope: `{summary['pair_scope']}`",
        f"- feature_dim: `{summary['feature_dim']}`",
        f"- utility_column: `{summary['utility_column']}`",
        f"- rb_penalty: `{summary['rb_penalty']}`",
        f"- reg_weight: `{summary['reg_weight']}`",
        f"- rank_weight: `{summary['rank_weight']}`",
        f"- state_available_ratio: `{summary['state_available_ratio']}`",
        f"- require_state_available: `{summary.get('require_state_available', False)}`",
        f"- split_strategy: `{summary.get('split_strategy', '')}`",
        "",
        "## Candidate Predictions",
        "",
        candidate_df.to_markdown(index=False),
        "",
        "## Group Metrics",
        "",
        group_metrics_df.to_markdown(index=False) if not group_metrics_df.empty else "No grouped metrics available.",
        "",
        "## Training History",
        "",
        history_df.to_markdown(index=False),
        "",
        "## Metrics",
        "",
        pd.DataFrame([summary["metrics"]]).to_markdown(index=False),
        "",
        "## Outputs",
        "",
    ]
    for key, value in summary["outputs"].items():
        lines.append(f"- {key}: `{value}`")
    path = output_dir / "world_model_v5_utility_ranking_smoke_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    df = ensure_decision_group_id(pd.read_csv(args.candidate_csv))
    if args.rb_penalty != 0.0 or args.utility_column == "resource_aware_utility":
        df = add_resource_aware_utility(df, args.rb_penalty)
    if args.utility_column not in df.columns:
        raise KeyError(f"utility column not found: {args.utility_column}")
    df["target_utility"] = df[args.utility_column].to_numpy(dtype=np.float32)
    state_feature_cols = []
    feature_mode = "action_only"
    if args.state_sample_index_csv is not None or args.state_dataset_npz is not None:
        sample_index_path = args.state_sample_index_csv or DEFAULT_STATE_SAMPLE_INDEX_CSV
        state_dataset_path = args.state_dataset_npz or DEFAULT_STATE_DATASET_NPZ
        sample_index = pd.read_csv(sample_index_path)
        state_arrays = load_state_arrays(state_dataset_path)
        df = enrich_candidates_with_state_features(df, sample_index, state_arrays)
        state_feature_cols = [col for col in df.columns if col.startswith("state_")]
        feature_mode = "state_action"
        if args.require_state_available:
            df = filter_state_available_groups(df)
            if df.empty:
                raise ValueError("no decision groups remain after requiring state_available for every candidate")
    if args.test_seeds:
        train_idx, test_idx = split_by_test_seeds(df, args.test_seeds)
        split_strategy = f"test_seeds_{','.join(str(seed) for seed in args.test_seeds)}"
    else:
        train_idx, test_idx = split_decision_groups(df, test_fraction=args.test_fraction, seed=args.split_seed)
        split_strategy = f"group_fraction_{args.test_fraction}_seed_{args.split_seed}"
    features = build_candidate_features(df, include_outcome_features=args.include_outcome_features)
    feature_mean, feature_std = fit_feature_standardizer(features[train_idx])
    features_std = apply_feature_standardizer(features, feature_mean, feature_std)
    utility = df["target_utility"].to_numpy(dtype=np.float32)
    model, train_pred, history = train_utility_head(
        features_std[train_idx],
        utility[train_idx],
        epochs=args.epochs,
        lr=args.lr,
        hidden=args.hidden,
        device=args.device,
        groups=df.iloc[train_idx]["decision_group_id"].to_numpy(),
        pair_scope=args.pair_scope,
        reg_weight=args.reg_weight,
        rank_weight=args.rank_weight,
    )
    pred = predict_utility(model, features_std, args.device)
    out_df = df.copy()
    out_df["v5_predicted_utility"] = pred
    out_df["split"] = "unused"
    out_df.loc[train_idx, "split"] = "train"
    out_df.loc[test_idx, "split"] = "test"
    global_metrics = ranking_metrics(utility, pred)
    train_metrics = ranking_metrics(utility[train_idx], pred[train_idx])
    test_metrics = ranking_metrics(utility[test_idx], pred[test_idx])
    group_rows = []
    for split_name, split_idx in [("train", train_idx), ("test", test_idx), ("all", np.arange(len(out_df)))]:
        part = out_df.iloc[split_idx].copy()
        group_summary = grouped_ranking_metrics(part)
        group_summary["split"] = split_name
        group_rows.append(group_summary)
    group_metrics_df = pd.DataFrame(group_rows)
    metrics = {
        **test_metrics,
        "group_top1_hit_mean": group_metrics_df.loc[group_metrics_df["split"].eq("test"), "top1_hit_mean"].iloc[0],
        "group_normalized_top1_regret_mean": group_metrics_df.loc[
            group_metrics_df["split"].eq("test"), "normalized_top1_regret_mean"
        ].iloc[0],
        "group_spearman_mean": group_metrics_df.loc[group_metrics_df["split"].eq("test"), "spearman_mean"].iloc[0],
    }
    metrics_path = args.output_dir / "world_model_v5_utility_ranking_smoke_predictions.csv"
    history_path = args.output_dir / "world_model_v5_utility_ranking_smoke_history.csv"
    group_metrics_path = args.output_dir / "world_model_v5_utility_ranking_smoke_group_metrics.csv"
    out_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    history.to_csv(history_path, index=False, encoding="utf-8-sig")
    group_metrics_df.to_csv(group_metrics_path, index=False, encoding="utf-8-sig")
    summary = {
        "device": str(args.device),
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "num_candidates": int(len(df)),
        "num_decision_groups": int(df["decision_group_id"].nunique()),
        "num_train_rows": int(len(train_idx)),
        "num_test_rows": int(len(test_idx)),
        "num_train_groups": int(out_df.loc[train_idx, "decision_group_id"].nunique()),
        "num_test_groups": int(out_df.loc[test_idx, "decision_group_id"].nunique()),
        "split_strategy": split_strategy,
        "include_outcome_features": bool(args.include_outcome_features),
        "feature_mode": feature_mode,
        "pair_scope": args.pair_scope,
        "utility_column": args.utility_column,
        "rb_penalty": float(args.rb_penalty),
        "reg_weight": float(args.reg_weight),
        "rank_weight": float(args.rank_weight),
        "feature_dim": int(features.shape[1]),
        "state_feature_columns": state_feature_cols,
        "state_available_ratio": float(df["state_available"].mean()) if "state_available" in df.columns else 0.0,
        "require_state_available": bool(args.require_state_available),
        "epochs": int(args.epochs),
        "metrics": metrics,
        "split_metrics": {
            "global": global_metrics,
            "train": train_metrics,
            "test": test_metrics,
        },
        "group_metrics": group_metrics_df.to_dict(orient="records"),
        "outputs": {
            "predictions_csv": display_path(metrics_path),
            "history_csv": display_path(history_path),
            "group_metrics_csv": display_path(group_metrics_path),
        },
    }
    report_path = write_report(summary, out_df, history, group_metrics_df, args.output_dir)
    summary["outputs"]["report_md"] = display_path(report_path)
    summary_path = args.output_dir / "world_model_v5_utility_ranking_smoke_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
