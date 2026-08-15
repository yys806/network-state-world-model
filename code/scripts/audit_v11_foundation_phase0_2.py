"""Audit PI-JWM v11-candidate foundation assumptions for phases 0-2.

This script is intentionally CPU-first and read-only with respect to model
weights.  It checks metric coverage, dual-graph/action feature availability,
coupled action-token consistency, action/target correlations, and a small
frozen-world-model fusion diagnostic.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from pi_jwm.v6_data import (  # noqa: E402
    V6WorldModelDataset,
    collate_v6_world_model_batch,
    load_world_model_arrays,
    make_normalization_stats,
)
from pi_jwm.v8_training import build_v8_model_from_arrays  # noqa: E402
from run_world_model_v8_full_training import add_event_memory_features, resolve_seed_splits  # noqa: E402


DEFAULT_EXPERIMENT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "experiments"
    / "pi_jwm_v9_expanded_v2_gpu_20260619"
    / "v2_hurdle_baseline"
)


ACTION_TOKEN_GROUPS = {
    "offload_count": ("offload_count",),
    "rb_coupled": ("rb_task_count", "rb_total"),
    "cpu_coupled": ("cpu_task_count", "cpu_total"),
    "return_count": ("return_count",),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit PI-JWM v11 foundation phases 0-2.")
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-fusion-samples", type=int, default=256)
    parser.add_argument("--skip-fusion", action="store_true")
    return parser.parse_args()


def choose_device(value: str) -> torch.device:
    if value == "cpu":
        return torch.device("cpu")
    if value == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_project_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def safe_float(value: object) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3 or float(np.std(x)) < 1e-12 or float(np.std(y)) < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(values, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return None
    return pearson(rankdata(x[mask]), rankdata(y[mask]))


def feature_names(arrays: dict[str, np.ndarray], key: str) -> list[str]:
    return [str(item) for item in arrays.get(key, np.array([], dtype=str)).tolist()]


def build_metric_contract(arrays: dict[str, np.ndarray]) -> list[dict]:
    link_features = set(feature_names(arrays, "link_features"))
    task_features = set(feature_names(arrays, "task_features"))
    node_features = set(feature_names(arrays, "node_features"))
    action_features = set(feature_names(arrays, "edge_action_features"))
    rows = [
        {
            "phase": "0",
            "metric": "active_rate_rmse",
            "status": "available",
            "evidence": "y_link_rate + y_link_active",
            "note": "Primary current v11-candidate score.",
        },
        {
            "phase": "0",
            "metric": "link_rate_rmse",
            "status": "available",
            "evidence": "y_link_rate and full link-rate predictions",
            "note": "Must be constrained while improving active-rate.",
        },
        {
            "phase": "0",
            "metric": "activity_precision_recall_f1",
            "status": "available",
            "evidence": "y_link_active",
            "note": "Low F1 has not explained most v11 ranking gains, but it remains a guardrail.",
        },
        {
            "phase": "0",
            "metric": "task_state_rmse",
            "status": "available",
            "evidence": "y_task",
            "note": "Task aggregate prediction exists.",
        },
        {
            "phase": "0",
            "metric": "node_state_rmse",
            "status": "available",
            "evidence": "y_node",
            "note": "Physical-state prediction exists.",
        },
        {
            "phase": "0",
            "metric": "rb_usage",
            "status": "available",
            "evidence": "allocated_rb_count in x_link; rb_task_count/rb_total in edge actions"
            if {"allocated_rb_count"} <= link_features and {"rb_task_count", "rb_total"} <= action_features
            else "missing one or more RB fields",
            "note": "Usable as evaluation/constraint signal.",
        },
        {
            "phase": "0",
            "metric": "cpu_usage",
            "status": "available_proxy",
            "evidence": "node cpu, total_task_cpu, cpu_task_count/cpu_total"
            if {"cpu"} <= node_features and {"total_task_cpu"} <= task_features and {"cpu_task_count", "cpu_total"} <= action_features
            else "missing one or more CPU fields",
            "note": "CPU allocation is represented, but true CPU service latency still needs interface-level definition.",
        },
        {
            "phase": "0",
            "metric": "queue_backlog",
            "status": "available_proxy",
            "evidence": "num_tasks/num_to_offload/num_computing/num_returning"
            if {"num_tasks", "num_to_offload", "num_computing", "num_returning"} <= task_features
            else "missing queue proxy fields",
            "note": "Queue proxy exists in aggregate task state.",
        },
        {
            "phase": "0",
            "metric": "delay",
            "status": "partial_proxy",
            "evidence": "mean_deadline/mean_priority" if {"mean_deadline", "mean_priority"} <= task_features else "missing deadline proxies",
            "note": "This is not true end-to-end delay; add explicit delay target before final decision-interface training.",
        },
        {
            "phase": "0",
            "metric": "constraint_violation",
            "status": "partial_proxy",
            "evidence": "valid_edge_node + action/resource fields",
            "note": "Can audit obvious invalid edge/resource actions; full simulator constraint outcome should be added later.",
        },
        {
            "phase": "0",
            "metric": "ood_uncertainty",
            "status": "missing_or_diagnostic_only",
            "evidence": "no ensemble/uncertainty target found in dataset arrays",
            "note": "Can approximate with train-distribution distance now; true uncertainty needs model support.",
        },
    ]
    return rows


def build_feature_audit(arrays: dict[str, np.ndarray]) -> list[dict]:
    rows: list[dict] = []
    specs = [
        ("physical_node", "x_node", "node_features"),
        ("information_edge", "x_link", "link_features"),
        ("task_state", "x_task", "task_features"),
        ("edge_action_history", "edge_a_hist", "edge_action_features"),
        ("edge_future_action", "edge_a_future", "edge_action_features"),
        ("node_target", "y_node", "node_features"),
        ("task_target", "y_task", "task_features"),
    ]
    for domain, array_key, feature_key in specs:
        values = arrays[array_key]
        names = feature_names(arrays, feature_key)
        rows.append(
            {
                "phase": "1",
                "domain": domain,
                "array": array_key,
                "shape": "x".join(str(part) for part in values.shape),
                "feature_count": len(names) if names else values.shape[-1],
                "features": "|".join(names),
                "nonfinite_count": int(np.size(values) - np.isfinite(values).sum()) if np.issubdtype(values.dtype, np.number) else "",
            }
        )
    rows.append(
        {
            "phase": "1",
            "domain": "graph_topology",
            "array": "edge_src_idx/edge_dst_idx/valid_edge_node",
            "shape": f"{arrays['edge_src_idx'].shape[0]} edges",
            "feature_count": "",
            "features": "",
            "valid_edge_count": int(np.asarray(arrays["valid_edge_node"]).sum()),
            "node_count": int(arrays["x_node"].shape[2]),
            "edge_count": int(arrays["x_link"].shape[2]),
        }
    )
    return rows


def build_coupled_consistency(arrays: dict[str, np.ndarray], split_indices: dict[str, np.ndarray]) -> tuple[list[dict], dict]:
    names = feature_names(arrays, "edge_action_features")
    name_to_idx = {name: idx for idx, name in enumerate(names)}
    future = arrays["edge_a_future"]
    rows: list[dict] = []
    summary: dict[str, object] = {"action_features": names, "groups": ACTION_TOKEN_GROUPS}
    for split, indices in split_indices.items():
        values = future[indices]
        for group, fields in ACTION_TOKEN_GROUPS.items():
            missing = [field for field in fields if field not in name_to_idx]
            if missing:
                rows.append({"phase": "1", "split": split, "group": group, "status": "missing", "missing": "|".join(missing)})
                continue
            group_values = [values[..., name_to_idx[field]] for field in fields]
            row = {
                "phase": "1",
                "split": split,
                "group": group,
                "status": "available",
                "positions": int(group_values[0].size),
            }
            for field, field_values in zip(fields, group_values):
                row[f"{field}_nonzero_rate"] = float(np.mean(field_values > 0.0))
                row[f"{field}_sum"] = float(np.sum(field_values))
            if len(fields) == 2:
                left, right = group_values
                row["both_zero_rate"] = float(np.mean((left == 0.0) & (right == 0.0)))
                row["both_positive_rate"] = float(np.mean((left > 0.0) & (right > 0.0)))
                row["count_positive_total_zero_rate"] = float(np.mean((left > 0.0) & (right == 0.0)))
                row["total_positive_count_zero_rate"] = float(np.mean((left == 0.0) & (right > 0.0)))
                row["pearson_count_total"] = safe_float(pearson(left, right))
                row["spearman_count_total"] = safe_float(spearman(left, right))
            rows.append(row)
    return rows, summary


def build_action_target_correlations(arrays: dict[str, np.ndarray], split_indices: dict[str, np.ndarray]) -> list[dict]:
    action_names = feature_names(arrays, "edge_action_features")
    task_names = feature_names(arrays, "task_features")
    rows: list[dict] = []
    target_specs = {
        "target_active_link_count": lambda idx: arrays["y_link_active"][idx].sum(axis=2),
        "target_active_rate_sum": lambda idx: (arrays["y_link_rate"][idx] * arrays["y_link_active"][idx]).sum(axis=2),
        "target_link_rate_sum": lambda idx: arrays["y_link_rate"][idx].sum(axis=2),
    }
    for task_name in ["num_tasks", "total_task_size", "total_task_cpu", "num_to_offload", "num_computing", "num_returning", "num_finished"]:
        if task_name in task_names:
            task_idx = task_names.index(task_name)
            target_specs[f"target_task_{task_name}"] = lambda idx, task_idx=task_idx: arrays["y_task"][idx, :, task_idx]
    for split, indices in split_indices.items():
        actions = arrays["edge_a_future"][indices]
        action_totals = {name: actions[..., pos].sum(axis=2) for pos, name in enumerate(action_names)}
        if {"rb_total", "cpu_total"} <= set(action_names):
            action_totals["step_rb_cpu_total"] = (
                action_totals["rb_total"] + action_totals["cpu_total"]
            )
        if {"rb_task_count", "cpu_task_count"} <= set(action_names):
            action_totals["step_rb_cpu_task_count"] = (
                action_totals["rb_task_count"] + action_totals["cpu_task_count"]
            )
        for action_name, action_value in action_totals.items():
            for target_name, target_fn in target_specs.items():
                target_value = target_fn(indices)
                rows.append(
                    {
                        "phase": "1",
                        "split": split,
                        "action_signal": action_name,
                        "target_signal": target_name,
                        "pearson": safe_float(pearson(action_value, target_value)),
                        "spearman": safe_float(spearman(action_value, target_value)),
                        "n": int(np.asarray(action_value).size),
                    }
                )
    return rows


def build_high_load_slices(arrays: dict[str, np.ndarray], split_indices: dict[str, np.ndarray]) -> list[dict]:
    action_names = feature_names(arrays, "edge_action_features")
    name_to_idx = {name: idx for idx, name in enumerate(action_names)}
    rows: list[dict] = []
    if "rb_total" not in name_to_idx or "cpu_total" not in name_to_idx:
        return rows
    for split, indices in split_indices.items():
        actions = arrays["edge_a_future"][indices]
        load = actions[..., name_to_idx["rb_total"]].sum(axis=2) + actions[..., name_to_idx["cpu_total"]].sum(axis=2)
        active_count = arrays["y_link_active"][indices].sum(axis=2)
        active_rate_sum = (arrays["y_link_rate"][indices] * arrays["y_link_active"][indices]).sum(axis=2)
        quantiles = np.quantile(load.reshape(-1), [0.0, 0.5, 0.75, 0.9, 0.95, 1.0])
        for label, lo, hi in [
            ("all", -np.inf, np.inf),
            ("p50_p75", quantiles[1], quantiles[2]),
            ("p75_p90", quantiles[2], quantiles[3]),
            ("p90_p95", quantiles[3], quantiles[4]),
            ("p95_max", quantiles[4], np.inf),
        ]:
            mask = (load >= lo) & (load <= hi)
            rows.append(
                {
                    "phase": "2",
                    "split": split,
                    "slice": label,
                    "count": int(mask.sum()),
                    "load_mean": float(load[mask].mean()) if mask.any() else None,
                    "active_link_count_mean": float(active_count[mask].mean()) if mask.any() else None,
                    "active_rate_sum_mean": float(active_rate_sum[mask].mean()) if mask.any() else None,
                }
            )
    return rows


def load_fusion_model(summary: dict, arrays: dict[str, np.ndarray], checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    config = summary["config"]
    activity_memory_dim = int(config.get("activity_memory_dim", 0))
    model = build_v8_model_from_arrays(
        arrays,
        hidden_dim=int(config["hidden_dim"]),
        graph_mode=config["graph_mode"],
        fusion_mode=config["fusion_mode"],
        fusion_num_heads=int(config.get("fusion_num_heads", 4)),
        active_rate_auxiliary=bool(config.get("active_rate_auxiliary", False)),
        active_rate_head_mode=config.get("active_rate_head_mode", "mlp"),
        num_rate_experts=int(config.get("num_rate_experts", 4)),
        rate_output_mode=config.get("model_rate_output_mode", "direct"),
        history_encoder=config.get("history_encoder", "mean"),
        latent_transition_mode=config.get("latent_transition_mode", "message_passing"),
        adaptive_edge_context=config.get("adaptive_edge_context", "none"),
        adaptive_edge_topk=int(config.get("adaptive_edge_topk", 8)),
        activity_memory_dim=activity_memory_dim,
        activity_memory_routing="activity_only" if activity_memory_dim > 0 else "none",
        return_message_diagnostics=True,
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def build_fusion_diagnostics(
    summary: dict,
    arrays: dict[str, np.ndarray],
    test_idx: np.ndarray,
    checkpoint_path: Path,
    device: torch.device,
    batch_size: int,
    max_samples: int,
) -> tuple[list[dict], dict]:
    if max_samples <= 0:
        return [], {"status": "skipped", "reason": "max_samples <= 0"}
    config = summary["config"]
    split = summary["split_seed_spec"]
    train_idx, _, _, _ = resolve_seed_splits(
        arrays["sample_seed"],
        train_seeds=split["train_seeds"],
        val_seeds=split["val_seeds"],
        test_seeds=split["test_seeds"],
    )
    stats = make_normalization_stats(
        arrays,
        train_idx,
        rate_target_transform=config.get("rate_target_transform", "raw"),
    )
    selected = np.asarray(test_idx[:max_samples], dtype=np.int64)
    dataset = V6WorldModelDataset(
        arrays,
        selected,
        stats,
        rate_target_transform=config.get("rate_target_transform", "raw"),
        future_action_mode=config.get("future_action_mode", "full"),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_v6_world_model_batch)
    model = load_fusion_model(summary, arrays, checkpoint_path, device)
    attention_parts = []
    weight_parts = []
    with torch.no_grad():
        for batch, _ in loader:
            batch = batch_to_device(batch, device)
            outputs = model(batch)
            if "message_edge_fusion_attention" in outputs:
                attention_parts.append(outputs["message_edge_fusion_attention"].detach().cpu().numpy())
            if "message_edge_fusion_weights" in outputs:
                weight_parts.append(outputs["message_edge_fusion_weights"].detach().cpu().numpy())
    rows: list[dict] = []
    modality = ["physical", "information", "action"]
    summary_out = {
        "status": "available",
        "fusion_mode": config.get("fusion_mode"),
        "samples": int(selected.size),
        "checkpoint": str(checkpoint_path),
    }
    if attention_parts:
        attention = np.concatenate(attention_parts, axis=0)
        matrix = attention.mean(axis=(0, 1))
        std = attention.std(axis=(0, 1))
        key_mean = attention.mean(axis=(0, 1, 2))
        summary_out["attention_key_mean"] = {modality[i]: float(key_mean[i]) for i in range(3)}
        for i, query in enumerate(modality):
            for j, key in enumerate(modality):
                rows.append(
                    {
                        "phase": "1",
                        "diagnostic": "fusion_attention",
                        "query": query,
                        "key": key,
                        "mean": float(matrix[i, j]),
                        "std": float(std[i, j]),
                    }
                )
    if weight_parts:
        weights = np.concatenate(weight_parts, axis=0)
        mean = weights.mean(axis=(0, 1))
        std = weights.std(axis=(0, 1))
        summary_out["weight_mean"] = {modality[i]: float(mean[i]) for i in range(3)}
        for i, name in enumerate(modality):
            rows.append(
                {
                    "phase": "1",
                    "diagnostic": "fusion_weights",
                    "modality": name,
                    "mean": float(mean[i]),
                    "std": float(std[i]),
                }
            )
    if not rows:
        summary_out["status"] = "missing"
        summary_out["reason"] = "model did not return fusion diagnostics"
    return rows, summary_out


def batch_to_device(batch, device: torch.device):
    from pi_jwm.v6_dual_graph import V6DualGraphBatch

    return V6DualGraphBatch(
        node_history=batch.node_history.to(device),
        physical_edge_history=batch.physical_edge_history.to(device),
        info_edge_history=batch.info_edge_history.to(device),
        action_history=batch.action_history.to(device),
        future_actions=batch.future_actions.to(device),
        task_history=batch.task_history.to(device),
        link_rate_baseline=batch.link_rate_baseline.to(device) if batch.link_rate_baseline is not None else None,
    )


def write_report(output_dir: Path, summary: dict) -> Path:
    lines = [
        "# PI-JWM v11 Foundation Phase 0-2 Audit",
        "",
        "This audit is CPU-first and uses frozen artifacts. It does not train a new policy or world model.",
        "",
        "## Key Facts",
        "",
        f"- Dataset: `{summary['dataset_dir']}`",
        f"- Experiment: `{summary['experiment_dir']}`",
        f"- Checkpoint: `{summary['checkpoint_path']}`",
        f"- Samples: train `{summary['split_counts']['train']}`, val `{summary['split_counts']['val']}`, test `{summary['split_counts']['test']}`",
        f"- Graph: nodes `{summary['shape_summary']['nodes']}`, edges `{summary['shape_summary']['edges']}`, horizon `{summary['shape_summary']['horizon']}`",
        "",
        "## Phase Readout",
        "",
        "- Phase 0 metric contract is written to `metric_contract.csv`; missing or proxy-only metrics are intentionally marked as gaps.",
        "- Phase 1 graph/action audit is written to `graph_feature_audit.csv`, `coupled_action_consistency.csv`, `action_target_correlations.csv`, and `fusion_diagnostics.csv`.",
        "- Phase 2 high-load slice evidence is written to `high_load_slices.csv`; action-ablation metrics should be run as the companion world-model sensitivity check.",
        "",
        "## Outputs",
        "",
    ]
    for name, path in summary["outputs"].items():
        lines.append(f"- {name}: `{path}`")
    path = output_dir / "foundation_phase0_2_audit.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    experiment_dir = args.experiment_dir if args.experiment_dir.is_absolute() else PROJECT_ROOT / args.experiment_dir
    summary_path = experiment_dir / "v8_full_training_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    dataset_dir = resolve_project_path(summary["dataset_dir"])
    arrays = load_world_model_arrays(dataset_dir)
    if summary["config"].get("use_event_memory_features", False):
        arrays = add_event_memory_features(arrays)
    split = summary["split_seed_spec"]
    train_idx, val_idx, test_idx, _ = resolve_seed_splits(
        arrays["sample_seed"],
        train_seeds=split["train_seeds"],
        val_seeds=split["val_seeds"],
        test_seeds=split["test_seeds"],
    )
    split_indices = {"train": train_idx, "val": val_idx, "test": test_idx}
    checkpoint_path = args.checkpoint_path or experiment_dir / "checkpoints" / "v8_dual_best.pt"

    metric_rows = build_metric_contract(arrays)
    feature_rows = build_feature_audit(arrays)
    consistency_rows, consistency_summary = build_coupled_consistency(arrays, split_indices)
    correlation_rows = build_action_target_correlations(arrays, split_indices)
    high_load_rows = build_high_load_slices(arrays, split_indices)
    if args.skip_fusion:
        fusion_rows, fusion_summary = [], {"status": "skipped", "reason": "--skip-fusion"}
    else:
        fusion_rows, fusion_summary = build_fusion_diagnostics(
            summary,
            arrays,
            test_idx,
            checkpoint_path,
            choose_device(args.device),
            batch_size=args.batch_size,
            max_samples=args.max_fusion_samples,
        )

    outputs = {
        "metric_contract_csv": output_dir / "metric_contract.csv",
        "graph_feature_audit_csv": output_dir / "graph_feature_audit.csv",
        "coupled_action_consistency_csv": output_dir / "coupled_action_consistency.csv",
        "action_target_correlations_csv": output_dir / "action_target_correlations.csv",
        "high_load_slices_csv": output_dir / "high_load_slices.csv",
        "fusion_diagnostics_csv": output_dir / "fusion_diagnostics.csv",
        "summary_json": output_dir / "summary.json",
    }
    write_csv(outputs["metric_contract_csv"], metric_rows)
    write_csv(outputs["graph_feature_audit_csv"], feature_rows)
    write_csv(outputs["coupled_action_consistency_csv"], consistency_rows)
    write_csv(outputs["action_target_correlations_csv"], correlation_rows)
    write_csv(outputs["high_load_slices_csv"], high_load_rows)
    write_csv(outputs["fusion_diagnostics_csv"], fusion_rows)

    out_summary = {
        "module": "audit_v11_foundation_phase0_2",
        "experiment_dir": str(experiment_dir),
        "dataset_dir": str(dataset_dir),
        "checkpoint_path": str(checkpoint_path),
        "split_counts": {
            "train": int(train_idx.size),
            "val": int(val_idx.size),
            "test": int(test_idx.size),
        },
        "shape_summary": {
            "samples": int(arrays["x_node"].shape[0]),
            "history": int(arrays["x_node"].shape[1]),
            "horizon": int(arrays["y_link_rate"].shape[1]),
            "nodes": int(arrays["x_node"].shape[2]),
            "edges": int(arrays["x_link"].shape[2]),
            "node_features": feature_names(arrays, "node_features"),
            "link_features": feature_names(arrays, "link_features"),
            "task_features": feature_names(arrays, "task_features"),
            "edge_action_features": feature_names(arrays, "edge_action_features"),
        },
        "metric_status_counts": {
            status: sum(1 for row in metric_rows if row["status"] == status)
            for status in sorted({row["status"] for row in metric_rows})
        },
        "coupled_summary": consistency_summary,
        "fusion_summary": fusion_summary,
        "outputs": {name: str(path) for name, path in outputs.items()},
    }
    report_path = write_report(output_dir, out_summary)
    out_summary["outputs"]["report_md"] = str(report_path)
    outputs["summary_json"].write_text(json.dumps(out_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
