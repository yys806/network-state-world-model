"""CPU-only diagnosis for the two failed P4 validation gates.

This script replays only the three expanded, non-locked CPU runs.  It does not
train, change model/data contracts, or read the locked-test split.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pi_jwm.formal_airfogsim_window_v1 import FormalAirFogSimWindowDataset, FormalWindowConfig
from pi_jwm.formal_dual_graph_world_model_v1 import FormalDualGraphWorldModel, FormalWorldModelConfig
from pi_jwm.airfogsim_tensor_v2 import EDGE_FEATURES


THRESHOLDS = (0.1, 0.3, 0.5, 0.7, 0.9)
RUN_NAMES = (
    "pi_jwm_formal_p4_cpu_h20_repair_expanded_seed20260830",
    "pi_jwm_formal_p4_cpu_h20_repair_expanded_seed20260831",
    "pi_jwm_formal_p4_cpu_h20_repair_expanded_seed20260832",
)


def _f1(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, Any]:
    predicted = scores >= float(threshold)
    labels = labels.astype(bool)
    tp = int(np.count_nonzero(predicted & labels))
    fp = int(np.count_nonzero(predicted & ~labels))
    fn = int(np.count_nonzero(~predicted & labels))
    denominator = 2 * tp + fp + fn
    return {
        "threshold": float(threshold),
        "f1": float(2 * tp / denominator) if denominator else None,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def _auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    labels = labels.astype(bool)
    positive = int(labels.sum())
    negative = int((~labels).sum())
    if not positive or not negative:
        return None
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks_sorted = np.arange(1, len(scores) + 1, dtype=np.float64)
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_scores)) + 1]
    ends = np.r_[starts[1:], len(scores)]
    for start, end in zip(starts, ends):
        ranks_sorted[start:end] = 0.5 * (start + 1 + end)
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = ranks_sorted
    return float((ranks[labels].sum() - positive * (positive + 1) / 2) / (positive * negative))


def _average_precision(scores: np.ndarray, labels: np.ndarray) -> float | None:
    labels = labels.astype(bool)
    if not labels.any():
        return None
    order = np.argsort(-scores, kind="mergesort")
    ranked = labels[order]
    cumulative = np.cumsum(ranked)
    positions = np.flatnonzero(ranked)
    return float(np.mean(cumulative[positions] / (positions + 1)))


def _load_model(run: Path, tensor_root: Path, stats: dict[str, Any], config: dict[str, Any]) -> torch.nn.Module:
    contract = json.loads((tensor_root / "tensor_contract.json").read_text(encoding="utf-8"))
    model = FormalDualGraphWorldModel(
        FormalWorldModelConfig(
            mode="coupled_dual_gnn",
            hidden_dim=int(config["hidden_dim"]),
            history_steps=int(config["history_steps"]),
            horizon_steps=int(config["horizon_steps"]),
            residual_state_prediction=True,
            zero_init_residual_state_heads=bool(config.get("zero_init_residual_state_heads", False)),
            residual_state_scale=float(config.get("residual_state_scale", 1.0)),
            deterministic_rule_layer=True,
            rule_layer_stats=stats,
            n_rb=int(contract.get("n_rb", 1)),
        )
    )
    checkpoint = torch.load(
        run / "checkpoints" / "coupled_dual_gnn_residual__best.pt",
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _action_rb(sample: dict[str, Any]) -> np.ndarray:
    """Aggregate RB action counts onto explicit physical-edge endpoints."""
    action = np.asarray(sample["future_action"]["task_action"])
    present = np.asarray(sample["future_action"]["task_action_present"]).astype(bool)
    source = np.asarray(sample["future_action"]["task_action_source_node_index"])
    target = np.asarray(sample["future_action"]["task_action_node_index"])
    edges = np.asarray(sample["static"]["physical_edge_endpoint_index"])
    horizon, _, _ = action.shape
    result = np.zeros((horizon, edges.shape[0]), dtype=np.float32)
    for step in range(horizon):
        rb = present[step] & (action[step, :, 1] > 0) & (source[step, :, 2] >= 0) & (target[step, :, 2] >= 0)
        if not rb.any():
            continue
        for task_index in np.flatnonzero(rb):
            match = (edges[:, 0] == source[step, task_index, 2]) & (edges[:, 1] == target[step, task_index, 2])
            result[step, match] += max(float(action[step, task_index, 3]), 0.0)
    return result


def _sample_index(dataset: FormalAirFogSimWindowDataset, sample_ids: list[str]) -> list[int]:
    lookup = {str(row["sample_id"]): index for index, row in enumerate(dataset.rows)}
    missing = [sample_id for sample_id in sample_ids if sample_id not in lookup]
    if missing:
        raise KeyError(f"sample IDs not found: {missing[:3]}")
    return [lookup[sample_id] for sample_id in sample_ids]


def _run_split(
    model: torch.nn.Module,
    dataset: FormalAirFogSimWindowDataset,
    sample_ids: list[str],
    stats: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, float]]:
    indices = _sample_index(dataset, sample_ids)
    loader = DataLoader(Subset(dataset, indices), batch_size=8, shuffle=False, num_workers=0)
    scores: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    predicted_rb: list[np.ndarray] = []
    target_rb: list[np.ndarray] = []
    action_rb: list[np.ndarray] = []
    persistence_rb: list[np.ndarray] = []
    rb_mask: list[np.ndarray] = []
    rb_index = EDGE_FEATURES.index("allocated_rb_count")
    rb_stats = stats["features"]["physical_edge_state"]
    rb_mean = float(rb_stats["mean"][rb_index])
    rb_scale = float(rb_stats["scale"][rb_index])
    with torch.no_grad():
        offset = 0
        for batch in loader:
            prediction = model(batch)
            target = batch["target"]
            static = batch["static"]
            activity = target.get("aggregate_link_activity", target["link_activity"]).bool()
            activity_mask = target.get("aggregate_link_activity_mask")
            if activity_mask is None:
                activity_mask = torch.ones_like(activity, dtype=torch.bool)
            edge_valid = torch.all(static["physical_edge_endpoint_index"] >= 0, dim=-1)
            valid = activity_mask & edge_valid[:, None, :].expand_as(activity)
            scores.append(torch.sigmoid(prediction["link_activity_logits"])[valid].cpu().numpy())
            labels.append(activity[valid].cpu().numpy())

            predicted_rb.append(
                prediction["physical_edge_state_mean"][..., rb_index].cpu().numpy() * rb_scale + rb_mean
            )
            target_rb.append(target["aggregate_rb_occupancy"].cpu().numpy())
            rb_mask.append(target["aggregate_rb_occupancy_mask"].cpu().numpy())
            for local_index in range(len(batch["target"]["aggregate_rb_occupancy"])):
                sample = dataset[indices[offset + local_index]]
                action_rb.append(_action_rb(sample))
                persistence_last = sample["history"]["aggregate_rb_occupancy"][-1].numpy()
                persistence_rb.append(np.broadcast_to(persistence_last, (target["aggregate_rb_occupancy"].shape[1], persistence_last.shape[0])).copy())
            offset += len(batch["target"]["aggregate_rb_occupancy"])

    flat_scores = np.concatenate(scores)
    flat_labels = np.concatenate(labels).astype(bool)
    fixed_rows = [_f1(flat_scores, flat_labels, threshold) for threshold in THRESHOLDS]
    calibrated_threshold = max(fixed_rows, key=lambda row: (row["f1"] if row["f1"] is not None else -1.0, -abs(row["threshold"] - 0.5)))["threshold"]

    predicted = np.concatenate(predicted_rb, axis=0).reshape(-1, predicted_rb[0].shape[-1])
    target = np.concatenate(target_rb, axis=0).reshape(-1, target_rb[0].shape[-1])
    action = np.concatenate(action_rb, axis=0)
    persistence = np.concatenate(persistence_rb, axis=0)
    observed = np.concatenate(rb_mask, axis=0).reshape(-1, rb_mask[0].shape[-1]).astype(bool)
    valid_edge = observed & np.isfinite(target)

    def rb_summary(values: np.ndarray) -> dict[str, float | int]:
        predicted_total = (values * valid_edge).sum(axis=1)
        target_total = (target * valid_edge).sum(axis=1)
        errors = np.abs(predicted_total - target_total)
        errors = errors[np.any(valid_edge, axis=1)]
        return {
            "mae": float(errors.mean()) if errors.size else 0.0,
            "count": int(errors.size),
            "mean_predicted_total": float(predicted_total.mean()),
            "mean_target_total": float(target_total.mean()),
        }

    action_error = np.abs((action - target) * valid_edge)
    action_total_error = np.abs((action * valid_edge).sum(axis=1) - (target * valid_edge).sum(axis=1))
    persistence_total_error = np.abs((persistence * valid_edge).sum(axis=1) - (target * valid_edge).sum(axis=1))
    result = {
        "sample_count": len(sample_ids),
        "valid_count": int(flat_labels.size),
        "positive_count": int(flat_labels.sum()),
        "negative_count": int((~flat_labels).sum()),
        "roc_auc": _auc(flat_scores, flat_labels),
        "average_precision": _average_precision(flat_scores, flat_labels),
        "fixed_thresholds": fixed_rows,
        "best_fixed_threshold": float(calibrated_threshold),
        "rb": {
            "learned": rb_summary(predicted),
            "action_derived": rb_summary(action),
            "persistence": rb_summary(persistence),
            "action_edge_mae": float(action_error[valid_edge].mean()) if action_error[valid_edge].size else 0.0,
            "action_total_mae": float(action_total_error.mean()) if action_total_error.size else 0.0,
            "persistence_total_mae": float(persistence_total_error.mean()) if persistence_total_error.size else 0.0,
            "action_exact_edge_fraction": float(np.mean(action[valid_edge] == target[valid_edge])) if action[valid_edge].size else 0.0,
            "action_observed_edge_count": int(valid_edge.sum()),
            "target_total_mean": float((target * valid_edge).sum(axis=1).mean()),
            "action_total_mean": float((action * valid_edge).sum(axis=1).mean()),
        },
    }
    return result, {"calibrated_threshold": float(calibrated_threshold)}


def diagnose(runs_root: Path, output: Path) -> dict[str, Any]:
    rows = []
    for run_name in RUN_NAMES:
        run = runs_root / run_name
        config = json.loads((run / "config.json").read_text(encoding="utf-8"))
        tensor_root = Path(config["tensor_root"])
        stats = json.loads((tensor_root / "normalization_stats.json").read_text(encoding="utf-8"))
        datasets = {
            split: FormalAirFogSimWindowDataset(
                tensor_root,
                split=split,
                config=FormalWindowConfig(int(config["history_steps"]), int(config["horizon_steps"])),
                stats=stats,
                normalize=True,
            )
            for split in ("validation", "calibration")
        }
        sample_ids = json.loads((run / "sample_ids.json").read_text(encoding="utf-8"))
        model = _load_model(run, tensor_root, stats, config)
        split_rows = {}
        for split in ("validation", "calibration"):
            split_rows[split], _ = _run_split(model, datasets[split], sample_ids[split], stats)
        rows.append({"run": run_name, "seed": int(config["seed"]), "splits": split_rows})
    report = {
        "schema_version": "PI-JWM-formal-p4-failure-diagnosis-v1",
        "device": "cpu",
        "runs": rows,
        "locked_test_accessed": False,
        "gpu_started": False,
        "scope": ["fixed validation link-activity thresholds", "action-derived RB occupancy alignment"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, default=CODE_ROOT / "artifacts" / "experiments")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(diagnose(args.runs_root, args.output), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
