"""CPU-only diagnosis of link-activity F1 instability for the repaired P4 runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pi_jwm.formal_airfogsim_window_v1 import FormalAirFogSimWindowDataset, FormalWindowConfig
from pi_jwm.formal_dual_graph_world_model_v1 import FormalDualGraphWorldModel, FormalWorldModelConfig


THRESHOLDS = (0.1, 0.3, 0.5, 0.7, 0.9)


def _auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    labels = labels.astype(bool)
    positives = int(labels.sum())
    negatives = int((~labels).sum())
    if not positives or not negatives:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    for value in np.unique(scores):
        tied = np.flatnonzero(scores == value)
        if len(tied) > 1:
            ranks[tied] = ranks[tied].mean()
    return float((ranks[labels].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def _average_precision(scores: np.ndarray, labels: np.ndarray) -> float | None:
    labels = labels.astype(bool)
    positives = int(labels.sum())
    if not positives:
        return None
    order = np.argsort(-scores, kind="mergesort")
    ranked = labels[order]
    cumulative = np.cumsum(ranked)
    positive_positions = np.flatnonzero(ranked)
    return float(np.mean(cumulative[positive_positions] / (positive_positions + 1)))


def _threshold_row(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, float | int | None]:
    predicted = scores >= threshold
    labels = labels.astype(bool)
    tp = int(np.count_nonzero(predicted & labels))
    fp = int(np.count_nonzero(predicted & ~labels))
    fn = int(np.count_nonzero(~predicted & labels))
    denominator = 2 * tp + fp + fn
    return {
        "threshold": threshold,
        "f1": float(2 * tp / denominator) if denominator else None,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def _collect(model: torch.nn.Module, dataset: FormalAirFogSimWindowDataset, sample_ids: list[str], stats: dict) -> tuple[np.ndarray, np.ndarray]:
    index = {str(row["sample_id"]): i for i, row in enumerate(dataset.rows)}
    subset = Subset(dataset, [index[sample_id] for sample_id in sample_ids])
    loader = DataLoader(subset, batch_size=1, shuffle=False, num_workers=0)
    scores: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            prediction = model(batch)
            target = batch["target"]
            static = batch["static"]
            activity = target.get("aggregate_link_activity", target["link_activity"]).bool()
            activity_mask = target.get("aggregate_link_activity_mask")
            if activity_mask is None:
                activity_mask = torch.ones_like(activity, dtype=torch.bool)
            edge_valid = torch.all(static["physical_edge_endpoint_index"] >= 0, dim=-1)
            valid = activity_mask.bool() & edge_valid[:, None, :].expand_as(activity)
            scores.append(torch.sigmoid(prediction["link_activity_logits"])[valid].cpu().numpy())
            labels.append(activity[valid].cpu().numpy())
    return np.concatenate(scores), np.concatenate(labels).astype(bool)


def diagnose(runs_root: Path, output_path: Path) -> dict:
    rows = []
    for run in sorted(runs_root.glob("pi_jwm_formal_p4_cpu_h20_repair*")):
        config = json.loads((run / "config.json").read_text(encoding="utf-8"))
        tensor_root = Path(config["tensor_root"])
        stats = json.loads((tensor_root / "normalization_stats.json").read_text(encoding="utf-8"))
        contract = FormalWindowConfig(
            history_steps=int(config["history_steps"]),
            horizon_steps=int(config["horizon_steps"]),
        )
        datasets = {
            split: FormalAirFogSimWindowDataset(
                tensor_root, split=split, config=contract, stats=stats, normalize=True
            )
            for split in ("validation", "calibration")
        }
        sample_ids = json.loads((run / "sample_ids.json").read_text(encoding="utf-8"))
        checkpoint = torch.load(
            run / "checkpoints" / "coupled_dual_gnn_residual__best.pt",
            map_location="cpu",
            weights_only=True,
        )
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
                n_rb=int(json.loads((tensor_root / "tensor_contract.json").read_text(encoding="utf-8")).get("n_rb", 1)),
            )
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        run_row = {"run": run.name, "seed": int(config["seed"]), "splits": {}}
        for split in ("validation", "calibration"):
            scores, labels = _collect(model, datasets[split], sample_ids[split], stats)
            run_row["splits"][split] = {
                "sample_count": len(sample_ids[split]),
                "valid_count": int(len(labels)),
                "positive_count": int(labels.sum()),
                "negative_count": int((~labels).sum()),
                "positive_rate": float(labels.mean()),
                "roc_auc": _auc(scores, labels),
                "average_precision": _average_precision(scores, labels),
                "score_quantiles": dict(zip(("q0", "q01", "q05", "q50", "q95", "q99", "q100"), np.quantile(scores, [0, .01, .05, .5, .95, .99, 1]).tolist())),
                "positive_score_quantiles": dict(zip(("q0", "q10", "q50", "q90", "q100"), np.quantile(scores[labels], [0, .1, .5, .9, 1]).tolist())),
                "negative_score_quantiles": dict(zip(("q0", "q10", "q50", "q90", "q100"), np.quantile(scores[~labels], [0, .1, .5, .9, 1]).tolist())),
                "mean_positive_score": float(scores[labels].mean()),
                "mean_negative_score": float(scores[~labels].mean()),
                "thresholds": [_threshold_row(scores, labels, threshold) for threshold in THRESHOLDS],
            }
        rows.append(run_row)
    report = {
        "schema_version": "PI-JWM-formal-p4-link-activity-diagnosis-v1",
        "device": "cpu",
        "locked_test_accessed": False,
        "gpu_started": False,
        "runs": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, default=CODE_ROOT / "artifacts" / "experiments")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(diagnose(args.runs_root, args.output), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
