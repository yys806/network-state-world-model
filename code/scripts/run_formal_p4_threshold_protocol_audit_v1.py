"""Audit link-activity threshold protocols on existing non-locked CPU runs.

This is read-only: it reuses existing checkpoints and calibration/validation
sample IDs.  Validation is never used to choose a threshold.
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
from pi_jwm.formal_binary_calibration_v1 import correct_positive_weighted_probability
from pi_jwm.formal_dual_graph_world_model_v1 import FormalDualGraphWorldModel, FormalWorldModelConfig

RUN_NAMES = (
    "pi_jwm_formal_p4_cpu_h20_repair_expanded_seed20260830",
    "pi_jwm_formal_p4_cpu_h20_repair_expanded_seed20260831",
    "pi_jwm_formal_p4_cpu_h20_repair_expanded_seed20260832",
)
RAW_CANDIDATES = (0.1, 0.3, 0.5, 0.7, 0.9)
CORRECTED_PROBABILITY_CANDIDATES = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5)
POS_WEIGHT = 50.0


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


def _select(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row["f1"] is not None]
    return max(valid, key=lambda row: (row["f1"], -abs(row["threshold"] - 0.5))) if valid else rows[0]


def _load_model(run: Path, config: dict[str, Any], stats: dict[str, Any], n_rb: int) -> torch.nn.Module:
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
            n_rb=n_rb,
        )
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model


def _collect(model: torch.nn.Module, dataset: FormalAirFogSimWindowDataset, sample_ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    index_by_id = {str(row["sample_id"]): index for index, row in enumerate(dataset.rows)}
    indices = [index_by_id[sample_id] for sample_id in sample_ids]
    loader = DataLoader(Subset(dataset, indices), batch_size=8, shuffle=False, num_workers=0)
    scores: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            prediction = model(batch)
            target = batch["target"]
            activity = target.get("aggregate_link_activity", target["link_activity"]).bool()
            activity_mask = target.get("aggregate_link_activity_mask")
            if activity_mask is None:
                activity_mask = torch.ones_like(activity, dtype=torch.bool)
            edge_valid = torch.all(batch["static"]["physical_edge_endpoint_index"] >= 0, dim=-1)
            valid = activity_mask.bool() & edge_valid[:, None, :].expand_as(activity)
            scores.append(torch.sigmoid(prediction["link_activity_logits"])[valid].cpu().numpy())
            labels.append(activity[valid].cpu().numpy())
    return np.concatenate(scores), np.concatenate(labels).astype(bool)


def audit(runs_root: Path, output: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for run_name in RUN_NAMES:
        run = runs_root / run_name
        config = json.loads((run / "config.json").read_text(encoding="utf-8"))
        if config.get("device") != "cpu" or config.get("locked_test_accessed"):
            raise ValueError(f"run is outside the allowed non-locked CPU scope: {run}")
        tensor_root = Path(config["tensor_root"])
        contract = json.loads((tensor_root / "tensor_contract.json").read_text(encoding="utf-8"))
        stats = json.loads((tensor_root / "normalization_stats.json").read_text(encoding="utf-8"))
        window = FormalWindowConfig(int(config["history_steps"]), int(config["horizon_steps"]))
        datasets = {
            split: FormalAirFogSimWindowDataset(tensor_root, split=split, config=window, stats=stats, normalize=True)
            for split in ("calibration", "validation")
        }
        sample_ids = json.loads((run / "sample_ids.json").read_text(encoding="utf-8"))
        model = _load_model(run, config, stats, int(contract.get("n_rb", 1)))
        calibration_scores, calibration_labels = _collect(model, datasets["calibration"], sample_ids["calibration"])
        validation_scores, validation_labels = _collect(model, datasets["validation"], sample_ids["validation"])
        calibration_probability = correct_positive_weighted_probability(torch.from_numpy(calibration_scores), POS_WEIGHT).numpy()
        validation_probability = correct_positive_weighted_probability(torch.from_numpy(validation_scores), POS_WEIGHT).numpy()

        raw_calibration_rows = [_f1(calibration_scores, calibration_labels, threshold) for threshold in RAW_CANDIDATES]
        raw_selected = _select(raw_calibration_rows)
        corrected_calibration_rows = [
            _f1(calibration_probability, calibration_labels, threshold)
            for threshold in CORRECTED_PROBABILITY_CANDIDATES
        ]
        corrected_selected = _select(corrected_calibration_rows)
        raw_equivalent_for_corrected = float(POS_WEIGHT * corrected_selected["threshold"] / (1.0 - corrected_selected["threshold"] + POS_WEIGHT * corrected_selected["threshold"]))
        protocols = {
            "raw_calibration_selected": {
                "selection_split": "calibration",
                "raw_threshold": float(raw_selected["threshold"]),
                "calibration": raw_selected,
                "validation": _f1(validation_scores, validation_labels, raw_selected["threshold"]),
            },
            "raw_fixed_0.5": {
                "selection_split": "pre_registered",
                "raw_threshold": 0.5,
                "calibration": _f1(calibration_scores, calibration_labels, 0.5),
                "validation": _f1(validation_scores, validation_labels, 0.5),
            },
            "ordinary_probability_fixed_0.5": {
                "selection_split": "pre_registered",
                "ordinary_probability_threshold": 0.5,
                "equivalent_raw_threshold": float(POS_WEIGHT / (1.0 + POS_WEIGHT)),
                "calibration": _f1(calibration_probability, calibration_labels, 0.5),
                "validation": _f1(validation_probability, validation_labels, 0.5),
            },
            "ordinary_probability_calibration_selected": {
                "selection_split": "calibration",
                "ordinary_probability_candidates": list(CORRECTED_PROBABILITY_CANDIDATES),
                "ordinary_probability_threshold": float(corrected_selected["threshold"]),
                "equivalent_raw_threshold": raw_equivalent_for_corrected,
                "calibration": corrected_selected,
                "validation": _f1(validation_probability, validation_labels, corrected_selected["threshold"]),
            },
        }
        rows.append({
            "run": run_name,
            "seed": int(config["seed"]),
            "pos_weight": POS_WEIGHT,
            "calibration_count": int(calibration_labels.size),
            "validation_count": int(validation_labels.size),
            "calibration_positive_rate": float(calibration_labels.mean()),
            "validation_positive_rate": float(validation_labels.mean()),
            "protocols": protocols,
        })
    report = {
        "schema_version": "PI-JWM-formal-p4-threshold-protocol-audit-v1",
        "device": "cpu",
        "runs": rows,
        "threshold_selection_policy": "calibration_only; validation is evaluation only",
        "pos_weight_probability_correction": "p = s / (50 - 49s)",
        "locked_test_accessed": False,
        "gpu_started": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, default=CODE_ROOT / "artifacts" / "experiments")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.runs_root, args.output), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
