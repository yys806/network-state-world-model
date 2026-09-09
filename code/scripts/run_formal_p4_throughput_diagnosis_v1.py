"""CPU-only, read-only diagnosis of aggregate throughput errors in P4.

The script replays existing non-locked CPU checkpoints and reports, for every
forecast step, learned aggregate rate, target aggregate rate, persistence
aggregate rate, absolute errors, and signed bias.  It does not train or alter
the model, tensor contract, training protocol, or locked-test state.
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

from pi_jwm.airfogsim_tensor_v2 import EDGE_FEATURES
from pi_jwm.formal_airfogsim_window_v1 import FormalAirFogSimWindowDataset, FormalWindowConfig
from pi_jwm.formal_dual_graph_world_model_v1 import FormalDualGraphWorldModel, FormalWorldModelConfig
from pi_jwm.formal_world_model_baselines_v1 import build_rule_prediction


RUN_NAMES = (
    "pi_jwm_formal_p4_cpu_h20_repair_expanded_seed20260830",
    "pi_jwm_formal_p4_cpu_h20_repair_expanded_seed20260831",
    "pi_jwm_formal_p4_cpu_h20_repair_expanded_seed20260832",
)


def summarize_stepwise_totals(
    predicted: np.ndarray,
    target: np.ndarray,
    persistence: np.ndarray,
    observed: np.ndarray,
) -> dict[str, Any]:
    """Summarize masked edge totals and signed errors for each forecast step."""

    arrays = (predicted, target, persistence, observed)
    if predicted.ndim != 3 or any(value.shape != predicted.shape for value in arrays):
        raise ValueError("predicted, target, persistence, and observed must have the same [batch, step, edge] shape")
    observed = observed.astype(bool, copy=False)
    predicted_total = np.where(observed, predicted, 0.0).sum(axis=2)
    target_total = np.where(observed, target, 0.0).sum(axis=2)
    persistence_total = np.where(observed, persistence, 0.0).sum(axis=2)

    def summarize(values: np.ndarray) -> dict[str, float]:
        return {
            "mean": float(values.mean()),
            "mae": float(np.abs(values).mean()),
            "bias": float(values.mean()),
        }

    steps: list[dict[str, Any]] = []
    for step in range(predicted.shape[1]):
        learned_error = predicted_total[:, step] - target_total[:, step]
        persistence_error = persistence_total[:, step] - target_total[:, step]
        observed_step = observed[:, step]
        steps.append(
            {
                "step": step + 1,
                "count": int(predicted.shape[0]),
                "observed_edge_count": int(observed_step.sum()),
                "target_total_mean": float(target_total[:, step].mean()),
                "learned_total_mean": float(predicted_total[:, step].mean()),
                "persistence_total_mean": float(persistence_total[:, step].mean()),
                "learned_total_mae": float(np.abs(learned_error).mean()),
                "persistence_total_mae": float(np.abs(persistence_error).mean()),
                "learned_bias": float(learned_error.mean()),
                "persistence_bias": float(persistence_error.mean()),
                "learned_better_fraction": float(np.mean(np.abs(learned_error) < np.abs(persistence_error))),
            }
        )

    learned_error = predicted_total - target_total
    persistence_error = persistence_total - target_total
    return {
        "batch_count": int(predicted.shape[0]),
        "step_count": int(predicted.shape[1]),
        "overall": {
            "target": summarize(target_total),
            "learned": summarize(predicted_total),
            "persistence": summarize(persistence_total),
            "learned_total_mae": float(np.abs(learned_error).mean()),
            "persistence_total_mae": float(np.abs(persistence_error).mean()),
            "learned_minus_persistence_mae": float(np.abs(learned_error).mean() - np.abs(persistence_error).mean()),
            "learned_bias": float(learned_error.mean()),
            "persistence_bias": float(persistence_error.mean()),
            "learned_better_fraction": float(np.mean(np.abs(learned_error) < np.abs(persistence_error))),
        },
        "steps": steps,
    }


def _subset_for_ids(dataset: FormalAirFogSimWindowDataset, sample_ids: list[str]) -> Subset:
    index_by_id = {str(row["sample_id"]): index for index, row in enumerate(dataset.rows)}
    missing = [sample_id for sample_id in sample_ids if sample_id not in index_by_id]
    if missing:
        raise KeyError(f"sample IDs not found: {missing[:3]}")
    return Subset(dataset, [index_by_id[sample_id] for sample_id in sample_ids])


def _load_model(run_dir: Path) -> torch.nn.Module:
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    tensor_root = Path(config["tensor_root"])
    stats = json.loads((tensor_root / "normalization_stats.json").read_text(encoding="utf-8"))
    contract = json.loads((tensor_root / "tensor_contract.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(
        run_dir / "checkpoints" / "coupled_dual_gnn_residual__best.pt",
        map_location="cpu",
        weights_only=True,
    )
    model = FormalDualGraphWorldModel(FormalWorldModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    if config.get("device") != "cpu" or config.get("locked_test_accessed"):
        raise ValueError(f"run is outside the CPU non-locked scope: {run_dir}")
    if int(contract.get("horizon_steps", config["horizon_steps"])) != int(config["horizon_steps"]):
        raise ValueError(f"run horizon disagrees with tensor contract: {run_dir}")
    return model, tensor_root, stats, config


def _collect_split(
    model: torch.nn.Module,
    dataset: FormalAirFogSimWindowDataset,
    sample_ids: list[str],
    stats: dict[str, Any],
    batch_size: int,
) -> dict[str, Any]:
    loader = DataLoader(_subset_for_ids(dataset, sample_ids), batch_size=batch_size, shuffle=False, num_workers=0)
    rate_index = EDGE_FEATURES.index("rate_sum")
    edge_stats = stats["features"]["physical_edge_state"]
    rate_mean = float(edge_stats["mean"][rate_index])
    rate_scale = float(edge_stats["scale"][rate_index])
    predicted_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    persistence_parts: list[np.ndarray] = []
    observed_parts: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            prediction = model(batch)
            persistence_prediction = build_rule_prediction("last_persistence", batch, stats)
            static = batch["static"]
            target = batch["target"]
            edge_valid = torch.all(static["physical_edge_endpoint_index"] >= 0, dim=-1)
            target_mask = target["aggregate_link_rate_sum_mask"].bool()
            observed = edge_valid[:, None, :] & target_mask
            predicted_rate = prediction["physical_edge_state_mean"][..., rate_index] * rate_scale + rate_mean
            persistence_rate = persistence_prediction["physical_edge_state_mean"][..., rate_index] * rate_scale + rate_mean
            predicted_parts.append(predicted_rate.cpu().numpy())
            target_parts.append(target["aggregate_link_rate_sum"].cpu().numpy())
            persistence_parts.append(persistence_rate.cpu().numpy())
            observed_parts.append(observed.cpu().numpy())
    predicted = np.concatenate(predicted_parts, axis=0)
    target = np.concatenate(target_parts, axis=0)
    persistence = np.concatenate(persistence_parts, axis=0)
    observed = np.concatenate(observed_parts, axis=0)
    result = summarize_stepwise_totals(predicted, target, persistence, observed)
    result["sample_ids_reused"] = True
    result["sample_count"] = len(sample_ids)
    result["unit"] = "Mbps"
    return result


def diagnose(runs_root: Path, output: Path) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    for run_name in RUN_NAMES:
        run_dir = runs_root / run_name
        model, tensor_root, stats, config = _load_model(run_dir)
        sample_ids = json.loads((run_dir / "sample_ids.json").read_text(encoding="utf-8"))
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
        runs.append(
            {
                "run": run_name,
                "seed": int(config["seed"]),
                "tensor_root": str(tensor_root.resolve()),
                "splits": {
                    # A fixed diagnostic batch keeps this read-only replay tractable;
                    # it does not change the trained model or any training protocol.
                    split: _collect_split(model, datasets[split], list(sample_ids[split]), stats, 8)
                    for split in ("validation", "calibration")
                },
            }
        )
    report = {
        "schema_version": "PI-JWM-formal-p4-throughput-diagnosis-v1",
        "device": "cpu",
        "training_performed": False,
        "gpu_started": False,
        "locked_test_accessed": False,
        "scope": "per-step aggregate link-rate totals against target and last-step persistence",
        "runs": runs,
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
