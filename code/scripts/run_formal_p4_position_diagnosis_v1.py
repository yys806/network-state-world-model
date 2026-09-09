"""CPU-only, validation-only diagnosis of P4 node-position rollout errors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pi_jwm.formal_airfogsim_window_v1 import FormalAirFogSimWindowDataset, FormalWindowConfig
from pi_jwm.formal_dual_graph_world_model_v1 import FormalDualGraphWorldModel, FormalWorldModelConfig


COORDINATES = ("x", "y", "z")
RUN_NAMES = (
    "seed_20260830",
    "seed_20260831",
    "seed_20260832",
)


def _mean_or_none(values: np.ndarray) -> float | None:
    return float(values.mean()) if values.size else None


def summarize_position_errors(
    target: np.ndarray,
    predicted: np.ndarray,
    persistence: np.ndarray,
    present: np.ndarray,
    node_types: np.ndarray,
) -> dict[str, Any]:
    """Summarize absolute position errors using only present target nodes."""
    target = np.asarray(target, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    persistence = np.asarray(persistence, dtype=np.float64)
    present = np.asarray(present, dtype=bool)
    if target.shape != predicted.shape or target.shape != persistence.shape:
        raise ValueError("target, predicted, and persistence shapes must match")
    if target.ndim != 3 or target.shape[-1] != 3 or present.shape != target.shape[:2]:
        raise ValueError("expected [step,node,3] states and [step,node] presence")
    node_types = np.asarray(node_types, dtype=object)
    if node_types.shape != (target.shape[1],):
        raise ValueError("node_types must have one entry per node")

    learned_error = np.abs(predicted - target)
    persistence_error = np.abs(persistence - target)
    result: dict[str, Any] = {
        "count": int(present.sum()),
        "mae": {},
        "persistence_mae": {},
        "delta": {},
        "mean_true_displacement": {},
        "mean_predicted_displacement": {},
        "by_node_type": {},
    }
    for coordinate, index in zip(COORDINATES, range(3)):
        mask = present
        learned_values = learned_error[..., index][mask]
        persistence_values = persistence_error[..., index][mask]
        result["mae"][coordinate] = _mean_or_none(learned_values)
        result["persistence_mae"][coordinate] = _mean_or_none(persistence_values)
        if learned_values.size and persistence_values.size:
            result["delta"][coordinate] = float(learned_values.mean() - persistence_values.mean())
        else:
            result["delta"][coordinate] = None
        true_displacement = np.abs(target[..., index] - persistence[..., index])[mask]
        predicted_displacement = np.abs(predicted[..., index] - persistence[..., index])[mask]
        result["mean_true_displacement"][coordinate] = _mean_or_none(true_displacement)
        result["mean_predicted_displacement"][coordinate] = _mean_or_none(predicted_displacement)

    for node_type in sorted({str(value) for value in node_types[present.any(axis=0)]}):
        type_mask = present & (node_types[None, :] == node_type)
        type_result: dict[str, Any] = {"count": int(type_mask.sum()), "mae": {}, "persistence_mae": {}, "delta": {}}
        for coordinate, index in zip(COORDINATES, range(3)):
            learned_values = learned_error[..., index][type_mask]
            persistence_values = persistence_error[..., index][type_mask]
            type_result["mae"][coordinate] = _mean_or_none(learned_values)
            type_result["persistence_mae"][coordinate] = _mean_or_none(persistence_values)
            type_result["delta"][coordinate] = (
                float(learned_values.mean() - persistence_values.mean())
                if learned_values.size and persistence_values.size else None
            )
        result["by_node_type"][node_type] = type_result
    return result


def _inverse_node(value: torch.Tensor, stats: Mapping[str, Any]) -> np.ndarray:
    node_stats = stats["features"]["node_state"]
    mean = torch.as_tensor(node_stats["mean"], dtype=value.dtype, device=value.device)
    scale = torch.as_tensor(node_stats["scale"], dtype=value.dtype, device=value.device)
    return (value * scale + mean)[..., :3].cpu().numpy()


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
    checkpoint = torch.load(run / "checkpoints" / "coupled_dual_gnn_residual__best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _indices(dataset: FormalAirFogSimWindowDataset, sample_ids: list[str]) -> list[int]:
    if any("locked_test" in str(sample_id).lower() for sample_id in sample_ids):
        raise ValueError("locked_test sample IDs are forbidden")
    lookup = {str(row["sample_id"]): index for index, row in enumerate(dataset.rows)}
    missing = [sample_id for sample_id in sample_ids if sample_id not in lookup]
    if missing:
        raise KeyError(f"validation sample IDs not found: {missing[:3]}")
    return [lookup[sample_id] for sample_id in sample_ids]


def diagnose_run(run: Path, tensor_root: Path) -> dict[str, Any]:
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    stats = json.loads((tensor_root / "normalization_stats.json").read_text(encoding="utf-8"))
    sample_ids = json.loads((run / "sample_ids.json").read_text(encoding="utf-8"))
    validation_ids = list(sample_ids["validation"])
    dataset = FormalAirFogSimWindowDataset(
        tensor_root,
        split="validation",
        config=FormalWindowConfig(int(config["history_steps"]), int(config["horizon_steps"])),
        stats=stats,
        normalize=True,
    )
    indices = _indices(dataset, validation_ids)
    loader = DataLoader(Subset(dataset, indices), batch_size=8, shuffle=False, num_workers=0)
    model = _load_model(run, tensor_root, stats, config)
    node_kind_names = ["vehicle", "uav", "rsu", "edge_server", "cloud"]
    per_step: list[list[dict[str, Any]]] = [[] for _ in range(int(config["horizon_steps"]))]
    with torch.no_grad():
        for batch in loader:
            prediction = model(batch)
            target = batch["target"]
            history = batch["history"]
            predicted = _inverse_node(prediction["node_state_mean"], stats)
            truth = _inverse_node(target["node_state"], stats)
            last = _inverse_node(history["node_state"][:, -1], stats)
            present = target["node_present"].bool().cpu().numpy()
            kinds = batch["static"]["node_kind_index"].cpu().numpy()
            for row in range(predicted.shape[0]):
                node_types = np.asarray(
                    [node_kind_names[int(kind)] if int(kind) >= 0 else "unknown" for kind in kinds[row]],
                    dtype=object,
                )
                for step in range(predicted.shape[1]):
                    summary = summarize_position_errors(
                        truth[row, step : step + 1],
                        predicted[row, step : step + 1],
                        np.broadcast_to(last[row], (1, last.shape[1], last.shape[2])),
                        present[row, step : step + 1],
                        node_types,
                    )
                    per_step[step].append(summary)

    def merge(
        summaries: list[dict[str, Any]],
        *,
        include_displacement: bool = True,
        include_types: bool = True,
    ) -> dict[str, Any]:
        # Reconstruct scalar means from counts so samples with more valid nodes are weighted correctly.
        merged: dict[str, Any] = {"count": sum(item["count"] for item in summaries), "mae": {}, "persistence_mae": {}, "delta": {}}
        if include_types:
            merged["by_node_type"] = {}
        scalar_keys = ["mae", "persistence_mae"]
        if include_displacement:
            merged["mean_true_displacement"] = {}
            merged["mean_predicted_displacement"] = {}
            scalar_keys.extend(["mean_true_displacement", "mean_predicted_displacement"])
        for key in scalar_keys:
            for coordinate in COORDINATES:
                values = [(item[key][coordinate], item["count"]) for item in summaries if item[key][coordinate] is not None]
                merged[key][coordinate] = float(sum(value * count for value, count in values) / sum(count for _, count in values)) if values else None
        for coordinate in COORDINATES:
            merged["delta"][coordinate] = (
                merged["mae"][coordinate] - merged["persistence_mae"][coordinate]
                if merged["mae"][coordinate] is not None and merged["persistence_mae"][coordinate] is not None else None
            )
        if include_types:
            for node_type in sorted({node_type for item in summaries for node_type in item["by_node_type"]}):
                type_items = [item["by_node_type"][node_type] for item in summaries if node_type in item["by_node_type"]]
                merged["by_node_type"][node_type] = merge(type_items, include_displacement=False, include_types=False)
        return merged

    return {"seed": int(config["seed"]), "sample_count": len(validation_ids), "per_step": {str(step + 1): merge(rows) for step, rows in enumerate(per_step)}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--tensor-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if "locked_test" in str(args.tensor_root).lower() or "locked_test" in str(args.runs_root).lower():
        raise SystemExit("locked_test paths are forbidden")
    runs = [args.runs_root / name for name in RUN_NAMES]
    report = {
        "schema_version": "PI-JWM-formal-p4-position-diagnosis-v1",
        "device": "cpu",
        "scope": "validation sample IDs from each existing GPU run; present target nodes only",
        "runs": [diagnose_run(run, args.tensor_root) for run in runs],
        "gpu_started": False,
        "locked_test_accessed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
