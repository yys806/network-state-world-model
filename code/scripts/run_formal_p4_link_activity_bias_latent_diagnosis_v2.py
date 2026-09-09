"""CPU-only bias versus recursive-latent diagnosis for P4 link activity."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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


ALLOWED_SPLITS = ("validation", "calibration")
FROZEN_PROBABILITY_THRESHOLD = 0.9
QUANTILES = {"q50": 0.50, "q90": 0.90, "q95": 0.95, "q99": 0.99, "q100": 1.0}


def probability_to_logit(probability: float) -> float:
    """Convert an open-interval probability to its raw-logit threshold."""
    if not 0.0 < probability < 1.0:
        raise ValueError("probability must be strictly between zero and one")
    return math.log(probability / (1.0 - probability))


def validate_split(split: str) -> str:
    if split not in ALLOWED_SPLITS:
        raise ValueError(f"split must be one of {ALLOWED_SPLITS}, got {split!r}")
    return split


def _distribution(values: np.ndarray) -> dict[str, float | None]:
    if not len(values):
        return {"mean": None, **{name: None for name in QUANTILES}}
    return {
        "mean": float(np.mean(values)),
        **{name: float(np.quantile(values, quantile)) for name, quantile in QUANTILES.items()},
    }


def summarize_negative_logits(
    raw_logits: np.ndarray,
    target_activity: np.ndarray,
    valid_edge_mask: np.ndarray,
    *,
    bias: float,
    probability_threshold: float = FROZEN_PROBABILITY_THRESHOLD,
) -> dict[str, Any]:
    """Summarize valid ground-truth-negative logits without changing the threshold."""
    raw = np.asarray(raw_logits, dtype=np.float64)
    target = np.asarray(target_activity, dtype=bool)
    valid = np.asarray(valid_edge_mask, dtype=bool)
    if raw.shape != target.shape or raw.shape != valid.shape:
        raise ValueError("raw_logits, target_activity, and valid_edge_mask must have the same shape")

    negatives = valid & ~target
    negative_raw = raw[negatives]
    pre_bias = negative_raw - float(bias)
    invariant_error_abs = (
        float(np.max(np.abs((negative_raw - pre_bias) - float(bias)))) if len(negative_raw) else 0.0
    )
    logit_threshold = probability_to_logit(probability_threshold)
    raw_fp = negative_raw >= logit_threshold
    zero_bias_fp = pre_bias >= logit_threshold
    negative_count = int(len(negative_raw))
    raw_count = int(np.count_nonzero(raw_fp))
    zero_bias_count = int(np.count_nonzero(zero_bias_fp))

    return {
        "negative_count": negative_count,
        "raw_logit": _distribution(negative_raw),
        "pre_bias_logit": _distribution(pre_bias),
        "frozen_probability_threshold": float(probability_threshold),
        "frozen_raw_logit_threshold": logit_threshold,
        "raw_false_positive_at_frozen_threshold": {
            "count": raw_count,
            "rate": float(raw_count / negative_count) if negative_count else None,
        },
        "zero_bias_counterfactual_at_frozen_threshold": {
            "count": zero_bias_count,
            "rate": float(zero_bias_count / negative_count) if negative_count else None,
            "eliminated_count": raw_count - zero_bias_count,
        },
        "invariant_error_abs": invariant_error_abs,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _best_checkpoint(run_dir: Path) -> Path:
    matches = sorted((run_dir / "checkpoints").glob("*best*.pt"))
    if len(matches) != 1:
        raise ValueError(f"expected exactly one best checkpoint under {run_dir / 'checkpoints'}, found {matches}")
    return matches[0]


def _model_from_run(config: dict[str, Any], stats: dict[str, Any], tensor_root: Path) -> FormalDualGraphWorldModel:
    tensor_contract = json.loads((tensor_root / "tensor_contract.json").read_text(encoding="utf-8"))
    return FormalDualGraphWorldModel(
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
            n_rb=int(tensor_contract.get("n_rb", 1)),
        )
    )


def _collect_horizons(
    model: torch.nn.Module,
    dataset: FormalAirFogSimWindowDataset,
    sample_ids: list[str],
    *,
    batch_size: int,
    bias: float,
) -> list[dict[str, Any]]:
    row_index = {str(row["sample_id"]): index for index, row in enumerate(dataset.rows)}
    missing = [sample_id for sample_id in sample_ids if sample_id not in row_index]
    if missing:
        raise ValueError(f"sample IDs missing from {dataset.split}: {missing[:3]}")
    loader = DataLoader(Subset(dataset, [row_index[sample_id] for sample_id in sample_ids]), batch_size=batch_size, shuffle=False, num_workers=0)
    raw_by_horizon: list[list[np.ndarray]] | None = None
    target_by_horizon: list[list[np.ndarray]] | None = None
    valid_by_horizon: list[list[np.ndarray]] | None = None
    model.eval()
    with torch.no_grad():
        for batch in loader:
            prediction = model(batch)
            raw = prediction["link_activity_logits"].detach().cpu().numpy()
            target = batch["target"]
            activity = target.get("aggregate_link_activity", target["link_activity"]).bool()
            activity_mask = target.get("aggregate_link_activity_mask")
            if activity_mask is None:
                activity_mask = torch.ones_like(activity, dtype=torch.bool)
            endpoint_index = batch["static"]["physical_edge_endpoint_index"]
            edge_valid = torch.all(endpoint_index >= 0, dim=-1)
            valid = activity_mask.bool() & edge_valid[:, None, :].expand_as(activity)
            target_np = activity.cpu().numpy()
            valid_np = valid.cpu().numpy()
            if raw_by_horizon is None:
                horizon_steps = raw.shape[1]
                raw_by_horizon = [[] for _ in range(horizon_steps)]
                target_by_horizon = [[] for _ in range(horizon_steps)]
                valid_by_horizon = [[] for _ in range(horizon_steps)]
            if raw.shape[1] != len(raw_by_horizon):
                raise ValueError("inconsistent horizon dimension across batches")
            for horizon_index in range(raw.shape[1]):
                raw_by_horizon[horizon_index].append(raw[:, horizon_index].reshape(-1))
                target_by_horizon[horizon_index].append(target_np[:, horizon_index].reshape(-1))
                valid_by_horizon[horizon_index].append(valid_np[:, horizon_index].reshape(-1))
    if raw_by_horizon is None:
        raise ValueError("no samples were collected")
    return [
        {
            "horizon": horizon_index + 1,
            **summarize_negative_logits(
                np.concatenate(raw_chunks),
                np.concatenate(target_by_horizon[horizon_index]),
                np.concatenate(valid_by_horizon[horizon_index]),
                bias=bias,
            ),
        }
        for horizon_index, raw_chunks in enumerate(raw_by_horizon)
    ]


def diagnose_run(run_dir: Path, *, batch_size: int) -> dict[str, Any]:
    config_path = run_dir / "config.json"
    sample_ids_path = run_dir / "sample_ids.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    sample_ids = json.loads(sample_ids_path.read_text(encoding="utf-8"))
    tensor_root = Path(config["tensor_root"])
    tensor_manifest_path = tensor_root / "manifest.json"
    if not tensor_manifest_path.is_file():
        tensor_manifest_path = tensor_root / "tensor_contract.json"
    stats = json.loads((tensor_root / "normalization_stats.json").read_text(encoding="utf-8"))
    checkpoint_path = _best_checkpoint(run_dir)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = _model_from_run(config, stats, tensor_root)
    model.load_state_dict(checkpoint["model_state_dict"])
    bias = float(model.link_activity_head.bias.detach().cpu().item())
    weight_l2 = float(torch.linalg.vector_norm(model.link_activity_head.weight.detach().cpu()).item())
    contract = FormalWindowConfig(history_steps=int(config["history_steps"]), horizon_steps=int(config["horizon_steps"]))
    if contract.horizon_steps != 20:
        raise ValueError(f"P4 v2 requires 20 horizons, got {contract.horizon_steps}")

    splits: dict[str, Any] = {}
    for split in ALLOWED_SPLITS:
        validate_split(split)
        if split not in sample_ids:
            raise ValueError(f"sample_ids.json does not contain required {split!r} IDs")
        dataset = FormalAirFogSimWindowDataset(tensor_root, split=split, config=contract, stats=stats, normalize=True)
        horizons = _collect_horizons(model, dataset, sample_ids[split], batch_size=batch_size, bias=bias)
        if len(horizons) != 20:
            raise ValueError(f"expected 20 horizons for {split}, got {len(horizons)}")
        splits[split] = {
            "sample_count": len(sample_ids[split]),
            "horizons": horizons,
            "highlight_horizons": {str(horizon): horizons[horizon - 1] for horizon in (1, 5, 20)},
        }

    invariant_error_abs = max(
        horizon["invariant_error_abs"]
        for split in splits.values()
        for horizon in split["horizons"]
    )
    if invariant_error_abs > 1e-6:
        raise AssertionError(f"raw - pre_bias must equal bias within 1e-6, got {invariant_error_abs}")
    return {
        "run": run_dir.name,
        "seed": int(config["seed"]),
        "provenance": {
            "config_sha256": _sha256(config_path),
            "sample_ids_sha256": _sha256(sample_ids_path),
            "best_checkpoint_sha256": _sha256(checkpoint_path),
            "tensor_manifest_sha256": _sha256(tensor_manifest_path),
            "tensor_root": str(tensor_root),
        },
        "link_activity_head": {"bias": bias, "weight_l2": weight_l2},
        "invariant_error_abs_max": invariant_error_abs,
        "splits": splits,
    }


def diagnose(run_dirs: list[Path], output_path: Path, *, batch_size: int = 16) -> dict[str, Any]:
    if batch_size <= 1:
        raise ValueError("batch_size must be greater than one for the v2 diagnostic")
    if not run_dirs:
        raise ValueError("at least one --run-dir is required")
    report = {
        "schema_version": "PI-JWM-formal-p4-link-bias-latent-diagnosis-v2",
        "execution_policy": {
            "device": "cpu",
            "training_started": False,
            "gpu_started": False,
            "locked_test_accessed": False,
        },
        "frozen_probability_threshold": FROZEN_PROBABILITY_THRESHOLD,
        "runs": [diagnose_run(Path(run_dir), batch_size=batch_size) for run_dir in run_dirs],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    print(json.dumps(diagnose(args.run_dir, args.output, batch_size=args.batch_size), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
