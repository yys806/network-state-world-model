"""Diagnose per-dimension value and coupling errors for a V11 policy checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from pi_jwm.v6_data import load_world_model_arrays, make_normalization_stats
from pi_jwm.v7_action_policy import V7ActionPolicy
from run_v11_discrete_value_policy import (
    V11DiscreteValuePolicyDataset,
    build_value_vocab,
    collate_discrete_policy_batch,
    collect_predictions,
)
from run_v7_action_policy import resolve_policy_seed_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def analyze(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    arrays = load_world_model_arrays(args.dataset_dir)
    train_idx, val_idx, test_idx, split = resolve_policy_seed_splits(
        arrays["sample_seed"],
        train_seeds=[*range(16), *range(20, 60)],
        val_seeds=[16, 17],
        test_seeds=[18, 19],
    )
    stats = make_normalization_stats(arrays, train_idx)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = type("Config", (), checkpoint["config"])()
    model = V7ActionPolicy(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    vocab = checkpoint.get("value_vocab") or build_value_vocab(arrays["edge_a_future"][train_idx])
    test_ds = V11DiscreteValuePolicyDataset(arrays, test_idx, stats, vocab)
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_discrete_policy_batch)
    pred = collect_predictions(model, loader, device, vocab)
    features = [str(x) for x in arrays["edge_action_features"].tolist()]
    active = pred["active"] > 0.5
    rows = []
    for dim, name in enumerate(features):
        mask = active[..., dim]
        true = pred["value_true"][..., dim][mask]
        value = pred["value_pred"][..., dim][mask]
        bin_true = pred["bin_true"][..., dim][mask]
        bin_pred = pred["bin_pred"][..., dim][mask]
        rows.append(
            {
                "dim": dim,
                "name": name,
                "positive_count": int(mask.sum()),
                "bin_accuracy": float(np.mean(bin_true == bin_pred)) if true.size else float("nan"),
                "rmse": float(np.sqrt(np.mean((value - true) ** 2))) if true.size else float("nan"),
                "mae": float(np.mean(np.abs(value - true))) if true.size else float("nan"),
                "true_mean": float(np.mean(true)) if true.size else float("nan"),
                "pred_mean": float(np.mean(value)) if true.size else float("nan"),
                "true_histogram": histogram(true),
                "pred_histogram": histogram(value),
            }
        )
    raw_pred = pred["value_pred"]
    coupling = {
        "true_offload_rb_task_equal": equality_rate(pred["value_true"][..., 0], pred["value_true"][..., 1]),
        "pred_offload_rb_task_equal": equality_rate(raw_pred[..., 0], raw_pred[..., 1]),
        "true_cpu_count_value_coactive": coactive_rate(pred["value_true"][..., 3], pred["value_true"][..., 4]),
        "pred_cpu_count_value_coactive": coactive_rate(raw_pred[..., 3], raw_pred[..., 4]),
    }
    result = {"split": split, "per_dim": rows, "coupling": coupling}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def histogram(values: np.ndarray) -> list[dict[str, float | int]]:
    unique, counts = np.unique(np.round(values.astype(np.float32), 6), return_counts=True)
    order = np.argsort(counts)[::-1]
    return [{"value": float(unique[i]), "count": int(counts[i])} for i in order]


def equality_rate(left: np.ndarray, right: np.ndarray) -> float:
    relevant = (left > 1e-9) | (right > 1e-9)
    return float(np.mean(np.isclose(left[relevant], right[relevant]))) if relevant.any() else float("nan")


def coactive_rate(left: np.ndarray, right: np.ndarray) -> float:
    relevant = (left > 1e-9) | (right > 1e-9)
    return float(np.mean((left[relevant] > 1e-9) == (right[relevant] > 1e-9))) if relevant.any() else float("nan")


def main() -> None:
    args = parse_args()
    result = analyze(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
