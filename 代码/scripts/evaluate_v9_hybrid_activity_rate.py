"""Post-hoc hybrid evaluator for PI-JWM v9 activity/rate separation.

This script combines an activity-gate model with a positive-rate model:
activity probability is taken from one checkpoint, positive-rate magnitude
from another, and final link-rate is prob * positive_rate. It is a diagnostic
for branch-specific routing before implementing a new single-model architecture.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pi_jwm.paths import ARTIFACTS_DIR
from pi_jwm.v6_data import (
    V6WorldModelDataset,
    collate_v6_world_model_batch,
    load_world_model_arrays,
    make_normalization_stats,
)
from pi_jwm.v6_metrics import active_rate_metrics, activity_metrics, regression_metrics
from pi_jwm.v8_training import (
    build_v8_model_from_arrays,
    collect_v8_predictions,
    denormalize_v8_link_rate_prediction,
)


DEFAULT_DATASET_DIR = (
    ARTIFACTS_DIR
    / "experiments"
    / "airfogsim_v0"
    / "datasets"
    / "world_model_dataset_active_heavy_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate hybrid PI-JWM v9 activity/rate checkpoints.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--activity-exp-dir", type=Path, required=True)
    parser.add_argument("--rate-exp-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="cpu")
    parser.add_argument("--train-seeds", default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15")
    parser.add_argument("--val-seeds", default="16,17")
    parser.add_argument("--test-seeds", default="18,19")
    return parser.parse_args()


def parse_seed_list(value: str) -> list[int]:
    return [int(part) for part in value.replace(",", " ").split() if part]


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


def resolve_by_seeds(sample_seed: np.ndarray, seeds: list[int]) -> np.ndarray:
    return np.where(np.isin(sample_seed, np.asarray(seeds, dtype=sample_seed.dtype)))[0]


def maybe_add_event_memory(arrays: dict[str, np.ndarray], exp_dir: Path) -> dict[str, np.ndarray]:
    summary = json.loads((exp_dir / "v8_full_training_summary.json").read_text(encoding="utf-8"))
    if summary.get("config", {}).get("use_event_memory_features"):
        from run_world_model_v8_full_training import add_event_memory_features

        return add_event_memory_features(arrays)
    return arrays


def load_model(exp_dir: Path, arrays: dict[str, np.ndarray], device: torch.device):
    summary_path = exp_dir / "v8_full_training_summary.json"
    checkpoint_path = exp_dir / "checkpoints" / "v8_dual_best.pt"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    config = summary["config"]
    model = build_v8_model_from_arrays(
        arrays,
        hidden_dim=int(config["hidden_dim"]),
        graph_mode=config["graph_mode"],
        fusion_mode=config["fusion_mode"],
        fusion_num_heads=int(config["fusion_num_heads"]),
        active_rate_auxiliary=bool(config["active_rate_auxiliary"]),
        active_rate_head_mode=config.get("active_rate_head_mode", "mlp"),
        num_rate_experts=int(config.get("num_rate_experts", 4)),
        rate_output_mode=config.get("model_rate_output_mode", "direct"),
        history_encoder=config.get("history_encoder", "mean"),
        latent_transition_mode=config.get("latent_transition_mode", "message_passing"),
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, summary


def collect_for_exp(
    exp_dir: Path,
    base_arrays: dict[str, np.ndarray],
    sample_indices: np.ndarray,
    device: torch.device,
    batch_size: int,
):
    arrays = maybe_add_event_memory(base_arrays, exp_dir)
    stats = make_normalization_stats(arrays, resolve_by_seeds(arrays["sample_seed"], parse_seed_list("0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15")))
    dataset = V6WorldModelDataset(arrays, sample_indices, stats)
    loader = DataLoader(Subset(dataset, range(len(dataset))), batch_size=batch_size, shuffle=False, collate_fn=collate_v6_world_model_batch)
    model, summary = load_model(exp_dir, arrays, device)
    predictions = collect_v8_predictions(
        model,
        loader,
        device,
        stats,
        rate_output_mode=summary["config"].get("rate_output_mode", "main"),
        inactive_rate_value=float(summary["config"].get("inactive_rate_value", 0.0)),
    )
    return predictions, summary, stats


def squeeze(values: np.ndarray) -> np.ndarray:
    return np.squeeze(values, axis=-1) if values.ndim >= 1 and values.shape[-1] == 1 else values


def evaluate_hybrid(activity_pred: dict[str, np.ndarray], rate_pred: dict[str, np.ndarray], threshold: float | None) -> dict:
    prob = activity_pred["link_activity_prob"]
    true_activity = activity_pred["link_activity_true"]
    true_rate = activity_pred["link_rate_true"]
    positive_rate = rate_pred.get("link_positive_rate_pred", rate_pred["link_rate_pred"])
    hybrid_rate = prob * positive_rate
    if threshold is None:
        from pi_jwm.v8_training import choose_activity_threshold

        threshold = choose_activity_threshold(prob, true_activity)
    return {
        "threshold": float(threshold),
        "activity": activity_metrics(prob, true_activity, threshold=float(threshold)),
        "link_rate": regression_metrics(hybrid_rate, true_rate),
        "active_rate": active_rate_metrics(hybrid_rate, true_rate, true_activity),
        "positive_rate_active": active_rate_metrics(positive_rate, true_rate, true_activity),
    }


def run(args: argparse.Namespace) -> dict:
    device = choose_device(args.device)
    base_arrays = load_world_model_arrays(args.dataset_dir)
    test_idx = resolve_by_seeds(base_arrays["sample_seed"], parse_seed_list(args.test_seeds))
    val_idx = resolve_by_seeds(base_arrays["sample_seed"], parse_seed_list(args.val_seeds))

    val_activity, activity_summary, _ = collect_for_exp(args.activity_exp_dir, base_arrays, val_idx, device, args.batch_size)
    val_rate, rate_summary, _ = collect_for_exp(args.rate_exp_dir, base_arrays, val_idx, device, args.batch_size)
    val_eval = evaluate_hybrid(val_activity, val_rate, threshold=None)

    test_activity, _, _ = collect_for_exp(args.activity_exp_dir, base_arrays, test_idx, device, args.batch_size)
    test_rate, _, _ = collect_for_exp(args.rate_exp_dir, base_arrays, test_idx, device, args.batch_size)
    test_eval = evaluate_hybrid(test_activity, test_rate, threshold=val_eval["threshold"])

    return {
        "activity_exp_dir": str(args.activity_exp_dir),
        "rate_exp_dir": str(args.rate_exp_dir),
        "activity_config": activity_summary["config"],
        "rate_config": rate_summary["config"],
        "val_eval": val_eval,
        "test_eval": test_eval,
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "hybrid_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = [
        {
            "split": "test",
            "threshold": summary["test_eval"]["threshold"],
            "precision": summary["test_eval"]["activity"]["precision"],
            "recall": summary["test_eval"]["activity"]["recall"],
            "f1": summary["test_eval"]["activity"]["f1"],
            "active_rate_rmse": summary["test_eval"]["active_rate"]["active_rmse"],
            "positive_rate_active_rmse": summary["test_eval"]["positive_rate_active"]["active_rmse"],
            "link_rate_rmse": summary["test_eval"]["link_rate"]["rmse"],
        }
    ]
    with (output_dir / "hybrid_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    row = rows[0]
    (output_dir / "hybrid_metrics.md").write_text(
        "\n".join(
            [
                "# PI-JWM v9 Hybrid Activity/Rate Evaluation",
                "",
                "| Precision | Recall | F1 | Active-rate RMSE | Positive-rate active RMSE | Link-rate RMSE |",
                "|---:|---:|---:|---:|---:|---:|",
                f"| {row['precision']:.6f} | {row['recall']:.6f} | {row['f1']:.6f} | {row['active_rate_rmse']:.6f} | {row['positive_rate_active_rmse']:.6f} | {row['link_rate_rmse']:.6f} |",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    summary = run(args)
    write_outputs(summary, args.output_dir)
    print(json.dumps(summary["test_eval"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
