"""Post-hoc hurdle gate calibration for PI-JWM v9 checkpoints.

This is a diagnostic evaluator: it does not retrain the model. It changes the
inference-time activity gate used in hurdle-style rate prediction:

    calibrated_rate = calibrated_activity_prob * positive_rate

The goal is to determine whether active-rate errors come from an overly
suppressed activity gate or from the positive-rate amplitude branch itself.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from pi_jwm.paths import ARTIFACTS_DIR
from pi_jwm.v6_data import (
    V6WorldModelDataset,
    collate_v6_world_model_batch,
    load_world_model_arrays,
    make_normalization_stats,
)
from pi_jwm.v6_metrics import active_rate_metrics, activity_metrics, regression_metrics
from pi_jwm.v8_training import build_v8_model_from_arrays, collect_v8_predictions, choose_activity_threshold


DEFAULT_DATASET_DIR = (
    ARTIFACTS_DIR
    / "experiments"
    / "airfogsim_v0"
    / "datasets"
    / "world_model_dataset_active_heavy_v1"
)
DEFAULT_OUTPUT_DIR = ARTIFACTS_DIR / "experiments" / "pi_jwm_v9_hurdle_gate_calibration_20260617"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate post-hoc PI-JWM v9 hurdle gate calibration.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--experiment", action="append", required=True)
    parser.add_argument(
        "--checkpoint-glob",
        default="v8_dual_best.pt",
        help="Checkpoint filename or glob under each experiment's checkpoints directory.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="cpu")
    parser.add_argument("--train-seeds", default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15")
    parser.add_argument("--val-seeds", default="16,17")
    parser.add_argument("--test-seeds", default="18,19")
    parser.add_argument("--temperatures", default="0.5,0.75,1.0,1.5,2.0")
    parser.add_argument("--powers", default="0.0,0.25,0.5,0.75,1.0")
    parser.add_argument(
        "--thresholds",
        default="",
        help="Optional fixed activity thresholds. When set, avoids per-grid threshold search.",
    )
    parser.add_argument("--positive-rate-scales", default="1.0")
    parser.add_argument("--rate-gate-modes", default="soft")
    parser.add_argument("--selective-min-activity-probs", default="")
    parser.add_argument("--selective-min-positive-rates", default="")
    parser.add_argument(
        "--selection-metric",
        choices=("active_rate_rmse", "composite", "constrained_active_rate", "constrained_composite"),
        default="active_rate_rmse",
    )
    parser.add_argument("--min-f1", type=float, default=0.0)
    parser.add_argument("--max-link-rmse", type=float, default=0.0)
    parser.add_argument("--f1-penalty-weight", type=float, default=1000.0)
    parser.add_argument("--link-penalty-weight", type=float, default=10.0)
    return parser.parse_args()


def parse_float_list(value: str) -> list[float]:
    return [float(part) for part in value.replace(",", " ").split() if part]


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


def resolve_experiment_path(name_or_path: str) -> Path:
    path = Path(name_or_path)
    if path.exists():
        return path
    artifact_path = ARTIFACTS_DIR / "experiments" / name_or_path
    if artifact_path.exists():
        return artifact_path
    raise FileNotFoundError(f"Experiment directory not found: {name_or_path}")


def calibrate_activity_gate(prob: np.ndarray, temperature: float = 1.0, power: float = 1.0) -> np.ndarray:
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if power < 0.0:
        raise ValueError("power must be non-negative")
    prob = np.asarray(prob, dtype=np.float64)
    clipped = np.clip(prob, 1e-6, 1.0 - 1e-6)
    if power == 0.0:
        return np.ones_like(clipped, dtype=np.float64)
    logits = np.log(clipped / (1.0 - clipped))
    calibrated = 1.0 / (1.0 + np.exp(-(logits / float(temperature))))
    return np.clip(calibrated**float(power), 0.0, 1.0)


def calibrate_selective_activity_gate(
    prob: np.ndarray,
    positive_rate: np.ndarray,
    temperature: float = 1.0,
    power: float = 1.0,
    min_activity_prob: float = 0.0,
    min_positive_rate: float = 0.0,
) -> np.ndarray:
    base = np.asarray(prob, dtype=np.float64)
    softened = calibrate_activity_gate(base, temperature=temperature, power=power)
    selected = (base >= float(min_activity_prob)) & (np.asarray(positive_rate, dtype=np.float64) >= float(min_positive_rate))
    return np.where(selected, softened, base)


def evaluate_gate_calibrated_rate(
    activity_prob: np.ndarray,
    positive_rate: np.ndarray,
    true_rate: np.ndarray,
    true_activity: np.ndarray,
    temperature: float,
    power: float,
    threshold: float | None = None,
    min_activity_prob: float = 0.0,
    min_positive_rate: float = 0.0,
    positive_rate_scale: float = 1.0,
    rate_gate_mode: str = "soft",
) -> dict:
    if positive_rate_scale <= 0.0:
        raise ValueError("positive_rate_scale must be positive")
    if rate_gate_mode not in {"soft", "hard"}:
        raise ValueError("rate_gate_mode must be one of: soft, hard")
    if min_activity_prob > 0.0 or min_positive_rate > 0.0:
        calibrated_prob = calibrate_selective_activity_gate(
            activity_prob,
            positive_rate,
            temperature=temperature,
            power=power,
            min_activity_prob=min_activity_prob,
            min_positive_rate=min_positive_rate,
        )
    else:
        calibrated_prob = calibrate_activity_gate(activity_prob, temperature=temperature, power=power)
    scaled_positive_rate = np.asarray(positive_rate, dtype=np.float64) * float(positive_rate_scale)
    if threshold is None:
        threshold = choose_activity_threshold(calibrated_prob, true_activity)
    if rate_gate_mode == "hard":
        rate_gate = (calibrated_prob >= float(threshold)).astype(np.float64)
    else:
        rate_gate = calibrated_prob
    calibrated_rate = rate_gate * scaled_positive_rate
    return {
        "temperature": float(temperature),
        "power": float(power),
        "min_activity_prob": float(min_activity_prob),
        "min_positive_rate": float(min_positive_rate),
        "positive_rate_scale": float(positive_rate_scale),
        "rate_gate_mode": rate_gate_mode,
        "threshold": float(threshold),
        "activity": activity_metrics(calibrated_prob, true_activity, threshold=float(threshold)),
        "link_rate": regression_metrics(calibrated_rate, true_rate),
        "active_rate": active_rate_metrics(calibrated_rate, true_rate, true_activity),
        "positive_rate_active": active_rate_metrics(scaled_positive_rate, true_rate, true_activity),
    }


def calibration_score(metrics: dict, selection_metric: str) -> float:
    active = float(metrics["active_rate"]["active_rmse"])
    if selection_metric == "active_rate_rmse":
        return active
    if selection_metric == "constrained_active_rate":
        return constrained_calibration_score(metrics)
    link = float(metrics["link_rate"]["rmse"])
    f1 = float(metrics["activity"]["f1"])
    composite = active + 0.2 * link - 10.0 * f1
    if selection_metric == "constrained_composite":
        return constrained_calibration_score(metrics, base_score=composite)
    return composite


def constrained_calibration_score(
    metrics: dict,
    min_f1: float = 0.0,
    max_link_rmse: float = 0.0,
    f1_penalty_weight: float = 1000.0,
    link_penalty_weight: float = 10.0,
    base_score: float | None = None,
) -> float:
    active = float(metrics["active_rate"]["active_rmse"])
    link = float(metrics["link_rate"]["rmse"])
    f1 = float(metrics["activity"]["f1"])
    score = active if base_score is None else float(base_score)
    f1_shortfall = max(0.0, float(min_f1) - f1)
    link_excess = max(0.0, link - float(max_link_rmse)) if max_link_rmse > 0.0 else 0.0
    return score + float(f1_penalty_weight) * f1_shortfall + float(link_penalty_weight) * link_excess


def select_best_row(
    rows: list[dict],
    selection_metric: str,
    min_f1: float = 0.0,
    max_link_rmse: float = 0.0,
    f1_penalty_weight: float = 1000.0,
    link_penalty_weight: float = 10.0,
) -> dict:
    if not rows:
        raise ValueError("rows must not be empty")

    def row_score(row: dict) -> float:
        metrics = {
            "active_rate": {"active_rmse": row["active_rate_rmse"]},
            "link_rate": {"rmse": row["link_rate_rmse"]},
            "activity": {"f1": row["f1"]},
        }
        if selection_metric == "active_rate_rmse":
            return float(row["active_rate_rmse"])
        if selection_metric == "composite":
            return float(row["score"])
        if selection_metric == "constrained_active_rate":
            return constrained_calibration_score(
                metrics,
                min_f1=min_f1,
                max_link_rmse=max_link_rmse,
                f1_penalty_weight=f1_penalty_weight,
                link_penalty_weight=link_penalty_weight,
            )
        if selection_metric == "constrained_composite":
            return constrained_calibration_score(
                metrics,
                min_f1=min_f1,
                max_link_rmse=max_link_rmse,
                f1_penalty_weight=f1_penalty_weight,
                link_penalty_weight=link_penalty_weight,
                base_score=float(row["score"]),
            )
        raise ValueError("unknown selection_metric")

    best = min(rows, key=row_score)
    best = dict(best)
    best["score"] = float(row_score(best))
    best["meets_constraints"] = bool(
        float(best["f1"]) >= float(min_f1)
        and (max_link_rmse <= 0.0 or float(best["link_rate_rmse"]) <= float(max_link_rmse))
    )
    return best


def maybe_add_event_memory(arrays: dict[str, np.ndarray], summary: dict) -> dict[str, np.ndarray]:
    if summary.get("config", {}).get("use_event_memory_features"):
        from run_world_model_v8_full_training import add_event_memory_features

        return add_event_memory_features(arrays)
    return arrays


def load_summary(exp_dir: Path) -> dict:
    summary_path = exp_dir / "v8_full_training_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary: {summary_path}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def resolve_checkpoint_paths(exp_dir: Path, checkpoint_glob: str = "v8_dual_best.pt") -> list[Path]:
    checkpoint_dir = exp_dir / "checkpoints"
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Missing checkpoint directory: {checkpoint_dir}")
    pattern = checkpoint_glob or "v8_dual_best.pt"
    pattern_path = Path(pattern)
    if pattern_path.is_absolute():
        matches = [pattern_path] if pattern_path.exists() else []
    elif any(char in pattern for char in "*?[]"):
        matches = list(checkpoint_dir.glob(pattern))
    else:
        candidate = checkpoint_dir / pattern
        matches = [candidate] if candidate.exists() else []
    if not matches:
        raise FileNotFoundError(f"No checkpoints matching {pattern!r} under {checkpoint_dir}")
    return sorted(matches, key=lambda path: (path.name != "v8_dual_best.pt", path.name))


def load_model(
    exp_dir: Path,
    arrays: dict[str, np.ndarray],
    summary: dict,
    device: torch.device,
    checkpoint_path: Path | None = None,
):
    checkpoint_path = checkpoint_path or resolve_checkpoint_paths(exp_dir)[0]
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
        activity_memory_dim=int(config.get("activity_memory_dim", 0)),
        activity_memory_routing="activity_only" if int(config.get("activity_memory_dim", 0)) > 0 else "none",
        adaptive_edge_context=config.get("adaptive_edge_context", "none"),
        adaptive_edge_topk=int(config.get("adaptive_edge_topk", 8)),
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def collect_for_experiment(
    exp_dir: Path,
    checkpoint_path: Path,
    base_arrays: dict[str, np.ndarray],
    sample_indices: np.ndarray,
    train_indices: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, np.ndarray], dict]:
    summary = load_summary(exp_dir)
    arrays = maybe_add_event_memory(base_arrays, summary)
    stats = make_normalization_stats(arrays, train_indices)
    dataset = V6WorldModelDataset(arrays, sample_indices, stats)
    loader = DataLoader(
        Subset(dataset, range(len(dataset))),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_v6_world_model_batch,
    )
    model = load_model(exp_dir, arrays, summary, device, checkpoint_path=checkpoint_path)
    predictions = collect_v8_predictions(
        model,
        loader,
        device,
        stats,
        rate_output_mode=summary["config"].get("rate_output_mode", "main"),
        inactive_rate_value=float(summary["config"].get("inactive_rate_value", 0.0)),
    )
    if "link_positive_rate_pred" not in predictions:
        raise ValueError(f"Experiment is not a hurdle-positive-rate model: {exp_dir}")
    return predictions, summary


def select_best_calibration(
    val_predictions: dict[str, np.ndarray],
    temperatures: list[float],
    powers: list[float],
    selection_metric: str,
    min_activity_probs: list[float] | None = None,
    min_positive_rates: list[float] | None = None,
    positive_rate_scales: list[float] | None = None,
    rate_gate_modes: list[str] | None = None,
    thresholds: list[float] | None = None,
    min_f1: float = 0.0,
    max_link_rmse: float = 0.0,
    f1_penalty_weight: float = 1000.0,
    link_penalty_weight: float = 10.0,
) -> tuple[dict, list[dict]]:
    rows = []
    min_activity_probs = min_activity_probs or [0.0]
    min_positive_rates = min_positive_rates or [0.0]
    positive_rate_scales = positive_rate_scales or [1.0]
    rate_gate_modes = rate_gate_modes or ["soft"]
    thresholds = thresholds or [None]
    for temperature in temperatures:
        for power in powers:
            for min_activity_prob in min_activity_probs:
                for min_positive_rate in min_positive_rates:
                    for positive_rate_scale in positive_rate_scales:
                        for rate_gate_mode in rate_gate_modes:
                            for threshold in thresholds:
                                metrics = evaluate_gate_calibrated_rate(
                                    val_predictions["link_activity_prob"],
                                    val_predictions["link_positive_rate_pred"],
                                    val_predictions["link_rate_true"],
                                    val_predictions["link_activity_true"],
                                    temperature=temperature,
                                    power=power,
                                    threshold=threshold,
                                    min_activity_prob=min_activity_prob,
                                    min_positive_rate=min_positive_rate,
                                    positive_rate_scale=positive_rate_scale,
                                    rate_gate_mode=rate_gate_mode,
                                )
                                score = calibration_score(metrics, selection_metric)
                                row = flatten_metrics(metrics, split="val", score=score)
                                rows.append(row)
    best = select_best_row(
        rows,
        selection_metric=selection_metric,
        min_f1=min_f1,
        max_link_rmse=max_link_rmse,
        f1_penalty_weight=f1_penalty_weight,
        link_penalty_weight=link_penalty_weight,
    )
    return best, rows


def flatten_metrics(metrics: dict, split: str, score: float | None = None) -> dict:
    row = {
        "split": split,
        "temperature": metrics["temperature"],
        "power": metrics["power"],
        "min_activity_prob": metrics.get("min_activity_prob", 0.0),
        "min_positive_rate": metrics.get("min_positive_rate", 0.0),
        "positive_rate_scale": metrics.get("positive_rate_scale", 1.0),
        "rate_gate_mode": metrics.get("rate_gate_mode", "soft"),
        "threshold": metrics["threshold"],
        "precision": metrics["activity"]["precision"],
        "recall": metrics["activity"]["recall"],
        "f1": metrics["activity"]["f1"],
        "active_rate_rmse": metrics["active_rate"]["active_rmse"],
        "positive_rate_active_rmse": metrics["positive_rate_active"]["active_rmse"],
        "link_rate_rmse": metrics["link_rate"]["rmse"],
    }
    if score is not None:
        row["score"] = float(score)
    return row


def run_experiment(
    args: argparse.Namespace,
    exp_dir: Path,
    checkpoint_path: Path,
    base_arrays: dict[str, np.ndarray],
    device: torch.device,
) -> dict:
    train_idx = resolve_by_seeds(base_arrays["sample_seed"], parse_seed_list(args.train_seeds))
    val_idx = resolve_by_seeds(base_arrays["sample_seed"], parse_seed_list(args.val_seeds))
    test_idx = resolve_by_seeds(base_arrays["sample_seed"], parse_seed_list(args.test_seeds))
    temperatures = parse_float_list(args.temperatures)
    powers = parse_float_list(args.powers)
    positive_rate_scales = parse_float_list(args.positive_rate_scales)
    rate_gate_modes = [part for part in args.rate_gate_modes.replace(",", " ").split() if part]
    min_activity_probs = parse_float_list(args.selective_min_activity_probs) or [0.0]
    min_positive_rates = parse_float_list(args.selective_min_positive_rates) or [0.0]
    thresholds = parse_float_list(getattr(args, "thresholds", ""))

    val_predictions, summary = collect_for_experiment(
        exp_dir,
        checkpoint_path,
        base_arrays,
        val_idx,
        train_idx,
        device,
        args.batch_size,
    )
    best, val_rows = select_best_calibration(
        val_predictions,
        temperatures,
        powers,
        args.selection_metric,
        min_activity_probs=min_activity_probs,
        min_positive_rates=min_positive_rates,
        positive_rate_scales=positive_rate_scales,
        rate_gate_modes=rate_gate_modes,
        thresholds=thresholds,
        min_f1=float(getattr(args, "min_f1", 0.0)),
        max_link_rmse=float(getattr(args, "max_link_rmse", 0.0)),
        f1_penalty_weight=float(getattr(args, "f1_penalty_weight", 1000.0)),
        link_penalty_weight=float(getattr(args, "link_penalty_weight", 10.0)),
    )
    test_predictions, _ = collect_for_experiment(
        exp_dir,
        checkpoint_path,
        base_arrays,
        test_idx,
        train_idx,
        device,
        args.batch_size,
    )
    test_metrics = evaluate_gate_calibrated_rate(
        test_predictions["link_activity_prob"],
        test_predictions["link_positive_rate_pred"],
        test_predictions["link_rate_true"],
        test_predictions["link_activity_true"],
        temperature=float(best["temperature"]),
        power=float(best["power"]),
        threshold=float(best["threshold"]),
        min_activity_prob=float(best.get("min_activity_prob", 0.0)),
        min_positive_rate=float(best.get("min_positive_rate", 0.0)),
        positive_rate_scale=float(best.get("positive_rate_scale", 1.0)),
        rate_gate_mode=best.get("rate_gate_mode", "soft"),
    )
    base_test_metrics = evaluate_gate_calibrated_rate(
        test_predictions["link_activity_prob"],
        test_predictions["link_positive_rate_pred"],
        test_predictions["link_rate_true"],
        test_predictions["link_activity_true"],
        temperature=1.0,
        power=1.0,
        positive_rate_scale=1.0,
        rate_gate_mode="soft",
        threshold=None,
    )
    bypass_test_metrics = evaluate_gate_calibrated_rate(
        test_predictions["link_activity_prob"],
        test_predictions["link_positive_rate_pred"],
        test_predictions["link_rate_true"],
        test_predictions["link_activity_true"],
        temperature=1.0,
        power=0.0,
        positive_rate_scale=1.0,
        rate_gate_mode="soft",
        threshold=None,
    )
    return {
        "experiment": str(exp_dir),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_name": checkpoint_path.name,
        "experiment_label": f"{exp_dir.name}/{checkpoint_path.stem}",
        "config": summary["config"],
        "selection_metric": args.selection_metric,
        "best_val": best,
        "val_grid": val_rows,
        "test_calibrated": flatten_metrics(test_metrics, split="test_calibrated"),
        "test_uncalibrated_gate": flatten_metrics(base_test_metrics, split="test_uncalibrated_gate"),
        "test_bypass_gate": flatten_metrics(bypass_test_metrics, split="test_bypass_gate"),
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "gate_calibration_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    rows = []
    for exp in summary["experiments"]:
        name = exp.get("experiment_label", Path(exp["experiment"]).name)
        for row_name in ("test_uncalibrated_gate", "test_calibrated", "test_bypass_gate"):
            rows.append({"experiment": name, "checkpoint": exp.get("checkpoint_name", ""), **exp[row_name]})
    with (output_dir / "gate_calibration_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "gate_calibration_metrics.md").open("w", encoding="utf-8") as f:
        f.write("# PI-JWM v9 Hurdle Gate Calibration\n\n")
        f.write("| Experiment | Checkpoint | split | temp | power | pos scale | rate gate | min prob | min pos | P | R | F1 | Active-rate RMSE | Positive-rate active RMSE | Link RMSE |\n")
        f.write("|---|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            f.write(
                "| {experiment} | {checkpoint} | {split} | {temperature:.6f} | {power:.6f} | {positive_rate_scale:.6f} | {rate_gate_mode} | {min_activity_prob:.6f} | {min_positive_rate:.6f} | {precision:.6f} | {recall:.6f} | {f1:.6f} | {active_rate_rmse:.6f} | {positive_rate_active_rmse:.6f} | {link_rate_rmse:.6f} |\n".format(
                    **row
                )
            )


def run(args: argparse.Namespace) -> dict:
    device = choose_device(args.device)
    arrays = load_world_model_arrays(args.dataset_dir)
    experiments = []
    for name in args.experiment:
        exp_dir = resolve_experiment_path(name)
        for checkpoint_path in resolve_checkpoint_paths(exp_dir, getattr(args, "checkpoint_glob", "v8_dual_best.pt")):
            experiments.append(run_experiment(args, exp_dir, checkpoint_path, arrays, device))
    return {
        "dataset_dir": str(args.dataset_dir),
        "device": str(device),
        "checkpoint_glob": getattr(args, "checkpoint_glob", "v8_dual_best.pt"),
        "temperatures": parse_float_list(args.temperatures),
        "powers": parse_float_list(args.powers),
        "thresholds": parse_float_list(getattr(args, "thresholds", "")),
        "positive_rate_scales": parse_float_list(args.positive_rate_scales),
        "rate_gate_modes": [part for part in args.rate_gate_modes.replace(",", " ").split() if part],
        "selective_min_activity_probs": parse_float_list(args.selective_min_activity_probs),
        "selective_min_positive_rates": parse_float_list(args.selective_min_positive_rates),
        "experiments": experiments,
    }


def main() -> None:
    args = parse_args()
    summary = run(args)
    write_outputs(summary, args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
