"""Post-hoc active-rate residual calibration for PI-JWM v8 checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Subset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from pi_jwm.paths import ARTIFACTS_DIR
from pi_jwm.v6_data import (
    V6WorldModelDataset,
    build_physical_edge_history,
    load_world_model_arrays,
    make_normalization_stats,
    split_by_seed,
)

from analyze_v8_active_rate_errors import (
    collect_predictions_for_experiment,
    flatten_active_rate_rows,
    resolve_experiment_path,
)


DEFAULT_DATASET_DIR = (
    ARTIFACTS_DIR
    / "experiments"
    / "airfogsim_v0"
    / "datasets"
    / "world_model_dataset_seed0_9_v0"
)
DEFAULT_EXPERIMENT = "pi_jwm_v8_gpu_m5_loss_balance_20260614/m5_recurrent_composite_balanced"
DEFAULT_OUTPUT_DIR = ARTIFACTS_DIR / "experiments" / "pi_jwm_v8_active_rate_residual_calibration_20260615"
DEFAULT_ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)


FEATURE_NAMES = (
    "pred_rate",
    "activity_prob",
    "last_rate",
    "action_l1",
    "action_nonzero",
    "horizon",
    "distance_3d",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate v8 active-rate residuals without test leakage.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--max-train-samples", type=int, default=999999)
    parser.add_argument("--max-val-samples", type=int, default=999999)
    parser.add_argument("--max-test-samples", type=int, default=999999)
    parser.add_argument("--train-seeds", default=None, help="Comma/space separated train seed ids; default uses seeds 0-7.")
    parser.add_argument("--val-seeds", default=None, help="Comma/space separated validation seed ids; default uses seed 8.")
    parser.add_argument("--test-seeds", default=None, help="Comma/space separated test seed ids; default uses seed 9.")
    return parser.parse_args()




def parse_seed_list(value: str | None) -> list[int] | None:
    if value is None:
        return None
    parts = [part for part in value.replace(",", " ").split() if part]
    if not parts:
        return None
    return [int(part) for part in parts]


def resolve_seed_splits(
    sample_seed: np.ndarray,
    train_seeds: list[int] | None = None,
    val_seeds: list[int] | None = None,
    test_seeds: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, list[int]]]:
    if train_seeds is None and val_seeds is None and test_seeds is None:
        train_idx, val_idx, test_idx = split_by_seed(sample_seed)
        spec = {"train_seeds": list(range(0, 8)), "val_seeds": [8], "test_seeds": [9]}
        return train_idx, val_idx, test_idx, spec
    train_seeds = list(range(0, 8)) if train_seeds is None else train_seeds
    val_seeds = [8] if val_seeds is None else val_seeds
    test_seeds = [9] if test_seeds is None else test_seeds
    sample_seed = np.asarray(sample_seed)
    train_idx = np.where(np.isin(sample_seed, train_seeds))[0]
    val_idx = np.where(np.isin(sample_seed, val_seeds))[0]
    test_idx = np.where(np.isin(sample_seed, test_seeds))[0]
    for name, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        if len(idx) == 0:
            raise ValueError(f"Custom seed split produced an empty {name} split.")
    spec = {
        "train_seeds": [int(seed) for seed in train_seeds],
        "val_seeds": [int(seed) for seed in val_seeds],
        "test_seeds": [int(seed) for seed in test_seeds],
    }
    return train_idx, val_idx, test_idx, spec

def choose_device(device_arg: str) -> torch.device:
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


def evaluate_rate_predictions(true: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
    true = np.asarray(true, dtype=np.float64).reshape(-1)
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    residual = pred - true
    return {
        "count": int(true.size),
        "rmse": float(np.sqrt(np.mean(residual**2))) if true.size else float("nan"),
        "mae": float(np.mean(np.abs(residual))) if true.size else float("nan"),
        "bias": float(np.mean(residual)) if true.size else float("nan"),
        "mean_true_rate": float(np.mean(true)) if true.size else float("nan"),
        "mean_pred_rate": float(np.mean(pred)) if true.size else float("nan"),
        "under_prediction_rate": float(np.mean(residual < 0.0)) if true.size else float("nan"),
    }


def select_best_candidate(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("candidate rows must not be empty")
    return min(rows, key=lambda row: (float(row["val_rmse"]), str(row["model"])))


def fit_ridge_residual_calibrator(
    train_rows: list[dict],
    val_rows: list[dict],
    alphas: Iterable[float] = DEFAULT_ALPHAS,
    include_edge_id: bool = True,
    num_edges: int = 188,
    name: str = "ridge_residual",
) -> tuple[dict, list[dict]]:
    x_train = build_feature_matrix(train_rows, include_edge_id=include_edge_id, num_edges=num_edges)
    x_val = build_feature_matrix(val_rows, include_edge_id=include_edge_id, num_edges=num_edges)
    y_train = np.asarray([float(row["true_rate"]) - float(row["pred_rate"]) for row in train_rows], dtype=np.float64)
    y_val_true = np.asarray([float(row["true_rate"]) for row in val_rows], dtype=np.float64)
    y_val_base = np.asarray([float(row["pred_rate"]) for row in val_rows], dtype=np.float64)

    x_mean, x_std = fit_standardizer(x_train)
    x_train_s = standardize(x_train, x_mean, x_std)
    x_val_s = standardize(x_val, x_mean, x_std)
    tuning_rows = []
    best = None
    for alpha in alphas:
        weights = fit_ridge(x_train_s, y_train, float(alpha))
        residual_pred = predict_ridge(x_val_s, weights)
        pred_val = np.clip(y_val_base + residual_pred, 0.0, None)
        metrics = evaluate_rate_predictions(y_val_true, pred_val)
        row = {
            "model": name,
            "alpha": float(alpha),
            "val_rmse": metrics["rmse"],
            "val_mae": metrics["mae"],
            "val_bias": metrics["bias"],
        }
        tuning_rows.append(row)
        if best is None or row["val_rmse"] < best["val_rmse"]:
            best = {**row, "weights": weights}
    calibrator = {
        "name": name,
        "kind": "ridge_residual",
        "include_edge_id": bool(include_edge_id),
        "num_edges": int(num_edges),
        "feature_names": list(FEATURE_NAMES) + (["edge_id_onehot"] if include_edge_id else []),
        "x_mean": x_mean,
        "x_std": x_std,
        "weights": best["weights"],
        "alpha": float(best["alpha"]),
        "val_rmse": float(best["val_rmse"]),
    }
    return calibrator, tuning_rows


def predict_calibrated_rates(calibrator: dict, rows: list[dict]) -> np.ndarray:
    base = np.asarray([float(row["pred_rate"]) for row in rows], dtype=np.float64)
    if calibrator["kind"] == "identity":
        return base
    if calibrator["kind"] == "global_bias":
        return np.clip(base + float(calibrator["bias_correction"]), 0.0, None)
    if calibrator["kind"] == "edge_bias":
        return predict_edge_bias(calibrator, rows, base)
    if calibrator["kind"] == "ridge_residual":
        x = build_feature_matrix(rows, bool(calibrator["include_edge_id"]), int(calibrator["num_edges"]))
        x_s = standardize(x, calibrator["x_mean"], calibrator["x_std"])
        residual = predict_ridge(x_s, calibrator["weights"])
        return np.clip(base + residual, 0.0, None)
    raise ValueError(f"Unknown calibrator kind: {calibrator['kind']}")


def build_identity_calibrator() -> dict:
    return {"name": "uncalibrated", "kind": "identity"}


def build_global_bias_calibrator(train_rows: list[dict]) -> dict:
    correction = np.mean([float(row["true_rate"]) - float(row["pred_rate"]) for row in train_rows])
    return {"name": "global_bias", "kind": "global_bias", "bias_correction": float(correction)}


def build_edge_bias_calibrator(train_rows: list[dict], min_count: int = 5) -> dict:
    global_bias = float(np.mean([float(row["true_rate"]) - float(row["pred_rate"]) for row in train_rows]))
    buckets: dict[int, list[float]] = {}
    for row in train_rows:
        buckets.setdefault(int(row["edge_id"]), []).append(float(row["true_rate"]) - float(row["pred_rate"]))
    edge_bias = {
        int(edge): float(np.mean(values))
        for edge, values in buckets.items()
        if len(values) >= min_count
    }
    return {
        "name": "edge_bias_min5",
        "kind": "edge_bias",
        "global_bias": global_bias,
        "edge_bias": edge_bias,
        "min_count": int(min_count),
    }


def predict_edge_bias(calibrator: dict, rows: list[dict], base: np.ndarray) -> np.ndarray:
    edge_bias = {int(k): float(v) for k, v in calibrator["edge_bias"].items()}
    global_bias = float(calibrator["global_bias"])
    correction = np.asarray([edge_bias.get(int(row["edge_id"]), global_bias) for row in rows], dtype=np.float64)
    return np.clip(base + correction, 0.0, None)


def build_feature_matrix(rows: list[dict], include_edge_id: bool = True, num_edges: int = 188) -> np.ndarray:
    values = []
    for row in rows:
        values.append(
            [
                float(row["pred_rate"]),
                float(row["activity_prob"]),
                float(row["last_rate"]),
                float(row["action_l1"]),
                float(row["action_nonzero"]),
                float(row["horizon"]),
                _finite_or_zero(float(row.get("distance_3d", 0.0))),
            ]
        )
    x = np.asarray(values, dtype=np.float64)
    if include_edge_id:
        onehot = np.zeros((len(rows), num_edges), dtype=np.float64)
        for i, row in enumerate(rows):
            edge_id = int(row["edge_id"])
            if 0 <= edge_id < num_edges:
                onehot[i, edge_id] = 1.0
        x = np.concatenate([x, onehot], axis=1)
    return x


def fit_standardizer(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return mean, std


def standardize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (np.asarray(x, dtype=np.float64) - mean) / std


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float64)], axis=1)
    eye = np.eye(x_aug.shape[1], dtype=np.float64)
    eye[-1, -1] = 0.0
    return np.linalg.solve(x_aug.T @ x_aug + alpha * eye, x_aug.T @ y.reshape(-1, 1))


def predict_ridge(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float64)], axis=1)
    return (x_aug @ weights).reshape(-1)


def _finite_or_zero(value: float) -> float:
    return value if np.isfinite(value) else 0.0


def collect_split_rows(
    exp_dir: Path,
    arrays: dict[str, np.ndarray],
    stats: dict,
    split_indices: np.ndarray,
    split_name: str,
    device: torch.device,
    batch_size: int,
) -> tuple[list[dict], dict]:
    ds = V6WorldModelDataset(arrays, split_indices, stats)
    subset = Subset(ds, range(len(split_indices)))
    physical_last = build_physical_edge_history(
        arrays["x_node"][split_indices],
        arrays["edge_src_idx"],
        arrays["edge_dst_idx"],
        arrays["valid_edge_node"],
    ).numpy()[:, -1]
    predictions, summary = collect_predictions_for_experiment(exp_dir, arrays, stats, subset, device, batch_size)
    rows = flatten_active_rate_rows(exp_dir.name, predictions, arrays, split_indices, physical_last)
    for row in rows:
        row["split"] = split_name
    return rows, summary


def evaluate_calibrators(calibrators: list[dict], split_rows: dict[str, list[dict]]) -> tuple[list[dict], list[dict]]:
    metric_rows = []
    prediction_rows = []
    for calibrator in calibrators:
        row = {"model": calibrator["name"]}
        for split_name in ("train", "val", "test"):
            rows = split_rows[split_name]
            true = np.asarray([float(item["true_rate"]) for item in rows], dtype=np.float64)
            pred = predict_calibrated_rates(calibrator, rows)
            metrics = evaluate_rate_predictions(true, pred)
            for key, value in metrics.items():
                row[f"{split_name}_{key}"] = value
            for source, calibrated_pred in zip(rows, pred):
                prediction_rows.append(
                    {
                        "calibrator": calibrator["name"],
                        "split": split_name,
                        "source_sample_idx": source["source_sample_idx"],
                        "sample_seed": source["sample_seed"],
                        "horizon": source["horizon"],
                        "edge_id": source["edge_id"],
                        "true_rate": source["true_rate"],
                        "pred_rate": source["pred_rate"],
                        "calibrated_pred_rate": float(calibrated_pred),
                        "residual_after": float(calibrated_pred - float(source["true_rate"])),
                        "last_rate": source["last_rate"],
                        "action_l1": source["action_l1"],
                        "activity_prob": source["activity_prob"],
                    }
                )
        row["val_rmse"] = row["val_rmse"]
        row["test_rmse"] = row["test_rmse"]
        metric_rows.append(row)
    return metric_rows, prediction_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def render_report(summary: dict) -> str:
    lines = [
        "# PI-JWM v8 Active-Rate Residual Calibration",
        "",
        "This report fits post-hoc residual calibrators on active train link-steps, selects by validation RMSE, and reports test metrics without test leakage.",
        "",
        "## Split Counts",
        "",
        "| split | active rows |",
        "|---|---:|",
    ]
    for split, count in summary["split_active_counts"].items():
        lines.append(f"| {split} | {count} |")
    lines.extend(["", "## Metrics", "", "| model | val RMSE | test RMSE | test MAE | test bias | test under-rate |", "|---|---:|---:|---:|---:|---:|"])
    for row in summary["metrics"]:
        lines.append(
            f"| {row['model']} | {row['val_rmse']:.3f} | {row['test_rmse']:.3f} | "
            f"{row['test_mae']:.3f} | {row['test_bias']:.3f} | {row['test_under_prediction_rate']:.3f} |"
        )
    best = summary["best_by_val"]
    lines.extend(
        [
            "",
            "## Selection",
            "",
            f"Selected calibrator by validation RMSE: `{best['model']}`.",
            "",
            "## Outputs",
            "",
        ]
    )
    for key, value in summary["outputs"].items():
        lines.append(f"- {key}: `{value}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    arrays = load_world_model_arrays(args.dataset_dir)
    train_idx, val_idx, test_idx, split_seed_spec = resolve_seed_splits(
        arrays["sample_seed"],
        train_seeds=parse_seed_list(args.train_seeds),
        val_seeds=parse_seed_list(args.val_seeds),
        test_seeds=parse_seed_list(args.test_seeds),
    )
    train_idx = train_idx[: min(args.max_train_samples, len(train_idx))]
    val_idx = val_idx[: min(args.max_val_samples, len(val_idx))]
    test_idx = test_idx[: min(args.max_test_samples, len(test_idx))]
    stats = make_normalization_stats(arrays, train_idx)
    exp_dir = resolve_experiment_path(args.experiment)

    split_rows = {}
    summary = None
    for split_name, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        rows, summary = collect_split_rows(exp_dir, arrays, stats, idx, split_name, device, args.batch_size)
        split_rows[split_name] = rows

    ridge_basic, ridge_basic_tuning = fit_ridge_residual_calibrator(
        split_rows["train"], split_rows["val"], include_edge_id=False, num_edges=arrays["y_link_rate"].shape[2], name="ridge_residual_basic"
    )
    ridge_edge, ridge_edge_tuning = fit_ridge_residual_calibrator(
        split_rows["train"], split_rows["val"], include_edge_id=True, num_edges=arrays["y_link_rate"].shape[2], name="ridge_residual_edge"
    )
    calibrators = [
        build_identity_calibrator(),
        build_global_bias_calibrator(split_rows["train"]),
        build_edge_bias_calibrator(split_rows["train"]),
        ridge_basic,
        ridge_edge,
    ]
    metric_rows, prediction_rows = evaluate_calibrators(calibrators, split_rows)
    best = select_best_candidate(metric_rows)
    tuning_rows = ridge_basic_tuning + ridge_edge_tuning

    raw_rows = split_rows["train"] + split_rows["val"] + split_rows["test"]
    write_csv(args.output_dir / "active_rate_split_predictions.csv", raw_rows)
    write_csv(args.output_dir / "calibration_metrics.csv", metric_rows)
    write_csv(args.output_dir / "calibrated_predictions.csv", prediction_rows)
    write_csv(args.output_dir / "ridge_tuning.csv", tuning_rows)

    summary_record = {
        "framework": "PI-JWM",
        "experiment": str(exp_dir),
        "checkpoint_summary": str(exp_dir / "v8_full_training_summary.json"),
        "best_epoch": int(summary["best_epoch"]),
        "split_active_counts": {split: len(rows) for split, rows in split_rows.items()},
        "metrics": metric_rows,
        "best_by_val": best,
        "calibrators": serialize_calibrators(calibrators),
        "outputs": {
            "raw_active_rows": str(args.output_dir / "active_rate_split_predictions.csv"),
            "metrics": str(args.output_dir / "calibration_metrics.csv"),
            "predictions": str(args.output_dir / "calibrated_predictions.csv"),
            "tuning": str(args.output_dir / "ridge_tuning.csv"),
            "report": str(args.output_dir / "active_rate_residual_calibration_report.md"),
        },
    }
    (args.output_dir / "active_rate_residual_calibration_summary.json").write_text(
        json.dumps(summary_record, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    (args.output_dir / "active_rate_residual_calibration_report.md").write_text(render_report(summary_record), encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "best_by_val": best["model"], "test_rmse": best["test_rmse"]}, indent=2))


def serialize_calibrators(calibrators: list[dict]) -> list[dict]:
    records = []
    for calibrator in calibrators:
        record = {key: value for key, value in calibrator.items() if key not in {"weights", "x_mean", "x_std"}}
        records.append(record)
    return records


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


if __name__ == "__main__":
    main()
