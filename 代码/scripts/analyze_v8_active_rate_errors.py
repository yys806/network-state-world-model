"""Diagnose active-link rate amplitude errors for PI-JWM v8 checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from statistics import NormalDist
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pi_jwm.paths import ARTIFACTS_DIR
from pi_jwm.v6_data import (
    V6WorldModelDataset,
    build_physical_edge_history,
    collate_v6_world_model_batch,
    load_world_model_arrays,
    make_normalization_stats,
    split_by_seed,
)
from pi_jwm.v8_training import build_v8_model_from_arrays, collect_v8_predictions


DEFAULT_DATASET_DIR = (
    ARTIFACTS_DIR
    / "experiments"
    / "airfogsim_v0"
    / "datasets"
    / "world_model_dataset_active_heavy_v1"
)
DEFAULT_OUTPUT_DIR = ARTIFACTS_DIR / "experiments" / "pi_jwm_v9_bottleneck_diagnosis_20260619"
DEFAULT_EXPERIMENTS = (
    "pi_jwm_v8_gpu_full_ablation_20260614/v8_base_cross_aux_mlp",
    "pi_jwm_v8_gpu_full_ablation_20260614/v8_recurrent_cross_aux",
    "pi_jwm_v8_gpu_m5_loss_balance_20260614/m5_recurrent_composite_balanced",
    "pi_jwm_v8_gpu_combined_stack_20260615/combined_stgcn_full_recurrent_m5",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export active-rate error diagnostics for PI-JWM v8 models.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--experiment", action="append", default=[])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--max-test-samples", type=int, default=999999)
    parser.add_argument("--train-seeds", default=",".join(str(seed) for seed in range(16)))
    parser.add_argument("--val-seeds", default="16,17")
    parser.add_argument("--test-seeds", default="18,19")
    parser.add_argument("--diagnosis-split", choices=("val", "test"), default="val")
    return parser.parse_args()


def resolve_diagnosis_indices(
    sample_seed: np.ndarray,
    train_seeds: list[int] | None = None,
    val_seeds: list[int] | None = None,
    test_seeds: list[int] | None = None,
    diagnosis_split: str = "val",
) -> tuple[np.ndarray, dict[str, object]]:
    from run_world_model_v8_full_training import resolve_seed_splits

    train_idx, val_idx, test_idx, spec = resolve_seed_splits(
        sample_seed,
        train_seeds=train_seeds,
        val_seeds=val_seeds,
        test_seeds=test_seeds,
    )
    if diagnosis_split not in {"val", "test"}:
        raise ValueError("diagnosis_split must be one of: val, test")
    indices = val_idx if diagnosis_split == "val" else test_idx
    return indices, {**spec, "diagnosis_split": diagnosis_split, "train_count": int(len(train_idx))}


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


def resolve_experiment_path(name_or_path: str) -> Path:
    path = Path(name_or_path)
    if path.exists():
        return path
    artifact_path = ARTIFACTS_DIR / "experiments" / name_or_path
    if artifact_path.exists():
        return artifact_path
    raise FileNotFoundError(f"Experiment directory not found: {name_or_path}")


def load_model_for_experiment(exp_dir: Path, arrays: dict[str, np.ndarray], device: torch.device):
    summary_path = exp_dir / "v8_full_training_summary.json"
    checkpoint_path = exp_dir / "checkpoints" / "v8_dual_best.pt"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary: {summary_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing best checkpoint: {checkpoint_path}")
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


def prepare_arrays_for_experiment(exp_dir: Path, arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    summary_path = exp_dir / "v8_full_training_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("config", {}).get("use_event_memory_features"):
        from run_world_model_v8_full_training import add_event_memory_features

        return add_event_memory_features(arrays)
    return arrays


def collect_predictions_for_experiment(
    exp_dir: Path,
    arrays: dict[str, np.ndarray],
    stats: dict,
    test_subset: Subset,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, np.ndarray], dict]:
    arrays = prepare_arrays_for_experiment(exp_dir, arrays)
    model, summary = load_model_for_experiment(exp_dir, arrays, device)
    config = summary["config"]
    loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, collate_fn=collate_v6_world_model_batch)
    predictions = collect_v8_predictions(
        model,
        loader,
        device,
        stats,
        rate_output_mode=config.get("rate_output_mode", "main"),
        inactive_rate_value=float(config.get("inactive_rate_value", 0.0)),
    )
    return predictions, summary


def flatten_active_rate_rows(
    model_name: str,
    predictions: dict[str, np.ndarray],
    arrays: dict[str, np.ndarray],
    sample_indices: np.ndarray,
    physical_edge_last: np.ndarray,
) -> list[dict[str, float | int | str]]:
    pred = _squeeze_last(predictions["link_rate_pred"])
    true = _squeeze_last(predictions["link_rate_true"])
    active = _squeeze_last(predictions["link_activity_true"]) > 0.5
    prob = _squeeze_last(predictions["link_activity_prob"])
    aux = _squeeze_last(predictions.get("link_active_rate_aux_pred", predictions["link_rate_pred"]))
    edge_src = arrays["edge_src_idx"]
    edge_dst = arrays["edge_dst_idx"]
    future_actions = arrays["edge_a_future"][sample_indices]
    x_link_last = arrays["x_link"][sample_indices, -1]
    rate_idx = _feature_index(arrays, "link_features", "rate_sum", default=1)
    rows = []
    sample_pos, horizon_pos, edge_pos = np.where(active)
    for sample_i, horizon_i, edge_i in zip(sample_pos, horizon_pos, edge_pos):
        true_rate = float(true[sample_i, horizon_i, edge_i])
        pred_rate = float(pred[sample_i, horizon_i, edge_i])
        residual = pred_rate - true_rate
        action = future_actions[sample_i, horizon_i, edge_i]
        physical = physical_edge_last[sample_i, edge_i]
        rows.append(
            {
                "model": model_name,
                "source_sample_idx": int(sample_indices[sample_i]),
                "sample_seed": int(arrays["sample_seed"][sample_indices[sample_i]]),
                "horizon": int(horizon_i + 1),
                "edge_id": int(edge_i),
                "edge_src": int(edge_src[edge_i]),
                "edge_dst": int(edge_dst[edge_i]),
                "true_rate": true_rate,
                "pred_rate": pred_rate,
                "aux_pred_rate": float(aux[sample_i, horizon_i, edge_i]),
                "residual": residual,
                "abs_error": abs(residual),
                "relative_error": abs(residual) / max(abs(true_rate), 1e-6),
                "activity_prob": float(prob[sample_i, horizon_i, edge_i]),
                "last_rate": float(x_link_last[sample_i, edge_i, rate_idx]),
                "action_l1": float(np.abs(action).sum()),
                "action_nonzero": int(np.count_nonzero(np.abs(action) > 1e-6)),
                "distance_3d": float(physical[3]) if physical.shape[0] > 3 else float("nan"),
            }
        )
    add_bucket_columns(rows)
    return rows


def add_bucket_columns(rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    true_rates = np.asarray([float(row["true_rate"]) for row in rows], dtype=np.float64)
    rate_q1, rate_q2 = np.quantile(true_rates, [1 / 3, 2 / 3])
    distances = np.asarray([float(row["distance_3d"]) for row in rows], dtype=np.float64)
    finite_dist = distances[np.isfinite(distances)]
    dist_q1, dist_q2 = np.quantile(finite_dist, [1 / 3, 2 / 3]) if finite_dist.size else (float("nan"), float("nan"))
    actions = np.asarray([float(row["action_l1"]) for row in rows], dtype=np.float64)
    action_q1, action_q2 = np.quantile(actions, [1 / 3, 2 / 3])
    for row in rows:
        row["rate_bucket"] = _tertile_label(float(row["true_rate"]), rate_q1, rate_q2)
        row["distance_bucket"] = _tertile_label(float(row["distance_3d"]), dist_q1, dist_q2)
        row["action_bucket"] = _tertile_label(float(row["action_l1"]), action_q1, action_q2)
        prob = float(row["activity_prob"])
        if prob >= 0.95:
            row["activity_confidence_bucket"] = "high_conf"
        elif prob >= 0.75:
            row["activity_confidence_bucket"] = "mid_conf"
        else:
            row["activity_confidence_bucket"] = "low_conf"
        row["signed_error_bucket"] = "under" if float(row["residual"]) < 0.0 else "over"


def summarize_bucket_metrics(
    rows: list[dict[str, float | int | str]],
    group_keys: Iterable[str],
) -> list[dict[str, float | int | str]]:
    groups: dict[tuple, list[dict[str, float | int | str]]] = {}
    group_keys = tuple(group_keys)
    for row in rows:
        key = tuple(row[name] for name in group_keys)
        groups.setdefault(key, []).append(row)
    summary = []
    for key, group_rows in sorted(groups.items(), key=lambda item: tuple(str(value) for value in item[0])):
        residual = np.asarray([float(row["residual"]) for row in group_rows], dtype=np.float64)
        abs_error = np.asarray([float(row["abs_error"]) for row in group_rows], dtype=np.float64)
        true_rate = np.asarray([float(row["true_rate"]) for row in group_rows], dtype=np.float64)
        record = {name: value for name, value in zip(group_keys, key)}
        record.update(
            {
                "count": int(len(group_rows)),
                "sse": float(np.sum(residual**2)),
                "rmse": float(np.sqrt(np.mean(residual**2))),
                "mae": float(np.mean(abs_error)),
                "bias": float(np.mean(residual)),
                "mean_true_rate": float(np.mean(true_rate)),
                "mean_pred_rate": float(np.mean([float(row["pred_rate"]) for row in group_rows])),
                "under_prediction_rate": float(np.mean(residual < 0.0)),
            }
        )
        summary.append(record)
    if summary:
        if "model" in group_keys:
            total_sse = {}
            for record in summary:
                model = str(record["model"])
                total_sse[model] = total_sse.get(model, 0.0) + float(record["sse"])
            for record in summary:
                denominator = total_sse[str(record["model"])]
                record["sse_share"] = float(record["sse"] / denominator) if denominator > 0.0 else 0.0
        else:
            denominator = sum(float(record["sse"]) for record in summary)
            for record in summary:
                record["sse_share"] = float(record["sse"] / denominator) if denominator > 0.0 else 0.0
    return summary


def diagnose_hurdle_predictions(predictions: dict[str, np.ndarray]) -> dict[str, float | int | str]:
    final_rate = _squeeze_last(predictions["link_rate_pred"]).astype(np.float64)
    positive_rate = _squeeze_last(predictions.get("link_positive_rate_pred", final_rate)).astype(np.float64)
    true_rate = _squeeze_last(predictions["link_rate_true"]).astype(np.float64)
    active = _squeeze_last(predictions["link_activity_true"]) > 0.5
    activity_prob = _squeeze_last(predictions["link_activity_prob"]).astype(np.float64)
    inactive = ~active
    if not np.any(active):
        return {
            "active_count": 0,
            "active_final_rmse": float("nan"),
            "active_positive_rmse": float("nan"),
            "gate_suppression_gap": float("nan"),
            "inactive_rate_mass_sum": float(np.abs(final_rate[inactive]).sum()),
            "inactive_rate_mass_mean": float(np.abs(final_rate[inactive]).mean()) if np.any(inactive) else 0.0,
            "inactive_activity_prob_mean": float(activity_prob[inactive].mean()) if np.any(inactive) else 0.0,
            "top20_active_sse_share": float("nan"),
            "top20_mean_underprediction_ratio": float("nan"),
            "log_rate_qq_r2": float("nan"),
            "recommended_method": "insufficient_active_data",
        }
    active_true = true_rate[active]
    active_final = final_rate[active]
    active_positive = positive_rate[active]
    final_error = active_final - active_true
    positive_error = active_positive - active_true
    final_rmse = float(np.sqrt(np.mean(final_error**2)))
    positive_rmse = float(np.sqrt(np.mean(positive_error**2)))
    tail_threshold = float(np.quantile(active_true, 0.8))
    tail = active_true >= tail_threshold
    total_sse = float(np.sum(final_error**2))
    tail_sse = float(np.sum(final_error[tail] ** 2))
    tail_sse_share = tail_sse / total_sse if total_sse > 0.0 else 0.0
    tail_under_ratio = float(np.mean((active_true[tail] - active_final[tail]) / np.maximum(active_true[tail], 1e-6)))
    qq_r2 = _normal_qq_r2(np.log(np.maximum(active_true, 1e-6)))
    if tail_sse_share >= 0.4 and tail_under_ratio >= 0.1:
        recommended = "lds"
    elif np.isfinite(qq_r2) and qq_r2 >= 0.95:
        recommended = "ziln"
    else:
        recommended = "balanced_mse"
    return {
        "active_count": int(active.sum()),
        "active_final_rmse": final_rmse,
        "active_positive_rmse": positive_rmse,
        "gate_suppression_gap": float(final_rmse - positive_rmse),
        "inactive_rate_mass_sum": float(np.abs(final_rate[inactive]).sum()),
        "inactive_rate_mass_mean": float(np.abs(final_rate[inactive]).mean()) if np.any(inactive) else 0.0,
        "inactive_activity_prob_mean": float(activity_prob[inactive].mean()) if np.any(inactive) else 0.0,
        "top20_rate_threshold": tail_threshold,
        "top20_active_sse_share": float(tail_sse_share),
        "top20_mean_underprediction_ratio": tail_under_ratio,
        "log_rate_qq_r2": qq_r2,
        "recommended_method": recommended,
    }


def _normal_qq_r2(values: np.ndarray) -> float:
    values = np.sort(np.asarray(values, dtype=np.float64))
    if values.size < 3 or np.allclose(values, values[0]):
        return float("nan")
    probabilities = (np.arange(values.size, dtype=np.float64) + 0.5) / values.size
    normal = NormalDist()
    theoretical = np.asarray([normal.inv_cdf(float(probability)) for probability in probabilities])
    correlation = np.corrcoef(values, theoretical)[0, 1]
    return float(correlation**2)


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def render_report(model_summary: list[dict], bucket_tables: dict[str, list[dict]]) -> str:
    lines = [
        "# PI-JWM v8 Active-Rate Error Diagnosis",
        "",
        "This report uses true active link-steps only and compares denormalized selected rate predictions.",
        "",
        "## Model Summary",
        "",
        "| model | active count | RMSE | MAE | bias | under-prediction rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in model_summary:
        lines.append(
            f"| {row['model']} | {row['count']} | {row['rmse']:.3f} | {row['mae']:.3f} | "
            f"{row['bias']:.3f} | {row['under_prediction_rate']:.3f} |"
        )
    for title, rows in bucket_tables.items():
        lines.extend(["", f"## {title}", ""])
        if not rows:
            lines.append("No rows.")
            continue
        keys = [key for key in rows[0].keys() if key not in {"rmse", "mae", "bias", "mean_true_rate", "mean_pred_rate", "under_prediction_rate"}]
        lines.append("| " + " | ".join(keys + ["count", "RMSE", "MAE", "bias", "mean true", "mean pred", "under rate"]) + " |")
        lines.append("| " + " | ".join(["---"] * len(keys) + ["---:"] * 7) + " |")
        for row in rows:
            prefix = [str(row[key]) for key in keys]
            metrics = [
                str(row["count"]),
                f"{row['rmse']:.3f}",
                f"{row['mae']:.3f}",
                f"{row['bias']:.3f}",
                f"{row['mean_true_rate']:.3f}",
                f"{row['mean_pred_rate']:.3f}",
                f"{row['under_prediction_rate']:.3f}",
            ]
            lines.append("| " + " | ".join(prefix + metrics) + " |")
    return "\n".join(lines) + "\n"


def _squeeze_last(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim == 4 and values.shape[-1] == 1:
        return values[..., 0]
    return values


def _feature_index(arrays: dict[str, np.ndarray], key: str, name: str, default: int) -> int:
    if key not in arrays:
        return default
    names = [str(item) for item in arrays[key]]
    return names.index(name) if name in names else default


def _tertile_label(value: float, q1: float, q2: float) -> str:
    if not np.isfinite(value) or not np.isfinite(q1) or not np.isfinite(q2):
        return "unknown"
    if value <= q1:
        return "low"
    if value <= q2:
        return "mid"
    return "high"


def main() -> None:
    from run_world_model_v8_full_training import parse_seed_list

    args = parse_args()
    device = choose_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    arrays = load_world_model_arrays(args.dataset_dir)
    diagnosis_idx, split_spec = resolve_diagnosis_indices(
        arrays["sample_seed"],
        train_seeds=parse_seed_list(args.train_seeds),
        val_seeds=parse_seed_list(args.val_seeds),
        test_seeds=parse_seed_list(args.test_seeds),
        diagnosis_split=args.diagnosis_split,
    )
    train_idx = np.where(np.isin(arrays["sample_seed"], split_spec["train_seeds"]))[0]
    stats = make_normalization_stats(arrays, train_idx)
    used_test_idx = diagnosis_idx[: min(args.max_test_samples, len(diagnosis_idx))]
    test_ds = V6WorldModelDataset(arrays, diagnosis_idx, stats)
    test_subset = Subset(test_ds, range(len(used_test_idx)))
    physical_last = build_physical_edge_history(
        arrays["x_node"][used_test_idx],
        arrays["edge_src_idx"],
        arrays["edge_dst_idx"],
        arrays["valid_edge_node"],
    ).numpy()[:, -1]

    experiment_names = args.experiment or list(DEFAULT_EXPERIMENTS)
    all_rows = []
    experiment_records = []
    hurdle_diagnosis = {}
    for name in experiment_names:
        exp_dir = resolve_experiment_path(name)
        predictions, summary = collect_predictions_for_experiment(
            exp_dir,
            arrays,
            stats,
            test_subset,
            device,
            args.batch_size,
        )
        model_name = exp_dir.name
        rows = flatten_active_rate_rows(model_name, predictions, arrays, used_test_idx, physical_last)
        all_rows.extend(rows)
        hurdle_diagnosis[model_name] = diagnose_hurdle_predictions(predictions)
        experiment_records.append(
            {
                "model": model_name,
                "summary_path": str(exp_dir / "v8_full_training_summary.json"),
                "best_epoch": int(summary["best_epoch"]),
                "row_count": len(rows),
            }
        )

    tables = {
        "by_rate_bucket": summarize_bucket_metrics(all_rows, ("model", "rate_bucket")),
        "by_horizon": summarize_bucket_metrics(all_rows, ("model", "horizon")),
        "by_edge": summarize_bucket_metrics(all_rows, ("model", "edge_id")),
        "by_distance_bucket": summarize_bucket_metrics(all_rows, ("model", "distance_bucket")),
        "by_action_bucket": summarize_bucket_metrics(all_rows, ("model", "action_bucket")),
        "by_activity_confidence": summarize_bucket_metrics(all_rows, ("model", "activity_confidence_bucket")),
    }
    model_summary = summarize_bucket_metrics(all_rows, ("model",))

    write_csv(args.output_dir / "active_rate_predictions.csv", all_rows)
    write_csv(args.output_dir / "model_summary.csv", model_summary)
    (args.output_dir / "bottleneck_diagnosis.json").write_text(
        json.dumps(hurdle_diagnosis, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "diagnosis_split.json").write_text(
        json.dumps(split_spec, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_csv(args.output_dir / "experiment_records.csv", experiment_records)
    for name, rows in tables.items():
        write_csv(args.output_dir / f"{name}.csv", rows)
    report = render_report(model_summary, tables)
    (args.output_dir / "active_rate_error_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "active_rows": len(all_rows)}, indent=2))


if __name__ == "__main__":
    main()
