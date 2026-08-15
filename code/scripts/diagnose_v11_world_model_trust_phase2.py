"""World-model trust diagnostics for PI-JWM v11-candidate phase 2.

The script evaluates a frozen PI-JWM world model without training.  It reports
overall action-ablation metrics, resource-load sliced errors, and sensitivity
of predictions to future-action removal.
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
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from evaluate_v9_action_ablation import ActionAblationDataset, load_model_for_experiment  # noqa: E402
from pi_jwm.v6_data import (  # noqa: E402
    V6WorldModelDataset,
    collate_v6_world_model_batch,
    load_world_model_arrays,
    make_normalization_stats,
)
from pi_jwm.v6_metrics import active_rate_metrics, activity_metrics, regression_metrics  # noqa: E402
from pi_jwm.v8_training import collect_v8_predictions  # noqa: E402
from run_world_model_v8_full_training import add_event_memory_features, resolve_seed_splits  # noqa: E402


DEFAULT_EXPERIMENT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "experiments"
    / "pi_jwm_v9_expanded_v2_gpu_20260619"
    / "v2_hurdle_baseline"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose frozen PI-JWM world-model trust.")
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means use full val/test splits.")
    parser.add_argument(
        "--modes",
        default="normal_actions,zero_future_actions,zero_history_actions",
        help="Comma/space separated action modes.",
    )
    return parser.parse_args()


def choose_device(value: str) -> torch.device:
    if value == "cpu":
        return torch.device("cpu")
    if value == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_modes(value: str) -> list[str]:
    return [part for part in value.replace(",", " ").split() if part]


def resolve_project_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def finite_float(value: object) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def pearson_or_none(x: np.ndarray, y: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3 or float(np.std(x)) < 1e-12 or float(np.std(y)) < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def choose_activity_threshold(prob: np.ndarray, true: np.ndarray) -> float:
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(0.01, 0.99, 99):
        f1 = activity_metrics(prob, true, threshold=float(threshold))["f1"]
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)
    return best_threshold


def overall_metric_row(mode: str, split: str, predictions: dict[str, np.ndarray], threshold: float) -> dict:
    activity = activity_metrics(
        predictions["link_activity_prob"],
        predictions["link_activity_true"],
        threshold=threshold,
    )
    link = regression_metrics(predictions["link_rate_pred"], predictions["link_rate_true"])
    active = active_rate_metrics(
        predictions["link_rate_pred"],
        predictions["link_rate_true"],
        predictions["link_activity_true"],
    )
    task = regression_metrics(predictions["task_pred"], predictions["task_true"])
    node = regression_metrics(predictions["node_pred"], predictions["node_true"])
    return {
        "mode": mode,
        "split": split,
        "threshold": float(threshold),
        "activity_precision": activity["precision"],
        "activity_recall": activity["recall"],
        "activity_f1": activity["f1"],
        "active_rate_rmse": active["active_rmse"],
        "active_rate_mae": active["active_mae"],
        "link_rate_rmse": link["rmse"],
        "task_rmse": task["rmse"],
        "node_rmse": node["rmse"],
    }


def resource_load(arrays: dict[str, np.ndarray], indices: np.ndarray) -> np.ndarray:
    action_names = [str(item) for item in arrays["edge_action_features"].tolist()]
    name_to_idx = {name: idx for idx, name in enumerate(action_names)}
    if "rb_total" not in name_to_idx or "cpu_total" not in name_to_idx:
        raise ValueError("edge_action_features must include rb_total and cpu_total")
    actions = arrays["edge_a_future"][indices]
    return actions[..., name_to_idx["rb_total"]].sum(axis=2) + actions[..., name_to_idx["cpu_total"]].sum(axis=2)


def slice_metrics(
    predictions: dict[str, np.ndarray],
    load: np.ndarray,
    quantiles: list[float] | None = None,
) -> list[dict]:
    quantiles = quantiles or [0.0, 0.5, 0.75, 0.9, 0.95, 1.0]
    q = np.quantile(load.reshape(-1), quantiles)
    specs = [("all", -np.inf, np.inf, True)]
    for label, lo, hi in [
        ("p00_p50", q[0], q[1]),
        ("p50_p75", q[1], q[2]),
        ("p75_p90", q[2], q[3]),
        ("p90_p95", q[3], q[4]),
    ]:
        if hi > lo:
            specs.append((label, lo, hi, False))
    specs.append(("p95_p100", q[4], np.inf, True))
    pred = np.squeeze(predictions["link_rate_pred"], axis=-1)
    true = np.squeeze(predictions["link_rate_true"], axis=-1)
    active = np.squeeze(predictions["link_activity_true"], axis=-1) > 0.5
    prob = np.squeeze(predictions["link_activity_prob"], axis=-1)
    rows: list[dict] = []
    for label, lo, hi, include_hi in specs:
        if include_hi:
            step_mask = (load >= lo) & (load <= hi)
        else:
            step_mask = (load >= lo) & (load < hi)
        if not step_mask.any():
            continue
        edge_mask = step_mask[:, :, None]
        active_edge_mask = edge_mask & active
        link_resid = pred[edge_mask.repeat(pred.shape[2], axis=2)] - true[edge_mask.repeat(true.shape[2], axis=2)]
        active_resid = pred[active_edge_mask] - true[active_edge_mask]
        rows.append(
            {
                "slice": label,
                "step_count": int(step_mask.sum()),
                "active_edge_count": int(active_edge_mask.sum()),
                "load_mean": float(load[step_mask].mean()),
                "link_rate_rmse": float(np.sqrt(np.mean(link_resid**2))) if link_resid.size else None,
                "active_rate_rmse": float(np.sqrt(np.mean(active_resid**2))) if active_resid.size else None,
                "activity_prob_mean": float(prob[edge_mask.repeat(prob.shape[2], axis=2)].mean()),
                "true_active_rate": float(active[edge_mask.repeat(active.shape[2], axis=2)].mean()),
            }
        )
    return rows


def sensitivity_rows(
    normal: dict[str, np.ndarray],
    ablated: dict[str, np.ndarray],
    load: np.ndarray,
    ablated_mode: str,
) -> list[dict]:
    prob_delta = np.abs(normal["link_activity_prob"] - ablated["link_activity_prob"])
    rate_delta = np.abs(normal["link_rate_pred"] - ablated["link_rate_pred"])
    active = normal["link_activity_true"] > 0.5
    load_edge = load[:, :, None, None]
    rows = [
        {
            "ablated_mode": ablated_mode,
            "scope": "all_edges",
            "mean_abs_activity_prob_delta": float(prob_delta.mean()),
            "mean_abs_link_rate_delta": float(rate_delta.mean()),
            "pearson_load_rate_delta": finite_float(pearson_or_none(load_edge.reshape(-1), rate_delta.mean(axis=2).reshape(-1))),
        },
        {
            "ablated_mode": ablated_mode,
            "scope": "true_active_edges",
            "mean_abs_activity_prob_delta": float(prob_delta[active].mean()) if active.any() else None,
            "mean_abs_link_rate_delta": float(rate_delta[active].mean()) if active.any() else None,
            "pearson_load_rate_delta": None,
        },
    ]
    return rows


def collect_predictions_for_mode(model, base_dataset, stats, mode: str, device, batch_size: int, config: dict):
    dataset = ActionAblationDataset(base_dataset, stats, mode)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_v6_world_model_batch)
    return collect_v8_predictions(
        model,
        loader,
        device,
        stats,
        rate_output_mode=config.get("rate_output_mode", "main"),
        inactive_rate_value=float(config.get("inactive_rate_value", 0.0)),
        hurdle_gate_temperature=float(config.get("eval_hurdle_gate_temperature", 1.0)),
        hurdle_gate_power=float(config.get("eval_hurdle_gate_power", 1.0)),
    )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    experiment_dir = args.experiment_dir if args.experiment_dir.is_absolute() else PROJECT_ROOT / args.experiment_dir
    summary = json.loads((experiment_dir / "v8_full_training_summary.json").read_text(encoding="utf-8"))
    dataset_dir = resolve_project_path(summary["dataset_dir"])
    arrays = load_world_model_arrays(dataset_dir)
    if summary["config"].get("use_event_memory_features", False):
        arrays = add_event_memory_features(arrays)
    split = summary["split_seed_spec"]
    train_idx, val_idx, test_idx, _ = resolve_seed_splits(
        arrays["sample_seed"],
        train_seeds=split["train_seeds"],
        val_seeds=split["val_seeds"],
        test_seeds=split["test_seeds"],
    )
    if args.max_samples > 0:
        val_idx = val_idx[: args.max_samples]
        test_idx = test_idx[: args.max_samples]
    config = summary["config"]
    stats = make_normalization_stats(arrays, train_idx, rate_target_transform=config.get("rate_target_transform", "raw"))
    val_base = V6WorldModelDataset(
        arrays,
        val_idx,
        stats,
        rate_target_transform=config.get("rate_target_transform", "raw"),
        future_action_mode=config.get("future_action_mode", "full"),
    )
    test_base = V6WorldModelDataset(
        arrays,
        test_idx,
        stats,
        rate_target_transform=config.get("rate_target_transform", "raw"),
        future_action_mode=config.get("future_action_mode", "full"),
    )
    device = choose_device(args.device)
    checkpoint_path = args.checkpoint_path or experiment_dir / "checkpoints" / "v8_dual_best.pt"
    model = load_model_for_experiment(summary, arrays, checkpoint_path, device)

    overall_rows: list[dict] = []
    predictions_by_mode: dict[str, dict[str, np.ndarray]] = {}
    for mode in parse_modes(args.modes):
        val_pred = collect_predictions_for_mode(model, val_base, stats, mode, device, args.batch_size, config)
        threshold = choose_activity_threshold(val_pred["link_activity_prob"], val_pred["link_activity_true"])
        overall_rows.append(overall_metric_row(mode, "val", val_pred, threshold))
        test_pred = collect_predictions_for_mode(model, test_base, stats, mode, device, args.batch_size, config)
        predictions_by_mode[mode] = test_pred
        overall_rows.append(overall_metric_row(mode, "test", test_pred, threshold))

    load = resource_load(arrays, test_idx)
    slice_rows: list[dict] = []
    if "normal_actions" in predictions_by_mode:
        for row in slice_metrics(predictions_by_mode["normal_actions"], load):
            row["mode"] = "normal_actions"
            row["split"] = "test"
            slice_rows.append(row)

    sens_rows: list[dict] = []
    normal = predictions_by_mode.get("normal_actions")
    if normal is not None:
        for mode, pred in predictions_by_mode.items():
            if mode != "normal_actions":
                sens_rows.extend(sensitivity_rows(normal, pred, load, mode))

    outputs = {
        "overall_metrics_csv": output_dir / "world_model_trust_overall_metrics.csv",
        "high_load_error_slices_csv": output_dir / "world_model_trust_high_load_slices.csv",
        "action_sensitivity_csv": output_dir / "world_model_trust_action_sensitivity.csv",
        "summary_json": output_dir / "world_model_trust_summary.json",
    }
    write_csv(outputs["overall_metrics_csv"], overall_rows)
    write_csv(outputs["high_load_error_slices_csv"], slice_rows)
    write_csv(outputs["action_sensitivity_csv"], sens_rows)
    out_summary = {
        "module": "diagnose_v11_world_model_trust_phase2",
        "experiment_dir": str(experiment_dir),
        "dataset_dir": str(dataset_dir),
        "checkpoint_path": str(checkpoint_path),
        "split_counts": {"train": int(train_idx.size), "val": int(val_idx.size), "test": int(test_idx.size)},
        "overall_metrics": overall_rows,
        "outputs": {name: str(path) for name, path in outputs.items()},
    }
    outputs["summary_json"].write_text(json.dumps(out_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
