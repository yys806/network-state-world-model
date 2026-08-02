"""Train and evaluate formal PI-JWM dual-graph models on a CUDA device."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pi_jwm.formal_airfogsim_window_v1 import (
    FormalAirFogSimWindowDataset,
    FormalWindowConfig,
    select_stratified_window_ids,
)
from pi_jwm.formal_dual_graph_world_model_v1 import (
    FormalDualGraphWorldModel,
    FormalWorldModelConfig,
)
from pi_jwm.formal_world_model_baselines_v1 import build_rule_prediction, method_registry
from pi_jwm.formal_world_model_loss_v1 import (
    FormalLossWeights,
    compute_training_class_weights,
    formal_world_model_loss,
)
from pi_jwm.formal_world_model_metrics_v1 import FormalMetricAccumulator
from run_formal_dual_graph_cpu_smoke_v1 import (
    _load_or_fit_stats,
    _manifest,
    _metric_value,
    _model_hash,
    _seed_everything,
    _sha256,
    _subset_for_ids,
    _write_json,
)


NONLOCKED_SPLITS = ("train", "validation", "calibration")
RULE_METHODS = ("zero_activity", "last_persistence")
MODEL_SPECS = {
    "pooled_gru": ("pooled_gru", False),
    "independent_dual_gnn": ("independent_dual_gnn", False),
    "coupled_dual_gnn": ("coupled_dual_gnn", False),
    "independent_dual_gnn_residual": ("independent_dual_gnn", True),
    "coupled_dual_gnn_residual": ("coupled_dual_gnn", True),
}
LEARNED_METHODS = tuple(MODEL_SPECS)


def _training_manifest(output_dir: Path) -> dict[str, Any]:
    manifest = _manifest(output_dir)
    manifest["schema_version"] = "PI-JWM-formal-training-manifest-v1"
    return manifest


def validate_gpu_protocol(splits: Iterable[str], device: str) -> None:
    requested = tuple(str(split) for split in splits)
    if "locked_test" in requested:
        raise ValueError("locked_test cannot be used by the GPU training protocol")
    unknown = set(requested) - set(NONLOCKED_SPLITS)
    if unknown:
        raise ValueError(f"unsupported GPU splits: {sorted(unknown)}")
    if not str(device).startswith("cuda"):
        raise ValueError("the formal GPU entry point requires a CUDA device")


def move_nested_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=device.type == "cuda")
    if isinstance(value, dict):
        return {key: move_nested_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_nested_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_nested_to_device(item, device) for item in value)
    return value


def _edge_valid(static: Mapping[str, torch.Tensor]) -> torch.Tensor:
    return torch.all(static["physical_edge_endpoint_index"] >= 0, dim=-1)


def _prediction(
    method: str,
    model: FormalDualGraphWorldModel | None,
    batch: Mapping[str, Any],
    stats: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    if method in RULE_METHODS:
        return build_rule_prediction(method, batch, stats)
    if model is None:
        raise ValueError(f"learned method {method} requires a model")
    return model(batch)


def _choose_link_threshold(
    method: str,
    model: FormalDualGraphWorldModel | None,
    loader: DataLoader,
    stats: Mapping[str, Any],
    device: torch.device,
) -> tuple[float, dict[str, Any], float]:
    scores: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    inference_seconds = 0.0
    if model is not None:
        model.eval()
    with torch.no_grad():
        for cpu_batch in loader:
            batch = move_nested_to_device(cpu_batch, device)
            started = time.perf_counter()
            prediction = _prediction(method, model, batch, stats)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_seconds += time.perf_counter() - started
            valid = _edge_valid(batch["static"])[:, None, :].expand_as(batch["target"]["link_activity"])
            scores.append(torch.sigmoid(prediction["link_activity_logits"])[valid].detach().cpu().numpy())
            labels.append(batch["target"]["link_activity"][valid].bool().detach().cpu().numpy())
    all_scores = np.concatenate(scores) if scores else np.empty((0,), dtype=np.float64)
    all_labels = np.concatenate(labels) if labels else np.empty((0,), dtype=bool)
    rows = []
    for threshold in (0.1, 0.3, 0.5, 0.7, 0.9):
        predicted = all_scores >= threshold
        tp = int(np.count_nonzero(predicted & all_labels))
        fp = int(np.count_nonzero(predicted & ~all_labels))
        fn = int(np.count_nonzero(~predicted & all_labels))
        denominator = 2 * tp + fp + fn
        rows.append(
            {
                "threshold": threshold,
                "f1": float(2 * tp / denominator) if denominator else None,
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
        )
    comparable = [row for row in rows if row["f1"] is not None]
    selected = max(comparable, key=lambda row: (row["f1"], -abs(row["threshold"] - 0.5))) if comparable else rows[2]
    return float(selected["threshold"]), {"candidates": rows, "selected": selected}, inference_seconds


def _evaluate(
    method: str,
    model: FormalDualGraphWorldModel | None,
    loader: DataLoader,
    stats: Mapping[str, Any],
    threshold: float,
    distribution_available: bool,
    device: torch.device,
) -> tuple[dict[str, Any], float]:
    accumulator = FormalMetricAccumulator(
        stats,
        threshold=threshold,
        distribution_available=distribution_available,
    )
    inference_seconds = 0.0
    if model is not None:
        model.eval()
    with torch.no_grad():
        for cpu_batch in loader:
            batch = move_nested_to_device(cpu_batch, device)
            started = time.perf_counter()
            prediction = _prediction(method, model, batch, stats)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_seconds += time.perf_counter() - started
            accumulator.update(prediction, batch["target"], batch["static"])
    return accumulator.finalize(), inference_seconds


def _mean_loss(
    model: FormalDualGraphWorldModel,
    loader: DataLoader,
    class_weights: Mapping[str, float],
    loss_weights: FormalLossWeights,
    device: torch.device,
) -> float | None:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for cpu_batch in loader:
            batch = move_nested_to_device(cpu_batch, device)
            prediction = model(batch)
            loss, _ = formal_world_model_loss(
                prediction,
                batch["target"],
                batch["static"],
                weights=loss_weights,
                class_weights=class_weights,
            )
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite validation loss")
            losses.append(float(loss.detach()))
    return float(np.mean(losses)) if losses else None


def _select_ids(
    dataset: FormalAirFogSimWindowDataset,
    limit: int | None,
    seed: int,
) -> list[str]:
    actual_limit = len(dataset) if limit is None else min(int(limit), len(dataset))
    if actual_limit < 0:
        raise ValueError("sample limits cannot be negative")
    return select_stratified_window_ids(dataset.rows, actual_limit, seed)


def run_formal_training(
    *,
    tensor_root: str | Path,
    output_dir: str | Path,
    device: str,
    learned_methods: Sequence[str] = LEARNED_METHODS,
    seed: int = 20260802,
    train_limit: int | None = None,
    evaluation_limit: int | None = None,
    hidden_dim: int = 64,
    epochs: int = 20,
    batch_size: int = 2,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-5,
    num_workers: int = 0,
) -> dict[str, Any]:
    unknown_methods = set(learned_methods) - set(MODEL_SPECS)
    if unknown_methods:
        raise ValueError(f"unsupported learned methods: {sorted(unknown_methods)}")
    if not learned_methods:
        raise ValueError("at least one learned method is required")
    if min(hidden_dim, epochs, batch_size) <= 0 or learning_rate <= 0 or weight_decay < 0:
        raise ValueError("training hyperparameters are invalid")
    requested_device = torch.device(device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    tensor_root = Path(tensor_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics").mkdir(exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)
    _seed_everything(seed)
    if requested_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    contract = json.loads((tensor_root / "tensor_contract.json").read_text(encoding="utf-8"))
    window_config = FormalWindowConfig(
        history_steps=int(contract["history_steps"]),
        horizon_steps=int(contract["horizon_steps"]),
    )
    stats = _load_or_fit_stats(tensor_root)
    datasets = {
        split: FormalAirFogSimWindowDataset(
            tensor_root,
            split=split,
            config=window_config,
            stats=stats,
            normalize=True,
        )
        for split in NONLOCKED_SPLITS
    }
    sample_ids = {
        "train": _select_ids(datasets["train"], train_limit, seed),
        "validation": _select_ids(datasets["validation"], evaluation_limit, seed + 1),
        "calibration": _select_ids(datasets["calibration"], evaluation_limit, seed + 2),
    }
    subsets = {split: _subset_for_ids(datasets[split], sample_ids[split]) for split in NONLOCKED_SPLITS}
    loaders = {
        split: DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=requested_device.type == "cuda",
        )
        for split, subset in subsets.items()
    }
    class_weight_report = compute_training_class_weights(subsets["train"])
    class_weights = class_weight_report["pos_weight"]
    loss_weights = FormalLossWeights()
    dataset_hash_source = tensor_root / "manifest.json"
    if not dataset_hash_source.is_file():
        dataset_hash_source = tensor_root / "tensor_contract.json"
    config = {
        "schema_version": "PI-JWM-formal-training-config-v1",
        "seed": seed,
        "device": str(requested_device),
        "history_steps": window_config.history_steps,
        "horizon_steps": window_config.horizon_steps,
        "hidden_dim": hidden_dim,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "num_workers": num_workers,
        "train_limit": train_limit,
        "evaluation_limit": evaluation_limit,
        "splits": list(NONLOCKED_SPLITS),
        "learned_methods": list(learned_methods),
        "loss_weights": asdict(loss_weights),
        "locked_test_accessed": False,
        "tensor_root": str(tensor_root.resolve()),
        "dataset_manifest_sha256": _sha256(dataset_hash_source),
    }
    registry = method_registry()
    _write_json(output_dir / "config.json", config)
    _write_json(output_dir / "method_registry.json", registry)
    _write_json(output_dir / "sample_ids.json", sample_ids)
    _write_json(output_dir / "class_weights.json", class_weight_report)

    histories: dict[str, list[dict[str, Any]]] = {}
    runtime: dict[str, dict[str, Any]] = {}
    models: dict[str, FormalDualGraphWorldModel] = {}
    for method in learned_methods:
        _seed_everything(seed)
        base_mode, residual_state_prediction = MODEL_SPECS[method]
        model_config = FormalWorldModelConfig(
            mode=base_mode,
            hidden_dim=hidden_dim,
            history_steps=window_config.history_steps,
            horizon_steps=window_config.horizon_steps,
            residual_state_prediction=residual_state_prediction,
        )
        model = FormalDualGraphWorldModel(model_config)
        initialization_hash = _model_hash(model)
        model.to(requested_device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        if requested_device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(requested_device)
        started = time.perf_counter()
        best_epoch = 0
        best_validation_loss = float("inf")
        best_state: dict[str, torch.Tensor] | None = None
        method_history: list[dict[str, Any]] = []
        for epoch in range(1, epochs + 1):
            model.train()
            train_losses: list[float] = []
            for cpu_batch in loaders["train"]:
                batch = move_nested_to_device(cpu_batch, requested_device)
                optimizer.zero_grad(set_to_none=True)
                prediction = model(batch)
                loss, _ = formal_world_model_loss(
                    prediction,
                    batch["target"],
                    batch["static"],
                    weights=loss_weights,
                    class_weights=class_weights,
                )
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite loss for {method} at epoch {epoch}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                train_losses.append(float(loss.detach()))
            mean_train_loss = float(np.mean(train_losses)) if train_losses else None
            validation_loss = _mean_loss(
                model, loaders["validation"], class_weights, loss_weights, requested_device
            )
            selection_loss = validation_loss if validation_loss is not None else mean_train_loss
            if selection_loss is None:
                raise RuntimeError("training and validation loaders are both empty")
            method_history.append(
                {
                    "epoch": epoch,
                    "mean_train_loss": mean_train_loss,
                    "mean_validation_loss": validation_loss,
                    "batch_count": len(train_losses),
                }
            )
            if selection_loss < best_validation_loss:
                best_epoch = epoch
                best_validation_loss = selection_loss
                best_state = copy.deepcopy(
                    {name: value.detach().cpu() for name, value in model.state_dict().items()}
                )
        if best_state is None:
            raise RuntimeError(f"no checkpoint was selected for {method}")
        checkpoint_path = output_dir / "checkpoints" / f"{method}__best.pt"
        torch.save(
            {
                "method": method,
                "model_config": model_config.__dict__,
                "run_config": config,
                "model_state_dict": best_state,
                "initialization_hash": initialization_hash,
                "best_epoch": best_epoch,
                "best_validation_loss": best_validation_loss,
            },
            checkpoint_path,
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        reloaded = FormalDualGraphWorldModel(model_config)
        reloaded.load_state_dict(checkpoint["model_state_dict"], strict=True)
        reloaded.to(requested_device)
        models[method] = reloaded
        histories[method] = method_history
        if requested_device.type == "cuda":
            torch.cuda.synchronize(requested_device)
            peak_device_bytes = int(torch.cuda.max_memory_allocated(requested_device))
        else:
            peak_device_bytes = 0
        runtime[method] = {
            "parameter_count": sum(parameter.numel() for parameter in reloaded.parameters()),
            "train_seconds": time.perf_counter() - started,
            "peak_device_memory_bytes": peak_device_bytes,
            "initialization_hash": initialization_hash,
            "best_epoch": best_epoch,
            "best_validation_loss": best_validation_loss,
            "checkpoint_reload_verified": True,
        }

    comparison_rows: list[dict[str, Any]] = []
    completed_methods = [*RULE_METHODS, *learned_methods]
    for method in completed_methods:
        model = models.get(method)
        threshold, threshold_report, threshold_seconds = _choose_link_threshold(
            method, model, loaders["validation"], stats, requested_device
        )
        distribution_available = bool(registry[method]["distribution_output"])
        validation_report, validation_seconds = _evaluate(
            method,
            model,
            loaders["validation"],
            stats,
            threshold,
            distribution_available,
            requested_device,
        )
        calibration_report, calibration_seconds = _evaluate(
            method,
            model,
            loaders["calibration"],
            stats,
            threshold,
            distribution_available,
            requested_device,
        )
        _write_json(output_dir / "metrics" / f"{method}__validation.json", validation_report)
        _write_json(output_dir / "metrics" / f"{method}__calibration.json", calibration_report)
        runtime.setdefault(method, {})
        runtime[method].update(
            {
                "threshold_selection_seconds": threshold_seconds,
                "validation_inference_seconds": validation_seconds,
                "calibration_inference_seconds": calibration_seconds,
            }
        )
        comparison_rows.append(
            {
                "method": method,
                "threshold": threshold,
                "validation_link_f1": _metric_value(validation_report, "event.link_activity.f1"),
                "calibration_link_f1": _metric_value(calibration_report, "event.link_activity.f1"),
                "validation_node_x_mae": _metric_value(validation_report, "state.node.x.mae"),
                "calibration_node_x_mae": _metric_value(calibration_report, "state.node.x.mae"),
                "validation_throughput_mae": _metric_value(validation_report, "system.communication_throughput.mae"),
                "calibration_throughput_mae": _metric_value(calibration_report, "system.communication_throughput.mae"),
                "validation_completion_rate_error": _metric_value(validation_report, "system.task_completion_rate.absolute_error"),
                "calibration_completion_rate_error": _metric_value(calibration_report, "system.task_completion_rate.absolute_error"),
                "validation_rb_occupancy_mae": _metric_value(validation_report, "resource.rb_occupancy.mae"),
                "calibration_rb_occupancy_mae": _metric_value(calibration_report, "resource.rb_occupancy.mae"),
                "validation_task_delay_mae": _metric_value(validation_report, "state.task.delay.mae"),
                "calibration_task_delay_mae": _metric_value(calibration_report, "state.task.delay.mae"),
                "validation_task_deadline_mae": _metric_value(validation_report, "state.task.deadline_remaining.mae"),
                "calibration_task_deadline_mae": _metric_value(calibration_report, "state.task.deadline_remaining.mae"),
                "validation_lifecycle_macro_f1": _metric_value(validation_report, "task.lifecycle.macro_f1"),
                "calibration_lifecycle_macro_f1": _metric_value(calibration_report, "task.lifecycle.macro_f1"),
                "validation_dag_unfinished_parent_mae": _metric_value(validation_report, "dag.unfinished_parent_count.mae"),
                "calibration_dag_unfinished_parent_mae": _metric_value(calibration_report, "dag.unfinished_parent_count.mae"),
                "best_epoch": runtime[method].get("best_epoch"),
                "best_validation_loss": runtime[method].get("best_validation_loss"),
                "parameter_count": runtime[method].get("parameter_count", 0),
                "train_seconds": runtime[method].get("train_seconds", 0.0),
                "peak_device_memory_bytes": runtime[method].get("peak_device_memory_bytes", 0),
            }
        )

    _write_json(output_dir / "training_history.json", histories)
    _write_json(output_dir / "runtime.json", runtime)
    with (output_dir / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison_rows[0]))
        writer.writeheader()
        writer.writerows(comparison_rows)
    summary = {
        "schema_version": "PI-JWM-formal-training-summary-v1",
        "training_run_complete": True,
        "gpu_execution": requested_device.type == "cuda",
        "formal_performance_claim_ready": False,
        "locked_test_accessed": False,
        "completed_methods": completed_methods,
        "sample_counts": {split: len(values) for split, values in sample_ids.items()},
        "result_boundary": "Nonlocked training evidence only; locked-test remains sealed.",
    }
    _write_json(output_dir / "run_summary.json", summary)
    _write_json(output_dir / "manifest.json", _training_manifest(output_dir))
    return summary


def run_gpu_training(**kwargs: Any) -> dict[str, Any]:
    device = str(kwargs.get("device", "cuda"))
    validate_gpu_protocol(NONLOCKED_SPLITS, device)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    kwargs["device"] = device
    return run_formal_training(**kwargs)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tensor-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--evaluation-limit", type=int)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--learned-methods",
        nargs="+",
        choices=tuple(MODEL_SPECS),
        default=list(LEARNED_METHODS),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = run_gpu_training(
        tensor_root=args.tensor_root,
        output_dir=args.output_dir,
        device=args.device,
        seed=args.seed,
        train_limit=args.train_limit,
        evaluation_limit=args.evaluation_limit,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        learned_methods=args.learned_methods,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
