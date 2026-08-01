"""Run the formal PI-JWM dual-graph CPU interface and comparison smoke."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pi_jwm.airfogsim_window_dataset_v2 import fit_training_stats
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
    compute_training_class_weights,
    formal_world_model_loss,
)
from pi_jwm.formal_world_model_metrics_v1 import FormalMetricAccumulator


CPU_METHODS = (
    "zero_activity",
    "last_persistence",
    "pooled_gru",
    "independent_dual_gnn",
    "coupled_dual_gnn",
)
LEARNED_METHODS = CPU_METHODS[2:]
CPU_SPLITS = ("train", "validation", "calibration")


def validate_cpu_protocol(splits: Iterable[str]) -> None:
    requested = tuple(str(split) for split in splits)
    if "locked_test" in requested:
        raise ValueError("locked_test cannot be used by the CPU development protocol")
    unknown = set(requested) - set(CPU_SPLITS)
    if unknown:
        raise ValueError(f"unsupported CPU splits: {sorted(unknown)}")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _model_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _subset_for_ids(dataset: FormalAirFogSimWindowDataset, sample_ids: list[str]) -> Subset:
    index_by_id = {str(row["sample_id"]): index for index, row in enumerate(dataset.rows)}
    missing = [sample_id for sample_id in sample_ids if sample_id not in index_by_id]
    if missing:
        raise ValueError(f"selected sample IDs are missing from the dataset: {missing[:3]}")
    return Subset(dataset, [index_by_id[sample_id] for sample_id in sample_ids])


def _prediction(
    method: str,
    model: FormalDualGraphWorldModel | None,
    batch: Mapping[str, Any],
    stats: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    if method in {"zero_activity", "last_persistence"}:
        return build_rule_prediction(method, batch, stats)
    if model is None:
        raise ValueError(f"learned method {method} requires a model")
    return model(batch)


def _edge_valid(static: Mapping[str, torch.Tensor]) -> torch.Tensor:
    return torch.all(static["physical_edge_endpoint_index"] >= 0, dim=-1)


def _choose_link_threshold(
    method: str,
    model: FormalDualGraphWorldModel | None,
    loader: DataLoader,
    stats: Mapping[str, Any],
) -> tuple[float, dict[str, Any], float]:
    scores: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    inference_seconds = 0.0
    if model is not None:
        model.eval()
    with torch.no_grad():
        for batch in loader:
            started = time.perf_counter()
            prediction = _prediction(method, model, batch, stats)
            inference_seconds += time.perf_counter() - started
            valid = _edge_valid(batch["static"])[:, None, :].expand_as(batch["target"]["link_activity"])
            scores.append(torch.sigmoid(prediction["link_activity_logits"])[valid].cpu().numpy())
            labels.append(batch["target"]["link_activity"][valid].bool().cpu().numpy())
    all_scores = np.concatenate(scores) if scores else np.empty((0,), dtype=np.float64)
    all_labels = np.concatenate(labels) if labels else np.empty((0,), dtype=bool)
    candidates = (0.1, 0.3, 0.5, 0.7, 0.9)
    rows = []
    for threshold in candidates:
        predicted = all_scores >= threshold
        tp = int(np.count_nonzero(predicted & all_labels))
        fp = int(np.count_nonzero(predicted & ~all_labels))
        fn = int(np.count_nonzero(~predicted & all_labels))
        denominator = 2 * tp + fp + fn
        f1 = float(2 * tp / denominator) if denominator else None
        rows.append({"threshold": threshold, "f1": f1, "tp": tp, "fp": fp, "fn": fn})
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
        for batch in loader:
            started = time.perf_counter()
            prediction = _prediction(method, model, batch, stats)
            inference_seconds += time.perf_counter() - started
            accumulator.update(prediction, batch["target"], batch["static"])
    return accumulator.finalize(), inference_seconds


def _metric_value(report: Mapping[str, Any], name: str) -> float | None:
    metric = report.get("horizons", {}).get("overall", {}).get("metrics", {}).get(name, {})
    return metric.get("value") if metric.get("status") == "computed" else None


def _load_or_fit_stats(tensor_root: Path) -> dict[str, Any]:
    path = tensor_root / "normalization_stats.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return fit_training_stats(tensor_root, split="train")


def _manifest(output_dir: Path) -> dict[str, Any]:
    files = {}
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path.name == "manifest.json":
            continue
        relative = path.relative_to(output_dir).as_posix()
        files[relative] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    return {"schema_version": "PI-JWM-formal-CPU-smoke-manifest-v1", "files": files}


def run_cpu_smoke(
    *,
    tensor_root: str | Path,
    output_dir: str | Path,
    mode: str = "interface_smoke",
    seed: int = 20260802,
    device: str = "cpu",
    train_limit: int | None = None,
    evaluation_limit: int | None = None,
    hidden_dim: int | None = None,
    epochs: int | None = None,
) -> dict[str, Any]:
    if mode not in {"interface_smoke", "comparison_smoke"}:
        raise ValueError("mode must be interface_smoke or comparison_smoke")
    if device != "cpu":
        raise ValueError("this runner is intentionally CPU-only")
    validate_cpu_protocol(CPU_SPLITS)
    tensor_root = Path(tensor_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics").mkdir(exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)
    defaults = {
        "interface_smoke": {"train_limit": 6, "evaluation_limit": 3, "hidden_dim": 16, "epochs": 1},
        "comparison_smoke": {"train_limit": 64, "evaluation_limit": 32, "hidden_dim": 32, "epochs": 2},
    }[mode]
    train_limit = int(train_limit if train_limit is not None else defaults["train_limit"])
    evaluation_limit = int(
        evaluation_limit if evaluation_limit is not None else defaults["evaluation_limit"]
    )
    hidden_dim = int(hidden_dim if hidden_dim is not None else defaults["hidden_dim"])
    epochs = int(epochs if epochs is not None else defaults["epochs"])
    if min(train_limit, evaluation_limit, hidden_dim, epochs) <= 0:
        raise ValueError("limits, hidden_dim, and epochs must be positive")

    _seed_everything(seed)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
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
        for split in CPU_SPLITS
    }
    limits = {"train": train_limit, "validation": evaluation_limit, "calibration": evaluation_limit}
    sample_ids = {
        split: select_stratified_window_ids(dataset.rows, min(limits[split], len(dataset)), seed + index)
        for index, (split, dataset) in enumerate(datasets.items())
    }
    subsets = {split: _subset_for_ids(datasets[split], sample_ids[split]) for split in CPU_SPLITS}
    loaders = {
        split: DataLoader(subset, batch_size=1, shuffle=False, num_workers=0)
        for split, subset in subsets.items()
    }
    class_weight_report = compute_training_class_weights(subsets["train"])
    class_weights = class_weight_report["pos_weight"]

    source_manifest = tensor_root / "manifest.json"
    dataset_hash_source = source_manifest if source_manifest.is_file() else tensor_root / "tensor_contract.json"
    config = {
        "schema_version": "PI-JWM-formal-CPU-smoke-config-v1",
        "mode": mode,
        "seed": seed,
        "device": device,
        "history_steps": window_config.history_steps,
        "horizon_steps": window_config.horizon_steps,
        "hidden_dim": hidden_dim,
        "epochs": epochs,
        "train_limit": train_limit,
        "evaluation_limit": evaluation_limit,
        "batch_size": 1,
        "learning_rate": 1e-3,
        "splits": list(CPU_SPLITS),
        "locked_test_accessed": False,
        "tensor_root": str(tensor_root.resolve()),
        "dataset_manifest_sha256": _sha256(dataset_hash_source),
    }
    registry = method_registry()
    _write_json(output_dir / "config.json", config)
    _write_json(output_dir / "method_registry.json", registry)
    _write_json(output_dir / "sample_ids.json", sample_ids)
    _write_json(output_dir / "class_weights.json", class_weight_report)

    models: dict[str, FormalDualGraphWorldModel] = {}
    training_history: dict[str, list[dict[str, Any]]] = {method: [] for method in LEARNED_METHODS}
    runtime: dict[str, dict[str, Any]] = {}
    for method in LEARNED_METHODS:
        _seed_everything(seed)
        model = FormalDualGraphWorldModel(
            FormalWorldModelConfig(
                mode=method,
                hidden_dim=hidden_dim,
                history_steps=window_config.history_steps,
                horizon_steps=window_config.horizon_steps,
            )
        )
        initialization_hash = _model_hash(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        tracemalloc.start()
        started = time.perf_counter()
        model.train()
        for epoch in range(epochs):
            losses = []
            for batch in loaders["train"]:
                optimizer.zero_grad(set_to_none=True)
                prediction = model(batch)
                loss, components = formal_world_model_loss(
                    prediction,
                    batch["target"],
                    batch["static"],
                    class_weights=class_weights,
                )
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite loss for {method}")
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))
            training_history[method].append(
                {
                    "epoch": epoch + 1,
                    "mean_loss": float(np.mean(losses)) if losses else None,
                    "batch_count": len(losses),
                }
            )
        train_seconds = time.perf_counter() - started
        _, peak_python_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        checkpoint = {
            "method": method,
            "config": config,
            "model_state_dict": model.state_dict(),
            "initialization_hash": initialization_hash,
        }
        torch.save(checkpoint, output_dir / "checkpoints" / f"{method}.pt")
        models[method] = model
        runtime[method] = {
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "train_seconds": train_seconds,
            "python_tracemalloc_peak_bytes": int(peak_python_bytes),
            "initialization_hash": initialization_hash,
        }

    metric_reports: dict[str, dict[str, Any]] = {}
    comparison_rows = []
    for method in CPU_METHODS:
        model = models.get(method)
        threshold, threshold_report, threshold_seconds = _choose_link_threshold(
            method, model, loaders["validation"], stats
        )
        distribution_available = bool(registry[method]["distribution_output"])
        validation_report, validation_seconds = _evaluate(
            method,
            model,
            loaders["validation"],
            stats,
            threshold,
            distribution_available,
        )
        calibration_report, calibration_seconds = _evaluate(
            method,
            model,
            loaders["calibration"],
            stats,
            threshold,
            distribution_available,
        )
        metric_reports[method] = {
            "threshold_selection": threshold_report,
            "validation": validation_report,
            "calibration": calibration_report,
        }
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
                "parameter_count": runtime[method].get("parameter_count", 0),
                "train_seconds": runtime[method].get("train_seconds", 0.0),
                "validation_inference_seconds": validation_seconds,
            }
        )

    _write_json(output_dir / "training_history.json", training_history)
    _write_json(output_dir / "runtime.json", runtime)
    with (output_dir / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison_rows[0]))
        writer.writeheader()
        writer.writerows(comparison_rows)
    summary = {
        "schema_version": "PI-JWM-formal-CPU-smoke-summary-v1",
        "cpu_smoke_ready": True,
        "formal_performance_claim_ready": False,
        "locked_test_accessed": False,
        "completed_cpu_methods": list(CPU_METHODS),
        "paper_baseline_status": registry["coupled_jepa_bou_chaaya_2026"]["stage"],
        "sample_counts": {split: len(values) for split, values in sample_ids.items()},
        "result_boundary": "CPU smoke verifies interfaces and metric computation, not converged performance.",
    }
    _write_json(output_dir / "run_summary.json", summary)
    _write_json(output_dir / "manifest.json", _manifest(output_dir))
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("interface_smoke", "comparison_smoke"), default="interface_smoke")
    parser.add_argument("--tensor-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--evaluation-limit", type=int)
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--epochs", type=int)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = run_cpu_smoke(
        tensor_root=args.tensor_root,
        output_dir=args.output_dir,
        mode=args.mode,
        seed=args.seed,
        device=args.device,
        train_limit=args.train_limit,
        evaluation_limit=args.evaluation_limit,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
