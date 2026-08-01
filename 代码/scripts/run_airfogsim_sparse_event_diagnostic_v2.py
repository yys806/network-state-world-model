from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import DataLoader


CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pi_jwm.airfogsim_smoke_model_v2 import MinimalDualGraphWorldModel, dual_graph_world_model_loss
from pi_jwm.airfogsim_sparse_diagnostics_v2 import (
    SparseDiagnosticAccumulator,
    build_last_persistence_prediction,
    build_zero_activity_prediction,
)
from pi_jwm.airfogsim_window_dataset_v2 import (
    AirFogSimTensorWindowDataset,
    fit_sparse_label_stats,
    fit_training_stats,
)


DEFAULT_DATASET = CODE_ROOT / "artifacts" / "datasets" / "airfogsim_tensor_v2_dev"
DEFAULT_OUTPUT = CODE_ROOT / "artifacts" / "small_experiments" / "exp08_airfogsim_sparse_event_diagnostic_v2"
ARM_NAMES = ("zero_activity", "last_persistence", "learned_unweighted", "learned_balanced")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_state_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _train_model(
    *,
    dataset_dir: Path,
    contract: Mapping[str, Any],
    stats: Mapping[str, Any],
    sparse_pos_weights: Mapping[str, float],
    epochs: int,
    batch_size: int,
    hidden_dim: int,
    learning_rate: float,
    random_seed: int,
) -> tuple[MinimalDualGraphWorldModel, dict[str, Any]]:
    _seed_everything(random_seed)
    dataset = AirFogSimTensorWindowDataset(
        dataset_dir, split="dev_train", stats=stats, normalize=True
    )
    if len(dataset) == 0:
        raise ValueError("dev_train split is empty")
    generator = torch.Generator().manual_seed(random_seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
    model = MinimalDualGraphWorldModel(
        hidden_dim=hidden_dim,
        horizon_steps=int(contract["horizon_steps"]),
    )
    initialization_sha256 = _model_state_hash(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    epochs_report: list[dict[str, Any]] = []
    sample_order: list[list[str]] = []
    for epoch in range(epochs):
        model.train()
        loss_sum = 0.0
        seen = 0
        epoch_order: list[str] = []
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            output = model(batch["history"])
            loss, _ = dual_graph_world_model_loss(
                output,
                batch["target"],
                batch["static"],
                sparse_pos_weights=sparse_pos_weights,
            )
            if not torch.isfinite(loss):
                raise ValueError("non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            current_batch = int(batch["seed"].shape[0])
            loss_sum += float(loss.detach()) * current_batch
            seen += current_batch
            epoch_order.extend(str(value) for value in batch["sample_id"])
        epochs_report.append(
            {"epoch": epoch + 1, "train_loss": loss_sum / seen, "sample_count": seen}
        )
        sample_order.append(epoch_order)
    return model, {
        "initialization_sha256": initialization_sha256,
        "data_order_seed": random_seed,
        "sample_order": sample_order,
        "epochs": epochs_report,
        "sparse_pos_weights": dict(sparse_pos_weights),
    }


def _evaluate_arm(
    *,
    dataset_dir: Path,
    split: str,
    stats: Mapping[str, Any],
    batch_size: int,
    prediction_fn: Callable[[Mapping[str, Any]], Mapping[str, torch.Tensor]],
) -> dict[str, Any]:
    dataset = AirFogSimTensorWindowDataset(
        dataset_dir, split=split, stats=stats, normalize=True
    )
    if len(dataset) == 0:
        raise ValueError(f"evaluation split {split} is empty")
    accumulator = SparseDiagnosticAccumulator(stats)
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            prediction = prediction_fn(batch)
            accumulator.update(prediction, batch["target"], batch["static"])
    return accumulator.finalize()


def _metric_text(value: Any) -> str:
    return "N/A" if value is None else f"{float(value):.4f}"


def _write_report(path: Path, evaluation: Mapping[str, Any], summary: Mapping[str, Any]) -> None:
    lines = [
        "# PI-JWM AirFogSim sparse-event diagnostic v2",
        "",
        f"- Diagnostic ready: `{str(summary['diagnostic_ready']).lower()}`",
        "- Scope: development-only four-arm diagnostic; no locked test set is used.",
        "- JEPA status: held until the base rollout model reliably exceeds both simple baselines.",
        "",
        "## Results",
        "",
        "| Arm | Split | Link F1 | Link AUPRC | Active rate MAE | Flow F1 | Task F1 | Lifecycle macro-F1 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, split_reports in evaluation.items():
        for split, report in split_reports.items():
            lines.append(
                f"| {arm} | {split} | {_metric_text(report['link_activity']['f1'])} | "
                f"{_metric_text(report['link_activity']['auprc'])} | "
                f"{_metric_text(report['active_only_rate']['mae'])} | "
                f"{_metric_text(report['presence']['flow']['f1'])} | "
                f"{_metric_text(report['presence']['task']['f1'])} | "
                f"{_metric_text(report['task_lifecycle']['macro_f1'])} |"
            )
    lines.extend(["", "Full per-feature physical-unit MAE and class support are in `evaluation.json`."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_diagnostic(
    *,
    dataset_dir: str | Path,
    output_dir: str | Path,
    epochs: int = 5,
    batch_size: int = 8,
    hidden_dim: int = 32,
    learning_rate: float = 1e-3,
    random_seed: int = 2026,
    eval_splits: Iterable[str] = ("dev_validation", "dev_calibration"),
) -> dict[str, Any]:
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    eval_splits = tuple(str(value) for value in eval_splits)
    contract = _read_json(dataset_dir / "tensor_contract.json")
    stats = fit_training_stats(dataset_dir, split="dev_train")
    class_stats = fit_sparse_label_stats(dataset_dir, split="dev_train", max_pos_weight=50.0)
    lifecycle_majority = int(class_stats["task_lifecycle"]["majority_index"])
    if lifecycle_majority < 0:
        lifecycle_majority = 0

    balanced_weights = {
        label: float(class_stats["labels"][label]["pos_weight"])
        for label in ("link_activity", "flow_present", "task_present")
    }
    learned_models: dict[str, MinimalDualGraphWorldModel] = {}
    training_history: dict[str, Any] = {}
    for arm, weights in (
        ("learned_unweighted", {label: 1.0 for label in balanced_weights}),
        ("learned_balanced", balanced_weights),
    ):
        model, history = _train_model(
            dataset_dir=dataset_dir,
            contract=contract,
            stats=stats,
            sparse_pos_weights=weights,
            epochs=int(epochs),
            batch_size=int(batch_size),
            hidden_dim=int(hidden_dim),
            learning_rate=float(learning_rate),
            random_seed=int(random_seed),
        )
        learned_models[arm] = model
        training_history[arm] = history

    if (
        training_history["learned_unweighted"]["initialization_sha256"]
        != training_history["learned_balanced"]["initialization_sha256"]
        or training_history["learned_unweighted"]["sample_order"]
        != training_history["learned_balanced"]["sample_order"]
    ):
        raise RuntimeError("learned-arm comparison is not deterministic")

    evaluation: dict[str, dict[str, Any]] = {arm: {} for arm in ARM_NAMES}
    for split in eval_splits:
        evaluation["zero_activity"][split] = _evaluate_arm(
            dataset_dir=dataset_dir,
            split=split,
            stats=stats,
            batch_size=int(batch_size),
            prediction_fn=lambda batch: build_zero_activity_prediction(
                batch, stats, lifecycle_majority_index=lifecycle_majority
            ),
        )
        evaluation["last_persistence"][split] = _evaluate_arm(
            dataset_dir=dataset_dir,
            split=split,
            stats=stats,
            batch_size=int(batch_size),
            prediction_fn=lambda batch: build_last_persistence_prediction(
                batch, horizon_steps=int(contract["horizon_steps"])
            ),
        )
        for arm, model in learned_models.items():
            model.eval()
            evaluation[arm][split] = _evaluate_arm(
                dataset_dir=dataset_dir,
                split=split,
                stats=stats,
                batch_size=int(batch_size),
                prediction_fn=lambda batch, current_model=model: current_model(batch["history"]),
            )

    config = {
        "dataset_dir": str(dataset_dir.resolve()),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "hidden_dim": int(hidden_dim),
        "learning_rate": float(learning_rate),
        "random_seed": int(random_seed),
        "eval_splits": list(eval_splits),
        "threshold": 0.5,
    }
    all_expected = all(split in evaluation[arm] for arm in ARM_NAMES for split in eval_splits)
    finite_training = all(
        math.isfinite(float(row["train_loss"]))
        for history in training_history.values()
        for row in history["epochs"]
    )
    summary = {
        "schema_version": "PI-JWM-AirFogSim-sparse-event-diagnostic-v2",
        "diagnostic_ready": bool(all_expected and finite_training),
        "formal_training_ready": False,
        "jepa_comparison_ready": False,
        "arms": {arm: {"evaluation_splits": list(evaluation[arm])} for arm in ARM_NAMES},
        "train_sample_count": int(class_stats["sample_count"]),
        "comparison_integrity": {
            "same_initialization": True,
            "same_sample_order": True,
            "class_stats_source_split": class_stats["source_split"],
        },
        "config": config,
    }

    _write_json(output_dir / "config.json", config)
    _write_json(output_dir / "normalization_stats.json", stats)
    _write_json(output_dir / "class_stats.json", class_stats)
    _write_json(output_dir / "training_history.json", training_history)
    _write_json(output_dir / "evaluation.json", evaluation)
    _write_json(output_dir / "summary.json", summary)
    for arm, model in learned_models.items():
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "contract": contract,
                "normalization_stats": stats,
                "class_stats": class_stats,
                "config": config,
            },
            output_dir / f"{arm}_model.pt",
        )
    _write_report(output_dir / "REPORT.md", evaluation, summary)
    evidence = sorted(path for path in output_dir.iterdir() if path.is_file() and path.name != "manifest.json")
    manifest = {
        "schema_version": "PI-JWM-AirFogSim-sparse-event-diagnostic-manifest-v2",
        "files": {
            path.name: {"sha256": _sha256(path), "size_bytes": path.stat().st_size}
            for path in evidence
        },
    }
    _write_json(output_dir / "manifest.json", manifest)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the PI-JWM AirFogSim sparse-event four-arm diagnostic.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--random-seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_diagnostic(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        random_seed=args.random_seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
