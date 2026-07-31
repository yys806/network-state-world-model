from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader


CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pi_jwm.airfogsim_smoke_model_v2 import MinimalDualGraphWorldModel, dual_graph_world_model_loss
from pi_jwm.airfogsim_tensor_v2 import EDGE_FEATURES, FLOW_FEATURES, NODE_FEATURES, TASK_FEATURES
from pi_jwm.airfogsim_window_dataset_v2 import AirFogSimTensorWindowDataset


DEFAULT_DATASET = CODE_ROOT / "artifacts" / "datasets" / "airfogsim_tensor_v2_dev"
DEFAULT_OUTPUT = CODE_ROOT / "artifacts" / "small_experiments" / "exp07_airfogsim_tensor_smoke_v2"
FEATURE_NAMES = {
    "node": NODE_FEATURES,
    "physical_edge": EDGE_FEATURES,
    "flow": FLOW_FEATURES,
    "task": TASK_FEATURES,
}


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


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _valid_mask(name: str, static: Mapping[str, torch.Tensor]) -> torch.Tensor:
    if name == "node":
        return static["node_kind_index"] >= 0
    if name == "physical_edge":
        return torch.all(static["physical_edge_endpoint_index"] >= 0, dim=-1)
    if name == "flow":
        return static["flow_valid"].bool()
    if name == "task":
        return static["task_valid"].bool()
    raise KeyError(name)


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def _evaluate(
    model: MinimalDualGraphWorldModel,
    loader: DataLoader,
    stats: Mapping[str, Any],
) -> dict[str, Any]:
    model.eval()
    loss_sum = 0.0
    sample_count = 0
    error_sums = {name: np.zeros((len(features),), dtype=np.float64) for name, features in FEATURE_NAMES.items()}
    error_counts = {name: 0 for name in FEATURE_NAMES}
    presence_counts = {name: {"tp": 0, "fp": 0, "fn": 0, "valid": 0} for name in FEATURE_NAMES}
    link_activity_counts = {"tp": 0, "fp": 0, "fn": 0, "valid": 0}
    active_rate_absolute_error = 0.0
    active_rate_squared_error = 0.0
    active_rate_count = 0
    with torch.no_grad():
        for batch in loader:
            output = model(batch["history"])
            loss, _ = dual_graph_world_model_loss(output, batch["target"], batch["static"])
            batch_size = int(batch["seed"].shape[0])
            loss_sum += float(loss.detach()) * batch_size
            sample_count += batch_size
            for name in FEATURE_NAMES:
                target_present = batch["target"][f"{name}_present"].bool()
                scale = torch.as_tensor(
                    stats["features"][f"{name}_state"]["scale"],
                    dtype=output[f"{name}_state"].dtype,
                )
                absolute_error = torch.abs(output[f"{name}_state"] - batch["target"][f"{name}_state"]) * scale
                masked_error = absolute_error * target_present.unsqueeze(-1)
                error_sums[name] += masked_error.sum(dim=(0, 1, 2)).cpu().numpy()
                error_counts[name] += int(target_present.sum())

                valid = _valid_mask(name, batch["static"])[:, None, :].expand_as(target_present)
                predicted = output[f"{name}_presence_logits"] >= 0.0
                truth = target_present
                counts = presence_counts[name]
                counts["tp"] += int((predicted & truth & valid).sum())
                counts["fp"] += int((predicted & ~truth & valid).sum())
                counts["fn"] += int((~predicted & truth & valid).sum())
                counts["valid"] += int(valid.sum())

            edge_stat = stats["features"]["physical_edge_state"]
            edge_mean = torch.as_tensor(edge_stat["mean"], dtype=output["physical_edge_state"].dtype)
            edge_scale = torch.as_tensor(edge_stat["scale"], dtype=output["physical_edge_state"].dtype)
            raw_edge_prediction = output["physical_edge_state"] * edge_scale + edge_mean
            raw_edge_target = batch["target"]["physical_edge_state"] * edge_scale + edge_mean
            edge_observed = batch["target"]["physical_edge_present"].bool()
            activity_index = EDGE_FEATURES.index("active_task_count")
            rate_index = EDGE_FEATURES.index("rate_sum")
            true_activity = raw_edge_target[..., activity_index] >= 0.5
            predicted_activity = raw_edge_prediction[..., activity_index] >= 0.5
            link_activity_counts["tp"] += int((predicted_activity & true_activity & edge_observed).sum())
            link_activity_counts["fp"] += int((predicted_activity & ~true_activity & edge_observed).sum())
            link_activity_counts["fn"] += int((~predicted_activity & true_activity & edge_observed).sum())
            link_activity_counts["valid"] += int(edge_observed.sum())
            active_rate_mask = true_activity & edge_observed
            active_rate_error = raw_edge_prediction[..., rate_index] - raw_edge_target[..., rate_index]
            active_rate_absolute_error += float(torch.abs(active_rate_error[active_rate_mask]).sum())
            active_rate_squared_error += float(torch.square(active_rate_error[active_rate_mask]).sum())
            active_rate_count += int(active_rate_mask.sum())
    state_mae: dict[str, Any] = {}
    presence: dict[str, Any] = {}
    for name, features in FEATURE_NAMES.items():
        denominator = error_counts[name]
        state_mae[name] = {
            feature: (float(error_sums[name][index] / denominator) if denominator else None)
            for index, feature in enumerate(features)
        }
        counts = presence_counts[name]
        precision = _safe_ratio(counts["tp"], counts["tp"] + counts["fp"])
        recall = _safe_ratio(counts["tp"], counts["tp"] + counts["fn"])
        f1 = None if precision is None or recall is None or precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
        presence[name] = {**counts, "precision": precision, "recall": recall, "f1": f1}
    predicted_positive = link_activity_counts["tp"] + link_activity_counts["fp"]
    actual_positive = link_activity_counts["tp"] + link_activity_counts["fn"]
    link_precision = _safe_ratio(link_activity_counts["tp"], predicted_positive)
    link_recall = _safe_ratio(link_activity_counts["tp"], actual_positive)
    if actual_positive and predicted_positive == 0:
        link_precision = 0.0
    link_f1 = (
        None
        if link_precision is None or link_recall is None
        else 0.0
        if link_precision + link_recall == 0
        else 2.0 * link_precision * link_recall / (link_precision + link_recall)
    )
    return {
        "sample_count": sample_count,
        "loss": loss_sum / sample_count if sample_count else None,
        "state_mae_physical_units": state_mae,
        "presence": presence,
        "link_activity": {
            **link_activity_counts,
            "precision": link_precision,
            "recall": link_recall,
            "f1": link_f1,
        },
        "active_only_rate": {
            "sample_count": active_rate_count,
            "mae": active_rate_absolute_error / active_rate_count if active_rate_count else None,
            "rmse": math.sqrt(active_rate_squared_error / active_rate_count) if active_rate_count else None,
            "unit": "AirFogSim rate unit",
        },
    }


def _write_report(path: Path, summary: Mapping[str, Any], evaluation: Mapping[str, Any]) -> None:
    lines = [
        "# AirFogSim dual-graph tensor v2 training smoke",
        "",
        f"- Smoke ready: `{str(summary['smoke_ready']).lower()}`",
        f"- Training split: `dev_train`, samples: `{summary['train_sample_count']}`",
        f"- Epochs: `{summary['config']['epochs']}`",
        "- Scope: development smoke only; this is not a formal model comparison.",
        "",
        "## Evaluation",
        "",
        "| Split | Samples | Loss | Link activity F1 | Active-only rate MAE | Flow F1 | Task F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split, report in evaluation.items():
        def value(name: str) -> str:
            metric = report["presence"][name]["f1"]
            return "N/A" if metric is None else f"{metric:.4f}"
        link_f1 = report["link_activity"]["f1"]
        rate_mae = report["active_only_rate"]["mae"]
        lines.append(
            f"| {split} | {report['sample_count']} | {report['loss']:.6f} | "
            f"{'N/A' if link_f1 is None else f'{link_f1:.4f}'} | "
            f"{'N/A' if rate_mae is None else f'{rate_mae:.4f}'} | {value('flow')} | {value('task')} |"
        )
    lines.extend(
        [
            "",
            "The JSON evaluation file contains per-feature MAE after inverse normalization.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_smoke(
    *,
    dataset_dir: str | Path,
    output_dir: str | Path,
    epochs: int = 1,
    batch_size: int = 8,
    hidden_dim: int = 32,
    learning_rate: float = 1e-3,
    random_seed: int = 2026,
    eval_splits: Iterable[str] = ("dev_validation", "dev_calibration"),
) -> dict[str, Any]:
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _seed_everything(int(random_seed))
    torch.set_num_threads(1)
    contract = _read_json(dataset_dir / "tensor_contract.json")
    stats = _read_json(dataset_dir / "normalization_stats.json")
    train_dataset = AirFogSimTensorWindowDataset(
        dataset_dir,
        split="dev_train",
        stats=stats,
        normalize=True,
    )
    if len(train_dataset) == 0:
        raise ValueError("dev_train split is empty")
    generator = torch.Generator().manual_seed(int(random_seed))
    train_loader = DataLoader(train_dataset, batch_size=int(batch_size), shuffle=True, generator=generator)
    model = MinimalDualGraphWorldModel(
        hidden_dim=int(hidden_dim),
        horizon_steps=int(contract["horizon_steps"]),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    training_history: list[dict[str, Any]] = []
    for epoch in range(int(epochs)):
        model.train()
        total_loss = 0.0
        seen = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            output = model(batch["history"])
            loss, _ = dual_graph_world_model_loss(output, batch["target"], batch["static"])
            if not torch.isfinite(loss):
                raise ValueError("non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            current_batch = int(batch["seed"].shape[0])
            total_loss += float(loss.detach()) * current_batch
            seen += current_batch
        training_history.append({"epoch": epoch + 1, "train_loss": total_loss / seen, "sample_count": seen})

    evaluation: dict[str, Any] = {}
    for split in eval_splits:
        dataset = AirFogSimTensorWindowDataset(dataset_dir, split=str(split), stats=stats, normalize=True)
        if len(dataset) == 0:
            raise ValueError(f"evaluation split {split} is empty")
        evaluation[str(split)] = _evaluate(model, DataLoader(dataset, batch_size=int(batch_size)), stats)
    finite_results = all(
        report["loss"] is not None and math.isfinite(float(report["loss"]))
        for report in evaluation.values()
    )
    smoke_ready = bool(training_history and finite_results)
    config = {
        "dataset_dir": str(dataset_dir.resolve()),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "hidden_dim": int(hidden_dim),
        "learning_rate": float(learning_rate),
        "random_seed": int(random_seed),
        "eval_splits": [str(value) for value in eval_splits],
    }
    summary = {
        "schema_version": "PI-JWM-AirFogSim-training-smoke-v2",
        "smoke_ready": smoke_ready,
        "formal_training_result": False,
        "scope": "development_pipeline_smoke",
        "train_sample_count": len(train_dataset),
        "training_history": training_history,
        "config": config,
        "evaluation_splits": list(evaluation),
    }
    _write_json(output_dir / "config.json", config)
    _write_json(output_dir / "training_history.json", training_history)
    _write_json(output_dir / "evaluation.json", evaluation)
    _write_json(output_dir / "summary.json", summary)
    torch.save({"model_state_dict": model.state_dict(), "contract": contract, "stats": stats, "config": config}, output_dir / "model.pt")
    _write_report(output_dir / "REPORT.md", summary, evaluation)
    evidence_paths = [
        output_dir / "config.json",
        output_dir / "training_history.json",
        output_dir / "evaluation.json",
        output_dir / "summary.json",
        output_dir / "model.pt",
        output_dir / "REPORT.md",
    ]
    manifest = {
        "schema_version": "PI-JWM-AirFogSim-training-smoke-manifest-v2",
        "files": {
            path.name: {"sha256": _sha256(path), "size_bytes": path.stat().st_size}
            for path in evidence_paths
        },
    }
    _write_json(output_dir / "manifest.json", manifest)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a one-epoch PI-JWM AirFogSim tensor-v2 training smoke.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--random-seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_smoke(
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
