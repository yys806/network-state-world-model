"""Frozen-window batching and validation metrics for R4 GPU screening."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .airfogsim_sparse_diagnostics_v2 import average_precision
from .airfogsim_teacher_tensor_v3 import (
    INFORMATION_EDGE_FEATURES,
    INFORMATION_NODE_FEATURES,
    PHYSICAL_EDGE_FEATURES,
    PHYSICAL_NODE_FEATURES,
)
from .airfogsim_tensor_v2 import FLOW_FEATURES, TASK_FEATURES
from .formal_airfogsim_graph_v1 import FORMAL_DAG_STATE_FEATURES
from .r3_preflight_data import (
    ExplicitStateBatch,
    R3Window,
    read_trajectory_index,
    select_r3_windows,
)
from .teacher_evaluation_v3 import SELECTION_COMPONENTS


R4_GPU_SCREENING_SCHEMA = "PIJWM-R4-GPU-Screening-v1"


@dataclass(frozen=True)
class FrozenScreeningProtocol:
    training_seed: int
    max_epochs: int
    patience: int
    effective_batch_size: int
    minimum_improvement: float


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_frozen_screening_protocol(
    evaluation_root: str | Path,
) -> FrozenScreeningProtocol:
    protocol = _read_json(Path(evaluation_root) / "fair_experiment_protocol.json")
    budget = protocol.get("budgets", {}).get("module_screening", {})
    common = protocol.get("common_training", {})
    seeds = budget.get("training_seeds")
    if seeds != [20260803]:
        raise ValueError("R4 module screening requires the frozen training seed")
    frozen = FrozenScreeningProtocol(
        training_seed=int(seeds[0]),
        max_epochs=int(budget.get("max_epochs", 0)),
        patience=int(budget.get("early_stopping_patience", 0)),
        effective_batch_size=int(common.get("batch_size", 0)),
        minimum_improvement=float(common.get("minimum_improvement", 0.0)),
    )
    if (
        frozen.max_epochs != 30
        or frozen.patience != 5
        or frozen.effective_batch_size != 32
        or frozen.minimum_improvement != 1.0e-4
        or not common.get("same_train_windows")
        or not common.get("same_optimizer_step_cap")
        or not common.get("same_common_output_heads")
    ):
        raise ValueError("R4 module-screening protocol has drifted from the frozen R2 budget")
    return frozen


def _window_rows(dataset_root: Path, split: str) -> list[dict[str, str]]:
    if split == "locked_test":
        raise ValueError("locked_test cannot be used by R4 module screening")
    if split not in {"train", "validation", "calibration"}:
        raise ValueError(f"unsupported R4 split: {split}")
    with (dataset_root / "window_index.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("split") == split]
    if not rows:
        raise ValueError(f"no frozen windows for split {split}")
    return rows


def build_training_window_schedule(
    dataset_root: str | Path,
    *,
    epochs: int,
    windows_per_epoch: int,
    seed: int,
) -> list[list[R3Window]]:
    """Choose one frozen 8->3 window per selected trajectory and epoch."""

    dataset_root = Path(dataset_root).resolve()
    if epochs <= 0 or windows_per_epoch <= 0:
        raise ValueError("epochs and windows_per_epoch must be positive")
    trajectory_rows = {
        int(row["seed"]): row
        for row in read_trajectory_index(dataset_root)
        if row.get("split") == "train" and row.get("v3_status") == "materialized"
    }
    rows_by_seed: dict[int, list[dict[str, str]]] = {}
    for row in _window_rows(dataset_root, "train"):
        rows_by_seed.setdefault(int(row["seed"]), []).append(row)
    eligible = sorted(set(trajectory_rows) & set(rows_by_seed))
    if windows_per_epoch > len(eligible):
        raise ValueError("windows_per_epoch exceeds the number of train trajectories")

    schedule: list[list[R3Window]] = []
    for epoch in range(epochs):
        generator = random.Random(int(seed) + epoch * 104729)
        selected_seeds = eligible.copy()
        generator.shuffle(selected_seeds)
        selected_seeds = selected_seeds[:windows_per_epoch]
        epoch_windows: list[R3Window] = []
        for environment_seed in selected_seeds:
            rows = rows_by_seed[environment_seed]
            token = hashlib.sha256(
                f"{seed}::{epoch}::{environment_seed}".encode("ascii")
            ).digest()
            row = rows[int.from_bytes(token[:8], "big") % len(rows)]
            trajectory = trajectory_rows[environment_seed]
            history_start = int(row["input_start_index"])
            history_end = int(row["input_end_index"])
            target_start = int(row["label_start_index"])
            target_end = int(row["label_end_index"])
            if history_end != target_start or target_end - target_start != 3:
                raise ValueError("frozen R1 training window is not an 8->3 window")
            epoch_windows.append(
                R3Window(
                    trajectory_id=str(trajectory["trajectory_id"]),
                    environment_seed=environment_seed,
                    split="train",
                    tensor_path=(
                        dataset_root
                        / str(trajectory["v3_seed_dir"])
                        / "trajectory_tensors.npz"
                    ),
                    history_start=history_start,
                    history_end=history_end,
                    target_start=target_start,
                    target_end=target_end,
                    horizon_steps=3,
                )
            )
        schedule.append(epoch_windows)
    return schedule


def build_validation_windows(
    dataset_root: str | Path,
    *,
    split: str = "validation",
    horizons: Sequence[int] = (1, 5, 20),
    seed: int = 20260803,
) -> list[R3Window]:
    if split == "locked_test":
        raise ValueError("locked_test cannot be used by R4 module screening")
    dataset_root = Path(dataset_root).resolve()
    rows = read_trajectory_index(dataset_root)
    trajectory_count = sum(
        row.get("split") == split and row.get("v3_status") == "materialized"
        for row in rows
    )
    if trajectory_count <= 0:
        raise ValueError(f"no materialized trajectories for split {split}")
    return select_r3_windows(
        dataset_root,
        rows,
        split=split,
        horizons=horizons,
        history_steps=8,
        per_horizon=trajectory_count,
        seed=seed,
    )


def collate_explicit_batches(batches: Sequence[ExplicitStateBatch]) -> ExplicitStateBatch:
    if not batches:
        raise ValueError("at least one explicit batch is required")

    def collate_namespace(name: str) -> dict[str, torch.Tensor]:
        namespaces = [getattr(batch, name) for batch in batches]
        keys = set(namespaces[0])
        if any(set(namespace) != keys for namespace in namespaces[1:]):
            raise ValueError(f"R4 batch namespace mismatch: {name}")
        return {key: torch.cat([namespace[key] for namespace in namespaces], dim=0) for key in keys}

    items = []
    for batch in batches:
        if "items" in batch.metadata:
            items.extend(batch.metadata["items"])
        else:
            items.append(dict(batch.metadata))
    return ExplicitStateBatch(
        history=collate_namespace("history"),
        history_action=collate_namespace("history_action"),
        future_action=collate_namespace("future_action"),
        target=collate_namespace("target"),
        static=collate_namespace("static"),
        metadata={"items": items, "batch_size": len(items)},
    )


def move_explicit_batch(
    batch: ExplicitStateBatch, device: str | torch.device
) -> ExplicitStateBatch:
    def move(namespace: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {key: value.to(device, non_blocking=True) for key, value in namespace.items()}

    return ExplicitStateBatch(
        history=move(batch.history),
        history_action=move(batch.history_action),
        future_action=move(batch.future_action),
        target=move(batch.target),
        static=move(batch.static),
        metadata=dict(batch.metadata),
    )


class R4ValidationAccumulator:
    """Pool the four frozen checkpoint-selection terms over validation windows."""

    def __init__(
        self,
        normalization_stats: Mapping[str, Any],
        *,
        selection_scales: Mapping[str, float],
    ) -> None:
        if normalization_stats.get("source_split") != "train":
            raise ValueError("R4 validation requires train-only normalization")
        if set(selection_scales) != set(SELECTION_COMPONENTS):
            raise ValueError("R4 selection scales are incomplete")
        self.stats = normalization_stats
        self.selection_scales = {key: float(value) for key, value in selection_scales.items()}
        if any(not math.isfinite(value) or value <= 0 for value in self.selection_scales.values()):
            raise ValueError("R4 selection scales must be finite and positive")
        self._continuous = {
            metric_id: {"numerator": 0.0, "count": 0, "kind": kind}
            for metric_id, kind in (
                ("state.physical_node.position.rmse", "rmse"),
                ("state.physical_node.motion.rmse", "rmse"),
                ("state.physical_edge.distance.rmse", "rmse"),
                ("state.physical_edge.relative_speed.rmse", "rmse"),
                ("state.information_node.queue.mae", "mae"),
                ("state.information_node.cpu_backlog.mae", "mae"),
                ("state.information_edge.rate.rmse", "rmse"),
                ("state.flow.remaining_data.mae", "mae"),
                ("state.task.deadline_remaining.mae", "mae"),
                ("state.dag.unfinished_parent_count.mae", "mae"),
            )
        }
        self._activity_scores: list[np.ndarray] = []
        self._activity_labels: list[np.ndarray] = []
        self._rate_abs = 0.0
        self._rate_count = 0
        self._lifecycle = np.zeros((5, 5), dtype=np.int64)
        self.window_count = 0

    def _scale(self, group: str, reference: torch.Tensor) -> torch.Tensor:
        return reference.new_tensor(self.stats["features"][group]["scale"])

    def _add(self, metric_id: str, error: torch.Tensor, valid: torch.Tensor) -> None:
        valid = valid.bool()
        if error.shape[: valid.ndim] != valid.shape:
            raise ValueError(f"R4 metric mask mismatch: {metric_id}")
        if not valid.any():
            return
        selected = error[valid]
        if not torch.isfinite(selected).all():
            raise ValueError(f"R4 validation metric is non-finite: {metric_id}")
        bucket = self._continuous[metric_id]
        if bucket["kind"] == "rmse":
            bucket["numerator"] += float(torch.square(selected).sum().item())
        else:
            bucket["numerator"] += float(torch.abs(selected).sum().item())
        bucket["count"] += int(selected.numel())

    def update(self, output: Any, batch: ExplicitStateBatch) -> None:
        prediction = output.predicted_explicit
        target = batch.target
        horizon = next(iter(prediction.values())).shape[1]
        self.window_count += int(next(iter(prediction.values())).shape[0])

        pn_error = (prediction["physical_node_state"] - target["physical_node_state"][:, :horizon]) * self._scale("physical_node_state", prediction["physical_node_state"])
        pn_valid = target["physical_node_present"][:, :horizon].bool()
        pn_mask = target["physical_node_feature_mask"][:, :horizon].bool()
        position_indices = [PHYSICAL_NODE_FEATURES.index(name) for name in ("x", "y", "z")]
        position_valid = pn_valid & pn_mask[..., position_indices].all(dim=-1)
        position_error = torch.linalg.vector_norm(pn_error[..., position_indices], dim=-1)
        self._add("state.physical_node.position.rmse", position_error, position_valid)
        speed = PHYSICAL_NODE_FEATURES.index("speed")
        self._add("state.physical_node.motion.rmse", pn_error[..., speed], pn_valid & pn_mask[..., speed])

        pe_error = (prediction["physical_edge_state"] - target["physical_edge_state"][:, :horizon]) * self._scale("physical_edge_state", prediction["physical_edge_state"])
        pe_valid = target["physical_edge_present"][:, :horizon].bool()
        pe_mask = target["physical_edge_feature_mask"][:, :horizon].bool()
        for metric_id, feature in (
            ("state.physical_edge.distance.rmse", "distance"),
            ("state.physical_edge.relative_speed.rmse", "relative_speed"),
        ):
            index = PHYSICAL_EDGE_FEATURES.index(feature)
            self._add(metric_id, pe_error[..., index], pe_valid & pe_mask[..., index])

        inode_error = (prediction["information_node_state"] - target["information_node_state"][:, :horizon]) * self._scale("information_node_state", prediction["information_node_state"])
        inode_valid = target["information_node_present"][:, :horizon].bool()
        inode_mask = target["information_node_feature_mask"][:, :horizon].bool()
        queue_indices = [
            INFORMATION_NODE_FEATURES.index(name)
            for name in ("unassigned_queue_count", "tx_queue_count", "return_queue_count")
        ]
        self._add(
            "state.information_node.queue.mae",
            inode_error[..., queue_indices],
            inode_valid.unsqueeze(-1) & inode_mask[..., queue_indices],
        )
        backlog = INFORMATION_NODE_FEATURES.index("cpu_backlog")
        self._add(
            "state.information_node.cpu_backlog.mae",
            inode_error[..., backlog],
            inode_valid & inode_mask[..., backlog],
        )

        iedge_error = (prediction["information_edge_state"] - target["information_edge_state"][:, :horizon]) * self._scale("information_edge_state", prediction["information_edge_state"])
        iedge_valid = target["information_edge_present"][:, :horizon].bool()
        iedge_mask = target["information_edge_feature_mask"][:, :horizon].bool()
        active_index = INFORMATION_EDGE_FEATURES.index("outcome.active_task_count")
        rate_index = INFORMATION_EDGE_FEATURES.index("outcome.rate_sum")
        self._add(
            "state.information_edge.rate.rmse",
            iedge_error[..., rate_index],
            iedge_valid & iedge_mask[..., rate_index],
        )
        activity_valid = target["information_link_activity_mask"][:, :horizon].bool()
        activity_target = target["information_link_activity"][:, :horizon].bool()
        activity_scores = torch.sigmoid(
            output.predicted_logits["information_link_activity"][:, :horizon]
        )
        self._activity_scores.append(activity_scores[activity_valid].detach().cpu().numpy())
        self._activity_labels.append(activity_target[activity_valid].detach().cpu().numpy())
        active_rate_valid = activity_valid & activity_target & iedge_mask[..., rate_index]
        if active_rate_valid.any():
            self._rate_abs += float(torch.abs(iedge_error[..., rate_index][active_rate_valid]).sum().item())
            self._rate_count += int(active_rate_valid.sum().item())

        flow_error = (prediction["data_flow_state"] - target["data_flow_state"][:, :horizon]) * self._scale("data_flow_state", prediction["data_flow_state"])
        flow_valid = target["data_flow_present"][:, :horizon].bool() & batch.static["data_flow_valid"].bool()[:, None, :]
        remaining = FLOW_FEATURES.index("remaining_data")
        self._add("state.flow.remaining_data.mae", flow_error[..., remaining], flow_valid)

        task_error = (prediction["task_state"] - target["task_state"][:, :horizon]) * self._scale("task_state", prediction["task_state"])
        task_valid = target["task_present"][:, :horizon].bool() & batch.static["task_valid"].bool()[:, None, :]
        deadline = TASK_FEATURES.index("deadline_remaining")
        self._add("state.task.deadline_remaining.mae", task_error[..., deadline], task_valid)

        dag_error = (prediction["task_dag_state"] - target["task_dag_state"][:, :horizon]) * self._scale("task_dag_state", prediction["task_dag_state"])
        dag_valid = target["task_dag_state_present"][:, :horizon].bool() & batch.static["task_valid"].bool()[:, None, :]
        unfinished = FORMAL_DAG_STATE_FEATURES.index("unfinished_parent_count")
        self._add("state.dag.unfinished_parent_count.mae", dag_error[..., unfinished], dag_valid)

        lifecycle_truth = target["task_lifecycle_index"][:, :horizon].long()
        lifecycle_prediction = output.predicted_logits["task_lifecycle"][:, :horizon].argmax(dim=-1)
        lifecycle_valid = task_valid & (lifecycle_truth >= 0) & (lifecycle_truth < 5)
        truth_np = lifecycle_truth[lifecycle_valid].detach().cpu().numpy()
        prediction_np = lifecycle_prediction[lifecycle_valid].detach().cpu().numpy()
        np.add.at(self._lifecycle, (truth_np, prediction_np), 1)

    def finalize(self) -> dict[str, Any]:
        metrics: dict[str, dict[str, Any]] = {}
        for metric_id, bucket in self._continuous.items():
            count = int(bucket["count"])
            if count:
                value = (
                    math.sqrt(float(bucket["numerator"]) / count)
                    if bucket["kind"] == "rmse"
                    else float(bucket["numerator"]) / count
                )
                metrics[metric_id] = {"status": "computed", "value": value, "numerator": float(bucket["numerator"]), "denominator": float(count), "count": count}
            else:
                metrics[metric_id] = {"status": "not_computable", "value": None, "numerator": None, "denominator": 0.0, "count": 0}

        scores = np.concatenate(self._activity_scores) if self._activity_scores else np.empty(0)
        labels = np.concatenate(self._activity_labels) if self._activity_labels else np.empty(0, dtype=bool)
        auprc = average_precision(scores, labels) if labels.size else None
        metrics["event.information_link_activity.auprc"] = {
            "status": "computed" if auprc is not None else "not_computable",
            "value": None if auprc is None else float(auprc),
            "count": int(labels.size),
        }
        metrics["link.active_only_rate.mae"] = {
            "status": "computed" if self._rate_count else "not_computable",
            "value": self._rate_abs / self._rate_count if self._rate_count else None,
            "numerator": self._rate_abs if self._rate_count else None,
            "denominator": float(self._rate_count),
            "count": self._rate_count,
        }
        class_f1 = []
        for index in range(5):
            tp = int(self._lifecycle[index, index])
            fp = int(self._lifecycle[:, index].sum() - tp)
            fn = int(self._lifecycle[index, :].sum() - tp)
            denominator = 2 * tp + fp + fn
            class_f1.append(0.0 if denominator == 0 else 2.0 * tp / denominator)
        lifecycle_count = int(self._lifecycle.sum())
        metrics["task.lifecycle.macro_f1"] = {
            "status": "computed" if lifecycle_count else "not_computable",
            "value": float(np.mean(class_f1)) if lifecycle_count else None,
            "count": lifecycle_count,
            "confusion_matrix": self._lifecycle.tolist(),
        }

        invalid = [
            metric_id
            for metric_id in SELECTION_COMPONENTS
            if metrics[metric_id]["status"] != "computed"
        ]
        if invalid:
            metrics["selection.required_continuous.normalized_error"] = {
                "status": "not_computable",
                "value": None,
                "missing_components": invalid,
            }
        else:
            normalized = [
                float(metrics[metric_id]["value"]) / self.selection_scales[metric_id]
                for metric_id in SELECTION_COMPONENTS
            ]
            metrics["selection.required_continuous.normalized_error"] = {
                "status": "computed",
                "value": float(np.mean(normalized)),
                "count": len(normalized),
                "components": dict(zip(SELECTION_COMPONENTS, normalized)),
            }

        required = (
            "event.information_link_activity.auprc",
            "link.active_only_rate.mae",
            "task.lifecycle.macro_f1",
            "selection.required_continuous.normalized_error",
        )
        eligible = all(metrics[name]["status"] == "computed" for name in required)
        if eligible:
            rate_scale = float(
                self.stats["features"]["information_edge_state"]["scale"]
                [INFORMATION_EDGE_FEATURES.index("outcome.rate_sum")]
            )
            score = 0.25 * (
                1.0 - float(metrics[required[0]]["value"])
                + float(metrics[required[1]]["value"]) / rate_scale
                + 1.0 - float(metrics[required[2]]["value"])
                + float(metrics[required[3]]["value"])
            )
        else:
            score = None
        return {
            "schema_version": R4_GPU_SCREENING_SCHEMA,
            "window_count": self.window_count,
            "candidate_eligible": eligible,
            "validation_protocol_score": score,
            "metrics": metrics,
        }


__all__ = [
    "FrozenScreeningProtocol",
    "R4_GPU_SCREENING_SCHEMA",
    "R4ValidationAccumulator",
    "build_training_window_schedule",
    "build_validation_windows",
    "collate_explicit_batches",
    "load_frozen_screening_protocol",
    "move_explicit_batch",
]
