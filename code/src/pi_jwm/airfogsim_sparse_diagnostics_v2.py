"""Shared baselines and metrics for the AirFogSim sparse-event diagnostic."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import torch

from .airfogsim_tensor_v2 import EDGE_FEATURES, FLOW_FEATURES, NODE_FEATURES, TASK_FEATURES


COMPONENT_FEATURE_NAMES = {
    "node": NODE_FEATURES,
    "physical_edge": EDGE_FEATURES,
    "flow": FLOW_FEATURES,
    "task": TASK_FEATURES,
}


def _logits_from_boolean(value: torch.Tensor) -> torch.Tensor:
    return torch.where(value.bool(), 20.0, -20.0).to(dtype=torch.float32)


def _repeat_last(value: torch.Tensor, horizon_steps: int) -> torch.Tensor:
    repeats = [1] * value.ndim
    repeats[1] = int(horizon_steps)
    return value[:, -1:].repeat(*repeats)


def _lifecycle_logits_from_last_history(
    batch: Mapping[str, Any],
    horizon_steps: int,
) -> torch.Tensor:
    lifecycle = _repeat_last(batch["history"]["task_lifecycle_index"], horizon_steps).long()
    logits = torch.full((*lifecycle.shape, 5), -20.0, dtype=torch.float32, device=lifecycle.device)
    valid = lifecycle >= 0
    if valid.any():
        logits[valid, lifecycle[valid]] = 20.0
    return logits


def build_last_persistence_prediction(
    batch: Mapping[str, Any],
    *,
    horizon_steps: int,
) -> dict[str, torch.Tensor]:
    """Repeat the last observed state, presence, activity, and lifecycle."""

    prediction: dict[str, torch.Tensor] = {}
    for name in COMPONENT_FEATURE_NAMES:
        prediction[f"{name}_state"] = _repeat_last(
            batch["history"][f"{name}_state"], horizon_steps
        )
        prediction[f"{name}_presence_logits"] = _logits_from_boolean(
            _repeat_last(batch["history"][f"{name}_present"], horizon_steps)
        )
    prediction["link_activity_logits"] = _logits_from_boolean(
        _repeat_last(batch["history"]["link_activity"], horizon_steps)
    )
    prediction["task_lifecycle_logits"] = _lifecycle_logits_from_last_history(
        batch, horizon_steps
    )
    return prediction


def _normalized_raw_zero(reference: torch.Tensor, stat: Mapping[str, Any]) -> torch.Tensor:
    mean = reference.new_tensor(stat["mean"])
    scale = reference.new_tensor(stat["scale"]).clamp_min(1e-6)
    value = -mean / scale
    return value.reshape(*([1] * (reference.ndim - 1)), -1).expand_as(reference).clone()


def build_zero_activity_prediction(
    batch: Mapping[str, Any],
    stats: Mapping[str, Any],
    *,
    lifecycle_majority_index: int,
) -> dict[str, torch.Tensor]:
    """Predict no future traffic while preserving nontraffic physical state."""

    horizon_steps = int(batch["target"]["node_state"].shape[1])
    prediction = build_last_persistence_prediction(batch, horizon_steps=horizon_steps)
    prediction["link_activity_logits"].fill_(-20.0)
    prediction["flow_presence_logits"].fill_(-20.0)
    prediction["task_presence_logits"].fill_(-20.0)
    prediction["flow_state"] = _normalized_raw_zero(
        prediction["flow_state"], stats["features"]["flow_state"]
    )
    prediction["task_state"] = _normalized_raw_zero(
        prediction["task_state"], stats["features"]["task_state"]
    )
    edge_stat = stats["features"]["physical_edge_state"]
    edge_mean = prediction["physical_edge_state"].new_tensor(edge_stat["mean"])
    edge_scale = prediction["physical_edge_state"].new_tensor(edge_stat["scale"]).clamp_min(1e-6)
    for feature in ("rate_sum", "active_task_count", "allocated_rb_count"):
        index = EDGE_FEATURES.index(feature)
        prediction["physical_edge_state"][..., index] = -edge_mean[index] / edge_scale[index]
    prediction["task_lifecycle_logits"].fill_(-20.0)
    if not 0 <= int(lifecycle_majority_index) < 5:
        raise ValueError("lifecycle_majority_index must be in [0, 4]")
    prediction["task_lifecycle_logits"][..., int(lifecycle_majority_index)] = 20.0
    return prediction


def average_precision(scores: Any, labels: Any) -> float | None:
    """Compute stepwise AUPRC while treating tied scores as one threshold."""

    scores_array = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels_array = np.asarray(labels, dtype=bool).reshape(-1)
    if scores_array.shape != labels_array.shape:
        raise ValueError("scores and labels must have equal shapes")
    if labels_array.sum() == 0:
        return None
    order = np.argsort(-scores_array, kind="mergesort")
    sorted_scores = scores_array[order]
    sorted_labels = labels_array[order]
    cumulative_true = np.cumsum(sorted_labels)
    threshold_ends = np.flatnonzero(np.r_[sorted_scores[1:] != sorted_scores[:-1], True])
    true_at_threshold = cumulative_true[threshold_ends]
    precision = true_at_threshold / (threshold_ends + 1)
    recall = true_at_threshold / labels_array.sum()
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


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


def _classification_report(tp: int, fp: int, fn: int, valid: int) -> dict[str, Any]:
    actual_positive = tp + fn
    predicted_positive = tp + fp
    precision = float(tp / predicted_positive) if predicted_positive else (0.0 if actual_positive else None)
    recall = float(tp / actual_positive) if actual_positive else None
    f1 = None
    if recall is not None and precision is not None:
        f1 = 0.0 if precision + recall == 0 else float(2 * precision * recall / (precision + recall))
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "valid": int(valid),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


class SparseDiagnosticAccumulator:
    def __init__(self, stats: Mapping[str, Any]) -> None:
        self.stats = stats
        self.sample_count = 0
        self.state_error_sums = {
            name: np.zeros(len(features), dtype=np.float64)
            for name, features in COMPONENT_FEATURE_NAMES.items()
        }
        self.state_counts = {name: 0 for name in COMPONENT_FEATURE_NAMES}
        self.presence_counts = {
            name: {"tp": 0, "fp": 0, "fn": 0, "valid": 0}
            for name in ("flow", "task")
        }
        self.link_counts = {"tp": 0, "fp": 0, "fn": 0, "valid": 0}
        self.link_scores: list[np.ndarray] = []
        self.link_labels: list[np.ndarray] = []
        self.rate_abs_sum = 0.0
        self.rate_sq_sum = 0.0
        self.rate_count = 0
        self.lifecycle_confusion = np.zeros((5, 5), dtype=np.int64)

    def update(
        self,
        prediction: Mapping[str, torch.Tensor],
        target: Mapping[str, torch.Tensor],
        static: Mapping[str, torch.Tensor],
    ) -> None:
        self.sample_count += int(target["node_state"].shape[0])
        for name in COMPONENT_FEATURE_NAMES:
            target_present = target[f"{name}_present"].bool()
            scale = prediction[f"{name}_state"].new_tensor(
                self.stats["features"][f"{name}_state"]["scale"]
            )
            error = torch.abs(prediction[f"{name}_state"] - target[f"{name}_state"]) * scale
            self.state_error_sums[name] += (
                error * target_present.unsqueeze(-1)
            ).sum(dim=(0, 1, 2)).detach().cpu().numpy()
            self.state_counts[name] += int(target_present.sum())

        for name in ("flow", "task"):
            truth = target[f"{name}_present"].bool()
            valid = _valid_mask(name, static)[:, None, :].expand_as(truth)
            predicted = prediction[f"{name}_presence_logits"] >= 0
            counts = self.presence_counts[name]
            counts["tp"] += int((predicted & truth & valid).sum())
            counts["fp"] += int((predicted & ~truth & valid).sum())
            counts["fn"] += int((~predicted & truth & valid).sum())
            counts["valid"] += int(valid.sum())

        truth_activity = target["link_activity"].bool()
        edge_valid = _valid_mask("physical_edge", static)[:, None, :].expand_as(truth_activity)
        activity_scores = torch.sigmoid(prediction["link_activity_logits"])
        predicted_activity = activity_scores >= 0.5
        self.link_counts["tp"] += int((predicted_activity & truth_activity & edge_valid).sum())
        self.link_counts["fp"] += int((predicted_activity & ~truth_activity & edge_valid).sum())
        self.link_counts["fn"] += int((~predicted_activity & truth_activity & edge_valid).sum())
        self.link_counts["valid"] += int(edge_valid.sum())
        self.link_scores.append(activity_scores[edge_valid].detach().cpu().numpy())
        self.link_labels.append(truth_activity[edge_valid].detach().cpu().numpy())

        edge_stat = self.stats["features"]["physical_edge_state"]
        edge_scale = prediction["physical_edge_state"].new_tensor(edge_stat["scale"])
        rate_index = EDGE_FEATURES.index("rate_sum")
        rate_error = (
            prediction["physical_edge_state"][..., rate_index]
            - target["physical_edge_state"][..., rate_index]
        ) * edge_scale[rate_index]
        rate_mask = truth_activity & edge_valid
        self.rate_abs_sum += float(torch.abs(rate_error[rate_mask]).sum())
        self.rate_sq_sum += float(torch.square(rate_error[rate_mask]).sum())
        self.rate_count += int(rate_mask.sum())

        lifecycle_truth = target["task_lifecycle_index"].long()
        lifecycle_mask = target["task_present"].bool() & (lifecycle_truth >= 0)
        lifecycle_predicted = prediction["task_lifecycle_logits"].argmax(dim=-1)
        truth_values = lifecycle_truth[lifecycle_mask].detach().cpu().numpy()
        predicted_values = lifecycle_predicted[lifecycle_mask].detach().cpu().numpy()
        for truth_value, predicted_value in zip(truth_values, predicted_values):
            self.lifecycle_confusion[int(truth_value), int(predicted_value)] += 1

    def finalize(self) -> dict[str, Any]:
        state_mae = {}
        for name, features in COMPONENT_FEATURE_NAMES.items():
            count = self.state_counts[name]
            state_mae[name] = {
                feature: (float(self.state_error_sums[name][index] / count) if count else None)
                for index, feature in enumerate(features)
            }
        presence = {
            name: _classification_report(**counts)
            for name, counts in self.presence_counts.items()
        }
        link_report = _classification_report(**self.link_counts)
        scores = np.concatenate(self.link_scores) if self.link_scores else np.empty((0,))
        labels = np.concatenate(self.link_labels) if self.link_labels else np.empty((0,), dtype=bool)
        link_report["auprc"] = average_precision(scores, labels)

        support = self.lifecycle_confusion.sum(axis=1)
        correct = int(np.trace(self.lifecycle_confusion))
        total = int(support.sum())
        class_f1: list[float | None] = []
        for index in range(5):
            tp = int(self.lifecycle_confusion[index, index])
            fp = int(self.lifecycle_confusion[:, index].sum() - tp)
            fn = int(self.lifecycle_confusion[index, :].sum() - tp)
            report = _classification_report(tp, fp, fn, int(support[index]))
            class_f1.append(report["f1"])
        supported_f1 = [value for index, value in enumerate(class_f1) if support[index] > 0 and value is not None]
        lifecycle = {
            "accuracy": float(correct / total) if total else None,
            "macro_f1": float(np.mean(supported_f1)) if supported_f1 else None,
            "support": support.astype(int).tolist(),
            "class_f1": class_f1,
            "valid": total,
        }
        return {
            "sample_count": self.sample_count,
            "state_mae_physical_units": state_mae,
            "presence": presence,
            "link_activity": link_report,
            "active_only_rate": {
                "sample_count": self.rate_count,
                "mae": self.rate_abs_sum / self.rate_count if self.rate_count else None,
                "rmse": math.sqrt(self.rate_sq_sum / self.rate_count) if self.rate_count else None,
                "unit": "AirFogSim rate unit",
            },
            "task_lifecycle": lifecycle,
        }


def evaluate_prediction_batches(
    prediction_batches: Iterable[Mapping[str, torch.Tensor]],
    batches: Iterable[Mapping[str, Any]],
    stats: Mapping[str, Any],
) -> dict[str, Any]:
    accumulator = SparseDiagnosticAccumulator(stats)
    for prediction, batch in zip(prediction_batches, batches):
        accumulator.update(prediction, batch["target"], batch["static"])
    return accumulator.finalize()
