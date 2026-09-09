"""Auditable horizon-wise metric suite for formal PI-JWM rollouts."""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
import torch

from .airfogsim_sparse_diagnostics_v2 import average_precision
from .airfogsim_tensor_v2 import EDGE_FEATURES, FLOW_FEATURES, NODE_FEATURES, TASK_FEATURES
from .formal_binary_calibration_v1 import (
    InverseTemperatureCalibration,
    binary_probability_metrics,
)


FEATURES = {
    "node": NODE_FEATURES,
    "physical_edge": EDGE_FEATURES,
    "flow": FLOW_FEATURES,
    "task": TASK_FEATURES,
}
UNITS = {
    "node": ["m", "m", "m", "m/s", "m/s^2", "cycles/s", "MB"],
    "physical_edge": ["m", "dB", "Mbps", "count", "RB"],
    "flow": ["MB", "MB", "MB", "MB", "s"],
    "task": ["MB", "MB", "cycles", "s", "score", "MB", "cycles", "s"],
    "task_dag": ["count", "count", "boolean"],
}
DAG_FEATURES = ("parent_count", "unfinished_parent_count", "release_ready")
BINARY_EVENT_NAMES = (
    "link_activity",
    "flow_present",
    "task_present",
    "dag_release",
    "dag_edge_present",
)


def not_computable(reason: str, required_fields: list[str]) -> dict[str, Any]:
    return {
        "value": None,
        "status": "not_computable",
        "numerator": None,
        "denominator": None,
        "count": 0,
        "unit": None,
        "source_fields": required_fields,
        "reason": reason,
    }


def _definition(unit: str | None, sources: list[str], denominator: str) -> dict[str, Any]:
    return {"unit": unit, "source_fields": sources, "denominator": denominator}


def metric_registry() -> dict[str, dict[str, Any]]:
    registry: dict[str, dict[str, Any]] = {}
    for name, features in {**FEATURES, "task_dag": DAG_FEATURES}.items():
        state_key = "task_dag_state" if name == "task_dag" else f"{name}_state"
        present_key = "task_dag_state_present" if name == "task_dag" else f"{name}_present"
        for index, feature in enumerate(features):
            unit = UNITS[name][index]
            sources = [state_key, present_key]
            registry[f"state.{name}.{feature}.mae"] = _definition(unit, sources, "valid target entities")
            registry[f"state.{name}.{feature}.rmse"] = _definition(unit, sources, "valid target entities")
            registry[f"uncertainty.{name}.{feature}.nll"] = _definition("nats", sources, "valid scalar targets")
            registry[f"uncertainty.{name}.{feature}.coverage_95"] = _definition("ratio", sources, "valid scalar targets")
            registry[f"uncertainty.{name}.{feature}.interval_width_95"] = _definition(unit, sources, "valid scalar targets")
    registry["state.physical_edge.rate_sum.rmse"] = _definition(
        "Mbps", ["physical_edge_state", "physical_edge_present"], "valid target physical edges"
    )
    event_sources = {
        "event.link_activity": ["aggregate_link_activity", "aggregate_link_activity_mask", "physical_edge_endpoint_index"],
        "event.flow_present": ["flow_present", "flow_valid"],
        "event.task_present": ["task_present", "task_valid"],
        "dag.release_ready": ["task_dag_state", "task_dag_state_present"],
        "dag.edge_presence": ["dag_edge_present", "dag_edge_valid"],
    }
    for prefix, sources in event_sources.items():
        for metric in ("precision", "recall", "f1", "auprc"):
            registry[f"{prefix}.{metric}"] = _definition("ratio", sources, "valid binary labels")
    for metric, unit in (("nll", "nats"), ("brier", "ratio"), ("ece", "ratio")):
        registry[f"event.link_activity.{metric}"] = _definition(
            unit,
            event_sources["event.link_activity"],
            "valid binary labels",
        )
    registry.update(
        {
            "task.lifecycle.accuracy": _definition("ratio", ["task_lifecycle_index", "task_present"], "valid task lifecycle labels"),
            "task.lifecycle.macro_f1": _definition("ratio", ["task_lifecycle_index", "task_present"], "supported lifecycle classes"),
            "task.lifecycle.support": _definition("count", ["task_lifecycle_index", "task_present"], "lifecycle class"),
            "link.active_only_rate.mae": _definition("Mbps", ["aggregate_link_rate_sum", "aggregate_link_rate_sum_mask", "aggregate_link_activity"], "observed active target links"),
            "link.active_only_rate.rmse": _definition("Mbps", ["aggregate_link_rate_sum", "aggregate_link_rate_sum_mask", "aggregate_link_activity"], "observed active target links"),
            "dag.unfinished_parent_count.mae": _definition("count", ["task_dag_state.unfinished_parent_count", "task_dag_state_present"], "valid DAG task states"),
            "system.task_completion_rate.absolute_error": _definition("ratio", ["task_lifecycle_logits", "task_lifecycle_index"], "valid task lifecycle labels"),
            "system.communication_throughput.mae": _definition("Mbps", ["aggregate_link_rate_sum", "aggregate_link_rate_sum_mask"], "observed rollout steps"),
            "resource.rb_occupancy.mae": _definition("RB", ["aggregate_rb_occupancy", "aggregate_rb_occupancy_mask"], "observed rollout steps"),
            "system.p95_latency": _definition("s", ["task_completion_timestamp"], "completed tasks"),
            "system.p99_latency": _definition("s", ["task_completion_timestamp"], "completed tasks"),
            "system.energy": _definition("J", ["realized_energy"], "evaluated rollout steps"),
            "system.fairness": _definition("ratio", ["user_service_history"], "frozen user population"),
            "decision.action_regret": _definition("utility", ["counterfactual_action_outcomes"], "evaluated candidate actions"),
        }
    )
    return registry


def _record(
    value: float | None,
    *,
    numerator: float | int | None,
    denominator: float | int | None,
    count: int,
    unit: str | None,
    sources: list[str],
    reason: str | None = None,
) -> dict[str, Any]:
    status = "computed" if value is not None and math.isfinite(float(value)) else "not_computable"
    return {
        "value": float(value) if status == "computed" else None,
        "status": status,
        "numerator": float(numerator) if numerator is not None else None,
        "denominator": float(denominator) if denominator is not None else None,
        "count": int(count),
        "unit": unit,
        "source_fields": sources,
        "reason": reason if status == "not_computable" else None,
    }


def _classification_records(
    prefix: str,
    counts: Mapping[str, int],
    scores: list[np.ndarray],
    labels: list[np.ndarray],
    sources: list[str],
) -> dict[str, dict[str, Any]]:
    tp, fp, fn, valid = (int(counts[key]) for key in ("tp", "fp", "fn", "valid"))
    predicted_positive = tp + fp
    actual_positive = tp + fn
    precision = tp / predicted_positive if predicted_positive else None
    recall = tp / actual_positive if actual_positive else None
    f1_denominator = 2 * tp + fp + fn
    f1 = 2 * tp / f1_denominator if f1_denominator else None
    all_scores = np.concatenate(scores) if scores else np.empty((0,), dtype=np.float64)
    all_labels = np.concatenate(labels) if labels else np.empty((0,), dtype=bool)
    auprc = average_precision(all_scores, all_labels)
    return {
        f"{prefix}.precision": _record(
            precision,
            numerator=tp,
            denominator=predicted_positive,
            count=valid,
            unit="ratio",
            sources=sources,
            reason="no predicted positives" if precision is None else None,
        ),
        f"{prefix}.recall": _record(
            recall,
            numerator=tp,
            denominator=actual_positive,
            count=valid,
            unit="ratio",
            sources=sources,
            reason="no positive labels" if recall is None else None,
        ),
        f"{prefix}.f1": _record(
            f1,
            numerator=2 * tp,
            denominator=f1_denominator,
            count=valid,
            unit="ratio",
            sources=sources,
            reason="no positive labels or predictions" if f1 is None else None,
        ),
        f"{prefix}.auprc": _record(
            auprc,
            numerator=None,
            denominator=valid,
            count=valid,
            unit="ratio",
            sources=sources,
            reason="AUPRC requires at least one positive label" if auprc is None else None,
        ),
    }


def _static_mask(name: str, static: Mapping[str, torch.Tensor]) -> torch.Tensor:
    if name == "node":
        return static["node_kind_index"] >= 0
    if name == "physical_edge":
        return torch.all(static["physical_edge_endpoint_index"] >= 0, dim=-1)
    if name == "flow":
        return static["flow_valid"].bool()
    if name == "task":
        return static["task_valid"].bool()
    raise KeyError(name)


class _MetricBucket:
    def __init__(self) -> None:
        all_features = {**FEATURES, "task_dag": DAG_FEATURES}
        self.state = {
            name: {
                "abs": np.zeros(len(features), dtype=np.float64),
                "sq": np.zeros(len(features), dtype=np.float64),
                "nll": np.zeros(len(features), dtype=np.float64),
                "covered": np.zeros(len(features), dtype=np.float64),
                "width": np.zeros(len(features), dtype=np.float64),
                "count": 0,
            }
            for name, features in all_features.items()
        }
        self.binary = {
            name: {
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "valid": 0,
                "scores": [],
                "probabilities": [],
                "labels": [],
            }
            for name in ("link_activity", "flow_present", "task_present", "dag_release", "dag_edge_present")
        }
        self.lifecycle = np.zeros((5, 5), dtype=np.int64)
        self.rate_abs = 0.0
        self.rate_sq = 0.0
        self.rate_count = 0
        self.system = {
            "completion_truth": 0,
            "completion_predicted": 0,
            "completion_valid": 0,
            "deadline_abs": 0.0,
            "throughput_abs": 0.0,
            "rb_abs": 0.0,
            "aggregate_count": 0,
        }

    def add_state(
        self,
        name: str,
        mean: torch.Tensor,
        log_variance: torch.Tensor,
        target: torch.Tensor,
        present: torch.Tensor,
        scale: torch.Tensor,
    ) -> None:
        normalized_error = mean - target
        physical_error = normalized_error * scale
        sigma = torch.exp(0.5 * log_variance)
        physical_sigma = sigma * scale
        feature_mask = present.unsqueeze(-1).to(mean.dtype)
        bucket = self.state[name]
        bucket["abs"] += (torch.abs(physical_error) * feature_mask).sum(dim=(0, 1)).detach().cpu().numpy()
        bucket["sq"] += (torch.square(physical_error) * feature_mask).sum(dim=(0, 1)).detach().cpu().numpy()
        nll = 0.5 * (log_variance + torch.square(normalized_error) * torch.exp(-log_variance))
        bucket["nll"] += (nll * feature_mask).sum(dim=(0, 1)).detach().cpu().numpy()
        covered = (torch.abs(normalized_error) <= 1.96 * sigma).to(mean.dtype)
        bucket["covered"] += (covered * feature_mask).sum(dim=(0, 1)).detach().cpu().numpy()
        bucket["width"] += (3.92 * physical_sigma * feature_mask).sum(dim=(0, 1)).detach().cpu().numpy()
        bucket["count"] += int(present.sum())

    def add_binary(
        self,
        name: str,
        logits: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor,
        threshold: float,
        *,
        probabilities: torch.Tensor | None = None,
        predicted: torch.Tensor | None = None,
    ) -> None:
        if probabilities is None:
            scores = torch.sigmoid(logits)
        else:
            if (
                not isinstance(probabilities, torch.Tensor)
                or probabilities.shape != logits.shape
                or torch.is_complex(probabilities)
                or not torch.isfinite(probabilities).all().item()
                or not torch.logical_and(probabilities >= 0.0, probabilities <= 1.0).all().item()
            ):
                raise ValueError("probabilities must match logits and be finite values in [0, 1]")
            scores = probabilities
        if predicted is None:
            predicted = scores >= threshold
        elif (
            not isinstance(predicted, torch.Tensor)
            or predicted.dtype is not torch.bool
            or predicted.shape != logits.shape
            or predicted.device != logits.device
        ):
            raise ValueError(
                "predicted must be a boolean tensor matching logits shape and device"
            )
        truth = target.bool()
        item = self.binary[name]
        item["tp"] += int((predicted & truth & valid).sum())
        item["fp"] += int((predicted & ~truth & valid).sum())
        item["fn"] += int((~predicted & truth & valid).sum())
        item["valid"] += int(valid.sum())
        item["scores"].append(scores[valid].detach().cpu().numpy())
        item["probabilities"].append(scores[valid].detach().cpu().numpy())
        item["labels"].append(truth[valid].detach().cpu().numpy())


class FormalMetricAccumulator:
    def __init__(
        self,
        stats: Mapping[str, Any],
        *,
        threshold: float = 0.5,
        thresholds: Mapping[str, float] | None = None,
        distribution_available: bool = True,
        link_probability_calibration: InverseTemperatureCalibration | None = None,
    ) -> None:
        if not 0.0 < threshold < 1.0:
            raise ValueError("threshold must be between zero and one")
        self.stats = stats
        self.threshold = float(threshold)
        self.thresholds = {name: self.threshold for name in BINARY_EVENT_NAMES}
        if thresholds is not None:
            unknown = set(thresholds) - set(BINARY_EVENT_NAMES)
            if unknown:
                raise ValueError(f"unsupported binary event thresholds: {sorted(unknown)}")
            for name, value in thresholds.items():
                value = float(value)
                if not 0.0 < value < 1.0:
                    raise ValueError(f"threshold for {name} must be between zero and one")
                self.thresholds[name] = value
        self.legacy_raw_thresholds = dict(self.thresholds)
        if link_probability_calibration is not None:
            self.thresholds["link_activity"] = (
                link_probability_calibration.map_raw_threshold(
                    self.legacy_raw_thresholds["link_activity"]
                )
            )
        self.distribution_available = bool(distribution_available)
        self.link_probability_calibration = link_probability_calibration
        self._horizon_buckets: list[_MetricBucket] = []
        self._overall = _MetricBucket()
        self.sample_count = 0

    def _scale(self, name: str, reference: torch.Tensor) -> torch.Tensor:
        if name == "task_dag":
            return reference.new_ones(3)
        return reference.new_tensor(self.stats["features"][f"{name}_state"]["scale"])

    def _mean(self, name: str, reference: torch.Tensor) -> torch.Tensor:
        if name == "task_dag":
            return reference.new_zeros(3)
        return reference.new_tensor(self.stats["features"][f"{name}_state"]["mean"])

    def _update_bucket(
        self,
        bucket: _MetricBucket,
        prediction: Mapping[str, torch.Tensor],
        target: Mapping[str, torch.Tensor],
        static: Mapping[str, torch.Tensor],
        horizon: int,
    ) -> None:
        for name in FEATURES:
            bucket.add_state(
                name,
                prediction[f"{name}_state_mean"][:, horizon],
                prediction[f"{name}_state_log_variance"][:, horizon],
                target[f"{name}_state"][:, horizon],
                target[f"{name}_present"][:, horizon].bool(),
                self._scale(name, prediction[f"{name}_state_mean"]),
            )
        bucket.add_state(
            "task_dag",
            prediction["task_dag_state_mean"][:, horizon],
            prediction["task_dag_state_log_variance"][:, horizon],
            target["task_dag_state"][:, horizon],
            target["task_dag_state_present"][:, horizon].bool(),
            self._scale("task_dag", prediction["task_dag_state_mean"]),
        )

        edge_valid = _static_mask("physical_edge", static).bool()
        flow_valid = _static_mask("flow", static).bool()
        task_valid = _static_mask("task", static).bool()
        dag_valid = static["dag_edge_valid"].bool()
        aggregate_activity = target.get("aggregate_link_activity", target["link_activity"]).bool()
        aggregate_activity_mask = target.get("aggregate_link_activity_mask")
        if aggregate_activity_mask is None:
            aggregate_activity_mask = edge_valid[:, None, :].expand_as(aggregate_activity)
        activity_valid = edge_valid & aggregate_activity_mask[:, horizon].bool()
        link_logits = prediction["link_activity_logits"][:, horizon]
        link_probabilities = None
        link_threshold = self.thresholds["link_activity"]
        if self.link_probability_calibration is not None:
            link_probabilities = self.link_probability_calibration.probabilities_from_weighted_logits(
                link_logits
            )
        legacy_link_decision = None
        if self.link_probability_calibration is not None:
            legacy_link_decision = (
                torch.sigmoid(link_logits)
                >= self.legacy_raw_thresholds["link_activity"]
            )
        bucket.add_binary(
            "link_activity",
            link_logits,
            aggregate_activity[:, horizon],
            activity_valid,
            link_threshold,
            probabilities=link_probabilities,
            predicted=legacy_link_decision,
        )
        bucket.add_binary(
            "flow_present",
            prediction["flow_presence_logits"][:, horizon],
            target["flow_present"][:, horizon],
            flow_valid,
            self.thresholds["flow_present"],
        )
        bucket.add_binary(
            "task_present",
            prediction["task_presence_logits"][:, horizon],
            target["task_present"][:, horizon],
            task_valid,
            self.thresholds["task_present"],
        )
        dag_state_present = target["task_dag_state_present"][:, horizon].bool() & task_valid
        bucket.add_binary(
            "dag_release",
            prediction["dag_release_logits"][:, horizon],
            target["task_dag_state"][:, horizon, :, 2] > 0.5,
            dag_state_present,
            self.thresholds["dag_release"],
        )
        bucket.add_binary(
            "dag_edge_present",
            prediction["dag_edge_presence_logits"][:, horizon],
            target["dag_edge_present"][:, horizon],
            dag_valid,
            self.thresholds["dag_edge_present"],
        )

        lifecycle_truth = target["task_lifecycle_index"][:, horizon].long()
        lifecycle_predicted = prediction["task_lifecycle_logits"][:, horizon].argmax(dim=-1)
        lifecycle_valid = target["task_present"][:, horizon].bool() & task_valid & (lifecycle_truth >= 0)
        for truth_value, predicted_value in zip(
            lifecycle_truth[lifecycle_valid].detach().cpu().numpy(),
            lifecycle_predicted[lifecycle_valid].detach().cpu().numpy(),
        ):
            bucket.lifecycle[int(truth_value), int(predicted_value)] += 1

        rate_index = list(EDGE_FEATURES).index("rate_sum")
        rate_scale = self._scale("physical_edge", prediction["physical_edge_state_mean"])[rate_index]
        edge_mean = self._mean("physical_edge", prediction["physical_edge_state_mean"])
        aggregate_rate = target.get(
            "aggregate_link_rate_sum", target["physical_edge_state"][..., rate_index]
        )
        aggregate_rate_mask = target.get("aggregate_link_rate_sum_mask")
        if aggregate_rate_mask is None:
            aggregate_rate_mask = edge_valid[:, None, :].expand_as(aggregate_rate)
        rate_observed = edge_valid & aggregate_rate_mask[:, horizon].bool()
        predicted_rate = (
            prediction["physical_edge_state_mean"][:, horizon, :, rate_index] * rate_scale
            + edge_mean[rate_index]
        )
        if "aggregate_link_rate_sum" in target:
            target_rate = aggregate_rate[:, horizon]
        else:
            target_rate = aggregate_rate[:, horizon] * rate_scale + edge_mean[rate_index]
        rate_error = predicted_rate - target_rate
        rate_mask = aggregate_activity[:, horizon].bool() & rate_observed
        bucket.rate_abs += float(torch.abs(rate_error[rate_mask]).sum())
        bucket.rate_sq += float(torch.square(rate_error[rate_mask]).sum())
        bucket.rate_count += int(rate_mask.sum())

        finished_index = 3
        bucket.system["completion_truth"] += int((lifecycle_truth[lifecycle_valid] == finished_index).sum())
        bucket.system["completion_predicted"] += int((lifecycle_predicted[lifecycle_valid] == finished_index).sum())
        bucket.system["completion_valid"] += int(lifecycle_valid.sum())
        deadline_index = list(TASK_FEATURES).index("deadline_remaining")
        task_scale = self._scale("task", prediction["task_state_mean"])
        deadline_error = (
            prediction["task_state_mean"][:, horizon, :, deadline_index]
            - target["task_state"][:, horizon, :, deadline_index]
        ) * task_scale[deadline_index]
        present_task = target["task_present"][:, horizon].bool() & task_valid
        if torch.any(present_task):
            bucket.system["deadline_abs"] += float(torch.abs(deadline_error[present_task]).mean())
        throughput_error = torch.abs(
            (predicted_rate * rate_observed).sum(dim=1) - (target_rate * rate_observed).sum(dim=1)
        )
        rb_index = list(EDGE_FEATURES).index("allocated_rb_count")
        rb_scale = self._scale("physical_edge", prediction["physical_edge_state_mean"])[rb_index]
        predicted_rb = (
            prediction["physical_edge_state_mean"][:, horizon, :, rb_index] * rb_scale
            + edge_mean[rb_index]
        )
        aggregate_rb = target.get(
            "aggregate_rb_occupancy", target["physical_edge_state"][..., rb_index]
        )
        aggregate_rb_mask = target.get("aggregate_rb_occupancy_mask")
        if aggregate_rb_mask is None:
            aggregate_rb_mask = edge_valid[:, None, :].expand_as(aggregate_rb)
        rb_observed = edge_valid & aggregate_rb_mask[:, horizon].bool()
        if "aggregate_rb_occupancy" in target:
            target_rb = aggregate_rb[:, horizon]
        else:
            target_rb = aggregate_rb[:, horizon] * rb_scale + edge_mean[rb_index]
        rb_error = torch.abs((predicted_rb * rb_observed).sum(dim=1) - (target_rb * rb_observed).sum(dim=1))
        bucket.system["throughput_abs"] += float(throughput_error.sum())
        bucket.system["rb_abs"] += float(rb_error.sum())
        bucket.system["aggregate_count"] += int(throughput_error.numel())

    def update(
        self,
        prediction: Mapping[str, torch.Tensor],
        target: Mapping[str, torch.Tensor],
        static: Mapping[str, torch.Tensor],
    ) -> None:
        horizon_steps = int(target["node_state"].shape[1])
        while len(self._horizon_buckets) < horizon_steps:
            self._horizon_buckets.append(_MetricBucket())
        self.sample_count += int(target["node_state"].shape[0])
        for horizon in range(horizon_steps):
            self._update_bucket(self._horizon_buckets[horizon], prediction, target, static, horizon)
            self._update_bucket(self._overall, prediction, target, static, horizon)

    def _finalize_bucket(self, bucket: _MetricBucket) -> dict[str, Any]:
        metrics: dict[str, dict[str, Any]] = {}
        for name, features in {**FEATURES, "task_dag": DAG_FEATURES}.items():
            state_key = "task_dag_state" if name == "task_dag" else f"{name}_state"
            present_key = "task_dag_state_present" if name == "task_dag" else f"{name}_present"
            values = bucket.state[name]
            count = int(values["count"])
            for index, feature in enumerate(features):
                sources = [state_key, present_key]
                unit = UNITS[name][index]
                if count:
                    mae = values["abs"][index] / count
                    rmse = math.sqrt(values["sq"][index] / count)
                    nll = values["nll"][index] / count
                    coverage = values["covered"][index] / count
                    width = values["width"][index] / count
                else:
                    mae = rmse = nll = coverage = width = None
                reason = "no valid target entities" if count == 0 else None
                metrics[f"state.{name}.{feature}.mae"] = _record(
                    mae, numerator=values["abs"][index], denominator=count, count=count, unit=unit, sources=sources, reason=reason
                )
                metrics[f"state.{name}.{feature}.rmse"] = _record(
                    rmse, numerator=values["sq"][index], denominator=count, count=count, unit=unit, sources=sources, reason=reason
                )
                metrics[f"uncertainty.{name}.{feature}.nll"] = _record(
                    nll, numerator=values["nll"][index], denominator=count, count=count, unit="nats", sources=sources, reason=reason
                )
                metrics[f"uncertainty.{name}.{feature}.coverage_95"] = _record(
                    coverage, numerator=values["covered"][index], denominator=count, count=count, unit="ratio", sources=sources, reason=reason
                )
                metrics[f"uncertainty.{name}.{feature}.interval_width_95"] = _record(
                    width, numerator=values["width"][index], denominator=count, count=count, unit=unit, sources=sources, reason=reason
                )

        binary_sources = {
            "link_activity": ["aggregate_link_activity", "aggregate_link_activity_mask", "physical_edge_endpoint_index"],
            "flow_present": ["flow_present", "flow_valid"],
            "task_present": ["task_present", "task_valid"],
            "dag_release": ["task_dag_state", "task_dag_state_present"],
            "dag_edge_present": ["dag_edge_present", "dag_edge_valid"],
        }
        binary_prefix = {
            "link_activity": "event.link_activity",
            "flow_present": "event.flow_present",
            "task_present": "event.task_present",
            "dag_release": "dag.release_ready",
            "dag_edge_present": "dag.edge_presence",
        }
        for name, item in bucket.binary.items():
            metrics.update(
                _classification_records(
                    binary_prefix[name], item, item["scores"], item["labels"], binary_sources[name]
                )
            )
        link_probability_sources = [
            "aggregate_link_activity",
            "aggregate_link_activity_mask",
            "physical_edge_endpoint_index",
        ]
        if self.link_probability_calibration is None:
            for name in ("nll", "brier", "ece"):
                metrics[f"event.link_activity.{name}"] = not_computable(
                    "raw_weighted_score is not formal event probability",
                    link_probability_sources,
                )
        elif bucket.binary["link_activity"]["valid"]:
            item = bucket.binary["link_activity"]
            probabilities = torch.from_numpy(np.concatenate(item["probabilities"]))
            labels = torch.from_numpy(np.concatenate(item["labels"]))
            probability_values = binary_probability_metrics(probabilities, labels)
            count = int(probability_values["count"])
            for name in ("nll", "brier", "ece"):
                value = float(probability_values[name])
                metrics[f"event.link_activity.{name}"] = _record(
                    value,
                    numerator=value * count,
                    denominator=count,
                    count=count,
                    unit="nats" if name == "nll" else "ratio",
                    sources=link_probability_sources,
                )
        else:
            for name in ("nll", "brier", "ece"):
                metrics[f"event.link_activity.{name}"] = not_computable(
                    "no valid link activity labels",
                    link_probability_sources,
                )

        support = bucket.lifecycle.sum(axis=1)
        total = int(support.sum())
        correct = int(np.trace(bucket.lifecycle))
        class_f1: list[float] = []
        for index in range(5):
            tp = int(bucket.lifecycle[index, index])
            fp = int(bucket.lifecycle[:, index].sum() - tp)
            fn = int(bucket.lifecycle[index, :].sum() - tp)
            denominator = 2 * tp + fp + fn
            if support[index] > 0 and denominator:
                class_f1.append(2 * tp / denominator)
        accuracy = correct / total if total else None
        macro_f1 = float(np.mean(class_f1)) if class_f1 else None
        metrics["task.lifecycle.accuracy"] = _record(
            accuracy, numerator=correct, denominator=total, count=total, unit="ratio", sources=["task_lifecycle_index", "task_present"], reason="no valid lifecycle labels" if not total else None
        )
        metrics["task.lifecycle.macro_f1"] = _record(
            macro_f1, numerator=None, denominator=len(class_f1), count=total, unit="ratio", sources=["task_lifecycle_index", "task_present"], reason="no supported lifecycle classes" if not class_f1 else None
        )
        metrics["task.lifecycle.support"] = {
            "value": support.astype(int).tolist(),
            "status": "computed",
            "numerator": None,
            "denominator": total,
            "count": total,
            "unit": "count",
            "source_fields": ["task_lifecycle_index", "task_present"],
            "reason": None,
        }

        rate_count = bucket.rate_count
        rate_mae = bucket.rate_abs / rate_count if rate_count else None
        rate_rmse = math.sqrt(bucket.rate_sq / rate_count) if rate_count else None
        metrics["link.active_only_rate.mae"] = _record(
            rate_mae, numerator=bucket.rate_abs, denominator=rate_count, count=rate_count, unit="Mbps", sources=["aggregate_link_rate_sum", "aggregate_link_rate_sum_mask", "aggregate_link_activity"], reason="no observed active target links" if not rate_count else None
        )
        metrics["link.active_only_rate.rmse"] = _record(
            rate_rmse, numerator=bucket.rate_sq, denominator=rate_count, count=rate_count, unit="Mbps", sources=["aggregate_link_rate_sum", "aggregate_link_rate_sum_mask", "aggregate_link_activity"], reason="no observed active target links" if not rate_count else None
        )
        unfinished = bucket.state["task_dag"]
        unfinished_count = int(unfinished["count"])
        unfinished_abs = unfinished["abs"][1]
        metrics["dag.unfinished_parent_count.mae"] = _record(
            unfinished_abs / unfinished_count if unfinished_count else None,
            numerator=unfinished_abs,
            denominator=unfinished_count,
            count=unfinished_count,
            unit="count",
            sources=["task_dag_state.unfinished_parent_count", "task_dag_state_present"],
            reason="no valid DAG task states" if not unfinished_count else None,
        )

        system = bucket.system
        completion_valid = int(system["completion_valid"])
        completion_error = (
            abs(system["completion_predicted"] - system["completion_truth"]) / completion_valid
            if completion_valid
            else None
        )
        metrics["system.task_completion_rate.absolute_error"] = _record(
            completion_error,
            numerator=abs(system["completion_predicted"] - system["completion_truth"]),
            denominator=completion_valid,
            count=completion_valid,
            unit="ratio",
            sources=["task_lifecycle_logits", "task_lifecycle_index"],
            reason="no valid lifecycle labels" if not completion_valid else None,
        )
        aggregate_count = int(system["aggregate_count"])
        metrics["system.communication_throughput.mae"] = _record(
            system["throughput_abs"] / aggregate_count if aggregate_count else None,
            numerator=system["throughput_abs"], denominator=aggregate_count, count=aggregate_count, unit="Mbps", sources=["aggregate_link_rate_sum", "aggregate_link_rate_sum_mask"], reason="no observed rollout steps" if not aggregate_count else None
        )
        metrics["resource.rb_occupancy.mae"] = _record(
            system["rb_abs"] / aggregate_count if aggregate_count else None,
            numerator=system["rb_abs"], denominator=aggregate_count, count=aggregate_count, unit="RB", sources=["aggregate_rb_occupancy", "aggregate_rb_occupancy_mask"], reason="no observed rollout steps" if not aggregate_count else None
        )

        unavailable = {
            "system.p95_latency": ("task completion timestamps cannot be reconstructed from a fixed three-step state head", ["task_completion_timestamp"]),
            "system.p99_latency": ("task completion timestamps cannot be reconstructed from a fixed three-step state head", ["task_completion_timestamp"]),
            "system.energy": ("the formal rollout target does not contain realized per-step energy", ["realized_energy"]),
            "system.fairness": ("a frozen user-level fairness population and aggregation window are not present", ["user_service_history"]),
            "decision.action_regret": ("counterfactual outcomes for alternative actions are not available", ["counterfactual_action_outcomes"]),
        }
        for name, (reason, fields) in unavailable.items():
            metrics[name] = not_computable(reason, fields)
        if not self.distribution_available:
            for name in list(metrics):
                if name.startswith("uncertainty."):
                    metrics[name] = not_computable(
                        "deterministic baseline does not output a predictive distribution",
                        metrics[name]["source_fields"],
                    )
        return metrics

    def finalize(self) -> dict[str, Any]:
        horizons = {
            f"k={index + 1}": {"metrics": self._finalize_bucket(bucket)}
            for index, bucket in enumerate(self._horizon_buckets)
        }
        horizons["overall"] = {"metrics": self._finalize_bucket(self._overall)}
        return {
            "schema_version": "PI-JWM-formal-metrics-v1",
            "sample_count": self.sample_count,
            "threshold": self.threshold,
            "thresholds": self.thresholds,
            "legacy_raw_thresholds": (
                None
                if self.link_probability_calibration is None
                else {"link_activity": self.legacy_raw_thresholds["link_activity"]}
            ),
            "mapped_probability_thresholds": (
                None
                if self.link_probability_calibration is None
                else {"link_activity": self.thresholds["link_activity"]}
            ),
            "link_probability_calibration": (
                None
                if self.link_probability_calibration is None
                else {
                    "pos_weight": self.link_probability_calibration.pos_weight,
                    "log_temperature": self.link_probability_calibration.log_temperature,
                    "temperature": self.link_probability_calibration.temperature,
                }
            ),
            "threshold_coordinate": {
                name: (
                    "event_probability"
                    if name == "link_activity"
                    and self.link_probability_calibration is not None
                    else (
                        "raw_weighted_score"
                        if name == "link_activity"
                        else "legacy_sigmoid_score"
                    )
                )
                for name in BINARY_EVENT_NAMES
            },
            "distribution_available": self.distribution_available,
            "registry": metric_registry(),
            "horizons": horizons,
        }


__all__ = ["FormalMetricAccumulator", "metric_registry", "not_computable"]
