"""Trajectory-level real system-outcome metrics for formal PI-JWM evaluation."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


SCHEMA_VERSION = "PI-JWM-formal-system-outcome-metrics-v1"


def _record(
    value: float | None,
    *,
    numerator: float | int | None,
    denominator: float | int | None,
    count: int,
    unit: str,
    sources: list[str],
    reason: str | None = None,
) -> dict[str, Any]:
    computed = value is not None and math.isfinite(float(value))
    return {
        "value": float(value) if computed else None,
        "status": "computed" if computed else "not_computable",
        "numerator": float(numerator) if numerator is not None else None,
        "denominator": float(denominator) if denominator is not None else None,
        "count": int(count),
        "unit": unit,
        "source_fields": sources,
        "reason": None if computed else reason,
    }


def _jain(values: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return None
    denominator = len(values) * float(np.square(values).sum())
    if denominator <= 0:
        return 0.0
    return float(values.sum() ** 2 / denominator)


def _require_same_shape(name: str, left: np.ndarray, right: np.ndarray) -> None:
    if left.shape != right.shape:
        raise ValueError(f"{name} arrays must have identical shapes")


def compute_system_outcome_metrics(
    *,
    true_completion_event: np.ndarray,
    predicted_completion_event: np.ndarray,
    true_completed_delay: np.ndarray,
    predicted_task_delay: np.ndarray,
    completed_delay_valid: np.ndarray,
    true_delivered_data: np.ndarray,
    predicted_delivered_data: np.ndarray,
    step_seconds: float,
    true_uav_energy: np.ndarray,
    predicted_uav_energy: np.ndarray | None,
    uav_energy_valid: np.ndarray,
    true_source_service: np.ndarray,
    predicted_source_service: np.ndarray,
    source_population_valid: np.ndarray,
    source_evaluable_task_count: np.ndarray,
) -> dict[str, Any]:
    """Compute auditable event and system metrics from one trajectory stream."""

    if not math.isfinite(float(step_seconds)) or float(step_seconds) <= 0:
        raise ValueError("step_seconds must be finite and positive")
    truth_event = np.asarray(true_completion_event, dtype=bool)
    predicted_event = np.asarray(predicted_completion_event, dtype=bool)
    _require_same_shape("completion event", truth_event, predicted_event)
    true_delay = np.asarray(true_completed_delay, dtype=np.float64)
    predicted_delay = np.asarray(predicted_task_delay, dtype=np.float64)
    delay_valid = np.asarray(completed_delay_valid, dtype=bool)
    for value in (true_delay, predicted_delay, delay_valid):
        _require_same_shape("completion delay", truth_event, value)

    metrics: dict[str, dict[str, Any]] = {}
    tp = int(np.count_nonzero(truth_event & predicted_event))
    fp = int(np.count_nonzero(~truth_event & predicted_event))
    fn = int(np.count_nonzero(truth_event & ~predicted_event))
    event_denominator = 2 * tp + fp + fn
    metrics["event.task_completion.f1"] = _record(
        2 * tp / event_denominator if event_denominator else None,
        numerator=2 * tp,
        denominator=event_denominator,
        count=int(truth_event.size),
        unit="ratio",
        sources=["task_completion_event", "task_lifecycle_logits"],
        reason="no true or predicted completion events" if not event_denominator else None,
    )

    aligned_mask = truth_event & delay_valid
    aligned_count = int(aligned_mask.sum())
    aligned_abs = float(np.abs(predicted_delay[aligned_mask] - true_delay[aligned_mask]).sum())
    metrics["system.latency.completed_task_delay.mae"] = _record(
        aligned_abs / aligned_count if aligned_count else None,
        numerator=aligned_abs,
        denominator=aligned_count,
        count=aligned_count,
        unit="s",
        sources=["completed_task_delay", "task_state.delay"],
        reason="no completed tasks with direct delay" if not aligned_count else None,
    )

    true_latency = true_delay[aligned_mask]
    predicted_latency = predicted_delay[predicted_event]
    predicted_latency = predicted_latency[np.isfinite(predicted_latency) & (predicted_latency >= 0)]
    latency_specs = {
        "mean": lambda value: float(np.mean(value)),
        "p95": lambda value: float(np.quantile(value, 0.95)),
        "p99": lambda value: float(np.quantile(value, 0.99)),
    }
    for name, aggregate in latency_specs.items():
        computable = len(true_latency) > 0 and len(predicted_latency) > 0
        absolute_error = (
            abs(aggregate(predicted_latency) - aggregate(true_latency)) if computable else None
        )
        metrics[f"system.latency.{name}.absolute_error"] = _record(
            absolute_error,
            numerator=absolute_error,
            denominator=1 if computable else None,
            count=min(len(true_latency), len(predicted_latency)),
            unit="s",
            sources=["completed_task_delay", "predicted_completion_event", "task_state.delay"],
            reason="true and predicted completed-task latency samples are required" if not computable else None,
        )

    true_data = np.asarray(true_delivered_data, dtype=np.float64)
    predicted_data = np.asarray(predicted_delivered_data, dtype=np.float64)
    _require_same_shape("delivered data", true_data, predicted_data)
    throughput_error = np.abs(predicted_data - true_data) / float(step_seconds)
    metrics["system.application_throughput.mae"] = _record(
        float(throughput_error.mean()) if throughput_error.size else None,
        numerator=float(throughput_error.sum()),
        denominator=int(throughput_error.size),
        count=int(throughput_error.size),
        unit="MB/s",
        sources=["delivered_data_total", "flow_state.delivered_this_slot"],
        reason="no evaluated trajectory steps" if not throughput_error.size else None,
    )

    true_energy = np.asarray(true_uav_energy, dtype=np.float64)
    energy_valid = np.asarray(uav_energy_valid, dtype=bool)
    _require_same_shape("UAV energy", true_energy, energy_valid)
    energy_names = (
        "system.uav_energy.mae",
        "system.uav_energy.rmse",
        "system.uav_energy.total_absolute_error",
    )
    if predicted_uav_energy is None:
        for name in energy_names:
            metrics[name] = _record(
                None,
                numerator=None,
                denominator=None,
                count=0,
                unit="AirFogSim energy unit",
                sources=["uav_energy_delta", "uav_energy_delta_prediction"],
                reason="the model does not output a UAV energy prediction",
            )
    else:
        predicted_energy = np.asarray(predicted_uav_energy, dtype=np.float64)
        _require_same_shape("UAV energy", true_energy, predicted_energy)
        energy_count = int(energy_valid.sum())
        energy_error = predicted_energy[energy_valid] - true_energy[energy_valid]
        energy_abs = float(np.abs(energy_error).sum())
        energy_sq = float(np.square(energy_error).sum())
        total_error = abs(
            float(predicted_energy[energy_valid].sum()) - float(true_energy[energy_valid].sum())
        )
        reason = "no valid UAV energy targets" if not energy_count else None
        metrics[energy_names[0]] = _record(
            energy_abs / energy_count if energy_count else None,
            numerator=energy_abs,
            denominator=energy_count,
            count=energy_count,
            unit="AirFogSim energy unit",
            sources=["uav_energy_delta", "uav_energy_delta_prediction"],
            reason=reason,
        )
        metrics[energy_names[1]] = _record(
            math.sqrt(energy_sq / energy_count) if energy_count else None,
            numerator=energy_sq,
            denominator=energy_count,
            count=energy_count,
            unit="AirFogSim energy unit",
            sources=["uav_energy_delta", "uav_energy_delta_prediction"],
            reason=reason,
        )
        metrics[energy_names[2]] = _record(
            total_error if energy_count else None,
            numerator=total_error,
            denominator=1 if energy_count else None,
            count=energy_count,
            unit="AirFogSim energy unit",
            sources=["uav_energy_delta", "uav_energy_delta_prediction"],
            reason=reason,
        )

    true_service = np.asarray(true_source_service, dtype=np.float64)
    predicted_service = np.asarray(predicted_source_service, dtype=np.float64)
    _require_same_shape("source service", true_service, predicted_service)
    population = np.asarray(source_population_valid, dtype=bool)
    if true_service.ndim != 2 or population.shape != (true_service.shape[1],):
        raise ValueError("source population mask must match the service node dimension")
    task_count = np.asarray(source_evaluable_task_count, dtype=np.float64)
    if task_count.shape != population.shape or np.any(task_count < 0):
        raise ValueError("source task counts must match the non-negative source population")
    if np.any(population & (task_count <= 0)):
        raise ValueError("every valid task source must have a positive task count")
    true_rate = true_service.sum(axis=0)[population] / task_count[population]
    predicted_rate = predicted_service.sum(axis=0)[population] / task_count[population]
    true_fairness = _jain(true_rate)
    predicted_fairness = _jain(predicted_rate)
    fairness_computable = true_fairness is not None and predicted_fairness is not None
    fairness_error = (
        abs(float(predicted_fairness) - float(true_fairness)) if fairness_computable else None
    )
    metrics["system.completion_fairness_jain.absolute_error"] = _record(
        fairness_error,
        numerator=fairness_error,
        denominator=1 if fairness_computable else None,
        count=int(population.sum()),
        unit="ratio",
        sources=[
            "source_on_time_service_delta",
            "source_evaluable_task_count",
            "task_source_node_index",
        ],
        reason="a nonempty fixed source population and task counts are required" if not fairness_computable else None,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": "teacher_forced_one_step_event_evaluation",
        "metrics": metrics,
    }


__all__ = ["SCHEMA_VERSION", "compute_system_outcome_metrics"]
