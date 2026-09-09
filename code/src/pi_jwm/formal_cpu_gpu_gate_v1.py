"""Frozen non-locked CPU evidence gate for formal GPU launch review."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


DEFAULT_GATE = {
    "minimum_independent_seed_count": 3,
    "minimum_calibration_link_f1_delta": 0.05,
    "maximum_validation_link_f1_regression": 0.05,
    "maximum_validation_node_x_mae_ratio": 1.25,
}
OPERATIONAL_METRICS = (
    "throughput_mae",
    "rb_occupancy_mae",
    "task_delay_mae",
)


def _mean(rows: list[Mapping[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows]
    return sum(values) / len(values)


def audit_cpu_to_gpu_gate(
    rows: Iterable[Mapping[str, Any]],
    *,
    criteria: Mapping[str, float | int] | None = None,
) -> dict[str, Any]:
    """Evaluate only predeclared CPU evidence conditions; never reads locked-test."""

    rows = [dict(row) for row in rows]
    if not rows:
        raise ValueError("at least one non-locked seed row is required")
    gate = {**DEFAULT_GATE, **(dict(criteria) if criteria is not None else {})}
    failed: list[str] = []
    seed_count = len({str(row["seed"]) for row in rows})
    if seed_count < int(gate["minimum_independent_seed_count"]):
        failed.append("minimum_independent_seed_count")
    if any(row.get("threshold_selection_split") != "calibration" for row in rows):
        failed.append("calibration_only_threshold_selection")

    validation_deltas = [
        float(row["validation_link_f1"]) - float(row["validation_persistence_link_f1"])
        for row in rows
    ]
    calibration_deltas = [
        float(row["calibration_link_f1"]) - float(row["calibration_persistence_link_f1"])
        for row in rows
    ]
    if _mean(rows, "calibration_link_f1") - _mean(rows, "calibration_persistence_link_f1") < float(gate["minimum_calibration_link_f1_delta"]):
        failed.append("minimum_calibration_link_f1_delta")
    if min(validation_deltas) < -float(gate["maximum_validation_link_f1_regression"]):
        failed.append("maximum_per_seed_validation_link_f1_regression")
    if _mean(rows, "validation_link_f1") < _mean(rows, "validation_persistence_link_f1"):
        failed.append("mean_validation_link_f1_noninferiority")

    node_ratio = _mean(rows, "validation_node_x_mae") / _mean(rows, "validation_persistence_node_x_mae")
    if node_ratio > float(gate["maximum_validation_node_x_mae_ratio"]):
        failed.append("maximum_validation_node_x_mae_ratio")

    operational_deltas: dict[str, float] = {}
    for metric in OPERATIONAL_METRICS:
        delta = _mean(rows, f"validation_{metric}") - _mean(rows, f"validation_persistence_{metric}")
        operational_deltas[metric] = delta
        if delta > 0.0:
            failed.append(f"validation_{metric}_non_regression")

    return {
        "schema_version": "PI-JWM-formal-cpu-gpu-gate-v1",
        "gpu_allowed": not failed,
        "failed_gates": failed,
        "criteria": gate,
        "observed": {
            "independent_seed_count": seed_count,
            "threshold_selection_splits": sorted({str(row.get("threshold_selection_split")) for row in rows}),
            "validation_link_f1_deltas": validation_deltas,
            "calibration_link_f1_deltas": calibration_deltas,
            "mean_validation_link_f1_delta": _mean(rows, "validation_link_f1") - _mean(rows, "validation_persistence_link_f1"),
            "mean_calibration_link_f1_delta": _mean(rows, "calibration_link_f1") - _mean(rows, "calibration_persistence_link_f1"),
            "validation_node_x_mae_ratio": node_ratio,
            "validation_operational_metric_deltas": operational_deltas,
        },
        "execution_policy": {
            "locked_test_accessed": False,
            "gpu_started": False,
            "formal_performance_claim_ready": False,
        },
    }


__all__ = ["DEFAULT_GATE", "OPERATIONAL_METRICS", "audit_cpu_to_gpu_gate"]
