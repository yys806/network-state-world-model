"""Machine-readable numeric checkpoint gates for formal non-locked P4 selection."""

from __future__ import annotations

from typing import Any, Mapping


SCHEMA_VERSION = "PI-JWM-formal-P4-checkpoint-gate-v1"


def _value(report: Mapping[str, Any], metric: str, horizon: str = "overall") -> float:
    row = report["horizons"][horizon]["metrics"][metric]
    if row.get("status") != "computed" or row.get("value") is None:
        raise ValueError(f"required P4 metric is unavailable: {horizon}/{metric}")
    return float(row["value"])


def _ratio(value: float, baseline: float) -> float:
    if baseline < 0:
        raise ValueError("MAE baseline must be non-negative")
    if baseline == 0:
        return 1.0 if value == 0 else float("inf")
    return value / baseline


def _lower_gate(name: str, value: float, minimum: float) -> dict[str, Any]:
    scale = max(abs(minimum), 1e-6)
    return {
        "name": name,
        "direction": "minimum",
        "value": value,
        "threshold": minimum,
        "passed": value >= minimum,
        "normalized_excess": max(0.0, minimum - value) / scale,
    }


def _upper_gate(name: str, value: float, maximum: float) -> dict[str, Any]:
    scale = max(abs(maximum), 1e-6)
    return {
        "name": name,
        "direction": "maximum",
        "value": value,
        "threshold": maximum,
        "passed": value <= maximum,
        "normalized_excess": max(0.0, value - maximum) / scale,
    }


def evaluate_p4_checkpoint_gates(
    *,
    validation: Mapping[str, Any],
    calibration: Mapping[str, Any],
    persistence_validation: Mapping[str, Any],
    persistence_calibration: Mapping[str, Any],
    validation_state_nll: float,
) -> dict[str, Any]:
    """Rank one checkpoint with the fixed P4 non-locked numeric gates."""

    val_link_delta = _value(validation, "event.link_activity.f1") - _value(
        persistence_validation, "event.link_activity.f1"
    )
    cal_link_delta = _value(calibration, "event.link_activity.f1") - _value(
        persistence_calibration, "event.link_activity.f1"
    )
    gates = [
        _lower_gate("validation_link_f1_delta", val_link_delta, -0.05),
        _lower_gate("calibration_link_f1_delta", cal_link_delta, 0.05),
        _upper_gate(
            "validation_node_x_mae_ratio",
            _ratio(
                _value(validation, "state.node.x.mae"),
                _value(persistence_validation, "state.node.x.mae"),
            ),
            1.25,
        ),
    ]
    for horizon in (5, 10, 20):
        gates.append(
            _upper_gate(
                f"validation_node_x_h{horizon}_mae_ratio",
                _ratio(
                    _value(validation, "state.node.x.mae", f"k={horizon}"),
                    _value(
                        persistence_validation,
                        "state.node.x.mae",
                        f"k={horizon}",
                    ),
                ),
                1.0,
            )
        )
    for name, metric in (
        ("validation_throughput_mae_ratio", "system.communication_throughput.mae"),
        ("validation_rb_occupancy_mae_ratio", "resource.rb_occupancy.mae"),
        ("validation_task_delay_mae_ratio", "state.task.delay.mae"),
    ):
        gates.append(
            _upper_gate(
                name,
                _ratio(_value(validation, metric), _value(persistence_validation, metric)),
                1.0,
            )
        )
    failed = [row["name"] for row in gates if not row["passed"]]
    maximum_excess = max((float(row["normalized_excess"]) for row in gates), default=0.0)
    return {
        "schema_version": SCHEMA_VERSION,
        "gates": gates,
        "failed_hard_gate_count": len(failed),
        "failed_hard_gates": failed,
        "maximum_normalized_gate_excess": maximum_excess,
        "validation_state_nll": float(validation_state_nll),
        "rank_key": [len(failed), maximum_excess, float(validation_state_nll)],
        "all_numeric_gates_passed": not failed,
        "scope": "single-seed checkpoint selection; cross-seed gates remain pending",
        "locked_test_accessed": False,
    }


__all__ = ["SCHEMA_VERSION", "evaluate_p4_checkpoint_gates"]
