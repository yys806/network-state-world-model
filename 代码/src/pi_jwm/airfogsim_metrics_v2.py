"""Directly computable PI-JWM metrics for AirFogSim dual-graph v2 data."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any


SCHEMA_VERSION = "PI-JWM-AirFogSim-metrics-v2"


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _metric(
    name: str,
    value: float | int | None,
    *,
    numerator: float | int | None,
    denominator: float | int | None,
    sample_count: int,
    unit: str,
    status: str = "available",
    source: str,
    reason: str | None = None,
) -> dict[str, Any]:
    row = {
        "name": name,
        "value": value,
        "numerator": numerator,
        "denominator": denominator,
        "sample_count": int(sample_count),
        "unit": unit,
        "status": status,
        "source": source,
    }
    if reason is not None:
        row["reason"] = reason
    return row


def _ratio_metric(
    name: str,
    numerator: float | int,
    denominator: float | int,
    *,
    sample_count: int,
    source: str,
) -> dict[str, Any]:
    if denominator <= 0:
        return _metric(
            name,
            None,
            numerator=numerator,
            denominator=denominator,
            sample_count=sample_count,
            unit="ratio",
            status="not_computable",
            source=source,
            reason="nonpositive_denominator",
        )
    return _metric(
        name,
        float(numerator) / float(denominator),
        numerator=numerator,
        denominator=denominator,
        sample_count=sample_count,
        unit="ratio",
        source=source,
    )


def _time_values(data: Mapping[str, Any]) -> list[float]:
    values = {
        float(row["observed_time"])
        for row in data.get("physical_node_snapshots", [])
        if row.get("observed_time") is not None
    }
    return sorted(values)


def _simulation_interval(data: Mapping[str, Any], time_values: list[float]) -> float | None:
    configured = data.get("simulation_interval")
    if configured is not None and float(configured) > 0:
        return float(configured)
    differences = [right - left for left, right in zip(time_values, time_values[1:]) if right > left]
    return min(differences) if differences else None


def _task_metrics(data: Mapping[str, Any], evaluation_end: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tasks = list(data.get("task_records", []))
    evaluable = [
        row
        for row in tasks
        if str(row.get("terminal_status")) in {"completed", "failed"}
        or (
            row.get("deadline_time") is not None
            and float(row["deadline_time"]) <= evaluation_end
        )
    ]
    completed = [row for row in evaluable if str(row.get("terminal_status")) == "completed"]
    failed = [row for row in evaluable if str(row.get("terminal_status")) != "completed"]
    deadline_violations = [
        row
        for row in evaluable
        if (
            "deadline" in str(row.get("failure_reason", "")).lower()
            or (
                str(row.get("terminal_status")) == "pending"
                and row.get("deadline_time") is not None
                and float(row["deadline_time"]) <= evaluation_end
            )
        )
    ]
    delays = [float(row["task_delay"]) for row in completed if row.get("task_delay") is not None]
    total_priority = sum(float(row.get("priority", 1.0)) for row in evaluable)
    completed_priority = sum(float(row.get("priority", 1.0)) for row in completed)

    metrics = [
        _ratio_metric(
            "task_completion_rate",
            len(completed),
            len(evaluable),
            sample_count=len(evaluable),
            source="AirFogSim Task terminal_status",
        ),
        _ratio_metric(
            "task_failure_rate",
            len(failed),
            len(evaluable),
            sample_count=len(evaluable),
            source="AirFogSim Task terminal_status and observation deadline",
        ),
        _ratio_metric(
            "deadline_violation_rate",
            len(deadline_violations),
            len(evaluable),
            sample_count=len(evaluable),
            source="AirFogSim Task failure_reason and deadline_time",
        ),
        _ratio_metric(
            "priority_weighted_completion_rate",
            completed_priority,
            total_priority,
            sample_count=len(evaluable),
            source="AirFogSim Task priority and terminal_status",
        ),
    ]
    for name, value in (
        ("successful_task_delay_mean", sum(delays) / len(delays) if delays else None),
        ("successful_task_delay_p95", _percentile(delays, 0.95)),
        ("successful_task_delay_p99", _percentile(delays, 0.99)),
    ):
        metrics.append(
            _metric(
                name,
                value,
                numerator=sum(delays) if name.endswith("mean") and delays else None,
                denominator=len(delays) if name.endswith("mean") and delays else None,
                sample_count=len(delays),
                unit="seconds",
                status="available" if delays else "not_computable",
                source="AirFogSim completed Task last operation minus arrival",
                reason=None if delays else "no_completed_task_delay",
            )
        )

    source_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for row in evaluable:
        source = str(row.get("source", "unknown"))
        source_counts[source][1] += 1
        source_counts[source][0] += str(row.get("terminal_status")) == "completed"
    source_rates = [completed_count / total for completed_count, total in source_counts.values() if total]
    fairness_denominator = len(source_rates) * sum(rate * rate for rate in source_rates)
    fairness = (sum(source_rates) ** 2 / fairness_denominator) if fairness_denominator > 0 else None
    metrics.append(
        _metric(
            "completion_fairness_jain",
            fairness,
            numerator=sum(source_rates) ** 2 if source_rates else None,
            denominator=fairness_denominator if source_rates else None,
            sample_count=len(source_rates),
            unit="ratio",
            status="available" if fairness is not None else "not_computable",
            source="per-source evaluable-task completion rates",
            reason=None if fairness is not None else "no_positive_per_source_completion_rate",
        )
    )
    return metrics, {
        "total_task_records": len(tasks),
        "evaluable_tasks": len(evaluable),
        "completed_tasks": len(completed),
        "failed_or_expired_tasks": len(failed),
        "right_censored_tasks": len(tasks) - len(evaluable),
    }


def _resource_metrics(
    data: Mapping[str, Any],
    *,
    evaluation_end: float,
    time_values: list[float],
    interval: float | None,
) -> list[dict[str, Any]]:
    events = list(data.get("transfer_events", []))
    delivered = sum(float(row.get("delivered_data", 0.0)) for row in events)
    metrics = [
        _ratio_metric(
            "information_throughput",
            delivered,
            evaluation_end,
            sample_count=len(events),
            source="AirFogSim runtime transfer events",
        )
    ]
    metrics[-1]["unit"] = "MB/s"

    edge_snapshots = list(data.get("physical_edge_snapshots", []))
    active_edges = sum(int(row.get("active_task_count", 0)) > 0 for row in edge_snapshots)
    metrics.append(
        _ratio_metric(
            "physical_link_active_ratio",
            active_edges,
            len(edge_snapshots),
            sample_count=len(edge_snapshots),
            source="AirFogSim physical edge snapshots",
        )
    )

    rb_rows = list(data.get("rb_ledger", []))
    n_rb_values = {int(row["n_rb"]) for row in rb_rows if row.get("n_rb") is not None}
    rb_slots = len(time_values)
    rb_capacity = next(iter(n_rb_values)) * rb_slots if len(n_rb_values) == 1 else 0
    rb_used = sum(len(row.get("rb_indices", [])) for row in rb_rows)
    rb_metric = _ratio_metric(
        "rb_utilization",
        rb_used,
        rb_capacity,
        sample_count=len(rb_rows),
        source="AirFogSim RB ledger over all simulated slots",
    )
    if len(n_rb_values) != 1:
        rb_metric.update(status="not_computable", value=None, reason="inconsistent_or_missing_rb_capacity")
    metrics.append(rb_metric)

    node_snapshots = list(data.get("physical_node_snapshots", []))
    cpu_rows = list(data.get("cpu_ledger", []))
    cpu_used = sum(float(row.get("allocated_cpu", 0.0)) * float(row.get("dt", 0.0)) for row in cpu_rows)
    cpu_capacity = (
        sum(float(row.get("cpu", 0.0) or 0.0) for row in node_snapshots) * float(interval)
        if interval is not None
        else 0.0
    )
    metrics.append(
        _ratio_metric(
            "cpu_utilization",
            cpu_used,
            cpu_capacity,
            sample_count=len(cpu_rows),
            source="AirFogSim CPU allocation ledger and per-slot node capacities",
        )
    )

    energy_rows = list(data.get("uav_energy_ledger", []))
    energy_used = sum(
        float(row.get("energy_before", 0.0)) - float(row.get("energy_after", 0.0))
        for row in energy_rows
    )
    metrics.append(
        _metric(
            "uav_energy_total",
            energy_used if energy_rows else None,
            numerator=energy_used if energy_rows else None,
            denominator=None,
            sample_count=len(energy_rows),
            unit="AirFogSim energy unit",
            status="available" if energy_rows else "not_computable",
            source="AirFogSim UAV energy manager before/after delta",
            reason=None if energy_rows else "no_uav_energy_rows",
        )
    )
    return metrics


def _constraint_metrics(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    tolerance = 1e-8
    task_rows = list(data.get("task_ledger", []))
    task_violations = sum(
        abs(
            float(row.get("remaining_before", 0.0))
            - float(row.get("delivered_data", 0.0))
            - float(row.get("remaining_after", 0.0))
        )
        > tolerance
        for row in task_rows
    )

    cpu_groups: dict[tuple[float, str], list[dict[str, Any]]] = defaultdict(list)
    for row in data.get("cpu_ledger", []):
        cpu_groups[(float(row.get("time", 0.0)), str(row.get("node_id", "")))].append(row)
    cpu_violations = 0
    for rows in cpu_groups.values():
        allocation = sum(float(row.get("allocated_cpu", 0.0)) for row in rows)
        capacities = {float(row.get("node_cpu_capacity", 0.0)) for row in rows}
        cpu_violations += len(capacities) != 1 or allocation > next(iter(capacities), 0.0) + tolerance

    energy_rows = list(data.get("uav_energy_ledger", []))
    energy_violations = 0
    for row in energy_rows:
        observed = float(row.get("energy_before", 0.0)) - float(row.get("energy_after", 0.0))
        expected = (
            float(row.get("is_flying", 0.0)) * float(row.get("fly_unit_cost", 0.0))
            + float(row.get("is_hovering", 0.0)) * float(row.get("hover_unit_cost", 0.0))
            + float(row.get("using_sensor_num", 0.0)) * float(row.get("sensing_unit_cost", 0.0))
            + float(row.get("sending_data_size", 0.0)) * float(row.get("send_unit_cost", 0.0))
            + float(row.get("receiving_data_size", 0.0)) * float(row.get("receive_unit_cost", 0.0))
        )
        energy_violations += abs(observed - expected) > tolerance

    return [
        _ratio_metric(
            "task_flow_conservation_violation_rate",
            task_violations,
            len(task_rows),
            sample_count=len(task_rows),
            source="task ledger conservation residual",
        ),
        _ratio_metric(
            "cpu_capacity_violation_rate",
            cpu_violations,
            len(cpu_groups),
            sample_count=len(cpu_groups),
            source="grouped CPU allocation versus node capacity",
        ),
        _ratio_metric(
            "energy_equation_violation_rate",
            energy_violations,
            len(energy_rows),
            sample_count=len(energy_rows),
            source="UAV energy before/after equation",
        ),
    ]


def _dag_metrics(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    dag_edges = list(data.get("task_dag_edges", []))
    dependency_flows = list(data.get("dependency_data_flows", []))
    coverage = _ratio_metric(
        "dependency_payload_coverage",
        len(dependency_flows),
        len(dag_edges),
        sample_count=len(dag_edges),
        source="explicit dependency_data flows over AirFogSim DAG edges",
    )
    if dependency_flows:
        completed = sum(str(row.get("status")) == "completed" for row in dependency_flows)
        delivery = _ratio_metric(
            "dependency_data_delivery_rate",
            completed,
            len(dependency_flows),
            sample_count=len(dependency_flows),
            source="explicit dependency_data flow status",
        )
    else:
        delivery = _metric(
            "dependency_data_delivery_rate",
            None,
            numerator=0,
            denominator=0,
            sample_count=0,
            unit="ratio",
            status="not_applicable",
            source="explicit dependency_data flow status",
            reason="AirFogSim DAG has precedence but no explicit dependency payload",
        )
    return [coverage, delivery]


def compute_airfogsim_metrics_v2(data: Mapping[str, Any]) -> dict[str, Any]:
    """Compute only metrics supported by direct simulator evidence."""

    time_values = _time_values(data)
    evaluation_end = float(
        data.get(
            "evaluation_end_time",
            max(time_values, default=max((float(row.get("observed_time", 0.0)) for row in data.get("task_records", [])), default=0.0)),
        )
    )
    interval = _simulation_interval(data, time_values)
    task_metrics, task_summary = _task_metrics(data, evaluation_end)
    metrics = task_metrics
    metrics.extend(
        _resource_metrics(
            data,
            evaluation_end=evaluation_end,
            time_values=time_values,
            interval=interval,
        )
    )
    metrics.extend(_constraint_metrics(data))
    metrics.extend(_dag_metrics(data))
    for name, reason in (
        ("uncertainty_coverage", "no predictive distribution or calibrated interval in simulator evidence"),
        ("action_regret", "no counterfactual outcomes for alternative actions in the same state"),
        ("ood_transfer_score", "single seed and scenario do not constitute an OOD split"),
    ):
        metrics.append(
            _metric(
                name,
                None,
                numerator=None,
                denominator=None,
                sample_count=0,
                unit="not available",
                status="not_computable",
                source="required PI-JWM evaluation interface",
                reason=reason,
            )
        )

    by_name = {row["name"]: row for row in metrics}
    completed_count = task_summary["completed_tasks"]
    energy_total = by_name["uav_energy_total"]["value"]
    energy_per_completed = (
        float(energy_total) / completed_count
        if energy_total is not None and completed_count > 0
        else None
    )
    metrics.append(
        _metric(
            "uav_energy_per_completed_task",
            energy_per_completed,
            numerator=energy_total,
            denominator=completed_count,
            sample_count=completed_count,
            unit="AirFogSim energy unit/task",
            status="available" if energy_per_completed is not None else "not_computable",
            source="UAV energy total divided by completed evaluable tasks",
            reason=None if energy_per_completed is not None else "no_completed_task_or_energy_rows",
        )
    )
    by_name = {row["name"]: row for row in metrics}
    completion = by_name["task_completion_rate"]["value"]
    fairness = by_name["completion_fairness_jain"]["value"]
    objectives = {
        "ordering": "lexicographic",
        "primary": {
            "name": "service_loss",
            "definition": "1 - task_completion_rate over evaluable tasks",
            "value": None if completion is None else 1.0 - float(completion),
        },
        "secondary": [
            {"name": "successful_task_delay_mean", "value": by_name["successful_task_delay_mean"]["value"]},
            {"name": "uav_energy_per_completed_task", "value": energy_per_completed},
            {"name": "rb_utilization", "value": by_name["rb_utilization"]["value"]},
            {"name": "cpu_utilization", "value": by_name["cpu_utilization"]["value"]},
            {"name": "fairness_loss", "value": None if fairness is None else 1.0 - float(fairness)},
        ],
        "hard_constraints": [
            "task_flow_conservation_violation_rate == 0",
            "cpu_capacity_violation_rate == 0",
            "energy_equation_violation_rate == 0",
        ],
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "evaluation_window": {
            "end_time": evaluation_end,
            "simulation_interval": interval,
            "observed_slots": len(time_values),
            **task_summary,
        },
        "metrics": metrics,
        "optimization_objectives": objectives,
    }
