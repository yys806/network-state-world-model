"""Reusable metrics for PI-JWM counterfactual AirFogSim diagnostics."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence


ENERGY_COMPONENTS = ("fly", "hover", "sensing", "receive", "send")


def reward_components(
    delta_done: float,
    delta_failed: float,
    throughput_delta: float,
    throughput_weight: float = 0.01,
) -> dict[str, float]:
    """Return an auditable decomposition of the existing task utility."""
    throughput = max(0.0, float(throughput_delta))
    reward_done = float(delta_done)
    reward_failed = -float(delta_failed)
    reward_throughput = float(throughput_weight) * math.log1p(throughput)
    return {
        "delta_done": float(delta_done),
        "delta_failed": float(delta_failed),
        "throughput_delta": throughput,
        "reward_done": reward_done,
        "reward_failed": reward_failed,
        "reward_throughput": reward_throughput,
        "task_utility": reward_done + reward_failed + reward_throughput,
    }


def energy_snapshot_totals(snapshot: Mapping) -> dict[str, float | int]:
    """Aggregate a simulator energy snapshot while retaining removed UAVs."""
    totals = {f"energy_{name}": 0.0 for name in ENERGY_COMPONENTS}
    totals.update({"energy_remaining": 0.0, "energy_num_active": 0, "energy_num_removed": 0})
    for item in snapshot.get("uavs", {}).values():
        status = str(item["status"])
        if status not in {"active", "removed"}:
            raise ValueError(f"unknown UAV energy status: {status}")
        totals[f"energy_num_{status}"] += 1
        totals["energy_remaining"] += float(item["remaining_energy"])
        consumption = item.get("last_consumption", {})
        for name in ENERGY_COMPONENTS:
            value = float(consumption.get(name, 0.0))
            if value < 0.0:
                raise ValueError(f"negative energy consumption for {name}: {value}")
            totals[f"energy_{name}"] += value
    totals["energy_total"] = sum(float(totals[f"energy_{name}"]) for name in ENERGY_COMPONENTS)
    return totals


def energy_step_metrics(before: Mapping, after: Mapping) -> dict[str, float | int]:
    """Describe one simulator step and expose its energy-conservation error."""
    before_totals = energy_snapshot_totals(before)
    effective_after = {"uavs": {}}
    before_uavs = before.get("uavs", {})
    for uav_id, item in after.get("uavs", {}).items():
        previous = before_uavs.get(uav_id)
        changed = previous is None or not math.isclose(
            float(previous["remaining_energy"]),
            float(item["remaining_energy"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        effective_after["uavs"][uav_id] = {
            "status": item["status"],
            "remaining_energy": item["remaining_energy"],
            "last_consumption": item.get("last_consumption", {}) if changed else {},
        }
    after_totals = energy_snapshot_totals(effective_after)
    result = dict(after_totals)
    result["energy_before"] = float(before_totals["energy_remaining"])
    result["energy_after"] = float(after_totals["energy_remaining"])
    observed_delta = result["energy_before"] - result["energy_after"]
    result["energy_balance_error"] = float(observed_delta - result["energy_total"])
    return result


def summarize_candidate_steps(
    rows: Iterable[Mapping],
    group_fields: Sequence[str] = ("seed", "decision_time", "candidate_id", "action_family"),
) -> list[dict]:
    """Aggregate auditable step rows into one row per candidate rollout."""
    grouped: dict[tuple, list[Mapping]] = {}
    for row in rows:
        key = tuple(row[field] for field in group_fields)
        grouped.setdefault(key, []).append(row)

    sum_fields = (
        "delta_done",
        "delta_failed",
        "throughput_delta",
        "reward_done",
        "reward_failed",
        "reward_throughput",
        "task_utility",
        "rb_total",
        "cpu_total",
        "energy_fly",
        "energy_hover",
        "energy_sensing",
        "energy_receive",
        "energy_send",
        "energy_total",
    )
    summaries = []
    for key, group in grouped.items():
        ordered = sorted(group, key=lambda row: int(row["step"]))
        summary = {field: value for field, value in zip(group_fields, key)}
        summary["num_steps"] = len(ordered)
        for field in sum_fields:
            summary[field] = sum(float(row.get(field, 0.0)) for row in ordered)
        summary["energy_before"] = float(ordered[0].get("energy_before", 0.0))
        summary["energy_after"] = float(ordered[-1].get("energy_after", 0.0))
        summary["max_abs_energy_balance_error"] = max(
            abs(float(row.get("energy_balance_error", 0.0))) for row in ordered
        )
        summary["action_applied"] = bool(any(bool(row.get("action_applied", False)) for row in ordered))
        summaries.append(summary)
    return summaries


def audit_diagnostic_rows(rows: Iterable[Mapping], tolerance: float = 1e-8) -> dict[str, int | bool]:
    """Count data-quality violations without silently dropping affected rows."""
    rows = list(rows)
    missing_numeric = 0
    negative_energy = 0
    reward_errors = 0
    balance_errors = 0
    invalid_actions = 0
    numeric_fields = (
        "reward_done",
        "reward_failed",
        "reward_throughput",
        "task_utility",
        "energy_total",
        "energy_balance_error",
    )
    for row in rows:
        values = []
        for field in numeric_fields:
            try:
                value = float(row[field])
            except (KeyError, TypeError, ValueError):
                missing_numeric += 1
                continue
            if not math.isfinite(value):
                missing_numeric += 1
            values.append((field, value))
        if float(row.get("energy_total", 0.0)) < -tolerance:
            negative_energy += 1
        reconstructed = sum(float(row.get(field, 0.0)) for field in (
            "reward_done", "reward_failed", "reward_throughput"
        ))
        if abs(reconstructed - float(row.get("task_utility", 0.0))) > tolerance:
            reward_errors += 1
        if abs(float(row.get("energy_balance_error", 0.0))) > tolerance:
            balance_errors += 1
        if not bool(row.get("action_applied", False)):
            invalid_actions += 1
    issue_count = missing_numeric + negative_energy + reward_errors + balance_errors + invalid_actions
    return {
        "num_rows": len(rows),
        "missing_numeric_values": missing_numeric,
        "negative_energy_rows": negative_energy,
        "reward_reconstruction_errors": reward_errors,
        "energy_balance_errors": balance_errors,
        "invalid_action_rows": invalid_actions,
        "passed": issue_count == 0,
    }


def paired_candidate_effects(
    rows: Iterable[Mapping],
    metric_fields: Sequence[str],
    group_fields: Sequence[str] = ("seed", "decision_time"),
) -> list[dict]:
    """Subtract each group's single default candidate from its alternatives."""
    grouped: dict[tuple, list[Mapping]] = {}
    for row in rows:
        key = tuple(row[field] for field in group_fields)
        grouped.setdefault(key, []).append(row)

    effects = []
    for key, group in grouped.items():
        defaults = [row for row in group if str(row.get("action_family")) == "default"]
        if len(defaults) != 1:
            raise ValueError(f"group {key} must contain exactly one default candidate")
        baseline = defaults[0]
        for row in group:
            if row is baseline:
                continue
            result = {field: row[field] for field in group_fields}
            result.update(
                {
                    "candidate_id": row.get("candidate_id"),
                    "action_family": row.get("action_family"),
                    "baseline_candidate_id": baseline.get("candidate_id"),
                }
            )
            for field in metric_fields:
                result[f"baseline_{field}"] = float(baseline[field])
                result[field] = float(row[field])
                result[f"effect_{field}"] = float(row[field]) - float(baseline[field])
            effects.append(result)
    return effects
