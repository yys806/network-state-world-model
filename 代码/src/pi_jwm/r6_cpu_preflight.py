"""Auditable CPU preflight for PI-JWM R6 strategy experiments.

This module deliberately does not train or rank a policy.  It verifies that the
frozen R5.1 working candidate roles and the non-locked AirFogSim evidence are
safe to use in the later R6 strategy stage.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


R6_CPU_PREFLIGHT_SCHEMA = "PIJWM-R6-CPU-Preflight-v1"
_METRIC_STATUSES = {"available", "not_applicable", "not_computable"}
_ROLE_LISTS = {
    "task_lifecycle_specialists": ["G"],
    "continuous_state_specialists": ["J"],
    "ablation_controls": ["F"],
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _finite_number(value: Any, *, field: str) -> float:
    _require(not isinstance(value, bool), f"{field} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite number") from exc
    _require(math.isfinite(result), f"{field} must be a finite number")
    return result


def validate_candidate_roles(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize the R5.1 working-candidate contract."""

    _require(
        payload.get("schema_version") == "PIJWM-R6-Working-Candidate-Freeze-v1",
        "candidate role schema is not the frozen R5.1 contract",
    )
    _require(payload.get("reference_control") == "B", "reference control must be B")
    _require(
        payload.get("primary_working_candidate") == "B",
        "primary working candidate must remain B during R6 preflight",
    )
    normalized = dict(payload)
    for field, expected in _ROLE_LISTS.items():
        observed = payload.get(field)
        _require(isinstance(observed, list), f"{field} must be a list")
        _require(sorted(observed) == expected, f"{field} must remain {expected}")
        normalized[field] = list(expected)

    expected_flags = {
        "r5_1_candidate_set_frozen": True,
        "r6_cpu_preflight_ready": True,
        "r6_gpu_strategy_training_ready": False,
        "final_method_frozen": False,
        "locked_test_accessed": False,
    }
    for field, expected in expected_flags.items():
        _require(payload.get(field) is expected, f"{field} must be {expected}")
        normalized[field] = expected
    return normalized


def _hard_constraint_violations(resource_validation: Mapping[str, Any]) -> list[str]:
    violations: list[str] = []
    gates = resource_validation.get("gates")
    if isinstance(gates, Mapping):
        required = (
            "cpu_valid",
            "rb_valid",
            "task_flow_conservation",
            "energy_equation_valid",
            "channel_energy_input_valid",
            "dependency_accounting_valid",
        )
        for name in required:
            if gates.get(name) is not True:
                violations.append(name)

    checks = resource_validation.get("checks")
    if isinstance(checks, Mapping):
        for name in (
            "cpu_capacity_violation_rate",
            "energy_equation_violation_rate",
            "task_flow_conservation_violation_rate",
        ):
            if name not in checks:
                continue
            value = _finite_number(checks[name], field=name)
            if value > 1e-12:
                violations.append(name)

    _require(isinstance(gates, Mapping) or isinstance(checks, Mapping), "resource validation has no auditable hard constraints")
    return violations


def audit_trajectory_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Audit one non-locked trajectory without opening any locked-test evidence."""

    summary = record.get("trajectory_summary")
    graph = record.get("graph_validation")
    resource = record.get("resource_validation")
    metrics = record.get("metric_results")
    _require(isinstance(summary, Mapping), "trajectory_summary is missing")
    _require(isinstance(graph, Mapping), "graph_validation is missing")
    _require(isinstance(resource, Mapping), "resource_validation is missing")
    _require(isinstance(metrics, Mapping), "metric_results is missing")

    split = str(summary.get("split", ""))
    _require(split != "locked_test", "locked-test trajectory access is forbidden in R6 CPU preflight")
    _require(split in {"train", "validation", "calibration"}, f"unsupported split: {split}")
    _require(graph.get("dual_graph_v2_ready") is True, "dual-graph validation did not pass")
    _require(resource.get("conservation_ready") is True, "resource conservation validation did not pass")

    checks = summary.get("checks")
    _require(isinstance(checks, Mapping), "trajectory summary checks are missing")
    for name in ("action_ledgers_present", "cpu_policy_trace_valid", "dag_precedence_only"):
        _require(checks.get(name) is True, f"trajectory check failed: {name}")

    violations = _hard_constraint_violations(resource)
    _require(not violations, f"hard constraint violation(s): {', '.join(violations)}")

    metric_items = metrics.get("metrics")
    _require(isinstance(metric_items, list) and metric_items, "metric results are empty")
    status_counts = {status: 0 for status in _METRIC_STATUSES}
    metric_names: set[str] = set()
    for item in metric_items:
        _require(isinstance(item, Mapping), "metric row must be an object")
        name = item.get("name")
        status = item.get("status")
        _require(isinstance(name, str) and name, "metric name is missing")
        _require(name not in metric_names, f"duplicate metric name: {name}")
        _require(status in _METRIC_STATUSES, f"unknown metric status for {name}: {status}")
        metric_names.add(name)
        status_counts[str(status)] += 1
        if status == "available":
            _finite_number(item.get("value"), field=f"metric {name}")
        else:
            _require(item.get("value") is None, f"unavailable metric {name} must use null, not zero")

    policy_id = summary.get("cpu_policy", summary.get("cpu_policy_id"))
    _require(isinstance(policy_id, str) and policy_id, "CPU policy identifier is missing")
    trajectory_id = summary.get("trajectory_id")
    _require(isinstance(trajectory_id, str) and trajectory_id, "trajectory_id is missing")
    seed = int(summary.get("seed"))
    cpu_action_count = int(summary.get("cpu_action_count", 0))
    _require(cpu_action_count >= 0, "cpu_action_count cannot be negative")

    scenario = summary.get("scenario", summary.get("scenario_id", ""))
    if isinstance(scenario, Mapping):
        scenario_id = str(scenario.get("scenario_id", ""))
    else:
        scenario_id = str(scenario)

    return {
        "trajectory_id": trajectory_id,
        "scenario_id": scenario_id,
        "seed": seed,
        "split": split,
        "policy_id": policy_id,
        "status": "audited",
        "dual_graph_v2_ready": True,
        "resource_conservation_ready": True,
        "cpu_action_count": cpu_action_count,
        "hard_constraint_violation_count": 0,
        "available_metric_count": status_counts["available"],
        "not_applicable_metric_count": status_counts["not_applicable"],
        "not_computable_metric_count": status_counts["not_computable"],
    }


def aggregate_metric_rows(rows: Iterable[Mapping[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Aggregate metrics while preserving unavailable-state semantics."""

    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (str(row["policy_id"]), str(row["split"]), str(row["metric_id"]))
        status = row.get("status")
        _require(status in _METRIC_STATUSES, f"unknown metric status: {status}")
        groups[key].append(row)

    result: dict[tuple[str, str, str], dict[str, Any]] = {}
    for key in sorted(groups):
        group = groups[key]
        values = [
            _finite_number(item.get("value"), field=f"metric {key[2]}")
            for item in group
            if item.get("status") == "available"
        ]
        statuses = {str(item.get("status")) for item in group}
        if values:
            result[key] = {
                "status": "available",
                "row_count": len(group),
                "available_count": len(values),
                "mean": statistics.fmean(values),
                "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "minimum": min(values),
                "maximum": max(values),
            }
        else:
            status = "not_computable" if "not_computable" in statuses else "not_applicable"
            result[key] = {
                "status": status,
                "row_count": len(group),
                "available_count": 0,
                "mean": None,
                "sample_std": None,
                "minimum": None,
                "maximum": None,
            }
    return result


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[Mapping[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_r6_cpu_preflight_bundle(
    output_dir: Path | str,
    *,
    candidate_roles: Mapping[str, Any],
    trajectory_rows: list[Mapping[str, Any]],
    metric_rows: list[Mapping[str, Any]],
    metric_summary: Mapping[tuple[str, str, str], Mapping[str, Any]],
    summary: Mapping[str, Any],
    input_binding: Mapping[str, Any],
) -> Path:
    """Write an immutable, hash-bound R6 CPU preflight bundle."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / "candidate_roles.json", dict(candidate_roles))

    trajectory_fields = [
        "trajectory_id", "scenario_id", "seed", "split", "policy_id", "status",
        "dual_graph_v2_ready", "resource_conservation_ready", "cpu_action_count",
        "hard_constraint_violation_count", "available_metric_count",
        "not_applicable_metric_count", "not_computable_metric_count",
    ]
    _write_csv(output / "trajectory_audit.csv", list(trajectory_rows), trajectory_fields)

    metric_fields = ["trajectory_id", "scenario_id", "seed", "split", "policy_id", "metric_id", "status", "value", "unit"]
    _write_csv(output / "metric_rows.csv", list(metric_rows), metric_fields)

    summary_rows: list[dict[str, Any]] = []
    for (policy_id, split, metric_id), values in sorted(metric_summary.items()):
        summary_rows.append({"policy_id": policy_id, "split": split, "metric_id": metric_id, **dict(values)})
    summary_fields = ["policy_id", "split", "metric_id", "status", "row_count", "available_count", "mean", "sample_std", "minimum", "maximum"]
    _write_csv(output / "metric_summary.csv", summary_rows, summary_fields)

    payload_summary = dict(summary)
    payload_summary.setdefault("schema_version", R6_CPU_PREFLIGHT_SCHEMA)
    payload_summary["input_binding"] = dict(input_binding)
    _write_json(output / "summary.json", payload_summary)

    readme = (
        "# PI-JWM R6 CPU strategy preflight\n\n"
        "This bundle audits only non-locked formal AirFogSim trajectories and the frozen R5.1 "
        "working candidate roles. Reported policy metrics are observed closed-loop lower-bound "
        "evidence; they are not a causal policy ranking and do not freeze the final PI-JWM method.\n\n"
        "Unavailable metrics retain `not_applicable` or `not_computable`; they are never replaced by zero.\n"
    )
    (output / "README.md").write_text(readme, encoding="utf-8")

    files: dict[str, dict[str, Any]] = {}
    for path in sorted(output.iterdir(), key=lambda item: item.name):
        if path.name == "manifest.json":
            continue
        files[path.name] = {"sha256": _sha256(path), "size_bytes": path.stat().st_size}
    manifest = {
        "schema_version": R6_CPU_PREFLIGHT_SCHEMA,
        "manifest_entry_count": len(files),
        "locked_test_accessed": False,
        "input_binding": dict(input_binding),
        "files": files,
    }
    _write_json(output / "manifest.json", manifest)
    return output
