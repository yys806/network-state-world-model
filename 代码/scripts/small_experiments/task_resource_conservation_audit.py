from __future__ import annotations

"""Task/resource conservation audit for PI-JWM small experiment 04."""

import argparse
import copy
import csv
import hashlib
import json
import platform
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def validate_conservation_ledger(
    rows: list[dict[str, Any]],
    tolerance: float = 1e-8,
) -> dict[str, Any]:
    residual_rows: list[dict[str, Any]] = []
    for row in rows:
        kind = str(row.get("kind"))
        remaining_before = float(row.get("remaining_before", 0.0))
        delivered = float(row.get("delivered_data", 0.0))
        if kind == "communication":
            available = float(row.get("planned_capacity", 0.0))
        elif kind == "compute":
            available = float(row.get("allocated_cpu", 0.0)) * float(row.get("dt", 0.0))
        else:
            available = 0.0
        expected_delivered = min(max(available, 0.0), max(remaining_before, 0.0))
        expected_remaining_after = max(remaining_before - delivered, 0.0)
        amount_residual = delivered - expected_delivered
        remaining_residual = float(row.get("remaining_after", 0.0)) - expected_remaining_after
        row_max = max(abs(amount_residual), abs(remaining_residual))
        residual_rows.append(
            {
                "record_id": str(row.get("record_id", "")),
                "amount_residual": amount_residual,
                "remaining_residual": remaining_residual,
                "max_abs_residual": row_max,
                "passed": row_max <= tolerance,
            }
        )
    failed = [row["record_id"] for row in residual_rows if not row["passed"]]
    return {
        "task_flow_conservation": bool(rows) and not failed,
        "max_abs_task_residual": max(
            (row["max_abs_residual"] for row in residual_rows),
            default=0.0,
        ),
        "failed_record_ids": failed,
        "residual_rows": residual_rows,
    }


def validate_shared_dependency_accounting(
    rows: list[dict[str, Any]],
    tolerance: float = 1e-8,
) -> dict[str, Any]:
    flows = [row for row in rows if row.get("kind") == "dependency_flow"]
    relations = [row for row in rows if row.get("kind") == "dependency_relation"]
    flow_counts = Counter(str(row.get("dependency_flow_id")) for row in flows)
    errors: list[str] = []
    if any(count > 1 for count in flow_counts.values()):
        errors.append("duplicate_dependency_flow")
    flows_by_id: dict[str, dict[str, Any]] = {}
    for row in flows:
        flows_by_id.setdefault(str(row.get("dependency_flow_id")), row)
    for relation in relations:
        if relation.get("dependency_status") != "arrived":
            continue
        flow_id = str(relation.get("dependency_flow_id"))
        flow = flows_by_id.get(flow_id)
        if flow is None:
            errors.append("missing_dependency_flow")
            continue
        relation_payload = float(relation.get("dependency_payload", 0.0))
        flow_payload = float(flow.get("dependency_payload", 0.0))
        if abs(relation_payload - flow_payload) > tolerance:
            errors.append("dependency_payload_mismatch")
    unique_physical = sum(
        float(row.get("physical_delivered_data", 0.0)) for row in flows_by_id.values()
    )
    logical_payload = sum(
        float(row.get("dependency_payload", 0.0))
        for row in relations
        if row.get("dependency_status") == "arrived"
    )
    return {
        "dependency_accounting_valid": not errors,
        "errors": sorted(set(errors)),
        "unique_dependency_flows": len(flows_by_id),
        "arrived_dependency_relations": sum(
            row.get("dependency_status") == "arrived" for row in relations
        ),
        "unique_physical_delivered_data": unique_physical,
        "logical_dependency_payload": logical_payload,
    }


def validate_rb_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    residual_rows: list[dict[str, Any]] = []
    for row in rows:
        record_id = str(row.get("record_id", ""))
        rb_indices = row.get("rb_indices")
        n_rb = row.get("n_rb")
        errors: list[str] = []
        if not isinstance(n_rb, int) or isinstance(n_rb, bool) or n_rb <= 0:
            errors.append("invalid_rb_capacity")
        if not isinstance(rb_indices, list):
            errors.append("invalid_rb_indices_type")
            rb_indices = []
        if len(rb_indices) != len(set(rb_indices)):
            errors.append("duplicate_rb_index")
        if any(
            not isinstance(index, int)
            or isinstance(index, bool)
            or not isinstance(n_rb, int)
            or isinstance(n_rb, bool)
            or index < 0
            or index >= n_rb
            for index in rb_indices
        ):
            errors.append("rb_index_out_of_range")
        residual_rows.append(
            {
                "record_id": record_id,
                "errors": sorted(set(errors)),
                "passed": not errors,
            }
        )
    failed = [row["record_id"] for row in residual_rows if not row["passed"]]
    return {
        "rb_valid": bool(rows) and not failed,
        "failed_record_ids": failed,
        "residual_rows": residual_rows,
    }


def validate_cpu_rows(
    rows: list[dict[str, Any]],
    tolerance: float = 1e-8,
) -> dict[str, Any]:
    allocations: dict[tuple[float, str], float] = defaultdict(float)
    capacities: dict[tuple[float, str], set[float]] = defaultdict(set)
    invalid_records: set[str] = set()
    records_by_group: dict[tuple[float, str], list[str]] = defaultdict(list)
    errors: list[str] = []
    for row in rows:
        record_id = str(row.get("record_id", ""))
        try:
            time = float(row["time"])
            node_id = str(row["node_id"])
            allocation = float(row["allocated_cpu"])
            capacity = float(row["node_cpu_capacity"])
        except (KeyError, TypeError, ValueError):
            invalid_records.add(record_id)
            errors.append("invalid_cpu_row")
            continue
        key = (time, node_id)
        records_by_group[key].append(record_id)
        allocations[key] += allocation
        capacities[key].add(capacity)
        if allocation < -tolerance:
            invalid_records.add(record_id)
            errors.append("negative_cpu_allocation")
        if capacity < -tolerance:
            invalid_records.add(record_id)
            errors.append("negative_cpu_capacity")

    group_rows: list[dict[str, Any]] = []
    max_oversubscription = 0.0
    for key, allocation in allocations.items():
        capacity_values = capacities[key]
        if len(capacity_values) != 1:
            invalid_records.update(records_by_group[key])
            errors.append("inconsistent_node_cpu_capacity")
            capacity = min(capacity_values, default=0.0)
        else:
            capacity = next(iter(capacity_values))
        oversubscription = max(allocation - capacity, 0.0)
        max_oversubscription = max(max_oversubscription, oversubscription)
        passed = oversubscription <= tolerance and len(capacity_values) == 1
        if not passed:
            invalid_records.update(records_by_group[key])
            if oversubscription > tolerance:
                errors.append("cpu_capacity_exceeded")
        group_rows.append(
            {
                "time": key[0],
                "node_id": key[1],
                "allocated_cpu_total": allocation,
                "node_cpu_capacity": capacity,
                "oversubscription": oversubscription,
                "passed": passed,
            }
        )
    return {
        "cpu_valid": bool(rows) and not invalid_records,
        "max_oversubscription": max_oversubscription,
        "failed_record_ids": sorted(invalid_records),
        "errors": sorted(set(errors)),
        "group_rows": group_rows,
    }


def validate_uav_energy_rows(
    rows: list[dict[str, Any]],
    tolerance: float = 1e-8,
) -> dict[str, Any]:
    residual_rows: list[dict[str, Any]] = []
    for row in rows:
        record_id = str(row.get("record_id", ""))
        try:
            expected_consumption = (
                float(row["is_flying"]) * float(row["fly_unit_cost"])
                + float(row["is_hovering"]) * float(row["hover_unit_cost"])
                + float(row["using_sensor_num"]) * float(row["sensing_unit_cost"])
                + float(row["sending_data_size"]) * float(row["send_unit_cost"])
                + float(row["receiving_data_size"]) * float(row["receive_unit_cost"])
            )
            observed_consumption = float(row["energy_before"]) - float(row["energy_after"])
            equation_residual = observed_consumption - expected_consumption
            sending_input_residual = float(row["sending_data_size"]) - float(
                row["event_sending_data_size"]
            )
            receiving_input_residual = float(row["receiving_data_size"]) - float(
                row["event_receiving_data_size"]
            )
            input_residual = max(abs(sending_input_residual), abs(receiving_input_residual))
            equation_valid = abs(equation_residual) <= tolerance
            channel_input_valid = input_residual <= tolerance
            errors: list[str] = []
        except (KeyError, TypeError, ValueError):
            equation_residual = float("inf")
            sending_input_residual = float("inf")
            receiving_input_residual = float("inf")
            input_residual = float("inf")
            equation_valid = False
            channel_input_valid = False
            errors = ["invalid_uav_energy_row"]
        residual_rows.append(
            {
                "record_id": record_id,
                "energy_equation_residual": equation_residual,
                "sending_input_residual": sending_input_residual,
                "receiving_input_residual": receiving_input_residual,
                "max_abs_channel_input_residual": input_residual,
                "energy_equation_valid": equation_valid,
                "channel_energy_input_valid": channel_input_valid,
                "errors": errors,
                "passed": equation_valid and channel_input_valid and not errors,
            }
        )
    return {
        "uav_energy_valid": bool(rows) and all(row["passed"] for row in residual_rows),
        "energy_equation_valid": bool(rows)
        and all(row["energy_equation_valid"] for row in residual_rows),
        "channel_energy_input_valid": bool(rows)
        and all(row["channel_energy_input_valid"] for row in residual_rows),
        "max_abs_energy_residual": max(
            (abs(row["energy_equation_residual"]) for row in residual_rows),
            default=0.0,
        ),
        "max_abs_channel_input_residual": max(
            (row["max_abs_channel_input_residual"] for row in residual_rows),
            default=0.0,
        ),
        "failed_record_ids": [row["record_id"] for row in residual_rows if not row["passed"]],
        "residual_rows": residual_rows,
    }


def build_metric_computability(
    contract_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    del contract_rows
    definitions = [
        ("task_completion_rate", "direct"),
        ("task_delay", "direct"),
        ("deadline_violation_rate", "direct"),
        ("throughput", "direct"),
        ("rb_utilization", "direct"),
        ("cpu_utilization", "direct"),
        ("uav_energy", "direct"),
        ("jain_fairness", "derived"),
        ("vehicle_energy", "not_modeled"),
        ("rsu_energy", "not_modeled"),
        ("cpu_compute_energy", "not_modeled"),
        ("storage_occupancy", "not_modeled"),
    ]
    return [
        {
            "field_id": field_id,
            "status": status,
            "fill_value": None if status == "not_modeled" else "computed_from_ledger",
        }
        for field_id, status in definitions
    ]


def build_transfer_ledgers(
    transfer_events: list[dict[str, Any]],
    n_rb: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    task_rows: list[dict[str, Any]] = []
    rb_rows: list[dict[str, Any]] = []
    for event in transfer_events:
        delivered = float(event.get("delivered_data", 0.0))
        remaining_before = float(event.get("remaining_before", 0.0))
        event_id = str(event.get("event_id", ""))
        task_rows.append(
            {
                "record_id": event_id,
                "kind": "communication",
                "phase": str(event.get("phase", "")),
                "time": float(event.get("time", 0.0)),
                "task_id": str(event.get("task_id", "")),
                "planned_capacity": float(event.get("planned_capacity", 0.0)),
                "remaining_before": remaining_before,
                "delivered_data": delivered,
                "remaining_after": max(remaining_before - delivered, 0.0),
                "evidence": "direct_runtime_task_and_channel_delta",
            }
        )
        rb_rows.append(
            {
                "record_id": f"rb::{event_id}",
                "time": float(event.get("time", 0.0)),
                "task_id": str(event.get("task_id", "")),
                "rb_indices": [int(value) for value in event.get("rb_indices", [])],
                "n_rb": int(n_rb),
                "evidence": "direct_runtime_scheduler_and_channel_event",
            }
        )
    return task_rows, rb_rows


def build_cpu_runtime_row(
    *,
    record_id: str,
    time_value: float,
    node_id: str,
    task_id: str,
    allocated_cpu: float,
    node_cpu_capacity: float,
    dt: float,
    task_cpu: float,
    computed_before: float,
    computed_after: float,
) -> dict[str, Any]:
    remaining_before = max(float(task_cpu) - float(computed_before), 0.0)
    delivered = max(float(computed_after) - float(computed_before), 0.0)
    return {
        "record_id": str(record_id),
        "kind": "compute",
        "time": float(time_value),
        "node_id": str(node_id),
        "task_id": str(task_id),
        "allocated_cpu": float(allocated_cpu),
        "node_cpu_capacity": float(node_cpu_capacity),
        "dt": float(dt),
        "task_cpu": float(task_cpu),
        "computed_before": float(computed_before),
        "computed_after": float(computed_after),
        "remaining_before": remaining_before,
        "delivered_data": delivered,
        "remaining_after": max(float(task_cpu) - float(computed_after), 0.0),
        "evidence": "direct_runtime_compute_delta",
    }


def build_uav_energy_runtime_row(
    *,
    record_id: str,
    time_value: float,
    uav_id: str,
    before: dict[str, Any],
    after: dict[str, Any],
    event_sending_data_size: float,
    event_receiving_data_size: float,
    costs: dict[str, Any],
) -> dict[str, Any]:
    return {
        "record_id": str(record_id),
        "time": float(time_value),
        "uav_id": str(uav_id),
        "energy_before": float(before["energy"]),
        "energy_after": float(after["energy"]),
        "is_flying": int(bool(after.get("is_flying", False))),
        "is_hovering": int(bool(after.get("is_hovering", False))),
        "using_sensor_num": int(after.get("using_sensor_num", 0)),
        "sending_data_size": float(after.get("sending_data_size", 0.0)),
        "receiving_data_size": float(after.get("receiving_data_size", 0.0)),
        "event_sending_data_size": float(event_sending_data_size),
        "event_receiving_data_size": float(event_receiving_data_size),
        "fly_unit_cost": float(costs["fly_unit_cost"]),
        "hover_unit_cost": float(costs["hover_unit_cost"]),
        "sensing_unit_cost": float(costs["sensing_unit_cost"]),
        "send_unit_cost": float(costs["send_unit_cost"]),
        "receive_unit_cost": float(costs["receive_unit_cost"]),
        "evidence": "direct_runtime_energy_manager_delta_and_channel_event",
    }


def assemble_runtime_conservation_bundle(
    evidence_result: dict[str, Any],
    *,
    cpu_rows: list[dict[str, Any]],
    energy_rows: list[dict[str, Any]],
    n_rb: int,
) -> dict[str, Any]:
    evidence_bundle = evidence_result["bundle"]
    task_rows, rb_rows = build_transfer_ledgers(
        list(evidence_bundle.get("transfer_events", [])), n_rb=n_rb
    )
    task_rows.extend(copy.deepcopy(cpu_rows))
    dependency_rows = [
        {"kind": "dependency_flow", **copy.deepcopy(row)}
        for row in evidence_bundle.get("dependency_flows", [])
    ]
    dependency_rows.extend(
        {"kind": "dependency_relation", **copy.deepcopy(row)}
        for row in evidence_bundle.get("ep_relations", [])
    )
    contract_rows = task_rows + rb_rows + cpu_rows + energy_rows
    return {
        "task_ledger": task_rows,
        "dependency_ledger": dependency_rows,
        "rb_ledger": rb_rows,
        "cpu_ledger": copy.deepcopy(cpu_rows),
        "uav_energy_ledger": copy.deepcopy(energy_rows),
        "metric_computability": build_metric_computability(contract_rows),
    }


def validate_exp04_bundle(
    bundle: dict[str, Any],
    tolerance: float = 1e-8,
) -> dict[str, Any]:
    reports = {
        "task_flow": validate_conservation_ledger(
            list(bundle.get("task_ledger", [])), tolerance=tolerance
        ),
        "dependency": validate_shared_dependency_accounting(
            list(bundle.get("dependency_ledger", [])), tolerance=tolerance
        ),
        "rb": validate_rb_rows(list(bundle.get("rb_ledger", []))),
        "cpu": validate_cpu_rows(
            list(bundle.get("cpu_ledger", [])), tolerance=tolerance
        ),
        "uav_energy": validate_uav_energy_rows(
            list(bundle.get("uav_energy_ledger", [])), tolerance=tolerance
        ),
    }
    gate_values = {
        "task_flow_conservation": reports["task_flow"]["task_flow_conservation"],
        "dependency_accounting_valid": reports["dependency"]["dependency_accounting_valid"],
        "rb_valid": reports["rb"]["rb_valid"],
        "cpu_valid": reports["cpu"]["cpu_valid"],
        "energy_equation_valid": reports["uav_energy"]["energy_equation_valid"],
        "channel_energy_input_valid": reports["uav_energy"]["channel_energy_input_valid"],
    }
    failed_gates = [name for name, passed in gate_values.items() if not passed]
    return {
        "conservation_ready": not failed_gates,
        "failed_gates": failed_gates,
        "gates": gate_values,
        "reports": reports,
    }


def build_exp04_corruption_report(
    bundle: dict[str, Any],
    tolerance: float = 1e-8,
) -> dict[str, Any]:
    baseline = validate_exp04_bundle(bundle, tolerance=tolerance)
    baseline_failed = set(baseline["failed_gates"])

    def gate_severity(report: dict[str, Any], gate: str) -> float:
        if gate == "task_flow_conservation":
            return float(report["reports"]["task_flow"]["max_abs_task_residual"])
        if gate == "rb_valid":
            return float(len(report["reports"]["rb"]["failed_record_ids"]))
        if gate == "cpu_valid":
            return float(report["reports"]["cpu"]["max_oversubscription"])
        if gate == "energy_equation_valid":
            return float(report["reports"]["uav_energy"]["max_abs_energy_residual"])
        return 0.0
    specifications = [
        ("task_byte_residual", "task_ledger", "task_flow_conservation"),
        ("illegal_rb", "rb_ledger", "rb_valid"),
        ("cpu_oversubscription", "cpu_ledger", "cpu_valid"),
        ("uav_energy_residual", "uav_energy_ledger", "energy_equation_valid"),
    ]
    cases: list[dict[str, Any]] = []
    for corruption_id, ledger_name, expected_gate in specifications:
        corrupted = copy.deepcopy(bundle)
        ledger = corrupted.get(ledger_name, [])
        if not ledger:
            cases.append(
                {
                    "corruption_id": corruption_id,
                    "expected_failed_gate": expected_gate,
                    "detected": False,
                    "details": f"{ledger_name} is empty",
                }
            )
            continue
        if corruption_id == "task_byte_residual":
            ledger[0]["delivered_data"] = (
                float(ledger[0].get("delivered_data", 0.0))
                + gate_severity(baseline, expected_gate)
                + 1.0
            )
        elif corruption_id == "illegal_rb":
            if baseline["gates"][expected_gate]:
                ledger[0]["rb_indices"] = [int(ledger[0].get("n_rb", 0))]
            else:
                ledger.append(
                    {
                        "record_id": "corrupt::illegal_rb",
                        "task_id": "corrupt",
                        "rb_indices": [1],
                        "n_rb": 1,
                    }
                )
        elif corruption_id == "cpu_oversubscription":
            ledger[0]["allocated_cpu"] = (
                float(ledger[0].get("node_cpu_capacity", 0.0))
                + gate_severity(baseline, expected_gate)
                + 1.0
            )
        elif corruption_id == "uav_energy_residual":
            ledger[0]["energy_after"] = (
                float(ledger[0].get("energy_after", 0.0))
                - gate_severity(baseline, expected_gate)
                - 1.0
            )
        validation = validate_exp04_bundle(corrupted, tolerance=tolerance)
        after_failed = set(validation["failed_gates"])
        newly_failed = after_failed - baseline_failed
        severity_before = gate_severity(baseline, expected_gate)
        severity_after = gate_severity(validation, expected_gate)
        relevant_failed = (
            expected_gate in newly_failed
            or (
                expected_gate in baseline_failed
                and severity_after > severity_before + tolerance
            )
        )
        unrelated_failed = sorted(newly_failed - {expected_gate})
        cases.append(
            {
                "corruption_id": corruption_id,
                "expected_failed_gate": expected_gate,
                "detected": relevant_failed and not unrelated_failed,
                "unrelated_failed_gates": unrelated_failed,
                "baseline_severity": severity_before,
                "corrupted_severity": severity_after,
            }
        )
    return {
        "all_corruptions_detected": bool(cases)
        and all(case["detected"] for case in cases),
        "baseline_failed_gates": sorted(baseline_failed),
        "cases": cases,
    }


def canonical_json_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_code_metadata() -> dict[str, Any]:
    code_root = Path(__file__).resolve().parents[2]
    paths = [
        Path(__file__).resolve(),
        Path(__file__).resolve().parent / "airfogsim_cross_graph_evidence_closure.py",
        code_root / "src" / "pi_jwm" / "airfogsim_contract_adapter.py",
        code_root / "reference" / "AirFogSim" / "airfogsim" / "airfogsim_env.py",
        code_root / "reference" / "AirFogSim" / "airfogsim" / "manager" / "task_manager.py",
        code_root / "reference" / "AirFogSim" / "airfogsim" / "manager" / "energy_manager.py",
        code_root / "reference" / "AirFogSim" / "airfogsim" / "manager" / "channel_manager_cp.py",
        code_root / "reference" / "AirFogSim" / "airfogsim" / "entities" / "task.py",
    ]
    hashes = {
        str(path.relative_to(code_root)).replace("\\", "/"): _sha256_file(path)
        for path in paths
        if path.exists()
    }
    return {
        "files": hashes,
        "aggregate_hash": canonical_json_hash(hashes),
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        if not fieldnames:
            return
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False, sort_keys=True)
                    if isinstance(value, (dict, list))
                    else value
                    for key, value in row.items()
                }
            )


def _report_markdown(
    validation: dict[str, Any],
    runtime_summary: dict[str, Any],
) -> str:
    lines = [
        "# 小实验04：任务与资源守恒审计",
        "",
        "## 冻结结果",
        "",
        f"- `experiment_completed`: `{str(validation['experiment_completed']).lower()}`",
        f"- `conservation_ready`: `{str(validation['conservation_ready']).lower()}`",
        f"- 失败门：`{validation['failed_gates']}`",
        f"- seed：`{runtime_summary.get('seed')}`",
        f"- 内部步数：`{runtime_summary.get('steps')}`",
        "",
        "## 守恒门",
        "",
        "| 检查 | 状态 |",
        "| --- | --- |",
    ]
    for name, passed in validation["gates"].items():
        lines.append(f"| `{name}` | {'通过' if passed else '失败'} |")
    lines.extend(
        [
            "",
            "## 证据边界",
            "",
            "该实验仅审计 AirFogSim 与 PI-JWM 已声明语义中实际建模的任务、RB、CPU 和 UAV 能量字段。车辆/RSU 能量与 CPU 计算能耗未建模，保留缺失掩码而非填零；通过本实验不等于真实网络资源模型完备。",
        ]
    )
    return "\n".join(lines) + "\n"


def run_exp04(
    output_dir: Path,
    seed: int,
    max_time: float,
    runtime_runner: Any,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime = runtime_runner(int(seed), float(max_time))
    repeat = runtime_runner(int(seed), float(max_time))
    bundle = copy.deepcopy(runtime["bundle"])
    if not bundle.get("metric_computability"):
        bundle["metric_computability"] = build_metric_computability([])
    repeat_bundle = copy.deepcopy(repeat["bundle"])
    if not repeat_bundle.get("metric_computability"):
        repeat_bundle["metric_computability"] = build_metric_computability([])
    bundle_hash = canonical_json_hash(bundle)
    repeat_bundle_hash = canonical_json_hash(repeat_bundle)
    config_hash = canonical_json_hash(runtime["config"])
    repeat_config_hash = canonical_json_hash(repeat["config"])
    reproducible = bundle_hash == repeat_bundle_hash and config_hash == repeat_config_hash
    validation = validate_exp04_bundle(bundle)
    validation["experiment_completed"] = True
    corruption = build_exp04_corruption_report(bundle)
    validation["corruption_detection_passed"] = corruption["all_corruptions_detected"]
    if not corruption["all_corruptions_detected"]:
        validation["failed_gates"].append("corruption_detection")
        validation["conservation_ready"] = False
    validation["gates"]["same_seed_reproducible"] = reproducible
    validation["reproducibility_passed"] = reproducible
    if not reproducible:
        validation["failed_gates"].append("same_seed_reproducible")
        validation["conservation_ready"] = False

    _write_json(output_dir / "bundle.json", bundle)
    _write_json(output_dir / "config_snapshot.json", runtime["config"])
    _write_json(output_dir / "runtime_summary.json", runtime["runtime_summary"])
    _write_json(output_dir / "validation_report.json", validation)
    _write_json(output_dir / "corruption_report.json", corruption)
    for key in (
        "task_ledger",
        "dependency_ledger",
        "rb_ledger",
        "cpu_ledger",
        "uav_energy_ledger",
        "metric_computability",
    ):
        _write_csv(output_dir / f"{key}.csv", list(bundle.get(key, [])))
    (output_dir / "REPORT.md").write_text(
        _report_markdown(validation, runtime["runtime_summary"]), encoding="utf-8"
    )
    frozen_files = sorted(
        path for path in output_dir.iterdir() if path.is_file() and path.name != "manifest.json"
    )
    manifest = {
        "schema_version": "AirFogSim-PIJWM-exp04-v1",
        "seed": int(seed),
        "max_time": float(max_time),
        "bundle_hash": bundle_hash,
        "config_hash": config_hash,
        "reproducibility": {
            "repeat_bundle_hash": repeat_bundle_hash,
            "repeat_config_hash": repeat_config_hash,
            "same_seed_bundle_hash_equal": bundle_hash == repeat_bundle_hash,
            "same_seed_config_hash_equal": config_hash == repeat_config_hash,
        },
        "source_code": _source_code_metadata(),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
        },
        "files": {path.name: _sha256_file(path) for path in frozen_files},
    }
    _write_json(output_dir / "manifest.json", manifest)
    return {**validation, "output_dir": str(output_dir)}


def run_airfogsim_conservation_seed(seed: int, max_time: float) -> dict[str, Any]:
    code_root = Path(__file__).resolve().parents[2]
    reference_root = code_root / "reference" / "AirFogSim"
    example_dir = reference_root / "examples"
    script_dir = Path(__file__).resolve().parent
    for path in (script_dir, reference_root, example_dir):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    from airfogsim import AirFogSimEnv
    import airfogsim_cross_graph_evidence_closure as exp03

    cpu_rows: list[dict[str, Any]] = []
    energy_rows: list[dict[str, Any]] = []
    n_rb_observed: list[int] = []
    original_computation = AirFogSimEnv._updateComputation
    original_energy = AirFogSimEnv._updateEnergy

    def observed_computation(env: Any) -> None:
        original_callback = env.alloc_cpu_callback
        task_snapshots: dict[str, dict[str, Any]] = {}
        allocations: dict[str, float] = {}

        def recording_callback(computing_tasks: dict[str, list[Any]], **kwargs: Any) -> dict[str, Any]:
            for node_id, tasks in computing_tasks.items():
                node = env._getNodeById(node_id)
                node_capacity = (
                    float(node.getFogProfile().get("cpu", 0.0)) if node is not None else 0.0
                )
                for task in tasks:
                    task_id = str(task.getTaskId())
                    task_snapshots[task_id] = {
                        "task": task,
                        "node_id": str(node_id),
                        "task_cpu": float(task.getTaskCPU()),
                        "computed_before": float(task.getComputedSize()),
                        "node_cpu_capacity": node_capacity,
                    }
            result = original_callback(computing_tasks, **kwargs)
            allocations.update({str(key): float(value) for key, value in result.items()})
            return result

        env.alloc_cpu_callback = recording_callback
        try:
            original_computation(env)
        finally:
            env.alloc_cpu_callback = original_callback
        time_value = float(env.simulation_time)
        for task_id, snapshot in task_snapshots.items():
            allocation = allocations.get(task_id, 0.0)
            if allocation == 0.0:
                continue
            cpu_rows.append(
                build_cpu_runtime_row(
                    record_id=f"cpu::{task_id}::{time_value:.6f}",
                    time_value=time_value,
                    node_id=snapshot["node_id"],
                    task_id=task_id,
                    allocated_cpu=allocation,
                    node_cpu_capacity=snapshot["node_cpu_capacity"],
                    dt=float(env.simulation_interval),
                    task_cpu=snapshot["task_cpu"],
                    computed_before=snapshot["computed_before"],
                    computed_after=float(snapshot["task"].getComputedSize()),
                )
            )
        n_rb_observed.append(int(env.channel_manager.n_RB))

    def observed_energy(env: Any) -> None:
        before_infos = copy.deepcopy(env.energy_manager._UAVs_energy_info)
        time_value = float(env.simulation_time)
        current_events = [
            event
            for event in getattr(env, "pi_jwm_transfer_events", [])
            if abs(float(event.get("time", -1.0)) - time_value) <= 1e-12
        ]
        event_sending: dict[str, float] = defaultdict(float)
        event_receiving: dict[str, float] = defaultdict(float)
        for event in current_events:
            event_sending[str(event.get("source"))] += float(event.get("planned_capacity", 0.0))
            event_receiving[str(event.get("target"))] += float(event.get("planned_capacity", 0.0))
        original_energy(env)
        after_infos = copy.deepcopy(env.energy_manager._UAVs_energy_info)
        after_infos.update(copy.deepcopy(env.energy_manager._removed_UAVs_energy_info))
        costs = {
            name: env.energy_manager.getConfig(name)
            for name in (
                "fly_unit_cost",
                "hover_unit_cost",
                "sensing_unit_cost",
                "send_unit_cost",
                "receive_unit_cost",
            )
        }
        for uav_id, before in before_infos.items():
            after = after_infos.get(uav_id)
            if after is None:
                continue
            energy_rows.append(
                build_uav_energy_runtime_row(
                    record_id=f"energy::{uav_id}::{time_value:.6f}",
                    time_value=time_value,
                    uav_id=str(uav_id),
                    before=before,
                    after=after,
                    event_sending_data_size=event_sending.get(str(uav_id), 0.0),
                    event_receiving_data_size=event_receiving.get(str(uav_id), 0.0),
                    costs=costs,
                )
            )
        n_rb_observed.append(int(env.channel_manager.n_RB))

    AirFogSimEnv._updateComputation = observed_computation
    AirFogSimEnv._updateEnergy = observed_energy
    try:
        evidence_result = exp03.run_airfogsim_evidence_seed(int(seed), float(max_time))
    finally:
        AirFogSimEnv._updateComputation = original_computation
        AirFogSimEnv._updateEnergy = original_energy

    n_rb = n_rb_observed[-1] if n_rb_observed else 0
    bundle = assemble_runtime_conservation_bundle(
        evidence_result,
        cpu_rows=cpu_rows,
        energy_rows=energy_rows,
        n_rb=n_rb,
    )
    config = copy.deepcopy(evidence_result["config"])
    config["pi_jwm_exp04"] = {
        "schema_version": "AirFogSim-PIJWM-exp04-v1",
        "seed": int(seed),
        "max_time": float(max_time),
        "observer": "in_process_stage_wrapper_no_core_source_modification",
        "dependency_semantics": "shared_parent_output",
    }
    source_summary = evidence_result["runtime_summary"]
    runtime_summary = {
        "seed": int(seed),
        "steps": int(source_summary.get("steps", 0)),
        "max_time": float(max_time),
        "n_rb": int(n_rb),
        "communication_rows": sum(row.get("kind") == "communication" for row in bundle["task_ledger"]),
        "compute_rows": len(cpu_rows),
        "dependency_rows": len(bundle["dependency_ledger"]),
        "rb_rows": len(bundle["rb_ledger"]),
        "uav_energy_rows": len(energy_rows),
        "source_exp03_transfer_events": int(source_summary.get("transfer_events", 0)),
        "source_exp03_ep_relations": int(source_summary.get("ep_relations", 0)),
    }
    return {
        "config": config,
        "bundle": bundle,
        "source_bundle": evidence_result["bundle"],
        "runtime_summary": runtime_summary,
    }


def _default_output_dir() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "artifacts"
        / "small_experiments"
        / "exp04_task_resource_conservation"
        / "conservation_v1"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run PI-JWM experiment 04 task/resource conservation audit."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-time", type=float, default=12.0)
    parser.add_argument("--output-dir", type=Path, default=_default_output_dir())
    args = parser.parse_args()
    result = run_exp04(
        output_dir=args.output_dir,
        seed=args.seed,
        max_time=args.max_time,
        runtime_runner=run_airfogsim_conservation_seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
