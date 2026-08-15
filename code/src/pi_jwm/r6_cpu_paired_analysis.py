"""Pairing and metric aggregation contracts for R6 CPU experiments."""

from __future__ import annotations

import math
import statistics
import csv
import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .r6_cpu_paired_policy import PAIRED_CPU_POLICY_IDS


def _finite(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def build_pair_key(record: Mapping[str, Any]) -> str:
    if str(record.get("split")) == "locked_test":
        raise ValueError("locked-test records are forbidden in paired analysis")
    scenario_id = str(record.get("scenario_id", ""))
    config = str(record.get("config_fingerprint", ""))
    if not scenario_id:
        raise ValueError("scenario_id is required for pair key")
    if not config:
        raise ValueError("config fingerprint is required for pair key")
    try:
        seed = int(record["seed"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("seed is required for pair key") from exc
    return f"{scenario_id}|seed={seed}|config={config}"


def validate_pair_records(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    scenario_seed_configs: dict[tuple[str, int], set[str]] = defaultdict(set)
    for record in records:
        key = build_pair_key(record)
        policy = str(record.get("policy_id", ""))
        if policy not in PAIRED_CPU_POLICY_IDS:
            raise ValueError(f"unknown paired policy: {policy}")
        groups[key].append(record)
        scenario_seed_configs[(str(record["scenario_id"]), int(record["seed"]))].add(str(record["config_fingerprint"]))
    if any(len(configs) != 1 for configs in scenario_seed_configs.values()):
        raise ValueError("paired records disagree on config fingerprint")
    complete = 0
    incomplete: list[str] = []
    duplicate: list[str] = []
    expected = set(PAIRED_CPU_POLICY_IDS)
    for key, group in groups.items():
        policies = [str(row["policy_id"]) for row in group]
        if len(policies) != len(set(policies)):
            duplicate.append(key)
        elif set(policies) == expected:
            complete += 1
        else:
            incomplete.append(key)
    if duplicate:
        raise ValueError(f"duplicate policy in pair group: {duplicate[0]}")
    if incomplete:
        raise ValueError(f"incomplete pair group: {incomplete[0]}")
    return {
        "pair_group_count": len(groups),
        "complete_pair_count": complete,
        "incomplete_pair_count": 0,
        "locked_test_accessed": False,
    }


def aggregate_paired_metric_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        status = str(row.get("status"))
        if status not in {"available", "not_applicable", "not_computable"}:
            raise ValueError(f"unknown metric status: {status}")
        groups[(str(row["policy_id"]), str(row["metric_id"]))].append(row)
    output: list[dict[str, Any]] = []
    for (policy_id, metric_id), group in sorted(groups.items()):
        values = [_finite(row.get("value"), f"{metric_id} value") for row in group if row.get("status") == "available"]
        statuses = {str(row.get("status")) for row in group}
        status = "available" if values else ("not_computable" if "not_computable" in statuses else "not_applicable")
        output.append(
            {
                "policy_id": policy_id,
                "metric_id": metric_id,
                "status": status,
                "row_count": len(group),
                "available_count": len(values),
                "mean": statistics.fmean(values) if values else None,
                "sample_std": statistics.stdev(values) if len(values) > 1 else (0.0 if values else None),
            }
        )
    return output


def compute_paired_deltas(
    rows: Iterable[Mapping[str, Any]], *, reference_policy: str = "equal_share"
) -> list[dict[str, Any]]:
    """Compute factual paired differences; never infer unavailable values."""

    groups: dict[tuple[str, str], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        groups[(str(row["pair_key"]), str(row["metric_id"]))][str(row["policy_id"])] = row
    output: list[dict[str, Any]] = []
    for (pair_key, metric_id), policies in sorted(groups.items()):
        reference = policies.get(reference_policy)
        if reference is None:
            continue
        for policy_id in sorted(policies):
            if policy_id == reference_policy:
                continue
            candidate = policies[policy_id]
            reference_status = str(reference.get("status"))
            candidate_status = str(candidate.get("status"))
            if reference_status == "available" and candidate_status == "available":
                reference_value = _finite(reference.get("value"), f"{metric_id} reference value")
                candidate_value = _finite(candidate.get("value"), f"{metric_id} candidate value")
                output.append(
                    {
                        "pair_key": pair_key,
                        "metric_id": metric_id,
                        "reference_policy": reference_policy,
                        "policy_id": policy_id,
                        "status": "available",
                        "reference_value": reference_value,
                        "value": candidate_value,
                        "delta_vs_equal_share": candidate_value - reference_value,
                    }
                )
            else:
                status = "not_computable" if "not_computable" in {reference_status, candidate_status} else "not_applicable"
                output.append(
                    {
                        "pair_key": pair_key,
                        "metric_id": metric_id,
                        "reference_policy": reference_policy,
                        "policy_id": policy_id,
                        "status": status,
                        "reference_value": None,
                        "value": None,
                        "delta_vs_equal_share": None,
                    }
                )
    return output


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[Mapping[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_paired_bundle(
    output_dir: Path | str,
    *,
    summary: Mapping[str, Any],
    pair_records: list[Mapping[str, Any]],
    action_rows: list[Mapping[str, Any]],
    metric_rows: list[Mapping[str, Any]],
    paired_deltas: list[Mapping[str, Any]],
    failures: list[Mapping[str, Any]],
    runtime_logs: list[Mapping[str, Any]],
    input_binding: Mapping[str, Any],
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / "summary.json", {**dict(summary), "input_binding": dict(input_binding)})
    pair_fields = [
        "pair_key", "scenario_id", "seed", "split", "policy_id", "config_fingerprint",
        "status", "cpu_action_count", "action_legal_rate", "hard_constraint_violation_count",
        "metric_count",
    ]
    _write_csv(output / "pair_records.csv", list(pair_records), pair_fields)
    action_fields = [
        "pair_key", "scenario_id", "seed", "split", "policy_id", "time", "task_id", "node_id",
        "policy_weight", "deadline_remaining", "queue_size", "allocated_cpu", "node_cpu_capacity",
        "allocated_fraction", "mechanism", "search_rank",
    ]
    _write_csv(output / "cpu_action_rows.csv", list(action_rows), action_fields)
    metric_fields = ["pair_key", "scenario_id", "seed", "split", "policy_id", "metric_id", "status", "value", "unit"]
    _write_csv(output / "metric_rows.csv", list(metric_rows), metric_fields)
    delta_fields = [
        "pair_key", "metric_id", "reference_policy", "policy_id", "status",
        "reference_value", "value", "delta_vs_equal_share",
    ]
    _write_csv(output / "paired_deltas.csv", list(paired_deltas), delta_fields)
    failure_fields = ["pair_key", "scenario_id", "seed", "split", "policy_id", "error_type", "error_message"]
    _write_csv(output / "failures.csv", list(failures), failure_fields)
    log_path = output / "runtime_logs.txt"
    with log_path.open("w", encoding="utf-8") as handle:
        for item in runtime_logs:
            handle.write(f"[{item.get('pair_key')}][{item.get('policy_id')}]\n")
            handle.write(str(item.get("stdout", "")))
            if not str(item.get("stdout", "")).endswith("\n"):
                handle.write("\n")
            if item.get("stderr"):
                handle.write("[stderr]\n")
                handle.write(str(item["stderr"]))
                if not str(item["stderr"]).endswith("\n"):
                    handle.write("\n")
    (output / "README.md").write_text(
        "# PI-JWM R6 CPU paired closed-loop baseline\n\n"
        "Runs share scenario, seed, configuration fingerprint, budget and protocol. "
        "Only CPU allocation differs. Paired deltas are factual differences; unavailable "
        "metrics remain not_applicable or not_computable. This is a CPU baseline, not a "
        "final policy freeze or a GPU strategy result.\n",
        encoding="utf-8",
    )
    files = {}
    for path in sorted(output.iterdir(), key=lambda item: item.name):
        if path.name == "manifest.json":
            continue
        files[path.name] = {"sha256": _sha256(path), "size_bytes": path.stat().st_size}
    _write_json(
        output / "manifest.json",
        {
            "schema_version": "PIJWM-R6-CPU-Paired-ClosedLoop-v1",
            "manifest_entry_count": len(files),
            "locked_test_accessed": False,
            "input_binding": dict(input_binding),
            "files": files,
        },
    )
    return output
