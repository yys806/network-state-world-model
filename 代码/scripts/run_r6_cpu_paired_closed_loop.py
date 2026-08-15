"""Run R6 same-scenario/same-seed CPU paired closed-loop baselines."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import sys
from contextlib import redirect_stderr, redirect_stdout
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping


SCRIPT_DIR = Path(__file__).resolve().parent
CODE_ROOT = SCRIPT_DIR.parent
SRC_ROOT = CODE_ROOT / "src"
for path in (SRC_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pi_jwm.formal_airfogsim_dataset_v1 import (  # noqa: E402
    build_formal_trajectory_specs,
)
from pi_jwm.r6_cpu_paired_analysis import (  # noqa: E402
    PAIRED_CPU_POLICY_IDS,
    aggregate_paired_metric_rows,
    build_pair_key,
    compute_paired_deltas,
    validate_pair_records,
    write_paired_bundle,
)
from pi_jwm.r6_cpu_paired_policy import PairedCpuPolicyAllocator  # noqa: E402


DATASET_ROOT = CODE_ROOT / "artifacts" / "datasets" / "airfogsim_formal_v1"
PREFLIGHT_ROOT = CODE_ROOT / "artifacts" / "formal_training" / "pi_jwm_r6_cpu_preflight_v1"
DEFAULT_OUTPUT = CODE_ROOT / "artifacts" / "formal_training" / "pi_jwm_r6_cpu_paired_closed_loop_v1"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(spec: Any, max_time: float) -> str:
    payload = {
        "protocol": "PIJWM-R6-CPU-Paired-ClosedLoop-v1",
        "scenario": spec.scenario.to_dict(),
        "seed": int(spec.seed),
        "max_time": float(max_time),
        "history": 8,
        "horizon": 3,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _runtime_components():
    from formal_airfogsim_runtime_v1 import run_formal_airfogsim_trajectory
    from build_formal_airfogsim_dataset_v1 import _build_graph, _metric_inputs
    from pi_jwm.airfogsim_dual_graph_v2 import validate_dual_graph_v2_bundle
    from pi_jwm.airfogsim_metrics_v2 import compute_airfogsim_metrics_v2
    import task_resource_conservation_audit as conservation_module

    return (
        run_formal_airfogsim_trajectory,
        _build_graph,
        _metric_inputs,
        validate_dual_graph_v2_bundle,
        compute_airfogsim_metrics_v2,
        conservation_module.validate_exp04_bundle,
    )


def invoke_runtime_quietly(runtime_runner: Any, *args: Any, **kwargs: Any) -> tuple[Any, str, str]:
    """Capture simulator console output so GBK terminals cannot abort a run."""

    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        result = runtime_runner(*args, **kwargs)
    return result, stdout.getvalue(), stderr.getvalue()


def _audit_action_rows(rows: list[Mapping[str, Any]]) -> tuple[float, int]:
    if not rows:
        return 1.0, 0
    legal = 0
    for row in rows:
        values = [row.get("allocated_cpu"), row.get("node_cpu_capacity"), row.get("allocated_fraction")]
        if not all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in values):
            continue
        if float(row["allocated_cpu"]) >= -1e-12 and 0.0 <= float(row["allocated_fraction"]) <= 1.0 + 1e-8:
            legal += 1
    return legal / len(rows), len(rows) - legal


def _metric_rows(pair_key: str, record: Mapping[str, Any], report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metric in report.get("metrics", []):
        rows.append(
            {
                "pair_key": pair_key,
                "scenario_id": record["scenario_id"],
                "seed": record["seed"],
                "split": record["split"],
                "policy_id": record["policy_id"],
                "metric_id": metric["name"],
                "status": metric["status"],
                "value": metric.get("value"),
                "unit": metric.get("unit", ""),
            }
        )
    return rows


def run(
    *,
    output_dir: Path,
    max_time: float = 30.0,
    splits: tuple[str, ...] = ("train", "validation", "calibration"),
    limit: int = 0,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite paired output: {output_dir}")
    if any(split == "locked_test" for split in splits):
        raise PermissionError("locked-test is forbidden in R6 paired CPU runs")
    if not splits or any(split not in {"train", "validation", "calibration"} for split in splits):
        raise ValueError("splits must be non-locked formal splits")
    runtime_runner, build_graph, metric_inputs, validate_graph, compute_metrics, validate_resource = _runtime_components()
    specs = [spec for spec in build_formal_trajectory_specs() if spec.split in splits]
    specs.sort(key=lambda spec: int(spec.seed))
    if limit > 0:
        specs = specs[:limit]
    if not specs:
        raise ValueError("no non-locked specs selected")

    pair_records: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    runtime_logs: list[dict[str, Any]] = []
    for base_spec in specs:
        config_fingerprint = _fingerprint(base_spec, max_time)
        base_record = {
            "scenario_id": base_spec.scenario.scenario_id,
            "seed": int(base_spec.seed),
            "split": base_spec.split,
            "config_fingerprint": config_fingerprint,
        }
        pair_key = build_pair_key(base_record)
        print(
            f"[R6 paired] scenario={base_record['scenario_id']} seed={base_record['seed']} split={base_record['split']}",
            flush=True,
        )
        for policy_id in PAIRED_CPU_POLICY_IDS:
            policy_spec = replace(
                base_spec,
                cpu_policy=policy_id,
                trajectory_id=f"{base_spec.scenario.scenario_id}__seed{base_spec.seed:03d}__{policy_id}",
            )
            stdout = ""
            stderr = ""
            try:
                print(f"[R6 paired] policy={policy_id}", flush=True)
                runtime, stdout, stderr = invoke_runtime_quietly(
                    runtime_runner,
                    policy_spec,
                    max_time=max_time,
                    allocator_factory=lambda selected_policy, seed: PairedCpuPolicyAllocator(selected_policy, seed=seed),
                )
                runtime_logs.append({"pair_key": pair_key, "policy_id": policy_id, "stdout": stdout, "stderr": stderr})
                resource = runtime["bundle"]
                graph = build_graph(runtime, policy_spec)
                graph_validation = validate_graph(graph)
                resource_validation = validate_resource(resource)
                metrics = compute_metrics(metric_inputs(resource, graph))
                cpu_rows = list(resource.get("cpu_ledger", []))
                legal_rate, action_violations = _audit_action_rows(cpu_rows)
                gates = resource_validation.get("gates", {})
                hard_violation_count = action_violations + sum(
                    1 for key in (
                        "cpu_valid", "rb_valid", "task_flow_conservation",
                        "energy_equation_valid", "channel_energy_input_valid", "dependency_accounting_valid",
                    )
                    if gates.get(key) is not True
                )
                record = {
                    **base_record,
                    "pair_key": pair_key,
                    "policy_id": policy_id,
                    "status": "completed",
                    "cpu_action_count": len(cpu_rows),
                    "action_legal_rate": legal_rate,
                    "hard_constraint_violation_count": hard_violation_count,
                    "metric_count": len(metrics.get("metrics", [])),
                    "graph_ready": graph_validation.get("dual_graph_v2_ready") is True,
                    "resource_ready": resource_validation.get("conservation_ready") is True,
                }
                pair_records.append(record)
                for row in cpu_rows:
                    action_rows.append({"pair_key": pair_key, "scenario_id": base_record["scenario_id"], "seed": base_record["seed"], "split": base_record["split"], **row})
                metric_rows.extend(_metric_rows(pair_key, record, metrics))
            except Exception as exc:  # preserve failures as first-class evidence
                runtime_logs.append({"pair_key": pair_key, "policy_id": policy_id, "stdout": stdout, "stderr": stderr})
                failures.append(
                    {
                        **base_record,
                        "pair_key": pair_key,
                        "policy_id": policy_id,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                )

    pair_validation = None
    if not failures:
        pair_validation = validate_pair_records(pair_records)
    else:
        pair_validation = {"pair_group_count": len(specs), "complete_pair_count": 0, "locked_test_accessed": False}
    hard_count = sum(int(row["hard_constraint_violation_count"]) for row in pair_records)
    status_counts = Counter(str(row["status"]) for row in metric_rows)
    summary = {
        "schema_version": "PIJWM-R6-CPU-Paired-ClosedLoop-v1",
        "paired_closed_loop_ready": not failures and hard_count == 0 and pair_validation["complete_pair_count"] == len(specs),
        "locked_test_accessed": False,
        "world_model_updated": False,
        "gpu_started": False,
        "policy_ids": list(PAIRED_CPU_POLICY_IDS),
        "selected_base_spec_count": len(specs),
        "run_count": len(pair_records),
        "failure_count": len(failures),
        "pair_validation": pair_validation,
        "hard_constraint_violation_count": hard_count,
        "action_legal_rate_min": min((float(row["action_legal_rate"]) for row in pair_records), default=0.0),
        "metric_status_counts": dict(status_counts),
        "claim_boundary": "Factual same-state CPU baseline and paired deltas only; no learned policy ranking or final method freeze.",
    }
    input_binding = {
        "dataset_manifest_sha256": _sha256(DATASET_ROOT / "manifest.json"),
        "dataset_protocol_sha256": _sha256(DATASET_ROOT / "protocol.json"),
        "r6_cpu_preflight_summary_sha256": _sha256(PREFLIGHT_ROOT / "summary.json"),
    }
    write_paired_bundle(
        output_dir,
        summary=summary,
        pair_records=pair_records,
        action_rows=action_rows,
        metric_rows=metric_rows,
        paired_deltas=compute_paired_deltas(metric_rows),
        failures=failures,
        runtime_logs=runtime_logs,
        input_binding=input_binding,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-time", type=float, default=30.0)
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "calibration"])
    parser.add_argument("--limit", type=int, default=0, help="limit base scenario-seed pairs; 0 means all selected")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run(
        output_dir=args.output_dir.resolve(),
        max_time=args.max_time,
        splits=tuple(args.splits),
        limit=args.limit,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["paired_closed_loop_ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
