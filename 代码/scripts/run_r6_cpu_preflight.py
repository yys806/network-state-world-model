"""Run the sealed, CPU-only PI-JWM R6 strategy preflight."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


SCRIPT_DIR = Path(__file__).resolve().parent
CODE_ROOT = SCRIPT_DIR.parent
SRC_ROOT = CODE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pi_jwm.r6_cpu_preflight import (  # noqa: E402
    R6_CPU_PREFLIGHT_SCHEMA,
    aggregate_metric_rows,
    audit_trajectory_record,
    validate_candidate_roles,
    write_r6_cpu_preflight_bundle,
)


DEFAULT_DATASET_ROOT = CODE_ROOT / "artifacts" / "datasets" / "airfogsim_formal_v1"
DEFAULT_R5_ROOT = CODE_ROOT / "artifacts" / "formal_training" / "pi_jwm_r5_module_confirmation_analysis_v1"
DEFAULT_OUTPUT = CODE_ROOT / "artifacts" / "formal_training" / "pi_jwm_r6_cpu_preflight_v1"


def _load_json(path: Path) -> dict[str, Any]:
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


def _verify_manifest_files(root: Path, manifest: Mapping[str, Any]) -> int:
    files = manifest.get("files")
    if not isinstance(files, Mapping) or not files:
        raise ValueError(f"manifest has no files: {root}")
    verified = 0
    for relative, evidence in files.items():
        if not isinstance(relative, str) or not isinstance(evidence, Mapping):
            raise ValueError(f"invalid manifest entry under {root}")
        path = root / relative
        if not path.is_file():
            raise ValueError(f"manifest file is missing: {path}")
        observed_size = path.stat().st_size
        expected_size = int(evidence.get("size_bytes", -1))
        if observed_size != expected_size:
            raise ValueError(f"manifest size mismatch: {path}")
        if _sha256(path) != evidence.get("sha256"):
            raise ValueError(f"manifest SHA-256 mismatch: {path}")
        verified += 1
    return verified


def _metric_rows(
    *,
    audit_row: Mapping[str, Any],
    metric_results: Mapping[str, Any],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for metric in metric_results["metrics"]:
        result.append(
            {
                "trajectory_id": audit_row["trajectory_id"],
                "scenario_id": audit_row["scenario_id"],
                "seed": audit_row["seed"],
                "split": audit_row["split"],
                "policy_id": audit_row["policy_id"],
                "metric_id": metric["name"],
                "status": metric["status"],
                "value": metric.get("value"),
                "unit": metric.get("unit", ""),
            }
        )
    return result


def run(dataset_root: Path, candidate_freeze: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing R6 bundle: {output_dir}")

    roles = validate_candidate_roles(_load_json(candidate_freeze))
    dataset_manifest_path = dataset_root / "manifest.json"
    dataset_manifest = _load_json(dataset_manifest_path)
    dataset_manifest_verified_files = _verify_manifest_files(dataset_root, dataset_manifest)

    validation = _load_json(dataset_root / "validation_report.json")
    if validation.get("formal_dataset_ready") is not True:
        raise ValueError("formal AirFogSim dataset did not pass its readiness gate")
    required_dataset_checks = (
        "action_ledgers_present",
        "all_dual_graphs_ready",
        "all_resource_ledgers_ready",
        "cpu_policy_trace_valid",
        "dag_precedence_only",
        "locked_test_excluded_from_metrics",
        "protocol_valid",
    )
    for name in required_dataset_checks:
        if validation.get("checks", {}).get(name) is not True:
            raise ValueError(f"formal dataset validation check failed: {name}")

    with (dataset_root / "trajectory_index.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        index_rows = list(csv.DictReader(handle))
    if len(index_rows) != 60:
        raise ValueError(f"formal trajectory count must be 60, observed {len(index_rows)}")
    index_split_counts = Counter(row["split"] for row in index_rows)
    expected_all_splits = {"train": 36, "validation": 12, "calibration": 6, "locked_test": 6}
    if dict(index_split_counts) != expected_all_splits:
        raise ValueError(f"formal split contract changed: {dict(index_split_counts)}")

    trajectory_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    verified_trajectory_manifest_files = 0
    # Important: skip locked-test rows before constructing or opening their directories.
    for index_row in index_rows:
        if index_row["split"] == "locked_test":
            continue
        trajectory_id = index_row["trajectory_id"]
        trajectory_root = dataset_root / "trajectories" / trajectory_id
        trajectory_manifest = _load_json(trajectory_root / "manifest.json")
        if trajectory_manifest.get("split") == "locked_test":
            raise ValueError("locked-test manifest reached through a non-locked index row")
        verified_trajectory_manifest_files += _verify_manifest_files(trajectory_root, trajectory_manifest)

        record = {
            "trajectory_summary": _load_json(trajectory_root / "trajectory_summary.json"),
            "graph_validation": _load_json(trajectory_root / "graph_validation.json"),
            "resource_validation": _load_json(trajectory_root / "resource_validation.json"),
            "metric_results": _load_json(trajectory_root / "metric_results.json"),
        }
        audit_row = audit_trajectory_record(record)
        for field in ("trajectory_id", "seed", "split", "policy_id"):
            expected = index_row[{"policy_id": "cpu_policy"}.get(field, field)]
            observed = str(audit_row[field])
            if observed != str(expected):
                raise ValueError(f"trajectory index mismatch for {trajectory_id}: {field}")
        trajectory_rows.append(audit_row)
        metric_rows.extend(_metric_rows(audit_row=audit_row, metric_results=record["metric_results"]))

    if len(trajectory_rows) != 54:
        raise ValueError(f"R6 must audit exactly 54 non-locked trajectories, observed {len(trajectory_rows)}")
    nonlocked_split_counts = Counter(row["split"] for row in trajectory_rows)
    if dict(nonlocked_split_counts) != {"train": 36, "validation": 12, "calibration": 6}:
        raise ValueError(f"non-locked split contract changed: {dict(nonlocked_split_counts)}")
    policy_counts = Counter(row["policy_id"] for row in trajectory_rows)
    if dict(policy_counts) != {"equal_share": 18, "deadline_aware": 18, "feasible_exploration": 18}:
        raise ValueError(f"CPU policy coverage changed: {dict(policy_counts)}")

    metric_summary = aggregate_metric_rows(metric_rows)
    status_counts = Counter(row["status"] for row in metric_rows)
    hard_constraint_violations = sum(int(row["hard_constraint_violation_count"]) for row in trajectory_rows)
    summary = {
        "schema_version": R6_CPU_PREFLIGHT_SCHEMA,
        "r6_cpu_preflight_ready": hard_constraint_violations == 0,
        "r6_gpu_strategy_training_ready": False,
        "final_method_frozen": False,
        "locked_test_accessed": False,
        "nonlocked_trajectory_count": len(trajectory_rows),
        "split_counts": dict(sorted(nonlocked_split_counts.items())),
        "policy_counts": dict(sorted(policy_counts.items())),
        "trajectory_audit_count": len(trajectory_rows),
        "hard_constraint_violation_count": hard_constraint_violations,
        "available_metric_row_count": status_counts["available"],
        "not_applicable_metric_row_count": status_counts["not_applicable"],
        "not_computable_metric_row_count": status_counts["not_computable"],
        "metric_group_count": len(metric_summary),
        "dataset_manifest_verified_file_count": dataset_manifest_verified_files,
        "trajectory_manifest_verified_file_count": verified_trajectory_manifest_files,
        "claim_boundary": "Observed closed-loop CPU-policy lower bound only; no causal policy ranking and no final-method freeze.",
    }
    if not summary["r6_cpu_preflight_ready"]:
        raise ValueError("R6 CPU preflight hard-constraint gate failed")

    r5_manifest = candidate_freeze.parent / "manifest.json"
    input_binding = {
        "dataset_manifest_sha256": _sha256(dataset_manifest_path),
        "dataset_protocol_sha256": _sha256(dataset_root / "protocol.json"),
        "candidate_freeze_sha256": _sha256(candidate_freeze),
        "r5_1_analysis_manifest_sha256": _sha256(r5_manifest),
    }
    write_r6_cpu_preflight_bundle(
        output_dir,
        candidate_roles=roles,
        trajectory_rows=trajectory_rows,
        metric_rows=metric_rows,
        metric_summary=metric_summary,
        summary=summary,
        input_binding=input_binding,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--candidate-freeze", type=Path, default=DEFAULT_R5_ROOT / "candidate_freeze.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run(args.dataset_root.resolve(), args.candidate_freeze.resolve(), args.output_dir.resolve())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
