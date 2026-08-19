"""Independent acceptance audit for the formal AirFogSim candidate dataset."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from .action_attempt_ledger_v1 import validate_attempt_records
from .airfogsim_dual_graph_v2 import validate_dual_graph_v2_bundle
from .formal_airfogsim_dataset_v1 import (
    CPU_POLICY_IDS,
    RESOURCE_ARMS,
    TrajectorySpec,
    build_formal_trajectory_specs,
    validate_formal_protocol,
)


AUDIT_SCHEMA_VERSION = "PIJWM-P2C-Formal-Data-Audit-v1"
EXPECTED_COLLECTOR_CONTRACT = "PIJWM-AirFogSim-Full-Collector-v2"
EXPECTED_SPLIT_COUNTS = Counter(
    {"train": 36, "validation": 12, "calibration": 6, "locked_test": 6}
)
REQUIRED_DIRECTIONS = frozenset(
    {"V2V", "V2U", "V2I", "U2V", "U2U", "U2I", "I2V", "I2U", "I2I"}
)
TRAJECTORY_FILES = frozenset(
    {
        "action_attempts.jsonl",
        "config_snapshot.json",
        "dual_graph_v2_bundle.json",
        "graph_validation.json",
        "metric_results.json",
        "resource_bundle.json",
        "resource_validation.json",
        "runtime_summary.json",
        "trajectory_summary.json",
    }
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _safe_relative_path(raw: object) -> PurePosixPath:
    text = str(raw).replace("\\", "/")
    pure = PurePosixPath(text)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"non-portable manifest path: {raw}")
    return pure


def _trajectory_directory(output_dir: Path, spec: TrajectorySpec) -> Path:
    root = output_dir / "locked_test" if spec.split == "locked_test" else output_dir
    return root / "trajectories" / spec.trajectory_id


def _manifest_errors(directory: Path, expected: TrajectorySpec) -> list[str]:
    errors: list[str] = []
    path = directory / "manifest.json"
    if not path.is_file():
        return [f"{expected.trajectory_id}: missing manifest.json"]
    try:
        manifest = _read_json(path)
        if manifest.get("trajectory_id") != expected.trajectory_id:
            errors.append(f"{expected.trajectory_id}: manifest trajectory_id mismatch")
        if manifest.get("split") != expected.split:
            errors.append(f"{expected.trajectory_id}: manifest split mismatch")
        files = manifest.get("files")
        if not isinstance(files, Mapping):
            return errors + [f"{expected.trajectory_id}: manifest files is not an object"]
        names = set()
        for raw_name, raw_record in files.items():
            relative = _safe_relative_path(raw_name)
            names.add(relative.as_posix())
            target = directory.joinpath(*relative.parts)
            if not target.is_file():
                errors.append(f"{expected.trajectory_id}: missing {relative}")
                continue
            if not isinstance(raw_record, Mapping):
                errors.append(f"{expected.trajectory_id}: invalid record for {relative}")
                continue
            if str(raw_record.get("sha256", "")).lower() != _sha256(target).lower():
                errors.append(f"{expected.trajectory_id}: hash mismatch for {relative}")
        missing = sorted(TRAJECTORY_FILES - names)
        extra = sorted(names - TRAJECTORY_FILES)
        if missing:
            errors.append(f"{expected.trajectory_id}: manifest missing files {missing}")
        if extra:
            errors.append(f"{expected.trajectory_id}: manifest has unexpected files {extra}")
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        errors.append(f"{expected.trajectory_id}: invalid manifest: {exc}")
    return errors


def validate_action_attempt_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    trajectory_id: str,
    expected_frames: int = 300,
) -> tuple[str, ...]:
    """Validate ledger schema plus trajectory/frame identity for one run."""

    errors = list(validate_attempt_records(rows))
    frame_indices: list[int] = []
    for index, row in enumerate(rows):
        if row.get("trajectory_id") != trajectory_id:
            errors.append(f"attempt[{index}] trajectory_id mismatch")
        try:
            frame_indices.append(int(row["frame_index"]))
        except (KeyError, TypeError, ValueError):
            errors.append(f"attempt[{index}] frame_index is invalid")
    if len(rows) != expected_frames:
        errors.append(f"expected {expected_frames} action attempts, got {len(rows)}")
    if frame_indices != list(range(expected_frames)):
        errors.append("action frame indices are not the contiguous 0..N-1 sequence")
    return tuple(errors)


def _read_attempts(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"blank action-attempt line {line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"action-attempt line {line_number} is not an object")
            rows.append(value)
    return rows


def _validate_trajectory(
    output_dir: Path,
    spec: TrajectorySpec,
    *,
    deep_graph_validation: bool,
) -> tuple[list[str], dict[str, Any]]:
    directory = _trajectory_directory(output_dir, spec)
    errors = _manifest_errors(directory, spec)
    evidence: dict[str, Any] = {
        "trajectory_id": spec.trajectory_id,
        "split": spec.split,
        "seed": spec.seed,
        "manifest_valid": not errors,
    }
    if errors:
        return errors, evidence
    try:
        summary = _read_json(directory / "trajectory_summary.json")
        runtime = _read_json(directory / "runtime_summary.json")
        graph_report = _read_json(directory / "graph_validation.json")
        resource_report = _read_json(directory / "resource_validation.json")
        attempts = _read_attempts(directory / "action_attempts.jsonl")
        identity_checks = {
            "summary_identity": (
                summary.get("trajectory_id") == spec.trajectory_id
                and int(summary.get("seed", -1)) == spec.seed
                and summary.get("split") == spec.split
                and summary.get("cpu_policy") == spec.cpu_policy
                and summary.get("resource_arm") == spec.resource_arm
                and summary.get("scenario") == spec.scenario.to_dict()
            ),
            "runtime_identity": (
                runtime.get("trajectory_id") == spec.trajectory_id
                and int(runtime.get("seed", -1)) == spec.seed
                and runtime.get("split") == spec.split
                and runtime.get("collector_contract") == EXPECTED_COLLECTOR_CONTRACT
                and runtime.get("formal_collector_ready") is True
                and int(runtime.get("steps", -1)) == 300
            ),
            "summary_checks": all(
                value is True for value in summary.get("checks", {}).values()
            ),
            "required_directions": REQUIRED_DIRECTIONS.issubset(
                set(summary.get("observed_physical_directions", []))
            ),
            "graph_validation_report": graph_report.get("dual_graph_v2_ready") is True
            and not graph_report.get("failed_checks"),
            "resource_validation_report": resource_report.get("conservation_ready") is True
            and not resource_report.get("failed_gates"),
        }
        attempt_errors = validate_action_attempt_rows(
            attempts, trajectory_id=spec.trajectory_id
        )
        identity_checks["action_attempts_valid"] = not attempt_errors
        if deep_graph_validation:
            graph = _read_json(directory / "dual_graph_v2_bundle.json")
            graph_validation = validate_dual_graph_v2_bundle(graph)
            identity_checks["graph_recomputed"] = bool(
                graph_validation.get("dual_graph_v2_ready")
            ) and not graph_validation.get("failed_checks")
        evidence.update(
            {
                "checks": identity_checks,
                "action_attempt_count": len(attempts),
                "graph_counts": summary.get("graph_counts", {}),
            }
        )
        for name, passed in identity_checks.items():
            if not passed:
                errors.append(f"{spec.trajectory_id}: {name} failed")
        errors.extend(
            f"{spec.trajectory_id}: {error}" for error in attempt_errors
        )
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        errors.append(f"{spec.trajectory_id}: invalid trajectory evidence: {exc}")
    return errors, evidence


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def audit_formal_dataset(
    output_dir: str | Path,
    *,
    deep_graph_validation: bool = True,
) -> dict[str, Any]:
    """Recompute acceptance gates from the materialized dataset artifacts."""

    root = Path(output_dir).resolve()
    specs = build_formal_trajectory_specs()
    protocol = validate_formal_protocol(specs)
    errors: list[str] = []
    checks: dict[str, bool] = {
        "dataset_directory_present": root.is_dir(),
        "protocol_valid": bool(protocol["protocol_valid"]),
    }
    if not root.is_dir():
        errors.append(f"dataset directory does not exist: {root}")
        checks.update(
            {
                "trajectory_set_complete": False,
                "trajectory_manifests_valid": False,
                "trajectory_evidence_valid": False,
                "top_level_indices_valid": False,
                "locked_test_excluded_from_metrics": False,
            }
        )
        failed = sorted(name for name, passed in checks.items() if not passed)
        return {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "audit_ready": False,
            "failed_checks": failed,
            "checks": checks,
            "errors": errors,
            "trajectory_count": 0,
            "formal_data_approved": False,
            "training_eligible": False,
            "locked_test_accessed": False,
        }

    expected_specs = [spec.to_dict() for spec in sorted(specs, key=lambda row: row.seed)]
    try:
        protocol_json = _read_json(root / "protocol.json")
        checks["protocol_artifact_matches_code"] = (
            protocol_json.get("trajectory_specs") == expected_specs
            and protocol_json.get("validation", {}).get("protocol_valid") is True
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        checks["protocol_artifact_matches_code"] = False
        errors.append(f"invalid protocol.json: {exc}")

    if any(path.name.startswith(".partial__") for path in root.rglob("*")):
        errors.append("partial trajectory directories remain")
    evidence_rows: list[dict[str, Any]] = []
    for spec in sorted(specs, key=lambda row: row.seed):
        trajectory_errors, evidence = _validate_trajectory(
            root, spec, deep_graph_validation=deep_graph_validation
        )
        errors.extend(trajectory_errors)
        evidence_rows.append(evidence)
    checks["trajectory_set_complete"] = len(evidence_rows) == 60 and all(
        _trajectory_directory(root, spec).is_dir() for spec in specs
    )
    checks["trajectory_manifests_valid"] = all(
        row.get("manifest_valid") is True for row in evidence_rows
    )
    checks["trajectory_evidence_valid"] = all(
        all(value is True for value in row.get("checks", {}).values())
        for row in evidence_rows
    )

    split_counts = Counter(spec.split for spec in specs)
    scenario_counts = Counter(spec.scenario.scenario_id for spec in specs)
    arm_counts = Counter(spec.resource_arm for spec in specs)
    checks["split_counts_36_12_6_6"] = split_counts == EXPECTED_SPLIT_COUNTS
    checks["scenario_counts_10_each"] = all(value == 10 for value in scenario_counts.values())
    checks["resource_arms_30_each"] = arm_counts == Counter({arm: 30 for arm in RESOURCE_ARMS})
    checks["resource_arms_5_each_per_scenario"] = all(
        Counter(
            spec.resource_arm
            for spec in specs
            if spec.scenario.scenario_id == scenario_id
        )
        == Counter({arm: 5 for arm in RESOURCE_ARMS})
        for scenario_id in scenario_counts
    )
    checks["cpu_policies_20_each"] = Counter(spec.cpu_policy for spec in specs) == Counter(
        {policy: 20 for policy in CPU_POLICY_IDS}
    )
    checks["seeds_unique"] = len({spec.seed for spec in specs}) == 60

    try:
        trajectory_rows = _read_csv_rows(root / "trajectory_index.csv")
        metric_rows = _read_csv_rows(root / "metrics_by_trajectory.csv")
        window_rows = _read_csv_rows(root / "window_index.csv")
        locked_window_rows = _read_csv_rows(root / "locked_test" / "window_index.csv")
        top_manifest = _read_json(root / "manifest.json")
        expected_ids = {spec.trajectory_id for spec in specs}
        expected_unlocked_seeds = {spec.seed for spec in specs if spec.split != "locked_test"}
        expected_locked_seeds = {spec.seed for spec in specs if spec.split == "locked_test"}
        metric_seeds = {int(row["seed"]) for row in metric_rows if row.get("seed") is not None}
        checks["top_level_indices_valid"] = (
            len(trajectory_rows) == 60
            and {row.get("trajectory_id") for row in trajectory_rows} == expected_ids
            and metric_seeds == expected_unlocked_seeds
            and len(metric_rows) >= len(expected_unlocked_seeds)
            and all(row.get("split") != "locked_test" for row in metric_rows)
            and all(row.get("split") != "locked_test" for row in window_rows)
            and all(row.get("split") == "locked_test" for row in locked_window_rows)
            and top_manifest.get("generation_completed") is True
            and top_manifest.get("source_manifest_present") is True
        )
        checks["locked_test_excluded_from_metrics"] = (
            metric_seeds == expected_unlocked_seeds
            and not metric_seeds.intersection(expected_locked_seeds)
            and all(row.get("split") != "locked_test" for row in metric_rows)
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        checks["top_level_indices_valid"] = False
        checks["locked_test_excluded_from_metrics"] = False
        errors.append(f"invalid top-level dataset artifact: {exc}")

    failed = sorted(name for name, passed in checks.items() if not passed)
    errors = errors[:100]
    ready = not failed and not errors
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "dataset_root": str(root),
        "audit_ready": ready,
        "failed_checks": failed,
        "checks": checks,
        "errors": errors,
        "trajectory_count": len(specs),
        "split_counts": dict(sorted(split_counts.items())),
        "scenario_counts": dict(sorted(scenario_counts.items())),
        "resource_arm_counts": dict(sorted(arm_counts.items())),
        "action_attempt_count": sum(
            int(row.get("action_attempt_count", 0)) for row in evidence_rows
        ),
        "locked_test_accessed": False,
        "formal_data_approved": ready,
        "training_eligible": False,
        "collector_contract": EXPECTED_COLLECTOR_CONTRACT,
    }
