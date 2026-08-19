"""Independent audit for the user-confirmed formal AirFogSim protocol."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from .formal_airfogsim_dataset_v1 import (
    RESOURCE_ARMS,
    SCHEMA_VERSION,
    TrajectorySpec,
    validate_formal_protocol,
)


AUDIT_SCHEMA_VERSION = "PIJWM-P2C-Formal-Protocol-Audit-v1"
EXPECTED_CONTRACT = "PIJWM-AirFogSim-Full-Collector-v2"
EXPECTED_FIXED_RUNTIME = {
    "simulation_interval": 0.1,
    "traffic_interval": 0.1,
    "max_n_UAVs": 1,
    "UAV_z_range": [100, 200],
    "UAV_speed_range": [10, 30],
    "max_n_cloudServers": 1,
    "channel_outage_model": "Rayleigh",
    "outage_snr_threshold": 10,
    "wired_edges": [],
    "failure_outage_semantics": (
        "natural_observation_only; no artificial failure injection in formal scenario matrix"
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _resolve_project_path(project_root: Path, raw: object) -> Path:
    text = str(raw).replace("\\", "/")
    pure = PurePosixPath(text)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"non-portable protocol evidence path: {raw}")
    return project_root.joinpath(*pure.parts)


def _scenario_rows(specs: Sequence[TrajectorySpec]) -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for spec in specs:
        scenario = spec.scenario
        row = {
            "scenario_id": scenario.scenario_id,
            "load_level": scenario.load_level,
            "density_level": scenario.density_level,
            "task_lambda": float(scenario.task_lambda),
            "max_vehicles": int(scenario.max_vehicles),
            "vehicle_arrival_lambda": float(scenario.vehicle_arrival_lambda),
        }
        previous = rows.setdefault(scenario.scenario_id, row)
        if previous != row:
            raise ValueError(f"scenario fields vary within specs: {scenario.scenario_id}")
    return [rows[key] for key in sorted(rows)]


def _proposal_scenario_rows(proposal: Mapping[str, Any]) -> list[dict[str, Any]]:
    matrix = _mapping(proposal.get("scenario_matrix_proposal"), name="scenario_matrix_proposal")
    raw_rows = matrix.get("scenarios")
    if not isinstance(raw_rows, list):
        raise ValueError("scenario_matrix_proposal.scenarios must be an array")
    rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        row = _mapping(raw, name="scenario row")
        rows.append(
            {
                "scenario_id": str(row.get("scenario_id")),
                "load_level": str(row.get("load_level")),
                "density_level": str(row.get("density_level")),
                "task_lambda": float(row.get("task_lambda")),
                "max_vehicles": int(row.get("max_vehicles")),
                "vehicle_arrival_lambda": float(row.get("vehicle_arrival_lambda")),
            }
        )
    return sorted(rows, key=lambda row: row["scenario_id"])


def _all_equal(values: Sequence[object], expected: object) -> bool:
    return bool(values) and all(value == expected for value in values)


def audit_formal_protocol(
    specs: Sequence[TrajectorySpec],
    freeze_proposal: Mapping[str, Any],
    *,
    project_root: str | Path,
    expected_collector_contract: str = EXPECTED_CONTRACT,
) -> dict[str, Any]:
    """Recompute the three P2-C freeze gates from code and calibration evidence."""

    root = Path(project_root).resolve()
    rows = list(specs)
    protocol = validate_formal_protocol(rows)
    matrix = _mapping(
        freeze_proposal.get("scenario_matrix_proposal"),
        name="scenario_matrix_proposal",
    )
    scale = _mapping(
        freeze_proposal.get("formal_scale_proposal"),
        name="formal_scale_proposal",
    )
    seed_split = _mapping(
        freeze_proposal.get("seed_split_proposal"),
        name="seed_split_proposal",
    )
    fixed_runtime = _mapping(
        freeze_proposal.get("fixed_runtime_proposal"),
        name="fixed_runtime_proposal",
    )
    resource = _mapping(
        freeze_proposal.get("resource_policy_proposal"),
        name="resource_policy_proposal",
    )

    expected_scenarios = _proposal_scenario_rows(freeze_proposal)
    actual_scenarios = _scenario_rows(rows)
    per_scenario_split = {
        "train": 6,
        "validation": 2,
        "calibration": 1,
        "locked_test": 1,
    }
    actual_split_counts = Counter(spec.split for spec in rows)
    actual_scenario_counts = Counter(spec.scenario.scenario_id for spec in rows)
    actual_arm_counts = Counter(spec.resource_arm for spec in rows)
    actual_arms_by_scenario = {
        scenario_id: Counter(
            spec.resource_arm
            for spec in rows
            if spec.scenario.scenario_id == scenario_id
        )
        for scenario_id in sorted(actual_scenario_counts)
    }
    expected_seed_values = [
        10000 + scenario_index * 100 + repetition
        for scenario_index in range(6)
        for repetition in range(10)
    ]
    actual_seed_values = [int(spec.seed) for spec in rows]
    calibration = _mapping(
        freeze_proposal.get("calibration_evidence"),
        name="calibration_evidence",
    )
    report_path = _resolve_project_path(root, calibration.get("report"))
    probe_path = _resolve_project_path(root, calibration.get("probe_rows"))
    calibration_checks = _mapping(calibration.get("checks"), name="calibration checks")
    calibration_evidence_verified = (
        report_path.is_file()
        and probe_path.is_file()
        and _sha256(report_path) == str(calibration.get("report_sha256", "")).upper()
        and _sha256(probe_path) == str(calibration.get("probe_rows_sha256", "")).upper()
        and all(value is True for value in calibration_checks.values())
    )

    checks = {
        "protocol_code_valid": bool(protocol["protocol_valid"]),
        "scenario_matrix_frozen": (
            matrix.get("status") == "frozen_by_user_confirmation_20260819"
            and actual_scenarios == expected_scenarios
            and len(actual_scenarios) == len(expected_scenarios) == 6
            and fixed_runtime.get("failure_outage_semantics")
            == EXPECTED_FIXED_RUNTIME["failure_outage_semantics"]
        ),
        "formal_scale_frozen": (
            str(scale.get("status", "")).startswith("frozen_by_user_confirmation_")
            and len(rows) == int(scale.get("total_trajectories", -1)) == 60
            and all(count == int(scale.get("trajectories_per_scenario", -1)) for count in actual_scenario_counts.values())
            and float(scale.get("seconds_per_trajectory", -1.0)) == 30.0
            and int(scale.get("steps_per_trajectory", -1)) == 300
        ),
        "formal_split_frozen": (
            str(seed_split.get("status", "")).startswith("frozen_by_user_confirmation_")
            and actual_split_counts == Counter(scale.get("total_split_counts", {}))
            and actual_seed_values == expected_seed_values
            and len(set(actual_seed_values)) == len(actual_seed_values)
            and not set(actual_seed_values).intersection(
                set(seed_split.get("reserved_development_seeds", []))
            )
        ),
        "resource_arms_frozen": (
            resource.get("policy_version") == "balanced_two_arm_v1"
            and resource.get("arms") == list(RESOURCE_ARMS)
            and all(
                actual_arms_by_scenario.get(scenario_id) == Counter({arm: 5 for arm in RESOURCE_ARMS})
                for scenario_id in actual_scenario_counts
            )
            and actual_arm_counts == Counter({arm: 30 for arm in RESOURCE_ARMS})
        ),
        "fixed_runtime_declared": all(
            fixed_runtime.get(key) == value for key, value in EXPECTED_FIXED_RUNTIME.items()
        ),
        "calibration_evidence_verified": calibration_evidence_verified,
        "locked_test_sealed": freeze_proposal.get("locked_test_accessed") is False
        and bool(seed_split.get("rule"))
        and "sealed" in str(seed_split.get("rule")),
        "collector_contract_declared": expected_collector_contract
        == EXPECTED_CONTRACT,
    }
    failed_checks = sorted(name for name, passed in checks.items() if not passed)
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "protocol_schema_version": SCHEMA_VERSION,
        "audit_ready": not failed_checks,
        "failed_checks": failed_checks,
        "checks": checks,
        "protocol_validation": protocol,
        "scenario_rows": actual_scenarios,
        "trajectory_count": len(rows),
        "split_counts": dict(sorted(actual_split_counts.items())),
        "resource_arm_counts": dict(sorted(actual_arm_counts.items())),
        "calibration_evidence": {
            "report": str(report_path.relative_to(root)).replace("\\", "/"),
            "probe_rows": str(probe_path.relative_to(root)).replace("\\", "/"),
            "report_sha256": _sha256(report_path) if report_path.is_file() else None,
            "probe_rows_sha256": _sha256(probe_path) if probe_path.is_file() else None,
        },
        "formal_data_approved": False,
        "training_eligible": False,
        "locked_test_accessed": False,
        "collector_contract": EXPECTED_CONTRACT,
    }
