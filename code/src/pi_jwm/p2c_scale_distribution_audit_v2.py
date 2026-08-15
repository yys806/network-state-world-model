"""P2-C v2 audit whose rejection evidence is recomputed from attempt ledger rows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .action_attempt_ledger_v1 import (
    MUTATION_STATES,
    RUN_ROLES,
    summarize_attempts,
    validate_attempt_records,
)
from .full_dual_graph_artifact_v2 import (
    SUCCESS_REQUIRED_FILES,
    validate_bundle_alignment,
)
from . import p2c_scale_distribution_audit_v1 as legacy


AUDIT_SCHEMA_VERSION = "PIJWM-P2C-Scale-Distribution-Audit-v2"
FORMAL_CONFIG_SCHEMA_VERSION = "PIJWM-P2C-Formal-Data-Config-Candidate-v2"
REQUIRED_ARTIFACT_FILES = SUCCESS_REQUIRED_FILES


class AuditContractError(ValueError):
    """A P2-B v2 bundle cannot support a truthful P2-C ledger audit."""


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AuditContractError(f"invalid JSON: {path.name}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise AuditContractError(f"{path.name} must be a JSON object")
    return value


def _load_jsonl(path: Path, *, what: str) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AuditContractError(
                    f"invalid {what} line {line_number}: {exc}"
                ) from exc
            if not isinstance(row, Mapping):
                raise AuditContractError(f"{what} line {line_number} must be an object")
            rows.append(row)
    return rows


def load_action_attempts(path: Path) -> list[Mapping[str, Any]]:
    rows = _load_jsonl(path, what="action_attempts.jsonl")
    if not rows:
        raise AuditContractError("action_attempts.jsonl is empty")
    errors = validate_attempt_records(rows)
    if errors:
        raise AuditContractError(f"attempt ledger invalid: {list(errors)}")
    return rows


def _role_report(summary: Mapping[str, object]) -> dict[str, object]:
    by_role = summary["by_run_role"]
    return {role: dict(by_role[role]) for role in RUN_ROLES}  # type: ignore[index]


def audit_bundle(
    bundle_dir: str | Path,
    *,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(bundle_dir)
    missing = [name for name in REQUIRED_ARTIFACT_FILES if not (root / name).is_file()]
    if missing:
        raise AuditContractError(f"bundle missing required files: {missing}")

    attempts = load_action_attempts(root / "action_attempts.jsonl")
    frames = _load_jsonl(root / "frames.jsonl", what="frames.jsonl")
    alignment_errors = validate_bundle_alignment(frames, attempts)
    if alignment_errors:
        raise AuditContractError(
            f"attempt/frame/replay alignment invalid: {list(alignment_errors)}"
        )

    try:
        report = legacy.audit_bundle(root, project_root=project_root)
    except legacy.AuditContractError as exc:
        raise AuditContractError(str(exc)) from exc
    summary = summarize_attempts(attempts)
    by_role = summary["by_run_role"]
    natural = by_role["natural_reference"]
    validation = _load_json(root / "validation_report.json")
    reported_summary = validation.get("action_rejection_count")

    blockers = set(str(value) for value in report.get("blocking_reasons", []))
    blockers.discard("action_rejection_rate_not_observed")
    blockers.update(
        {
            "scenario_matrix_not_frozen",
            "formal_scale_not_frozen",
            "formal_split_not_frozen",
        }
    )
    report["schema_version"] = AUDIT_SCHEMA_VERSION
    report["audit_scope"] = (
        "CPU-only read-only audit of P2-B v2 attempt ledger; no formal-data approval"
    )
    report["blocking_reasons"] = sorted(blockers)
    report["attempt_ledger_evidence"] = {
        "source_file": "action_attempts.jsonl",
        "schema_and_transition_validation_passed": True,
        "frame_and_replay_alignment_passed": True,
        "binary_conservation_passed": summary["binary_conservation_passed"],
        "by_run_role": _role_report(summary),
    }
    report["rejection_quarantine"] = {
        "source": "independently_recomputed_from_action_attempts_jsonl",
        "action_attempt_count": natural["attempt_count"],
        "action_accepted_count": natural["accepted_count"],
        "action_rejection_count": natural["rejected_count"],
        "action_rejection_rate": natural["rejection_rate"],
        "natural_reference_quarantined_count": natural["quarantined_count"],
        "natural_reference_mutation_counts": dict(natural["mutation_counts"]),
        "all_role_quarantined_count": sum(
            int(by_role[role]["quarantined_count"]) for role in RUN_ROLES
        ),
        "all_role_mutation_counts": {
            state: sum(
                int(by_role[role]["mutation_counts"][state]) for role in RUN_ROLES
            )
            for state in MUTATION_STATES
        },
        "reported_summary_rejection_count_ignored": reported_summary,
        "status": "observed_from_natural_reference_ledger",
    }
    observed_validation = report.get("observed_facts", {}).get("validation")
    if isinstance(observed_validation, dict):
        observed_validation["rejection_count"] = natural["rejected_count"]
        observed_validation["rejection_source"] = "action_attempts.jsonl"
        observed_validation["reported_summary_rejection_count_ignored"] = reported_summary
    candidate = legacy.build_candidate_formal_data_config(report)
    candidate["schema_version"] = FORMAL_CONFIG_SCHEMA_VERSION
    candidate["source_audit_schema"] = AUDIT_SCHEMA_VERSION
    candidate["source_scope"] = (
        "P2-B v2 CPU-only nontraining preflight with attempt ledger; not a formal dataset"
    )
    candidate["formal_output_directory"]["relative_path"] = (
        "code/artifacts/formal_data/pi_jwm_v4_formal_candidate_v2"
    )
    report["candidate_formal_data_config"] = candidate
    report["audit_status"] = (
        "blocked" if report["blocking_reasons"] else "passed_for_candidate_review"
    )
    return report
