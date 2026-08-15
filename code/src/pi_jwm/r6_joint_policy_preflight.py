"""Readiness gates and self-auditing bundle writer for R6 joint policy."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


R6_GPU_READINESS_SCHEMA = "PIJWM-R6-joint-policy-GPU-readiness-v1"


def _is_sha256(value: str) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


@dataclass(frozen=True)
class GPUReadinessEvidence:
    joint_candidate_contract_passed: bool
    offload_nonnoop_count: int
    rb_nonnoop_count: int
    cpu_nonnoop_count: int
    hard_constraint_rejection_passed: bool
    actor_critic_update_passed: bool
    ppo_update_passed: bool
    real_rollout_transition_count: int
    identity_continuity_passed: bool
    reward_recomputation_passed: bool
    gae_reference_passed: bool
    world_model_sha256_before: str
    world_model_sha256_after: str
    locked_test_accessed: bool
    gpu_used: bool
    dataset_regenerated: bool
    world_model_retrained: bool
    regression_passed: bool

    def __post_init__(self) -> None:
        for field in (
            "offload_nonnoop_count",
            "rb_nonnoop_count",
            "cpu_nonnoop_count",
            "real_rollout_transition_count",
        ):
            if int(getattr(self, field)) < 0:
                raise ValueError(f"{field} must be nonnegative")
        for field in ("world_model_sha256_before", "world_model_sha256_after"):
            if not _is_sha256(getattr(self, field)):
                raise ValueError(f"{field} must be a lowercase SHA-256 digest")


@dataclass(frozen=True)
class GPUReadinessAssessment:
    schema_version: str
    r6_gpu_strategy_training_ready: bool
    blockers: tuple[str, ...]
    gate_results: Mapping[str, bool]
    locked_test_accessed: bool
    gpu_used: bool
    dataset_regenerated: bool
    world_model_retrained: bool
    final_method_frozen: bool = False


def assess_gpu_readiness(evidence: GPUReadinessEvidence) -> GPUReadinessAssessment:
    gates = {
        "joint_candidate_contract": bool(evidence.joint_candidate_contract_passed),
        "real_nonnoop_action_evidence": min(
            evidence.offload_nonnoop_count,
            evidence.rb_nonnoop_count,
            evidence.cpu_nonnoop_count,
        )
        > 0,
        "hard_constraint_rejection": bool(evidence.hard_constraint_rejection_passed),
        "finite_policy_updates": bool(
            evidence.actor_critic_update_passed and evidence.ppo_update_passed
        ),
        "real_multistep_rollout": bool(
            evidence.real_rollout_transition_count >= 2
            and evidence.identity_continuity_passed
        ),
        "reward_recomputation": bool(evidence.reward_recomputation_passed),
        "gae_reference": bool(evidence.gae_reference_passed),
        "regression": bool(evidence.regression_passed),
        "stage_boundary": bool(
            evidence.world_model_sha256_before == evidence.world_model_sha256_after
            and not evidence.locked_test_accessed
            and not evidence.gpu_used
            and not evidence.dataset_regenerated
            and not evidence.world_model_retrained
        ),
    }
    blockers = tuple(name for name, passed in gates.items() if not passed)
    return GPUReadinessAssessment(
        schema_version=R6_GPU_READINESS_SCHEMA,
        r6_gpu_strategy_training_ready=not blockers,
        blockers=blockers,
        gate_results=gates,
        locked_test_accessed=bool(evidence.locked_test_accessed),
        gpu_used=bool(evidence.gpu_used),
        dataset_regenerated=bool(evidence.dataset_regenerated),
        world_model_retrained=bool(evidence.world_model_retrained),
    )


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    values = [dict(row) for row in rows]
    fields: list[str] = []
    for row in values:
        for key in row:
            if str(key) not in fields:
                fields.append(str(key))
    if not fields:
        fields = ["empty"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in values:
            writer.writerow(row)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_gpu_readiness_bundle(
    output_dir: str | Path,
    *,
    assessment: GPUReadinessAssessment,
    evidence: GPUReadinessEvidence,
    input_bindings: Mapping[str, str],
    protocol_payload: Mapping[str, Any],
    action_rows: Sequence[Mapping[str, Any]],
    transition_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    bindings = {str(key): str(value) for key, value in input_bindings.items()}
    invalid = sorted(key for key, value in bindings.items() if not _is_sha256(value))
    if invalid:
        raise ValueError(f"input bindings are not SHA-256 digests: {invalid}")
    summary = asdict(assessment)
    summary["evidence_counts"] = {
        "offload_nonnoop": evidence.offload_nonnoop_count,
        "rb_nonnoop": evidence.rb_nonnoop_count,
        "cpu_nonnoop": evidence.cpu_nonnoop_count,
        "real_rollout_transitions": evidence.real_rollout_transition_count,
    }
    _write_json(output / "summary.json", summary)
    _write_json(output / "evidence.json", asdict(evidence))
    _write_json(output / "input_bindings.json", bindings)
    _write_json(output / "gpu_training_protocol.json", dict(protocol_payload))
    _write_csv(output / "action_ledger.csv", action_rows)
    _write_csv(output / "transition_ledger.csv", transition_rows)
    paths = [
        output / "summary.json",
        output / "evidence.json",
        output / "input_bindings.json",
        output / "gpu_training_protocol.json",
        output / "action_ledger.csv",
        output / "transition_ledger.csv",
    ]
    rows = [
        {"path": path.name, "sha256": _sha256(path), "bytes": path.stat().st_size}
        for path in paths
    ]
    self_check = all(_sha256(output / row["path"]) == row["sha256"] for row in rows)
    manifest = {
        "schema_version": "PIJWM-R6-joint-policy-GPU-readiness-bundle-v1",
        "files": rows,
        "self_check_passed": self_check,
        "r6_gpu_strategy_training_ready": assessment.r6_gpu_strategy_training_ready,
        "locked_test_accessed": assessment.locked_test_accessed,
        "gpu_used": assessment.gpu_used,
    }
    _write_json(output / "manifest.json", manifest)
    return manifest
