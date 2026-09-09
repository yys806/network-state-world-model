"""Theory-code consistency gates for the selected formal aggregate candidate."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


SCHEMA_VERSION = "PI-JWM-formal-candidate-consistency-audit-v1"
_REQUIRED_CHECKS = (
    "aggregate_boundary_matches",
    "residual_config_matches",
    "task_history_conditioning_matches",
    "future_action_conditioning_matches",
    "deterministic_rule_update_implemented",
    "rule_layer_input_contract_ready",
    "recursive_rule_output_applied",
    "cpu_inner_rule_work_conserving",
    "no_future_state_endpoint_leakage",
    "candidate_checkpoint_rule_layer_enabled",
)


def evaluate_consistency_claims(observed: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the pre-training theory-code consistency gate to observed facts."""

    checks = {name: bool(observed.get(name, False)) for name in _REQUIRED_CHECKS}
    critical_mismatches: list[str] = []
    if not checks["aggregate_boundary_matches"]:
        critical_mismatches.append("aggregate_boundary_mismatch")
    if not checks["residual_config_matches"]:
        critical_mismatches.append("residual_configuration_mismatch")
    if not checks["task_history_conditioning_matches"]:
        critical_mismatches.append("task_history_conditioning_mismatch")
    if not checks["future_action_conditioning_matches"]:
        critical_mismatches.append("future_action_conditioning_mismatch")
    if not checks["deterministic_rule_update_implemented"]:
        critical_mismatches.append("per_step_deterministic_rule_update_missing")
    if not checks["rule_layer_input_contract_ready"]:
        critical_mismatches.append("rule_layer_input_contract_incomplete")
    if not checks["recursive_rule_output_applied"]:
        critical_mismatches.append("recursive_rule_output_not_applied")
    if not checks["cpu_inner_rule_work_conserving"]:
        critical_mismatches.append("cpu_inner_rule_not_work_conserving")
    if not checks["no_future_state_endpoint_leakage"]:
        critical_mismatches.append("future_state_endpoint_leakage")
    if not checks["candidate_checkpoint_rule_layer_enabled"]:
        critical_mismatches.append("candidate_checkpoint_requires_rule_layer_retraining")

    ready = not critical_mismatches
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "ready_for_review" if ready else "blocked",
        "checks": checks,
        "critical_mismatches": critical_mismatches,
        "sidecar": {
            "field": "per_rb_target_sidecar",
            "consumed_by_current_model": bool(observed.get("per_rb_outputs_consumed", False)),
            "required_for_aggregate_baseline": False,
        },
        "launch_gates": {
            "cpu_training_allowed": ready,
            "gpu_allowed": ready,
            "formal_performance_claim_allowed": False,
            "locked_test_allowed": False,
        },
    }


__all__ = ["SCHEMA_VERSION", "evaluate_consistency_claims"]
