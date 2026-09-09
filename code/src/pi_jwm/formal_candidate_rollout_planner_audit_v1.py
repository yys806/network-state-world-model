"""Mechanism gate for a candidate-action PI-JWM rollout planner claim."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


SCHEMA_VERSION = "PI-JWM-formal-candidate-rollout-planner-audit-v1"
_REQUIRED_CHECKS = (
    "legal_candidate_actions_generated",
    "common_belief_reused_per_candidate",
    "world_model_invoked_per_candidate",
    "candidate_action_injected_into_rollout",
    "candidate_future_state_extracted",
    "candidate_task_cost_risk_extracted",
    "selection_uses_predicted_future_objective",
    "selected_first_action_feedback_replanning",
    "world_model_action_conditioned",
)
_MISMATCHES = {
    "legal_candidate_actions_generated": "legal_candidate_generation_missing",
    "common_belief_reused_per_candidate": "common_starting_belief_per_candidate_missing",
    "world_model_invoked_per_candidate": "candidate_world_model_rollout_missing",
    "candidate_action_injected_into_rollout": "candidate_action_rollout_injection_missing",
    "candidate_future_state_extracted": "candidate_future_state_extraction_missing",
    "candidate_task_cost_risk_extracted": "candidate_task_cost_risk_extraction_missing",
    "selection_uses_predicted_future_objective": "predicted_future_objective_selection_missing",
    "selected_first_action_feedback_replanning": "selected_first_action_feedback_replanning_missing",
    "world_model_action_conditioned": "action_conditioned_world_model_missing",
}


def evaluate_candidate_rollout_planner_claims(observed: Mapping[str, Any]) -> dict[str, Any]:
    """Gate planner claims from explicit, independently observed mechanism facts."""

    checks = {name: bool(observed.get(name, False)) for name in _REQUIRED_CHECKS}
    critical_mismatches = [
        _MISMATCHES[name] for name in _REQUIRED_CHECKS if not checks[name]
    ]
    ready = not critical_mismatches
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "ready_for_review" if ready else "blocked",
        "checks": checks,
        "critical_mismatches": critical_mismatches,
        "method_boundary": {
            "required_definition": (
                "For each legal candidate from one common belief/history, invoke the "
                "action-conditioned world model with that candidate, extract its predicted "
                "future state/task/cost/risk outcomes, select using those outcomes, execute "
                "only the selected first action, then feed back and replan."
            ),
            "direct_candidate_scorer_is_rollout_planner": False,
        },
        "launch_gates": {
            "planner_gpu_experiment_allowed": ready,
            "planner_method_freeze_allowed": ready,
            "formal_performance_claim_allowed": False,
            "locked_test_allowed": False,
        },
    }


__all__ = ["SCHEMA_VERSION", "evaluate_candidate_rollout_planner_claims"]
