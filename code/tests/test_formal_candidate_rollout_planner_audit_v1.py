from __future__ import annotations

import sys
import unittest
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class CandidateRolloutPlannerAuditV1Tests(unittest.TestCase):
    def test_static_inspection_keeps_direct_scoring_distinct_from_rollout(self) -> None:
        scripts_root = Path(__file__).resolve().parents[1] / "scripts"
        if str(scripts_root) not in sys.path:
            sys.path.insert(0, str(scripts_root))
        from run_formal_candidate_rollout_planner_audit_v1 import (
            inspect_candidate_planner_sources,
        )

        inspection = inspect_candidate_planner_sources(
            policy_source="candidate_descriptors self.candidate_encoder self.scorer",
            model_source=(
                'def forward(self): future_action["task_action"] '
                'range(self.config.horizon_steps)'
            ),
            transition_source="JointTransition candidate_index",
        )

        self.assertTrue(
            inspection["source_observations"]["direct_candidate_descriptor_scorer_detected"]
        )
        self.assertTrue(inspection["observed"]["world_model_action_conditioned"])
        self.assertFalse(inspection["observed"]["world_model_invoked_per_candidate"])
        self.assertFalse(
            inspection["observed"]["selection_uses_predicted_future_objective"]
        )

    def test_direct_candidate_scorer_is_not_promoted_to_a_rollout_planner(self) -> None:
        from pi_jwm.formal_candidate_rollout_planner_audit_v1 import (
            evaluate_candidate_rollout_planner_claims,
        )

        report = evaluate_candidate_rollout_planner_claims(
            {
                "legal_candidate_actions_generated": False,
                "common_belief_reused_per_candidate": False,
                "world_model_invoked_per_candidate": False,
                "candidate_action_injected_into_rollout": False,
                "candidate_future_state_extracted": False,
                "candidate_task_cost_risk_extracted": False,
                "selection_uses_predicted_future_objective": False,
                "selected_first_action_feedback_replanning": False,
                "world_model_action_conditioned": True,
            }
        )

        self.assertEqual("blocked", report["status"])
        self.assertIn(
            "candidate_world_model_rollout_missing", report["critical_mismatches"]
        )
        self.assertIn(
            "predicted_future_objective_selection_missing",
            report["critical_mismatches"],
        )
        self.assertFalse(report["launch_gates"]["planner_gpu_experiment_allowed"])
        self.assertFalse(report["launch_gates"]["locked_test_allowed"])

    def test_all_required_mechanisms_are_required_for_review(self) -> None:
        from pi_jwm.formal_candidate_rollout_planner_audit_v1 import (
            evaluate_candidate_rollout_planner_claims,
        )

        report = evaluate_candidate_rollout_planner_claims(
            {
                "legal_candidate_actions_generated": True,
                "common_belief_reused_per_candidate": True,
                "world_model_invoked_per_candidate": True,
                "candidate_action_injected_into_rollout": True,
                "candidate_future_state_extracted": True,
                "candidate_task_cost_risk_extracted": True,
                "selection_uses_predicted_future_objective": True,
                "selected_first_action_feedback_replanning": True,
                "world_model_action_conditioned": True,
            }
        )

        self.assertEqual("ready_for_review", report["status"])
        self.assertEqual([], report["critical_mismatches"])
        self.assertTrue(report["launch_gates"]["planner_gpu_experiment_allowed"])
        self.assertFalse(report["launch_gates"]["formal_performance_claim_allowed"])


if __name__ == "__main__":
    unittest.main()
