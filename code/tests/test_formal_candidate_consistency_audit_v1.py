from __future__ import annotations

import sys
import unittest
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class FormalCandidateConsistencyAuditV1Tests(unittest.TestCase):
    def test_source_observation_distinguishes_conditioning_from_rule_layer(self):
        scripts_root = Path(__file__).resolve().parents[1] / "scripts"
        if str(scripts_root) not in sys.path:
            sys.path.insert(0, str(scripts_root))
        from run_formal_candidate_consistency_audit_v1 import inspect_model_source

        source = '''
        residual_bases = {name: history[f"{name}_state"][:, -1] for name in COMPONENT_FEATURES}
        for step in range(self.config.horizon_steps):
            raw_action = future_action["task_action"][:, step].clone()
            task_state = self.task_encoder(history["task_state"])
            task = self.task_history(task_state)
            mean = residual_bases["node"] + self.config.residual_state_scale * mean
        '''
        observed = inspect_model_source(source)

        self.assertTrue(observed["residual_anchor_present"])
        self.assertTrue(observed["task_history_conditioning_present"])
        self.assertTrue(observed["future_action_conditioning_present"])
        self.assertFalse(observed["deterministic_rule_update_present"])

    def test_protocol_config_view_ignores_seed_only_variation(self):
        scripts_root = Path(__file__).resolve().parents[1] / "scripts"
        if str(scripts_root) not in sys.path:
            sys.path.insert(0, str(scripts_root))
        from run_formal_candidate_consistency_audit_v1 import protocol_config_view

        left = {"seed": 1, "learning_rate": 3e-4, "loss_weights": {"state_mae": 0.05}}
        right = {"seed": 2, "learning_rate": 3e-4, "loss_weights": {"state_mae": 0.05}}
        self.assertEqual(protocol_config_view(left), protocol_config_view(right))

    def test_rule_layer_input_contract_blocks_when_physical_context_is_missing(self):
        scripts_root = Path(__file__).resolve().parents[1] / "scripts"
        if str(scripts_root) not in sys.path:
            sys.path.insert(0, str(scripts_root))
        from run_formal_candidate_consistency_audit_v1 import assess_rule_layer_input_contract

        report = assess_rule_layer_input_contract(
            {
                "stats_passed_into_model": False,
                "future_source_endpoint_mapping": False,
                "future_service_outcome": False,
                "deterministic_target_masks": False,
            }
        )

        self.assertEqual("blocked", report["status"])
        self.assertEqual(4, len(report["missing_requirements"]))

    def test_missing_rule_layer_blocks_new_training_and_final_claims(self):
        from pi_jwm.formal_candidate_consistency_audit_v1 import evaluate_consistency_claims

        report = evaluate_consistency_claims(
            {
                "aggregate_boundary_matches": True,
                "residual_config_matches": True,
                "task_history_conditioning_matches": True,
                "future_action_conditioning_matches": True,
                "deterministic_rule_update_implemented": False,
                "per_rb_outputs_consumed": False,
            }
        )

        self.assertEqual("blocked", report["status"])
        self.assertIn("per_step_deterministic_rule_update_missing", report["critical_mismatches"])
        self.assertFalse(report["launch_gates"]["gpu_allowed"])
        self.assertFalse(report["launch_gates"]["formal_performance_claim_allowed"])

    def test_aggregate_baseline_boundary_does_not_require_per_rb_consumption(self):
        from pi_jwm.formal_candidate_consistency_audit_v1 import evaluate_consistency_claims

        report = evaluate_consistency_claims(
            {
                "aggregate_boundary_matches": True,
                "residual_config_matches": True,
                "task_history_conditioning_matches": True,
                "future_action_conditioning_matches": True,
                "deterministic_rule_update_implemented": True,
                "rule_layer_input_contract_ready": True,
                "recursive_rule_output_applied": True,
                "cpu_inner_rule_work_conserving": True,
                "no_future_state_endpoint_leakage": True,
                "candidate_checkpoint_rule_layer_enabled": True,
                "per_rb_outputs_consumed": False,
            }
        )

        self.assertEqual("ready_for_review", report["status"])
        self.assertNotIn("per_rb_consumption_missing", report["critical_mismatches"])
        self.assertTrue(report["launch_gates"]["gpu_allowed"])
        self.assertFalse(report["launch_gates"]["formal_performance_claim_allowed"])

    def test_ready_candidate_is_blocked_by_endpoint_leakage(self):
        from pi_jwm.formal_candidate_consistency_audit_v1 import evaluate_consistency_claims

        observed = {
            "aggregate_boundary_matches": True,
            "residual_config_matches": True,
            "task_history_conditioning_matches": True,
            "future_action_conditioning_matches": True,
            "deterministic_rule_update_implemented": True,
            "rule_layer_input_contract_ready": True,
            "recursive_rule_output_applied": True,
            "cpu_inner_rule_work_conserving": True,
            "no_future_state_endpoint_leakage": False,
            "candidate_checkpoint_rule_layer_enabled": True,
        }
        report = evaluate_consistency_claims(observed)
        self.assertEqual("blocked", report["status"])
        self.assertIn("future_state_endpoint_leakage", report["critical_mismatches"])


if __name__ == "__main__":
    unittest.main()
