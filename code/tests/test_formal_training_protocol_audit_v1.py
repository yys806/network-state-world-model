from __future__ import annotations

import sys
import unittest
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class FormalTrainingProtocolAuditV1Tests(unittest.TestCase):
    def test_aggregate_baseline_passes_without_claiming_per_rb_consumption(self):
        from pi_jwm.formal_training_protocol_audit_v1 import audit_training_protocol

        report = audit_training_protocol(
            contract_mode="aggregate_baseline",
            tensor_contract={"max_physical_edges": 4, "n_rb": 2},
            tensor_validation={"formal_tensor_ready": True, "failed_checks": []},
            normalization_stats={"source_split": "train"},
            window_keys={
                "aggregate_link_activity",
                "aggregate_link_activity_mask",
                "aggregate_link_rate_sum",
                "aggregate_link_rate_sum_mask",
                "aggregate_rb_occupancy",
                "aggregate_rb_occupancy_mask",
                "link_activity_by_rb",
                "link_activity_mask_by_rb",
                "link_rate_by_rb",
                "link_rate_by_rb_mask",
            },
            model_outputs={"link_activity_logits", "physical_edge_state_mean"},
            loss_targets={
                "aggregate_link_activity",
                "aggregate_link_activity_mask",
                "aggregate_link_rate_sum",
                "aggregate_link_rate_sum_mask",
                "aggregate_rb_occupancy",
                "aggregate_rb_occupancy_mask",
            },
            metric_sources={
                "aggregate_link_activity",
                "aggregate_link_rate_sum",
                "aggregate_rb_occupancy",
            },
            locked_test_materialized=False,
        )

        self.assertTrue(report["formal_training_ready"])
        self.assertEqual("aggregate_baseline", report["contract_mode"])
        self.assertEqual("per_rb_target_sidecar", report["sidecar"]["field"])
        self.assertEqual("retained_diagnostic_only", report["sidecar"]["status"])
        self.assertEqual([], report["critical_mismatches"])

        from pi_jwm.formal_training_protocol_audit_v1 import build_training_protocol_freeze

        freeze = build_training_protocol_freeze(report)
        self.assertTrue(freeze["launch_gates"]["cpu_training_allowed"])
        self.assertFalse(freeze["launch_gates"]["gpu_allowed"])
        self.assertEqual(
            "pending_cpu_baseline_vs_persistence_gate",
            freeze["launch_gates"]["gpu_blocker"],
        )

    def test_blocks_when_model_loss_and_metrics_do_not_consume_per_rb_contract(self):
        from pi_jwm.formal_training_protocol_audit_v1 import audit_training_protocol

        report = audit_training_protocol(
            tensor_contract={
                "max_physical_edges": 4,
                "n_rb": 2,
            },
            tensor_validation={"formal_tensor_ready": True, "failed_checks": []},
            normalization_stats={"source_split": "train"},
            window_keys={
                "link_activity_by_rb",
                "link_activity_mask_by_rb",
                "link_rate_by_rb",
                "link_rate_by_rb_mask",
            },
            model_outputs={"link_activity_logits"},
            loss_targets={"link_activity"},
            metric_sources={"link_activity", "physical_edge_state.rate_sum"},
            locked_test_materialized=False,
        )

        self.assertFalse(report["formal_training_ready"])
        self.assertEqual("blocked", report["status"])
        self.assertIn("model_missing_per_rb_outputs", report["critical_mismatches"])
        self.assertIn("loss_missing_per_rb_targets_or_masks", report["critical_mismatches"])
        self.assertIn("metrics_missing_per_rb_definitions", report["critical_mismatches"])

    def test_passes_only_when_all_per_rb_consumers_and_safety_gates_are_present(self):
        from pi_jwm.formal_training_protocol_audit_v1 import audit_training_protocol

        report = audit_training_protocol(
            tensor_contract={"max_physical_edges": 4, "n_rb": 2},
            tensor_validation={"formal_tensor_ready": True, "failed_checks": []},
            normalization_stats={"source_split": "train"},
            window_keys={
                "link_activity_by_rb",
                "link_activity_mask_by_rb",
                "link_rate_by_rb",
                "link_rate_by_rb_mask",
            },
            model_outputs={"link_activity_by_rb_logits", "link_rate_by_rb_mean"},
            loss_targets={
                "link_activity_by_rb",
                "link_activity_mask_by_rb",
                "link_rate_by_rb",
                "link_rate_by_rb_mask",
            },
            metric_sources={"link_activity_by_rb", "link_rate_by_rb"},
            locked_test_materialized=False,
        )

        self.assertTrue(report["formal_training_ready"])
        self.assertEqual("ready_for_protocol_review", report["status"])
        self.assertEqual([], report["critical_mismatches"])

    def test_freeze_records_blocked_launch_policy_and_fixed_data_protocol(self):
        from pi_jwm.formal_training_protocol_audit_v1 import build_training_protocol_freeze

        freeze = build_training_protocol_freeze(
            {
                "status": "blocked",
                "formal_training_ready": False,
                "critical_mismatches": ["model_missing_per_rb_outputs"],
                "contract": {
                    "history_steps": 8,
                    "horizon_steps": 3,
                    "n_rb": 50,
                    "max_physical_edges": 1980,
                    "normalization_source_split": "train",
                    "locked_test_policy": "untensorized_until_explicit_unlock",
                },
            }
        )

        self.assertEqual("blocked", freeze["protocol_status"])
        self.assertEqual(["train", "validation", "calibration"], freeze["data_protocol"]["splits"])
        self.assertFalse(freeze["launch_gates"]["gpu_allowed"])
        self.assertFalse(freeze["launch_gates"]["locked_test_allowed"])
        self.assertEqual(["model_missing_per_rb_outputs"], freeze["critical_mismatches"])


if __name__ == "__main__":
    unittest.main()
