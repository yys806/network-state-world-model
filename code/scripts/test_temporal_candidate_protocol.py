from __future__ import annotations

import unittest
import sys
from pathlib import Path

import pandas as pd


SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from audit_temporal_candidate_protocol import build_pretraining_gate


class TemporalCandidateProtocolAuditTest(unittest.TestCase):
    def test_gate_allows_formal_labels_but_not_selector_gpu_training(self):
        summary = {
            "seeds": [0, 1, 2, 3, 4],
            "quality_audit": {
                "passed": True,
                "missing_numeric_values": 0,
                "negative_energy_rows": 0,
                "reward_reconstruction_errors": 0,
                "energy_balance_errors": 0,
                "invalid_action_rows": 0,
            },
        }
        candidates = pd.DataFrame(
            [
                {"seed": 0, "decision_time": 1.0, "action_family": "default", "action_applied": True},
                {
                    "seed": 0,
                    "decision_time": 1.0,
                    "action_family": "cpu_scale",
                    "action_applied": True,
                    "action_supported": True,
                    "action_changed": True,
                },
                {"seed": 1, "decision_time": 2.0, "action_family": "default", "action_applied": True},
                {
                    "seed": 1,
                    "decision_time": 2.0,
                    "action_family": "rb_count",
                    "action_applied": True,
                    "action_supported": False,
                    "action_changed": False,
                },
            ]
        )
        groups = pd.DataFrame(
            [
                {"seed": 0, "decision_time": 1.0, "is_nontrivial": True},
                {"seed": 1, "decision_time": 2.0, "is_nontrivial": False},
            ]
        )

        gate = build_pretraining_gate(summary, candidates, groups)

        self.assertEqual(gate["status"], "ready_for_formal_label_generation")
        self.assertTrue(gate["bridge_smoke_gate_passed"])
        self.assertFalse(gate["gpu_selector_training_allowed"])
        self.assertEqual(gate["non_default_applicability_rate"], 1.0)
        self.assertEqual(gate["supported_non_default_candidates"], 1)
        self.assertEqual(gate["nontrivial_group_ratio"], 0.5)
        self.assertFalse(gate["matched_test_accessed"])
        self.assertFalse(gate["external_holdout_accessed"])

    def test_gate_rejects_measurement_failure(self):
        summary = {
            "seeds": [0],
            "quality_audit": {"passed": False, "invalid_action_rows": 1},
        }
        candidates = pd.DataFrame(
            [
                {"seed": 0, "decision_time": 1.0, "action_family": "default", "action_applied": True},
                {"seed": 0, "decision_time": 1.0, "action_family": "rb_count", "action_applied": False},
            ]
        )
        groups = pd.DataFrame(
            [{"seed": 0, "decision_time": 1.0, "is_nontrivial": False}]
        )

        gate = build_pretraining_gate(summary, candidates, groups)

        self.assertEqual(gate["status"], "measurement_gate_failed")
        self.assertFalse(gate["bridge_smoke_gate_passed"])


if __name__ == "__main__":
    unittest.main()
