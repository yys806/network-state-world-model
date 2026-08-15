from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class TemporalCandidatePolicyTest(unittest.TestCase):
    def test_formal_policy_rejects_task_id_payloads(self):
        from pi_jwm.temporal_candidate_policy import validate_temporal_candidate

        with self.assertRaisesRegex(ValueError, "task-ID payload"):
            validate_temporal_candidate(
                {
                    "action_protocol": "causal_policy_v1",
                    "action_family": "rb_count",
                    "rb_scale": 0.5,
                    "policy_coverage": 1,
                    "rb_plan": {"task-1": [0]},
                }
            )

    def test_converts_legacy_descriptor_without_task_ids(self):
        from pi_jwm.temporal_candidate_policy import to_causal_policy_candidate

        result = to_causal_policy_candidate(
            {
                "candidate_id": "mixed_task_alt_rb_2",
                "action_family": "mixed_offload_rb",
                "rb_scale": 0.5,
                "rb_plan": {"task-1": [0, 1]},
                "offload_overrides": {"task-1": "node-2"},
                "num_offload_overrides": 1,
            }
        )

        self.assertEqual(result["action_protocol"], "causal_policy_v1")
        self.assertEqual(result["policy_coverage"], 1)
        self.assertEqual(result["policy_rank"], 1)
        self.assertEqual(result["rb_plan"], {})
        self.assertEqual(result["offload_overrides"], {})
        self.assertNotIn("task-1", repr(result))

    def test_legacy_default_id_overrides_mislabeled_family(self):
        from pi_jwm.temporal_candidate_policy import to_causal_policy_candidate

        result = to_causal_policy_candidate(
            {
                "candidate_id": "default_no_rb",
                "action_family": "rb_count",
                "rb_scale": 0.0,
                "rb_plan": {},
            }
        )

        self.assertEqual(result["candidate_id"], "default")
        self.assertEqual(result["action_family"], "default")
        self.assertEqual(result["policy_coverage"], 0)

    def test_rejects_nonfinite_or_nonpositive_scales(self):
        from pi_jwm.temporal_candidate_policy import validate_temporal_candidate

        for value in (0.0, -1.0, math.nan, math.inf):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "rb_scale"):
                    validate_temporal_candidate(
                        {
                            "action_protocol": "causal_policy_v1",
                            "action_family": "rb_count",
                            "rb_scale": value,
                            "policy_coverage": 1,
                        }
                    )

    def test_scaled_counts_respect_capacity_and_stable_order(self):
        from pi_jwm.temporal_candidate_policy import project_scaled_counts

        result = project_scaled_counts(
            {"task-b": 2, "task-a": 2},
            scale=1.5,
            capacity=5,
        )

        self.assertEqual(result, {"task-a": 3, "task-b": 2})
        self.assertEqual(sum(result.values()), 5)
        self.assertTrue(all(value >= 0 for value in result.values()))

    def test_scaled_counts_change_downward_and_handle_empty_support(self):
        from pi_jwm.temporal_candidate_policy import project_scaled_counts

        self.assertEqual(
            project_scaled_counts({"a": 2, "b": 2}, scale=0.5, capacity=8),
            {"a": 1, "b": 1},
        )
        self.assertEqual(project_scaled_counts({}, scale=0.5, capacity=8), {})

    def test_active_step_audit_separates_applicable_from_changed(self):
        from pi_jwm.temporal_candidate_policy import summarize_active_action_steps

        unchanged = summarize_active_action_steps(
            [
                {"action_applicable": True, "action_changed": False},
                {"action_applicable": True, "action_changed": False},
            ]
        )
        changed = summarize_active_action_steps(
            [
                {"action_applicable": True, "action_changed": False},
                {"action_applicable": True, "action_changed": True},
            ]
        )

        self.assertEqual(unchanged, {"action_applicable": True, "action_changed": False})
        self.assertEqual(changed, {"action_applicable": True, "action_changed": True})
        self.assertEqual(
            summarize_active_action_steps([]),
            {"action_applicable": False, "action_changed": False},
        )


if __name__ == "__main__":
    unittest.main()
