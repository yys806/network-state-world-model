from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
from torch import nn

SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
for path in (SCRIPTS_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _horizon(negative_count: int, raw_fp: int) -> dict[str, object]:
    return {
        "negative_count": negative_count,
        "raw_false_positive_at_frozen_threshold": {"count": raw_fp},
    }


class _FeedbackOnlyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.state_feedback = nn.ModuleDict({"physical_edge": nn.Identity()})


class DirectEdgeRuleFeedbackInterventionV1Tests(unittest.TestCase):
    def test_physical_edge_feedback_hook_zeroes_only_its_output_and_restores(self):
        from run_formal_p4_direct_edge_rule_feedback_intervention_v1 import (
            suppress_direct_physical_edge_rule_feedback,
        )

        model = _FeedbackOnlyModel()
        payload = torch.tensor([[1.0, -2.0]])
        self.assertTrue(torch.equal(model.state_feedback["physical_edge"](payload), payload))

        handle = suppress_direct_physical_edge_rule_feedback(model)
        try:
            self.assertTrue(
                torch.equal(
                    model.state_feedback["physical_edge"](payload),
                    torch.zeros_like(payload),
                )
            )
        finally:
            handle.remove()

        self.assertTrue(torch.equal(model.state_feedback["physical_edge"](payload), payload))

    def test_sufficiency_requires_both_h20_conditions(self):
        from run_formal_p4_direct_edge_rule_feedback_intervention_v1 import (
            classify_direct_path_sufficiency,
        )

        baseline = {
            "zero_bias_counterfactual_at_frozen_threshold": {"count": 100},
            "pre_bias_logit": {"q99": 5.0},
        }
        sufficient = {
            "zero_bias_counterfactual_at_frozen_threshold": {"count": 10},
            "pre_bias_logit": {"q99": 2.19},
        }
        not_sufficient_fp = {
            "zero_bias_counterfactual_at_frozen_threshold": {"count": 11},
            "pre_bias_logit": {"q99": 1.0},
        }
        not_sufficient_q99 = {
            "zero_bias_counterfactual_at_frozen_threshold": {"count": 1},
            "pre_bias_logit": {"q99": 2.1972245773362196},
        }

        self.assertEqual(
            classify_direct_path_sufficiency(baseline, sufficient),
            "sufficient_to_explain",
        )
        self.assertEqual(
            classify_direct_path_sufficiency(baseline, not_sufficient_fp),
            "not_sufficient_to_explain",
        )
        self.assertEqual(
            classify_direct_path_sufficiency(baseline, not_sufficient_q99),
            "not_sufficient_to_explain",
        )

    def test_contract_accepts_validation_only(self):
        from run_formal_p4_direct_edge_rule_feedback_intervention_v1 import (
            validate_intervention_contract,
        )

        self.assertEqual(
            validate_intervention_contract(
                {
                    "seed": 20260831,
                    "horizon_steps": 20,
                    "split": "validation",
                    "locked_test_accessed": False,
                }
            ),
            "validation",
        )
        with self.assertRaises(ValueError):
            validate_intervention_contract(
                {
                    "seed": 20260831,
                    "horizon_steps": 20,
                    "split": "calibration",
                    "locked_test_accessed": False,
                }
            )

    def test_contract_rejects_seed_horizon_or_locked_test_mismatch(self):
        from run_formal_p4_direct_edge_rule_feedback_intervention_v1 import (
            validate_intervention_contract,
        )

        for invalid in (
            {"seed": 1, "horizon_steps": 20, "split": "validation", "locked_test_accessed": False},
            {"seed": 20260831, "horizon_steps": 19, "split": "validation", "locked_test_accessed": False},
            {"seed": 20260831, "horizon_steps": 20, "split": "validation", "locked_test_accessed": True},
        ):
            with self.assertRaises(ValueError):
                validate_intervention_contract(invalid)

    def test_contract_rejects_missing_locked_test_accessed(self):
        from run_formal_p4_direct_edge_rule_feedback_intervention_v1 import (
            validate_intervention_contract,
        )

        with self.assertRaises(ValueError):
            validate_intervention_contract(
                {
                    "seed": 20260831,
                    "horizon_steps": 20,
                    "split": "validation",
                }
            )

    def test_baseline_alignment_rejects_negative_count_and_false_positive_mismatch(self):
        from run_formal_p4_direct_edge_rule_feedback_intervention_v1 import (
            assert_v2_baseline_alignment,
        )

        expected = [_horizon(20 + index, 2 + index) for index in range(20)]
        self.assertIsNone(assert_v2_baseline_alignment(expected, expected))
        with self.assertRaises(AssertionError):
            assert_v2_baseline_alignment([_horizon(19, 2), *expected[1:]], expected)
        with self.assertRaises(AssertionError):
            assert_v2_baseline_alignment([_horizon(20, 1), *expected[1:]], expected)

    def test_v2_input_provenance_rejects_checkpoint_or_sample_id_drift(self):
        from run_formal_p4_direct_edge_rule_feedback_intervention_v1 import (
            assert_v2_input_provenance,
        )

        expected = {
            "config_sha256": "config",
            "sample_ids_sha256": "samples",
            "best_checkpoint_sha256": "checkpoint",
            "tensor_manifest_sha256": "manifest",
        }
        self.assertIsNone(assert_v2_input_provenance(expected, expected))
        drifted = {**expected, "best_checkpoint_sha256": "different-checkpoint"}
        with self.assertRaises(AssertionError):
            assert_v2_input_provenance(drifted, expected)

    def test_not_sufficient_decision_has_non_causal_boundary_text(self):
        from run_formal_p4_direct_edge_rule_feedback_intervention_v1 import (
            sufficiency_decision_boundary,
        )

        boundary = sufficiency_decision_boundary("not_sufficient_to_explain")
        self.assertIn("direct path alone", boundary["interpretation"])
        self.assertIn("does not mean no contribution", boundary["interpretation"])
        self.assertIn("does not authorize", boundary["interpretation"])
        self.assertFalse(boundary["authorizes_other_path_attribution"])
        self.assertFalse(boundary["authorizes_repair"])


if __name__ == "__main__":
    unittest.main()
