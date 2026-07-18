import sys
import unittest
from pathlib import Path

import numpy as np
import torch


CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class DecisionAlignedTargetsTest(unittest.TestCase):
    def test_targets_use_ranked_default_sse_and_masked_oracle(self):
        from pi_jwm.v11_objective_aligned_selector import build_decision_aligned_targets
        from pi_jwm.v11_selector import CandidateOutcome

        outcome = CandidateOutcome(
            active_sse=np.asarray(
                [[100.0, 80.0, 20.0], [9.0, 9.0, 12.0]], dtype=np.float32
            ),
            active_count=np.asarray([2, 1]),
            default_index=1,
        )
        mask = np.asarray([[True, True, True], [True, True, False]])

        targets = build_decision_aligned_targets(outcome, mask, weight_cap=5.0)

        np.testing.assert_allclose(targets.candidate_benefit[0], [-20.0, 0.0, 60.0])
        self.assertTrue(np.isnan(targets.candidate_benefit[1, 2]))
        np.testing.assert_allclose(targets.opportunity, [60.0, 0.0])
        self.assertEqual(targets.benefit_scale, 60.0)
        self.assertEqual(targets.positive_opportunity.tolist(), [True, False])

    def test_zero_active_rows_are_audited_but_not_trainable(self):
        from pi_jwm.v11_objective_aligned_selector import build_decision_aligned_targets
        from pi_jwm.v11_selector import CandidateOutcome

        outcome = CandidateOutcome(
            active_sse=np.zeros((1, 2), dtype=np.float32),
            active_count=np.zeros((1,), dtype=np.int64),
            default_index=1,
        )

        targets = build_decision_aligned_targets(
            outcome, np.ones((1, 2), dtype=bool)
        )

        self.assertFalse(targets.valid_sample[0])
        self.assertEqual(targets.sample_weight[0], 0.0)

    def test_weight_cap_limits_high_gain_outlier(self):
        from pi_jwm.v11_objective_aligned_selector import build_decision_aligned_targets
        from pi_jwm.v11_selector import CandidateOutcome

        outcome = CandidateOutcome(
            active_sse=np.asarray([[9.0, 10.0], [0.0, 1000.0]], dtype=np.float32),
            active_count=np.ones(2, dtype=np.int64),
            default_index=1,
        )

        targets = build_decision_aligned_targets(
            outcome,
            np.ones((2, 2), dtype=bool),
            weight_cap=5.0,
            benefit_scale=1.0,
        )

        self.assertEqual(float(targets.sample_weight.max()), 5.25)

    def test_default_candidate_must_be_available(self):
        from pi_jwm.v11_objective_aligned_selector import build_decision_aligned_targets
        from pi_jwm.v11_selector import CandidateOutcome

        outcome = CandidateOutcome(
            active_sse=np.ones((1, 2), dtype=np.float32),
            active_count=np.ones(1, dtype=np.int64),
            default_index=1,
        )

        with self.assertRaisesRegex(ValueError, "ranked default"):
            build_decision_aligned_targets(
                outcome, np.asarray([[True, False]], dtype=bool)
            )


class OpportunityBenefitRankerTest(unittest.TestCase):
    def test_weighted_listwise_prioritizes_high_impact_sample(self):
        from pi_jwm.v11_objective_aligned_selector import (
            weighted_listwise_benefit_loss,
        )

        benefit = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
        mask = torch.ones_like(benefit, dtype=torch.bool)
        high_first = weighted_listwise_benefit_loss(
            torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
            benefit,
            mask,
            torch.tensor([10.0, 1.0]),
        )
        high_second = weighted_listwise_benefit_loss(
            torch.tensor([[0.0, 1.0], [0.0, 1.0]]),
            benefit,
            mask,
            torch.tensor([10.0, 1.0]),
        )

        self.assertLess(float(high_first), float(high_second))

    def test_zero_weight_rows_do_not_affect_listwise_loss(self):
        from pi_jwm.v11_objective_aligned_selector import (
            weighted_listwise_benefit_loss,
        )

        predicted = torch.tensor([[2.0, 0.0], [100.0, -100.0]])
        benefit = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
        mask = torch.ones_like(benefit, dtype=torch.bool)

        combined = weighted_listwise_benefit_loss(
            predicted, benefit, mask, torch.tensor([1.0, 0.0])
        )
        first_only = weighted_listwise_benefit_loss(
            predicted[:1], benefit[:1], mask[:1], torch.tensor([1.0])
        )

        torch.testing.assert_close(combined, first_only)

    def test_model_is_candidate_permutation_equivariant(self):
        from pi_jwm.v11_objective_aligned_selector import OpportunityBenefitRanker

        torch.manual_seed(4)
        model = OpportunityBenefitRanker(5, 3, hidden_dim=8)
        model.eval()
        candidate = torch.randn(2, 4, 5)
        context = torch.randn(2, 3)
        mask = torch.ones(2, 4, dtype=torch.bool)
        permutation = torch.tensor([2, 0, 3, 1])

        original = model(candidate, context, mask)
        permuted = model(candidate[:, permutation], context, mask[:, permutation])

        inverse = torch.argsort(permutation)
        for field in ("predicted_candidate_benefit", "candidate_uncertainty"):
            torch.testing.assert_close(original[field], permuted[field][:, inverse])
        torch.testing.assert_close(
            original["predicted_opportunity"], permuted["predicted_opportunity"]
        )
        torch.testing.assert_close(
            original["opportunity_uncertainty"],
            permuted["opportunity_uncertainty"],
        )

    def test_masked_candidate_cannot_receive_a_rankable_benefit(self):
        from pi_jwm.v11_objective_aligned_selector import OpportunityBenefitRanker

        model = OpportunityBenefitRanker(2, 2, hidden_dim=4)
        output = model(
            torch.ones(1, 2, 2),
            torch.ones(1, 2),
            torch.tensor([[True, False]]),
        )

        self.assertLess(
            output["predicted_candidate_benefit"][0, 1].detach().item(), -1e8
        )
        self.assertEqual(
            output["candidate_uncertainty"][0, 1].detach().item(), 0.0
        )


if __name__ == "__main__":
    unittest.main()
