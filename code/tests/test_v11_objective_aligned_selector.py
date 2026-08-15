import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def synthetic_batch_and_outcome():
    from pi_jwm.v11_selector import CandidateBatch, CandidateOutcome

    rng = np.random.default_rng(4)
    batch = CandidateBatch(
        context=rng.normal(size=(8, 3)).astype(np.float32),
        candidate_features=rng.normal(size=(8, 4, 5)).astype(np.float32),
        candidate_mask=np.ones((8, 4), dtype=bool),
        stage=np.asarray(["offload", "compute", "return", "unknown"] * 2),
        feature_names=("a", "b", "c", "d", "e"),
        candidate_names=("identity", "ranked", "repair_a", "repair_b"),
        context_feature_names=("ctx_a", "ctx_b", "ctx_c"),
    )
    active_sse = np.asarray(
        [
            [9.0, 10.0, 4.0, 12.0],
            [5.0, 8.0, 10.0, 7.0],
            [4.0, 6.0, 3.0, 8.0],
            [7.0, 9.0, 10.0, 2.0],
            [12.0, 10.0, 11.0, 8.0],
            [3.0, 5.0, 4.0, 6.0],
            [8.0, 8.0, 8.0, 8.0],
            [11.0, 12.0, 7.0, 13.0],
        ],
        dtype=np.float32,
    )
    outcome = CandidateOutcome(
        active_sse=active_sse,
        active_count=np.ones((8,), dtype=np.int64),
        default_index=1,
    )
    return batch, outcome


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


class ObjectiveAlignedFitTest(unittest.TestCase):
    def test_fit_rejects_cache_without_active_targets(self):
        from pi_jwm.v11_objective_aligned_selector import (
            fit_objective_aligned_selector,
        )
        from pi_jwm.v11_selector import CandidateOutcome

        batch, _ = synthetic_batch_and_outcome()
        outcome = CandidateOutcome(
            active_sse=np.zeros((8, 4), dtype=np.float32),
            active_count=np.zeros((8,), dtype=np.int64),
            default_index=1,
        )

        with self.assertRaisesRegex(ValueError, "active target"):
            fit_objective_aligned_selector(batch, outcome, hidden_dim=8, epochs=1)

    def test_fit_is_deterministic_and_freezes_train_only_scales(self):
        from pi_jwm.v11_objective_aligned_selector import (
            fit_objective_aligned_selector,
        )

        batch, outcome = synthetic_batch_and_outcome()
        first = fit_objective_aligned_selector(
            batch,
            outcome,
            hidden_dim=8,
            weight_cap=5.0,
            epochs=3,
            seed=17,
            group_ids=np.arange(8) % 2,
        )
        second = fit_objective_aligned_selector(
            batch,
            outcome,
            hidden_dim=8,
            weight_cap=5.0,
            epochs=3,
            seed=17,
            group_ids=np.arange(8) % 2,
        )

        self.assertEqual(first.benefit_scale, second.benefit_scale)
        np.testing.assert_array_equal(first.candidate_mean, second.candidate_mean)
        np.testing.assert_array_equal(first.context_scale, second.context_scale)
        for left, right in zip(first.model.parameters(), second.model.parameters()):
            torch.testing.assert_close(left, right)

    def test_uniform_impact_ablation_is_recorded_in_fitted_model(self):
        from pi_jwm.v11_objective_aligned_selector import (
            fit_objective_aligned_selector,
        )

        batch, outcome = synthetic_batch_and_outcome()
        fitted = fit_objective_aligned_selector(
            batch,
            outcome,
            hidden_dim=8,
            epochs=1,
            seed=17,
            impact_weighting=False,
        )

        self.assertFalse(fitted.impact_weighting)

    def test_prediction_returns_original_sse_units_and_shapes(self):
        from pi_jwm.v11_objective_aligned_selector import (
            fit_objective_aligned_selector,
            predict_objective_aligned_selector,
        )

        batch, outcome = synthetic_batch_and_outcome()
        fitted = fit_objective_aligned_selector(
            batch, outcome, hidden_dim=8, epochs=2, seed=17
        )

        prediction = predict_objective_aligned_selector(fitted, batch)

        self.assertEqual(prediction["predicted_candidate_benefit"].shape, (8, 4))
        self.assertEqual(prediction["candidate_uncertainty"].shape, (8, 4))
        self.assertEqual(prediction["predicted_opportunity"].shape, (8,))
        self.assertEqual(prediction["opportunity_uncertainty"].shape, (8,))
        for values in prediction.values():
            self.assertTrue(np.all(np.isfinite(values)))

    def test_checkpoint_loader_rejects_configuration_digest_mismatch(self):
        from pi_jwm.v11_objective_aligned_selector import (
            fit_objective_aligned_selector,
            load_objective_aligned_checkpoint,
            save_objective_aligned_checkpoint,
        )

        batch, outcome = synthetic_batch_and_outcome()
        fitted = fit_objective_aligned_selector(
            batch, outcome, hidden_dim=8, epochs=1, seed=17
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selector.pt"
            save_objective_aligned_checkpoint(
                path, fitted, configuration_digest="a" * 64, training_seed=17
            )

            with self.assertRaisesRegex(ValueError, "digest"):
                load_objective_aligned_checkpoint(
                    path, expected_configuration_digest="b" * 64
                )

    def test_checkpoint_roundtrip_preserves_predictions(self):
        from pi_jwm.v11_objective_aligned_selector import (
            fit_objective_aligned_selector,
            load_objective_aligned_checkpoint,
            predict_objective_aligned_selector,
            save_objective_aligned_checkpoint,
        )

        batch, outcome = synthetic_batch_and_outcome()
        fitted = fit_objective_aligned_selector(
            batch, outcome, hidden_dim=8, epochs=1, seed=29
        )
        expected = predict_objective_aligned_selector(fitted, batch)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selector.pt"
            save_objective_aligned_checkpoint(
                path, fitted, configuration_digest="c" * 64, training_seed=29
            )
            restored, metadata = load_objective_aligned_checkpoint(
                path, expected_configuration_digest="c" * 64
            )
            actual = predict_objective_aligned_selector(restored, batch)

        self.assertEqual(metadata["training_seed"], 29)
        for name in expected:
            np.testing.assert_array_equal(expected[name], actual[name])


class ObjectiveAlignedDecisionTest(unittest.TestCase):
    def test_calibration_uses_only_fixed_opportunity_quantiles(self):
        from pi_jwm.v11_objective_aligned_selector import (
            calibrate_opportunity_threshold,
        )
        from pi_jwm.v11_selector import CandidateOutcome

        opportunity_lcb = np.asarray([0.0, 1.0, 2.0, 3.0])
        candidate_choice = np.asarray([1, 1, 1, 1])
        outcome = CandidateOutcome(
            active_sse=np.asarray(
                [[4.0, 1.0], [4.0, 5.0], [4.0, 1.0], [4.0, 5.0]],
                dtype=np.float32,
            ),
            active_count=np.ones(4, dtype=np.int64),
            default_index=0,
        )

        result = calibrate_opportunity_threshold(
            opportunity_lcb, candidate_choice, outcome
        )

        self.assertIn(result.quantile, (0.0, 0.25, 0.5, 0.75, 0.9))
        self.assertEqual(len(result.curve), 5)

    def test_selection_requires_positive_opportunity_and_candidate_lcb(self):
        from pi_jwm.v11_objective_aligned_selector import select_objective_aligned

        decision = select_objective_aligned(
            ensemble_candidate_benefit=np.asarray(
                [[[0.0, 4.0]], [[0.0, 4.0]], [[0.0, 4.0]]], dtype=np.float32
            ),
            ensemble_candidate_uncertainty=np.zeros((3, 1, 2), dtype=np.float32),
            ensemble_opportunity=np.asarray([[5.0], [5.0], [5.0]], dtype=np.float32),
            ensemble_opportunity_uncertainty=np.zeros((3, 1), dtype=np.float32),
            candidate_mask=np.ones((1, 2), dtype=bool),
            default_index=0,
            opportunity_threshold=1.0,
        )

        self.assertEqual(decision.candidate_index[0], 1)
        self.assertFalse(decision.deferred[0])
        self.assertEqual(decision.defer_reason[0], "")

    def test_selection_defers_when_opportunity_is_below_threshold(self):
        from pi_jwm.v11_objective_aligned_selector import select_objective_aligned

        decision = select_objective_aligned(
            ensemble_candidate_benefit=np.asarray([[[0.0, 4.0]]], dtype=np.float32),
            ensemble_candidate_uncertainty=np.zeros((1, 1, 2), dtype=np.float32),
            ensemble_opportunity=np.asarray([[0.5]], dtype=np.float32),
            ensemble_opportunity_uncertainty=np.zeros((1, 1), dtype=np.float32),
            candidate_mask=np.ones((1, 2), dtype=bool),
            default_index=0,
            opportunity_threshold=1.0,
        )

        self.assertEqual(decision.candidate_index[0], 0)
        self.assertEqual(decision.defer_reason[0], "opportunity_below_threshold")

    def test_selection_defers_when_candidate_lcb_is_not_positive(self):
        from pi_jwm.v11_objective_aligned_selector import select_objective_aligned

        decision = select_objective_aligned(
            ensemble_candidate_benefit=np.asarray([[[0.0, 0.1]]], dtype=np.float32),
            ensemble_candidate_uncertainty=np.asarray([[[0.0, 1.0]]], dtype=np.float32),
            ensemble_opportunity=np.asarray([[5.0]], dtype=np.float32),
            ensemble_opportunity_uncertainty=np.zeros((1, 1), dtype=np.float32),
            candidate_mask=np.ones((1, 2), dtype=bool),
            default_index=0,
            opportunity_threshold=1.0,
        )

        self.assertEqual(decision.candidate_index[0], 0)
        self.assertEqual(decision.defer_reason[0], "candidate_nonpositive_lcb")

    def test_pareto_dominated_candidate_is_not_selected(self):
        from pi_jwm.v11_objective_aligned_selector import select_objective_aligned

        decision = select_objective_aligned(
            ensemble_candidate_benefit=np.asarray(
                [[[0.0, 5.0, 4.0]]], dtype=np.float32
            ),
            ensemble_candidate_uncertainty=np.zeros((1, 1, 3), dtype=np.float32),
            ensemble_opportunity=np.asarray([[6.0]], dtype=np.float32),
            ensemble_opportunity_uncertainty=np.zeros((1, 1), dtype=np.float32),
            candidate_mask=np.ones((1, 3), dtype=bool),
            default_index=0,
            opportunity_threshold=1.0,
            task_delta=np.asarray([[0.0, -1.0, 1.0]], dtype=np.float32),
            energy_delta=np.asarray([[0.0, 2.0, 1.0]], dtype=np.float32),
        )

        self.assertEqual(decision.candidate_index[0], 2)


if __name__ == "__main__":
    unittest.main()
