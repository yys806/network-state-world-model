import unittest

import numpy as np
import torch


class InteractionBenefitRankerTest(unittest.TestCase):
    def _inputs(self):
        torch.manual_seed(7)
        candidate = torch.randn(3, 4, 5)
        context = torch.randn(3, 6)
        tokens = torch.randn(3, 4, 5, 7)
        token_mask = torch.tensor(
            [
                [[1, 1, 1, 0, 0], [1, 1, 0, 0, 0], [1, 1, 1, 1, 0], [0, 0, 0, 0, 0]],
                [[1, 1, 0, 0, 0], [1, 1, 1, 0, 0], [1, 0, 0, 0, 0], [1, 1, 1, 1, 1]],
                [[1, 0, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 1, 0, 0], [1, 1, 1, 1, 0]],
            ],
            dtype=torch.bool,
        )
        tokens = tokens.masked_fill(~token_mask[..., None], 0.0)
        candidate_mask = torch.ones(3, 4, dtype=torch.bool)
        stage = torch.tensor([1, 2, 3], dtype=torch.long)
        return candidate, context, tokens, token_mask, candidate_mask, stage

    def test_token_permutation_does_not_change_predictions(self):
        from pi_jwm.v11_interaction_selector import InteractionCandidateBenefitRanker

        model = InteractionCandidateBenefitRanker(5, 6, 7, hidden_dim=16, dropout=0.0)
        model.eval()
        inputs = self._inputs()
        original = model(*inputs)
        permutation = torch.tensor([2, 0, 4, 1, 3])
        permuted_tokens = inputs[2][:, :, permutation]
        permuted_mask = inputs[3][:, :, permutation]
        permuted = model(
            inputs[0], inputs[1], permuted_tokens, permuted_mask, inputs[4], inputs[5]
        )
        for name in ("candidate_score", "predicted_benefit", "candidate_sign_logit"):
            torch.testing.assert_close(original[name], permuted[name])
        torch.testing.assert_close(original["opportunity_logit"], permuted["opportunity_logit"])

    def test_candidate_permutation_is_equivariant(self):
        from pi_jwm.v11_interaction_selector import InteractionCandidateBenefitRanker

        model = InteractionCandidateBenefitRanker(5, 6, 7, hidden_dim=16, dropout=0.0)
        model.eval()
        inputs = self._inputs()
        original = model(*inputs)
        permutation = torch.tensor([2, 0, 3, 1])
        permuted = model(
            inputs[0][:, permutation],
            inputs[1],
            inputs[2][:, permutation],
            inputs[3][:, permutation],
            inputs[4][:, permutation],
            inputs[5],
        )
        for name in ("candidate_score", "predicted_benefit", "candidate_sign_logit"):
            torch.testing.assert_close(original[name][:, permutation], permuted[name])
        torch.testing.assert_close(original["opportunity_logit"], permuted["opportunity_logit"])

    def test_padding_token_values_are_ignored(self):
        from pi_jwm.v11_interaction_selector import InteractionCandidateBenefitRanker

        model = InteractionCandidateBenefitRanker(5, 6, 7, hidden_dim=16, dropout=0.0)
        model.eval()
        inputs = self._inputs()
        original = model(*inputs)
        changed_tokens = inputs[2].clone()
        changed_tokens[~inputs[3]] = 1e6
        changed = model(
            inputs[0], inputs[1], changed_tokens, inputs[3], inputs[4], inputs[5]
        )
        for name, expected in original.items():
            torch.testing.assert_close(expected, changed[name])


class OpportunityMaskedLossTest(unittest.TestCase):
    def test_ranking_loss_uses_only_positive_opportunity_rows(self):
        from pi_jwm.v11_interaction_selector import interaction_selector_loss

        outputs = {
            "opportunity_logit": torch.zeros(2),
            "candidate_score": torch.tensor([[0.0, 2.0, -1.0], [100.0, -100.0, 50.0]]),
            "predicted_benefit": torch.zeros(2, 3),
            "candidate_sign_logit": torch.zeros(2, 3),
        }
        benefit = torch.tensor([[0.0, 4.0, -2.0], [0.0, 0.0, -1.0]])
        mask = torch.ones(2, 3, dtype=torch.bool)
        first = interaction_selector_loss(outputs, benefit, mask, default_index=0)
        changed = {name: value.clone() for name, value in outputs.items()}
        changed["candidate_score"][1] = torch.tensor([-1e6, 1e6, 5e5])
        second = interaction_selector_loss(changed, benefit, mask, default_index=0)
        self.assertEqual(first["opportunity_row_count"], 1)
        self.assertAlmostEqual(float(first["ranking"]), float(second["ranking"]), places=6)
        self.assertTrue(np.isfinite(float(first["loss"])))


class InteractionTrainingProtocolTest(unittest.TestCase):
    def _dataset(self):
        from pi_jwm.v11_interactions import CandidateInteractionBatch
        from pi_jwm.v11_selector import CandidateBatch, CandidateOutcome

        rng = np.random.default_rng(13)
        sample_count, candidate_count, token_count = 12, 3, 4
        candidate_mask = np.ones((sample_count, candidate_count), dtype=bool)
        context = rng.normal(size=(sample_count, 4)).astype(np.float32)
        candidate = rng.normal(size=(sample_count, candidate_count, 5)).astype(np.float32)
        tokens = rng.normal(size=(sample_count, candidate_count, token_count, 6)).astype(
            np.float32
        )
        token_mask = np.ones((sample_count, candidate_count, token_count), dtype=bool)
        token_mask[:, :, -1] = False
        tokens[~token_mask] = 0.0
        edge_index = np.broadcast_to(
            np.arange(token_count, dtype=np.int32), token_mask.shape
        ).copy()
        edge_index[~token_mask] = -1
        batch = CandidateBatch(
            context=context,
            candidate_features=candidate,
            candidate_mask=candidate_mask,
            stage=np.asarray(["offload", "compute", "return"] * 4),
            feature_names=tuple(f"candidate_{i}" for i in range(5)),
            context_feature_names=tuple(f"context_{i}" for i in range(4)),
        )
        interaction = CandidateInteractionBatch(
            tokens=tokens,
            token_mask=token_mask,
            edge_index=edge_index,
            token_feature_names=tuple(f"token_{i}" for i in range(6)),
        )
        active_sse = np.full((sample_count, candidate_count), 10.0, dtype=np.float32)
        active_sse[:, 0] = 8.0
        active_sse[:, 1] = np.where(np.arange(sample_count) % 2 == 0, 2.0, 9.0)
        active_sse[:, 2] = np.where(np.arange(sample_count) % 2 == 1, 3.0, 9.0)
        outcome = CandidateOutcome(
            active_sse=active_sse,
            active_count=np.ones(sample_count, dtype=np.int64),
            default_index=0,
        )
        return batch, outcome, interaction

    def test_normalizer_ignores_padding_tokens(self):
        from pi_jwm.v11_interaction_selector import fit_interaction_normalizer

        batch, _, interaction = self._dataset()
        first = fit_interaction_normalizer(batch, interaction)
        changed = interaction.tokens.copy()
        changed[~interaction.token_mask] = 1e8
        object.__setattr__(interaction, "tokens", changed)
        second = fit_interaction_normalizer(batch, interaction)
        np.testing.assert_allclose(first.token_mean, second.token_mean)
        np.testing.assert_allclose(first.token_scale, second.token_scale)

    def test_safe_selection_applies_all_three_gates(self):
        from pi_jwm.v11_interaction_selector import select_interaction_candidates

        opportunity = np.asarray([0.9, 0.4, 0.9], dtype=np.float32)
        score = np.asarray([[0.0, 3.0, 2.0], [0.0, 4.0, 3.0], [0.0, 2.0, 5.0]])
        sign = np.asarray([[0.0, 0.8, 0.9], [0.0, 0.9, 0.9], [0.0, 0.8, 0.9]])
        mask = np.ones((3, 3), dtype=bool)
        pareto_allowed = np.asarray(
            [[1, 1, 1], [1, 1, 1], [1, 1, 0]], dtype=bool
        )
        choice = select_interaction_candidates(
            opportunity,
            score,
            sign,
            mask,
            default_index=0,
            opportunity_threshold=0.5,
            sign_threshold=0.7,
            pareto_allowed=pareto_allowed,
        )
        np.testing.assert_array_equal(choice, np.asarray([1, 0, 1]))

    def test_small_fit_is_deterministic(self):
        from pi_jwm.v11_interaction_selector import (
            fit_interaction_selector,
            predict_interaction_selector,
        )

        batch, outcome, interaction = self._dataset()
        first = fit_interaction_selector(
            batch, outcome, interaction, hidden_dim=8, epochs=2, batch_size=4, seed=17
        )
        second = fit_interaction_selector(
            batch, outcome, interaction, hidden_dim=8, epochs=2, batch_size=4, seed=17
        )
        first_prediction = predict_interaction_selector(first, batch, interaction, batch_size=5)
        second_prediction = predict_interaction_selector(second, batch, interaction, batch_size=5)
        for name in first_prediction:
            np.testing.assert_allclose(first_prediction[name], second_prediction[name])


if __name__ == "__main__":
    unittest.main()
