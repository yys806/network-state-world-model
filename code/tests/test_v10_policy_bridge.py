import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


class V10PolicyBridgeDecoderTest(unittest.TestCase):
    def test_bridge_result_protocol_distinguishes_autonomous_and_reference_modes(self):
        from evaluate_v10_policy_bridge import make_bridge_result_protocol

        deployable = make_bridge_result_protocol(
            action_generator="policy",
            action_decoder="threshold",
            mode="predicted_all",
            fit_splits=("train", "val"),
        )
        reference = make_bridge_result_protocol(
            action_generator="policy",
            action_decoder="threshold",
            mode="true_first_pred_rest",
            fit_splits=("train", "val"),
        )

        self.assertEqual(deployable["result_kind"], "deployable")
        self.assertEqual(reference["result_kind"], "true_future_reference")
        self.assertEqual(reference["evaluation_split"], "test")

    def test_decode_action_activity_topk_selects_budgeted_edges_per_horizon_and_action(self):
        from evaluate_v10_policy_bridge import decode_action_activity_topk

        prob = torch.tensor(
            [
                [[0.8, 0.1], [0.3, 0.9], [0.7, 0.4], [0.2, 0.6]],
                [[0.1, 0.5], [0.9, 0.4], [0.6, 0.8], [0.3, 0.7]],
            ],
            dtype=torch.float32,
        )
        counts = torch.tensor([[2, 1], [1, 2]], dtype=torch.long)

        decoded = decode_action_activity_topk(prob, counts)

        expected = torch.tensor(
            [
                [[True, False], [False, True], [True, False], [False, False]],
                [[False, False], [True, False], [False, True], [False, True]],
            ]
        )
        self.assertTrue(torch.equal(decoded.cpu(), expected))

    def test_summarize_action_count_budget_uses_validation_distribution(self):
        from evaluate_v10_policy_bridge import summarize_action_count_budget

        true_value = np.zeros((3, 2, 4, 2), dtype=np.float32)
        true_value[0, 0, [0, 1], 0] = 1.0
        true_value[1, 0, [2], 0] = 1.0
        true_value[2, 0, [0, 1, 2], 0] = 1.0
        true_value[0, 1, [3], 1] = 1.0
        true_value[1, 1, [2, 3], 1] = 1.0

        mean_budget = summarize_action_count_budget(true_value, reducer="mean", quantile=0.5)
        median_budget = summarize_action_count_budget(true_value, reducer="quantile", quantile=0.5)

        np.testing.assert_array_equal(mean_budget, np.array([[2, 0], [0, 1]], dtype=np.int64))
        np.testing.assert_array_equal(median_budget, np.array([[2, 0], [0, 1]], dtype=np.int64))

    def test_summarize_edge_count_budget_counts_active_edges_once(self):
        from evaluate_v10_policy_bridge import summarize_edge_count_budget

        true_value = np.zeros((3, 1, 4, 3), dtype=np.float32)
        true_value[0, 0, 0, [0, 1]] = 1.0
        true_value[0, 0, 1, [1, 2]] = 1.0
        true_value[1, 0, 2, [0, 2]] = 1.0
        true_value[2, 0, [0, 1, 2], 0] = 1.0

        mean_budget = summarize_edge_count_budget(true_value, reducer="mean", quantile=0.5)
        q50_budget = summarize_edge_count_budget(true_value, reducer="quantile", quantile=0.5)

        np.testing.assert_array_equal(mean_budget, np.array([2], dtype=np.int64))
        np.testing.assert_array_equal(q50_budget, np.array([2], dtype=np.int64))

    def test_decode_edge_activity_topk_selects_edges_and_expands_dims(self):
        from evaluate_v10_policy_bridge import decode_edge_activity_topk

        prob = torch.tensor(
            [
                [[0.1, 0.2], [0.8, 0.1], [0.3, 0.4]],
                [[0.9, 0.1], [0.2, 0.7], [0.1, 0.2]],
            ],
            dtype=torch.float32,
        )
        counts = torch.tensor([1, 2], dtype=torch.long)

        decoded = decode_edge_activity_topk(prob, counts)

        expected = torch.tensor(
            [
                [[False, False], [True, True], [False, False]],
                [[True, True], [True, True], [False, False]],
            ]
        )
        self.assertTrue(torch.equal(decoded.cpu(), expected))

    def test_decode_edge_threshold_topk_requires_threshold_and_budget(self):
        from evaluate_v10_policy_bridge import decode_edge_threshold_topk

        prob = torch.tensor(
            [
                [[0.6, 0.2], [0.4, 0.1], [0.9, 0.1]],
                [[0.3, 0.2], [0.2, 0.1], [0.1, 0.2]],
            ],
            dtype=torch.float32,
        )
        counts = torch.tensor([1, 1], dtype=torch.long)

        decoded = decode_edge_threshold_topk(prob, counts, threshold=0.5)

        expected = torch.tensor(
            [
                [[False, False], [False, False], [True, True]],
                [[False, False], [False, False], [False, False]],
            ]
        )
        self.assertTrue(torch.equal(decoded.cpu(), expected))

    def test_edge_decoders_can_use_separate_edge_probability(self):
        from evaluate_v10_policy_bridge import decode_edge_activity_topk_np, decode_edge_threshold_topk_np

        prob = np.zeros((1, 1, 3, 2), dtype=np.float32)
        edge_prob = np.array([[[0.1, 0.9, 0.6]]], dtype=np.float32)
        counts = np.array([[1]], dtype=np.int64)

        topk = decode_edge_activity_topk_np(prob, counts, edge_prob=edge_prob)
        threshold_topk = decode_edge_threshold_topk_np(prob, counts, threshold=0.5, edge_prob=edge_prob)

        np.testing.assert_array_equal(topk, np.array([[[False, True, False]]], dtype=bool))
        np.testing.assert_array_equal(threshold_topk, np.array([[[False, True, False]]], dtype=bool))

    def test_probability_mass_decoder_learns_scale_from_validation_counts(self):
        from evaluate_v10_policy_bridge import (
            calibrate_probability_budget_scales,
            decode_action_activity_topk_np,
        )

        prob = np.array(
            [
                [[[[0.9], [0.8], [0.1], [0.1]]]],
                [[[[0.7], [0.2], [0.1], [0.1]]]],
            ],
            dtype=np.float32,
        ).reshape(2, 1, 4, 1)
        true_value = np.zeros_like(prob, dtype=np.float32)
        true_value[0, 0, [0, 1], 0] = 1.0
        true_value[1, 0, [0], 0] = 1.0

        scales = calibrate_probability_budget_scales(prob, true_value)
        counts = np.rint(prob.sum(axis=2) * scales.reshape(1, 1, 1)).astype(np.int64)
        decoded = decode_action_activity_topk_np(prob, counts)

        self.assertGreater(scales[0, 0], 0.0)
        np.testing.assert_array_equal(decoded, true_value > 0.0)

    def test_edge_probability_mass_decoder_learns_scale_from_validation_counts(self):
        from evaluate_v10_policy_bridge import (
            calibrate_edge_probability_budget_scales,
            decode_edge_activity_topk_np,
        )

        prob = np.array(
            [
                [[0.9, 0.1], [0.8, 0.1], [0.1, 0.1], [0.1, 0.1]],
                [[0.7, 0.1], [0.2, 0.1], [0.1, 0.1], [0.1, 0.1]],
            ],
            dtype=np.float32,
        ).reshape(2, 1, 4, 2)
        true_value = np.zeros_like(prob, dtype=np.float32)
        true_value[0, 0, [0, 1], :] = 1.0
        true_value[1, 0, [0], :] = 1.0

        scales = calibrate_edge_probability_budget_scales(prob, true_value)
        edge_score = prob.max(axis=-1)
        counts = np.rint(edge_score.sum(axis=2) * scales.reshape(1, 1)).astype(np.int64)
        decoded = decode_edge_activity_topk_np(prob, counts)

        self.assertGreater(scales[0], 0.0)
        np.testing.assert_array_equal(decoded, np.any(true_value > 0.0, axis=-1))

    def test_edge_probability_mass_calibration_uses_edge_probability_when_available(self):
        from evaluate_v10_policy_bridge import calibrate_edge_probability_budget_scales, decode_edge_activity_topk_np

        prob = np.full((1, 1, 3, 2), 0.9, dtype=np.float32)
        edge_prob = np.array([[[0.1, 0.95, 0.2]]], dtype=np.float32)
        true_value = np.zeros_like(prob, dtype=np.float32)
        true_value[0, 0, 1, :] = 1.0

        scales = calibrate_edge_probability_budget_scales(prob, true_value, edge_prob=edge_prob)
        counts = np.rint(edge_prob.sum(axis=2) * scales.reshape(1, 1)).astype(np.int64)
        decoded = decode_edge_activity_topk_np(prob, counts, edge_prob=edge_prob)

        np.testing.assert_array_equal(decoded, np.array([[[False, True, False]]], dtype=bool))

    def test_make_action_decoder_config_passes_edge_probability_to_mass_calibrator(self):
        from evaluate_v10_policy_bridge import make_action_decoder_config

        prob = np.full((1, 1, 3, 2), 0.9, dtype=np.float32)
        edge_prob = np.array([[[0.1, 0.95, 0.2]]], dtype=np.float32)
        true_value = np.zeros_like(prob, dtype=np.float32)
        true_value[0, 0, 1, :] = 1.0

        config = make_action_decoder_config(
            "edge_probability_mass_topk",
            {"prob": prob, "edge_prob": edge_prob, "value_true": true_value},
            budget_quantile=0.5,
        )

        counts = np.rint(edge_prob.sum(axis=2) * config.probability_budget_scales.reshape(1, 1)).astype(np.int64)
        self.assertEqual(int(counts[0, 0]), 1)

    def test_scale_action_value_groups_scales_rb_and_cpu_dimensions(self):
        from evaluate_v10_policy_bridge import scale_action_value_groups

        value = torch.ones((1, 2, 6), dtype=torch.float32)

        scaled = scale_action_value_groups(value, rb_dim_scale=0.5, cpu_dim_scale=2.0)

        expected = torch.tensor([1.0, 0.5, 0.5, 2.0, 2.0, 1.0], dtype=torch.float32)
        self.assertTrue(torch.equal(scaled[0, 0], expected))

    def test_step_total_calibrator_scales_rb_and_cpu_totals_by_predicted_count(self):
        from evaluate_v10_policy_bridge import apply_step_total_calibration_np, fit_step_total_calibrator

        pred = np.zeros((2, 1, 3, 6), dtype=np.float32)
        true = np.zeros_like(pred)
        pred[0, 0, [0, 1], 2] = [10.0, 10.0]
        pred[0, 0, [0, 1], 4] = [1.0, 1.0]
        pred[1, 0, [0], 2] = [4.0]
        pred[1, 0, [0], 4] = [2.0]
        true[0, 0, [0, 1], 2] = [30.0, 30.0]
        true[0, 0, [0, 1], 4] = [3.0, 3.0]
        true[1, 0, [0], 2] = [8.0]
        true[1, 0, [0], 4] = [4.0]

        calibrator = fit_step_total_calibrator(pred, true, quantile=0.5)
        calibrated = apply_step_total_calibration_np(pred, calibrator)

        self.assertAlmostEqual(float(calibrated[0, 0, :, 2].sum()), 60.0, places=5)
        self.assertAlmostEqual(float(calibrated[0, 0, :, 4].sum()), 6.0, places=5)
        self.assertAlmostEqual(float(calibrated[1, 0, :, 2].sum()), 8.0, places=5)
        self.assertAlmostEqual(float(calibrated[1, 0, :, 4].sum()), 4.0, places=5)

    def test_step_total_calibrator_keeps_zero_count_steps_zero(self):
        from evaluate_v10_policy_bridge import apply_step_total_calibration_np, fit_step_total_calibrator

        pred = np.zeros((1, 1, 2, 6), dtype=np.float32)
        true = np.ones_like(pred)

        calibrator = fit_step_total_calibrator(pred, true, quantile=0.5)
        calibrated = apply_step_total_calibration_np(pred, calibrator)

        self.assertEqual(float(calibrated[..., 2].sum()), 0.0)
        self.assertEqual(float(calibrated[..., 4].sum()), 0.0)

    def test_policy_step_total_controller_uses_predicted_totals(self):
        from evaluate_v10_policy_bridge import StepTotalCalibrator, apply_step_total_calibration_np

        actions = np.zeros((1, 1, 2, 6), dtype=np.float32)
        actions[0, 0, :, 2] = [10.0, 10.0]
        actions[0, 0, :, 4] = [1.0, 1.0]
        step_total_pred = np.array([[[2.0, 60.0, 8.0]]], dtype=np.float32)

        calibrated = apply_step_total_calibration_np(
            actions,
            StepTotalCalibrator(name="policy_step_total"),
            step_total_pred=step_total_pred,
        )

        self.assertAlmostEqual(float(calibrated[0, 0, :, 2].sum()), 60.0, places=5)
        self.assertAlmostEqual(float(calibrated[0, 0, :, 4].sum()), 8.0, places=5)

    def test_logistic_edge_reranker_scores_positive_edges_higher(self):
        from evaluate_v10_policy_bridge import fit_logistic_edge_reranker, predict_edge_reranker_scores

        prob = np.zeros((4, 1, 3, 2), dtype=np.float32)
        edge_prob = np.array(
            [
                [[0.9, 0.1, 0.2]],
                [[0.8, 0.2, 0.1]],
                [[0.2, 0.9, 0.1]],
                [[0.1, 0.8, 0.2]],
            ],
            dtype=np.float32,
        )
        value_pred = prob + edge_prob[..., None]
        value_true = np.zeros_like(prob)
        value_true[0, 0, 0, :] = 1.0
        value_true[1, 0, 0, :] = 1.0
        value_true[2, 0, 1, :] = 1.0
        value_true[3, 0, 1, :] = 1.0

        model = fit_logistic_edge_reranker(
            {"prob": prob, "edge_prob": edge_prob, "value_pred": value_pred, "value_true": value_true},
            max_train_rows=0,
            seed=7,
        )
        scores = predict_edge_reranker_scores(model, {"prob": prob, "edge_prob": edge_prob, "value_pred": value_pred})

        self.assertGreater(float(scores[0, 0, 0]), float(scores[0, 0, 1]))
        self.assertGreater(float(scores[2, 0, 1]), float(scores[2, 0, 0]))

    def test_edge_count_regressor_predicts_clipped_sample_step_counts(self):
        from evaluate_v10_policy_bridge import fit_edge_count_regressor, predict_edge_count_regressor

        sample_count = 100
        prob = np.zeros((sample_count, 1, 4, 2), dtype=np.float32)
        edge_prob = np.zeros((sample_count, 1, 4), dtype=np.float32)
        value_true = np.zeros_like(prob)
        expected_counts = np.array([sample_idx % 5 for sample_idx in range(sample_count)], dtype=np.int64)
        for sample_idx, count in enumerate(expected_counts):
            edge_prob[sample_idx, 0, :count] = 0.9
            value_true[sample_idx, 0, :count, :] = 1.0

        model = fit_edge_count_regressor(
            {"prob": prob, "edge_prob": edge_prob, "value_pred": prob, "value_true": value_true},
            seed=11,
        )
        counts = predict_edge_count_regressor(model, {"prob": prob, "edge_prob": edge_prob, "value_pred": prob})

        self.assertEqual(tuple(counts.shape), (sample_count, 1))
        self.assertTrue(np.all(counts >= 0))
        self.assertTrue(np.all(counts <= 4))
        self.assertGreaterEqual(float(np.corrcoef(counts.reshape(-1), expected_counts)[0, 1]), 0.5)

    def test_edge_count_offset_increases_counts_but_keeps_edge_limit(self):
        from evaluate_v10_policy_bridge import apply_edge_count_offset

        counts = np.array([[0, 1, 3]], dtype=np.int64)

        shifted = apply_edge_count_offset(counts, offset=2, num_edges=4)

        np.testing.assert_array_equal(shifted, np.array([[2, 3, 4]], dtype=np.int64))

    def test_repeat_first_future_actions_copies_known_first_step(self):
        from evaluate_v10_policy_bridge import repeat_first_future_actions

        true_future = torch.arange(2 * 3 * 2, dtype=torch.float32).reshape(2, 3, 2)

        repeated = repeat_first_future_actions(true_future)

        self.assertTrue(torch.equal(repeated[0], true_future[0]))
        self.assertTrue(torch.equal(repeated[1], true_future[0]))

    def test_repeat_last_history_actions_extends_latest_observed_action(self):
        from evaluate_v10_policy_bridge import repeat_last_history_actions

        history = torch.arange(4 * 3 * 2, dtype=torch.float32).reshape(4, 3, 2)

        repeated = repeat_last_history_actions(history, horizon=3)

        self.assertEqual(tuple(repeated.shape), (3, 3, 2))
        self.assertTrue(torch.equal(repeated[0], history[-1]))
        self.assertTrue(torch.equal(repeated[2], history[-1]))

    def test_policy_activity_true_value_uses_policy_mask_and_true_magnitude(self):
        policy_active = torch.tensor(
            [
                [[True, False], [False, True]],
                [[False, False], [True, True]],
            ]
        )
        true_future = torch.arange(2 * 2 * 2, dtype=torch.float32).reshape(2, 2, 2) + 1.0

        decoded = torch.where(policy_active, true_future, torch.zeros_like(true_future))

        self.assertEqual(float(decoded[0, 0, 0]), 1.0)
        self.assertEqual(float(decoded[0, 0, 1]), 0.0)
        self.assertEqual(float(decoded[1, 1, 0]), 7.0)
        self.assertEqual(float(decoded[1, 1, 1]), 8.0)

    def test_fit_action_value_prototype_uses_positive_values_only(self):
        from evaluate_v10_policy_bridge import fit_action_value_prototype

        values = np.zeros((3, 2, 2, 1), dtype=np.float32)
        values[:, 0, 0, 0] = np.array([0.0, 2.0, 6.0], dtype=np.float32)
        values[:, 0, 1, 0] = np.array([10.0, 0.0, 0.0], dtype=np.float32)
        values[:, 1, 0, 0] = np.array([1.0, 3.0, 5.0], dtype=np.float32)

        mean_proto = fit_action_value_prototype(values, decoder="train_mean", value_quantile=0.75)
        median_proto = fit_action_value_prototype(values, decoder="train_median", value_quantile=0.75)
        q75_proto = fit_action_value_prototype(values, decoder="train_q75", value_quantile=0.75)

        np.testing.assert_allclose(mean_proto[:, 0], np.array([6.0, 3.0], dtype=np.float32))
        np.testing.assert_allclose(median_proto[:, 0], np.array([6.0, 3.0], dtype=np.float32))
        np.testing.assert_allclose(q75_proto[:, 0], np.array([8.0, 4.0], dtype=np.float32))

    def test_edge_median_value_prototype_falls_back_when_edge_has_no_positive(self):
        from evaluate_v10_policy_bridge import fit_action_value_prototype

        values = np.zeros((3, 1, 2, 1), dtype=np.float32)
        values[:, 0, 0, 0] = np.array([0.0, 2.0, 6.0], dtype=np.float32)
        values[:, 0, 1, 0] = 0.0

        proto = fit_action_value_prototype(values, decoder="train_edge_median", value_quantile=0.75)

        self.assertEqual(tuple(proto.shape), (1, 2, 1))
        self.assertAlmostEqual(float(proto[0, 0, 0]), 4.0)
        self.assertAlmostEqual(float(proto[0, 1, 0]), 4.0)

    def test_dim_scaled_value_prototype_uses_validation_policy_mask(self):
        from evaluate_v10_policy_bridge import fit_action_value_prototype

        train = np.zeros((2, 1, 2, 1), dtype=np.float32)
        train[:, 0, :, 0] = np.array([[2.0, 0.0], [2.0, 0.0]], dtype=np.float32)
        val_predictions = {
            "prob": np.array([[[[0.9], [0.1]]]], dtype=np.float32),
            "value_true": np.array([[[[4.0], [100.0]]]], dtype=np.float32),
        }

        proto = fit_action_value_prototype(
            train,
            decoder="train_median_dim_scaled",
            value_quantile=0.75,
            val_predictions=val_predictions,
            policy_threshold=0.5,
        )

        self.assertEqual(tuple(proto.shape), (1, 1))
        self.assertAlmostEqual(float(proto[0, 0]), 4.0, places=5)

    def test_step_scaled_value_prototype_can_choose_different_step_scales(self):
        from evaluate_v10_policy_bridge import fit_action_value_prototype

        train = np.full((2, 2, 1, 1), 2.0, dtype=np.float32)
        val_predictions = {
            "prob": np.ones((1, 2, 1, 1), dtype=np.float32),
            "value_true": np.array([[[[2.0]], [[4.0]]]], dtype=np.float32),
        }

        proto = fit_action_value_prototype(
            train,
            decoder="train_median_step_scaled",
            value_quantile=0.75,
            val_predictions=val_predictions,
            policy_threshold=0.5,
        )

        self.assertAlmostEqual(float(proto[0, 0]), 2.0, places=5)
        self.assertAlmostEqual(float(proto[1, 0]), 4.0, places=5)

    def test_codebook_quantile_value_prototype_uses_positive_quantiles(self):
        from evaluate_v10_policy_bridge import fit_action_value_prototype

        values = np.zeros((4, 1, 2, 1), dtype=np.float32)
        values[:, 0, 0, 0] = np.array([0.0, 2.0, 6.0, 10.0], dtype=np.float32)
        values[:, 0, 1, 0] = np.array([14.0, 0.0, 0.0, 0.0], dtype=np.float32)

        codebook = fit_action_value_prototype(
            values,
            decoder="train_codebook_quantile",
            value_quantile=0.75,
            value_codebook_size=3,
        )

        self.assertEqual(tuple(codebook.shape), (1, 1, 3))
        np.testing.assert_allclose(codebook[0, 0], np.array([5.0, 8.0, 11.0], dtype=np.float32))

    def test_project_policy_value_to_codebook_uses_nearest_step_action_bin(self):
        from evaluate_v10_policy_bridge import project_policy_value_to_codebook

        policy_value = torch.tensor(
            [
                [[1.0, 9.0], [7.0, 30.0]],
                [[3.0, 14.0], [11.0, 1.0]],
            ],
            dtype=torch.float32,
        )
        codebook = torch.tensor(
            [
                [[0.0, 5.0, 10.0], [10.0, 20.0, 40.0]],
                [[2.0, 8.0, 12.0], [0.0, 4.0, 16.0]],
            ],
            dtype=torch.float32,
        )

        decoded = project_policy_value_to_codebook(policy_value, codebook)

        expected = torch.tensor(
            [
                [[0.0, 10.0], [5.0, 20.0]],
                [[2.0, 16.0], [12.0, 0.0]],
            ],
            dtype=torch.float32,
        )
        self.assertTrue(torch.equal(decoded, expected))

    def test_decode_discrete_policy_value_uses_checkpoint_vocab(self):
        from evaluate_v10_policy_bridge import decode_discrete_policy_value

        vocab = {
            "values": [[1.0, 2.0, 0.0], [5.0, 10.0, 20.0]],
            "sizes": [2, 3],
            "max_bins": 3,
        }
        logits = torch.tensor(
            [
                [[[[0.1, 0.9, 9.0], [0.1, 0.2, 0.3]], [[0.8, 0.2, 9.0], [3.0, 2.0, 1.0]]]],
            ],
            dtype=torch.float32,
        )

        decoded = decode_discrete_policy_value(logits, vocab)

        expected = torch.tensor([[[[2.0, 20.0], [1.0, 5.0]]]], dtype=torch.float32)
        self.assertTrue(torch.equal(decoded, expected))

    def test_decode_coupled_policy_value_uses_checkpoint_vocab(self):
        from evaluate_v10_policy_bridge import decode_coupled_policy_value

        vocab = {
            "mode": "coupled_tokens",
            "groups": [[0], [1, 2], [3, 4], [5]],
            "values": [
                [[0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
                [[0, 0, 0, 0, 0, 0], [0, 1, 25, 0, 0, 0], [0, 2, 50, 0, 0, 0]],
                [[0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 10, 0], [0, 0, 0, 2, 20, 0]],
                [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0]],
            ],
            "sizes": [2, 3, 3, 2],
            "max_tokens": 3,
            "action_dim": 6,
        }
        logits = torch.zeros((1, 1, 2, 4, 3), dtype=torch.float32)
        logits[0, 0, 0, 1, 2] = 4.0
        logits[0, 0, 0, 2, 1] = 4.0
        logits[0, 0, 1, 1, 1] = 4.0
        logits[0, 0, 1, 3, 1] = 4.0

        decoded = decode_coupled_policy_value(logits, vocab)

        expected = torch.tensor([[[[0, 2, 50, 1, 10, 0], [0, 1, 25, 0, 0, 1]]]], dtype=torch.float32)
        self.assertTrue(torch.equal(decoded, expected))

    def test_decode_hierarchical_policy_value_uses_checkpoint_vocab(self):
        from evaluate_v10_policy_bridge import decode_hierarchical_policy_value

        vocab = {
            "mode": "hierarchical_tokens",
            "groups": [[0], [1, 2], [3, 4], [5]],
            "count_values": [[0, 1, 2], [0, 1, 2], [0, 1, 0], [0, 1, 0]],
            "total_values": [[0, 1, 2], [0, 25, 50], [0, 10, 0], [0, 1, 0]],
            "count_sizes": [3, 3, 2, 2],
            "total_sizes": [3, 3, 2, 2],
            "max_count_tokens": 3,
            "max_total_tokens": 3,
            "action_dim": 6,
        }
        count_logits = torch.zeros((1, 1, 1, 4, 3), dtype=torch.float32)
        total_logits = torch.zeros((1, 1, 1, 4, 3), dtype=torch.float32)
        count_logits[0, 0, 0, 1, 2] = 5.0
        total_logits[0, 0, 0, 1, 2] = 5.0
        count_logits[0, 0, 0, 2, 1] = 5.0
        total_logits[0, 0, 0, 2, 1] = 5.0

        decoded = decode_hierarchical_policy_value(count_logits, total_logits, vocab)

        expected = torch.tensor([[[[0, 2, 50, 1, 10, 0]]]], dtype=torch.float32)
        self.assertTrue(torch.equal(decoded, expected))

    def test_limit_eval_indices_keeps_zero_as_unlimited(self):
        from evaluate_v10_policy_bridge import limit_eval_indices

        indices = np.arange(5, dtype=np.int64)

        np.testing.assert_array_equal(limit_eval_indices(indices, 0), indices)
        np.testing.assert_array_equal(limit_eval_indices(indices, -1), indices)
        np.testing.assert_array_equal(limit_eval_indices(indices, 3), np.array([0, 1, 2], dtype=np.int64))

    def test_active_aware_sample_limit_prefers_active_samples(self):
        from evaluate_v10_policy_bridge import limit_eval_indices_active_aware

        indices = np.arange(5, dtype=np.int64)
        edge_a_future = np.zeros((5, 1, 1, 1), dtype=np.float32)
        y_link_active = np.zeros((5, 1, 1), dtype=np.float32)
        edge_a_future[3, 0, 0, 0] = 1.0
        y_link_active[1, 0, 0] = 1.0

        selected = limit_eval_indices_active_aware(indices, 3, edge_a_future, y_link_active)

        np.testing.assert_array_equal(selected, np.array([1, 3, 0], dtype=np.int64))

    def test_pretrim_bridge_arrays_keeps_train_and_limited_eval_with_local_indices(self):
        from evaluate_v10_policy_bridge import load_bridge_arrays_and_splits

        with TemporaryDirectory() as tmp:
            dataset_dir = Path(tmp)
            sample_seed = np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)
            x_node = np.arange(6 * 2, dtype=np.float32).reshape(6, 2)
            static = np.array([10, 11], dtype=np.int32)
            np.savez(
                dataset_dir / "world_model_dataset_v0_samples.npz",
                sample_seed=sample_seed,
                x_node=x_node,
                static=static,
            )

            arrays, train_idx, val_idx, test_idx = load_bridge_arrays_and_splits(
                dataset_dir,
                {"train_seeds": [0], "val_seeds": [1], "test_seeds": [2]},
                max_val_samples=1,
                max_test_samples=1,
                pretrim_arrays=True,
            )

        self.assertEqual(train_idx.tolist(), [0, 1])
        self.assertEqual(val_idx.tolist(), [2])
        self.assertEqual(test_idx.tolist(), [3])
        self.assertEqual(arrays["sample_seed"].tolist(), [0, 0, 1, 2])
        np.testing.assert_allclose(arrays["x_node"], x_node[[0, 1, 2, 4]])
        np.testing.assert_array_equal(arrays["static"], static)


if __name__ == "__main__":
    unittest.main()
