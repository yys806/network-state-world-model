import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


class V11PolicyActionProxyTest(unittest.TestCase):
    def test_step_total_rmse_uses_selected_action_dimensions(self):
        from analyze_v11_policy_action_proxy import step_total_rmse

        pred = np.zeros((2, 2, 3, 6), dtype=np.float32)
        true = np.zeros_like(pred)
        pred[:, :, :, 2] = 1.0
        true[:, :, :, 2] = 2.0

        rmse = step_total_rmse(pred, true, dim=2)

        self.assertAlmostEqual(rmse, 3.0, places=6)

    def test_action_proxy_metrics_reports_activity_and_totals(self):
        from analyze_v11_policy_action_proxy import compute_action_proxy_metrics

        true = np.zeros((1, 1, 4, 6), dtype=np.float32)
        true[0, 0, [0, 1], 2] = [10.0, 20.0]
        true[0, 0, [0, 1], 4] = [2.0, 4.0]
        pred = np.zeros_like(true)
        pred[0, 0, [0, 2], 2] = [10.0, 5.0]
        pred[0, 0, [0, 2], 4] = [2.0, 1.0]

        metrics = compute_action_proxy_metrics(pred, true)

        self.assertAlmostEqual(metrics["activity_f1"], 0.5, places=6)
        self.assertAlmostEqual(metrics["rb_total_sum_ratio"], 0.5, places=6)
        self.assertAlmostEqual(metrics["cpu_total_sum_ratio"], 0.5, places=6)
        self.assertGreater(metrics["rb_total_step_rmse"], 0.0)

    def test_project_policy_value_to_codebook_np_uses_nearest_bin(self):
        from analyze_v11_policy_action_proxy import project_policy_value_to_codebook_np

        policy_value = np.array([[[[1.2, 7.7], [4.9, 11.0]]]], dtype=np.float32)
        codebook = np.array([[[1.0, 5.0], [8.0, 10.0]]], dtype=np.float32)

        projected = project_policy_value_to_codebook_np(policy_value, codebook)

        np.testing.assert_allclose(projected, np.array([[[[1.0, 8.0], [5.0, 10.0]]]], dtype=np.float32))

    def test_decode_values_supports_codebook_quantile(self):
        from analyze_v11_policy_action_proxy import decode_values
        from evaluate_v10_policy_bridge import ActionValueDecoderConfig

        policy_value = np.array([[[[1.2, 7.7]]]], dtype=np.float32)
        codebook = np.array([[[1.0, 5.0], [8.0, 10.0]]], dtype=np.float32)
        config = ActionValueDecoderConfig(name="train_codebook_quantile", prototype=codebook, value_scale=0.5)

        decoded = decode_values(policy_value, config)

        np.testing.assert_allclose(decoded, np.array([[[[0.5, 4.0]]]], dtype=np.float32))

    def test_decode_activity_np_expands_edge_topk_to_action_dims(self):
        from analyze_v11_policy_action_proxy import decode_activity_np
        from evaluate_v10_policy_bridge import ActionDecoderConfig

        prob = np.array([[[[0.1, 0.2], [0.8, 0.1], [0.3, 0.4]]]], dtype=np.float32)
        config = ActionDecoderConfig(name="edge_val_quantile_topk", count_budget=np.array([1], dtype=np.int64))

        active = decode_activity_np(prob, np.zeros_like(prob), config, threshold=0.5)

        expected = np.array([[[[False, False], [True, True], [False, False]]]], dtype=bool)
        np.testing.assert_array_equal(active, expected)

    def test_decode_activity_np_edge_threshold_topk_filters_low_confidence_edges(self):
        from analyze_v11_policy_action_proxy import decode_activity_np
        from evaluate_v10_policy_bridge import ActionDecoderConfig

        prob = np.array([[[[0.6, 0.2], [0.4, 0.1], [0.9, 0.1]]]], dtype=np.float32)
        config = ActionDecoderConfig(name="edge_threshold_topk", count_budget=np.array([2], dtype=np.int64))

        active = decode_activity_np(prob, np.zeros_like(prob), config, threshold=0.5)

        expected = np.array([[[[True, True], [False, False], [True, True]]]], dtype=bool)
        np.testing.assert_array_equal(active, expected)

    def test_apply_group_value_scales_affects_rb_and_cpu_groups(self):
        from analyze_v11_policy_action_proxy import apply_group_value_scales

        values = np.ones((1, 1, 1, 6), dtype=np.float32)

        scaled = apply_group_value_scales(values, rb_dim_scale=0.5, cpu_dim_scale=2.0)

        np.testing.assert_allclose(scaled[0, 0, 0], np.array([1.0, 0.5, 0.5, 2.0, 2.0, 1.0], dtype=np.float32))

    def test_average_policy_predictions_averages_scores_and_keeps_truth(self):
        from analyze_v11_policy_action_proxy import average_policy_predictions

        first = {
            "prob": np.ones((1, 1, 2, 1), dtype=np.float32),
            "edge_prob": np.array([[[0.2, 0.8]]], dtype=np.float32),
            "value_pred": np.ones((1, 1, 2, 1), dtype=np.float32),
            "value_true": np.array([[[[3.0], [4.0]]]], dtype=np.float32),
        }
        second = {
            "prob": np.zeros((1, 1, 2, 1), dtype=np.float32),
            "edge_prob": np.array([[[0.6, 0.4]]], dtype=np.float32),
            "value_pred": np.full((1, 1, 2, 1), 3.0, dtype=np.float32),
            "value_true": first["value_true"].copy(),
        }

        averaged = average_policy_predictions([first, second])

        np.testing.assert_allclose(averaged["prob"], 0.5)
        np.testing.assert_allclose(averaged["edge_prob"], np.array([[[0.4, 0.6]]], dtype=np.float32))
        np.testing.assert_allclose(averaged["value_pred"], 2.0)
        np.testing.assert_array_equal(averaged["value_true"], first["value_true"])

    def test_build_edge_reranker_features_uses_edge_and_policy_scores(self):
        from analyze_v11_policy_action_proxy import build_edge_reranker_features

        predictions = {
            "prob": np.array([[[[0.1, 0.9], [0.4, 0.6]]]], dtype=np.float32),
            "edge_prob": np.array([[[0.8, 0.2]]], dtype=np.float32),
            "value_pred": np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32),
        }

        features = build_edge_reranker_features(predictions)

        self.assertEqual(features.shape[0], 2)
        self.assertGreater(features.shape[1], 6)
        self.assertAlmostEqual(float(features[0, 0]), 0.8, places=6)
        self.assertAlmostEqual(float(features[1, 0]), 0.2, places=6)

    def test_choose_threshold_by_f1_selects_best_candidate(self):
        from analyze_v11_policy_action_proxy import choose_threshold_by_f1

        scores = np.array([0.9, 0.8, 0.2, 0.1], dtype=np.float32)
        true = np.array([True, False, True, False])

        threshold = choose_threshold_by_f1(scores, true, candidates=np.array([0.15, 0.5, 0.85], dtype=np.float32))

        self.assertAlmostEqual(threshold, 0.15, places=6)

    def test_build_edge_count_features_returns_one_row_per_sample_step(self):
        from analyze_v11_policy_action_proxy import build_edge_count_features

        predictions = {
            "prob": np.array(
                [
                    [[[0.9], [0.2], [0.1]], [[0.4], [0.3], [0.2]]],
                    [[[0.1], [0.2], [0.3]], [[0.9], [0.8], [0.1]]],
                ],
                dtype=np.float32,
            ),
            "edge_prob": np.array(
                [
                    [[0.9, 0.2, 0.1], [0.4, 0.3, 0.2]],
                    [[0.1, 0.2, 0.3], [0.9, 0.8, 0.1]],
                ],
                dtype=np.float32,
            ),
        }

        features = build_edge_count_features(predictions)

        self.assertEqual(features.shape[0], 4)
        self.assertGreaterEqual(features.shape[1], 10)
        self.assertGreater(float(features[0, 0]), float(features[1, 0]))

    def test_edge_count_regressor_predicts_clipped_sample_step_counts(self):
        from analyze_v11_policy_action_proxy import fit_edge_count_regressor, predict_edge_count_regressor

        edge_prob = np.array(
            [
                [[0.95, 0.90, 0.05]],
                [[0.80, 0.75, 0.10]],
                [[0.10, 0.05, 0.01]],
                [[0.20, 0.10, 0.05]],
            ],
            dtype=np.float32,
        )
        predictions = {
            "prob": edge_prob[:, :, :, None],
            "edge_prob": edge_prob,
            "value_true": np.zeros((4, 1, 3, 1), dtype=np.float32),
        }
        predictions["value_true"][0, 0, [0, 1], 0] = 1.0
        predictions["value_true"][1, 0, [0, 1], 0] = 1.0

        model = fit_edge_count_regressor(predictions, seed=7)
        counts = predict_edge_count_regressor(model, predictions)

        self.assertEqual(counts.shape, (4, 1))
        self.assertTrue(np.all((0 <= counts) & (counts <= 3)))
        self.assertGreaterEqual(int(counts[0, 0]), int(counts[2, 0]))


if __name__ == "__main__":
    unittest.main()
