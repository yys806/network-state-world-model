import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


class V11RbTotalValueHeadTest(unittest.TestCase):
    def test_build_step_codebooks_uses_train_positive_values_only(self):
        from run_v11_rb_total_value_head import build_step_codebooks

        truth = np.zeros((2, 3, 3, 6), dtype=np.float32)
        truth[:, 1, :, 2] = np.array([[0.0, 10.0, 20.0], [30.0, 40.0, 50.0]], dtype=np.float32)
        truth[:, 2, :, 2] = np.array([[0.0, 100.0, 200.0], [300.0, 400.0, 500.0]], dtype=np.float32)

        codebooks = build_step_codebooks(truth, steps=(1, 2), bin_count=3)

        self.assertEqual(set(codebooks), {1, 2})
        self.assertTrue(np.all(np.diff(codebooks[1].values) >= 0.0))
        self.assertTrue(np.all(np.diff(codebooks[2].values) >= 0.0))
        self.assertGreaterEqual(float(codebooks[1].values[0]), 10.0)
        self.assertLessEqual(float(codebooks[1].values[-1]), 50.0)
        self.assertGreaterEqual(float(codebooks[2].values[0]), 100.0)
        self.assertLessEqual(float(codebooks[2].values[-1]), 500.0)

    def test_make_rb_total_examples_focuses_on_predicted_active_target_steps(self):
        from run_v11_rb_total_value_head import build_step_codebooks, make_rb_total_examples

        baseline = np.zeros((1, 3, 3, 6), dtype=np.float32)
        truth = np.zeros_like(baseline)
        baseline[0, 0, :, 2] = [9.0, 9.0, 9.0]
        baseline[0, 1, :, 2] = [0.0, 11.0, 21.0]
        baseline[0, 2, :, 2] = [31.0, 0.0, 41.0]
        baseline[..., 1] = 2.0
        baseline[..., 4] = 3.0
        truth[0, 1, :, 2] = [0.0, 10.0, 20.0]
        truth[0, 2, :, 2] = [30.0, 0.0, 40.0]
        codebooks = build_step_codebooks(truth, steps=(1, 2), bin_count=2)

        examples = make_rb_total_examples(baseline, truth, codebooks, steps=(1, 2))

        self.assertEqual(examples.features.shape[0], 4)
        self.assertEqual(examples.coordinates.tolist(), [[0, 1, 1], [0, 1, 2], [0, 2, 0], [0, 2, 2]])
        self.assertEqual(examples.labels.shape, (4,))
        self.assertTrue(np.all(examples.true_values > 0.0))
        self.assertTrue(np.all(examples.baseline_values > 0.0))

    def test_apply_value_head_repair_preserves_masks_dims_and_step0(self):
        from run_v11_rb_total_value_head import apply_value_head_repair

        actions = np.ones((1, 3, 3, 6), dtype=np.float32)
        actions[..., 2] = np.array([[[7.0, 8.0, 9.0], [0.0, 10.0, 20.0], [30.0, 0.0, 40.0]]], dtype=np.float32)
        coordinates = np.array([[0, 1, 1], [0, 1, 2], [0, 2, 0], [0, 2, 1], [0, 2, 2]], dtype=np.int64)
        values = np.array([100.0, 200.0, 300.0, 999.0, 400.0], dtype=np.float32)
        confidence = np.array([0.9, 0.1, 0.8, 0.99, 0.7], dtype=np.float32)

        repaired = apply_value_head_repair(actions, coordinates, values, confidence, min_confidence=0.5)

        self.assertTrue(np.allclose(repaired[:, 0, :, 2], actions[:, 0, :, 2]))
        self.assertEqual(float(repaired[0, 1, 1, 2]), 100.0)
        self.assertEqual(float(repaired[0, 1, 2, 2]), 20.0)
        self.assertEqual(float(repaired[0, 2, 0, 2]), 300.0)
        self.assertEqual(float(repaired[0, 2, 1, 2]), 0.0)
        self.assertEqual(float(repaired[0, 2, 2, 2]), 400.0)
        for dim in (0, 1, 3, 4, 5):
            self.assertTrue(np.allclose(repaired[..., dim], actions[..., dim]))

    def test_codebook_decode_supports_argmax_expected_and_conservative(self):
        from run_v11_rb_total_value_head import StepCodebook, decode_probabilities

        codebook = StepCodebook(step=1, values=np.array([10.0, 20.0, 40.0], dtype=np.float32), edges=np.array([15.0, 30.0], dtype=np.float32))
        probabilities = np.array([[0.1, 0.7, 0.2], [0.2, 0.2, 0.6]], dtype=np.float32)

        argmax_values, argmax_conf = decode_probabilities(probabilities, codebook, decoder="argmax")
        expected_values, expected_conf = decode_probabilities(probabilities, codebook, decoder="expected")
        conservative_values, conservative_conf = decode_probabilities(probabilities, codebook, decoder="conservative")

        self.assertTrue(np.allclose(argmax_values, [20.0, 40.0]))
        self.assertTrue(np.allclose(argmax_conf, [0.7, 0.6]))
        self.assertTrue(np.allclose(expected_values, [23.0, 30.0]))
        self.assertTrue(np.allclose(expected_conf, [0.7, 0.6]))
        self.assertTrue(np.allclose(conservative_values, [20.0, 20.0]))
        self.assertTrue(np.allclose(conservative_conf, [0.7, 0.6]))

    def test_limit_indices_keeps_prefix_when_limit_is_set(self):
        from run_v11_rb_total_value_head import limit_indices

        indices = np.arange(10, dtype=np.int64)

        self.assertTrue(np.array_equal(limit_indices(indices, None), indices))
        self.assertTrue(np.array_equal(limit_indices(indices, 4), np.array([0, 1, 2, 3], dtype=np.int64)))
        self.assertTrue(np.array_equal(limit_indices(indices, 20), indices))

    def test_append_state_features_matches_coordinates(self):
        from run_v11_rb_total_value_head import append_state_features

        base = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        state = np.arange(2 * 3 * 5, dtype=np.float32).reshape(2, 3, 5)
        coordinates = np.array([[0, 1, 2], [1, 2, 0]], dtype=np.int64)

        combined = append_state_features(base, state, coordinates)

        self.assertEqual(combined.shape, (2, 7))
        self.assertTrue(np.allclose(combined[:, :2], base))
        self.assertTrue(np.allclose(combined[0, 2:], state[0, 2]))
        self.assertTrue(np.allclose(combined[1, 2:], state[1, 0]))

    def test_make_teacher_critical_labels_requires_truth_and_positive_improvement(self):
        from run_v11_rb_total_value_head import make_teacher_critical_labels

        coordinates = np.array(
            [
                [0, 1, 0],
                [0, 1, 1],
                [0, 2, 0],
                [1, 1, 0],
            ],
            dtype=np.int64,
        )
        truth = np.zeros((2, 3, 2, 6), dtype=np.float32)
        truth[0, 1, 0, 2] = 50.0
        truth[0, 1, 1, 2] = 50.0
        truth[0, 2, 0, 2] = 50.0
        truth[1, 1, 0, 2] = 0.0
        improvement = np.zeros((2, 3), dtype=np.float32)
        improvement[0, 1] = 100.0
        improvement[0, 2] = -1.0
        improvement[1, 1] = 100.0

        labels = make_teacher_critical_labels(
            coordinates,
            truth,
            improvement,
            min_effective_value=1.0,
            min_improvement=0.0,
        )

        self.assertTrue(np.array_equal(labels, np.array([1, 1, 0, 0], dtype=np.int64)))

    def test_streaming_fit_stats_matches_numpy_fit_stats(self):
        from pi_jwm.v6_data import fit_stats
        from run_v11_rb_total_value_head import streaming_fit_stats

        values = np.arange(4 * 3 * 2, dtype=np.float32).reshape(4, 3, 2)

        expected_mean, expected_std = fit_stats(values)
        mean, std = streaming_fit_stats((values[:2], values[2:]))

        self.assertTrue(np.allclose(mean, expected_mean, atol=1e-6))
        self.assertTrue(np.allclose(std, expected_std, atol=1e-6))

    def test_select_context_indices_can_limit_after_stats(self):
        from run_v11_rb_total_value_head import select_context_indices

        train = np.arange(10, dtype=np.int64)
        val = np.arange(20, 30, dtype=np.int64)
        test = np.arange(40, 50, dtype=np.int64)

        stats_train, splits = select_context_indices(
            train,
            val,
            test,
            max_train_samples=3,
            max_val_samples=4,
            max_test_samples=5,
            limit_after_stats=True,
        )

        self.assertTrue(np.array_equal(stats_train, train))
        self.assertTrue(np.array_equal(splits["train"], np.array([0, 1, 2], dtype=np.int64)))
        self.assertTrue(np.array_equal(splits["val"], np.array([20, 21, 22, 23], dtype=np.int64)))
        self.assertTrue(np.array_equal(splits["test"], np.array([40, 41, 42, 43, 44], dtype=np.int64)))

    def test_first_order_edge_improvement_uses_negative_gradient_delta(self):
        from run_v11_rb_total_value_head import first_order_edge_improvement

        gradient = np.array([[[[0.0, 0.0, 2.0], [0.0, 0.0, -3.0]]]], dtype=np.float32)
        baseline = np.array([[[[0.0, 0.0, 5.0], [0.0, 0.0, 5.0]]]], dtype=np.float32)
        truth = np.array([[[[0.0, 0.0, 3.0], [0.0, 0.0, 7.0]]]], dtype=np.float32)

        improvement = first_order_edge_improvement(gradient, baseline, truth, rb_dim=2)

        self.assertTrue(np.allclose(improvement, np.array([[[4.0, 6.0]]], dtype=np.float32)))

    def test_make_edge_teacher_labels_is_coordinate_specific(self):
        from run_v11_rb_total_value_head import make_edge_teacher_labels

        coordinates = np.array([[0, 1, 0], [0, 1, 1], [0, 2, 0]], dtype=np.int64)
        truth = np.zeros((1, 3, 2, 6), dtype=np.float32)
        truth[0, 1, 0, 2] = 50.0
        truth[0, 1, 1, 2] = 50.0
        truth[0, 2, 0, 2] = 50.0
        improvement = np.zeros((1, 3, 2), dtype=np.float32)
        improvement[0, 1, 0] = 10.0
        improvement[0, 1, 1] = -1.0
        improvement[0, 2, 0] = 0.5

        labels = make_edge_teacher_labels(
            coordinates,
            truth,
            improvement,
            min_effective_value=1.0,
            min_improvement=1.0,
        )

        self.assertTrue(np.array_equal(labels, np.array([1, 0, 0], dtype=np.int64)))

    def test_make_balanced_indices_oversamples_positive_class(self):
        from run_v11_rb_total_value_head import make_balanced_indices

        labels = np.array([0, 0, 0, 0, 1], dtype=np.int64)
        indices = make_balanced_indices(labels, positive_multiplier=3, seed=1)

        self.assertEqual(int(np.sum(labels[indices] == 1)), 3)
        self.assertEqual(int(np.sum(labels[indices] == 0)), 4)

    def test_binary_focal_loss_downweights_easy_examples(self):
        import torch

        from run_v11_rb_total_value_head import binary_focal_cross_entropy

        logits = torch.tensor([[4.0, -4.0], [-4.0, 4.0]], dtype=torch.float32)
        labels = torch.tensor([0, 1], dtype=torch.long)
        focal = binary_focal_cross_entropy(logits, labels, gamma=2.0)
        ce = torch.nn.functional.cross_entropy(logits, labels)

        self.assertLess(float(focal), float(ce))

    def test_retrieval_repair_predicts_nearest_supported_value_with_distance_confidence(self):
        from run_v11_rb_total_value_head import fit_retrieval_prototypes, predict_retrieval_values

        features = np.array([[0.0, 0.0], [10.0, 10.0]], dtype=np.float32)
        labels = np.array([1, 1], dtype=np.int64)
        values = np.array([25.0, 50.0], dtype=np.float32)
        model = fit_retrieval_prototypes(features, labels, values, max_prototypes=10)

        query = np.array([[0.1, 0.1], [9.0, 9.0], [100.0, 100.0]], dtype=np.float32)
        pred, confidence, distance = predict_retrieval_values(query, model)

        self.assertTrue(np.allclose(pred, [25.0, 50.0, 50.0]))
        self.assertGreater(float(confidence[0]), float(confidence[2]))
        self.assertLess(float(distance[0]), float(distance[2]))

    def test_apply_topk_score_repair_uses_highest_scores_only(self):
        from run_v11_rb_total_value_head import apply_topk_score_repair

        actions = np.ones((1, 3, 3, 6), dtype=np.float32)
        actions[..., 2] = np.array([[[1.0, 2.0, 3.0], [10.0, 20.0, 30.0], [40.0, 50.0, 60.0]]], dtype=np.float32)
        coordinates = np.array([[0, 1, 0], [0, 1, 1], [0, 2, 0], [0, 2, 2]], dtype=np.int64)
        values = np.array([100.0, 200.0, 300.0, 400.0], dtype=np.float32)
        scores = np.array([0.1, 0.9, 0.8, 0.2], dtype=np.float32)

        repaired = apply_topk_score_repair(actions, coordinates, values, scores, top_k=2)

        self.assertEqual(float(repaired[0, 1, 0, 2]), 10.0)
        self.assertEqual(float(repaired[0, 1, 1, 2]), 200.0)
        self.assertEqual(float(repaired[0, 2, 0, 2]), 300.0)
        self.assertEqual(float(repaired[0, 2, 2, 2]), 60.0)

    def test_pairwise_ranker_scores_high_targets_above_low_targets(self):
        from run_v11_rb_total_value_head import predict_scores, train_pairwise_ranker

        features = np.array(
            [
                [-2.0],
                [-1.0],
                [1.0],
                [2.0],
            ],
            dtype=np.float32,
        )
        targets = np.array([0.0, 0.1, 5.0, 6.0], dtype=np.float32)

        model = train_pairwise_ranker(
            features,
            targets,
            hidden_dim=16,
            epochs=120,
            lr=0.01,
            batch_size=8,
            seed=3,
            pairs_per_epoch=16,
            min_target_gap=0.5,
        )
        scores = predict_scores(model, features)

        self.assertGreater(float(np.min(scores[2:])), float(np.max(scores[:2])))


if __name__ == "__main__":
    unittest.main()
