import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


class V11RbTotalValueReconstructionTest(unittest.TestCase):
    def test_value_target_roundtrip_for_residual_log_ratio_and_per_task(self):
        from diagnose_v11_rb_total_value_reconstruction import invert_value_target, make_value_target

        baseline = np.array([10.0, 20.0, 0.0], dtype=np.float32)
        true = np.array([15.0, 10.0, 5.0], dtype=np.float32)
        rb_count = np.array([3.0, 2.0, 0.0], dtype=np.float32)

        for mode in ("abs", "residual", "log_ratio", "per_task", "per_task_residual"):
            target = make_value_target(mode, true, baseline, rb_count)
            decoded = invert_value_target(mode, target, baseline, rb_count)
            self.assertTrue(np.allclose(decoded, true, atol=1e-5), mode)

    def test_count_total_constraint_preserves_inactive_and_clips_per_task(self):
        from diagnose_v11_rb_total_value_reconstruction import apply_count_total_constraint

        values = np.array([100.0, 100.0, 100.0, -5.0], dtype=np.float32)
        baseline = np.array([10.0, 20.0, 0.0, 30.0], dtype=np.float32)
        rb_count = np.array([2.0, 0.0, 4.0, 3.0], dtype=np.float32)

        constrained = apply_count_total_constraint(values, baseline, rb_count, max_rb_per_task=20.0)

        self.assertTrue(np.allclose(constrained, np.array([40.0, 20.0, 0.0, 0.0], dtype=np.float32)))

    def test_audit_coordinate_values_detects_mismatch(self):
        from diagnose_v11_rb_total_value_reconstruction import audit_coordinate_values

        truth = np.zeros((1, 3, 2, 6), dtype=np.float32)
        truth[0, 1, 0, 2] = 7.0
        truth[0, 2, 1, 2] = 9.0
        coordinates = np.array([[0, 1, 0], [0, 2, 1]], dtype=np.int64)
        values = np.array([7.0, 8.0], dtype=np.float32)

        audit = audit_coordinate_values(coordinates, values, truth)

        self.assertEqual(audit["row_count"], 2)
        self.assertEqual(audit["mismatch_count"], 1)
        self.assertAlmostEqual(audit["max_abs_error"], 1.0)



    def test_neural_value_decoder_learns_weighted_residual_shape(self):
        from diagnose_v11_rb_total_value_reconstruction import fit_neural_value_decoder, predict_neural_value_decoder

        features = np.array([[-2.0], [-1.0], [1.0], [2.0]], dtype=np.float32)
        targets = np.array([-2.0, -1.0, 1.0, 2.0], dtype=np.float32)
        weights = np.array([1.0, 1.0, 4.0, 4.0], dtype=np.float32)

        model = fit_neural_value_decoder(
            features,
            targets,
            sample_weight=weights,
            hidden_dim=16,
            epochs=180,
            lr=0.01,
            batch_size=4,
            seed=5,
        )
        predictions = predict_neural_value_decoder(model, features)

        self.assertEqual(predictions.shape, targets.shape)
        self.assertGreater(float(np.min(predictions[2:])), float(np.max(predictions[:2])))

    def test_value_fit_mask_can_include_all_predicted_active_rows(self):
        from diagnose_v11_rb_total_value_reconstruction import make_value_fit_mask

        true_value = np.array([0.0, 0.5, 2.0, 10.0], dtype=np.float32)

        positive_only = make_value_fit_mask(true_value, min_effective_value=1.0, fit_mode="positive_only")
        all_rows = make_value_fit_mask(true_value, min_effective_value=1.0, fit_mode="all")

        self.assertTrue(np.array_equal(positive_only, np.array([False, False, True, True])))
        self.assertTrue(np.array_equal(all_rows, np.array([True, True, True, True])))

    def test_prepare_value_training_data_uses_fit_mode_for_targets(self):
        from diagnose_v11_rb_total_value_reconstruction import prepare_value_training_data

        features = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
        true_value = np.array([0.0, 5.0, 10.0], dtype=np.float32)
        baseline = np.array([4.0, 4.0, 4.0], dtype=np.float32)
        rb_count = np.array([2.0, 2.0, 2.0], dtype=np.float32)

        fit_features, target, weight, mask = prepare_value_training_data(
            features,
            true_value,
            baseline,
            rb_count,
            target_mode="per_task_residual",
            min_effective_value=1.0,
            fit_mode="all",
            weight_power=0.5,
        )

        self.assertTrue(np.array_equal(mask, np.array([True, True, True])))
        self.assertEqual(fit_features.shape, (3, 1))
        self.assertTrue(np.allclose(target, np.array([-2.0, 0.5, 3.0], dtype=np.float32)))
        self.assertEqual(weight.shape, (3,))

    def test_support_sample_weight_balances_rare_positive_rows(self):
        from diagnose_v11_rb_total_value_reconstruction import make_support_sample_weight

        labels = np.array([0, 0, 0, 1], dtype=np.float32)
        weight = make_support_sample_weight(labels)

        self.assertEqual(weight.shape, labels.shape)
        self.assertGreater(float(weight[-1]), float(weight[0]))
        self.assertAlmostEqual(float(np.mean(weight)), 1.0, places=5)

    def test_select_support_gated_topk_indices_applies_probability_floor(self):
        from diagnose_v11_rb_total_value_reconstruction import select_support_gated_topk_indices

        coordinates = np.array(
            [
                [0, 1, 0],
                [0, 1, 1],
                [0, 1, 2],
                [1, 1, 0],
            ],
            dtype=np.int64,
        )
        scores = np.array([10.0, 9.0, 8.0, 7.0], dtype=np.float32)
        support = np.array([0.1, 0.9, 0.8, 0.2], dtype=np.float32)

        selected = select_support_gated_topk_indices(
            coordinates,
            scores,
            support,
            top_k=2,
            scope="per_sample",
            min_support_probability=0.5,
        )

        self.assertTrue(np.array_equal(selected, np.array([1, 2], dtype=np.int64)))

    def test_fit_support_classifier_predicts_probabilities(self):
        from diagnose_v11_rb_total_value_reconstruction import fit_support_classifier, predict_support_probability

        features = np.array([[-2.0], [-1.0], [1.0], [2.0]], dtype=np.float32)
        labels = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)

        model = fit_support_classifier("rf", features, labels, seed=3, rf_trees=20)
        probability = predict_support_probability(model, features)

        self.assertEqual(probability.shape, labels.shape)
        self.assertTrue(np.all((probability >= 0.0) & (probability <= 1.0)))
        self.assertGreater(float(np.mean(probability[labels > 0.5])), float(np.mean(probability[labels < 0.5])))

    def test_make_step_support_labels_broadcasts_to_candidate_rows(self):
        from diagnose_v11_rb_total_value_reconstruction import make_step_support_labels

        coordinates = np.array(
            [
                [0, 1, 0],
                [0, 1, 2],
                [0, 2, 1],
                [1, 1, 0],
            ],
            dtype=np.int64,
        )
        true_value = np.array([0.0, 5.0, 0.0, 0.0], dtype=np.float32)

        keys, labels, row_labels = make_step_support_labels(coordinates, true_value, min_effective_value=1.0)

        self.assertTrue(np.array_equal(keys, np.array([[0, 1], [0, 2], [1, 1]], dtype=np.int64)))
        self.assertTrue(np.array_equal(labels, np.array([1.0, 0.0, 0.0], dtype=np.float32)))
        self.assertTrue(np.array_equal(row_labels, np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)))

    def test_make_step_scheduler_features_uses_step_action_totals(self):
        from diagnose_v11_rb_total_value_reconstruction import make_step_scheduler_features

        actions = np.zeros((2, 3, 2, 6), dtype=np.float32)
        actions[0, 1, 0, 2] = 10.0
        actions[0, 1, 1, 4] = 3.0
        actions[1, 2, 0, 2] = 5.0
        keys = np.array([[0, 1], [1, 2]], dtype=np.int64)

        features = make_step_scheduler_features(actions, keys)

        self.assertEqual(features.shape[0], 2)
        self.assertGreater(float(features[0, 1]), float(features[1, 1]))
        self.assertGreater(float(features[0, 2]), 0.0)
if __name__ == "__main__":
    unittest.main()
