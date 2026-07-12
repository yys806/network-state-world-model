import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


class V11SchedulerPatternGenerationTest(unittest.TestCase):
    def test_make_step_keys_enumerates_requested_future_steps(self):
        from diagnose_v11_scheduler_pattern_generation import make_step_keys

        actions = np.zeros((2, 4, 3, 6), dtype=np.float32)

        keys = make_step_keys(actions, steps=(1, 3))

        self.assertTrue(
            np.array_equal(
                keys,
                np.array([[0, 1], [0, 3], [1, 1], [1, 3]], dtype=np.int64),
            )
        )

    def test_retrieve_nearest_step_patterns_returns_matching_action_matrix(self):
        from diagnose_v11_scheduler_pattern_generation import retrieve_nearest_step_patterns

        prototype_features = np.array([[0.0, 0.0], [10.0, 10.0]], dtype=np.float32)
        prototype_actions = np.zeros((2, 3, 6), dtype=np.float32)
        prototype_actions[1, :, 2] = 7.0
        query_features = np.array([[9.0, 9.0]], dtype=np.float32)

        retrieved, nearest = retrieve_nearest_step_patterns(query_features, prototype_features, prototype_actions)

        self.assertEqual(nearest.tolist(), [1])
        self.assertTrue(np.allclose(retrieved[0, :, 2], 7.0))

    def test_apply_step_pattern_replacement_preserves_step0_and_can_limit_dims(self):
        from diagnose_v11_scheduler_pattern_generation import apply_step_pattern_replacement

        actions = np.zeros((1, 3, 2, 6), dtype=np.float32)
        actions[0, 0, :, 2] = 3.0
        actions[0, 1, :, 2] = 4.0
        replacements = np.ones((1, 2, 6), dtype=np.float32) * 9.0
        keys = np.array([[0, 1]], dtype=np.int64)

        replaced = apply_step_pattern_replacement(actions, keys, replacements, mode="rb_cpu")

        self.assertTrue(np.allclose(replaced[0, 0, :, 2], 3.0))
        self.assertTrue(np.allclose(replaced[0, 1, :, 2], 9.0))
        self.assertTrue(np.allclose(replaced[0, 1, :, 4], 9.0))
        self.assertTrue(np.allclose(replaced[0, 1, :, 0], 0.0))

    def test_apply_step_group_total_scaling_preserves_support_and_matches_totals(self):
        from diagnose_v11_scheduler_pattern_generation import apply_step_group_total_scaling

        actions = np.zeros((1, 3, 2, 6), dtype=np.float32)
        actions[0, 0, :, 2] = 3.0
        actions[0, 1, 0, 2] = 2.0
        actions[0, 1, 1, 2] = 6.0
        actions[0, 1, 0, 4] = 1.0
        targets = np.array([[16.0, 4.0]], dtype=np.float32)
        keys = np.array([[0, 1]], dtype=np.int64)

        scaled = apply_step_group_total_scaling(actions, keys, targets, mode="rb_cpu")

        self.assertTrue(np.allclose(scaled[0, 0, :, 2], 3.0))
        self.assertAlmostEqual(float(np.sum(scaled[0, 1, :, 2])), 16.0, places=5)
        self.assertAlmostEqual(float(np.sum(scaled[0, 1, :, 4])), 4.0, places=5)
        self.assertEqual(bool(scaled[0, 1, 1, 4] > 0.0), False)

    def test_fit_group_total_model_predicts_two_totals(self):
        from diagnose_v11_scheduler_pattern_generation import fit_group_total_model, predict_group_totals

        features = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
        targets = np.array([[0.0, 0.0], [2.0, 1.0], [4.0, 2.0], [6.0, 3.0]], dtype=np.float32)

        model = fit_group_total_model("rf", features, targets, seed=7, rf_trees=20)
        predictions = predict_group_totals(model, features)

        self.assertEqual(predictions.shape, targets.shape)
        self.assertTrue(np.all(predictions >= 0.0))


if __name__ == "__main__":
    unittest.main()
