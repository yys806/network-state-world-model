import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))


class V11TemplatePolicySelectorTest(unittest.TestCase):
    def test_template_advantage_weights_emphasize_high_reward_rows(self):
        from compare_v11_template_policy_selector import template_advantage_weights

        rewards = np.array([0.0, 1.0, 4.0], dtype=np.float32)
        weights = template_advantage_weights(rewards, baseline=1.0, temperature=1.0, max_weight=10.0)

        self.assertLess(float(weights[0]), float(weights[1]))
        self.assertLess(float(weights[1]), float(weights[2]))
        self.assertLessEqual(float(weights[2]), 10.0)

    def test_sample_step_feature_rows_keep_group_order(self):
        from compare_v11_template_policy_selector import make_sample_step_feature_rows

        actions = np.zeros((2, 3, 2, 6), dtype=np.float32)
        actions[0, 1, :, 2] = [1.0, 3.0]
        actions[1, 2, :, 2] = [5.0, 0.0]
        coords = np.array([[0, 1, 0], [0, 1, 1], [1, 2, 0]], dtype=np.int64)
        score = np.array([0.5, 1.5, 2.0], dtype=np.float32)
        value = np.array([2.0, 4.0, 6.0], dtype=np.float32)

        keys, features = make_sample_step_feature_rows(actions, coords, score, value)

        self.assertTrue(np.array_equal(keys, np.array([[0, 1], [1, 2]], dtype=np.int64)))
        self.assertEqual(features.shape[0], 2)
        self.assertAlmostEqual(float(features[0, 0]), 4.0, places=5)
        self.assertAlmostEqual(float(features[1, 0]), 5.0, places=5)
        self.assertGreater(float(features[0, 3]), 0.0)

    def test_apply_template_assignments_preserves_step0_and_support(self):
        from compare_v11_template_policy_selector import SchedulerTemplate, apply_template_assignments

        actions = np.zeros((1, 3, 3, 6), dtype=np.float32)
        actions[0, 0, :, 2] = [7.0, 8.0, 9.0]
        actions[0, 1, :, 2] = [2.0, 4.0, 0.0]
        coords = np.array([[0, 1, 0], [0, 1, 1]], dtype=np.int64)
        values = np.array([20.0, 20.0], dtype=np.float32)
        score = np.array([0.1, 0.9], dtype=np.float32)
        template = SchedulerTemplate(top_k=1, alpha=1.0, step_total_cap_scale=1.5, edge_value_cap_scale=2.0)

        repaired = apply_template_assignments(
            actions,
            coords,
            values,
            score,
            np.array([[0, 1]], dtype=np.int64),
            np.array([0], dtype=np.int64),
            [template],
        )

        self.assertTrue(np.allclose(repaired[0, 0, :, 2], actions[0, 0, :, 2]))
        self.assertGreater(float(repaired[0, 1, 0, 2]), 0.0)
        self.assertGreater(float(repaired[0, 1, 1, 2]), 4.0)
        self.assertAlmostEqual(float(repaired[0, 1, 2, 2]), 0.0, places=5)
        self.assertLessEqual(float(np.sum(repaired[0, 1, :, 2])), 9.0 + 1e-5)

    def test_sample_step_rewards_use_active_edges_when_available(self):
        from compare_v11_template_policy_selector import sample_step_rewards_from_predictions

        predictions = {
            'link_rate_true': np.array([[[[1.0], [100.0]], [[3.0], [4.0]]]], dtype=np.float32),
            'link_rate_pred': np.array([[[[2.0], [0.0]], [[3.0], [8.0]]]], dtype=np.float32),
            'link_activity_true': np.array([[[[1.0], [0.0]], [[0.0], [0.0]]]], dtype=np.float32),
        }
        group_keys = np.array([[0, 0], [0, 1]], dtype=np.int64)

        rewards = sample_step_rewards_from_predictions(predictions, group_keys, link_penalty_weight=0.0)

        self.assertAlmostEqual(float(rewards[0]), -1.0, places=5)
        self.assertLess(float(rewards[1]), -2.0)


if __name__ == '__main__':
    unittest.main()
