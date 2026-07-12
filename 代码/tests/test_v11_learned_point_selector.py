import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


class V11LearnedPointSelectorTest(unittest.TestCase):
    def test_make_step_feature_matrix_uses_action_summaries(self):
        from compare_v11_learned_point_selector import make_step_feature_matrix

        actions = np.zeros((2, 1, 2, 3, 6), dtype=np.float32)
        actions[0, 0, 0, :, 2] = [1.0, 2.0, 0.0]
        actions[1, 0, 0, :, 2] = [0.0, 5.0, 0.0]
        actions[1, 0, 1, :, 4] = [3.0, 0.0, 0.0]

        features = make_step_feature_matrix(actions)

        self.assertEqual(features.shape[0], 2)
        self.assertEqual(features.shape[1], 1 + 2 * 3)
        self.assertAlmostEqual(float(features[0, 1]), 2.0)
        self.assertAlmostEqual(float(features[0, 2]), 3.0)
        self.assertAlmostEqual(float(features[0, 4]), 1.0)
        self.assertAlmostEqual(float(features[1, 6]), 3.0)

    def test_mix_actions_by_step_labels_selects_per_step_point(self):
        from compare_v11_learned_point_selector import mix_actions_by_step_labels

        actions = np.zeros((2, 1, 2, 1, 6), dtype=np.float32)
        actions[0, :, :, :, 2] = 1.0
        actions[1, :, :, :, 2] = 3.0
        labels = np.asarray([[0, 1]], dtype=np.int64)

        mixed = mix_actions_by_step_labels(actions, labels)

        self.assertAlmostEqual(float(mixed[0, 0, 0, 2]), 1.0)
        self.assertAlmostEqual(float(mixed[0, 1, 0, 2]), 3.0)

    def test_row_label_accepts_candidate_or_name(self):
        from compare_v11_learned_point_selector import row_label

        self.assertEqual(row_label({"candidate": "learned"}), "learned")
        self.assertEqual(row_label({"name": "point"}), "point")

    def test_limit_indices_keeps_prefix_without_mutating_full_array(self):
        from compare_v11_learned_point_selector import limit_indices

        full = np.arange(6)

        limited = limit_indices(full, 3)

        self.assertEqual(limited.tolist(), [0, 1, 2])
        self.assertEqual(full.tolist(), [0, 1, 2, 3, 4, 5])


if __name__ == "__main__":
    unittest.main()
