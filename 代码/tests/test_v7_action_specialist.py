import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


class V7ActionSpecialistTest(unittest.TestCase):
    def test_downsample_keeps_all_positive_rows_and_caps_negatives(self):
        from run_v7_action_specialist import make_balanced_training_indices

        active = np.array([True, False, False, True, False, False, False, True])
        selected = make_balanced_training_indices(active, max_neg_per_pos=1, seed=7)

        self.assertTrue(set(np.where(active)[0]).issubset(set(selected.tolist())))
        self.assertEqual(int((~active[selected]).sum()), 3)
        self.assertEqual(len(np.unique(selected)), len(selected))

    def test_binary_metrics_handles_empty_positive_predictions(self):
        from run_v7_action_specialist import binary_metrics

        metrics = binary_metrics(
            pred=np.array([False, False, False]),
            true=np.array([True, False, True]),
        )

        self.assertEqual(metrics["tp"], 0)
        self.assertEqual(metrics["fn"], 2)
        self.assertEqual(metrics["precision"], 0.0)
        self.assertEqual(metrics["recall"], 0.0)
        self.assertEqual(metrics["f1"], 0.0)

    def test_decode_budgeted_topk_selects_count_limited_edges_per_step(self):
        from run_v7_action_specialist import decode_budgeted_topk

        prob = np.array(
            [
                [0.8, 0.1],
                [0.3, 0.9],
                [0.7, 0.4],
                [0.2, 0.6],
            ],
            dtype=np.float32,
        )
        counts = np.array([[2, 1]], dtype=np.int64)

        decoded = decode_budgeted_topk(prob, counts, num_edges=4)

        expected = np.array(
            [
                [True, False],
                [False, True],
                [True, False],
                [False, False],
            ]
        )
        np.testing.assert_array_equal(decoded, expected)


if __name__ == "__main__":
    unittest.main()
