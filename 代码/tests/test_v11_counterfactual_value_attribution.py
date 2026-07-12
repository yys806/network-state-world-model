import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


class V11CounterfactualValueAttributionTest(unittest.TestCase):
    def test_dim_replacement_preserves_predicted_activity_mask(self):
        from diagnose_v11_counterfactual_value_attribution import make_counterfactual_actions

        baseline = np.zeros((1, 2, 2, 3), dtype=np.float32)
        baseline[0, 0, 0, 1] = 5.0
        baseline[0, 1, 1, 1] = 7.0
        baseline[0, 1, 0, 2] = 9.0
        truth = np.arange(12, dtype=np.float32).reshape(1, 2, 2, 3) + 100.0

        replaced = make_counterfactual_actions(baseline, truth, "dim:1")

        self.assertEqual(replaced[0, 0, 0, 1], truth[0, 0, 0, 1])
        self.assertEqual(replaced[0, 1, 1, 1], truth[0, 1, 1, 1])
        self.assertEqual(replaced[0, 0, 1, 1], 0.0)
        self.assertEqual(replaced[0, 1, 0, 2], baseline[0, 1, 0, 2])

    def test_step_replacement_keeps_other_steps_unchanged(self):
        from diagnose_v11_counterfactual_value_attribution import make_counterfactual_actions

        baseline = np.ones((1, 3, 2, 2), dtype=np.float32)
        baseline[:, 2] = 0.0
        truth = np.full_like(baseline, 10.0)

        replaced = make_counterfactual_actions(baseline, truth, "step:1")

        self.assertTrue(np.allclose(replaced[:, 1], 10.0))
        self.assertTrue(np.allclose(replaced[:, 0], baseline[:, 0]))
        self.assertTrue(np.allclose(replaced[:, 2], baseline[:, 2]))

    def test_step_dim_replacement_targets_single_slice(self):
        from diagnose_v11_counterfactual_value_attribution import make_counterfactual_actions

        baseline = np.ones((1, 2, 2, 3), dtype=np.float32)
        truth = np.full_like(baseline, 8.0)

        replaced = make_counterfactual_actions(baseline, truth, "step_dim:1:2")

        self.assertTrue(np.allclose(replaced[:, 1, :, 2], 8.0))
        self.assertTrue(np.allclose(replaced[:, 1, :, 0], 1.0))
        self.assertTrue(np.allclose(replaced[:, 0, :, 2], 1.0))

    def test_unknown_mode_is_rejected(self):
        from diagnose_v11_counterfactual_value_attribution import make_counterfactual_actions

        with self.assertRaises(ValueError):
            make_counterfactual_actions(np.ones((1, 1, 1, 1)), np.ones((1, 1, 1, 1)), "bad")

    def test_raw_future_from_normalized_inverts_stats(self):
        import torch

        from diagnose_v11_counterfactual_value_attribution import raw_future_from_normalized

        stats = {"edge_a_future": (np.array([[10.0, 20.0]], dtype=np.float32), np.array([[2.0, 4.0]], dtype=np.float32))}
        normalized = torch.tensor([[[1.0, -1.0], [0.0, 0.5]]], dtype=torch.float32)

        raw = raw_future_from_normalized(normalized, stats)

        expected = np.array([[[12.0, 16.0], [10.0, 22.0]]], dtype=np.float32)
        self.assertTrue(np.allclose(raw, expected))


if __name__ == "__main__":
    unittest.main()
