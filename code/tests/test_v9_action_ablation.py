import sys
import unittest
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


class V9ActionAblationTest(unittest.TestCase):
    def test_apply_action_ablation_zeroes_raw_future_or_history_actions(self):
        from evaluate_v9_action_ablation import apply_action_ablation
        from pi_jwm.v6_dual_graph import V6DualGraphBatch

        batch = V6DualGraphBatch(
            node_history=torch.ones(2, 3),
            physical_edge_history=torch.ones(2, 3),
            info_edge_history=torch.ones(2, 3),
            action_history=torch.full((2, 3, 4), 9.0),
            future_actions=torch.full((1, 3, 4), 7.0),
            task_history=torch.ones(2, 3),
            link_rate_baseline=None,
        )
        stats = {
            "edge_a_hist": (np.full((1, 2, 3, 4), 2.0, dtype=np.float32), np.full((1, 2, 3, 4), 4.0, dtype=np.float32)),
            "edge_a_future": (np.full((1, 1, 3, 4), 3.0, dtype=np.float32), np.full((1, 1, 3, 4), 2.0, dtype=np.float32)),
        }

        zero_future = apply_action_ablation(batch, stats, "zero_future_actions")
        zero_history = apply_action_ablation(batch, stats, "zero_history_actions")
        normal = apply_action_ablation(batch, stats, "normal_actions")

        self.assertIs(normal, batch)
        self.assertTrue(torch.allclose(zero_future.future_actions, torch.full((1, 3, 4), -1.5)))
        self.assertTrue(torch.allclose(zero_future.action_history, batch.action_history))
        self.assertTrue(torch.allclose(zero_history.action_history, torch.full((2, 3, 4), -0.5)))
        self.assertTrue(torch.allclose(zero_history.future_actions, batch.future_actions))

    def test_apply_action_ablation_rejects_unknown_mode(self):
        from evaluate_v9_action_ablation import apply_action_ablation
        from pi_jwm.v6_dual_graph import V6DualGraphBatch

        batch = V6DualGraphBatch(
            node_history=torch.ones(1, 1),
            physical_edge_history=torch.ones(1, 1),
            info_edge_history=torch.ones(1, 1),
            action_history=torch.ones(1, 1, 1),
            future_actions=torch.ones(1, 1, 1),
            task_history=torch.ones(1, 1),
            link_rate_baseline=None,
        )

        with self.assertRaisesRegex(ValueError, "Unknown action ablation mode"):
            apply_action_ablation(batch, {}, "bad_mode")


if __name__ == "__main__":
    unittest.main()
