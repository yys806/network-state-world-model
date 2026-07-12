import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _make_batch():
    from pi_jwm.v6_dual_graph import V6DualGraphBatch

    return V6DualGraphBatch(
        node_history=torch.randn(2, 3, 4, 6),
        physical_edge_history=torch.randn(2, 3, 5, 8),
        info_edge_history=torch.randn(2, 3, 5, 5),
        action_history=torch.randn(2, 3, 5, 3),
        future_actions=torch.randn(2, 2, 5, 3),
        task_history=torch.randn(2, 3, 7),
    )


class V7DualGraphFusionTest(unittest.TestCase):
    def test_gated_fusion_returns_normalized_modality_weights(self):
        from pi_jwm.v6_dual_graph import V6DualGraphConfig, V6DualGraphRollout

        config = V6DualGraphConfig(
            node_dim=6,
            physical_edge_dim=8,
            info_edge_dim=5,
            action_dim=3,
            task_dim=7,
            hidden_dim=16,
            horizon=2,
            fusion_mode="gated",
            return_fusion_diagnostics=True,
        )

        outputs = V6DualGraphRollout(config)(_make_batch())

        self.assertEqual(outputs["node"].shape, (2, 2, 4, 6))
        self.assertIn("fusion_weights", outputs)
        self.assertEqual(outputs["fusion_weights"].shape, (2, 5, 3))
        self.assertTrue(torch.all(outputs["fusion_weights"] >= 0.0))
        torch.testing.assert_close(
            outputs["fusion_weights"].sum(dim=-1),
            torch.ones(2, 5),
            atol=1e-6,
            rtol=1e-6,
        )

    def test_cross_attention_fusion_returns_modality_attention_matrix(self):
        from pi_jwm.v6_dual_graph import V6DualGraphConfig, V6DualGraphRollout

        config = V6DualGraphConfig(
            node_dim=6,
            physical_edge_dim=8,
            info_edge_dim=5,
            action_dim=3,
            task_dim=7,
            hidden_dim=16,
            horizon=2,
            fusion_mode="cross_attention",
            fusion_num_heads=4,
            return_fusion_diagnostics=True,
        )

        outputs = V6DualGraphRollout(config)(_make_batch())

        self.assertEqual(outputs["link_rate"].shape, (2, 2, 5, 1))
        self.assertIn("fusion_attention", outputs)
        self.assertEqual(outputs["fusion_attention"].shape, (2, 5, 3, 3))
        torch.testing.assert_close(
            outputs["fusion_attention"].sum(dim=-1),
            torch.ones(2, 5, 3),
            atol=1e-6,
            rtol=1e-6,
        )

    def test_hybrid_attention_fusion_returns_attention_and_residual_gate(self):
        from pi_jwm.v6_dual_graph import V6DualGraphConfig, V6DualGraphRollout

        config = V6DualGraphConfig(
            node_dim=6,
            physical_edge_dim=8,
            info_edge_dim=5,
            action_dim=3,
            task_dim=7,
            hidden_dim=16,
            horizon=2,
            fusion_mode="hybrid_attention",
            fusion_num_heads=4,
            return_fusion_diagnostics=True,
        )

        outputs = V6DualGraphRollout(config)(_make_batch())

        self.assertEqual(outputs["link_activity_logit"].shape, (2, 2, 5, 1))
        self.assertIn("fusion_attention", outputs)
        self.assertIn("fusion_residual_gate", outputs)
        self.assertEqual(outputs["fusion_attention"].shape, (2, 5, 3, 3))
        self.assertEqual(outputs["fusion_residual_gate"].shape, (2, 5, 1))
        self.assertTrue(torch.all(outputs["fusion_residual_gate"] >= 0.0))
        self.assertTrue(torch.all(outputs["fusion_residual_gate"] <= 1.0))
        torch.testing.assert_close(
            outputs["fusion_attention"].sum(dim=-1),
            torch.ones(2, 5, 3),
            atol=1e-6,
            rtol=1e-6,
        )

    def test_invalid_fusion_mode_raises(self):
        from pi_jwm.v6_dual_graph import V6DualGraphConfig, V6DualGraphRollout

        config = V6DualGraphConfig(
            node_dim=6,
            physical_edge_dim=8,
            info_edge_dim=5,
            action_dim=3,
            task_dim=7,
            hidden_dim=16,
            horizon=2,
            fusion_mode="bad",
        )

        with self.assertRaises(ValueError):
            V6DualGraphRollout(config)


if __name__ == "__main__":
    unittest.main()
