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


class V7ActivityGatedRateHeadTest(unittest.TestCase):
    def test_activity_gated_rate_head_multiplies_rate_value_by_activity_probability(self):
        from pi_jwm.v6_dual_graph import V6DualGraphConfig, V6DualGraphRollout

        torch.manual_seed(20260611)
        config = V6DualGraphConfig(
            node_dim=6,
            physical_edge_dim=8,
            info_edge_dim=5,
            action_dim=3,
            task_dim=7,
            hidden_dim=16,
            horizon=2,
            rate_head_mode="activity_gated",
        )

        outputs = V6DualGraphRollout(config)(_make_batch())

        self.assertIn("link_rate_value", outputs)
        self.assertIn("link_activity_prob", outputs)
        self.assertEqual(outputs["link_rate_value"].shape, (2, 2, 5, 1))
        self.assertEqual(outputs["link_activity_prob"].shape, (2, 2, 5, 1))
        expected = outputs["link_activity_prob"] * outputs["link_rate_value"]
        torch.testing.assert_close(outputs["link_rate"], expected, atol=1e-6, rtol=1e-6)

    def test_residual_activity_gated_rate_head_adds_gated_residual_to_base_rate(self):
        from pi_jwm.v6_dual_graph import V6DualGraphConfig, V6DualGraphRollout

        torch.manual_seed(20260614)
        config = V6DualGraphConfig(
            node_dim=6,
            physical_edge_dim=8,
            info_edge_dim=5,
            action_dim=3,
            task_dim=7,
            hidden_dim=16,
            horizon=2,
            rate_head_mode="residual_activity_gated",
            rate_gate_temperature=2.0,
            rate_gate_floor=0.05,
        )

        outputs = V6DualGraphRollout(config)(_make_batch())

        self.assertIn("link_rate_base", outputs)
        self.assertIn("link_rate_residual", outputs)
        self.assertIn("link_activity_gate", outputs)
        self.assertEqual(outputs["link_rate_base"].shape, (2, 2, 5, 1))
        self.assertEqual(outputs["link_rate_residual"].shape, (2, 2, 5, 1))
        self.assertEqual(outputs["link_activity_gate"].shape, (2, 2, 5, 1))
        self.assertTrue(torch.all(outputs["link_activity_gate"] >= 0.05))
        self.assertTrue(torch.all(outputs["link_activity_gate"] <= 1.0))
        expected = outputs["link_rate_base"] + outputs["link_activity_gate"] * outputs["link_rate_residual"]
        torch.testing.assert_close(outputs["link_rate"], expected, atol=1e-6, rtol=1e-6)

    def test_direct_rate_head_keeps_existing_output_interface(self):
        from pi_jwm.v6_dual_graph import V6DualGraphConfig, V6DualGraphRollout

        config = V6DualGraphConfig(
            node_dim=6,
            physical_edge_dim=8,
            info_edge_dim=5,
            action_dim=3,
            task_dim=7,
            hidden_dim=16,
            horizon=2,
            rate_head_mode="direct",
        )

        outputs = V6DualGraphRollout(config)(_make_batch())

        self.assertIn("link_rate", outputs)
        self.assertNotIn("link_rate_value", outputs)
        self.assertNotIn("link_activity_prob", outputs)
        self.assertNotIn("link_rate_base", outputs)
        self.assertNotIn("link_rate_residual", outputs)
        self.assertNotIn("link_activity_gate", outputs)

    def test_active_rate_auxiliary_head_outputs_active_rate_prediction(self):
        from pi_jwm.v6_dual_graph import V6DualGraphConfig, V6DualGraphRollout

        config = V6DualGraphConfig(
            node_dim=6,
            physical_edge_dim=8,
            info_edge_dim=5,
            action_dim=3,
            task_dim=7,
            hidden_dim=16,
            horizon=2,
            active_rate_auxiliary=True,
        )

        outputs = V6DualGraphRollout(config)(_make_batch())

        self.assertIn("link_active_rate_aux", outputs)
        self.assertEqual(outputs["link_active_rate_aux"].shape, (2, 2, 5, 1))
        self.assertEqual(outputs["link_rate"].shape, outputs["link_active_rate_aux"].shape)

    def test_invalid_rate_gate_parameters_raise(self):
        from pi_jwm.v6_dual_graph import V6DualGraphConfig, V6DualGraphRollout

        base_kwargs = dict(
            node_dim=6,
            physical_edge_dim=8,
            info_edge_dim=5,
            action_dim=3,
            task_dim=7,
            hidden_dim=16,
            horizon=2,
            rate_head_mode="residual_activity_gated",
        )

        with self.assertRaises(ValueError):
            V6DualGraphRollout(V6DualGraphConfig(**base_kwargs, rate_gate_temperature=0.0))
        with self.assertRaises(ValueError):
            V6DualGraphRollout(V6DualGraphConfig(**base_kwargs, rate_gate_floor=-0.1))
        with self.assertRaises(ValueError):
            V6DualGraphRollout(V6DualGraphConfig(**base_kwargs, rate_gate_floor=1.0))

    def test_invalid_rate_head_mode_raises(self):
        from pi_jwm.v6_dual_graph import V6DualGraphConfig, V6DualGraphRollout

        config = V6DualGraphConfig(
            node_dim=6,
            physical_edge_dim=8,
            info_edge_dim=5,
            action_dim=3,
            task_dim=7,
            hidden_dim=16,
            horizon=2,
            rate_head_mode="bad",
        )

        with self.assertRaises(ValueError):
            V6DualGraphRollout(config)


if __name__ == "__main__":
    unittest.main()
