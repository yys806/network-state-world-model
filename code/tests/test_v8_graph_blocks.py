import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


class V8GraphBlocksTest(unittest.TestCase):
    def _make_states(self):
        torch.manual_seed(20260614)
        batch_size = 2
        num_nodes = 4
        num_edges = 5
        hidden_dim = 8
        return {
            "node_state": torch.randn(batch_size, num_nodes, hidden_dim),
            "physical_edge_state": torch.randn(batch_size, num_edges, hidden_dim),
            "info_edge_state": torch.randn(batch_size, num_edges, hidden_dim),
            "action_state": torch.randn(batch_size, num_edges, hidden_dim),
            "edge_src_idx": torch.tensor([0, 0, 1, 2, 3]),
            "edge_dst_idx": torch.tensor([1, 2, 2, 3, 0]),
        }

    def test_edge_update_block_preserves_edge_shape_and_returns_gate(self):
        from pi_jwm.v8_graph_blocks import EdgeUpdateBlock, V8GraphBlockConfig

        states = self._make_states()
        block = EdgeUpdateBlock(V8GraphBlockConfig(hidden_dim=8))

        updated_edge, diagnostics = block(**states)

        self.assertEqual(updated_edge.shape, states["physical_edge_state"].shape)
        self.assertIn("edge_fusion_weights", diagnostics)
        self.assertEqual(diagnostics["edge_fusion_weights"].shape, (2, 5, 3))
        torch.testing.assert_close(
            diagnostics["edge_fusion_weights"].sum(dim=-1),
            torch.ones(2, 5),
            atol=1e-6,
            rtol=1e-6,
        )

    def test_edge_update_block_can_use_cross_attention_fusion(self):
        from pi_jwm.v8_graph_blocks import EdgeUpdateBlock, V8GraphBlockConfig

        states = self._make_states()
        block = EdgeUpdateBlock(V8GraphBlockConfig(hidden_dim=8, fusion_mode="cross_attention", fusion_num_heads=2))

        updated_edge, diagnostics = block(**states)

        self.assertEqual(updated_edge.shape, states["physical_edge_state"].shape)
        self.assertIn("edge_fusion_attention", diagnostics)
        self.assertEqual(diagnostics["edge_fusion_attention"].shape, (2, 5, 3, 3))
        torch.testing.assert_close(
            diagnostics["edge_fusion_attention"].sum(dim=-1),
            torch.ones(2, 5, 3),
            atol=1e-6,
            rtol=1e-6,
        )

    def test_edge_update_block_rejects_bad_fusion_mode(self):
        from pi_jwm.v8_graph_blocks import EdgeUpdateBlock, V8GraphBlockConfig

        with self.assertRaises(ValueError):
            EdgeUpdateBlock(V8GraphBlockConfig(hidden_dim=8, fusion_mode="bad"))

    def test_dual_graph_message_passing_returns_node_and_edge_states(self):
        from pi_jwm.v8_graph_blocks import DualGraphMessagePassing, V8GraphBlockConfig

        states = self._make_states()
        block = DualGraphMessagePassing(V8GraphBlockConfig(hidden_dim=8))

        updated_node, updated_edge, diagnostics = block(**states)

        self.assertEqual(updated_node.shape, states["node_state"].shape)
        self.assertEqual(updated_edge.shape, states["physical_edge_state"].shape)
        self.assertIn("node_in_degree", diagnostics)
        self.assertEqual(diagnostics["node_in_degree"].shape, (4,))
        self.assertTrue(torch.isfinite(updated_node).all())
        self.assertTrue(torch.isfinite(updated_edge).all())

    def test_dual_graph_message_passing_rejects_bad_graph_mode(self):
        from pi_jwm.v8_graph_blocks import DualGraphMessagePassing, V8GraphBlockConfig

        with self.assertRaises(ValueError):
            DualGraphMessagePassing(V8GraphBlockConfig(hidden_dim=8, graph_mode="bad"))

    def test_message_passing_is_equivariant_to_edge_order(self):
        from pi_jwm.v8_graph_blocks import DualGraphMessagePassing, V8GraphBlockConfig

        states = self._make_states()
        block = DualGraphMessagePassing(V8GraphBlockConfig(hidden_dim=8))

        updated_node, updated_edge, _ = block(**states)

        permutation = torch.tensor([2, 4, 1, 3, 0])
        permuted_states = {
            **states,
            "physical_edge_state": states["physical_edge_state"][:, permutation],
            "info_edge_state": states["info_edge_state"][:, permutation],
            "action_state": states["action_state"][:, permutation],
            "edge_src_idx": states["edge_src_idx"][permutation],
            "edge_dst_idx": states["edge_dst_idx"][permutation],
        }
        permuted_node, permuted_edge, _ = block(**permuted_states)

        torch.testing.assert_close(permuted_node, updated_node, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(permuted_edge[:, torch.argsort(permutation)], updated_edge, atol=1e-6, rtol=1e-6)

    def test_message_passing_uses_endpoint_indices(self):
        from pi_jwm.v8_graph_blocks import DualGraphMessagePassing, V8GraphBlockConfig

        states = self._make_states()
        block = DualGraphMessagePassing(V8GraphBlockConfig(hidden_dim=8))
        original_node, original_edge, _ = block(**states)
        rewired = {
            **states,
            "edge_dst_idx": torch.tensor([2, 3, 0, 1, 2]),
        }
        rewired_node, rewired_edge, _ = block(**rewired)

        self.assertGreater(float((original_node - rewired_node).detach().abs().sum()), 0.0)
        self.assertGreater(float((original_edge - rewired_edge).detach().abs().sum()), 0.0)

    def test_v8_rollout_keeps_world_model_output_interface(self):
        from pi_jwm.v6_dual_graph import V6DualGraphBatch
        from pi_jwm.v8_full_world_model import V8FullWorldModelConfig, V8FullWorldModelRollout

        batch = V6DualGraphBatch(
            node_history=torch.randn(2, 3, 4, 6),
            physical_edge_history=torch.randn(2, 3, 5, 8),
            info_edge_history=torch.randn(2, 3, 5, 5),
            action_history=torch.randn(2, 3, 5, 3),
            future_actions=torch.randn(2, 2, 5, 3),
            task_history=torch.randn(2, 3, 7),
        )
        config = V8FullWorldModelConfig(
            node_dim=6,
            physical_edge_dim=8,
            info_edge_dim=5,
            action_dim=3,
            task_dim=7,
            hidden_dim=16,
            horizon=2,
            edge_src_idx=torch.tensor([0, 0, 1, 2, 3]),
            edge_dst_idx=torch.tensor([1, 2, 2, 3, 0]),
            fusion_mode="cross_attention",
            fusion_num_heads=4,
            return_message_diagnostics=True,
        )

        outputs = V8FullWorldModelRollout(config)(batch)

        self.assertEqual(outputs["node"].shape, (2, 2, 4, 6))
        self.assertEqual(outputs["link_activity_logit"].shape, (2, 2, 5, 1))
        self.assertEqual(outputs["link_rate"].shape, (2, 2, 5, 1))
        self.assertEqual(outputs["task"].shape, (2, 2, 7))
        self.assertIn("message_node_in_degree", outputs)
        self.assertEqual(outputs["message_node_in_degree"].shape, (4,))

    def test_v8_rollout_is_conditioned_on_each_future_action_step(self):
        from pi_jwm.v6_dual_graph import V6DualGraphBatch
        from pi_jwm.v8_full_world_model import V8FullWorldModelConfig, V8FullWorldModelRollout

        torch.manual_seed(20260711)
        shared = {
            "node_history": torch.randn(1, 3, 4, 6),
            "physical_edge_history": torch.randn(1, 3, 5, 8),
            "info_edge_history": torch.randn(1, 3, 5, 5),
            "action_history": torch.randn(1, 3, 5, 3),
            "task_history": torch.randn(1, 3, 7),
        }
        baseline_actions = torch.zeros(1, 2, 5, 3)
        changed_actions = baseline_actions.clone()
        changed_actions[:, 0, :, 0] = 1.0
        baseline = V6DualGraphBatch(future_actions=baseline_actions, **shared)
        changed = V6DualGraphBatch(future_actions=changed_actions, **shared)
        model = V8FullWorldModelRollout(
            V8FullWorldModelConfig(
                node_dim=6,
                physical_edge_dim=8,
                info_edge_dim=5,
                action_dim=3,
                task_dim=7,
                hidden_dim=16,
                horizon=2,
                edge_src_idx=torch.tensor([0, 0, 1, 2, 3]),
                edge_dst_idx=torch.tensor([1, 2, 2, 3, 0]),
            )
        )

        baseline_output = model(baseline)
        changed_output = model(changed)

        self.assertGreater(
            float(
                (baseline_output["link_rate"][:, 1] - changed_output["link_rate"][:, 1])
                .detach()
                .abs()
                .sum()
            ),
            0.0,
        )

    def test_v8_rollout_can_return_active_rate_auxiliary_head(self):
        from pi_jwm.v6_dual_graph import V6DualGraphBatch
        from pi_jwm.v8_full_world_model import V8FullWorldModelConfig, V8FullWorldModelRollout

        batch = V6DualGraphBatch(
            node_history=torch.randn(2, 3, 4, 6),
            physical_edge_history=torch.randn(2, 3, 5, 8),
            info_edge_history=torch.randn(2, 3, 5, 5),
            action_history=torch.randn(2, 3, 5, 3),
            future_actions=torch.randn(2, 2, 5, 3),
            task_history=torch.randn(2, 3, 7),
        )
        config = V8FullWorldModelConfig(
            node_dim=6,
            physical_edge_dim=8,
            info_edge_dim=5,
            action_dim=3,
            task_dim=7,
            hidden_dim=16,
            horizon=2,
            edge_src_idx=torch.tensor([0, 0, 1, 2, 3]),
            edge_dst_idx=torch.tensor([1, 2, 2, 3, 0]),
            active_rate_auxiliary=True,
        )

        outputs = V8FullWorldModelRollout(config)(batch)

        self.assertIn("link_active_rate_aux", outputs)
        self.assertEqual(outputs["link_active_rate_aux"].shape, (2, 2, 5, 1))

    def test_v8_rollout_can_use_hurdle_rate_output_mode(self):
        from pi_jwm.v6_dual_graph import V6DualGraphBatch
        from pi_jwm.v8_full_world_model import V8FullWorldModelConfig, V8FullWorldModelRollout

        batch = V6DualGraphBatch(
            node_history=torch.randn(2, 3, 4, 6),
            physical_edge_history=torch.randn(2, 3, 5, 8),
            info_edge_history=torch.randn(2, 3, 5, 5),
            action_history=torch.randn(2, 3, 5, 3),
            future_actions=torch.randn(2, 2, 5, 3),
            task_history=torch.randn(2, 3, 7),
        )
        config = V8FullWorldModelConfig(
            node_dim=6,
            physical_edge_dim=8,
            info_edge_dim=5,
            action_dim=3,
            task_dim=7,
            hidden_dim=16,
            horizon=2,
            edge_src_idx=torch.tensor([0, 0, 1, 2, 3]),
            edge_dst_idx=torch.tensor([1, 2, 2, 3, 0]),
            rate_output_mode="hurdle_soft",
        )

        outputs = V8FullWorldModelRollout(config)(batch)

        self.assertIn("link_positive_rate", outputs)
        self.assertEqual(outputs["link_positive_rate"].shape, (2, 2, 5, 1))
        self.assertNotIn("link_active_mass_rate", outputs)
        expected = torch.sigmoid(outputs["link_activity_logit"]) * outputs["link_positive_rate"]
        torch.testing.assert_close(outputs["link_rate"], expected, atol=1e-6, rtol=1e-6)

    def test_v8_rollout_can_allocate_active_rate_mass(self):
        from pi_jwm.v6_dual_graph import V6DualGraphBatch
        from pi_jwm.v8_full_world_model import V8FullWorldModelConfig, V8FullWorldModelRollout

        torch.manual_seed(20260617)
        batch = V6DualGraphBatch(
            node_history=torch.randn(2, 3, 4, 6),
            physical_edge_history=torch.randn(2, 3, 5, 8),
            info_edge_history=torch.randn(2, 3, 5, 5),
            action_history=torch.randn(2, 3, 5, 3),
            future_actions=torch.randn(2, 2, 5, 3),
            task_history=torch.randn(2, 3, 7),
        )
        config = V8FullWorldModelConfig(
            node_dim=6,
            physical_edge_dim=8,
            info_edge_dim=5,
            action_dim=3,
            task_dim=7,
            hidden_dim=16,
            horizon=2,
            edge_src_idx=torch.tensor([0, 0, 1, 2, 3]),
            edge_dst_idx=torch.tensor([1, 2, 2, 3, 0]),
            rate_output_mode="hurdle_mass",
        )

        outputs = V8FullWorldModelRollout(config)(batch)

        self.assertIn("link_positive_rate", outputs)
        self.assertIn("link_active_mass_rate", outputs)
        self.assertIn("link_active_mass_total", outputs)
        self.assertEqual(outputs["link_active_mass_rate"].shape, (2, 2, 5, 1))
        self.assertEqual(outputs["link_active_mass_total"].shape, (2, 2, 1, 1))
        torch.testing.assert_close(
            outputs["link_active_mass_rate"].sum(dim=2, keepdim=True),
            outputs["link_active_mass_total"],
            atol=1e-5,
            rtol=1e-5,
        )
        torch.testing.assert_close(outputs["link_rate"], outputs["link_active_mass_rate"], atol=1e-6, rtol=1e-6)

    def test_v8_rollout_can_route_event_memory_to_activity_head_only(self):
        from pi_jwm.v6_dual_graph import V6DualGraphBatch
        from pi_jwm.v8_full_world_model import V8FullWorldModelConfig, V8FullWorldModelRollout

        torch.manual_seed(20260616)
        base_info = torch.randn(2, 3, 5, 5)
        batch_a = V6DualGraphBatch(
            node_history=torch.randn(2, 3, 4, 6),
            physical_edge_history=torch.randn(2, 3, 5, 8),
            info_edge_history=torch.cat([base_info, torch.zeros(2, 3, 5, 3)], dim=-1),
            action_history=torch.randn(2, 3, 5, 3),
            future_actions=torch.randn(2, 2, 5, 3),
            task_history=torch.randn(2, 3, 7),
        )
        batch_b = V6DualGraphBatch(
            node_history=batch_a.node_history,
            physical_edge_history=batch_a.physical_edge_history,
            info_edge_history=torch.cat([base_info, torch.ones(2, 3, 5, 3)], dim=-1),
            action_history=batch_a.action_history,
            future_actions=batch_a.future_actions,
            task_history=batch_a.task_history,
        )
        config = V8FullWorldModelConfig(
            node_dim=6,
            physical_edge_dim=8,
            info_edge_dim=8,
            action_dim=3,
            task_dim=7,
            hidden_dim=16,
            horizon=2,
            edge_src_idx=torch.tensor([0, 0, 1, 2, 3]),
            edge_dst_idx=torch.tensor([1, 2, 2, 3, 0]),
            activity_memory_dim=3,
            activity_memory_routing="activity_only",
            rate_output_mode="hurdle_soft",
        )
        model = V8FullWorldModelRollout(config)

        outputs_a = model(batch_a)
        outputs_b = model(batch_b)

        self.assertEqual(outputs_a["link_activity_logit"].shape, (2, 2, 5, 1))
        activity_delta = (outputs_a["link_activity_logit"] - outputs_b["link_activity_logit"]).detach().abs().sum()
        self.assertGreater(float(activity_delta), 0.0)
        torch.testing.assert_close(outputs_a["link_positive_rate"], outputs_b["link_positive_rate"], atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(outputs_a["node"], outputs_b["node"], atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(outputs_a["task"], outputs_b["task"], atol=1e-6, rtol=1e-6)

    def test_v8_rollout_can_use_moe_active_rate_auxiliary_head(self):
        from pi_jwm.v6_dual_graph import V6DualGraphBatch
        from pi_jwm.v8_full_world_model import V8FullWorldModelConfig, V8FullWorldModelRollout

        batch = V6DualGraphBatch(
            node_history=torch.randn(2, 3, 4, 6),
            physical_edge_history=torch.randn(2, 3, 5, 8),
            info_edge_history=torch.randn(2, 3, 5, 5),
            action_history=torch.randn(2, 3, 5, 3),
            future_actions=torch.randn(2, 2, 5, 3),
            task_history=torch.randn(2, 3, 7),
        )
        config = V8FullWorldModelConfig(
            node_dim=6,
            physical_edge_dim=8,
            info_edge_dim=5,
            action_dim=3,
            task_dim=7,
            hidden_dim=16,
            horizon=2,
            edge_src_idx=torch.tensor([0, 0, 1, 2, 3]),
            edge_dst_idx=torch.tensor([1, 2, 2, 3, 0]),
            active_rate_auxiliary=True,
            active_rate_head_mode="moe",
            num_rate_experts=3,
        )

        outputs = V8FullWorldModelRollout(config)(batch)

        self.assertIn("link_active_rate_aux", outputs)
        self.assertIn("rate_expert_weights", outputs)
        self.assertEqual(outputs["link_active_rate_aux"].shape, (2, 2, 5, 1))
        self.assertEqual(outputs["rate_expert_weights"].shape, (2, 2, 5, 3))
        torch.testing.assert_close(
            outputs["rate_expert_weights"].sum(dim=-1),
            torch.ones(2, 2, 5),
            atol=1e-6,
            rtol=1e-6,
        )

    def test_v8_rollout_can_use_temporal_conv_history_encoder(self):
        from pi_jwm.v6_dual_graph import V6DualGraphBatch
        from pi_jwm.v8_full_world_model import V8FullWorldModelConfig, V8FullWorldModelRollout

        batch = V6DualGraphBatch(
            node_history=torch.randn(2, 4, 4, 6),
            physical_edge_history=torch.randn(2, 4, 5, 8),
            info_edge_history=torch.randn(2, 4, 5, 5),
            action_history=torch.randn(2, 4, 5, 3),
            future_actions=torch.randn(2, 2, 5, 3),
            task_history=torch.randn(2, 4, 7),
        )
        config = V8FullWorldModelConfig(
            node_dim=6,
            physical_edge_dim=8,
            info_edge_dim=5,
            action_dim=3,
            task_dim=7,
            hidden_dim=16,
            horizon=2,
            edge_src_idx=torch.tensor([0, 0, 1, 2, 3]),
            edge_dst_idx=torch.tensor([1, 2, 2, 3, 0]),
            history_encoder="temporal_conv",
        )
        model = V8FullWorldModelRollout(config)

        outputs = model(batch)

        self.assertEqual(outputs["node"].shape, (2, 2, 4, 6))
        self.assertEqual(model.config.history_encoder, "temporal_conv")
        self.assertIsNotNone(model.node_temporal_encoder)

    def test_v8_rollout_can_use_stgcn_light_history_encoder(self):
        from pi_jwm.v6_dual_graph import V6DualGraphBatch
        from pi_jwm.v8_full_world_model import V8FullWorldModelConfig, V8FullWorldModelRollout

        batch = V6DualGraphBatch(
            node_history=torch.randn(2, 4, 4, 6),
            physical_edge_history=torch.randn(2, 4, 5, 8),
            info_edge_history=torch.randn(2, 4, 5, 5),
            action_history=torch.randn(2, 4, 5, 3),
            future_actions=torch.randn(2, 2, 5, 3),
            task_history=torch.randn(2, 4, 7),
        )
        config = V8FullWorldModelConfig(
            node_dim=6,
            physical_edge_dim=8,
            info_edge_dim=5,
            action_dim=3,
            task_dim=7,
            hidden_dim=16,
            horizon=2,
            edge_src_idx=torch.tensor([0, 0, 1, 2, 3]),
            edge_dst_idx=torch.tensor([1, 2, 2, 3, 0]),
            history_encoder="stgcn_light",
        )
        model = V8FullWorldModelRollout(config)

        outputs = model(batch)

        self.assertEqual(outputs["node"].shape, (2, 2, 4, 6))
        self.assertEqual(outputs["link_rate"].shape, (2, 2, 5, 1))
        self.assertEqual(model.config.history_encoder, "stgcn_light")
        self.assertIsNotNone(model.history_message_passing)
        self.assertIsNotNone(model.node_temporal_encoder)
        self.assertIsNotNone(model.physical_edge_temporal_encoder)

    def test_v8_rollout_can_use_stgcn_full_history_encoder(self):
        from pi_jwm.v6_dual_graph import V6DualGraphBatch
        from pi_jwm.v8_full_world_model import V8FullWorldModelConfig, V8FullWorldModelRollout

        batch = V6DualGraphBatch(
            node_history=torch.randn(2, 4, 4, 6),
            physical_edge_history=torch.randn(2, 4, 5, 8),
            info_edge_history=torch.randn(2, 4, 5, 5),
            action_history=torch.randn(2, 4, 5, 3),
            future_actions=torch.randn(2, 2, 5, 3),
            task_history=torch.randn(2, 4, 7),
        )
        config = V8FullWorldModelConfig(
            node_dim=6,
            physical_edge_dim=8,
            info_edge_dim=5,
            action_dim=3,
            task_dim=7,
            hidden_dim=16,
            horizon=2,
            edge_src_idx=torch.tensor([0, 0, 1, 2, 3]),
            edge_dst_idx=torch.tensor([1, 2, 2, 3, 0]),
            history_encoder="stgcn_full",
        )
        model = V8FullWorldModelRollout(config)

        outputs = model(batch)

        self.assertEqual(outputs["node"].shape, (2, 2, 4, 6))
        self.assertEqual(outputs["link_rate"].shape, (2, 2, 5, 1))
        self.assertEqual(model.config.history_encoder, "stgcn_full")
        self.assertIsNotNone(model.node_stgcn_full_encoder)
        self.assertIsNotNone(model.edge_stgcn_full_encoder)
        self.assertIsNot(model.node_stgcn_full_encoder, model.history_message_passing)

    def test_v8_rollout_can_use_recurrent_latent_transition(self):
        from pi_jwm.v6_dual_graph import V6DualGraphBatch
        from pi_jwm.v8_full_world_model import V8FullWorldModelConfig, V8FullWorldModelRollout

        batch = V6DualGraphBatch(
            node_history=torch.randn(2, 4, 4, 6),
            physical_edge_history=torch.randn(2, 4, 5, 8),
            info_edge_history=torch.randn(2, 4, 5, 5),
            action_history=torch.randn(2, 4, 5, 3),
            future_actions=torch.randn(2, 3, 5, 3),
            task_history=torch.randn(2, 4, 7),
        )
        config = V8FullWorldModelConfig(
            node_dim=6,
            physical_edge_dim=8,
            info_edge_dim=5,
            action_dim=3,
            task_dim=7,
            hidden_dim=16,
            horizon=3,
            edge_src_idx=torch.tensor([0, 0, 1, 2, 3]),
            edge_dst_idx=torch.tensor([1, 2, 2, 3, 0]),
            latent_transition_mode="recurrent",
        )
        model = V8FullWorldModelRollout(config)

        outputs = model(batch)

        self.assertEqual(outputs["node"].shape, (2, 3, 4, 6))
        self.assertEqual(outputs["link_rate"].shape, (2, 3, 5, 1))
        self.assertEqual(outputs["task"].shape, (2, 3, 7))
        self.assertEqual(model.config.latent_transition_mode, "recurrent")
        self.assertIsNotNone(model.node_latent_rollout)
        self.assertIsNotNone(model.edge_latent_rollout)

    def test_v8_rollout_can_use_sparse_adaptive_edge_context(self):
        from pi_jwm.v6_dual_graph import V6DualGraphBatch
        from pi_jwm.v8_full_world_model import V8FullWorldModelConfig, V8FullWorldModelRollout

        batch = V6DualGraphBatch(
            node_history=torch.randn(2, 4, 4, 6),
            physical_edge_history=torch.randn(2, 4, 5, 8),
            info_edge_history=torch.randn(2, 4, 5, 5),
            action_history=torch.randn(2, 4, 5, 3),
            future_actions=torch.randn(2, 2, 5, 3),
            task_history=torch.randn(2, 4, 7),
        )
        config = V8FullWorldModelConfig(
            node_dim=6,
            physical_edge_dim=8,
            info_edge_dim=5,
            action_dim=3,
            task_dim=7,
            hidden_dim=16,
            horizon=2,
            edge_src_idx=torch.tensor([0, 0, 1, 2, 3]),
            edge_dst_idx=torch.tensor([1, 2, 2, 3, 0]),
            adaptive_edge_context="sparse_attention",
            adaptive_edge_topk=2,
            return_message_diagnostics=True,
        )
        model = V8FullWorldModelRollout(config)

        outputs = model(batch)

        self.assertEqual(outputs["link_rate"].shape, (2, 2, 5, 1))
        self.assertEqual(model.config.adaptive_edge_context, "sparse_attention")
        self.assertIsNotNone(model.adaptive_edge_context)
        self.assertIn("adaptive_edge_context_attention", outputs)
        attention = outputs["adaptive_edge_context_attention"]
        self.assertEqual(attention.shape, (2, 5, 5))
        torch.testing.assert_close(attention.sum(dim=-1), torch.ones(2, 5), atol=1e-6, rtol=1e-6)
        self.assertLessEqual(int((attention > 0.0).sum(dim=-1).max()), 2)

    def test_v8_rollout_rejects_bad_latent_transition_mode(self):
        from pi_jwm.v8_full_world_model import V8FullWorldModelConfig, V8FullWorldModelRollout

        with self.assertRaises(ValueError):
            V8FullWorldModelRollout(
                V8FullWorldModelConfig(
                    node_dim=6,
                    physical_edge_dim=8,
                    info_edge_dim=5,
                    action_dim=3,
                    task_dim=7,
                    hidden_dim=16,
                    horizon=2,
                    edge_src_idx=torch.tensor([0, 0, 1, 2, 3]),
                    edge_dst_idx=torch.tensor([1, 2, 2, 3, 0]),
                    latent_transition_mode="bad",
                )
            )


if __name__ == "__main__":
    unittest.main()
