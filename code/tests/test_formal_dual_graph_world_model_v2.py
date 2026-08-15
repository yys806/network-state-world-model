from __future__ import annotations

import copy
import unittest

import torch

from test_formal_dual_graph_world_model_v1 import fake_formal_batch


class FormalDirectedDynamicWorldModelV2Tests(unittest.TestCase):
    def test_cfe_and_dag_soft_weights_attenuate_single_relation_messages(self):
        from pi_jwm.formal_dual_graph_world_model_v2 import FormalDirectedDynamicWorldModelV2

        flow = torch.tensor([[[2.0]]])
        edge = torch.tensor([[[8.0]]])
        full_flow, full_edge = FormalDirectedDynamicWorldModelV2._cfe_messages(
            flow, edge, torch.tensor([[[1.0]]])
        )
        soft_flow, soft_edge = FormalDirectedDynamicWorldModelV2._cfe_messages(
            flow, edge, torch.tensor([[[0.25]]])
        )
        torch.testing.assert_close(soft_flow, full_flow * 0.25)
        torch.testing.assert_close(soft_edge, full_edge * 0.25)

        task = torch.tensor([[[4.0], [10.0]]])
        dag_index = torch.tensor([[0], [1]])
        full_task, full_relation = FormalDirectedDynamicWorldModelV2._dag_messages(
            task, dag_index, torch.tensor([[1.0]])
        )
        soft_task, soft_relation = FormalDirectedDynamicWorldModelV2._dag_messages(
            task, dag_index, torch.tensor([[0.25]])
        )
        torch.testing.assert_close(soft_task, full_task * 0.25)
        torch.testing.assert_close(soft_relation, full_relation * 0.25)

    def test_information_agent_history_does_not_read_physical_node_state(self):
        from pi_jwm.formal_dual_graph_world_model_v2 import (
            FormalDirectedDynamicWorldModelConfig,
            FormalDirectedDynamicWorldModelV2,
        )

        torch.manual_seed(17)
        model = FormalDirectedDynamicWorldModelV2(
            FormalDirectedDynamicWorldModelConfig(
                hidden_dim=8,
                history_steps=3,
                horizon_steps=2,
                use_cross_coupling=False,
            )
        ).eval()
        baseline = fake_formal_batch()
        changed_physical = copy.deepcopy(baseline)
        changed_physical["history"]["node_state"].add_(10000.0)

        with torch.no_grad():
            agent_baseline = model.encode_information_agent_history(
                baseline["history"], baseline["static"]
            )
            agent_changed = model.encode_information_agent_history(
                changed_physical["history"], changed_physical["static"]
            )

        torch.testing.assert_close(agent_baseline, agent_changed)

    def test_information_agent_history_changes_with_flow_state_and_direction(self):
        from pi_jwm.formal_dual_graph_world_model_v2 import (
            FormalDirectedDynamicWorldModelConfig,
            FormalDirectedDynamicWorldModelV2,
        )

        torch.manual_seed(19)
        model = FormalDirectedDynamicWorldModelV2(
            FormalDirectedDynamicWorldModelConfig(
                hidden_dim=8,
                history_steps=3,
                horizon_steps=2,
                use_cross_coupling=False,
            )
        ).eval()
        baseline = fake_formal_batch()
        changed_flow = copy.deepcopy(baseline)
        changed_flow["history"]["flow_state"][:, :, 0].add_(3.0)
        reversed_flow = copy.deepcopy(baseline)
        reversed_flow["static"]["flow_endpoint_index"][:, 0] = torch.tensor([1, 0])

        with torch.no_grad():
            agent_baseline = model.encode_information_agent_history(
                baseline["history"], baseline["static"]
            )
            agent_changed_flow = model.encode_information_agent_history(
                changed_flow["history"], changed_flow["static"]
            )
            agent_reversed_flow = model.encode_information_agent_history(
                reversed_flow["history"], reversed_flow["static"]
            )

        self.assertFalse(torch.allclose(agent_baseline, agent_changed_flow))
        self.assertFalse(torch.allclose(agent_baseline, agent_reversed_flow))

    def test_reversing_physical_and_information_edges_changes_rollout(self):
        from pi_jwm.formal_dual_graph_world_model_v2 import (
            FormalDirectedDynamicWorldModelConfig,
            FormalDirectedDynamicWorldModelV2,
        )

        torch.manual_seed(23)
        model = FormalDirectedDynamicWorldModelV2(
            FormalDirectedDynamicWorldModelConfig(
                hidden_dim=8,
                history_steps=3,
                horizon_steps=2,
            )
        ).eval()
        baseline = fake_formal_batch()
        reversed_physical = copy.deepcopy(baseline)
        reversed_physical["static"]["physical_edge_endpoint_index"][:, 0] = torch.tensor([1, 0])
        reversed_information = copy.deepcopy(baseline)
        reversed_information["static"]["flow_endpoint_index"][:, 0] = torch.tensor([1, 0])

        with torch.no_grad():
            base_output = model(baseline)
            physical_output = model(reversed_physical)
            information_output = model(reversed_information)

        self.assertFalse(
            torch.allclose(base_output["node_state_mean"], physical_output["node_state_mean"])
        )
        self.assertFalse(
            torch.allclose(base_output["flow_state_mean"], information_output["flow_state_mean"])
        )

    def test_cross_coupling_is_the_only_physical_to_information_path(self):
        from pi_jwm.formal_dual_graph_world_model_v2 import (
            FormalDirectedDynamicWorldModelConfig,
            FormalDirectedDynamicWorldModelV2,
        )

        baseline = fake_formal_batch()
        changed_physical = copy.deepcopy(baseline)
        changed_physical["history"]["node_state"][:, :, 0].add_(50.0)
        torch.manual_seed(29)
        uncoupled = FormalDirectedDynamicWorldModelV2(
            FormalDirectedDynamicWorldModelConfig(
                hidden_dim=8,
                history_steps=3,
                horizon_steps=2,
                use_cross_coupling=False,
            )
        ).eval()
        torch.manual_seed(29)
        coupled = FormalDirectedDynamicWorldModelV2(
            FormalDirectedDynamicWorldModelConfig(
                hidden_dim=8,
                history_steps=3,
                horizon_steps=2,
                use_cross_coupling=True,
            )
        ).eval()

        with torch.no_grad():
            uncoupled_base = uncoupled(baseline)["flow_state_mean"]
            uncoupled_changed = uncoupled(changed_physical)["flow_state_mean"]
            coupled_base = coupled(baseline)["flow_state_mean"]
            coupled_changed = coupled(changed_physical)["flow_state_mean"]

        torch.testing.assert_close(uncoupled_base, uncoupled_changed)
        self.assertFalse(torch.allclose(coupled_base, coupled_changed))

    def test_disabling_cip_and_cfe_blocks_all_physical_state_paths_to_agents(self):
        from pi_jwm.formal_dual_graph_world_model_v2 import (
            FormalDirectedDynamicWorldModelConfig,
            FormalDirectedDynamicWorldModelV2,
        )

        torch.manual_seed(30)
        model = FormalDirectedDynamicWorldModelV2(
            FormalDirectedDynamicWorldModelConfig(
                hidden_dim=8,
                history_steps=3,
                horizon_steps=2,
                use_cross_coupling=True,
            )
        ).eval()
        model.agent_cip_gate.forward = lambda current, message: torch.zeros_like(message)
        model.flow_cfe_gate.forward = lambda current, message: torch.zeros_like(message)

        baseline = fake_formal_batch()
        changed_physical = copy.deepcopy(baseline)
        changed_physical["history"]["node_state"].add_(100.0)

        captured: list[torch.Tensor] = []
        handle = model.agent_transition.register_forward_hook(
            lambda module, inputs, output: captured.append(output.detach().clone())
        )
        try:
            with torch.no_grad():
                model(baseline)
                baseline_agents = torch.stack(captured)
                captured.clear()
                model(changed_physical)
                changed_agents = torch.stack(captured)
        finally:
            handle.remove()

        torch.testing.assert_close(baseline_agents, changed_agents)

    def test_future_entity_weights_and_direct_cfe_are_recomputed(self):
        from pi_jwm.formal_directed_graph_ops_v2 import direct_bearer_candidates
        from pi_jwm.formal_dual_graph_world_model_v2 import (
            FormalDirectedDynamicWorldModelConfig,
            FormalDirectedDynamicWorldModelV2,
        )

        torch.manual_seed(31)
        model = FormalDirectedDynamicWorldModelV2(
            FormalDirectedDynamicWorldModelConfig(
                hidden_dim=8,
                history_steps=3,
                horizon_steps=2,
            )
        ).eval()
        for name in ("physical_edge", "flow", "task"):
            torch.nn.init.zeros_(model.presence_heads[name].weight)
            torch.nn.init.zeros_(model.presence_heads[name].bias)
        batch = fake_formal_batch()

        with torch.no_grad():
            output = model(batch)

        historical_edge = batch["history"]["physical_edge_present"][:, -1].float()
        historical_flow = (
            batch["history"]["flow_present"][:, -1].float()
            * batch["static"]["flow_valid"].float()
        )
        torch.testing.assert_close(output["rollout_edge_weight"][:, 0], historical_edge)
        torch.testing.assert_close(output["rollout_flow_weight"][:, 0], historical_flow)
        edge_valid = (
            batch["static"]["physical_edge_endpoint_index"][..., 0] >= 0
        ).float()
        torch.testing.assert_close(
            output["rollout_edge_weight"][:, 1],
            0.5 * edge_valid,
        )
        expected_flow_step_two = 0.5 * batch["static"]["flow_valid"].float()
        torch.testing.assert_close(output["rollout_flow_weight"][:, 1], expected_flow_step_two)
        candidate = direct_bearer_candidates(
            batch["static"]["flow_endpoint_index"],
            batch["static"]["physical_edge_endpoint_index"],
            batch["static"]["flow_valid"].float(),
            edge_valid,
        )
        expected_cfe_step_two = (
            candidate
            * expected_flow_step_two.unsqueeze(-1)
            * output["rollout_edge_weight"][:, 1].unsqueeze(1)
        )
        torch.testing.assert_close(output["rollout_cfe_weight"][:, 1], expected_cfe_step_two)
        self.assertEqual(0.0, float(output["rollout_cfe_weight"][0, 1, 0, 1]))

    def test_future_presence_probabilities_change_the_rollout_main_path(self):
        from pi_jwm.formal_dual_graph_world_model_v2 import (
            FormalDirectedDynamicWorldModelConfig,
            FormalDirectedDynamicWorldModelV2,
        )

        torch.manual_seed(35)
        model = FormalDirectedDynamicWorldModelV2(
            FormalDirectedDynamicWorldModelConfig(
                hidden_dim=8,
                history_steps=3,
                horizon_steps=2,
            )
        ).eval()
        batch = fake_formal_batch()

        with torch.no_grad():
            for head in model.presence_heads.values():
                head.weight.zero_()
                head.bias.fill_(8.0)
            model.dag_edge_presence_head.weight.zero_()
            model.dag_edge_presence_head.bias.fill_(8.0)
            high_presence = model(batch)

            for head in model.presence_heads.values():
                head.bias.fill_(-8.0)
            model.dag_edge_presence_head.bias.fill_(-8.0)
            low_presence = model(batch)

        self.assertFalse(
            torch.allclose(
                high_presence["flow_state_mean"][:, 1],
                low_presence["flow_state_mean"][:, 1],
            )
        )
        self.assertFalse(
            torch.allclose(
                high_presence["task_state_mean"][:, 1],
                low_presence["task_state_mean"][:, 1],
            )
        )

    def test_complete_interface_declares_deterministic_v2_boundary(self):
        from pi_jwm.formal_dual_graph_world_model_v2 import (
            FormalDirectedDynamicWorldModelConfig,
            FormalDirectedDynamicWorldModelV2,
        )

        model = FormalDirectedDynamicWorldModelV2(
            FormalDirectedDynamicWorldModelConfig(
                hidden_dim=8,
                history_steps=3,
                horizon_steps=2,
                use_system_energy_head=True,
            )
        )
        output = model(fake_formal_batch())

        self.assertEqual("directed_dynamic_v2_1", model.model_version)
        self.assertEqual("deterministic", model.latent_dynamics)
        expected_shapes = {
            "node_state_mean": (2, 2, 4, 7),
            "physical_edge_state_mean": (2, 2, 3, 5),
            "flow_state_mean": (2, 2, 2, 5),
            "task_state_mean": (2, 2, 3, 8),
            "task_dag_state_mean": (2, 2, 3, 3),
            "task_lifecycle_logits": (2, 2, 3, 5),
            "dag_edge_presence_logits": (2, 2, 2),
            "uav_energy_delta_mean": (2, 2, 4),
        }
        for name, shape in expected_shapes.items():
            self.assertEqual(shape, tuple(output[name].shape), name)
        for value in output.values():
            self.assertTrue(torch.isfinite(value).all())

    def test_future_action_changes_v2_but_future_target_does_not(self):
        from pi_jwm.formal_dual_graph_world_model_v2 import (
            FormalDirectedDynamicWorldModelConfig,
            FormalDirectedDynamicWorldModelV2,
        )

        torch.manual_seed(37)
        model = FormalDirectedDynamicWorldModelV2(
            FormalDirectedDynamicWorldModelConfig(
                hidden_dim=8, history_steps=3, horizon_steps=2
            )
        ).eval()
        batch = fake_formal_batch()
        changed_action = copy.deepcopy(batch)
        changed_action["future_action"]["task_action"][:, 0, 0, 0] = 1.0
        changed_action["future_action"]["task_action_present"][:, 0, 0] = True
        changed_target = copy.deepcopy(batch)
        changed_target["target"]["node_state"].fill_(99999.0)

        with torch.no_grad():
            baseline = model(batch)["task_state_mean"]
            action_conditioned = model(changed_action)["task_state_mean"]
            target_changed = model(changed_target)["task_state_mean"]

        self.assertFalse(torch.allclose(baseline, action_conditioned))
        torch.testing.assert_close(baseline, target_changed)

    def test_v1_loss_accepts_v2_output_and_backpropagates(self):
        from pi_jwm.formal_dual_graph_world_model_v2 import (
            FormalDirectedDynamicWorldModelConfig,
            FormalDirectedDynamicWorldModelV2,
        )
        from pi_jwm.formal_world_model_loss_v1 import formal_world_model_loss

        batch = fake_formal_batch()
        batch["target"]["link_activity"] = batch["target"][
            "physical_edge_present"
        ].clone()
        batch["target"]["task_lifecycle_index"] = torch.full_like(
            batch["target"]["task_present"], 2, dtype=torch.long
        )
        model = FormalDirectedDynamicWorldModelV2(
            FormalDirectedDynamicWorldModelConfig(
                hidden_dim=8, history_steps=3, horizon_steps=2
            )
        )
        loss, components = formal_world_model_loss(
            model(batch), batch["target"], batch["static"]
        )
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(components["total_loss"]))
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))
        self.assertTrue(
            all(
                torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
                if parameter.grad is not None
            )
        )


if __name__ == "__main__":
    unittest.main()
