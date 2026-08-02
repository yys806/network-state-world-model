from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

import torch


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def fake_formal_batch() -> dict:
    torch.manual_seed(11)
    batch, history, horizon = 2, 3, 2
    counts = {"node": 4, "physical_edge": 3, "flow": 2, "task": 3}
    features = {"node": 7, "physical_edge": 5, "flow": 5, "task": 8}
    history_data = {}
    target = {}
    for name, count in counts.items():
        present = torch.ones(batch, history, count, dtype=torch.bool)
        present[..., -1] = False
        history_data[f"{name}_state"] = (
            torch.randn(batch, history, count, features[name]) * present.unsqueeze(-1)
        )
        history_data[f"{name}_present"] = present
        target[f"{name}_state"] = torch.randn(batch, horizon, count, features[name])
        target[f"{name}_present"] = present[:, :horizon].clone()
    history_data["task_lifecycle_index"] = torch.zeros(batch, history, counts["task"], dtype=torch.long)
    history_data["task_action"] = torch.zeros(batch, history, counts["task"], 8)
    history_data["task_action_present"] = torch.zeros(batch, history, counts["task"], dtype=torch.bool)
    history_data["task_node_index"] = torch.tensor(
        [[[[0, 1, -1, -1], [1, 2, -1, -1], [-1, -1, -1, -1]]] * history] * batch
    )
    history_data["task_action_node_index"] = history_data["task_node_index"].clone()
    history_data["flow_bearer_mask"] = torch.tensor(
        [[[[True, False, False], [False, True, False]]] * history] * batch
    )
    history_data["flow_bearer_edge_index"] = torch.zeros(batch, history, counts["flow"], dtype=torch.long)
    history_data["task_dag_state"] = torch.randn(batch, history, counts["task"], 3)
    history_data["task_dag_state_present"] = history_data["task_present"].clone()
    history_data["dag_edge_present"] = torch.tensor([[[True, False]] * history] * batch)

    future_action = {
        "task_action": torch.zeros(batch, horizon, counts["task"], 8),
        "task_action_present": torch.zeros(batch, horizon, counts["task"], dtype=torch.bool),
        "task_action_node_index": torch.tensor(
            [[[[0, 1, -1, -1], [1, 2, -1, -1], [-1, -1, -1, -1]]] * horizon] * batch
        ),
    }
    static = {
        "node_kind_index": torch.tensor([[0, 1, 2, -1]] * batch),
        "physical_edge_endpoint_index": torch.tensor(
            [[[0, 1], [1, 2], [-1, -1]]] * batch
        ),
        "flow_endpoint_index": torch.tensor([[[0, 1], [1, 2]]] * batch),
        "flow_valid": torch.tensor([[True, False]] * batch),
        "task_valid": torch.tensor([[True, True, False]] * batch),
        "dag_edge_index": torch.tensor([[[0, 1], [1, -1]]] * batch),
        "dag_edge_valid": torch.tensor([[True, False]] * batch),
        "agent_node_index": torch.tensor([[0, 1, 2, -1]] * batch),
    }
    target["task_dag_state"] = torch.randn(batch, horizon, counts["task"], 3)
    target["task_dag_state_present"] = target["task_present"].clone()
    target["dag_edge_present"] = torch.tensor([[[True, False]] * horizon] * batch)
    return {
        "history": history_data,
        "future_action": future_action,
        "static": static,
        "target": target,
    }


class FormalDualGraphWorldModelV1Tests(unittest.TestCase):
    def test_all_learning_modes_return_the_same_complete_interface(self):
        from pi_jwm.formal_dual_graph_world_model_v1 import (
            FormalDualGraphWorldModel,
            FormalWorldModelConfig,
        )

        batch = fake_formal_batch()
        for mode in ("pooled_gru", "independent_dual_gnn", "coupled_dual_gnn"):
            with self.subTest(mode=mode):
                model = FormalDualGraphWorldModel(
                    FormalWorldModelConfig(mode=mode, hidden_dim=8, history_steps=3, horizon_steps=2)
                )
                output = model(batch)
                self.assertEqual((2, 2, 4, 7), tuple(output["node_state_mean"].shape))
                self.assertEqual((2, 2, 3, 5), tuple(output["physical_edge_state_mean"].shape))
                self.assertEqual((2, 2, 2, 5), tuple(output["flow_state_mean"].shape))
                self.assertEqual((2, 2, 3, 8), tuple(output["task_state_mean"].shape))
                self.assertEqual((2, 2, 3, 3), tuple(output["task_dag_state_mean"].shape))
                self.assertEqual((2, 2, 3), tuple(output["task_lifecycle_logits"].shape[:3]))
                self.assertEqual(5, output["task_lifecycle_logits"].shape[-1])
                self.assertEqual((2, 2, 2), tuple(output["dag_edge_presence_logits"].shape))
                self.assertEqual((2, 2, 3), tuple(output["dag_release_logits"].shape))
                for value in output.values():
                    self.assertTrue(torch.isfinite(value).all(), mode)
                for name in ("node", "physical_edge", "flow", "task", "task_dag"):
                    log_variance = output[f"{name}_state_log_variance"]
                    self.assertGreaterEqual(float(log_variance.detach().min()), -8.0)
                    self.assertLessEqual(float(log_variance.detach().max()), 5.0)

    def test_residual_state_prediction_anchors_zero_delta_to_last_observation(self):
        from pi_jwm.formal_dual_graph_world_model_v1 import (
            FormalDualGraphWorldModel,
            FormalWorldModelConfig,
        )

        batch = fake_formal_batch()
        residual = FormalDualGraphWorldModel(
            FormalWorldModelConfig(
                mode="coupled_dual_gnn",
                hidden_dim=8,
                history_steps=3,
                horizon_steps=2,
                residual_state_prediction=True,
            )
        )
        absolute = FormalDualGraphWorldModel(
            FormalWorldModelConfig(
                mode="coupled_dual_gnn",
                hidden_dim=8,
                history_steps=3,
                horizon_steps=2,
            )
        )
        for model in (residual, absolute):
            for head in model.state_heads.values():
                torch.nn.init.zeros_(head.mean.weight)
                torch.nn.init.zeros_(head.mean.bias)
            torch.nn.init.zeros_(model.dag_state_head.mean.weight)
            torch.nn.init.zeros_(model.dag_state_head.mean.bias)
            model.eval()

        with torch.no_grad():
            residual_output = residual(batch)
            absolute_output = absolute(batch)

        for name in ("node", "physical_edge", "flow", "task"):
            expected = batch["history"][f"{name}_state"][:, -1:].expand(
                -1, 2, -1, -1
            )
            torch.testing.assert_close(residual_output[f"{name}_state_mean"], expected)
            torch.testing.assert_close(
                absolute_output[f"{name}_state_mean"],
                torch.zeros_like(absolute_output[f"{name}_state_mean"]),
            )
        dag_expected = batch["history"]["task_dag_state"][:, -1:].expand(-1, 2, -1, -1)
        torch.testing.assert_close(residual_output["task_dag_state_mean"], dag_expected)
        torch.testing.assert_close(
            absolute_output["task_dag_state_mean"],
            torch.zeros_like(absolute_output["task_dag_state_mean"]),
        )

    def test_optional_uav_energy_head_returns_nonnegative_node_aligned_distribution(self):
        from pi_jwm.formal_dual_graph_world_model_v1 import (
            FormalDualGraphWorldModel,
            FormalWorldModelConfig,
        )

        batch = fake_formal_batch()
        model = FormalDualGraphWorldModel(
            FormalWorldModelConfig(
                mode="coupled_dual_gnn",
                hidden_dim=8,
                history_steps=3,
                horizon_steps=2,
                use_system_energy_head=True,
            )
        )
        output = model(batch)

        self.assertEqual((2, 2, 4), tuple(output["uav_energy_delta_mean"].shape))
        self.assertEqual((2, 2, 4), tuple(output["uav_energy_delta_log_variance"].shape))
        self.assertTrue(torch.all(output["uav_energy_delta_mean"] >= 0))
        self.assertTrue(torch.isfinite(output["uav_energy_delta_mean"]).all())

    def test_future_action_changes_rollout_but_future_target_does_not(self):
        from pi_jwm.formal_dual_graph_world_model_v1 import (
            FormalDualGraphWorldModel,
            FormalWorldModelConfig,
        )

        batch = fake_formal_batch()
        model = FormalDualGraphWorldModel(
            FormalWorldModelConfig(mode="coupled_dual_gnn", hidden_dim=8, history_steps=3, horizon_steps=2)
        )
        model.eval()
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

    def test_forward_backpropagates_through_graph_and_rollout(self):
        from pi_jwm.formal_dual_graph_world_model_v1 import (
            FormalDualGraphWorldModel,
            FormalWorldModelConfig,
        )

        model = FormalDualGraphWorldModel(
            FormalWorldModelConfig(mode="coupled_dual_gnn", hidden_dim=8, history_steps=3, horizon_steps=2)
        )
        output = model(fake_formal_batch())
        loss = sum(value.float().square().mean() for value in output.values())
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))
        self.assertTrue(
            all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None)
        )

    def test_mode_and_ablation_switches_define_effective_coupling(self):
        from pi_jwm.formal_dual_graph_world_model_v1 import (
            FormalDualGraphWorldModel,
            FormalWorldModelConfig,
        )

        independent = FormalDualGraphWorldModel(
            FormalWorldModelConfig(mode="independent_dual_gnn", hidden_dim=8, history_steps=3, horizon_steps=2)
        )
        coupled = FormalDualGraphWorldModel(
            FormalWorldModelConfig(mode="coupled_dual_gnn", hidden_dim=8, history_steps=3, horizon_steps=2)
        )
        no_cross = FormalDualGraphWorldModel(
            FormalWorldModelConfig(
                mode="coupled_dual_gnn",
                hidden_dim=8,
                history_steps=3,
                horizon_steps=2,
                use_cross_coupling=False,
            )
        )
        self.assertFalse(independent.cross_coupling_enabled)
        self.assertTrue(coupled.cross_coupling_enabled)
        self.assertFalse(no_cross.cross_coupling_enabled)

        for config in (
            FormalWorldModelConfig(mode="coupled_dual_gnn", hidden_dim=8, history_steps=3, horizon_steps=2, physical_only=True),
            FormalWorldModelConfig(mode="coupled_dual_gnn", hidden_dim=8, history_steps=3, horizon_steps=2, information_only=True),
            FormalWorldModelConfig(mode="coupled_dual_gnn", hidden_dim=8, history_steps=3, horizon_steps=2, use_dag=False),
            FormalWorldModelConfig(mode="coupled_dual_gnn", hidden_dim=8, history_steps=3, horizon_steps=2, use_cpu_action=False),
        ):
            with self.subTest(config=config):
                output = FormalDualGraphWorldModel(config)(fake_formal_batch())
                self.assertTrue(all(torch.isfinite(value).all() for value in output.values()))


if __name__ == "__main__":
    unittest.main()
