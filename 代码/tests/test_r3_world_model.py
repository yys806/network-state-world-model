from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

import torch


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def make_batch(*, horizon: int = 20, action_value: float = 0.0):
    from pi_jwm.r3_preflight_data import ExplicitStateBatch

    batch, history, nodes, edges, flows, tasks = 1, 2, 3, 4, 2, 2
    generator = torch.Generator().manual_seed(11)

    def state(shape):
        return torch.randn(shape, generator=generator)

    history_values = {
        "physical_node_state": state((batch, history, nodes, 9)),
        "physical_node_feature_mask": torch.ones(batch, history, nodes, 9, dtype=torch.bool),
        "physical_node_present": torch.ones(batch, history, nodes, dtype=torch.bool),
        "physical_edge_state": state((batch, history, edges, 7)),
        "physical_edge_feature_mask": torch.ones(batch, history, edges, 7, dtype=torch.bool),
        "physical_edge_present": torch.ones(batch, history, edges, dtype=torch.bool),
        "information_node_state": state((batch, history, nodes, 7)),
        "information_node_feature_mask": torch.ones(batch, history, nodes, 7, dtype=torch.bool),
        "information_node_present": torch.ones(batch, history, nodes, dtype=torch.bool),
        "information_edge_state": state((batch, history, edges, 18)),
        "information_edge_feature_mask": torch.ones(batch, history, edges, 18, dtype=torch.bool),
        "information_edge_present": torch.ones(batch, history, edges, dtype=torch.bool),
        "data_flow_state": state((batch, history, flows, 5)),
        "data_flow_present": torch.ones(batch, history, flows, dtype=torch.bool),
        "task_state": state((batch, history, tasks, 8)),
        "task_present": torch.ones(batch, history, tasks, dtype=torch.bool),
        "task_lifecycle_index": torch.ones(batch, history, tasks, dtype=torch.long),
        "task_dag_state": state((batch, history, tasks, 3)),
        "task_dag_state_present": torch.ones(batch, history, tasks, dtype=torch.bool),
    }
    target = {
        key: (
            value[:, -1:].expand(batch, horizon, *value.shape[2:]).clone()
            if value.ndim >= 3
            else value
        )
        for key, value in history_values.items()
    }
    future_action = {
        "task_action": torch.full((batch, horizon, tasks, 8), action_value),
        "task_action_present": torch.ones(batch, horizon, tasks, dtype=torch.bool),
        "task_action_information_node_index": torch.tensor(
            [[[[0, 1, 1, 0], [1, 2, 2, 1]]]], dtype=torch.long
        ).expand(batch, horizon, tasks, 4).clone(),
    }
    history_action = {
        "task_action": torch.zeros(batch, history, tasks, 8),
        "task_action_present": torch.ones(batch, history, tasks, dtype=torch.bool),
        "task_action_information_node_index": torch.tensor(
            [[[[0, 1, 1, 0], [1, 2, 2, 1]]]], dtype=torch.long
        ).expand(batch, history, tasks, 4).clone(),
    }
    static = {
        "physical_node_kind_index": torch.tensor([[0, 1, 2]], dtype=torch.long),
        "physical_edge_endpoint_index": torch.tensor(
            [[[0, 1], [1, 2], [2, 0], [0, 2]]], dtype=torch.long
        ),
        "information_edge_endpoint_index": torch.tensor(
            [[[0, 1], [1, 2], [2, 0], [0, 2]]], dtype=torch.long
        ),
        "cip_agent_node_index": torch.tensor([[0, 1, 2]], dtype=torch.long),
        "cep_information_to_physical_edge_index": torch.tensor([[0, 1, 2, 3]], dtype=torch.long),
        "cfl_information_edge_index": torch.tensor([[0, 2]], dtype=torch.long),
        "data_flow_valid": torch.ones(batch, flows, dtype=torch.bool),
        "task_valid": torch.ones(batch, tasks, dtype=torch.bool),
    }
    return ExplicitStateBatch(
        history=history_values,
        history_action=history_action,
        future_action=future_action,
        target=target,
        static=static,
        metadata={"trajectory_id": "fixture", "split": "train"},
    )


class R3WorldModelTests(unittest.TestCase):
    def test_reference_model_returns_explicit_and_belief_sequences(self):
        from pi_jwm.r3_world_model import R3ReferenceConfig, R3ReferenceWorldModel

        torch.manual_seed(3)
        model = R3ReferenceWorldModel(R3ReferenceConfig(hidden_dim=8, history_steps=2))
        output = model(make_batch(horizon=5), rollout_steps=5)
        self.assertEqual((1, 5, 3, 9), tuple(output.predicted_explicit["physical_node_state"].shape))
        self.assertEqual((1, 5, 4, 18), tuple(output.predicted_explicit["information_edge_state"].shape))
        self.assertEqual((1, 5, 8), tuple(output.predicted_belief.joint_latent.shape))
        self.assertEqual((1, 5, 3, 8), tuple(output.predicted_belief.physical_latent.shape))
        self.assertEqual((1, 5, 3, 8), tuple(output.predicted_belief.information_latent.shape))
        self.assertEqual((1, 5, 8), tuple(output.predicted_belief.business_latent.shape))

    def test_cip_cep_cfl_change_coupled_belief_but_not_no_coupling_control(self):
        from pi_jwm.r3_world_model import R3ReferenceConfig, R3ReferenceWorldModel

        torch.manual_seed(5)
        coupled = R3ReferenceWorldModel(
            R3ReferenceConfig(hidden_dim=8, history_steps=2, use_cross_graph_coupling=True)
        )
        control = R3ReferenceWorldModel(
            R3ReferenceConfig(hidden_dim=8, history_steps=2, use_cross_graph_coupling=False)
        )
        control.load_state_dict(coupled.state_dict())
        original = make_batch(horizon=1)
        changed = copy.deepcopy(original)
        changed.static["cip_agent_node_index"] = torch.tensor([[2, 0, 1]])
        changed.static["cep_information_to_physical_edge_index"] = torch.tensor([[3, 2, 1, 0]])
        changed.static["cfl_information_edge_index"] = torch.tensor([[3, 1]])

        coupled_left = coupled(original, rollout_steps=1).predicted_belief.joint_latent
        coupled_right = coupled(changed, rollout_steps=1).predicted_belief.joint_latent
        control_left = control(original, rollout_steps=1).predicted_belief.joint_latent
        control_right = control(changed, rollout_steps=1).predicted_belief.joint_latent
        self.assertGreater(torch.max(torch.abs(coupled_left - coupled_right)).item(), 0.0)
        torch.testing.assert_close(control_left, control_right)

    def test_future_action_changes_next_belief(self):
        from pi_jwm.r3_world_model import R3ReferenceConfig, R3ReferenceWorldModel

        torch.manual_seed(7)
        model = R3ReferenceWorldModel(R3ReferenceConfig(hidden_dim=8, history_steps=2))
        left = model(make_batch(horizon=1, action_value=0.0), rollout_steps=1)
        right = model(make_batch(horizon=1, action_value=1.0), rollout_steps=1)
        self.assertGreater(
            torch.max(
                torch.abs(
                    left.predicted_belief.joint_latent
                    - right.predicted_belief.joint_latent
                )
            ).item(),
            0.0,
        )

    def test_historical_action_changes_inferred_belief(self):
        from pi_jwm.r3_world_model import R3ReferenceConfig, R3ReferenceWorldModel

        torch.manual_seed(17)
        model = R3ReferenceWorldModel(R3ReferenceConfig(hidden_dim=8, history_steps=2))
        left = make_batch(horizon=1)
        right = copy.deepcopy(left)
        right.history_action["task_action"].fill_(1.0)
        left_belief = model.infer_belief(left).joint
        right_belief = model.infer_belief(right).joint
        self.assertGreater(torch.max(torch.abs(left_belief - right_belief)).item(), 0.0)

    def test_twenty_step_rollout_does_not_read_future_target(self):
        from pi_jwm.r3_world_model import R3ReferenceConfig, R3ReferenceWorldModel

        torch.manual_seed(13)
        model = R3ReferenceWorldModel(R3ReferenceConfig(hidden_dim=8, history_steps=2))
        first_batch = make_batch(horizon=20)
        second_batch = copy.deepcopy(first_batch)
        for key, value in second_batch.target.items():
            if torch.is_floating_point(value):
                second_batch.target[key] = torch.randn_like(value) * 1000.0
        first = model(first_batch, rollout_steps=20)
        second = model(second_batch, rollout_steps=20)
        torch.testing.assert_close(
            first.predicted_belief.joint_latent,
            second.predicted_belief.joint_latent,
        )
        self.assertTrue(torch.isfinite(first.predicted_belief.joint_latent).all())
        self.assertEqual(20, first.predicted_belief.joint_latent.shape[1])

    def test_unknown_component_name_is_rejected(self):
        from pi_jwm.r3_world_model import R3ReferenceConfig, R3ReferenceWorldModel

        with self.assertRaisesRegex(ValueError, "field_encoder"):
            R3ReferenceWorldModel(
                R3ReferenceConfig(
                    hidden_dim=8,
                    history_steps=2,
                    field_encoder="unimplemented_jepa_placeholder",
                )
            )


if __name__ == "__main__":
    unittest.main()
