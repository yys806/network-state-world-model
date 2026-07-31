from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

import torch


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def fake_batch():
    torch.manual_seed(7)
    batch, history, horizon = 2, 3, 2
    counts = {"node": 4, "physical_edge": 5, "flow": 3, "task": 6}
    features = {"node": 7, "physical_edge": 5, "flow": 5, "task": 8}
    history_data = {}
    target = {}
    for name in counts:
        present = torch.zeros(batch, history, counts[name], dtype=torch.bool)
        present[..., : max(1, counts[name] - 1)] = True
        state = torch.randn(batch, history, counts[name], features[name]) * present.unsqueeze(-1)
        history_data[f"{name}_state"] = state
        history_data[f"{name}_present"] = present
        future_present = present[:, :horizon].clone()
        future_state = torch.randn(batch, horizon, counts[name], features[name]) * future_present.unsqueeze(-1)
        target[f"{name}_state"] = future_state
        target[f"{name}_present"] = future_present
    history_data["task_action"] = torch.zeros(batch, history, counts["task"], 5)
    history_data["task_action_present"] = torch.zeros(batch, history, counts["task"], dtype=torch.bool)
    static = {
        "node_kind_index": torch.tensor([[0, 1, 2, -1]] * batch),
        "physical_edge_endpoint_index": torch.tensor([[[0, 1], [1, 2], [2, 0], [0, 2], [-1, -1]]] * batch),
        "flow_valid": torch.tensor([[True, True, False]] * batch),
        "task_valid": torch.tensor([[True, True, True, True, True, False]] * batch),
    }
    return history_data, target, static


class AirFogSimSmokeModelV2Tests(unittest.TestCase):
    def test_forward_produces_all_dual_graph_rollout_shapes(self):
        from pi_jwm.airfogsim_smoke_model_v2 import MinimalDualGraphWorldModel

        history, _, _ = fake_batch()
        model = MinimalDualGraphWorldModel(hidden_dim=16, horizon_steps=2)
        output = model(history)

        self.assertEqual((2, 2, 4, 7), tuple(output["node_state"].shape))
        self.assertEqual((2, 2, 5, 5), tuple(output["physical_edge_state"].shape))
        self.assertEqual((2, 2, 3, 5), tuple(output["flow_state"].shape))
        self.assertEqual((2, 2, 6, 8), tuple(output["task_state"].shape))
        self.assertEqual((2, 2, 6), tuple(output["task_presence_logits"].shape))

    def test_masked_loss_ignores_padding_values_and_backpropagates(self):
        from pi_jwm.airfogsim_smoke_model_v2 import MinimalDualGraphWorldModel, dual_graph_world_model_loss

        history, target, static = fake_batch()
        model = MinimalDualGraphWorldModel(hidden_dim=16, horizon_steps=2)
        output = model(history)
        loss, metrics = dual_graph_world_model_loss(output, target, static)
        changed = copy.deepcopy(target)
        changed["task_state"] = changed["task_state"].clone()
        changed["task_state"][~changed["task_present"]] = 99999.0
        changed_loss, _ = dual_graph_world_model_loss(output, changed, static)

        self.assertTrue(torch.isfinite(loss))
        self.assertAlmostEqual(float(loss.detach()), float(changed_loss.detach()), places=6)
        self.assertIn("task_state_mae", metrics)
        loss.backward()
        self.assertTrue(any(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters()))

    def test_action_history_changes_rollout(self):
        from pi_jwm.airfogsim_smoke_model_v2 import MinimalDualGraphWorldModel

        history, _, _ = fake_batch()
        changed = {key: value.clone() for key, value in history.items()}
        changed["task_action"][:, -1, 0, 0] = 1.0
        changed["task_action_present"][:, -1, 0] = True
        model = MinimalDualGraphWorldModel(hidden_dim=16, horizon_steps=2)
        model.eval()
        with torch.no_grad():
            baseline = model(history)["task_state"]
            action_conditioned = model(changed)["task_state"]
        self.assertFalse(torch.allclose(baseline, action_conditioned))


if __name__ == "__main__":
    unittest.main()
