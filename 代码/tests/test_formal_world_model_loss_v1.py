from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

import torch


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from test_formal_dual_graph_world_model_v1 import fake_formal_batch


def _complete_target(batch: dict) -> dict:
    target = batch["target"]
    target["link_activity"] = target["physical_edge_present"].clone()
    target["task_lifecycle_index"] = torch.full_like(
        target["task_present"], 2, dtype=torch.long
    )
    return target


class FormalWorldModelLossV1Tests(unittest.TestCase):
    def test_complete_masked_objective_is_finite_and_backpropagates(self):
        from pi_jwm.formal_dual_graph_world_model_v1 import (
            FormalDualGraphWorldModel,
            FormalWorldModelConfig,
        )
        from pi_jwm.formal_world_model_loss_v1 import formal_world_model_loss

        batch = fake_formal_batch()
        _complete_target(batch)
        model = FormalDualGraphWorldModel(
            FormalWorldModelConfig(mode="coupled_dual_gnn", hidden_dim=8, history_steps=3, horizon_steps=2)
        )
        loss, components = formal_world_model_loss(model(batch), batch["target"], batch["static"])
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        for key in (
            "node_state_nll",
            "physical_edge_state_nll",
            "flow_state_nll",
            "task_state_nll",
            "task_dag_state_nll",
            "link_activity_bce",
            "task_lifecycle_ce",
            "dag_release_bce",
            "dag_edge_presence_bce",
            "total_loss",
        ):
            self.assertIn(key, components)
            self.assertTrue(torch.isfinite(components[key]))
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_padding_target_values_do_not_change_continuous_loss(self):
        from pi_jwm.formal_dual_graph_world_model_v1 import (
            FormalDualGraphWorldModel,
            FormalWorldModelConfig,
        )
        from pi_jwm.formal_world_model_loss_v1 import formal_world_model_loss

        batch = fake_formal_batch()
        _complete_target(batch)
        model = FormalDualGraphWorldModel(
            FormalWorldModelConfig(mode="pooled_gru", hidden_dim=8, history_steps=3, horizon_steps=2)
        )
        prediction = model(batch)
        original, _ = formal_world_model_loss(prediction, batch["target"], batch["static"])
        changed = copy.deepcopy(batch["target"])
        for name in ("node", "physical_edge", "flow", "task"):
            changed[f"{name}_state"][~changed[f"{name}_present"]] = 99999.0
        changed["task_dag_state"][~changed["task_dag_state_present"]] = 99999.0
        changed_loss, _ = formal_world_model_loss(prediction, changed, batch["static"])

        self.assertAlmostEqual(float(original.detach()), float(changed_loss.detach()), places=5)

    def test_empty_target_component_has_zero_loss_and_finite_total(self):
        from pi_jwm.formal_dual_graph_world_model_v1 import (
            FormalDualGraphWorldModel,
            FormalWorldModelConfig,
        )
        from pi_jwm.formal_world_model_loss_v1 import formal_world_model_loss

        batch = fake_formal_batch()
        _complete_target(batch)
        batch["target"]["flow_present"].zero_()
        batch["static"]["flow_valid"].zero_()
        prediction = FormalDualGraphWorldModel(
            FormalWorldModelConfig(mode="pooled_gru", hidden_dim=8, history_steps=3, horizon_steps=2)
        )(batch)
        loss, components = formal_world_model_loss(prediction, batch["target"], batch["static"])

        self.assertEqual(0.0, float(components["flow_state_nll"]))
        self.assertEqual(0, int(components["flow_state_valid_count"]))
        self.assertTrue(torch.isfinite(loss))

    def test_training_class_weights_reject_non_train_samples(self):
        from pi_jwm.formal_world_model_loss_v1 import compute_training_class_weights

        batch = fake_formal_batch()
        _complete_target(batch)
        sample = {
            "split": "validation",
            "target": {key: value[0] for key, value in batch["target"].items()},
            "static": {key: value[0] for key, value in batch["static"].items()},
        }
        with self.assertRaisesRegex(ValueError, "train"):
            compute_training_class_weights([sample])

    def test_training_class_weights_are_computed_from_masked_train_labels(self):
        from pi_jwm.formal_world_model_loss_v1 import compute_training_class_weights

        batch = fake_formal_batch()
        _complete_target(batch)
        sample = {
            "split": "train",
            "target": {key: value[0] for key, value in batch["target"].items()},
            "static": {key: value[0] for key, value in batch["static"].items()},
        }
        weights = compute_training_class_weights([sample], max_pos_weight=20.0)

        self.assertEqual("train", weights["source_split"])
        self.assertIn("link_activity", weights["pos_weight"])
        self.assertIn("flow_present", weights["pos_weight"])
        self.assertIn("task_present", weights["pos_weight"])
        self.assertGreaterEqual(weights["pos_weight"]["flow_present"], 1.0)
        self.assertLessEqual(weights["pos_weight"]["flow_present"], 20.0)


if __name__ == "__main__":
    unittest.main()
