from __future__ import annotations

import copy
import math
import sys
import unittest
from pathlib import Path

import torch


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from test_formal_dual_graph_world_model_v1 import fake_formal_batch


def _activity_batch(*, observed: bool = True) -> dict:
    batch = fake_formal_batch()
    history = batch["history"]
    history["physical_edge_present"][..., -1] = True
    horizon = batch["future_action"]["task_action"].shape[1]
    activity = torch.zeros_like(history["physical_edge_present"])
    activity[..., 0] = True
    history["aggregate_link_activity"] = activity
    history["aggregate_link_activity_mask"] = torch.full_like(activity, observed)
    batch["target"]["aggregate_link_activity"] = activity[:, -horizon:].clone()
    batch["target"]["aggregate_link_activity_mask"] = torch.full_like(
        batch["target"]["aggregate_link_activity"], observed
    )
    return batch


class FormalLinkActivityPersistenceResidualV1Tests(unittest.TestCase):
    def _model(self, **kwargs):
        from pi_jwm.formal_dual_graph_world_model_v1 import (
            FormalDualGraphWorldModel,
            FormalWorldModelConfig,
        )

        values = dict(
            mode="coupled_dual_gnn",
            hidden_dim=8,
            history_steps=3,
            horizon_steps=2,
            link_activity_method="persistence_residual_v1",
            link_activity_pos_weight=50.0,
            link_activity_missing_history_prior=0.25,
        )
        values.update(kwargs)
        config = FormalWorldModelConfig(**values)
        return FormalDualGraphWorldModel(config)

    def test_zero_delta_preserves_observed_history_persistence_logit(self):
        model = self._model()
        torch.nn.init.zeros_(model.link_activity_head.weight)
        torch.nn.init.zeros_(model.link_activity_head.bias)
        with torch.no_grad():
            output = model(_activity_batch(observed=True))
        expected = 20.0
        torch.testing.assert_close(
            output["link_activity_logits"][..., 0],
            torch.full_like(output["link_activity_logits"][..., 0], expected),
        )

    def test_h2_uses_previous_prediction_and_not_future_target(self):
        batch = _activity_batch(observed=True)
        changed = copy.deepcopy(batch)
        changed["target"]["aggregate_link_activity"].fill_(False)
        changed["target"]["aggregate_link_activity_mask"].fill_(False)
        model = self._model()
        torch.nn.init.zeros_(model.link_activity_head.weight)
        torch.nn.init.constant_(model.link_activity_head.bias, 1.0)
        with torch.no_grad():
            first = model(batch)["link_activity_logits"]
            second = model(changed)["link_activity_logits"]
        torch.testing.assert_close(first, second)
        torch.testing.assert_close(first[:, 1] - first[:, 0], torch.ones_like(first[:, 0]))

    def test_raw_logit_is_unweighted_event_logit_plus_pos_weight_log(self):
        model = self._model()
        torch.nn.init.zeros_(model.link_activity_head.weight)
        torch.nn.init.zeros_(model.link_activity_head.bias)
        with torch.no_grad():
            raw = model(_activity_batch(observed=True))["link_activity_logits"]
        corrected = raw - math.log(50.0)
        torch.testing.assert_close(
            corrected[..., 0],
            torch.full_like(corrected[..., 0], 20.0 - math.log(50.0)),
        )

    def test_missing_history_uses_train_only_prior(self):
        model = self._model(link_activity_missing_history_prior=0.25)
        torch.nn.init.zeros_(model.link_activity_head.weight)
        torch.nn.init.zeros_(model.link_activity_head.bias)
        with torch.no_grad():
            raw = model(_activity_batch(observed=False))["link_activity_logits"]
        expected = math.log(0.25 / 0.75) + math.log(50.0)
        torch.testing.assert_close(
            raw[..., 0], torch.full_like(raw[..., 0], expected), atol=1e-6, rtol=0
        )

    def test_default_absolute_method_keeps_existing_head_semantics(self):
        from pi_jwm.formal_dual_graph_world_model_v1 import (
            FormalDualGraphWorldModel,
            FormalWorldModelConfig,
        )

        config = FormalWorldModelConfig(
            mode="coupled_dual_gnn", hidden_dim=8, history_steps=3, horizon_steps=2
        )
        model = FormalDualGraphWorldModel(config)
        self.assertEqual("absolute_v1", config.link_activity_method)
        batch = fake_formal_batch()
        with torch.no_grad():
            output = model(batch)
        self.assertEqual((2, 2, 3), tuple(output["link_activity_logits"].shape))


if __name__ == "__main__":
    unittest.main()
