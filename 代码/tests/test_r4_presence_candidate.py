from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

import torch


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
TEST_ROOT = Path(__file__).resolve().parent
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from test_r3_world_model import make_batch


class R4SoftPresenceTests(unittest.TestCase):
    def _model(self):
        from pi_jwm.r4_module_registry import make_single_module_config
        from pi_jwm.r4_world_model import build_r4_world_model

        return build_r4_world_model(
            make_single_module_config(
                "presence",
                "soft_predicted_presence_v1",
                hidden_dim=8,
                history_steps=2,
            )
        )

    def test_soft_presence_executes_finite_long_rollout(self):
        model = self._model()
        output = model(make_batch(horizon=20), rollout_steps=20)
        self.assertEqual((1, 20, 8), tuple(output.predicted_belief.joint_latent.shape))
        self.assertTrue(torch.isfinite(output.predicted_belief.joint_latent).all())

    def test_predicted_presence_first_affects_the_next_rollout_step(self):
        torch.manual_seed(79)
        low = self._model()
        high = self._model()
        high.load_state_dict(low.state_dict())
        for model, bias in ((low, -20.0), (high, 20.0)):
            for name in (
                "physical_node_present",
                "physical_edge_present",
                "information_node_present",
                "information_edge_present",
            ):
                model.backend.presence_heads[name].bias.data.fill_(bias)

        batch = make_batch(horizon=2)
        low_output = low(batch, rollout_steps=2).predicted_belief.joint_latent
        high_output = high(batch, rollout_steps=2).predicted_belief.joint_latent
        torch.testing.assert_close(low_output[:, 0], high_output[:, 0])
        self.assertGreater(
            torch.max(torch.abs(low_output[:, 1] - high_output[:, 1])).item(),
            0.0,
        )

    def test_soft_presence_does_not_read_future_presence_targets(self):
        model = self._model()
        left = make_batch(horizon=5)
        right = copy.deepcopy(left)
        for key in (
            "physical_node_present",
            "physical_edge_present",
            "information_node_present",
            "information_edge_present",
        ):
            right.target[key] = ~right.target[key]
        torch.testing.assert_close(
            model(left, rollout_steps=5).predicted_belief.joint_latent,
            model(right, rollout_steps=5).predicted_belief.joint_latent,
        )


if __name__ == "__main__":
    unittest.main()
