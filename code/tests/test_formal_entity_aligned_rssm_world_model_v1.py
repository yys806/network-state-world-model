from __future__ import annotations

import copy
import unittest

import torch

from test_formal_dual_graph_world_model_v1 import fake_formal_batch
from test_formal_world_model_loss_v1 import _complete_target


def _add_motion(batch):
    node = batch["history"]["node_state"]
    batch["history"]["node_motion_state"] = torch.zeros(
        (*node.shape[:-1], 6), dtype=node.dtype
    )
    batch["history"]["node_motion_mask"] = torch.ones(
        (*node.shape[:-1], 6), dtype=torch.bool
    )
    batch["history"]["node_motion_state"][..., 0] = 0.5
    return batch


class FormalEntityAlignedRSSMWorldModelTests(unittest.TestCase):
    def _model(self, *, zero_init=False):
        from pi_jwm.formal_entity_aligned_rssm_world_model_v1 import (
            FormalEntityAlignedRSSMConfig,
            FormalEntityAlignedRSSMWorldModel,
        )

        return FormalEntityAlignedRSSMWorldModel(
            FormalEntityAlignedRSSMConfig(
                hidden_dim=8,
                stochastic_dim=4,
                history_steps=3,
                horizon_steps=2,
                overshooting_distance=2,
                zero_init_residual_state_heads=zero_init,
                slot_seconds=1.0,
                node_position_scale=(1.0, 1.0, 1.0),
            )
        )

    def test_eval_is_prior_only_and_keeps_entity_latents(self):
        batch = _add_motion(fake_formal_batch())
        _complete_target(batch)
        changed = copy.deepcopy(batch)
        changed["target"]["node_state"] += 10.0
        model = self._model().eval()

        with torch.no_grad():
            first = model(batch)
            second = model(changed)

        self.assertTrue(first["rssm_deployment_prior_only"])
        self.assertEqual((2, 2, 4, 4), tuple(first["rssm_node_rollout_prior_mean"].shape))
        self.assertEqual((2, 2, 3), tuple(first["rssm_link_activity_correction"].shape))
        torch.testing.assert_close(first["node_state_mean"], second["node_state_mean"])
        self.assertFalse(any(key.startswith("training_") for key in first))

    def test_posterior_target_change_remains_entity_aligned(self):
        batch = _add_motion(fake_formal_batch())
        _complete_target(batch)
        changed = copy.deepcopy(batch)
        changed["target"]["node_state"][:, :, 0] += 7.0
        model = self._model().train()

        torch.manual_seed(19)
        first = model(batch)
        torch.manual_seed(19)
        second = model(changed)

        delta = (
            first["rssm_node_posterior_mean"]
            - second["rssm_node_posterior_mean"]
        ).abs().sum(dim=-1)
        self.assertTrue(torch.all(delta[:, :, 0] > 0))
        torch.testing.assert_close(delta[:, :, 1:], torch.zeros_like(delta[:, :, 1:]))
        torch.testing.assert_close(first["node_state_mean"], second["node_state_mean"])

    def test_link_correction_is_per_edge(self):
        batch = _add_motion(fake_formal_batch())
        _complete_target(batch)
        model = self._model().eval()

        with torch.no_grad():
            prediction = model(batch)

        correction = prediction["rssm_link_activity_correction"]
        self.assertEqual(tuple(prediction["link_activity_logits"].shape), tuple(correction.shape))
        self.assertGreater(float((correction[..., 0] - correction[..., 1]).abs().max()), 0.0)

    def test_zero_initialization_and_full_loss_gradients(self):
        from pi_jwm.formal_world_model_loss_v1 import FormalLossWeights, formal_world_model_loss

        batch = _add_motion(fake_formal_batch())
        _complete_target(batch)
        model = self._model(zero_init=True).train()
        for head in model.prior_state_heads.values():
            self.assertTrue(torch.equal(head.weight, torch.zeros_like(head.weight)))
        prediction = model(batch)
        loss, components = formal_world_model_loss(
            prediction,
            batch["target"],
            batch["static"],
            weights=FormalLossWeights(
                rssm_kl=0.1,
                rssm_teacher_reconstruction=0.5,
                rssm_overshooting=0.1,
            ),
        )
        loss.backward()

        for key in ("rssm_kl", "rssm_teacher_reconstruction", "rssm_overshooting"):
            self.assertIn(key, components)
        for name in (
            "transitions.node.weight_ih",
            "priors.physical_edge.weight",
            "posteriors.task.weight",
            "teacher_state_heads.flow.weight",
        ):
            gradient = dict(model.named_parameters())[name].grad
            self.assertIsNotNone(gradient, name)
            self.assertGreater(float(gradient.abs().sum()), 0.0, name)

    def test_missing_motion_contract_is_rejected(self):
        batch = fake_formal_batch()
        _complete_target(batch)
        with self.assertRaisesRegex(ValueError, "node_motion"):
            self._model()(batch)

    def test_invalid_entities_receive_zero_rssm_corrections(self):
        batch = _add_motion(fake_formal_batch())
        _complete_target(batch)
        model = self._model().eval()
        with torch.no_grad():
            prediction = model(batch)

        self.assertTrue(torch.equal(
            prediction["rssm_node_state_correction"][:, :, -1],
            torch.zeros_like(prediction["rssm_node_state_correction"][:, :, -1]),
        ))
        self.assertTrue(torch.equal(
            prediction["rssm_link_activity_correction"][:, :, -1],
            torch.zeros_like(prediction["rssm_link_activity_correction"][:, :, -1]),
        ))


if __name__ == "__main__":
    unittest.main()
