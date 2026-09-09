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
from test_formal_world_model_loss_v1 import _complete_target


class FormalCompleteRSSMWorldModelV1Tests(unittest.TestCase):
    def _model(self):
        from pi_jwm.formal_complete_rssm_world_model_v1 import (
            FormalCompleteRSSMConfig,
            FormalCompleteRSSMWorldModel,
        )

        return FormalCompleteRSSMWorldModel(
            FormalCompleteRSSMConfig(
                hidden_dim=8,
                stochastic_dim=5,
                history_steps=3,
                horizon_steps=2,
                overshooting_distance=2,
            )
        )

    def test_zero_init_contract_covers_rssm_continuous_residual_decoders(self):
        from pi_jwm.formal_complete_rssm_world_model_v1 import (
            FormalCompleteRSSMConfig,
            FormalCompleteRSSMWorldModel,
        )

        model = FormalCompleteRSSMWorldModel(
            FormalCompleteRSSMConfig(
                hidden_dim=8,
                stochastic_dim=5,
                history_steps=3,
                horizon_steps=2,
                zero_init_residual_state_heads=True,
            )
        )
        heads = [
            *model.rssm_prior_state_heads.values(),
            *model.rssm_teacher_state_heads.values(),
            model.rssm_prior_dag_head,
            model.rssm_teacher_dag_head,
        ]
        for head in heads:
            self.assertTrue(torch.equal(head.weight, torch.zeros_like(head.weight)))
            self.assertTrue(torch.equal(head.bias, torch.zeros_like(head.bias)))

    def test_eval_is_prior_only_and_does_not_read_future_target(self):
        batch = fake_formal_batch()
        _complete_target(batch)
        changed = copy.deepcopy(batch)
        for key, value in changed["target"].items():
            if torch.is_floating_point(value):
                changed["target"][key] = value + 17.0
            elif value.dtype == torch.bool:
                changed["target"][key] = ~value
        model = self._model().eval()

        with torch.no_grad():
            first = model(batch)
            second = model(changed)

        self.assertTrue(first["rssm_deployment_prior_only"])
        self.assertFalse(any(key.startswith("training_") for key in first))
        for key, value in first.items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(value, second[key])

    def test_training_teacher_uses_target_while_prior_prediction_does_not(self):
        batch = fake_formal_batch()
        _complete_target(batch)
        changed = copy.deepcopy(batch)
        changed["target"]["node_state"] = changed["target"]["node_state"] + 9.0
        model = self._model().train()

        torch.manual_seed(23)
        first = model(batch)
        torch.manual_seed(23)
        second = model(changed)

        torch.testing.assert_close(first["node_state_mean"], second["node_state_mean"])
        torch.testing.assert_close(first["rssm_rollout_prior_mean"], second["rssm_rollout_prior_mean"])
        self.assertFalse(
            torch.allclose(
                first["training_node_state_mean"],
                second["training_node_state_mean"],
            )
        )
        self.assertFalse(
            torch.allclose(
                first["training_link_activity_logits"],
                second["training_link_activity_logits"],
            )
        )
        self.assertEqual((2, 2, 5), tuple(first["rssm_posterior_mean"].shape))
        self.assertEqual((2, 1, 5), tuple(first["rssm_overshooting_prior_mean"].shape))

    def test_prior_output_exposes_exact_node_state_correction_for_safety_loss(self):
        batch = fake_formal_batch()
        _complete_target(batch)
        model = self._model().train()

        torch.manual_seed(29)
        prediction = model(batch)
        base = model.base(
            {
                "history": batch["history"],
                "future_action": batch["future_action"],
                "static": batch["static"],
            }
        )

        self.assertIn("rssm_node_state_correction", prediction)
        torch.testing.assert_close(
            base["node_state_mean"] + prediction["rssm_node_state_correction"],
            prediction["node_state_mean"],
        )

    def test_node_x_safety_loss_sends_gradient_to_correction_not_base(self):
        from pi_jwm.formal_world_model_loss_v1 import (
            FormalLossWeights,
            formal_world_model_loss,
        )

        batch = fake_formal_batch()
        target = _complete_target(batch)
        model = self._model().train()
        prediction = model(batch)
        correction = prediction["rssm_node_state_correction"]
        base_x = (prediction["node_state_mean"][..., 0] - correction[..., 0]).detach()
        target["node_state"][..., 0] = base_x
        weights = FormalLossWeights(
            state_nll=0.0,
            state_mae=0.0,
            presence=0.0,
            sparse_event=0.0,
            lifecycle=0.0,
            dag=0.0,
            active_rate_mae=0.0,
            rb_occupancy_mae=0.0,
            task_delay_mae=0.0,
            task_deadline_mae=0.0,
            uav_energy_nll=0.0,
            uav_energy_mae=0.0,
            node_x_residual_non_degradation=1.0,
        )

        loss, _ = formal_world_model_loss(
            prediction, target, batch["static"], weights=weights
        )
        model.zero_grad(set_to_none=True)
        loss.backward()

        correction_grad = model.rssm_prior_state_heads["node"].weight.grad
        self.assertIsNotNone(correction_grad)
        self.assertGreater(float(correction_grad.abs().sum()), 0.0)
        base_gradient = sum(
            float(parameter.grad.abs().sum())
            for name, parameter in model.named_parameters()
            if name.startswith("base.") and parameter.grad is not None
        )
        self.assertEqual(0.0, base_gradient)

    def test_prior_distribution_is_conditioned_on_candidate_action(self):
        batch = fake_formal_batch()
        _complete_target(batch)
        changed = copy.deepcopy(batch)
        changed["future_action"]["task_action_present"][:, :, 0] = True
        changed["future_action"]["task_action"][:, :, 0, 0] = 3.0
        model = self._model().eval()

        with torch.no_grad():
            first = model(batch)
            second = model(changed)

        self.assertFalse(
            torch.allclose(
                first["rssm_rollout_prior_mean"],
                second["rssm_rollout_prior_mean"],
            )
        )
        self.assertFalse(
            torch.allclose(
                first["link_activity_logits"],
                second["link_activity_logits"],
            )
        )

    def test_formal_loss_trains_prior_posterior_teacher_and_overshooting(self):
        from pi_jwm.formal_world_model_loss_v1 import FormalLossWeights, formal_world_model_loss

        batch = fake_formal_batch()
        _complete_target(batch)
        model = self._model().train()
        prediction = model(batch)
        loss, components = formal_world_model_loss(
            prediction,
            batch["target"],
            batch["static"],
            weights=FormalLossWeights(
                rssm_kl=1.0,
                rssm_teacher_reconstruction=0.5,
                rssm_overshooting=0.25,
            ),
        )
        loss.backward()

        for key in ("rssm_kl", "rssm_teacher_reconstruction", "rssm_overshooting"):
            self.assertIn(key, components)
            self.assertTrue(torch.isfinite(components[key]))
        for name in (
            "rssm_transition.weight_ih",
            "rssm_prior.weight",
            "rssm_posterior.weight",
            "rssm_teacher_state_heads.node.weight",
        ):
            parameter = dict(model.named_parameters())[name]
            self.assertIsNotNone(parameter.grad, name)
            self.assertGreater(float(parameter.grad.abs().sum()), 0.0, name)


if __name__ == "__main__":
    unittest.main()
