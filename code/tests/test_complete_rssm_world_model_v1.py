from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

import torch


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
TEST_ROOT = Path(__file__).resolve().parent
for root in (SRC_ROOT, TEST_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from test_r3_objective import objective_batch


class CompleteRSSMWorldModelTests(unittest.TestCase):
    def _model(self):
        from pi_jwm.r4_module_registry import make_single_module_config
        from pi_jwm.r4_world_model import build_r4_world_model

        return build_r4_world_model(
            make_single_module_config(
                "dynamics", "complete_graph_rssm_v1", hidden_dim=8, history_steps=2
            )
        )

    def test_has_separate_step_prior_and_posterior_paths(self):
        model = self._model().eval()
        output = model(objective_batch(horizon=5), rollout_steps=5)
        parameters = output.probabilistic_parameters
        for key in (
            "posterior_path_prior_mean",
            "posterior_path_posterior_mean",
            "rollout_prior_mean",
        ):
            self.assertEqual((1, 5, 8), tuple(parameters[key].shape))
        self.assertTrue(output.execution_metadata["training_posterior_teacher"])
        self.assertTrue(output.execution_metadata["deployment_prior_only"])

    def test_prediction_path_does_not_read_future_targets(self):
        torch.manual_seed(31)
        model = self._model().eval()
        left_batch = objective_batch(horizon=20)
        right_batch = copy.deepcopy(left_batch)
        for key, value in right_batch.target.items():
            if torch.is_floating_point(value):
                right_batch.target[key] = torch.randn_like(value) * 1000.0
        left = model(left_batch, rollout_steps=20)
        right = model(right_batch, rollout_steps=20)
        torch.testing.assert_close(
            left.predicted_belief.joint_latent, right.predicted_belief.joint_latent
        )
        torch.testing.assert_close(
            left.probabilistic_parameters["rollout_prior_mean"],
            right.probabilistic_parameters["rollout_prior_mean"],
        )

    def test_complete_objective_has_balanced_step_kl_and_gradients(self):
        from pi_jwm.r4_objective import compute_r4_objective

        model = self._model()
        report = compute_r4_objective(model(objective_batch(horizon=3), rollout_steps=3), objective_batch(horizon=3))
        self.assertEqual("computed", report.auxiliary_terms["rssm_teacher_reconstruction"].status)
        self.assertEqual("computed", report.auxiliary_terms["rssm_step_kl_balanced"].status)
        self.assertEqual("computed", report.auxiliary_terms["rssm_overshooting_consistency"].status)
        report.total.backward()
        parameter_map = dict(model.named_parameters())
        for prefix in ("backend.transition", "backend.prior", "backend.posterior"):
            gradients = [
                parameter.grad
                for name, parameter in parameter_map.items()
                if name.startswith(prefix) and parameter.grad is not None
            ]
            self.assertTrue(gradients, prefix)
            self.assertTrue(any(torch.count_nonzero(g).item() > 0 for g in gradients), prefix)

    def test_training_teacher_reconstruction_is_present_and_target_conditioned(self):
        torch.manual_seed(53)
        model = self._model()
        batch = objective_batch(horizon=3)
        output = model(batch, rollout_steps=3)
        self.assertIsNotNone(output.training_predicted_explicit)
        self.assertIsNotNone(output.training_predicted_logits)
        changed = copy.deepcopy(batch)
        for key, value in changed.target.items():
            if torch.is_floating_point(value):
                changed.target[key] = torch.randn_like(value) * 1000.0
        changed_output = model(changed, rollout_steps=3)
        self.assertGreater(
            torch.max(
                torch.abs(
                    output.training_predicted_explicit["physical_node_state"]
                    - changed_output.training_predicted_explicit["physical_node_state"]
                )
            ).item(),
            0.0,
        )
        # Deployment-visible prior outputs are deterministic and target
        # independent when sampling is disabled.
        model.eval()
        torch.manual_seed(54)
        deployment = model(batch, rollout_steps=3)
        torch.manual_seed(54)
        changed_deployment = model(changed, rollout_steps=3)
        torch.testing.assert_close(
            deployment.predicted_explicit["physical_node_state"],
            changed_deployment.predicted_explicit["physical_node_state"],
        )

    def test_h20_is_finite_and_target_changes_training_posterior_only(self):
        torch.manual_seed(41)
        model = self._model().eval()
        left_batch = objective_batch(horizon=20)
        right_batch = copy.deepcopy(left_batch)
        for key, value in right_batch.target.items():
            if torch.is_floating_point(value):
                right_batch.target[key] = torch.randn_like(value) * 1000.0
        left = model(left_batch, rollout_steps=20)
        right = model(right_batch, rollout_steps=20)
        self.assertTrue(torch.isfinite(left.predicted_belief.joint_latent).all())
        self.assertGreater(
            torch.max(
                torch.abs(
                    left.probabilistic_parameters["posterior_path_posterior_mean"]
                    - right.probabilistic_parameters["posterior_path_posterior_mean"]
                )
            ).item(),
            0.0,
        )
        torch.testing.assert_close(
            left.probabilistic_parameters["rollout_prior_mean"],
            right.probabilistic_parameters["rollout_prior_mean"],
        )


if __name__ == "__main__":
    unittest.main()
