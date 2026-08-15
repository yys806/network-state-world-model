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

from test_r3_objective import objective_batch


class R4GraphRSSMTests(unittest.TestCase):
    def _model(self):
        from pi_jwm.r4_module_registry import make_single_module_config
        from pi_jwm.r4_world_model import build_r4_world_model

        return build_r4_world_model(
            make_single_module_config(
                "dynamics",
                "graph_rssm_v1",
                hidden_dim=8,
                history_steps=2,
            )
        )

    def test_rssm_reports_context_posterior_and_prior_only_rollout(self):
        model = self._model().eval()
        output = model(objective_batch(horizon=5), rollout_steps=5)

        self.assertTrue(output.execution_metadata["deployment_prior_only"])
        self.assertEqual(
            (1, 5, 8),
            tuple(output.probabilistic_parameters["rollout_prior_mean"].shape),
        )
        self.assertEqual(
            (1, 8),
            tuple(output.probabilistic_parameters["context_posterior_mean"].shape),
        )
        self.assertTrue(
            all(
                torch.isfinite(value).all()
                for value in output.probabilistic_parameters.values()
            )
        )

    def test_rssm_open_loop_does_not_read_future_targets(self):
        torch.manual_seed(67)
        model = self._model().eval()
        left_batch = objective_batch(horizon=20)
        right_batch = copy.deepcopy(left_batch)
        for key, value in right_batch.target.items():
            if torch.is_floating_point(value):
                right_batch.target[key] = torch.randn_like(value) * 1000.0
        left = model(left_batch, rollout_steps=20)
        right = model(right_batch, rollout_steps=20)
        torch.testing.assert_close(
            left.predicted_belief.joint_latent,
            right.predicted_belief.joint_latent,
        )
        torch.testing.assert_close(
            left.probabilistic_parameters["rollout_prior_mean"],
            right.probabilistic_parameters["rollout_prior_mean"],
        )

    def test_rssm_future_action_changes_prior_rollout(self):
        model = self._model().eval()
        left_batch = objective_batch(horizon=2)
        right_batch = copy.deepcopy(left_batch)
        right_batch.future_action["task_action"].add_(1.0)
        left = model(left_batch, rollout_steps=2)
        right = model(right_batch, rollout_steps=2)
        self.assertGreater(
            torch.max(
                torch.abs(
                    left.probabilistic_parameters["rollout_prior_mean"]
                    - right.probabilistic_parameters["rollout_prior_mean"]
                )
            ).item(),
            0.0,
        )

    def test_rssm_objective_adds_finite_kl_and_reaches_rssm_parameters(self):
        from pi_jwm.r4_objective import compute_r4_objective

        model = self._model()
        batch = objective_batch(horizon=2)
        report = compute_r4_objective(model(batch, rollout_steps=2), batch)
        report.total.backward()

        self.assertEqual("computed", report.auxiliary_terms["rssm_kl"].status)
        self.assertTrue(torch.isfinite(report.total))
        rssm_gradients = [
            parameter.grad
            for name, parameter in model.named_parameters()
            if "rssm" in name and parameter.grad is not None
        ]
        self.assertTrue(rssm_gradients)
        self.assertTrue(
            any(torch.count_nonzero(gradient).item() > 0 for gradient in rssm_gradients)
        )


if __name__ == "__main__":
    unittest.main()
