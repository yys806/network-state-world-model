from __future__ import annotations

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


class R4OutputHeadTests(unittest.TestCase):
    def _hurdle_model(self):
        from pi_jwm.r4_module_registry import make_single_module_config
        from pi_jwm.r4_world_model import build_r4_world_model

        return build_r4_world_model(
            make_single_module_config(
                "head",
                "hurdle_active_rate_v1",
                hidden_dim=8,
                history_steps=2,
                information_rate_mean=10.0,
                information_rate_scale=2.0,
            )
        )

    def _heteroscedastic_model(self):
        from pi_jwm.r4_module_registry import make_single_module_config
        from pi_jwm.r4_world_model import build_r4_world_model

        return build_r4_world_model(
            make_single_module_config(
                "head",
                "heteroscedastic_typed_v1",
                hidden_dim=8,
                history_steps=2,
            )
        )

    def test_heteroscedastic_head_returns_finite_log_variance_per_continuous_target(self):
        model = self._heteroscedastic_model()
        output = model(objective_batch(horizon=5), rollout_steps=5)
        for name, prediction in output.predicted_explicit.items():
            key = f"{name}_log_variance"
            self.assertIn(key, output.probabilistic_parameters)
            self.assertEqual(
                tuple(prediction.shape),
                tuple(output.probabilistic_parameters[key].shape),
            )
            self.assertTrue(torch.isfinite(output.probabilistic_parameters[key]).all())

    def test_heteroscedastic_objective_adds_masked_nll_and_backpropagates(self):
        from pi_jwm.r4_objective import compute_r4_objective

        model = self._heteroscedastic_model()
        batch = objective_batch(horizon=2)
        batch.target["physical_node_feature_mask"][..., 0] = False
        report = compute_r4_objective(model(batch, rollout_steps=2), batch)
        report.total.backward()

        self.assertEqual(
            "computed",
            report.auxiliary_terms["heteroscedastic_nll"].status,
        )
        gradients = [
            parameter.grad
            for name, parameter in model.named_parameters()
            if "heteroscedastic" in name and parameter.grad is not None
        ]
        self.assertTrue(gradients)
        self.assertTrue(
            any(torch.count_nonzero(gradient).item() > 0 for gradient in gradients)
        )

    def test_hurdle_requires_train_only_rate_normalization(self):
        from pi_jwm.r4_module_registry import make_single_module_config

        with self.assertRaisesRegex(ValueError, "normalization"):
            make_single_module_config("head", "hurdle_active_rate_v1")

    def test_hurdle_returns_positive_raw_rate_and_normalized_expected_rate(self):
        model = self._hurdle_model()
        output = model(objective_batch(horizon=2), rollout_steps=2)
        raw_mean = output.probabilistic_parameters["active_rate_raw_mean"]
        self.assertTrue((raw_mean > 0.0).all())
        activity_probability = torch.sigmoid(
            output.predicted_logits["information_link_activity"]
        )
        expected_normalized = (activity_probability * raw_mean - 10.0) / 2.0
        torch.testing.assert_close(
            output.predicted_explicit["information_edge_state"][..., 12],
            expected_normalized,
        )

    def test_hurdle_objective_uses_active_positive_raw_rate_only(self):
        from pi_jwm.r4_objective import compute_r4_objective

        model = self._hurdle_model()
        batch = objective_batch(horizon=2)
        batch.target["information_link_activity"].fill_(True)
        batch.target["information_link_activity_mask"].fill_(True)
        batch.target["information_edge_state"][..., 12] = 1.0
        report = compute_r4_objective(model(batch, rollout_steps=2), batch)
        report.total.backward()

        self.assertEqual(
            "computed",
            report.auxiliary_terms["hurdle_active_rate_nll"].status,
        )
        gradients = [
            parameter.grad
            for name, parameter in model.named_parameters()
            if "hurdle" in name and parameter.grad is not None
        ]
        self.assertTrue(gradients)
        self.assertTrue(
            any(torch.count_nonzero(gradient).item() > 0 for gradient in gradients)
        )

    def test_hurdle_reports_not_computable_when_window_has_no_active_rate(self):
        from pi_jwm.r4_objective import compute_r4_objective

        model = self._hurdle_model()
        batch = objective_batch(horizon=1)
        batch.target["information_link_activity"].fill_(False)
        report = compute_r4_objective(model(batch, rollout_steps=1), batch)
        term = report.auxiliary_terms["hurdle_active_rate_nll"]
        self.assertEqual("not_computable", term.status)
        self.assertEqual(0, term.count)
        self.assertTrue(torch.isfinite(report.total))

    def test_hurdle_accepts_active_links_with_zero_realized_rate(self):
        from pi_jwm.r4_objective import compute_r4_objective

        model = self._hurdle_model()
        batch = objective_batch(horizon=1)
        batch.target["information_link_activity"].fill_(True)
        batch.target["information_link_activity_mask"].fill_(True)
        rate_mean = model.config.information_rate_mean
        rate_scale = model.config.information_rate_scale
        batch.target["information_edge_state"][..., 12] = -rate_mean / rate_scale

        report = compute_r4_objective(model(batch, rollout_steps=1), batch)

        term = report.auxiliary_terms["hurdle_active_rate_nll"]
        self.assertEqual("not_computable", term.status)
        self.assertEqual(0, term.count)
        self.assertTrue(torch.isfinite(report.total))


if __name__ == "__main__":
    unittest.main()
