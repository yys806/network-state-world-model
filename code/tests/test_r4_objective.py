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


class R4ObjectiveTests(unittest.TestCase):
    def test_reference_objective_preserves_r3_terms_and_backpropagates(self):
        from pi_jwm.r4_module_registry import reference_r4_config
        from pi_jwm.r4_objective import compute_r4_objective
        from pi_jwm.r4_world_model import build_r4_world_model

        torch.manual_seed(47)
        model = build_r4_world_model(
            reference_r4_config(hidden_dim=8, history_steps=2)
        )
        batch = objective_batch(horizon=5)
        report = compute_r4_objective(model(batch, rollout_steps=5), batch)
        report.total.backward()

        self.assertTrue(torch.isfinite(report.total))
        self.assertFalse(report.auxiliary_terms)
        self.assertIn("information_link_activity", report.terms)
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))

    def test_no_coupling_objective_uses_same_frozen_target_terms(self):
        from pi_jwm.r4_module_registry import make_single_module_config
        from pi_jwm.r4_objective import compute_r4_objective
        from pi_jwm.r4_world_model import build_r4_world_model

        model = build_r4_world_model(
            make_single_module_config(
                "coupling",
                "no_cross_graph_coupling_v1",
                hidden_dim=8,
                history_steps=2,
            )
        )
        batch = objective_batch(horizon=1)
        report = compute_r4_objective(model(batch, rollout_steps=1), batch)
        self.assertEqual("computed", report.terms["task_lifecycle"].status)
        self.assertTrue(torch.isfinite(report.total))


if __name__ == "__main__":
    unittest.main()
