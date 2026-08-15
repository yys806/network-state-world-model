from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from test_r3_world_model import make_batch


def objective_batch(*, horizon: int = 2):
    batch = make_batch(horizon=horizon, action_value=0.25)
    activity = batch.target["information_edge_state"][..., 11] > 0.0
    batch.target["information_link_activity"] = activity
    batch.target["information_link_activity_mask"] = (
        batch.target["information_edge_present"].bool()
        & batch.target["information_edge_feature_mask"][..., 11].bool()
    )
    return batch


class R3ObjectiveTests(unittest.TestCase):
    def test_objective_reports_explicit_terms_and_finite_total(self):
        from pi_jwm.r3_objective import compute_r3_objective
        from pi_jwm.r3_world_model import R3ReferenceConfig, R3ReferenceWorldModel

        torch.manual_seed(17)
        model = R3ReferenceWorldModel(R3ReferenceConfig(hidden_dim=8, history_steps=2))
        batch = objective_batch(horizon=2)
        report = compute_r3_objective(model(batch, rollout_steps=2), batch)

        self.assertTrue(torch.isfinite(report.total))
        self.assertEqual("computed", report.terms["physical_node_state"].status)
        self.assertEqual("computed", report.terms["information_link_activity"].status)
        self.assertEqual("computed", report.terms["task_lifecycle"].status)
        self.assertGreater(report.terms["physical_node_state"].count, 0)

    def test_empty_target_group_is_not_computable_not_zero(self):
        from pi_jwm.r3_objective import compute_r3_objective
        from pi_jwm.r3_world_model import R3ReferenceConfig, R3ReferenceWorldModel

        model = R3ReferenceWorldModel(R3ReferenceConfig(hidden_dim=8, history_steps=2))
        batch = objective_batch(horizon=1)
        batch.target["information_edge_feature_mask"].zero_()
        batch.target["information_link_activity_mask"].zero_()
        report = compute_r3_objective(model(batch, rollout_steps=1), batch)

        term = report.terms["information_edge_state"]
        self.assertEqual("not_computable", term.status)
        self.assertIsNone(term.value)
        self.assertEqual(0, term.count)

    def test_illegal_lifecycle_label_is_rejected(self):
        from pi_jwm.r3_objective import compute_r3_objective
        from pi_jwm.r3_world_model import R3ReferenceConfig, R3ReferenceWorldModel

        model = R3ReferenceWorldModel(R3ReferenceConfig(hidden_dim=8, history_steps=2))
        batch = objective_batch(horizon=1)
        batch.target["task_lifecycle_index"][0, 0, 0] = 5
        with self.assertRaisesRegex(ValueError, "task_lifecycle_index"):
            compute_r3_objective(model(batch, rollout_steps=1), batch)

    def test_reference_backward_reaches_all_required_modules(self):
        from pi_jwm.r3_objective import compute_r3_objective
        from pi_jwm.r3_world_model import R3ReferenceConfig, R3ReferenceWorldModel

        torch.manual_seed(23)
        model = R3ReferenceWorldModel(R3ReferenceConfig(hidden_dim=8, history_steps=2))
        batch = objective_batch(horizon=2)
        report = compute_r3_objective(model(batch, rollout_steps=2), batch)
        report.total.backward()

        parameters = dict(model.named_parameters())
        for prefix in (
            "physical_node_encoder",
            "information_edge_encoder",
            "coupler",
            "action_encoder",
            "task_transition",
            "state_heads.physical_node_state",
        ):
            gradients = [
                parameter.grad
                for name, parameter in parameters.items()
                if name.startswith(prefix) and parameter.grad is not None
            ]
            self.assertTrue(gradients, prefix)
            self.assertTrue(
                any(torch.count_nonzero(gradient).item() > 0 for gradient in gradients),
                prefix,
            )


if __name__ == "__main__":
    unittest.main()
