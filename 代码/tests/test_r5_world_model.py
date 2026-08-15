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
from test_r3_world_model import make_batch
from test_r4_dag_candidate import dag_batch


def dag_objective_batch():
    batch = objective_batch(horizon=1)
    batch.static["dag_edge_index"] = torch.tensor(
        [[[0, 1]]], dtype=torch.long
    )
    batch.static["dag_edge_valid"] = torch.ones(1, 1, dtype=torch.bool)
    batch.history["dag_edge_present"] = torch.ones(
        1, 2, 1, dtype=torch.bool
    )
    batch.target["dag_edge_present"] = torch.ones(
        1, 1, 1, dtype=torch.bool
    )
    return batch


class R5WorldModelTests(unittest.TestCase):
    def _model(self, combination_id: str):
        from pi_jwm.r5_world_model import build_r5_world_model

        return build_r5_world_model(
            combination_id,
            hidden_dim=8,
            history_steps=2,
        )

    def test_all_approved_combinations_execute_common_rollout_contract(self):
        for combination_id in "ABCDE":
            with self.subTest(combination_id=combination_id):
                torch.manual_seed(101)
                model = self._model(combination_id).eval()
                batch = dag_batch() if combination_id == "D" else make_batch(horizon=5)
                horizon = 1 if combination_id == "D" else 5
                output = model(batch, rollout_steps=horizon)
                self.assertEqual(combination_id, model.combination_id)
                self.assertEqual(horizon, output.predicted_belief.joint_latent.shape[1])
                self.assertTrue(torch.isfinite(output.predicted_belief.joint_latent).all())
                self.assertEqual(model.config.component_names(), model.component_registry())

    def test_c_combines_rssm_and_heteroscedastic_objectives_and_gradients(self):
        from pi_jwm.r4_objective import compute_r4_objective

        torch.manual_seed(103)
        model = self._model("C")
        batch = objective_batch(horizon=2)
        output = model(batch, rollout_steps=2)
        self.assertIn("context_prior_mean", output.probabilistic_parameters)
        self.assertIn(
            "physical_node_state_log_variance", output.probabilistic_parameters
        )
        report = compute_r4_objective(output, batch)
        report.total.backward()
        self.assertEqual("computed", report.auxiliary_terms["rssm_kl"].status)
        self.assertEqual(
            "computed", report.auxiliary_terms["heteroscedastic_nll"].status
        )
        for marker in ("rssm", "heteroscedastic"):
            gradients = [
                parameter.grad
                for name, parameter in model.named_parameters()
                if marker in name and parameter.grad is not None
            ]
            self.assertTrue(gradients, marker)
            self.assertTrue(
                any(torch.count_nonzero(value).item() > 0 for value in gradients),
                marker,
            )

    def test_d_combines_rssm_and_explicit_dag_gradients(self):
        from pi_jwm.r4_objective import compute_r4_objective

        torch.manual_seed(107)
        model = self._model("D")
        batch = dag_objective_batch()
        report = compute_r4_objective(model(batch, rollout_steps=1), batch)
        report.total.backward()
        self.assertEqual("computed", report.auxiliary_terms["rssm_kl"].status)
        for marker in ("rssm", "dag_"):
            gradients = [
                parameter.grad
                for name, parameter in model.named_parameters()
                if marker in name and parameter.grad is not None
            ]
            self.assertTrue(gradients, marker)
            self.assertTrue(
                any(torch.count_nonzero(value).item() > 0 for value in gradients),
                marker,
            )

    def test_e_predicted_presence_changes_second_rssm_rollout_step(self):
        torch.manual_seed(109)
        low = self._model("E").eval()
        high = self._model("E").eval()
        high.load_state_dict(low.state_dict())
        for model, bias in ((low, -20.0), (high, 20.0)):
            presence_backend = model.backend.base
            for name in (
                "physical_node_present",
                "physical_edge_present",
                "information_node_present",
                "information_edge_present",
            ):
                presence_backend.presence_heads[name].bias.data.fill_(bias)

        batch = make_batch(horizon=2)
        low_output = low(batch, rollout_steps=2).predicted_belief.joint_latent
        high_output = high(batch, rollout_steps=2).predicted_belief.joint_latent
        torch.testing.assert_close(low_output[:, 0], high_output[:, 0])
        self.assertGreater(
            torch.max(torch.abs(low_output[:, 1] - high_output[:, 1])).item(),
            0.0,
        )

    def test_unknown_combination_fails_before_model_construction(self):
        from pi_jwm.r5_world_model import build_r5_world_model

        with self.assertRaisesRegex(ValueError, "unknown R5 combination"):
            build_r5_world_model("F")


if __name__ == "__main__":
    unittest.main()
