from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from test_formal_dual_graph_world_model_v1 import fake_formal_batch
from test_formal_world_model_metrics_v1 import identity_stats


class FormalWorldModelBaselinesV1Tests(unittest.TestCase):
    def test_rule_baselines_match_the_learned_output_shapes(self):
        from pi_jwm.formal_world_model_baselines_v1 import build_rule_prediction

        batch = fake_formal_batch()
        for method in ("zero_activity", "last_persistence"):
            with self.subTest(method=method):
                output = build_rule_prediction(method, batch, identity_stats())
                self.assertEqual((2, 2, 4, 7), tuple(output["node_state_mean"].shape))
                self.assertEqual((2, 2, 3, 5), tuple(output["physical_edge_state_mean"].shape))
                self.assertEqual((2, 2, 2, 5), tuple(output["flow_state_mean"].shape))
                self.assertEqual((2, 2, 3, 8), tuple(output["task_state_mean"].shape))
                self.assertEqual((2, 2, 3, 3), tuple(output["task_dag_state_mean"].shape))
                self.assertEqual((2, 2, 2), tuple(output["dag_edge_presence_logits"].shape))
                self.assertTrue(all(torch.isfinite(value).all() for value in output.values()))

    def test_last_persistence_repeats_the_last_observation(self):
        from pi_jwm.formal_world_model_baselines_v1 import build_rule_prediction

        batch = fake_formal_batch()
        output = build_rule_prediction("last_persistence", batch, identity_stats())

        expected = batch["history"]["node_state"][:, -1:].expand(-1, 2, -1, -1)
        torch.testing.assert_close(output["node_state_mean"], expected)
        expected_dag = batch["history"]["dag_edge_present"][:, -1:].expand(-1, 2, -1)
        self.assertTrue(torch.equal(output["dag_edge_presence_logits"] >= 0, expected_dag))

    def test_zero_activity_turns_off_traffic_and_task_events(self):
        from pi_jwm.formal_world_model_baselines_v1 import build_rule_prediction

        output = build_rule_prediction("zero_activity", fake_formal_batch(), identity_stats())

        self.assertTrue(torch.all(output["link_activity_logits"] < 0))
        self.assertTrue(torch.all(output["flow_presence_logits"] < 0))
        self.assertTrue(torch.all(output["task_presence_logits"] < 0))
        self.assertTrue(torch.all(output["dag_release_logits"] < 0))
        self.assertTrue(torch.all(output["dag_edge_presence_logits"] < 0))

    def test_method_registry_separates_cpu_methods_from_paper_baseline(self):
        from pi_jwm.formal_world_model_baselines_v1 import method_registry

        registry = method_registry()
        cpu_methods = [name for name, item in registry.items() if item["stage"] == "cpu_ready"]

        self.assertEqual(
            {
                "zero_activity",
                "last_persistence",
                "pooled_gru",
                "independent_dual_gnn",
                "coupled_dual_gnn",
            },
            set(cpu_methods),
        )
        self.assertEqual("gpu_pending", registry["coupled_jepa_bou_chaaya_2026"]["stage"])
        self.assertEqual("complete_paper_method_adapter", registry["coupled_jepa_bou_chaaya_2026"]["role"])
        self.assertFalse(registry["zero_activity"]["distribution_output"])
        self.assertTrue(registry["coupled_dual_gnn"]["distribution_output"])
        self.assertEqual(
            "local_interface_ready",
            registry["coupled_directed_dynamic_v2"]["stage"],
        )
        self.assertFalse(
            registry["coupled_directed_dynamic_v2"]["residual_state_prediction"]
        )
        self.assertTrue(
            registry["coupled_directed_dynamic_residual_v2"][
                "residual_state_prediction"
            ]
        )
        self.assertEqual(
            "deterministic",
            registry["coupled_directed_dynamic_v2"]["latent_dynamics"],
        )

    def test_rule_baseline_uncertainty_is_reported_as_not_computable(self):
        from pi_jwm.formal_world_model_baselines_v1 import build_rule_prediction
        from pi_jwm.formal_world_model_metrics_v1 import FormalMetricAccumulator
        from test_formal_world_model_metrics_v1 import perfect_prediction

        batch = fake_formal_batch()
        perfect_prediction(batch)
        prediction = build_rule_prediction("last_persistence", batch, identity_stats())
        accumulator = FormalMetricAccumulator(identity_stats(), distribution_available=False)
        accumulator.update(prediction, batch["target"], batch["static"])
        metrics = accumulator.finalize()["horizons"]["overall"]["metrics"]

        self.assertEqual("not_computable", metrics["uncertainty.node.x.nll"]["status"])
        self.assertIn("deterministic", metrics["uncertainty.node.x.nll"]["reason"])


if __name__ == "__main__":
    unittest.main()
