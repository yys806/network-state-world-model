from __future__ import annotations

import copy
import math
import sys
import unittest
from pathlib import Path

import torch


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from test_formal_dual_graph_world_model_v1 import fake_formal_batch
from test_formal_world_model_loss_v1 import _complete_target


def perfect_prediction(batch: dict) -> dict[str, torch.Tensor]:
    target = _complete_target(batch)
    target["task_dag_state"][..., 2] = target["task_dag_state_present"].to(torch.float32)
    prediction = {}
    for name in ("node", "physical_edge", "flow", "task"):
        prediction[f"{name}_state_mean"] = target[f"{name}_state"].clone()
        prediction[f"{name}_state_log_variance"] = torch.zeros_like(target[f"{name}_state"])
        prediction[f"{name}_presence_logits"] = torch.where(
            target[f"{name}_present"], 20.0, -20.0
        )
    prediction["task_dag_state_mean"] = target["task_dag_state"].clone()
    prediction["task_dag_state_log_variance"] = torch.zeros_like(target["task_dag_state"])
    prediction["link_activity_logits"] = torch.where(target["link_activity"], 20.0, -20.0)
    prediction["task_lifecycle_logits"] = torch.full(
        (*target["task_lifecycle_index"].shape, 5), -20.0
    )
    prediction["task_lifecycle_logits"][..., 2] = 20.0
    prediction["dag_release_logits"] = torch.where(
        target["task_dag_state"][..., 2] > 0.5, 20.0, -20.0
    )
    prediction["dag_edge_presence_logits"] = torch.where(
        target["dag_edge_present"], 20.0, -20.0
    )
    return prediction


def identity_stats() -> dict:
    feature_counts = {"node": 7, "physical_edge": 5, "flow": 5, "task": 8}
    return {
        "features": {
            f"{name}_state": {"mean": [0.0] * count, "scale": [1.0] * count}
            for name, count in feature_counts.items()
        }
    }


class FormalWorldModelMetricsV1Tests(unittest.TestCase):
    def test_perfect_prediction_reports_zero_errors_and_perfect_events(self):
        from pi_jwm.formal_world_model_metrics_v1 import FormalMetricAccumulator

        batch = fake_formal_batch()
        prediction = perfect_prediction(batch)
        accumulator = FormalMetricAccumulator(identity_stats())
        accumulator.update(prediction, batch["target"], batch["static"])
        report = accumulator.finalize()

        for horizon in ("k=1", "k=2", "overall"):
            metrics = report["horizons"][horizon]["metrics"]
            self.assertEqual(0.0, metrics["state.node.x.mae"]["value"])
            self.assertEqual(0.0, metrics["state.node.x.rmse"]["value"])
            self.assertEqual(1.0, metrics["event.link_activity.f1"]["value"])
            self.assertEqual(1.0, metrics["event.flow_present.f1"]["value"])
            self.assertEqual(1.0, metrics["event.task_present.f1"]["value"])
            self.assertEqual(1.0, metrics["task.lifecycle.accuracy"]["value"])
            self.assertEqual(1.0, metrics["dag.release_ready.f1"]["value"])
            self.assertEqual(0.0, metrics["dag.unfinished_parent_count.mae"]["value"])
            self.assertEqual(1.0, metrics["uncertainty.node.x.coverage_95"]["value"])

    def test_state_errors_are_restored_to_physical_units(self):
        from pi_jwm.formal_world_model_metrics_v1 import FormalMetricAccumulator

        batch = fake_formal_batch()
        prediction = perfect_prediction(batch)
        prediction["node_state_mean"] = prediction["node_state_mean"].clone()
        prediction["node_state_mean"][..., 0] += 2.0
        stats = identity_stats()
        stats["features"]["node_state"]["scale"][0] = 10.0
        accumulator = FormalMetricAccumulator(stats)
        accumulator.update(prediction, batch["target"], batch["static"])
        metric = accumulator.finalize()["horizons"]["overall"]["metrics"]["state.node.x.mae"]

        self.assertEqual(20.0, metric["value"])
        self.assertEqual("m", metric["unit"])
        self.assertGreater(metric["count"], 0)
        self.assertEqual("computed", metric["status"])

    def test_active_only_rate_uses_only_true_active_links(self):
        from pi_jwm.formal_world_model_metrics_v1 import FormalMetricAccumulator

        batch = fake_formal_batch()
        prediction = perfect_prediction(batch)
        target = batch["target"]
        target["link_activity"].zero_()
        target["link_activity"][:, :, 0] = True
        prediction["link_activity_logits"] = torch.where(target["link_activity"], 20.0, -20.0)
        prediction["physical_edge_state_mean"] = target["physical_edge_state"].clone()
        prediction["physical_edge_state_mean"][:, :, 0, 2] += 3.0
        accumulator = FormalMetricAccumulator(identity_stats())
        accumulator.update(prediction, target, batch["static"])
        metrics = accumulator.finalize()["horizons"]["overall"]["metrics"]

        self.assertEqual(3.0, metrics["link.active_only_rate.mae"]["value"])
        self.assertEqual(3.0, metrics["link.active_only_rate.rmse"]["value"])

    def test_degenerate_labels_and_unavailable_system_metrics_are_explicit(self):
        from pi_jwm.formal_world_model_metrics_v1 import FormalMetricAccumulator

        batch = fake_formal_batch()
        prediction = perfect_prediction(batch)
        batch["target"]["link_activity"].zero_()
        prediction["link_activity_logits"].zero_()
        accumulator = FormalMetricAccumulator(identity_stats())
        accumulator.update(prediction, batch["target"], batch["static"])
        report = accumulator.finalize()
        metrics = report["horizons"]["overall"]["metrics"]

        self.assertEqual("not_computable", metrics["event.link_activity.auprc"]["status"])
        self.assertIsNone(metrics["event.link_activity.auprc"]["value"])
        for name in (
            "system.p95_latency",
            "system.p99_latency",
            "system.energy",
            "system.fairness",
            "decision.action_regret",
        ):
            self.assertEqual("not_computable", metrics[name]["status"])
            self.assertIsNone(metrics[name]["value"])
            self.assertTrue(metrics[name]["reason"])
        self.assertFalse(any(math.isnan(value["value"]) for value in metrics.values() if isinstance(value.get("value"), float)))

    def test_metric_registry_declares_units_sources_and_denominators(self):
        from pi_jwm.formal_world_model_metrics_v1 import metric_registry

        registry = metric_registry()
        definition = registry["state.physical_edge.rate_sum.rmse"]
        self.assertEqual("Mbps", definition["unit"])
        self.assertEqual(["physical_edge_state", "physical_edge_present"], definition["source_fields"])
        self.assertTrue(definition["denominator"])
        for name in (
            "event.link_activity.f1",
            "event.flow_present.auprc",
            "task.lifecycle.macro_f1",
            "dag.release_ready.f1",
            "link.active_only_rate.rmse",
            "system.communication_throughput.mae",
            "decision.action_regret",
        ):
            self.assertIn(name, registry)


if __name__ == "__main__":
    unittest.main()
