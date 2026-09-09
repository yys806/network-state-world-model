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
    def test_aggregate_rate_metrics_ignore_unobserved_edge_values(self):
        from pi_jwm.formal_world_model_metrics_v1 import FormalMetricAccumulator

        batch = fake_formal_batch()
        prediction = perfect_prediction(batch)
        target = batch["target"]
        target["aggregate_link_activity"] = target["link_activity"].clone()
        target["aggregate_link_activity_mask"] = torch.ones_like(target["link_activity"], dtype=torch.bool)
        target["aggregate_link_rate_sum"] = target["physical_edge_state"][..., 2].clone()
        target["aggregate_link_rate_sum_mask"] = torch.ones_like(target["link_activity"], dtype=torch.bool)
        target["aggregate_rb_occupancy"] = target["physical_edge_state"][..., 4].clone()
        target["aggregate_rb_occupancy_mask"] = torch.ones_like(target["link_activity"], dtype=torch.bool)
        target["aggregate_link_rate_sum_mask"][:, :, 0] = False
        prediction["physical_edge_state_mean"] = target["physical_edge_state"].clone()
        prediction["physical_edge_state_mean"][:, :, 0, 2] += 100.0

        accumulator = FormalMetricAccumulator(identity_stats())
        accumulator.update(prediction, target, batch["static"])
        metrics = accumulator.finalize()["horizons"]["overall"]["metrics"]

        self.assertEqual(0.0, metrics["link.active_only_rate.mae"]["value"])
        self.assertEqual(0.0, metrics["system.communication_throughput.mae"]["value"])

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

    def test_aggregate_rb_metric_restores_prediction_mean_but_not_raw_target(self):
        from pi_jwm.formal_world_model_metrics_v1 import FormalMetricAccumulator

        batch = fake_formal_batch()
        prediction = perfect_prediction(batch)
        target = batch["target"]
        rb_index = 4
        valid_edges = batch["static"]["physical_edge_endpoint_index"][..., 0] >= 0
        raw_rb = torch.full_like(target["link_activity"], 4.0, dtype=torch.float32)
        raw_rb_mask = valid_edges[:, None, :].expand_as(raw_rb)
        target["aggregate_rb_occupancy"] = raw_rb
        target["aggregate_rb_occupancy_mask"] = raw_rb_mask
        prediction["physical_edge_state_mean"] = prediction[
            "physical_edge_state_mean"
        ].clone()
        prediction["physical_edge_state_mean"][..., rb_index] = 0.0

        stats = identity_stats()
        stats["features"]["physical_edge_state"]["mean"][rb_index] = 4.0
        stats["features"]["physical_edge_state"]["scale"][rb_index] = 3.0
        accumulator = FormalMetricAccumulator(stats)
        accumulator.update(prediction, target, batch["static"])
        metric = accumulator.finalize()["horizons"]["overall"]["metrics"][
            "resource.rb_occupancy.mae"
        ]

        self.assertEqual(0.0, metric["value"])
        self.assertEqual("RB", metric["unit"])

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
        for name, unit in (
            ("event.link_activity.nll", "nats"),
            ("event.link_activity.brier", "ratio"),
            ("event.link_activity.ece", "ratio"),
        ):
            self.assertEqual(unit, registry[name]["unit"])
            self.assertEqual(
                [
                    "aggregate_link_activity",
                    "aggregate_link_activity_mask",
                    "physical_edge_endpoint_index",
                ],
                registry[name]["source_fields"],
            )

    def test_computed_aggregate_metric_records_use_aggregate_sources(self):
        from pi_jwm.formal_world_model_metrics_v1 import FormalMetricAccumulator

        batch = fake_formal_batch()
        prediction = perfect_prediction(batch)
        target = batch["target"]
        target["aggregate_link_activity"] = target["link_activity"].clone()
        target["aggregate_link_activity_mask"] = torch.ones_like(target["link_activity"], dtype=torch.bool)
        target["aggregate_link_rate_sum"] = target["physical_edge_state"][..., 2].clone()
        target["aggregate_link_rate_sum_mask"] = torch.ones_like(target["link_activity"], dtype=torch.bool)
        target["aggregate_rb_occupancy"] = target["physical_edge_state"][..., 4].clone()
        target["aggregate_rb_occupancy_mask"] = torch.ones_like(target["link_activity"], dtype=torch.bool)

        report = FormalMetricAccumulator(identity_stats())
        report.update(prediction, target, batch["static"])
        metrics = report.finalize()["horizons"]["overall"]["metrics"]

        self.assertEqual(
            ["aggregate_link_rate_sum", "aggregate_link_rate_sum_mask", "aggregate_link_activity"],
            metrics["link.active_only_rate.mae"]["source_fields"],
        )
        self.assertEqual(
            ["aggregate_rb_occupancy", "aggregate_rb_occupancy_mask"],
            metrics["resource.rb_occupancy.mae"]["source_fields"],
        )
        self.assertEqual(
            ["aggregate_link_rate_sum", "aggregate_link_rate_sum_mask"],
            metrics["system.communication_throughput.mae"]["source_fields"],
        )

    def test_binary_events_accept_independent_thresholds(self):
        from pi_jwm.formal_world_model_metrics_v1 import FormalMetricAccumulator

        batch = fake_formal_batch()
        target = _complete_target(batch)
        target["task_dag_state"][..., 2] = 1.0
        prediction = perfect_prediction(batch)
        for name in (
            "link_activity_logits",
            "flow_presence_logits",
            "task_presence_logits",
            "dag_release_logits",
            "dag_edge_presence_logits",
        ):
            prediction[name].fill_(1.38629436112)

        report = FormalMetricAccumulator(
            identity_stats(),
            thresholds={
                "link_activity": 0.9,
                "flow_present": 0.5,
                "task_present": 0.5,
                "dag_release": 0.5,
                "dag_edge_present": 0.9,
            },
        )
        report.update(prediction, target, batch["static"])
        metrics = report.finalize()["horizons"]["overall"]["metrics"]

        self.assertEqual(0.0, metrics["event.link_activity.f1"]["value"])
        self.assertGreater(metrics["event.flow_present.f1"]["value"], 0.0)
        self.assertGreater(metrics["event.task_present.f1"]["value"], 0.0)
        self.assertGreater(metrics["dag.release_ready.f1"]["value"], 0.0)
        self.assertEqual(0.0, metrics["dag.edge_presence.f1"]["value"])

    def test_link_calibration_reports_formal_probability_metrics_without_changing_decisions(self):
        from pi_jwm.formal_binary_calibration_v1 import InverseTemperatureCalibration
        from pi_jwm.formal_world_model_metrics_v1 import FormalMetricAccumulator

        batch = fake_formal_batch()
        prediction = perfect_prediction(batch)
        target = batch["target"]
        target["aggregate_link_activity"] = target["link_activity"].clone()
        target["aggregate_link_activity_mask"] = torch.ones_like(
            target["link_activity"], dtype=torch.bool
        )
        target["aggregate_link_activity_mask"][:, :, -1] = False
        event_probability = torch.where(
            target["aggregate_link_activity"].bool(), 0.8, 0.2
        )
        prediction["link_activity_logits"] = (
            math.log(50.0) + torch.logit(event_probability)
        )
        calibration = InverseTemperatureCalibration(50.0, 0.0)
        raw_threshold = 0.9
        mapped_threshold = calibration.map_raw_threshold(raw_threshold)
        legacy = FormalMetricAccumulator(
            identity_stats(), thresholds={"link_activity": raw_threshold}
        )
        calibrated = FormalMetricAccumulator(
            identity_stats(),
            thresholds={"link_activity": raw_threshold},
            link_probability_calibration=calibration,
        )
        legacy.update(prediction, target, batch["static"])
        calibrated.update(prediction, target, batch["static"])
        legacy_report = legacy.finalize()
        report = calibrated.finalize()

        self.assertEqual("event_probability", report["threshold_coordinate"]["link_activity"])
        self.assertEqual("legacy_sigmoid_score", report["threshold_coordinate"]["flow_present"])
        self.assertEqual(50.0, report["link_probability_calibration"]["pos_weight"])
        self.assertEqual(1.0, report["link_probability_calibration"]["temperature"])
        self.assertAlmostEqual(mapped_threshold, report["thresholds"]["link_activity"], places=12)
        self.assertEqual(raw_threshold, report["legacy_raw_thresholds"]["link_activity"])
        self.assertAlmostEqual(
            mapped_threshold,
            report["mapped_probability_thresholds"]["link_activity"],
            places=12,
        )
        expected_count = int(target["aggregate_link_activity_mask"].sum())
        for horizon in ("overall", "k=1", "k=2"):
            metrics = report["horizons"][horizon]["metrics"]
            legacy_metrics = legacy_report["horizons"][horizon]["metrics"]
            for name in (
                "event.link_activity.nll",
                "event.link_activity.brier",
                "event.link_activity.ece",
            ):
                self.assertEqual("computed", metrics[name]["status"])
                self.assertTrue(math.isfinite(metrics[name]["value"]))
                self.assertEqual(expected_count if horizon == "overall" else expected_count // 2, metrics[name]["count"])
            for name in (
                "event.link_activity.precision",
                "event.link_activity.recall",
                "event.link_activity.f1",
            ):
                self.assertEqual(legacy_metrics[name], metrics[name])

    def test_link_probability_metrics_are_not_computable_without_calibration(self):
        from pi_jwm.formal_world_model_metrics_v1 import FormalMetricAccumulator

        batch = fake_formal_batch()
        report = FormalMetricAccumulator(identity_stats())
        report.update(perfect_prediction(batch), batch["target"], batch["static"])
        metrics = report.finalize()["horizons"]["overall"]["metrics"]

        for name in (
            "event.link_activity.nll",
            "event.link_activity.brier",
            "event.link_activity.ece",
        ):
            self.assertEqual("not_computable", metrics[name]["status"])
            self.assertIn("raw_weighted_score is not formal event probability", metrics[name]["reason"])
        self.assertIsNone(report.finalize()["legacy_raw_thresholds"])
        self.assertIsNone(report.finalize()["mapped_probability_thresholds"])
        self.assertEqual("raw_weighted_score", report.finalize()["threshold_coordinate"]["link_activity"])
        self.assertEqual("legacy_sigmoid_score", report.finalize()["threshold_coordinate"]["flow_present"])

    def test_link_calibration_maps_the_single_legacy_threshold_input(self):
        from pi_jwm.formal_binary_calibration_v1 import InverseTemperatureCalibration
        from pi_jwm.formal_world_model_metrics_v1 import FormalMetricAccumulator

        calibration = InverseTemperatureCalibration(50.0, 0.0)
        report = FormalMetricAccumulator(
            identity_stats(), link_probability_calibration=calibration
        ).finalize()

        self.assertEqual(0.5, report["legacy_raw_thresholds"]["link_activity"])
        self.assertAlmostEqual(
            calibration.map_raw_threshold(0.5),
            report["thresholds"]["link_activity"],
            places=12,
        )

    def test_link_calibration_preserves_legacy_tie_decisions_for_identity_and_nonidentity_temperature(self):
        from pi_jwm.formal_binary_calibration_v1 import InverseTemperatureCalibration
        from pi_jwm.formal_world_model_metrics_v1 import FormalMetricAccumulator

        batch = fake_formal_batch()
        prediction = perfect_prediction(batch)
        target = batch["target"]
        target["aggregate_link_activity"] = target["link_activity"].clone()
        target["aggregate_link_activity_mask"] = torch.ones_like(
            target["link_activity"], dtype=torch.bool
        )
        tie_logit = torch.logit(torch.tensor(0.9, dtype=torch.float64))
        prediction["link_activity_logits"] = torch.full_like(
            target["link_activity"], tie_logit, dtype=torch.float64
        )
        legacy = FormalMetricAccumulator(
            identity_stats(), thresholds={"link_activity": 0.9}
        )
        legacy.update(prediction, target, batch["static"])
        legacy_metrics = legacy.finalize()["horizons"]["overall"]["metrics"]

        for log_temperature in (0.0, math.log(2.0)):
            calibrated = FormalMetricAccumulator(
                identity_stats(),
                thresholds={"link_activity": 0.9},
                link_probability_calibration=InverseTemperatureCalibration(
                    50.0, log_temperature
                ),
            )
            calibrated.update(prediction, target, batch["static"])
            metrics = calibrated.finalize()["horizons"]["overall"]["metrics"]
            for name in (
                "event.link_activity.precision",
                "event.link_activity.recall",
                "event.link_activity.f1",
            ):
                self.assertEqual(legacy_metrics[name], metrics[name])

    def test_metric_bucket_rejects_invalid_provided_binary_probabilities(self):
        from pi_jwm.formal_world_model_metrics_v1 import _MetricBucket

        bucket = _MetricBucket()
        logits = torch.zeros(2)
        target = torch.tensor([0, 1])
        valid = torch.tensor([True, True])
        for probabilities in (
            torch.tensor([0.2]),
            torch.tensor([float("nan"), 0.2]),
            torch.tensor([0.2, 1.1]),
        ):
            with self.assertRaises(ValueError):
                bucket.add_binary(
                    "link_activity",
                    logits,
                    target,
                    valid,
                    0.5,
                    probabilities=probabilities,
                )

    def test_metric_bucket_rejects_invalid_provided_binary_decisions(self):
        from pi_jwm.formal_world_model_metrics_v1 import _MetricBucket

        bucket = _MetricBucket()
        logits = torch.zeros(2)
        target = torch.tensor([0, 1])
        valid = torch.tensor([True, True])
        for predicted in (
            torch.tensor([0.0, 1.0]),
            torch.tensor([True]),
            torch.empty(2, dtype=torch.bool, device="meta"),
        ):
            with self.assertRaises(ValueError):
                bucket.add_binary(
                    "link_activity",
                    logits,
                    target,
                    valid,
                    0.5,
                    predicted=predicted,
                )


if __name__ == "__main__":
    unittest.main()
