from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class FormalSystemOutcomeMetricsV1Tests(unittest.TestCase):
    def test_computes_event_latency_throughput_energy_and_fairness_metrics(self):
        from pi_jwm.formal_system_outcome_metrics_v1 import compute_system_outcome_metrics

        true_event = np.asarray([[0, 0], [1, 0], [0, 1]], dtype=bool)
        predicted_event = np.asarray([[0, 1], [1, 0], [0, 0]], dtype=bool)
        true_delay = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 3.0]])
        predicted_delay = np.asarray([[0.0, 4.0], [1.5, 0.0], [0.0, 2.5]])
        report = compute_system_outcome_metrics(
            true_completion_event=true_event,
            predicted_completion_event=predicted_event,
            true_completed_delay=true_delay,
            predicted_task_delay=predicted_delay,
            completed_delay_valid=true_event,
            true_delivered_data=np.asarray([1.0, 2.0, 0.0]),
            predicted_delivered_data=np.asarray([1.0, 1.0, 0.0]),
            step_seconds=0.1,
            true_uav_energy=np.asarray([[1.0, 0.0], [2.0, 0.0], [1.0, 0.0]]),
            predicted_uav_energy=np.asarray([[1.5, 0.0], [1.5, 0.0], [1.0, 0.0]]),
            uav_energy_valid=np.asarray([[1, 0], [1, 0], [1, 0]], dtype=bool),
            true_source_service=np.asarray([[0, 0], [1, 0], [0, 1]], dtype=float),
            predicted_source_service=np.asarray([[0, 1], [1, 0], [0, 0]], dtype=float),
            source_population_valid=np.asarray([1, 1], dtype=bool),
            source_evaluable_task_count=np.asarray([1, 1], dtype=float),
        )
        metrics = report["metrics"]

        self.assertAlmostEqual(0.5, metrics["event.task_completion.f1"]["value"])
        self.assertAlmostEqual(0.5, metrics["system.latency.completed_task_delay.mae"]["value"])
        self.assertAlmostEqual(0.75, metrics["system.latency.mean.absolute_error"]["value"])
        self.assertAlmostEqual(0.975, metrics["system.latency.p95.absolute_error"]["value"])
        self.assertAlmostEqual(0.995, metrics["system.latency.p99.absolute_error"]["value"])
        self.assertAlmostEqual(
            10.0 / 3.0, metrics["system.application_throughput.mae"]["value"]
        )
        self.assertAlmostEqual(1.0 / 3.0, metrics["system.uav_energy.mae"]["value"])
        self.assertAlmostEqual(
            math.sqrt(0.5 / 3.0), metrics["system.uav_energy.rmse"]["value"]
        )
        self.assertEqual(0.0, metrics["system.uav_energy.total_absolute_error"]["value"])
        self.assertAlmostEqual(
            0.0, metrics["system.completion_fairness_jain.absolute_error"]["value"]
        )
        for value in metrics.values():
            self.assertIn(value["status"], {"computed", "not_computable"})
            if value["status"] == "computed":
                self.assertTrue(math.isfinite(float(value["value"])))

    def test_missing_energy_prediction_is_explicitly_not_computable(self):
        from pi_jwm.formal_system_outcome_metrics_v1 import compute_system_outcome_metrics

        event = np.asarray([[True]], dtype=bool)
        report = compute_system_outcome_metrics(
            true_completion_event=event,
            predicted_completion_event=event,
            true_completed_delay=np.asarray([[1.0]]),
            predicted_task_delay=np.asarray([[1.0]]),
            completed_delay_valid=event,
            true_delivered_data=np.asarray([1.0]),
            predicted_delivered_data=np.asarray([1.0]),
            step_seconds=0.1,
            true_uav_energy=np.asarray([[1.0]]),
            predicted_uav_energy=None,
            uav_energy_valid=np.asarray([[True]]),
            true_source_service=np.asarray([[1.0]]),
            predicted_source_service=np.asarray([[1.0]]),
            source_population_valid=np.asarray([True]),
            source_evaluable_task_count=np.asarray([1.0]),
        )

        for name in (
            "system.uav_energy.mae",
            "system.uav_energy.rmse",
            "system.uav_energy.total_absolute_error",
        ):
            self.assertEqual("not_computable", report["metrics"][name]["status"])
            self.assertIn("energy", report["metrics"][name]["reason"])

    def test_fairness_uses_per_source_completion_rates_not_raw_completion_counts(self):
        from pi_jwm.formal_system_outcome_metrics_v1 import compute_system_outcome_metrics

        report = compute_system_outcome_metrics(
            true_completion_event=np.asarray([[True]]),
            predicted_completion_event=np.asarray([[True]]),
            true_completed_delay=np.asarray([[1.0]]),
            predicted_task_delay=np.asarray([[1.0]]),
            completed_delay_valid=np.asarray([[True]]),
            true_delivered_data=np.asarray([1.0]),
            predicted_delivered_data=np.asarray([1.0]),
            step_seconds=0.1,
            true_uav_energy=np.asarray([[1.0]]),
            predicted_uav_energy=np.asarray([[1.0]]),
            uav_energy_valid=np.asarray([[True]]),
            true_source_service=np.asarray([[1.0, 1.0]]),
            predicted_source_service=np.asarray([[1.0, 0.0]]),
            source_population_valid=np.asarray([True, True]),
            source_evaluable_task_count=np.asarray([1.0, 2.0]),
        )

        self.assertAlmostEqual(
            0.4,
            report["metrics"]["system.completion_fairness_jain.absolute_error"]["value"],
        )


if __name__ == "__main__":
    unittest.main()
