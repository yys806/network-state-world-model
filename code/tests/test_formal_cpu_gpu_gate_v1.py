from __future__ import annotations

import sys
import unittest
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class FormalCpuGpuGateV1Tests(unittest.TestCase):
    def test_gate_requires_three_seeds_and_reports_each_failed_condition(self):
        from pi_jwm.formal_cpu_gpu_gate_v1 import audit_cpu_to_gpu_gate

        rows = [
            {
                "seed": 1,
                "threshold_selection_split": "calibration",
                "validation_link_f1": 0.70,
                "validation_persistence_link_f1": 0.68,
                "calibration_link_f1": 0.30,
                "calibration_persistence_link_f1": 0.20,
                "validation_node_x_mae": 1.0,
                "validation_persistence_node_x_mae": 1.0,
                "validation_throughput_mae": 1.0,
                "validation_persistence_throughput_mae": 1.0,
                "validation_rb_occupancy_mae": 1.0,
                "validation_persistence_rb_occupancy_mae": 1.0,
                "validation_task_delay_mae": 1.0,
                "validation_persistence_task_delay_mae": 1.0,
            },
            {
                "seed": 2,
                "threshold_selection_split": "validation",
                "validation_link_f1": 0.70,
                "validation_persistence_link_f1": 0.68,
                "calibration_link_f1": 0.30,
                "calibration_persistence_link_f1": 0.20,
                "validation_node_x_mae": 1.0,
                "validation_persistence_node_x_mae": 1.0,
                "validation_throughput_mae": 1.0,
                "validation_persistence_throughput_mae": 1.0,
                "validation_rb_occupancy_mae": 1.0,
                "validation_persistence_rb_occupancy_mae": 1.0,
                "validation_task_delay_mae": 1.0,
                "validation_persistence_task_delay_mae": 1.0,
            },
        ]
        report = audit_cpu_to_gpu_gate(rows)

        self.assertFalse(report["gpu_allowed"])
        self.assertIn("minimum_independent_seed_count", report["failed_gates"])
        self.assertIn("calibration_only_threshold_selection", report["failed_gates"])
        self.assertEqual(2, report["observed"]["independent_seed_count"])

    def test_gate_passes_only_when_all_frozen_conditions_hold(self):
        from pi_jwm.formal_cpu_gpu_gate_v1 import audit_cpu_to_gpu_gate

        rows = []
        for seed in (1, 2, 3):
            rows.append(
                {
                    "seed": seed,
                    "threshold_selection_split": "calibration",
                    "validation_link_f1": 0.72,
                    "validation_persistence_link_f1": 0.68,
                    "calibration_link_f1": 0.30,
                    "calibration_persistence_link_f1": 0.20,
                    "validation_node_x_mae": 1.1,
                    "validation_persistence_node_x_mae": 1.0,
                    "validation_throughput_mae": 0.9,
                    "validation_persistence_throughput_mae": 1.0,
                    "validation_rb_occupancy_mae": 0.9,
                    "validation_persistence_rb_occupancy_mae": 1.0,
                    "validation_task_delay_mae": 0.9,
                    "validation_persistence_task_delay_mae": 1.0,
                }
            )
        report = audit_cpu_to_gpu_gate(rows)

        self.assertTrue(report["gpu_allowed"])
        self.assertEqual([], report["failed_gates"])


if __name__ == "__main__":
    unittest.main()
