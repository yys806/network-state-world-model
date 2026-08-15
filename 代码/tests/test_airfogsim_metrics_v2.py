from __future__ import annotations

import sys
import unittest
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pi_jwm.airfogsim_metrics_v2 import compute_airfogsim_metrics_v2


def fixture() -> dict:
    return {
        "task_records": [
            {
                "id": "t0",
                "source": "p0",
                "arrival_time": 0.0,
                "deadline_time": 2.0,
                "priority": 2.0,
                "terminal_status": "completed",
                "task_delay": 1.0,
            },
            {
                "id": "t1",
                "source": "p0",
                "arrival_time": 0.0,
                "deadline_time": 2.0,
                "priority": 1.0,
                "terminal_status": "failed",
                "task_delay": None,
                "failure_reason": "Task fails due to out of deadline.",
            },
            {
                "id": "t2",
                "source": "p1",
                "arrival_time": 3.0,
                "deadline_time": 10.0,
                "priority": 1.0,
                "terminal_status": "pending",
                "task_delay": None,
            },
        ],
        "transfer_events": [
            {"event_id": "e0", "delivered_data": 2.0, "time": 1.0},
            {"event_id": "e1", "delivered_data": 1.0, "time": 2.0},
        ],
        "physical_node_snapshots": [
            {"id": "p0", "cpu": 10.0, "observed_time": 1.0},
            {"id": "p1", "cpu": 10.0, "observed_time": 1.0},
            {"id": "p0", "cpu": 10.0, "observed_time": 2.0},
            {"id": "p1", "cpu": 10.0, "observed_time": 2.0},
            {"id": "p0", "cpu": 10.0, "observed_time": 3.0},
            {"id": "p1", "cpu": 10.0, "observed_time": 3.0},
        ],
        "physical_edge_snapshots": [
            {"id": "e", "observed_time": 1.0, "active_task_count": 1},
            {"id": "e", "observed_time": 2.0, "active_task_count": 0},
            {"id": "e", "observed_time": 3.0, "active_task_count": 1},
        ],
        "rb_ledger": [
            {"record_id": "rb0", "time": 1.0, "n_rb": 4, "rb_indices": [0, 1]},
            {"record_id": "rb1", "time": 2.0, "n_rb": 4, "rb_indices": [0]},
        ],
        "cpu_ledger": [
            {
                "record_id": "cpu0",
                "time": 1.0,
                "node_id": "p0",
                "allocated_cpu": 5.0,
                "node_cpu_capacity": 10.0,
                "dt": 1.0,
            }
        ],
        "uav_energy_ledger": [
            {
                "record_id": "energy0",
                "energy_before": 10.0,
                "energy_after": 8.0,
                "is_flying": 1,
                "fly_unit_cost": 2.0,
                "is_hovering": 0,
                "hover_unit_cost": 1.0,
                "using_sensor_num": 0,
                "sensing_unit_cost": 1.0,
                "sending_data_size": 0.0,
                "send_unit_cost": 0.1,
                "receiving_data_size": 0.0,
                "receive_unit_cost": 0.1,
            }
        ],
        "task_ledger": [
            {
                "record_id": "task0",
                "remaining_before": 2.0,
                "delivered_data": 1.5,
                "remaining_after": 0.5,
            }
        ],
        "task_dag_edges": [{"id": "d0"}, {"id": "d1"}],
        "dependency_data_flows": [],
        "evaluation_end_time": 3.0,
        "simulation_interval": 1.0,
    }


class AirFogSimMetricsV2Tests(unittest.TestCase):
    def test_computes_task_resource_energy_and_constraint_metrics(self):
        report = compute_airfogsim_metrics_v2(fixture())
        metrics = {row["name"]: row for row in report["metrics"]}

        self.assertEqual(0.5, metrics["task_completion_rate"]["value"])
        self.assertEqual(1.0, metrics["successful_task_delay_p99"]["value"])
        self.assertEqual(2.0 / 3.0, metrics["physical_link_active_ratio"]["value"])
        self.assertEqual(0.25, metrics["rb_utilization"]["value"])
        self.assertEqual(5.0 / 60.0, metrics["cpu_utilization"]["value"])
        self.assertEqual(2.0, metrics["uav_energy_total"]["value"])
        self.assertEqual(0.0, metrics["task_flow_conservation_violation_rate"]["value"])
        self.assertEqual("available", metrics["task_completion_rate"]["status"])
        self.assertEqual(2, metrics["task_completion_rate"]["denominator"])

    def test_keeps_unsupported_world_model_metrics_explicit(self):
        report = compute_airfogsim_metrics_v2(fixture())
        metrics = {row["name"]: row for row in report["metrics"]}

        self.assertEqual("not_computable", metrics["uncertainty_coverage"]["status"])
        self.assertIsNone(metrics["uncertainty_coverage"]["value"])
        self.assertEqual("not_computable", metrics["action_regret"]["status"])
        self.assertEqual("not_applicable", metrics["dependency_data_delivery_rate"]["status"])
        self.assertEqual(0.0, metrics["dependency_payload_coverage"]["value"])
        self.assertEqual("service_loss", report["optimization_objectives"]["primary"]["name"])

    def test_right_censored_tasks_are_excluded_from_completion_denominator(self):
        data = fixture()
        data["task_records"].append(
            {
                "id": "t3",
                "source": "p1",
                "arrival_time": 2.5,
                "deadline_time": 9.0,
                "terminal_status": "pending",
                "task_delay": None,
            }
        )

        report = compute_airfogsim_metrics_v2(data)
        metrics = {row["name"]: row for row in report["metrics"]}

        self.assertEqual(2, metrics["task_completion_rate"]["denominator"])
        self.assertEqual(2, report["evaluation_window"]["right_censored_tasks"])


if __name__ == "__main__":
    unittest.main()
