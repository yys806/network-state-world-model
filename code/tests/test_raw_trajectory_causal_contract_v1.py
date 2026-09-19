from __future__ import annotations

import sys
import unittest
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pi_jwm.raw_trajectory_causal_contract_v1 import (  # noqa: E402
    aggregate_slot_outcomes,
    attach_canonical_acceleration,
    partition_tasks_at_decision,
)


class CausalTaskBoundaryTests(unittest.TestCase):
    def test_future_tasks_are_internal_metadata_not_observable_tasks(self):
        tasks = [
            {"task_id": "arrived", "arrival_time_s": 2.4, "lifecycle": "waiting_to_offload"},
            {"task_id": "future", "arrival_time_s": 3.2, "lifecycle": "to_generate"},
        ]

        observable, internal_future = partition_tasks_at_decision(tasks, 2.4)

        self.assertEqual(["arrived"], [row["task_id"] for row in observable])
        self.assertEqual(["future"], [row["task_id"] for row in internal_future])
        self.assertEqual("internal_future_schedule", internal_future[0]["visibility"])


class CanonicalAccelerationTests(unittest.TestCase):
    def test_first_observation_is_masked_and_later_value_uses_only_history(self):
        first = attach_canonical_acceleration(
            [{"entity_id": "uav-0", "speed_mps": 10.0, "raw_simulator_acceleration_mps2": -100.0}],
            previous_speed_by_entity=None,
            delta_t_s=None,
        )[0]
        self.assertEqual(-100.0, first["raw_simulator_acceleration_mps2"])
        self.assertIsNone(first["canonical_acceleration_mps2"])
        self.assertFalse(first["canonical_acceleration_observed_mask"])
        self.assertEqual(
            "NO_PREVIOUS_SPEED_IN_TRAJECTORY",
            first["canonical_acceleration_missing_reason"],
        )

        second = attach_canonical_acceleration(
            [{"entity_id": "uav-0", "speed_mps": 11.0, "raw_simulator_acceleration_mps2": -10.0}],
            previous_speed_by_entity={"uav-0": 10.0},
            delta_t_s=0.1,
        )[0]
        self.assertAlmostEqual(10.0, second["canonical_acceleration_mps2"])
        self.assertTrue(second["canonical_acceleration_observed_mask"])
        self.assertIsNone(second["canonical_acceleration_missing_reason"])


class SlotOutcomeTests(unittest.TestCase):
    def test_delivered_data_and_served_cpu_are_aggregated_per_task(self):
        outcome = aggregate_slot_outcomes(
            transfer_events=[
                {"task_id": "task-a", "delivered_data": 0.3},
                {"task_id": "task-a", "delivered_data": 0.2},
            ],
            computed_before={"task-a": 0.1, "task-b": 0.0},
            computed_after={"task-a": 0.4, "task-b": 0.2},
        )
        self.assertAlmostEqual(0.5, outcome["delivered_data_by_task"]["task-a"])
        self.assertAlmostEqual(0.3, outcome["served_cpu_work_by_task"]["task-a"])
        self.assertAlmostEqual(0.2, outcome["served_cpu_work_by_task"]["task-b"])


if __name__ == "__main__":
    unittest.main()
