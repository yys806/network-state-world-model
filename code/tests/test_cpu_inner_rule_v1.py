from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pi_jwm.cpu_inner_rule_v1 import (  # noqa: E402
    CPU_INNER_RULE_VERSION,
    CpuTaskDemand,
    allocate_work_conserving_cpu,
)


class CpuInnerRuleV1Test(unittest.TestCase):
    def allocate(self, tasks, capacities, slot_seconds=1.0):
        return allocate_work_conserving_cpu(tasks, capacities, slot_seconds)

    def test_empty_input_has_empty_auditable_output(self) -> None:
        decision = self.allocate([], {}, slot_seconds=0.5)
        self.assertEqual(decision.rule_version, CPU_INNER_RULE_VERSION)
        self.assertEqual(decision.slot_seconds, 0.5)
        self.assertEqual(decision.allocations, ())
        self.assertEqual(decision.node_summaries, ())
        self.assertEqual(decision.as_allocation_dict(), {})

    def test_zero_capacity_retains_tasks_with_zero_allocations(self) -> None:
        tasks = [
            CpuTaskDemand("task_b", "node_0", 2.0),
            CpuTaskDemand("task_a", "node_0", 1.0),
        ]
        decision = self.allocate(tasks, {"node_0": 0.0})
        self.assertEqual(tuple(row.task_id for row in decision.allocations), ("task_a", "task_b"))
        self.assertEqual(decision.as_allocation_dict(), {"task_a": 0.0, "task_b": 0.0})
        self.assertEqual(decision.node_summaries[0].total_allocated_cpu, 0.0)

    def test_demand_below_capacity_is_capped_by_remaining_work(self) -> None:
        tasks = [
            CpuTaskDemand("task_a", "node_0", 1.0),
            CpuTaskDemand("task_b", "node_0", 2.0),
        ]
        decision = self.allocate(tasks, {"node_0": 10.0}, slot_seconds=1.0)
        self.assertEqual(decision.as_allocation_dict(), {"task_a": 1.0, "task_b": 2.0})
        self.assertEqual(decision.node_summaries[0].total_allocated_cpu, 3.0)

    def test_demand_above_capacity_uses_equal_water_level(self) -> None:
        tasks = [
            CpuTaskDemand("task_a", "node_0", 10.0),
            CpuTaskDemand("task_b", "node_0", 10.0),
        ]
        decision = self.allocate(tasks, {"node_0": 6.0})
        self.assertEqual(decision.as_allocation_dict(), {"task_a": 3.0, "task_b": 3.0})
        self.assertEqual(decision.node_summaries[0].water_level, 3.0)

    def test_small_demand_is_saturated_before_equal_sharing_remainder(self) -> None:
        tasks = [
            CpuTaskDemand("task_c", "node_0", 10.0),
            CpuTaskDemand("task_a", "node_0", 1.0),
            CpuTaskDemand("task_b", "node_0", 10.0),
        ]
        decision = self.allocate(tasks, {"node_0": 7.0})
        self.assertEqual(decision.as_allocation_dict(), {"task_a": 1.0, "task_b": 3.0, "task_c": 3.0})

    def test_multiple_nodes_are_independent_and_stably_sorted(self) -> None:
        tasks = [
            CpuTaskDemand("task_z", "node_b", 9.0),
            CpuTaskDemand("task_b", "node_a", 4.0),
            CpuTaskDemand("task_a", "node_a", 4.0),
        ]
        decision = self.allocate(tasks, {"node_b": 2.0, "node_a": 4.0})
        self.assertEqual(
            tuple((row.node_id, row.task_id) for row in decision.allocations),
            (("node_a", "task_a"), ("node_a", "task_b"), ("node_b", "task_z")),
        )
        self.assertEqual(decision.as_allocation_dict(), {"task_a": 2.0, "task_b": 2.0, "task_z": 2.0})

    def test_repeated_calls_are_identical(self) -> None:
        tasks = [
            CpuTaskDemand("task_c", "node_0", 1.0 / 3.0),
            CpuTaskDemand("task_a", "node_0", 2.0 / 3.0),
            CpuTaskDemand("task_b", "node_0", 7.0 / 3.0),
        ]
        first = self.allocate(tasks, {"node_0": 1.7}, slot_seconds=0.7)
        second = self.allocate(reversed(tasks), {"node_0": 1.7}, slot_seconds=0.7)
        self.assertEqual(first, second)

    def test_conservation_for_mixed_floating_point_demands(self) -> None:
        tasks = [
            CpuTaskDemand("task_a", "node_0", 0.1),
            CpuTaskDemand("task_b", "node_0", 0.2),
            CpuTaskDemand("task_c", "node_0", 0.9),
        ]
        slot_seconds = 0.3
        capacity = 2.0
        decision = self.allocate(tasks, {"node_0": capacity}, slot_seconds)
        allocations = decision.allocations
        self.assertTrue(all(math.isfinite(row.allocated_cpu) and row.allocated_cpu >= 0.0 for row in allocations))
        self.assertTrue(all(row.served_work <= row.remaining_work + 1e-12 for row in allocations))
        self.assertLessEqual(math.fsum(row.allocated_cpu for row in allocations), capacity + 1e-12)
        self.assertAlmostEqual(math.fsum(row.allocated_cpu for row in allocations), capacity, places=12)

    def test_candidate_post_communication_task_set_changes_allocation(self) -> None:
        candidate_local = [CpuTaskDemand("task_a", "vehicle_0", 4.0)]
        candidate_offload = [CpuTaskDemand("task_a", "rsu_0", 4.0)]
        capacities = {"vehicle_0": 1.0, "rsu_0": 3.0}
        local = self.allocate(candidate_local, capacities)
        offload = self.allocate(candidate_offload, capacities)
        self.assertEqual(local.as_allocation_dict()["task_a"], 1.0)
        self.assertEqual(offload.as_allocation_dict()["task_a"], 3.0)
        self.assertNotEqual(local.allocations, offload.allocations)

    def test_invalid_inputs_fail_fast(self) -> None:
        valid = [CpuTaskDemand("task_a", "node_0", 1.0)]
        invalid_tasks = (
            [CpuTaskDemand("", "node_0", 1.0)],
            [CpuTaskDemand("task_a", "", 1.0)],
            [CpuTaskDemand("task_a", "node_0", -1.0)],
            [CpuTaskDemand("task_a", "node_0", float("nan"))],
        )
        for tasks in invalid_tasks:
            with self.subTest(tasks=tasks), self.assertRaises(ValueError):
                self.allocate(tasks, {"node_0": 1.0})
        for slot in (0.0, -1.0, float("inf")):
            with self.subTest(slot=slot), self.assertRaises(ValueError):
                self.allocate(valid, {"node_0": 1.0}, slot_seconds=slot)
        for capacity in (-1.0, float("nan")):
            with self.subTest(capacity=capacity), self.assertRaises(ValueError):
                self.allocate(valid, {"node_0": capacity})

    def test_duplicate_task_id_and_missing_capacity_are_rejected(self) -> None:
        duplicates = [
            CpuTaskDemand("task_a", "node_0", 1.0),
            CpuTaskDemand("task_a", "node_1", 1.0),
        ]
        with self.assertRaisesRegex(ValueError, "duplicate task_id"):
            self.allocate(duplicates, {"node_0": 1.0, "node_1": 1.0})
        with self.assertRaisesRegex(ValueError, "missing CPU capacity"):
            self.allocate([CpuTaskDemand("task_a", "node_missing", 1.0)], {"node_0": 1.0})

    def test_nonfinite_individual_or_aggregate_demand_rate_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "demand_rate"):
            self.allocate(
                [CpuTaskDemand("task_a", "node_0", 1e308)],
                {"node_0": 1.0},
                slot_seconds=1e-308,
            )
        with self.assertRaisesRegex(ValueError, "total demand_rate"):
            self.allocate(
                [
                    CpuTaskDemand("task_a", "node_0", 1e308),
                    CpuTaskDemand("task_b", "node_0", 1e308),
                ],
                {"node_0": 1e308},
                slot_seconds=1.0,
            )


if __name__ == "__main__":
    unittest.main()
