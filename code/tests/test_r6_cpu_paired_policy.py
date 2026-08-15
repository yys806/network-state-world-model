from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pi_jwm.r6_cpu_paired_policy import (  # noqa: E402
    PAIRED_CPU_POLICY_IDS,
    PairedCpuPolicyAllocator,
    project_cpu_allocations,
)


class _Node:
    def __init__(self, capacity: float) -> None:
        self.capacity = capacity

    def getFogProfile(self):
        return {"cpu": self.capacity}


class _Env:
    simulation_time = 1.0

    def __init__(self) -> None:
        self.nodes = {"RSU_0": _Node(10.0)}

    def _getNodeById(self, node_id: str):
        return self.nodes.get(node_id)


class _Task:
    def __init__(self, task_id: str, deadline: float, cpu: float) -> None:
        self.task_id = task_id
        self.deadline = deadline
        self.cpu = cpu

    def getTaskId(self):
        return self.task_id

    def getTaskArrivalTime(self):
        return 0.0

    def getTaskDeadline(self):
        return self.deadline

    def getTaskCPU(self):
        return self.cpu

    def getComputedSize(self):
        return 0.0


class R6PairedCpuPolicyTest(unittest.TestCase):
    def test_policy_ids_are_explicit_and_allocator_is_deterministic(self) -> None:
        self.assertEqual(
            PAIRED_CPU_POLICY_IDS,
            ("equal_share", "deadline_aware", "feasible_exploration", "local_search"),
        )
        tasks = {"RSU_0": [_Task("task_b", 10.0, 1.0), _Task("task_a", 2.0, 1.0)]}
        first = PairedCpuPolicyAllocator("local_search", seed=7).allocate(_Env(), tasks)
        second = PairedCpuPolicyAllocator("local_search", seed=7).allocate(_Env(), tasks)
        self.assertEqual(first.allocations, second.allocations)
        self.assertEqual(first.rows, second.rows)

    def test_projection_is_finite_nonnegative_and_capacity_safe(self) -> None:
        result = project_cpu_allocations({"a": 3.0, "b": 1.0}, 2.0)
        self.assertTrue(all(math.isfinite(value) and value >= 0 for value in result.values()))
        self.assertLessEqual(sum(result.values()), 2.0 + 1e-12)
        with self.assertRaisesRegex(ValueError, "capacity"):
            project_cpu_allocations({"a": 1.0}, -1.0)

    def test_all_policies_produce_audit_rows_with_no_oversubscription(self) -> None:
        tasks = {"RSU_0": [_Task("task_a", 2.0, 1.0), _Task("task_b", 10.0, 1.0)]}
        for policy_id in PAIRED_CPU_POLICY_IDS:
            decision = PairedCpuPolicyAllocator(policy_id, seed=3).allocate(_Env(), tasks)
            self.assertEqual({row["policy_id"] for row in decision.rows}, {policy_id})
            self.assertLessEqual(sum(decision.allocations.values()), 10.0 + 1e-12)

    def test_unknown_policy_and_nonfinite_inputs_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown"):
            PairedCpuPolicyAllocator("bad", seed=0)
        with self.assertRaisesRegex(ValueError, "finite"):
            project_cpu_allocations({"a": float("nan")}, 2.0)


if __name__ == "__main__":
    unittest.main()
