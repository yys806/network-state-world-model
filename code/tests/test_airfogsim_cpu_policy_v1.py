from __future__ import annotations

import sys
import unittest
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class FakeTask:
    def __init__(
        self,
        task_id: str,
        *,
        arrival: float,
        deadline: float,
        task_cpu: float = 10.0,
        computed: float = 0.0,
        priority: float = 1.0,
    ) -> None:
        self.task_id = task_id
        self.arrival = arrival
        self.deadline = deadline
        self.task_cpu = task_cpu
        self.computed = computed
        self.priority = priority

    def getTaskId(self) -> str:
        return self.task_id

    def getTaskArrivalTime(self) -> float:
        return self.arrival

    def getTaskDeadline(self) -> float:
        return self.deadline

    def getTaskCPU(self) -> float:
        return self.task_cpu

    def getComputedSize(self) -> float:
        return self.computed

    def getTaskPriority(self) -> float:
        return self.priority


class FakeNode:
    def __init__(self, cpu: float) -> None:
        self.cpu = cpu

    def getFogProfile(self) -> dict[str, float]:
        return {"cpu": self.cpu}


class FakeEnv:
    def __init__(self, capacities: dict[str, float], current_time: float = 1.0) -> None:
        self.nodes = {node_id: FakeNode(cpu) for node_id, cpu in capacities.items()}
        self.simulation_time = current_time

    def _getNodeById(self, node_id: str):
        return self.nodes.get(node_id)


def task_set() -> dict[str, list[FakeTask]]:
    return {
        "RSU_0": [
            FakeTask("urgent", arrival=0.0, deadline=1.2),
            FakeTask("relaxed", arrival=0.0, deadline=10.0),
            FakeTask("middle", arrival=0.0, deadline=4.0),
        ]
    }


class CpuPolicyAllocatorTests(unittest.TestCase):
    def test_all_policies_are_capacity_safe_and_deterministic(self):
        from pi_jwm.airfogsim_cpu_policy_v1 import CPU_POLICY_IDS, CpuPolicyAllocator

        env = FakeEnv({"RSU_0": 10.0})
        tasks = task_set()
        for policy in CPU_POLICY_IDS:
            with self.subTest(policy=policy):
                left = CpuPolicyAllocator(policy, seed=7).allocate(env, tasks)
                right = CpuPolicyAllocator(policy, seed=7).allocate(env, tasks)
                self.assertEqual(left.allocations, right.allocations)
                self.assertLessEqual(sum(left.allocations.values()), 10.0 + 1e-9)
                self.assertTrue(all(value >= 0.0 for value in left.allocations.values()))
                self.assertEqual(3, len(left.rows))

    def test_deadline_aware_prioritizes_urgent_task(self):
        from pi_jwm.airfogsim_cpu_policy_v1 import CpuPolicyAllocator

        result = CpuPolicyAllocator("deadline_aware", seed=0).allocate(
            FakeEnv({"RSU_0": 10.0}),
            task_set(),
        )

        self.assertGreater(result.allocations["urgent"], result.allocations["middle"])
        self.assertGreater(result.allocations["middle"], result.allocations["relaxed"])

    def test_exploration_is_nonuniform_and_keeps_auditable_rows(self):
        from pi_jwm.airfogsim_cpu_policy_v1 import CpuPolicyAllocator

        result = CpuPolicyAllocator("feasible_exploration", seed=11).allocate(
            FakeEnv({"RSU_0": 9.0}),
            task_set(),
        )

        rounded = {round(value, 8) for value in result.allocations.values()}
        self.assertGreater(len(rounded), 1)
        self.assertEqual(
            {
                "allocated_cpu",
                "allocated_fraction",
                "deadline_remaining",
                "node_cpu_capacity",
                "node_id",
                "policy_id",
                "policy_weight",
                "queue_size",
                "task_id",
            },
            set(result.rows[0]),
        )

    def test_policies_are_independent_across_nodes(self):
        from pi_jwm.airfogsim_cpu_policy_v1 import CpuPolicyAllocator

        tasks = {
            "vehicle_0": [FakeTask("v0", arrival=0.0, deadline=2.0)],
            "RSU_0": [FakeTask("r0", arrival=0.0, deadline=2.0)],
        }
        result = CpuPolicyAllocator("equal_share", seed=0).allocate(
            FakeEnv({"vehicle_0": 2.0, "RSU_0": 10.0}),
            tasks,
        )

        self.assertEqual({"v0": 2.0, "r0": 10.0}, result.allocations)

    def test_zero_capacity_and_unknown_nodes_produce_no_action(self):
        from pi_jwm.airfogsim_cpu_policy_v1 import CpuPolicyAllocator

        result = CpuPolicyAllocator("equal_share", seed=0).allocate(
            FakeEnv({"zero": 0.0}),
            {
                "zero": [FakeTask("z0", arrival=0.0, deadline=2.0)],
                "missing": [FakeTask("m0", arrival=0.0, deadline=2.0)],
            },
        )

        self.assertEqual({}, result.allocations)
        self.assertEqual([], result.rows)

    def test_unknown_policy_and_nonfinite_capacity_are_rejected(self):
        from pi_jwm.airfogsim_cpu_policy_v1 import CpuPolicyAllocator

        with self.assertRaises(ValueError):
            CpuPolicyAllocator("unknown", seed=0)
        with self.assertRaises(ValueError):
            CpuPolicyAllocator("equal_share", seed=0).allocate(
                FakeEnv({"RSU_0": float("nan")}),
                task_set(),
            )


if __name__ == "__main__":
    unittest.main()
