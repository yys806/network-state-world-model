from __future__ import annotations

import sys
import unittest
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
AIRFOGSIM_ROOT = CODE_ROOT / "reference" / "AirFogSim"
for path in (SRC_ROOT,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pi_jwm.airfogsim_cpu_inner_rule_v1 import (  # noqa: E402
    allocate_airfogsim_cpu,
    load_airfogsim_task_class_from_source,
    make_airfogsim_cpu_callback,
)
from pi_jwm.cpu_inner_rule_v1 import CpuTaskDemand, allocate_work_conserving_cpu  # noqa: E402


class _Node:
    def __init__(self, capacity: float | None) -> None:
        self.capacity = capacity

    def getFogProfile(self):
        return {} if self.capacity is None else {"cpu": self.capacity}


class _Env:
    simulation_interval = 0.5

    def __init__(self, capacities: dict[str, float | None]) -> None:
        self.nodes = {node_id: _Node(capacity) for node_id, capacity in capacities.items()}

    def _getNodeById(self, node_id: str):
        return self.nodes.get(node_id)


Task = load_airfogsim_task_class_from_source(AIRFOGSIM_ROOT)


def _task(task_id: str, node_id: str, total_cpu: float, computed_cpu: float = 0.0):
    task = Task(
        task_id=task_id,
        task_node_id=node_id,
        task_cpu=total_cpu,
        task_size=1.0,
        task_deadline=10.0,
        task_priority=1.0,
        task_arrival_time=0.0,
    )
    task.setAssignedTo(node_id)
    task.setAttribute("_routes", [node_id])
    task.setAttribute("_computed_size", computed_cpu)
    task.startToCompute(0.0)
    return task


class AirFogSimCpuInnerRuleV1Test(unittest.TestCase):
    def test_real_task_callback_matches_pure_rule(self) -> None:
        env = _Env({"RSU_0": 6.0})
        tasks = {
            "RSU_0": [
                _task("task_b", "RSU_0", total_cpu=5.0, computed_cpu=1.0),
                _task("task_a", "RSU_0", total_cpu=1.0),
            ]
        }
        callback_result = allocate_airfogsim_cpu(env, tasks, slot_seconds=0.5)
        pure_result = allocate_work_conserving_cpu(
            [
                CpuTaskDemand("task_b", "RSU_0", 4.0),
                CpuTaskDemand("task_a", "RSU_0", 1.0),
            ],
            {"RSU_0": 6.0},
            0.5,
        )
        self.assertEqual(callback_result.decision, pure_result)
        self.assertEqual(callback_result.allocations, pure_result.as_allocation_dict())
        self.assertEqual(callback_result.source_task_classes, ("airfogsim.entities.task.Task",))

    def test_callback_signature_uses_environment_interval_by_default(self) -> None:
        env = _Env({"RSU_0": 2.0})
        tasks = {"RSU_0": [_task("task_a", "RSU_0", total_cpu=3.0)]}
        callback = make_airfogsim_cpu_callback(env)
        self.assertEqual(callback(tasks), {"task_a": 2.0})

    def test_assignment_and_current_node_mismatch_are_rejected(self) -> None:
        env = _Env({"RSU_0": 2.0, "RSU_1": 2.0})
        assigned_elsewhere = _task("task_a", "RSU_0", total_cpu=1.0)
        assigned_elsewhere.setAssignedTo("RSU_1")
        with self.assertRaisesRegex(ValueError, "assigned node mismatch"):
            allocate_airfogsim_cpu(env, {"RSU_0": [assigned_elsewhere]})

        located_elsewhere = _task("task_b", "RSU_0", total_cpu=1.0)
        located_elsewhere.setAttribute("_routes", ["RSU_1"])
        with self.assertRaisesRegex(ValueError, "current node mismatch"):
            allocate_airfogsim_cpu(env, {"RSU_0": [located_elsewhere]})

    def test_missing_node_and_missing_capacity_are_rejected(self) -> None:
        task = _task("task_a", "RSU_0", total_cpu=1.0)
        with self.assertRaisesRegex(ValueError, "node not found"):
            allocate_airfogsim_cpu(_Env({}), {"RSU_0": [task]})
        with self.assertRaisesRegex(ValueError, "missing CPU capacity"):
            allocate_airfogsim_cpu(_Env({"RSU_0": None}), {"RSU_0": [task]})

    def test_overcomputed_task_is_rejected_instead_of_clamped(self) -> None:
        task = _task("task_a", "RSU_0", total_cpu=1.0, computed_cpu=1.5)
        with self.assertRaisesRegex(ValueError, "computed CPU exceeds total CPU"):
            allocate_airfogsim_cpu(_Env({"RSU_0": 2.0}), {"RSU_0": [task]})


if __name__ == "__main__":
    unittest.main()
