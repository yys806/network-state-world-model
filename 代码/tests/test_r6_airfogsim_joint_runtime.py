from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pi_jwm.r6_airfogsim_joint_runtime import (  # noqa: E402
    apply_prepared_candidate,
    prepare_joint_action_step,
)
from pi_jwm.r6_joint_action import CPUAllocation, JointActionCandidate  # noqa: E402


class FakeTask:
    def __init__(self, task_id: str, node_id: str, *, assigned: str | None = None) -> None:
        self.task_id = task_id
        self.node_id = node_id
        self.assigned = assigned
        self.current = assigned or node_id
        self.target = "UAV_0"

    def getTaskId(self):
        return self.task_id

    def getTaskNodeId(self):
        return self.node_id

    def getAssignedTo(self):
        return self.assigned

    def getCurrentNodeId(self):
        return self.current

    def getTaskDeadline(self):
        return 0.5 if self.task_id == "Task_3" else 1.0

    def getTaskArrivalTime(self):
        return 0.0

    def getTaskPriority(self):
        return 2.0 if self.task_id == "Task_3" else 1.0

    def getTaskSize(self):
        return 2.0

    def isComputing(self):
        return self.assigned is not None

    def isComputed(self):
        return False

    def changeOffloadTo(self, target, route, time):
        self.target = target


class FakeTaskManager:
    def __init__(self) -> None:
        self.offload = FakeTask("Task_1", "vehicle_0")
        self.compute = [
            FakeTask("Task_3", "vehicle_2", assigned="UAV_0"),
            FakeTask("Task_4", "vehicle_3", assigned="UAV_0"),
        ]

    def getComputingTasks(self):
        return {"UAV_0": list(self.compute)}

    def getTaskByTaskId(self, task_id):
        if task_id == "Task_1":
            return self.offload
        for task in self.compute:
            if task.task_id == task_id:
                return task
        return None


class FakeTaskScheduler:
    def __init__(self, events):
        self.events = events

    def getAllToOffloadTaskInfos(self, env, check_dependency=False):
        self.events.append(f"capture_dependency_{check_dependency}")
        return [
            {
                "task_id": "Task_1",
                "task_node_id": "vehicle_0",
                "task_deadline": 1.0,
                "task_arrival_time": 0.0,
                "task_priority": 2.0,
                "task_size": 2.0,
            }
        ]


class FakeEntityScheduler:
    def getNeighborNodeInfosById(self, env, node_id, sorted_by="distance", max_num=10):
        return [
            {"id": "UAV_0", "fog_profile": {"cpu": 12.0}},
            {"id": "RSU_0", "fog_profile": {"cpu": 50.0}},
        ]

    def getDistanceBetweenNodes(self, env, source, target):
        return {"UAV_0": 10.0, "RSU_0": 20.0}[target]

    def getNodeInfoById(self, env, node_id):
        return {"id": node_id, "fog_profile": {"cpu": 12.0}}


class FakeCommScheduler:
    def __init__(self, events):
        self.events = events

    def getNumberOfRB(self, env):
        return 6

    def getEstimatedRateBetweenNodeIds(self, env, source, target):
        rate = 4.0 if target[0] == "UAV_0" else 8.0
        return np.asarray([[rate]], dtype=np.float64), 0.0

    def setCommunicationWithRB(self, env, task_id, rb_ids):
        self.events.append(f"rb_{task_id}")
        env.activated_offloading_tasks_with_RB_Nos[task_id] = list(rb_ids)


class FakeCompScheduler:
    def __init__(self, events):
        self.events = events

    def setComputingCallBack(self, env, callback):
        self.events.append("cpu_callback")
        env.alloc_cpu_callback = callback


class FakeEnv:
    def __init__(self, events) -> None:
        self.events = events
        self.simulation_time = 0.2
        self.task_manager = FakeTaskManager()
        self.activated_offloading_tasks_with_RB_Nos = {}
        self.cpu_callback_count = 0

        def baseline_callback(computing_tasks, **kwargs):
            self.cpu_callback_count += 1
            return {
                task.getTaskId(): 6.0
                for tasks in computing_tasks.values()
                for task in tasks
            }

        self.alloc_cpu_callback = baseline_callback

    def _getNodeById(self, node_id):
        if node_id != "UAV_0":
            return None

        class Node:
            @staticmethod
            def getFogProfile():
                return {"cpu": 12.0}

        return Node()


class FakeAlgorithm:
    def __init__(self, events) -> None:
        self.events = events
        self.taskScheduler = FakeTaskScheduler(events)
        self.entityScheduler = FakeEntityScheduler()
        self.commScheduler = FakeCommScheduler(events)
        self.compScheduler = FakeCompScheduler(events)
        self.offload_rows = []
        self.rb_rows = []
        self.cpu_rows = []
        self.seed = 507

    def scheduleStep(self, env):
        self.events.append("schedule_default")
        env.activated_offloading_tasks_with_RB_Nos = {"Task_1": [0, 1]}
        self.offload_rows.append(
            {
                "seed": self.seed,
                "time": 0.3,
                "task_id": "Task_1",
                "source_node_id": "vehicle_0",
                "target_node_id": "UAV_0",
            }
        )
        self.rb_rows.append(
            {
                "seed": self.seed,
                "time": 0.3,
                "task_id": "Task_1",
                "current_node_id": "vehicle_0",
                "assigned_to": "UAV_0",
                "rb_count": 2,
                "rb_indices": "0 1",
            }
        )


class R6AirFogSimJointRuntimeTest(unittest.TestCase):
    def test_prepare_captures_dag_before_default_and_builds_complete_candidates(self) -> None:
        events: list[str] = []
        env = FakeEnv(events)
        algorithm = FakeAlgorithm(events)
        prepared = prepare_joint_action_step(
            env,
            algorithm,
            scenario_id="load_high__density_dense__r07",
            seed=507,
            slot=2,
            split="validation",
            max_candidates=6,
        )
        self.assertEqual(["capture_dependency_True", "schedule_default"], events)
        self.assertEqual(6, len(prepared.candidates.candidates))
        self.assertEqual({"Task_1"}, {row.task_id for row in prepared.context.default_rb_plan})
        self.assertEqual({"Task_3", "Task_4"}, {row.task_id for row in prepared.context.compute_tasks})

    def test_apply_uses_offload_then_rb_then_cpu_and_respects_capacity(self) -> None:
        events: list[str] = []
        env = FakeEnv(events)
        algorithm = FakeAlgorithm(events)
        prepared = prepare_joint_action_step(
            env,
            algorithm,
            scenario_id="load_high__density_dense__r07",
            seed=507,
            slot=2,
            split="validation",
            max_candidates=6,
        )
        selected = next(
            index
            for index, candidate in enumerate(prepared.candidates.candidates)
            if candidate.template_id == "rate_aware"
        )
        record = apply_prepared_candidate(env, algorithm, prepared, candidate_index=selected)
        self.assertEqual(0, env.cpu_callback_count)
        self.assertEqual("RSU_0", env.task_manager.offload.target)
        self.assertEqual([0, 1, 2, 3, 4, 5], env.activated_offloading_tasks_with_RB_Nos["Task_1"])
        allocation = env.alloc_cpu_callback(env.task_manager.getComputingTasks())
        self.assertEqual(1, env.cpu_callback_count)
        self.assertAlmostEqual(12.0, sum(allocation.values()))
        self.assertEqual("RSU_0", algorithm.offload_rows[-1]["target_node_id"])
        self.assertEqual("0 1 2 3 4 5", algorithm.rb_rows[-1]["rb_indices"])
        self.assertEqual({"Task_3", "Task_4"}, {row["task_id"] for row in algorithm.cpu_rows})
        self.assertTrue(all(row["action_source"] == "selected_joint_candidate" for row in algorithm.cpu_rows))
        self.assertEqual(0, record.hard_violation_count)
        self.assertTrue(record.offload_changed)
        self.assertTrue(record.rb_changed)
        self.assertTrue(record.cpu_changed)
        self.assertLess(events.index("schedule_default"), events.index("rb_Task_1"))
        self.assertLess(events.index("rb_Task_1"), events.index("cpu_callback"))

    def test_invalid_candidate_is_rejected_before_runtime_mutation(self) -> None:
        events: list[str] = []
        env = FakeEnv(events)
        algorithm = FakeAlgorithm(events)
        prepared = prepare_joint_action_step(
            env,
            algorithm,
            scenario_id="load_high__density_dense__r07",
            seed=507,
            slot=2,
            split="validation",
            max_candidates=6,
        )
        base = prepared.candidates.candidates[1]
        bad = JointActionCandidate.create(
            candidate_id="bad_cpu",
            template_id="deadline_first",
            offload=base.offload,
            rb=base.rb,
            cpu=(CPUAllocation("UAV_0", "Task_3", 20.0), CPUAllocation("UAV_0", "Task_4", 20.0)),
        )
        prepared = prepared.with_candidates((prepared.candidates.candidates[0], bad))
        before_events = list(events)
        with self.assertRaisesRegex(ValueError, "CPU capacity"):
            apply_prepared_candidate(env, algorithm, prepared, candidate_index=1)
        self.assertEqual(before_events, events)
        self.assertEqual("UAV_0", env.task_manager.offload.target)


if __name__ == "__main__":
    unittest.main()
