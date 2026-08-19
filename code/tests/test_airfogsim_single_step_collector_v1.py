from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pi_jwm.airfogsim_single_step_collector_v1 import (  # noqa: E402
    SingleStepRecorder,
    execute_candidate,
)
from pi_jwm.single_step_collector_contract_v1 import (  # noqa: E402
    CandidateAction,
    OffloadAction,
    RbAssignment,
)


class FakeTask:
    def __init__(self, task_id="task0", node_id="veh0"):
        self._task_id = task_id
        self._node_id = node_id
        self._computed = 0.0

    def getTaskId(self):
        return self._task_id

    def getAssignedTo(self):
        return self._node_id

    def getCurrentNodeId(self):
        return self._node_id

    def getTaskCPU(self):
        return 2.0

    def getComputedSize(self):
        return self._computed


class FakeNode:
    def getFogProfile(self):
        return {"cpu": 2.0}


class FakeTaskManager:
    def __init__(self):
        self.compute_callback = None

    def invoke_compute_callback(self, computing_tasks):
        return self.compute_callback(computing_tasks)


class FakeEnv:
    simulation_interval = 1.0
    simulation_time = 0.0

    def __init__(self):
        self.task_manager = FakeTaskManager()
        self.nodes = {"veh0": FakeNode()}
        self.calls = []
        self.communication_setter_calls = []
        self.offload_setter_calls = []
        self.activated_offloading_tasks_with_RB_Nos = {}
        self.stepped = False

    def _getNodeById(self, node_id):
        return self.nodes.get(node_id)

    def step(self):
        self.calls.append("env_step")
        self.stepped = True
        if self.task_manager.compute_callback is not None:
            task = FakeTask("task0", "veh0")
            task._computed = 0.5
            self.task_manager.compute_callback({"veh0": [task]})
        return False


class FakeTaskScheduler:
    @staticmethod
    def setTaskOffloading(env, task_node_id, task_id, target_node_id, route=None):
        env.calls.append("offload_setter")
        env.offload_setter_calls.append((task_node_id, task_id, target_node_id, route))
        return True


class FakeCommunicationScheduler:
    @staticmethod
    def setCommunicationWithRB(env, task_id, rb_nos):
        env.calls.append("rb_setter")
        env.communication_setter_calls.append((task_id, list(rb_nos)))
        env.activated_offloading_tasks_with_RB_Nos[task_id] = list(rb_nos)


class FakeComputationScheduler:
    @staticmethod
    def setComputingCallBack(env, callback):
        env.calls.append("cpu_callback_installed")
        env.task_manager.compute_callback = callback


class AirFogSimSingleStepCollectorV1Tests(unittest.TestCase):
    def test_pre_action_observer_runs_after_validation_and_before_any_setter(self):
        env = FakeEnv()
        action = CandidateAction(
            candidate_id="remote",
            offloads=(OffloadAction("veh0", "task0", "veh0", ("veh0",)),),
            rb_assignments=(RbAssignment(0, 0, 0, 0),),
        )

        def observe():
            env.calls.append("decision_time_csi_read")
            return {"channel_attenuation_db": [1.0]}

        result = execute_candidate(
            env,
            action,
            task_ids=("task0",),
            node_ids=("veh0",),
            edge_count=1,
            flow_count=1,
            n_rb=1,
            task_scheduler=FakeTaskScheduler,
            communication_scheduler=FakeCommunicationScheduler,
            computation_scheduler=FakeComputationScheduler,
            pre_action_observer=observe,
        )

        self.assertEqual(
            [
                "decision_time_csi_read",
                "cpu_callback_installed",
                "offload_setter",
                "rb_setter",
                "env_step",
            ],
            env.calls,
        )
        self.assertEqual(
            {"channel_attenuation_db": [1.0]}, result.pre_action_observation
        )
        self.assertEqual(
            (
                "action_validated",
                "decision_time_observation_captured",
                "cpu_callback_installed",
                "action_setters_called",
                "env_step_started",
                "env_step_finished",
            ),
            result.temporal_trace,
        )

    def test_invalid_action_never_calls_pre_action_observer_or_setters(self):
        env = FakeEnv()
        action = CandidateAction(
            candidate_id="remote",
            offloads=(OffloadAction("veh0", "task0", "veh0", ("veh0",)),),
            rb_assignments=(RbAssignment(0, 0, 0, 99),),
        )

        def observe():
            env.calls.append("decision_time_csi_read")
            return {}

        with self.assertRaisesRegex(ValueError, "resource"):
            execute_candidate(
                env,
                action,
                task_ids=("task0",),
                node_ids=("veh0",),
                edge_count=1,
                flow_count=1,
                n_rb=1,
                task_scheduler=FakeTaskScheduler,
                communication_scheduler=FakeCommunicationScheduler,
                computation_scheduler=FakeComputationScheduler,
                pre_action_observer=observe,
            )
        self.assertEqual([], env.calls)

    def test_cpu_callback_receives_candidate_compute_set_and_records_ledger(self):
        env = FakeEnv()
        recorder = SingleStepRecorder(env, candidate_id="local")
        recorder.install_cpu_callback(FakeComputationScheduler)
        allocations = env.task_manager.invoke_compute_callback(
            {"veh0": [FakeTask("task0", "veh0")]}
        )
        self.assertEqual(allocations, {"task0": 2.0})
        self.assertEqual(recorder.cpu_rows[0]["rule_version"], "PIJWM-CPU-Inner-Rule-v1")
        self.assertEqual(recorder.cpu_rows[0]["task_ids"], ["task0"])
        self.assertEqual(recorder.cpu_rows[0]["node_summaries"][0]["capacity"], 2.0)

    def test_cpu_callback_accepts_injected_formal_policy_allocator(self):
        env = FakeEnv()

        def allocator(_env, _tasks):
            return SimpleNamespace(
                allocations={"task0": 1.0},
                rows=[
                    {
                        "task_id": "task0",
                        "node_id": "veh0",
                        "policy_id": "deadline_aware",
                        "policy_weight": 2.0,
                        "deadline_remaining": 1.0,
                        "queue_size": 1,
                        "allocated_cpu": 1.0,
                        "node_cpu_capacity": 2.0,
                        "allocated_fraction": 0.5,
                    }
                ],
            )

        recorder = SingleStepRecorder(env, candidate_id="formal", cpu_allocator=allocator)
        recorder.install_cpu_callback(FakeComputationScheduler)
        allocations = env.task_manager.invoke_compute_callback(
            {"veh0": [FakeTask("task0", "veh0")]}
        )

        self.assertEqual(allocations, {"task0": 1.0})
        self.assertEqual(
            recorder.cpu_rows[0]["policy_rows"][0]["policy_id"],
            "deadline_aware",
        )

    def test_invalid_rb_is_rejected_before_airfogsim_setter(self):
        env = FakeEnv()
        action = CandidateAction(
            candidate_id="remote",
            offloads=(OffloadAction("veh0", "task0", "veh0", ("veh0",)),),
            rb_assignments=(RbAssignment(0, 0, 0, 99),),
        )
        with self.assertRaisesRegex(ValueError, "resource"):
            execute_candidate(
                env,
                action,
                task_ids=("task0",),
                node_ids=("veh0",),
                edge_count=1,
                flow_count=1,
                n_rb=1,
                task_scheduler=FakeTaskScheduler,
                communication_scheduler=FakeCommunicationScheduler,
                computation_scheduler=FakeComputationScheduler,
            )
        self.assertEqual(env.communication_setter_calls, [])
        self.assertEqual(env.offload_setter_calls, [])

    def test_execute_candidate_calls_real_boundaries_and_step(self):
        env = FakeEnv()
        action = CandidateAction(
            candidate_id="local",
            offloads=(OffloadAction("veh0", "task0", "veh0", ("veh0",)),),
            rb_assignments=(),
        )
        result = execute_candidate(
            env,
            action,
            task_ids=("task0",),
            node_ids=("veh0",),
            edge_count=0,
            flow_count=1,
            n_rb=1,
            task_scheduler=FakeTaskScheduler,
            communication_scheduler=FakeCommunicationScheduler,
            computation_scheduler=FakeComputationScheduler,
        )
        self.assertTrue(env.stepped)
        self.assertEqual(result.candidate_id, "local")
        self.assertEqual(env.offload_setter_calls[0][1], "task0")
        self.assertEqual(result.cpu_rows[0]["computed_after"], {"task0": 0.5})


if __name__ == "__main__":
    unittest.main()
