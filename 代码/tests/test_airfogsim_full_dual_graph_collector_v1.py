from __future__ import annotations

import os
import sys
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[1]
for path in (CODE_ROOT / "src", CODE_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pi_jwm.airfogsim_full_dual_graph_collector_v1 import (  # noqa: E402
    execute_full_collector_step,
)
from pi_jwm.airfogsim_full_dual_graph_frame_builder_v1 import (  # noqa: E402
    BuiltFrameDecision,
    build_frame_decision,
)
from pi_jwm.airfogsim_full_dual_graph_observer_v1 import AirFogSimSnapshot  # noqa: E402
from pi_jwm.full_dual_graph_collector_contract_v1 import (  # noqa: E402
    CarryingHop,
    CollectorContractError,
    DecisionRow,
    JointFrameAction,
    LogicalFlow,
    PhysicalEdge,
    PhysicalNode,
    RbAllocation,
    SnapshotPhase,
    TaskLifecycle,
    TaskSnapshot,
)
from pi_jwm.full_dual_graph_vocabulary_v1 import (  # noqa: E402
    FullTrajectoryVocabulary,
    RouteRevisionLedger,
)
import run_p2_single_step_collector_preflight_v1 as p2_runner  # noqa: E402


class FakeNode:
    def __init__(self, cpu=2.0, position=(0.0, 0.0, 0.0)):
        self.cpu = cpu
        self.position = position

    def getFogProfile(self):
        return {"cpu": self.cpu}

    def getPosition(self):
        return self.position


class FakeTask:
    def __init__(self, task_id, source, lifecycle, route=(), destination=None):
        self.task_id = task_id
        self.source = source
        self.current = source
        self.lifecycle = lifecycle
        self.route = list(route)
        self.destination = destination or source
        self.transmitted = 0.0
        self.computed = 0.0

    def getTaskId(self): return self.task_id
    def getTaskNodeId(self): return self.source
    def getCurrentNodeId(self): return self.current
    def getToOffloadRoute(self): return list(self.route)
    def getToReturnNodeId(self): return self.destination
    def getTaskArrivalTime(self): return 0.0
    def getTaskSize(self): return 100.0
    def getReturnedSize(self): return 20.0
    def getTransmittedSize(self): return self.transmitted
    def getTaskCPU(self): return 10.0
    def getComputedSize(self): return self.computed
    def getAssignedTo(self): return self.current
    def isReturning(self): return self.lifecycle in {TaskLifecycle.WAITING_TO_RETURN, TaskLifecycle.RETURNING}


class FakeTaskManager:
    def __init__(self, tasks):
        self.tasks = {task.task_id: task for task in tasks}

    def getTaskByTaskId(self, task_id):
        return self.tasks.get(task_id)


class FakeChannelManager:
    n_RB = 2
    sig2_dB = -114.0
    sig2 = 1e-12
    RB_bandwidth = 2.0

    def __init__(self):
        shape = (2, 1, 2)
        self.V2U_Interference = np.full(shape, 2e-12, dtype=float)
        self.V2U_SINR = np.full(shape, 8.0, dtype=float)
        self.is_V2U_outage = np.zeros(shape, dtype=bool)
        self.U2V_Interference = np.full((1, 2, 2), 2e-12, dtype=float)
        self.U2V_SINR = np.full((1, 2, 2), 8.0, dtype=float)
        self.is_U2V_outage = np.zeros((1, 2, 2), dtype=bool)
        self.transmission_totals = None

    def getCSI(self, tx_idx, rx_idx, tx_type, rx_type):
        del tx_idx, rx_idx, tx_type, rx_type
        return np.asarray([70.0, 71.0], dtype=float)

    def getRateByChannelType(self, tx_idx, rx_idx, channel_type, rb_indices):
        del rx_idx, channel_type
        return np.asarray([5.0 + tx_idx + rb for rb in rb_indices], dtype=float)

    def setThisTimeslotTransSize(self, sending, receiving):
        self.transmission_totals = (dict(sending), dict(receiving))


class FakeEnergyManager:
    _fly_unit_cost = 1.0
    _hover_unit_cost = 0.5
    _sensing_unit_cost = 0.1
    _send_unit_cost = 0.01
    _receive_unit_cost = 0.02

    def __init__(self):
        self._UAVs_energy_info = {}
        self._removed_UAVs_energy_info = {}


class FakeEnv:
    traffic_interval = 0.1
    simulation_interval = 0.1
    simulation_time = 1.0

    def __init__(self, tasks, *, remove_target=False):
        self.tasks = {task.task_id: task for task in tasks}
        self.task_manager = FakeTaskManager(tasks)
        self.vehicles = {"v0": FakeNode(), "v1": FakeNode(position=(1.0, 0.0, 0.0))}
        self.UAVs = {"u0": FakeNode(cpu=5.0, position=(5.0, 0.0, 0.0))}
        self.RSUs = {}
        self.cloudServers = {}
        self.vehicle_ids_as_index = ["v0", "v1"]
        self.uav_ids_as_index = ["u0"]
        self.rsu_ids_as_index = []
        self.cloud_server_ids_as_index = []
        self.channel_manager = FakeChannelManager()
        self.energy_manager = FakeEnergyManager()
        self.alloc_cpu_callback = None
        self.task_return_routes = {}
        self.activated_offloading_tasks_with_RB_Nos = {}
        self.step_calls = 0
        self.trace = []
        self.remove_target = remove_target

    def _getNodeById(self, node_id):
        return self.vehicles.get(node_id) or self.UAVs.get(node_id)

    def _getNodeTypeById(self, node_id):
        if node_id in self.vehicles: return "V"
        if node_id in self.UAVs: return "U"
        return None

    def _getNodeIdxById(self, node_id):
        if node_id in self.vehicles: return self.vehicle_ids_as_index.index(node_id)
        if node_id in self.UAVs: return self.uav_ids_as_index.index(node_id)
        return -1

    def _updateTraffics(self):
        self.trace.extend(("traffic_update_started", "traffic_update_finished"))
        if self.remove_target:
            self.UAVs.pop("u0", None)
            self.uav_ids_as_index = []

    def _updateTask(self):
        for task_id, route in self.task_return_routes.items():
            self.tasks[task_id].route = list(route)
            self.tasks[task_id].lifecycle = TaskLifecycle.RETURNING
        self.task_return_routes = {}

    def _updateWirelessCommunication(self):
        return None

    def _allocate_communication_RBs(self, assignments):
        profiles = {}
        for task_id, rb_nos in assignments.items():
            task = self.tasks[task_id]
            target = task.getToOffloadRoute()[0]
            source_type = self._getNodeTypeById(task.getCurrentNodeId())
            target_type = self._getNodeTypeById(target)
            if source_type is None or target_type is None:
                continue
            profiles[task_id] = {
                "task": task,
                "tx_idx": self._getNodeIdxById(task.getCurrentNodeId()),
                "rx_idx": self._getNodeIdxById(target),
                "channel_type": f"{source_type}2{target_type}",
                "RB_Nos": list(rb_nos),
            }
        return profiles

    def _compute_communication_rate(self, activated):
        del activated

    def _execute_communication(self, activated):
        del activated

    def step(self):
        self.step_calls += 1
        self._updateTraffics()
        self._updateTask()
        self._updateWirelessCommunication()
        if self.alloc_cpu_callback is not None:
            self.alloc_cpu_callback({})
        self.simulation_time += self.simulation_interval
        return False


class SpySchedulers:
    def __init__(self, *, fail_return=False):
        self.calls = []
        self.fail_return = fail_return

    def setComputingCallBack(self, env, callback):
        self.calls.append("cpu")
        env.alloc_cpu_callback = callback

    def setTaskOffloading(self, env, task_node_id, task_id, target_node_id, route=None):
        self.calls.append(("offload", task_id))
        task = env.tasks[task_id]
        task.route = list(route)
        task.current = task_node_id
        task.lifecycle = TaskLifecycle.OFFLOADING
        return True

    def setTaskReturnRoute(self, env, task_id, route):
        self.calls.append(("return", task_id))
        if self.fail_return:
            raise RuntimeError("return setter failed")
        env.task_return_routes[task_id] = list(route)

    def setCommunicationWithRB(self, env, task_id, rb_nos):
        self.calls.append(("rb", task_id, tuple(rb_nos)))
        env.activated_offloading_tasks_with_RB_Nos[task_id] = list(rb_nos)


def physical_snapshot(tasks, phase=SnapshotPhase.DECISION, *, include_target=True):
    nodes = [
        PhysicalNode("v0", "V", True, (0.0, 0.0, 0.0)),
        PhysicalNode("v1", "V", True, (1.0, 0.0, 0.0)),
    ]
    edges = []
    if include_target:
        nodes.append(PhysicalNode("u0", "U", True, (5.0, 0.0, 0.0)))
        edges.extend(
            [
                PhysicalEdge("physical::v0::u0::V2U", "v0", "u0", "V2U", True),
                PhysicalEdge("physical::v1::u0::V2U", "v1", "u0", "V2U", True),
                PhysicalEdge("physical::u0::v0::U2V", "u0", "v0", "U2V", True),
            ]
        )
    task_rows = tuple(
        TaskSnapshot(
            item.task_id,
            item.source,
            item.lifecycle,
            item.current,
            tuple(item.route),
            item.destination,
            0.0,
        )
        for item in tasks
    )
    return AirFogSimSnapshot(phase, 1.0, tuple(nodes), tuple(edges), task_rows, (), ())


def built_for(snapshot, frame_index=1, n_rb=2):
    return build_frame_decision(
        snapshot,
        trajectory_id="traj0",
        frame_index=frame_index,
        seed=1,
        n_rb=n_rb,
        vocabulary=FullTrajectoryVocabulary(),
        route_revisions=RouteRevisionLedger(),
        node_cpu={"v0": 2.0, "v1": 2.0, "u0": 5.0},
        node_distance={("v0", "v1"): 1.0, ("v0", "u0"): 5.0, ("v1", "u0"): 4.0, ("u0", "v0"): 5.0},
    )


def phase_observer(decision, env):
    def observe(_env, *, phase):
        env.trace.append(f"{phase.value}_snapshot_captured")
        if phase == SnapshotPhase.EXECUTION and env.remove_target:
            return physical_snapshot(tuple(env.tasks.values()), phase, include_target=False)
        return replace(decision, phase=phase, simulation_time=env.simulation_time)
    return observe


class FullCollectorExecutorTests(unittest.TestCase):
    def test_invalid_action_calls_no_setter_or_step(self):
        task = FakeTask("task0", "v0", TaskLifecycle.WAITING_TO_OFFLOAD)
        snap = physical_snapshot([task])
        built = built_for(snap)
        bad = replace(
            built,
            action=replace(
                built.action,
                rb_allocations=(replace(built.action.rb_allocations[0], rb_index=99),),
            ),
        )
        env = FakeEnv([task])
        schedulers = SpySchedulers()

        with self.assertRaisesRegex(CollectorContractError, "rb_out_of_range"):
            execute_full_collector_step(
                env,
                bad,
                trajectory_id="traj0",
                task_scheduler=schedulers,
                communication_scheduler=schedulers,
                computation_scheduler=schedulers,
                observer=phase_observer(snap, env),
            )
        self.assertEqual([], schedulers.calls)
        self.assertEqual(0, env.step_calls)

    def test_setter_order_execution_snapshot_and_outcome(self):
        task = FakeTask("task0", "v0", TaskLifecycle.WAITING_TO_OFFLOAD)
        snap = physical_snapshot([task])
        built = built_for(snap)
        env = FakeEnv([task])
        schedulers = SpySchedulers()

        result = execute_full_collector_step(
            env,
            built,
            trajectory_id="traj0",
            task_scheduler=schedulers,
            communication_scheduler=schedulers,
            computation_scheduler=schedulers,
            observer=phase_observer(snap, env),
        )

        self.assertEqual(["cpu", ("offload", "task0"), ("rb", "task0", (0,))], schedulers.calls)
        self.assertTrue(result.stepped)
        self.assertEqual(SnapshotPhase.EXECUTION, result.execution_snapshot.phase)
        self.assertEqual(SnapshotPhase.OUTCOME, result.outcome_snapshot.phase)
        self.assertLess(result.temporal_trace.index("execution_snapshot_captured"), result.temporal_trace.index("env_step_finished"))

    def test_partial_setter_failure_is_quarantined_without_step_or_retry(self):
        offload = FakeTask("off", "v0", TaskLifecycle.WAITING_TO_OFFLOAD)
        returning = FakeTask("ret", "u0", TaskLifecycle.WAITING_TO_RETURN, destination="v0")
        snap = physical_snapshot([offload, returning])
        built = built_for(snap, frame_index=0)
        env = FakeEnv([offload, returning])
        schedulers = SpySchedulers(fail_return=True)

        result = execute_full_collector_step(
            env,
            built,
            trajectory_id="traj0",
            task_scheduler=schedulers,
            communication_scheduler=schedulers,
            computation_scheduler=schedulers,
            observer=phase_observer(snap, env),
        )

        self.assertTrue(result.quarantined)
        self.assertEqual("quarantined_after_partial_setter_failure", result.quarantine_reason)
        self.assertFalse(result.stepped)
        self.assertFalse(result.training_eligible)
        self.assertIsNone(result.action)
        self.assertEqual(0, env.step_calls)
        self.assertEqual(1, sum(call == ("return", "ret") for call in schedulers.calls))

    def test_cross_transmitter_rb_reuse_records_direct_per_rb_fields(self):
        a = FakeTask("a", "v0", TaskLifecycle.OFFLOADING, route=("u0",))
        b = FakeTask("b", "v1", TaskLifecycle.OFFLOADING, route=("u0",))
        snap = physical_snapshot([a, b])
        flows = (
            LogicalFlow("flow::traj0::a::offload::0", "traj0", "a", "offload", 0),
            LogicalFlow("flow::traj0::b::offload::0", "traj0", "b", "offload", 0),
        )
        hops = (
            CarryingHop("hop::a", flows[0].flow_id, 0, "v0", "u0", "physical::v0::u0::V2U", "wireless"),
            CarryingHop("hop::b", flows[1].flow_id, 0, "v1", "u0", "physical::v1::u0::V2U", "wireless"),
        )
        decisions = tuple(
            DecisionRow(t.task_id, t.lifecycle, True, "continue_current_route", "u0", ("u0",), flow.flow_id, hop.hop_id, None, None, False)
            for t, flow, hop in zip((a, b), flows, hops)
        )
        action = JointFrameAction(0, decisions, flows, hops, tuple(RbAllocation(flow.flow_id, hop.hop_id, 0) for flow, hop in zip(flows, hops)))
        built = BuiltFrameDecision(action, tuple({"task_id": t.task_id, "lifecycle": t.lifecycle, "requires_route_setter": False} for t in (a, b)), "interference_reuse")
        env = FakeEnv([a, b])
        schedulers = SpySchedulers()

        result = execute_full_collector_step(
            env, built, trajectory_id="traj0", task_scheduler=schedulers,
            communication_scheduler=schedulers, computation_scheduler=schedulers,
            observer=phase_observer(snap, env),
        )

        self.assertEqual(2, len(result.transfer_rows))
        self.assertEqual({0}, {row["rb_index"] for row in result.transfer_rows})
        self.assertTrue(all(row["observed_mask"] for row in result.transfer_rows))
        self.assertTrue(all("interference_plus_noise_mw" in row and "sinr_db" in row and "outage" in row and "rate_per_s" in row for row in result.transfer_rows))
        self.assertEqual(({"v0": 0.5, "v1": 0.6000000000000001}, {"u0": 1.1}), env.channel_manager.transmission_totals)

    def test_return_route_uses_frozen_destination(self):
        task = FakeTask("ret", "u0", TaskLifecycle.WAITING_TO_RETURN, destination="v0")
        snap = physical_snapshot([task])
        built = built_for(snap, frame_index=0)
        env = FakeEnv([task])
        schedulers = SpySchedulers()
        result = execute_full_collector_step(
            env, built, trajectory_id="traj0", task_scheduler=schedulers,
            communication_scheduler=schedulers, computation_scheduler=schedulers,
            observer=phase_observer(snap, env),
        )
        self.assertTrue(result.stepped)
        self.assertIn(("return", "ret"), schedulers.calls)
        self.assertEqual(("v0",), built.action.decisions[0].route_nodes)

    def test_node_disappearance_after_decision_is_retained_as_runtime_outcome(self):
        task = FakeTask("task0", "v0", TaskLifecycle.WAITING_TO_OFFLOAD)
        snap = physical_snapshot([task])
        built = built_for(snap)
        env = FakeEnv([task], remove_target=True)
        schedulers = SpySchedulers()
        result = execute_full_collector_step(
            env, built, trajectory_id="traj0", task_scheduler=schedulers,
            communication_scheduler=schedulers, computation_scheduler=schedulers,
            observer=phase_observer(snap, env),
        )
        self.assertTrue(result.stepped)
        self.assertFalse(result.quarantined)
        self.assertTrue(any(row["missing_reason"] == "endpoint_absent_at_execution" for row in result.transfer_rows))


class RealAirFogSimFullCollectorTests(unittest.TestCase):
    @staticmethod
    def _runtime_inputs(env):
        from pi_jwm.airfogsim_full_dual_graph_observer_v1 import observe_airfogsim_snapshot
        snap = observe_airfogsim_snapshot(env, phase=SnapshotPhase.DECISION)
        node_cpu = {
            node.node_id: float(env._getNodeById(node.node_id).getFogProfile().get("cpu", 0.0))
            for node in snap.nodes
        }
        node_distance = {
            (edge.source_id, edge.target_id): float(env.getDistanceBetweenNodesById(edge.source_id, edge.target_id))
            for edge in snap.physical_edges
            if edge.edge_type != "wired"
        }
        return snap, node_cpu, node_distance

    def test_natural_remote_action_executes_one_real_step(self):
        old_cwd = Path.cwd()
        env = None
        try:
            os.chdir(p2_runner.EXAMPLE_DIR)
            env, task_sched, comm_sched, comp_sched, _ = p2_runner._build_environment(3, 5.0)
            ready, _ = p2_runner._warm_to_branch(env)
            snap, node_cpu, node_distance = self._runtime_inputs(env)
            built = build_frame_decision(
                snap, trajectory_id="real-seed3", frame_index=1, seed=3,
                n_rb=int(env.channel_manager.n_RB), vocabulary=FullTrajectoryVocabulary(),
                route_revisions=RouteRevisionLedger(), node_cpu=node_cpu, node_distance=node_distance,
            )
            before = float(env.simulation_time)
            result = execute_full_collector_step(
                env, built, trajectory_id="real-seed3", task_scheduler=task_sched,
                communication_scheduler=comm_sched, computation_scheduler=comp_sched,
            )
            selected = [row for row in built.action.decisions if row.task_id == str(ready.getTaskId())]
            self.assertTrue(selected and selected[0].selected)
            self.assertTrue(result.stepped)
            self.assertGreater(float(env.simulation_time), before)
            self.assertTrue(result.transfer_rows)
        finally:
            if env is not None:
                env.close()
            os.chdir(old_cwd)

    def test_local_action_has_real_task_transition_without_flow_or_rb(self):
        old_cwd = Path.cwd()
        env = None
        try:
            os.chdir(p2_runner.EXAMPLE_DIR)
            env, task_sched, comm_sched, comp_sched, _ = p2_runner._build_environment(4, 5.0)
            ready, _ = p2_runner._warm_to_branch(env)
            snap, node_cpu, node_distance = self._runtime_inputs(env)
            built = build_frame_decision(
                snap, trajectory_id="real-local", frame_index=0, seed=4,
                n_rb=int(env.channel_manager.n_RB), vocabulary=FullTrajectoryVocabulary(),
                route_revisions=RouteRevisionLedger(), node_cpu=node_cpu, node_distance=node_distance,
            )
            selected = next(row for row in built.action.decisions if row.task_id == str(ready.getTaskId()))
            self.assertEqual(str(ready.getTaskNodeId()), selected.target_node_id)
            self.assertEqual((), built.action.flows)
            self.assertEqual((), built.action.rb_allocations)
            result = execute_full_collector_step(
                env, built, trajectory_id="real-local", task_scheduler=task_sched,
                communication_scheduler=comm_sched, computation_scheduler=comp_sched,
            )
            self.assertTrue(result.stepped)
            self.assertEqual((), result.transfer_rows)
            self.assertTrue(any(str(ready.getTaskId()) in row["task_ids"] for row in result.cpu_rows))
        finally:
            if env is not None:
                env.close()
            os.chdir(old_cwd)

    def test_return_fixture_uses_real_task_and_frozen_destination(self):
        old_cwd = Path.cwd()
        env = None
        try:
            os.chdir(p2_runner.EXAMPLE_DIR)
            env, task_sched, comm_sched, comp_sched, _ = p2_runner._build_environment(5, 5.0)
            ready, _ = p2_runner._warm_to_branch(env)
            task_id = str(ready.getTaskId())
            source = str(ready.getTaskNodeId())
            compute_node = sorted(env.RSUs)[0]
            env.task_manager._waiting_to_offload_tasks[source].remove(ready)
            ready.setAttribute("_assigned_to", compute_node)
            ready.setAttribute("_routes", [compute_node])
            ready.setAttribute("_computed_size", float(ready.getTaskCPU()))
            ready.setAttribute("_required_returned_size", max(float(ready.getReturnedSize()), 1.0))
            ready.setAttribute("_start_to_transmit_time", float(env.simulation_time))
            ready.setAttribute("_to_offload_route", [])
            env.task_manager._waiting_to_return_tasks.setdefault(compute_node, []).append(ready)
            snap, node_cpu, node_distance = self._runtime_inputs(env)
            built = build_frame_decision(
                snap, trajectory_id="real-return", frame_index=0, seed=5,
                n_rb=int(env.channel_manager.n_RB), vocabulary=FullTrajectoryVocabulary(),
                route_revisions=RouteRevisionLedger(), node_cpu=node_cpu, node_distance=node_distance,
            )
            selected = next(row for row in built.action.decisions if row.task_id == task_id)
            self.assertEqual((source,), selected.route_nodes)
            result = execute_full_collector_step(
                env, built, trajectory_id="real-return", task_scheduler=task_sched,
                communication_scheduler=comm_sched, computation_scheduler=comp_sched,
            )
            self.assertTrue(result.stepped)
            self.assertTrue(any(row["task_id"] == task_id for row in result.transfer_rows))
        finally:
            if env is not None:
                env.close()
            os.chdir(old_cwd)


if __name__ == "__main__":
    unittest.main()
