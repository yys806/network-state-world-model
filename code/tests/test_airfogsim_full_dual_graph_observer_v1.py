from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
SCRIPTS_ROOT = CODE_ROOT / "scripts"
for path in (SRC_ROOT, SCRIPTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pi_jwm.airfogsim_full_dual_graph_observer_v1 import (  # noqa: E402
    capture_execution_snapshot,
    observe_airfogsim_snapshot,
)
from pi_jwm.full_dual_graph_collector_contract_v1 import (  # noqa: E402
    CollectorContractError,
    SnapshotPhase,
    TaskLifecycle,
)
import run_p2_single_step_collector_preflight_v1 as p2_runner  # noqa: E402


class FakeNode:
    def __init__(self, position=(0.0, 0.0, 0.0)):
        self._position = position

    def getPosition(self):
        return self._position


class FakeTask:
    def __init__(
        self,
        task_id: str,
        *,
        task_node_id: str = "v0",
        current_node_id: str = "v0",
        route=(),
        return_destination_id: str | None = "v0",
        arrival_time: float = 0.0,
    ):
        self._task_id = task_id
        self._task_node_id = task_node_id
        self._current_node_id = current_node_id
        self._route = list(route)
        self._return_destination_id = return_destination_id
        self._arrival_time = arrival_time

    def getTaskId(self):
        return self._task_id

    def getTaskNodeId(self):
        return self._task_node_id

    def getCurrentNodeId(self):
        return self._current_node_id

    def getToOffloadRoute(self):
        return list(self._route)

    def getToReturnNodeId(self):
        return self._return_destination_id

    def getTaskArrivalTime(self):
        return self._arrival_time


class FakeTaskManager:
    def __init__(self):
        tasks = {
            name: FakeTask(name)
            for name in (
                "to_generate",
                "waiting",
                "offloading",
                "computing",
                "waiting_return",
                "returning",
                "done",
                "failed",
            )
        }
        self._to_generate_task_infos = {"v0": [tasks["to_generate"]]}
        self._waiting_to_offload_tasks = {"v0": [tasks["waiting"]]}
        self._offloading_tasks = {"v0": [tasks["offloading"]]}
        self._computing_tasks = {"v0": [tasks["computing"]]}
        self._waiting_to_return_tasks = {"v0": [tasks["waiting_return"]]}
        self._returning_tasks = {"v0": [tasks["returning"]]}
        self._done_tasks = {"v0": [tasks["done"]]}
        self._out_of_ddl_tasks = {"v0": [tasks["failed"]]}
        dag = nx.DiGraph()
        dag.add_edge("waiting", "offloading")
        self._task_dependencies = {"v0": dag}


class FakeChannelManager:
    n_RB = 2

    def getCSI(self, tx_index, rx_index, tx_type, rx_type):
        del tx_index, rx_index
        return np.asarray([len(tx_type), len(rx_type)], dtype=float)


class FakeWiredManager:
    def __init__(self, links=()):
        self._links = set(links)

    def hasLink(self, source, target):
        return (source, target) in self._links


def fake_observer_env():
    env = SimpleNamespace(
        simulation_time=3.0,
        vehicles={"v0": FakeNode((1.0, 2.0, 0.0))},
        UAVs={},
        RSUs={},
        cloudServers={},
        vehicle_ids_as_index=["v0"],
        uav_ids_as_index=[],
        rsu_ids_as_index=[],
        cloud_server_ids_as_index=[],
        channel_manager=FakeChannelManager(),
        wired_manager=FakeWiredManager(),
        task_manager=FakeTaskManager(),
    )
    env._getNodeIdxById = lambda node_id: env.vehicle_ids_as_index.index(node_id)
    return env


class AirFogSimFullObserverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._old_cwd = Path.cwd()
        os.chdir(p2_runner.EXAMPLE_DIR)
        cls.env, *_ = p2_runner._build_environment(seed=0, max_time=0.3)
        cls.env.alloc_cpu_callback = lambda _: {}
        cls.env.step()

    @classmethod
    def tearDownClass(cls):
        try:
            cls.env.close()
        finally:
            os.chdir(cls._old_cwd)

    def test_extracts_complete_current_physical_structure_from_real_environment(self):
        channel_manager = self.env.channel_manager
        for name in dir(channel_manager):
            if name.endswith("_active_links"):
                getattr(channel_manager, name).fill(False)

        snapshot = observe_airfogsim_snapshot(self.env, phase=SnapshotPhase.DECISION)
        expected_nodes = {
            **{node_id: "V" for node_id in self.env.vehicles},
            **{node_id: "U" for node_id in self.env.UAVs},
            **{node_id: "I" for node_id in self.env.RSUs},
            **{node_id: "C" for node_id in self.env.cloudServers},
        }
        self.assertEqual(expected_nodes, {node.node_id: node.node_type for node in snapshot.nodes})
        self.assertTrue(all(node.position is not None for node in snapshot.nodes))

        wireless_nodes = {
            node_id: node_type
            for node_id, node_type in expected_nodes.items()
            if node_type in "VUI"
        }
        expected_wireless = {
            (source, target, f"{wireless_nodes[source]}2{wireless_nodes[target]}")
            for source in wireless_nodes
            for target in wireless_nodes
            if source != target
        }
        observed_wireless = {
            (edge.source_id, edge.target_id, edge.edge_type)
            for edge in snapshot.physical_edges
            if edge.edge_type != "wired"
        }
        self.assertEqual(expected_wireless, observed_wireless)
        self.assertTrue(all(edge.present for edge in snapshot.physical_edges))
        self.assertEqual(len(expected_wireless), len(snapshot.channel_rows))
        self.assertTrue(all(row["source_method"] == "channel_manager.getCSI" for row in snapshot.channel_rows))

        expected_wired = {
            (source, target)
            for source in (*self.env.RSUs, *self.env.cloudServers)
            for target in (*self.env.RSUs, *self.env.cloudServers)
            if source != target and self.env.wired_manager.hasLink(source, target)
        }
        observed_wired = {
            (edge.source_id, edge.target_id)
            for edge in snapshot.physical_edges
            if edge.edge_type == "wired"
        }
        self.assertEqual(expected_wired, observed_wired)
        cloud_ids = set(self.env.cloudServers)
        self.assertFalse(
            any(
                edge.edge_type != "wired"
                and (edge.source_id in cloud_ids or edge.target_id in cloud_ids)
                for edge in snapshot.physical_edges
            )
        )

    def test_extracts_all_task_lifecycles_and_precedence_only_dag(self):
        snapshot = observe_airfogsim_snapshot(
            fake_observer_env(), phase=SnapshotPhase.DECISION
        )

        self.assertEqual(
            {
                "to_generate": TaskLifecycle.TO_GENERATE,
                "waiting": TaskLifecycle.WAITING_TO_OFFLOAD,
                "offloading": TaskLifecycle.OFFLOADING,
                "computing": TaskLifecycle.COMPUTING,
                "waiting_return": TaskLifecycle.WAITING_TO_RETURN,
                "returning": TaskLifecycle.RETURNING,
                "done": TaskLifecycle.DONE,
                "failed": TaskLifecycle.FAILED,
            },
            {task.task_id: task.lifecycle for task in snapshot.tasks},
        )
        self.assertEqual(
            "v0",
            next(task for task in snapshot.tasks if task.task_id == "returning").return_destination_id,
        )
        self.assertEqual(1, len(snapshot.dag_edges))
        self.assertEqual("not_modeled", snapshot.dag_edges[0].communication_mapping)

    def test_execution_snapshot_does_not_read_stale_fast_fading_arrays(self):
        env = fake_observer_env()

        snapshot = observe_airfogsim_snapshot(env, phase=SnapshotPhase.EXECUTION)

        self.assertEqual((), snapshot.channel_rows)
        outcome = observe_airfogsim_snapshot(env, phase=SnapshotPhase.OUTCOME)
        self.assertEqual((), outcome.channel_rows)

    def test_initial_structure_keeps_edge_and_masks_csi_when_node_index_is_unavailable(self):
        env = fake_observer_env()
        env.UAVs = {"u0": FakeNode((2.0, 0.0, 0.0))}

        def index(node_id):
            if node_id == "u0":
                raise ValueError("u0 is not in list")
            return 0

        env._getNodeIdxById = index

        snapshot = observe_airfogsim_snapshot(env, phase=SnapshotPhase.DECISION)
        row = next(
            item
            for item in snapshot.channel_rows
            if item["source_id"] == "v0" and item["target_id"] == "u0"
        )

        self.assertIn(
            ("v0", "u0"),
            {(edge.source_id, edge.target_id) for edge in snapshot.physical_edges},
        )
        self.assertIs(row["observed_mask"], False)
        self.assertEqual((), row["channel_attenuation_db"])
        self.assertEqual("NODE_INDEX_UNAVAILABLE_AT_DECISION", row["missing_reason"])
        self.assertIsNone(row["source_method"])

    def test_execution_hook_captures_after_traffic_before_task_and_restores(self):
        trace = ["decision_snapshot_captured"]

        class HookEnv:
            traffic_interval = 0.1
            simulation_interval = 0.1

            def _updateTraffics(self):
                trace.extend(("traffic_update_started", "traffic_update_finished"))

            def step(self):
                self._updateTraffics()
                trace.append("task_update_started")

        env = HookEnv()
        original_function = env._updateTraffics.__func__

        def observer():
            trace.append("execution_snapshot_captured")
            return "execution"

        with capture_execution_snapshot(env, observer) as captures:
            env.step()

        self.assertEqual(
            [
                "decision_snapshot_captured",
                "traffic_update_started",
                "traffic_update_finished",
                "execution_snapshot_captured",
                "task_update_started",
            ],
            trace,
        )
        self.assertEqual(["execution"], captures)
        self.assertIs(original_function, env._updateTraffics.__func__)

    def test_execution_hook_restores_after_step_failure(self):
        class HookEnv:
            traffic_interval = 0.1
            simulation_interval = 0.1

            def _updateTraffics(self):
                return None

            def step(self):
                self._updateTraffics()
                raise RuntimeError("step failed")

        env = HookEnv()
        original_function = env._updateTraffics.__func__
        with self.assertRaisesRegex(RuntimeError, "step failed"):
            with capture_execution_snapshot(env, lambda: "execution"):
                env.step()
        self.assertIs(original_function, env._updateTraffics.__func__)

    def test_execution_hook_rejects_multiple_simulation_substeps(self):
        env = SimpleNamespace(
            traffic_interval=0.2,
            simulation_interval=0.1,
            _updateTraffics=lambda: None,
        )

        with self.assertRaisesRegex(CollectorContractError, "multi_substep_not_supported"):
            with capture_execution_snapshot(env, lambda: "execution"):
                pass


if __name__ == "__main__":
    unittest.main()
